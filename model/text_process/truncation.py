"""Stage 2: text length normalization.

- Length distribution statistics (char / token views: mean, median, P95, max, over-limit ratio)
- Head-tail truncation: total length capped at max_tokens (incl. [CLS]/[SEP]); 60% of the
  budget keeps the head and 40% keeps the tail
"""
import logging
import math
import re
import statistics
from typing import Any, Dict, List, Sequence

logger = logging.getLogger("clinical_text_processing")


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (e.g. median / 95th)."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def length_statistics(lengths: Sequence[int], max_tokens: int = 512) -> Dict[str, float]:
    """
    Length distribution statistics (shared by the char and token views).
    Returns: count, mean, median, 95th percentile, max, over-limit ratio.
    """
    n = len(lengths)
    if n == 0:
        return {
            "count": 0, "mean": float("nan"), "median": float("nan"),
            "p95": float("nan"), "max": float("nan"), "over_ratio": float("nan"),
        }
    s = sorted(lengths)
    over_ratio = sum(1 for x in s if x > max_tokens) / n
    return {
        "count": n,
        "mean": round(statistics.mean(s), 2),
        "median": float(statistics.median(s)),
        "p95": round(_percentile(s, 0.95), 2),
        "max": float(max(s)),
        "over_ratio": round(over_ratio, 4),
    }


class ClinicalTextTokenizer:
    """
    Wrapper around the clinical text tokenizer.

    Production: HuggingFace `hfl/chinese-macbert-base` (as in the paper);
    requires `pip install transformers` and network access.

    When offline / transformers is unavailable, it falls back to a
    character-level tokenizer: 1 Chinese character ~ 1 token (consistent
    with Chinese BERT behaviour). It is only meant for local testing and
    does not affect the correctness of the truncation logic.
    """

    CLS, SEP, PAD = 0, 1, 2

    def __init__(self, model_name: str = "hfl/chinese-macbert-base", offline: bool = False):
        self.model_name = model_name
        self.backend = "transformers"
        self._hf = None
        self._char_vocab: Dict[str, int] = {
            "[PAD]": self.PAD, "[CLS]": self.CLS, "[SEP]": self.SEP,
        }
        if not offline:
            try:
                from transformers import AutoTokenizer

                self._hf = AutoTokenizer.from_pretrained(model_name)
            except Exception as exc:  # not installed / no network, etc.
                logger.warning("Cannot load HuggingFace tokenizer (%s); fall back to char-level.", exc)
                self.backend = "char-fallback"
        else:
            self.backend = "char-fallback"

    def encode(self, text: str) -> List[int]:
        """Encode text into a token id list (without [CLS]/[SEP])."""
        if self._hf is not None:
            return self._hf(text, add_special_tokens=False)["input_ids"]
        ids: List[int] = []
        # Consecutive letters/digits form one token (approx. WordPiece);
        # anything else is split per character (1 Chinese char ~ 1 token)
        for tok in re.findall(r"[A-Za-z0-9]+|.", text):
            if tok not in self._char_vocab:
                self._char_vocab[tok] = len(self._char_vocab)
            ids.append(self._char_vocab[tok])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a token id sequence back to text (special tokens removed)."""
        if self._hf is not None:
            return self._hf.decode(list(ids), skip_special_tokens=True)
        rev = {v: k for k, v in self._char_vocab.items()}
        return "".join(
            rev.get(i, "") for i in ids if i not in (self.CLS, self.SEP, self.PAD)
        )

    def add_special_tokens(self, ids: Sequence[int]) -> List[int]:
        """Wrap the sequence with [CLS] / [SEP]."""
        if self._hf is not None:
            return self._hf.build_inputs_with_special_tokens(list(ids))
        return [self.CLS] + list(ids) + [self.SEP]


class HeadTailTruncator:
    """
    Head-tail truncation:
      - Total input length = max_tokens (512), with 2 tokens reserved for
        [CLS]/[SEP]
      - Of the remaining budget, 60% keeps the head (head_ratio) and 40%
        keeps the tail (1 - head_ratio)
      - Sequences within the limit are kept intact (special tokens added)
    """

    def __init__(
        self,
        tokenizer: ClinicalTextTokenizer,
        max_tokens: int = 512,
        head_ratio: float = 0.6,
    ):
        assert 0 < head_ratio < 1
        assert max_tokens > 2
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.head_ratio = head_ratio
        self.tail_ratio = 1.0 - head_ratio

    def truncate(self, text: str) -> Dict[str, Any]:
        """
        Apply head-tail truncation to a single text.

        Returns: input_ids (incl. [CLS]/[SEP]), raw_len, final_len,
        truncated (whether truncation happened), head_kept / tail_kept.
        """
        ids = self.tokenizer.encode(text)
        raw_len = len(ids)
        budget = self.max_tokens - 2  # reserve [CLS]/[SEP]

        if raw_len <= budget:
            return {
                "input_ids": self.tokenizer.add_special_tokens(ids),
                "raw_len": raw_len,
                "final_len": raw_len + 2,
                "truncated": False,
                "head_kept": raw_len,
                "tail_kept": 0,
            }

        head_len = int(budget * self.head_ratio)   # 60% budget -> head
        tail_len = budget - head_len               # 40% budget -> tail
        new_ids = ids[:head_len] + ids[-tail_len:]
        result = {
            "input_ids": self.tokenizer.add_special_tokens(new_ids),
            "raw_len": raw_len,
            "final_len": len(new_ids) + 2,
            "truncated": True,
            "head_kept": head_len,
            "tail_kept": tail_len,
        }
        assert result["final_len"] <= self.max_tokens
        return result
