"""Multimodal data: ECG .npz + clinical text .csv loading / alignment / splitting / tokenization / Dataset.

Generic load_npz / split_data live in model.common.data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _norm_key(v):
    """Normalize filename-like keys so different suffixes still match.
    Example: "123.dat" and "123.med" -> "123"."""
    s = str(v).strip().lower()
    base = s.replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".dat", ".med", ".mat", ".csv", ".hea"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def load_text_csv(path: str):
    """Load clinical text csv: expected columns filename(optional), text, label."""
    import pandas as pd

    df = pd.read_csv(path)
    needed = ["text", "label"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Text CSV is missing column '{c}'; available columns: {list(df.columns)}")
    pid_col = None
    for c in ("filename", "file", "record_id", "ecg_name", "name"):
        if c in df.columns:
            pid_col = c
            break
    return df, pid_col


def align_ecg_text(ecg_data: dict, text_df, pid_col=None):
    """Align ECG records (by patient_ids) with text rows (by filename column).

    If pid_col is None, row-wise alignment is assumed (text csv rows must be in
    the same order as ECG records).
    Labels are taken from the ECG .npz (DSA gold standard); csv labels are used
    only for consistency checking.
    Returns (ecg, labels, texts, keys) with ecg.shape = (N, leads, L).
    """
    ecg = ecg_data["ecg"]
    labels = ecg_data.get("labels")
    if labels is None:
        raise KeyError("ECG .npz is missing labels")
    if ecg.ndim != 3:
        raise SystemExit(f"ecg must be 3D (N, leads, L), got shape={ecg.shape}")
    if len(ecg) != len(labels):
        raise SystemExit(f"ecg and labels length mismatch: {len(ecg)} vs {len(labels)}")

    texts = text_df["text"].astype(str).tolist()
    text_labels = text_df["label"].astype(int).tolist()

    if pid_col is None:
        if len(texts) != len(ecg):
            raise SystemExit(
                "Text CSV has no filename column and its row count differs from ECG, cannot align: "
                f"text={len(texts)} ecg={len(ecg)}"
            )
        print("[Align] Text CSV has no filename column; aligning by row order.")
        return ecg, np.asarray(labels, dtype=np.int64), texts, None

    raw_pids = ecg_data.get("patient_ids")
    if raw_pids is None:
        raw_pids = [str(i) for i in range(len(ecg))]
    if len(raw_pids) != len(ecg):
        raise SystemExit(
            f"patient_ids length ({len(raw_pids)}) differs from ecg ({len(ecg)}), cannot align."
        )
    ecg_keys = [_norm_key(p) for p in raw_pids]
    text_keys = [_norm_key(v) for v in text_df[pid_col]]

    key_to_text = {}
    for k, t, y in zip(text_keys, texts, text_labels):
        key_to_text.setdefault(k, (t, y))

    idx_ecg, out_texts, out_labels = [], [], []
    dropped = 0
    mismatch = 0
    for i, k in enumerate(ecg_keys):
        if k not in key_to_text:
            dropped += 1
            continue
        t, y_csv = key_to_text[k]
        y_npz = int(labels[i])
        if y_csv != y_npz:
            mismatch += 1
        idx_ecg.append(i)
        out_texts.append(t)
        out_labels.append(y_npz)
    if dropped:
        print(f"[Align] {dropped} ECG records have no matching text and were dropped.")
    if mismatch:
        print(f"[Align] Warning: {mismatch} records have text CSV labels inconsistent with ECG npz labels; npz (DSA gold standard) takes precedence.")
    if not idx_ecg:
        raise SystemExit("No ECG records match any text; please check filename/patient_ids consistency.")

    print(f"[Align] Successfully aligned {len(idx_ecg)} records.")
    return (ecg[np.asarray(idx_ecg)],
            np.asarray(out_labels, dtype=np.int64),
            out_texts,
            [ecg_keys[i] for i in idx_ecg])


# ---------------------------------------------------------------------------
# Text tokenization: MacBERT + head(60%)/tail(40%) truncation to 512 tokens
# ---------------------------------------------------------------------------

class HeadTailTokenizer:
    """MacBERT tokenizer with the paper's head-tail truncation strategy:
    total length capped at 512 tokens (incl. [CLS]/[SEP]); 60% of the budget
    keeps the head, 40% keeps the tail."""

    def __init__(self, model_name_or_path: str, max_tokens: int = 512, head_ratio: float = 0.6):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_tokens = int(max_tokens)
        self.head_ratio = float(head_ratio)
        self.budget = self.max_tokens - 2  # reserve [CLS]/[SEP]

    def encode(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.budget:
            return self.tokenizer.build_inputs_with_special_tokens(ids)
        head_len = int(self.budget * self.head_ratio)
        tail_len = self.budget - head_len
        ids = ids[:head_len] + ids[-tail_len:]
        return self.tokenizer.build_inputs_with_special_tokens(ids)


class MultimodalDataset(Dataset):
    """Dataset for (ecg, text) pairs. mode: 'fusion' | 'text' | 'ecg'.
    tokenized_ids: pre-tokenized text (list of lists of token ids) so that
    tokenization runs once before training instead of once per epoch."""

    def __init__(self, ecg, tokenized_ids, labels, mode: str = "fusion"):
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.tokenized_ids = tokenized_ids
        self.mode = mode
        if mode in ("fusion", "ecg"):
            self.ecg = torch.from_numpy(np.asarray(ecg, dtype=np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {"y": self.labels[i]}
        if self.mode in ("fusion", "ecg"):
            item["ecg"] = self.ecg[i]
        if self.mode in ("fusion", "text"):
            ids = self.tokenized_ids[i]
            item["input_ids"] = torch.tensor(ids, dtype=torch.long)
            item["attention_mask"] = torch.ones(len(ids), dtype=torch.long)
        return item


def collate_multimodal(batch):
    out = {"y": torch.stack([b["y"] for b in batch])}
    if "ecg" in batch[0]:
        out["ecg"] = torch.stack([b["ecg"] for b in batch])
    if "input_ids" in batch[0]:
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["input_ids"])] = b["input_ids"]
            mask[i, : len(b["attention_mask"])] = b["attention_mask"]
        out["input_ids"] = ids
        out["attention_mask"] = mask
    return out
