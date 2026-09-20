"""End-to-end text processing pipeline: extraction/QC → concatenation → length analysis → head-tail truncation."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from model.text_process.config import TextProcessingConfig
from model.text_process.extraction import build_text_sequences, load_emr_export, qc_documents
from model.text_process.truncation import (
    ClinicalTextTokenizer,
    HeadTailTruncator,
    length_statistics,
)


class ClinicalTextProcessor:
    """Clinical text pipeline: extraction/QC -> concatenation -> length analysis -> truncation."""

    def __init__(self, cfg: Optional[TextProcessingConfig] = None):
        self.cfg = cfg or TextProcessingConfig()
        self.tokenizer = ClinicalTextTokenizer(self.cfg.model_name, offline=self.cfg.offline)
        self.truncator = HeadTailTruncator(
            self.tokenizer, self.cfg.max_tokens, self.cfg.head_ratio
        )

    def run(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        docs = load_emr_export(input_path)
        kept, dropped = qc_documents(docs, self.cfg)
        sequences = build_text_sequences(kept, self.cfg)

        char_lens = [len(s) for s in sequences.values()]
        token_lens = [len(self.tokenizer.encode(s)) for s in sequences.values()]
        char_stats = length_statistics(char_lens, self.cfg.max_tokens)
        token_stats = length_statistics(token_lens, self.cfg.max_tokens)

        truncated = {pid: self.truncator.truncate(seq) for pid, seq in sequences.items()}

        return {
            "input_docs": len(docs),
            "kept_docs": len(kept),
            "dropped_docs": len(dropped),
            "patients": len(sequences),
            "char_stats": char_stats,
            "token_stats": token_stats,
            "sequences": sequences,
            "truncated": truncated,
        }
