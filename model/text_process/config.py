"""Pipeline configuration: extraction/QC rules + length-normalization parameters."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TextProcessingConfig:
    """Pipeline configuration."""

    # ---- Stage 1: extraction & QC ----
    # Whitelist of document types kept for concatenation. The values are
    # Chinese on purpose: they match the document type names used in
    # Chinese EMR data. Any other type is dropped.
    allowed_doc_types: Tuple[str, ...] = ("入院记录", "检验结果")
    # DSA exclusion: a document is dropped when its type name or content
    # hits any of these keywords. Chinese keywords match Chinese clinical
    # text; "dsa" additionally covers the English abbreviation.
    dsa_keywords: Tuple[str, ...] = (
        "dsa",
        "数字减影血管造影", "造影报告", "造影结果",
        "冠脉造影", "冠状动脉造影", "介入造影",
    )
    drop_empty: bool = True          # Drop documents with empty content
    tag_segments: bool = True        # Prefix each segment with a type tag
    segment_separator: str = "\n"    # Separator between concatenated segments

    # ---- Stage 2: length normalization ----
    model_name: str = "hfl/chinese-macbert-base"  # MacBERT used in the paper
    max_tokens: int = 512                          # BERT input limit
    head_ratio: float = 0.6                        # Budget ratio for the head (60%)
    tail_ratio: float = 0.4                        # Budget ratio for the tail (40%)
    offline: bool = False                          # Skip model download; use char fallback

    def __post_init__(self) -> None:
        assert 0 < self.head_ratio < 1, "head_ratio must be in (0, 1)"
        assert self.max_tokens > 2, "max_tokens must be > 2 (room for [CLS]/[SEP])"
        if abs((self.head_ratio + self.tail_ratio) - 1.0) > 1e-9:
            raise ValueError("head_ratio + tail_ratio must equal 1")
