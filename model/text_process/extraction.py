"""Stage 1: clinical text extraction and QC.

- Load documents exported from an EMR system (jsonl / json / csv)
- Type whitelist + DSA keyword double exclusion (prevent gold-standard leakage)
- Concatenate all documents of the same patient into a single text sequence in chronological order
"""
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from model.text_process.config import TextProcessingConfig
from model.text_process.models import ClinicalDocument

logger = logging.getLogger("clinical_text_processing")


_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
)


def _parse_time(raw: Any) -> Optional[datetime]:
    """Parse an EMR time field into datetime; return None on failure."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text in ("", "nan", "NaT", "None"):
        return None
    if isinstance(raw, datetime):
        return raw
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:  # ISO format, e.g. 2024-03-01T09:30:00
        return datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Cannot parse time field: %r", raw)
        return None


def _row_to_doc(row: Dict[str, Any], source: str = "") -> Optional[ClinicalDocument]:
    patient_id = str(row.get("patient_id", "")).strip()
    doc_type = str(row.get("doc_type", "")).strip()
    content = str(row.get("content", "")).strip()
    if not patient_id or not doc_type:
        logger.warning("Skip record missing patient_id / doc_type: %s", row)
        return None
    return ClinicalDocument(
        patient_id=patient_id,
        doc_type=doc_type,
        content=content,
        recorded_at=_parse_time(row.get("recorded_at")),
        doc_id=str(row.get("doc_id", "")).strip(),
        source=source,
    )


def load_emr_export(path: Union[str, Path], encoding: str = "utf-8") -> List[ClinicalDocument]:
    """
    Load documents exported from an EMR system and return a document list.

    Two formats are supported:
      - JSON Lines / JSON array (recommended): each record contains
        patient_id, doc_type, content, recorded_at (optional), doc_id (optional)
      - CSV: header contains patient_id, doc_type, content,
        recorded_at (optional), doc_id (optional)

    In production, adapt an enterprise EMR API query into this function
    (just return a List[ClinicalDocument]); the downstream QC and
    concatenation logic stays unchanged.
    """
    path = Path(path)
    docs: List[ClinicalDocument] = []

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                clean = {k.strip(): v for k, v in row.items()}
                doc = _row_to_doc(clean, source=str(path))
                if doc is not None:
                    docs.append(doc)
        return docs

    text = path.read_text(encoding=encoding).strip()
    if not text:
        return docs

    if text.startswith("["):  # JSON array
        records = json.loads(text)
        for rec in records:
            doc = _row_to_doc(rec, source=str(path))
            if doc is not None:
                docs.append(doc)
    else:  # JSON Lines
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skip unparseable JSON line: %.80s", line)
                continue
            doc = _row_to_doc(rec, source=str(path))
            if doc is not None:
                docs.append(doc)
    return docs


def qc_documents(
    docs: Iterable[ClinicalDocument],
    cfg: TextProcessingConfig,
) -> Tuple[List[ClinicalDocument], List[Tuple[ClinicalDocument, str]]]:
    """
    Quality control rules:
      1. Type whitelist: only admission records / lab results are kept
      2. DSA exclusion: drop a document whose type or content hits an
         exclusion keyword (gold-standard leakage protection)
      3. Drop empty documents
      4. Deduplicate by (patient_id, doc_id)

    Returns (kept, [(dropped_doc, reason), ...]).
    """
    kept: List[ClinicalDocument] = []
    dropped: List[Tuple[ClinicalDocument, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for doc in docs:
        if doc is None:
            continue
        if doc.doc_type not in cfg.allowed_doc_types:
            dropped.append((doc, f"Type not in whitelist: {doc.doc_type}"))
            continue
        content_l = doc.content.lower()
        hit = next(
            (kw for kw in cfg.dsa_keywords if kw in content_l or kw in doc.doc_type.lower()),
            None,
        )
        if hit is not None:
            dropped.append((doc, f"Hits DSA exclusion keyword: {hit}"))
            continue
        if cfg.drop_empty and not doc.content.strip():
            dropped.append((doc, "Empty content"))
            continue
        key = (doc.patient_id, doc.doc_id or doc.content)
        if key in seen:
            dropped.append((doc, "Duplicate document"))
            continue
        seen.add(key)
        kept.append(doc)

    logger.info("QC done: kept %d, dropped %d", len(kept), len(dropped))
    return kept, dropped


def build_text_sequences(
    docs: Iterable[ClinicalDocument],
    cfg: TextProcessingConfig,
) -> Dict[str, str]:
    """
    Concatenate all documents of the same patient into one text sequence
    in chronological order.

    Documents without a timestamp are appended at the end with a warning;
    ties are sorted stably by doc_id.
    """
    by_patient: Dict[str, List[ClinicalDocument]] = {}
    for doc in docs:
        by_patient.setdefault(doc.patient_id, []).append(doc)

    sequences: Dict[str, str] = {}
    for pid, doc_list in by_patient.items():
        with_time = [d for d in doc_list if d.is_valid_time()]
        no_time = [d for d in doc_list if not d.is_valid_time()]
        if no_time:
            logger.warning("Patient %s has %d documents without time; appended at end", pid, len(no_time))
        ordered = sorted(with_time, key=lambda d: (d.recorded_at, d.doc_id)) + no_time

        segments: List[str] = []
        for d in ordered:
            text = d.content.strip()
            if cfg.tag_segments:
                # The Chinese doc type itself acts as the segment marker
                segments.append(f"【{d.doc_type}】{text}")
            else:
                segments.append(text)
        sequences[pid] = cfg.segment_separator.join(segments)
    return sequences
