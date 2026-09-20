"""Data model: a single EMR clinical document."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ClinicalDocument:
    """A single EMR document / report."""

    patient_id: str
    doc_type: str
    content: str
    recorded_at: Optional[datetime] = None
    doc_id: str = ""
    source: str = ""

    def is_valid_time(self) -> bool:
        return self.recorded_at is not None
