
import hashlib
from typing import Iterable

import pandas as pd


def ensure_doc_id(df: pd.DataFrame, *, prefer_cols: Iterable[str] = ("doc_id", "id", "filename"), text_col: str = "text") -> pd.Series:
    """Return a stable doc_id Series.
    Priority: 1) an existing doc_id/id/filename column (first hit) 2) SHA1(text)[:16] fallback.
    """
    for c in prefer_cols:
        if c in df.columns:
            return df[c].astype(str)
    texts = df[text_col].astype(str) if text_col in df.columns else df.apply(lambda r: str(r.values), axis=1)
    return texts.apply(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
