"""Raw ECG record loading: a directory of ``.med`` / ``.dat``(+``.hea``) files.

Reads the record files straight off disk and returns ``ecg:(N,leads,L)``,
``patient_ids`` and ``labels`` for the training scripts.

Two record kinds are supported:

  ``.med``            Machine-computed median beat. Raw int16, sample-major
                      ``(L, leads)`` interleaved. The file carries no header, so
                      neither the sampling rate nor the gain is known from it;
                      pass ``med_gain`` to convert to mV, otherwise the values
                      stay in ADC units.
  ``.dat`` + ``.hea`` WFDB record, format 16. Gain / baseline / units are read
                      from the header and applied, so the values come out in mV.

Both are converted to float32 ``(N, leads, L)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Suffixes treated as ECG records when scanning a directory.
RECORD_SUFFIXES: Tuple[str, ...] = (".med", ".dat", ".hea")

#: Column names accepted when reading a labels csv.
_ID_COLUMNS = ("id", "patient_id", "filename", "file", "record", "record_id",
               "ecg_name", "name")
_LABEL_COLUMNS = ("label", "omi", "y", "target")


# ---------------------------------------------------------------------------
# .med  (median beat, no header)
# ---------------------------------------------------------------------------

def read_med(path: str | Path, num_leads: int = 12) -> np.ndarray:
    """Read a ``.med`` median-beat file -> float32 ``(leads, L)``.

    The layout is int16 sample-major: all ``num_leads`` values of the first
    sample, then the second sample, and so on -- the same interleaving WFDB uses
    for format-16 records.
    """
    raw = np.fromfile(str(path), dtype="<i2")
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    if raw.size % num_leads:
        raise ValueError(
            f"{path}: {raw.size} int16 samples is not divisible by {num_leads} leads"
        )
    return raw.reshape(-1, num_leads).T.astype(np.float32)


# ---------------------------------------------------------------------------
# WFDB  (.dat + .hea, format 16)
# ---------------------------------------------------------------------------

def _parse_gain_field(field: str) -> Tuple[float, float, str]:
    """Parse a WFDB gain field such as ``1000.0(0)/mV`` -> (gain, baseline, units).

    ``units`` is empty when the field has no ``/unit`` part.
    """
    units = ""
    if "/" in field:
        field, units = field.split("/", 1)
    baseline = 0.0
    if "(" in field:
        field, rest = field.split("(", 1)
        try:
            baseline = float(rest.rstrip(")"))
        except ValueError:
            baseline = 0.0
    try:
        gain = float(field)
    except ValueError:
        gain = 1.0
    # WFDB convention: gain 0 means "uncalibrated", i.e. leave the ADC values.
    if gain == 0:
        gain = 1.0
    return gain, baseline, units


def parse_wfdb_header(hea_path: str | Path) -> Dict:
    """Parse a WFDB ``.hea`` header into a dict.

    Only the fields this pipeline needs are extracted: signal count, sampling
    rate, sample count, and per-signal file / format / gain / baseline / name.
    Raises on anything it cannot faithfully reproduce (non-format-16, per-signal
    byte offsets, mismatched per-signal files), rather than silently guessing.
    """
    lines = [ln.strip() for ln in Path(hea_path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{hea_path}: empty header")

    head = lines[0].split()
    if len(head) < 4:
        raise ValueError(f"{hea_path}: malformed header line: {lines[0]!r}")
    record, nsig, fs, nsamp = head[0], int(head[1]), float(head[2]), int(head[3])
    if len(lines) < 1 + nsig:
        raise ValueError(f"{hea_path}: expected {nsig} signal lines, found {len(lines) - 1}")

    signals: List[Dict] = []
    for line in lines[1:1 + nsig]:
        f = line.split()
        if len(f) < 3:
            raise ValueError(f"{hea_path}: malformed signal line: {line!r}")
        fmt = f[1]
        if not fmt.startswith("16"):
            raise SystemExit(
                f"{hea_path}: only WFDB format 16 int16 records are supported, got format {fmt!r}"
            )
        gain, baseline, units = _parse_gain_field(f[2])
        signals.append({
            "file": f[0],
            "fmt": fmt,
            "gain": gain,
            "baseline": baseline,
            "units": units,
            "adc_zero": int(f[4]) if len(f) > 4 else 0,
            # Byte offset is an optional field that only appears when the record
            # is NOT interleaved; a trailing description field is present when
            # the line has at least 9 columns.
            "offset": int(f[8]) if len(f) >= 10 else 0,
            "name": f[-1] if len(f) >= 9 else "",
        })

    files = {s["file"] for s in signals}
    if len(files) != 1:
        raise SystemExit(
            f"{hea_path}: signals span several files {sorted(files)}, which is not supported"
        )
    if any(s["offset"] != 0 for s in signals):
        raise SystemExit(f"{hea_path}: per-signal byte offsets are not supported (record is not interleaved)")

    return {
        "record": record,
        "nsig": nsig,
        "fs": fs,
        "nsamp": nsamp,
        "dat_file": signals[0]["file"],
        "signals": signals,
    }


def read_wfdb(hea_path: str | Path, num_leads: int = 12) -> Tuple[np.ndarray, float]:
    """Read a WFDB record -> (float32 ``(leads, L)`` in mV, sampling rate)."""
    hdr = parse_wfdb_header(hea_path)
    if hdr["nsig"] != num_leads:
        raise SystemExit(f"{hea_path}: header declares {hdr['nsig']} leads, expected {num_leads}")

    dat_path = Path(hea_path).with_name(hdr["dat_file"])
    if not dat_path.exists():
        dat_path = Path(hea_path).with_suffix(".dat")
    if not dat_path.exists():
        raise SystemExit(f"{hea_path}: data file {hdr['dat_file']} not found")

    raw = np.fromfile(str(dat_path), dtype="<i2")
    need = hdr["nsamp"] * num_leads
    if raw.size < need:
        raise SystemExit(
            f"{dat_path}: {raw.size} int16 samples, header declares {hdr['nsamp']} samples x "
            f"{num_leads} leads = {need}"
        )

    # Interleaved: sample-major, all leads of one sample then the next.
    m = raw[:need].reshape(hdr["nsamp"], num_leads).astype(np.float32)
    gains = np.array([s["gain"] for s in hdr["signals"]], dtype=np.float32)
    baselines = np.array([s["baseline"] for s in hdr["signals"]], dtype=np.float32)
    return ((m - baselines) / gains).T, hdr["fs"]


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def scan_ecg_dir(root: str | Path, prefer: Sequence[str] = ("med", "dat")) -> List[Dict]:
    """List the records under ``root``, one entry per record id.

    ``prefer`` decides which kind wins when a record has both; a ``.dat`` record
    is only usable when its ``.hea`` sits next to it.
    """
    root = Path(root)
    by_id: Dict[str, Dict[str, Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in RECORD_SUFFIXES:
            by_id.setdefault(p.stem, {})[p.suffix.lower()] = p

    records: List[Dict] = []
    skipped: List[str] = []
    for rid in sorted(by_id):
        files = by_id[rid]
        chosen: Optional[Dict] = None
        reasons: List[str] = []
        for kind in prefer:                       # order matters: first match wins
            if kind == "med" and ".med" in files:
                chosen = {"id": rid, "kind": "med", "path": files[".med"]}
                break
            if kind == "dat" and ".dat" in files:
                if ".hea" not in files:
                    reasons.append(".dat without .hea")   # try the next preference
                    continue
                chosen = {"id": rid, "kind": "dat", "path": files[".hea"]}
                break
        if chosen is not None:
            records.append(chosen)
        else:
            why = ", ".join(reasons) if reasons else ",".join(sorted(files))
            skipped.append(f"{rid} ({why})")

    if skipped:
        print(f"[ECG] Skipped {len(skipped)} record(s) with no usable file: "
              f"{', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    if not records:
        raise SystemExit(
            f"{root}: no usable ECG records found (looked for "
            f"{', '.join(prefer)} among {RECORD_SUFFIXES})"
        )
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[ECG] Found {len(records)} record(s) under {root}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return records


# ---------------------------------------------------------------------------
# Length fitting
# ---------------------------------------------------------------------------

def qrs_index(x: np.ndarray) -> int:
    """Index of the QRS complex: the peak of the lead-wise vector magnitude."""
    mag = np.sqrt((np.asarray(x, dtype=np.float64) ** 2).sum(axis=0))
    return int(np.argmax(mag))


def fit_length(x: np.ndarray, target_len: Optional[int], align: str = "qrs") -> np.ndarray:
    """Crop (or pad) ``(leads, L)`` to ``target_len`` samples.

    ``align`` picks the window: ``'qrs'`` centers it on the QRS complex,
    ``'center'`` on the record, ``'start'`` on the first sample. The default
    ``--seq-len 500`` feeds the paper's 500-sample (1 s at 500 Hz) median beat
    as-is (a .med record is already 500 samples); smaller targets apply a
    QRS-centred crop, which keeps the complex intact even when it is not
    exactly in the middle of every record.
    """
    l = x.shape[-1]
    if target_len is None or target_len == l:
        return x

    if target_len > l:                      # pad (only meaningful for alignment)
        pad = target_len - l
        if align in ("qrs", "center"):
            left = pad // 2
            print(f"[ECG] Warning: record has {l} samples, padding to {target_len}")
        else:
            left = 0
        return np.pad(x, ((0, 0), (left, pad - left)), mode="constant")

    if align == "start":
        start = 0
    elif align == "center":
        start = (l - target_len) // 2
    else:
        start = qrs_index(x) - target_len // 2
        start = max(0, min(start, l - target_len))
    return x[:, start:start + target_len]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_ecg(x: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Normalize one record ``(leads, L)``.

    ``'zscore'`` per-lead zero mean / unit variance (removes the lead-to-lead
    amplitude spread); ``'none'`` leaves the values alone. Both raw sources are
    already in a physical unit by this point -- mV for WFDB, or ADC counts scaled
    by ``med_gain`` for ``.med`` -- so no third mode is needed here.
    """
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x
    if mode != "zscore":
        raise ValueError(f"unknown normalization: {mode!r}")
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.where(sd < 1e-6, np.float32(1.0), sd)


# ---------------------------------------------------------------------------
# Labels csv
# ---------------------------------------------------------------------------

def _parse_label(raw, path, rid) -> int:
    """Coerce a label cell to 0/1, accepting the usual boolean spellings."""
    text = str(raw).strip().lower()
    try:
        return int(float(text))
    except ValueError:
        pass
    if text in ("true", "yes", "y", "阳性"):
        return 1
    if text in ("false", "no", "n", "阴性"):
        return 0
    raise SystemExit(f"{path}: cannot read label {raw!r} for record {rid!r} as 0/1")


def load_labels_csv(path: str | Path) -> Dict[str, int]:
    """Read an ``id,label`` csv -> ``{record_id: int}``. Column names are matched loosely."""
    import csv

    out: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        id_col = next((cols[c] for c in _ID_COLUMNS if c in cols), None)
        label_col = next((cols[c] for c in _LABEL_COLUMNS if c in cols), None)
        if id_col is None or label_col is None:
            raise SystemExit(
                f"{path}: need an id column ({'/'.join(_ID_COLUMNS)}) and a label column "
                f"({'/'.join(_LABEL_COLUMNS)}); found {reader.fieldnames}"
            )
        for row in reader:
            rid, lab = row.get(id_col), row.get(label_col)
            if rid is None or lab is None or str(lab).strip() == "":
                continue
            rid = str(rid).strip()
            for suffix in RECORD_SUFFIXES:
                if rid.lower().endswith(suffix):
                    rid = rid[: -len(suffix)]
                    break
            out[rid] = _parse_label(lab, path, rid)
    if not out:
        raise SystemExit(f"{path}: no usable rows")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_ecg_source(
    path: str | Path,
    num_leads: int = 12,
    target_len: Optional[int] = None,
    align: str = "qrs",
    norm: str = "zscore",
    med_gain: float = 1000.0,
    prefer: Sequence[str] = ("med", "dat"),
    labels_csv: Optional[str | Path] = None,
    fs: Optional[float] = None,
) -> Dict:
    """Load ECG records from a directory of raw records.

    Returns ``{"ecg": (N, leads, L) float32, "patient_ids": (N,) str,
    "labels": (N,) int64 or None, "source": str}``.

    The directory is scanned for ``.med`` / ``.dat``+``.hea`` records, each read,
    length-fitted and normalized. Labels come from ``labels_csv`` when given --
    raw record files have no labels of their own.
    """
    p = Path(path)

    if not p.is_dir():
        raise SystemExit(
            f"{p}: expected a directory of .med / .dat records"
        )

    records = scan_ecg_dir(p, prefer=prefer)
    signals, ids = [], []
    failed: List[str] = []

    for rec in records:
        try:
            if rec["kind"] == "med":
                x = read_med(rec["path"], num_leads=num_leads)
                if norm == "mv":
                    x = x / float(med_gain)      # .med has no header, so no gain
            else:
                x, hdr_fs = read_wfdb(rec["path"], num_leads=num_leads)
                if fs is not None and abs(hdr_fs - fs) > 1e-6:
                    print(f"[ECG] Warning: {rec['id']} header sampling rate is {hdr_fs} Hz, "
                          f"expected {fs} Hz")
            if x.shape[0] != num_leads:
                raise SystemExit(f"{rec['id']}: {x.shape[0]} leads, expected {num_leads}")
            x = fit_length(x, target_len, align=align)
            # z-score is applied after length fitting; 'mv' was already applied above.
            if norm == "zscore":
                x = normalize_ecg(x, "zscore")
        except (ValueError, SystemExit) as exc:
            failed.append(f"{rec['id']}: {exc}")
            continue
        signals.append(x)
        ids.append(rec["id"])

    if failed:
        print(f"[ECG] {len(failed)} record(s) failed to load and were dropped:")
        for msg in failed[:5]:
            print(f"      {msg}")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")
    if not signals:
        raise SystemExit(f"{p}: every record failed to load")

    lengths = {s.shape[-1] for s in signals}
    if len(lengths) != 1:
        raise SystemExit(
            f"{p}: records have differing lengths {sorted(lengths)}; pass a fixed --ecg-len "
            f"to make them stackable"
        )

    ecg = np.stack(signals).astype(np.float32)
    labels = None
    if labels_csv is not None:
        table = load_labels_csv(labels_csv)
        missing = [i for i in ids if i not in table]
        if missing:
            print(f"[ECG] {len(missing)} record(s) have no label in {labels_csv} "
                  f"(e.g. {', '.join(missing[:5])})")
        keep = [i for i in ids if i in table]
        if not keep:
            raise SystemExit(f"{labels_csv}: none of the {len(ids)} record ids appear in the labels csv")
        index = {rid: k for k, rid in enumerate(ids)}
        sel = np.array([index[i] for i in keep])
        ecg = ecg[sel]
        ids = keep
        labels = np.array([table[i] for i in keep], dtype=np.int64)
        print(f"[ECG] Labels matched for {len(keep)}/{len(records)} record(s)")

    print(f"[ECG] Loaded {len(ids)} record(s) from {p}: shape={ecg.shape}, "
          f"norm={norm}, target_len={target_len or 'source'}")
    return {
        "ecg": ecg,
        "patient_ids": np.asarray(ids),
        "labels": labels,
        "source": f"{p}",
    }


def load_ecg_from_cfg(cfg, path: Optional[str] = None, use_labels: bool = True) -> Dict:
    """``load_ecg_source`` driven by a training script's argparse namespace.

    Both entry points share the same ``--ecg-*`` flags, so this keeps them in
    sync. ``--seq-len`` doubles as the target length: it is the model's input
    length, and it is what the record is cropped/padded to.
    """
    prefer = getattr(cfg, "ecg_prefer", "med,dat")
    if isinstance(prefer, str):
        prefer = tuple(s.strip().lower() for s in prefer.split(",") if s.strip())
    target_len = getattr(cfg, "seq_len", 500) or None
    return load_ecg_source(
        path if path is not None else cfg.data,
        num_leads=getattr(cfg, "num_leads", 12),
        target_len=target_len,
        align=getattr(cfg, "ecg_align", "qrs"),
        norm=getattr(cfg, "ecg_norm", "zscore"),
        med_gain=getattr(cfg, "ecg_med_gain", 1000.0),
        prefer=prefer,
        labels_csv=(getattr(cfg, "labels", None) if use_labels else None),
    )
