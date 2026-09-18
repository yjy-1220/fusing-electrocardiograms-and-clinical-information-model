"""General-purpose data utilities: ECG .npz loading and (patient-level) stratified splitting.

Extracted from the duplicated code in the original train_ecg_omi.py and
train_multimodal_omi.py.
"""
from typing import List, Tuple

import numpy as np


def load_npz(path: str):
    """Load a .npz file: returns dict{ecg, labels, patient_ids?}.

    patient_ids is returned with its original dtype (int / float / str all
    supported); downstream split_data only relies on its grouping info.
    """
    data = np.load(path, allow_pickle=False)
    ecg = np.asarray(data["ecg"], dtype=np.float32)
    out = {"ecg": ecg}
    if "labels" in data.files:
        out["labels"] = np.asarray(data["labels"], dtype=np.int64)
    if "patient_ids" in data.files:
        out["patient_ids"] = np.asarray(data["patient_ids"])
    return out


def _stratified_split(labels: np.ndarray, frac: float, seed: int):
    """Stratified record-level sampling: returns (kept indices, drawn indices).
    When a class has more than one sample, at least one sample of that class is
    always kept in the kept (training) fold."""
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _draw(n, size):
        if n <= 1:
            return 0
        return min(max(1, int(round(n * size))), n - 1)

    n_pos_out = _draw(len(pos), frac)
    n_neg_out = _draw(len(neg), frac)
    out = np.concatenate([pos[:n_pos_out], neg[:n_neg_out]])
    keep = np.setdiff1d(np.arange(len(labels)), out)
    return keep, out


def split_data(patient_keys, labels, test_frac: float, val_frac: float, seed: int):
    """
    Paper-style stratified splitting:
      Total data -> 9:1 training set / held-out test set
      Training set -> 9:1 internal training set / internal validation set
    When patient_keys are provided, stratification is done per patient
    (all records of one patient stay in the same split) to avoid data leakage.
    Returns (train_idx, val_idx, test_idx).
    """
    if patient_keys is None:
        train_idx, test_idx = _stratified_split(labels, test_frac, seed)
        train_labels = labels[train_idx]
        train2_idx, val_idx = _stratified_split(train_labels, val_frac / (1.0 - test_frac), seed + 1)
        train_idx = train_idx[train2_idx]
        return train_idx, val_idx, test_idx

    # Patient level: label a patient as positive if any of its records is positive
    unique_patients, inverse = np.unique(patient_keys, return_inverse=True)
    patient_label = np.zeros(len(unique_patients), dtype=np.int64)
    np.maximum.at(patient_label, inverse, labels)

    pat_train, pat_test = _stratified_split(patient_label, test_frac, seed)
    pat_train_labels = patient_label[pat_train]
    pat_train2, pat_val = _stratified_split(pat_train_labels, val_frac / (1.0 - test_frac), seed + 1)
    # pat_train2 / pat_val are local indices into pat_train; map both back to
    # global patient indices (val must be mapped BEFORE pat_train is reindexed).
    pat_val = pat_train[pat_val]
    pat_train = pat_train[pat_train2]

    def _to_records(pat_idx):
        return np.where(np.isin(inverse, pat_idx))[0]

    return _to_records(pat_train), _to_records(pat_val), _to_records(pat_test)


def stratified_kfold_indices(labels: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold (positive/negative split evenly): returns
    [(train_idx, val_idx), ...]."""
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_folds = np.array_split(pos, n_splits)
    neg_folds = np.array_split(neg, n_splits)
    folds = []
    for k in range(n_splits):
        val = np.concatenate([pos_folds[k], neg_folds[k]])
        train = np.setdiff1d(np.arange(len(labels)), val)
        folds.append((train, val))
    return folds
