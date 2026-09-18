"""Threshold search: static optimal threshold + entropy-based group dynamic thresholds.

Dependencies: numpy / pandas / sklearn
"""
import numpy as np
import pandas as pd
from itertools import combinations, product
from typing import Iterable, List, Tuple, Optional

from sklearn.metrics import f1_score, roc_curve, precision_recall_curve


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Return TP, TN, FP, FN counts for binary labels in {0,1}."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return tp, tn, fp, fn


def score_at_threshold(y_true, y_score, threshold: float, objective: str = "f1") -> float:
    """Evaluate a threshold with objective in {'f1', 'youden_j'}. """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= float(threshold)).astype(int)
    obj = str(objective).lower()
    if obj == "f1":
        return float(f1_score(y_true, pred))
    if obj == "youden_j":
        tp, tn, fp, fn = _binary_confusion_counts(y_true, pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(tpr - fpr)
    raise ValueError(f"Unsupported objective: {objective}")


def extract_positive_probs(probs, pos_label=1):
    """Extract the positive-class (label=1) probabilities. Works with DataFrame / ndarray outputs."""
    if isinstance(probs, pd.DataFrame):
        if pos_label in probs.columns:
            return probs[pos_label].to_numpy()
        return probs.iloc[:, -1].to_numpy()
    probs_arr = np.asarray(probs)
    if probs_arr.ndim == 1:
        return probs_arr
    if probs_arr.ndim == 2:
        return probs_arr[:, -1]
    raise ValueError(f"Unsupported probability output shape: {probs_arr.shape}")


def search_best_threshold(
    y_true,
    y_score,
    thr_grid=None,
    objective: str = "f1",
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    tie_breaker: str = "none",
):
    """Search the decision threshold that maximizes the objective, with precision/recall constraints.

    objective: 'f1' or 'youden_j'
    tie_breaker: 'recall' / 'precision' / 'none'
    """
    if thr_grid is None:
        _, _, thr = precision_recall_curve(y_true, y_score)
        thr_grid = np.concatenate(([0.0], thr, [1.0]))
    # For ROC-style objectives, prefer the threshold candidates from roc_curve (cover all operating points).
    if str(objective).lower() == "youden_j":
        _, _, roc_thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
        roc_thr = np.asarray(roc_thr, dtype=float)
        roc_thr = roc_thr[np.isfinite(roc_thr)]
        if roc_thr.size > 0:
            thr_grid = np.unique(np.clip(np.concatenate((thr_grid, roc_thr)), 0.0, 1.0))

    min_precision = float(min_precision)
    min_recall = float(min_recall)
    tie_breaker = str(tie_breaker).lower()

    best_thr, best_score = 0.5, -1.0
    best_rec, best_pre = -1.0, -1.0
    has_feasible = False

    def _metrics_at_threshold(t: float):
        y_t = np.asarray(y_true).astype(int)
        y_s = np.asarray(y_score, dtype=float)
        pred = (y_s >= float(t)).astype(int)
        tp, tn, fp, fn = _binary_confusion_counts(y_t, pred)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return float(precision), float(recall)

    for t in thr_grid:
        score = score_at_threshold(y_true, y_score, t, objective=objective)
        precision, recall = _metrics_at_threshold(t)
        feasible = (precision + 1e-12 >= min_precision) and (recall + 1e-12 >= min_recall)
        if feasible:
            has_feasible = True

        # If a feasible threshold exists, compare only within the feasible set; otherwise fall back to all candidates.
        if has_feasible and (not feasible):
            continue

        if score > best_score + 1e-12:
            best_score, best_thr = score, t
            best_rec, best_pre = recall, precision
            continue

        if abs(score - best_score) <= 1e-12:
            if tie_breaker == "recall" and recall > best_rec + 1e-12:
                best_score, best_thr = score, t
                best_rec, best_pre = recall, precision
            elif tie_breaker == "precision" and precision > best_pre + 1e-12:
                best_score, best_thr = score, t
                best_rec, best_pre = recall, precision
    return float(best_thr), float(best_score)


# ---------------------------------------------------------------------------
# Group dynamic thresholds (based on the probability entropy prob_entropy)
# ---------------------------------------------------------------------------

def prob_entropy(p: np.ndarray) -> np.ndarray:
    """Binary probability entropy H(p) = -p log p - (1-p) log(1-p), maximum ln(2)≈0.693."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def apply_dynamic_thresholds(
    probs: np.ndarray,
    bins: List[Tuple[float, float, float]],
    default_thr: float = 0.5,
):
    """Dynamic thresholds grouped by entropy. bins = [(lo, hi, thr), ...]; samples falling in (lo, hi] use thr."""
    p = np.asarray(probs, dtype=float)
    ent = prob_entropy(p)
    used = np.full_like(p, fill_value=float(default_thr), dtype=float)
    for lo, hi, thr in bins:
        mask = (ent > float(lo)) & (ent <= float(hi))
        used[mask] = float(thr)
    preds = (p >= used).astype(int)
    return preds, used, ent


def _build_bin_threshold_grid(p_bin: np.ndarray, max_k: int = 50) -> np.ndarray:
    """Build a candidate threshold set from the probabilities in this bin (quantile sampling, at most max_k)."""
    if p_bin.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0.0, 1.0, int(max(2, max_k)))
    cand = np.quantile(p_bin, qs)
    cand = np.unique(np.clip(np.concatenate(([0.0], cand, [1.0])), 0.0, 1.0))
    return cand


def search_best_dynamic_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    bins: List[Tuple[float, float, float]],
    *,
    default_thr: float = 0.5,
    max_thr_per_bin: int = 50,
    method: str = "auto",
    max_bruteforce_states: int = 10000,
    n_coord_passes: int = 3,
    objective: str = "f1",
):
    """Search the best threshold per bin under fixed entropy boundaries (maximizing the given objective).

    Returns:
      - opt_bins: [(lo, hi, thr*), ...] optimized thresholds
      - best_score: global objective score of opt_bins on (y_true, probs)
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    ent = prob_entropy(p)
    bin_masks = []
    for lo, hi, _ in bins:
        bin_masks.append((ent > float(lo)) & (ent <= float(hi)))

    cand_list = []
    for mask in bin_masks:
        cand = _build_bin_threshold_grid(p[mask], max_k=max_thr_per_bin)
        cand = np.unique(np.append(cand, float(default_thr)))
        cand_list.append(cand)

    current_thrs = [float(thr) if len(bins) > i else float(default_thr)
                    for i, (_, _, thr) in enumerate(bins)]

    def eval_objective(thrs: List[float]) -> float:
        used = np.full_like(p, fill_value=float(default_thr), dtype=float)
        for (mask, t) in zip(bin_masks, thrs):
            used[mask] = float(t)
        pred = (p >= used).astype(int)
        obj = str(objective).lower()
        if obj == "f1":
            return float(f1_score(y_true, pred))
        if obj == "youden_j":
            tp, tn, fp, fn = _binary_confusion_counts(y_true, pred)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return float(tpr - fpr)
        raise ValueError(f"Unsupported objective: {objective}")

    total_states = 1
    for c in cand_list:
        total_states *= len(c)
        if total_states > max_bruteforce_states:
            break

    best_thrs = list(current_thrs)
    best_score = eval_objective(best_thrs)

    if total_states <= max_bruteforce_states:
        for combo in product(*cand_list):
            sv = eval_objective(combo)
            if sv > best_score:
                best_score = sv
                best_thrs = list(combo)
    else:
        # Coordinate descent: greedily pick per-bin thresholds that improve the global objective
        for _ in range(max(1, int(n_coord_passes))):
            improved = False
            for i in range(len(cand_list)):
                cand = cand_list[i]
                base_thrs = list(best_thrs)
                local_best = best_score
                local_thr = base_thrs[i]
                for t in cand:
                    base_thrs[i] = float(t)
                    sv = eval_objective(base_thrs)
                    if sv > local_best + 1e-12:
                        local_best = sv
                        local_thr = float(t)
                if local_best > best_score + 1e-12:
                    best_score = local_best
                    best_thrs[i] = local_thr
                    improved = True
            if not improved:
                break

    opt_bins = [(float(lo), float(hi), float(t)) for (lo, hi, _), t in zip(bins, best_thrs)]
    return opt_bins, float(best_score)


def search_best_bins_and_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    initial_bins: Optional[List[Tuple[float, float, float]]] = None,
    n_bins: Optional[int] = None,
    default_thr: float = 0.5,
    max_thr_per_bin: int = 50,
    method: str = "auto",
    boundary_quantiles: Optional[List[float]] = None,
    max_boundary_states: int = 1000,
    min_bin_frac: float = 0.01,
    n_coord_passes: int = 0,
    objective: str = "f1",
):
    """Jointly search entropy boundaries and per-bin thresholds with a fixed bin count to maximize the objective.

    Returns:
      - opt_bins: [(lo, hi, thr*), ...] optimized boundaries and thresholds (last bin's hi uses 9.9 as a placeholder)
      - best_score: global objective score achieved with opt_bins
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    N = p.size
    assert N == y_true.size, "y_true and probs must have the same length"
    assert N > 0, "Empty sample"

    ent = prob_entropy(p)
    max_ent = float(np.log(2.0))  # ~0.693

    if boundary_quantiles is None:
        boundary_quantiles = list(np.linspace(0.1, 0.9, 9))
    boundary_quantiles = [q for q in boundary_quantiles if 0.0 < q < 1.0]

    cand_vals = np.quantile(ent, boundary_quantiles)
    eps = 1e-6
    cand_vals = np.unique(np.clip(cand_vals, eps, max_ent - eps))

    if n_bins is None:
        if initial_bins is not None and len(initial_bins) > 0:
            n_bins = len(initial_bins)
        else:
            n_bins = 3
    n_boundaries = n_bins - 1
    if n_boundaries <= 0:
        # Single bin: degenerate to searching one threshold directly
        solo_bins = [(0.0, 9.9, default_thr)]
        return search_best_dynamic_thresholds(y_true, p, solo_bins, default_thr=default_thr,
                                              max_thr_per_bin=max_thr_per_bin, method=method,
                                              objective=objective)[0:2]

    combos = list(combinations(cand_vals, n_boundaries))

    def _bins_to_bounds(bins_def):
        return tuple(float(b[1]) for b in bins_def[:-1]) if bins_def and len(bins_def) > 1 else tuple()
    init_bounds = _bins_to_bounds(initial_bins) if initial_bins else None
    if init_bounds and init_bounds not in combos:
        combos.append(init_bounds)

    if len(combos) > max_boundary_states > 0:
        step = int(np.ceil(len(combos) / float(max_boundary_states)))
        combos = combos[::max(1, step)]

    best_bins = initial_bins if initial_bins is not None else [
        (0.0,) + (cand_vals[:n_boundaries].tolist() + [9.9])
    ]
    best_score = -1.0

    def _evaluate_with_bounds(bounds_tuple):
        bounds = list(bounds_tuple)
        bounds.sort()
        los = [0.0] + bounds
        his = bounds + [9.9]
        bins_def = [(float(lo), float(hi), float(default_thr)) for lo, hi in zip(los, his)]

        masks = []
        for lo, hi, _ in bins_def:
            masks.append((ent > lo) & (ent <= (hi if hi < 9.9 else 1e9)))
        counts = [int(m.sum()) for m in masks]
        if any(c < max(1, int(np.ceil(min_bin_frac * N))) for c in counts):
            return None, None

        opt_bins, scorev = search_best_dynamic_thresholds(
            y_true, p, bins_def, default_thr=default_thr,
            max_thr_per_bin=max_thr_per_bin, method=method,
            objective=objective
        )
        return opt_bins, scorev

    for comb in combos:
        opt_bins, scorev = _evaluate_with_bounds(comb)
        if scorev is None:
            continue
        if scorev > best_score + 1e-12:
            best_score = scorev
            best_bins = opt_bins

    # Optional: coordinate-wise refinement of the boundaries
    if n_coord_passes and len(cand_vals) >= n_boundaries:
        for _ in range(max(1, int(n_coord_passes))):
            improved = False
            cur_bounds = list(_bins_to_bounds(best_bins))
            if len(cur_bounds) != n_boundaries:
                break
            for i in range(n_boundaries):
                local_best_score = best_score
                local_best_bins = None
                for b in cand_vals:
                    trial_bounds = list(cur_bounds)
                    trial_bounds[i] = float(b)
                    trial_bounds_sorted = sorted(trial_bounds)
                    if tuple(trial_bounds_sorted) == tuple(cur_bounds):
                        continue
                    opt_bins, scorev = _evaluate_with_bounds(trial_bounds_sorted)
                    if scorev is None:
                        continue
                    if scorev > local_best_score + 1e-12:
                        local_best_score = scorev
                        local_best_bins = opt_bins
                if local_best_bins is not None and local_best_score > best_score + 1e-12:
                    best_score = local_best_score
                    best_bins = local_best_bins
                    cur_bounds = list(_bins_to_bounds(best_bins))
                    improved = True
            if not improved:
                break

    return best_bins, float(best_score)
