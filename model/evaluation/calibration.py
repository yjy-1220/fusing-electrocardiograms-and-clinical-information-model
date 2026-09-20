"""Probability calibration: Platt (logistic regression) / Isotonic / auto / both.

Dependencies: numpy / sklearn
"""
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def fit_platt(oof_p, y):
    """Fit Platt calibration (logistic-regression calibrated probabilities). Returns (lr_model, calibrated probabilities)."""
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(oof_p.reshape(-1, 1), y)
    calib_probs = lr.predict_proba(oof_p.reshape(-1, 1))[:, 1]
    return lr, calib_probs


def fit_isotonic(oof_p, y):
    """Fit Isotonic calibration (monotonic-regression calibrated probabilities). Returns (iso_model, calibrated probabilities)."""
    iso_model = IsotonicRegression(out_of_bounds='clip')
    iso_model.fit(oof_p, y)
    calib_probs = iso_model.predict(oof_p)
    return iso_model, calib_probs


def _score_probs(y_true, probs, metric: str, joint_pr_weight: float = 0.6, joint_roc_weight: float = 0.4):
    """Score probabilities by the requested metric. Returns (score, higher_is_better)."""
    metric = str(metric).lower()
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, probs)), True
    if metric in ("pr_auc", "average_precision"):
        return float(average_precision_score(y_true, probs)), True
    if metric in ("joint_roc_pr", "joint_pr_roc", "joint"):
        roc = float(roc_auc_score(y_true, probs))
        pr = float(average_precision_score(y_true, probs))
        score = float(joint_pr_weight * pr + joint_roc_weight * roc)
        return score, True
    if metric == "log_loss":
        return float(log_loss(y_true, probs)), False
    raise ValueError(f"Unknown calibration selection metric: {metric}")


def _pick_best_calibration(y_true, candidates, selection_metric: str = "roc_auc",
                           joint_pr_weight: float = 0.6, joint_roc_weight: float = 0.4):
    """Pick the best candidate by the selection metric."""
    metric = str(selection_metric).lower()
    if metric in ("", "auto", "eval_metric"):
        metric = "roc_auc"
    best_name = None
    best_probs = None
    best_score = None
    scores = {}
    higher_is_better = True

    for name, probs in candidates.items():
        if probs is None:
            continue
        score, higher_is_better = _score_probs(y_true, probs, metric,
                                               joint_pr_weight, joint_roc_weight)
        scores[name] = float(score)
        if best_name is None:
            best_name, best_probs, best_score = name, probs, score
            continue
        if higher_is_better and score > best_score + 1e-12:
            best_name, best_probs, best_score = name, probs, score
        elif (not higher_is_better) and score < best_score - 1e-12:
            best_name, best_probs, best_score = name, probs, score

    if best_name is None:
        raise ValueError("No calibration candidate available")
    return best_name, best_probs, float(best_score), metric, scores


def calibrate_probs(oof_probs, y_oof, method: str = "auto",
                    selection_metric: str = "roc_auc",
                    joint_pr_weight: float = 0.6, joint_roc_weight: float = 0.4):
    """Fit calibration models on out-of-fold (OOF) probabilities.

    Parameters
    ----------
    oof_probs : array
        Raw positive-class probabilities on the train/validation set (used to fit the calibrator).
    y_oof : array
        Corresponding labels.
    method : str
        "platt" / "isotonic" / "auto" (auto-selection including the raw candidate) / "both" (fit both and select).
    selection_metric : str
        "roc_auc" / "pr_auc" / "log_loss" / "joint_roc_pr" / "auto".
    joint_pr_weight, joint_roc_weight : float
        Only used when selection_metric="joint_roc_pr".

    Returns
    -------
    dict with oof_calib, lr_cal, iso, calib_method, used_calib_for_thr, ...
    """
    oof_calib = None
    oof_calib_platt = None
    oof_calib_iso = None
    lr_cal = None
    iso = None
    used_calib_for_thr = None
    calib_selection_metric = selection_metric
    calib_selection_scores = {}
    selected_calib_score = None

    def _score(probs):
        return _score_probs(y_oof, probs, calib_selection_metric,
                            joint_pr_weight, joint_roc_weight)[0]

    if method == "platt":
        lr_cal, oof_calib_platt = fit_platt(oof_probs, y_oof)
        oof_calib = oof_calib_platt
        used_calib_for_thr = "platt"
        selected_calib_score = _score(oof_calib)
        calib_selection_scores = {"platt": float(selected_calib_score)}
        print("Using Platt calibration (logistic regression)")
    elif method == "isotonic":
        try:
            iso, oof_calib_iso = fit_isotonic(oof_probs, y_oof)
            oof_calib = oof_calib_iso
            used_calib_for_thr = "isotonic"
            selected_calib_score = _score(oof_calib)
            calib_selection_scores = {"isotonic": float(selected_calib_score)}
            print("Using Isotonic calibration (monotonic regression)")
        except Exception as e:
            print(f"Isotonic calibration failed, using Platt fallback: {e}")
            lr_cal, oof_calib_platt = fit_platt(oof_probs, y_oof)
            oof_calib = oof_calib_platt
            used_calib_for_thr = "platt"
            selected_calib_score = _score(oof_calib)
            calib_selection_scores = {"platt": float(selected_calib_score)}
    elif method == "auto":
        # Include raw as a candidate: when calibration hurts ranking, auto-fallback to raw.
        oof_raw = np.asarray(oof_probs, dtype=float)
        try:
            iso, oof_calib_iso = fit_isotonic(oof_probs, y_oof)
            print("Auto mode: fitting Isotonic calibration")
        except Exception as e:
            print(f"Auto mode: Isotonic failed: {e}")
            oof_calib_iso = None
        try:
            lr_cal, oof_calib_platt = fit_platt(oof_probs, y_oof)
            print("Auto mode: fitting Platt calibration")
        except Exception as e:
            print(f"Auto mode: Platt failed: {e}")
            oof_calib_platt = None

        used_calib_for_thr, oof_calib, selected_calib_score, calib_selection_metric, calib_selection_scores = _pick_best_calibration(
            y_oof,
            {"raw": oof_raw, "platt": oof_calib_platt, "isotonic": oof_calib_iso},
            selection_metric, joint_pr_weight, joint_roc_weight,
        )
        print(f"Auto mode: selected {used_calib_for_thr} by {calib_selection_metric}, score={selected_calib_score:.6f}")
    elif method == "both":
        oof_raw = np.asarray(oof_probs, dtype=float)
        try:
            iso, oof_calib_iso = fit_isotonic(oof_probs, y_oof)
            print("Both mode: fitting Isotonic calibration")
        except Exception as e:
            print(f"Both mode: Isotonic failed: {e}")
            oof_calib_iso = None
        lr_cal, oof_calib_platt = fit_platt(oof_probs, y_oof)
        print("Both mode: fitting Platt calibration")
        used_calib_for_thr, oof_calib, selected_calib_score, calib_selection_metric, calib_selection_scores = _pick_best_calibration(
            y_oof,
            {"raw": oof_raw, "platt": oof_calib_platt, "isotonic": oof_calib_iso},
            selection_metric, joint_pr_weight, joint_roc_weight,
        )
        print(f"Both mode: selected {used_calib_for_thr} by {calib_selection_metric}, score={selected_calib_score:.6f}")
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return {
        "oof_calib": oof_calib,
        "oof_calib_platt": oof_calib_platt,
        "oof_calib_iso": oof_calib_iso,
        "lr_cal": lr_cal,
        "iso": iso,
        "calib_method": method,
        "used_calib_for_thr": used_calib_for_thr,
        "calib_selection_metric": calib_selection_metric,
        "calib_selection_scores": calib_selection_scores,
        "selected_calib_score": selected_calib_score,
    }


def apply_calibration(raw_probs, calib_info):
    """Apply the fitted calibration model to new probabilities."""
    method = calib_info["calib_method"]
    used_thr = calib_info["used_calib_for_thr"]
    lr_cal = calib_info["lr_cal"]
    iso = calib_info["iso"]

    if method == "platt":
        return lr_cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        return iso.predict(raw_probs)
    elif method == "auto":
        if used_thr == "raw":
            return raw_probs
        if used_thr == "isotonic":
            return iso.predict(raw_probs)
        else:
            return lr_cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    elif method == "both":
        if used_thr == "raw":
            return raw_probs
        if used_thr == "isotonic" and iso is not None:
            return iso.predict(raw_probs)
        else:
            return lr_cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
