"""Evaluation toolkit: threshold search / dynamic thresholds / plotting / probability calibration.



Submodules:
    thresholds.py   static threshold search + entropy-based group dynamic thresholds
    plotting.py     ROC / PR / confusion matrix / calibration curve / CV ROC summary plotting
    calibration.py  Platt / Isotonic probability calibration
    utils.py        stable document ID generation
"""
import warnings


warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model.evaluation.thresholds import (
    _binary_confusion_counts,
    score_at_threshold,
    extract_positive_probs,
    search_best_threshold,
    prob_entropy,
    apply_dynamic_thresholds,
    _build_bin_threshold_grid,
    search_best_dynamic_thresholds,
    search_best_bins_and_thresholds,
)
from model.evaluation.utils import ensure_doc_id
from model.evaluation.plotting import (
    plot_save_roc,
    plot_save_pr,
    plot_save_confusion,
    plot_save_calibration,
    plot_save_cv_roc_summary,
)
from model.evaluation.calibration import (
    fit_platt,
    fit_isotonic,
    _score_probs,
    _pick_best_calibration,
    calibrate_probs,
    apply_calibration,
)

__all__ = [
    # threshold search
    "_binary_confusion_counts", "score_at_threshold", "extract_positive_probs",
    "search_best_threshold",
    # group dynamic thresholds
    "prob_entropy", "apply_dynamic_thresholds", "_build_bin_threshold_grid",
    "search_best_dynamic_thresholds", "search_best_bins_and_thresholds",
    # plotting
    "plot_save_roc", "plot_save_pr", "plot_save_confusion",
    "plot_save_calibration", "plot_save_cv_roc_summary",
    # probability calibration
    "fit_platt", "fit_isotonic", "_score_probs", "_pick_best_calibration",
    "calibrate_probs", "apply_calibration",
    # other
    "ensure_doc_id",
]
