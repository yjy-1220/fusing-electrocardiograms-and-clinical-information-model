"""Evaluation toolkit: threshold search / dynamic thresholds / risk stratification / plotting / probability calibration.



Submodules:
    thresholds.py           static threshold search + entropy-based group dynamic thresholds
    risk_stratification.py  paper's three-tier dual-threshold Pareto search (5%/75%)
    plotting.py             ROC / PR / confusion matrix / calibration curve / CV ROC summary plotting
    calibration.py          Platt / Isotonic probability calibration
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
from model.evaluation.risk_stratification import (
    group_metrics,
    search_dual_thresholds,
    evaluate_risk_tiers,
)
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
    # risk stratification
    "group_metrics", "search_dual_thresholds", "evaluate_risk_tiers",
    # plotting
    "plot_save_roc", "plot_save_pr", "plot_save_confusion",
    "plot_save_calibration", "plot_save_cv_roc_summary",
    # probability calibration
    "fit_platt", "fit_isotonic", "_score_probs", "_pick_best_calibration",
    "calibrate_probs", "apply_calibration",
]
