"""General-purpose evaluation utilities (compatible entry point).

The actual implementation has been split by responsibility into the
model/evaluation/ package:
  - thresholds.py    threshold search / group dynamic thresholds
  - plotting.py      ROC / PR / confusion matrix / calibration curve / CV ROC summary plotting
  - calibration.py   probability calibration (Platt / Isotonic / auto / both)
  - utils.py         stable document ID generation


"""
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
