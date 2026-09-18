"""General-purpose evaluation metrics: random seed / AUROC / confusion matrix metrics / DeLong test and confidence interval.


"""
import math
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix the random seed (Python / NumPy / PyTorch, including CUDA)."""
    import torch  

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Mean ranks (handles ties) without depending on scipy."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auroc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, equivalent to the normalized Mann-Whitney U statistic."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary metrics from predictions: sensitivity/specificity/PPV/NPV/F1/accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    def _safe(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "ppv": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "f1": _safe(2 * tp, 2 * tp + fp + fn),
        "accuracy": _safe(tp + tn, tp + fp + fn + tn),
    }


# ---------------------------------------------------------------------------
# DeLong test and confidence interval (placement-based structure components,
# handles ties)
# ---------------------------------------------------------------------------

def _delong_placements(y_true: np.ndarray, scores: np.ndarray):
    """
    Returns (auc, theta10, theta01, idx_pos, idx_neg)
      theta10[i] = probability that the i-th positive score exceeds a random
                   negative score (half ties)
      theta01[j] = probability that the j-th negative score is below a random
                   positive score (half ties)
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    m = len(idx_pos)
    n = len(idx_neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes to be present")
    s_pos = scores[idx_pos]
    s_neg = scores[idx_neg]
    theta10 = np.zeros(m, dtype=np.float64)
    theta01 = np.zeros(n, dtype=np.float64)
    for i in range(m):
        gt = s_pos[i] > s_neg
        eq = s_pos[i] == s_neg
        theta10[i] = gt.mean() + 0.5 * eq.mean()
    for j in range(n):
        lt = s_neg[j] < s_pos
        eq = s_neg[j] == s_pos
        theta01[j] = lt.mean() + 0.5 * eq.mean()
    auc = float(theta10.mean())
    return auc, theta10, theta01, idx_pos, idx_neg


def _normal_ppf(p: float) -> float:
    """Standard normal quantile (Acklam rational approximation, error < 1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (erf-based)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delong_auc_ci(y_true: np.ndarray, scores: np.ndarray, alpha: float = 0.05):
    """AUROC with a DeLong confidence interval (95% by default).
    Returns (auc, lower, upper)."""
    auc, theta10, theta01, _, _ = _delong_placements(y_true, scores)
    m = len(theta10)
    n = len(theta01)
    v10 = float(theta10.var(ddof=1)) if m > 1 else 0.0
    v01 = float(theta01.var(ddof=1)) if n > 1 else 0.0
    var_auc = v10 / m + v01 / n
    z = _normal_ppf(1.0 - alpha / 2.0)
    half = math.sqrt(max(var_auc, 0.0)) * z
    return auc, auc - half, auc + half


def delong_test(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray):
    """DeLong test for the AUROC difference between two correlated predictions.
    Returns (z, p_value, auc1, auc2)."""
    auc1, t10_1, t01_1, _, _ = _delong_placements(y_true, scores1)
    auc2, t10_2, t01_2, _, _ = _delong_placements(y_true, scores2)
    m = len(t10_1)
    n = len(t01_1)
    v10_1 = float(t10_1.var(ddof=1)) if m > 1 else 0.0
    v01_1 = float(t01_1.var(ddof=1)) if n > 1 else 0.0
    v10_2 = float(t10_2.var(ddof=1)) if m > 1 else 0.0
    v01_2 = float(t01_2.var(ddof=1)) if n > 1 else 0.0
    cov10 = float(np.cov(t10_1, t10_2, ddof=1)[0, 1]) if m > 1 else 0.0
    cov01 = float(np.cov(t01_1, t01_2, ddof=1)[0, 1]) if n > 1 else 0.0
    var_diff = (v10_1 + v10_2 - 2 * cov10) / m + (v01_1 + v01_2 - 2 * cov01) / n
    se = math.sqrt(max(var_diff, 0.0))
    z = (auc1 - auc2) / se if se > 0 else float("nan")
    p = 2.0 * (1.0 - _normal_cdf(abs(z))) if not math.isnan(z) else float("nan")
    return z, p, auc1, auc2
