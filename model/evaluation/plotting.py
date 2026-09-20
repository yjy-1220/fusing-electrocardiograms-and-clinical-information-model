"""Plotting: ROC / PR / confusion matrix / calibration curve / multi-fold CV ROC summary.

Dependencies: numpy / matplotlib / sklearn
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, brier_score_loss,
)


def plot_save_roc(y_true, y_score, savepath, n_bootstrap: int = 1000, random_state: int = 42):
    """Plot and save an ROC curve with bootstrap 95% CI in the legend. Returns AUROC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    rng = np.random.default_rng(int(random_state))
    boot_aucs = []
    n = y_true.shape[0]
    for _ in range(max(1, int(n_bootstrap))):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if np.unique(y_b).size < 2:
            continue
        s_b = y_score[idx]
        fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
        boot_aucs.append(float(auc(fpr_b, tpr_b)))

    if boot_aucs:
        auc_low, auc_high = np.percentile(boot_aucs, [2.5, 97.5])
    else:
        auc_low, auc_high = roc_auc, roc_auc

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}, {auc_low:.4f}-{auc_high:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    return roc_auc


def plot_save_pr(y_true, y_score, savepath):
    """Plot and save a Precision-Recall curve. Returns PR AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    return pr_auc


def plot_save_confusion(y_true, y_pred, savepath, labels=[0, 1]):
    """Plot and save a confusion matrix. Returns the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    return cm


def plot_save_calibration(y_true, y_score, savepath, n_bins=10):
    """Plot and save a probability calibration curve (with Brier score annotation)."""
    brier = brier_score_loss(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    frac_pos = []
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1]) if i < n_bins - 1 else (y_score >= bins[i])
        if mask.sum() == 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            frac_pos.append(np.nan)
        else:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            frac_pos.append(y_true[mask].mean())
    plt.figure(figsize=(6, 5))
    plt.plot(bin_centers, frac_pos, marker='o', label=f'Empirical (Brier={brier:.3f})')
    plt.plot([0, 1], [0, 1], "--", color="gray", label='Ideal')
    plt.xlabel("Predicted prob")
    plt.ylabel("Observed freq")
    plt.title("Calibration (OOF/Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_save_cv_roc_summary(fold_roc_items, savepath, title="ROC Curves of 5-Fold CV on Training Set"):
    """Plot per-fold ROC curves + mean curve + ±1 std band.

    Args:
        fold_roc_items: list of dicts, each containing keys: fpr, tpr, roc_auc.
        savepath: output path.
        title: plot title.

    Returns:
        dict with mean_auc/std_auc/auc_low/auc_high, or None when no valid input.
    """
    if not fold_roc_items:
        return None

    mean_fpr = np.linspace(0.0, 1.0, 201)
    interp_tprs = []
    aucs = []

    plt.figure(figsize=(6.4, 5.6))
    valid_count = 0

    for item in fold_roc_items:
        fpr = np.asarray(item.get("fpr", []), dtype=float)
        tpr = np.asarray(item.get("tpr", []), dtype=float)
        roc_auc = item.get("roc_auc", None)
        if fpr.size < 2 or tpr.size < 2 or roc_auc is None:
            continue

        valid_count += 1
        aucs.append(float(roc_auc))
        plt.plot(fpr, tpr, lw=1.4, alpha=0.35, label=f"Fold {valid_count} (AUROC = {float(roc_auc):.3f})")

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interp_tprs.append(interp_tpr)

    if not interp_tprs:
        plt.close()
        return None

    interp_tprs = np.asarray(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = interp_tprs.std(axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = float(np.std(aucs)) if aucs else 0.0
    auc_low = max(0.0, mean_auc - std_auc)
    auc_high = min(1.0, mean_auc + std_auc)

    plt.plot(mean_fpr, mean_tpr, color="tab:blue", lw=2.2,
             label=f"Mean (AUROC = {mean_auc:.3f}, {auc_low:.3f}-{auc_high:.3f})")

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="tab:blue", alpha=0.18, label="+- 1 Std. Dev.")

    plt.plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", lw=1.2, label="Chance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

    return {
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "auc_low": float(auc_low),
        "auc_high": float(auc_high),
        "n_folds": int(valid_count),
    }
