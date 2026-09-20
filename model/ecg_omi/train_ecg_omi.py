"""Single-modal ECG training entry point (M_ECG): pretrain → fine-tune → threshold → evaluate.

Usage:
    python model/ecg_omi/train_ecg_omi.py \
        --data Data/med_data --labels Data/labels.csv \
        --pretrain-data Data/unlabeled_med_data \
        --out runs/ecg_omi


"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution via `python model/ecg_omi/train_ecg_omi.py`:
# insert the repo root into sys.path so the `model` package can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.common.data import split_data
from model.common.ecg_io import load_ecg_from_cfg
from model.common.metrics import set_seed, delong_auc_ci, confusion_metrics
from model.common.training import calibrate_threshold, load_weights, run_finetuning
from model.ecg_omi.data import ECGDataset
from model.ecg_omi.training import (
    predict_proba,
    run_cross_validation,
    run_pretraining,
    _build_model,
)

# Force UTF-8 for the Windows console so log messages print correctly
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-modal ECG model (M_ECG) training: pretrain -> fine-tune -> threshold -> evaluate")
    p.add_argument("--data", type=str, default=None,
                   help="labeled OMI records: a directory of .med, or .dat+.hea files")
    p.add_argument("--labels", type=str, default=None,
                   help="labels csv (id,label); required, since the record files carry no labels")
    p.add_argument("--pretrain-data", type=str, default=None,
                   help="unlabeled ECG records: a directory of .med, or .dat+.hea files")
    p.add_argument("--out", type=str, default="runs/ecg_omi", help="output directory")
    p.add_argument("--no-pretrain", action="store_true", help="skip self-supervised pretraining")
    p.add_argument("--pretrain-ckpt", type=str, default=None, help="path to existing pretrained weights")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-leads", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=500,
                   help="samples per lead fed to the encoder; raw records are cropped/padded to "
                        "this length. Default 500 = the paper's 500-sample (1 s at 500 Hz) median "
                        "beat; 0 keeps the source length (500 for .med, 5000 for .dat)")

    # Raw-record reading (only used when --data is a directory)
    p.add_argument("--ecg-align", type=str, default="qrs", choices=["qrs", "center", "start"],
                   help="where to centre the window when cropping to --seq-len")
    p.add_argument("--ecg-norm", type=str, default="zscore", choices=["zscore", "mv", "none"],
                   help="per-record normalization when reading raw records")
    p.add_argument("--ecg-med-gain", type=float, default=1000.0,
                   help="ADC units per mV for .med files (they carry no header); used by --ecg-norm mv")
    p.add_argument("--ecg-prefer", type=str, default="med,dat",
                   help="which record kind wins when one id has several files")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--mlp-layers", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-pretrain", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--cv-epochs", type=int, default=15,
                   help="max epochs per cross-validation fold (paper Table 2: CV Max Epochs = 15; "
                        "the final fine-tune uses --epochs)")
    p.add_argument("--cv-patience", type=int, default=5,
                   help="early-stop patience per cross-validation fold (paper Table 2: "
                        "CV Early-stop Patience = 5; the final fine-tune uses --patience)")
    p.add_argument("--lr-ecg", type=float, default=2e-5)
    p.add_argument("--lr-pretrain", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cv-folds", type=int, default=5, help="0 to skip cross-validation")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = args

    set_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"[Config] device={device}, seed={cfg.seed}, out={out_dir}")

    # ---- 1. Data loading (.med / .dat+.hea record directory) ----
    if not cfg.data:
        raise SystemExit("Provide --data pointing to a directory of ECG records.")
    data = load_ecg_from_cfg(cfg)
    ecg, labels = data["ecg"], data["labels"]
    patient_ids = data.get("patient_ids")
    if labels is None:
        raise SystemExit(
            "The ECG source carries no labels. The record files have none of their own, "
            "so pass --labels <csv with an id column and a label column>."
        )
    unlabeled_ecg = None
    if not cfg.no_pretrain:
        if cfg.pretrain_ckpt:
            print(f"[Data] Using existing pretrained weights: {cfg.pretrain_ckpt}")
        elif cfg.pretrain_data:
            unlabeled_ecg = load_ecg_from_cfg(cfg, cfg.pretrain_data, use_labels=False)["ecg"]
        else:
            print("[Data] No --pretrain-data provided; skipping pretraining (random init).")
    if ecg.ndim != 3:
        raise SystemExit(f"ecg must be a 3D array (N, leads, L), got {ecg.shape}")
    print(f"[Data] N={len(ecg)}, shape={ecg.shape}, OMI prevalence={labels.mean():.3f}")

    # ---- 2. Splitting: train / internal validation / held-out test (patient-level) ----
    train_idx, val_idx, test_idx = split_data(patient_ids, labels,
                                              test_frac=0.1, val_frac=0.1, seed=cfg.seed)
    ecg_tr, y_tr = ecg[train_idx], labels[train_idx]
    ecg_va, y_va = ecg[val_idx], labels[val_idx]
    ecg_te, y_te = ecg[test_idx], labels[test_idx]
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(test OMI prevalence={y_te.mean():.3f})")

    # ---- 3. Self-supervised pretraining (optional) ----
    pretrain_ckpt = cfg.pretrain_ckpt
    if not cfg.no_pretrain and pretrain_ckpt is None and unlabeled_ecg is not None:
        pretrain_ckpt = run_pretraining(unlabeled_ecg, cfg, device, out_dir)
    elif pretrain_ckpt is not None:
        pretrain_ckpt = str(Path(pretrain_ckpt))

    # ---- 4. Cross-validation (optional; used by the paper for hyper-parameter search) ----
    if cfg.cv_folds > 0:
        run_cross_validation(ecg_tr, y_tr, cfg, device, pretrain_ckpt,
                             cv_epochs=cfg.cv_epochs, cv_patience=cfg.cv_patience)

    # ---- 5. Final fine-tuning: internal training set + early stop on validation set ----
    model = _build_model(cfg, device)
    if pretrain_ckpt is not None:
        model.encoder.load_state_dict(load_weights(pretrain_ckpt, device))
        print("[Finetune] Encoder initialized from pretrained weights")
    tr_loader = DataLoader(ECGDataset(ecg_tr, y_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(ECGDataset(ecg_va, y_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    best_auc, _ = run_finetuning(model, tr_loader, va_loader, cfg, device, tag="final")
    print(f"[Finetune] Best validation AUROC = {best_auc:.4f}")

    # ---- 6. Decision-threshold calibration (maximize F1 on the validation set) ----
    y_va_all, p_va_all = predict_proba(model, va_loader, device)
    threshold = calibrate_threshold(y_va_all, p_va_all)

    # ---- 7. Evaluation on the held-out test set ----
    te_loader = DataLoader(ECGDataset(ecg_te, y_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    y_test, p_test = predict_proba(model, te_loader, device)

    auc, lo, hi = delong_auc_ci(y_test, p_test)
    y_pred = (p_test >= threshold).astype(int)
    cm = confusion_metrics(y_test, y_pred)

    print("\n=============== Evaluation on the held-out test set ===============")
    print(f"AUROC          = {auc:.4f}  (95% CI: {lo:.4f}-{hi:.4f}, DeLong)")
    for k, v in cm.items():
        print(f"{k:<13} = {v:.4f}")
    print(f"threshold(F1)  = {threshold:.4f}")

    results = {
        "seed": cfg.seed,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "auc": auc, "auc_ci": [lo, hi],
        "threshold": threshold,
        "val_auc_best": best_auc,
        "metrics_at_threshold": cm,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    model_ckpt = out_dir / "ecg_omi_model.pt"
    torch.save({"model": model.state_dict(), "threshold": threshold, "cfg": vars(cfg)}, model_ckpt)
    print(f"[Save] Model -> {model_ckpt}")
    print(f"[Save] Metrics -> {out_dir / 'metrics.json'}")

    # To compare the AUROC of two models on the same test set with DeLong's test,
    # pass the predictions of both models:
    # z, p, auc1, auc2 = delong_test(y_test, p_model_a, p_model_b)


if __name__ == "__main__":
    main()
