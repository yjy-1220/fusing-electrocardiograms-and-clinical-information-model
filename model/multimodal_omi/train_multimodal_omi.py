
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution via `python model/multimodal_omi/train_multimodal_omi.py`:
# insert the repo root into sys.path so the `model` package can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.common.data import split_data
from model.common.ecg_io import load_ecg_from_cfg
from model.common.metrics import set_seed, delong_auc_ci, confusion_metrics
from model.common.training import calibrate_threshold, load_weights
from model.evaluation.risk_stratification import evaluate_risk_tiers, search_dual_thresholds
from model.multimodal_omi.data import (
    align_ecg_text,
    collate_multimodal,
    HeadTailTokenizer,
    load_text_csv,
    MultimodalDataset,
)
from model.multimodal_omi.training import (
    build_model,
    predict_proba,
    run_cross_validation,
    run_finetuning_multimodal,
)

# Force UTF-8 for the Windows console so log messages print correctly
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodal OMI model (M_multi): ECG Wav2Vec + MacBERT + "
                    "transformer fusion, paper-consistent pipeline (no embeddings)")
    p.add_argument("--data", type=str, required=True,
                   help="ECG records: a directory of .med, or .dat+.hea files")
    p.add_argument("--labels", type=str, default=None,
                   help="labels csv (id,label); optional, since labels can also come "
                        "from the --text csv")
    p.add_argument("--text", type=str, required=True,
                   help="clinical text .csv (filename[, text, label])")
    p.add_argument("--macbert-path", type=str, default="Data/chinese-macbert-base",
                   help="local MacBERT weight directory (pytorch_model.bin + config + vocab)")
    p.add_argument("--pretrain-ckpt", type=str, default=None,
                   help="pretrained Wav2Vec ECG encoder weight file "
                        "(from ecg_omi/train_ecg_omi.py, e.g. pretrain_encoder.pt)")
    p.add_argument("--mode", type=str, default="fusion",
                   choices=["fusion", "text", "ecg"],
                   help="fusion = M_multi; text = M_clin; ecg = M_ECG (ablation)")
    p.add_argument("--out", type=str, default="runs/multimodal_omi")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # ECG waveform
    p.add_argument("--num-leads", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=500,
                   help="samples per lead fed to the encoder; raw records are cropped/padded to "
                        "this length. Default 500 = the paper's 500-sample (1 s at 500 Hz) median "
                        "beat; 0 keeps the source length (500 for .med, 5000 for .dat)")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)

    # Raw-record reading (only used when --data is a directory)
    p.add_argument("--ecg-align", type=str, default="qrs", choices=["qrs", "center", "start"],
                   help="where to centre the window when cropping to --seq-len")
    p.add_argument("--ecg-norm", type=str, default="zscore", choices=["zscore", "mv", "none"],
                   help="per-record normalization when reading raw records")
    p.add_argument("--ecg-med-gain", type=float, default=1000.0,
                   help="ADC units per mV for .med files (they carry no header); used by --ecg-norm mv")
    p.add_argument("--ecg-prefer", type=str, default="med,dat",
                   help="which record kind wins when one id has several files")

    # ECG MLP / fusion
    p.add_argument("--mlp-layers", type=int, default=5)
    p.add_argument("--fusion-layers", type=int, default=5)

    # Data split (paper: 9:1 train/test, then 9:1 train/valid)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--cv-epochs", type=int, default=15,
                   help="max epochs per cross-validation fold (paper Table 2: CV Max Epochs = 15; "
                        "the final fine-tune uses --epochs)")
    p.add_argument("--cv-patience", type=int, default=5,
                   help="early-stop patience per cross-validation fold (paper Table 2: "
                        "CV Early-stop Patience = 5; the final fine-tune uses --patience)")
    p.add_argument("--lr-common", type=float, default=2e-5,
                   help="LR for ECG branch / fusion / classification head")
    p.add_argument("--lr-text-top", type=float, default=1.5e-5,
                   help="MacBERT top-layer LR (layerwise decay)")
    p.add_argument("--lr-text-bottom", type=float, default=4.7e-6,
                   help="MacBERT bottom-layer / embeddings LR (approx 1.5e-5 * 0.9^11)")
    p.add_argument("--lr-text-decay", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cv-folds", type=int, default=5, help="0 to skip cross-validation")
    p.add_argument("--threshold-objective", type=str, default="f1",
                   choices=["f1", "youden_j"])
    p.add_argument("--risk-strat", action="store_true",
                   help="run the paper's three-tier risk stratification (Section 2.4): "
                        "dual-threshold Pareto search with rule-out/rule-in purity and coverage "
                        "constraints on the internal-validation set, applied to the test set "
                        "(final scheme e.g. low-risk <=5% / high-risk >=75%)")
    return p.parse_args()


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"[Config] mode={cfg.mode}, device={device}, seed={cfg.seed}, out={out_dir}")
    print(f"[Config] MacBERT={cfg.macbert_path}, pretrain_ckpt={cfg.pretrain_ckpt}")

    # ---- 1. Data loading & alignment ----
    ecg_data = load_ecg_from_cfg(cfg)
    text_df, pid_col = load_text_csv(cfg.text)
    ecg, labels, texts, keys = align_ecg_text(ecg_data, text_df, pid_col)
    print(f"[Data] N={len(ecg)}, ecg shape={ecg.shape}, "
          f"OMI prevalence={labels.mean():.3f}, text mean len={np.mean([len(t) for t in texts]):.0f} chars")

    # ---- 2. Splitting: train / internal validation / held-out test ----
    train_idx, val_idx, test_idx = split_data(keys, labels,
                                              test_frac=cfg.test_frac,
                                              val_frac=cfg.val_frac,
                                              seed=cfg.seed)
    ecg_tr, y_tr = ecg[train_idx], labels[train_idx]
    ecg_va, y_va = ecg[val_idx], labels[val_idx]
    ecg_te, y_te = ecg[test_idx], labels[test_idx]
    text_tr = [texts[i] for i in train_idx]
    text_va = [texts[i] for i in val_idx]
    text_te = [texts[i] for i in test_idx]
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(test OMI prevalence={y_te.mean():.3f})")

    # ---- 3. Tokenizer (MacBERT, head 60% + tail 40%, 512 tokens) ----
    tokenizer = HeadTailTokenizer(cfg.macbert_path, max_tokens=512, head_ratio=0.6)
    # Pre-tokenize once (instead of once per epoch).
    train_ids = [tokenizer.encode(t) for t in text_tr]
    val_ids = [tokenizer.encode(t) for t in text_va]
    test_ids = [tokenizer.encode(t) for t in text_te]
    print(f"[Tokenize] train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # ---- 4. Model & pretrained weights ----
    model = build_model(cfg, device)
    if cfg.pretrain_ckpt and cfg.mode in ("fusion", "ecg"):
        if not Path(cfg.pretrain_ckpt).exists():
            raise SystemExit(f"Pretrained weight file does not exist: {cfg.pretrain_ckpt}")
        model.ecg_encoder.load_state_dict(load_weights(cfg.pretrain_ckpt, device))
        print(f"[Weights] ECG encoder initialized from {cfg.pretrain_ckpt}")
    elif cfg.mode in ("fusion", "ecg"):
        print("[Weights] No --pretrain-ckpt provided; ECG encoder is randomly initialized")

    # ---- 5. Cross-validation (optional; paper reports 5-fold CV AUROC) ----
    cv_auc = None
    if cfg.cv_folds > 0:
        cv_auc = run_cross_validation(ecg_tr, train_ids, y_tr, cfg, device,
                                      cfg.pretrain_ckpt,
                                      cv_epochs=cfg.cv_epochs, cv_patience=cfg.cv_patience)

    # ---- 6. Final fine-tuning: internal train + early stop on validation ----
    tr_loader = DataLoader(
        MultimodalDataset(ecg_tr, train_ids, y_tr, cfg.mode),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_multimodal,
    )
    va_loader = DataLoader(
        MultimodalDataset(ecg_va, val_ids, y_va, cfg.mode),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal,
    )
    te_loader = DataLoader(
        MultimodalDataset(ecg_te, test_ids, y_te, cfg.mode),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal,
    )
    best_auc, history = run_finetuning_multimodal(model, tr_loader, va_loader, cfg, device, tag="final")
    print(f"[Finetune] Best validation AUROC = {best_auc:.4f}")

    # ---- 7. Threshold calibration (F1-maximizing on validation set) ----
    y_va_all, p_va_all = predict_proba(model, va_loader, device)
    threshold = calibrate_threshold(y_va_all, p_va_all, objective=cfg.threshold_objective)

    # ---- 8. Evaluation on the held-out test set ----
    y_test, p_test = predict_proba(model, te_loader, device)
    auc, lo, hi = delong_auc_ci(y_test, p_test)
    y_pred = (p_test >= threshold).astype(int)
    cm = confusion_metrics(y_test, y_pred)

    print("\n=============== Evaluation on the held-out test set ===============")
    print(f"AUROC          = {auc:.4f}  (95% CI: {lo:.4f}-{hi:.4f}, DeLong)")
    for k, v in cm.items():
        print(f"{k:<13} = {v:.4f}")
    print(f"threshold({cfg.threshold_objective}) = {threshold:.4f}")

    # ---- 8b. Optional three-tier risk stratification (paper Section 2.4) ----
    # Thresholds are optimized exclusively on the internal-validation set and
    # then applied to the blinded test set.
    risk = None
    if cfg.risk_strat:
        res = search_dual_thresholds(y_va_all, p_va_all)
        te_risk = evaluate_risk_tiers(y_test, p_test, res["t_low"], res["t_high"])
        risk = {
            "cutoffs": [res["t_low"], res["t_high"]],
            "satisfied_all_constraints": res["satisfied_all"],
            "constraints": res["constraints"],
            "validation": res["metrics"],
            "test": te_risk,
        }
        print("\n=============== Three-tier risk stratification (test set) ===============")
        for k in ("n_low", "n_mid", "n_high"):
            print(f"  {k:<8} = {te_risk[k]}")
        print(f"  low-risk  NPV (rule-out purity)  = {te_risk['npv_low']:.4f}  "
              f"(OMI {te_risk['omi_low']})")
        print(f"  high-risk PPV (rule-in purity)   = {te_risk['ppv_high']:.4f}  "
              f"(OMI {te_risk['omi_high']})")
        print(f"  rule-out sensitivity            = {te_risk['rule_out_sensitivity']:.4f}  "
              f"(missed rate {te_risk['missed_rate']:.4f})")

    # ---- 9. Save model + metrics ----
    results = {
        "seed": cfg.seed,
        "mode": cfg.mode,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "auc": auc, "auc_ci": [lo, hi],
        "cv_auc": cv_auc,
        "threshold": threshold,
        "threshold_objective": cfg.threshold_objective,
        "val_auc_best": best_auc,
        "metrics_at_threshold": cm,
        "risk_stratification": risk,
        "training": {
            "batch_size": cfg.batch_size, "epochs": cfg.epochs, "patience": cfg.patience,
            "lr_common": cfg.lr_common,
            "lr_text_top": cfg.lr_text_top, "lr_text_bottom": cfg.lr_text_bottom,
            "lr_text_decay": cfg.lr_text_decay,
            "weight_decay": cfg.weight_decay, "warmup_frac": cfg.warmup_frac,
            "optimizer": "AdamW", "schedule": "cosine_decay + linear warmup",
            "precision": "FP16" if device.type == "cuda" else "fp32",
            "loss": "binary_cross_entropy",
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    model_ckpt = out_dir / f"multimodal_omi_model_{cfg.mode}.pt"
    torch.save({
        "model": model.state_dict(),
        "threshold": threshold,
        "cfg": vars(cfg),
        "metrics": results,
    }, model_ckpt)
    print(f"[Save] Model -> {model_ckpt}")
    print(f"[Save] Metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
