## Directory structure

```
model/
├── common_utils.py               
├── evaluation/                   # general-purpose evaluation utilities
│   ├── thresholds.py             #   threshold search / group dynamic thresholds
│   ├── risk_stratification.py    #   paper's three-tier dual-threshold Pareto search (5%/75%)
│   ├── plotting.py               #   ROC / PR / confusion matrix / calibration curve plotting
│   └── calibration.py            #   probability calibration (Platt / Isotonic)
├── common/                       # base code shared by both training scripts
│   ├── metrics.py                #   random seed / AUROC / confusion matrix metrics / DeLong test
│   ├── data.py                   #   ECG loading and stratified splitting
│   ├── ecg_io.py                 #   raw .med / .dat+.hea record loading 
│   ├── ecg_encoder.py            #   Wav2Vec-style ECG encoder
│   └── training.py               #   AMP / weight loading / LR scheduling / fine-tuning loop / threshold calibration
├── ecg_omi/
│   ├── data.py                   # ECG Dataset and time-shift augmentation
│   ├── models.py                 # M_ECG model definition (encoder + classification head / SimCLR head)
│   ├── training.py               # pretraining / cross-validation / prediction
│   └── train_ecg_omi.py          # [Entry] single-modal M_ECG: pretrain → fine-tune → evaluate
├── text_process/
│   ├── config.py                 # pipeline configuration
│   ├── models.py                 # document data model
│   ├── extraction.py             # text extraction / QC / per-patient concatenation
│   ├── truncation.py             # length statistics + head-tail truncation
│   ├── pipeline.py               # end-to-end pipeline
│   └── clinical_text_processing.py # [Entry] clinical text extraction/QC + head-tail truncation
└── multimodal_omi/
    ├── data.py                   # ECG/text loading, alignment, tokenization, Dataset
    ├── models.py                 # fusion model definition (ECG + MacBERT + Transformer fusion)
    ├── training.py               # grouped optimizer / training loop / cross-validation / prediction
    └── train_multimodal_omi.py   # [Entry] main pipeline: multimodal fusion training
```

## Pipeline

`model/multimodal_omi/train_multimodal_omi.py` directly loads the **raw 12-lead waveforms** and the **raw clinical text**;
both encoders are initialized from **pretrained model weight files** and fine-tuned end to end:

- **ECG branch**: raw waveform → custom Wav2Vec encoder (5×Conv1d+BN+ReLU → 12-layer Transformer
  → mean pooling → 512 dims), initialized with the weight file produced by pretraining with
  `ecg_omi/train_ecg_omi.py` (`pretrain_encoder.pt`) → 5-layer MLP (GELU, dropout 0.2) → linear projection to 256 dims.
- **Text branch**: Chinese-MacBERT-Base (local weight directory `Data/chinese-macbert-base`) →
  text-CLS (768 dims) → linear projection to 256 dims; head 60% + tail 40% truncated to 512 tokens.
- **Fusion**: `[fusion-CLS, ECG-token, text-token]` 3-token sequence → 5-layer Transformer
  (8-head self-attention, 256 dims, GeGLU FFN 256→512→256, dropout 0.1) → fusion-CLS → single linear classification head.
- **Training**: AdamW (wd=0.01), cosine decay + linear warmup 0.1, batch 128, FP16, BCE;
  ECG branch / fusion / classification head lr=2e-5; MacBERT layerwise LR decay (top layer 1.5e-5, ×0.9 per layer → bottom ≈4.7e-6).
- **Evaluation**: optional 5-fold CV AUROC (the CV stage:
  15 epochs / early-stop patience 5);
  validation-set F1-optimal threshold; held-out test-set AUROC (DeLong 95% CI) + confusion matrix metrics;
  optional three-tier risk stratification (`--risk-strat`, 5%/75% dual-threshold Pareto search).
- `--mode fusion|text|ecg` selects M_multi / M_clin / M_ECG.

### Running

```bash
# 1) (optional, M_ECG) First run SimCLR pretraining on unlabeled ECGs to produce ECG encoder weights
python model/ecg_omi/train_ecg_omi.py \
    --data Data/med_data --labels Data/labels.csv \
    --pretrain-data Data/unlabeled_med_data \
    --out runs/ecg_omi

# 2) Multimodal fusion training
python model/multimodal_omi/train_multimodal_omi.py \
    --data Data/med_data \
    --text Data/text.csv \
    --macbert-path Data/chinese-macbert-base \
    --pretrain-ckpt runs/ecg_omi/pretrain_encoder.pt \
    --out runs/multimodal_omi
```

### Record format

`--data` points at a **directory of raw records**, read directly at load time:

- `.med` — int16, sample-major `(L, leads)` interleaved, no header.
- `.dat` + `.hea` — WFDB format 16; gain/baseline are read from the header and applied, so values come out in mV.
- Records are cropped/padded to `--seq-len` samples per lead (`--ecg-align qrs|center|start` picks the window; the
  default `qrs` centers it on the QRS complex). The default `--seq-len 500` 
  (1 s at 500 Hz) median beat; smaller values crop it QRS-centred without cutting the complex off, and
  `--seq-len 0` keeps the source length.
- `--ecg-norm zscore|mv|none` normalizes each record (`zscore` per lead is the default).
- `--ecg-prefer med,dat` decides which kind wins when one record id has several files.

### Input format

- `--data`: a directory of `.med` records, or of `.dat`+`.hea` records .
- `--labels`: `.csv` containing an id column (`id` / `patient_id` / `filename` / `record` / …) and a label column
  (`label` / `omi` / `y` / `target`); required for the ECG-only entry point, while the multimodal entry point
  falls back to the `label` column of `--text`.
- `--text`: `.csv` containing `filename` (optional, matched against `patient_ids`), `text`, `label`.
- `--macbert-path`: local MacBERT weight directory (containing `pytorch_model.bin`, `config.json`, `vocab.txt`).
- `--pretrain-ckpt`: Wav2Vec ECG encoder pretrained weights (`pretrain_encoder.pt` saved by `train_ecg_omi.py`).

## Shared utilities

The evaluation utilities live in `model/evaluation/` (compatible entry `model/common_utils.py`) and can be reused directly by the evaluation stage of every model:

- **Threshold search** (`thresholds.py`): `search_best_threshold` (F1 / Youden's J, with precision/recall constraints and tie preferences),
  `score_at_threshold`, `extract_positive_probs`
- **Group dynamic thresholds** (`thresholds.py`): `prob_entropy`, `apply_dynamic_thresholds`,
  `search_best_dynamic_thresholds` (search thresholds on fixed entropy boundaries),
  `search_best_bins_and_thresholds` (joint search of boundaries + thresholds)
- **Risk stratification** (`risk_stratification.py`): `search_dual_thresholds` (the paper's dual-threshold
  Pareto-front search with rule-out/rule-in purity, extreme-group coverage constraints and integer rounding),
  `evaluate_risk_tiers`, `group_metrics`
- **Plotting** (`plotting.py`): `plot_save_roc` (bootstrap 95% CI), `plot_save_pr`, `plot_save_confusion`,
  `plot_save_calibration`, `plot_save_cv_roc_summary` (multi-fold mean±std band)
- **Probability calibration** (`calibration.py`): `calibrate_probs` (platt / isotonic / auto / both, auto-selected by metric),
  `apply_calibration`, `fit_platt`, `fit_isotonic`

Shared training code lives in `model/common/`: `metrics.py` (AUROC / DeLong / confusion matrix metrics),
`data.py` (patient-level stratified splitting), `ecg_io.py` (raw `.med` / `.dat`+`.hea`
record loading: header parsing, QRS-centred length fitting, normalization, labels csv),
`ecg_encoder.py` (Wav2Vec ECG encoder), `training.py` (AMP / LR scheduling / fine-tuning loop / threshold calibration).

Example:

```python
from model.common_utils import search_best_threshold, calibrate_probs, apply_calibration

thr, score = search_best_threshold(y_valid, p_valid, objective="f1")
info = calibrate_probs(p_oof, y_oof, method="auto", selection_metric="roc_auc")
p_cal = apply_calibration(p_test, info)
```
