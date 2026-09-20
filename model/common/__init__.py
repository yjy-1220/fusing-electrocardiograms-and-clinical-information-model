"""Base code package shared by the two training scripts.

    metrics.py      random seed / AUROC / confusion matrix metrics / DeLong test
    data.py         patient-level stratified splitting
    ecg_io.py       raw .med / .dat+.hea record loading (directory sources)
    ecg_encoder.py  Wav2Vec-style ECG encoder (Conv stack + Transformer)
    training.py     AMP / weight loading / LR scheduling / fine-tuning loop / threshold calibration
"""
from model.common.metrics import (
    set_seed,
    _rankdata,
    auroc_score,
    confusion_metrics,
    _delong_placements,
    _normal_ppf,
    _normal_cdf,
    delong_auc_ci,
    delong_test,
)
from model.common.data import (
    _stratified_split,
    split_data,
    stratified_kfold_indices,
)
from model.common.ecg_io import (
    read_med,
    read_wfdb,
    parse_wfdb_header,
    scan_ecg_dir,
    fit_length,
    normalize_ecg,
    load_labels_csv,
    load_ecg_source,
    load_ecg_from_cfg,
)
from model.common.ecg_encoder import (
    ConvBlock,
    ConvFeatureExtractor,
    ECGWav2VecEncoder,
)
from model.common.training import (
    make_amp,
    load_weights,
    WarmupCosineLR,
    GroupedWarmupCosineLR,
    calibrate_threshold,
    run_finetuning,
)

__all__ = [
    # metrics
    "set_seed", "_rankdata", "auroc_score", "confusion_metrics",
    "_delong_placements", "_normal_ppf", "_normal_cdf", "delong_auc_ci", "delong_test",
    # data
    "_stratified_split", "split_data", "stratified_kfold_indices",
    # ecg_io
    "read_med", "read_wfdb", "parse_wfdb_header", "scan_ecg_dir", "fit_length",
    "normalize_ecg", "load_labels_csv", "load_ecg_source", "load_ecg_from_cfg",
    # ecg_encoder
    "ConvBlock", "ConvFeatureExtractor", "ECGWav2VecEncoder",
    # training
    "make_amp", "load_weights", "WarmupCosineLR", "GroupedWarmupCosineLR",
    "calibrate_threshold", "run_finetuning",
]
