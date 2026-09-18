"""OMI multimodal model codebase (ECG + clinical text fusion).

Module layout:
    evaluation/      evaluation utilities (threshold search / plotting / probability calibration)
    common/          base code shared by both training scripts (metrics / data / ECG encoder / training facilities)
    ecg_omi/         single-modal M_ECG training
    multimodal_omi/  multimodal fusion M_multi training
    text_process/    clinical text extraction / QC / length normalization
"""
