"""Multimodal training flow: grouped (layerwise-LR) optimizer / dict-batch training loop / cross-validation / prediction.

Generic AMP, schedulers and the fine-tuning loop live in model.common.training;
this module plugs the multimodal-specific batch format and layerwise LR into
run_finetuning through its hook parameters.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.common.metrics import auroc_score
from model.common.training import GroupedWarmupCosineLR, load_weights, make_amp, run_finetuning

from model.multimodal_omi.data import MultimodalDataset, collate_multimodal
from model.multimodal_omi.models import MultimodalOMIClassifier


def build_grouped_optimizer(model: MultimodalOMIClassifier, cfg):
    """
    Paper modular learning-rate configuration:
      - MacBERT backbone : layerwise decay, top lr = lr_text_top (1.5e-5),
        factor 0.9 per lower layer -> bottom ~4.7e-6; embeddings/pooler use
        the bottom lr.
      - ECG branch / fusion module / classification head : lr_common = 2e-5.
    All groups share AdamW(weight_decay=0.01).
    """
    bert = model.text_encoder
    groups = []
    n_layers = len(bert.encoder.layer)
    # BERT encoder.layer[0] is the bottom layer, [n-1] is the top layer.
    for i, layer in enumerate(bert.encoder.layer):
        lr = float(cfg.lr_text_top) * (float(cfg.lr_text_decay) ** (n_layers - 1 - i))
        groups.append({"params": layer.parameters(), "lr": lr})
    groups.append({"params": bert.embeddings.parameters(), "lr": float(cfg.lr_text_bottom)})
    groups.append({"params": bert.pooler.parameters(), "lr": float(cfg.lr_text_bottom)})

    if model.mode in ("fusion", "text"):
        groups.append({"params": model.text_proj.parameters(), "lr": float(cfg.lr_common)})
    if model.mode in ("fusion", "ecg"):
        groups.append({"params": model.ecg_encoder.parameters(), "lr": float(cfg.lr_common)})
        groups.append({"params": model.ecg_mlp.parameters(), "lr": float(cfg.lr_common)})
    if model.mode == "fusion":
        groups.append({"params": [model.fusion_cls], "lr": float(cfg.lr_common)})
        groups.append({"params": model.fusion_layers.parameters(), "lr": float(cfg.lr_common)})
    groups.append({"params": model.classifier.parameters(), "lr": float(cfg.lr_common)})

    return torch.optim.AdamW(groups, lr=float(cfg.lr_common), weight_decay=float(cfg.weight_decay))


def _run_epoch(model, loader, device, autocast_ctx, scaler, optimizer=None,
               scheduler=None, grad_clip=None, step_counter=None):
    """Run one epoch (forward/backward) for dict-batch loaders.
    Pass an optimizer to train, None to evaluate."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batch = 0
    all_y, all_p = [], []

    for batch in loader:
        y = batch["y"].to(device)
        with autocast_ctx():
            if model.mode == "fusion":
                logits = model(
                    ecg=batch["ecg"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).squeeze(1)
            elif model.mode == "text":
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).squeeze(1)
            else:  # ecg
                logits = model(ecg=batch["ecg"].to(device)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and step_counter is not None:
                scheduler.step(step_counter["n"])
                step_counter["n"] += 1

        total_loss += float(loss.item())
        n_batch += 1
        all_y.append(y.detach().cpu().numpy())
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    return total_loss / max(1, n_batch), auroc_score(y_all, p_all)


def run_finetuning_multimodal(model, train_loader, val_loader, cfg, device, tag: str = ""):
    """Multimodal-specific fine-tuning: MacBERT layerwise LR + grouped warmup-cosine + dict batches.

    Reuses the early-stopping loop of model.common.training.run_finetuning, replacing
    only the three hooks: epoch function, optimizer and scheduler.
    """
    return run_finetuning(
        model, train_loader, val_loader, cfg, device, tag=tag,
        run_epoch=_run_epoch,
        build_optimizer=lambda m, c: build_grouped_optimizer(m, c),
        build_scheduler=lambda opt, total_steps: GroupedWarmupCosineLR(opt, total_steps, cfg.warmup_frac),
    )


def run_cross_validation(ecg, tokenized_texts, labels, cfg, device, pretrain_ckpt,
                         cv_epochs=None, cv_patience=None):
    """Stratified 5-fold CV on the training set (paper reports CV AUROC).

    The paper decouples the CV budget from the final fine-tuning: the
    cross-validation stage runs 15 epochs with early-stop patience 5
    (Table 2: CV Max Epochs = 15 / CV Early-stop Patience = 5), while the final
    model trains up to 40 epochs with patience 10 (--epochs / --patience).
    Pass cv_epochs / cv_patience to apply the CV budget; when omitted, the fold
    loop falls back to cfg.epochs / cfg.patience.
    """
    import copy
    from sklearn.model_selection import StratifiedKFold

    cv_cfg = copy.copy(cfg)
    if cv_epochs is not None:
        cv_cfg.epochs = int(cv_epochs)
    if cv_patience is not None:
        cv_cfg.patience = int(cv_patience)

    skf = StratifiedKFold(n_splits=int(cfg.cv_folds), shuffle=True, random_state=cfg.seed)
    aucs = []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        model = build_model(cfg, device)
        if pretrain_ckpt is not None:
            model.ecg_encoder.load_state_dict(load_weights(pretrain_ckpt, device))
        tr_loader = DataLoader(
            MultimodalDataset(ecg[tr], [tokenized_texts[i] for i in tr], labels[tr], cfg.mode),
            batch_size=cfg.batch_size, shuffle=True, num_workers=0,
            collate_fn=collate_multimodal,
        )
        va_loader = DataLoader(
            MultimodalDataset(ecg[va], [tokenized_texts[i] for i in va], labels[va], cfg.mode),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
            collate_fn=collate_multimodal,
        )
        auc, _ = run_finetuning_multimodal(model, tr_loader, va_loader, cv_cfg, device, tag=f"cv-fold-{k}")
        aucs.append(auc)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_auc = float(np.mean(aucs))
    print(f"[CV] {cfg.cv_folds}-fold AUROC (epochs<={cv_cfg.epochs}, patience={cv_cfg.patience}): "
          f"{[f'{a:.4f}' for a in aucs]} | mean {mean_auc:.4f}")
    return mean_auc


def predict_proba(model, loader, device):
    """Run inference batch by batch over a dict-batch DataLoader; returns (y_true, sigmoid probabilities)."""
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["y"].numpy()
            if model.mode == "fusion":
                logits = model(
                    ecg=batch["ecg"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).squeeze(1)
            elif model.mode == "text":
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).squeeze(1)
            else:
                logits = model(ecg=batch["ecg"].to(device)).squeeze(1)
            all_y.append(y)
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def build_model(cfg, device):
    model = MultimodalOMIClassifier(
        macbert_path=cfg.macbert_path,
        mode=cfg.mode,
        num_leads=cfg.num_leads,
        ecg_hidden=cfg.hidden_dim,
        ecg_layers=cfg.num_layers,
        mlp_layers=cfg.mlp_layers,
        mlp_dropout=0.2,
        proj_dim=256,
        fusion_layers=cfg.fusion_layers,
        fusion_heads=8,
        fusion_dropout=0.1,
    ).to(device)
    return model
