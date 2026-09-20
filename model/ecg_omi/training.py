"""Single-modal ECG training flow: SimCLR pretraining / cross-validation / prediction.

Generic AMP, schedulers and the fine-tuning loop live in model.common.training;
this module keeps the epoch loop and pretraining logic tied to the ECG data
format ((x, y) tuple batches).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.common.data import stratified_kfold_indices
from model.common.ecg_encoder import ECGWav2VecEncoder
from model.common.metrics import auroc_score
from model.common.training import WarmupCosineLR, load_weights, make_amp, run_finetuning

from model.ecg_omi.data import ECGDataset, time_shift_batch
from model.ecg_omi.models import ECGClassifier, ECGSimCLR, nt_xent_loss


def _run_epoch(model, loader, device, autocast_ctx, scaler, optimizer=None,
               scheduler=None, grad_clip=None, step_counter=None, pretrain=False):
    """Run one epoch (forward/backward). Pass an optimizer to train, None to evaluate.
    When pretrain=True, run SimCLR contrastive learning (returns only the loss)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batch = 0
    all_y, all_p = [], []

    for batch in loader:
        if pretrain:
            x = batch.to(device)
            with torch.no_grad():
                x1 = time_shift_batch(x)
                x2 = time_shift_batch(x)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

        with autocast_ctx():
            if pretrain:
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=0.1)
            else:
                logits = model(x).squeeze(1)
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
        if not pretrain:
            all_y.append(y.detach().cpu().numpy())
            all_p.append(torch.sigmoid(logits).detach().cpu().numpy())

    if pretrain:
        return total_loss / max(1, n_batch)
    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    return total_loss / max(1, n_batch), auroc_score(y_all, p_all)


def run_pretraining(ecg_unlabeled, cfg, device, out_dir: Path):
    """SimCLR contrastive pretraining: Adam lr=1e-4, cosine annealing,
    gradient clipping=1, early stopping on validation loss."""
    rng = np.random.default_rng(cfg.seed + 7)
    n = len(ecg_unlabeled)
    val_size = max(2, int(n * 0.05))
    val_idx = rng.choice(n, size=val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    train_ds = ECGDataset(ecg_unlabeled[train_idx])
    val_ds = ECGDataset(ecg_unlabeled[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_pretrain, shuffle=True,
                              num_workers=0, drop_last=len(train_ds) >= cfg.batch_size_pretrain)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_pretrain, shuffle=False, num_workers=0)

    encoder = ECGWav2VecEncoder(cfg.num_leads, cfg.hidden_dim, cfg.num_layers, dropout=0.1)
    model = ECGSimCLR(encoder, proj_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_pretrain)
    autocast_ctx, scaler = make_amp(device)
    total_steps = len(train_loader) * cfg.pretrain_epochs
    scheduler = WarmupCosineLR(optimizer, total_steps, warmup_frac=0.0, base_lr=cfg.lr_pretrain)
    step_counter = {"n": 0}

    best_val_loss = float("inf")
    patience_left = cfg.patience
    best_state = None
    print(f"[Pretrain] unlabeled={n}, epochs<={cfg.pretrain_epochs}, batch={cfg.batch_size_pretrain}")

    for epoch in range(1, cfg.pretrain_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, device, autocast_ctx, scaler,
                                optimizer, scheduler, cfg.grad_clip, step_counter, pretrain=True)
        val_loss = _run_epoch(model, val_loader, device, autocast_ctx, None, pretrain=True)
        print(f"  epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        ckpt = out_dir / "pretrain_encoder.pt"
        torch.save(best_state, ckpt)
        print(f"[Pretrain] best encoder saved -> {ckpt}")
        return ckpt
    return None


def _build_model(cfg, device):
    model = ECGClassifier(cfg.num_leads, cfg.hidden_dim, cfg.num_layers,
                          cfg.mlp_layers, dropout=0.2).to(device)
    return model


def run_cross_validation(ecg, labels, cfg, device, pretrain_ckpt,
                         cv_epochs=None, cv_patience=None):
    """Stratified 5-fold cross-validation on the training set to evaluate AUROC
    (used by the paper for hyper-parameter search).

    The paper decouples the CV budget from the final fine-tuning: the
    cross-validation stage runs 15 epochs with early-stop patience 5
    (Table 2: CV Max Epochs = 15 / CV Early-stop Patience = 5), while the final
    model trains up to 40 epochs with patience 10 (--epochs / --patience).
    Pass cv_epochs / cv_patience to apply the CV budget; when omitted, the fold
    loop falls back to cfg.epochs / cfg.patience.
    """
    import copy

    cv_cfg = copy.copy(cfg)
    if cv_epochs is not None:
        cv_cfg.epochs = int(cv_epochs)
    if cv_patience is not None:
        cv_cfg.patience = int(cv_patience)

    folds = stratified_kfold_indices(labels, cfg.cv_folds, cfg.seed)
    aucs = []
    for k, (tr, va) in enumerate(folds):
        model = _build_model(cfg, device)
        if pretrain_ckpt is not None:
            model.encoder.load_state_dict(load_weights(pretrain_ckpt, device))
        tr_loader = DataLoader(ECGDataset(ecg[tr], labels[tr]), batch_size=cfg.batch_size,
                               shuffle=True, num_workers=0)
        va_loader = DataLoader(ECGDataset(ecg[va], labels[va]), batch_size=cfg.batch_size,
                               shuffle=False, num_workers=0)
        auc, _ = run_finetuning(model, tr_loader, va_loader, cv_cfg, device, tag=f"cv-fold-{k+1}")
        aucs.append(auc)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_auc = float(np.mean(aucs))
    print(f"[CV] {cfg.cv_folds}-fold AUROC (epochs<={cv_cfg.epochs}, patience={cv_cfg.patience}): "
          f"{[f'{a:.4f}' for a in aucs]} | mean {mean_auc:.4f}")
    return mean_auc


def predict_proba(model, loader, device):
    """Run inference batch by batch over a DataLoader; returns (y_true, sigmoid probabilities)."""
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).squeeze(1)
            all_y.append(y.numpy())
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_p)
