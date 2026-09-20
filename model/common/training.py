"""General training infrastructure: AMP / weight loading / LR scheduling / fine-tuning loop / threshold calibration.


"""
import contextlib
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.metrics import auroc_score


# ---------------------------------------------------------------------------
# AMP and weight loading (compatible with torch 1.x / 2.x)
# ---------------------------------------------------------------------------

def make_amp(device: torch.device):
    """Return (autocast_ctx_factory, scaler_or_None), compatible with torch 1.x / 2.x."""
    if device.type != "cuda":
        return contextlib.nullcontext, None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        return (lambda: torch.autocast(device_type="cuda"), scaler)
    except (AttributeError, TypeError):
        from torch.cuda.amp import autocast, GradScaler  # fallback for torch < 2.0
        return (autocast, GradScaler(enabled=True))


def load_weights(state_dict_path, device):
    """Load a state dict, compatible with torch>=2.0 (weights_only) and older versions."""
    try:
        return torch.load(state_dict_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(state_dict_path, map_location=device)


# ---------------------------------------------------------------------------
# Learning-rate scheduling
# ---------------------------------------------------------------------------

class WarmupCosineLR:
    """Linear warmup + cosine annealing (updated per step).
    total_steps is the total number of batches. Single-group learning-rate variant."""

    def __init__(self, optimizer, total_steps: int, warmup_frac: float, base_lr: float):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = int(self.total_steps * warmup_frac)
        self.base_lr = base_lr

    def step(self, step: int) -> float:
        if step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


class GroupedWarmupCosineLR:
    """Linear warmup + cosine annealing applied to EVERY param group with its
    own base learning rate (per step). Multimodal layerwise learning-rate variant."""

    def __init__(self, optimizer, total_steps: int, warmup_frac: float):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(self.total_steps * warmup_frac)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, step: int) -> float:
        if step < self.warmup_steps:
            factor = (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * factor
        return factor


# ---------------------------------------------------------------------------
# Single-epoch training loop (standard BCE, batches are (x, y) tuples)
# ---------------------------------------------------------------------------

def _run_epoch_standard(model, loader, device, autocast_ctx, scaler, optimizer=None,
                        scheduler=None, grad_clip=None, step_counter=None):
    """Run one epoch (forward/backward) for models with (x, y) batches.
    Pass an optimizer to train, None to evaluate."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batch = 0
    all_y, all_p = [], []

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        with autocast_ctx():
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
        all_y.append(y.detach().cpu().numpy())
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    return total_loss / max(1, n_batch), auroc_score(y_all, p_all)


# ---------------------------------------------------------------------------
# Fine-tuning loop (early stopping on validation AUROC)
# ---------------------------------------------------------------------------

def run_finetuning(model, train_loader, val_loader, cfg, device, tag: str = "",
                   run_epoch=None, build_optimizer=None, build_scheduler=None):
    """Supervised fine-tuning with early stopping on validation AUROC.

    Default (ECG single-modal): AdamW wd=0.01, lr=cfg.lr_ecg, warmup 0.1 + cosine, AMP, BCE.
    The multimodal path replaces three optional hooks:
      - run_epoch:          epoch function that handles dict batches
      - build_optimizer:    builds the grouped layerwise-LR optimizer
      - build_scheduler:    builds the grouped warmup-cosine scheduler
    Returns (best_val_auc, history).
    """
    autocast_ctx, scaler = make_amp(device)

    if build_optimizer is None:
        def build_optimizer(m, c):
            return torch.optim.AdamW(m.parameters(), lr=c.lr_ecg, weight_decay=c.weight_decay)
    if build_scheduler is None:
        def build_scheduler(opt, total_steps):
            return WarmupCosineLR(opt, total_steps, cfg.warmup_frac, cfg.lr_ecg)
    if run_epoch is None:
        run_epoch = _run_epoch_standard

    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = build_scheduler(optimizer, total_steps)
    step_counter = {"n": 0}

    best_auc = -1.0
    patience_left = cfg.patience
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_loader, device, autocast_ctx, scaler,
                                  optimizer, scheduler, cfg.grad_clip, step_counter)
        val_loss, val_auc = run_epoch(model, val_loader, device, autocast_ctx, None)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_auc": val_auc})
        print(f"  [{tag}] epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_auc {val_auc:.4f} | {time.time()-t0:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_left = cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [{tag}] early stop at epoch {epoch} (best_auc={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, history


# ---------------------------------------------------------------------------
# Decision-threshold calibration (maximize F1 or Youden's J on the validation set)
# ---------------------------------------------------------------------------

def calibrate_threshold(y_val: np.ndarray, scores_val: np.ndarray,
                        objective: str = "f1") -> float:
    """Scan decision thresholds on the validation set.
    objective: 'f1' (paper's convention) or 'youden_j'."""
    thresholds = np.unique(scores_val)
    if len(thresholds) > 500:
        thresholds = np.quantile(scores_val, np.linspace(0, 1, 500))
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        pred = (scores_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        if objective == "youden_j":
            tn = int(((pred == 0) & (y_val == 0)).sum())
            score = (tp / (tp + fn) if (tp + fn) > 0 else 0.0) - \
                    (fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        else:  # f1
            score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if score > best_s:
            best_s, best_t = score, float(t)
    print(f"[Threshold] best {objective}={best_s:.4f} at threshold={best_t:.4f}")
    return best_t
