"""Phase 4 — evaluation metrics for binary entity tasks.

We use sklearn directly (relbench.metrics.log_loss calls a nonexistent `np.sigmoid` on this numpy and
expects raw logits). AP + AUROC are the imbalance-robust headline numbers for driver-dnf; log_loss is
kept because the Phase-6 CMI estimator is `E[log_loss_null - log_loss_full]`.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """logits/target `[N]` -> {ap, auroc, logloss}. Robust to a single-class batch (returns nan there)."""
    prob = torch.sigmoid(logits.detach().float().cpu()).numpy()
    return binary_metrics_prob(prob, target.detach().cpu().numpy().astype(int))


def binary_metrics_prob(prob, y) -> dict[str, float]:
    """Probabilities + integer labels -> {ap, auroc, logloss} (for the LightGBM baseline)."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    out: dict[str, float] = {}
    if len(np.unique(y)) < 2:
        out["ap"] = float("nan")
        out["auroc"] = float("nan")
    else:
        out["ap"] = float(average_precision_score(y, prob))
        out["auroc"] = float(roc_auc_score(y, prob))
    out["logloss"] = float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return out


def regression_metrics(pred, target) -> dict[str, float]:
    """pred/target `[N]` (original units; torch or array) -> {mae, rmse, r2}."""
    p = pred.detach().float().cpu().numpy().ravel() if torch.is_tensor(pred) else np.asarray(pred, float).ravel()
    y = target.detach().float().cpu().numpy().ravel() if torch.is_tensor(target) else np.asarray(target, float).ravel()
    diff = p - y
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    var = float(np.var(y))
    r2 = float(1.0 - np.mean(diff ** 2) / var) if var > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}
