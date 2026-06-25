"""Losses — masked task losses over the seeds that carry a label (binary BCE, regression MSE)."""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def masked_bce(logits: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    """logits/target/has_target `[B]` -> scalar BCE over labelled seeds (0 if none)."""
    m = has_target
    if m.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[m], target[m].to(logits.dtype))


def masked_mse(pred: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    """pred/target/has_target `[B]` -> scalar MSE over labelled seeds (0 if none). ``target`` is assumed
    already standardized by the caller (regression targets are z-scored for training stability)."""
    m = has_target
    if m.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[m], target[m].to(pred.dtype))


def task_loss(out: Tensor, target: Tensor, has_target: Tensor, task_type: str) -> Tensor:
    """Dispatch by task type: BCE for ``binary``, MSE for ``regression`` (target pre-standardized)."""
    if task_type == "regression":
        return masked_mse(out, target, has_target)
    return masked_bce(out, target, has_target)
