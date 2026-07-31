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


def masked_l1(pred: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    """pred/target/has_target `[B]` -> scalar L1 over labelled seeds (0 if none). ``target`` is assumed
    already standardized by the caller.

    RelBench scores regression on **MAE**, but training minimised MSE. Those have different optima —
    MSE's is the conditional mean, L1's the conditional median — and the penalty grows with target
    skew, which is severe here: rel-trial/study-adverse has skew 39.1 and rel-event/user-attendance
    5.0. On study-adverse the mean-vs-median NMAE gap alone (0.0455) is nearly *twice* our whole
    deficit to RT (0.024), so aligning the loss with the metric is the cheapest thing that could
    close it (see ``results/WHY_NOT_RT.md`` §4)."""
    m = has_target
    if m.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[m], target[m].to(pred.dtype))


def masked_huber(pred: Tensor, target: Tensor, has_target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber: L2 within ``delta`` of the target, L1 beyond. Median-seeking like L1 in the tail but
    with a smooth gradient near zero, so it does not inherit L1's constant-magnitude gradient. On a
    z-scored target ``delta=1.0`` is one train-std."""
    m = has_target
    if m.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[m], target[m].to(pred.dtype), delta=delta)


#: Regression objectives selectable at train time. ``mse`` is the historical default — every result
#: before 2026-07-31 used it — so it stays the default and any change must be explicit.
REGRESSION_LOSSES = {"mse": masked_mse, "l1": masked_l1, "huber": masked_huber}


def task_loss(out: Tensor, target: Tensor, has_target: Tensor, task_type: str,
              regression_loss: str = "mse") -> Tensor:
    """Dispatch by task type: BCE for ``binary``, ``regression_loss`` for ``regression``.

    ``regression_loss`` is ignored for binary tasks (there is one sensible objective) rather than
    raising, so a runner can pass it unconditionally.
    """
    if task_type == "regression":
        try:
            fn = REGRESSION_LOSSES[regression_loss]
        except KeyError:
            raise ValueError(f"unknown regression_loss {regression_loss!r}; "
                             f"expected one of {sorted(REGRESSION_LOSSES)}") from None
        return fn(out, target, has_target)
    return masked_bce(out, target, has_target)
