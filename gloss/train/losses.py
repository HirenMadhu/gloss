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


def masked_pairwise_auc(logits: Tensor, target: Tensor, has_target: Tensor,
                        margin: float = 1.0) -> Tensor:
    """Squared-hinge pairwise AUC surrogate over labelled seeds (0 if either class is absent).

    ``mean_{i in pos, j in neg} relu(margin - (s_i - s_j))**2``. AUROC *is* the fraction of
    (pos, neg) pairs ranked correctly, so this optimises the metric directly, where BCE optimises
    calibrated probability and only ranks correctly as a side effect. This is the core of the
    AUC-margin objective without LibAUC's learnable ``(a, b, alpha)`` or its PESG optimizer — keeping
    AdamW means a change here is attributable to the loss alone (``results/WHY_NOT_RT.md``).

    Two caveats worth knowing before reading a result:

    * The pairs are **within-batch only**, so the estimate is biased by batch composition. At
      batch 64 and the measured positive rates (0.17-0.88) every batch has pairs, but a task at
      1000:1 imbalance would need a sampler, not just this loss.
    * The squared hinge has a different **scale** from BCE, which acts like a change in effective
      learning rate — and this model is lr-fragile (33/69 cells beat RT at 3e-4 vs 15/71 at 1e-3).
      The sweep varies lr, so this is covered rather than confounded, but a single-lr comparison
      would not be trustworthy.
    """
    m = has_target
    if m.sum() == 0:
        return logits.sum() * 0.0
    s, y = logits[m], target[m].to(logits.dtype)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        # A single-class batch carries no ranking information. Return a real zero that still tracks
        # the graph, so DDP/grad-accum see a gradient of the right shape rather than a detached 0.
        return logits.sum() * 0.0
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # [P, N]
    return (margin - diff).clamp_min(0).pow(2).mean()


#: Binary objectives selectable at train time. ``bce`` is the historical default.
BINARY_LOSSES = {"bce": masked_bce, "auc": masked_pairwise_auc}


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
              regression_loss: str = "mse", binary_loss: str = "bce") -> Tensor:
    """Dispatch by task type: ``binary_loss`` for ``binary``, ``regression_loss`` for ``regression``.

    The objective for the *other* task type is ignored rather than rejected, so a runner can pass
    both unconditionally without knowing the task kind.
    """
    table, name = ((REGRESSION_LOSSES, regression_loss) if task_type == "regression"
                   else (BINARY_LOSSES, binary_loss))
    kind = "regression_loss" if task_type == "regression" else "binary_loss"
    try:
        fn = table[name]
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(table)}") from None
    return fn(out, target, has_target)
