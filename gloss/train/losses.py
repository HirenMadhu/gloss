"""Phase 4 — losses. Masked binary cross-entropy over seeds that carry a label."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_bce(logits: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    """logits/target/has_target `[B]` -> scalar BCE over labelled seeds (0 if none)."""
    m = has_target
    if m.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[m], target[m].to(logits.dtype))
