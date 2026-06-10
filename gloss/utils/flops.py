"""Parameter / FLOPs accounting — keep the model honest about the '<=~30M params' budget (CLAUDE.md)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def count_params(module: "nn.Module", trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def param_summary(module: "nn.Module") -> str:
    total = count_params(module, trainable_only=False)
    train = count_params(module, trainable_only=True)
    return f"params: {train/1e6:.2f}M trainable / {total/1e6:.2f}M total"
