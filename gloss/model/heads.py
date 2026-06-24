"""Phase 3 — task heads. A seed-row readout for entity tasks (binary/regression).

DOC-RT v1 reads out the seed entity row by mean-pooling its (now context-aware, after the RT substrate's
global attention) cells, then projects to logits. The masked-cell prediction head used by RT for
pretraining belongs to a later phase; only the supervised readout head is needed for the Phase-3 DoD.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import CellBatch


class EntityHead(nn.Module):
    """Mean-pool the seed root row's cells and project to ``out_dim`` logits."""

    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, cell_states: Tensor, cb: CellBatch) -> Tensor:
        """cell_states ``[B, S, d]`` -> logits ``[B, out_dim]`` from the seed row's cells."""
        w = cb.is_seed_cell.to(cell_states.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (cell_states * w.unsqueeze(-1)).sum(dim=1)       # [B, d]
        return self.mlp(pooled)
