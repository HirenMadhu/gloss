"""Retired: `EntityHead` — the RT-style readout that mean-pools the SEED ROW'S CELLS.

Reachable in the live tree until 2026-08-12 behind ``head.mode: seed_cells``, and forced on by
``arch: rt``. It existed for the changes.md Phase 0 parity check — "is the two-level gain from the
row tokens, or merely from changing the readout?" — which `RowTokenHead` won. Kept because the
`arch: rt` result directories cannot be regenerated without it.

Body is deliberately identical to `RowTokenHead`; only the readout differs.
"""
from __future__ import annotations

from torch import Tensor, nn

from gloss.data.collate import CellBatch


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
