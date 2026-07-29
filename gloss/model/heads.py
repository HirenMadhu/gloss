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


class RowTokenHead(nn.Module):
    """changes.md §3.8 — read out the seed **row token** instead of pooling its cells.

    .. math:: \\text{logit} = W_2\\,\\mathrm{GELU}(W_1\\,\\mathrm{LayerNorm}(u_{\\text{root}}))

    ``u_root`` is selected by ``row_is_root``, which the row graph guarantees is True for exactly one
    row per seed. This replaces :class:`EntityHead`'s mean over the (typically ~6) seed cells; the old
    head stays reachable behind ``head.mode: seed_cells`` for the Phase 0 parity check, so the two are
    comparable rather than one silently replacing the other.

    Same body as :class:`EntityHead` on purpose — only the *readout* changes, so a Phase-0 difference
    is attributable to the readout and not to head capacity.
    """

    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, row_states: Tensor, cb: CellBatch) -> Tensor:
        """row_states ``[B, R, d]`` -> logits ``[B, out_dim]`` from the seed root row."""
        root = cb.row_is_root.to(row_states.dtype)
        # exactly one root per seed is a row-graph invariant; clamp defensively so a malformed batch
        # degrades to a mean over roots rather than dividing by zero
        root = root / root.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (row_states * root.unsqueeze(-1)).sum(dim=1)     # [B, d]
        return self.mlp(pooled)
