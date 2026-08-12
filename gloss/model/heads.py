"""Phase 3 — the task head. Reads the seed row's ROW TOKEN and projects to logits.

`EntityHead` (mean-pool the seed row's *cells*, the RT-style readout) is retired to
``archive/multi-level/`` along with the ``head.mode: seed_cells`` switch it served: it existed only
for the Phase 0 parity check, which `row_token` won. The masked-cell pretraining head belongs to a
later phase and does not exist yet.
"""
from __future__ import annotations

from torch import Tensor, nn

from ..data.collate import CellBatch


class RowTokenHead(nn.Module):
    """changes.md §3.8 — read out the seed **row token** instead of pooling its cells.

    .. math:: \\text{logit} = W_2\\,\\mathrm{GELU}(W_1\\,\\mathrm{LayerNorm}(u_{\\text{root}}))

    ``u_root`` is selected by ``row_is_root``, which the row graph guarantees is True for exactly one
    row per seed. It replaced a mean over the (typically ~6) seed cells; both heads had an identical
    body on purpose, so the Phase-0 difference was attributable to the readout and not to capacity.
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
