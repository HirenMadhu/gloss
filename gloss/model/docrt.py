"""The model: RT cell encoder + RT relational-attention substrate + task head.

    cells  = CellEncoder(cb)                 # [B, S, d]   value (dtype enc) + RT name token
    cells  = RTSubstrate(cells, cb)          # [B, S, d]   relational attention (col/feat/nbr/full)
    logits = EntityHead(cells, cb)           # [B, out_dim] seed-row readout

This is plain RT (names-only). MoRE adds a Mixture-of-Experts FFN inside the substrate (Phase B), which
is the only new mechanism; the cell token and masks are RT's, unchanged.
"""
from __future__ import annotations

from torch import Tensor, nn

from ..data.collate import CellBatch
from ..data.graph import GraphBundle
from .column_encoder import CellEncoder
from .heads import EntityHead
from .rt_substrate import RTSubstrate


class DOCRT(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb,
        *,
        d_model: int = 256,
        n_blocks: int = 8,
        n_heads: int = 8,
        d_ff: int | None = None,
        enc_channels: int | None = None,
        out_dim: int = 1,
    ):
        super().__init__()
        self.encoder = CellEncoder(bundle, name_emb, d_model=d_model, enc_channels=enc_channels)
        self.substrate = RTSubstrate(d_model=d_model, n_blocks=n_blocks, n_heads=n_heads, d_ff=d_ff)
        self.head = EntityHead(d_model, out_dim)

    def forward(self, cb: CellBatch) -> Tensor:
        x = self.encoder(cb)                     # [B, S, d]
        x = self.substrate(x, cb)                # [B, S, d]
        return self.head(x, cb)                  # [B, out_dim]
