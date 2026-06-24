"""Phase 2-3 (DOC-RT) — the model: documentation-conditioned cell encoder + RT substrate + task head.

    cells  = CellEncoder(cb, grounding)      # [B, S, d]   FiLM(d_c) on values + RT name token
    cells  = RTSubstrate(cells, cb)          # [B, S, d]   relational attention (col/feat/nbr/full)
    logits = EntityHead(cells, cb)           # [B, out_dim] seed-row readout

The only documentation entry point is the FiLM in the cell encoder; switching the grounding regime
(full / null / shuffled / name_only) is the entire docs-on-vs-docs-off mechanism.
"""
from __future__ import annotations

from torch import Tensor, nn

from ..data.collate import CellBatch
from ..data.graph import GraphBundle
from ..docs.grounding import GroundingResult
from .column_encoder import CellEncoder
from .heads import EntityHead
from .rt_substrate import RTSubstrate


class DOCRT(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        *,
        d_model: int = 256,
        d_text: int = 2560,
        n_blocks: int = 8,
        n_heads: int = 8,
        d_ff: int | None = None,
        enc_channels: int | None = None,
        out_dim: int = 1,
    ):
        super().__init__()
        self.encoder = CellEncoder(bundle, d_model=d_model, d_text=d_text, enc_channels=enc_channels)
        self.substrate = RTSubstrate(d_model=d_model, n_blocks=n_blocks, n_heads=n_heads, d_ff=d_ff)
        self.head = EntityHead(d_model, out_dim)

    def forward(self, cb: CellBatch, grounding: GroundingResult) -> Tensor:
        x = self.encoder(cb, grounding)          # [B, S, d]
        x = self.substrate(x, cb)                # [B, S, d]
        return self.head(x, cb)                  # [B, out_dim]
