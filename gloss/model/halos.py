"""Phase 3 — HALOS-minimal: doc-conditioned node encoder + doc-generated geometry + attention stack.

forward(gb, grounding) -> seed logits. The geometry table is compiled once per forward from the bias
generator (over the per-DB metapath set) and broadcast to every layer/pair. ``doc_per_metapath`` (pooled
FK-role docs) makes the geometry a function of documentation; if omitted the generator uses its learned
null-doc (geometry is then driven by the metapath embedding alone — still FK-role-distinct).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import GlossBatch
from ..data.graph import GraphBundle
from ..docs.grounding import GroundingResult
from .attention import RelationalAttention
from .bias_generator import BiasGenerator, GeometryTable
from .column_encoder import ColumnEncoder
from .heads import EntityHead


class HALOSLayer(nn.Module):
    """Pre-norm relational-attention block + FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelationalAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, h: Tensor, gb: GlossBatch, geom: GeometryTable, *, log_t_ctx=None) -> Tensor:
        h = h + self.attn(self.norm1(h), gb, geom, log_t_ctx=log_t_ctx)
        h = h + self.ff(self.norm2(h))
        return h * gb.pad_mask.to(h.device).unsqueeze(-1)


class HALOS(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        *,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_text: int = 2560,
        n_freq: int = 16,
        sigma_floor: float = 0.1,
        absolute_anchor: bool = False,
        out_dim: int = 1,
    ):
        super().__init__()
        self.absolute_anchor = absolute_anchor
        self.encoder = ColumnEncoder(bundle, d_model=d_model, d_text=d_text, n_freq=n_freq)
        self.bias_gen = BiasGenerator(
            bundle.num_metapaths, n_heads, d_text=d_text,
            sigma_floor=sigma_floor, absolute_anchor=absolute_anchor,
        )
        self.layers = nn.ModuleList([HALOSLayer(d_model, n_heads) for _ in range(n_layers)])
        self.head = EntityHead(d_model, out_dim)

    def compile_geometry(self, doc_per_metapath: Tensor | None = None) -> GeometryTable:
        return self.bias_gen.compile(doc_per_metapath)

    def forward(
        self,
        gb: GlossBatch,
        grounding: GroundingResult,
        *,
        doc_per_metapath: Tensor | None = None,
        return_node_states: bool = False,
    ):
        h = self.encoder(gb, grounding)                                   # [B, N, d_model]
        geom = self.compile_geometry(doc_per_metapath)
        log_t_ctx = None
        if self.absolute_anchor:
            log_t_ctx = torch.log(gb.t_ctx.clamp_min(1e-12)).to(h.dtype).to(h.device)
        for layer in self.layers:
            h = layer(h, gb, geom, log_t_ctx=log_t_ctx)
        logits = self.head(h, gb)
        return (logits, h) if return_node_states else logits


def build_doc_per_metapath(bundle: GraphBundle, grounding: GroundingResult) -> Tensor:
    """Pool FK-role docs into a ``[num_metapaths, d_text]`` table (reserved/missing rows -> zeros).

    Each relation metapath id maps to a canonical fkey column; we mean-pool the grounded ``fk::*::<col>``
    embeddings over every table that uses that column. This is what makes the compiled geometry a
    function of documentation.
    """
    P = bundle.num_metapaths
    d_text = grounding.d_text
    table = torch.zeros(P, d_text)
    # canonical column -> metapath id
    col_to_mp = bundle.metapath_id  # {col: id}
    # collect fk doc keys per canonical column
    by_col: dict[str, list[str]] = {}
    for key in grounding.keys:
        if key.startswith("fk::"):
            _, _table, _col = key.split("::")
            by_col.setdefault(_col, []).append(key)
    for col, mp_id in col_to_mp.items():
        keys = by_col.get(col, [])
        if not keys:
            continue
        emb, _rel, grounded = grounding.gather(keys)
        if grounded.any():
            table[mp_id] = emb[grounded].mean(dim=0)
    return table
