"""The model: RT cell encoder + relational signature + two-level substrate + row-token head.

    cells, _   = CellEncoder(cb)                 # [B, S, d]   dtype value emb + RT name token
    z          = RelationalSignature(cb)         # [B, S, d_sig]   value-free routing signature
    c, r, aux  = TwoLevelSubstrate(cells, cb, z) # [B,S,d], [B,R,d]
    logits     = RowTokenHead(r, cb)             # [B, out_dim]    seed root-row readout

``forward`` returns ``(logits, aux)`` where ``aux`` is the summed router-orthogonality loss (0 for
``dense``). Two ``route_on`` settings survive, and they are the headline comparison:

    signature : route the cell MoE on z, the value-free relational signature (the method)
    dense     : no cell MoE at all — a single shared SwiGLU, i.e. RT's FFN

The other four arms (``hybrid``/``hidden``/``value``/``identity``) and the single-level ``arch: rt``
substrate are retired to ``archive/multi-level/``. The row-level MoE is always on and always routed
on the row signature; it has no dense arm, because ``row_ffn: dense`` was a phase-0 switch.
"""
from __future__ import annotations

from torch import nn

from ..data.collate import CellBatch
from ..data.graph import GraphBundle
from ..text.schema import build_column_modality_ids
from .column_encoder import CellEncoder
from .heads import RowTokenHead
from .schema_free_encoder import SchemaFreeCellEncoder
from .signature import RelationalSignature
from .two_level import TwoLevelSubstrate

ROUTE_ONS = ("signature", "dense")

#: How a cell's VALUE is projected. ``per_column`` is torch_frame's stype encoders, which keep
#: per-column weights (10.6M on rel-trial at d_text=384) and therefore cannot transfer; ``schema_free``
#: shares one projection per datatype, RT-style, leaving zero learned parameters sized by the schema.
#: ``per_column`` stays the default because every reported result was measured on it.
CELL_ENCODERS = ("per_column", "schema_free")


class MoRE(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb,
        *,
        d_model: int = 256,
        d_sig: int = 128,
        n_blocks: int = 8,
        n_heads: int = 8,
        d_ff: int | None = None,
        enc_channels: int | None = None,
        out_dim: int = 1,
        route_on: str = "signature",
        num_experts: int = 4,
        k: int = 2,
        table_name_emb=None,
        role_name_emb=None,
        cat_name_emb=None,
        cell_encoder: str = "per_column",
        max_hop: int = 8,
        **two_level_kw,
    ):
        super().__init__()
        assert route_on in ROUTE_ONS, f"unknown route_on={route_on!r}"
        if cell_encoder not in CELL_ENCODERS:
            raise ValueError(f"unknown cell_encoder {cell_encoder!r}; expected {CELL_ENCODERS}")
        if table_name_emb is None or role_name_emb is None:
            raise ValueError(
                "MoRE needs table_name_emb and role_name_emb (build them with "
                "gloss.text.schema.build_table_name_embeddings / role_name_embeddings_with_none; "
                "scripts/build_schema_cache.py caches them)."
            )
        self.route_on = route_on
        self.cell_encoder_kind = cell_encoder
        if cell_encoder == "schema_free":
            self.encoder = SchemaFreeCellEncoder(bundle, name_emb, cat_name_emb,
                                                 d_model=d_model, enc_channels=enc_channels)
        else:
            self.encoder = CellEncoder(bundle, name_emb, d_model=d_model,
                                       enc_channels=enc_channels)
        modality_id, n_stypes = build_column_modality_ids(bundle)
        self.signature = RelationalSignature(name_emb, modality_id, n_stypes, d_sig=d_sig)
        self.substrate = TwoLevelSubstrate(
            d_model, d_ff or 4 * d_model, d_sig,
            table_name_emb, role_name_emb, name_emb,
            n_blocks=n_blocks, n_heads=n_heads, max_hop=max_hop,
            cell_ffn="dense" if route_on == "dense" else "moe",
            cell_route_dim=d_sig,
            cell_num_experts=num_experts, cell_k=k,
            **two_level_kw,
        )
        self.head = RowTokenHead(d_model, out_dim)

    def category_table(self, node_type: str):
        """The per-DB categorical embedding matrix for ``node_type``, or None.

        Exposed so :class:`~gloss.model.mlm_head.MaskedCellHead` can tie its logits to it instead of
        owning a vocabulary-shaped output layer of its own.
        """
        if self.cell_encoder_kind == "schema_free":
            # One frozen, schema-independent table of category-LABEL embeddings, so the decode head
            # scores against real semantics on a database it never saw instead of untrained vectors.
            return self.encoder.category_table(node_type)
        enc = self.encoder.cell_encoders
        if node_type not in enc:
            return None
        # nn.ModuleDict has no `.get`; use membership + indexing or this silently returns None.
        sub = getattr(enc[node_type], "encoder_dict", None)
        if sub is None or "categorical" not in sub:
            return None
        return sub["categorical"].emb.weight

    def forward(self, cb: CellBatch, *, cell_mask=None, return_cells: bool = False):
        """-> ``(logits, aux)``, or ``(logits, aux, cells)`` when ``return_cells``.

        ``cell_mask`` withholds those cells' values (masked-cell pretraining); ``return_cells`` hands
        back the ``[B, S, d_model]`` cell states the substrate already computes and this method
        otherwise discards, which is what the reconstruction head reads.
        """
        x = self.encoder(cb, cell_mask)
        z = self.signature(cb)
        cells, rows, aux, _diag = self.substrate(x, cb, z=z)
        logits = self.head(rows, cb)                          # [B, out_dim], scalar aux
        return (logits, aux, cells) if return_cells else (logits, aux)
