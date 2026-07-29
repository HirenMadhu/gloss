"""The model: RT cell encoder + relational signature + RT substrate (+ MoE FFN) + task head.

    cells, value = CellEncoder(cb)             # [B, S, d]   value (dtype enc) + RT name token
    z            = RelationalSignature(cb)     # [B, S, d_sig]   value-free routing signature
    cells, aux   = RTSubstrate(cells, cb, ...) # [B, S, d]   relational attention + MoE FFN
    logits       = EntityHead(cells, cb)       # [B, out_dim]    seed-row readout

MoRE's only new mechanism is the Mixture-of-Experts FFN in the substrate, routed on ``z`` (or, in the
ablation arms, the hidden / value / identity / signature+hidden). ``forward`` returns ``(logits, aux)``
where ``aux`` is the summed router-orthogonality loss (0 for the dense arms). ``route_on``:

    signature : route on z (the method)                          identity : learned per-column id (no transfer)
    hybrid    : route on [z ; h] (signature+hidden)              dense    : plain RT (no MoE)
    hidden    : route on the block's evolving hidden state       dense_wide: param-matched dense (d_ff x k)
    value     : route on the cell's value component

The MoE FFN carries optional ablation additions (all off by default): a shared always-on expert
(``use_shared``), a cosine router (``cosine``/``tau``), Top-P adaptive expert count (``top_p``), and a
hierarchical two-level gate (``hmoe`` with ``n_groups``/``experts_per_group``/``k2``).
"""
from __future__ import annotations

from torch import nn

from ..data.collate import CellBatch
from ..data.graph import GraphBundle
from ..text.schema import build_column_modality_ids
from .column_encoder import CellEncoder
from .heads import EntityHead, RowTokenHead
from .rt_substrate import RTSubstrate
from .signature import RelationalSignature

ROUTE_ONS = ("signature", "hybrid", "hidden", "value", "identity", "dense", "dense_wide")
ARCHS = ("rt", "two_level")


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
        moe_placement: str = "all",
        use_shared: bool = False,
        cosine: bool = False,
        tau: float = 0.3,
        top_p: float | None = None,
        hmoe: bool = False,
        n_groups: int = 4,
        experts_per_group: int = 2,
        k2: int = 1,
        # --- two-level additions (changes.md); arch='rt' ignores all of them ---
        arch: str = "rt",
        table_name_emb=None,
        role_name_emb=None,
        time_mode: str = "buckets",
        max_hop: int = 8,
        head_mode: str = "row_token",
        **two_level_kw,
    ):
        super().__init__()
        assert route_on in ROUTE_ONS, f"unknown route_on={route_on!r}"
        assert arch in ARCHS, f"unknown arch={arch!r}"
        self.route_on = route_on
        self.arch = arch
        self.needs_value = route_on == "value"
        self.encoder = CellEncoder(bundle, name_emb, d_model=d_model, enc_channels=enc_channels)
        if route_on in ("signature", "hybrid"):                  # hybrid = [z ; h] needs z too
            modality_id, n_stypes = build_column_modality_ids(bundle)
            self.signature = RelationalSignature(name_emb, modality_id, n_stypes, d_sig=d_sig,
                                                 time_mode=time_mode)
        else:
            self.signature = None
        self.id_emb = nn.Embedding(int(name_emb.shape[0]), d_sig) if route_on == "identity" else None

        if arch == "rt":
            self.substrate = RTSubstrate(
                d_model=d_model, n_blocks=n_blocks, n_heads=n_heads, d_ff=d_ff,
                route_on=route_on, d_sig=d_sig, num_experts=num_experts, k=k,
                moe_placement=moe_placement,
                use_shared=use_shared, cosine=cosine, tau=tau, top_p=top_p,
                hmoe=hmoe, n_groups=n_groups, experts_per_group=experts_per_group, k2=k2,
            )
            self.head = EntityHead(d_model, out_dim)
            self.head_mode = "seed_cells"
        else:
            from .two_level import TwoLevelSubstrate

            if table_name_emb is None or role_name_emb is None:
                raise ValueError(
                    "arch='two_level' needs P0.4's table_name_emb and role_name_emb (build them with "
                    "gloss.text.schema.build_table_name_embeddings / role_name_embeddings_with_none; "
                    "scripts/build_schema_cache.py caches them)."
                )
            # The cell-level MoE keeps routing on the CELL signature `z`, so its router input width is
            # d_sig for signature/identity and d_model for hidden/value — same contract as arch='rt'.
            cell_route_dim = d_sig if route_on in ("signature", "identity") else d_model
            self.substrate = TwoLevelSubstrate(
                d_model, d_ff or 4 * d_model, d_sig,
                table_name_emb, role_name_emb, name_emb,
                n_blocks=n_blocks, n_heads=n_heads, max_hop=max_hop,
                cell_ffn="dense" if route_on in ("dense", "dense_wide") else "moe",
                cell_route_dim=cell_route_dim,
                cell_num_experts=num_experts, cell_k=k,
                **two_level_kw,
            )
            self.head_mode = head_mode
            self.head = (RowTokenHead(d_model, out_dim) if head_mode == "row_token"
                         else EntityHead(d_model, out_dim))

    def forward(self, cb: CellBatch):
        out = self.encoder(cb, return_value=self.needs_value)
        x, value_feat = out if self.needs_value else (out, None)
        z = self.signature(cb) if self.signature is not None else None
        id_e = self.id_emb(cb.col_idxs.clamp(min=0)) if self.id_emb is not None else None

        if self.arch == "rt":
            x, aux = self.substrate(x, cb, z=z, value_feat=value_feat, id_emb=id_e)
            return self.head(x, cb), aux                          # [B, out_dim], scalar aux

        cells, rows, aux, diag = self.substrate(x, cb, z=z)
        # `head.mode: seed_cells` stays reachable so Phase 0 can attribute a delta to the READOUT
        # rather than to the row tokens themselves (§3.8).
        readout = rows if self.head_mode == "row_token" else cells
        return self.head(readout, cb), aux
