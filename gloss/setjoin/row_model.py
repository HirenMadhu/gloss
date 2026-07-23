"""RowModel: ONE hierarchical encoder over a per-seed SET of denormalized wide rows.

Retires SetJoin's wide/set split. Per seed the input is a set of "one big table" (OBT) rows (Row 0 =
the seed's own m2o closure; Rows 1..K = each direct child with the seed's closure repeated in). The
signature-routed ``MoEFFN`` (the one mechanism, reused via ``_MoEStack``/``MoELayer`` from ``model.py``)
is applied at TWO granularities:

    cell_grid  = encode + tag cells                         [B, M_rows, C, d]
    cell_enc   = within-row MHSA + MoE (cell signature)     [B, M_rows, C, d]   (checkpointed)
    row_emb    = CascadeCellPool(cells -> one row vector)   [B, M_rows, d]
    row_enc    = row-level MHSA + MoE (row signature)       [B, M_rows, d]
    agg        = mean over rows (default) | slot pool       [B, d]
    logits     = MLP( [agg (; counts)] )                    [B, out_dim]

``route_on`` arms (signature | hidden | dense) and the ``aux`` router-orthogonality balance are exactly
as in SetJoin. ``SetJoin`` and ``to_join_batch`` are untouched; this is additive.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.graph import FK_NONE, GraphBundle
from ..model.column_encoder import CellEncoder
from .collate import RowSetBatch
from .model import ROUTE_ARMS, RowPool, _MoEStack
from .paths import child_rels, row_paths
from .recency import N_RECENCY_BINS, recency_bins

AGGREGATES = ("mean", "slot")
ROW_POOLS = ("slot", "gated")


# --------------------------------------------------------------------------------------------------
# value-free routing signatures (grid-native analogues of WideSignature / ElemSignature)
# --------------------------------------------------------------------------------------------------
class RowCellSignature(nn.Module):
    """Value-free cell signature over the ``[B, M_rows, C]`` grid — the MoRE cell signature WITH the
    join-path term (unlike ``WideSignature.type_cells``, which drops it):

        z = RMSNorm( W_s·name_emb[col]·has_col + ψ(modality)·has_col + φ(recency) + π(row-path) )
    """

    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, n_paths: int, d_sig: int):
        super().__init__()
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.schema_proj = nn.Linear(int(name_emb.shape[1]), d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_sig)
        self.path_emb = nn.Embedding(max(n_paths, 1), d_sig)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, rb: RowSetBatch) -> Tensor:          # -> [B, M_rows, C, d_sig]
        col = rb.cell_col_id.clamp(min=0)
        B, M, C = col.shape
        name = self.name_emb.index_select(0, col.reshape(-1)).view(B, M, C, -1)
        mod = self.modality_id.index_select(0, col.reshape(-1)).view(B, M, C)
        has_col = (rb.cell_col_id >= 0).unsqueeze(-1).float()
        z = (self.schema_proj(name) + self.stype_emb(mod)) * has_col
        z = z + self.path_emb(rb.cell_path_id.clamp(min=0))
        z = z + self.recency_emb(recency_bins(rb.seed_time, rb.cell_row_time, rb.cell_is_timed))
        return self.norm(z)


class RowSignature(nn.Module):
    """Value-free row signature over ``[B, M_rows]`` — the ElemSignature analog, NO null prepend
    (RowModel has no null token; Row 0 is always real):

        z = RMSNorm( table type + FK role + φ(recency) + hop )
    """

    def __init__(self, num_node_types: int, num_fk_roles: int, d_sig: int):
        super().__init__()
        self.table_emb = nn.Embedding(num_node_types, d_sig)
        self.fk_emb = nn.Embedding(num_fk_roles, d_sig)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_sig)
        self.hop_emb = nn.Embedding(4, d_sig)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, rb: RowSetBatch) -> Tensor:          # -> [B, M_rows, d_sig]
        z = self.table_emb(rb.row_table_id.clamp(min=0)) + self.fk_emb(rb.row_fk_role) \
            + self.recency_emb(recency_bins(rb.seed_time, rb.row_row_time, rb.row_is_timed)) \
            + self.hop_emb(rb.row_hop.clamp(min=0, max=3))
        return self.norm(z)


# --------------------------------------------------------------------------------------------------
# cardinality-aware measure pool (a standalone reimplementation of SlotReadout's measure formula —
# NOT a reuse of SlotReadout, whose forward returns a combined [B, d]) and the C->k1->k2->1 cascade
# --------------------------------------------------------------------------------------------------
class MeasurePoolStage(nn.Module):
    """One measure-pool stage: ``n_query`` learned, sigmoid-gated queries pool ``T`` tokens and KEEP the
    count (mean stays recoverable as SUM/(COUNT+eps)). Input/output are 3-D ``[batch, T, d] ->
    [batch, n_query, d]``. Cardinality (``count``) is only informative when ``T`` varies across the
    batch (stage 1 over a row's real cells); later cascade stages pool a fixed count."""

    def __init__(self, d: int, n_query: int, channels=("mean", "count", "sum"),
                 seed_cond: bool = False):
        super().__init__()
        self.d, self.eps, self.channels = d, 1e-6, tuple(channels)
        self.query = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_seed = nn.Linear(d, d) if seed_cond else None
        n_vec = sum(1 for c in self.channels if c in ("mean", "sum"))
        self.w_o = nn.Linear(n_vec * d + (1 if "count" in self.channels else 0), d)
        self.norm = nn.RMSNorm(d)

    def forward(self, H: Tensor, valid: Tensor | None, seed_repr: Tensor | None = None) -> Tensor:
        q = self.query
        if self.w_seed is not None and seed_repr is not None:
            q = q + self.w_seed(seed_repr).unsqueeze(-2)               # [batch, n_query, d]
        content = q @ self.w_k(H).transpose(-1, -2) / (self.d ** 0.5)  # [batch, n_query, T]
        attn = torch.sigmoid(content)
        if valid is not None:
            attn = attn * valid.unsqueeze(-2).to(attn.dtype)
        V = self.w_v(H)
        m = attn.sum(dim=-1)                                          # [batch, n_query]  soft COUNT
        S = attn @ V                                                  # [batch, n_query, d]  SUM
        n_real = (valid.sum(-1) if valid is not None
                  else torch.full(m.shape[:-1] + (1,), H.shape[-2], device=H.device)).float()
        parts = []
        for c in self.channels:
            if c == "mean":
                parts.append(S / (m.unsqueeze(-1) + self.eps))
            elif c == "sum":
                parts.append(S / torch.log1p(n_real).clamp(min=self.eps).reshape(*m.shape[:-1], 1, 1))
            elif c == "count":
                parts.append(torch.log1p(m).unsqueeze(-1))
        return self.norm(self.w_o(torch.cat(parts, dim=-1)))


class CascadeCellPool(nn.Module):
    """``C -> k1 -> k2 -> 1`` cascade of measure-pool stages: pool a wide row's cells to one vector.
    ``widths = (k1, k2, ..., 1)``; the final width must be 1. Only the first stage is seed-conditionable
    and only its count channel sees a variable cardinality (a row's real-cell count)."""

    def __init__(self, d: int, widths, channels=("mean", "count", "sum"), seed_cond: bool = False):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        assert widths and widths[-1] == 1, f"cascade must end in width 1, got {widths}"
        self.stages = nn.ModuleList(
            MeasurePoolStage(d, w, channels, seed_cond=(seed_cond and i == 0))
            for i, w in enumerate(widths))

    def forward(self, H: Tensor, valid: Tensor | None, seed_repr: Tensor | None = None) -> Tensor:
        x = self.stages[0](H, valid, seed_repr)
        for st in self.stages[1:]:
            x = st(x, None)
        return x[..., 0, :]                                           # width-1 -> [..., d]


class RowModel(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb: Tensor,
        entity_table: str,
        *,
        d_model: int = 256,
        enc_channels: int | None = None,
        n_cell_layers: int = 2,
        n_row_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        route_on: str = "signature",
        num_experts: int = 4,
        k: int = 2,
        d_sig: int = 64,
        d_ff: int | None = None,
        use_shared: bool = False,
        aggregate: str = "mean",
        use_counts: bool = False,
        agg_slots: int = 4,
        row_pool: str = "slot",
        cell_slots=(8, 2),
        readout_channels=("mean", "count", "sum"),
        checkpoint_cells: bool = False,
    ):
        super().__init__()
        if route_on not in ROUTE_ARMS:
            raise ValueError(f"route_on must be one of {ROUTE_ARMS}, got {route_on!r}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        if row_pool not in ROW_POOLS:
            raise ValueError(f"row_pool must be one of {ROW_POOLS}, got {row_pool!r}")
        self.d_model = d_model
        self.entity_table = entity_table
        self.route_on = route_on
        self.aggregate = aggregate
        self.use_counts = bool(use_counts)
        self.row_pool_arm = row_pool
        self.encoder = CellEncoder(bundle, name_emb, d_model=d_model, enc_channels=enc_channels)

        rp = row_paths(bundle, entity_table, depth=2)
        self.n_child_rels = len(child_rels(bundle, entity_table))
        self.path_emb = nn.Embedding(rp.n_row_paths, d_model)
        self.table_emb = nn.Embedding(bundle.num_node_types, d_model)
        self.fk_emb = nn.Embedding(bundle.num_fk_roles, d_model)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_model)
        self.hop_emb = nn.Embedding(4, d_model)
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        if route_on == "signature":
            from ..text.schema import build_column_modality_ids

            modality_id, n_stypes = build_column_modality_ids(bundle)
            self.cell_sig = RowCellSignature(self.encoder.name_emb, modality_id, n_stypes,
                                             rp.n_row_paths, d_sig)
            self.row_sig = RowSignature(bundle.num_node_types, bundle.num_fk_roles, d_sig)
        else:
            self.cell_sig = self.row_sig = None
        d_route = d_sig if route_on == "signature" else d_model
        moe = dict(route_on=route_on, d_ff=d_ff or 2 * d_model, d_route=d_route,
                   num_experts=num_experts, k=k, use_shared=use_shared)

        self.cell_enc = _MoEStack(d_model, n_cell_layers, n_heads, dropout,
                                  checkpoint=checkpoint_cells, **moe)
        if row_pool == "slot":
            self.pool = CascadeCellPool(d_model, tuple(cell_slots) + (1,), readout_channels)
        else:
            self.pool = RowPool(d_model)
        self.row_norm = nn.LayerNorm(d_model)
        self.row_enc = _MoEStack(d_model, n_row_layers, n_heads, dropout, **moe)

        if aggregate == "slot":
            self.agg_pool = CascadeCellPool(d_model, (agg_slots, 1), readout_channels, seed_cond=True)
        self.w_cnt = nn.Linear(self.n_child_rels, d_model) if (use_counts and self.n_child_rels) else None
        head_in = d_model * (2 if self.w_cnt is not None else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, rb: RowSetBatch) -> tuple[Tensor, Tensor]:
        dev = self.encoder.name_emb.device
        B, M, C, d = rb.num_seeds, rb.m_rows, rb.cells_per_row, self.d_model

        # ---- Step A: encode each node type ONCE; scatter (gather) into the cell grid ----
        grid = torch.zeros(B, M, C, d, device=dev)
        for nt, tf in rb.tf_dict.items():
            if nt not in self.encoder.cell_encoders or nt not in rb.cell_placement:
                continue
            cell, _ = self.encoder.encode_type(nt, tf)                # [n, C_nt, d]
            b_i, m_i, c_i, r_i, col_i = rb.cell_placement[nt]
            grid[b_i, m_i, c_i] = cell[r_i, col_i].to(grid.dtype)     # .to() is a no-op off AMP
        rec = recency_bins(rb.seed_time, rb.cell_row_time, rb.cell_is_timed)
        grid = grid + self.path_emb(rb.cell_path_id.clamp(min=0)) + self.recency_emb(rec)
        marker = self.missing_emb.view(1, 1, 1, -1) + self.table_emb(rb.cell_table_id.clamp(min=0))
        grid = grid + rb.cell_missing.unsqueeze(-1).float() * marker
        grid = grid * rb.cell_mask.unsqueeze(-1).float()

        # ---- Step B: cell encoder over REAL rows only (compaction => no all-pad-query NaN) ----
        z_cell = self.cell_sig(rb) if self.cell_sig is not None else None
        rmask = rb.row_mask                                           # [B, M]
        real = grid[rmask]                                            # [n_real, C, d]
        cmask = rb.cell_mask[rmask]                                   # [n_real, C]
        z_real = z_cell[rmask] if z_cell is not None else None
        real = self.cell_enc(real, ~cmask, z_real)
        real = real * cmask.unsqueeze(-1).to(real.dtype)             # re-zero pad cells before pooling
        grid = grid.clone()
        grid[rmask] = real.to(grid.dtype)

        # ---- Step C: cell -> row pool ----
        real_rows = grid[rmask]                                       # [n_real, C, d]
        if self.row_pool_arm == "slot":
            pooled = self.pool(real_rows, cmask)                     # [n_real, d]
        else:
            pooled = self.pool(real_rows * cmask.unsqueeze(-1).float())
        row_emb = torch.zeros(B, M, d, device=dev)
        row_emb[rmask] = pooled.to(row_emb.dtype)

        # ---- Step D: row encoder (+ row tags, no null token) ----
        row_rec = recency_bins(rb.seed_time, rb.row_row_time, rb.row_is_timed)
        row_emb = row_emb + self.fk_emb(rb.row_fk_role) + self.table_emb(rb.row_table_id.clamp(min=0)) \
            + self.recency_emb(row_rec) + self.hop_emb(rb.row_hop.clamp(min=0, max=3))
        row_emb = self.row_norm(row_emb) * rmask.unsqueeze(-1).float()
        z_row = self.row_sig(rb) if self.row_sig is not None else None
        row_enc = self.row_enc(row_emb, ~rmask, z_row)               # [B, M, d]

        # ---- Step E: aggregate rows -> one prediction embedding ----
        if self.aggregate == "mean":
            agg = (row_enc * rmask.unsqueeze(-1).float()).sum(1) / rmask.sum(1, keepdim=True).clamp(min=1)
        else:                                                        # slot: seed-conditioned by Row 0
            agg = self.agg_pool(row_enc, rmask, seed_repr=row_enc[:, 0])
        if self.w_cnt is not None:
            agg = torch.cat([agg, self.w_cnt(torch.log1p(rb.child_counts))], dim=-1)

        logits = self.head(agg)
        aux = self.cell_enc.ortho_loss() + self.row_enc.ortho_loss()
        if not torch.is_tensor(aux):
            aux = logits.new_zeros(())                               # dense arm
        return logits, aux
