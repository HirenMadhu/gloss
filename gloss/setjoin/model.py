"""The SetJoin model: wide-row tabular transformer + seed-conditioned set transformer, with the
Mixture-of-Relational-Experts FFN inside every layer ("route on semantics, transform the content" on
the single-table substrate).

    seed_repr = WideEncoder( wide cells + path/recency tags, missing markers | z_wide )   # CLS readout
    E_n       = RowPool(child_n cells) + Σ_p RowPool(parent_p cells) + fk/table/Δt/hop tags
    context   = SetEncoder( {E_n} ∪ {null} | seed_repr, z_elem )                           # PMA readout
    logits    = MLP([seed_repr ; context ; W_cnt log1p(child_counts)])

Every transformer layer's FFN is MoRE's ``MoEFFN`` (reused verbatim from ``gloss/model/moe.py``): a
pool of SwiGLU experts whose top-k router reads a **value-free signature** while the experts transform
the evolving hidden state. Wide tokens ARE cells, so they route on the true MoRE signature — frozen-LM
column-name embedding + modality + recency (+ join path). Set elements are rows, so they route on the
row-level analog — table type + FK role + recency (+ hop). Balance is the router-orthogonality loss,
summed over layers and returned as ``aux``.

``route_on`` arms: ``signature`` (the method) | ``hidden`` (router reads the normed hidden state) |
``dense`` (plain FFN layers, aux=0 — the v2-gate baseline). Cell encoding reuses
``CellEncoder.encode_type`` verbatim; each sampled node type is encoded ONCE and shared between the
wide grid and the set elements. No positional encoding on the set path — permutation invariance of the
pooled context is a tested contract. ``table_emb`` on elements is load-bearing (``fk_role_id`` is keyed
by canonical FK column name and collides across tables).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.graph import GraphBundle
from ..model.column_encoder import CellEncoder
from ..model.moe import MoEFFN
from .collate import JoinBatch
from .paths import child_rels, m2o_paths
from .recency import N_RECENCY_BINS, recency_bins

ROUTE_ARMS = ("signature", "hidden", "dense")


class RowPool(nn.Module):
    """Gated attention pool over a row's C cells -> one row vector (shared across node types; the
    frozen name tokens inside the cells already disambiguate columns and tables)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, 1)

    def forward(self, cells: Tensor) -> Tensor:            # [n, C, d] -> [n, d]
        att = self.w2(torch.tanh(self.w1(cells))).softmax(dim=1)
        return (att * cells).sum(dim=1)


def _encoder_layer(d_model: int, n_heads: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )


class MoELayer(nn.Module):
    """Pre-norm attention + MoE-FFN layer (the SetJoin analog of MoRE's RelationalBlock): the router
    reads the value-free per-token signature ``z`` (or the normed hidden state for the ``hidden`` arm);
    the experts transform the hidden state."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_route: int, *,
                 num_experts: int = 4, k: int = 2, dropout: float = 0.1,
                 route_on: str = "signature"):
        super().__init__()
        self.route_on = route_on
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MoEFFN(d_model, d_ff, d_route, num_experts=num_experts, k=k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, z: Tensor | None, key_padding_mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        y, _ = self.ffn(h, z if self.route_on == "signature" else h)
        return x + self.drop(y)


class _MoEStack(nn.Module):
    """A stack of dense TransformerEncoder layers (``route_on='dense'``) or MoELayers, one interface."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, *,
                 route_on: str, d_ff: int, d_route: int, num_experts: int, k: int):
        super().__init__()
        self.route_on = route_on
        if route_on == "dense":
            self.layers = nn.TransformerEncoder(_encoder_layer(d_model, n_heads, dropout), n_layers,
                                                enable_nested_tensor=False)
        else:
            self.layers = nn.ModuleList(
                MoELayer(d_model, n_heads, d_ff, d_route, num_experts=num_experts, k=k,
                         dropout=dropout, route_on=route_on)
                for _ in range(n_layers))

    def forward(self, x: Tensor, pad: Tensor, z: Tensor | None) -> Tensor:
        if self.route_on == "dense":
            return self.layers(x, src_key_padding_mask=pad)
        for lyr in self.layers:
            x = lyr(x, z, pad)
        return x

    def ortho_loss(self) -> Tensor | float:
        if self.route_on == "dense":
            return 0.0
        return sum(lyr.ffn.ortho_loss() for lyr in self.layers)


class WideSignature(nn.Module):
    """Value-free routing signature for the wide tokens — the MoRE cell signature on JoinBatch fields:

        z = RMSNorm( W_s·name_emb[col] + ψ(modality) + φ(recency) + π(join path) )

    Markers/pads keep only path+recency (no name/modality — there is no column). A learned CLS
    signature is prepended to match the encoder's CLS token. Nothing value-dependent enters z.
    """

    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, n_paths: int, d_sig: int):
        super().__init__()
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.schema_proj = nn.Linear(int(name_emb.shape[1]), d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_sig)
        self.path_emb = nn.Embedding(max(n_paths, 1), d_sig)
        self.cls_sig = nn.Parameter(torch.randn(1, 1, d_sig) * 0.02)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, jb: JoinBatch) -> Tensor:            # -> [B, W+1, d_sig]
        col = jb.wide_col_idxs.clamp(min=0)
        B, W = col.shape
        flat = col.reshape(-1)
        name = self.name_emb.index_select(0, flat).view(B, W, -1)
        mod = self.modality_id.index_select(0, flat).view(B, W)
        has_col = (jb.wide_col_idxs >= 0).unsqueeze(-1).float()
        z = (self.schema_proj(name) + self.stype_emb(mod)) * has_col
        z = z + self.path_emb(jb.wide_path_idxs.clamp(min=0))
        z = z + self.recency_emb(recency_bins(jb.seed_time, jb.wide_row_time, jb.wide_is_timed))
        z = self.norm(z)
        return torch.cat([self.cls_sig.expand(B, -1, -1), z], dim=1)

    def type_cells(self, col_gids: Tensor, rec_bins: Tensor) -> Tensor:
        """Value-free cell signatures for one node type's TensorFrame rows -> ``[n, C, d_sig]``
        (same signature space as the wide tokens; no path term — these are set-side cells).
        ``col_gids [C]`` = global column ids, ``rec_bins [n]`` = per-row Δt recency bins."""
        name = self.name_emb.index_select(0, col_gids)                    # [C, d_text]
        mod = self.modality_id.index_select(0, col_gids)                  # [C]
        z = (self.schema_proj(name) + self.stype_emb(mod)).unsqueeze(0) \
            + self.recency_emb(rec_bins).unsqueeze(1)                     # [n, C, d_sig]
        return self.norm(z)


class ElemSignature(nn.Module):
    """Value-free routing signature for set elements — the row-level signature analog:

        z = RMSNorm( table type + FK role + φ(recency) + hop )

    A learned null-element signature is prepended to match the encoder's null token.
    """

    def __init__(self, num_node_types: int, num_fk_roles: int, d_sig: int):
        super().__init__()
        self.table_emb = nn.Embedding(num_node_types, d_sig)
        self.fk_emb = nn.Embedding(num_fk_roles, d_sig)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_sig)
        self.hop_emb = nn.Embedding(4, d_sig)
        self.null_sig = nn.Parameter(torch.randn(1, 1, d_sig) * 0.02)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, jb: JoinBatch) -> Tensor:            # -> [B, N+1, d_sig]
        z = self.table_emb(jb.elem_table_idxs) + self.fk_emb(jb.elem_rel_idxs) \
            + self.recency_emb(recency_bins(jb.seed_time, jb.elem_row_time, jb.elem_is_timed)) \
            + self.hop_emb(jb.elem_hop.clamp(min=0, max=3))
        z = self.norm(z)
        return torch.cat([self.null_sig.expand(z.shape[0], -1, -1), z], dim=1)


class AxialCellBlock(nn.Module):
    """One axial block over a per-type cell grid ``[B, R, C, d]`` (R = the seed's sampled rows of this
    table, C = its columns): **row-level** attention (cells of one row attend across columns), then
    **column-level** attention (same column attends across the seed's rows — RT's same-column pattern),
    then the (MoE-)FFN. Pre-norm residuals. Pad positions are re-zeroed via ``torch.where`` after every
    sublayer: fully-masked pad queries produce NaN in SDPA, and a NaN value behind a zero attention
    weight still poisons the weighted sum, so pads must never carry NaN into the next sublayer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_route: int, *,
                 num_experts: int = 4, k: int = 2, dropout: float = 0.1,
                 route_on: str = "signature"):
        super().__init__()
        self.route_on = route_on
        self.norm_r = nn.LayerNorm(d_model)
        self.attn_r = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(d_model)
        self.attn_c = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        if route_on == "dense":
            self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        else:
            self.ffn = MoEFFN(d_model, d_ff, d_route, num_experts=num_experts, k=k)
        self.drop = nn.Dropout(dropout)

    def forward(self, grid: Tensor, valid: Tensor, z: Tensor | None) -> Tensor:
        B, R, C, d = grid.shape
        keep = valid.unsqueeze(-1).unsqueeze(-1)                          # [B, R, 1, 1]
        zero = grid.new_zeros(())
        # row-level: attend across the C columns of each row
        h = self.norm_r(grid).reshape(B * R, C, d)
        kpm_r = (~valid).reshape(B * R, 1).expand(B * R, C)
        a, _ = self.attn_r(h, h, h, key_padding_mask=kpm_r, need_weights=False)
        grid = torch.where(keep, grid + self.drop(a.view(B, R, C, d)), zero)
        # column-level: attend across the seed's R rows within each column
        h = self.norm_c(grid).permute(0, 2, 1, 3).reshape(B * C, R, d)
        kpm_c = (~valid).unsqueeze(1).expand(B, C, R).reshape(B * C, R)
        a, _ = self.attn_c(h, h, h, key_padding_mask=kpm_c, need_weights=False)
        a = a.view(B, C, R, d).permute(0, 2, 1, 3)
        grid = torch.where(keep, grid + self.drop(a), zero)
        # (MoE-)FFN
        h = self.norm_f(grid)
        if self.route_on == "dense":
            y = self.ffn(h)
        else:
            y, _ = self.ffn(h, z if self.route_on == "signature" else h)
        return torch.where(keep, grid + self.drop(y), zero)


class AxialCellEncoder(nn.Module):
    """Shared across node types: scatter one type's cells ``[n, C, d]`` into per-seed grids
    ``[B, R, C, d]`` (disjoint sampling ⇒ every row copy belongs to exactly one seed), run
    ``n_layers`` axial blocks, gather back. This supplies RT's same-row and same-column interaction
    patterns at cell granularity — the set-level attention downstream supplies the cross-table one."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, *,
                 route_on: str, d_ff: int, d_route: int, num_experts: int, k: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            AxialCellBlock(d_model, n_heads, d_ff, d_route, num_experts=num_experts, k=k,
                           dropout=dropout, route_on=route_on)
            for _ in range(n_layers))

    def forward(self, cell: Tensor, seg: Tensor, num_seeds: int, z: Tensor | None) -> Tensor:
        n, C, d = cell.shape
        order = torch.argsort(seg, stable=True)
        sseg = seg[order]
        counts = torch.bincount(sseg, minlength=num_seeds)
        R = int(counts.max())
        offs = torch.cumsum(counts, 0) - counts
        slot = torch.arange(n, device=cell.device) - offs[sseg]
        grid = cell.new_zeros(num_seeds, R, C, d)
        grid[sseg, slot] = cell[order]
        valid = torch.zeros(num_seeds, R, dtype=torch.bool, device=cell.device)
        valid[sseg, slot] = True
        zg = None
        if z is not None:
            zg = z.new_zeros(num_seeds, R, C, z.shape[-1])
            zg[sseg, slot] = z[order]
        for blk in self.blocks:
            grid = blk(grid, valid, zg)
        out = torch.empty_like(cell)
        out[order] = grid[sseg, slot]
        return out

    def ortho_loss(self) -> Tensor | float:
        return sum((blk.ffn.ortho_loss() for blk in self.blocks
                    if isinstance(blk.ffn, MoEFFN)), 0.0)


class WideEncoder(nn.Module):
    """Wide cell grid ``[B, W, d]`` (+ pad mask) -> ``seed_repr [B, d]`` via a prepended CLS token.

    The CLS is never key-masked, so a degenerate all-pad wide row still yields a finite readout.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, *,
                 route_on: str = "dense", d_ff: int | None = None, d_route: int | None = None,
                 num_experts: int = 4, k: int = 2):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.stack = _MoEStack(d_model, n_layers, n_heads, dropout, route_on=route_on,
                               d_ff=d_ff or 2 * d_model, d_route=d_route or d_model,
                               num_experts=num_experts, k=k)

    def forward(self, h: Tensor, is_pad: Tensor, z: Tensor | None = None) -> Tensor:
        B = h.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h.device), is_pad], dim=1)
        return self.stack(x, pad, z)[:, 0]

    def ortho_loss(self):
        return self.stack.ortho_loss()


class SetEncoder(nn.Module):
    """Element set ``[B, N, d]`` (+ mask) -> pooled ``context [B, d]``.

    A learned ``null_elem`` is concatenated and never masked (attention always has ≥1 key; empty sets
    stay well-defined and trainable). ``n_layers`` self-attention layers contextualize the set (no
    positional encoding — permutation invariance is a contract), then a seed-conditioned PMA readout:
    ``n_pma`` learned queries, each shifted by a projection of ``seed_repr``, cross-attend into the set.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, n_pma: int, dropout: float, *,
                 route_on: str = "dense", d_ff: int | None = None, d_route: int | None = None,
                 num_experts: int = 4, k: int = 2):
        super().__init__()
        self.null_elem = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.stack = _MoEStack(d_model, n_layers, n_heads, dropout, route_on=route_on,
                               d_ff=d_ff or 2 * d_model, d_route=d_route or d_model,
                               num_experts=num_experts, k=k)
        self.pma_emb = nn.Parameter(torch.randn(1, n_pma, d_model) * 0.02)
        self.w_q = nn.Linear(d_model, d_model)
        self.pma = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(n_pma * d_model, d_model)

    def forward(self, E: Tensor, mask: Tensor, seed_repr: Tensor, z: Tensor | None = None) -> Tensor:
        B = E.shape[0]
        x = torch.cat([self.null_elem.expand(B, -1, -1), E], dim=1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=E.device), ~mask], dim=1)
        x = self.stack(x, kpm, z)
        q = self.pma_emb.expand(B, -1, -1) + self.w_q(seed_repr).unsqueeze(1)
        ctx, _ = self.pma(q, x, x, key_padding_mask=kpm, need_weights=False)
        return self.out(ctx.reshape(B, -1))

    def ortho_loss(self):
        return self.stack.ortho_loss()


class SetJoin(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb: Tensor,
        entity_table: str,
        *,
        d_model: int = 128,
        enc_channels: int | None = None,
        n_wide_layers: int = 2,
        n_set_layers: int = 2,
        n_heads: int = 4,
        n_pma: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        route_on: str = "signature",
        num_experts: int = 4,
        k: int = 2,
        d_sig: int = 64,
        d_ff: int | None = None,
        n_axial_layers: int = 0,
    ):
        super().__init__()
        if route_on not in ROUTE_ARMS:
            raise ValueError(f"route_on must be one of {ROUTE_ARMS}, got {route_on!r}")
        self.d_model = d_model
        self.entity_table = entity_table
        self.route_on = route_on
        self.encoder = CellEncoder(bundle, name_emb, d_model=d_model, enc_channels=enc_channels)

        n_paths = len(m2o_paths(bundle, entity_table, depth=2))
        self.n_child_rels = len(child_rels(bundle, entity_table))
        self.path_emb = nn.Embedding(n_paths, d_model)
        self.fk_emb = nn.Embedding(bundle.num_fk_roles, d_model)
        self.table_emb = nn.Embedding(bundle.num_node_types, d_model)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_model)
        self.hop_emb = nn.Embedding(4, d_model)
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        # value-free routing signatures (signature arm only; hidden routes on the normed state)
        if route_on == "signature":
            from ..text.schema import build_column_modality_ids

            modality_id, n_stypes = build_column_modality_ids(bundle)
            self.wide_sig = WideSignature(self.encoder.name_emb, modality_id, n_stypes,
                                          n_paths, d_sig)
            self.elem_sig = ElemSignature(bundle.num_node_types, bundle.num_fk_roles, d_sig)
        else:
            self.wide_sig = self.elem_sig = None
        d_route = d_sig if route_on == "signature" else d_model

        moe = dict(route_on=route_on, d_ff=d_ff, d_route=d_route, num_experts=num_experts, k=k)
        self.row_pool = RowPool(d_model)
        self.wide_enc = WideEncoder(d_model, n_wide_layers, n_heads, dropout, **moe)
        self.set_enc = SetEncoder(d_model, n_set_layers, n_heads, n_pma, dropout, **moe)
        self.elem_norm = nn.LayerNorm(d_model)
        self.w_cnt = nn.Linear(self.n_child_rels, d_model) if self.n_child_rels else None
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )
        # created LAST so the RNG draw order of every pre-existing module is unchanged when
        # n_axial_layers=0 (keeps v2/v3 arms bit-reproducible under the same seed)
        if n_axial_layers > 0:
            self.axial = AxialCellEncoder(d_model, n_axial_layers, n_heads, dropout,
                                          route_on=route_on, d_ff=d_ff or 2 * d_model,
                                          d_route=d_route, num_experts=num_experts, k=k)
        else:
            self.axial = None

    def forward(self, jb: JoinBatch) -> tuple[Tensor, Tensor]:
        dev = self.encoder.name_emb.device
        B, W, N, d = jb.num_seeds, jb.wide_len, jb.set_size, self.d_model

        # ---- encode each sampled node type ONCE; share between wide grid and set elements ----
        h = torch.zeros(B, W, d, device=dev)
        row_embs: dict[str, Tensor] = {}
        for nt, tf in jb.tf_dict.items():
            in_wide, in_set = nt in jb.wide_placement, nt in jb.elem_rows
            if nt not in self.encoder.cell_encoders or not (in_wide or in_set):
                continue
            cell, _ = self.encoder.encode_type(nt, tf)                  # [n, C, d]
            if self.axial is not None:                                  # row+column cell attention
                seg_t = jb.row_seg[nt]
                z_t = None
                if self.wide_sig is not None:
                    gids = self.encoder._col_gids[nt].to(dev)
                    bins = recency_bins(jb.seed_time[seg_t], jb.row_times[nt], jb.row_timed[nt])
                    z_t = self.wide_sig.type_cells(gids, bins)
                cell = self.axial(cell, seg_t, B, z_t)
            if in_wide:
                b_idx, w_idx, row_idx, col_idx = jb.wide_placement[nt]
                h[b_idx, w_idx] = cell[row_idx, col_idx]
            if in_set:
                row_embs[nt] = self.row_pool(cell)                      # [n, d]

        # ---- wide row: + path/recency tags; markers get missing_emb + the missing table's tag ----
        wide_rec = recency_bins(jb.seed_time, jb.wide_row_time, jb.wide_is_timed)
        h = h + self.path_emb(jb.wide_path_idxs.clamp(min=0)) + self.recency_emb(wide_rec)
        marker = self.missing_emb.view(1, 1, -1) + self.table_emb(jb.wide_table_idxs.clamp(min=0))
        h = h + jb.wide_missing.unsqueeze(-1).float() * marker
        h = h * (~jb.wide_is_pad).unsqueeze(-1).float()
        z_w = self.wide_sig(jb) if self.wide_sig is not None else None
        seed_repr = self.wide_enc(h, jb.wide_is_pad, z_w)               # [B, d]

        # ---- set elements: additive row scatter (child @ FK_NONE + flattened parents @ fk role) ----
        E = torch.zeros(B, N, d, device=dev)
        for nt, (b_idx, n_idx, row_idx, path_idx) in jb.elem_rows.items():
            if nt not in row_embs:
                continue
            E.index_put_((b_idx, n_idx), row_embs[nt][row_idx] + self.fk_emb(path_idx),
                         accumulate=True)
        elem_rec = recency_bins(jb.seed_time, jb.elem_row_time, jb.elem_is_timed)
        E = E + self.fk_emb(jb.elem_rel_idxs) + self.table_emb(jb.elem_table_idxs) \
            + self.recency_emb(elem_rec) + self.hop_emb(jb.elem_hop.clamp(min=0, max=3))
        E = self.elem_norm(E) * jb.elem_mask.unsqueeze(-1).float()
        z_e = self.elem_sig(jb) if self.elem_sig is not None else None
        context = self.set_enc(E, jb.elem_mask, seed_repr, z_e)         # [B, d]

        # ---- head ----
        if self.w_cnt is not None:
            cnt = self.w_cnt(torch.log1p(jb.child_counts))
        else:
            cnt = torch.zeros(B, d, device=dev)
        logits = self.head(torch.cat([seed_repr, context, cnt], dim=-1))
        aux = self.wide_enc.ortho_loss() + self.set_enc.ortho_loss()    # router orthogonality balance
        if self.axial is not None:
            aux = aux + self.axial.ortho_loss()
        if not torch.is_tensor(aux):
            aux = logits.new_zeros(())                                  # dense arm
        return logits, aux
