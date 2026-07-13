"""The SetJoin model: wide-row tabular transformer + seed-conditioned set transformer.

    seed_repr = WideEncoder( wide cells + path/recency tags, missing markers )        # CLS readout
    E_n       = RowPool(child_n cells) + Σ_p RowPool(parent_p cells) + fk/table/Δt/hop tags
    context   = SetEncoder( {E_n} ∪ {null_elem} | seed_repr )                          # PMA readout
    logits    = MLP([seed_repr ; context ; W_cnt log1p(child_counts)])

Cell encoding reuses ``CellEncoder.encode_type`` verbatim (frozen-LM name token + pytorch-frame stype
value encoder); each sampled node type is encoded ONCE and shared between the wide grid (cell-granular
scatter) and the set elements (row-pooled additive scatter). The set path uses no positional encoding
anywhere — permutation invariance of the pooled context is a tested contract. ``table_emb`` on elements
is load-bearing, not decorative: ``fk_role_id`` is keyed by canonical FK *column name* and collides
across tables. ``aux`` is always 0 — kept so the ``forward -> (logits, aux)`` contract matches MoRE and
the training scaffolding forks stay thin.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.graph import GraphBundle
from ..model.column_encoder import CellEncoder
from .collate import JoinBatch
from .paths import child_rels, m2o_paths
from .recency import N_RECENCY_BINS, recency_bins


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


class WideEncoder(nn.Module):
    """Wide cell grid ``[B, W, d]`` (+ pad mask) -> ``seed_repr [B, d]`` via a prepended CLS token.

    The CLS is never key-masked, so a degenerate all-pad wide row still yields a finite readout.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.TransformerEncoder(_encoder_layer(d_model, n_heads, dropout), n_layers,
                                            enable_nested_tensor=False)

    def forward(self, h: Tensor, is_pad: Tensor) -> Tensor:
        B = h.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h.device), is_pad], dim=1)
        return self.layers(x, src_key_padding_mask=pad)[:, 0]


class SetEncoder(nn.Module):
    """Element set ``[B, N, d]`` (+ mask) -> pooled ``context [B, d]``.

    A learned ``null_elem`` is concatenated and never masked (attention always has ≥1 key; empty sets
    stay well-defined and trainable). ``n_layers`` self-attention layers contextualize the set (no
    positional encoding — permutation invariance is a contract), then a seed-conditioned PMA readout:
    ``n_pma`` learned queries, each shifted by a projection of ``seed_repr``, cross-attend into the set.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, n_pma: int, dropout: float):
        super().__init__()
        self.null_elem = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.TransformerEncoder(_encoder_layer(d_model, n_heads, dropout), n_layers,
                                            enable_nested_tensor=False)
        self.pma_emb = nn.Parameter(torch.randn(1, n_pma, d_model) * 0.02)
        self.w_q = nn.Linear(d_model, d_model)
        self.pma = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(n_pma * d_model, d_model)

    def forward(self, E: Tensor, mask: Tensor, seed_repr: Tensor) -> Tensor:
        B = E.shape[0]
        x = torch.cat([self.null_elem.expand(B, -1, -1), E], dim=1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=E.device), ~mask], dim=1)
        x = self.layers(x, src_key_padding_mask=kpm)
        q = self.pma_emb.expand(B, -1, -1) + self.w_q(seed_repr).unsqueeze(1)
        ctx, _ = self.pma(q, x, x, key_padding_mask=kpm, need_weights=False)
        return self.out(ctx.reshape(B, -1))


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
    ):
        super().__init__()
        self.d_model = d_model
        self.entity_table = entity_table
        self.encoder = CellEncoder(bundle, name_emb, d_model=d_model, enc_channels=enc_channels)

        n_paths = len(m2o_paths(bundle, entity_table, depth=2))
        self.n_child_rels = len(child_rels(bundle, entity_table))
        self.path_emb = nn.Embedding(n_paths, d_model)
        self.fk_emb = nn.Embedding(bundle.num_fk_roles, d_model)
        self.table_emb = nn.Embedding(bundle.num_node_types, d_model)
        self.recency_emb = nn.Embedding(N_RECENCY_BINS, d_model)
        self.hop_emb = nn.Embedding(4, d_model)
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        self.row_pool = RowPool(d_model)
        self.wide_enc = WideEncoder(d_model, n_wide_layers, n_heads, dropout)
        self.set_enc = SetEncoder(d_model, n_set_layers, n_heads, n_pma, dropout)
        self.elem_norm = nn.LayerNorm(d_model)
        self.w_cnt = nn.Linear(self.n_child_rels, d_model) if self.n_child_rels else None
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

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
        seed_repr = self.wide_enc(h, jb.wide_is_pad)                    # [B, d]

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
        context = self.set_enc(E, jb.elem_mask, seed_repr)              # [B, d]

        # ---- head ----
        if self.w_cnt is not None:
            cnt = self.w_cnt(torch.log1p(jb.child_counts))
        else:
            cnt = torch.zeros(B, d, device=dev)
        logits = self.head(torch.cat([seed_repr, context, cnt], dim=-1))
        return logits, logits.new_zeros(())
