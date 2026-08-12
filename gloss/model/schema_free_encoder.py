"""A cell encoder with **no learned parameter sized by any particular database**.

The RT cell token is unchanged::

    x_{u,c} = W_v · Enc_dtype(v_{u,c})  +  W_name · name_c

What changes is only the first term. ``CellEncoder`` delegates it to torch_frame's stype encoders,
which keep **per-column** weights: one ``Linear(d_text, enc)`` for every text column (47 of them on
rel-trial, 10.25M parameters), one learned vector per (column, category), a per-column
``weight[n_cols, out]`` for numerics, and a per-column ``weight[n_cols, 7, out_size, enc]`` for
timestamps. None of that can transfer — column 12 of rel-f1 is not column 12 of rel-stack — so a
pretrained model needs 10.6M of per-database weights re-initialised on every new schema.

Here each stype gets **one shared projection**, exactly as RT does (``W_d`` is datatype-specific, not
column-specific), and everything that must remain per-column becomes frozen **data** that any new
database regenerates without gradients:

===============  ==============================================  ==========================
stype            value -> enc_channels                            per-DB piece (frozen data)
===============  ==============================================  ==========================
numerical        ``(v - mu_c)/sigma_c`` -> shared ``Linear(1,e)``  mu_c, sigma_c (col_stats)
categorical      frozen label embedding -> shared ``Linear(dt,e)`` the label table + offsets
text/embedding   frozen cell embedding -> shared ``Linear(dv,e)``  none
timestamp        (year - mu)/sigma + cyclic -> shared projection   ONE global mu/sigma
===============  ==============================================  ==========================

The consequence to be aware of: once every numerical column shares one ``Linear(1, e)``, the *only*
thing distinguishing ``price of product`` from ``age of customer`` is ``W_name · name_c``. That is
RT's bet — the value carries the magnitude, the frozen name carries what it is the magnitude *of* —
and it is the same signal the MoE router already routes on, so the architecture gets more internally
consistent rather than less. It is still a real capacity reduction and is meant to be measured
against :class:`~gloss.model.column_encoder.CellEncoder`, not assumed.

**Timestamp normalization is global**, not per column. ``TimestampEncoder`` subtracts each column's
own ``YEAR_RANGE[0]``; with a shared projection that would feed inconsistent scales through one
weight, so year is standardized by a single mean/std over every timestamp in the database (RT
likewise "normalize[s] globally ... the global mean and standard deviation of timestamps"). The other
six fields are calendar-cyclic and already schema-free.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import CellBatch, column_vocab, feature_col_names
from ..text.schema import N_STYPES, build_column_modality_ids, category_index

#: (year, month, day, dayofweek, hour, minute, second) — torch_frame's `TimestampTensorMapper` layout.
N_TIME_FIELDS = 7
#: Calendar periods for the six cyclic fields; constants of the Gregorian calendar, not of any DB.
CYCLIC_PERIODS = (12.0, 31.0, 7.0, 24.0, 60.0, 60.0)


def _frozen(module: nn.Module, name: str, t: Tensor) -> None:
    """Non-persistent, so no `state_dict` entry is sized by C, n_categories or n_tables."""
    module.register_buffer(name, t.detach(), persistent=False)


class SchemaFreeCellEncoder(nn.Module):
    """Drop-in alternative to :class:`~gloss.model.column_encoder.CellEncoder`, zero per-DB weights.

    ``cat_emb`` is the frozen category-label table from
    :func:`gloss.text.schema.build_category_name_embeddings`; pass ``None`` on a database with no
    categorical columns.
    """

    def __init__(self, bundle, name_emb: Tensor, cat_emb: Tensor | None = None, *,
                 d_model: int = 256, enc_channels: int | None = None, time_out_size: int = 8):
        super().__init__()
        from torch_frame.data.stats import StatType

        self.d_model = d_model
        self.d_text = int(name_emb.shape[1])
        self.enc_channels = enc_channels or d_model
        e = self.enc_channels
        self.time_out_size = time_out_size

        vocab = column_vocab(bundle)
        C = len(vocab)
        cat_ix, _texts = category_index(bundle)

        # ---- per-column frozen DATA (regenerable on any schema, no gradients) ----
        num_mean = torch.zeros(C)
        num_std = torch.ones(C)
        cat_base = torch.zeros(C, dtype=torch.long)
        years: list[float] = []
        for nt in bundle.node_types:
            tf = bundle.data[nt].tf
            for st, cols in tf.col_names_dict.items():
                nm = str(st).rsplit(".", 1)[-1]
                for c in cols:
                    gid = vocab.get((nt, c))
                    if gid is None:
                        continue
                    stats = bundle.col_stats_dict[nt][c]
                    if nm == "numerical":
                        num_mean[gid] = float(stats.get(StatType.MEAN, 0.0))
                        num_std[gid] = float(stats.get(StatType.STD, 1.0)) + 1e-6
                    elif nm == "categorical":
                        cat_base[gid] = cat_ix[(nt, c)][0]
                    elif nm == "timestamp":
                        yr = stats.get(StatType.YEAR_RANGE)
                        if yr is not None:
                            years.extend(float(y) for y in yr)
        # ONE global year normalizer for the whole database (see the module docstring).
        if years:
            y = torch.tensor(years, dtype=torch.float32)
            year_mean, year_std = float(y.mean()), float(y.std()) + 1e-6
        else:
            year_mean, year_std = 2000.0, 1.0
        _frozen(self, "num_mean", num_mean)
        _frozen(self, "num_std", num_std)
        _frozen(self, "cat_base", cat_base)
        _frozen(self, "year_stats", torch.tensor([year_mean, year_std]))
        _frozen(self, "name_emb", name_emb.detach().to(torch.float32))
        _frozen(self, "cat_emb", (cat_emb if cat_emb is not None
                                  else torch.zeros(1, self.d_text)).to(torch.float32))
        _frozen(self, "cyclic_periods", torch.tensor(CYCLIC_PERIODS))
        modality_id, _ = build_column_modality_ids(bundle)
        _frozen(self, "modality_id", modality_id.to(torch.long))

        # column layout per node type: which slot of `feature_col_names` each stype's block occupies
        self._layout: dict[str, dict[str, list[int]]] = {}
        self._gids: dict[str, Tensor] = {}
        for nt in bundle.node_types:
            tf = bundle.data[nt].tf
            sorted_cols = feature_col_names(tf)
            self._gids[nt] = torch.tensor([vocab[(nt, c)] for c in sorted_cols], dtype=torch.long)
            per: dict[str, list[int]] = {}
            for st, cols in tf.col_names_dict.items():
                per[str(st).rsplit(".", 1)[-1]] = [sorted_cols.index(c) for c in cols]
            self._layout[nt] = per

        d_val = int(self.cat_emb.shape[-1])
        self.d_value_text = _value_text_dim(bundle) or d_val
        # ---- the shared, portable projections: ONE per datatype ----
        self.proj_num = nn.Linear(1, e)
        self.proj_cat = nn.Linear(d_val, e)
        self.proj_text = nn.Linear(self.d_value_text, e)
        self.time_weight = nn.Parameter(torch.randn(N_TIME_FIELDS, time_out_size, e) * 0.01)
        self.time_bias = nn.Parameter(torch.zeros(e))

        self.w_v = nn.Linear(e, d_model)
        self.name_proj = nn.Linear(self.d_text, d_model)
        self.mask_emb = nn.Embedding(N_STYPES, d_model)

    # ------------------------------------------------------------------ per-stype value encoders
    def _encode_numerical(self, feat: Tensor, gids: Tensor) -> Tensor:
        z = (feat - self.num_mean[gids]) / self.num_std[gids]           # [n, C] frozen per-col stats
        return self.proj_num(torch.nan_to_num(z).unsqueeze(-1))          # -> [n, C, e], ONE weight

    def _encode_categorical(self, feat: Tensor, gids: Tensor) -> Tensor:
        # feat is the class index, -1 for NaN; row 0 of the frozen table is the unknown slot.
        idx = torch.where(feat < 0, torch.zeros_like(feat), feat + self.cat_base[gids].unsqueeze(0))
        return self.proj_cat(self.cat_emb[idx.clamp(0, self.cat_emb.shape[0] - 1)])

    def _encode_text(self, feat) -> Tensor:
        v = feat.values.view(feat.num_rows, -1, self.d_value_text) if hasattr(feat, "values") else feat
        return self.proj_text(v.to(self.proj_text.weight.dtype))

    def _encode_timestamp(self, feat: Tensor) -> Tensor:
        """``[n, C, 7]`` -> ``[n, C, e]``. Year standardized GLOBALLY; the rest calendar-cyclic."""
        f = feat.to(torch.float32)
        mean, std = self.year_stats[0], self.year_stats[1]
        year = ((f[..., :1] - mean) / std).clamp(-8.0, 8.0)              # one normalizer, whole DB
        ang = 2 * torch.pi * f[..., 1:] / self.cyclic_periods
        # [n, C, 7, out_size]: year as a ramp, the six cyclic fields as (sin, cos, harmonics)
        k = torch.arange(1, self.time_out_size + 1, device=f.device, dtype=f.dtype)
        year_feat = year.unsqueeze(-1) * k
        cyc = torch.where((k.long() % 2 == 0), torch.cos(ang.unsqueeze(-1) * ((k + 1) // 2)),
                          torch.sin(ang.unsqueeze(-1) * ((k + 1) // 2)))
        parts = torch.cat([year_feat, cyc], dim=-2)                      # [n, C, 7, out_size]
        return torch.einsum("ncfo,foe->nce", parts, self.time_weight) + self.time_bias

    # --------------------------------------------------------------------------------- forward
    def encode_type(self, nt: str, tf) -> Tensor:
        """All rows of one node type -> ``[n, C, d_model]`` in ``feature_col_names`` order."""
        import torch_frame

        gids = self._gids[nt].to(self.name_emb.device)
        n = tf.num_rows
        out = torch.zeros(n, gids.numel(), self.enc_channels, device=self.name_emb.device)
        handlers = {
            "numerical": (torch_frame.numerical, self._encode_numerical, True),
            "categorical": (torch_frame.categorical, self._encode_categorical, True),
            "embedding": (torch_frame.embedding, self._encode_text, False),
            "timestamp": (torch_frame.timestamp, self._encode_timestamp, False),
        }
        for name, pos in self._layout[nt].items():
            spec = handlers.get(name)
            if spec is None or not pos:
                continue                     # multicategorical & friends: no shared encoder yet
            st, fn, needs_gids = spec
            feat = tf.feat_dict.get(st)
            if feat is None:
                continue
            idx = torch.tensor(pos, device=out.device, dtype=torch.long)
            block = fn(feat, gids[idx]) if needs_gids else fn(feat)
            out[:, idx, :] = block.to(out.dtype)
        name = self.name_emb.index_select(0, gids)
        return self.w_v(out) + self.name_proj(name).unsqueeze(0)

    def masked_token(self, cb: CellBatch) -> Tensor:
        gid = cb.col_idxs.clamp_min(0)
        B, S = gid.shape
        flat = gid.reshape(-1)
        name = self.name_emb.index_select(0, flat).view(B, S, -1)
        mod = self.modality_id.index_select(0, flat).view(B, S)
        return self.mask_emb(mod) + self.name_proj(name)

    def forward(self, cb: CellBatch, cell_mask: Tensor | None = None) -> Tensor:
        dev = self.name_emb.device
        h = torch.zeros(cb.num_seeds, cb.seq_len, self.d_model, device=dev)
        for nt, tf in cb.tf_dict.items():
            if nt not in self._layout or nt not in cb.cell_placement:
                continue
            cell = self.encode_type(nt, tf)
            b_idx, s_idx, row_idx, col_idx = (t.to(dev) for t in cb.cell_placement[nt])
            h[b_idx, s_idx] = cell[row_idx, col_idx].to(h.dtype)
        if cell_mask is not None:
            h = torch.where(cell_mask.to(dev).unsqueeze(-1), self.masked_token(cb).to(h.dtype), h)
        return h * (~cb.is_padding).to(dev).unsqueeze(-1)

    def category_table(self, node_type: str | None = None) -> Tensor:
        """The frozen label table the masked-cell head ties its logits to (schema-independent)."""
        return self.cat_emb


def _value_text_dim(bundle) -> int | None:
    """Width of the frozen cell-text embeddings, which must agree across every database."""
    from torch_frame.data.stats import StatType

    for nt in bundle.node_types:
        stats = bundle.col_stats_dict[nt]
        for c, s in stats.items():
            d = s.get(StatType.EMB_DIM)
            if d is not None:
                return int(d)
    return None
