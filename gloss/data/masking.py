"""Masked-cell prediction: which cells may be masked, which are, and what their targets are.

RT (arXiv:2510.06377) masks **exactly one cell per sequence** — the seed row's designated target
column (``rt-v1/rustler/src/fly.rs``: ``masks[seq_i] = is_target``); its ``main`` branch added a random
rate and shipped it at 0. We keep that seed-target cell, because it is the shape MoRE's root-row
readout is already built around, and *add* Bernoulli-``p`` masking of the remaining maskable cells for
a denser signal. Both halves are switches, so ``p_random=0`` is the RT-faithful arm and
``seed_target=False`` is plain BERT-style corruption.

**What may be masked, and why the exclusions are not arbitrary:**

``numerical``
    Target is ``(v - MEAN) / (STD + 1e-6)`` from ``col_stats`` — the *same* normalization
    ``LinearEncoder.encode_forward`` applies on the input side, so prediction and input share units.
``categorical``
    Target is the raw category index. Vocabularies here are small (max 42 across rel-trial, max 3 on
    rel-stack, and rel-f1 has exactly one categorical column), so this term is real but minor.
``timestamp`` — **excluded.**
    relbench keeps a table's time column in the TensorFrame *and* republishes it as ``row_time`` on
    every cell of the row. Predicting it is reading it.
``text_embedded`` / ``embedding`` — **excluded.**
    RT's own position (``"masking text not supported"``). The target would be a frozen encoder's
    output, not a fact about the database.
``multicategorical`` — **excluded.** No decode head for a variable-length label set.

Also dropped: columns whose ``STD`` is ~0 (a z-score of a constant is not a prediction), single-class
categoricals, and the ``__const__`` filler relbench inserts for tables with no real features.

**NaN cells never become targets.** They are removed from the candidate pool *before* the mask is
drawn, so a masked position always has a label — RT does the same (``"Never mark such cells as
targets -- they have no valid label"``). A seed row with no maskable cell therefore contributes no
seed target at all, which is a legal and *common* outcome, not an error: rel-f1's ``drivers`` and
``constructors`` have zero maskable columns, and so do 7 of rel-trial's 15 tables — including
``facilities_studies`` at 1.87M rows. The count is returned so a run can log it instead of quietly
training on fewer targets than it thinks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..text.schema import STYPE_ORDER, stype_id

#: Stype ids that carry a reconstruction target, in the pinned `STYPE_ORDER` id space.
NUMERICAL_ID = stype_id("numerical")
CATEGORICAL_ID = stype_id("categorical")
MASKABLE_STYPES = (NUMERICAL_ID, CATEGORICAL_ID)

#: relbench's placeholder column for a table with no real features (`make_pkey_fkey_graph`), which is
#: the constant 1.0 and therefore has nothing to predict.
CONST_COL = "__const__"

_MIN_STD = 1e-8


@dataclass
class ColumnTargetSpec:
    """Static, per-bundle description of every column, indexed by **global column id**.

    Global column id is ``collate.column_vocab``'s ``(table, column) -> int``, i.e. exactly what a
    :class:`~gloss.data.collate.CellBatch` carries in ``col_idxs``. Every tensor here is ``[C]``, so a
    batch's per-cell property is one ``index_select`` away and no per-cell Python runs at train time.
    """

    maskable: Tensor      # bool  — may be chosen as a prediction target
    stype: Tensor         # long  — pinned STYPE_ORDER id (see gloss/text/schema.py)
    within_idx: Tensor    # long  — column's index inside tf.feat_dict[stype] (-1 if none)
    table: Tensor         # long  — node-type id, for gathering out of the right TensorFrame
    num_mean: Tensor      # float — col_stats MEAN (0 for non-numerical)
    num_std: Tensor       # float — col_stats STD + 1e-6 (1 for non-numerical)
    n_cat: Tensor         # long  — category count (0 for non-categorical)
    cat_base: Tensor      # long  — row of the shared EmbeddingEncoder table where this column starts
    node_types: tuple     # node-type name per table id, so gather() can walk tf_dict

    @property
    def num_columns(self) -> int:
        return int(self.maskable.numel())

    def to(self, device) -> "ColumnTargetSpec":
        f = lambda t: t.to(device)
        return ColumnTargetSpec(
            maskable=f(self.maskable), stype=f(self.stype), within_idx=f(self.within_idx),
            table=f(self.table), num_mean=f(self.num_mean), num_std=f(self.num_std),
            n_cat=f(self.n_cat), cat_base=f(self.cat_base), node_types=self.node_types,
        )

    def summary(self) -> str:
        n = self.num_columns
        m = int(self.maskable.sum())
        cat = int((self.maskable & (self.stype == CATEGORICAL_ID)).sum())
        return (f"{m}/{n} maskable ({m - cat} numerical, {cat} categorical); "
                f"max categories {int(self.n_cat.max()) if n else 0}")


def _cat_offsets(cell_encoder, nt: str) -> Tensor | None:
    """The per-column base row of ``nt``'s categorical embedding table, or None if it has none.

    torch_frame's ``EmbeddingEncoder`` keeps ONE ``nn.Embedding`` for all of a table's categorical
    columns, with row 0 reserved for NaN and a per-column ``offset`` buffer; column ``j``'s classes
    live at ``offset[j] + 1 ... offset[j] + n_cat_j``. Tying the decode head's logits to that slice is
    what keeps the head free of any vocabulary-shaped parameter, so it survives an unseen schema.
    """
    enc = getattr(cell_encoder, "cell_encoders", None)
    if enc is None or nt not in enc:
        return None
    # `encoder_dict` is an nn.ModuleDict, which implements __contains__/__getitem__ but NOT .get --
    # a `.get(...)` here silently returned None for every table, so the offsets fell back to the
    # cumsum and the cross-check below never actually ran.
    sub = getattr(enc[nt], "encoder_dict", None)
    if sub is None or "categorical" not in sub:
        return None
    emb = sub["categorical"]
    if not hasattr(emb, "offset"):
        return None
    return emb.offset.detach().cpu().to(torch.long) + 1


def build_column_target_spec(bundle, cell_encoder=None) -> ColumnTargetSpec:
    """Build the per-column target table for ``bundle``. Cheap; call once and reuse.

    ``cell_encoder`` supplies the authoritative categorical offsets. Without it they are derived from
    the ``col_stats`` class-count cumsum, which must agree — and is asserted to when both are present,
    because a silent disagreement would train the head against the wrong slice of the table.
    """
    import torch_frame
    from torch_frame.data.stats import StatType

    from .collate import column_vocab, feature_col_names

    vocab = column_vocab(bundle)
    C = len(vocab)
    # The schema-free encoder ties to ONE global, frozen category-label table and already publishes
    # the per-column base as a [C] buffer; torch_frame's encoder instead uses per-TABLE offsets into
    # a learned per-table embedding. Both are "where this column's classes start", but in different
    # tables, so the head must be told which one it is scoring against.
    global_cat_base = getattr(cell_encoder, "cat_base", None)
    if global_cat_base is not None and int(global_cat_base.numel()) != C:
        global_cat_base = None

    maskable = torch.zeros(C, dtype=torch.bool)
    stype = torch.zeros(C, dtype=torch.long)
    within = torch.full((C,), -1, dtype=torch.long)
    table = torch.full((C,), -1, dtype=torch.long)
    mean = torch.zeros(C, dtype=torch.float32)
    std = torch.ones(C, dtype=torch.float32)
    n_cat = torch.zeros(C, dtype=torch.long)
    cat_base = torch.zeros(C, dtype=torch.long)

    node_types = [""] * (max(bundle.node_type_id.values()) + 1) if bundle.node_type_id else []
    for nt, i in bundle.node_type_id.items():
        node_types[i] = nt

    for nt in bundle.node_types:
        tf = bundle.data[nt].tf
        stats = bundle.col_stats_dict[nt]
        tid = bundle.node_type_id[nt]
        offsets = (None if global_cat_base is not None or cell_encoder is None
                   else _cat_offsets(cell_encoder, nt))
        running = 1                                # row 0 of the shared table is the NaN slot
        for st, cols in tf.col_names_dict.items():
            sid = stype_id(st)
            is_num = st == torch_frame.numerical
            is_cat = st == torch_frame.categorical
            for j, c in enumerate(cols):
                if (nt, c) not in vocab:
                    continue
                gid = vocab[(nt, c)]
                stype[gid], within[gid], table[gid] = sid, j, tid
                if is_num:
                    m = stats[c].get(StatType.MEAN)
                    s = stats[c].get(StatType.STD)
                    if m is not None and s is not None:
                        mean[gid], std[gid] = float(m), float(s) + 1e-6
                        maskable[gid] = c != CONST_COL and float(s) > _MIN_STD
                elif is_cat:
                    k = len(stats[c][StatType.COUNT][0])
                    n_cat[gid] = k
                    base = int(offsets[j]) if offsets is not None else running
                    if offsets is not None:
                        assert base == running, (
                            f"categorical offset disagreement on {nt}.{c}: encoder says {base}, "
                            f"col_stats cumsum says {running}. The decode head ties its logits to "
                            f"this slice, so a mismatch trains against the wrong classes."
                        )
                    cat_base[gid] = int(global_cat_base[gid]) if global_cat_base is not None else base
                    running += k
                    maskable[gid] = k >= 2

    return ColumnTargetSpec(maskable=maskable, stype=stype, within_idx=within, table=table,
                            num_mean=mean, num_std=std, n_cat=n_cat, cat_base=cat_base,
                            node_types=tuple(node_types))


def _raw_values(cb, spec: ColumnTargetSpec) -> tuple[Tensor, Tensor]:
    """``([B,S] float raw value, [B,S] bool has-a-value)`` gathered from the batch's TensorFrames.

    Values live per node type in ``cb.tf_dict``; ``cb.cell_placement[nt]`` is the ``(b, s, row, col)``
    scatter map the cell encoder uses, and ``col`` there indexes ``feature_col_names`` — the same
    sorted order the spec's ``within_idx`` was built from, keyed through the global column id.
    """
    import torch_frame

    B, S = cb.col_idxs.shape
    dev = cb.col_idxs.device
    val = torch.zeros(B, S, dtype=torch.float32, device=dev)
    ok = torch.zeros(B, S, dtype=torch.bool, device=dev)

    gid = cb.col_idxs.clamp_min(0)
    cell_stype = spec.stype.to(dev).index_select(0, gid.reshape(-1)).view(B, S)
    cell_within = spec.within_idx.to(dev).index_select(0, gid.reshape(-1)).view(B, S)

    for nt, (b_idx, s_idx, row_idx, _col_idx) in cb.cell_placement.items():
        tf = cb.tf_dict.get(nt)
        if tf is None:
            continue
        b_idx, s_idx, row_idx = b_idx.to(dev), s_idx.to(dev), row_idx.to(dev)
        for st, key in ((torch_frame.numerical, NUMERICAL_ID),
                        (torch_frame.categorical, CATEGORICAL_ID)):
            feat = tf.feat_dict.get(st)
            if feat is None:
                continue
            sel = cell_stype[b_idx, s_idx] == key
            if not bool(sel.any()):
                continue
            bb, ss, rr = b_idx[sel], s_idx[sel], row_idx[sel]
            ww = cell_within[bb, ss]
            v = feat[rr, ww]
            # NaN for numerical, -1 for categorical -- torch_frame's two missing-value conventions.
            good = ~torch.isnan(v) if v.is_floating_point() else v >= 0
            val[bb, ss] = v.to(torch.float32)
            ok[bb, ss] = good
    return val, ok


def maskable_cells(cb, spec: ColumnTargetSpec) -> Tensor:
    """``[B,S]`` bool: real, target-bearing, non-missing cells — the pool a mask may draw from."""
    dev = cb.col_idxs.device
    gid = cb.col_idxs.clamp_min(0)
    B, S = gid.shape
    ok_col = spec.maskable.to(dev).index_select(0, gid.reshape(-1)).view(B, S)
    _v, has_value = _raw_values(cb, spec)
    return (~cb.is_padding) & ok_col & has_value


def sample_cell_mask(
    cb,
    spec: ColumnTargetSpec,
    *,
    p_random: float = 0.15,
    seed_target: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """-> ``(mask [B,S] bool, is_seed_target [B,S] bool)``.

    ``seed_target`` masks exactly one cell of each seed root row (RT's objective); ``p_random`` masks
    that fraction of the *remaining* candidates. A seed whose root row has no maskable cell simply
    gets no seed target — see the module docstring for how routine that is.
    """
    cand = maskable_cells(cb, spec)
    dev = cand.device
    B, S = cand.shape

    is_seed_target = torch.zeros_like(cand)
    if seed_target:
        seed_cand = cand & cb.is_seed_cell
        # One uniform draw per seed among its own candidates: score the candidates randomly and take
        # the argmax. Rows with no candidate score -inf everywhere and are dropped by `any()` below.
        noise = torch.rand(B, S, device=dev, generator=generator)
        noise = noise.masked_fill(~seed_cand, -1.0)
        pick = noise.argmax(dim=1)
        has_pick = seed_cand.any(dim=1)
        is_seed_target[torch.arange(B, device=dev)[has_pick], pick[has_pick]] = True

    mask = is_seed_target.clone()
    if p_random > 0:
        draw = torch.rand(B, S, device=dev, generator=generator) < p_random
        mask |= cand & draw & ~is_seed_target
    return mask, is_seed_target


@dataclass
class MaskedTargets:
    """Flat, per-masked-cell targets. ``b``/``s`` index back into the ``[B,S]`` cell grid."""

    b: Tensor           # long  [M]
    s: Tensor           # long  [M]
    gid: Tensor         # long  [M] global column id
    stype: Tensor       # long  [M]
    y_num: Tensor       # float [M] z-scored value (0 where not numerical)
    y_cat: Tensor       # long  [M] class index (-1 where not categorical)
    is_seed: Tensor     # bool  [M] this was the seed row's designated target
    n_seeds_without_target: int

    def __len__(self) -> int:
        return int(self.b.numel())


def gather_masked_targets(cb, spec: ColumnTargetSpec, mask: Tensor,
                          is_seed_target: Tensor) -> MaskedTargets:
    """Collect the label for every masked cell.

    Numerical labels are z-scored with the column's TRAIN-split ``col_stats``, matching both RT
    (§3.1: ``r = (v - mu_c) / sigma_c``) and, more importantly, what the *input* encoder already does —
    so the head regresses in the units the model sees.
    """
    dev = mask.device
    val, _ok = _raw_values(cb, spec)
    b, s = mask.nonzero(as_tuple=True)
    gid = cb.col_idxs[b, s].clamp_min(0)

    st = spec.stype.to(dev).index_select(0, gid)
    mean = spec.num_mean.to(dev).index_select(0, gid)
    std = spec.num_std.to(dev).index_select(0, gid)
    raw = val[b, s]

    y_num = torch.where(st == NUMERICAL_ID, (raw - mean) / std, torch.zeros_like(raw))
    y_cat = torch.where(st == CATEGORICAL_ID, raw.to(torch.long),
                        torch.full_like(raw, -1, dtype=torch.long))

    if is_seed_target.any():
        missing = int((~(maskable_cells(cb, spec) & cb.is_seed_cell).any(dim=1)).sum())
    else:
        missing = int(cb.num_seeds)
    return MaskedTargets(b=b, s=s, gid=gid, stype=st, y_num=y_num, y_cat=y_cat,
                         is_seed=is_seed_target[b, s], n_seeds_without_target=missing)
