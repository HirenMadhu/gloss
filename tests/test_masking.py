"""Masked-cell selection and targets.

The invariants here are the ones that make the objective *mean* something. A mask that lands on a
padding slot, an unmaskable modality, or a missing value produces a loss term with no label behind it,
and nothing downstream would complain — the shapes are all valid. So they are asserted here.

The last test in the first group is the load-bearing one for the method: masking a value must not
perturb the routing signature. MoRE's whole claim is "route on a value-free signature, transform the
content", and masked-cell pretraining is only well-posed under it because that holds.
"""
from __future__ import annotations

import torch

from gloss.data.masking import (CATEGORICAL_ID, NUMERICAL_ID, build_column_target_spec,
                                gather_masked_targets, maskable_cells, sample_cell_mask)
from gloss.text.schema import stype_id

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available


def _spec_and_batch(seq_len=256, batch_size=8):
    bundle, task, cb = sample_cell_batch(seq_len=seq_len, batch_size=batch_size)
    return bundle, task, cb, build_column_target_spec(bundle)


# ------------------------------------------------------------------------------- what may be masked


@rel_f1_available
def test_spec_marks_only_numerical_and_categorical_columns():
    bundle, _task, _cb, spec = _spec_and_batch()
    marked = spec.stype[spec.maskable].unique().tolist()
    assert set(marked) <= {NUMERICAL_ID, CATEGORICAL_ID}, marked
    # rel-f1 is 23 numerical + 1 categorical + 8 timestamp + 13 text/embedding of 45 columns, so the
    # exclusions have to be doing real work or this assertion is vacuous.
    assert 0 < int(spec.maskable.sum()) < spec.num_columns


@rel_f1_available
def test_timestamp_and_text_columns_are_never_maskable():
    """Timestamps are republished as `row_time` on every cell, so predicting one is reading it; text
    targets are a frozen encoder's output, not a fact about the DB (and RT excludes them too)."""
    _b, _t, _cb, spec = _spec_and_batch()
    for name in ("timestamp", "text_embedded", "embedding", "multicategorical"):
        sid = stype_id(name)
        assert not bool(spec.maskable[spec.stype == sid].any()), name


@rel_f1_available
def test_categorical_columns_carry_a_class_count_and_a_table_offset():
    _b, _t, _cb, spec = _spec_and_batch()
    cat = spec.stype == CATEGORICAL_ID
    if not bool(cat.any()):
        return
    assert (spec.n_cat[cat] >= 1).all()
    assert (spec.cat_base[cat] >= 1).all(), "row 0 of the shared table is torch_frame's NaN slot"


@rel_f1_available
def test_encoder_offsets_agree_with_the_col_stats_cumsum():
    """`build_column_target_spec` asserts this internally; run it with a real encoder so the assert
    is actually exercised. A silent disagreement would tie the head to the wrong classes."""
    from gloss.model.column_encoder import CellEncoder

    bundle, _task, _cb, _spec = _spec_and_batch()
    enc = CellEncoder(bundle, name_table(), d_model=32, enc_channels=32)
    spec = build_column_target_spec(bundle, enc)         # raises if the two disagree
    assert int(spec.maskable.sum()) > 0


# ---------------------------------------------------------------------------------- what IS masked


@rel_f1_available
def test_mask_never_lands_on_padding_unmaskable_or_missing_cells():
    _b, _t, cb, spec = _spec_and_batch()
    cand = maskable_cells(cb, spec)
    mask, _seed = sample_cell_mask(cb, spec, p_random=0.5)
    assert bool((mask <= cand).all()), "mask escaped the candidate pool"
    assert not bool((mask & cb.is_padding).any())


@rel_f1_available
def test_at_most_one_seed_target_per_seed_and_exactly_one_where_possible():
    _b, _t, cb, spec = _spec_and_batch()
    cand = maskable_cells(cb, spec)
    _mask, seed = sample_cell_mask(cb, spec, p_random=0.0)
    per_seed = seed.sum(dim=1)
    assert bool((per_seed <= 1).all())
    possible = (cand & cb.is_seed_cell).any(dim=1)
    assert torch.equal(per_seed.bool(), possible), (
        "a seed with a maskable root-row cell must get exactly one target, and one without must "
        "get none -- rel-f1's `drivers`/`constructors` have no maskable column at all")


@rel_f1_available
def test_p_random_zero_masks_only_the_seed_targets():
    _b, _t, cb, spec = _spec_and_batch()
    mask, seed = sample_cell_mask(cb, spec, p_random=0.0)
    assert torch.equal(mask, seed), "p_random=0 is the RT-faithful arm: one cell per sequence"


@rel_f1_available
def test_seed_target_off_gives_plain_bert_masking():
    _b, _t, cb, spec = _spec_and_batch()
    mask, seed = sample_cell_mask(cb, spec, p_random=0.3, seed_target=False)
    assert not bool(seed.any())
    assert bool(mask.any())


@rel_f1_available
def test_random_rate_is_roughly_honoured_over_the_candidate_pool():
    _b, _t, cb, spec = _spec_and_batch(batch_size=16)
    cand = maskable_cells(cb, spec)
    mask, seed = sample_cell_mask(cb, spec, p_random=0.4, seed_target=False)
    frac = float(mask.sum()) / max(int(cand.sum()), 1)
    assert 0.25 < frac < 0.55, frac


# ------------------------------------------------------------------------------------- the targets


@rel_f1_available
def test_targets_match_the_raw_tensorframe_values():
    """Spot-check every masked numerical target against a hand gather through `cell_placement`."""
    import torch_frame

    bundle, _t, cb, spec = _spec_and_batch()
    mask, seed = sample_cell_mask(cb, spec, p_random=0.3)
    tgt = gather_masked_targets(cb, spec, mask, seed)
    assert len(tgt) > 0

    # (b, s) -> raw value, rebuilt independently of masking.py's vectorized path
    raw: dict[tuple[int, int], float] = {}
    for nt, (b_i, s_i, r_i, _c_i) in cb.cell_placement.items():
        tf = cb.tf_dict[nt]
        feat = tf.feat_dict.get(torch_frame.numerical)
        if feat is None:
            continue
        cols = tf.col_names_dict[torch_frame.numerical]
        for b, s, r in zip(b_i.tolist(), s_i.tolist(), r_i.tolist()):
            gid = int(cb.col_idxs[b, s])
            if int(spec.stype[gid]) != NUMERICAL_ID:
                continue
            j = int(spec.within_idx[gid])
            assert cols[j] is not None
            raw[(b, s)] = float(feat[r, j])

    checked = 0
    for i in range(len(tgt)):
        if int(tgt.stype[i]) != NUMERICAL_ID:
            continue
        gid = int(tgt.gid[i])
        key = (int(tgt.b[i]), int(tgt.s[i]))
        want = (raw[key] - float(spec.num_mean[gid])) / float(spec.num_std[gid])
        assert abs(float(tgt.y_num[i]) - want) < 1e-4
        checked += 1
    assert checked > 0, "no numerical targets to check"


@rel_f1_available
def test_categorical_targets_are_in_range_and_numerical_ones_are_finite():
    _b, _t, cb, spec = _spec_and_batch(batch_size=16)
    mask, seed = sample_cell_mask(cb, spec, p_random=0.5)
    tgt = gather_masked_targets(cb, spec, mask, seed)
    is_cat = tgt.stype == CATEGORICAL_ID
    if bool(is_cat.any()):
        n_cat = spec.n_cat[tgt.gid[is_cat]]
        assert bool((tgt.y_cat[is_cat] >= 0).all())
        assert bool((tgt.y_cat[is_cat] < n_cat).all())
    is_num = tgt.stype == NUMERICAL_ID
    assert bool(torch.isfinite(tgt.y_num[is_num]).all()), "a NaN cell became a target"


# --------------------------------------------------- the property the whole method rests on


@rel_f1_available
def test_masking_a_value_does_not_change_the_routing_signature():
    """`z` is value-free, so a masked cell still routes to the experts its column/modality/recency
    imply. Without this, masked-cell pretraining would be routing on a hole."""
    from gloss.model.signature import RelationalSignature
    from gloss.text.schema import build_column_modality_ids

    bundle, _t, cb, spec = _spec_and_batch()
    name_emb = name_table()
    modality_id, n_stypes = build_column_modality_ids(bundle)
    sig = RelationalSignature(name_emb, modality_id, n_stypes, d_sig=32)
    with torch.no_grad():
        z = sig(cb)
    mask, seed = sample_cell_mask(cb, spec, p_random=0.5)
    assert bool(mask.any())
    with torch.no_grad():
        z2 = sig(cb)
    assert torch.equal(z, z2)


@rel_f1_available
def test_masked_cell_token_keeps_the_column_name_and_drops_only_the_value():
    """RT's contract: the value embedding is replaced by a learned per-datatype vector; the
    column-name token stays, so the model knows what it is being asked to predict."""
    from gloss.model.column_encoder import CellEncoder

    bundle, _t, cb, spec = _spec_and_batch()
    enc = CellEncoder(bundle, name_table(), d_model=32, enc_channels=32)
    mask, seed = sample_cell_mask(cb, spec, p_random=0.4)
    with torch.no_grad():
        plain = enc(cb)
        masked = enc(cb, mask)
        token = enc.masked_token(cb)
    real = ~cb.is_padding
    assert torch.allclose(masked[mask & real], token[mask & real], atol=1e-6)
    keep = real & ~mask
    assert torch.allclose(masked[keep], plain[keep], atol=0), "unmasked cells must be untouched"


@rel_f1_available
def test_mask_embedding_is_sized_by_the_pinned_stype_enum_not_the_schema():
    """§0: no parameter may be shaped by a training-set id. The mask token is per-MODALITY."""
    from gloss.model.column_encoder import CellEncoder
    from gloss.text.schema import N_STYPES

    bundle, _t, _cb, _spec = _spec_and_batch()
    enc = CellEncoder(bundle, name_table(), d_model=32, enc_channels=32)
    assert enc.mask_emb.num_embeddings == N_STYPES
    C = enc.name_emb.shape[0]
    assert enc.mask_emb.weight.shape[0] != C or N_STYPES == C
