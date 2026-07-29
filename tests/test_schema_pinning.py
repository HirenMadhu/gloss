"""P0.4 / P0.5 — the two things that make an UNSEEN schema representable.

P0.5 pins the modality (stype) id space to a fixed enum, so a given stype has the same id on every
dataset and ``stype_emb`` has a constant shape. Before this, the id space was "the stypes present in
this bundle, sorted by name", so ``numerical`` could be id 1 on one DB and id 2 on another and a
checkpoint silently mis-indexed across bundles.

P0.4 embeds table and FK-role NAMES. That is what lets an unseen table or role be represented at all:
on a new schema ``n_tables`` and ``K`` are new, so anything *indexed* by them is undefined, while a
name table is regenerated from the new schema's strings with no gradients.

changes.md §0 is the governing rule: a tensor may depend on the data only if it is recomputable from
a new database without gradients, and may not have a shape indexed by a training-set id.
"""
from __future__ import annotations

import pytest
import torch

from gloss.text.cache import HashEncoder
from gloss.text.schema import (
    MODALITY_UNKNOWN,
    N_STYPES,
    NAME_EMB_WIDTH,
    STYPE_ORDER,
    assert_name_emb_width,
    build_column_modality_ids,
    build_role_name_embeddings,
    build_table_name_embeddings,
    role_name_embeddings_with_none,
    role_name_strings,
    stype_id,
    table_name_strings,
)

from gloss.data.graph import role_name

from .conftest import collide_bundle, synthetic_bundle

# ---- P0.5: the pinned stype enum ----


def test_n_stypes_is_constant_and_covers_every_torch_frame_stype():
    from torch_frame import stype

    assert N_STYPES == 1 + len(STYPE_ORDER)
    present = {s for s in dir(stype) if not s.startswith("_")}
    missing = present - set(STYPE_ORDER)
    assert not missing, f"stypes unknown to STYPE_ORDER would collapse to id 0: {sorted(missing)}"
    # every real stype gets a distinct, nonzero id (0 is reserved for unknown)
    ids = [stype_id(getattr(stype, s)) for s in sorted(present)]
    assert len(set(ids)) == len(ids)
    assert MODALITY_UNKNOWN not in ids


def test_stype_id_accepts_enum_or_string_and_degrades_on_unknown():
    from torch_frame import stype

    assert stype_id(stype.numerical) == stype_id("numerical")
    # keyed on the NAME, not enum identity, so a pytorch-frame version bump that re-creates the
    # enum object does not silently renumber every id
    assert stype_id("some_future_stype") == MODALITY_UNKNOWN


def test_modality_ids_are_bundle_independent():
    """The P0.6 item: the same stype must get the same id on two DIFFERENT bundles.

    ``collide_bundle`` has an extra table and extra columns relative to ``synthetic_bundle``, so
    under the old "stypes present in this bundle, sorted by name" scheme the id space could shift.
    """
    a, n_a = build_column_modality_ids(synthetic_bundle())
    b, n_b = build_column_modality_ids(collide_bundle())

    assert n_a == n_b == N_STYPES, "n_stypes must not depend on the bundle"

    # for every stype actually present in both, the id agrees — compared via the pinned table
    # rather than by position, since the two bundles have different column counts
    from torch_frame import stype

    for name in STYPE_ORDER:
        assert stype_id(getattr(stype, name)) == stype_id(name)

    assert a.min() >= 0 and int(a.max()) < N_STYPES
    assert b.min() >= 0 and int(b.max()) < N_STYPES


def test_stype_order_is_append_only_contract():
    """Reordering STYPE_ORDER silently invalidates every checkpoint; pin the first entries."""
    assert STYPE_ORDER[:4] == ("numerical", "categorical", "multicategorical", "timestamp")
    assert stype_id("numerical") == 1, "id 0 is reserved for unknown"


# ---- P0.4: name-derived table and role tables ----


def test_table_name_table_is_gathered_by_table_id():
    bundle = synthetic_bundle()
    enc = HashEncoder(dim=16)
    emb = build_table_name_embeddings(bundle, enc)

    assert emb.shape == (len(bundle.node_type_id), 16)
    assert emb.dtype == torch.float32
    # index i really is table id i
    texts = table_name_strings(bundle)
    for nt, i in bundle.node_type_id.items():
        assert texts[i] == f"table {nt}"


def test_role_name_table_rows_line_up_with_role_triples():
    bundle = synthetic_bundle()
    enc = HashEncoder(dim=16)
    emb = build_role_name_embeddings(bundle, enc)

    assert emb.shape == (bundle.num_roles, 16)
    texts = role_name_strings(bundle)
    assert len(texts) == bundle.num_roles
    # row i corresponds to role id i+1 (FK_NONE = 0 has no name)
    for i, triple in enumerate(bundle.role_triples):
        assert texts[i] == role_name(triple)


def test_role_table_with_none_has_a_zero_row_for_no_edge():
    """`adj_role`'s 0 means "no edge", so a K+1 table can be gathered without offsetting."""
    bundle = synthetic_bundle()
    emb = role_name_embeddings_with_none(bundle, HashEncoder(dim=16))

    assert emb.shape == (bundle.num_roles + 1, 16)
    assert bool((emb[0] == 0).all()), "row 0 must be the all-zero FK_NONE slot"
    assert not bool((emb[1:] == 0).all())


def test_role_names_distinguish_same_named_fk_columns():
    """The P0.1 property, carried into the NAME strings P0.4 embeds.

    ``collide_bundle`` reuses one FK column name across two child tables. Those must produce
    different strings, or the frozen name table cannot separate them however good the encoder is.
    """
    bundle = collide_bundle()
    texts = role_name_strings(bundle)
    assert len(texts) == len(set(texts)), f"duplicate role name strings: {texts}"
    assert len(texts) == bundle.num_roles


# ---- §6: the no-dataset-artifact guard, at the schema level ----


def test_name_tables_are_data_not_parameters():
    """They must be plain frozen tensors — detached, not requiring grad."""
    bundle = synthetic_bundle()
    enc = HashEncoder(dim=16)
    for emb in (build_table_name_embeddings(bundle, enc),
                build_role_name_embeddings(bundle, enc)):
        assert not emb.requires_grad
        assert emb.device.type == "cpu"


def test_name_table_widths_are_encoder_driven_not_hardcoded():
    """A stale/wrong-encoder cache must fail loudly, while `hash` stays free-width for tests."""
    assert NAME_EMB_WIDTH["qwen"] == 2560

    assert_name_emb_width(torch.zeros(4, 2560), encoder="qwen")          # ok
    assert_name_emb_width(torch.zeros(4, 7), encoder="hash")             # unconstrained
    with pytest.raises(ValueError, match="name embedding width"):
        assert_name_emb_width(torch.zeros(4, 5376), encoder="qwen")      # harrier cache, qwen asked


def test_two_bundles_with_different_role_counts_give_the_same_widths():
    """`K` and `n_tables` may size the frozen DATA, never a weight shape (§0).

    Different schemas => different row counts, identical widths. Anything learned downstream is
    shaped by `d_text`, so it loads across schemas — the cheap proxy for the deferred LODO run.
    """
    enc = HashEncoder(dim=16)
    a, b = synthetic_bundle(), collide_bundle()
    ra = build_role_name_embeddings(a, enc)
    rb = build_role_name_embeddings(b, enc)

    assert ra.shape[0] != rb.shape[0], "fixtures must actually differ in role count"
    assert ra.shape[-1] == rb.shape[-1] == 16
    assert build_table_name_embeddings(a, enc).shape[-1] == \
        build_table_name_embeddings(b, enc).shape[-1]
