"""Task-free seed sampling: temporal safety, seed-table selection, and the round-robin schedule.

The temporal assertion here is the one that matters most in the whole pretraining stack. The graph is
materialized with ``upto_test_timestamp=False`` — the *entire* database, including the test period —
so nothing structural stops a pretraining seed from reaching a test-period row. If that cap is ever
removed, every downstream number becomes contaminated and no test elsewhere would notice.
"""
from __future__ import annotations

import torch

from gloss.data.collate import to_cell_batch
from gloss.data.masking import build_column_target_spec, maskable_cells, sample_cell_mask
from gloss.data.pretrain_loader import (build_pretrain_stream, build_seed_tables, maskable_tables)

from ._relf1 import bundle_and_task
from .conftest import rel_f1_available


def _bundle_spec():
    bundle, _task = bundle_and_task()
    return bundle, build_column_target_spec(bundle)


# ---------------------------------------------------------------------------- temporal safety


@rel_f1_available
def test_train_seed_times_never_reach_the_validation_period():
    """The cap that makes pretraining honest. Combined with the sampler's `row_time <= seed_time`,
    it is what guarantees no val/test-period cell can enter a pretraining batch."""
    from relbench.datasets import get_dataset

    bundle, spec = _bundle_spec()
    val_ts = int(get_dataset("rel-f1", download=False).val_timestamp.timestamp())
    for t in build_seed_tables(bundle, spec, "train"):
        assert int(t.time.max()) <= val_ts, (t.node_type, int(t.time.max()), val_ts)


@rel_f1_available
def test_val_seeds_are_a_later_window_than_train_seeds():
    bundle, spec = _bundle_spec()
    tr = {t.node_type: t for t in build_seed_tables(bundle, spec, "train")}
    va = {t.node_type: t for t in build_seed_tables(bundle, spec, "val")}
    shared = [n for n in tr if n in va and tr[n].timed and va[n].timed]
    assert shared, "no timed table in both splits"
    for n in shared:
        assert int(va[n].time.min()) >= int(tr[n].time.max()) - 1, n


@rel_f1_available
def test_no_cell_is_dated_after_its_seed():
    """The sampler's own guarantee, re-checked on the label-free path (it was only ever tested on the
    task-driven one, in test_leakage.py)."""
    bundle, spec = _bundle_spec()
    stream = build_pretrain_stream(bundle, spec, split="train", steps=3, batch_size=8, seed=0)
    for nt, raw in stream:
        cb = to_cell_batch(raw, bundle, nt, seq_len=512, max_fk=5)
        late = (cb.row_time > cb.seed_time.unsqueeze(1)) & ~cb.is_padding & cb.is_timed
        assert not bool(late.any()), f"{nt}: {int(late.sum())} cells dated after their seed"


# ------------------------------------------------------------------------ which tables seed


@rel_f1_available
def test_tables_without_a_maskable_column_are_never_seed_tables():
    """rel-f1's `drivers` and `constructors` have no numerical or categorical column, so a seed drawn
    from them could never produce a seed target. They still appear as sampled neighbours."""
    bundle, spec = _bundle_spec()
    per_table = maskable_tables(bundle, spec)
    assert per_table["drivers"] == 0 and per_table["constructors"] == 0
    names = {t.node_type for t in build_seed_tables(bundle, spec, "train")}
    assert "drivers" not in names and "constructors" not in names
    assert "results" in names, "the widest table (10 maskable columns) must be a seed table"


@rel_f1_available
def test_seed_weight_scales_with_maskable_columns_not_just_rows():
    bundle, spec = _bundle_spec()
    tabs = {t.node_type: t for t in build_seed_tables(bundle, spec, "train")}
    per_table = maskable_tables(bundle, spec)
    for name, t in tabs.items():
        assert t.weight == float(per_table[name]) * float(len(t))


@rel_f1_available
def test_untimed_tables_get_a_sampled_cutoff_rather_than_being_dropped():
    """A static table has no moment of its own, but dropping it would lose a legitimate seed source;
    a drawn cutoff gives it causal context instead."""
    bundle, spec = _bundle_spec()
    tabs = {t.node_type: t for t in build_seed_tables(bundle, spec, "train")}
    untimed = [t for t in tabs.values() if not t.timed]
    assert untimed, "rel-f1's `circuits` is untimed and has 3 maskable columns"
    for t in untimed:
        assert int(t.time.min()) > 0 and t.time.numel() == len(t)


# --------------------------------------------------------------------------- the mixed stream


@rel_f1_available
def test_the_table_schedule_is_deterministic_given_the_seed():
    """Every DDP rank must pick the SAME table at the same step. With per-table encoders, ranks
    disagreeing would leave different parameters unused on different ranks, which DDP cannot
    reconcile."""
    bundle, spec = _bundle_spec()
    a = build_pretrain_stream(bundle, spec, split="train", steps=64, batch_size=4, seed=3)
    b = build_pretrain_stream(bundle, spec, split="train", steps=64, batch_size=4, seed=3)
    c = build_pretrain_stream(bundle, spec, split="train", steps=64, batch_size=4, seed=4)
    assert a.table_schedule() == b.table_schedule()
    assert a.table_schedule() != c.table_schedule()


@rel_f1_available
def test_stream_yields_the_requested_number_of_steps_with_the_table_name():
    bundle, spec = _bundle_spec()
    stream = build_pretrain_stream(bundle, spec, split="train", steps=5, batch_size=4, seed=0)
    assert len(stream) == 5
    got = list(stream)
    assert len(got) == 5
    for nt, raw in got:
        assert isinstance(nt, str) and nt in stream.loaders


@rel_f1_available
def test_batches_carry_no_labels_but_do_carry_a_seed_time():
    """No task table, so `y` never arrives; `seed_time` still does, straight from `input_time`."""
    bundle, spec = _bundle_spec()
    stream = build_pretrain_stream(bundle, spec, split="train", steps=2, batch_size=8, seed=0)
    for nt, raw in stream:
        assert "y" not in raw[nt]
        assert "seed_time" in raw[nt]
        cb = to_cell_batch(raw, bundle, nt, seq_len=512, max_fk=5)
        assert not bool(cb.has_target.any())
        assert bool(cb.row_is_root.sum(dim=1).eq(1).all()), "exactly one root row per seed"


@rel_f1_available
def test_seeds_from_maskable_tables_actually_get_seed_targets():
    """The payoff of weighting by maskable rows: on the task-driven loader rel-f1 yields ZERO seed
    targets (entity table `drivers` has no maskable column); here every seed gets one."""
    bundle, spec = _bundle_spec()
    stream = build_pretrain_stream(bundle, spec, split="train", steps=3, batch_size=8, seed=0)
    for nt, raw in stream:
        cb = to_cell_batch(raw, bundle, nt, seq_len=1024, max_fk=5)
        _mask, seed = sample_cell_mask(cb, spec, p_random=0.0,
                                       generator=torch.Generator().manual_seed(0))
        assert int(seed.sum()) == cb.num_seeds, (
            f"{nt}: {int(seed.sum())}/{cb.num_seeds} seeds got a target")
        assert int(maskable_cells(cb, spec).sum()) > 0
