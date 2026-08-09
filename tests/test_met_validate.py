"""``MultiEmbeddingTensor.validate`` repair policy — re-read before rebase.

The rule these tests pin is an ordering, not a numeric result: a tensor that fails validation must be
**re-read** first, and only rebased if the failure survives. Re-reading the same storage cannot change
data; rebasing rewrites which slice of ``values`` each column points at, so a rebase applied to a
tensor that was only *transiently* unreadable silently corrupts embeddings instead of crashing. The
2026-08-08 dumps in :func:`test_the_four_failing_dumps_satisfy_every_assert` are the evidence that the
transient case is real.
"""
from __future__ import annotations

import pytest
import torch

from gloss.data import graph as G

MET = pytest.importorskip("torch_frame.data.multi_embedding_tensor").MultiEmbeddingTensor


@pytest.fixture(autouse=True)
def _zero_counters():
    before = dict(G.MET_REPAIRS)
    G.MET_REPAIRS.update(reread=0, rebase=0)
    yield
    G.MET_REPAIRS.update(before)


def make_met(offset, values, num_cols=None, num_rows=None):
    off = torch.tensor(offset, dtype=torch.long)
    return MET(num_rows=values.shape[0] if num_rows is None else num_rows,
               num_cols=off.numel() - 1 if num_cols is None else num_cols,
               values=values, offset=off)


# --------------------------------------------------------------- the named checks ---
def test_named_checks_pass_on_a_well_formed_tensor():
    assert G._met_failed_checks(torch.tensor([0, 3, 5]), torch.zeros(4, 5), 2) == []


@pytest.mark.parametrize("offset,values,num_cols,expected", [
    ([2, 5, 7], torch.zeros(4, 5), 2, "offset[0]==0"),
    ([0, 3, 5], torch.zeros(4, 5), 7, "len(offset)==num_cols+1"),
    ([[0, 3], [5, 7]], torch.zeros(4, 5), 3, "offset.ndim==1"),
    ([0, 3, 5], torch.zeros(4, 5, 6), 2, "values.ndim==2 or values.numel()==0"),
])
def test_each_broken_invariant_is_named(offset, values, num_cols, expected):
    """Upstream is a bare assert chain, so a failure carries no name. These are the names."""
    bad = G._met_failed_checks(torch.tensor(offset, dtype=torch.long), values, num_cols)
    assert expected in bad


def test_none_offset_is_reported_not_raised():
    assert len(G._met_failed_checks(None, torch.zeros(4, 5), 2)) == 3


def test_zero_dim_offset_does_not_raise_typeerror():
    """`len()` on a 0-dim tensor raises rather than asserting, which would mask the real fault."""
    bad = G._met_failed_checks(torch.tensor(0), torch.zeros(4, 5), 2)
    assert "offset.ndim==1" in bad


def test_the_four_failing_dumps_satisfy_every_assert():
    """The diagnosis, as a test.

    These are the verbatim layouts from ``29054792_7``, ``29054796_1`` and ``29054798_{1,5}``, all of
    which died inside ``validate``. Every one of them satisfies all four asserts, so the assert cannot
    have been looking at this state — the storage changed underneath it. If this test ever starts
    failing, the shared-memory-race explanation is wrong and the repair policy needs revisiting.
    """
    dumps = [
        ([0, 32, 64, 96, 128], (512, 128), 4),
        ([0, 32], (5616, 32), 1),
        ([0, 32, 64, 96, 128, 160], (285, 160), 5),
        ([0, 32, 64, 96, 128], (0, 128), 4),
    ]
    for offset, shape, num_cols in dumps:
        bad = G._met_failed_checks(torch.tensor(offset), torch.zeros(*shape), num_cols)
        assert bad == [], f"{offset} / {shape} / {num_cols} -> {bad}"


# --------------------------------------------------------------- the repair policy ---
def test_a_valid_tensor_takes_no_repair_path():
    make_met([0, 32, 64], torch.randn(8, 64))
    assert G.MET_REPAIRS == {"reread": 0, "rebase": 0}


def test_a_transient_failure_is_repaired_by_reread_not_rebase(monkeypatch):
    """The race: the check fails once, then the same storage reads clean. Data must not be rewritten."""
    calls = {"n": 0}
    real = G._met_failed_checks

    def flaky(off, values, num_cols):
        calls["n"] += 1
        return ["offset[0]==0"] if calls["n"] == 1 else real(off, values, num_cols)

    monkeypatch.setattr(G, "_met_failed_checks", flaky)
    values = torch.randn(8, 64)
    met = make_met([0, 32, 64], values.clone())

    assert G.MET_REPAIRS == {"reread": 1, "rebase": 0}
    assert torch.equal(met.offset, torch.tensor([0, 32, 64]))
    assert torch.equal(met.values, values), "a re-read repair must not touch the value buffer"


def test_a_persistent_shift_still_rebases_and_preserves_each_column():
    """When the base really is shifted and stays shifted, the old repair is still available."""
    values = torch.randn(8, 64)                       # w == T == 64, so only the base moves
    met = make_met([32, 64, 96], values.clone(), num_cols=2)
    assert G.MET_REPAIRS == {"reread": 0, "rebase": 1}
    assert torch.equal(met.offset, torch.tensor([0, 32, 64]))
    assert torch.equal(met.values, values)


def test_a_shift_with_orphan_leading_columns_drops_them():
    """w == k + T: the first k value-columns are unreferenced, so the rebase must slice them off."""
    k, T = 16, 64
    values = torch.randn(8, k + T)
    met = make_met([k, k + 32, k + T], values.clone(), num_cols=2)
    assert G.MET_REPAIRS["rebase"] == 1
    assert torch.equal(met.offset, torch.tensor([0, 32, 64]))
    assert torch.equal(met.values, values[:, k:])


def test_a_zero_column_shift_rebases_without_touching_values():
    """num_cols == 0: nothing indexes `values`, so rebasing to [0] cannot lose data."""
    values = torch.randn(8, 64)
    met = make_met([32], values.clone(), num_cols=0)
    assert G.MET_REPAIRS["rebase"] == 1
    assert torch.equal(met.offset, torch.tensor([0]))
    assert torch.equal(met.values, values)


def test_rebase_is_refused_when_more_than_the_base_is_wrong():
    """A rebase only explains a shifted base. Any second broken invariant means it is a guess."""
    with pytest.raises(RuntimeError) as e:
        make_met([32, 64, 96], torch.randn(8, 64), num_cols=5)
    assert "neither a re-read nor a rebase" in str(e.value)
    assert "len(offset)==num_cols+1" in str(e.value)
    assert G.MET_REPAIRS["rebase"] == 0


def test_an_unrecognised_value_width_raises_with_the_layout():
    """k=32, T=64, w=50 matches neither known layout -> refuse, and say so with numbers attached."""
    with pytest.raises(RuntimeError) as e:
        make_met([32, 64, 96], torch.randn(8, 50), num_cols=2)
    msg = str(e.value)
    assert "unrecognised layout" in msg
    assert "k=32" in msg and "T=64" in msg and "(8, 50)" in msg
    assert "re-reads" in msg, "must state that the race was ruled out before blaming the layout"
    assert G.MET_REPAIRS["rebase"] == 0


def test_the_giveup_message_reports_both_check_snapshots(monkeypatch):
    """Entry state *and* post-re-read state, because the difference between them is the diagnosis."""
    monkeypatch.setattr(G, "_met_failed_checks",
                        lambda off, values, nc: ["offset.ndim==1"])
    with pytest.raises(RuntimeError) as e:
        make_met([0, 32, 64], torch.randn(8, 64))
    msg = str(e.value)
    assert "failed_checks_on_entry=['offset.ndim==1']" in msg
    assert "rereads=['offset.ndim==1']" in msg


def test_reread_is_retried_more_than_once(monkeypatch):
    """One retry would miss a race that takes two beats to settle; the loop is what buys the margin."""
    calls = {"n": 0}
    real = G._met_failed_checks

    def flaky(off, values, num_cols):
        calls["n"] += 1
        return ["offset[0]==0"] if calls["n"] <= 2 else real(off, values, num_cols)

    monkeypatch.setattr(G, "_met_failed_checks", flaky)
    monkeypatch.setattr(G, "_MET_REREAD_TRIES", 3)
    make_met([0, 32, 64], torch.randn(8, 64))
    assert G.MET_REPAIRS == {"reread": 1, "rebase": 0}


def test_met_repair_counts_is_a_copy():
    counts = G.met_repair_counts()
    counts["rebase"] = 99
    assert G.MET_REPAIRS["rebase"] == 0
