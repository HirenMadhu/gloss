"""Hermetic tests for the row-level graph (changes.md P0.2 / P0.3 / P0.6).

Contracts, all asserted on synthetic fixtures and re-checked on a real rel-f1 batch when cached:
``adj_role`` is symmetric with the direction flipped, every non-root row is joined to something,
row timestamps never exceed the seed time (the leakage assert — never relax it), the three row
counts agree, and hop is a BFS distance bounded by the sampler depth.
"""
from __future__ import annotations

import pytest
import torch

from gloss.data.collate import to_cell_batch
from gloss.data.row_graph import self_loop_id

from .conftest import (
    ENTITY,
    chain_bundle,
    make_chain_batch,
    make_synth_batch,
    rel_f1_available,
    synthetic_bundle,
)


def _cb(seq_len=16, max_fk=2, max_rows=8, **kw):
    return to_cell_batch(make_synth_batch(), synthetic_bundle(), ENTITY,
                         seq_len=seq_len, max_fk=max_fk, max_rows=max_rows, **kw)


def _chain_cb(max_rows=8, **kw):
    return to_cell_batch(make_chain_batch(), chain_bundle(), ENTITY,
                         seq_len=32, max_fk=2, max_rows=max_rows, **kw)


# ---- shapes / dtypes (the §6 "Shapes" contract) ----

def test_row_field_shapes_and_dtypes():
    cb = _cb()
    B, R = cb.num_seeds, cb.max_rows
    assert (cb.num_rows.shape, cb.num_rows.dtype) == ((B,), torch.int64)
    for name, dt in (("row_valid", torch.bool), ("row_table", torch.int64),
                     ("row_time_r", torch.float64), ("row_is_timed", torch.bool),
                     ("row_hop", torch.int64), ("row_in_role", torch.int64),
                     ("row_is_root", torch.bool)):
        t = getattr(cb, name)
        assert t.shape == (B, R) and t.dtype == dt, name
    assert cb.adj_role.shape == (B, R, R) and cb.adj_role.dtype == torch.int64
    # cell_row is an alias, not a copy
    assert cb.cell_row is cb.node_idxs
    # pad slots carry the documented fill values
    pad = ~cb.row_valid
    assert (cb.row_table[pad] == -1).all()
    assert (cb.row_time_r[pad] == 0).all()
    assert not cb.row_is_root[pad].any()


# ---- adjacency ----

def test_adj_role_symmetry_with_direction_flip():
    cb = _cb()
    K = synthetic_bundle().num_roles
    a = cb.adj_role
    child = (a >= 1) & (a <= K)
    parent = (a >= K + 1) & (a <= 2 * K)
    at = a.transpose(1, 2)
    # r sees s as a CHILD  <=>  s sees r as a PARENT, via the SAME role id
    assert torch.equal(child, (at >= K + 1) & (at <= 2 * K))
    assert torch.equal(parent, (at >= 1) & (at <= K))
    assert torch.equal(a[child], at[child] - K)
    # values stay inside the encoding
    assert int(a.max()) <= 2 * K + 1 and int(a.min()) == 0


def test_self_loops_present_on_every_valid_row_only():
    cb = _cb()
    K = synthetic_bundle().num_roles
    diag = torch.diagonal(cb.adj_role, dim1=1, dim2=2)
    assert torch.equal(diag == self_loop_id(K), cb.row_valid)
    # so no row is ever fully masked (self-attention always admissible)
    assert bool(cb.row_mask()[cb.row_valid].any(dim=-1).all())
    # and nothing points at a pad slot
    assert not bool((cb.adj_role != 0)[~cb.row_valid].any())
    assert not bool((cb.adj_role != 0).transpose(1, 2)[~cb.row_valid].any())


def test_every_non_root_valid_row_has_an_edge():
    for cb, bundle in ((_cb(), synthetic_bundle()), (_chain_cb(), chain_bundle())):
        K = bundle.num_roles
        off_diag = (cb.adj_role != 0) & (cb.adj_role != self_loop_id(K))
        has_edge = off_diag.any(dim=2)
        assert bool(has_edge[cb.row_valid & ~cb.row_is_root].all())


def test_row_counts_agree_across_adj_role_num_rows_and_node_idxs():
    cb = _cb()
    K = synthetic_bundle().num_roles
    from_adj = (torch.diagonal(cb.adj_role, dim1=1, dim2=2) == self_loop_id(K)).sum(dim=1)
    assert torch.equal(from_adj, cb.num_rows)
    assert torch.equal(cb.row_valid.sum(dim=1), cb.num_rows)
    for b in range(cb.num_seeds):
        seen = cb.node_idxs[b][~cb.is_padding[b]].unique()
        assert set(seen.tolist()) == set(cb.row_valid[b].nonzero().flatten().tolist())


def test_dual_fk_roles_survive_into_adj_role():
    """The two FKs into `user` are different roles, so the seed row sees two differently-labelled
    children — the row-level payoff of P0.1."""
    b = synthetic_bundle()
    cb = _cb()
    buyer = b.edge_fk_role(("event", "f2p_buyer", "user"))
    seller = b.edge_fk_role(("event", "f2p_seller", "user"))
    root = int(cb.row_is_root[0].nonzero()[0])
    labels = cb.adj_role[0, root][cb.row_valid[0]]
    assert buyer in labels.tolist() and seller in labels.tolist()
    assert sorted(cb.row_in_role[0][cb.row_valid[0] & ~cb.row_is_root[0]].tolist()) == \
        sorted([buyer, seller])
    assert int(cb.row_in_role[0][root]) == 0     # root was not reached by any role


# ---- hop ----

def test_root_and_hop():
    cb = _cb()
    assert torch.equal(cb.row_is_root.sum(dim=1), torch.ones(cb.num_seeds, dtype=torch.long))
    assert (cb.row_hop[cb.row_is_root] == 0).all()
    assert int(cb.row_hop.max()) == 1                     # star: seed + its two children
    assert (cb.row_hop[cb.row_valid] >= 0).all()


def test_two_hop_chain_from_reverse_only_edges():
    """The sampler fills only the store it traversed, so a 2-hop chain arrives entirely in the
    `rev_f2p_*` edge types. The row graph must still be complete, connected and correctly hopped."""
    bundle = chain_bundle()
    cb = _chain_cb(num_hops=2)
    K = bundle.num_roles
    tid = bundle.node_type_id
    for b in range(cb.num_seeds):
        valid = cb.row_valid[b].nonzero().flatten().tolist()
        assert len(valid) == 3
        hop = {int(cb.row_table[b, r]): int(cb.row_hop[b, r]) for r in valid}
        assert hop == {tid["user"]: 0, tid["event"]: 1, tid["comment"]: 2}
        role = {int(cb.row_table[b, r]): int(cb.row_in_role[b, r]) for r in valid}
        assert role[tid["user"]] == 0
        assert role[tid["event"]] == bundle.edge_fk_role(("event", "f2p_buyer", "user"))
        assert role[tid["comment"]] == bundle.edge_fk_role(("comment", "f2p_eventId", "event"))
        # comment is a child of event, event a child of user; user and comment are NOT adjacent
        r_by_t = {int(cb.row_table[b, r]): r for r in valid}
        assert 1 <= int(cb.adj_role[b, r_by_t[tid["event"]], r_by_t[tid["comment"]]]) <= K
        assert int(cb.adj_role[b, r_by_t[tid["user"]], r_by_t[tid["comment"]]]) == 0
    # forward stores were empty, so the cell-level f2p neighbours are genuinely absent here —
    # adj_role is NOT derived from them (see row_graph.py).
    assert not bool((cb.f2p_nbr_idxs >= 0).any())


def test_hop_bound_is_asserted():
    with pytest.raises(AssertionError, match="exceeds sampler depth"):
        _chain_cb(num_hops=1)


# ---- leakage + capacity ----

def _n_row_leaks(cb) -> int:
    """Rows dated after their seed. Two gates, both load-bearing:

    * ``row_is_timed`` — untimed rows carry the sentinel 0, and real timestamps are UNIX seconds
      that go **negative** (rel-f1 starts in 1950), so an ungated comparison flags every untimed row.
    * ``~row_is_root`` — the ROOT row is the query entity, which RelBench includes regardless of its
      own timestamp. Measured on rel-event/user-attendance: **35.0%** of task rows have the `users`
      row's own time after the seed time (median +9.7 days, worst +152 days). That is the benchmark's
      definition of the query, not leakage from a neighbour, so scoping it out is correct — but it
      must be scoped explicitly, or this assert fires on a third of rel-event and looks like a bug.

    Neighbour leakage — the thing that would actually be wrong — is still fully covered.
    """
    later = cb.row_time_r > cb.seed_time.unsqueeze(1)
    return int((later & cb.row_is_timed & cb.row_valid & ~cb.row_is_root).sum())


def test_row_time_never_exceeds_seed_time():
    cb = _cb()
    assert _n_row_leaks(cb) == 0
    # untimed rows are flagged, not silently "time 0"
    assert bool((cb.row_time_r[~cb.row_is_timed] == 0).all())
    assert bool(cb.row_is_timed[cb.row_valid & ~cb.row_is_root].all())


def test_nat_sentinel_timestamp_is_marked_untimed_not_ancient():
    """A missing timestamp must arrive as *untimed*, not as a finite date in 1677.

    ``to_unix_time`` maps NaT to ``NAT_UNIX_SENTINEL`` (-9223372037). Before this was gated, such a
    row kept the table's blanket ``is_timed=True`` and encoded as ~336 years before the seed, giving
    tau = 23.08 — outside changes.md §3.1's universal [0, 22] band. Measured on rel-event: 64918 rows
    in event_attendees and 58 in users.
    """
    from gloss.data.collate import NAT_UNIX_SENTINEL

    def build(t1):
        return to_cell_batch(
            make_synth_batch(seed_time=100.0, event_times=(10.0, t1, 30.0, 40.0)),
            synthetic_bundle(), ENTITY, seq_len=16, max_fk=2, max_rows=8)

    ctl = build(20.0)                            # control: a perfectly ordinary timestamp
    cb = build(float(NAT_UNIX_SENTINEL))         # same batch, that one row's time missing

    # the sentinel never survives as a real timestamp
    assert not bool((cb.row_valid & (cb.row_time_r == float(NAT_UNIX_SENTINEL))).any()), \
        "NaT sentinel leaked through as a real timestamp"
    assert bool((cb.row_time_r[~cb.row_is_timed] == 0).all())

    # differential: exactly ONE more untimed row than the control. The fixture has untimed rows of
    # its own (untimed tables), so an absolute count would be testing the fixture, not the fix.
    n_ctl = int((ctl.row_valid & ~ctl.row_is_timed).sum())
    n_cb = int((cb.row_valid & ~cb.row_is_timed).sum())
    assert n_cb == n_ctl + 1, f"expected {n_ctl} + 1 untimed rows, got {n_cb}"
    # ...and the same number of valid rows: the row is reclassified, never dropped
    assert int(cb.row_valid.sum()) == int(ctl.row_valid.sum())
    # and it does not register as a leak either way
    assert _n_row_leaks(cb) == 0

    # tau stays inside the band that changes.md §3.1 calls universal
    delta = (cb.seed_time.unsqueeze(1) - cb.row_time_r).clamp_min(0.0)
    tau = torch.log1p(delta[cb.row_valid & cb.row_is_timed])
    assert bool((tau <= 22.0).all()), f"tau escaped [0, 22]: max {float(tau.max())}"


def test_planted_row_leak_is_detectable():
    cb = to_cell_batch(make_synth_batch(seed_time=100.0, event_times=(10.0, 150.0, 30.0, 40.0)),
                       synthetic_bundle(), ENTITY, seq_len=16, max_fk=2, max_rows=8)
    assert _n_row_leaks(cb) == 1


def test_max_rows_asserts_and_never_clamps():
    with pytest.raises(AssertionError, match="max_rows_per_seed exceeded"):
        _cb(max_rows=2)          # each seed has 3 rows
    cb = _cb(max_rows=3)
    assert int(cb.num_rows.max()) == 3


def test_max_rows_none_fits_r_to_the_batch():
    """``max_rows=None`` (the default) sizes R to the batch instead of asserting against a constant.

    The fixed 160 was a rel-f1 measurement; rel-event needs 162+ and died on it. Fitting must (a) never
    raise however many rows arrive, (b) report the R it actually built, and (c) leave the row fields
    identical to a hand-picked exact cap — the padding it drops is inert.
    """
    cb = _cb(max_rows=None)
    need = int(cb.num_rows.max())
    assert cb.max_rows == need == 3
    assert cb.adj_role.shape == (cb.num_seeds, need, need)
    assert cb.row_valid.shape == (cb.num_seeds, need)

    exact = _cb(max_rows=need)
    assert torch.equal(cb.adj_role, exact.adj_role)
    assert torch.equal(cb.row_hop, exact.row_hop)
    assert torch.equal(cb.row_in_role, exact.row_in_role)
    assert torch.equal(cb.row_is_root, exact.row_is_root)

    # ...and a generous fixed cap agrees on the populated prefix — fitting only drops pad slots.
    padded = _cb(max_rows=32)
    assert padded.max_rows == 32
    assert torch.equal(padded.adj_role[:, :need, :need], cb.adj_role)
    assert torch.equal(padded.row_valid[:, :need], cb.row_valid)
    assert not bool(padded.row_valid[:, need:].any())


def test_default_max_rows_is_fit_to_batch():
    """The module default must be the fitting one — the bug was that MAX_ROWS=160 (a rel-f1 number)
    was baked in as a cross-dataset constant AND unreachable from config, so it could only be found
    by crashing a job."""
    from gloss.data.collate import MAX_ROWS

    assert MAX_ROWS is None
    cb = to_cell_batch(make_synth_batch(), synthetic_bundle(), ENTITY, seq_len=16, max_fk=2)
    assert cb.max_rows == int(cb.num_rows.max())


# ---- real data ----

@rel_f1_available
def test_rel_f1_row_graph_invariants():
    from ._relf1 import bundle_and_task, sample_cell_batch

    bundle, _task = bundle_and_task()
    _b, _t, cb = sample_cell_batch()
    K = bundle.num_roles
    a = cb.adj_role
    at = a.transpose(1, 2)
    child = (a >= 1) & (a <= K)
    assert torch.equal(child, (at >= K + 1) & (at <= 2 * K))
    assert torch.equal(a[child], at[child] - K)
    assert torch.equal(cb.row_valid.sum(dim=1), cb.num_rows)
    assert torch.equal((torch.diagonal(a, dim1=1, dim2=2) == self_loop_id(K)).sum(dim=1), cb.num_rows)
    assert int(cb.num_rows.max()) <= cb.max_rows
    # exactly one root, hop is a real BFS distance bounded by the sampler depth (fanout (6, 6))
    assert torch.equal(cb.row_is_root.sum(dim=1), torch.ones(cb.num_seeds, dtype=torch.long))
    assert (cb.row_hop[cb.row_is_root] == 0).all()
    assert (cb.row_hop[cb.row_valid] >= 0).all() and int(cb.row_hop.max()) <= 2
    # leakage: never relax this
    assert _n_row_leaks(cb) == 0
    assert bool((cb.row_time_r[~cb.row_is_timed] == 0).all())
    # rows agree with the cells that reference them
    for b in range(cb.num_seeds):
        seen = set(cb.node_idxs[b][~cb.is_padding[b]].unique().tolist())
        assert seen <= set(cb.row_valid[b].nonzero().flatten().tolist())
    # every non-root row is attached, and role labels span more than the 4 ids the old vocab had
    off_diag = (a != 0) & (a != self_loop_id(K))
    assert bool(off_diag.any(dim=2)[cb.row_valid & ~cb.row_is_root].all())
    assert len(set(cb.row_in_role[cb.row_valid & ~cb.row_is_root].tolist())) >= 2
