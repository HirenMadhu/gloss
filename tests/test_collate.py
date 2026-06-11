"""Phase 0 — collate: dense pairwise batch invariants on the synthetic dual-FK fixture."""
from __future__ import annotations

import math

import torch

from gloss.data.collate import to_gloss_batch
from gloss.data.graph import FK_NONE, MP_SELF


def _gb(dualfk_batch, dualfk_bundle):
    return to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)


def test_node_counts_and_padding(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    assert gb.num_seeds == 2
    # each segment: 1 user + 2 events = 3 real nodes
    assert gb.n_max == 3
    assert int(gb.pad_mask.sum()) == 6
    assert gb.pad_mask.all()  # no padding in this balanced fixture (3 each)
    # exactly one seed per segment
    assert gb.is_seed.sum(dim=1).tolist() == [1, 1]


def test_seed_time_and_timed_flags(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    assert torch.allclose(gb.seed_time, torch.tensor([100.0, 100.0], dtype=torch.float64))
    # users timeless, events timed -> 2 timed per segment
    assert gb.is_timed.sum(dim=1).tolist() == [2, 2]


def test_diagonal_is_self_metapath(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    for b in range(gb.num_seeds):
        for i in range(gb.n_max):
            assert gb.metapath_id[b, i, i].item() == MP_SELF
            assert gb.attend_mask[b, i, i].item() is True or bool(gb.attend_mask[b, i, i])


def test_attend_mask_symmetric_and_within_subgraph(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    am = gb.attend_mask
    assert torch.equal(am, am.transpose(1, 2)), "attend mask must be symmetric"
    # no attention across segments is possible (dense tensors are per-segment), and all within-seg
    # event<->event are 2-hop via the shared user, so fully connected 3-node subgraph -> all True
    assert am.sum().item() == 2 * 3 * 3


def test_fk_role_on_one_hop_pairs(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    bundle = dualfk_bundle
    buyer = bundle.relation_fk_role("f2p_buyer")
    seller = bundle.relation_fk_role("f2p_seller")
    # find the user (seed) local index and the two event indices in segment 0
    seg = 0
    types = gb.node_type_id[seg]
    user_idx = (types == bundle.node_type_id["user"]).nonzero().flatten().tolist()
    event_idx = (types == bundle.node_type_id["event"]).nonzero().flatten().tolist()
    assert len(user_idx) == 1 and len(event_idx) == 2
    u = user_idx[0]
    roles = {gb.fk_role_id[seg, u, e].item() for e in event_idx} | {
        gb.fk_role_id[seg, e, u].item() for e in event_idx
    }
    assert buyer in roles and seller in roles  # both FK roles present and distinct
    # event<->event (2-hop) carries no direct fk role
    e0, e1 = event_idx
    assert gb.fk_role_id[seg, e0, e1].item() == FK_NONE


def test_temporal_valid_and_tau_formula(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    seg = 0
    types = gb.node_type_id[seg]
    bundle = dualfk_bundle
    event_idx = (types == bundle.node_type_id["event"]).nonzero().flatten().tolist()
    e0, e1 = event_idx
    # event<->event pair is both-timed and attends -> temporal_valid; user pairs are timeless -> not
    assert bool(gb.temporal_valid[seg, e0, e1])
    user_idx = (types == bundle.node_type_id["user"]).nonzero().flatten().item()
    assert not bool(gb.temporal_valid[seg, user_idx, e0])
    # tau = log(dt / T_ctx); with a single nonzero gap, T_ctx = that gap -> tau = 0
    dt = gb.dt[seg, e0, e1].item()
    assert dt == abs(10.0 - 20.0)
    assert math.isclose(gb.tau[seg, e0, e1].item(), math.log(dt / gb.t_ctx[seg].item()), abs_tol=1e-9)


def test_no_temporal_valid_crosses_seed_time(dualfk_batch, dualfk_bundle):
    gb = _gb(dualfk_batch, dualfk_bundle)
    # every timed node's row_time must be <= its seed_time (leak-free fixture)
    bad = (gb.is_timed & (gb.row_time > gb.seed_time.view(-1, 1))).sum().item()
    assert bad == 0
