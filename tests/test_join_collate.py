"""SetJoin collate contracts: wide m2o walk, markers, union set, parent flattening, caps."""
from __future__ import annotations

import torch

from gloss.data.collate import column_vocab
from gloss.data.graph import FK_NONE
from gloss.setjoin.collate import to_join_batch
from gloss.setjoin.paths import child_rels, m2o_paths, parent_rels, setjoin_neighbors

from ._join_fixtures import ENTITY as ORDER
from ._join_fixtures import chain_bundle, make_chain_batch, make_hop2_batch
from .conftest import ENTITY as USER
from .conftest import make_synth_batch, rel_f1_available, synthetic_bundle


def _jb(wide_len=16, set_size=8, **kw):
    return to_join_batch(make_chain_batch(**kw), chain_bundle(), ORDER,
                         wide_len=wide_len, set_size=set_size)


# ---------- schema maps ----------

def test_schema_maps():
    b = chain_bundle()
    assert parent_rels(b, "order") == [("order", "customer", "customer")]
    assert parent_rels(b, "payment") == [("payment", "method", "method"),
                                         ("payment", "order", "order")]
    assert child_rels(b, "order") == [("payment", "order", "order")]
    paths = m2o_paths(b, "order")
    assert paths[0] == ()                                  # seed first == path_id 0
    assert paths[1] == (("order", "customer", "customer"),)
    assert paths[2] == (("order", "customer", "customer"), ("customer", "region", "region"))
    assert len(paths) == 3


def test_m2o_paths_self_fk_terminates():
    from gloss.data.graph import GraphBundle, _build_vocabs

    nts = ["emp"]
    ets = [("emp", "f2p_boss", "emp"), ("emp", "rev_f2p_boss", "emp")]
    ntid, fkid, mpid = _build_vocabs(nts, ets)
    b = GraphBundle(dataset_name="synthetic-selffk", data=None, col_stats_dict={}, node_types=nts,
                    edge_types=ets, node_type_id=ntid, fk_role_id=fkid, metapath_id=mpid)
    paths = m2o_paths(b, "emp", depth=2)
    assert len(paths) == 3                                 # (), (boss,), (boss, boss)


def test_setjoin_neighbors_dict():
    # PyG budgets src-side neighbors per DST frontier node: forward f2p types (dst=parent) pull
    # CHILDREN and get the fanout; rev types (dst=child) pull the unique parent and get [1, 1].
    b = chain_bundle()
    nn = setjoin_neighbors(b, fanout=32)
    assert set(nn) == set(b.edge_types)
    assert nn[("order", "f2p_customer", "customer")] == [32, 0]   # customer's child orders (o2m)
    assert nn[("payment", "f2p_order", "order")] == [32, 0]       # order's child payments (o2m)
    assert nn[("order", "rev_f2p_order", "payment")] == [1, 1]    # payment's parent order (m2o)
    assert nn[("customer", "rev_f2p_customer", "order")] == [1, 1]


def test_setjoin_neighbors_fanout2():
    # hop-2 arm: three layers so the co-child walk seed→child→parent→child is reachable
    b = chain_bundle()
    nn = setjoin_neighbors(b, fanout=32, fanout2=8)
    assert nn[("payment", "f2p_order", "order")] == [32, 8, 8]
    assert nn[("order", "rev_f2p_order", "payment")] == [1, 1, 1]
    assert setjoin_neighbors(b, fanout=32, fanout2=0) == setjoin_neighbors(b, fanout=32)


@rel_f1_available
def test_sampler_multiplicity_and_parent_flattening_rel_f1():
    """Regression test for the v1-gate fanout inversion: on real sampled data the union set must
    contain MANY children per seed (not ~1 per relation) and elements must flatten their non-seed
    parents (races/constructors for a driver's results)."""
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout=16),
                         batch_size=64, shuffle=True)
    jb = to_join_batch(next(iter(loader)), bundle, task.entity_table, wide_len=64, set_size=64)
    assert float(jb.child_counts.max()) > 1.0, "children capped at 1 per relation: fanout inverted?"
    assert float(jb.elem_mask.sum(1).max()) >= 8, "union sets implausibly small on rel-f1"
    assert "races" in jb.elem_rows, "elements lost their flattened parents: rev hop-2 budget missing?"


# ---------- wide seed row ----------

def test_wide_paths_cells_and_markers():
    jb = _jb()
    vocab = column_vocab(chain_bundle())
    ntid = chain_bundle().node_type_id
    # segment 0: 2 order cells (path 0) + c_name (path 1) + r_code (path 2), no markers
    real0 = (~jb.wide_is_pad[0] & ~jb.wide_missing[0])
    assert int(real0.sum()) == 4 and int(jb.wide_missing[0].sum()) == 0
    assert jb.wide_path_idxs[0, :4].tolist() == [0, 0, 1, 2]
    assert jb.wide_col_idxs[0, :4].tolist() == [
        vocab[("order", "o_amt")], vocab[("order", "o_qty")],
        vocab[("customer", "c_name")], vocab[("region", "r_code")]]
    # segment 1: 2 order cells + 2 missing markers (customer path and customer→region path)
    assert int((~jb.wide_is_pad[1] & ~jb.wide_missing[1]).sum()) == 2
    assert int(jb.wide_missing[1].sum()) == 2
    marker_slots = jb.wide_missing[1].nonzero().squeeze(-1)
    assert jb.wide_path_idxs[1, marker_slots].tolist() == [1, 2]
    assert jb.wide_table_idxs[1, marker_slots].tolist() == [ntid["customer"], ntid["region"]]
    assert (jb.wide_col_idxs[1, marker_slots] == -1).all()
    # wide row_time carried from the node (orders timed at 50/60)
    assert float(jb.wide_row_time[0, 0]) == 50.0 and bool(jb.wide_is_timed[0, 0])


def test_wide_truncation_keeps_seed_cells_first():
    jb = _jb(wide_len=3)
    # seg 0: 4 cells > 3; seg 1: 2 cells + 2 markers > 3 (markers count toward the cap)
    assert jb.wide_truncated == 2
    assert jb.wide_path_idxs[0, :3].tolist() == [0, 0, 1]  # seed cells survive, region dropped
    assert jb.wide_path_idxs[1, :3].tolist() == [0, 0, 1] and bool(jb.wide_missing[1, 2])


def test_wide_placement_scatter_map():
    jb = _jb()
    b_idx, w_idx, row_idx, col_idx = jb.wide_placement["order"]
    assert b_idx.tolist() == [0, 0, 1, 1] and w_idx.tolist() == [0, 1, 0, 1]
    assert row_idx.tolist() == [0, 0, 1, 1] and col_idx.tolist() == [0, 1, 0, 1]
    assert set(jb.wide_placement) == {"order", "customer", "region"}   # markers place nothing


# ---------- union set ----------

def test_union_set_from_rev_only_edges():
    # payments are attached via rev_f2p_order ONLY — the collate must still find them
    jb = _jb()
    assert jb.elem_mask[0].sum() == 2 and jb.elem_mask[1].sum() == 1
    assert jb.child_counts.tolist() == [[2.0], [1.0]]
    role = chain_bundle().fk_role_id["order"]
    ntid = chain_bundle().node_type_id["payment"]
    assert jb.elem_rel_idxs[0, :2].tolist() == [role, role]
    assert jb.elem_table_idxs[0, :2].tolist() == [ntid, ntid]
    assert (jb.elem_hop[0, :2] == 1).all()


def test_elements_sorted_most_recent_first_and_capped():
    jb = _jb(set_size=1)
    assert jb.set_truncated == 1                           # segment 0 had 2 children
    assert int(jb.elem_mask[0].sum()) == 1
    assert float(jb.elem_row_time[0, 0]) == 20.0           # P1 (t=20) beats P0 (t=10)


def test_elem_parent_flattening_and_seed_exclusion():
    jb = _jb()
    # every element carries its own payment row at path FK_NONE
    b_idx, n_idx, row_idx, path_idx = jb.elem_rows["payment"]
    assert (path_idx == FK_NONE).all() and len(b_idx) == 3
    # P0's method parent is flattened in at the method FK's role id
    role_method = chain_bundle().fk_role_id["method"]
    b_idx, n_idx, row_idx, path_idx = jb.elem_rows["method"]
    assert b_idx.tolist() == [0] and row_idx.tolist() == [0]
    assert path_idx.tolist() == [role_method]
    p_slot = int(n_idx[0])
    assert float(jb.elem_row_time[0, p_slot]) == 10.0      # attached to P0's element
    # payments' order parents are the SEEDS -> excluded by n_id; nothing scatters into "order" elements
    assert "order" not in jb.elem_rows


def test_no_cross_product():
    # hop-1 contract: under the cap, hop-1 elements are 1:1 with sampled children (child_counts is
    # hop-1-only by definition — the same numbers as the pre-hop2 total-count contract when hop2 off)
    jb = _jb()
    assert float(((jb.elem_hop == 1) & jb.elem_mask).sum()) == float(jb.child_counts.sum())
    assert float(jb.elem_mask.sum()) == float(jb.child_counts.sum())
    jb2 = to_join_batch(make_hop2_batch(), chain_bundle(), ORDER, wide_len=16, set_size=8, hop2=True)
    assert float(((jb2.elem_hop == 1) & jb2.elem_mask).sum()) == float(jb2.child_counts.sum())


# ---------- hop-2 union elements (default off) ----------

def _jb2(set_size=8, hop2=True, per_relation_cap=None, **kw):
    return to_join_batch(make_hop2_batch(**kw), chain_bundle(), ORDER,
                         wide_len=16, set_size=set_size, hop2=hop2,
                         per_relation_cap=per_relation_cap)


def test_hop2_default_off_is_unchanged():
    # hop2=False on a batch CONTAINING hop-2 rows: only the direct children become elements
    jb = _jb2(hop2=False)
    assert int(jb.elem_mask.sum()) == 2                       # P1, P0 only
    assert (jb.elem_hop[jb.elem_mask] == 1).all()
    assert jb.elem_row_time[0, :2].tolist() == [20.0, 10.0]   # most-recent-first, as before
    assert "order" not in jb.elem_rows and "refund" not in jb.elem_rows


def test_hop2_grandchild_sibling_and_cochild_tagged():
    jb = _jb2()
    b = chain_bundle()
    assert int(jb.elem_mask.sum()) == 5                       # P1 P0 | O2 P3 RF0
    assert jb.elem_hop[0, :5].tolist() == [1, 1, 2, 2, 2]
    # hop-1 first, then hop-2 by recency: O2 (t=40) > P3 (t=15) > RF0 (t=12)
    assert jb.elem_row_time[0, :5].tolist() == [20.0, 10.0, 40.0, 15.0, 12.0]
    assert jb.elem_table_idxs[0, 2:5].tolist() == [
        b.node_type_id["order"], b.node_type_id["payment"], b.node_type_id["refund"]]
    # role = LAST traversed FK: sibling via customer, co-child via method, grandchild via payment
    assert jb.elem_rel_idxs[0, 2:5].tolist() == [
        b.fk_role_id["customer"], b.fk_role_id["method"], b.fk_role_id["payment"]]


def test_hop2_seed_and_duplicate_dedup():
    jb = _jb2()
    # the sibling family reaches a duplicate copy of the seed (n_id 0) — never re-emitted; the
    # co-child family reaches P0 again (already hop-1) — not duplicated. Filter to each element's
    # OWN row (path FK_NONE) — the scatter also carries flattened parents (e.g. P0 under RF0).
    def own_rows(nt):
        _, _, row_idx, path_idx = jb.elem_rows[nt]
        return sorted(int(r) for r, p in zip(row_idx, path_idx) if int(p) == FK_NONE)

    assert own_rows("order") == [1]                           # O2 only; rows 0/2 (seed copies) never
    assert own_rows("payment") == [0, 1, 2]                   # P0, P1, P3 exactly once each
    assert own_rows("refund") == [0]                          # RF0 (P0 reappears only as its parent)


def test_hop2_sort_hop1_first_and_cap():
    # cap 3: both hop-1 payments survive; only the most recent hop-2 element (O2) fits
    jb = _jb2(set_size=3)
    assert jb.set_truncated == 1
    assert jb.elem_hop[0, :3].tolist() == [1, 1, 2]
    assert jb.elem_row_time[0, :3].tolist() == [20.0, 10.0, 40.0]


def test_hop2_child_counts_hop1_only():
    jb = _jb2()
    assert jb.child_counts.tolist() == [[2.0]]                # P0, P1 — hop-2 rows never counted


# ---------- per-relation recency cap (default off) ----------

def test_per_relation_cap_keeps_most_recent():
    # chain fixture: 2 payments in segment 0 (t=10, 20); cap 1 keeps the most recent only,
    # child_counts stays PRE-cap (the model keeps the true-count signal)
    jb = to_join_batch(make_chain_batch(), chain_bundle(), ORDER,
                       wide_len=16, set_size=8, per_relation_cap=1)
    assert int(jb.elem_mask[0].sum()) == 1
    assert float(jb.elem_row_time[0, 0]) == 20.0              # P1 survives, P0 capped away
    assert jb.child_counts.tolist() == [[2.0], [1.0]]         # pre-cap counts intact
    assert jb.set_truncated == 0                              # the cap is not set_size truncation


def test_per_relation_cap_is_per_relation():
    # dual-FK fixture: buyer + seller relations; cap 1 keeps one element of EACH role
    jb = to_join_batch(make_synth_batch(), synthetic_bundle(), USER, wide_len=8, set_size=8,
                       per_relation_cap=1)
    fk = synthetic_bundle().fk_role_id
    assert int(jb.elem_mask[0].sum()) == 2
    assert sorted(jb.elem_rel_idxs[0, :2].tolist()) == sorted([fk["buyer"], fk["seller"]])


def test_per_relation_cap_default_off_unchanged():
    a = to_join_batch(make_chain_batch(), chain_bundle(), ORDER, wide_len=16, set_size=8)
    b = to_join_batch(make_chain_batch(), chain_bundle(), ORDER, wide_len=16, set_size=8,
                      per_relation_cap=None)
    assert torch.equal(a.elem_mask, b.elem_mask) and torch.equal(a.elem_row_time, b.elem_row_time)
    assert torch.equal(a.elem_rel_idxs, b.elem_rel_idxs)


def test_per_relation_cap_composes_with_hop2():
    # groups are (hop, table, role): capping at 1 keeps the most recent of EACH hop-2 family
    # (order-sibling, payment-co-child, refund-grandchild are separate groups) and 1 hop-1 payment
    jb = _jb2(per_relation_cap=1)
    assert int(jb.elem_mask.sum()) == 4                       # P1 | O2, P3, RF0
    assert jb.elem_hop[0, :4].tolist() == [1, 2, 2, 2]
    assert jb.elem_row_time[0, :4].tolist() == [20.0, 40.0, 15.0, 12.0]
    assert jb.child_counts.tolist() == [[2.0]]                # still pre-cap


def test_dualfk_distinct_rel_ids_and_seed_exclusion():
    # conftest dual-FK schema: events reach the seed user via buyer AND seller relations
    jb = to_join_batch(make_synth_batch(), synthetic_bundle(), USER, wide_len=8, set_size=8)
    fk = synthetic_bundle().fk_role_id
    assert int(jb.elem_mask[0].sum()) == 2 and int(jb.elem_mask[1].sum()) == 2
    assert sorted(jb.elem_rel_idxs[0, :2].tolist()) == sorted([fk["buyer"], fk["seller"]])
    # user (seed) has no m2o parents; elements are event rows only, seed never re-injected
    assert set(jb.elem_rows) == {"event"}
    # wide row: user is parentless -> exactly its own cell, no markers
    assert int((~jb.wide_is_pad[0]).sum()) == 1 and int(jb.wide_missing.sum()) == 0
    assert jb.child_counts.shape == (2, 2)                 # R=2 child relations (buyer, seller)


# ---------- dtypes / bookkeeping ----------

def test_per_row_metadata_for_axial_grids():
    jb = _jb()
    # every TensorFrame row carries its seed/segment + time (payment rows: segs 0,0,1; timed)
    assert jb.row_seg["payment"].tolist() == [0, 0, 1]
    assert jb.row_times["payment"].tolist() == [10.0, 20.0, 30.0]
    assert jb.row_timed["payment"].all()
    assert jb.row_seg["order"].tolist() == [0, 1]
    assert not jb.row_timed["customer"].any()               # untimed table
    assert set(jb.row_seg) == set(jb.tf_dict)


def test_shapes_dtypes_and_input_id():
    jb = _jb()
    assert jb.wide_col_idxs.dtype == torch.long and jb.elem_rel_idxs.dtype == torch.long
    assert jb.wide_row_time.dtype == torch.float64 and jb.elem_row_time.dtype == torch.float64
    assert jb.child_counts.dtype == torch.float32
    assert jb.input_id.tolist() == [-1, -1]                # absent in fixtures
    assert jb.seed_time.tolist() == [100.0, 100.0]
    for t in (jb.wide_col_idxs, jb.wide_is_pad, jb.wide_missing):
        assert t.shape == (2, 16)
    for t in (jb.elem_mask, jb.elem_rel_idxs, jb.elem_row_time, jb.elem_hop):
        assert t.shape == (2, 8)
    assert "JoinBatch" in jb.pretty_shapes()
