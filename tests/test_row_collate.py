"""R1 tests for the unified RowModel collate (``to_row_set_batch`` / ``RowSetBatch`` / ``row_paths``).

Hermetic on the chain fixture (payment→order→customer→region + side ``method`` parent, seed=order) and a
purpose-built dual-FK-to-entity batch. No network, no pytorch-frame stats — the collate reads column
identities, not values.
"""
from __future__ import annotations

import torch

from gloss.data.graph import _build_vocabs, GraphBundle
from gloss.setjoin.collate import to_row_set_batch
from gloss.setjoin.paths import child_rels, parent_rels, row_paths

from ._join_fixtures import chain_bundle, make_chain_batch
from .conftest import _tf


# --------------------------------------------------------------------------------------------------
# a dual-FK-to-entity batch: an event with BOTH FKs to user present (buyer=seed, seller=a non-seed user)
# --------------------------------------------------------------------------------------------------
_DFK_NODE_TYPES = ["event", "user"]
_DFK_EDGE_TYPES = [
    ("event", "f2p_buyer", "user"),
    ("user", "rev_f2p_buyer", "event"),
    ("event", "f2p_seller", "user"),
    ("user", "rev_f2p_seller", "event"),
]


def _dfk_bundle() -> GraphBundle:
    from torch_geometric.data import HeteroData

    node_type_id, fk_role_id, metapath_id = _build_vocabs(_DFK_NODE_TYPES, _DFK_EDGE_TYPES)
    g = HeteroData()
    g["user"].tf = _tf(1, ["u_attr"])
    g["event"].tf = _tf(1, ["e_x", "e_y"])
    return GraphBundle(
        dataset_name="synthetic-dualfk-entity", data=g, col_stats_dict={},
        node_types=_DFK_NODE_TYPES, edge_types=_DFK_EDGE_TYPES,
        node_type_id=node_type_id, fk_role_id=fk_role_id, metapath_id=metapath_id,
    )


def _dfk_batch(seed_time: float = 100.0):
    """1 seed user0 (n_id 0); event E0 (buyer=user0=seed, seller=user1 a non-seed neighbour)."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["user"].num_nodes = 2                          # row 0 = seed, row 1 = the seller neighbour
    d["user"].batch = torch.tensor([0, 0])
    d["user"].seed_time = torch.tensor([seed_time], dtype=torch.float64)
    d["user"].n_id = torch.tensor([0, 1])
    d["user"].tf = _tf(2, ["u_attr"])

    d["event"].num_nodes = 1
    d["event"].batch = torch.tensor([0])
    d["event"].time = torch.tensor([50.0], dtype=torch.float64)
    d["event"].n_id = torch.tensor([10])
    d["event"].tf = _tf(1, ["e_x", "e_y"])

    d["event", "f2p_buyer", "user"].edge_index = torch.tensor([[0], [0]])    # E0 buyer -> user0 (seed)
    d["event", "f2p_seller", "user"].edge_index = torch.tensor([[0], [1]])   # E0 seller -> user1
    d["user", "rev_f2p_buyer", "event"].edge_index = torch.tensor([[0], [0]])  # user0's child E0
    return d


# --------------------------------------------------------------------------------------------------
# row_paths
# --------------------------------------------------------------------------------------------------
def test_row_paths_namespace_is_total_and_cached():
    b = chain_bundle()
    rp = row_paths(b, "order")
    # block A = m2o paths (seed's own row + customer + region)
    assert len(rp.m2o) == 3 and rp.self_id == 3
    # block C covers EVERY parent relation of the child type (payment): order AND method
    prks = parent_rels(b, "payment")
    assert set(rp.parent_lookup) == set(prks)
    assert rp.n_row_paths == 3 + 1 + len(prks)
    ids = list(range(rp.self_id)) + [rp.self_id] + list(rp.parent_lookup.values())
    assert sorted(ids) == list(range(rp.n_row_paths))            # disjoint & total
    assert row_paths(b, "order") is rp                            # cached


def test_row_paths_keeps_second_fk_to_entity():
    """The blocker: a dual-FK child (event.buyer AND event.seller both -> user) must have a path id
    for the SECOND relation, not a schema-level 'non-seed parent' exclusion."""
    b = _dfk_bundle()
    rp = row_paths(b, "user")
    prks = parent_rels(b, "event")                               # both point at the entity `user`
    assert all(pk[2] == "user" for pk in prks) and len(prks) == 2
    assert set(rp.parent_lookup) == set(prks)                    # BOTH kept


# --------------------------------------------------------------------------------------------------
# membership & OBT denormalization
# --------------------------------------------------------------------------------------------------
def test_membership_row0_and_one_row_per_child():
    b = chain_bundle()
    rb = to_row_set_batch(make_chain_batch(), b, "order", m_rows=16, cells_per_row=16)
    assert rb.row_mask[:, 0].all()                               # Row 0 present for every seed
    # seg 0 has 2 children (P0, P1) -> 3 rows; seg 1 has 1 child (P2) -> 2 rows
    assert int(rb.row_mask[0].sum()) == 3
    assert int(rb.row_mask[1].sum()) == 2
    assert rb.row_hop[0, 0] == 0 and (rb.row_hop[0, 1:3] == 1).all()


def test_obt_seed_closure_byte_identical_and_child_parent_kept():
    b = chain_bundle()
    rb = to_row_set_batch(make_chain_batch(), b, "order", m_rows=16, cells_per_row=16)
    rp = row_paths(b, "order")
    vocab_method = None
    for seg in (0, 1):
        L = int(rb.cell_mask[seg, 0].sum())                     # Row 0's cell count = seed closure len
        for mrow in range(1, int(rb.row_mask[seg].sum())):
            for t in (rb.cell_col_id, rb.cell_path_id, rb.cell_table_id, rb.cell_missing,
                      rb.cell_is_timed):
                assert torch.equal(t[seg, mrow, :L], t[seg, 0, :L])   # byte-identical seed cells
            assert torch.equal(rb.cell_row_time[seg, mrow, :L], rb.cell_row_time[seg, 0, :L])
    # seg 0: P0's row (Row 2, since most-recent-first = P1,P0) carries its method parent (a NON-seed
    # parent) at the method block-C path id; the order parent (== seed) is NOT separately emitted.
    from gloss.data.collate import column_vocab
    v = column_vocab(b)
    mid = v[("method", "m_kind")]
    method_prk = next(pk for pk in parent_rels(b, "payment") if pk[2] == "method")
    method_pid = rp.parent_lookup[method_prk]
    row = rb.cell_path_id[0, 2]
    hit = (rb.cell_col_id[0, 2] == mid) & (row == method_pid) & rb.cell_mask[0, 2]
    assert int(hit.sum()) == 1
    # the seed (order) never appears as a child's *parent* path id (only in the closure, path 0..2)
    order_prk = next(pk for pk in parent_rels(b, "payment") if pk[2] == "order")
    assert not bool((rb.cell_path_id[0, 2] == rp.parent_lookup[order_prk]).any())


def test_dual_fk_second_parent_is_emitted_with_defined_path_id():
    b = _dfk_bundle()
    rp = row_paths(b, "user")
    rb = to_row_set_batch(_dfk_batch(), b, "user", m_rows=8, cells_per_row=8)
    from gloss.data.collate import column_vocab
    v = column_vocab(b)
    u_attr = v[("user", "u_attr")]
    seller_prk = next(pk for pk in parent_rels(b, "event") if pk[1] == "seller")
    seller_pid = rp.parent_lookup[seller_prk]
    # E0 is Row 1 (Row 0 = the seed user's own row). Its seller parent user1 must be present.
    row1 = rb.cell_path_id[0, 1]
    hit = (rb.cell_col_id[0, 1] == u_attr) & (row1 == seller_pid) & rb.cell_mask[0, 1]
    assert int(hit.sum()) == 1
    assert 0 <= seller_pid < rp.n_row_paths


# --------------------------------------------------------------------------------------------------
# leakage, caps, shapes
# --------------------------------------------------------------------------------------------------
def _n_leaks(rb) -> int:
    bad = (rb.cell_row_time > rb.seed_time[:, None, None]) & rb.cell_is_timed & rb.cell_mask
    return int(bad.sum())


def test_no_leakage_and_planted_leak_detected():
    b = chain_bundle()
    assert _n_leaks(to_row_set_batch(make_chain_batch(), b, "order")) == 0
    planted = make_chain_batch(payment_times=(10.0, 20.0, 150.0))   # P2 (seg 1) at t=150 > 100
    assert _n_leaks(to_row_set_batch(planted, b, "order")) > 0


def test_caps_and_truncation_bookkeeping():
    b = chain_bundle()
    rb = to_row_set_batch(make_chain_batch(), b, "order", m_rows=2, cells_per_row=3)
    # m_rows=2 -> seg 0 (2 children) keeps Row 0 + 1 child, dropping one -> row_truncated
    assert rb.row_truncated >= 1
    assert rb.row_mask.shape == (2, 2) and rb.cell_mask.shape == (2, 2, 3)
    # cells_per_row=3 truncates the wider rows (P0 row has closure(4)+P0+M0 = 6 cells)
    assert rb.cell_truncated >= 1


def test_all_path_ids_in_range_and_shapes_dtypes():
    b = chain_bundle()
    rp = row_paths(b, "order")
    rb = to_row_set_batch(make_chain_batch(), b, "order", m_rows=16, cells_per_row=16)
    real = rb.cell_path_id[rb.cell_mask]
    assert int(real.min()) >= 0 and int(real.max()) < rp.n_row_paths
    assert rb.cell_col_id.dtype == torch.long and rb.cell_row_time.dtype == torch.float64
    assert rb.cell_mask.dtype == torch.bool and rb.child_counts.dtype == torch.float32
    assert (rb.input_id == -1).all()                            # fixtures carry no split-table id
    assert rb.row_table_id[0, 0] == b.node_type_id["order"]     # Row 0 anchored at the entity table
    assert rb.row_fk_role[0, 0] == 0                            # FK_NONE for the seed's own row


def test_to_device_and_pretty_shapes():
    b = chain_bundle()
    rb = to_row_set_batch(make_chain_batch(), b, "order")
    rb2 = rb.to("cpu")                                          # no-op move exercises the carry
    assert rb2.cell_mask.shape == rb.cell_mask.shape
    assert "RowSetBatch" in rb.pretty_shapes()
