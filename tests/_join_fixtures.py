"""Hermetic fixtures for the SetJoin collate: a 3-level chain schema with a side parent.

Schema (FKs point right):   payment ─f2p_order→ order ─f2p_customer→ customer ─f2p_region→ region
                            payment ─f2p_method→ method
                            refund  ─f2p_payment→ payment          (hop-2 grandchild family)

Seed = ``order``. This exercises, without any network or pytorch-frame stats:
* the depth-2 m2o wide-row walk (order + customer + region cells, path-tagged),
* the missing-parent marker (segment 1's order has no customer edge),
* the union set built from REV-ONLY edge instances (payments are attached only via
  ``rev_f2p_order``, the direction a real PyG sampler stores them under),
* element parent flattening with seed exclusion (a payment's ``order`` parent IS the seed → excluded;
  its ``method`` parent is kept).

Like ``conftest``, TensorFrames carry only ``col_names_dict`` — the collate reads column identities,
not values, so no stats are needed.
"""
from __future__ import annotations

import torch

from gloss.data.graph import GraphBundle, _build_vocabs

from .conftest import _tf

ENTITY = "order"
NODE_TYPES = ["customer", "method", "order", "payment", "refund", "region"]
EDGE_TYPES = [
    ("order", "f2p_customer", "customer"),
    ("customer", "rev_f2p_customer", "order"),
    ("customer", "f2p_region", "region"),
    ("region", "rev_f2p_region", "customer"),
    ("payment", "f2p_order", "order"),
    ("order", "rev_f2p_order", "payment"),
    ("payment", "f2p_method", "method"),
    ("method", "rev_f2p_method", "payment"),
    ("refund", "f2p_payment", "payment"),
    ("payment", "rev_f2p_payment", "refund"),
]
ORDER_COLS = ["o_amt", "o_qty"]
CUSTOMER_COLS = ["c_name"]
REGION_COLS = ["r_code"]
PAYMENT_COLS = ["p_val"]
METHOD_COLS = ["m_kind"]
REFUND_COLS = ["rf_amt"]
COLS = {"order": ORDER_COLS, "customer": CUSTOMER_COLS, "region": REGION_COLS,
        "payment": PAYMENT_COLS, "method": METHOD_COLS, "refund": REFUND_COLS}


def _schema_graph():
    from torch_geometric.data import HeteroData

    d = HeteroData()
    for nt, cols in COLS.items():
        d[nt].tf = _tf(1, cols)
    return d


def chain_bundle() -> GraphBundle:
    node_type_id, fk_role_id, metapath_id = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    return GraphBundle(
        dataset_name="synthetic-chain",
        data=_schema_graph(),
        col_stats_dict={},
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


def make_chain_batch(seed_time: float = 100.0, payment_times=(10.0, 20.0, 30.0)):
    """A 2-seed disjoint batch.

    Segment 0: order O0 (t=50) ⋈ customer C0 ⋈ region R0; children payments P0 (t=payment_times[0],
    method M0) and P1 (t=payment_times[1], no method edge).
    Segment 1: order O1 (t=60), NO customer edge (→ missing markers for both m2o paths); child P2.

    Payments are attached to their orders ONLY via ``rev_f2p_order`` instances (traversal direction of a
    real sampler); the f2p direction is present for payment→method and order→customer→region.
    """
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["order"].num_nodes = 2
    d["order"].batch = torch.tensor([0, 1])
    d["order"].seed_time = torch.tensor([seed_time, seed_time], dtype=torch.float64)
    d["order"].time = torch.tensor([50.0, 60.0], dtype=torch.float64)
    d["order"].n_id = torch.tensor([0, 1])
    d["order"].tf = _tf(2, ORDER_COLS)

    d["customer"].num_nodes = 1
    d["customer"].batch = torch.tensor([0])
    d["customer"].n_id = torch.tensor([100])
    d["customer"].tf = _tf(1, CUSTOMER_COLS)

    d["region"].num_nodes = 1
    d["region"].batch = torch.tensor([0])
    d["region"].n_id = torch.tensor([200])
    d["region"].tf = _tf(1, REGION_COLS)

    d["payment"].num_nodes = 3
    d["payment"].batch = torch.tensor([0, 0, 1])
    d["payment"].time = torch.tensor(list(payment_times), dtype=torch.float64)
    d["payment"].n_id = torch.tensor([300, 301, 302])
    d["payment"].tf = _tf(3, PAYMENT_COLS)

    d["method"].num_nodes = 1
    d["method"].batch = torch.tensor([0])
    d["method"].n_id = torch.tensor([400])
    d["method"].tf = _tf(1, METHOD_COLS)

    # m2o closure: forward (traversal) direction
    d["order", "f2p_customer", "customer"].edge_index = torch.tensor([[0], [0]])
    d["customer", "f2p_region", "region"].edge_index = torch.tensor([[0], [0]])
    d["payment", "f2p_method", "method"].edge_index = torch.tensor([[0], [0]])
    # children: REV-ONLY instances (order -> payment), as a real sampler stores them
    d["order", "rev_f2p_order", "payment"].edge_index = torch.tensor([[0, 0, 1], [0, 1, 2]])
    return d


def make_hop2_batch(seed_time: float = 100.0, refund_time: float = 12.0):
    """A 1-seed disjoint batch exercising every hop-2 family and both dedup rules.

    Seed O0 (t=50, n_id 0) ⋈ C0 ⋈ R0. Hop-1 children: P0 (t=10, parent M0) and P1 (t=20).
    Hop-2 candidates:
    * grandchild  RF0 (t=refund_time, n_id 500)  — child of P0 via ``refund→payment``;
    * sibling     O2  (t=40, n_id 1)             — child of C0 (the seed's parent) via ``rev_f2p_customer``;
    * co-child    P3  (t=15, n_id 303)           — child of M0 (a hop-1 child's parent) via ``rev_f2p_method``;
    * EXCLUDED: a duplicate copy of the seed (order store row 2, n_id 0) attached as C0's child, and
      P0 reattached as M0's child (already a hop-1 element) — both must be deduped by n_id.
    """
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["order"].num_nodes = 3                                  # row 0 = seed O0, 1 = O2, 2 = seed dup
    d["order"].batch = torch.tensor([0, 0, 0])
    d["order"].seed_time = torch.tensor([seed_time], dtype=torch.float64)
    d["order"].time = torch.tensor([50.0, 40.0, 50.0], dtype=torch.float64)
    d["order"].n_id = torch.tensor([0, 1, 0])
    d["order"].tf = _tf(3, ORDER_COLS)

    d["customer"].num_nodes = 1
    d["customer"].batch = torch.tensor([0])
    d["customer"].n_id = torch.tensor([100])
    d["customer"].tf = _tf(1, CUSTOMER_COLS)

    d["region"].num_nodes = 1
    d["region"].batch = torch.tensor([0])
    d["region"].n_id = torch.tensor([200])
    d["region"].tf = _tf(1, REGION_COLS)

    d["payment"].num_nodes = 3                                # rows: P0, P1, P3 (co-child)
    d["payment"].batch = torch.tensor([0, 0, 0])
    d["payment"].time = torch.tensor([10.0, 20.0, 15.0], dtype=torch.float64)
    d["payment"].n_id = torch.tensor([300, 301, 303])
    d["payment"].tf = _tf(3, PAYMENT_COLS)

    d["method"].num_nodes = 1
    d["method"].batch = torch.tensor([0])
    d["method"].n_id = torch.tensor([400])
    d["method"].tf = _tf(1, METHOD_COLS)

    d["refund"].num_nodes = 1
    d["refund"].batch = torch.tensor([0])
    d["refund"].time = torch.tensor([refund_time], dtype=torch.float64)
    d["refund"].n_id = torch.tensor([500])
    d["refund"].tf = _tf(1, REFUND_COLS)

    d["order", "f2p_customer", "customer"].edge_index = torch.tensor([[0], [0]])
    d["customer", "f2p_region", "region"].edge_index = torch.tensor([[0], [0]])
    d["payment", "f2p_method", "method"].edge_index = torch.tensor([[0], [0]])       # P0 -> M0
    d["order", "rev_f2p_order", "payment"].edge_index = torch.tensor([[0, 0], [0, 1]])
    # hop-2 families (traversal direction, as a 3-layer sampler stores them):
    d["customer", "rev_f2p_customer", "order"].edge_index = torch.tensor([[0, 0], [1, 2]])
    d["method", "rev_f2p_method", "payment"].edge_index = torch.tensor([[0, 0], [2, 0]])
    d["payment", "rev_f2p_payment", "refund"].edge_index = torch.tensor([[0], [0]])
    return d
