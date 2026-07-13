"""Hermetic fixtures for the SetJoin collate: a 3-level chain schema with a side parent.

Schema (FKs point right):   payment ─f2p_order→ order ─f2p_customer→ customer ─f2p_region→ region
                            payment ─f2p_method→ method

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
NODE_TYPES = ["customer", "method", "order", "payment", "region"]
EDGE_TYPES = [
    ("order", "f2p_customer", "customer"),
    ("customer", "rev_f2p_customer", "order"),
    ("customer", "f2p_region", "region"),
    ("region", "rev_f2p_region", "customer"),
    ("payment", "f2p_order", "order"),
    ("order", "rev_f2p_order", "payment"),
    ("payment", "f2p_method", "method"),
    ("method", "rev_f2p_method", "payment"),
]
ORDER_COLS = ["o_amt", "o_qty"]
CUSTOMER_COLS = ["c_name"]
REGION_COLS = ["r_code"]
PAYMENT_COLS = ["p_val"]
METHOD_COLS = ["m_kind"]
COLS = {"order": ORDER_COLS, "customer": CUSTOMER_COLS, "region": REGION_COLS,
        "payment": PAYMENT_COLS, "method": METHOD_COLS}


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
