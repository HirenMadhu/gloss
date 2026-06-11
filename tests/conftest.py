"""Shared hermetic fixtures — tiny synthetic graphs so the unit/invariance tests need no network.

The headline fixture is a **dual-FK** graph: an ``event`` table with two foreign keys into ``user``
(``buyer`` and ``seller``), which rel-f1 lacks. It powers both the collate/shape tests and the Phase-3
``test_fk_role`` (distinct compiled geometry per FK role).
"""
from __future__ import annotations

import pytest
import torch

from gloss.data.graph import GraphBundle, _build_vocabs

ENTITY = "user"
NODE_TYPES = ["event", "user"]
EDGE_TYPES = [
    ("event", "f2p_buyer", "user"),
    ("user", "rev_f2p_buyer", "event"),
    ("event", "f2p_seller", "user"),
    ("user", "rev_f2p_seller", "event"),
]


def _bundle() -> GraphBundle:
    node_type_id, fk_role_id, metapath_id = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    return GraphBundle(
        dataset_name="synthetic-dualfk",
        data=None,
        col_stats_dict={},
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


def make_dualfk_batch(seed_time: float = 100.0, event_times=(10.0, 20.0, 30.0, 40.0)):
    """A 2-seed disjoint batch. Each segment: 1 timeless ``user`` seed + 2 timed ``event`` nodes,
    one linked via ``buyer`` and one via ``seller``. All event times <= seed_time (leak-free)."""
    from torch_geometric.data import HeteroData

    et = torch.tensor(event_times, dtype=torch.float64)
    d = HeteroData()
    d["user"].num_nodes = 2
    d["user"].batch = torch.tensor([0, 1])
    d["user"].seed_time = torch.tensor([seed_time, seed_time], dtype=torch.float64)
    d["user"].n_id = torch.tensor([0, 1])
    d["user"].tf = torch.zeros(2, 1)

    d["event"].num_nodes = 4
    d["event"].batch = torch.tensor([0, 0, 1, 1])
    d["event"].time = et
    d["event"].n_id = torch.tensor([10, 11, 12, 13])
    d["event"].tf = torch.zeros(4, 1)

    # segment 0: events 0(buyer),1(seller) -> user 0 ; segment 1: events 2(buyer),3(seller) -> user 1
    d["event", "f2p_buyer", "user"].edge_index = torch.tensor([[0, 2], [0, 1]])
    d["event", "f2p_seller", "user"].edge_index = torch.tensor([[1, 3], [0, 1]])
    d["user", "rev_f2p_buyer", "event"].edge_index = torch.tensor([[0, 1], [0, 2]])
    d["user", "rev_f2p_seller", "event"].edge_index = torch.tensor([[0, 1], [1, 3]])
    return d


@pytest.fixture
def dualfk_bundle() -> GraphBundle:
    return _bundle()


@pytest.fixture
def dualfk_batch():
    return make_dualfk_batch()


@pytest.fixture
def entity_table() -> str:
    return ENTITY


# ---- optional integration fixture on the cached real rel-f1 (offline; skip if absent) ----
def _rel_f1_cached() -> bool:
    from pathlib import Path

    try:
        from relbench.datasets import get_relbench_cache_dir
    except Exception:
        return False
    return (Path(get_relbench_cache_dir()) / "rel-f1").exists()


rel_f1_available = pytest.mark.skipif(not _rel_f1_cached(), reason="rel-f1 not cached locally")
