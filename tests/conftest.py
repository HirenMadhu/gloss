"""Shared hermetic fixtures — tiny synthetic relational graphs so the unit tests need no network.

The synthetic graph is a **dual-FK** schema: an ``event`` table with two foreign keys into ``user``
(``buyer`` and ``seller``). It exercises the cell-token collate (placement, f2p neighbors, relational
masks) and the leakage rule. Minimal ``TensorFrame``s carry only ``col_names_dict`` (the collate reads
column identities, not values), so no pytorch-frame stats are needed. Encoder/FiLM tests that need real
dtype stats are guarded by the cached real rel-f1.

A second, additive fixture (``COLLIDE_*`` / :func:`collide_bundle`) adds a ``review`` table whose FK
column is *also* named ``buyer`` — the same-name-different-child-table case that the column-keyed role
vocabulary silently merged (changes.md P0.1). It is separate from the dual-FK bundle on purpose: the
dual-FK fixture is consumed by unrelated tests and must not change shape.
"""
from __future__ import annotations

from pathlib import Path

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
USER_COLS = ["u_attr"]
EVENT_COLS = ["e_x", "e_y"]


def _tf(n: int, cols: list[str]):
    import torch_frame

    return torch_frame.TensorFrame(
        feat_dict={torch_frame.numerical: torch.zeros(n, len(cols))},
        col_names_dict={torch_frame.numerical: list(cols)},
    )


def _schema_graph():
    """The 'full graph' the bundle points at (only its TensorFrame col names are read, for the vocab)."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["user"].tf = _tf(2, USER_COLS)
    d["event"].tf = _tf(4, EVENT_COLS)
    return d


def synthetic_bundle() -> GraphBundle:
    node_type_id, fk_role_id, metapath_id = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    return GraphBundle(
        dataset_name="synthetic-dualfk",
        data=_schema_graph(),
        col_stats_dict={},
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


def make_synth_batch(seed_time: float = 100.0, event_times=(10.0, 20.0, 30.0, 40.0)):
    """A 2-seed disjoint batch. Each segment: 1 untimed ``user`` seed + 2 timed ``event`` rows, one via
    ``buyer`` and one via ``seller``. All event times <= seed_time (leak-free)."""
    from torch_geometric.data import HeteroData

    et = torch.tensor(event_times, dtype=torch.float64)
    d = HeteroData()
    d["user"].num_nodes = 2
    d["user"].batch = torch.tensor([0, 1])
    d["user"].seed_time = torch.tensor([seed_time, seed_time], dtype=torch.float64)
    d["user"].n_id = torch.tensor([0, 1])
    d["user"].tf = _tf(2, USER_COLS)

    d["event"].num_nodes = 4
    d["event"].batch = torch.tensor([0, 0, 1, 1])
    d["event"].time = et
    d["event"].n_id = torch.tensor([10, 11, 12, 13])
    d["event"].tf = _tf(4, EVENT_COLS)

    # f2p (child=event -> parent=user). seg 0: events 0,1 -> user 0 ; seg 1: events 2,3 -> user 1
    d["event", "f2p_buyer", "user"].edge_index = torch.tensor([[0, 2], [0, 1]])
    d["event", "f2p_seller", "user"].edge_index = torch.tensor([[1, 3], [0, 1]])
    d["user", "rev_f2p_buyer", "event"].edge_index = torch.tensor([[0, 1], [0, 2]])
    d["user", "rev_f2p_seller", "event"].edge_index = torch.tensor([[0, 1], [1, 3]])
    return d


# ---- P0.1: same-named FK column in two different child tables (must NOT share a role id) ----
COLLIDE_NODE_TYPES = ["event", "review", "user"]
COLLIDE_EDGE_TYPES = EDGE_TYPES + [
    ("review", "f2p_buyer", "user"),       # same column NAME as event.buyer, different child table
    ("user", "rev_f2p_buyer", "review"),
]
REVIEW_COLS = ["r_score"]


def collide_bundle() -> GraphBundle:
    """Bundle whose schema has ``buyer`` as an FK column of two different child tables."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["user"].tf = _tf(2, USER_COLS)
    d["event"].tf = _tf(4, EVENT_COLS)
    d["review"].tf = _tf(2, REVIEW_COLS)
    node_type_id, fk_role_id, metapath_id = _build_vocabs(COLLIDE_NODE_TYPES, COLLIDE_EDGE_TYPES)
    return GraphBundle(
        dataset_name="synthetic-collide",
        data=d,
        col_stats_dict={},
        node_types=COLLIDE_NODE_TYPES,
        edge_types=COLLIDE_EDGE_TYPES,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


# ---- P0.2/P0.3: a 2-hop chain whose edges are sampled in the REVERSE direction only ----
# user (seed) <- event (child via buyer) <- comment (child via eventId). PyG records a traversed edge
# under the edge type it was traversed in, so expanding outward from the seed populates only the
# `rev_f2p_*` stores; the row graph must still be complete and connected.
CHAIN_NODE_TYPES = ["comment", "event", "user"]
CHAIN_EDGE_TYPES = [
    ("event", "f2p_buyer", "user"),
    ("user", "rev_f2p_buyer", "event"),
    ("comment", "f2p_eventId", "event"),
    ("event", "rev_f2p_eventId", "comment"),
]
COMMENT_COLS = ["c_txt"]


def chain_bundle() -> GraphBundle:
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["user"].tf = _tf(2, USER_COLS)
    d["event"].tf = _tf(2, EVENT_COLS)
    d["comment"].tf = _tf(2, COMMENT_COLS)
    node_type_id, fk_role_id, metapath_id = _build_vocabs(CHAIN_NODE_TYPES, CHAIN_EDGE_TYPES)
    return GraphBundle(
        dataset_name="synthetic-chain",
        data=d,
        col_stats_dict={},
        node_types=CHAIN_NODE_TYPES,
        edge_types=CHAIN_EDGE_TYPES,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


def make_chain_batch(seed_time: float = 100.0):
    """2 seeds; each segment is user -> event -> comment, all times <= seed_time."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    for nt, cols, n in (("user", USER_COLS, 2), ("event", EVENT_COLS, 2), ("comment", COMMENT_COLS, 2)):
        d[nt].num_nodes = n
        d[nt].batch = torch.tensor([0, 1])
        d[nt].n_id = torch.tensor([0, 1])
        d[nt].tf = _tf(n, cols)
    d["user"].seed_time = torch.tensor([seed_time, seed_time], dtype=torch.float64)
    d["event"].time = torch.tensor([10.0, 20.0], dtype=torch.float64)
    d["comment"].time = torch.tensor([5.0, 15.0], dtype=torch.float64)
    # reverse stores only (as the sampler would fill them when expanding away from the seed)
    d["user", "rev_f2p_buyer", "event"].edge_index = torch.tensor([[0, 1], [0, 1]])
    d["event", "rev_f2p_eventId", "comment"].edge_index = torch.tensor([[0, 1], [0, 1]])
    return d


@pytest.fixture
def synth_bundle() -> GraphBundle:
    return synthetic_bundle()


@pytest.fixture
def synth_batch():
    return make_synth_batch()


@pytest.fixture
def entity_table() -> str:
    return ENTITY


# ---- optional integration fixture on the cached real rel-f1 (offline; skip if absent) ----
def _rel_f1_cached() -> bool:
    try:
        from relbench.datasets import get_relbench_cache_dir
    except Exception:
        return False
    return (Path(get_relbench_cache_dir()) / "rel-f1").exists()


rel_f1_available = pytest.mark.skipif(not _rel_f1_cached(), reason="rel-f1 not cached locally")
