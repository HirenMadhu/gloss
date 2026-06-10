"""test_shapes.py — TokenBatch / registry shapes end-to-end (impl §9). Model shapes added in Phase 3."""
from __future__ import annotations

import torch

from gloss.data.collate import collate_subgraphs
from gloss.data.relbench_graph import MASK_NEIGHBOR, MASK_SEED
from gloss.data.synthetic import make_synthetic_bundle, make_synthetic_dualfk
from gloss.data.relbench_graph import HeteroTemporalGraph, SchemaRegistry


def _batch(bundle, B=8, fanout=(6, 6), max_cells=512):
    sampler = bundle.make_sampler(num_neighbors=list(fanout), max_cells=max_cells, seed=0)
    seeds = bundle.task.get_table("train").df.head(B)
    subs = [sampler.sample("entity", r.id, r.date, r.target) for r in seeds.itertuples()]
    return subs, collate_subgraphs(subs, bundle.db, bundle.registry)


def test_tokenbatch_shapes_consistent():
    bundle, _ = make_synthetic_bundle(seed=3)
    subs, tb = _batch(bundle, B=8)
    B, T = tb.value_num.shape
    assert B == 8 and T >= 1
    for name, t in tb.__dict__.items():
        assert t.shape == (B, T), f"{name} has shape {tuple(t.shape)} != {(B, T)}"
    assert tb.pad_mask.dtype == torch.bool
    assert tb.pad_mask.any(), "no real cells"


def test_ids_within_registry_range():
    bundle, _ = make_synthetic_bundle(seed=4)
    _, tb = _batch(bundle)
    reg = bundle.registry
    real = tb.pad_mask
    assert int(tb.col_global_id[real].max()) < reg.num_cols
    assert int(tb.node_type_id[real].max()) < reg.num_tables
    assert int(tb.fk_role_id[real].max()) < reg.num_fk_roles


def test_fk_role_assigned_and_distinct():
    """event.entity_id is an FK cell (role>0); plain value columns are role 0."""
    bundle, _ = make_synthetic_bundle(seed=5)
    reg = bundle.registry
    assert reg.fk_role_id[("event", "entity_id")] >= 1
    assert ("event", "col_X") not in reg.fk_role_id  # value column -> not an FK


def test_dualfk_distinct_roles():
    """Two FKs into the same table get DISTINCT fk_role_ids (RT dual-FK fix; stress-test #6)."""
    db, planted = make_synthetic_dualfk(seed=0)
    _, reg = HeteroTemporalGraph.build(db)
    buyer = reg.fk_role_id[("transaction", "buyer_id")]
    seller = reg.fk_role_id[("transaction", "seller_id")]
    assert buyer != seller and buyer >= 1 and seller >= 1
    # both point at the same parent table, but are distinguishable by role id
    assert db.table_dict["transaction"].fkey_col_to_pkey_table["buyer_id"] == "users"
    assert db.table_dict["transaction"].fkey_col_to_pkey_table["seller_id"] == "users"


def test_seed_row_present_and_neighbors_sampled():
    bundle, _ = make_synthetic_bundle(seed=6)
    subs, _ = _batch(bundle)
    kinds = {r.mask_kind for sg in subs for r in sg.rows}
    assert MASK_SEED in kinds
    # at least one subgraph should pull event history (P->F neighbors)
    assert any(r.mask_kind == MASK_NEIGHBOR for sg in subs for r in sg.rows)
