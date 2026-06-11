"""Phase 0 — graph vocabularies (fk-role distinctness is the load-bearing contract)."""
from __future__ import annotations

from gloss.data.graph import FK_NONE, MP_MULTIHOP, MP_PAD, MP_SELF, _build_vocabs
from tests.conftest import EDGE_TYPES, NODE_TYPES, rel_f1_available


def test_node_type_vocab_sorted_and_contiguous():
    nt, _, _ = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    assert nt == {"event": 0, "user": 1}


def test_dual_fk_relations_get_distinct_ids(dualfk_bundle):
    b = dualfk_bundle
    # the two FKs into `user` must be distinct roles — the whole point
    assert b.relation_fk_role("f2p_buyer") != b.relation_fk_role("f2p_seller")
    assert b.relation_metapath("f2p_buyer") != b.relation_metapath("f2p_seller")
    # forward and reverse of the SAME FK share a role (direction-independent)
    assert b.relation_fk_role("f2p_buyer") == b.relation_fk_role("rev_f2p_buyer")
    assert b.relation_metapath("f2p_buyer") == b.relation_metapath("rev_f2p_buyer")
    # reserved ids are not reused
    _, fk, mp = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    assert FK_NONE == 0 and all(v >= 1 for v in fk.values())
    assert {MP_PAD, MP_SELF, MP_MULTIHOP} == {0, 1, 2}
    assert all(v >= 3 for v in mp.values())
    # canonical: 2 distinct FK columns (buyer, seller), not 4 relations
    assert len(fk) == 2 and len(mp) == 2


def test_bundle_counts(dualfk_bundle):
    b = dualfk_bundle
    assert b.num_node_types == 2
    assert b.num_fk_roles == 1 + 2        # + FK_NONE; 2 canonical FK columns (buyer, seller)
    assert b.num_metapaths == 3 + 2       # + reserved
    assert b.relation_fk_role("nonexistent") == FK_NONE
    assert b.relation_metapath("nonexistent") == MP_MULTIHOP


@rel_f1_available
def test_real_rel_f1_fk_roles_from_edge_relations():
    from gloss.data.graph import build_gloss_graph

    bundle = build_gloss_graph("rel-f1")
    # FK role derives from the f2p_<col> edge relation, canonicalized to the column name.
    cols = set(bundle.fk_role_id)
    assert {"driverId", "constructorId", "raceId"} <= cols
    # distinct fkey columns => distinct ids; forward/reverse share a role
    assert bundle.relation_fk_role("f2p_driverId") != bundle.relation_fk_role("f2p_constructorId")
    assert bundle.relation_fk_role("f2p_driverId") == bundle.relation_fk_role("rev_f2p_driverId")
