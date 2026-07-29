"""Phase 0 — graph vocabularies (fk-role distinctness is the load-bearing contract).

Roles are keyed on the triple ``(child_table, fk_column, parent_table)`` (changes.md P0.1), so a
relation *name* identifies a role only when its FK column occurs in exactly one child table.
"""
from __future__ import annotations

import pytest

from gloss.data.graph import (
    FK_NONE,
    MP_MULTIHOP,
    MP_PAD,
    MP_SELF,
    _build_vocabs,
    canonical_edge_role,
    role_name,
)
from tests.conftest import (
    COLLIDE_EDGE_TYPES,
    COLLIDE_NODE_TYPES,
    EDGE_TYPES,
    NODE_TYPES,
    collide_bundle,
    rel_f1_available,
)


def test_node_type_vocab_sorted_and_contiguous():
    nt, _, _ = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    assert nt == {"event": 0, "user": 1}


def test_canonical_edge_role_is_direction_free():
    assert canonical_edge_role(("event", "f2p_buyer", "user")) == ("event", "buyer", "user")
    assert canonical_edge_role(("user", "rev_f2p_buyer", "event")) == ("event", "buyer", "user")
    assert role_name(("event", "buyer", "user")) == "table event column buyer references table user"


def test_dual_fk_relations_get_distinct_ids(synth_bundle):
    b = synth_bundle
    # the two FKs into `user` must be distinct roles — the whole point
    assert b.relation_fk_role("f2p_buyer") != b.relation_fk_role("f2p_seller")
    assert b.relation_metapath("f2p_buyer") != b.relation_metapath("f2p_seller")
    # forward and reverse of the SAME FK share a role (direction-independent)
    assert b.relation_fk_role("f2p_buyer") == b.relation_fk_role("rev_f2p_buyer")
    assert b.relation_metapath("f2p_buyer") == b.relation_metapath("rev_f2p_buyer")
    # ... and via the triple API, which is the one that is always unambiguous
    assert b.edge_fk_role(("event", "f2p_buyer", "user")) == b.edge_fk_role(("user", "rev_f2p_buyer", "event"))
    assert b.edge_fk_role(("event", "f2p_buyer", "user")) != b.edge_fk_role(("event", "f2p_seller", "user"))
    # reserved ids are not reused
    _, fk, mp = _build_vocabs(NODE_TYPES, EDGE_TYPES)
    assert FK_NONE == 0 and all(v >= 1 for v in fk.values())
    assert {MP_PAD, MP_SELF, MP_MULTIHOP} == {0, 1, 2}
    assert all(v >= 3 for v in mp.values())
    # 2 role triples (event.buyer->user, event.seller->user), not 4 relations
    assert len(fk) == 2 and len(mp) == 2


def test_same_named_fk_column_in_two_child_tables_is_two_roles():
    """P0.1's uncovered case: `buyer` is an FK column of BOTH `event` and `review`.

    The column-keyed vocabulary merged these onto one id (raceId x5 on rel-f1, nct_id x10 on
    rel-trial). Under the triple they are two roles, and the ambiguous *name* raises instead of
    silently returning one of them.
    """
    b = collide_bundle()
    ev = b.edge_fk_role(("event", "f2p_buyer", "user"))
    rv = b.edge_fk_role(("review", "f2p_buyer", "user"))
    assert ev != rv and ev >= 1 and rv >= 1
    # every FK edge gets its own id: 3 triples from 6 edge types (2 columns, 3 child/parent pairs)
    assert b.num_roles == 3 and b.num_fk_roles == 4
    assert len(set(b.fk_role_id.values())) == 3
    assert sorted(b.fk_role_id.values()) == [1, 2, 3]
    # reverse edge types map onto the same three ids, not new ones
    assert b.edge_fk_role(("user", "rev_f2p_buyer", "review")) == rv
    assert b.edge_fk_role(("user", "rev_f2p_buyer", "event")) == ev
    # the name alone no longer identifies a role -> loud failure, not a silent merge
    with pytest.raises(ValueError, match="ambiguous"):
        b.relation_fk_role("f2p_buyer")
    with pytest.raises(ValueError, match="ambiguous"):
        b.relation_metapath("rev_f2p_buyer")
    # an unambiguous name still resolves
    assert b.relation_fk_role("f2p_seller") == b.edge_fk_role(("event", "f2p_seller", "user"))
    # role_triples is ordered by id, so it can index a [K, d_text] name table (P0.4)
    assert [b.fk_role_id[t] for t in b.role_triples] == [1, 2, 3]
    _, fk, mp = _build_vocabs(COLLIDE_NODE_TYPES, COLLIDE_EDGE_TYPES)
    assert len(fk) == 3 and len(mp) == 3


def test_bundle_counts(synth_bundle):
    b = synth_bundle
    assert b.num_node_types == 2
    assert b.num_fk_roles == 1 + 2        # + FK_NONE; 2 role triples (buyer, seller)
    assert b.num_roles == 2               # K
    assert b.num_metapaths == 3 + 2       # + reserved
    assert b.relation_fk_role("nonexistent") == FK_NONE
    assert b.relation_metapath("nonexistent") == MP_MULTIHOP
    assert b.edge_fk_role(("event", "f2p_nonexistent", "user")) == FK_NONE


@rel_f1_available
def test_real_rel_f1_fk_roles_from_edge_relations():
    from gloss.data.graph import build_gloss_graph

    bundle = build_gloss_graph("rel-f1")
    # one role per (child, fk_column, parent) triple — rel-f1 has 13 FK edges, so 13 roles
    assert bundle.num_roles == 13, bundle.role_triples
    cols = {t[1] for t in bundle.fk_role_id}
    assert {"driverId", "constructorId", "raceId"} <= cols
    # raceId is the collision case: 5 child tables reference `races` through it, and they are 5 roles
    race = [t for t in bundle.fk_role_id if t[1] == "raceId"]
    assert len(race) == 5 and len({bundle.fk_role_id[t] for t in race}) == 5
    # distinct roles => distinct ids; forward/reverse of the SAME edge share a role
    et = ("results", "f2p_driverId", "drivers")
    assert bundle.edge_fk_role(et) == bundle.edge_fk_role(("drivers", "rev_f2p_driverId", "results"))
    assert bundle.edge_fk_role(et) != bundle.edge_fk_role(("results", "f2p_constructorId", "constructors"))
    # the name-only API refuses to guess where the schema is ambiguous
    with pytest.raises(ValueError, match="ambiguous"):
        bundle.relation_fk_role("f2p_raceId")
