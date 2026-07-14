"""Schema-level join maps for SetJoin.

Everything here is a pure function of the DB *schema* (``bundle.edge_types``), computed once and cached
on the bundle. A relation is identified by the triple ``(child_type, fk_col, parent_type)`` — NOT by the
bare canonical column: ``fk_role_id`` is keyed by column name and collides across tables (results.driverId
and qualifying.driverId share a role id), so the triple is the unambiguous schema key while the role id
stays the (table-tag-disambiguated) embedding index.

Direction convention (relbench): a forward edge type is ``(child, f2p_<col>, parent)`` — the src table
holds the FK. Reverse types are ``rev_f2p_<col>``. Schema enumeration uses forward triples only; batch
edge *instances* must be read from ALL edge types (PyG stores sampled edges under the type traversed —
children arrive via ``rev_*``), which is `collate.to_join_batch`'s job.
"""
from __future__ import annotations

from ..data.graph import GraphBundle, canonical_relation, is_forward_relation

# A schema relation key: (child_type, canonical fk column, parent_type).
RelKey = tuple[str, str, str]


def fk_relations(bundle: GraphBundle) -> list[RelKey]:
    """All FK relations as sorted, deduped ``(child_type, fk_col, parent_type)`` triples."""
    cached = getattr(bundle, "_setjoin_fk_relations", None)
    if cached is not None:
        return cached
    rels = sorted({(src, canonical_relation(rel), dst) for (src, rel, dst) in bundle.edge_types
                   if is_forward_relation(rel)})
    try:
        bundle._setjoin_fk_relations = rels  # type: ignore[attr-defined]
    except Exception:
        pass
    return rels


def parent_rels(bundle: GraphBundle, nt: str) -> list[RelKey]:
    """The many-to-one relations OUT of ``nt`` (``nt`` holds the FK) — safe to flatten (≤1 row each)."""
    return [r for r in fk_relations(bundle) if r[0] == nt]


def child_rels(bundle: GraphBundle, nt: str) -> list[RelKey]:
    """The one-to-many relations INTO ``nt`` — each contributes a *set* of child rows. The sorted order
    here defines the ``child_counts`` column order of :class:`~gloss.setjoin.collate.JoinBatch`."""
    return [r for r in fk_relations(bundle) if r[2] == nt]


def m2o_paths(bundle: GraphBundle, entity_table: str, depth: int = 2) -> list[tuple[RelKey, ...]]:
    """Deterministic BFS enumeration of the seed's many-to-one join paths, up to ``depth`` FK hops.

    ``[()]`` (the seed's own row) first, then 1-hop parent paths, then 2-hop, in sorted relation order —
    the list index is the wide row's ``path_id`` (so path 0 == seed, matching the seed-cells-first
    truncation convention). The depth cap terminates self-referencing FKs.
    """
    cache = getattr(bundle, "_setjoin_m2o_paths", None)
    if cache is None:
        cache = {}
        try:
            bundle._setjoin_m2o_paths = cache  # type: ignore[attr-defined]
        except Exception:
            pass
    key = (entity_table, depth)
    if key in cache:
        return cache[key]
    paths: list[tuple[RelKey, ...]] = [()]
    frontier: list[tuple[tuple[RelKey, ...], str]] = [((), entity_table)]
    for _ in range(depth):
        nxt: list[tuple[tuple[RelKey, ...], str]] = []
        for path, nt in frontier:
            for r in parent_rels(bundle, nt):
                p = path + (r,)
                paths.append(p)
                nxt.append((p, r[2]))
        frontier = nxt
    cache[key] = paths
    return paths


def path_target_type(entity_table: str, path: tuple[RelKey, ...]) -> str:
    """The table a wide-row path lands on (used to tag missing-parent marker slots)."""
    return path[-1][2] if path else entity_table


def setjoin_neighbors(bundle: GraphBundle, fanout: int = 64, depth: int = 2) -> dict:
    """Per-edge-type ``num_neighbors`` that samples exactly SetJoin's neighborhood and nothing else.

    PyG DIRECTION SEMANTICS (the v1 gate inverted this — every seed got ~1 child per relation and
    elements lost their flattened parents): ``num_neighbors[et]`` budgets how many **src**-side
    neighbors are drawn for each frontier node of type ``et.dst`` (edges are sampled *into* the
    frontier, message-passing direction). Hence:

    * forward ``(child, f2p_<col>, parent)`` types expand a PARENT-side frontier node by sampling its
      CHILD rows → ``[fanout, 0, ...]``: up to ``fanout`` most-recent children per relation at hop 1,
      and NO deeper o2m expansion (hop-2 budget 0 = no children-of-children, no siblings of parents).
    * reverse ``(parent, rev_f2p_<col>, child)`` types expand a CHILD-side frontier node by sampling
      its parent (unique per FK) → ``[1] * depth``: the seed's parents at hop 1, grandparents and each
      child row's own flattened parents at hop 2 (the m2o closure).
    """
    return {et: ([fanout] + [0] * (depth - 1) if is_forward_relation(et[1]) else [1] * depth)
            for et in bundle.edge_types}
