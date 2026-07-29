"""Row-level adjacency for the two-level substrate (changes.md P0.2 / P0.3).

The cell-token batch is flat: rows exist only implicitly, as the set of cells sharing a
``node_idxs`` value. The two-level encoder needs rows as first-class tokens, which means a per-seed
row graph: who is present, when, how far from the seed, and by which FK role each pair is joined.

This module builds exactly that, from data the collate already has:

* a **row slot** is a per-seed local node index ``r in [0, R)`` — the same index ``node_idxs``
  carries, so ``cell_row = node_idxs`` needs no renumbering;
* ``adj_role[b, r, s]`` encodes the joined pair *and its direction* in one int64 (see below);
* ``row_hop`` is BFS distance from the seed root over ``adj_role != 0``, undirected — the sampler
  gives no hop annotation (relbench's ``CustomNodeLoader`` drops PyG's ``num_sampled_nodes``), so BFS
  is the only route. This is the verified prototype from ``scripts/probe_sampler_causality.py``.

``adj_role`` encoding, with ``K = bundle.num_roles``::

    0           no edge
    1 .. K      s is a CHILD of r via role s->r     (r holds the referenced pkey)
    K+1 .. 2K   s is a PARENT of r via role r->s    (r holds the FK)
    2K+1        r == s (self loop; present for every valid row)

The admissibility mask for row attention is ``(adj_role != 0)`` AND-ed with ``row_valid`` on both
axes. Self-loops are always present, so no row is ever fully masked.

**Edges come from every sampled edge type, both directions.** PyG's sampler records a traversed edge
under the edge type it was traversed in, and the forward ``f2p_*`` store is *not* guaranteed to hold
the transpose of the sampled ``rev_f2p_*`` edges. Reading forward relations only (which is what
``f2p_nbr_idxs`` does, and must keep doing — it is a frozen model input) would therefore leave rows
disconnected from the root. Both directions are canonicalized to ``(child, parent, role)`` here.
"""
from __future__ import annotations

import torch
from torch import Tensor

ADJ_NONE = 0


def self_loop_id(num_roles: int) -> int:
    """``2K+1`` — the ``adj_role`` value on the diagonal."""
    return 2 * num_roles + 1


def child_of(role: Tensor | int, num_roles: int):
    """``adj_role`` value for "s is a CHILD of r via role s->r"."""
    return role


def parent_of(role: Tensor | int, num_roles: int):
    """``adj_role`` value for "s is a PARENT of r via role r->s"."""
    return role + num_roles


def undirected_role(adj: Tensor, num_roles: int) -> Tensor:
    """Strip direction from ``adj_role`` values: ``[1,2K] -> [1,K]``; 0 and the self loop -> 0."""
    out = torch.where(adj > num_roles, adj - num_roles, adj)
    return torch.where((adj <= 0) | (adj > 2 * num_roles), torch.zeros_like(out), out)


def build_row_graph(
    *,
    seg: Tensor,            # [N] int64  per sampled node: seed segment b
    local_node: Tensor,     # [N] int64  per sampled node: row slot r within its segment
    table_id: Tensor,       # [N] int64  per sampled node: node-type id
    row_t: Tensor,          # [N] float64 per sampled node: timestamp (0 if untimed)
    timed: Tensor,          # [N] bool
    is_root: Tensor,        # [N] bool   the seed row of its segment (exactly one per segment)
    child_g: Tensor,        # [E] int64  global node index of the FK-holding (child) endpoint
    parent_g: Tensor,       # [E] int64  global node index of the referenced (parent) endpoint
    edge_role: Tensor,      # [E] int64  role id in [1, K]
    num_seeds: int,
    max_rows: int,
    num_roles: int,
    num_hops: int | None = None,
) -> dict[str, Tensor]:
    """Build the per-seed row graph. Returns the P0.2/P0.3 ``CellBatch`` fields as a dict.

    ``max_rows`` (R) is a hard cap: exceeding it **asserts**, it never clamps — a silently dropped row
    would break the ``cell_row`` alias and hide a sampler/fanout change.
    """
    B, R, K = num_seeds, max_rows, num_roles
    dev = seg.device

    num_rows = torch.bincount(seg, minlength=B)[:B]
    over = int(num_rows.max()) if num_rows.numel() else 0
    assert over <= R, (
        f"max_rows_per_seed exceeded: {over} rows on some seed but R={R}. Raise data.collate.max_rows "
        f"(R is fanout-coupled: it grows with data.sampler.num_neighbors). Do not clamp."
    )

    flat = seg * R + local_node                                   # [N] slot index into [B*R]
    row_valid = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_table = torch.full((B * R,), -1, dtype=torch.long, device=dev)
    row_time = torch.zeros(B * R, dtype=torch.float64, device=dev)
    row_is_timed = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_is_root = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_valid[flat] = True
    row_table[flat] = table_id
    row_time[flat] = row_t
    row_is_timed[flat] = timed
    row_is_root[flat] = is_root
    row_valid = row_valid.view(B, R)
    row_table = row_table.view(B, R)
    row_time = row_time.view(B, R)
    row_is_timed = row_is_timed.view(B, R)
    row_is_root = row_is_root.view(B, R)

    n_root = row_is_root.sum(dim=1)
    assert bool((n_root == 1).all()), f"expected exactly one root row per seed, got {n_root.tolist()}"

    # ---- adjacency ----
    adj = torch.zeros(B * R * R, dtype=torch.long, device=dev)
    if child_g.numel():
        b = seg[child_g]
        keep = (seg[parent_g] == b) & (child_g != parent_g)       # same seed subgraph, no self-FK
        b, rc, rp, role = b[keep], local_node[child_g][keep], local_node[parent_g][keep], edge_role[keep]
        if rc.numel():
            # One int64 per *unordered* pair, so the two directions can never disagree. Duplicate
            # pairs (dual FK onto the same parent row, or a 2-cycle) resolve to the smallest role id;
            # a stable sort makes the choice deterministic.
            lo = torch.minimum(rc, rp)
            hi = torch.maximum(rc, rp)
            pair = b * (R * R) + lo * R + hi
            _, order = torch.sort(pair * (K + 2) + role, stable=True)
            pair_s = pair[order]
            first = torch.ones_like(pair_s, dtype=torch.bool)
            first[1:] = pair_s[1:] != pair_s[:-1]
            sel = order[first]
            b, rc, rp, role = b[sel], rc[sel], rp[sel], role[sel]
            adj[b * (R * R) + rp * R + rc] = child_of(role, K)    # from the parent: s is my child
            adj[b * (R * R) + rc * R + rp] = parent_of(role, K)   # from the child: s is my parent
    adj = adj.view(B, R, R)
    diag = torch.arange(R, device=dev)
    adj = adj * (row_valid.unsqueeze(2) & row_valid.unsqueeze(1)).long()  # never point at a pad slot
    adj[:, diag, diag] = row_valid.long() * self_loop_id(K)

    # ---- BFS from the root (undirected over adj != 0), P0.3 ----
    adjb = (adj != 0)
    adjb[:, diag, diag] = False                                    # self loops carry no hop
    row_hop = torch.full((B, R), -1, dtype=torch.long, device=dev)
    row_in_role = torch.zeros((B, R), dtype=torch.long, device=dev)
    row_hop[row_is_root] = 0
    frontier = row_is_root.clone()
    h = 0
    while bool(frontier.any()):
        h += 1
        assert h <= R, "BFS did not terminate"
        cand = frontier.unsqueeze(2) & adjb                        # [B,R,R]: r (frontier) -> s
        new = cand.any(dim=1) & (row_hop < 0) & row_valid
        if not bool(new.any()):
            break
        pred = cand.to(torch.uint8).argmax(dim=1)                  # first frontier predecessor of s
        reached_by = undirected_role(adj.gather(1, pred.unsqueeze(1)).squeeze(1), K)
        row_hop[new] = h
        row_in_role[new] = reached_by[new]
        frontier = new

    unreached = row_valid & (row_hop < 0)
    assert not bool(unreached.any()), (
        f"{int(unreached.sum())} sampled row(s) are not reachable from their seed root — the row graph "
        f"is disconnected, which means an edge type was dropped when building adj_role."
    )
    if num_hops is not None:
        assert int(row_hop.max()) <= num_hops, (
            f"row_hop max {int(row_hop.max())} exceeds sampler depth {num_hops}"
        )

    return {
        "num_rows": num_rows,
        "row_valid": row_valid,
        "row_table": row_table,
        "row_time": row_time,
        "row_is_timed": row_is_timed,
        "row_hop": row_hop,
        "row_in_role": row_in_role,
        "row_is_root": row_is_root,
        "adj_role": adj,
    }
