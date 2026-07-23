"""SetJoin collate: a per-seed-disjoint sampled ``HeteroData`` minibatch -> a :class:`JoinBatch`.

Per seed the batch carries two structures (see setjoin_implementation.md for the normative contract):

* the **wide seed row** — CELL tokens ``[B, W]``: the seed's cells plus the many-to-one closure
  (parents, grandparents) flattened in, each cell tagged with its join-path id. A path whose walk dies
  (NULL FK / sampler miss / temporal exclusion of a timed parent) emits exactly ONE marker slot
  (``wide_missing``) so the model can tell "FK is absent" from "columns truncated".
* the **union set** — ROW elements ``[B, N]``: all 1-hop child rows across all child relations in one
  table-tagged set, most-recent-first, truncated to ``set_size``. Each element's value content is an
  ADDITIVE row scatter (``elem_rows``): the child row itself at path ``FK_NONE`` plus each of the
  child's own flattened parents at that FK's role id — excluding the seed itself (by global ``n_id``:
  ``disjoint=True`` samples a tree, so the seed can reappear as a duplicate node copy).

PyG facts this code depends on (stress-tested, do not "simplify" away):
* sampled edges are stored under the edge type **traversed** — children arrive via ``rev_f2p_*`` types,
  so adjacency is built from ALL edge types, canonicalized by direction and deduped;
* the entity-type store spans seeds + sampled entity-type neighbors; seeds are rows
  ``[:seed_time.numel()]`` and ``input_id`` is per-seed ``[B]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.collate import _seed_time_per_segment, column_vocab, feature_col_names
from ..data.graph import FK_NONE, GraphBundle, canonical_relation, is_forward_relation
from .paths import child_rels, m2o_paths, parent_rels, path_target_type, row_paths, slot_relations

PAD = -1


@dataclass
class JoinBatch:
    num_seeds: int                 # B
    wide_len: int                  # W (fixed cap)
    set_size: int                  # N (fixed cap)
    # ---- wide seed row: CELL tokens [B, W] ----
    wide_col_idxs: Tensor          # long; global (table, column) id via column_vocab (pad/marker = -1)
    wide_table_idxs: Tensor        # long; node-type id (markers: the MISSING table's id; pad = -1)
    wide_path_idxs: Tensor         # long; m2o join-path id, 0 = seed's own row (pad = -1)
    wide_is_pad: Tensor            # bool; True = pad slot
    wide_missing: Tensor           # bool; True = absent-parent marker slot
    wide_row_time: Tensor          # float64; 0 where untimed/pad (gate with wide_is_timed)
    wide_is_timed: Tensor          # bool
    # ---- union set: ROW elements [B, N] ----
    elem_mask: Tensor              # bool; True = real element
    elem_rel_idxs: Tensor          # long; fk_role_id of the child->seed relation (pad 0)
    elem_table_idxs: Tensor        # long; child node-type id (pad 0)
    elem_row_time: Tensor          # float64
    elem_is_timed: Tensor          # bool
    elem_hop: Tensor               # long; 1 = direct child, 2 = grandchild/sibling/co-child (hop2 arm)
    set_group_idx: Tensor          # long [B, N]; slot-attention GROUP-BY slot (paths.slot_relations); pad = -1
    # ---- per child relation [B, R] (order = paths.child_rels; post-sampler, PRE-truncation) ----
    child_counts: Tensor           # float32
    # ---- per seed [B] ----
    seed_time: Tensor              # float64
    target: Tensor                 # float32
    has_target: Tensor             # bool
    input_id: Tensor               # long; split-table row index (-1 when absent, e.g. fixtures)
    # ---- value encoding ----
    tf_dict: dict                  # node_type -> torch_frame.TensorFrame (as sampled)
    wide_placement: dict           # node_type -> (b, w, row_in_tf, col_pos) — CELL scatter into [B, W, d]
    elem_rows: dict                # node_type -> (b, n, row_in_tf, path_id) — additive ROW scatter into [B, N, d]
    # ---- per-TensorFrame-row metadata [n_t] (axial row/column attention grids; disjoint sampling
    # means every row copy belongs to exactly one seed) ----
    row_seg: dict                  # node_type -> long [n_t]; the row's seed/segment index
    row_times: dict                # node_type -> float64 [n_t]; 0 untimed (gate with row_timed)
    row_timed: dict                # node_type -> bool [n_t]
    wide_truncated: int = 0        # seeds whose wide row hit the cap with cells left over
    set_truncated: int = 0         # seeds whose union set hit the cap with children left over

    def to(self, device) -> "JoinBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return JoinBatch(
            num_seeds=self.num_seeds, wide_len=self.wide_len, set_size=self.set_size,
            wide_col_idxs=mv(self.wide_col_idxs), wide_table_idxs=mv(self.wide_table_idxs),
            wide_path_idxs=mv(self.wide_path_idxs), wide_is_pad=mv(self.wide_is_pad),
            wide_missing=mv(self.wide_missing), wide_row_time=mv(self.wide_row_time),
            wide_is_timed=mv(self.wide_is_timed),
            elem_mask=mv(self.elem_mask), elem_rel_idxs=mv(self.elem_rel_idxs),
            elem_table_idxs=mv(self.elem_table_idxs), elem_row_time=mv(self.elem_row_time),
            elem_is_timed=mv(self.elem_is_timed), elem_hop=mv(self.elem_hop),
            set_group_idx=mv(self.set_group_idx),
            child_counts=mv(self.child_counts),
            seed_time=mv(self.seed_time), target=mv(self.target), has_target=mv(self.has_target),
            input_id=mv(self.input_id),
            tf_dict={k: v.to(device) for k, v in self.tf_dict.items()},
            wide_placement={k: tuple(t.to(device) for t in v) for k, v in self.wide_placement.items()},
            elem_rows={k: tuple(t.to(device) for t in v) for k, v in self.elem_rows.items()},
            row_seg={k: v.to(device) for k, v in self.row_seg.items()},
            row_times={k: v.to(device) for k, v in self.row_times.items()},
            row_timed={k: v.to(device) for k, v in self.row_timed.items()},
            wide_truncated=self.wide_truncated, set_truncated=self.set_truncated,
        )

    def pretty_shapes(self) -> str:
        n_cells = int((~self.wide_is_pad & ~self.wide_missing).sum())
        n_markers = int(self.wide_missing.sum())
        per_seed = self.elem_mask.sum(dim=1)
        lines = [
            f"JoinBatch: B={self.num_seeds} W={self.wide_len} N={self.set_size}",
            f"  wide: real cells={n_cells}  missing markers={n_markers}  "
            f"truncated seeds={self.wide_truncated}",
            f"  set:  elements={int(per_seed.sum())} "
            f"(hop-2: {int((self.elem_mask & (self.elem_hop == 2)).sum())})  "
            f"empty-set seeds={int((per_seed == 0).sum())}  "
            f"truncated seeds={self.set_truncated}",
            f"  child_counts {tuple(self.child_counts.shape)}  targets={int(self.has_target.sum())}",
            f"  tf_dict types: {sorted(self.tf_dict)}  elem_rows types: {sorted(self.elem_rows)}",
        ]
        return "\n".join(lines)


def to_join_batch(
    batch,
    bundle: GraphBundle,
    entity_table: str,
    *,
    wide_len: int = 128,
    set_size: int = 128,
    hop2: bool = False,
    per_relation_cap: int | None = None,
) -> JoinBatch:
    """Convert a disjoint sampled ``HeteroData`` minibatch into a dense :class:`JoinBatch`.

    ``hop2=True`` (needs a ``fanout2>0`` sampler) additionally admits two hop-2 element families —
    **grandchildren** (children of the seed's children) and **siblings/co-children** (children of the
    seed's parents and of the children's non-seed parents), tagged ``elem_hop=2`` with the role id of
    the last traversed FK. Hop-1 elements keep priority under the ``set_size`` cap, and hop-2
    candidates are deduped by global n_id against the seed and all hop-1 elements. Default off —
    the emitted batch is then bit-identical to the 1-hop MVP.

    ``per_relation_cap=N`` (the anti-dilution arm) keeps only the ``N`` most-recent candidates per
    ``(hop, table, FK role)`` group before the merged sort, so one high-volume relation can't crowd
    the rest out of the set. ``child_counts`` stays PRE-cap (the model keeps the true-count signal).
    Default off — no behavior change.
    """
    vocab = column_vocab(bundle)
    node_types = [nt for nt in bundle.node_types if nt in batch.node_types and batch[nt].num_nodes > 0]

    # ---- flatten nodes across types (stable order); mirror of to_cell_batch lines 142-176 ----
    type_slice: dict[str, slice] = {}
    seg_list, rowt_list, timed_list, nid_list, intype_list, typeidx_list = [], [], [], [], [], []
    offset = 0
    for ti, nt in enumerate(node_types):
        st = batch[nt]
        n = st.num_nodes
        type_slice[nt] = slice(offset, offset + n)
        offset += n
        seg_list.append(st.batch.to(torch.long))
        if "time" in st:
            rowt_list.append(st.time.to(torch.float64))
            timed_list.append(torch.ones(n, dtype=torch.bool))
        else:
            rowt_list.append(torch.zeros(n, dtype=torch.float64))
            timed_list.append(torch.zeros(n, dtype=torch.bool))
        nid_list.append(st.n_id.to(torch.long) if "n_id" in st else torch.arange(n))
        intype_list.append(torch.arange(n, dtype=torch.long))
        typeidx_list.append(torch.full((n,), ti, dtype=torch.long))

    seg = torch.cat(seg_list)
    rowt = torch.cat(rowt_list)
    timed = torch.cat(timed_list)
    nid = torch.cat(nid_list)
    intype = torch.cat(intype_list)
    typeidx = torch.cat(typeidx_list)

    B = int(seg.max().item()) + 1
    ent = batch[entity_table]
    seed_time = _seed_time_per_segment(ent, B)

    # seeds = first seed_time.numel() rows of the entity store (the store may also hold entity-type
    # neighbors); map each segment to its seed's flat index and global n_id.
    n_seeds = ent.seed_time.numel()
    ent_start = type_slice[entity_table].start
    seed_flat = torch.full((B,), -1, dtype=torch.long)
    seed_flat[ent.batch[:n_seeds].to(torch.long)] = torch.arange(n_seeds) + ent_start
    assert int((seed_flat < 0).sum()) == 0, "could not locate a seed node for every segment"
    seed_nid = nid[seed_flat]                                             # [B]

    # labels (AttachTargetTransform puts `y` on the entity store)
    target = torch.zeros(B, dtype=torch.float32)
    has_target = torch.zeros(B, dtype=torch.bool)
    if "y" in ent:
        y = ent.y.to(torch.float32)
        yb = ent.batch[: y.numel()] if y.numel() != ent.batch.numel() else ent.batch
        target[yb] = y
        has_target[yb] = True
    if "input_id" in ent:
        input_id = ent.input_id.to(torch.long)[:B]
    else:
        input_id = torch.full((B,), -1, dtype=torch.long)

    # ---- adjacency over ALL edge types, canonicalized by direction, same-segment guard, deduped ----
    # (children arrive via rev_* types — a forward-only walk would find an empty union set)
    parent_of: dict[tuple, dict[int, int]] = {}
    children_of: dict[tuple, dict[int, list[int]]] = {}
    seen: set[tuple] = set()
    for (src, rel, dst) in batch.edge_types:
        ei = batch[(src, rel, dst)].edge_index
        if ei.numel() == 0:
            continue
        if is_forward_relation(rel):
            child_nt, parent_nt, c_loc, p_loc = src, dst, ei[0], ei[1]
        else:
            child_nt, parent_nt, c_loc, p_loc = dst, src, ei[1], ei[0]
        if child_nt not in type_slice or parent_nt not in type_slice:
            continue
        rk = (child_nt, canonical_relation(rel), parent_nt)
        gc = (c_loc.to(torch.long) + type_slice[child_nt].start).tolist()
        gp = (p_loc.to(torch.long) + type_slice[parent_nt].start).tolist()
        p_of = parent_of.setdefault(rk, {})
        c_of = children_of.setdefault(rk, {})
        for c, p in zip(gc, gp):
            if int(seg[c]) != int(seg[p]):
                continue
            key = (rk, c, p)
            if key in seen:
                continue
            seen.add(key)
            p_of[c] = p
            c_of.setdefault(p, []).append(c)

    # ---- wide seed row: walk the m2o paths (seed path () first — survives truncation) ----
    paths = m2o_paths(bundle, entity_table, depth=2)
    feat_cols = {nt: feature_col_names(batch[nt].tf) for nt in node_types}

    wide_col_idxs = torch.full((B, wide_len), PAD, dtype=torch.long)
    wide_table_idxs = torch.full((B, wide_len), PAD, dtype=torch.long)
    wide_path_idxs = torch.full((B, wide_len), PAD, dtype=torch.long)
    wide_is_pad = torch.ones(B, wide_len, dtype=torch.bool)
    wide_missing = torch.zeros(B, wide_len, dtype=torch.bool)
    wide_row_time = torch.zeros(B, wide_len, dtype=torch.float64)
    wide_is_timed = torch.zeros(B, wide_len, dtype=torch.bool)
    wide_place: dict[str, list[list[int]]] = {nt: [[], [], [], []] for nt in node_types}
    wide_truncated = 0

    for b in range(B):
        w = 0
        overflow = False
        emitted_nids: set[int] = set()
        for path_id, path in enumerate(paths):
            node = int(seed_flat[b])
            for rk in path:
                nxt = parent_of.get(rk, {}).get(node)
                if nxt is None:
                    node = -1
                    break
                node = nxt
            if node < 0:
                # absent parent -> ONE marker slot tagged with the missing table's type id
                if w >= wide_len:
                    overflow = True
                    break
                miss_nt = path_target_type(entity_table, path)
                wide_missing[b, w] = True
                wide_is_pad[b, w] = False
                wide_path_idxs[b, w] = path_id
                wide_table_idxs[b, w] = bundle.node_type_id[miss_nt]
                w += 1
                continue
            key = int(nid[node])
            if key in emitted_nids:                       # diamond/self-FK: same row via two paths
                continue
            emitted_nids.add(key)
            nt = node_types[int(typeidx[node])]
            cols = feat_cols[nt]
            if not cols:
                continue
            row_in_type = int(intype[node])
            for col_pos, cname in enumerate(cols):
                if w >= wide_len:
                    overflow = True
                    break
                wide_col_idxs[b, w] = vocab[(nt, cname)]
                wide_table_idxs[b, w] = bundle.node_type_id[nt]
                wide_path_idxs[b, w] = path_id
                wide_is_pad[b, w] = False
                wide_row_time[b, w] = float(rowt[node])
                wide_is_timed[b, w] = bool(timed[node])
                wide_place[nt][0].append(b)
                wide_place[nt][1].append(w)
                wide_place[nt][2].append(row_in_type)
                wide_place[nt][3].append(col_pos)
                w += 1
            if overflow:
                break
        wide_truncated += int(overflow)

    # ---- union set: 1-hop children across all child relations, most-recent-first, capped ----
    crels = child_rels(bundle, entity_table)
    R = len(crels)
    child_counts = torch.zeros(B, R, dtype=torch.float32)
    _, slot_lookup, _ = slot_relations(bundle)          # (table_id, role_id) -> GROUP-BY slot index

    elem_mask = torch.zeros(B, set_size, dtype=torch.bool)
    elem_rel_idxs = torch.zeros(B, set_size, dtype=torch.long)
    elem_table_idxs = torch.zeros(B, set_size, dtype=torch.long)
    elem_row_time = torch.zeros(B, set_size, dtype=torch.float64)
    elem_is_timed = torch.zeros(B, set_size, dtype=torch.bool)
    elem_hop = torch.ones(B, set_size, dtype=torch.long)
    set_group_idx = torch.full((B, set_size), -1, dtype=torch.long)
    elem_place: dict[str, list[list[int]]] = {}
    set_truncated = 0

    def elem_append(nt: str, b: int, n: int, row_in_type: int, path_id: int) -> None:
        p = elem_place.setdefault(nt, [[], [], [], []])
        p[0].append(b)
        p[1].append(n)
        p[2].append(row_in_type)
        p[3].append(path_id)

    # sort: hop-1 before hop-2, then most recent first, untimed last, flat index as a stable
    # deterministic tiebreak (identical to the 1-hop ordering when every item is hop 1)
    def _key(it):
        return (it[2], 0 if timed[it[0]] else 1,
                -float(rowt[it[0]]) if timed[it[0]] else 0.0, it[0])

    for b in range(B):
        sf = int(seed_flat[b])
        items: list[tuple[int, int, int]] = []            # (flat idx, fk role id, hop)
        for r_idx, rk in enumerate(crels):
            kids = children_of.get(rk, {}).get(sf, [])
            child_counts[b, r_idx] = float(len(kids))
            role = bundle.fk_role_id.get(rk[1], FK_NONE)
            items.extend((c, role, 1) for c in kids)
        if hop2:
            seen_nids = {int(seed_nid[b])}
            seen_nids.update(int(nid[c]) for c, _, _ in items)
            h1 = [c for c, _, _ in items]
            cand: list[tuple[int, int]] = []              # (flat idx, role of the last traversed FK)
            for c in h1:                                  # grandchildren first (deterministic priority)
                for rk2 in child_rels(bundle, node_types[int(typeidx[c])]):
                    cand.extend((g, bundle.fk_role_id.get(rk2[1], FK_NONE))
                                for g in children_of.get(rk2, {}).get(c, []))
            pseen: set[int] = set()                       # parents dedup by n_id (disjoint copies)
            parents: list[int] = []
            walk = [(sf, entity_table)] + [(c, node_types[int(typeidx[c])]) for c in h1]
            for node, nt in walk:
                for prk in parent_rels(bundle, nt):
                    p = parent_of.get(prk, {}).get(node)
                    if p is None or int(nid[p]) == int(seed_nid[b]) or int(nid[p]) in pseen:
                        continue
                    pseen.add(int(nid[p]))
                    parents.append(p)
            for p in parents:                             # siblings + co-children via each parent
                for rk3 in child_rels(bundle, node_types[int(typeidx[p])]):
                    cand.extend((s, bundle.fk_role_id.get(rk3[1], FK_NONE))
                                for s in children_of.get(rk3, {}).get(p, []))
            for f, role in cand:
                key = int(nid[f])
                if key in seen_nids:                      # never the seed, a hop-1 element, or a dup
                    continue
                seen_nids.add(key)
                items.append((f, role, 2))
        if per_relation_cap is not None:                  # keep the cap most-recent per group
            groups: dict[tuple, list] = {}
            for it in items:
                groups.setdefault((it[2], int(typeidx[it[0]]), it[1]), []).append(it)
            items = [it for g in groups.values() for it in sorted(g, key=_key)[:per_relation_cap]]
        items.sort(key=_key)
        if len(items) > set_size:
            set_truncated += 1
            items = items[:set_size]
        for n, (c, role, hop) in enumerate(items):
            child_nt = node_types[int(typeidx[c])]
            table_id = bundle.node_type_id[child_nt]
            elem_mask[b, n] = True
            elem_rel_idxs[b, n] = role
            elem_table_idxs[b, n] = table_id
            elem_row_time[b, n] = float(rowt[c])
            elem_is_timed[b, n] = bool(timed[c])
            elem_hop[b, n] = hop
            set_group_idx[b, n] = slot_lookup[(table_id, role)]
            elem_append(child_nt, b, n, int(intype[c]), FK_NONE)        # the element row itself
            for prk in parent_rels(bundle, child_nt):                   # + its flattened parents
                p = parent_of.get(prk, {}).get(c)
                if p is None:
                    continue                                            # missing parent: contributes nothing
                if int(nid[p]) == int(seed_nid[b]):
                    continue                                            # never re-inject the seed (by n_id)
                elem_append(node_types[int(typeidx[p])], b, n, int(intype[p]),
                            bundle.fk_role_id.get(prk[1], FK_NONE))

    def pack(place: dict[str, list[list[int]]]) -> dict[str, tuple[Tensor, ...]]:
        return {nt: tuple(torch.tensor(x, dtype=torch.long) for x in p)
                for nt, p in place.items() if p[0]}

    # per-TensorFrame-row metadata for the axial (row/column) attention grids
    row_seg = {nt: seg[type_slice[nt]].clone() for nt in node_types}
    row_times = {nt: rowt[type_slice[nt]].clone() for nt in node_types}
    row_timed = {nt: timed[type_slice[nt]].clone() for nt in node_types}

    return JoinBatch(
        num_seeds=B, wide_len=wide_len, set_size=set_size,
        wide_col_idxs=wide_col_idxs, wide_table_idxs=wide_table_idxs, wide_path_idxs=wide_path_idxs,
        wide_is_pad=wide_is_pad, wide_missing=wide_missing,
        wide_row_time=wide_row_time, wide_is_timed=wide_is_timed,
        elem_mask=elem_mask, elem_rel_idxs=elem_rel_idxs, elem_table_idxs=elem_table_idxs,
        elem_row_time=elem_row_time, elem_is_timed=elem_is_timed, elem_hop=elem_hop,
        set_group_idx=set_group_idx,
        child_counts=child_counts,
        seed_time=seed_time, target=target, has_target=has_target, input_id=input_id,
        tf_dict={nt: batch[nt].tf for nt in node_types},
        wide_placement=pack(wide_place), elem_rows=pack(elem_place),
        row_seg=row_seg, row_times=row_times, row_timed=row_timed,
        wide_truncated=wide_truncated, set_truncated=set_truncated,
    )


# ======================================================================================
# Unified RowModel collate: one SET of denormalized wide rows per seed (see row_model.py)
# ======================================================================================


@dataclass
class RowSetBatch:
    """A per-seed SET of denormalized "one big table" (OBT) wide rows for :class:`RowModel`.

    Per seed: **Row 0** = the seed's own many-to-one closure; **Rows 1..K** = each *direct* (hop-1)
    child, each denormalized by repeating the seed + the seed's full m2o closure into the row (so a
    child cell can attend to the seed's cells within its own row). Every cell is a token tagged by its
    ``(column, table, join-path, recency)`` identity, so heterogeneous wide rows coexist in one grid.

    The seed-closure cells occupy the SAME leading slots of every row and are byte-identical across a
    seed's rows by construction (Row 0's cells are emitted first in each row). The seed is excluded
    from a child's flattened parents at RUNTIME by global n_id (``disjoint=True`` samples a tree, so
    the seed can reappear as a duplicate node copy) — never at schema enumeration, so a child's SECOND
    FK-to-entity (e.g. rel-event ``user_friends.friend``) is kept.
    """

    num_seeds: int                 # B
    m_rows: int                    # M_rows (fixed cap on wide rows / seed)
    cells_per_row: int             # C (fixed cap on cells / wide row)
    # ---- cell grid [B, M_rows, C] ----
    cell_col_id: Tensor            # long; global (table, column) id via column_vocab (pad/marker = -1)
    cell_table_id: Tensor          # long; node-type id (markers: the MISSING table's id; pad = -1)
    cell_path_id: Tensor           # long; RowPaths global path id (pad = -1)
    cell_missing: Tensor           # bool; True = absent-parent marker slot
    cell_row_time: Tensor          # float64; 0 where untimed/pad (gate with cell_is_timed)
    cell_is_timed: Tensor          # bool
    cell_mask: Tensor              # bool; True = real cell (marker counts as real, not pad)
    # ---- per wide row [B, M_rows] ----
    row_mask: Tensor               # bool; True = real wide row (Row 0 always True)
    row_table_id: Tensor           # long; row anchor's table (Row 0 = entity table; pad = -1)
    row_fk_role: Tensor            # long; child->seed fk_role_id (Row 0 = FK_NONE = 0)
    row_hop: Tensor                # long; 0 = seed's own row, 1 = direct child
    row_row_time: Tensor           # float64; anchor node time (Row 0 = seed's own row time)
    row_is_timed: Tensor           # bool
    # ---- per child relation [B, R] (order = paths.child_rels; post-sampler, PRE-truncation) ----
    child_counts: Tensor           # float32
    # ---- per seed [B] ----
    seed_time: Tensor              # float64
    target: Tensor                 # float32
    has_target: Tensor             # bool
    input_id: Tensor               # long; split-table row index (-1 when absent, e.g. fixtures)
    # ---- value encoding ----
    tf_dict: dict                  # node_type -> torch_frame.TensorFrame (as sampled)
    cell_placement: dict           # node_type -> (b, mrow, cpos, row_in_tf, col_pos) CELL scatter
    cell_truncated: int = 0        # rows whose cells hit the C cap with cells left over
    row_truncated: int = 0         # seeds whose row set hit the M_rows cap with children left over

    def to(self, device) -> "RowSetBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return RowSetBatch(
            num_seeds=self.num_seeds, m_rows=self.m_rows, cells_per_row=self.cells_per_row,
            cell_col_id=mv(self.cell_col_id), cell_table_id=mv(self.cell_table_id),
            cell_path_id=mv(self.cell_path_id), cell_missing=mv(self.cell_missing),
            cell_row_time=mv(self.cell_row_time), cell_is_timed=mv(self.cell_is_timed),
            cell_mask=mv(self.cell_mask),
            row_mask=mv(self.row_mask), row_table_id=mv(self.row_table_id),
            row_fk_role=mv(self.row_fk_role), row_hop=mv(self.row_hop),
            row_row_time=mv(self.row_row_time), row_is_timed=mv(self.row_is_timed),
            child_counts=mv(self.child_counts),
            seed_time=mv(self.seed_time), target=mv(self.target), has_target=mv(self.has_target),
            input_id=mv(self.input_id),
            tf_dict={k: v.to(device) for k, v in self.tf_dict.items()},
            cell_placement={k: tuple(t.to(device) for t in v) for k, v in self.cell_placement.items()},
            cell_truncated=self.cell_truncated, row_truncated=self.row_truncated,
        )

    def pretty_shapes(self) -> str:
        n_cells = int((self.cell_mask & ~self.cell_missing).sum())
        n_markers = int(self.cell_missing.sum())
        per_seed = self.row_mask.sum(dim=1)
        lines = [
            f"RowSetBatch: B={self.num_seeds} M_rows={self.m_rows} C={self.cells_per_row}",
            f"  rows: total={int(per_seed.sum())}  min/seed={int(per_seed.min())}  "
            f"max/seed={int(per_seed.max())}  truncated seeds={self.row_truncated}",
            f"  cells: real={n_cells}  missing markers={n_markers}  "
            f"truncated rows={self.cell_truncated}",
            f"  child_counts {tuple(self.child_counts.shape)}  targets={int(self.has_target.sum())}",
            f"  tf_dict types: {sorted(self.tf_dict)}  placement types: {sorted(self.cell_placement)}",
        ]
        return "\n".join(lines)


def to_row_set_batch(
    batch,
    bundle: GraphBundle,
    entity_table: str,
    *,
    m_rows: int = 128,
    cells_per_row: int = 48,
) -> RowSetBatch:
    """Convert a disjoint sampled ``HeteroData`` minibatch into a dense :class:`RowSetBatch`.

    The sampler / temporal cutoff / adjacency are IDENTICAL to :func:`to_join_batch` (this duplicates
    that setup so ``to_join_batch`` stays byte-for-byte intact). The one behavioral change is the
    packing: instead of (one wide row) + (a pre-pooled union set), emit a SET of denormalized wide
    rows, repeating the seed's m2o closure into every child row (the OBT denormalization).
    """
    vocab = column_vocab(bundle)
    node_types = [nt for nt in bundle.node_types if nt in batch.node_types and batch[nt].num_nodes > 0]

    # ---- flatten nodes across types (stable order) — duplicated from to_join_batch ----
    type_slice: dict[str, slice] = {}
    seg_list, rowt_list, timed_list, nid_list, intype_list, typeidx_list = [], [], [], [], [], []
    offset = 0
    for ti, nt in enumerate(node_types):
        st = batch[nt]
        n = st.num_nodes
        type_slice[nt] = slice(offset, offset + n)
        offset += n
        seg_list.append(st.batch.to(torch.long))
        if "time" in st:
            rowt_list.append(st.time.to(torch.float64))
            timed_list.append(torch.ones(n, dtype=torch.bool))
        else:
            rowt_list.append(torch.zeros(n, dtype=torch.float64))
            timed_list.append(torch.zeros(n, dtype=torch.bool))
        nid_list.append(st.n_id.to(torch.long) if "n_id" in st else torch.arange(n))
        intype_list.append(torch.arange(n, dtype=torch.long))
        typeidx_list.append(torch.full((n,), ti, dtype=torch.long))

    seg = torch.cat(seg_list)
    rowt = torch.cat(rowt_list)
    timed = torch.cat(timed_list)
    nid = torch.cat(nid_list)
    intype = torch.cat(intype_list)
    typeidx = torch.cat(typeidx_list)

    B = int(seg.max().item()) + 1
    ent = batch[entity_table]
    seed_time = _seed_time_per_segment(ent, B)

    n_seeds = ent.seed_time.numel()
    ent_start = type_slice[entity_table].start
    seed_flat = torch.full((B,), -1, dtype=torch.long)
    seed_flat[ent.batch[:n_seeds].to(torch.long)] = torch.arange(n_seeds) + ent_start
    assert int((seed_flat < 0).sum()) == 0, "could not locate a seed node for every segment"
    seed_nid = nid[seed_flat]                                             # [B]

    target = torch.zeros(B, dtype=torch.float32)
    has_target = torch.zeros(B, dtype=torch.bool)
    if "y" in ent:
        y = ent.y.to(torch.float32)
        yb = ent.batch[: y.numel()] if y.numel() != ent.batch.numel() else ent.batch
        target[yb] = y
        has_target[yb] = True
    if "input_id" in ent:
        input_id = ent.input_id.to(torch.long)[:B]
    else:
        input_id = torch.full((B,), -1, dtype=torch.long)

    # ---- adjacency over ALL edge types (children arrive via rev_*) — duplicated ----
    parent_of: dict[tuple, dict[int, int]] = {}
    children_of: dict[tuple, dict[int, list[int]]] = {}
    seen: set[tuple] = set()
    for (src, rel, dst) in batch.edge_types:
        ei = batch[(src, rel, dst)].edge_index
        if ei.numel() == 0:
            continue
        if is_forward_relation(rel):
            child_nt, parent_nt, c_loc, p_loc = src, dst, ei[0], ei[1]
        else:
            child_nt, parent_nt, c_loc, p_loc = dst, src, ei[1], ei[0]
        if child_nt not in type_slice or parent_nt not in type_slice:
            continue
        rk = (child_nt, canonical_relation(rel), parent_nt)
        gc = (c_loc.to(torch.long) + type_slice[child_nt].start).tolist()
        gp = (p_loc.to(torch.long) + type_slice[parent_nt].start).tolist()
        p_of = parent_of.setdefault(rk, {})
        c_of = children_of.setdefault(rk, {})
        for c, p in zip(gc, gp):
            if int(seg[c]) != int(seg[p]):
                continue
            key = (rk, c, p)
            if key in seen:
                continue
            seen.add(key)
            p_of[c] = p
            c_of.setdefault(p, []).append(c)

    # ---- schema maps ----
    rp = row_paths(bundle, entity_table, depth=2)
    feat_cols = {nt: feature_col_names(batch[nt].tf) for nt in node_types}
    crels = child_rels(bundle, entity_table)
    R = len(crels)
    C = cells_per_row

    # A cell record: (col_id, table_id, path_id, missing, row_time, is_timed, nt|None, row_in_tf, col_pos)
    def node_cells(node: int, path_id: int) -> list[tuple]:
        nt = node_types[int(typeidx[node])]
        cols = feat_cols[nt]
        rt, tm, rit = float(rowt[node]), bool(timed[node]), int(intype[node])
        tid = bundle.node_type_id[nt]
        return [(vocab[(nt, cname)], tid, path_id, False, rt, tm, nt, rit, cp)
                for cp, cname in enumerate(cols)]

    def marker_cell(miss_nt: str, path_id: int) -> tuple:
        return (PAD, bundle.node_type_id[miss_nt], path_id, True, 0.0, False, None, -1, -1)

    def seed_closure(b: int) -> tuple[list[tuple], set[int]]:
        """Row 0 = the seed's m2o closure (block-A path ids); returns (cells, emitted n_ids)."""
        cells: list[tuple] = []
        nids: set[int] = set()
        for path_id, path in enumerate(rp.m2o):
            node = int(seed_flat[b])
            dead = False
            for rk in path:
                nxt = parent_of.get(rk, {}).get(node)
                if nxt is None:
                    dead = True
                    break
                node = nxt
            if dead:
                cells.append(marker_cell(path_target_type(entity_table, path), path_id))
                continue
            key = int(nid[node])
            if key in nids:                                   # diamond/self-FK: same row via two paths
                continue
            nids.add(key)
            if not feat_cols[node_types[int(typeidx[node])]]:
                continue
            cells.extend(node_cells(node, path_id))
        return cells, nids

    def child_cells(child: int, closure_nids: set[int], seed_nid_b: int) -> list[tuple]:
        """A child's own anchor (SELF id) + its non-seed flattened parents (block-C ids)."""
        cells: list[tuple] = []
        emitted = set(closure_nids)                           # per-row dedup spans the spliced closure
        ck = int(nid[child])
        if ck not in emitted:
            emitted.add(ck)
            cells.extend(node_cells(child, rp.self_id))
        child_nt = node_types[int(typeidx[child])]
        for prk in parent_rels(bundle, child_nt):
            p = parent_of.get(prk, {}).get(child)
            if p is None:
                continue
            if int(nid[p]) == seed_nid_b:                     # never re-inject the seed (runtime n_id)
                continue
            pk = int(nid[p])
            if pk in emitted:
                continue
            emitted.add(pk)
            cells.extend(node_cells(p, rp.parent_lookup[prk]))
        return cells

    def _key(flat_idx: int):                                  # most-recent-first, untimed last, stable
        return (0 if timed[flat_idx] else 1,
                -float(rowt[flat_idx]) if timed[flat_idx] else 0.0, flat_idx)

    # ---- allocate grid tensors ----
    cell_col_id = torch.full((B, m_rows, C), PAD, dtype=torch.long)
    cell_table_id = torch.full((B, m_rows, C), PAD, dtype=torch.long)
    cell_path_id = torch.full((B, m_rows, C), PAD, dtype=torch.long)
    cell_missing = torch.zeros(B, m_rows, C, dtype=torch.bool)
    cell_row_time = torch.zeros(B, m_rows, C, dtype=torch.float64)
    cell_is_timed = torch.zeros(B, m_rows, C, dtype=torch.bool)
    cell_mask = torch.zeros(B, m_rows, C, dtype=torch.bool)
    row_mask = torch.zeros(B, m_rows, dtype=torch.bool)
    row_table_id = torch.full((B, m_rows), PAD, dtype=torch.long)
    row_fk_role = torch.zeros(B, m_rows, dtype=torch.long)
    row_hop = torch.zeros(B, m_rows, dtype=torch.long)
    row_row_time = torch.zeros(B, m_rows, dtype=torch.float64)
    row_is_timed = torch.zeros(B, m_rows, dtype=torch.bool)
    child_counts = torch.zeros(B, R, dtype=torch.float32)
    place: dict[str, list[list[int]]] = {nt: [[], [], [], [], []] for nt in node_types}
    cell_truncated = 0
    row_truncated = 0

    def write_row(b: int, mrow: int, anchor: int, hop: int, fk_role: int, cells: list[tuple]) -> None:
        nonlocal cell_truncated
        row_mask[b, mrow] = True
        row_hop[b, mrow] = hop
        row_fk_role[b, mrow] = fk_role
        row_table_id[b, mrow] = bundle.node_type_id[node_types[int(typeidx[anchor])]]
        row_row_time[b, mrow] = float(rowt[anchor])
        row_is_timed[b, mrow] = bool(timed[anchor])
        if len(cells) > C:
            cell_truncated += 1
            cells = cells[:C]
        for cpos, (col_id, tid, pid, miss, rt, tm, nt, rit, cp) in enumerate(cells):
            cell_col_id[b, mrow, cpos] = col_id
            cell_table_id[b, mrow, cpos] = tid
            cell_path_id[b, mrow, cpos] = pid
            cell_missing[b, mrow, cpos] = miss
            cell_row_time[b, mrow, cpos] = rt
            cell_is_timed[b, mrow, cpos] = tm
            cell_mask[b, mrow, cpos] = True
            if nt is not None:
                place[nt][0].append(b)
                place[nt][1].append(mrow)
                place[nt][2].append(cpos)
                place[nt][3].append(rit)
                place[nt][4].append(cp)

    for b in range(B):
        sf = int(seed_flat[b])
        s_cells, s_nids = seed_closure(b)
        # Row 0 = the seed's own wide row (its m2o closure)
        write_row(b, 0, sf, hop=0, fk_role=FK_NONE, cells=s_cells)
        # Rows 1..K = direct children, most-recent-first, capped
        items: list[tuple[int, int]] = []                     # (child flat idx, fk role id)
        for r_idx, rk in enumerate(crels):
            kids = children_of.get(rk, {}).get(sf, [])
            child_counts[b, r_idx] = len(kids)                # PRE-truncation
            role = bundle.fk_role_id.get(rk[1], FK_NONE)
            items.extend((c, role) for c in kids)
        items.sort(key=lambda it: _key(it[0]))
        cap = m_rows - 1
        if len(items) > cap:
            row_truncated += 1
            items = items[:cap]
        for j, (child, role) in enumerate(items):
            row = s_cells + child_cells(child, s_nids, int(seed_nid[b]))
            write_row(b, j + 1, child, hop=1, fk_role=role, cells=row)

    def pack(p: dict[str, list[list[int]]]) -> dict[str, tuple[Tensor, ...]]:
        return {nt: tuple(torch.tensor(x, dtype=torch.long) for x in lists)
                for nt, lists in p.items() if lists[0]}

    return RowSetBatch(
        num_seeds=B, m_rows=m_rows, cells_per_row=C,
        cell_col_id=cell_col_id, cell_table_id=cell_table_id, cell_path_id=cell_path_id,
        cell_missing=cell_missing, cell_row_time=cell_row_time, cell_is_timed=cell_is_timed,
        cell_mask=cell_mask,
        row_mask=row_mask, row_table_id=row_table_id, row_fk_role=row_fk_role, row_hop=row_hop,
        row_row_time=row_row_time, row_is_timed=row_is_timed,
        child_counts=child_counts,
        seed_time=seed_time, target=target, has_target=has_target, input_id=input_id,
        tf_dict={nt: batch[nt].tf for nt in node_types},
        cell_placement=pack(place),
        cell_truncated=cell_truncated, row_truncated=row_truncated,
    )
