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
from .paths import child_rels, m2o_paths, parent_rels, path_target_type

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
    elem_hop: Tensor               # long; 1 everywhere (MVP; reserved for deeper union sets)
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
            f"  set:  elements={int(per_seed.sum())}  empty-set seeds={int((per_seed == 0).sum())}  "
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
) -> JoinBatch:
    """Convert a disjoint sampled ``HeteroData`` minibatch into a dense :class:`JoinBatch`."""
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

    elem_mask = torch.zeros(B, set_size, dtype=torch.bool)
    elem_rel_idxs = torch.zeros(B, set_size, dtype=torch.long)
    elem_table_idxs = torch.zeros(B, set_size, dtype=torch.long)
    elem_row_time = torch.zeros(B, set_size, dtype=torch.float64)
    elem_is_timed = torch.zeros(B, set_size, dtype=torch.bool)
    elem_hop = torch.ones(B, set_size, dtype=torch.long)
    elem_place: dict[str, list[list[int]]] = {}
    set_truncated = 0

    def elem_append(nt: str, b: int, n: int, row_in_type: int, path_id: int) -> None:
        p = elem_place.setdefault(nt, [[], [], [], []])
        p[0].append(b)
        p[1].append(n)
        p[2].append(row_in_type)
        p[3].append(path_id)

    for b in range(B):
        items: list[tuple[int, int]] = []                 # (child flat idx, child-relation index)
        for r_idx, rk in enumerate(crels):
            kids = children_of.get(rk, {}).get(int(seed_flat[b]), [])
            child_counts[b, r_idx] = float(len(kids))
            items.extend((c, r_idx) for c in kids)
        # most recent first; untimed last; flat index as a stable deterministic tiebreak
        items.sort(key=lambda it: (0 if timed[it[0]] else 1,
                                   -float(rowt[it[0]]) if timed[it[0]] else 0.0, it[0]))
        if len(items) > set_size:
            set_truncated += 1
            items = items[:set_size]
        for n, (c, r_idx) in enumerate(items):
            rk = crels[r_idx]
            child_nt = node_types[int(typeidx[c])]
            elem_mask[b, n] = True
            elem_rel_idxs[b, n] = bundle.fk_role_id.get(rk[1], FK_NONE)
            elem_table_idxs[b, n] = bundle.node_type_id[child_nt]
            elem_row_time[b, n] = float(rowt[c])
            elem_is_timed[b, n] = bool(timed[c])
            elem_append(child_nt, b, n, int(intype[c]), FK_NONE)        # the child row itself
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
        child_counts=child_counts,
        seed_time=seed_time, target=target, has_target=has_target, input_id=input_id,
        tf_dict={nt: batch[nt].tf for nt in node_types},
        wide_placement=pack(wide_place), elem_rows=pack(elem_place),
        row_seg=row_seg, row_times=row_times, row_timed=row_timed,
        wide_truncated=wide_truncated, set_truncated=set_truncated,
    )
