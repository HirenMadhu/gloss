"""Phase 0 collate (DOC-RT): a per-seed-disjoint sampled ``HeteroData`` minibatch -> a dense, padded
**cell-token** batch (RT substrate).

RT (Relational Transformer, arXiv 2510.06377) tokenizes every *cell* (row x feature-column) and attends
over a flat ``[B, S]`` sequence with relational masks. We replicate that contract on top of relbench's
PyG temporal sampler:

  per-cell  ``[B, S]``           : node_idxs (local row), col_idxs (table,column), table_idxs,
                                   is_padding, is_seed_cell, row_time, is_timed, n_id
  per-cell  ``[B, S, max_fk]``   : f2p_nbr_idxs — local row indices this cell's row references via FK
  per-seed  ``[B]``              : seed_time, target, has_target
  encoding  (per node type)      : tf_dict (TensorFrame) + cell_placement (scatter map) so Phase-2's
                                   cell encoder can place per-cell value vectors into the [B, S, d] grid.

The four RT relational masks (same-column / same-row + forward-FK / reverse-FK / global) are derived
from these index tensors inside the model (``rt_substrate.py``), recomputed each forward.

``seq_len`` is a **fixed** cap (pad/truncate) — RT's Rust sampler does the same; a fixed S keeps the
attention masks (and any FlexAttention block masks) a single compiled shape. Cells of the seed root row
are emitted first so a readout head always sees them and they survive truncation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .graph import GraphBundle, is_forward_relation

PAD = -1


def feature_col_names(tf) -> list[str]:
    """Canonical, deterministic feature-column order for a TensorFrame: sorted across all stypes.

    Both the collate (which assigns each cell a column slot) and the cell encoder (which reorders the
    pytorch-frame output) use this, so a cell's ``col_idx`` agrees on both sides without the collate
    having to know pytorch-frame's internal concatenation order.
    """
    names: list[str] = []
    for cols in tf.col_names_dict.values():
        names.extend(cols)
    return sorted(names)


def column_vocab(bundle: GraphBundle) -> dict[tuple[str, str], int]:
    """Global ``(node_type, column) -> id`` map (stable sorted). ``col_idxs`` use it so two cells share
    a column id iff they are the *same column of the same table* — exactly RT's ``same_col_table``."""
    cached = getattr(bundle, "_column_vocab", None)
    if cached is not None:
        return cached
    pairs: list[tuple[str, str]] = []
    for nt in bundle.node_types:
        tf = bundle.data[nt].tf
        for c in feature_col_names(tf):
            pairs.append((nt, c))
    vocab = {p: i for i, p in enumerate(sorted(pairs))}
    try:
        bundle._column_vocab = vocab  # type: ignore[attr-defined]
    except Exception:
        pass
    return vocab


@dataclass
class CellBatch:
    num_seeds: int                 # B
    seq_len: int                   # S
    max_fk: int
    # per-cell [B, S]
    node_idxs: Tensor              # long; local row index within the seed subgraph (pad = -1)
    col_idxs: Tensor               # long; global (table, column) id (pad = -1)
    table_idxs: Tensor             # long; node-type id (pad = -1)
    is_padding: Tensor             # bool; True = pad slot
    is_seed_cell: Tensor           # bool; cell belongs to the seed root row
    row_time: Tensor               # float64; 0 where untimed/pad (gate with is_timed)
    is_timed: Tensor               # bool
    n_id: Tensor                   # long; global node id (pad = -1) — debugging/leakage
    # per-cell [B, S, max_fk]
    f2p_nbr_idxs: Tensor           # long; local row idxs referenced via FK (pad = -1)
    # per-seed [B]
    seed_time: Tensor              # float64
    target: Tensor                 # float32
    has_target: Tensor             # bool
    # for value encoding (scatter): per node type
    tf_dict: dict                  # node_type -> torch_frame.TensorFrame
    cell_placement: dict           # node_type -> (b_idx, s_idx, row_idx, col_idx) LongTensors

    def to(self, device) -> "CellBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return CellBatch(
            num_seeds=self.num_seeds, seq_len=self.seq_len, max_fk=self.max_fk,
            node_idxs=mv(self.node_idxs), col_idxs=mv(self.col_idxs), table_idxs=mv(self.table_idxs),
            is_padding=mv(self.is_padding), is_seed_cell=mv(self.is_seed_cell),
            row_time=mv(self.row_time), is_timed=mv(self.is_timed), n_id=mv(self.n_id),
            f2p_nbr_idxs=mv(self.f2p_nbr_idxs),
            seed_time=mv(self.seed_time), target=mv(self.target), has_target=mv(self.has_target),
            tf_dict={k: v.to(device) for k, v in self.tf_dict.items()},
            cell_placement={k: tuple(t.to(device) for t in v) for k, v in self.cell_placement.items()},
        )

    def pretty_shapes(self) -> str:
        lines = [f"CellBatch: B={self.num_seeds} S={self.seq_len} max_fk={self.max_fk}"]
        for name in ("node_idxs", "col_idxs", "table_idxs", "is_padding", "is_seed_cell",
                     "row_time", "is_timed", "f2p_nbr_idxs"):
            t = getattr(self, name)
            lines.append(f"  {name:14s} {tuple(t.shape)} {t.dtype}")
        n_real = int((~self.is_padding).sum())
        lines.append(f"  real cells={n_real}  seed cells={int(self.is_seed_cell.sum())}  "
                     f"targets={int(self.has_target.sum())}")
        lines.append(f"  tf_dict types: {sorted(self.tf_dict)}")
        return "\n".join(lines)


def _seed_time_per_segment(entity_store, num_segments: int) -> Tensor:
    seed_time = entity_store.seed_time.to(torch.float64)
    batch = entity_store.batch
    out = torch.full((num_segments,), float("nan"), dtype=torch.float64)
    if seed_time.numel() == batch.numel():
        out[batch] = seed_time
    else:
        out[batch[: seed_time.numel()]] = seed_time
    assert not torch.isnan(out).any(), "could not recover seed_time for every segment"
    return out


def to_cell_batch(
    batch,
    bundle: GraphBundle,
    entity_table: str,
    *,
    seq_len: int = 1024,
    max_fk: int = 5,
) -> CellBatch:
    """Convert a disjoint sampled ``HeteroData`` minibatch into a dense :class:`CellBatch`."""
    vocab = column_vocab(bundle)
    node_types = [nt for nt in bundle.node_types if nt in batch.node_types and batch[nt].num_nodes > 0]

    # ---- flatten nodes across types (stable order); track per-node (type, in-type row) ----
    type_slice: dict[str, slice] = {}
    seg_list, ntype_list, rowt_list, timed_list, seed_list, nid_list = [], [], [], [], [], []
    intype_list, typeidx_list = [], []
    offset = 0
    for ti, nt in enumerate(node_types):
        st = batch[nt]
        n = st.num_nodes
        type_slice[nt] = slice(offset, offset + n)
        offset += n
        seg_list.append(st.batch.to(torch.long))
        ntype_list.append(torch.full((n,), bundle.node_type_id[nt], dtype=torch.long))
        if "time" in st:
            rowt_list.append(st.time.to(torch.float64))
            timed_list.append(torch.ones(n, dtype=torch.bool))
        else:
            rowt_list.append(torch.zeros(n, dtype=torch.float64))
            timed_list.append(torch.zeros(n, dtype=torch.bool))
        is_seed = torch.zeros(n, dtype=torch.bool)
        if nt == entity_table and "seed_time" in st:
            is_seed[: st.seed_time.numel()] = True
        seed_list.append(is_seed)
        nid_list.append(st.n_id.to(torch.long) if "n_id" in st else torch.arange(n))
        intype_list.append(torch.arange(n, dtype=torch.long))
        typeidx_list.append(torch.full((n,), ti, dtype=torch.long))

    seg = torch.cat(seg_list)
    ntype = torch.cat(ntype_list)
    rowt = torch.cat(rowt_list)
    timed = torch.cat(timed_list)
    is_seed_node = torch.cat(seed_list)
    nid = torch.cat(nid_list)
    intype = torch.cat(intype_list)
    typeidx = torch.cat(typeidx_list)
    total = seg.numel()

    B = int(seg.max().item()) + 1
    seed_time = _seed_time_per_segment(batch[entity_table], B)

    # per-seed local node index (0..R_b-1), stable
    local_node = torch.empty(total, dtype=torch.long)
    order = torch.argsort(seg, stable=True)
    seen = torch.zeros(B, dtype=torch.long)
    for gi in order.tolist():
        s = int(seg[gi])
        local_node[gi] = int(seen[s])
        seen[s] += 1

    # labels (attached by AttachTargetTransform to the entity store as `y`)
    ent = batch[entity_table]
    target = torch.zeros(B, dtype=torch.float32)
    has_target = torch.zeros(B, dtype=torch.bool)
    if "y" in ent:
        y = ent.y.to(torch.float32)
        yb = ent.batch[: y.numel()] if y.numel() != ent.batch.numel() else ent.batch
        target[yb] = y
        has_target[yb] = True

    # ---- f2p neighbors: per global node, the local-node idxs it references via forward FK edges ----
    f2p: list[list[int]] = [[] for _ in range(total)]

    def flat_index(nt: str, local_in_store: Tensor) -> Tensor:
        return local_in_store + type_slice[nt].start

    for (src, rel, dst) in bundle.edge_types:
        if not is_forward_relation(rel):
            continue
        if (src, rel, dst) not in batch.edge_types:
            continue
        ei = batch[(src, rel, dst)].edge_index
        if ei.numel() == 0:
            continue
        # f2p_<col>: src = table holding the FK (child), dst = referenced pkey table (parent).
        gchild = flat_index(src, ei[0])
        gparent = flat_index(dst, ei[1])
        for c, p in zip(gchild.tolist(), gparent.tolist()):
            if int(seg[c]) == int(seg[p]) and len(f2p[c]) < max_fk:
                f2p[c].append(int(local_node[p]))

    # ---- enumerate cells; seed-row cells first so they survive truncation ----
    feat_cols = {nt: feature_col_names(batch[nt].tf) for nt in node_types}
    s_counter = [0] * B
    place: dict[str, list[list[int]]] = {nt: [[], [], [], []] for nt in node_types}  # b,s,row,col
    node_idxs = torch.full((B, seq_len), PAD, dtype=torch.long)
    col_idxs = torch.full((B, seq_len), PAD, dtype=torch.long)
    table_idxs = torch.full((B, seq_len), PAD, dtype=torch.long)
    is_padding = torch.ones(B, seq_len, dtype=torch.bool)
    is_seed_cell = torch.zeros(B, seq_len, dtype=torch.bool)
    row_time = torch.zeros(B, seq_len, dtype=torch.float64)
    is_timed = torch.zeros(B, seq_len, dtype=torch.bool)
    n_id_g = torch.full((B, seq_len), PAD, dtype=torch.long)
    f2p_g = torch.full((B, seq_len, max_fk), PAD, dtype=torch.long)

    # node visiting order: per seed, seed-root nodes first, then by local_node
    visit = sorted(
        range(total),
        key=lambda g: (int(seg[g]), 0 if bool(is_seed_node[g]) else 1, int(local_node[g])),
    )
    for g in visit:
        b = int(seg[g])
        nt = node_types[int(typeidx[g])]
        cols = feat_cols[nt]
        if not cols:
            continue
        row_in_type = int(intype[g])
        ln = int(local_node[g])
        nbrs = f2p[g]
        for col_pos, _cname in enumerate(cols):
            s = s_counter[b]
            if s >= seq_len:
                break
            s_counter[b] = s + 1
            node_idxs[b, s] = ln
            col_idxs[b, s] = vocab[(nt, _cname)]
            table_idxs[b, s] = bundle.node_type_id[nt]
            is_padding[b, s] = False
            is_seed_cell[b, s] = bool(is_seed_node[g])
            row_time[b, s] = float(rowt[g])
            is_timed[b, s] = bool(timed[g])
            n_id_g[b, s] = int(nid[g])
            for k, nb in enumerate(nbrs):
                f2p_g[b, s, k] = nb
            place[nt][0].append(b)
            place[nt][1].append(s)
            place[nt][2].append(row_in_type)
            place[nt][3].append(col_pos)

    cell_placement = {
        nt: (torch.tensor(p[0], dtype=torch.long), torch.tensor(p[1], dtype=torch.long),
             torch.tensor(p[2], dtype=torch.long), torch.tensor(p[3], dtype=torch.long))
        for nt, p in place.items() if p[0]
    }
    tf_dict = {nt: batch[nt].tf for nt in node_types}

    return CellBatch(
        num_seeds=B, seq_len=seq_len, max_fk=max_fk,
        node_idxs=node_idxs, col_idxs=col_idxs, table_idxs=table_idxs,
        is_padding=is_padding, is_seed_cell=is_seed_cell, row_time=row_time, is_timed=is_timed,
        n_id=n_id_g, f2p_nbr_idxs=f2p_g,
        seed_time=seed_time, target=target, has_target=has_target,
        tf_dict=tf_dict, cell_placement=cell_placement,
    )
