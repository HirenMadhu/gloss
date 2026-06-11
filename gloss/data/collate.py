"""Phase 0 collate: a per-seed-disjoint sampled HeteroData minibatch -> a dense ``GlossBatch``.

The model attends over node pairs within each seed's sampled subgraph. We therefore turn the merged
(but disjoint) PyG batch into **dense, padded, per-seed** tensors:

  node-level   ``[B, N_max]``           : node_type_id, seg/pad mask, row_time, is_timed, is_seed, n_id
  pairwise     ``[B, N_max, N_max]``    : attend_mask, dt, tau, temporal_valid, metapath_id, fk_role_id
  per-seed     ``[B]``                  : seed_time, T_ctx

Cell *features* are not encoded here — Phase 2's column encoder consumes ``tf_dict`` (the per-type
``TensorFrame``s) and scatters the resulting node vectors into the ``[B, N_max, d]`` grid using
``placement``.

Dimensionless time (the scale-equivariance carrier): ``tau_uw = log(dt_uw / T_ctx)`` for pairs where
both endpoints are timestamped (``temporal_valid``); ``T_ctx`` = median nonzero gap in that seed's
subgraph. Everything else (timeless node, zero gap, >1-hop, padding) is left to the structural-bias
bucket downstream — here we just set ``temporal_valid=False`` and ``tau=0`` for those pairs.

Time math is done in ``float64`` so the global time-rescale invariance test passes at 1e-5.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .graph import FK_NONE, MP_MULTIHOP, MP_PAD, MP_SELF, GraphBundle, is_forward_relation


@dataclass
class GlossBatch:
    num_seeds: int                 # B
    n_max: int
    # node-level [B, N_max]
    node_type_id: Tensor           # long; pad = -1
    pad_mask: Tensor               # bool; True = real node
    is_seed: Tensor                # bool
    is_timed: Tensor               # bool
    row_time: Tensor               # float64; 0 where untimed/pad (use is_timed to gate)
    n_id: Tensor                   # long; global node id in the full graph (pad = -1)
    # per-seed [B]
    seed_time: Tensor              # float64
    t_ctx: Tensor                  # float64; median nonzero gap per subgraph (>=1)
    # pairwise [B, N_max, N_max]
    attend_mask: Tensor            # bool; True = j may attend to i (<=2-hop or self, same subgraph)
    metapath_id: Tensor            # long; {PAD,SELF,MULTIHOP} + per-relation
    fk_role_id: Tensor             # long; relation fk-role for 1-hop pairs, else FK_NONE
    dt: Tensor                     # float64; |row_time_i - row_time_j|, 0 where not temporal_valid
    tau: Tensor                    # float64; log(dt / T_ctx), 0 where not temporal_valid
    temporal_valid: Tensor         # bool; both endpoints timestamped AND attend
    # carried for Phase 2 encoding: per node-type TensorFrame + grid placement
    tf_dict: dict                  # node_type -> torch_frame.TensorFrame (rows in store order)
    placement: dict                # node_type -> (seg LongTensor[n_t], localpos LongTensor[n_t])

    def to(self, device) -> "GlossBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return GlossBatch(
            num_seeds=self.num_seeds, n_max=self.n_max,
            node_type_id=mv(self.node_type_id), pad_mask=mv(self.pad_mask), is_seed=mv(self.is_seed),
            is_timed=mv(self.is_timed), row_time=mv(self.row_time), n_id=mv(self.n_id),
            seed_time=mv(self.seed_time), t_ctx=mv(self.t_ctx),
            attend_mask=mv(self.attend_mask), metapath_id=mv(self.metapath_id),
            fk_role_id=mv(self.fk_role_id), dt=mv(self.dt), tau=mv(self.tau),
            temporal_valid=mv(self.temporal_valid),
            tf_dict={k: v.to(device) for k, v in self.tf_dict.items()},
            placement={k: (s.to(device), p.to(device)) for k, (s, p) in self.placement.items()},
        )

    def pretty_shapes(self) -> str:
        lines = [f"GlossBatch: B={self.num_seeds} N_max={self.n_max}"]
        for name in ("node_type_id", "pad_mask", "is_seed", "is_timed", "row_time", "n_id"):
            lines.append(f"  {name:14s} {tuple(getattr(self, name).shape)} {getattr(self, name).dtype}")
        for name in ("seed_time", "t_ctx"):
            lines.append(f"  {name:14s} {tuple(getattr(self, name).shape)} {getattr(self, name).dtype}")
        for name in ("attend_mask", "metapath_id", "fk_role_id", "dt", "tau", "temporal_valid"):
            lines.append(f"  {name:14s} {tuple(getattr(self, name).shape)} {getattr(self, name).dtype}")
        n_real = int(self.pad_mask.sum())
        n_attend = int(self.attend_mask.sum())
        n_tv = int(self.temporal_valid.sum())
        lines.append(f"  real nodes={n_real}  attend pairs={n_attend}  temporal_valid pairs={n_tv}")
        lines.append(f"  tf_dict types: {sorted(self.tf_dict)}")
        return "\n".join(lines)


def _seed_time_per_segment(entity_store, num_segments: int) -> Tensor:
    """Map segment index -> seed time (float64). Handles seed_time sized to all-ent-nodes or to-seeds."""
    seed_time = entity_store.seed_time.to(torch.float64)
    batch = entity_store.batch
    out = torch.full((num_segments,), float("nan"), dtype=torch.float64)
    if seed_time.numel() == batch.numel():
        out[batch] = seed_time
    else:  # seeds are the first `num_segments` entity nodes in disjoint order
        out[batch[:seed_time.numel()]] = seed_time
    assert not torch.isnan(out).any(), "could not recover seed_time for every segment"
    return out


def to_gloss_batch(batch, bundle: GraphBundle, entity_table: str, *, max_nodes: int = 4096) -> GlossBatch:
    """Convert a disjoint sampled ``HeteroData`` minibatch into a dense :class:`GlossBatch`."""
    node_types = [nt for nt in bundle.node_types if nt in batch.node_types and batch[nt].num_nodes > 0]

    # ---- flatten nodes across types (stable order) ----
    seg_list, ntype_list, rowt_list, timed_list, seed_list, nid_list = [], [], [], [], [], []
    type_slice: dict[str, slice] = {}
    offset = 0
    for nt in node_types:
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
            # seeds carry seed_time; in disjoint mode entity store == the B seeds (deduped per segment)
            is_seed[: st.seed_time.numel()] = True
        seed_list.append(is_seed)
        nid_list.append(st.n_id.to(torch.long) if "n_id" in st else torch.arange(n))

    seg = torch.cat(seg_list)
    ntype = torch.cat(ntype_list)
    rowt = torch.cat(rowt_list)
    timed = torch.cat(timed_list)
    is_seed_flat = torch.cat(seed_list)
    nid = torch.cat(nid_list)
    total = seg.numel()

    B = int(seg.max().item()) + 1
    seed_time = _seed_time_per_segment(batch[entity_table], B)

    # local index within each segment (0..n_b-1) and per-segment counts
    local_pos = torch.empty(total, dtype=torch.long)
    counts = torch.zeros(B, dtype=torch.long)
    # stable per-segment enumeration
    order = torch.argsort(seg, stable=True)
    seen = torch.zeros(B, dtype=torch.long)
    for gi in order.tolist():
        s = int(seg[gi])
        local_pos[gi] = seen[s]
        seen[s] += 1
    counts = seen
    n_max = int(counts.max().item())
    if n_max > max_nodes:
        n_max = max_nodes  # truncation guard (rare; logged by caller if hit)

    # global flat idx -> (segment, local) ; record placement per type for Phase-2 scatter
    placement: dict[str, tuple[Tensor, Tensor]] = {}
    for nt in node_types:
        sl = type_slice[nt]
        placement[nt] = (seg[sl].clone(), local_pos[sl].clone())

    # ---- dense node-level tensors ----
    def grid_long(fill):
        return torch.full((B, n_max), fill, dtype=torch.long)

    node_type_id = grid_long(-1)
    pad_mask = torch.zeros(B, n_max, dtype=torch.bool)
    is_seed_g = torch.zeros(B, n_max, dtype=torch.bool)
    is_timed_g = torch.zeros(B, n_max, dtype=torch.bool)
    row_time_g = torch.zeros(B, n_max, dtype=torch.float64)
    n_id_g = grid_long(-1)

    keep = local_pos < n_max
    bb = seg[keep]
    ll = local_pos[keep]
    node_type_id[bb, ll] = ntype[keep]
    pad_mask[bb, ll] = True
    is_seed_g[bb, ll] = is_seed_flat[keep]
    is_timed_g[bb, ll] = timed[keep]
    row_time_g[bb, ll] = rowt[keep]
    n_id_g[bb, ll] = nid[keep]

    # ---- 1-hop relation matrix (per segment, dense) from edges ----
    # rel_mat[b, i, j] = metapath id of a *directed* 1-hop edge i<-j (j attends to i); 0 if none.
    rel_mat = torch.zeros(B, n_max, n_max, dtype=torch.long)      # stores per-relation metapath id
    fk_mat = torch.zeros(B, n_max, n_max, dtype=torch.long)
    adj = torch.zeros(B, n_max, n_max, dtype=torch.bool)

    def flat_index(nt: str, local_in_store: Tensor) -> Tensor:
        return local_in_store + type_slice[nt].start

    for (src, rel, dst) in bundle.edge_types:
        # process only forward relations; we add both attention directions ourselves, and the
        # canonical FK role is direction-independent (so rev_* would just double-write).
        if not is_forward_relation(rel):
            continue
        if (src, rel, dst) not in batch.edge_types:
            continue
        ei = batch[(src, rel, dst)].edge_index
        if ei.numel() == 0:
            continue
        # edge_index row 0 = src local, row 1 = dst local (PyG convention: src -> dst)
        gsrc = flat_index(src, ei[0])
        gdst = flat_index(dst, ei[1])
        bsrc, bdst = seg[gsrc], seg[gdst]
        # disjoint => bsrc == bdst; guard anyway
        same = bsrc == bdst
        gsrc, gdst, bs = gsrc[same], gdst[same], bsrc[same]
        ls, ld = local_pos[gsrc], local_pos[gdst]
        ok = (ls < n_max) & (ld < n_max)
        bs, ls, ld = bs[ok], ls[ok], ld[ok]
        mp = bundle.relation_metapath(rel)
        fk = bundle.relation_fk_role(rel)
        # make attention symmetric (either endpoint may attend to the other)
        for (a, b_) in ((ls, ld), (ld, ls)):
            adj[bs, a, b_] = True
            rel_mat[bs, a, b_] = mp
            fk_mat[bs, a, b_] = fk

    # ---- reachability (<=2 hop) + self ----
    eye = torch.eye(n_max, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    adj_f = adj.float()
    two_hop = (torch.bmm(adj_f, adj_f) > 0)
    valid_pair = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)   # both real
    reach = (adj | two_hop | eye) & valid_pair
    attend_mask = reach

    metapath_id = torch.full((B, n_max, n_max), MP_PAD, dtype=torch.long)
    metapath_id[two_hop & valid_pair] = MP_MULTIHOP        # 2-hop default
    onehop = adj & valid_pair
    metapath_id[onehop] = rel_mat[onehop]                  # 1-hop overrides with its relation
    diag = eye & valid_pair
    metapath_id[diag] = MP_SELF                            # self overrides
    metapath_id[~attend_mask] = MP_PAD

    fk_role_id = torch.zeros(B, n_max, n_max, dtype=torch.long)
    fk_role_id[onehop] = fk_mat[onehop]

    # ---- dimensionless time on attendable, both-timed pairs ----
    rt_i = row_time_g.unsqueeze(2)         # [B, N, 1]
    rt_j = row_time_g.unsqueeze(1)         # [B, 1, N]
    both_timed = is_timed_g.unsqueeze(2) & is_timed_g.unsqueeze(1)
    temporal_valid = both_timed & attend_mask & ~diag
    dt = torch.zeros(B, n_max, n_max, dtype=torch.float64)
    dt[temporal_valid] = (rt_i - rt_j).abs()[temporal_valid]

    t_ctx = torch.ones(B, dtype=torch.float64)
    tau = torch.zeros(B, n_max, n_max, dtype=torch.float64)
    for b in range(B):
        pair_gaps = dt[b][temporal_valid[b] & (dt[b] > 0)]
        # also fold in node recencies (seed_time - row_time): this keeps T_ctx a REAL timescale (so it
        # scales with a global clock rescale) even for subgraphs with no valid pairwise gap — otherwise
        # the fallback T_ctx=1.0 would make the node-time term log((seed-row)/1) non-invariant.
        rec = (seed_time[b] - row_time_g[b])[is_timed_g[b]]
        rec = rec[rec > 0]
        all_gaps = torch.cat([pair_gaps, rec])
        if all_gaps.numel() > 0:
            t_ctx[b] = torch.median(all_gaps)
        tv = temporal_valid[b] & (dt[b] > 0)
        tau[b][tv] = torch.log(dt[b][tv] / t_ctx[b])
    # pairs with dt==0 but both timed are not "temporal" (same instant) -> structural bucket
    temporal_valid = temporal_valid & (dt > 0)

    tf_dict = {nt: batch[nt].tf for nt in node_types}

    return GlossBatch(
        num_seeds=B, n_max=n_max,
        node_type_id=node_type_id, pad_mask=pad_mask, is_seed=is_seed_g, is_timed=is_timed_g,
        row_time=row_time_g, n_id=n_id_g, seed_time=seed_time, t_ctx=t_ctx,
        attend_mask=attend_mask, metapath_id=metapath_id, fk_role_id=fk_role_id,
        dt=dt, tau=tau, temporal_valid=temporal_valid,
        tf_dict=tf_dict, placement=placement,
    )
