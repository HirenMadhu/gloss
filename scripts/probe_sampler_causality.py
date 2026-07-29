"""probe_sampler_causality.py — resolve changes.md §9.2 empirically.

Question: does ``NeighborSampler(time_attr="time", temporal_strategy="last", disjoint=True)``
filter a sampled row's timestamp against its **parent** (the node it was expanded from) or only
against the **seed**?

This decides whether Phase 1's ``t4`` (path-causal masking) is a *mask* change or a *sampler*
change. If filtering is seed-only, paths inside a sampled subgraph are not internally
time-ordered, and masking cannot create an ordering the sample never had.

Method: sample real minibatches, reconstruct each node's hop from ``num_sampled_nodes``, and for
every sampled edge joining hop ``h`` to hop ``h+1`` compare the two timestamps. A single
``t_deeper > t_shallower`` with both ``<= t_seed`` settles it: seed-only.

    .venv/bin/python scripts/probe_sampler_causality.py --datasets rel-f1 rel-trial rel-event
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hops_by_bfs(batch, entity_table: str, n_seeds: int):
    """Per-node hop distance from the seed, by BFS over the sampled subgraph (undirected).

    relbench's ``CustomNodeLoader`` does not forward PyG's ``num_sampled_nodes``, so hop is
    recovered from the edges themselves. Disjoint sampling keeps segments disconnected, so one
    multi-source BFS over the whole batch assigns every node its distance from *its own* seed.

    This is the prototype of changes.md P0.3; returns ``{node_type: LongTensor[n]}``, -1 = unreached.
    """
    import torch

    offset, total = {}, 0
    for nt in batch.node_types:
        offset[nt] = total
        total += int(batch[nt].num_nodes)

    src, dst = [], []
    for et in batch.edge_types:
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        s = ei[0] + offset[et[0]]
        d = ei[1] + offset[et[2]]
        src.append(torch.cat([s, d]))
        dst.append(torch.cat([d, s]))
    if not src:
        return {nt: torch.full((int(batch[nt].num_nodes),), -1, dtype=torch.long)
                for nt in batch.node_types}
    src = torch.cat(src)
    dst = torch.cat(dst)

    hop = torch.full((total,), -1, dtype=torch.long)
    frontier = torch.zeros(total, dtype=torch.bool)
    frontier[offset[entity_table]:offset[entity_table] + n_seeds] = True
    hop[frontier] = 0

    h = 0
    while bool(frontier.any()):
        h += 1
        reach = torch.zeros(total, dtype=torch.bool)
        reach[dst[frontier[src]]] = True
        frontier = reach & (hop < 0)
        hop[frontier] = h

    return {nt: hop[offset[nt]:offset[nt] + int(batch[nt].num_nodes)] for nt in batch.node_types}


def probe(dataset: str, *, batches: int, batch_size: int, num_neighbors: list[int]) -> None:
    import torch

    from gloss.data.collate import _seed_time_per_segment
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.eval.ablation import dataset_tasks

    bundle = build_gloss_graph(dataset)
    tasks = dataset_tasks([dataset], ["leaderboard"]).get(dataset, [])
    if not tasks:
        tasks = dataset_tasks([dataset]).get(dataset, [])
    task_name = tasks[0]

    from relbench.tasks import get_task

    task = get_task(dataset, task_name, download=True)
    loader = make_loader(bundle, task, "train", batch_size=batch_size,
                         num_neighbors=num_neighbors, shuffle=False)

    n_pairs = 0            # hop-adjacent edges with both endpoints timed
    n_violate_parent = 0   # deeper node strictly LATER than the node it hangs off
    n_violate_seed = 0     # any node later than its seed (would be a real leak)
    worst = 0.0
    max_hop = 0            # sanity: must not exceed len(num_neighbors)
    n_unreached = 0        # nodes BFS never reached (should be 0)
    n_strictly_earlier = 0 # deeper row strictly EARLIER than its parent
    n_equal = 0            # deeper row shares its parent's timestamp

    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        # disjoint sampling: every node carries `batch` = its seed segment, and the seed times
        # live on the entity store only, so recover them per segment exactly as collate.py does.
        ent = batch[task.entity_table]
        n_seg = int(ent.batch.max()) + 1
        seg_seed_time = _seed_time_per_segment(ent, n_seg)
        n_seeds = int(ent.seed_time.numel())

        hop = hops_by_bfs(batch, task.entity_table, n_seeds)
        time, seedt = {}, {}
        for nt in batch.node_types:
            store = batch[nt]
            time[nt] = store.time.to(torch.float64) if "time" in store else None
            seedt[nt] = (seg_seed_time[store.batch.to(torch.long)]
                         if "batch" in store else None)
            h = hop[nt]
            if bool((h >= 0).any()):
                max_hop = max(max_hop, int(h[h >= 0].max()))
            n_unreached += int((h < 0).sum())

        for et in batch.edge_types:
            src_t, _, dst_t = et
            ei = batch[et].edge_index
            if ei.numel() == 0:
                continue
            ts, td = time[src_t], time[dst_t]
            if ts is None or td is None:
                continue
            s, d = ei[0], ei[1]
            hs, hd = hop[src_t][s], hop[dst_t][d]
            timed = (ts[s] > 0) & (td[d] > 0) & (hs >= 0) & (hd >= 0)

            # deeper endpoint was expanded FROM the shallower one
            deeper_is_dst = timed & (hd == hs + 1)
            deeper_is_src = timed & (hs == hd + 1)

            for sel, t_deep, t_shal in ((deeper_is_dst, td[d], ts[s]),
                                        (deeper_is_src, ts[s], td[d])):
                if not bool(sel.any()):
                    continue
                dt = (t_deep - t_shal)[sel]
                n_pairs += int(sel.sum())
                bad = dt > 0
                n_violate_parent += int(bad.sum())
                # sign breakdown: if every pair is dt == 0 the test is UNINFORMATIVE — the rows
                # simply share a timestamp, which proves nothing about the sampler's policy.
                n_strictly_earlier += int((dt < 0).sum())
                n_equal += int((dt == 0).sum())
                if bool(bad.any()):
                    worst = max(worst, float(dt[bad].max()))

        for nt in batch.node_types:
            t, st = time[nt], seedt[nt]
            if t is None or st is None:
                continue
            m = t > 0
            n_violate_seed += int((t[m] > st[m]).sum())

    print(f"=== {dataset} / {task_name}  ({batches} batches x {batch_size} seeds, "
          f"fanout={num_neighbors})")
    pct = lambda n: 100.0 * n / max(n_pairs, 1)
    print(f"  hop-adjacent timed edge pairs   {n_pairs}")
    print(f"  deeper row EARLIER than parent  {n_strictly_earlier}  ({pct(n_strictly_earlier):.2f}%)")
    print(f"  deeper row EQUAL to parent      {n_equal}  ({pct(n_equal):.2f}%)")
    print(f"  deeper row LATER than its parent {n_violate_parent}"
          f"   ({pct(n_violate_parent):.2f}%)"
          f"   worst +{worst / 86400.0:.1f} days")
    print(f"  any row later than its SEED      {n_violate_seed}   (must be 0 — leakage assert)")
    print(f"  max BFS hop {max_hop} (fanout depth {len(num_neighbors)}), unreached nodes {n_unreached}")
    if n_violate_parent > 0:
        verdict = "SEED-ONLY -> paths are NOT internally time-ordered; t4 is a SAMPLER change"
    elif n_strictly_earlier == 0:
        verdict = ("UNINFORMATIVE -> every hop-adjacent pair shares a timestamp; this dataset "
                   "cannot distinguish the two policies")
    else:
        verdict = "PARENT-FILTERED -> paths already time-ordered; t4 is a pure MASK change"
    print(f"  VERDICT: {verdict}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-neighbors", type=int, nargs="+", default=[12, 12])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    for ds in args.datasets:
        probe(ds, batches=args.batches, batch_size=args.batch_size,
              num_neighbors=args.num_neighbors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
