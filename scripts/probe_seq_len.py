"""probe_seq_len.py — measure cells-per-seed demand against the fixed ``seq_len`` cap.

``to_cell_batch`` enumerates a seed's sampled rows into a FIXED ``[B, seq_len]`` cell axis and drops
the overflow with a bare ``if s >= seq_len: break`` — no counter, no warning. So a run can silently
train on a fraction of each neighbourhood and still look completely healthy.

``seq_len=512`` comes from a rel-f1 measurement (~353 cells/seed, per CLAUDE.md). Whether it holds on
rel-trial (15 tables) or rel-event (44.8M rows) was never measured. That is the same shape of mistake
as ``MAX_ROWS=160`` — a rel-f1 number asserted on every DB — which killed 57 jobs (amendments.md §9.1);
the difference is that ``max_rows`` asserted and this one truncates in silence.

Seed-row cells are emitted first, so truncation eats NEIGHBOUR cells: the relational context is
exactly what gets lost, which is the part MoRE is supposed to exploit.

    .venv/bin/python scripts/probe_seq_len.py --dataset rel-trial --batches 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cells_per_seed(raw, bundle, feat_cols: dict[str, int]) -> list[int]:
    """Per-seed DEMANDED cell count = sum over sampled rows of that table's feature-column count.

    Mirrors the enumeration in ``to_cell_batch`` (a node with no feature columns is skipped) but
    without the ``seq_len`` break, so it reports true demand rather than what survived.
    """
    import torch

    per_seed: dict[int, int] = {}
    for nt in bundle.node_types:
        if nt not in raw.node_types or raw[nt].num_nodes == 0:
            continue
        ncols = feat_cols.get(nt, 0)
        if not ncols:
            continue
        seg = raw[nt].batch.to(torch.long)
        counts = torch.bincount(seg)
        for b, n in enumerate(counts.tolist()):
            if n:
                per_seed[b] = per_seed.get(b, 0) + n * ncols
    return [per_seed.get(b, 0) for b in range(max(per_seed) + 1)] if per_seed else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--tasks", nargs="*", default=None, help="default: all leaderboard entity tasks")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--batches", type=int, default=20, help="0 = the whole split")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512, help="the cap to report overflow against")
    ap.add_argument("--num-neighbors", type=int, nargs="*", default=[12, 12])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.tasks import get_task

    from gloss.data.collate import feature_col_names
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.eval.ablation import dataset_tasks

    tasks = args.tasks or dataset_tasks([args.dataset], ["leaderboard"])[args.dataset]
    bundle = build_gloss_graph(args.dataset)

    print(f"# {args.dataset}  seq_len={args.seq_len}  fanout={args.num_neighbors}")
    for tname in tasks:
        task = get_task(args.dataset, tname, download=True)
        for split in args.splits:
            loader = make_loader(bundle, task, split, batch_size=args.batch_size,
                                 num_workers=0, shuffle=False, num_neighbors=args.num_neighbors)
            feat_cols: dict[str, int] = {}
            allc: list[int] = []
            for i, raw in enumerate(loader):
                if not feat_cols:
                    feat_cols = {nt: len(feature_col_names(raw[nt].tf))
                                 for nt in bundle.node_types if nt in raw.node_types}
                if args.batches and i >= args.batches:
                    break
                allc.extend(cells_per_seed(raw, bundle, feat_cols))
            if not allc:
                print(f"  {tname:20} {split:5}  no cells")
                continue
            allc.sort()
            n = len(allc)
            over = sum(c > args.seq_len for c in allc)
            # what fraction of all demanded cells the cap actually keeps
            kept = sum(min(c, args.seq_len) for c in allc) / max(sum(allc), 1)
            print(f"  {tname:20} {split:5}  seeds={n:6d}  median={allc[n // 2]:6d}  "
                  f"p90={allc[int(n * 0.9)]:6d}  max={allc[-1]:6d}  "
                  f"over_cap={over / n:6.1%}  cells_kept={kept:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
