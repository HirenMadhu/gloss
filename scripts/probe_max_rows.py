"""probe_max_rows.py — measure rows-per-seed (the row-graph axis R) per dataset/task/split.

``build_row_graph`` asserts ``max_rows_per_seed <= R`` and never clamps, so R has to be set from a
measurement, not a guess. ``MAX_ROWS = 160`` was measured on rel-f1 only (max 65 at fanout [12,12]);
rel-event blows through it (162 observed at validation), which killed every rel-event run of the
two-level arrays.

This bypasses ``to_cell_batch`` entirely — it counts rows straight off the sampled batch, so it can
report the true max even when that max would trip the assert.

    .venv/bin/python scripts/probe_max_rows.py --dataset rel-event
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows_per_seed(raw, bundle) -> list[int]:
    """Per-seed sampled-node counts, summed over node types — exactly the ``seg`` bincount that
    ``build_row_graph`` asserts on, but computed without going through the assert."""
    import torch

    segs = [raw[nt].batch.to(torch.long) for nt in bundle.node_types
            if nt in raw.node_types and raw[nt].num_nodes > 0]
    if not segs:
        return []
    seg = torch.cat(segs)
    return torch.bincount(seg, minlength=int(seg.max()) + 1).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-event")
    ap.add_argument("--tasks", nargs="*", default=None, help="default: all leaderboard entity tasks")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--batches", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-neighbors", type=int, nargs="*", default=[12, 12])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.eval.ablation import dataset_tasks

    tasks = args.tasks or dataset_tasks([args.dataset], ["leaderboard"])[args.dataset]
    bundle = build_gloss_graph(args.dataset)

    overall = 0
    for tname in tasks:
        task = get_task(args.dataset, tname, download=True)
        for split in args.splits:
            loader = make_loader(bundle, task, split, batch_size=args.batch_size,
                                 num_workers=0, shuffle=False, num_neighbors=args.num_neighbors)
            mx, tot, nb = 0, 0, 0
            for i, raw in enumerate(loader):
                if args.batches and i >= args.batches:
                    break
                counts = rows_per_seed(raw, bundle)
                if counts:
                    mx = max(mx, max(counts))
                    tot += len(counts)
                nb += 1
            overall = max(overall, mx)
            print(f"{args.dataset:12} {tname:18} {split:5}  seeds={tot:7d}  max_rows_per_seed={mx}",
                  flush=True)

    print(f"\n=== {args.dataset} @ num_neighbors={args.num_neighbors}: "
          f"OVERALL max rows/seed = {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
