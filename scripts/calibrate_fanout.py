"""calibrate_fanout.py — pick `num_neighbors` per database so sequences are actually full.

    source scripts/env.sh
    .venv/bin/python scripts/calibrate_fanout.py --datasets rel-f1 rel-trial --seq-len 1024

At the finetune-era default ``num_neighbors=[12, 12]`` a sampled sequence is **6-20% full** at
``seq_len=1024`` on every database measured (61-205 cells/seed; rel-f1's widest seed table is the
outlier at ~58%). Raising ``seq_len`` without raising the fanout just buys padding — RT fills its
1024-cell context with ``max_bfs_width=256``, not with a bigger cap.

So the knob is the fanout, and the right value differs per database because it depends on the schema's
branching factor. This measures occupancy and cost across a few candidates and prints the smallest
fanout that fills the target band, which is what the production launcher should then be given.

It also reports **cells/second**, because a richer fanout is not free: `to_cell_batch` is the training
bottleneck (121-353 ms/batch against the sampler's 50-60 ms) and its cost scales with cells/seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CANDIDATES = [[6, 6], [12, 12], [24, 24], [48, 48], [96, 96], [128, 128]]


def main() -> int:
    args = _argparser().parse_args()
    warnings.filterwarnings("ignore")

    import torch

    from gloss.data.collate import to_cell_batch
    from gloss.data.masking import build_column_target_spec, maskable_cells
    from gloss.data.pretrain_loader import build_pretrain_stream
    from gloss.data.graph import build_gloss_graph
    from gloss.utils.paths import graph_cache_dir

    out: dict = {}
    for ds in args.datasets:
        cache = graph_cache_dir(ds, args.text_encoder)
        if not cache.exists():
            print(f"[{ds}] no graph cache at {cache}; skipping", flush=True)
            continue
        bundle = build_gloss_graph(ds, cache_dir=str(cache), text_encoder=args.text_encoder)
        spec = build_column_target_spec(bundle)
        rows = []
        print(f"\n=== {ds}  (seq_len={args.seq_len}, target {args.lo:.0%}-{args.hi:.0%} full)",
              flush=True)
        print(f"{'fanout':>12} {'cells/seed':>11} {'occupancy':>10} {'R':>5} "
              f"{'maskable':>9} {'trunc':>6} {'s/batch':>8} {'cells/s':>9}", flush=True)
        for fan in args.fanouts or CANDIDATES:
            try:
                stream = build_pretrain_stream(bundle, spec, split="train", steps=args.batches,
                                               batch_size=args.batch_size, num_neighbors=fan,
                                               seed=0)
                cells = occ = maskn = rmax = 0.0
                trunc = 0
                t0 = time.time()
                n = 0
                for key, raw in stream:
                    cb = to_cell_batch(raw, bundle, key, seq_len=args.seq_len, max_fk=5)
                    real = int((~cb.is_padding).sum())
                    cells += real / cb.num_seeds
                    occ += real / (cb.num_seeds * cb.seq_len)
                    maskn += int(maskable_cells(cb, spec).sum()) / cb.num_seeds
                    rmax = max(rmax, cb.max_rows)
                    # a seed that filled every slot almost certainly lost cells to the cap
                    trunc += int(((~cb.is_padding).sum(1) == cb.seq_len).sum())
                    n += 1
                dt = time.time() - t0
                cells, occ, maskn = cells / n, occ / n, maskn / n
                rate = cells * args.batch_size * n / max(dt, 1e-9)
                rows.append(dict(fanout=fan, cells_per_seed=cells, occupancy=occ, max_rows=int(rmax),
                                 maskable_per_seed=maskn, truncated=trunc,
                                 sec_per_batch=dt / n, cells_per_s=rate))
                print(f"{str(fan):>12} {cells:11.0f} {occ:9.1%} {int(rmax):5d} "
                      f"{maskn:9.0f} {trunc:6d} {dt/n:8.2f} {rate:9.0f}", flush=True)
                if occ > args.hi:
                    break                       # anything larger only truncates harder
            except Exception as e:               # OOM or a sampler limit: report, do not abort
                print(f"{str(fan):>12}  FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                break
        # Preference order. Truncation is the thing to avoid — a truncated seed silently loses cells,
        # and seed-row cells are emitted first only so THEY survive, not the rest. So: in-band and
        # clean, else in-band with the least truncation, else the fullest sequence that has not
        # overshot the ceiling. Picking plain max-occupancy would choose the arm that truncates
        # hardest, which is the opposite of the intent.
        in_band = [r for r in rows if args.lo <= r["occupancy"] <= args.hi]
        clean = [r for r in in_band if not r["truncated"]]
        if clean:
            pick = clean[0]
        elif in_band:
            pick = min(in_band, key=lambda r: r["truncated"])
        elif rows:
            under = [r for r in rows if r["occupancy"] <= args.hi]
            pick = max(under or rows, key=lambda r: r["occupancy"])
        else:
            pick = None
        if pick:
            print(f"  -> recommended --num-neighbors {' '.join(map(str, pick['fanout']))} "
                  f"({pick['occupancy']:.0%} full, {pick['cells_per_s']:.0f} cells/s)", flush=True)
        out[ds] = {"rows": rows, "recommended": pick["fanout"] if pick else None}

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


def _argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1"])
    ap.add_argument("--text-encoder", default="minilm")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--fanouts", type=int, nargs="+", action="append", default=None,
                    help="repeatable, e.g. --fanouts 12 12 --fanouts 32 32")
    ap.add_argument("--lo", type=float, default=0.60, help="target occupancy floor")
    ap.add_argument("--hi", type=float, default=0.90, help="target occupancy ceiling")
    ap.add_argument("--out", default=None)
    return ap


if __name__ == "__main__":
    raise SystemExit(main())
