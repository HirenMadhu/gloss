"""run_horizon.py — multi-horizon study: how does TEST performance decay predicting k=1..10
timestamps ahead, for BOTH substrates (single-table SetJoin and multi-table MoRE)?

    .venv/bin/python scripts/run_horizon.py --list                    # grid size (54)
    .venv/bin/python scripts/run_horizon.py --index N [--out-dir D]   # one (dataset, task, model, seed)
    .venv/bin/python scripts/run_horizon.py --plot  [--out-dir D]     # aggregate + save the curves PNG
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="aggregate records + save horizon_curves.png")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--encoder", default="harrier")
    ap.add_argument("--fanout", type=int, default=64)
    ap.add_argument("--wide-len", type=int, default=128)
    ap.add_argument("--set-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128, help="setjoin batch (more uses --more-batch-size)")
    ap.add_argument("--more-batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    ap.add_argument("--out-dir", default="results/horizon")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from gloss.setjoin.horizon import horizon_grid, plot_horizon_curves, run_config_horizon

    out_dir = Path(args.out_dir)
    if args.list:
        print(len(horizon_grid(seeds=args.seeds)))
        return 0
    if args.index is not None:
        run_config_horizon(
            args.index, seeds=args.seeds, encoder=args.encoder, fanout=args.fanout,
            wide_len=args.wide_len, set_size=args.set_size, batch_size=args.batch_size,
            more_batch_size=args.more_batch_size, max_epochs=args.epochs,
            num_workers=args.num_workers, out_dir=out_dir,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        return 0
    if args.plot:
        records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_horizon_*.json"))]
        if not records:
            print(f"no horizon records in {out_dir}")
            return 1
        png = plot_horizon_curves(records, out_dir / "horizon_curves.png")
        print(f"{len(records)} records -> {png}")
        return 0
    print("pass --list, --index N, or --plot")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
