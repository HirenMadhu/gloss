"""run_setjoin_grid.py — SetJoin backbone sweep (phase S10), SLURM-array friendly.

    .venv/bin/python scripts/run_setjoin_grid.py --list                        # array size (486)
    .venv/bin/python scripts/run_setjoin_grid.py --index $SLURM_ARRAY_TASK_ID  # one (config, task, seed)
    .venv/bin/python scripts/run_setjoin_grid.py --aggregate                   # winners vs RT / MoRE-best

Fixed knobs (not swept): route_on=signature, experts=4, k=2, d_sig=64, heads=4, n_pma=4, fanout=64,
wide_len=128, set_size=128, lr=3e-4, wd=0.01, lambda_ortho=0.5, encoder=harrier.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print array size and exit")
    ap.add_argument("--aggregate", action="store_true", help="winners vs RT / MoRE grid best")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    ap.add_argument("--out-dir", default="results/setjoin_grid")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from gloss.setjoin.grid import aggregate_grid, jobs, run_index

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    if args.list:
        print(len(jobs(args.seeds)))
        return 0
    if args.aggregate:
        print(aggregate_grid(out_dir))
        return 0
    if args.index is None:
        print("pass --index N, --list, or --aggregate")
        return 2
    rec = run_index(args.index, seeds=args.seeds, epochs=args.epochs,
                    num_workers=args.num_workers, out_dir=out_dir,
                    limit_train_batches=args.limit_train_batches,
                    limit_val_batches=args.limit_val_batches)
    print({k: rec.get(k) for k in ("config_idx", "dataset", "task", "seed", "task_type",
                                   "batch_size", "n_params", "test_roc_auc", "test_nmae")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
