"""run_ablation.py — the routing-signal ablation runner (SLURM array friendly).

One ``(dataset, task, signal, seed)`` config per process so a job array runs them in parallel;
``--aggregate`` reduces the per-config JSONs to per-(dataset, task) tables.

    .venv/bin/python scripts/run_ablation.py --list                        # array size
    .venv/bin/python scripts/run_ablation.py --index $SLURM_ARRAY_TASK_ID  # run one config (full training)
    .venv/bin/python scripts/run_ablation.py --aggregate                   # print the tables

Smoke one arm locally (rel-f1, hash encoder, tiny) before the GPU array:
    .venv/bin/python scripts/run_ablation.py --index 0 --datasets rel-f1 --seeds 1 \
        --encoder hash --epochs 1 --batch-size 16 --num-workers 0 --seq-len 256
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.eval import ablation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print array size and exit")
    ap.add_argument("--aggregate", action="store_true", help="reduce result JSONs to tables")
    ap.add_argument("--datasets", nargs="+", default=list(ablation.DEFAULT_DATASETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--encoder", default="qwen", help="'qwen' | 'hash' | registry label | HF id")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lambda-ortho", type=float, default=0.5)
    ap.add_argument("--signals", nargs="+", default=list(ablation.ROUTING_SIGNALS),
                    help="routing arms to run (default: all six); e.g. --signals signature hybrid")
    # architecture additions (S/C/P/H) — a whole array run carries one addition config into its out-dir;
    # the variant label (router + tags) keeps it distinct in --aggregate.
    ap.add_argument("--use-shared", action="store_true", help="S: always-on shared expert")
    ap.add_argument("--cosine", action="store_true", help="C: cosine/normalized router over learnable keys")
    ap.add_argument("--tau", type=float, default=0.3, help="cosine router temperature")
    ap.add_argument("--top-p", type=float, default=None, help="P: adaptive expert count (cumulative mass)")
    ap.add_argument("--hmoe", action="store_true", help="H: hierarchical two-level gate (HMoEFFN)")
    ap.add_argument("--n-groups", type=int, default=4, help="HMoE level-1 group count")
    ap.add_argument("--experts-per-group", type=int, default=2, help="HMoE experts per group")
    ap.add_argument("--k2", type=int, default=1, help="HMoE within-group top-k2")
    ap.add_argument("--baseline", default="dense",
                    help="reference arm for the Δ column in --aggregate (e.g. dense | signature | hybrid)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write/read result JSONs (default results/ablation)")
    ap.add_argument("--limit-train-batches", type=float, default=None,
                    help="cap train batches/epoch (int count, or float fraction in (0,1]) for big tasks")
    ap.add_argument("--limit-val-batches", type=float, default=None,
                    help="cap val batches/epoch for best-val monitoring (int count or fraction); TEST eval stays full")
    ap.add_argument("--no-test", action="store_true", help="skip the held-out test eval")
    ap.add_argument("--split", default="test", choices=["test", "val"], help="which split to tabulate")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out_dir = Path(args.out_dir) if args.out_dir else None
    signals = tuple(args.signals)
    if args.list:
        print(len(ablation.enumerate_configs(args.datasets, args.seeds, signals)))
        return 0
    if args.aggregate:
        print(ablation.format_table(ablation.load_records(out_dir), split=args.split,
                                    baseline=args.baseline))
        return 0
    if args.index is None:
        print("pass --index N (run one config), --list, or --aggregate")
        return 2

    d_text = 64 if args.encoder == "hash" else 2560
    ltb = args.limit_train_batches
    if ltb is not None and ltb >= 1:
        ltb = int(ltb)
    lvb = args.limit_val_batches
    if lvb is not None and lvb >= 1:
        lvb = int(lvb)
    rec = ablation.run_config(
        args.index, datasets=tuple(args.datasets), seeds=args.seeds, signals=signals,
        encoder=args.encoder, d_text=d_text, max_epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, seq_len=args.seq_len, num_experts=args.num_experts,
        k=args.k, lambda_ortho=args.lambda_ortho, test=not args.no_test,
        use_shared=args.use_shared, cosine=args.cosine, tau=args.tau, top_p=args.top_p,
        hmoe=args.hmoe, n_groups=args.n_groups, experts_per_group=args.experts_per_group, k2=args.k2,
        out_dir=out_dir, limit_train_batches=ltb, limit_val_batches=lvb,
    )
    print({kk: rec.get(kk) for kk in ("dataset", "task", "signal", "variant", "seed", "task_type")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
