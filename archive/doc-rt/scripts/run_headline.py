"""run_headline.py — the four-regime DOC-RT headline gate (full / null / shuffled / name_only).

One ``(regime, seed)`` config per process so a SLURM job array runs them in parallel; ``--aggregate``
reduces the per-config JSONs to the headline table.

    .venv/bin/python scripts/run_headline.py --list                       # array size (= seeds x 4)
    .venv/bin/python scripts/run_headline.py --index $SLURM_ARRAY_TASK_ID  # run one config (full training)
    .venv/bin/python scripts/run_headline.py --aggregate                  # print the headline table

Smoke one arm locally (hash encoder, tiny) before the real GPU array:
    .venv/bin/python scripts/run_headline.py --index 0 --seeds 1 --encoder hash --epochs 1 \
        --batch-size 64 --num-workers 0
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402

log = get_logger("gloss.headline")


def _model_kwargs(cfg) -> dict:
    m = cfg.model
    return dict(
        d_model=int(m.d_model), n_blocks=int(m.n_blocks), n_heads=int(m.n_heads),
        d_ff=int(m.get("d_ff", 4 * int(m.d_model))),
        enc_channels=int(m.get("enc_channels", int(m.d_model))),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print array size and exit")
    ap.add_argument("--aggregate", action="store_true", help="reduce result JSONs to the headline table")
    ap.add_argument("--encoder", default="qwen",
                    help="'qwen' | 'hash' | a registry label (e.g. 'harrier') | a raw HF model id")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--test", action="store_true",
                    help="also score the held-out RelBench TEST split (results/headline_test*/)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from gloss.eval.ablation import (
        enumerate_configs,
        format_table,
        format_test_table,
        load_records,
        results_dir,
        run_config,
    )

    out_dir = results_dir(args.encoder, args.test)
    if args.list:
        print(len(enumerate_configs(args.seeds)))
        return 0
    if args.aggregate:
        recs = load_records(out_dir)
        print(format_test_table(recs) if args.test else format_table(recs))
        return 0
    if args.index is None:
        log.error("pass --index N, --list, or --aggregate")
        return 2

    cfg = load_config(args.config)
    d_text = 64 if args.encoder == "hash" else 2560
    rec = run_config(
        args.index,
        dataset=str(cfg.data.dataset), task_name=str(cfg.data.task), seeds=args.seeds,
        encoder=args.encoder, d_text=d_text, model_kwargs=_model_kwargs(cfg),
        num_neighbors=list(cfg.data.sampler.num_neighbors),
        seq_len=int(cfg.data.collate.seq_len), max_fk=int(cfg.data.collate.max_fk),
        batch_size=args.batch_size or int(cfg.train.batch_size),
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        max_epochs=args.epochs or int(cfg.train.max_epochs), num_workers=args.num_workers,
        sim_threshold=float(cfg.docs.grounding.sim_threshold), test=args.test, out_dir=out_dir,
    )
    keys = ("regime", "seed", "ap", "auroc", "test_ap", "test_auroc") if args.test \
        else ("regime", "seed", "ap", "auroc")
    log.info("config[%d] = %s", args.index, {k: rec.get(k) for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
