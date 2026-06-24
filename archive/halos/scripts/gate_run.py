"""gate_run.py — one GATE-1 config per process (for a SLURM job array), + an aggregation mode.

Configs are independent (regime x geometry_mode x seed), so a job array runs them in parallel across
GPUs; each writes results/gate/<index>.json; `--aggregate` reduces them to the H1/H2 tables.

    python scripts/gate_run.py --list                       # how many array tasks
    python scripts/gate_run.py --index $SLURM_ARRAY_TASK_ID  # run one config (full training)
    python scripts/gate_run.py --aggregate                  # print H1/H2 tables from the JSONs
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402

log = get_logger("gloss.gate")
OUT = Path(__file__).resolve().parents[1] / "results" / "gate"


def enumerate_configs(seeds: int):
    """Unique (regime, geometry_mode, seed). H1 = 3 regimes x generated; H2 adds full x free_learned."""
    cfgs = []
    for s in range(seeds):
        for regime in ("full", "shuffled_spans", "null"):
            cfgs.append({"regime": regime, "geometry_mode": "generated", "seed": s})
        cfgs.append({"regime": "full", "geometry_mode": "free_learned", "seed": s})
    return cfgs


def run_one(args) -> int:
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    dataset, task_name = str(cfg.data.dataset), str(cfg.data.task)
    configs = enumerate_configs(args.seeds)
    c = configs[args.index]
    OUT.mkdir(parents=True, exist_ok=True)

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.train.finetune import docs_for_regime, train_prebuilt

    log.info("config[%d] = %s | epochs=%d d_model=%d n_layers=%d", args.index, c, args.epochs,
             args.d_model, args.n_layers)
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g, doc_mp = docs_for_regime(bundle, dataset, c["regime"], encoder=args.encoder, d_text=2560)
    mk = dict(d_model=args.d_model, n_heads=int(cfg.model.n_heads), n_layers=args.n_layers,
              d_text=2560, n_freq=int(cfg.model.geometry.get("n_freq", 16)),
              sigma_floor=float(cfg.model.geometry.sigma_floor), geometry_mode=c["geometry_mode"])
    _m, metrics = train_prebuilt(
        bundle, task, g, doc_mp, model_kwargs=mk,
        num_neighbors=list(cfg.data.sampler.num_neighbors), batch_size=args.batch_size,
        max_epochs=args.epochs, seed=c["seed"], num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
    )
    rec = {**c, "ap": metrics.get("val/ap"), "auroc": metrics.get("val/auroc"),
           "logloss": metrics.get("val/logloss")}
    (OUT / f"{args.index:03d}.json").write_text(json.dumps(rec))
    log.info("config[%d] done: %s", args.index, {k: rec[k] for k in ("ap", "auroc")})
    return 0


def aggregate() -> int:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not recs:
        log.error("no results in %s", OUT)
        return 1
    print(f"\n{len(recs)} runs collected.")

    def agg(rows, key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return (stats.mean(xs), (stats.pstdev(xs) if len(xs) > 1 else 0.0)) if xs else (float("nan"), 0.0)

    print("\n=== H1: doc regime (geometry_mode=generated) ===")
    print(f"{'regime':16s} {'AP mean±std':>20s} {'AUROC mean±std':>20s} {'n':>4s}")
    for regime in ("full", "shuffled_spans", "null"):
        rows = [r for r in recs if r["regime"] == regime and r["geometry_mode"] == "generated"]
        (apm, aps), (aum, aus) = agg(rows, "ap"), agg(rows, "auroc")
        print(f"{regime:16s} {apm:9.4f} ± {aps:.4f}   {aum:9.4f} ± {aus:.4f} {len(rows):4d}")

    print("\n=== H2: geometry mode (regime=full) ===")
    print(f"{'mode':16s} {'AP mean±std':>20s} {'AUROC mean±std':>20s} {'n':>4s}")
    for mode in ("generated", "free_learned"):
        rows = [r for r in recs if r["regime"] == "full" and r["geometry_mode"] == mode]
        (apm, aps), (aum, aus) = agg(rows, "ap"), agg(rows, "auroc")
        print(f"{mode:16s} {apm:9.4f} ± {aps:.4f}   {aum:9.4f} ± {aus:.4f} {len(rows):4d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    args = ap.parse_args()
    if args.list:
        print(len(enumerate_configs(args.seeds)))
        return 0
    if args.aggregate:
        return aggregate()
    if args.index is not None:
        return run_one(args)
    log.error("pass --index N, --list, or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
