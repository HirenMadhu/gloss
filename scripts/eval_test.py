"""eval_test.py — train HALOS (full docs, generated geometry) and evaluate on the RelBench TEST set,
for a leaderboard-comparable number (vs GelGT 0.7608 +/- 0.0175 on rel-f1 driver-dnf).

One seed per process (SLURM array) -> results/test_eval/<idx>.json; `--aggregate` -> mean +/- std.

  python scripts/eval_test.py --index $SLURM_ARRAY_TASK_ID --seeds 5
  python scripts/eval_test.py --aggregate
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

log = get_logger("gloss.eval_test")
OUT = Path(__file__).resolve().parents[1] / "results" / "test_eval"


def run_one(args) -> int:
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    dataset, task_name = str(cfg.data.dataset), str(cfg.data.task)
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.test_eval import evaluate_test
    from gloss.train.finetune import docs_for_regime, train_prebuilt

    feat = bool(args.doc_feature)
    geo = bool(args.doc_geometry)
    log.info("seed=%d regime=%s fusion(feature=%s,geometry=%s) epochs=%d", args.index, args.regime, feat, geo, args.epochs)
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g, doc_mp = docs_for_regime(bundle, dataset, args.regime, encoder=args.encoder, d_text=2560)
    mk = dict(d_model=args.d_model, n_heads=int(cfg.model.n_heads), n_layers=args.n_layers, d_text=2560,
              n_freq=int(cfg.model.geometry.get("n_freq", 16)), sigma_floor=float(cfg.model.geometry.sigma_floor),
              doc_cross_attn_feature=feat, doc_cross_attn_geometry=geo)
    module, val = train_prebuilt(bundle, task, g, doc_mp, model_kwargs=mk,
                                 num_neighbors=list(cfg.data.sampler.num_neighbors), batch_size=args.batch_size,
                                 max_epochs=args.epochs, seed=args.index, num_workers=args.num_workers)
    module.eval()
    test = evaluate_test(lambda gb: module(gb), bundle, task,
                         num_neighbors=list(cfg.data.sampler.num_neighbors),
                         batch_size=args.batch_size, device=module.device)
    rec = {"seed": args.index, "regime": args.regime, "feature": feat, "geometry": geo,
           "test_roc_auc": test.get("roc_auc"), "test_ap": test.get("average_precision"),
           "val_auroc": val.get("val/auroc"), "val_ap": val.get("val/ap")}
    (OUT / f"{args.index:03d}.json").write_text(json.dumps(rec))
    log.info("seed %d done: TEST roc_auc=%.4f ap=%.4f | val_auroc=%.4f",
             args.index, rec["test_roc_auc"], rec["test_ap"], rec["val_auroc"] or float("nan"))
    return 0


def aggregate() -> int:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not recs:
        log.error("no results in %s", OUT)
        return 1

    def agg(key):
        xs = [r[key] for r in recs if r.get(key) is not None]
        return (stats.mean(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)

    tm, ts = agg("test_roc_auc")
    am, as_ = agg("test_ap")
    vm, vs = agg("val_auroc")
    print(f"\nHALOS rel-f1 driver-dnf — {len(recs)} seeds")
    print(f"  TEST roc_auc : {tm:.4f} +/- {ts:.4f}   (GelGT table: 0.7608 +/- 0.0175)")
    print(f"  TEST AP      : {am:.4f} +/- {as_:.4f}")
    print(f"  (val auroc   : {vm:.4f} +/- {vs:.4f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--doc-feature", action="store_true")
    ap.add_argument("--doc-geometry", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    if args.list:
        print(args.seeds)
        return 0
    if args.aggregate:
        return aggregate()
    if args.index is not None:
        return run_one(args)
    log.error("pass --index, --list, or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
