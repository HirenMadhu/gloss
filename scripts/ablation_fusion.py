"""ablation_fusion.py — does the doc cross-attention architecture help/hurt vs the FiLM baseline?

Configs: fusion in {film, feature, geometry, both} under regime=full + a null-doc floor (cross-attn is
inert under null, so null is run once with film). One config per process (SLURM array) -> JSON;
`--aggregate` reduces to a table.

  python scripts/ablation_fusion.py --list
  python scripts/ablation_fusion.py --index $SLURM_ARRAY_TASK_ID
  python scripts/ablation_fusion.py --aggregate

Honest caveat: rel-f1 is doc-null (GATE 1), so the doc effect should be ~0 for ALL fusions; this ablation
mainly answers 'does the cross-attention architecture stay competitive / not hurt?'. The doc *advantage*
needs a doc-load-bearing setting (transfer / synthetic), which is the next build.
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

log = get_logger("gloss.ablation")
OUT = Path(__file__).resolve().parents[1] / "results" / "ablation_fusion"
FUSION = {"film": (False, False), "feature": (True, False), "geometry": (False, True), "both": (True, True)}


def enumerate_configs(seeds: int):
    cfgs = []
    for s in range(seeds):
        for fusion in ("film", "feature", "geometry", "both"):
            cfgs.append({"regime": "full", "fusion": fusion, "seed": s})
        cfgs.append({"regime": "null", "fusion": "film", "seed": s})   # doc floor (cross-attn inert)
    return cfgs


def run_one(args) -> int:
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    dataset, task_name = str(cfg.data.dataset), str(cfg.data.task)
    c = enumerate_configs(args.seeds)[args.index]
    OUT.mkdir(parents=True, exist_ok=True)
    feat, geo = FUSION[c["fusion"]]

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.train.finetune import docs_for_regime, train_prebuilt

    log.info("config[%d] = %s -> feature=%s geometry=%s", args.index, c, feat, geo)
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g, doc_mp = docs_for_regime(bundle, dataset, c["regime"], encoder=args.encoder, d_text=2560)
    mk = dict(d_model=args.d_model, n_heads=int(cfg.model.n_heads), n_layers=args.n_layers, d_text=2560,
              n_freq=int(cfg.model.geometry.get("n_freq", 16)), sigma_floor=float(cfg.model.geometry.sigma_floor),
              doc_cross_attn_feature=feat, doc_cross_attn_geometry=geo)
    _m, metrics = train_prebuilt(bundle, task, g, doc_mp, model_kwargs=mk,
                                 num_neighbors=list(cfg.data.sampler.num_neighbors), batch_size=args.batch_size,
                                 max_epochs=args.epochs, seed=c["seed"], num_workers=args.num_workers)
    rec = {**c, "ap": metrics.get("val/ap"), "auroc": metrics.get("val/auroc"), "logloss": metrics.get("val/logloss")}
    (OUT / f"{args.index:03d}.json").write_text(json.dumps(rec))
    log.info("config[%d] done: %s", args.index, {k: rec[k] for k in ("ap", "auroc")})
    return 0


def aggregate() -> int:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not recs:
        log.error("no results in %s", OUT)
        return 1
    print(f"\n{len(recs)} runs.\n{'config':22s} {'AP mean±std':>20s} {'AUROC mean±std':>20s} {'n':>3s}")

    def agg(rows, k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return (stats.mean(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)

    def show(label, rows):
        (apm, aps), (aum, aus) = agg(rows, "ap"), agg(rows, "auroc")
        print(f"{label:22s} {apm:9.4f} ± {aps:.4f}   {aum:9.4f} ± {aus:.4f} {len(rows):3d}")

    for fusion in ("film", "feature", "geometry", "both"):
        show(f"full / {fusion}", [r for r in recs if r["regime"] == "full" and r["fusion"] == fusion])
    show("null / film (floor)", [r for r in recs if r["regime"] == "null"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    if args.list:
        print(len(enumerate_configs(args.seeds)))
        return 0
    if args.aggregate:
        return aggregate()
    if args.index is not None:
        return run_one(args)
    log.error("pass --index, --list, or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
