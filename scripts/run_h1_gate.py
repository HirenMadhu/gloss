"""run_h1_gate.py — Phase-5 GATE 1 (efficient: graph + groundings built ONCE, reused across configs).

H1: same HALOS under docs.regime in {full, shuffled_spans, null} -> val AP/AUROC.
H2: doc-generated geometry vs free-learned per-relation bias (regime=full).

    .venv/bin/python scripts/run_h1_gate.py --seeds 2 --epochs 3 --limit-train-batches 60
"""
from __future__ import annotations

import argparse
import statistics as stats
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402

log = get_logger("gloss.h1_gate")


def _agg(rows):
    f = lambda xs: (stats.mean(xs), (stats.pstdev(xs) if len(xs) > 1 else 0.0))
    return f([r[0] for r in rows]), f([r[1] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "hash"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=int, default=60)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    dataset, task_name = str(cfg.data.dataset), str(cfg.data.task)
    d_text = 2560 if args.encoder == "qwen" else 64

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.train.finetune import docs_for_regime, train_prebuilt

    log.info("building graph + groundings once ...")
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    # build each grounding (+doc table) ONCE and reuse across seeds/modes
    groundings = {r: docs_for_regime(bundle, dataset, r, encoder=args.encoder, d_text=d_text)
                  for r in ("full", "shuffled_spans", "null")}

    def run(regime, geometry_mode, seed):
        mk = dict(d_model=args.d_model, n_heads=int(cfg.model.n_heads), n_layers=args.n_layers,
                  d_text=d_text, n_freq=int(cfg.model.geometry.get("n_freq", 16)),
                  sigma_floor=float(cfg.model.geometry.sigma_floor), geometry_mode=geometry_mode)
        g, doc_mp = groundings[regime]
        _m, metrics = train_prebuilt(
            bundle, task, g, doc_mp, model_kwargs=mk,
            num_neighbors=list(cfg.data.sampler.num_neighbors), batch_size=256,
            max_epochs=args.epochs, limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches, seed=seed,
        )
        return metrics.get("val/ap", float("nan")), metrics.get("val/auroc", float("nan"))

    print("\n=== H1: doc regime (geometry_mode=generated) ===")
    print(f"{'regime':16s} {'AP mean±std':>20s} {'AUROC mean±std':>20s}")
    for regime in ("full", "shuffled_spans", "null"):
        (apm, aps), (aum, aus) = _agg([run(regime, "generated", s) for s in range(args.seeds)])
        print(f"{regime:16s} {apm:9.4f} ± {aps:.4f}   {aum:9.4f} ± {aus:.4f}")

    print("\n=== H2: geometry mode (regime=full) ===")
    print(f"{'mode':16s} {'AP mean±std':>20s} {'AUROC mean±std':>20s}")
    for mode in ("generated", "free_learned"):
        (apm, aps), (aum, aus) = _agg([run("full", mode, s) for s in range(args.seeds)])
        print(f"{mode:16s} {apm:9.4f} ± {aps:.4f}   {aum:9.4f} ± {aus:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
