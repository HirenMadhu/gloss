"""run_h1_gate.py — Phase-5 GATE 1.

H1: train the SAME HALOS under docs.regime in {full, shuffled_spans, null} and compare val AP/AUROC.
H2: doc-generated geometry vs a free-learned per-relation bias (matched-ish), under the full regime.

Reduced defaults (few seeds, limited batches) so it runs under CPU contention; scale --seeds/--epochs
for the real gate. Reports mean +/- std; the decision + caveats go into PROGRESS.md by hand.

    .venv/bin/python scripts/run_h1_gate.py --seeds 3 --epochs 2 --limit-train-batches 50
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


def _run(cfg, *, regime, geometry_mode, encoder, epochs, d_model, n_layers, limit, seed):
    from gloss.train.finetune import train

    d_text = 2560 if encoder == "qwen" else 64
    mk = dict(d_model=d_model, n_heads=int(cfg.model.n_heads), n_layers=n_layers, d_text=d_text,
              n_freq=int(cfg.model.geometry.get("n_freq", 16)),
              sigma_floor=float(cfg.model.geometry.sigma_floor), geometry_mode=geometry_mode)
    _m, metrics = train(
        dataset=str(cfg.data.dataset), task_name=str(cfg.data.task), regime=regime, encoder=encoder,
        model_kwargs=mk, batch_size=256, max_epochs=epochs, limit_train_batches=limit, seed=seed,
    )
    return metrics.get("val/ap", float("nan")), metrics.get("val/auroc", float("nan"))


def _agg(rows):
    ap = [r[0] for r in rows]
    au = [r[1] for r in rows]
    f = lambda xs: (stats.mean(xs), (stats.pstdev(xs) if len(xs) > 1 else 0.0))
    return f(ap), f(au)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "hash"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=int, default=50)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)

    common = dict(cfg=cfg, encoder=args.encoder, epochs=args.epochs, d_model=args.d_model,
                  n_layers=args.n_layers, limit=args.limit_train_batches)

    print("\n=== H1: doc regime (geometry_mode=generated) ===")
    print(f"{'regime':16s} {'AP mean±std':>18s} {'AUROC mean±std':>18s}")
    for regime in ("full", "shuffled_spans", "null"):
        rows = [_run(regime=regime, geometry_mode="generated", seed=s, **common) for s in range(args.seeds)]
        (apm, aps), (aum, aus) = _agg(rows)
        print(f"{regime:16s} {apm:8.4f}±{aps:.4f}    {aum:8.4f}±{aus:.4f}")

    print("\n=== H2: geometry mode (regime=full) ===")
    print(f"{'mode':16s} {'AP mean±std':>18s} {'AUROC mean±std':>18s}")
    for mode in ("generated", "free_learned"):
        rows = [_run(regime="full", geometry_mode=mode, seed=s, **common) for s in range(args.seeds)]
        (apm, aps), (aum, aus) = _agg(rows)
        print(f"{mode:16s} {apm:8.4f}±{aps:.4f}    {aum:8.4f}±{aus:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
