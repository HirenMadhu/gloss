"""run_train.py — Phase-0 DoD + single-arm trainer.

    # P0 DoD: build rel-f1, sample a leakage-safe minibatch, build a CellBatch, print shapes, forward:
    .venv/bin/python scripts/run_train.py --dry-run

    # train one arm and report val metrics (Phase 3):
    .venv/bin/python scripts/run_train.py --train --regime full  --encoder qwen
    .venv/bin/python scripts/run_train.py --train --regime null  --encoder qwen --baseline
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402
from gloss.utils.seeding import seed_everything  # noqa: E402

log = get_logger("gloss.run_train")


def _model_kwargs(cfg, d_text: int) -> dict:
    m = cfg.model
    return dict(
        d_model=int(m.d_model), d_text=d_text, n_blocks=int(m.n_blocks),
        n_heads=int(m.n_heads), d_ff=int(m.get("d_ff", 4 * int(m.d_model))),
        enc_channels=int(m.get("enc_channels", int(m.d_model))),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--dry-run", action="store_true", help="sample one batch, print shapes, forward")
    ap.add_argument("--train", action="store_true", help="train one arm + report val metrics")
    ap.add_argument("--regime", default="full", choices=["full", "null", "shuffled", "name_only"])
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "hash"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None, help="override config train.num_workers")
    ap.add_argument("--seq-len", type=int, default=None, help="override config seq_len (smaller=faster)")
    ap.add_argument("--baseline", action="store_true", help="also run the LightGBM floor")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    dataset = str(cfg.data.dataset)
    task_name = str(cfg.data.task)
    num_neighbors = list(cfg.data.sampler.num_neighbors)
    seq_len = int(args.seq_len or cfg.data.collate.seq_len)
    max_fk = int(cfg.data.collate.max_fk)

    if args.dry_run:
        return _dry_run(cfg, dataset, task_name, num_neighbors, seq_len, max_fk, args)
    if args.train:
        return _train(cfg, dataset, task_name, num_neighbors, seq_len, max_fk, args)
    log.error("pass --dry-run (Phase 0) or --train (Phase 3)")
    return 2


def _dry_run(cfg, dataset, task_name, num_neighbors, seq_len, max_fk, args) -> int:
    import torch

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.model.docrt import DOCRT
    from gloss.train.finetune import docs_for_regime
    from relbench.tasks import get_task

    log.info(f"building {dataset} graph ...")
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=num_neighbors,
                         batch_size=args.batch_size or 8, shuffle=False)
    raw = next(iter(loader))
    # small fixed cap for a quick CPU forward
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=min(seq_len, 384), max_fk=max_fk)
    print(cb.pretty_shapes())

    # leakage check: no timed cell with row_time > its seed's time
    st = cb.seed_time.unsqueeze(1)
    bad = int(((cb.row_time > st) & cb.is_timed & ~cb.is_padding).sum())
    print(f"leakage check (row_time > seed_time): {bad}")

    g = docs_for_regime(dataset, args.regime, encoder="hash", d_text=64, sim_threshold=0.0)
    model = DOCRT(bundle, d_model=128, d_text=g.d_text, n_blocks=2, n_heads=4, d_ff=256, enc_channels=128)
    with torch.no_grad():
        logits = model(cb, g)
    print(f"forward OK: logits {tuple(logits.shape)}  finite={bool(torch.isfinite(logits).all())}")
    return 0


def _train(cfg, dataset, task_name, num_neighbors, seq_len, max_fk, args) -> int:
    from gloss.train.finetune import docs_for_regime, train_prebuilt
    from gloss.data.graph import build_gloss_graph
    from relbench.tasks import get_task

    d_text = 2560 if args.encoder == "qwen" else 64
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g = docs_for_regime(dataset, args.regime, encoder=args.encoder, d_text=d_text,
                        sim_threshold=float(cfg.docs.grounding.sim_threshold))
    mk = _model_kwargs(cfg, g.d_text)
    _, metrics = train_prebuilt(
        bundle, task, g, model_kwargs=mk,
        num_neighbors=num_neighbors, seq_len=seq_len, max_fk=max_fk,
        batch_size=args.batch_size or int(cfg.train.batch_size),
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        max_epochs=args.epochs or int(cfg.train.max_epochs), seed=int(cfg.seed),
        num_workers=args.num_workers if args.num_workers is not None else int(cfg.train.num_workers),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    print(f"[{args.regime}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "val" in k))

    if args.baseline:
        from gloss.eval.baselines import run_lightgbm_baseline

        floor = run_lightgbm_baseline(bundle, task, num_neighbors=num_neighbors)
        print("LightGBM floor:", {k: round(float(v), 4) for k, v in floor.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
