"""run_finetune.py — Phase-0 DoD entry point.

`--dry-run` builds the rel-f1 graph, samples one leakage-safe disjoint minibatch, collates it into a
dense GlossBatch, and prints shapes. (Actual fine-tuning lands in Phase 4.)

    .venv/bin/python scripts/run_finetune.py --dry-run
    .venv/bin/python scripts/run_finetune.py --dry-run --config rel-f1 --batch-size 8
"""
from __future__ import annotations

import argparse
import sys
import warnings

# make `gloss` importable when run as a script
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402
from gloss.utils.seeding import seed_everything  # noqa: E402

log = get_logger("gloss.dry_run")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="sample one batch and print shapes")
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--split", default="train")
    # training mode (Phase 4)
    ap.add_argument("--train", action="store_true", help="fine-tune HALOS + report val metrics")
    ap.add_argument("--regime", default="full", choices=["full", "null", "shuffled_spans"])
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "hash"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--baseline", action="store_true", help="also run the LightGBM floor")
    args = ap.parse_args()

    if args.train:
        return _train(args)
    if not args.dry_run:
        log.error("pass --dry-run (Phase 0) or --train (Phase 4)")
        return 2

    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    from relbench.tasks import get_task

    from gloss.data.collate import to_gloss_batch
    from gloss.data.graph import build_gloss_graph, make_loader

    log.info("building graph for dataset=%s task=%s", cfg.data.dataset, cfg.data.task)
    bundle = build_gloss_graph(cfg.data.dataset)
    log.info(
        "graph: %d node types, %d edge types | fk_roles=%d metapaths=%d",
        bundle.num_node_types, len(bundle.edge_types), bundle.num_fk_roles, bundle.num_metapaths,
    )

    task = get_task(cfg.data.dataset, cfg.data.task, download=False)
    loader = make_loader(
        bundle, task, args.split,
        num_neighbors=list(cfg.data.sampler.num_neighbors),
        batch_size=args.batch_size,
        shuffle=False,
    )
    raw = next(iter(loader))
    gb = to_gloss_batch(raw, bundle, task.entity_table, max_nodes=int(cfg.data.sampler.max_nodes))
    print(gb.pretty_shapes())

    # leakage sanity (the headline invariant)
    rt_i = gb.row_time.unsqueeze(2)
    seedt = gb.seed_time.view(-1, 1, 1)
    bad = (gb.is_timed.unsqueeze(2) & (rt_i > seedt)).sum().item()
    log.info("leakage check: timestamped nodes with row_time > seed_time = %d (expect 0)", bad)
    return 0


def _train(args) -> int:
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    from relbench.tasks import get_task

    from gloss.train.finetune import train

    d_text = 2560 if args.encoder == "qwen" else 64
    mk = dict(d_model=args.d_model, n_heads=int(cfg.model.n_heads), n_layers=args.n_layers,
              d_text=d_text, n_freq=int(cfg.model.geometry.get("n_freq", 16)),
              sigma_floor=float(cfg.model.geometry.sigma_floor))
    log.info("training HALOS regime=%s encoder=%s epochs=%d d_model=%d n_layers=%d",
             args.regime, args.encoder, args.epochs, args.d_model, args.n_layers)
    _module, metrics = train(
        dataset=str(cfg.data.dataset), task_name=str(cfg.data.task), regime=args.regime,
        encoder=args.encoder, model_kwargs=mk, batch_size=args.batch_size if args.batch_size > 8 else 256,
        max_epochs=args.epochs, limit_train_batches=args.limit_train_batches, seed=int(cfg.seed),
    )
    val = {k: v for k, v in metrics.items() if k.startswith("val/")}
    log.info("HALOS val metrics: %s", {k: round(v, 4) for k, v in val.items()})

    if args.baseline:
        from gloss.data.graph import build_gloss_graph
        from gloss.eval.baselines import run_lightgbm_baseline

        bundle = build_gloss_graph(str(cfg.data.dataset))
        task = get_task(str(cfg.data.dataset), str(cfg.data.task), download=False)
        bm = run_lightgbm_baseline(bundle, task, num_neighbors=list(cfg.data.sampler.num_neighbors))
        log.info("LightGBM floor: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bm.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
