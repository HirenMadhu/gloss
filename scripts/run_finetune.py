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
    args = ap.parse_args()

    if not args.dry_run:
        log.error("only --dry-run is implemented in Phase 0 (training is Phase 4)")
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


if __name__ == "__main__":
    raise SystemExit(main())
