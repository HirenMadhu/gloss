#!/usr/bin/env python
"""run_finetune.py — Phase-4 fine-tuning entry point (TRAINING is a Phase-4 stub).

For now its ``--dry-run`` is the **Phase-0 Definition of Done**: load a (synthetic or RelBench) task,
sample a batch of leakage-safe subgraphs, collate to a TokenBatch, and print shapes. This proves the
data path end-to-end before any model exists.

    python scripts/run_finetune.py --dry-run                       # synthetic (offline)
    python scripts/run_finetune.py --dry-run --dataset rel-f1 --task driver-dnf
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")


def _load_bundle(args):
    if args.dataset == "synthetic":
        from gloss.data.synthetic import make_synthetic_bundle

        bundle, planted = make_synthetic_bundle(seed=args.seed)
        print(f"[synthetic] planted_truth = {planted.as_dict()}")
        return bundle
    from gloss.data.relbench_graph import load_task_bundle

    return load_task_bundle(args.dataset, args.task, download=True)


def dry_run(args) -> None:
    from gloss.data.collate import collate_subgraphs
    from gloss.utils.seeding import seed_everything

    seed_everything(args.seed)
    bundle = _load_bundle(args)
    print(f"[bundle] dataset={bundle.dataset} task={bundle.task_name} "
          f"tables={bundle.registry.num_tables} cols={bundle.registry.num_cols} "
          f"fk_roles={bundle.registry.num_fk_roles}")

    sampler = bundle.make_sampler(num_neighbors=args.num_neighbors, max_cells=args.max_cells, seed=args.seed)
    seeds = bundle.task.get_table("train").df.head(args.batch_seeds)
    subs = [
        sampler.sample(bundle.entity_table, getattr(r, bundle.entity_col),
                       getattr(r, bundle.time_col), getattr(r, bundle.target_col))
        for r in seeds.itertuples(index=False)
    ]
    tb = collate_subgraphs(subs, bundle.db, bundle.registry)

    rows_per_seed = [len(sg.rows) for sg in subs]
    print(f"[batch] B={tb.batch_size} T_max={tb.max_cells} "
          f"real_cells={int(tb.pad_mask.sum())} rows/seed(min/mean/max)="
          f"{min(rows_per_seed)}/{sum(rows_per_seed)/len(rows_per_seed):.1f}/{max(rows_per_seed)}")
    print("[shapes]")
    for name, shape in tb.shapes().items():
        print(f"   {name:16s} {shape}")
    # leakage sanity on the materialized batch
    real = tb.pad_mask
    bad = ((tb.row_time > tb.seed_time) & real).sum().item()
    print(f"[leakage] cells with row_time>seed_time among real cells: {bad} (must be 0)")
    assert bad == 0
    print("DRY_RUN_OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--task", default="driver-dnf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-seeds", dest="batch_seeds", type=int, default=16)
    p.add_argument("--num-neighbors", dest="num_neighbors", type=int, nargs="+", default=[12, 12])
    p.add_argument("--max-cells", dest="max_cells", type=int, default=4096)
    args = p.parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        raise SystemExit("Training is a Phase-4 stub. Use --dry-run for the Phase-0 DoD.")


if __name__ == "__main__":
    main()
