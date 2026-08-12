"""run_train.py — dry-run substrate check + single-arm trainer (plain RT).

    # build rel-f1, sample a leakage-safe minibatch, build a CellBatch, print shapes, forward:
    .venv/bin/python scripts/run_train.py --dry-run

    # train one arm and report val metrics:
    .venv/bin/python scripts/run_train.py --train --encoder hash
    .venv/bin/python scripts/run_train.py --train --encoder qwen --baseline
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


def _model_kwargs(cfg, args) -> dict:
    m = cfg.model
    return dict(
        d_model=int(m.d_model), n_blocks=int(m.n_blocks),
        n_heads=int(m.n_heads), d_ff=int(m.get("d_ff", 4 * int(m.d_model))),
        enc_channels=int(m.get("enc_channels", int(m.d_model))),
        d_sig=int(args.d_sig), num_experts=int(args.num_experts), k=int(args.k),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--dry-run", action="store_true", help="sample one batch, print shapes, forward")
    ap.add_argument("--train", action="store_true", help="train one arm + report val metrics")
    ap.add_argument("--encoder", default="hash", choices=["qwen", "hash"],
                    help="frozen encoder for the column-name table (hash=dev, qwen=real)")

    ap.add_argument("--route-on", default="dense",
                    choices=["signature", "dense"],
                    help="MoE routing arm (dense = plain RT; hybrid = signature+hidden)")
    ap.add_argument("--dataset", default=None, help="override config data.dataset")
    ap.add_argument("--task", default=None, help="override config data.task")
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--d-sig", type=int, default=128)
    ap.add_argument("--lambda-ortho", type=float, default=0.5)









    ap.add_argument("--test", action="store_true", help="also score the held-out RelBench test split")
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
    dataset = str(args.dataset or cfg.data.dataset)
    task_name = str(args.task or cfg.data.task)
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
    from gloss.model.more import MoRE
    from gloss.train.finetune import name_embeddings
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

    print(f"row graph: num_rows max {int(cb.num_rows.max())} of R={cb.row_valid.shape[1]}"
          f"  hop max {int(cb.row_hop.max())}"
          f"  distinct in-roles {int(cb.row_in_role.unique().numel())}")
    # the row-level leakage assert must be scoped: untimed rows carry sentinel 0 and real UNIX
    # seconds go negative, and the ROOT is the query entity, which RelBench includes regardless
    # of its own timestamp (35% of rel-event task rows are dated after their seed).
    rbad = int(((cb.row_time_r > cb.seed_time.unsqueeze(1)) & cb.row_is_timed
                & cb.row_valid & ~cb.row_is_root).sum())
    print(f"row leakage check (non-root timed rows after seed): {rbad}")

    name_emb = name_embeddings(bundle, dataset, encoder=args.encoder, d_text=64)
    extra = {}
    from gloss.text.cache import HashEncoder
    from gloss.text.schema import (
        build_table_name_embeddings,
        role_name_embeddings_with_none,
    )

    enc = HashEncoder(dim=64) if args.encoder == "hash" else _name_enc(dataset, args)
    extra = {
        "table_name_emb": build_table_name_embeddings(bundle, enc),
        "role_name_emb": role_name_embeddings_with_none(bundle, enc),
        **_two_level_kwargs(cfg),
    }

    model = MoRE(bundle, name_emb, d_model=128, d_sig=64, n_blocks=2, n_heads=4, d_ff=256,
                 enc_channels=128, route_on="signature", **extra)
    with torch.no_grad():
        logits, aux = model(cb)
    print(f"forward OK (two_level): logits {tuple(logits.shape)}  "
          f"finite={bool(torch.isfinite(logits).all())}  aux={float(aux):.4f}")
    return 0



def _name_enc(dataset: str, args):
    """The frozen encoder for the table/role name tables (P0.4), matching --encoder."""
    from gloss.train.finetune import _name_encoder

    return _name_encoder(dataset, encoder=args.encoder,
                         d_text=2560 if args.encoder == "qwen" else 64)


def _two_level_kwargs(cfg) -> dict:
    """Flatten `model.two_level` (changes.md §4) into MoRE/TwoLevelSubstrate kwargs.

    Config keys are grouped by level for readability; the modules take a flat kwarg list. Missing
    sections fall back to the Phase 0a defaults, which is what makes `--arch two_level` alone
    reproduce Phase 0a rather than the full design.
    """
    tl = getattr(getattr(cfg, "model", object()), "two_level", None)
    if tl is None:
        return {}
    cell = getattr(tl, "cell", None)
    row = getattr(tl, "row", None)
    return {
        "max_hop": int(getattr(tl, "max_hop", 8)),
        "cell_attn_backend": str(getattr(cell, "backend", "sdpa")),
        "pool_slots": int(getattr(row, "pool_slots", 4)),
        "row_num_experts": int(getattr(row, "num_experts", 4)),
        "row_k": int(getattr(row, "top_k", 2)),
        "row_use_shared": bool(getattr(row, "use_shared", True)),
        "lambda_ortho": float(getattr(row, "lambda_ortho", 0.5)),
        "lambda_balance": float(getattr(row, "lambda_balance", 0.01)),
    }


def _train(cfg, dataset, task_name, num_neighbors, seq_len, max_fk, args) -> int:
    from gloss.data.graph import build_gloss_graph
    from gloss.train.finetune import name_embeddings, train_prebuilt
    from relbench.tasks import get_task

    d_text = 2560 if args.encoder == "qwen" else 64
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    name_emb = name_embeddings(bundle, dataset, encoder=args.encoder, d_text=d_text)
    mk = _model_kwargs(cfg, args)
    batch_size = args.batch_size or int(cfg.train.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.train.num_workers)
    module, metrics = train_prebuilt(
        bundle, task, name_emb, model_kwargs=mk, route_on=args.route_on, lambda_ortho=args.lambda_ortho,
        num_neighbors=num_neighbors, seq_len=seq_len, max_fk=max_fk,
        batch_size=batch_size,
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        max_epochs=args.epochs or int(cfg.train.max_epochs), seed=int(cfg.seed),
        num_workers=num_workers,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    print(f"[{args.route_on}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "val" in k))

    if args.test:
        from gloss.eval.test_eval import evaluate_split

        tm = evaluate_split(module, bundle, task, "test", num_neighbors=num_neighbors,
                            seq_len=seq_len, max_fk=max_fk, batch_size=batch_size, num_workers=num_workers)
        print("[test] " + "  ".join(f"{k}={v:.4f}" for k, v in tm.items()))

    if args.baseline:
        from gloss.eval.baselines import run_lightgbm_baseline

        floor = run_lightgbm_baseline(bundle, task, num_neighbors=num_neighbors)
        print("LightGBM floor:", {k: round(float(v), 4) for k, v in floor.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
