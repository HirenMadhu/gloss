"""run_pretrain.py — masked-cell pretraining.

    source scripts/env.sh
    .venv/bin/python scripts/run_pretrain.py --datasets rel-f1 --max-steps 200 \
        --batch-size 16 --seq-len 512 --d-model 128 --n-blocks 2 --d-ff 512 --wandb

Budgeted in **steps, not epochs**: the corpus is ~110M candidate seeds across the 7 DBs, so an epoch
is not a meaningful unit. RT's anchor is 50k steps at batch 256, context 1024.

`--text-encoder minilm` is the default here, unlike everywhere else in the repo, because free-text
cell values are otherwise `HashTextEmbedder` noise — roughly half of every schema (see
`scripts/prep_data.py`'s header). Pass `--text-encoder hash` to reproduce the old behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1"])
    ap.add_argument("--holdout", default=None,
                    help="LODO: pretrain on every dataset EXCEPT this one")
    ap.add_argument("--encoder", default="qwen", help="frozen encoder for column NAMES")
    ap.add_argument("--text-encoder", default="minilm", help="frozen encoder for CELL VALUES")
    # objective
    ap.add_argument("--p-random", type=float, default=0.15)
    ap.add_argument("--no-seed-target", action="store_true",
                    help="drop RT's one-cell-per-sequence seed target; plain BERT masking")
    ap.add_argument("--lambda-cat", type=float, default=1.0)
    ap.add_argument("--lambda-ortho", type=float, default=0.5)
    # model
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--d-sig", type=int, default=128)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--num-shared", type=int, default=2, help="cell-level shared experts")
    ap.add_argument("-k", "--top-k", type=int, default=2)
    ap.add_argument("--row-num-experts", type=int, default=4)
    ap.add_argument("--row-num-shared", type=int, default=1)
    ap.add_argument("--dispatch", default="sparse", choices=["dense", "sparse"])
    ap.add_argument("--cell-encoder", default="schema_free", choices=["per_column", "schema_free"],
                    help="schema_free shares one projection per datatype, so the ENTIRE model "
                         "transfers to an unseen schema; per_column is torch_frame's per-column "
                         "encoders (10.6M non-transferable params on rel-trial) kept as the "
                         "comparison arm.")
    ap.add_argument("--cell-attn-backend", default="flex", choices=["sdpa", "flex"])
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-mean-aux", action="store_true",
                    help="sum aux over blocks instead of averaging (lambda then scales with depth)")
    # optimization
    ap.add_argument("--max-steps", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--accum", type=int, default=1)
    # data
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-fk", type=int, default=5)
    ap.add_argument("--num-neighbors", type=int, nargs="+", default=[12, 12])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-steps", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    # bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true", help="continue from the run's train_state.pt")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="more-pretrain")
    ap.add_argument("--devices", type=int, default=1)
    return ap


def resolve_datasets(args) -> list[str]:
    from gloss.eval.ablation import LEADERBOARD_TASKS

    all_dbs = ["rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"]
    if args.holdout:
        if args.holdout not in all_dbs:
            raise SystemExit(f"unknown --holdout {args.holdout!r}; expected one of {all_dbs}")
        return [d for d in all_dbs if d != args.holdout]
    if args.datasets == ["all"]:
        return all_dbs
    assert LEADERBOARD_TASKS is not None
    return list(args.datasets)


def main() -> int:
    args = build_argparser().parse_args()
    warnings.filterwarnings("ignore")

    import pytorch_lightning as pl
    import torch

    from gloss.data.graph import build_gloss_graph
    from gloss.data.pretrain_loader import build_pretrain_stream
    from gloss.text.schema import (build_table_name_embeddings, role_name_embeddings_with_none)
    from gloss.train.finetune import _name_encoder
    from gloss.train.pretrain import MoREPretrainLitModule, PretrainCheckpoint
    from gloss.utils.paths import ckpt_dir, graph_cache_dir
    from gloss.utils.seeding import seed_everything

    datasets = resolve_datasets(args)
    if len(datasets) > 1:
        raise SystemExit(
            f"--datasets resolved to {datasets}. The MODEL is no longer the obstacle — with "
            "--cell-encoder schema_free its state_dict is identical across schemas — but the loader "
            "and the frozen name/label tables are still built per bundle, so one process holds one "
            "database. Multi-DB is the next phase; pass exactly one dataset for now.")
    ds = datasets[0]
    run = args.run_name or f"{ds}-d{args.d_model}b{args.n_blocks}e{args.num_experts}s{args.seed}"
    seed_everything(args.seed)

    cache = str(graph_cache_dir(ds, args.text_encoder))
    if not Path(cache).exists():
        raise SystemExit(
            f"no graph cache at {cache}. Build it first:\n"
            f"  sbatch scripts/prep.sh --datasets {ds} --text-encoder {args.text_encoder} "
            f"--no-download")
    bundle = build_gloss_graph(ds, cache_dir=cache, text_encoder=args.text_encoder)

    enc = _name_encoder(ds, encoder=args.encoder, d_text=2560 if args.encoder == "qwen" else 64)
    from gloss.text.schema import build_category_name_embeddings, build_column_name_embeddings
    name_emb = build_column_name_embeddings(bundle, enc, kind="query")
    tab = build_table_name_embeddings(bundle, enc, kind="query")
    role = role_name_embeddings_with_none(bundle, enc, kind="query")
    # Category LABELS as text. Cheap (383 distinct values corpus-wide) and cached like the rest, but
    # only the schema-free encoder consumes it.
    cat = (build_category_name_embeddings(bundle, enc, kind="query")
           if args.cell_encoder == "schema_free" else None)

    model_kwargs = dict(
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads, d_ff=args.d_ff,
        d_sig=args.d_sig, enc_channels=args.d_model, num_experts=args.num_experts, k=args.top_k,
        cell_num_shared=args.num_shared, row_num_experts=args.row_num_experts,
        row_num_shared=args.row_num_shared, dispatch=args.dispatch,
        cell_attn_backend=args.cell_attn_backend, grad_checkpoint=args.grad_checkpoint,
        mean_aux=not args.no_mean_aux, table_name_emb=tab, role_name_emb=role,
        cell_encoder=args.cell_encoder, cat_name_emb=cat,
    )
    module = MoREPretrainLitModule(
        bundle, name_emb, model_kwargs=model_kwargs, p_random=args.p_random,
        seed_target=not args.no_seed_target, lambda_cat=args.lambda_cat,
        lambda_ortho=args.lambda_ortho, lr=args.lr, weight_decay=args.weight_decay,
        max_steps=args.max_steps, warmup_frac=args.warmup_frac, seq_len=args.seq_len,
        max_fk=args.max_fk, mask_seed=args.seed,
    )
    n_par = sum(p.numel() for p in module.parameters())
    print(f"[{run}] {ds}: {n_par/1e6:.1f}M params | {module.spec.summary()}", flush=True)

    common = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors,
                  num_workers=args.num_workers, seed=args.seed)
    train_stream = build_pretrain_stream(bundle, module.spec, split="train",
                                         steps=args.max_steps * args.accum, **common)
    val_stream = build_pretrain_stream(bundle, module.spec, split="val",
                                       steps=args.val_steps, **common)
    print(f"[{run}] seed tables: {sorted(train_stream.loaders)}", flush=True)

    root = ckpt_dir(run)
    cfg = {"run": run, "dataset": ds, "args": vars(args), "model_kwargs":
           {k: v for k, v in model_kwargs.items() if not hasattr(v, "shape")}, "n_params": n_par}
    ckpt = PretrainCheckpoint(root, ds, cfg, every_n_steps=args.ckpt_every)

    logger = False
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=run, save_dir=str(root))
        logger.log_hyperparams(cfg["args"] | {"n_params": n_par})

    trainer = pl.Trainer(
        max_steps=args.max_steps, accelerator="auto", devices=args.devices,
        precision=args.precision, logger=logger, enable_checkpointing=False,
        enable_model_summary=False, log_every_n_steps=10, callbacks=[ckpt],
        gradient_clip_val=1.0, accumulate_grad_batches=args.accum,
        val_check_interval=args.val_every, num_sanity_val_steps=0,
        limit_val_batches=args.val_steps,
    )
    resume = root / "train_state.pt" if args.resume else None
    if resume and resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        print(f"[{run}] resuming from step {st['global_step']}", flush=True)
    trainer.fit(module, train_stream, val_stream)

    print(f"[{run}] checkpoints -> {root}", flush=True)
    print("PRETRAIN_OK " + json.dumps({"run": run, "steps": int(trainer.global_step),
                                       "best_val": ckpt.best}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
