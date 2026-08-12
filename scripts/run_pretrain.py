"""run_pretrain.py — masked-cell pretraining, single- or multi-database.

    source scripts/env.sh
    .venv/bin/python scripts/run_pretrain.py --datasets rel-f1 --max-steps 200 --wandb
    .venv/bin/python scripts/run_pretrain.py --datasets all  --max-steps 50000     # one model, 7 DBs
    .venv/bin/python scripts/run_pretrain.py --holdout rel-f1 --max-steps 50000    # LODO fold

Budgeted in **steps, not epochs**: the corpus is ~110M candidate seeds across the 7 DBs, so an epoch
is not a meaningful unit. RT's anchor is 50k steps at batch 256, context 1024.

Two defaults differ from the rest of the repo, both deliberately. `--text-encoder minilm`, because
free-text cell values are otherwise `HashTextEmbedder` noise — roughly half of every schema (see
`scripts/prep_data.py`'s header). And `--cell-encoder schema_free`, because torch_frame's per-column
stype encoders cannot transfer and multi-DB training is impossible with them.

Prints one machine-readable line on success, `PRETRAIN_OK {...}`, carrying `best_val` — that is the
metric an autoresearch loop should optimize.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALL_DBS = ["rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1"], help="names, or 'all'")
    ap.add_argument("--holdout", default=None, help="LODO: pretrain on every dataset EXCEPT this one")
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
                         "transfers to an unseen schema and several DBs can share one set of "
                         "weights; per_column is torch_frame's per-column encoders, kept as the "
                         "comparison arm and single-DB only.")
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
    ap.add_argument("--max-seeds-per-table", type=int, default=None,
                    help="cap each table's seed pool (RT-J uses 100k) so one huge table cannot own "
                         "the mixture")
    # bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true", help="continue from the run's train_state.pt")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="more-pretrain")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--metric-out", default=None,
                    help="write the final metrics as JSON here (for autoresearch)")
    return ap


def resolve_datasets(args) -> list[str]:
    if args.holdout:
        if args.holdout not in ALL_DBS:
            raise SystemExit(f"unknown --holdout {args.holdout!r}; expected one of {ALL_DBS}")
        return [d for d in ALL_DBS if d != args.holdout]
    if list(args.datasets) == ["all"]:
        return list(ALL_DBS)
    return list(args.datasets)


def default_run_name(args, datasets: list[str]) -> str:
    if args.run_name:
        return args.run_name
    who = f"hold-{args.holdout}" if args.holdout else (
        "-".join(datasets) if len(datasets) <= 2 else f"all{len(datasets)}")
    return f"{who}-d{args.d_model}b{args.n_blocks}e{args.num_experts}s{args.seed}"


def main() -> int:
    args = build_argparser().parse_args()
    warnings.filterwarnings("ignore")

    import pytorch_lightning as pl

    from gloss.data.graph import build_gloss_graph
    from gloss.data.masking import build_column_target_spec
    from gloss.data.pretrain_loader import build_multi_pretrain_stream
    from gloss.model.schema_bank import SchemaBank, SchemaTables
    from gloss.text.schema import (build_category_name_embeddings, build_column_name_embeddings,
                                   build_table_name_embeddings, role_name_embeddings_with_none)
    from gloss.train.finetune import _name_encoder
    from gloss.train.pretrain import MoREPretrainLitModule, PretrainCheckpoint
    from gloss.utils.paths import ckpt_dir, graph_cache_dir
    from gloss.utils.seeding import seed_everything

    datasets = resolve_datasets(args)
    run = default_run_name(args, datasets)
    seed_everything(args.seed)
    if len(datasets) > 1 and args.cell_encoder != "schema_free":
        raise SystemExit(
            "multi-database training requires --cell-encoder schema_free. torch_frame's per-column "
            "stype encoders are sized by one database's columns, so one set of weights cannot serve "
            "several; the schema-free encoder makes the state_dict identical across schemas.")

    # ---- per-database frozen tables. The WEIGHTS are shared; only these differ. ----
    bundles, embs = {}, {}
    for ds in datasets:
        cache = graph_cache_dir(ds, args.text_encoder)
        if not cache.exists():
            raise SystemExit(
                f"no graph cache at {cache}. Build it first:\n"
                f"  sbatch scripts/prep.sh --datasets {ds} --text-encoder {args.text_encoder} "
                f"--no-download")
        bundles[ds] = build_gloss_graph(ds, cache_dir=str(cache), text_encoder=args.text_encoder)
        enc = _name_encoder(ds, encoder=args.encoder,
                            d_text=2560 if args.encoder == "qwen" else 64)
        embs[ds] = dict(
            name=build_column_name_embeddings(bundles[ds], enc, kind="query"),
            table=build_table_name_embeddings(bundles[ds], enc, kind="query"),
            role=role_name_embeddings_with_none(bundles[ds], enc, kind="query"),
            cat=(build_category_name_embeddings(bundles[ds], enc, kind="query")
                 if args.cell_encoder == "schema_free" else None),
        )

    anchor = datasets[0]
    model_kwargs = dict(
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads, d_ff=args.d_ff,
        d_sig=args.d_sig, enc_channels=args.d_model, num_experts=args.num_experts, k=args.top_k,
        cell_num_shared=args.num_shared, row_num_experts=args.row_num_experts,
        row_num_shared=args.row_num_shared, dispatch=args.dispatch,
        cell_attn_backend=args.cell_attn_backend, grad_checkpoint=args.grad_checkpoint,
        mean_aux=not args.no_mean_aux, cell_encoder=args.cell_encoder,
        table_name_emb=embs[anchor]["table"], role_name_emb=embs[anchor]["role"],
        cat_name_emb=embs[anchor]["cat"],
    )
    module = MoREPretrainLitModule(
        bundles[anchor], embs[anchor]["name"], bundles=bundles, model_kwargs=model_kwargs,
        p_random=args.p_random, seed_target=not args.no_seed_target, lambda_cat=args.lambda_cat,
        lambda_ortho=args.lambda_ortho, lr=args.lr, weight_decay=args.weight_decay,
        max_steps=args.max_steps, warmup_frac=args.warmup_frac, seq_len=args.seq_len,
        max_fk=args.max_fk, mask_seed=args.seed,
    )

    # One SchemaTables per DB, built by temporarily standing the model up on each. Only the frozen
    # buffers differ, so this costs a few table gathers, not a model.
    specs = {}
    if len(datasets) > 1:
        from gloss.model.more import MoRE

        tables = {anchor: SchemaTables.from_model(module.model, anchor)}
        specs[anchor] = module.spec
        for ds in datasets[1:]:
            kw = dict(model_kwargs, table_name_emb=embs[ds]["table"],
                      role_name_emb=embs[ds]["role"], cat_name_emb=embs[ds]["cat"])
            probe = MoRE(bundles[ds], embs[ds]["name"], route_on="signature", **kw)
            tables[ds] = SchemaTables.from_model(probe, ds)
            specs[ds] = build_column_target_spec(bundles[ds], probe.encoder)
            del probe
        module.bank = SchemaBank(tables)
        module.specs = specs
    else:
        specs[anchor] = module.spec

    n_par = sum(p.numel() for p in module.parameters())
    print(f"[{run}] {len(datasets)} DB(s): {datasets}", flush=True)
    print(f"[{run}] {n_par/1e6:.1f}M params | " +
          " | ".join(f"{d}: {specs[d].summary()}" for d in datasets), flush=True)

    common = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors,
                  num_workers=args.num_workers, seed=args.seed,
                  max_seeds_per_table=args.max_seeds_per_table)
    train_stream = build_multi_pretrain_stream(bundles, specs, split="train",
                                               steps=args.max_steps * args.accum, **common)
    val_stream = build_multi_pretrain_stream(bundles, specs, split="val",
                                             steps=args.val_steps, **common)
    print(f"[{run}] {len(train_stream.names)} seed tables in the mixture", flush=True)

    root = ckpt_dir(run)
    cfg = {"run": run, "datasets": datasets, "args": vars(args), "n_params": n_par,
           "model_kwargs": {k: v for k, v in model_kwargs.items() if not hasattr(v, "shape")}}
    ckpt = PretrainCheckpoint(root, anchor, cfg, every_n_steps=args.ckpt_every)

    logger = False
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=run, save_dir=str(root))
        logger.log_hyperparams(cfg["args"] | {"n_params": n_par, "n_datasets": len(datasets)})

    strategy = "auto"
    if args.devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        # find_unused_parameters is required, twice over: sparse dispatch leaves an expert with no
        # routed tokens without a gradient (the dense combine always produced a zero one), and a
        # batch only touches the node types its sample happened to contain.
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_steps=args.max_steps, accelerator="auto", devices=args.devices, strategy=strategy,
        precision=args.precision, logger=logger, enable_checkpointing=False,
        enable_model_summary=False, log_every_n_steps=10, callbacks=[ckpt],
        gradient_clip_val=1.0, accumulate_grad_batches=args.accum,
        val_check_interval=args.val_every, num_sanity_val_steps=0,
        limit_val_batches=args.val_steps,
    )
    ckpt_path = root / "train_state.pt"
    if args.resume and ckpt_path.exists():
        print(f"[{run}] resume state present at {ckpt_path}", flush=True)
    trainer.fit(module, train_stream, val_stream)

    metrics = {"run": run, "datasets": datasets, "steps": int(trainer.global_step),
               "best_val": ckpt.best, "n_params": n_par}
    if args.metric_out:
        Path(args.metric_out).write_text(json.dumps(metrics, indent=2))
    print(f"[{run}] checkpoints -> {root}", flush=True)
    print("PRETRAIN_OK " + json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
