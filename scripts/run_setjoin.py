"""run_setjoin.py — SetJoin: dry-run substrate check, single-task trainer, and the gate array runner.

    # build rel-f1, sample with the SetJoin fanout, build a JoinBatch, print shapes (+ forward in S2+):
    .venv/bin/python scripts/run_setjoin.py --dry-run [--collate-only]

    # train one task (S3+):
    .venv/bin/python scripts/run_setjoin.py --train --dataset rel-f1 --task driver-dnf --encoder hash --test

    # the gate array (S4+):
    .venv/bin/python scripts/run_setjoin.py --list | --index N | --aggregate | --compare
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gloss.utils.logging import get_logger  # noqa: E402
from gloss.utils.seeding import seed_everything  # noqa: E402

log = get_logger("gloss.run_setjoin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="sample one batch, print JoinBatch shapes, forward")
    ap.add_argument("--collate-only", action="store_true", help="dry-run: skip the model forward")
    ap.add_argument("--train", action="store_true", help="train one (dataset, task) + report val metrics")
    ap.add_argument("--list", action="store_true", help="print the gate grid size")
    ap.add_argument("--index", type=int, default=None, help="run one gate grid cell (SLURM array task)")
    ap.add_argument("--aggregate", action="store_true", help="aggregate gate results in --out-dir")
    ap.add_argument("--compare", action="store_true", help="compare gate results vs leaderboard baselines")
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-dnf")
    ap.add_argument("--encoder", default="hash", help="frozen column-name encoder (hash|qwen|harrier|HF id)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="gate grid: seeds per (dataset, task)")
    ap.add_argument("--route-on", default="signature", choices=["signature", "hidden", "dense"],
                    help="MoE FFN routing arm (signature = the method; dense = plain FFN, the v2 arm)")
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--d-sig", type=int, default=64, help="signature width (signature arm)")
    ap.add_argument("--lambda-ortho", type=float, default=0.5, help="router orthogonality weight")
    ap.add_argument("--n-axial-layers", type=int, default=0,
                    help="row+column cell-attention blocks per node type (0 = off; the three-level arm uses 1)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-wide-layers", type=int, default=2)
    ap.add_argument("--n-set-layers", type=int, default=2)
    ap.add_argument("--n-pma", type=int, default=4)
    ap.add_argument("--wide-len", type=int, default=128)
    ap.add_argument("--set-size", type=int, default=128)
    ap.add_argument("--fanout", type=int, default=64, help="children sampled per o2m relation at hop 1")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--limit-val-batches", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="also score the held-out RelBench test split")
    ap.add_argument("--out-dir", default="results/setjoin_v1")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)

    if args.dry_run:
        return _dry_run(args)
    if args.train:
        return _train(args)
    if args.list or args.index is not None or args.aggregate or args.compare:
        return _gate(args)
    log.error("pass --dry-run, --train, --list, --index N, --aggregate, or --compare")
    return 2


def _build(dataset: str):
    from gloss.data.graph import build_gloss_graph
    from gloss.utils.paths import graph_cache_dir

    return build_gloss_graph(dataset, cache_dir=str(graph_cache_dir(dataset)))


def _dry_run(args) -> int:
    from relbench.tasks import get_task

    from gloss.data.graph import make_loader
    from gloss.setjoin.collate import to_join_batch
    from gloss.setjoin.paths import setjoin_neighbors

    log.info(f"building {args.dataset} graph ...")
    bundle = _build(args.dataset)
    task = get_task(args.dataset, args.task, download=False)
    nn = setjoin_neighbors(bundle, fanout=args.fanout)
    try:
        loader = make_loader(bundle, task, "train", num_neighbors=nn, batch_size=8, shuffle=False)
        raw = next(iter(loader))
        log.info("per-edge-type num_neighbors dict accepted by the sampler")
    except Exception as exc:
        log.warning(f"dict num_neighbors rejected ({exc!r}); falling back to flat [fanout, 8]")
        loader = make_loader(bundle, task, "train", num_neighbors=[args.fanout, 8],
                             batch_size=8, shuffle=False)
        raw = next(iter(loader))
    jb = to_join_batch(raw, bundle, task.entity_table, wide_len=args.wide_len, set_size=args.set_size)
    print(jb.pretty_shapes())

    st = jb.seed_time.unsqueeze(1)
    leaks = int(((jb.wide_row_time > st) & jb.wide_is_timed & ~jb.wide_is_pad).sum()) + \
        int(((jb.elem_row_time > st) & jb.elem_is_timed & jb.elem_mask).sum())
    print(f"leakage check (row_time > seed_time): {leaks}")

    if args.collate_only:
        return 0

    import torch

    from gloss.setjoin.model import SetJoin
    from gloss.train.finetune import name_embeddings

    name_emb = name_embeddings(bundle, args.dataset, encoder=args.encoder, d_text=64)
    model = SetJoin(bundle, name_emb, task.entity_table, d_model=args.d_model, n_heads=args.n_heads,
                    n_wide_layers=args.n_wide_layers, n_set_layers=args.n_set_layers, n_pma=args.n_pma,
                    route_on=args.route_on, num_experts=args.num_experts, k=args.k, d_sig=args.d_sig,
                    n_axial_layers=args.n_axial_layers)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        logits, aux = model(jb)
    print(f"forward OK: logits {tuple(logits.shape)}  finite={bool(torch.isfinite(logits).all())}  "
          f"aux={float(aux):.4f}  params={n_params/1e6:.2f}M")
    assert n_params < 30e6, "param budget exceeded"
    return 0


def _train(args) -> int:
    from relbench.tasks import get_task

    from gloss.setjoin.train import train_prebuilt_setjoin
    from gloss.train.finetune import name_embeddings

    bundle = _build(args.dataset)
    task = get_task(args.dataset, args.task, download=False)
    d_text = 2560 if args.encoder != "hash" else 64
    name_emb = name_embeddings(bundle, args.dataset, encoder=args.encoder, d_text=d_text)
    module, metrics = train_prebuilt_setjoin(
        bundle, task, name_emb,
        model_kwargs=dict(d_model=args.d_model, n_heads=args.n_heads,
                          n_wide_layers=args.n_wide_layers, n_set_layers=args.n_set_layers,
                          n_pma=args.n_pma, route_on=args.route_on, num_experts=args.num_experts,
                          k=args.k, d_sig=args.d_sig, n_axial_layers=args.n_axial_layers),
        fanout=args.fanout, wide_len=args.wide_len, set_size=args.set_size,
        lambda_ortho=args.lambda_ortho,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.epochs, seed=args.seed, num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
    )
    print("[setjoin] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "val" in k))
    if args.test:
        from gloss.setjoin.eval import evaluate_split

        tm = evaluate_split(module, bundle, task, "test", fanout=args.fanout,
                            wide_len=args.wide_len, set_size=args.set_size,
                            batch_size=args.batch_size, num_workers=args.num_workers)
        print("[test] " + "  ".join(f"{k}={v:.4f}" for k, v in tm.items()))
    return 0


def _gate(args) -> int:
    from gloss.setjoin import runner

    out_dir = Path(args.out_dir)
    if args.list:
        print(len(runner.gate_grid(seeds=args.seeds)))
        return 0
    if args.index is not None:
        runner.run_config(
            args.index, seeds=args.seeds, encoder=args.encoder,
            model_kwargs=dict(d_model=args.d_model, n_heads=args.n_heads,
                              n_wide_layers=args.n_wide_layers, n_set_layers=args.n_set_layers,
                              n_pma=args.n_pma, route_on=args.route_on,
                              num_experts=args.num_experts, k=args.k, d_sig=args.d_sig,
                              n_axial_layers=args.n_axial_layers),
            fanout=args.fanout, wide_len=args.wide_len, set_size=args.set_size,
            lambda_ortho=args.lambda_ortho,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            max_epochs=args.epochs, num_workers=args.num_workers,
            limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
            out_dir=out_dir, test=True,
        )
        return 0
    from gloss.eval.ablation import format_table, load_records

    records = load_records(out_dir)
    if args.aggregate:
        print(format_table(records, split="test",
                           baseline=runner.variant_of(dict(route_on=args.route_on))))
        return 0
    if args.compare:
        print(runner.compare_table(records))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
