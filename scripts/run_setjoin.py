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
    ap.add_argument("--use-shared", action="store_true",
                    help="add an always-on shared expert to every MoE FFN (the S addition)")
    ap.add_argument("--d-sig", type=int, default=64, help="signature width (signature arm)")
    ap.add_argument("--lambda-ortho", type=float, default=0.5, help="router orthogonality weight")
    ap.add_argument("--n-axial-layers", type=int, default=0,
                    help="row+column cell-attention blocks per node type (0 = off; the three-level arm uses 1)")
    ap.add_argument("--readout", default="pma", choices=["pma", "measure", "slot"],
                    help="set-encoder readout arm (pma = current PMA; measure/slot = GROUP-BY pooling)")
    ap.add_argument("--slot-mode", default="hard", choices=["hard", "soft", "iterative"],
                    help="slot readout: hard GROUP BY | soft (learned gamma) | iterative (GRU)")
    ap.add_argument("--slot-group-key", default="table_fkrole", help="slot grouping key")
    ap.add_argument("--slot-iters", type=int, default=3, help="iterative slot refinement steps")
    ap.add_argument("--slot-gamma-init", type=float, default=2.0, help="initial key-bias strength")
    ap.add_argument("--no-slot-gamma-learnable", dest="slot_gamma_learnable", action="store_false",
                    help="freeze the slot key-bias gamma")
    ap.add_argument("--no-slot-seed-cond", dest="slot_seed_cond", action="store_false",
                    help="do not seed-condition the slot init")
    ap.add_argument("--readout-channels", default="mean,count,sum",
                    help="comma-separated measure channels ⊆ {mean,count,sum,max}")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-wide-layers", type=int, default=2)
    ap.add_argument("--n-set-layers", type=int, default=2)
    ap.add_argument("--n-pma", type=int, default=4)
    ap.add_argument("--wide-len", type=int, default=128)
    ap.add_argument("--set-size", type=int, default=128)
    # ---- unified RowModel (--model rowmodel): one hierarchical encoder over a set of wide rows ----
    ap.add_argument("--model", default="setjoin", choices=["setjoin", "rowmodel"],
                    help="setjoin = two-stream (wide + set); rowmodel = unified hierarchy over wide rows")
    ap.add_argument("--m-rows", type=int, default=128, help="rowmodel: wide rows per seed (cap)")
    ap.add_argument("--cells-per-row", type=int, default=48, help="rowmodel: cells per wide row (cap)")
    ap.add_argument("--n-cell-layers", type=int, default=2, help="rowmodel: cell-encoder MoE layers")
    ap.add_argument("--n-row-layers", type=int, default=2, help="rowmodel: row-encoder MoE layers")
    ap.add_argument("--row-aggregate", default="mean", choices=["mean", "slot"],
                    help="rowmodel: rows->prediction pool (mean default; slot = count-aware measure pool)")
    ap.add_argument("--use-counts", action="store_true",
                    help="rowmodel: concat w_cnt(log1p(child_counts)) at the head (count-matched ablation)")
    ap.add_argument("--agg-slots", type=int, default=4, help="rowmodel: slot-aggregate query count")
    ap.add_argument("--cell-slots", default="8,2",
                    help="rowmodel: cell->row cascade widths (C->k1->k2->1), comma-separated")
    ap.add_argument("--row-pool", default="slot", choices=["slot", "gated"],
                    help="rowmodel: cell->row pool (slot = cascade measure pool; gated = the RowPool control)")
    ap.add_argument("--checkpoint-cells", action="store_true",
                    help="rowmodel: gradient-checkpoint the cell MoE layers (memory; SLURM path)")
    ap.add_argument("--fanout", type=int, default=64, help="children sampled per o2m relation at hop 1")
    ap.add_argument("--fanout2", type=int, default=0,
                    help="hop-2 o2m fanout (0 = off; >0 adds grandchild/sibling/co-child elements)")
    ap.add_argument("--per-relation-cap", type=int, default=None,
                    help="keep only the N most-recent elements per (hop, table, FK role) group")
    ap.add_argument("--d-ff", type=int, default=None, help="FFN width (default 2*d_model)")
    ap.add_argument("--diff", default=None, metavar="DIR2",
                    help="print per-task deltas: --out-dir results vs DIR2 results")
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
    if args.list or args.index is not None or args.aggregate or args.compare or args.diff:
        return _gate(args)
    log.error("pass --dry-run, --train, --list, --index N, --aggregate, or --compare")
    return 2


def _build(dataset: str):
    from gloss.data.graph import build_gloss_graph
    from gloss.utils.paths import graph_cache_dir

    return build_gloss_graph(dataset, cache_dir=str(graph_cache_dir(dataset)))


def _readout_kwargs(args) -> dict:
    """The slot/measure readout model_kwargs (parallel to route_on); readout=pma keeps current behavior."""
    return dict(readout=args.readout, slot_mode=args.slot_mode, slot_group_key=args.slot_group_key,
                slot_iters=args.slot_iters, slot_gamma_init=args.slot_gamma_init,
                slot_gamma_learnable=args.slot_gamma_learnable, slot_seed_cond=args.slot_seed_cond,
                readout_channels=tuple(c for c in args.readout_channels.split(",") if c))


def _cell_slots(spec: str) -> tuple[int, ...]:
    return tuple(int(w) for w in spec.split(",") if w.strip())


def _model_kwargs(args) -> dict:
    """The constructor kwargs for the chosen model (SetJoin vs RowModel have disjoint knobs)."""
    if args.model == "rowmodel":
        return dict(d_model=args.d_model, n_cell_layers=args.n_cell_layers,
                    n_row_layers=args.n_row_layers, n_heads=args.n_heads, route_on=args.route_on,
                    num_experts=args.num_experts, k=args.k, d_sig=args.d_sig, d_ff=args.d_ff,
                    use_shared=args.use_shared, aggregate=args.row_aggregate, use_counts=args.use_counts,
                    agg_slots=args.agg_slots, row_pool=args.row_pool,
                    cell_slots=_cell_slots(args.cell_slots), checkpoint_cells=args.checkpoint_cells)
    return dict(d_model=args.d_model, n_heads=args.n_heads, n_wide_layers=args.n_wide_layers,
                n_set_layers=args.n_set_layers, n_pma=args.n_pma, route_on=args.route_on,
                num_experts=args.num_experts, k=args.k, d_sig=args.d_sig, d_ff=args.d_ff,
                n_axial_layers=args.n_axial_layers, use_shared=args.use_shared, **_readout_kwargs(args))


def _dry_run(args) -> int:
    from relbench.tasks import get_task

    from gloss.data.graph import make_loader
    from gloss.setjoin.collate import to_join_batch, to_row_set_batch
    from gloss.setjoin.paths import setjoin_neighbors

    log.info(f"building {args.dataset} graph ...")
    bundle = _build(args.dataset)
    task = get_task(args.dataset, args.task, download=False)
    nn = setjoin_neighbors(bundle, fanout=args.fanout, fanout2=args.fanout2)
    try:
        loader = make_loader(bundle, task, "train", num_neighbors=nn, batch_size=8, shuffle=False)
        raw = next(iter(loader))
        log.info("per-edge-type num_neighbors dict accepted by the sampler")
    except Exception as exc:
        log.warning(f"dict num_neighbors rejected ({exc!r}); falling back to flat [fanout, 8]")
        loader = make_loader(bundle, task, "train", num_neighbors=[args.fanout, 8],
                             batch_size=8, shuffle=False)
        raw = next(iter(loader))

    if args.model == "rowmodel":
        rb = to_row_set_batch(raw, bundle, task.entity_table, m_rows=args.m_rows,
                              cells_per_row=args.cells_per_row)
        print(rb.pretty_shapes())
        leaks = int(((rb.cell_row_time > rb.seed_time[:, None, None]) & rb.cell_is_timed
                     & rb.cell_mask).sum())
        batch = rb
    else:
        jb = to_join_batch(raw, bundle, task.entity_table, wide_len=args.wide_len,
                           set_size=args.set_size, hop2=args.fanout2 > 0,
                           per_relation_cap=args.per_relation_cap)
        print(jb.pretty_shapes())
        st = jb.seed_time.unsqueeze(1)
        leaks = int(((jb.wide_row_time > st) & jb.wide_is_timed & ~jb.wide_is_pad).sum()) + \
            int(((jb.elem_row_time > st) & jb.elem_is_timed & jb.elem_mask).sum())
        batch = jb
    print(f"leakage check (row_time > seed_time): {leaks}")

    if args.collate_only:
        return 0

    import torch

    from gloss.train.finetune import name_embeddings

    name_emb = name_embeddings(bundle, args.dataset, encoder=args.encoder, d_text=64)
    if args.model == "rowmodel":
        from gloss.setjoin.row_model import RowModel

        model = RowModel(bundle, name_emb, task.entity_table, **_model_kwargs(args))
    else:
        from gloss.setjoin.model import SetJoin

        model = SetJoin(bundle, name_emb, task.entity_table, **_model_kwargs(args))
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        logits, aux = model(batch)
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
        bundle, task, name_emb, model_kwargs=_model_kwargs(args),
        fanout=args.fanout, fanout2=args.fanout2, per_relation_cap=args.per_relation_cap,
        wide_len=args.wide_len, set_size=args.set_size,
        lambda_ortho=args.lambda_ortho,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.epochs, seed=args.seed, num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        model=args.model, m_rows=args.m_rows, cells_per_row=args.cells_per_row,
    )
    print("[setjoin] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "val" in k))
    if args.test:
        from gloss.setjoin.eval import evaluate_split

        tm = evaluate_split(module, bundle, task, "test", fanout=args.fanout, fanout2=args.fanout2,
                            per_relation_cap=args.per_relation_cap,
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
            model_kwargs=_model_kwargs(args),
            fanout=args.fanout, fanout2=args.fanout2, per_relation_cap=args.per_relation_cap,
            wide_len=args.wide_len,
            set_size=args.set_size, lambda_ortho=args.lambda_ortho,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            max_epochs=args.epochs, num_workers=args.num_workers,
            limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
            out_dir=out_dir, test=True,
            model=args.model, m_rows=args.m_rows, cells_per_row=args.cells_per_row,
        )
        return 0
    from gloss.eval.ablation import format_table, load_records

    records = load_records(out_dir)
    if args.aggregate:
        print(format_table(records, split="test",
                           baseline=runner.variant_of(_model_kwargs(args), fanout2=args.fanout2,
                                                      per_relation_cap=args.per_relation_cap,
                                                      model=args.model)))
        return 0
    if args.compare:
        print(runner.compare_table(records))
        return 0
    if args.diff:
        print(runner.diff_table(records, load_records(Path(args.diff)),
                                label_a=str(out_dir), label_b=args.diff))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
