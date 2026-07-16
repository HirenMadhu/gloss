"""Multi-horizon evaluation: how fast does performance decay when predicting FURTHER into the future?

RelBench defines each example as (entity, seed_time) with the label over [seed_time, seed_time +
task.timedelta). Horizon-k evaluation keeps the (held-out) labels of the split table but moves each
seed's *as-of feature time* BACK by ``k * task.timedelta``: the model must predict the window that
starts k timestamps ahead of everything it is allowed to see. ``k = 0`` is the standard evaluation
(predict the next timestamp); ``k = 9`` predicts the 10th-next. Nothing is retrained — a model trained
on the next-step objective is probed at longer ranges, so the curve isolates how much of the signal is
short-range recency vs longer-range structure.

Works for BOTH substrates: ``kind="setjoin"`` (a `SetJoinLitModule`, JoinBatch path) and
``kind="more"`` (a `MoRELitModule`, CellBatch path). Leakage safety is inherited: the sampler's as-of
filter runs on the shifted seed times, so a horizon-k prediction sees strictly less history.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.graph import GraphBundle, coerce_binary_target


def shifted_task_table(task, split: str, k: int):
    """The split's task table with seed times moved back ``k * task.timedelta`` (labels/rows/order
    unchanged, so ``input_id`` alignment and ``task.evaluate`` row order still match the original)."""
    table = coerce_binary_target(task.get_table(split), task)
    if k == 0:
        return table
    df = table.df.copy()
    df[table.time_col] = df[table.time_col] - k * task.timedelta
    return type(table)(df=df, fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                       pkey_col=table.pkey_col, time_col=table.time_col)


def make_shifted_loader(bundle: GraphBundle, task, split: str, k: int, *,
                        num_neighbors, batch_size: int = 128, num_workers: int = 0):
    """`gloss.data.graph.make_loader` with the horizon-k shifted seed times (same sampler recipe)."""
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.modeling.loader import CustomNodeLoader, NeighborSampler

    inp = get_node_train_table_input(shifted_task_table(task, split, k), task)
    sampler = NeighborSampler(bundle.data, num_neighbors=num_neighbors, time_attr="time",
                              temporal_strategy="last", disjoint=True)
    return CustomNodeLoader(bundle.data, node_sampler=sampler, input_nodes=inp.nodes,
                            input_time=inp.time, transform=inp.transform,
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def predict_shifted(module, bundle: GraphBundle, task, split: str, k: int, *, kind: str,
                    num_neighbors, batch_size: int = 128, num_workers: int = 0,
                    wide_len: int = 128, set_size: int = 128, seq_len: int = 512, max_fk: int = 5,
                    device=None) -> np.ndarray:
    """Horizon-k predictions aligned to the ORIGINAL split table rows (probability for binary,
    de-standardized value for regression). ``kind`` picks the collate/model path."""
    device = device or next(module.parameters()).device
    was_training = module.training
    module.eval()
    loader = make_shifted_loader(bundle, task, split, k, num_neighbors=num_neighbors,
                                 batch_size=batch_size, num_workers=num_workers)
    n = len(task.get_table(split).df)
    pred = np.full(n, np.nan, dtype=np.float64)
    for raw in loader:
        if kind == "setjoin":
            from .collate import to_join_batch

            b = to_join_batch(raw, bundle, task.entity_table,
                              wide_len=wide_len, set_size=set_size).to(device)
        elif kind == "more":
            from ..data.collate import to_cell_batch

            b = to_cell_batch(raw, bundle, task.entity_table,
                              seq_len=seq_len, max_fk=max_fk).to(device)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        out = module(b).float().cpu()                         # [B] per-seed raw output
        if getattr(module, "task_type", "binary") == "regression":
            out = out * module.target_std + module.target_mean
        else:
            out = torch.sigmoid(out)
        gid = raw[task.entity_table].input_id.cpu().numpy()   # per-seed split-table row index
        pred[gid] = out.numpy()
    if was_training:
        module.train()
    assert not np.isnan(pred).any(), f"{split} h{k}: {int(np.isnan(pred).sum())} seeds unpredicted"
    return pred


def horizon_curve(module, bundle: GraphBundle, task, *, kind: str, horizons=range(10),
                  split: str = "test", train_std: float | None = None, **cfg) -> dict:
    """-> ``{k: {metric: value, ...}}`` for each horizon; regression additionally gets ``nmae`` when
    ``train_std`` is given. Labels come from the ORIGINAL (unshifted) split table for every k."""
    target_table = coerce_binary_target(task.get_table(split, mask_input_cols=False), task)
    out: dict[int, dict] = {}
    for k in horizons:
        pred = predict_shifted(module, bundle, task, split, k, kind=kind, **cfg)
        m = {kk: float(v) for kk, v in task.evaluate(pred, target_table).items()}
        if train_std is not None and "mae" in m:
            m["nmae"] = m["mae"] / max(float(train_std), 1e-6)
        out[int(k)] = m
    return out


# ---------------------------------------------------------------------------------------------
# gate-style runner: 9 leaderboard tasks x seeds x {setjoin, more}, one cell per SLURM array task
# ---------------------------------------------------------------------------------------------

HORIZONS = tuple(range(10))                    # k = 0 (next timestamp) .. 9 (10th-next)

# fixed training configs (one per substrate; NOT tuned per task):
# - setjoin: the v4 gate arm (signature MoE + one axial row/column block)
# - more:    the grid-search modal winner family (d128 x 8 blocks, 8 experts, signature router)
SETJOIN_CFG = dict(d_model=128, n_heads=4, n_wide_layers=2, n_set_layers=2, n_pma=4,
                   route_on="signature", num_experts=4, k=2, d_sig=64, n_axial_layers=1)
MORE_CFG = dict(d_model=128, n_blocks=8, n_heads=4, d_ff=512, enc_channels=128,
                d_sig=128, num_experts=8, k=2)


def horizon_grid(seeds: int = 3) -> list[dict]:
    from ..eval.ablation import build_grid, dataset_tasks
    from .runner import GATE_DATASETS

    return build_grid(dataset_tasks(GATE_DATASETS, ["leaderboard"]), seeds,
                      signals=("setjoin", "more"))


def run_config_horizon(
    index: int,
    *,
    seeds: int = 3,
    encoder: str = "harrier",
    fanout: int = 64,
    wide_len: int = 128,
    set_size: int = 128,
    batch_size: int = 128,
    more_batch_size: int = 64,
    max_epochs: int = 30,
    num_workers: int = 8,
    out_dir=None,
    limit_train_batches=None,
    limit_val_batches=None,
) -> dict:
    """Train ONE (dataset, task, model, seed) cell with its substrate's fixed config, then evaluate
    the full horizon curve k=0..9 on TEST. Idempotent (skip-if-done); OOM halves the batch."""
    import json
    from pathlib import Path

    from ..eval.ablation import RESULTS_ROOT
    from ..train.finetune import name_embeddings, target_stats, task_kind
    from ..utils.paths import graph_cache_dir

    grid = horizon_grid(seeds)
    if index >= len(grid):
        print(f"index {index} >= grid size {len(grid)}; nothing to do")
        return {}
    c = grid[index]
    model_kind = c["signal"]
    out_dir = Path(out_dir or RESULTS_ROOT / "horizon")
    out_path = out_dir / f"{index:04d}_horizon_{model_kind}.json"
    if out_path.exists():
        print(f"{out_path} exists; skipping")
        return json.loads(out_path.read_text())

    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph

    bundle = build_gloss_graph(c["dataset"], cache_dir=str(graph_cache_dir(c["dataset"])))
    task = get_task(c["dataset"], c["task"], download=False)
    d_text = 64 if encoder == "hash" else 2560
    name_emb = name_embeddings(bundle, c["dataset"], encoder=encoder, d_text=d_text)
    kind = task_kind(task)
    std = target_stats(task)[1] if kind == "regression" else None

    def _train(bs):
        if model_kind == "setjoin":
            from .paths import setjoin_neighbors
            from .train import train_prebuilt_setjoin

            module, _ = train_prebuilt_setjoin(
                bundle, task, name_emb, model_kwargs=dict(SETJOIN_CFG),
                fanout=fanout, wide_len=wide_len, set_size=set_size, batch_size=bs,
                max_epochs=max_epochs, seed=c["seed"], num_workers=num_workers,
                limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
            cfg = dict(num_neighbors=setjoin_neighbors(bundle, fanout), batch_size=bs,
                       wide_len=wide_len, set_size=set_size, num_workers=num_workers)
        else:
            from ..train.finetune import train_prebuilt

            module, _ = train_prebuilt(
                bundle, task, name_emb, model_kwargs=dict(MORE_CFG), route_on="signature",
                num_neighbors=[12, 12], seq_len=512, max_fk=5, batch_size=bs,
                max_epochs=max_epochs, seed=c["seed"], num_workers=num_workers,
                limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
            cfg = dict(num_neighbors=[12, 12], batch_size=bs, seq_len=512, max_fk=5,
                       num_workers=num_workers)
        return module, cfg

    bs = batch_size if model_kind == "setjoin" else more_batch_size
    while True:
        try:
            module, cfg = _train(bs)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    curves = horizon_curve(module, bundle, task, kind=model_kind, horizons=HORIZONS,
                           train_std=std, **cfg)
    rec = {**c, "model": model_kind, "task_type": kind, "batch_size": bs,
           "horizons": {str(k): v for k, v in curves.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec))
    return rec


def plot_horizon_curves(records: list[dict], out_png) -> str:
    """3x3 grid of horizon curves (x = steps ahead 1..10, y = AUROC x100 or NMAE), one line per
    substrate (seed mean, +-95% CI band). Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({(r["dataset"], r["task"]) for r in records})
    labels = {"setjoin": "single table", "more": "multi table"}
    colors = {"setjoin": "tab:blue", "more": "tab:orange"}
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for ax, (ds, tk) in zip(axes.flat, tasks):
        recs = [r for r in records if (r["dataset"], r["task"]) == (ds, tk)]
        ttype = recs[0]["task_type"]
        metric = "roc_auc" if ttype == "binary" else "nmae"
        scale = 100.0 if ttype == "binary" else 1.0
        for model in ("setjoin", "more"):
            runs = [r for r in recs if r["model"] == model]
            if not runs:
                continue
            ks = sorted(int(k) for k in runs[0]["horizons"])
            ys = np.array([[r["horizons"][str(k)].get(metric, np.nan) * scale for k in ks]
                           for r in runs])
            mean, n = np.nanmean(ys, 0), ys.shape[0]
            ci = 1.96 * np.nanstd(ys, 0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            x = np.array(ks) + 1                            # "steps ahead": k=0 -> 1
            ax.plot(x, mean, marker="o", ms=3, color=colors[model], label=labels[model])
            ax.fill_between(x, mean - ci, mean + ci, color=colors[model], alpha=0.15, lw=0)
        ax.set_title(f"{ds} / {tk}", fontsize=10)
        ax.set_xlabel("timestamps ahead")
        ax.set_ylabel("AUROC x100 (higher=better)" if ttype == "binary" else "NMAE (lower=better)",
                      fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Forecast horizon vs TEST performance (features as-of k timestamps before the label window)",
                 fontsize=11)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)
