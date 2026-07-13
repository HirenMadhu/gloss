"""The SetJoin gate runner: one (dataset, task, seed) per SLURM array task, + the compare table.

Grid = the 9 RT-reported leaderboard entity tasks (rel-f1, rel-trial, rel-event) × seeds, signal
``"setjoin"`` — reusing `eval/ablation.py`'s grid/aggregate bookkeeping so records round-trip through
`aggregate`/`format_table` unchanged. Records store **`test_nmae` = test_mae / train-std at run time**
(the leaderboard regression metric; the pipeline's `task.evaluate` reports raw MAE). `compare_table`
prints SetJoin vs RT-from-scratch / GelGT (`results/leaderboard_baselines.json`) and the MoRE
grid-search best (partial arch sweep, PROGRESS/recap §3) as a reference.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..eval.ablation import RESULTS_ROOT, aggregate, build_grid, dataset_tasks

GATE_DATASETS = ("rel-f1", "rel-trial", "rel-event")
RESULTS = RESULTS_ROOT / "setjoin_v1"

# MoRE (signature router) best per task from the PARTIAL architecture grid search (872/2592 configs,
# some winners 1-2 seeds — winner's-curse caveat). Display-only reference for compare_table.
MORE_GRID_BEST = {
    ("rel-f1", "driver-dnf"): 82.9, ("rel-f1", "driver-top3"): 90.6,
    ("rel-trial", "study-outcome"): 69.4, ("rel-event", "user-repeat"): 79.5,
    ("rel-event", "user-ignore"): 87.3,
    ("rel-f1", "driver-position"): 0.395, ("rel-trial", "study-adverse"): 0.161,
    ("rel-trial", "site-success"): 0.840, ("rel-event", "user-attendance"): 0.399,
}


def gate_grid(seeds: int = 3) -> list[dict]:
    return build_grid(dataset_tasks(GATE_DATASETS, ["leaderboard"]), seeds, signals=("setjoin",))


def attach_nmae(rec: dict, train_std: float) -> dict:
    """Store the leaderboard regression metric at run time (raw MAE / TRAIN-split target std)."""
    if rec.get("task_type") == "regression" and rec.get("test_mae") is not None:
        rec["test_nmae"] = rec["test_mae"] / max(float(train_std), 1e-6)
    return rec


def run_config(
    index: int,
    *,
    seeds: int = 3,
    encoder: str = "harrier",
    model_kwargs: dict | None = None,
    fanout: int = 64,
    wide_len: int = 128,
    set_size: int = 128,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 30,
    num_workers: int = 8,
    out_dir: Path | None = None,
    test: bool = True,
    limit_train_batches: float | int | None = None,
    limit_val_batches: float | int | None = None,
) -> dict:
    """Train the single (dataset, task, seed) gate cell at ``index`` and persist its metrics."""
    import torch
    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph
    from ..train.finetune import name_embeddings, target_stats, task_kind
    from ..utils.paths import graph_cache_dir
    from .eval import evaluate_split
    from .train import train_prebuilt_setjoin

    grid = gate_grid(seeds)
    if index >= len(grid):
        print(f"index {index} >= grid size {len(grid)}; nothing to do")
        return {}
    c = grid[index]
    out_dir = out_dir or RESULTS
    out_path = out_dir / f"{index:04d}_setjoin.json"
    if out_path.exists():                                  # idempotent resubmits (gridsearch precedent)
        print(f"{out_path} exists; skipping")
        return json.loads(out_path.read_text())

    bundle = build_gloss_graph(c["dataset"], cache_dir=str(graph_cache_dir(c["dataset"])))
    task = get_task(c["dataset"], c["task"], download=False)
    d_text = 64 if encoder == "hash" else 2560
    name_emb = name_embeddings(bundle, c["dataset"], encoder=encoder, d_text=d_text)
    kind = task_kind(task)

    bs = batch_size
    while True:                                            # halve on CUDA OOM (floor 8)
        try:
            module, metrics = train_prebuilt_setjoin(
                bundle, task, name_emb, model_kwargs=model_kwargs,
                fanout=fanout, wide_len=wide_len, set_size=set_size, batch_size=bs,
                lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, seed=c["seed"],
                num_workers=num_workers,
                limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    rec = {**c, "variant": "setjoin", "task_type": kind, "batch_size": bs,
           "fanout": fanout, "wide_len": wide_len, "set_size": set_size}
    rec.update({f"val_{k.split('/')[-1]}": v for k, v in metrics.items() if k.startswith("val/")})
    if test:
        try:
            tm = evaluate_split(module, bundle, task, "test", fanout=fanout, wide_len=wide_len,
                                set_size=set_size, batch_size=bs, num_workers=num_workers)
            rec.update({f"test_{k}": v for k, v in tm.items()})
            attach_nmae(rec, target_stats(task)[1])
        except Exception as exc:               # never lose the trained val metrics to a test-eval failure
            rec["test_error"] = repr(exc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec))
    return rec


def compare_table(records: list[dict], baselines_path: Path | str | None = None) -> str:
    """SetJoin (seed mean ± 95% CI) vs RT-from-scratch / GelGT / MoRE grid best, per leaderboard task.

    Binary: AUROC × 100 (higher better). Regression: NMAE (lower better), read from the run-time
    ``test_nmae`` field.
    """
    if not records:
        return "(no records)"
    baselines_path = Path(baselines_path or RESULTS_ROOT / "leaderboard_baselines.json")
    base = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    agg = aggregate(records, keys=("test_roc_auc", "test_nmae"))
    task_types = {(r["dataset"], r["task"]): r.get("task_type", "binary") for r in records}

    lines = [f"{len(records)} runs.  binary: AUROC x100 (higher better)  |  "
             "regression: NMAE (lower better)", ""]
    header = f"{'task':34s} {'SetJoin':>16s} {'RT(scratch)':>12s} {'GelGT':>8s} {'MoRE best':>10s}"
    lines += [header, "-" * len(header)]
    for (ds, tk) in sorted(task_types):
        ttype = task_types[(ds, tk)]
        binary = ttype == "binary"
        key, scale = ("test_roc_auc", 100.0) if binary else ("test_nmae", 1.0)
        mean, _sd, ci, n = agg.get((ds, tk, "setjoin"), {}).get(key, (float("nan"), 0, 0, 0))
        side = base.get("classification" if binary else "regression", {}).get(f"{ds} {tk}", {})
        rt = float(side["RT_from_scratch"]) if "RT_from_scratch" in side else float("nan")
        gg = float(side["GelGT"]) if "GelGT" in side else float("nan")
        more = MORE_GRID_BEST.get((ds, tk), float("nan"))
        ours = mean * scale
        beats = (ours > rt) if binary else (ours < rt)
        mark = "" if ours != ours or rt != rt else (" *" if beats else "")
        lines.append(f"{ds + ' / ' + tk:34s} {ours:10.3f}±{ci * scale:.3f}(n={n})"
                     f" {rt:>12.3f} {gg:>8.3f} {more:>10.3f}{mark}")
    lines += ["", "* = beats RT (from scratch), the target baseline"]
    return "\n".join(lines)
