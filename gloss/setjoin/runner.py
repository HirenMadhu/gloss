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


def variant_of(model_kwargs: dict | None, *, fanout2: int = 0,
               per_relation_cap: int | None = None) -> str:
    """Arm label: ``setjoin`` for the dense arm (back-compat with the v1/v2 records), else
    ``setjoin-<route_on>`` (e.g. ``setjoin-signature`` — the MoE method), with a ``-shared``
    suffix for the always-on shared expert (the S addition), ``-hop2`` for the hop-2 union arm,
    and ``-cap<N>`` for the per-relation recency cap."""
    mk = model_kwargs or {}
    route_on = mk.get("route_on", "signature")
    if route_on == "dense":
        label = "setjoin"
    else:
        label = f"setjoin-{route_on}" + ("-shared" if mk.get("use_shared") else "")
    if fanout2 > 0:
        label += "-hop2"
    if per_relation_cap is not None:
        label += f"-cap{per_relation_cap}"
    return label


def run_config(
    index: int,
    *,
    seeds: int = 3,
    encoder: str = "harrier",
    model_kwargs: dict | None = None,
    fanout: int = 64,
    fanout2: int = 0,
    wide_len: int = 128,
    set_size: int = 128,
    lambda_ortho: float = 0.5,
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
    variant = variant_of(model_kwargs, fanout2=fanout2)
    out_dir = out_dir or RESULTS
    out_path = out_dir / f"{index:04d}_{variant}.json"
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
                fanout=fanout, fanout2=fanout2, wide_len=wide_len, set_size=set_size,
                lambda_ortho=lambda_ortho,
                batch_size=bs, lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                seed=c["seed"], num_workers=num_workers,
                limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    mk = model_kwargs or {}
    rec = {**c, "variant": variant, "task_type": kind, "batch_size": bs,
           "fanout": fanout, "fanout2": fanout2, "wide_len": wide_len, "set_size": set_size,
           "route_on": mk.get("route_on", "signature"),
           "num_experts": mk.get("num_experts", 4),
           "n_axial_layers": mk.get("n_axial_layers", 0),
           "use_shared": bool(mk.get("use_shared", False)),
           "d_model": mk.get("d_model", 128), "d_ff": mk.get("d_ff"),
           "n_wide_layers": mk.get("n_wide_layers", 2), "n_set_layers": mk.get("n_set_layers", 2),
           "n_params": sum(p.numel() for p in module.model.parameters()),
           "lambda_ortho": lambda_ortho}
    rec.update({f"val_{k.split('/')[-1]}": v for k, v in metrics.items() if k.startswith("val/")})
    if test:
        try:
            tm = evaluate_split(module, bundle, task, "test", fanout=fanout, fanout2=fanout2,
                                wide_len=wide_len, set_size=set_size, batch_size=bs,
                                num_workers=num_workers)
            rec.update({f"test_{k}": v for k, v in tm.items()})
            attach_nmae(rec, target_stats(task)[1])
        except Exception as exc:               # never lose the trained val metrics to a test-eval failure
            rec["test_error"] = repr(exc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec))
    return rec


def diff_table(records_a: list[dict], records_b: list[dict],
               label_a: str = "A", label_b: str = "B") -> str:
    """Per-task seed-mean deltas between two gate result sets (A − B for AUROC, B − A would flip
    sign; regression deltas are NMAE so NEGATIVE = A better). Each side may hold one variant only."""
    from ..eval.ablation import _variant_of

    def side(records):
        vs = {_variant_of(r) for r in records}
        if len(vs) > 1:
            raise ValueError(f"records mix variants {sorted(vs)}")
        agg = aggregate(records, keys=("test_roc_auc", "test_nmae"))
        ttypes = {(r["dataset"], r["task"]): r.get("task_type", "binary") for r in records}
        return vs.pop() if vs else "?", agg, ttypes

    va, agg_a, tt_a = side(records_a)
    vb, agg_b, tt_b = side(records_b)
    lines = [f"{label_a} = {va}  vs  {label_b} = {vb}   (Δ = {label_a} − {label_b}; "
             "AUROC↑: positive Δ better, NMAE↓: negative Δ better)", ""]
    for (ds, tk) in sorted(set(tt_a) | set(tt_b)):
        binary = tt_a.get((ds, tk), tt_b.get((ds, tk))) == "binary"
        key, scale = ("test_roc_auc", 100.0) if binary else ("test_nmae", 1.0)
        ma = agg_a.get((ds, tk, va), {}).get(key, (float("nan"),) * 4)[0] * scale
        mb = agg_b.get((ds, tk, vb), {}).get(key, (float("nan"),) * 4)[0] * scale
        unit = "AUROC" if binary else "NMAE"
        lines.append(f"{ds + ' / ' + tk:34s} {unit:5s} {ma:8.3f} {mb:8.3f}  Δ={ma - mb:+7.3f}")
    return "\n".join(lines)


def compare_table(records: list[dict], baselines_path: Path | str | None = None,
                  variant: str | None = None) -> str:
    """SetJoin (seed means) vs RT-from-scratch / GelGT / MoRE grid best, per leaderboard task.

    Binary: AUROC × 100 (higher better). Regression: NMAE (lower better), read from the run-time
    ``test_nmae`` field. ``variant`` picks the arm when a results dir mixes several (``setjoin`` =
    dense, ``setjoin-signature`` = the MoE method); with a single-variant dir it is auto-detected.
    """
    if not records:
        return "(no records)"
    from ..eval.ablation import _variant_of

    present = sorted({_variant_of(r) for r in records})
    if variant is None:
        if len(present) > 1:
            raise ValueError(f"results dir mixes variants {present}; pass variant=...")
        variant = present[0]
    records = [r for r in records if _variant_of(r) == variant]
    baselines_path = Path(baselines_path or RESULTS_ROOT / "leaderboard_baselines.json")
    base = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    agg = aggregate(records, keys=("test_roc_auc", "test_nmae"))
    task_types = {(r["dataset"], r["task"]): r.get("task_type", "binary") for r in records}

    lines = [f"{len(records)} runs.  binary: AUROC x100 (higher better)  |  "
             "regression: NMAE (lower better)", ""]
    # column names: "single table" = SetJoin (this method), "multi table" = MoRE grid best (the
    # repo's cell-token relational method), RT = RT (from scratch), GelGT = task-specific SOTA.
    # Per row the BEST method is bold and the SECOND BEST underlined (ANSI; paper convention).
    BOLD, UL, END = "\033[1m", "\033[4m", "\033[0m"
    widths = (14, 8, 8, 12)
    header = (f"{'task':34s} {'single table':>{widths[0]}s} {'RT':>{widths[1]}s} "
              f"{'GelGT':>{widths[2]}s} {'multi table':>{widths[3]}s}")
    lines += [header, "-" * len(header)]
    for (ds, tk) in sorted(task_types):
        ttype = task_types[(ds, tk)]
        binary = ttype == "binary"
        key, scale = ("test_roc_auc", 100.0) if binary else ("test_nmae", 1.0)
        mean, _sd, _ci, _n = agg.get((ds, tk, variant), {}).get(key, (float("nan"), 0, 0, 0))
        side = base.get("classification" if binary else "regression", {}).get(f"{ds} {tk}", {})
        rt = float(side["RT_from_scratch"]) if "RT_from_scratch" in side else float("nan")
        gg = float(side["GelGT"]) if "GelGT" in side else float("nan")
        more = MORE_GRID_BEST.get((ds, tk), float("nan"))
        ours = mean * scale
        vals = [ours, rt, gg, more]
        cells = [f"{v:>{w}.3f}" for v, w in zip(vals, widths)]      # pad BEFORE ANSI wrapping
        ranked = sorted(((i, v) for i, v in enumerate(vals) if v == v),
                        key=lambda t: -t[1] if binary else t[1])
        if ranked:
            i = ranked[0][0]
            cells[i] = BOLD + cells[i] + END
        if len(ranked) > 1:
            i = ranked[1][0]
            cells[i] = UL + cells[i] + END
        beats = (ours > rt) if binary else (ours < rt)
        mark = "" if ours != ours or rt != rt else (" *" if beats else "")
        lines.append(f"{ds + ' / ' + tk:34s} {' '.join(cells)}{mark}")
    lines += ["", "bold = best, underline = 2nd best; * = single table beats RT (the target baseline)"]
    return "\n".join(lines)
