"""The routing-signal ablation: the SAME MoRE encoder under each routing arm.

    signature  : route the MoE on the value-free relational signature (THE method)
    hidden     : route on the block's evolving hidden state (leak-free; weaker transfer)
    value      : route on the cell's value component
    identity   : route on a learned per-column id embedding (cannot transfer)
    dense      : plain RT, no MoE (the in-distribution control)
    dense_wide : param-matched dense (d_ff x k) — to show gains aren't just parameters

Headline claim: ``signature >= value/dense`` in-distribution (and ``signature >> identity`` on held-out
schemas, a later phase). Honesty: the dense-combine MVP runs every expert on every token, so a MoE arm
costs ~M x FFN FLOPs (not k x). We therefore report vanilla ``dense`` AND param-matched ``dense_wide``
and do **not** claim active-FLOP parity (that needs sparse dispatch).

Each ``(dataset, task, signal, seed)`` is an independent run (one SLURM array task) that writes one JSON;
``aggregate``/``format_table`` reduce them to per-``(dataset, task)`` tables (seed mean ± 95% CI) of the
task's primary metric, with the lift of each signal over ``dense``.
"""
from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path

ROUTING_SIGNALS = ("signature", "hidden", "value", "identity", "dense", "dense_wide")
DEFAULT_DATASETS = ("rel-f1", "rel-stack", "rel-trial")
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
RESULTS = RESULTS_ROOT / "ablation"

# primary (headline) metric per task type and its direction (RelBench metric key names).
PRIMARY = {"binary": "roc_auc", "regression": "mae"}
LOWER_BETTER = {"mae", "rmse", "logloss"}


def entity_tasks(dataset: str) -> list[str]:
    """Sorted RelBench entity tasks for ``dataset`` — binary + regression only (link tasks excluded)."""
    from relbench.base import TaskType
    from relbench.tasks import get_task, get_task_names

    keep = (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION)
    out = []
    for name in sorted(get_task_names(dataset)):
        try:
            task = get_task(dataset, name, download=False)
        except Exception:
            continue
        if task.task_type in keep:
            out.append(name)
    return out


def dataset_tasks(datasets) -> dict[str, list[str]]:
    return {ds: entity_tasks(ds) for ds in datasets}


def build_grid(dataset_tasks_map: dict, seeds: int, signals=ROUTING_SIGNALS) -> list[dict]:
    """The full ``(dataset, task, signal, seed)`` grid — the SLURM array iterates this by index."""
    return [
        {"dataset": ds, "task": t, "signal": sig, "seed": s}
        for ds, tasks in dataset_tasks_map.items()
        for t in tasks
        for sig in signals
        for s in range(seeds)
    ]


def enumerate_configs(datasets, seeds: int, signals=ROUTING_SIGNALS) -> list[dict]:
    return build_grid(dataset_tasks(datasets), seeds, signals)


def run_config(
    index: int,
    *,
    datasets=DEFAULT_DATASETS,
    seeds: int,
    signals=ROUTING_SIGNALS,
    encoder: str = "qwen",
    d_text: int = 2560,
    model_kwargs: dict | None = None,
    num_neighbors: list[int] | None = None,
    seq_len: int = 512,
    max_fk: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 10,
    num_workers: int = 8,
    lambda_ortho: float = 0.5,
    num_experts: int = 4,
    k: int = 2,
    out_dir: Path | None = None,
    test: bool = True,
) -> dict:
    """Train the single ``(dataset, task, signal, seed)`` config at ``index`` and persist its metrics.

    Records validation metrics (per task type) and, if ``test``, the held-out RelBench **test** score via
    ``task.evaluate`` (leaderboard-comparable) — the gate saves no checkpoints, so test eval reuses the
    in-memory module."""
    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph
    from ..train.finetune import name_embeddings, task_kind, train_prebuilt

    c = build_grid(dataset_tasks(datasets), seeds, signals)[index]
    bundle = build_gloss_graph(c["dataset"])
    task = get_task(c["dataset"], c["task"], download=False)
    name_emb = name_embeddings(bundle, c["dataset"], encoder=encoder, d_text=d_text)
    mk = dict(model_kwargs or {})
    mk.update(num_experts=num_experts, k=k)
    module, metrics = train_prebuilt(
        bundle, task, name_emb, model_kwargs=mk, route_on=c["signal"], lambda_ortho=lambda_ortho,
        num_neighbors=num_neighbors, seq_len=seq_len, max_fk=max_fk, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, seed=c["seed"], num_workers=num_workers,
    )
    rec = {**c, "task_type": task_kind(task)}
    rec.update({f"val_{kk.split('/')[-1]}": v for kk, v in metrics.items() if kk.startswith("val/")})
    if test:
        from .test_eval import evaluate_split

        tm = evaluate_split(module, bundle, task, "test", num_neighbors=num_neighbors,
                            seq_len=seq_len, max_fk=max_fk, batch_size=batch_size, num_workers=num_workers)
        rec.update({f"test_{kk}": v for kk, v in tm.items()})
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{index:04d}.json").write_text(json.dumps(rec))
    return rec


def _agg(rows: list[dict], key: str) -> tuple[float, float, float, int]:
    """(mean, std, 95% CI half-width, n) over the present, non-NaN values of ``key``."""
    xs = [r[key] for r in rows
          if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    if not xs:
        return float("nan"), 0.0, 0.0, 0
    mean = stats.mean(xs)
    sd = stats.stdev(xs) if len(xs) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return mean, sd, ci, len(xs)


def aggregate(records: list[dict], keys: tuple[str, ...]) -> dict[tuple, dict]:
    """-> ``{(dataset, task, signal): {metric: (mean, std, ci, n)}}`` for each metric in ``keys``."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["dataset"], r["task"], r["signal"]), []).append(r)
    return {gk: {key: _agg(rows, key) for key in keys} for gk, rows in groups.items()}


def load_records(out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or RESULTS
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def format_table(records: list[dict], *, split: str = "test") -> str:
    """Per-``(dataset, task)`` table of the primary metric (seed mean ± 95% CI), lift of each arm vs dense."""
    if not records:
        return "(no records)"
    info: dict[tuple, dict] = {}
    for r in records:
        dt = (r["dataset"], r["task"])
        info.setdefault(dt, {"type": r.get("task_type", "binary"), "signals": set()})
        info[dt]["signals"].add(r["signal"])
    lines = [f"{len(records)} runs collected.", ""]
    for dt in sorted(info):
        ds, task = dt
        ttype = info[dt]["type"]
        metric = PRIMARY.get(ttype, "roc_auc")
        key = f"{split}_{metric}"
        lower = metric in LOWER_BETTER
        agg = aggregate([r for r in records if (r["dataset"], r["task"]) == dt], keys=(key,))
        dense = agg.get((ds, task, "dense"), {}).get(key, (float("nan"),))[0]
        lines.append(f"=== {ds} / {task}  ({ttype}; {split} {metric} {'↓' if lower else '↑'}) ===")
        for sig in ROUTING_SIGNALS:
            gk = (ds, task, sig)
            if gk not in agg:
                continue
            m, _sd, ci, n = agg[gk][key]
            lift = (dense - m) if lower else (m - dense)        # positive lift = better than dense
            lines.append(f"  {sig:11s} {m:9.4f} ± {ci:.4f} (n={n})   Δvs dense={lift:+.4f}")
        lines.append("")
    return "\n".join(lines)
