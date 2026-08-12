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

ROUTING_SIGNALS = ("signature", "dense")
# Display order for the tables — the six canonical arms plus hybrid (=signature+hidden). Architecture-
# addition variants (e.g. ``signature+S``) sort after these, alphabetically.
ROUTER_DISPLAY = ("dense", "signature")
DEFAULT_DATASETS = ("rel-f1", "rel-stack", "rel-trial")
# The 9 entity tasks RT (from scratch) / GelGT report on the RelBench leaderboard (the only ones with an
# external baseline to compare against). `--tasks leaderboard` restricts a run to these.
LEADERBOARD_TASKS = {
    "rel-f1": ("driver-dnf", "driver-top3", "driver-position"),
    "rel-trial": ("study-outcome", "study-adverse", "site-success"),
    "rel-event": ("user-repeat", "user-ignore", "user-attendance"),
}
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
RESULTS = RESULTS_ROOT / "ablation"


def variant_label(route_on: str, *, use_shared=False, cosine=False, top_p=None, hmoe=False) -> str:
    """A stable arm label that distinguishes architecture-addition runs sharing one ``route_on``.

    Base arms keep their bare name (``signature``, ``dense``, …); additions append S/C/P/H tags
    (``signature+S``, ``signature+SCPH``) so the ablation runner can put several addition configs of the
    *same* router in one out-dir without them colliding under a single ``route_on`` in ``aggregate``.
    """
    tags = "".join(t for t, on in
                   (("S", use_shared), ("C", cosine), ("P", top_p is not None), ("H", hmoe)) if on)
    return f"{route_on}+{tags}" if tags else route_on


def _variant_of(rec: dict) -> str:
    """The grouping label for a record — its ``variant`` if present, else the bare ``signal`` (back-compat)."""
    return rec.get("variant") or rec["signal"]

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


def dataset_tasks(datasets, tasks=None) -> dict[str, list[str]]:
    """Entity tasks per dataset (sorted). ``tasks`` optionally filters: a list of task names to keep, or
    the literal ``["leaderboard"]`` to keep only the RT-reported leaderboard tasks (``LEADERBOARD_TASKS``).
    Filtering preserves ``entity_tasks`` order so the grid index stays stable."""
    if tasks and list(tasks) == ["leaderboard"]:
        keep_by_ds = {ds: set(LEADERBOARD_TASKS.get(ds, ())) for ds in datasets}
        return {ds: [t for t in entity_tasks(ds) if t in keep_by_ds[ds]] for ds in datasets}
    keep = set(tasks) if tasks else None
    return {ds: [t for t in entity_tasks(ds) if keep is None or t in keep] for ds in datasets}


def build_grid(dataset_tasks_map: dict, seeds: int, signals=ROUTING_SIGNALS) -> list[dict]:
    """The full ``(dataset, task, signal, seed)`` grid — the SLURM array iterates this by index."""
    return [
        {"dataset": ds, "task": t, "signal": sig, "seed": s}
        for ds, tasks in dataset_tasks_map.items()
        for t in tasks
        for sig in signals
        for s in range(seeds)
    ]


def enumerate_configs(datasets, seeds: int, signals=ROUTING_SIGNALS, tasks=None) -> list[dict]:
    return build_grid(dataset_tasks(datasets, tasks), seeds, signals)


#: Keys every result JSON must carry so a finished run can be checked against the config it was meant
#: to run. Do not shrink this list.
FINGERPRINT_KEYS = ("encoder", "d_text", "d_model", "n_blocks", "n_heads", "d_ff", "enc_channels",
                    "num_experts", "k", "lr", "max_epochs", "batch_size", "seq_len")


def run_fingerprint(*, encoder, d_text, model_kwargs, num_experts, k, lr, max_epochs,
                    batch_size, seq_len) -> dict:
    """The run's identity, to be merged into its result record alongside the scores.

    Records used to carry `(dataset, task, signal, seed, arch)` and metrics only — no `encoder`, no
    model shape. That is not a cosmetic gap: it means a finished result cannot be checked against the
    config it was supposed to run, so you have to trust the submit line. Trusting the submit line is
    exactly how two grid arrays reached completion on the wrong architecture AND the wrong encoder
    before anyone read a result back (`amendments.md` §9.3), and why `results/two_level_full/` can
    only be shown to be qwen by inference rather than by record.

    `run_gridsearch.py` writes the same fields; keep the two runners in step.
    """
    mk = model_kwargs or {}
    out = {"encoder": encoder, "d_text": d_text,
           "num_experts": num_experts, "k": k, "lr": lr,
           "max_epochs": max_epochs, "batch_size": batch_size, "seq_len": seq_len}
    for key in ("d_model", "n_blocks", "n_heads", "d_ff", "enc_channels"):
        out[key] = mk.get(key)
    assert set(out) == set(FINGERPRINT_KEYS), "fingerprint drifted from FINGERPRINT_KEYS"
    return out


def run_config(
    index: int,
    *,
    datasets=DEFAULT_DATASETS,
    seeds: int,
    signals=ROUTING_SIGNALS,
    tasks=None,
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
    use_shared: bool = False,
    cosine: bool = False,
    tau: float = 0.3,
    top_p: float | None = None,
    hmoe: bool = False,
    n_groups: int = 4,
    experts_per_group: int = 2,
    k2: int = 1,
    out_dir: Path | None = None,
    test: bool = True,
    limit_train_batches: float | int | None = None,
    limit_val_batches: float | int | None = None,
    arch: str = "rt",
    two_level: dict | None = None,
) -> dict:
    """Train the single ``(dataset, task, signal, seed)`` config at ``index`` and persist its metrics.

    Records validation metrics (per task type) and, if ``test``, the held-out RelBench **test** score via
    ``task.evaluate`` (leaderboard-comparable) — the gate saves no checkpoints, so test eval reuses the
    in-memory module."""
    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph
    from ..train.finetune import name_embeddings, task_kind, train_prebuilt
    from ..utils.paths import graph_cache_dir

    grid = build_grid(dataset_tasks(datasets, tasks), seeds, signals)
    if index >= len(grid):
        print(f"index {index} >= grid size {len(grid)}; nothing to do")
        return {}
    c = grid[index]
    graph_cache = str(graph_cache_dir(c["dataset"]))
    bundle = build_gloss_graph(c["dataset"], cache_dir=graph_cache)
    task = get_task(c["dataset"], c["task"], download=False)
    name_emb = name_embeddings(bundle, c["dataset"], encoder=encoder, d_text=d_text)
    mk = dict(model_kwargs or {})
    mk.update(num_experts=num_experts, k=k, use_shared=use_shared, cosine=cosine, tau=tau, top_p=top_p,
              hmoe=hmoe, n_groups=n_groups, experts_per_group=experts_per_group, k2=k2)
    if arch == "two_level":
        # P0.4's table/role name tables come from the SAME frozen encoder + cache as the column table,
        # so they cost nothing here (content-hash keyed) and stay name-derived => an unseen schema works.
        from ..text.schema import build_table_name_embeddings, role_name_embeddings_with_none
        from ..train.finetune import _name_encoder

        enc = _name_encoder(c["dataset"], encoder=encoder, d_text=d_text)
        mk.update(
            arch="two_level",
            table_name_emb=build_table_name_embeddings(bundle, enc, kind="query"),
            role_name_emb=role_name_embeddings_with_none(bundle, enc, kind="query"),
            **(two_level or {}),
        )
    module, metrics = train_prebuilt(
        bundle, task, name_emb, model_kwargs=mk, route_on=c["signal"], lambda_ortho=lambda_ortho,
        num_neighbors=num_neighbors, seq_len=seq_len, max_fk=max_fk, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, seed=c["seed"], num_workers=num_workers,
        limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
    )
    # variant = router arm + S/C/P/H tags, so several addition configs of one router don't collide in a
    # shared out-dir (dense/dense_wide carry no additions -> variant == signal).
    variant = variant_label(c["signal"], use_shared=use_shared, cosine=cosine, top_p=top_p, hmoe=hmoe)
    if arch != "rt":
        # arch goes in the VARIANT, not just a field: the variant is the filename key and the aggregate
        # grouping key, so without this a two_level run would overwrite / be averaged together with an
        # RT run of the same (index, router). They are different architectures and must never merge.
        variant = f"{variant}@{arch}"
    rec = {**c, "variant": variant, "arch": arch, "task_type": task_kind(task),
           **run_fingerprint(encoder=encoder, d_text=d_text, model_kwargs=mk, num_experts=num_experts,
                             k=k, lr=lr, max_epochs=max_epochs, batch_size=batch_size,
                             seq_len=seq_len)}
    if arch == "two_level":
        # self-describing results: which phase switches produced this number
        rec["two_level"] = {kk: vv for kk, vv in (two_level or {}).items()
                            if not kk.endswith("_emb")}
    rec.update({f"val_{kk.split('/')[-1]}": v for kk, v in metrics.items() if kk.startswith("val/")})
    if test:
        from .test_eval import evaluate_split

        try:
            tm = evaluate_split(module, bundle, task, "test", num_neighbors=num_neighbors,
                                seq_len=seq_len, max_fk=max_fk, batch_size=batch_size, num_workers=num_workers)
            rec.update({f"test_{kk}": v for kk, v in tm.items()})
        except Exception as exc:  # never lose the (trained) val metrics to a test-eval failure
            rec["test_error"] = repr(exc)
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    # variant in the filename so several addition configs of one router (same grid index) can share an
    # out-dir without colliding — the additions study puts base + S/C/P/H runs in one directory.
    (out_dir / f"{index:04d}_{variant}.json").write_text(json.dumps(rec))
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
    """-> ``{(dataset, task, variant): {metric: (mean, std, ci, n)}}`` for each metric in ``keys``.

    Groups by the arm ``variant`` (router + S/C/P/H tags) so architecture-addition runs are kept distinct;
    records without a ``variant`` fall back to their bare ``signal`` (back-compat)."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["dataset"], r["task"], _variant_of(r)), []).append(r)
    return {gk: {key: _agg(rows, key) for key in keys} for gk, rows in groups.items()}


def load_records(out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or RESULTS
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def _variant_sort_key(variant: str):
    """Canonical router arms first (ROUTER_DISPLAY order), then addition variants alphabetically."""
    return (ROUTER_DISPLAY.index(variant) if variant in ROUTER_DISPLAY else len(ROUTER_DISPLAY), variant)


def format_table(records: list[dict], *, split: str = "test", baseline: str = "dense") -> str:
    """Per-``(dataset, task)`` table of the primary metric (seed mean ± 95% CI), each arm's lift over the
    ``baseline`` arm. Rows are the arm ``variant``s present (router + S/C/P/H tags). ``baseline`` is the
    reference variant for the Δ column (``dense`` for the routing study; a base router for the additions
    study, e.g. ``signature`` or ``hybrid``)."""
    if not records:
        return "(no records)"
    info: dict[tuple, dict] = {}
    for r in records:
        dt = (r["dataset"], r["task"])
        info.setdefault(dt, {"type": r.get("task_type", "binary"), "variants": set()})
        info[dt]["variants"].add(_variant_of(r))
    lines = [f"{len(records)} runs collected.", ""]
    for dt in sorted(info):
        ds, task = dt
        ttype = info[dt]["type"]
        metric = PRIMARY.get(ttype, "roc_auc")
        key = f"{split}_{metric}"
        lower = metric in LOWER_BETTER
        agg = aggregate([r for r in records if (r["dataset"], r["task"]) == dt], keys=(key,))
        base = agg.get((ds, task, baseline), {}).get(key, (float("nan"),))[0]
        lines.append(f"=== {ds} / {task}  ({ttype}; {split} {metric} {'↓' if lower else '↑'}) ===")
        width = max((len(v) for v in info[dt]["variants"]), default=11)
        for variant in sorted(info[dt]["variants"], key=_variant_sort_key):
            gk = (ds, task, variant)
            if gk not in agg:
                continue
            m, _sd, ci, n = agg[gk][key]
            lift = (base - m) if lower else (m - base)          # positive lift = better than baseline
            lines.append(f"  {variant:{width}s} {m:9.4f} ± {ci:.4f} (n={n})   Δvs {baseline}={lift:+.4f}")
        lines.append("")
    return "\n".join(lines)
