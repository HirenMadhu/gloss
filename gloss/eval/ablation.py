"""Phase 4 — the headline gate: four documentation regimes on the SAME DOC-RT encoder.

    full      : every column conditioned on its grounded prose documentation        (docs ON)
    null      : FiLM doc turned off (d_null everywhere); RT name tokens kept         (names-only floor)
    shuffled  : placebo — doc spans deranged across columns (length/topology matched, meaning destroyed)
    name_only : condition on the column-name string embedding instead of prose docs  (RELATE-style)

How to read the gate:
    full > null      => documentation carries signal beyond what the schema/names already give;
    full > shuffled  => it is the *meaning* of the docs, not their mere presence/length;
    full > name_only => prose docs beat the bare column-name string.
``full ≈ null`` is a legitimate, honest negative (e.g. coded columns like ``statusId`` are already
learnable as plain categorical embeddings, so prose adds little on rel-f1).

Each ``(regime, seed)`` is an independent run (one SLURM array task) that writes
``results/headline/<index>.json``; ``aggregate``/``format_table`` reduce the JSONs to a per-regime
AP/AUROC table with the seed mean ± std and a 95 % CI, plus the lift over the ``null`` floor.
"""
from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path

REGIMES = ("full", "null", "shuffled", "name_only")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "headline"


def enumerate_configs(seeds: int, regimes: tuple[str, ...] = REGIMES) -> list[dict]:
    """All independent ``(regime, seed)`` runs — the SLURM array iterates this list by index."""
    return [{"regime": r, "seed": s} for s in range(seeds) for r in regimes]


def run_config(
    index: int,
    *,
    dataset: str,
    task_name: str,
    seeds: int,
    encoder: str = "qwen",
    d_text: int = 2560,
    model_kwargs: dict | None = None,
    num_neighbors: list[int] | None = None,
    seq_len: int = 1024,
    max_fk: int = 5,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 10,
    num_workers: int = 8,
    sim_threshold: float = 0.60,
    out_dir: Path | None = None,
) -> dict:
    """Train the single ``(regime, seed)`` config at ``index`` and persist its val metrics as JSON."""
    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph
    from ..train.finetune import docs_for_regime, train_prebuilt

    c = enumerate_configs(seeds)[index]
    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    grounding = docs_for_regime(dataset, c["regime"], encoder=encoder, d_text=d_text,
                                sim_threshold=sim_threshold)
    mk = dict(model_kwargs or {})
    mk.setdefault("d_text", grounding.d_text)
    _module, metrics = train_prebuilt(
        bundle, task, grounding, model_kwargs=mk, num_neighbors=num_neighbors,
        seq_len=seq_len, max_fk=max_fk, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, seed=c["seed"], num_workers=num_workers,
    )
    rec = {
        **c, "dataset": dataset, "task": task_name,
        "ap": metrics.get("val/ap"), "auroc": metrics.get("val/auroc"),
        "logloss": metrics.get("val/logloss"),
    }
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{index:03d}.json").write_text(json.dumps(rec))
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


def aggregate(records: list[dict], regimes: tuple[str, ...] = REGIMES) -> dict[str, dict]:
    """-> ``{regime: {metric: (mean, std, ci, n)}}`` for metric in (ap, auroc, logloss)."""
    return {
        regime: {k: _agg([r for r in records if r.get("regime") == regime], k)
                 for k in ("ap", "auroc", "logloss")}
        for regime in regimes
    }


def load_records(out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or RESULTS
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def format_table(records: list[dict], regimes: tuple[str, ...] = REGIMES) -> str:
    agg = aggregate(records, regimes)
    lines = [
        f"{len(records)} runs collected.",
        "",
        "=== HEADLINE: documentation regime (same DOC-RT encoder) ===",
        f"{'regime':12s} {'AP  mean±std (95%CI)':>26s} {'AUROC  mean±std (95%CI)':>28s} {'n':>3s}",
    ]
    for regime in regimes:
        apm, aps, apci, n = agg[regime]["ap"]
        aum, aus, auci, _ = agg[regime]["auroc"]
        lines.append(f"{regime:12s} {apm:7.4f}±{aps:.4f}({apci:.4f})   "
                     f"{aum:7.4f}±{aus:.4f}({auci:.4f}) {n:3d}")

    nap = agg.get("null", {}).get("ap", (float("nan"),))[0]
    nau = agg.get("null", {}).get("auroc", (float("nan"),))[0]
    lines += ["", "Δ vs null (documentation lift):"]
    for regime in regimes:
        if regime == "null":
            continue
        apm = agg[regime]["ap"][0]
        aum = agg[regime]["auroc"][0]
        lines.append(f"  {regime:12s} ΔAP={apm - nap:+.4f}  ΔAUROC={aum - nau:+.4f}")
    return "\n".join(lines)
