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
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
RESULTS = RESULTS_ROOT / "headline"
RESULTS_TEST = RESULTS_ROOT / "headline_test"


def results_dir(encoder: str = "qwen", test: bool = False) -> Path:
    """Per-encoder output dir so swapping the text encoder never mixes results. ``qwen`` keeps the
    original ``headline`` / ``headline_test`` dirs (back-compat); any other encoder gets an
    ``_<encoder>`` suffix (e.g. ``headline_test_harrier``)."""
    base = "headline_test" if test else "headline"
    suffix = "" if encoder == "qwen" else f"_{encoder.replace('/', '__')}"
    return RESULTS_ROOT / f"{base}{suffix}"


def enumerate_configs(seeds: int, regimes: tuple[str, ...] = REGIMES) -> list[dict]:
    """All independent ``(regime, seed)`` runs — the SLURM array iterates this list by index."""
    return [{"regime": r, "seed": s} for s in range(seeds) for r in regimes]


def run_config(*args, **kwargs):
    """Per-config training runner — **rebuilt in Phase D** as the routing-signal ablation
    (signature / hidden / value / identity / dense). Intentionally not implemented during the Phase-A
    pivot; the pure enumeration/aggregation helpers below are what the current tests exercise, and the
    DOC-RT four-regime runner is archived under ``archive/doc-rt/``."""
    raise NotImplementedError(
        "run_config is rebuilt in Phase D (routing-signal ablation); the DOC-RT regime runner is in "
        "archive/doc-rt/."
    )


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


def aggregate(
    records: list[dict],
    regimes: tuple[str, ...] = REGIMES,
    keys: tuple[str, ...] = ("ap", "auroc", "logloss"),
) -> dict[str, dict]:
    """-> ``{regime: {metric: (mean, std, ci, n)}}`` for each metric in ``keys``."""
    return {
        regime: {k: _agg([r for r in records if r.get("regime") == regime], k) for k in keys}
        for regime in regimes
    }


def load_records(out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or RESULTS
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def _format(records, regimes, ap_key, auroc_key, title) -> str:
    agg = aggregate(records, regimes, keys=(ap_key, auroc_key))
    lines = [
        f"{len(records)} runs collected.",
        "",
        title,
        f"{'regime':12s} {'AP  mean±std (95%CI)':>26s} {'AUROC  mean±std (95%CI)':>28s} {'n':>3s}",
    ]
    for regime in regimes:
        apm, aps, apci, n = agg[regime][ap_key]
        aum, aus, auci, _ = agg[regime][auroc_key]
        lines.append(f"{regime:12s} {apm:7.4f}±{aps:.4f}({apci:.4f})   "
                     f"{aum:7.4f}±{aus:.4f}({auci:.4f}) {n:3d}")

    nap = agg.get("null", {}).get(ap_key, (float("nan"),))[0]
    nau = agg.get("null", {}).get(auroc_key, (float("nan"),))[0]
    lines += ["", "Δ vs null (documentation lift):"]
    for regime in regimes:
        if regime == "null":
            continue
        apm = agg[regime][ap_key][0]
        aum = agg[regime][auroc_key][0]
        lines.append(f"  {regime:12s} ΔAP={apm - nap:+.4f}  ΔAUROC={aum - nau:+.4f}")
    return "\n".join(lines)


def format_table(records: list[dict], regimes: tuple[str, ...] = REGIMES) -> str:
    """The headline (VALIDATION) table — same DOC-RT encoder across the four regimes."""
    return _format(records, regimes, "ap", "auroc",
                   "=== HEADLINE: documentation regime (same DOC-RT encoder) — VALIDATION ===")


def format_test_table(records: list[dict], regimes: tuple[str, ...] = REGIMES) -> str:
    """The held-out RelBench TEST table (leaderboard-comparable, via ``task.evaluate``)."""
    return _format(records, regimes, "test_ap", "test_auroc",
                   "=== HEADLINE: documentation regime (same DOC-RT encoder) — TEST ===")
