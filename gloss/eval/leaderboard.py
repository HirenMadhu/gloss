"""External comparison targets: the RelBench leaderboard's RT (from scratch) and GelGT rows.

``results/leaderboard_baselines.json`` is the verbatim scrape of every leaderboard task (written by
``scripts/fetch_leaderboard.py``). This module projects it onto the 9 entity tasks we actually run
and pins down the metric conventions, which differ from what our training pipeline reports:

* **binary** — leaderboard is **AUROC in percent** (higher better); we report a 0-1 fraction, so
  displaying ours means ``x * 100``.
* **regression** — leaderboard is **NMAE = MAE / train-std** (lower better); we report **raw MAE**,
  so comparing means dividing by the train-split target std first. Skipping that normalization is
  the one mistake that makes regression look wildly off against the leaderboard.

Use :func:`to_display` / :func:`beats` rather than hand-rolling either conversion.
"""

from __future__ import annotations

import json
from pathlib import Path

from gloss.eval.ablation import LEADERBOARD_TASKS, RESULTS_ROOT

#: Leaderboard methods we compare against, in display order. Keys are the JSON's identifier-safe forms.
METHODS: dict[str, str] = {"RT (from scratch)": "RT_from_scratch", "GelGT": "GelGT"}

BASELINES = RESULTS_ROOT / "leaderboard_baselines.json"

# Which leaderboard table each of our task types lives in.
_SECTION = {"binary": "classification", "regression": "regression"}
_METRIC = {"binary": "auroc", "regression": "nmae"}


def _task_types() -> dict[str, str]:
    """``{"rel-f1/driver-dnf": "binary", ...}`` for the 9 tasks, without importing relbench.

    The scrape tells us the type: a task appears in exactly one of the two tables.
    """
    doc = _raw()
    out = {}
    for ds, tasks in LEADERBOARD_TASKS.items():
        for tk in tasks:
            for ttype, section in _SECTION.items():
                if f"{ds} {tk}" in doc.get(section, {}):
                    out[f"{ds}/{tk}"] = ttype
    return out


def _raw(path: Path | None = None) -> dict:
    p = path or BASELINES
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run `python scripts/fetch_leaderboard.py` to scrape the leaderboard."
        )
    return json.loads(p.read_text())


def load(path: Path | None = None) -> dict[str, dict]:
    """Baselines for our 9 leaderboard tasks, keyed ``"<dataset>/<task>"``.

    Each value is ``{"type": "binary"|"regression", "metric": "auroc"|"nmae",
    "RT (from scratch)": float, "GelGT": float}``. Tasks the leaderboard does not report for a
    method are simply absent from that task's dict, so callers must use ``.get``.
    """
    doc = _raw(path)
    out: dict[str, dict] = {}
    for ds, tasks in LEADERBOARD_TASKS.items():
        for tk in tasks:
            key, col = f"{ds}/{tk}", f"{ds} {tk}"
            for ttype, section in _SECTION.items():
                row = doc.get(section, {}).get(col)
                if row is None:
                    continue
                entry: dict = {"type": ttype, "metric": _METRIC[ttype]}
                for name, jkey in METHODS.items():
                    if row.get(jkey) not in (None, "", "-"):
                        entry[name] = float(row[jkey])
                out[key] = entry
    return out


def to_display(value: float, task_type: str, *, train_std: float | None = None) -> float:
    """Convert a pipeline metric into the leaderboard's units.

    ``binary``: pass ``roc_auc`` (0-1) -> percent. ``regression``: pass **raw MAE** plus the
    train-split target std -> NMAE. If the caller already normalized, omit ``train_std``.
    """
    if task_type == "binary":
        return value * 100.0
    if train_std is not None:
        if train_std <= 0:
            raise ValueError(f"train_std must be positive, got {train_std}")
        return value / train_std
    return value


def beats(ours_display: float, baseline: float | None, task_type: str) -> bool | None:
    """Did we beat ``baseline``? ``None`` when the leaderboard reports no value for that cell.

    Direction is set by the metric: AUROC higher-better, NMAE lower-better.
    """
    if baseline is None:
        return None
    return ours_display > baseline if task_type == "binary" else ours_display < baseline


def summary_line(key: str, ours_display: float, entry: dict) -> str:
    """One-line ``ours vs each method`` comparison for aggregate tables."""
    parts = []
    for name in METHODS:
        base = entry.get(name)
        won = beats(ours_display, base, entry["type"])
        parts.append(f"{name}={base if base is not None else 'n/a'}"
                     + ("" if won is None else "  BEAT" if won else "  no"))
    unit = "AUROC↑" if entry["type"] == "binary" else "NMAE↓"
    return f"{key} ({unit}) ours={ours_display:.4f}   " + "   ".join(parts)
