#!/usr/bin/env python
"""Aggregate a two-level grid-search output directory into leaderboard-comparable tables.

Reads the per-job JSONs written by ``run_gridsearch.py`` and reports, per task, every config's
score in the **leaderboard's units** — AUROC x100 (higher better) for binary, NMAE (lower better)
for regression. It never re-derives those units by hand: binary goes through
``leaderboard.to_display`` and regression uses the record's own ``test_nmae``, which the pipeline
writes alongside raw MAE. A record missing ``test_nmae`` is reported as such rather than silently
falling back to raw MAE — that substitution is exactly the mistake that makes regression look
wildly off (see ``gloss/eval/leaderboard.py``).

Two things this deliberately does NOT do:

* **No cross-task averaging.** AUROC and NMAE point in opposite directions and live on different
  scales; a mean over them is meaningless. Configs are ranked by *how many tasks they win*, and
  the per-task numbers are always shown.
* **No "best config" claim without a caveat.** The grid runs ONE seed, and the headline's own
  3-seed run showed cv up to 28.5% driven by a single collapsed seed. A per-task argmin over 8
  single-seed configs is therefore partly selecting noise; ``--spread`` prints the gap between the
  best and second-best config so it can be read against that.

Usage:
    python scripts/aggregate_gridsearch.py                       # both encoders, all tasks
    python scripts/aggregate_gridsearch.py --dir results/tl_grid_qwen
    python scripts/aggregate_gridsearch.py --spread
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gloss.eval import leaderboard as lb  # noqa: E402

DEFAULT_DIRS = {"qwen": ROOT / "results/tl_grid_qwen", "harrier": ROOT / "results/tl_grid_harrier"}

# The headline 27-run array (`results/two_level_full/`), 3 seeds, d_model=256/n_blocks=8, qwen.
# Kept here so the grid can be read against the configuration it was launched to explain.
HEADLINE = {
    "rel-f1/driver-dnf": 71.37, "rel-f1/driver-top3": 89.69, "rel-f1/driver-position": 0.4735,
    "rel-trial/study-outcome": 60.91, "rel-trial/study-adverse": 0.1763,
    "rel-trial/site-success": 0.9670, "rel-event/user-repeat": 67.93,
    "rel-event/user-ignore": 82.41, "rel-event/user-attendance": 0.5536,
}


def config_label(rec: dict) -> str:
    """`d_model/n_blocks @ lr` — the three axes this grid actually sweeps."""
    return f"{rec['d_model']}/{rec['n_blocks']} @ {rec['lr']:g}"


def display_score(rec: dict) -> float | None:
    """The record's score in leaderboard units, or None if it cannot be expressed in them."""
    if rec.get("task_type") == "binary":
        auc = rec.get("test_roc_auc")
        return None if auc is None else lb.to_display(auc, "binary")
    nmae = rec.get("test_nmae")            # never fall back to raw test_mae — see module docstring
    return None if nmae is None else float(nmae)


def load_dir(d: Path) -> dict[tuple[str, int], dict]:
    """`{(dataset/task, config_idx): record}`. Later seeds of one cell would overwrite; the grid is
    single-seed, so assert that rather than silently keeping one arbitrarily."""
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(d.glob("*.json")):
        rec = json.loads(f.read_text())
        if "dataset" not in rec:            # a skipped/aborted index
            continue
        key = (f"{rec['dataset']}/{rec['task']}", rec["config_idx"])
        if key in out:
            raise RuntimeError(f"{f}: duplicate cell {key} (seed {rec.get('seed')}) — this "
                               "aggregator assumes one seed per (task, config)")
        out[key] = rec
    return out


def fingerprint(recs: dict) -> dict:
    """What the records SAY they are. Trusting the submit line instead is how two arrays finished
    on the wrong architecture and encoder (amendments.md §9.3)."""
    keys = ("arch", "phase", "encoder")
    seen = {k: sorted({str(r.get(k)) for r in recs.values()}) for k in keys}
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, action="append", default=None,
                    help="grid output dir (repeatable); default: both tl_grid_{qwen,harrier}")
    ap.add_argument("--spread", action="store_true",
                    help="show the best-vs-second-best gap per task (single-seed noise check)")
    args = ap.parse_args()

    dirs = {d.name.replace("tl_grid_", ""): d for d in args.dir} if args.dir else DEFAULT_DIRS
    base = lb.load()

    grids = {}
    for name, d in dirs.items():
        if not d.exists():
            print(f"!! {d} does not exist — skipping")
            continue
        recs = load_dir(d)
        if not recs:
            print(f"!! {d} has no records — skipping")
            continue
        grids[name] = recs
        fp = fingerprint(recs)
        n_cfg = len({c for _, c in recs})
        n_task = len({t for t, _ in recs})
        # A record can exist, hold val metrics, and still have NO test score: `run_index` catches a
        # test-eval crash into a `test_error` field, so the job exits 0 and sacct says COMPLETED.
        # Counting files therefore overstates coverage — report scoreable cells, not records.
        broken = {k: r for k, r in recs.items() if display_score(r) is None}
        print(f"== {name}: {len(recs)} records, {n_task} tasks x {n_cfg} configs, "
              f"{len(recs) - len(broken)} scoreable "
              f"({n_task * n_cfg - len(recs)} absent, {len(broken)} with no test metric)   {fp}")
        for (task, c), r in sorted(broken.items()):
            why = str(r.get("test_error", "no test_* key")).strip().splitlines()[-1][:90]
            print(f"     !! {task} cfg{c} ({config_label(r)}): {why}")
    if not grids:
        print("nothing to aggregate")
        return 1

    tasks = sorted({t for recs in grids.values() for t, _ in recs})
    cfg_idxs = sorted({c for recs in grids.values() for _, c in recs})
    labels = {}
    for recs in grids.values():
        for (_, c), r in recs.items():
            labels[c] = config_label(r)

    # ---- per-task table: every config, both encoders, vs the external baselines ----------------
    wins = {name: {c: 0 for c in cfg_idxs} for name in grids}
    for task in tasks:
        entry = base.get(task, {})
        ttype = entry.get("type", "binary")
        unit = "AUROC x100 (higher better)" if ttype == "binary" else "NMAE (lower better)"
        rt, gel = entry.get("RT (from scratch)"), entry.get("GelGT")
        print(f"\n### {task} — {unit}")
        print(f"    RT (from scratch)={rt}   GelGT={gel}   headline 256/8={HEADLINE.get(task)}")
        rows = []
        for c in cfg_idxs:
            cells = {}
            for name, recs in grids.items():
                rec = recs.get((task, c))
                cells[name] = display_score(rec) if rec else None
            rows.append((c, cells))
        # rank within each encoder so a "best" is per-encoder, not mixed
        for name in grids:
            vals = [(c, cells[name]) for c, cells in rows if cells[name] is not None]
            if not vals:
                continue
            best = (min if ttype == "regression" else max)(vals, key=lambda cv: cv[1])
            wins[name][best[0]] += 1
        hdr = f"    {'config':>16} " + " ".join(f"{n:>10}" for n in grids)
        print(hdr)
        for c, cells in rows:
            line = f"    {labels.get(c, c):>16} "
            for name in grids:
                v = cells[name]
                if v is None:
                    line += f"{'--':>10} "
                else:
                    mark = ""
                    if lb.beats(v, rt, ttype):
                        mark = "*"          # beats RT (from scratch)
                    line += f"{v:>9.4f}{mark:1} " if ttype == "regression" else f"{v:>9.2f}{mark:1} "
            print(line)
        if args.spread:
            for name in grids:
                vals = sorted([cells[name] for _, cells in rows if cells[name] is not None],
                              reverse=(ttype == "binary"))
                if len(vals) >= 2:
                    print(f"      {name}: best={vals[0]:.4f} 2nd={vals[1]:.4f} "
                          f"gap={abs(vals[0] - vals[1]):.4f}  worst={vals[-1]:.4f}")

    # ---- which config wins most tasks (NOT a cross-task average — see docstring) ---------------
    print("\n### tasks won per config (per encoder; ties not broken)")
    print(f"    {'config':>16} " + " ".join(f"{n:>10}" for n in grids))
    for c in cfg_idxs:
        print(f"    {labels.get(c, c):>16} " + " ".join(f"{wins[name][c]:>10}" for name in grids))

    # ---- how each encoder's best-per-task compares to the external baselines -------------------
    print("\n### best-of-grid per task vs the leaderboard  (* = beats RT from scratch)")
    for name, recs in grids.items():
        beat_rt = beat_gel = n = 0
        for task in tasks:
            entry = base.get(task, {})
            ttype = entry.get("type", "binary")
            vals = [display_score(recs[(task, c)]) for c in cfg_idxs if (task, c) in recs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            n += 1
            best = (min if ttype == "regression" else max)(vals)
            beat_rt += bool(lb.beats(best, entry.get("RT (from scratch)"), ttype))
            beat_gel += bool(lb.beats(best, entry.get("GelGT"), ttype))
        print(f"    {name:>10}: beats RT on {beat_rt}/{n}, beats GelGT on {beat_gel}/{n} "
              f"(best-of-8 per task, 1 seed — optimistic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
