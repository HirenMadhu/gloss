#!/usr/bin/env python
"""Paired AUC-surrogate-vs-BCE comparison on the 5 binary tasks (companion to `compare_reg_loss.py`).

Same pairing logic as the regression comparison: the AUC sweep reuses the BCE sweep's hparam grid
unchanged, so every (task, config) cell exists under both objectives and the only within-cell
difference is the training loss. Unpaired best-vs-best would confound the loss swap with config
reselection.

The prediction here is *weaker* than the L1 one, and worth stating up front so the result is read
honestly. L1 had a quantified target — the mean-vs-median NMAE gap, measured per task before the
run. The pairwise squared-hinge surrogate has no such precomputed number: BCE is already a proper
scoring rule whose optimum (the true conditional probability) is a perfect AUROC ranker, so at
convergence there is nothing to fix. The surrogate can only help where optimization, not the
objective, is the binding constraint — most plausibly under heavy class imbalance, where BCE's
gradient is dominated by the majority class. So: gains should track **positive-class rate**, and a
uniform gain across all five tasks would be as suspicious here as it would have been there.

    python scripts/compare_bin_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregate_gridsearch import config_label, display_score, load_dir  # noqa: E402

from gloss.eval import leaderboard as lb  # noqa: E402

PAIRS = {"qwen": ("results/tl_grid_qwen", "results/tl_bin_auc_qwen"),
         "harrier": ("results/tl_grid_harrier", "results/tl_bin_auc_harrier")}

TASKS = ["rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-trial/study-outcome",
         "rel-event/user-repeat", "rel-event/user-ignore"]


def pos_rate(task: str) -> float | None:
    """TRAIN-split positive-class rate — the axis the prediction is stated against."""
    try:
        from relbench.tasks import get_task
        ds, tk = task.split("/")
        t = get_task(ds, tk, download=False)
        y = t.get_table("train").df[t.target_col]
        return float((y.astype(str).isin(["1", "True", "true", "t"]) | (y == 1)).mean())
    except Exception:
        return None


def main() -> int:
    base = lb.load()
    rates = {t: pos_rate(t) for t in TASKS}
    for enc, (bce_dir, auc_dir) in PAIRS.items():
        bce_p, auc_p = ROOT / bce_dir, ROOT / auc_dir
        if not auc_p.exists() or not any(auc_p.glob("*.json")):
            print(f"== {enc}: no AUC records yet at {auc_dir}")
            continue
        bce, auc = load_dir(bce_p), load_dir(auc_p)
        print(f"\n{'=' * 92}\n{enc.upper()}  —  AUROC x100, HIGHER better.  "
              f"d = AUC - BCE  (positive = surrogate better)")
        for task in TASKS:
            cells = sorted({c for (tk, c) in auc if tk == task})
            if not cells:
                continue
            rt = base.get(task, {}).get("RT (from scratch)")
            gel = base.get(task, {}).get("GelGT")
            pr = rates.get(task)
            print(f"\n  {task}   pos_rate={'?' if pr is None else f'{pr:.3f}'}   "
                  f"RT={rt}   GelGT={gel}")
            print(f"    {'config':>16} {'BCE':>8} {'AUC':>8} {'d':>8}")
            deltas = []
            for c in cells:
                a, b = bce.get((task, c)), auc.get((task, c))
                va = display_score(a) if a else None
                vb = display_score(b) if b else None
                if va is None or vb is None:
                    lbl = config_label(b or a) if (a or b) else str(c)
                    print(f"    {lbl:>16} {'--' if va is None else f'{va:8.2f}'} "
                          f"{'--' if vb is None else f'{vb:8.2f}'} {'--':>8}")
                    continue
                d = vb - va
                deltas.append(d)
                print(f"    {config_label(b):>16} {va:8.2f} {vb:8.2f} {d:+8.2f}")
            if deltas:
                won = sum(d > 0 for d in deltas)
                bb = max(display_score(r) for (tk, c), r in bce.items()
                         if tk == task and display_score(r) is not None)
                ba = max(display_score(r) for (tk, c), r in auc.items()
                         if tk == task and display_score(r) is not None)
                print(f"    -> AUC better on {won}/{len(deltas)} cells, "
                      f"mean d={sum(deltas) / len(deltas):+.2f};  best BCE={bb:.2f} best AUC={ba:.2f} "
                      f"({'AUC' if ba > bb else 'BCE'} wins)"
                      + (f"   [beats RT {rt}]" if rt is not None and max(bb, ba) > rt else ""))
    print("\nPrediction under test: gains should track positive-class rate (imbalance), not appear "
          "uniformly. BCE is already a proper scoring rule, so a uniform gain would need a different "
          "explanation than the objective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
