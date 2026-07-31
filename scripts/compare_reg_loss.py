#!/usr/bin/env python
"""Paired L1-vs-MSE comparison on the 4 regression tasks (`results/WHY_NOT_RT.md` §4).

The L1 sweep reuses the MSE sweep's hparam grid **unchanged**, so every (task, config) cell exists
under both objectives and the comparison is paired — the only thing that differs within a cell is
the training loss. Unpaired best-vs-best would confound the loss change with config reselection.

The prediction being tested is specific, not "L1 is better": MSE's optimum is the conditional mean,
L1's the conditional median, so the gain should scale with **target skew**. study-adverse (skew
39.1) and user-attendance (5.0) should move; driver-position (0.52) and site-success (0.24) should
not. site-success is underfitting (most cells sit near the 0.9713 constant-predictor NMAE), which is
a different cause that L1 cannot address. A result where all four move equally would *undercut* the
mechanism rather than support it.

    python scripts/compare_reg_loss.py
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

PAIRS = {"qwen": ("results/tl_grid_qwen", "results/tl_reg_l1_qwen"),
         "harrier": ("results/tl_grid_harrier", "results/tl_reg_l1_harrier")}

# Train-target skew and the constant-predictor NMAE floor, both measured in this session.
SKEW = {"rel-f1/driver-position": 0.52, "rel-trial/study-adverse": 39.11,
        "rel-trial/site-success": 0.24, "rel-event/user-attendance": 5.03}
CONST = {"rel-f1/driver-position": 0.6327, "rel-trial/study-adverse": 0.1697,
         "rel-trial/site-success": 0.9713, "rel-event/user-attendance": 0.3444}


def main() -> int:
    base = lb.load()
    for enc, (mse_dir, l1_dir) in PAIRS.items():
        mse_p, l1_p = ROOT / mse_dir, ROOT / l1_dir
        if not l1_p.exists() or not any(l1_p.glob("*.json")):
            print(f"== {enc}: no L1 records yet at {l1_dir}")
            continue
        mse, l1 = load_dir(mse_p), load_dir(l1_p)
        print(f"\n{'=' * 88}\n{enc.upper()}  —  NMAE, lower better.  d = MSE - L1  (positive = L1 better)")
        for task in sorted(SKEW, key=lambda t: -SKEW[t]):
            cells = sorted({c for (tk, c) in l1 if tk == task})
            if not cells:
                continue
            rt = base.get(task, {}).get("RT (from scratch)")
            gel = base.get(task, {}).get("GelGT")
            print(f"\n  {task}   skew={SKEW[task]}   const={CONST[task]:.4f}   RT={rt}   GelGT={gel}")
            print(f"    {'config':>16} {'MSE':>8} {'L1':>8} {'d':>8}")
            deltas = []
            for c in cells:
                a, b = mse.get((task, c)), l1.get((task, c))
                va = display_score(a) if a else None
                vb = display_score(b) if b else None
                if va is None or vb is None:
                    lbl = config_label(b or a) if (a or b) else str(c)
                    print(f"    {lbl:>16} {'--' if va is None else f'{va:8.4f}'} "
                          f"{'--' if vb is None else f'{vb:8.4f}'} {'--':>8}")
                    continue
                d = va - vb
                deltas.append(d)
                print(f"    {config_label(b):>16} {va:8.4f} {vb:8.4f} {d:+8.4f}")
            if deltas:
                won = sum(d > 0 for d in deltas)
                bm = min(display_score(r) for (tk, c), r in mse.items()
                         if tk == task and display_score(r) is not None)
                bl = min(display_score(r) for (tk, c), r in l1.items()
                         if tk == task and display_score(r) is not None)
                print(f"    -> L1 better on {won}/{len(deltas)} cells, "
                      f"mean d={sum(deltas) / len(deltas):+.4f};  best MSE={bm:.4f} best L1={bl:.4f} "
                      f"({'L1' if bl < bm else 'MSE'} wins)"
                      + (f"   [beats RT {rt}]" if rt is not None and min(bl, bm) < rt else ""))
    print("\nPrediction under test: gains should track skew (study-adverse 39.1, user-attendance 5.0 "
          "move; driver-position 0.52, site-success 0.24 do not).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
