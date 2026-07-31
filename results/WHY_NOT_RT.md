# Why doesn't MoRE clearly beat RT? — evidence, 2026-07-31

All numbers from the completed two-level grid (140 cells) + `results/rt_arch_grid_cfg0-7_qwen/`.

## 0. First, the premise needs tightening

Selecting each config on **val** (the honest protocol) rather than on test, both encoders beat
published RT on **5/9, not 6/9**. Only `user-ignore` flips, and it flips hard — qwen's best-val
config scores 78.70 on test while its best-test config scores 89.77, an 11.06-point gap. Every other
task's test-selection inflation is ≤ 2.5 points.

**But there is no controlled RT comparison at all.** `run_gridsearch.py:147` hardcodes
`route_on="signature"`, so `results/rt_arch_grid_cfg0-7_qwen/` is *single-level MoRE on the RT
substrate*, not RT. A true in-codebase RT is `arch=rt, route_on=dense`, which the standing rule
forbids. So every "vs RT" number compares against a published model trained under a different
protocol (epochs, sampler, tuning). That is the single largest weakness in the current evidence.

## 1. The row level adds nothing (two-level ≈ single-level)

Best-of-grid, qwen, on the 6 tasks both grids cover (+ = two-level better):

| task | 1-level best | 2-level best | Δ |
|---|---|---|---|
| rel-f1/driver-dnf | 83.82 | 83.77 | −0.05 |
| rel-f1/driver-top3 | 90.41 | 91.15 | **+0.73** |
| rel-f1/driver-position | 0.4084 | 0.4073 | +0.001 |
| rel-trial/site-success | 0.8344 | 0.8775 | −0.043 |
| rel-trial/study-adverse | 0.1503 | 0.1547 | −0.004 |
| rel-trial/study-outcome | 72.09 | 70.06 | −2.03 |

5 of 6 tied or worse. Whatever is limiting performance, **it is not the absence of the row level** —
and the row MoE, the branch's whole contribution, is not paying for itself.

## 2. The router has NEVER been measured

`gloss/eval/diagnostics.py` implements `expert_usage`, `specialization_probe` and
`mean_active_experts`. **None is called anywhere in the grid or headline pipeline**, and no result
JSON contains a routing field — the keys are config + val/test metrics only. The gate also saves no
checkpoints, so this cannot be probed post-hoc; it needs a re-run with the diagnostics wired in.

So the method's central claim — that routing on the relational signature specialises the experts —
is **completely unmeasured**. If the router has collapsed to one expert, MoRE ≈ RT with a wider FFN,
and every result here is consistent with that. This is exactly the RGCN null `CLAUDE.md` predicts:
RT's frozen-LM name token may already let one shared FFN absorb every column.

## 3. On the tasks it loses, MoRE is barely above a constant predictor

NMAE of the best *constant* prediction (train mean or median, whichever is better), scored through
RelBench's own `task.evaluate` on test:

| task | const | GelGT | RT | ours | headroom closed (ours / RT) |
|---|---|---|---|---|---|
| rel-f1/driver-position | 0.6327 | 0.5315 | 0.4775 | **0.3950** | 235% / 153% |
| rel-trial/study-adverse | 0.1697 | 0.1255 | 0.1306 | 0.1547 | **34%** / 88% |
| rel-trial/site-success | 0.9713 | 0.7324 | 0.7341 | 0.8210 | **63%** / 99% |
| rel-event/user-attendance | 0.3444 | 0.3167 | 0.5040 | 0.3853 | −147% / −576% |

Two consequences:

* On **site-success** and **study-adverse** the model is not being narrowly outperformed — it is
  **failing to learn**. Most site-success grid cells sit at 0.88–0.98 against a 0.9713 constant.
  That is an optimization/underfitting signature, not a subtle architectural deficit.
* On **user-attendance**, *both* RT (0.504) and MoRE (0.3853) are **worse than predicting the train
  median** (0.3444); only GelGT (0.3167) beats it. Our "W vs RT" there is not evidence of learning
  and should not be counted as a win.

## 4. Loss/metric mismatch explains much of the regression deficit

Training uses **MSE** (`train/losses.py::task_loss` → `masked_mse`) on z-scored targets; RelBench
scores regression on **MAE**. MSE's optimum is the conditional *mean*, MAE's is the *median* — the
penalty scales with target skew:

| task | target skew | NMAE(mean) | NMAE(median) | gap |
|---|---|---|---|---|
| rel-trial/study-adverse | **39.11** | 0.2152 | 0.1697 | 0.0455 |
| rel-event/user-attendance | **5.03** | 0.6209 | 0.3444 | 0.2765 |
| rel-f1/driver-position | 0.52 | 0.6478 | 0.6327 | 0.0151 |
| rel-trial/site-success | 0.24 | 0.9824 | 0.9713 | 0.0111 |

On study-adverse our deficit to RT is 0.1547 − 0.1306 = **0.024**, while the mean-vs-median gap alone
is **0.0455** — the mismatch is nearly *twice* the deficit, so an MAE-aligned loss (L1/Huber on the
standardized target) could plausibly close it outright. Same story on user-attendance (gap 0.2765).

This does **not** explain site-success (skew 0.24) — that one is underfitting, cause 3.

## 4b. RULED OUT: `seq_len` truncation is *not* starving the losing tasks

Worth checking because it is the same shape of mistake as `MAX_ROWS=160` — a rel-f1 number applied to
every DB — except that `max_rows` asserted loudly while `to_cell_batch` drops overflow cells with a
bare `if s >= seq_len: break`: no counter, no warning. A run could have been training on a fraction
of each neighbourhood and looked perfectly healthy. Measured with `scripts/probe_seq_len.py`
(fanout `[12,12]`, `seq_len=512`), reporting demanded cells/seed:

| dataset | median | p90 | max | seeds over cap | cells kept |
|---|---|---|---|---|---|
| rel-f1 (3 tasks) | 249–288 | 290–336 | **374** | 0.0% | **100%** |
| rel-trial (3 tasks) | 71–109 | 113–317 | **317** | 0.0% | **100%** |

**Nothing is truncated on either DB.** rel-trial is in fact the *sparsest* — site-success has a median
of 109 cells/seed against a 512 cap, and study-adverse 71. So the two worst tasks are not
information-starved by the cap; they simply are not learning from what they already get, which
sharpens cause 3 (underfitting) rather than competing with it.

It also means the losses are not explained by neighbourhood size: rel-trial gets *less* context per
seed than rel-f1 and loses, while rel-f1 gets the most and wins 3/3. (rel-event measurement pending —
its loader is slow; it is the one DB where a large fanout could still overflow.)

Related size facts, for the record: train rows are site-success 151,407 and study-adverse 43,335 —
the two **largest** training sets are two of the three losses, so this is not a shortage of labels or
of optimizer steps either. Distinct column names: rel-f1 **45**, rel-event **134**. We win where the
schema is *least* diverse, which is the opposite of what a schema-routing mechanism should predict.

## 5. lr fragility, and a fixed 10-epoch budget

Cells beating RT: **33/69 at lr=3e-4 vs 15/71 at 1e-3**; depth is irrelevant (25/72 at 2 blocks vs
23/68 at 4). At 1e-3 the model collapses outright on several tasks (driver-dnf 63.6, driver-top3
65.6, study-outcome 53.4), matching the headline's one-collapsed-seed-in-three instability. A model
this lr-sensitive at 10 fixed epochs is being measured partly on its optimization robustness.

## What would actually settle it, cheapest first

1. **Wire the diagnostics into the runner** (`expert_usage`, `mean_active_experts`) and re-run a few
   cells. Near-zero cost, and it tests the method's core claim. If the router has collapsed, nothing
   else here matters.
2. **Swap MSE → L1/Huber** for regression and re-run the 4 regression tasks. Directly targets a
   quantified 0.0455 / 0.2765 NMAE penalty.
3. **Run the in-codebase dense control** (`arch=rt, route_on=dense`) — requires lifting the standing
   never-run-dense rule. Without it no result can be attributed to the MoE.
4. Multi-seed confirmation of the winning configs; the grid is 1 seed and the headline showed cv to
   28.5%.
