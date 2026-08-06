# Pre-registration — recency order-statistic channel (`x`)

Written **before** the arms were launched, so the predictions can falsify rather than be fitted to
the result. Companion to the spec; measurements below are from `scripts/probe_role_window.py`
(val split, CPU, no training).

## What was verified first, not assumed (spec §6)

| question | answer | consequence for the build |
|---|---|---|
| recency unit | `τ = log1p(Δ_seconds)`, UNIT = 1 s | reuse `TimeLadder.tau`; the spec formula and the existing convention agree to **0.0** on rel-f1 / rel-trial / rel-event, so there is one convention, not two |
| existing `x` channel | none — no `κ`, no gated-max path anywhere in `gloss/` | this is a new module, not an edit |
| effective cap | w = 12 holds, but rel-event role 1 reaches **13** | a row can be reached as a child via two parents, so the union exceeds the per-hop cap → `sat` is `|C| ≥ w`, never `== w` |
| `Δ ≥ 0` | **0 violations in 7,474 child rows**; rel-event has **1 root row** with Δ < 0 | the assert is scoped to child rows. Scoped to all rows it fires on the known §9.10 clamped-row class (`was_clamped` / `b_clamped` exist *because* those rows are real) and kills every rel-event run |
| row set vs `seq_len` | the row graph is built from all sampled nodes **before** cell enumeration | `seq_len=512` truncates *cells*, not rows, so `sat` measures sampler fanout — the intended quantity — and not cell truncation |

## Role table (measured, val split)

Saturation is the fraction of `(row, role)` groups with `|C| ≥ 12`.

**rel-f1** (`driver-top3`, seed rows) — the channel has the most to read here:

| role | mean fanout | max | sat @12 |
|---|---|---|---|
| 6 | 10.84 | 12 | **0.840** |
| 10 | 10.93 | 12 | **0.856** |
| 12 | 11.23 | 12 | **0.886** |

All 8 reached roles are **timed** (0 untimed children).

**rel-trial** (`study-adverse`, all rows) — near-zero truncation:

| role | mean fanout | max | sat @12 |
|---|---|---|---|
| 7 | 3.75 | 12 | 0.188 |
| 6 | 1.11 | 12 | 0.009 |
| 1,2,3,5,8,9,14,15 | 1.0–2.1 | ≤7 | **0.000** |

All roles timed.

**rel-event** (`user-repeat`, all rows):

| role | mean fanout | max | sat @12 | timed |
|---|---|---|---|---|
| 2 | 7.44 | 12 | 0.406 | yes |
| 6 | 7.27 | 12 | 0.351 | **no (0/560)** |
| 7 | 2.17 | 12 | 0.106 | **no (0/1296)** |
| 1 | 1.24 | **13** | 0.021 | yes |
| 3,4,5 | 1.3–8.2 | ≤11 | 0.000 | yes |

## Predictions

Role level:
- timed **and** saturating → the channel has a window to read
- timed **and** never saturating → no truncation, no rate, no effect
- untimed → no effect by construction (null path, `x = 0`)

Task level, as specified:
- **should move:** `study-adverse` (strongest), `user-ignore`, `study-outcome`
- **should not move:** `user-repeat` — its two heaviest roles (6, 7) are untimed, confirmed above
- **uncertain:** `driver-top3`, `driver-dnf`. A regression here is not a refutation (rel-f1 has the
  +2.87 log-space fanout drift and is where censoring acted as a regulariser)
- **no prediction:** `user-attendance`, `site-success`, `driver-position` — the census marks these
  UNINFORMATIVE

**One prediction I am registering against the spec.** The spec calls `study-adverse` the strongest
mover, but its measured saturation is **0.188 on one role and ~0 on the rest**. The spec's own logic
says a max age only encodes *rate* when the window is truncated. So either `study-adverse` moves for a
reason other than the rate story (`x_max` as raw history span, which `x_flags` will not capture), or it
does not move at all. rel-f1 is the DB where the mechanism has the most to bite on (sat 0.84–0.89), and
it is the one the spec is least confident about. If the ranking comes out **rel-f1 > rel-trial**, the
mechanism is real but the census-based task ranking was wrong.

Falsifier: uniform improvement including `user-repeat` and the three uninformative tasks ⇒ capacity or
regularisation, not the mechanism. Check `x_shuffle` immediately.

## Arms and protocol

| arm | out-dir |
|---|---|
| `base` | `results/x_base_s1` |
| `x_full` | `results/x_full_s1` |
| `x_flags` | `results/x_flags_s1` |
| `x_shuffle` | `results/x_shuffle_s1` |

Held fixed: `d_model=128 / n_blocks=2`, lr `3e-4`, batch 512, 80 epochs, patience 24, qwen, AUC
surrogate (binary) / L1 (regression), `seq_len=512`, seed **1**, all 9 leaderboard tasks.

**`base` is re-run inside this array rather than reused** from `results/tl_bs512_qwen_s3`. Identical
`(config, seed)` cells re-run in a different array differ by up to **2.12 AUC** and **0.015 NMAE**
(measured: `tl_bs512_qwen` vs `tl_shape_b512_qwen`, 10 overlapping cells) — the global seed does not
pin the neighbour sampler or worker order. A cross-array pairing would therefore compare the
mechanism against that noise floor.

Seed 1 is the best-ranked of the four seeds run at b512/80ep (mean rank 1.875 over 8 tasks, vs 2.33 /
2.33 / 3.13 for seeds 0 / 42 / 2).

**One seed, as requested.** Deltas smaller than ~1 AUC / ~0.01 NMAE on rel-event and rel-trial are
inside the single-run noise floor measured above and must not be read as effects.

## What to check before reading any accuracy number

- `x_kappa_max` drifting toward 0 ⇒ the soft max collapsed to a **mean**; no order statistic is being
  computed and the arm silently stopped testing the hypothesis. Init is +4.0 / −4.0, both stamped on
  the record next to the final value.
- `x_alpha` near 0 ⇒ the channel is unused and the arm is a no-op.
- `x_role_sat_rate` / `x_role_untimed_rate` on val vs the offline table above — the prediction is
  checked against what happened at runtime, not against the census.
- `x_max_std` ≈ 0 for a role ⇒ the window is not varying and there was nothing to read.
