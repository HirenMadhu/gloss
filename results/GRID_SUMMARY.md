# Two-level grid search — 140/144 cells (`29030571` qwen, `29030572` harrier)

`signature@two_level`, `--phase full`, **1 seed**, 10 epochs, TEST set, 9 RelBench leaderboard entity
tasks × 8 configs × 2 encoders. Grid axes: `d_model ∈ {128,256}` × `n_blocks ∈ {2,4}` × `lr ∈
{3e-4,1e-3}` (`n_heads=8`, `d_ff=4·d_model`, `num_experts=4`, `enc_channels=d_model`).

Regenerate with `python scripts/aggregate_gridsearch.py --spread`. Records are self-describing —
`arch=two_level, phase=full, encoder=∈{qwen,harrier}` verified from the JSONs, not the submit line.

**Coverage: 70/72 scoreable per encoder.** Four cells hit the zero-column MET bug at TEST eval
(fixed in `eb488e4`); their stale records are parked in `results/_test_error_pre_eb488e4/` and the
cells are re-running as `29033908`/`29033909`. Two of them (qwen `256/4@3e-4` on driver-position and
user-attendance) could still move a per-task best, so treat those two rows as provisional.

## Best config per task, vs the leaderboard

`W/L` is against **RT (from scratch) / GelGT** respectively. Neither is protocol-matched to this grid
(see caveats); both are published numbers.

| task | metric | RT | GelGT | headline 256/8 | qwen best (config) | vs RT/Gel | harrier best (config) | vs RT/Gel |
|---|---|---|---|---|---|---|---|---|
| rel-f1/driver-dnf | AUROC↑ | 78.7 | 76.1 | 71.37 | 83.77 (128/4 @ 3e-4) | **W / W** | 83.51 (128/2 @ 3e-4) | **W / W** |
| rel-f1/driver-top3 | AUROC↑ | 82.7 | 84.1 | 89.69 | 91.15 (256/2 @ 3e-4) | **W / W** | 92.00 (128/2 @ 3e-4) | **W / W** |
| rel-f1/driver-position | NMAE↓ | 0.4775 | 0.5315 | 0.4735 | 0.4073 (128/2 @ 1e-3) | **W / W** | 0.3950 (256/2 @ 1e-3) | **W / W** |
| rel-trial/study-outcome | AUROC↑ | 68.6 | 72.5 | 60.91 | 70.06 (128/4 @ 3e-4) | W / L | 69.19 (256/4 @ 3e-4) | W / L |
| rel-trial/study-adverse | NMAE↓ | 0.1306 | 0.1255 | 0.1763 | 0.1547 (256/2 @ 3e-4) | L / L | 0.1549 (128/2 @ 3e-4) | L / L |
| rel-trial/site-success | NMAE↓ | 0.7341 | 0.7324 | 0.9670 | 0.8775 (128/4 @ 3e-4) | L / L | 0.8210 (256/4 @ 3e-4) | L / L |
| rel-event/user-repeat | AUROC↑ | 79.7 | 83.6 | 67.93 | 78.83 (256/2 @ 3e-4) | L / L | 79.45 (128/4 @ 3e-4) | L / L |
| rel-event/user-ignore | AUROC↑ | 85.1 | 87.8 | 82.41 | 89.77 (256/4 @ 1e-3) | **W / W** | 86.84 (256/2 @ 1e-3) | W / L |
| rel-event/user-attendance | NMAE↓ | 0.504 | 0.3167 | 0.5536 | 0.3974 (128/2 @ 3e-4) | W / L | 0.3853 (128/4 @ 1e-3) | W / L |

**Both encoders beat RT on 6 of 9; against GelGT it is qwen 4/9 and harrier 3/9.**

The three losses are the *same three* for both encoders and lose to **both** methods: user-repeat (a
near-tie vs RT, 79.45 vs 79.7) and the two rel-trial regressions, site-success and study-adverse,
which lose by a wide margin.

GelGT is the harder bar and it splits the wins along a clean line: **the three rel-f1 tasks beat both
methods**, everything else either splits or loses. The two tasks that beat RT but lose to GelGT are
study-outcome (70.06 vs 72.5) and user-attendance (0.3974 vs 0.3167) — the latter is the widest GelGT
margin on the board, so it is the single "win" that most overstates itself when only RT is quoted.

## The capacity confound is confirmed — and it was mostly the learning rate

The headline's negative result (2/9 vs RT) was hypothesised to be a capacity/lr mismatch rather than
a broken mechanism. The grid supports that: **95 of 140 cells beat the 68.9M-param `256/8` headline**,
and the per-task best improves on it everywhere except driver-top3.

Cells beating RT, split by axis:

| axis | value | cells beating RT |
|---|---|---|
| **lr** | **3e-4** | **33/69** |
| **lr** | **1e-3** | **15/71** |
| d_model | 128 | 28/71 |
| d_model | 256 | 20/69 |
| n_blocks | 2 | 25/72 |
| n_blocks | 4 | 23/68 |

**`lr` is the dominant axis by a wide margin; depth is irrelevant over this range.** 13 of the 18
per-task winners are at `3e-4`. At `1e-3` the model does not merely underperform, it *collapses* on
several tasks — driver-dnf 63.57, driver-top3 65.60, user-repeat 60.00, study-outcome 53.43 — which
matches the headline's one-collapsed-seed-in-three instability signature. The two-level model is
**lr-fragile**, and the headline used the fragile end of the range at 2.3× the parameter budget.

d_model=128 edges out 256, consistent with the earlier RT arch grid. Both `128/*` configs are inside
`CLAUDE.md`'s ~30M cap; `256/4` (35.5M) is not.

## Encoder: qwen vs harrier is a wash

6/9 vs RT either way. Per-task, harrier wins 5 and qwen 4, with most gaps far inside the seed noise
the headline measured (cv up to 28.5%). **This grid does not resolve the encoder question** — at 1
seed it cannot. The one non-trivial gap is user-ignore (qwen 89.77 vs harrier 86.84), and it comes
from a single `1e-3` cell in the collapse-prone regime, so it is the least trustworthy cell to read.

## What this does NOT establish

* **Single seed.** Per-task best-of-8 at 1 seed selects noise as well as signal, and the headline
  showed this model's seed spread is large and *asymmetric* (one collapsed seed, not a spread). The
  6/9 figures are **best-of-8 and therefore optimistic**; a multi-seed confirmation of the winning
  configs is required before any of it is claimed.
* **No matched RT control in this codebase.** The comparison is against the published leaderboard.
  The internal RT grid (`results/rt_arch_grid_cfg0-7_qwen/`) is `d_model=128/n_blocks=4` and is not
  capacity-matched to every cell here.
* **Not bisectable.** A win cannot be attributed to any individual two-level switch without
  `--phase phase0a`/`phase0b` runs.
* rel-trial regression (site-success, study-adverse) is beaten by RT under *every* config and both
  encoders — that is the one consistent negative and is not a tuning artifact.
