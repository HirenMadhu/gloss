# Two-level headline array — 27/27 runs (`29029490` + `29030109`)

`signature@two_level`, `--phase full`, qwen, 9 RelBench leaderboard entity tasks x 3 seeds, 10 epochs,
TEST set. Architecture: **MoRE defaults, `d_model=256`, `n_blocks=8`** — no config search was run.

Binary = AUROC x100 (higher better). Regression = **NMAE = MAE / train-target-std** (lower better);
the pipeline stores raw `test_mae`, so NMAE is computed here with `target_stats(task)[1]`.

## vs the RelBench leaderboard

| dataset/task | metric | MoRE 2-level (3 seeds) | RT (scratch) | GelGT | vs RT | |
|---|---|---|---|---|---|---|
| rel-f1/driver-top3 | AUROC | **89.69 ± 2.74** | 82.70 | 84.1 | +6.99 | WIN |
| rel-f1/driver-position | NMAE | **0.4735 ± 0.1348** | 0.4775 | 0.531 | −0.0040 | win (noise) |
| rel-f1/driver-dnf | AUROC | 71.37 ± 2.26 | 78.70 | 76.1 | −7.33 | loss |
| rel-trial/study-outcome | AUROC | 60.91 ± 7.70 | 68.60 | 72.5 | −7.69 | loss |
| rel-trial/study-adverse | NMAE | 0.1763 ± 0.0105 | 0.1306 | 0.126 | +0.0457 | loss |
| rel-trial/site-success | NMAE | 0.9670 ± 0.0263 | 0.7341 | 0.732 | +0.2329 | loss |
| rel-event/user-ignore | AUROC | 82.41 ± 1.41 | 85.10 | 87.8 | −2.69 | loss |
| rel-event/user-repeat | AUROC | 67.93 ± 12.86 | 79.70 | 83.6 | −11.77 | loss |
| rel-event/user-attendance | NMAE | 0.5536 ± 0.0225 | 0.5040 | 0.317 | +0.0496 | loss |

**2 of 9 vs RT (from scratch), and the driver-position "win" (−0.004) is far inside its own ±0.135.**

## vs OUR OWN RT, same codebase (`results/rt_arch_grid_cfg0-7_qwen/`)

The mislabelled RT grid turned out to be the useful internal control — same repo, same tasks, same
qwen cache, `d_model=128/n_blocks=4`, best of 8 configs at 1 seed:

| dataset/task | 2-level (3 seeds) | our RT best-of-8 | our RT median |
|---|---|---|---|
| rel-f1/driver-dnf | 71.37 | 83.82 | 82.17 |
| rel-f1/driver-top3 | 89.69 | 90.41 | 88.38 |
| rel-f1/driver-position | 0.4735 | 0.4084 | 0.4325 |
| rel-trial/study-outcome | 60.91 | 72.09 | 70.07 |
| rel-trial/study-adverse | 0.1763 | 0.1503 | 0.1556 |
| rel-trial/site-success | 0.9670 | 0.8344 | 0.9298 |

**0 of 6.** And it is not a best-of-8 optimism artifact — the two-level model also loses to our RT's
*median* config on all six.

## The result is confounded by capacity, and the first grid config proves it

The headline ran the MoRE default `d_model=256, n_blocks=8`. A two-level block has **six** sublayers
to RT's five, so `n_blocks=8` is 48 sublayers. Grid config 0 — the *smallest* point, `d_model=128,
n_blocks=2, lr=3e-4`, 1 seed — beats the headline on **8 of 9 tasks**:

| dataset/task | 2L 256/8 (3 seeds) | 2L 128/2 (1 seed) | RT (scratch) |
|---|---|---|---|
| rel-f1/driver-dnf | 71.37 | **83.23** | 78.70 |
| rel-f1/driver-top3 | 89.69 | **90.99** | 82.70 |
| rel-f1/driver-position | 0.4735 | **0.4143** | 0.4775 |
| rel-trial/study-outcome | 60.91 | **67.79** | 68.60 |
| rel-trial/study-adverse | 0.1763 | **0.1692** | 0.1306 |
| rel-trial/site-success | 0.9670 | **0.9214** | 0.7341 |
| rel-event/user-repeat | 67.93 | **77.81** | 79.70 |
| rel-event/user-attendance | 0.5536 | **0.3974** | 0.5040 |
| rel-event/user-ignore | 82.41 | 79.99 | 85.10 |

At `128/2` the model beats RT (from scratch) on **4 of 9** (driver-dnf, driver-top3, driver-position,
user-attendance) rather than 2, and several losses shrink to near-ties (study-outcome 67.8 vs 68.6,
user-repeat 77.8 vs 79.7).

**So the 27-run table above should NOT be read as the verdict on the two-level architecture.** It is
the verdict on one over-sized configuration of it. The grid (`29030571`) exists to settle this and
still has 7 of 8 configs to run.

## Seed instability is the other story

Coefficient of variation across the 3 seeds:

| dataset/task | cv | seeds |
|---|---|---|
| rel-f1/driver-position | 28.5% | 0.3918, **0.6290**, 0.3996 |
| rel-event/user-repeat | 18.9% | **53.09**, 74.71, 75.99 |
| rel-trial/study-outcome | 12.6% | 62.89, **52.42**, 67.43 |
| rel-trial/study-adverse | 5.9% | 0.1705, 0.1700, 0.1884 |
| rel-event/user-attendance | 4.1% | 0.5490, 0.5338, 0.5781 |
| rel-f1/driver-dnf | 3.2% | 73.19, 68.83, 72.08 |
| rel-f1/driver-top3 | 3.1% | 91.00, 91.54, 86.55 |
| rel-trial/site-success | 2.7% | 0.9825, 0.9366, 0.9820 |
| rel-event/user-ignore | 1.7% | 83.95, 82.09, 81.20 |

The top three are **one collapsed seed out of three**, not uniform spread — a training-stability
signature, consistent with the 48-sublayer depth. It also means **single-seed grid numbers are noisy**
and the grid's per-task winners will need a multi-seed confirmation run before anything is claimed.

## Caveats that bound every number here

* No RT baseline was run in *this* configuration (the user cancelled it; 6 partial runs sit in
  `results/baseline_qwen/`). The comparisons above use the published leaderboard and the 128/4 RT
  grid — neither is capacity-matched to `256/8`.
* A null or negative two-level result is **not bisectable** without `--phase phase0a` / `phase0b`
  runs to attribute it to individual switches.
* 10 epochs, one `lr`. More parameters at a fixed lr can simply be undertrained.
