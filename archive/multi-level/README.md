# archive/multi-level — retired on 2026-08-12

Everything here was reachable in the live tree and is not part of the adopted model. It is kept
rather than deleted because **result directories under `results/` cannot be regenerated without it**.
Nothing here is imported by `gloss/`; these files are not on the import path and are not tested.

The adopted model is the two-level (cell, row) encoder with the flex cell-attention backend, the
additive `Broadcast`, and `route_on: signature`. Every switch below had exactly one live value in
every reported result, which is why collapsing them changed no behaviour.

## What is here and why it went

| retired | why | results that used it |
|---|---|---|
| `model/rt_substrate.py` (`RTSubstrate`, four relational masks, `MaskedAttention`) | the single-level baseline; two-level beat it on 5 of the 6 tasks where both ran, and the leaderboard's RT-from-scratch row covers the comparison | `arch: rt` records |
| `model/entity_head.py` (`EntityHead`) | the RT-style seed-cell readout; `head.mode: seed_cells` existed for the Phase 0 parity check, which `RowTokenHead` won | `arch: rt` records |
| `model/row_to_cell.py` (`RowToCellAttention`) | `broadcast: attention`. Measured worse on 7 of 8 tasks; user-ignore collapsed to 76.34 ±6.23 vs 83.44 | `results/r2c_attn/` |
| `model/recency_stats.py` (`RecencyOrderChannel`) | the `recency_channel` x-arms; never adopted | `results/x_base_*`, `results/x_full_*` |
| `model/moe_ablations.py` (S/C/P/H) | shared expert / cosine router / top-p / hierarchical gate — the superseded v2 ablation era, reachable only from `run_ablation.py` | v2 ablation records |
| `scripts/run_ablation.py`, `scripts/run_ablation_phases.py` | the phase-0a/0b/full ladder and the S/C/P/H runner | v2 ablation records |
| `scripts/capture_parity_baseline.py`, `tests/test_parity.py`, `tests/fixtures/` | the §6 bit-for-bit `arch: rt` parity guard — it guarded a refactor that has completed | — |
| `scripts/measure_substrate.py`, `scripts/bench_arch.py` | measured/benchmarked the single-level substrate and the rt-vs-two_level A/B | — |
| `tests/test_row_to_cell.py`, `test_recency_stats.py`, `test_hmoe.py` | tests for the above | — |

## Switches collapsed rather than archived

These had no separate class to move; they were `if` branches, now pinned to the value every reported
run used. To recover one, read this table and the git history before that date — do not re-derive.

- `cell.attention`: `four_mask` → **`full`** (padding-only mask)
- `cell.rope_time`: `false` → **`true`**
- `time.mode`: `buckets` (20 log-decade bins, 14 empty on rel-f1) → **`rope`**
- `row.pool_query`: `mean` / `signature` / `hidden` → **`hybrid`**
- `row.role_bias`: `none` → **`name_derived`**
- `row.time_bias`: `none` / `fixed_basis` → **`rope`**
- `row.ffn`: `dense` → **`moe`**
- `broadcast`: `film` / `none` / `attention` → **`additive`**
- `head.mode`: `seed_cells` → **`row_token`**
- `route_on`: `hybrid` / `hidden` / `value` / `identity` / `dense_wide` → **`signature`** (with
  `dense` kept, because `signature` vs `dense` is the headline comparison)

## What was deliberately NOT removed

- **The `sdpa` cell backend.** `flex_attention` requires CUDA and `head_dim >= 16`, so `sdpa` is the
  only path the CPU test suite can run — including the test that asserts flex matches it. It is a
  non-default fallback, not a live arm.
- **`route_on: dense`.** It is the RT-FFN control for the headline claim, not an ablation arm.
- **`RowMoE(use_shared=True)`.** The row experts stay shared+routed while the cell experts are
  routed-only; that asymmetry is measured, not an oversight.
