# Large-batch arm — pre-registered predictions, 2026-08-03

Written **before** the arrays finish, so the results can falsify it. The last two loss sweeps each
carried a stated prediction: L1's held exactly (gains tracked target skew, 28/28 cells on the two
high-skew tasks, flat on the two low-skew ones) and the AUC surrogate's was **falsified** (I
predicted gains would track class imbalance; they tracked whether BCE had collapsed). Both were
useful *because* they were stated in advance.

## What is being changed, and against what

Everything so far ran at **batch 64, 10 epochs, patience 3** (confirmed from the sbatch submit lines
of every array). Two new arms, both at `d_model=128, n_blocks=2`, qwen, 1 seed, all 9 leaderboard
tasks, `--reg-loss l1 --bin-loss auc` (the two objectives the sweeps selected):

| arm | batch | epochs | patience | optimizer steps | samples seen |
|---|---|---|---|---|---|
| existing (`tl_reg_l1_*`, `tl_bin_auc_*`) | 64 | 10 | 3 | 1x | 1x |
| **A. `tl_bs512_qwen`** | **512** | **80** | **24** | 1x | 8x |
| **B. `tl_bs64_e80_qwen`** | 64 | **80** | **24** | 8x | 8x |

Arm B is not padding — it is what makes arm A interpretable. A alone changes batch size *and* the
compute budget together, so a gain would be unattributable. B holds the budget at A's level with the
old batch. Read as a 2x2 against the existing runs:

* **A > existing but A ≈ B** → the batch is irrelevant; it was only ever the 10-epoch budget, i.e.
  **underfitting**, which is what `WHY_NOT_RT.md` §3 already diagnosed on site-success.
* **A > B** → large batch genuinely helps, beyond the extra compute.
* **A ≈ B ≈ existing** → 10 epochs was already enough and the ceiling is architectural.

`patience` scales with `epochs` for a reason that is easy to get wrong: at 8x the batch an epoch is
1/8 the steps, so leaving patience at 3 would stop 8x earlier *in steps* and quietly undo the longer
budget. Both arms therefore use patience 24.

**Restriction to 128/2 is required, not a budget compromise.** `probe_batch_size.py` measured 512 as
fitting only at 128/2 (38.3 GiB); 256/4 OOMs above 256. Reaching an effective 512 there via
`--accum 2` would be *wrong for the binary tasks*: the pairwise AUC surrogate forms its pairs within
a batch, so accumulation leaves the pair count at 256's level while the record would read 512.

`--lr-set scaled` = {3e-4, 1e-3, 3e-3}. 3e-4 is retained as the unscaled control so "the optimum lr
moved" stays separable from "big batches help".

## Predictions

**P1 — the lr optimum moves up with batch size, and crosses over.** At batch 64 the sweep found
3e-4 far better than 1e-3 (33/69 cells beat RT vs 15/71), and 1e-3 collapsed outright on several
tasks. Batch 64 -> 512 is 8x; linear scaling puts the optimum near 2.4e-3, sqrt scaling near 8.5e-4.
So: **3e-3 should be the worst lr in arm B and among the best in arm A.** This is the sharpest
falsifiable claim here — a crossover, not a level shift. If 3e-3 is bad in both, batch size is not
buying gradient-noise reduction at this scale.

**P2 — site-success is the test case for underfitting, and it discriminates between A and B.** Most
of its grid cells sit at 0.88-0.98 NMAE against a 0.9713 constant predictor, with 151,407 train rows
and a median of only 109 cells/seed (well under the 512 cap) — so it is neither label-starved nor
context-truncated. If 8x the compute does not move it off ~0.82 toward RT's 0.734 in **either** arm,
underfitting is refuted as the explanation and the cause is architectural or in the readout.

**P3 — the AUC surrogate should gain superlinearly from a real batch increase, unlike every other
loss here.** Its gradient is built from in-batch positive-negative pairs, which grow as O(B^2 p(1-p))
rather than O(B). At batch 64 with user-ignore's 0.169 positive rate a batch holds ~11 x 53 = 583
pairs; at 512 it holds ~87 x 425 = 36,975, a 63x increase. **user-ignore is the specific prediction**:
it is the one binary task where the AUC surrogate *lost* to BCE (-0.65 best-of, 0/6 cells at qwen),
and it is also the most imbalanced, i.e. the most pair-starved at batch 64. If the pair-count
mechanism is right, user-ignore is where arm A helps most, and it must NOT show the same gain in arm
B. Caveat, stated up front: positive rate alone already failed to predict the BCE->AUC gain
(driver-top3 at 0.171 gained, user-ignore at 0.169 lost), so P3 rests on the pair-count argument
rather than on imbalance as such, and a null here is a real null.

**P4 — no prediction for rel-event, by construction.** `probe_seq_len.py` measured ~57% of every
rel-event seed's cells being silently dropped at `seq_len=512`. Its three tasks are reported for
completeness and must not be read as evidence either way until the fanout is fixed.

## Also newly wired in: router diagnostics

`expert_usage` / `mean_active_experts` / `specialization_probe` have existed since Phase D and
**have never been called** — no result JSON in this repo contains a routing field, and the grid saves
no checkpoints, so nothing already run can be probed after the fact (`WHY_NOT_RT.md` §2). Every
record from these arms now carries `router_usage`, `router_entropy_norm` (1.0 = uniform, ->0 =
collapsed onto one expert), `router_mean_active_k` and `router_distinct_experts`.

This tests the method's central claim directly, and it is the one measurement that could make the
rest moot: **if usage has collapsed onto a single expert, MoRE is RT with a wider FFN**, and every
comparison in this repo is consistent with that — the RGCN null `CLAUDE.md` names as the honest
negative outcome. The diagnostic is exception-guarded and records `router_error` on failure; a
measurement must not be able to destroy the result it was bolted onto.

## Reading the results

`batch_size` is stamped per record and `run_index` halves it on CUDA OOM, so **check it before
comparing** — a job that silently fell back to 256 is no longer step-matched to the rest of arm A.
`epochs`, `patience` and `grid_set` are now stamped too; before this they were not, and a 10-epoch
run was indistinguishable from an 80-epoch one in its own JSON.
