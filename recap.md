# RECAP — MoRE / `multi-level`, 2026-07-29 (user back 2026-07-30; **may lose this node**)

Self-contained handoff, possibly into a fresh conversation. Repo:
`/gpfs/milgram/project/ying_rex/hm638/gloss` (Milgram). Use `.venv/bin/python`.
Current node `r818u23n11`. **The user expects to give up this node soon — assume the running array may
die and that scratch paths need re-checking (§2 symlinks).**

Supersedes the 2026-07-02 recap; its still-valid hard rules are preserved in §2. **Read §2.**

---

## 0. What this is, in 30 seconds

**MoRE** = Relational Transformer substrate + a **Mixture-of-Experts FFN** whose router reads a
**value-free relational signature** (frozen-LM name embedding ⊕ modality ⊕ recency). Goal: **beat
`RT (from scratch)` and `GelGT` on the RelBench leaderboard**, per task.

The **`multi-level` branch** (current, 22 commits ahead of `main`) replaces the flat cell-token
substrate with a **two-level (cell, row)** encoder: row tokens, a row signature, temporal RoPE at both
levels, name-derived FK-role biases, one full cell attention instead of four masked ones, a
cross-attention row encoder, and **MoE at BOTH levels**.

**Read in order:** `CLAUDE.md` → `changes.md` (the plan) → **`amendments.md` (where the plan was
measurably wrong — six falsified claims + five bugs, each with its measurement)**. `report.md`, cited
throughout `changes.md`, **does not exist in the repo**; every `report X##` citation is unverifiable.
Never trust a `changes.md` figure that `amendments.md` supersedes.

---

## 1. WHAT IS RUNNING RIGHT NOW — four chained jobs

**The 8-GPU QOS cap is the binding constraint.** All of these SHARE 8 concurrent slots — submitting
more arrays does not add throughput, it only changes ordering. Ordered so the comparable (qwen)
results land first.

| job | what | notes |
|---|---|---|
| `29029490` | **headline**: 27 two-level runs, `--phase full`, qwen | `results/two_level_full/` |
| `29029525` | harrier **table/role** cache build, **h100-pinned** | see the a40 warning below |
| `29029522` | **qwen grid**: 72 jobs | `results/tl_grid_qwen/` |
| `29029526` | harrier grid: 72 jobs, `afterok:29029525` | `results/tl_grid_harrier/` — **may not finish** |

**Grid** (`run_gridsearch.py --arch two_level`): `d_model {128,256} × n_blocks {2,4} × lr {3e-4,1e-3}`
= 8 configs × 9 tasks × **1 seed**, 10 epochs. Targeted at the *losses*, not at polishing a winner:
the earlier RT grid found `d_model=128` best while the headline runs use 256, and a two-level block has
**six** sublayers to RT's five, so `n_blocks=8` is 48 sublayers.

> **⚠ harrier needs an h100, NOT `--gpus=1`.** harrier is Gemma-3-27B in **bf16 ≈ 54 GB**; the `gpu`
> partition carries a40 (**46 GB**) *and* h100. An a40 assignment OOMs the cache build, and then
> `afterok` leaves the whole harrier grid in `DependencyNeverSatisfied` — a silent, hard-to-diagnose
> stall. Always pass `--gpus=h100:1` for any harrier job.
>
> **Cache state:** qwen is COMPLETE (67 entries = 45 cols + 9 tables + 13 roles). harrier had only 45
> (columns only) — P0.4's table/role strings were never embedded with it, hence job `29029525`.

### Aggregate the grids
```bash
.venv/bin/python scripts/run_gridsearch.py --aggregate --arch two_level --seeds 1 \
    --out-dir results/tl_grid_qwen        # best config per task vs RT; regression rows carry NMAE
```

---

## 1a-bis. 2026-07-30 — what failed, what was fixed, what is running now

61 of the 2026-07-29 array tasks failed; see `amendments.md` §9 for the full diagnosis.

* **57 failures, all rel-event:** `max_rows_per_seed exceeded: 162 rows but R=160`. `MAX_ROWS=160`
  was a rel-f1 measurement asserted on every DB, and `train/loop.py` never passed the config key, so
  the assert's own advice was a no-op. Fixed in `e9d98a1`: `max_rows=None` fits R to each batch
  (semantics-preserving — padding rows are masked everywhere — and ~6x cheaper on rel-f1).
* **4 failures:** the intermittent MET offset assert (`num_workers>0` only). Its layout is still
  unknown; the fall-through now raises the actual numbers, because exceptions cross the DataLoader
  worker boundary and `print()` does not.
* **Both grid arrays ran the WRONG experiment** — `arch=rt`, `encoder=qwen`, RT configs 0-7. Neither
  `--arch two_level` nor `--encoder harrier` reached the jobs. Their output is parked in
  `results/rt_arch_grid_cfg0-7_qwen{,_DUPLICATE}/` with a README.

| job | what | out-dir |
|---|---|---|
| `29030109` | headline rel-event redo, indices 18-26 | `results/two_level_full/` (was 18/27) |
| `29030122` | **two-level** grid, qwen, 72 jobs, `afterany:29030109` | `results/tl_grid_qwen/` |
| `29030123` | **two-level** grid, harrier, 72 jobs, `afterany:29030122` | `results/tl_grid_harrier/` |

Chained, not parallel: all three share the same 8-GPU QOS cap, so extra arrays add ordering, not
throughput. qwen first because it is the one comparable to the headline runs.

```bash
sbatch --array=0-71%8 scripts/run_gridsearch.sh --arch two_level --phase full \
    --seeds 1 --epochs 10 --encoder qwen --out-dir results/tl_grid_qwen
```

> **Before letting any array run to completion, fingerprint its first finished record.** §9.3 cost 96
> GPU-jobs and a whole harrier cache build because nobody did:
> ```bash
> .venv/bin/python -c "import json,glob;d=json.load(open(sorted(glob.glob('results/tl_grid_qwen/*.json'))[0]));print({k:d.get(k) for k in ('arch','phase','encoder','d_model','n_blocks','lr')})"
> ```
> `arch`, `phase`, `lr` and `encoder` are in every JSON precisely so this is a one-liner.

The harrier schema cache **is** complete: its key set is byte-identical to the known-good qwen cache
for all three DBs (67/134/134), so `29029525` did its job and no model load happens at train time.

---

## 1b. The headline array (29029490)

```
sbatch --array=0-26%8 scripts/run_ablation.sh \
  --datasets rel-f1 rel-trial rel-event --tasks leaderboard \
  --signals signature --seeds 3 --encoder qwen \
  --arch two_level --phase full \
  --out-dir results/two_level_full
```
27 runs = 9 leaderboard tasks × 3 seeds, `signature` router, qwen, `gpu`/h100:1, `%8`.
`--phase full` = cell RoPE + one full cell attention + name-derived role bias + row RoPE + **MoE at both
levels** (cell routed-only, row **shared+routed**).

### Check / resume
```bash
cd /gpfs/milgram/project/ying_rex/hm638/gloss
squeue -j 29029490 -h -o "%T" | sort | uniq -c
ls results/two_level_full/*.json | wc -l                 # of 27
for f in logs/slurm/gloss_abl_29029490_*.err; do grep -qE "Traceback" "$f" && echo "FAIL $f"; done
.venv/bin/python scripts/run_ablation.py --aggregate \
  --datasets rel-f1 rel-trial rel-event --tasks leaderboard --signals signature --seeds 3 \
  --out-dir results/two_level_full --baseline signature
```
**If it died, resubmit the identical command** — there is an idempotent skip-if-done guard keyed on the
output filename, so finished configs are not recomputed.

### Results so far (8 of 27) — **MIXED, and the honest read is 1 win / 2 losses on rel-f1**

NMAE-correct (regression converted as `test_mae / train-std`; the pipeline stores RAW mae):

| task | metric | two-level | our RT | RT (lb) | GelGT | |
|---|---|---|---|---|---|---|
| rel-f1/driver-top3 | AUROC | **89.69** (3 sd) | 82.65 | 82.70 | 84.10 | **beats both** |
| rel-f1/driver-dnf | AUROC | 70.46 (2 sd) | 77.40 | 78.70 | 76.10 | **loses ~8** |
| rel-f1/driver-position | NMAE | 0.6290 (1 sd) | 0.4303 | 0.477 | 0.531 | **loses badly (+32%)** |

**Do not generalise from driver-top3.** Its first two seeds (0.910, 0.915) looked like a clean win; the
third came in at **0.8655**, so the real spread is ~0.05. And the two other rel-f1 tasks go the *other*
way by a similar margin. **Our RT reproduces the leaderboard closely** (82.65 vs 82.70; 77.40 vs 78.70),
so the reference is trustworthy and these losses are real, not a broken baseline.

That is what the grid search is for: testing whether the losses are a **capacity/LR mismatch** rather
than the mechanism being wrong.

**One matched-seed comparison exists.** A partial RT baseline (`results/baseline_qwen/`, 6 runs,
cancelled on the user's instruction) covered driver-top3 **seed 1** at identical encoder/epochs/batch/
seq_len — only the architecture differs:

| driver-top3, seed 1 | test AUROC |
|---|---|
| `arch=rt` | 0.7882 |
| `arch=two_level` | **0.9154** |

**Why this is NOT yet a result:** 2 seeds of 3, on **1 task of 9**, and driver-top3 is the **noisiest
task in the set** — RT's own two seeds spanned 7.7 AUROC points (0.788 / 0.865). A jump this size should
raise suspicion before enthusiasm. The real test is rel-trial and rel-event (different schemas; rel-event
is where the NaT fix actually changed the data). **Wait for the aggregate.**

---

## 2. HARD RULES / GOTCHAS (preserved — do not violate)

- **NEVER run `dense` or `dense_wide`.** The user was emphatic. RT-from-scratch on the leaderboard IS the
  dense baseline. Aggregate against the base router (`--baseline signature`).
- **Only the 9 RT-reported leaderboard tasks** (`--tasks leaderboard`), never all 18.
- **Metrics:** binary = **AUROC** (leaderboard ×100); regression = **NMAE = MAE / train-std** (lower
  better). `run_ablation.py` stores **raw `test_mae`, NOT NMAE** — divide by
  `finetune.target_stats(task)[1]`. `gloss/eval/leaderboard.py` pins this conversion in code; use it
  rather than converting by hand.
- **`gloss/data/graph.py:_patch_multiembedding_offset` — DO NOT lose it.** With `num_workers>0`,
  embedding columns return from a DataLoader worker with a shifted `MultiEmbeddingTensor.offset` →
  torch_frame `assert offset[0]==0` crash. The patch rebases content-preservingly and **deliberately
  falls through** on layouts it cannot verify ("no silent fix" — correct; a wrong rebase corrupts
  embeddings instead of crashing).
  **→ KNOWN WORKAROUND: re-run the few affected configs with `--num-workers 0` (bulletproof).**
  This is sporadic (~6 configs in a previous 432-job study), so it is a per-config retry, not a blocker.
- **8-GPU cap is a hard QOS limit** (`QOSMaxGRESPerUser`); submitting more just queues. `%8` is right.
- **`run_ablation.py` has NO OOM-retry** (unlike `run_gridsearch.py`). If a config OOMs on h100 at batch
  64, it just fails — watch for missing result files and re-run with a smaller batch.
- **This cluster = Milgram: `gpu` partition, `h100:1`.** `run_ablation.sh` / `prep.sh` used to hardcode
  another cluster's `gpu_h200`/`h200:1` and **could never schedule**; both are now FIXED in the committed
  scripts (`fb0ccfb`). `sinfo`: `gpu` = a40:4 + h100:4. No h200 here.
- **rel-event cache symlinks — fragile, re-check after any scratch cleanup.** `env.sh` repoints caches to
  `~/scratch60/gloss/{relbench,graph_cache,schema_cache}`. rel-f1/rel-trial auto-download; **rel-event
  cannot** (manual Kaggle file). Required:
  `~/scratch60/gloss/relbench/rel-event -> ~/scratch60/relbench/rel-event`,
  `.../graph_cache/rel-event -> <repo>/data/graph_cache/rel-event`,
  `.../schema_cache/rel-event -> <repo>/data/schema_cache/rel-event`.
  If they vanish every rel-event task fails with `RuntimeError: Dataset not found ...
  event-recommendation-engine-challenge.zip`. **Re-create the symlinks; do NOT auto-download rel-event.**
- **`data/schema_cache/` is NOT the cache the cluster reads.** `env.sh` sets `GLOSS_SCHEMA_CACHE` to
  scratch60; the repo-relative dir is a hermetic-test fallback holding stale dev leftovers. An earlier
  encoder-coverage table was read from the wrong one and was wrong. Always check
  `~/scratch60/gloss/schema_cache/`.
- **Milgram has NO working GitHub push auth** (HTTPS no creds, SSH key unregistered; `fetch` works).
  Commits stay local — the user pushes from an authenticated machine.
- **`results/` is gitignored.** Regenerate leaderboard numbers with `scripts/fetch_leaderboard.py`.

---

## 3. WHAT IS BUILT — all of `changes.md` §1–§4. **211 tests green.**

| item | file |
|---|---|
| P0.1 role vocabulary, **triple-keyed** `(child, col, parent)` | `data/graph.py` |
| P0.2 `adj_role` row adjacency, P0.3 hop (BFS) | `data/row_graph.py`, `data/collate.py` |
| P0.4 table/role **name** embeddings (qwen-cached, all 3 DBs) | `text/schema.py` |
| P0.5 **pinned** stype enum (`N_STYPES = 10`, bundle-independent) | `text/schema.py` |
| Fixed-frequency time ladder + `b_untimed` + `b_clamped` | `model/time_encoding.py` |
| RowSignature / RowPool / RowAttention / Broadcast / RowMoE | `model/row_level.py` |
| TwoLevelBlock / TwoLevelSubstrate, cell RoPE, full cell attention | `model/two_level.py` |
| `RowTokenHead` (§3.8) | `model/heads.py` |
| `arch ∈ {rt, two_level}` | `model/more.py` |
| §4 config (`model.two_level`, `time`) | `configs/default.yaml` |
| `--arch` / `--phase {phase0a,phase0b,full}` | `scripts/run_ablation.py`, `eval/ablation.py` |
| bit-for-bit parity guard | `tests/test_parity.py`, `tests/fixtures/` |
| leaderboard comparison (RT + GelGT) | `eval/leaderboard.py`, `scripts/fetch_leaderboard.py` |

`rt_substrate.py` is **untouched** so `arch: rt` stays a valid A/B baseline — that is what the parity
guard protects.

**Diagnostic scripts (all reusable):** `measure_substrate.py` (R, mask density, τ stats, ladder audit) ·
`probe_sampler_causality.py` (does `t4` exist) · `probe_fk_role_collision.py` (role-vocabulary regression
check) · `probe_relevent_time.py` (NaT / clamped-Δ) · `probe_met_offset.py` (the offset assert) ·
`capture_parity_baseline.py`.

---

## 4. THE FINDINGS THAT MATTER (full detail + measurements in `amendments.md`)

1. **P0.1 was a vocabulary bug, not plumbing.** Role ids were keyed on the FK **column name alone**:
   rel-f1 had 4 roles instead of 13, rel-trial **6 instead of 15** (all ten `nct_id → studies` relations
   collapsed to one), rel-event 4 instead of 7. **Phase 2's role bias would have been a guaranteed null
   for a mechanical reason.** Fixed; `probe_fk_role_collision.py` asserts 13/15/7.
2. **NaT timestamps encoded as the year 1677.** `to_unix_time` maps `NaT` to `pd.Timestamp.min//1e9` =
   `-9223372037` — a *finite* sentinel — and `collate.py` marked whole timed *tables* as timed. ~65k
   rel-event rows read as "336 years before the seed" (τ = 23.08). Fixed with a per-**row** plausibility
   floor (1800-01-01). **Every pre-existing rel-event number was computed under this bug.**
3. **`seq_len=512` BINDS on rel-event** (p90 = max = 512, ≥10% of seeds truncated). The "0% truncation"
   claim was generalised from the only two DBs then measured. Phase 5's headroom premise fails there.
4. **`t4` (path-causal masking) is a SAMPLER change, not a mask change.** rel-event: 55% of hop-adjacent
   pairs have the child dated LATER than its parent (worst +197.6 d). Masking cannot impose an ordering
   the sample never had. **Not implementable as `changes.md` specifies.**
5. **The ladder band `[0.05, 5.0]` was wrong** — derived from τ's *range* (22) when what matters is its
   *spread* (σ ≈ 1.6–1.9), leaving 2 of 8 channels dead. Now `[0.3, 5.0]`, measured.
6. **Seed rows had Δ clamped to 0** (35% of rel-event task rows are dated after their own seed), making
   "the query row" identical to "maximally recent" — and that τ=0 mass was the *sole* cause of the
   ladder's wraparound. Fixed with `b_clamped` + an indicator channel in `feats()`.
7. **The parity guard works and is correctly scoped.** It fired on P0.5 (7 failures, `stype_emb 2→10`,
   64/95 params differing) and was re-captured deliberately; it stayed correctly quiet on `b_clamped`,
   which exists only on the two-level path.
8. **val ≪ test is expected here, not a bug.** RelBench splits are temporal and *disjoint eras* — rel-f1
   train 1994–2004, val 2005–2008, **test 2010–2013**. With ~600-row splits, ±0.03–0.04 AUROC noise, and
   val being a *selected* max over epochs, per-task gaps under ~0.05 are indistinguishable from noise.
   **Weight the 9-task aggregate, never a single task.**

---

## 5. OPEN THREADS / TODO

1. **[watch] the array.** Aggregate when it fills (§1). Expect driver-top3 to be noisy.
2. **[bug, sporadic] the offset assert.** RT baseline index 6 (rel-f1/driver-top3 seed 0) died on it; the
   two-level run of the *same index* **succeeded** → nondeterministic. Reproduces **only with
   `num_workers>0`**; all 3 splits collate cleanly at `workers=0`; the patch's own `off.numel() >= 2`
   guard **excludes** the failing case, so the hypothesis is a **zero-column** MET with nonzero offset
   base — **UNCONFIRMED**, because worker `print()` does not survive. To confirm, make
   `probe_met_offset.py` **write the record to a file** from inside the worker. **Do not widen the patch
   on inference.** Practical fix meanwhile: re-run that config with `--num-workers 0`.
3. **[decision] no RT baseline was run** (user cancelled; 6 partial runs kept). Consequence: a null
   two-level result is **not bisectable** — run `--phase phase0a` then `phase0b` to attribute. `phase0b`
   differs from `phase0a` in exactly one switch (test-enforced).
4. **[open, §3.7] Phase 4 arm design.** `r1` (both levels) differs from `r0` in **two** ways: the row
   level exists *and* its experts are shared+routed. Isolating the shared expert needs its own arm.
5. **[open] `PROGRESS.md` not appended** for this work (CLAUDE.md asks for it per phase).
6. **[deferred, unchanged]** LODO zero-shot transfer (`signature ≫ identity`), masked-cell pretraining,
   true sparse MoE dispatch, recommendation/link tasks, the recency-axis ablation.

---

## 6. KEY FILES

- `CLAUDE.md` — normative rules. Updated: MoE is now at **two granularities**; cell experts routed-only,
  row experts shared+routed (intentional asymmetry, commented so it is not "harmonised" away).
- `changes.md` — the two-level plan (§1 P0, §3 math, §4 config, §5 phases, §6 tests, §9 resolve-first).
- **`amendments.md`** — what was wrong with it, measured. **Read before trusting `changes.md`.**
- `gloss/eval/ablation.py` — `run_config` (one array task), `variant_label`, `aggregate`, `format_table`.
  **`arch` is part of the `variant`**, so `two_level` and `rt` never share a filename or get averaged
  together (test-enforced).
- `results/leaderboard_baselines.json` — RT-from-scratch + GelGT, all 9 tasks (re-scraped, zero drift).
- `results/two_level_full/` — the run in flight. `results/baseline_qwen/` — 6 partial RT runs.

```bash
.venv/bin/python -m pytest tests/ -q                                # 211 passed; must stay green
.venv/bin/python scripts/run_train.py --dry-run --arch two_level    # real-data forward + row-graph summary
```
