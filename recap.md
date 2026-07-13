# RECAP — MoRE, 2026-07-02 (user back ~2026-07-04)

Self-contained handoff. Picking this up possibly in a fresh conversation. Repo:
`/gpfs/milgram/project/ying_rex/hm638/gloss` (Milgram cluster). Use `.venv/bin/python`.

---

## 0. What this project is (30-sec version)
**MoRE** = Relational Transformer (RT) substrate + a **Mixture-of-Experts FFN** whose router reads a
cell's **value-free relational signature** (frozen LM column-name embedding + modality + recency). Goal:
**beat `RT (from scratch)`** on the RelBench leaderboard (per task). See `idea.md`, `implementation.md`,
`CLAUDE.md`, and `moe_ablation.md` (the v2 ablation design).

---

## 1. WHAT IS RUNNING RIGHT NOW — GRID SEARCH RESUME (array 28977028)
Pivoted back to the **architecture grid search** (2026-07-05). Resumed the missing **1,672 of 2,592** grid
jobs (920 were already done) as array **28977028** (`%8`, `gpu`/h100:1). `run_gridsearch.py` now has an
**idempotent skip-if-done guard**. Results → `results/gridsearch/NNNNN.json`; aggregate with
`run_gridsearch.py --aggregate`. rel-event loads fine here (symlink fix §2).

**Why the pivot:** the **v2 S/C/P/H additions ablation is COMPLETE and a negative result** — no addition
consistently beats its base router (`signature`/`hidden`); only `+S` (shared expert) helps regression a
little (driver-position 0.398, user-attendance 0.408, both < RT). hidden×HMoE trains fine with the NaN
guard (§2) but doesn't win. Full 16-arm results in `results/v2_add_signature/` + `results/v2_add_hidden/`
(aggregate: `run_ablation.py --aggregate --out-dir results/v2_add_<router> --baseline <router>`).
driver-top3 test-AUROC is missing (transient relbench eval SQL glitch — works in isolation; re-run to
recover). The grid search (best `signature` configs beat RT 6/9, all small `d_model=128`) is the
productive axis.

--- older detail (the v2 ablation, now finished) ---
16 SLURM arrays, **432 jobs total**, `%8` (hard 8-GPU cap on this cluster), on **`gpu` partition / h100:1**.
Submitted 2026-07-02. IDs in `~/scratch60/.gloss_v2_jids`.

**Design:** two routers × the **S/C/P/H additions ladder** (forward select + backward leave-one-out).
- Routers: **`signature`** (the method) and **`hidden`**. NO `hybrid`, NO `identity`, **NEVER `dense`**.
- Additions: **S** shared expert (`--use-shared`), **C** cosine router (`--cosine`), **P** Top-P
  (`--top-p 0.7`), **H** HMoE (`--hmoe`).
- 8 configs per router (Full−H == R*+S+C+P, so 8 not 9):
  `base` · `+S` · `+SC` · `+SCP` · `Full(+SCPH)` · `Full−S(+CPH)` · `Full−C(+SPH)` · `Full−P(+SCH)`.
- **Datasets: rel-f1, rel-trial, rel-event — LEADERBOARD tasks ONLY (9 = 3/dataset), via `--tasks
  leaderboard`.** (rel-f1: driver-dnf/-top3/-position; rel-trial: study-outcome/-adverse, site-success;
  rel-event: user-repeat/-ignore, user-attendance.) User rule: only run tasks with an RT/GelGT baseline.
  3 seeds. `--epochs 30` (a CAP; early stopping patience=3 stops most runs earlier). Encoder `harrier`
  (schema caches built for all 3 DBs). N per array = 27 (9 tasks × 1 router × 3 seeds), `--array=0-26%8`.
- Output: `results/v2_add_signature/` (8 arrays) and `results/v2_add_hidden/` (8 arrays). Files named
  `{index}_{variant}.json` (e.g. `0000_signature+SCP.json`) so variants share a dir without collision.

**ETA ~1.5 days** at 8-wide/30-epoch (9 tasks, early stopping).

### Check progress / aggregate
```bash
cd /gpfs/milgram/project/ying_rex/hm638/gloss
squeue -u $USER -h -r -t RUNNING -o '%i' | grep -c '_[0-9]'      # GPUs in use (<=8)
ls results/v2_add_signature/*.json results/v2_add_hidden/*.json | wc -l   # of 864
# additions Δ vs the BASE ROUTER (never dense):
.venv/bin/python scripts/run_ablation.py --aggregate --out-dir results/v2_add_signature --baseline signature
.venv/bin/python scripts/run_ablation.py --aggregate --out-dir results/v2_add_hidden    --baseline hidden
```

---

## 2. HARD RULES / GOTCHAS (do not violate)
- **NEVER run `dense` or `dense_wide`.** The user was emphatic. RT-from-scratch on the leaderboard IS the
  dense baseline. Baseline aggregates on the base router (`--baseline signature` / `--baseline hidden`).
- **Metrics:** binary = **AUROC** (leaderboard shows ×100); regression = **NMAE = MAE / train-std**
  (lower better). The pipeline (`task.evaluate`) reports **raw MAE**, NOT NMAE — divide by
  `finetune.target_stats(task)[1]` (the train-std) to compare to the leaderboard. `run_gridsearch.py`
  already stored `test_nmae`; `run_ablation.py` stores raw `test_mae` — convert at read time.
- **Uncommitted collation fix in `gloss/data/graph.py`** (`_patch_multiembedding_offset`) — DO NOT lose it.
  With `num_workers>0`, rel-event embedding columns come back from a DataLoader worker with a shifted
  `MultiEmbeddingTensor.offset` → torch_frame `assert offset[0]==0` crash. The patch rebases the offset
  content-preservingly (no-op when offset[0]==0). Verified + tests green (66 passed). It's active for the
  v2 runs (they use `num_workers=8`). **Not on GitHub yet** (see §5).
- **8-GPU cap** is a hard QOS limit (`QOSMaxGRESPerUser`) — submitting more just queues (`%8` is fine).
- **This cluster = Milgram: `gpu` partition, `h100:1`.** The committed `run_ablation.sh`/`prep.sh` have
  the OTHER cluster's `gpu_h200`/`h200:1` baked in — override at submit with `--partition=gpu --gpus=h100:1`
  (that's what the 16 arrays did). No h200 here.
- **rel-event cache paths (2026-07-03 fix).** The pulled `env.sh` repoints all caches to
  `~/scratch60/gloss/{relbench,graph_cache,schema_cache}`. rel-f1/rel-trial auto-download there, but
  **rel-event can't** (needs a manual Kaggle file) → every rel-event task failed with `RuntimeError:
  Dataset not found ... event-recommendation-engine-challenge.zip`. Fix = **symlinks** pointing rel-event
  in the new roots at the real data:
  `~/scratch60/gloss/relbench/rel-event -> ~/scratch60/relbench/rel-event`,
  `.../graph_cache/rel-event -> <repo>/data/graph_cache/rel-event`,
  `.../schema_cache/rel-event -> <repo>/data/schema_cache/rel-event`.
  **If these symlinks vanish (scratch cleanup) or you re-`prep` rel-event, it'll break again** — re-create
  the symlinks (do NOT try to auto-download rel-event).

---

## 3. RESULTS SO FAR (context) — the arch grid search (CANCELLED, partial)
Before v2, an **architecture grid search** ran (arrays 28970564 + rerun 28975076, now cancelled). It swept
`signature`-router × {d_model,n_blocks,n_heads,d_ff,enc_channels,num_experts} × 9 RT tasks × 3 seeds.
**872/2592 partial results in `results/gridsearch/`.** Headline from that partial run:

**Best MoRE (signature) vs baselines — beats RT (from scratch) on 6/9, GelGT on 3/9:**
| Task | GelGT | RT(scratch) | MoRE best | >RT | >GelGT |
|---|---|---|---|---|---|
| rel-f1/driver-dnf (AUROC↑) | 76.1 | 78.7 | **82.9** | ✅ | ✅ |
| rel-f1/driver-top3 (AUROC↑) | 84.1 | 82.7 | **90.6** | ✅ | ✅ |
| rel-trial/study-outcome (AUROC↑) | 72.5 | 68.6 | 69.4 | ✅ | ❌ |
| rel-event/user-repeat (AUROC↑) | 83.6 | 79.7 | 79.5 | ❌ | ❌ |
| rel-event/user-ignore (AUROC↑) | 87.8 | 85.1 | 87.3 | ✅ | ❌ |
| rel-f1/driver-position (NMAE↓) | 0.532 | 0.478 | **0.395** | ✅ | ✅ |
| rel-trial/study-adverse (NMAE↓) | 0.126 | 0.131 | 0.161 | ❌ | ❌ |
| rel-trial/site-success (NMAE↓) | 0.732 | 0.734 | 0.840 | ❌ | ❌ |
| rel-event/user-attendance (NMAE↓) | 0.317 | 0.504 | 0.399 | ✅ | ❌ |

Notable: **best configs were all SMALL** (`d_model=128`, ~5–7M active params, top-2 of M experts). The
256/512 tiers hadn't beaten them when cancelled. Caveat: some winners were 1–2 seeds (winner's curse).

**Baselines saved: `results/leaderboard_baselines.json`** — RT (from scratch) + GelGT for ALL 21 RelBench
tasks (classification AUROC%, regression NMAE), pulled from HF leaderboard. Cross-checked: GelGT leaderboard
numbers match the GelGT paper exactly (classification direct; regression = paper-MAE ÷ train-std). RT (from
scratch) is the **target to beat**.

---

## 4. GIT STATE
- On `main`, HEAD = `2db8f45` ("ablation-v2: hybrid + S/C/P/H"), pulled from GitHub (fast-forward).
- **Uncommitted (keep all):** `gloss/data/graph.py` (collation fix, §2); `gloss/eval/ablation.py` +
  `scripts/run_ablation.py` (the **`--tasks leaderboard`** filter + `LEADERBOARD_TASKS`);
  `gloss/train/finetune.py` (**`gradient_clip_val=1.0`**) + `gloss/train/loop.py` (**non-finite-grad guard**,
  see below); `recap.md` (this file).
- **hidden×HMoE NaN (2026-07-04).** The 4 arms combining `hidden` router × `--hmoe`
  (Full/Full-S/Full-C/Full-P) **diverge to NaN during training** — specifically **NaN *gradients* while the
  forward stays finite** (so `gradient_clip_val=1.0`, which I also added, does NOT fix it — clipping a NaN
  grad stays NaN). Crashes at validation on the 5 binary tasks (sklearn rejects NaN preds). `signature`×HMoE
  and `hidden`×non-HMoE are fine. **Could not reproduce locally** (CPU/hash/1-batch hits a spurious
  numerical-encoder NaN that even `dense` triggers — unfaithful). This matches `moe_ablation.md`'s own
  prediction that **H is the highest-variance / most-likely-unstable addition.**
  - **Mitigation:** a non-finite-gradient guard in `loop.py` (`on_before_optimizer_step` → `nan_to_num_`
    the grads → 0) so NaN can't poison the weights → the job completes instead of crashing. Re-running the
    4 arms as **28976714–717**. **CAVEAT: if their AUROC comes out ~0.5 (init-level), the arm genuinely
    doesn't train under `hidden` routing — report hidden×HMoE as unstable rather than trusting the number.**
- **sporadic collation `offset[0]` asserts** (~6) slip past the graph.py patch (a 3rd layout). Re-run those
  few with `--num-workers 0` (bulletproof; arrays 28976701–703).
- Earlier session commits (`2bd515f`, `617aad9`: gitignore fix that TRACKS `gloss/data/graph.py`, the
  't'/'f' boolean-target fix, prep OOM fix, grid-search runner) are on GitHub via the user's push.

---

## 5. OPEN THREADS / TODO when you return
1. **v2 ablation results** — aggregate both dirs (§1). Look for: does **S (shared expert)** help
   (predicted robust win)? Do C/P/H help on complex `rel-trial` but hurt simple `rel-f1`? signature vs
   hidden router (the M0/base rows). Compare abs numbers to RT/GelGT in `leaderboard_baselines.json`.
2. **Commit the graph.py collation fix + push.** Push from an authenticated machine — **Milgram has NO
   working GitHub push auth** (HTTPS no creds, SSH key not registered; `git fetch` works so repo is public).
3. **Arch grid search** was cancelled at 34% (best small configs beat RT 6/9). Decide: resume, or fold the
   winning small-config insight into v2. `results/gridsearch/` has 872 partial JSONs +
   `scripts/run_gridsearch.py --aggregate`.
4. **run_ablation.py has NO OOM-retry** (unlike run_gridsearch.py). If HMoE (8 experts) OOMs at batch 64 on
   rel-event/h100, those arms fail — watch `results/v2_add_*/` for missing HMoE variants; reduce batch or
   add retry if so.

---

## 6. KEY FILES
- `moe_ablation.md` — the v2 ablation design (routers, S/C/P/H, tiers, predictions). **Read this first.**
- `gloss/eval/ablation.py` — `run_config` (one array task), `variant_label`, `format_table` (aggregate).
- `scripts/run_ablation.py` / `.sh` — the array runner + CLI flags (`--use-shared/--cosine/--top-p/--hmoe/
  --baseline/--signals`).
- `gloss/model/moe.py` — `MoEFFN` (+ shared/cosine/top-p), `HMoEFFN`, `ortho_loss`.
- `gloss/model/more.py`, `rt_substrate.py` — model + relational blocks (`route_feat` incl. `hybrid`).
- `gloss/data/graph.py` — graph build + `make_loader` + the collation fix.
- `results/leaderboard_baselines.json` — RT (from scratch) + GelGT, all tasks.
- Memory: `~/.claude/projects/.../memory/` (never-run-dense, leaderboard-metric-nmae, gridsearch-inflight).
