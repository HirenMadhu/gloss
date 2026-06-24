# PROGRESS

HALOS / `gloss` — v3 (documentation-conditioned geometry). Build log; one section per phase.

## Bootstrap (env + restructure)
- Spec rewritten v2→v3 (method paper; doc-generated attention geometry is the core). Rewrote `CLAUDE.md`
  to v3; updated project memory.
- Env was already provisioned (`.venv`, py3.12, torch 2.8.0+cu128, PyG 2.8, torch_frame 0.3, relbench,
  sentence-transformers, transformers, lightgbm, shap, lightning, hydra, wandb). Added build deps
  (`wheel ninja einops`) and the PyG compiled extensions (`pyg-lib`, `torch-scatter`, `torch-sparse`
  for `torch-2.8.0+cu128`) — the relbench temporal sampler needs `pyg-lib`.
- `tests/test_env.py`: imports + cuda12/cxx11abi asserts green; flash-attn xfail until built.
- flash-attn: prior build failed (venv lacked `wheel`). Fixed deps + resubmitted `scripts/build_flash_attn.sh`
  (SLURM job; archs 86;90). Its only consumer is the frozen Qwen encoder — off the Phase 0-3 critical path.
- Deleted the stale v2 scaffold (`proxy/ext/audit/train`, v2 model/eval, structured-card tests). New v3
  tree under `gloss/{data,docs,model,eval,utils}`; configs `default.yaml` + `rel-f1.yaml`.

## Phase 0 — substrate ✅
- `gloss/data/graph.py`: `build_gloss_graph` (wraps `make_pkey_fkey_graph` + `get_stype_proposal`,
  cheap deterministic `HashTextEmbedder` for cell text), per-DB vocabs (`node_type_id`; `fk_role_id` /
  `metapath_id` **canonicalized to the fkey column** so forward/reverse share a role and dual FKs stay
  distinct); `make_loader` (per-seed **disjoint** leakage-safe temporal `NeighborSampler`, `time_attr=time`,
  `temporal_strategy=last`).
- `gloss/data/collate.py`: `to_gloss_batch` → dense `GlossBatch` (node-level `[B,N_max]`; pairwise
  `[B,N_max,N_max]` attend_mask / metapath_id / fk_role_id / dt / tau / temporal_valid; per-seed
  seed_time + T_ctx). `tau = log(dt/T_ctx)` on both-timed attendable pairs (float64); timeless / zero-gap /
  >1-hop / pad → structural bucket (temporal_valid=False). Carries per-type `TensorFrame`s + `placement`
  for Phase-2 encoding.
- `gloss/utils/{seeding,config,logging}.py`.
- `scripts/run_finetune.py --dry-run`: builds rel-f1, samples a disjoint batch, prints shapes, leakage check = 0.
- **Tests (hermetic; real rel-f1 guarded by cache):** test_env, test_graph, test_collate, test_leakage,
  test_shapes — **36 passed, 1 xfailed**.
- **Decisions / deferrals:** self-label task-table nodes deferred to Phase 4 (relbench's graph excludes the
  task table; self-labels are a transfer-phase concern). Attention is dense within a seed's subgraph over
  ≤2-hop-reachable pairs (1-hop = FK relation, 2-hop = MULTIHOP bucket, else masked).
- DoD met: `run_finetune.py --dry-run` prints a rel-f1 batch; leakage/shape tests green.

## Phase 1 — doc corpus + grounding ✅
- `doc_corpus/rel-f1/docs.md` + `meta.yaml`: **Tier-2, blind** senior-dev prose (authored from schema +
  F1 domain knowledge only; no task/label/target referenced; ~65% coverage target; `constructor_results`
  + some columns deliberately undocumented). meta carries the blind attestation for the audit.
- `gloss/docs/corpus.py`: load/validate (blind ⇒ attestation required), sentence-chunk into spans,
  enumerate schema elements (table / column / fk_role) from a relbench DB or a spec dict, coverage report.
- `gloss/docs/grounding.py`: chunk→embed→top-K cosine→softmax-pool → `d_e`, `rel_e`; `d_null` fallback
  below threshold; regimes `full | null | shuffled_spans`. **Placebo fix:** permuting span rows is a no-op
  (sims recomputed), so `shuffled_spans` permutes the *element→doc assignment* (derangement) — same
  coverage/length, decorrelated meaning.
- `gloss/docs/cache.py`: `QwenEncoder` (frozen Qwen3-Embedding-4B via sentence-transformers, instruction
  on queries only), `HashEncoder` (dev/tests), idempotent content-hash `EmbeddingCache`.
- `scripts/build_doc_cache.py`: real Qwen run → **d_text=2560 confirmed**, idempotent (cache reload, no
  model reload), coverage emitted.
- **Calibration finding:** Qwen sims sit ~0.52–0.85 (high baseline), so the spec's 0.3 threshold grounds
  everything (100%). Calibrated `sim_threshold=0.60` → **0.725 partial coverage** (table .78 / col .67 /
  fk .92). Encoder-specific knob; revisit at the H1 gate (Phase 5).
- **Tests:** test_corpus, test_grounding (controllable BoW encoder + HashEncoder; null fallback, placebo
  decorrelation, determinism, cache idempotency) — all green. Full suite: **48 passed, 1 xfailed**.

## Phase 2 — doc-conditioned node encoder ✅
- `gloss/model/time_encoding.py`: `node_tau` (τ_u = log((seed−row)/T_ctx), float64, valid only for
  timed positive-gap rows; **scale-invariant** by construction) + `BochnerTime` (learnable Fourier
  features of τ).
- `gloss/model/column_encoder.py`: per-cell embeddings via pytorch-frame `StypeWiseFeatureEncoder`
  (reused, not reinvented) → **per-column FiLM** `x_c=γ(d_c)⊙W_v Enc(v_c)+β(d_c)` → **doc-keyed
  attention pool** over columns → `h_u = pool + E_type(t) + W_t φ(τ_u)`. Ungrounded columns / null
  regime fall back to a learned `d_null`.
- **Tests:** test_time_encoding (formula, scale-invariance, Bochner) hermetic; test_column_encoder
  (guarded rel-f1): finite `[B,N,d_model]`, pad=0, and **FiLM responds to full↔null regime**.
- DoD met. Full suite: **53 passed, 1 xfailed**.

## Phase 3 — the core operator (doc-generated geometry) ✅
- `gloss/model/bias_generator.py` (**CORE**): `g_θ` maps `ctx(p)=[E_metapath(p); doc(p)]` → per-head
  `(a,μ,σ,b)` (σ=softplus+floor). `compile()` runs once per DB over the metapath set → `GeometryTable`
  (recomputed each forward; gradients flow). `absolute_anchor` emits an extra `log T_ctx` coefficient.
- `gloss/model/attention.py`: `B_h(i,j)=b + temporal_valid·a·exp(−(τ−μ)²/2σ²)`; additive bias via
  `F.scaled_dot_product_attention` (SDPA mem-efficient; flash-attn can't take per-pair bias). Diagonal
  kept finite so padded rows don't NaN.
- `gloss/model/halos.py`: `ColumnEncoder` → compile geometry → `HALOSLayer` (pre-norm attn + FFN) stack →
  `EntityHead` seed readout. `build_doc_per_metapath` pools FK-role docs into the geometry context.
- `gloss/model/heads.py`, `gloss/eval/geometry_report.py` (+ `scripts/run_geometry_report.py`): the
  readable per-FK-role kernel exhibit (renders untrained).
- **Invariance fix:** absolute timestamp *cell features* are dropped (time enters only via τ); and
  `T_ctx` now folds in node recencies `(seed−row)` so it always scales with the clock (the old 1.0
  fallback made the node-time term non-invariant). Result: **logits bit-identical under t→c·t (1e-5)**.
- **Tests:** test_bias_generator, test_attention, test_fk_role (dual-FK fixture: distinct geometry +
  role-swap changes preds), test_scale_equivariance (full rel-f1 model invariant; absolute_anchor breaks
  it). DoD: forward runs, geometry report renders. **65 passed, 1 xfailed.**

## Phase 4 — training loop + LightGBM floor ✅ (preliminary numbers)
- `gloss/train/{datamodule,loop,finetune,losses}.py`: Lightning `DataModule` over the disjoint temporal
  loaders + `HALOSLitModule` (BCEWithLogits; raw batch → `GlossBatch` in `transfer_batch_to_device`);
  `make_loader` now attaches labels via `AttachTargetTransform` → `GlossBatch.target/has_target`.
- `gloss/eval/metrics.py`: val AP/AUROC/log_loss via **sklearn** (relbench's `log_loss` is buggy on this
  numpy). `gloss/eval/baselines.py`: LightGBM-on-aggregated-raw-features floor.
- `scripts/run_finetune.py --train` (+ `--baseline`, `--regime`, `--encoder`).
- **Tests:** `test_train.py` — overfits a single batch to <0.1 loss (GPU), val metrics finite. Suite green.
- **DoD numbers (PRELIMINARY — undertrained):** the flash-attn build saturated the 8 CPUs (the neighbor
  sampler is CPU-bound), forcing a tiny config (d_model=128, n_layers=4, 2 epochs × 20 train batches).
  - HALOS-full val: **AP 0.813, AUROC 0.556** (near chance — far from converged).
  - LightGBM floor val: **AP 0.929, AUROC 0.786** (n_val=566, 31 feats).
  - ⇒ The loop works end-to-end and produces val metrics, but HALOS needs a **proper run on clean CPU**
    (post-build) before it's competitive — and **before GATE 1 (H1) is meaningful** (a near-chance model
    can't resolve a 1–3% doc effect). H1/H2 machinery (`scripts/run_h1_gate.py`, `geometry_mode` toggle)
    is built and unit-tested; run it once the build frees the CPU.

## Phase 5 — GATE 1 (H1 + H2): machinery built; result is a SMOKE run, NOT a verdict
- `scripts/run_h1_gate.py` (graph + 3 groundings built ONCE, reused across configs via
  `finetune.train_prebuilt`); `model.geometry_mode ∈ {generated, free_learned}` for H2.
- **FlexAttention** backend added (`attn_impl={sdpa,flex}`) + GPU parity test — the "flash for HALOS".
- **Smoke gate (1 seed, 1 epoch, 12 train batches, d_model=64 — UNDERTRAINED, near chance):**
  | H1 regime | val AP | val AUROC |   | H2 mode | val AP | val AUROC |
  |---|---|---|---|---|---|---|
  | full | 0.788 | 0.477 |   | generated | 0.788 | 0.477 |
  | shuffled_spans | 0.788 | 0.479 |   | free_learned | 0.746 | 0.429 |
  | null | 0.786 | 0.473 |   | | | |
- **NOT a GO/NO-GO.** All AUROCs ≈ 0.5 (model barely trained), so H1 cannot resolve any doc effect and
  H2 is noise. The blocker is **compute in this interactive session**: the PyG disjoint neighbor sampler
  is CPU-bound (~2 min per short run), and the flash-attn build was saturating the 8 CPUs for ~1 h.
- **To get a real gate:** run on **SLURM (H100 + more CPUs)** with proper training (≥5 epochs to
  convergence, full batches, ≥5 seeds). The code is ready (`run_h1_gate.py`); only compute is missing.
  Reminder: rel-f1 is well-named, so even a converged H1 may be small — the existence proof is Phase-6 synthetic.

## flash-attn build (Stage A)
- Built + installed `flash_attn 2.8.3` from source on the A40 node. **BUT targeted sm_90 (H100) only** —
  flash-attn maps Ampere→sm_80 and my `ARCHS="86;90"` dropped it, so the kernel errors on this A40
  ("no kernel image"). Fixed both build scripts to `ARCHS="80;90"` for a future rebuild. Off the critical
  path (HALOS uses SDPA/Flex; Qwen cache already built), so not rebuilt now. `test_env` checks import +
  arch-guards the kernel.

## GATE 1 — proper run submitted to SLURM (multi-GPU array)
- `scripts/gate_run.py` (one config per process: `--index`, `--list`, `--aggregate`) +
  `scripts/run_gate.sh` (`#SBATCH --array=0-19%4`, h100:1 each, 12 CPUs, num_workers=8, 10 epochs,
  d_model=256/n_layers=8, full batches). 20 configs = 5 seeds × {full,shuffled,null}-generated + full-free_learned.
- Submitted: **job array 28963902_[0-19%4]** (4 H100s in parallel). Independent runs → array, not DDP.
- Harvest when done: `.venv/bin/python scripts/gate_run.py --aggregate` → H1/H2 tables (mean±std over 5 seeds);
  paste into this file with the Go/No-go decision + the well-named-rel-f1 caveat.

## GATE 1 — RESULT (job array 28963902; 19/20 configs, proper training to convergence)
HALOS trained to **AUROC ≈ 0.795** ≈ the LightGBM floor (0.786) — the model is competitive, not broken.

**H1 — doc regime (generated geometry, 5 seeds):**
| regime | val AP (mean±std) | val AUROC (mean±std) |
|---|---|---|
| full | 0.9212 ± 0.0044 | **0.7949 ± 0.0027** |
| shuffled_spans | 0.9228 ± 0.0033 | 0.7915 ± 0.0019 |
| null | 0.9187 ± 0.0054 | 0.7913 ± 0.0017 |

**H2 — geometry mode (regime=full):**
| mode | val AP | val AUROC | n |
|---|---|---|---|
| generated | 0.9212 ± 0.0044 | 0.7949 ± 0.0027 | 5 |
| free_learned | 0.9200 ± 0.0036 | 0.7933 ± 0.0024 | 5 (backfill 28964162 done) |

**Verdict: no measurable documentation effect on rel-f1.** full ≈ shuffled ≈ null (AUROC Δ ≤ 0.4%, within
~1–1.5 σ; AP ordering inconsistent — full < shuffled). H2: generated ≈ free_learned in-DB. This is the
**expected outcome for a well-named schema** (spec calibration ~1–3%; here <0.5%) and is consistent with
risk #1 ("docs redundant on real DBs"): the measurement *is* the result, not a failure.

**Decision: GO (to Phase 6), with the conclusion that rel-f1 is the wrong place to see the effect.** The
gate confirms the pipeline trains competitively and grounding isn't feeding noise (full not worse than
null). The mechanism's leverage must be shown where docs are load-bearing: the **Phase-6 synthetic
existence proof** (docs the ONLY disambiguator) + poorly-named/coded schemas and **OOD transfer (H2-OOD,
Phase 7)**. Do NOT conclude the method fails from one well-named DB.

Note: config 7 (seed 1, free_learned) crashed on a flaky pytorch-frame `MultiEmbeddingTensor` index
assertion under multi-worker slicing (other free_learned seeds fine) — resubmitted.

## Extension — depth-matched text tower + doc cross-attention (user-requested architecture)
A second, **trainable** text encoder over the FROZEN-cached Qwen span memory, with cross-attention into
the graph transformer. Both paths **default OFF** (FiLM/pooled-d_e path unchanged); config
`model.doc_cross_attn.{feature,geometry}` or `run_finetune.py --doc-feature/--doc-geometry`.
- `gloss/model/text_tower.py`: `TextTower` (L self-attn blocks, **L = graph n_layers**, over projected
  span memory `[M,d_text]→[M,d_model]`, per-layer states T¹..Tᴸ) + `DocCrossAttention` (queries attend a
  doc memory; empty memory → exactly zero, so the `null` regime contributes nothing).
- **Feature-side** (`HALOSLayer`): graph nodes cross-attend Tˡ per layer (self-attn → cross-attn → FFN).
- **Geometry-side** (`bias_generator`): the metapath query cross-attends Tᴸ to build `ctx(p)` for g_θ —
  richer docs *for generating geometry* (on-thesis), vs RELATE-like feature-side.
- `grounding.py`: `GroundingResult.span_emb`/`span_memory()` expose the raw span set (empty under `null`).
- **Verified:** two-tower forward finite; full vs null differ (cross-attn responds to docs); **exact
  scale-equivariance still holds with both paths ON** (docs are time-independent) — 0.0 logit diff under
  t→c·t. Tests: `test_text_tower.py` (+ existing suite). **75 passed, 1 skipped.**
- **Caveats:** feature-side ≈ RELATE — keep doc-generated *geometry* as the headline (geometry-side is the
  novel use). The cross-attn *placebo* (shuffled_spans for a SET-consuming tower) is not yet a true
  control — meaningful contrast for now is full vs null. Evaluate on Phase-6 synthetic / poorly-named DBs,
  NOT rel-f1 (GATE 1 showed docs are redundant there regardless of fusion).

## Multi-task TEST-set sanity check (vs GelGT table) — HALOS is competitive, NOT SOTA
5 seeds each, RelBench test split via `task.evaluate` (leaderboard-comparable):

| dataset / task | HALOS test roc_auc | GelGT | Δ |
|---|---|---|---|
| rel-f1 / driver-dnf (full) | 0.8247 ± 0.0052 | 0.7608 | +0.064 (win) |
| rel-f1 / driver-top3 (full) | 0.6791 ± 0.0238 | 0.8408 | −0.162 (loss) |
| rel-trial / study-outcome (null) | 0.6947 ± 0.0071 | 0.7254 | −0.030 (loss) |

**Verdict: the driver-dnf win does NOT generalize.** HALOS wins one, loses two → a competitive but
task-dependent architecture, not a SOTA-beater. (Consistent with the spec's explicit non-goal of beating
SOTA accuracy; the contribution is mechanism + invariances + transfer, not leaderboard wins.)
- driver-dnf rewards recent-failure-history → fits HALOS's temporal-geometry attention; driver-top3 needs
  driver/car *quality* (cell content), where our HashEmbedder cell-text (noise) + weak id-column handling
  under-powers it.
- study-outcome run in NULL regime (no docs) on a doc-rich domain — real rel-trial docs might help.
- **Pipeline bug caught + fixed:** the first driver-top3 run gave 0.42 (below chance) because 5 concurrent
  array tasks raced downloading the task parquet and corrupted it. Lesson: pre-cache every task serially
  before a multi-seed array. Clean re-run -> 0.679.
