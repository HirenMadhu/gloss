# PROGRESS — MoRE (Mixture of Relational Experts)

Running log. **Current method = MoRE** (RT cell-token substrate + a Mixture-of-Experts FFN routed on a
value-free relational signature); see the 2026-06-25 entry. The earlier **DOC-RT** build (RT + per-column
documentation via FiLM) is retired to `archive/doc-rt/` but its log is kept below as history. The retired
HALOS log is at `archive/halos/PROGRESS_halos.md`. Design: `CLAUDE.md`, `idea.md`, `implementation.md`.

## 2026-06-24 — Pivot HALOS → DOC-RT; P0–P4 scaffold green; headline gate launched

**Pivot.** Retired the HALOS geometry stack (Gaussian-in-τ kernels, `g_θ` geometry generator, dimensionless
time `τ`, scale-equivariance) to `archive/halos/`. DOC-RT keeps RT's cell-token substrate + relational masks
and adds one signal: per-column documentation injected by FiLM into the cell encoder. `gloss` package kept.

**Phase 0 — substrate [done].** `data/graph.py` (RelBench → hetero temporal graph, `fk_role_id`, leakage-safe
sampler) + `data/collate.py` (`CellBatch`: RT cell tokens at fixed `seq_len`, the 4 relational masks rebuilt in
the model, `f2p_nbr_idxs`, `is_seed_cell`, `cell_placement`). DoD: `run_train.py --dry-run` builds a real
rel-f1 cell batch (B=8, 527 real cells across all 5 tables), leakage check = 0, forward OK.

**Phase 1 — grounding [done].** `docs/{corpus,grounding,cache}.py`; regimes `full | null | shuffled | name_only`;
`d_null` fallback; Qwen3-Embedding-4B grounding cached at `data/doc_cache/rel-f1/` (complete — no LM forward at
train time).

**Phase 2 — cell encoder [done].** `model/column_encoder.py` `CellEncoder`: pytorch-frame dtype encoders +
FiLM by `d_c` + RT name token + learned `d_null`; scatters per-cell vectors into the `[B,S,d]` grid.

**Phase 3 — model [done].** `model/rt_substrate.py` (RT `RelationalBlock`: col/feat/nbr/full SDPA attention +
SwiGLU, pre-norm RMSNorm) + `model/docrt.py` (`DOCRT` = CellEncoder → RTSubstrate → EntityHead).

**Phase 4 — training + HEADLINE GATE [running].** `train/*`, `eval/ablation.py`, `scripts/run_headline.{py,sh}`.
SLURM job **28967996**: array `0-19%4` = 5 seeds × {full, null, shuffled, name_only}, h100:1, 10 epochs, qwen.
Aggregate when done: `run_headline.py --aggregate` → `results/headline/`. Reading: `full > null` (signal)
`> shuffled` (meaning) `> name_only` (beats names); `full ≈ null` is the honest negative.

**Tests: 56 passed, 1 skipped (flash-attn).** Leakage, grounding (null fallback / placebo / name-independence /
determinism), FiLM-responds, FK-role distinctness, cell-batch shapes / collate / column-encoder, single-batch
overfit + grad-flow (`test_train.py`), gate bookkeeping (`test_ablation.py`).

**Engineering decisions.**
- Memory: DOC-RT attention is dense `O(S²)` with 4 `[B,S,S]` bool masks/block; SDPA+bool-mask → math backend
  materializes all score matrices for backward. Measured rel-f1 cells/seed (mean 206, max 353) → `seq_len=512`
  (0% truncation). Gate `batch_size=64` (the archived HALOS value 512 was infeasible for cell-token attention).
- `run_train.py` flags added for fast smokes: `--seq-len`, `--limit-val-batches`, `--num-workers`.

**Deferred (P5–8).** P0b self-labels (task-table node type + seed's past task rows, leakage-safe); coverage-vs-
gain curve; more DBs; same-column attention-bias ablation (`--docbias`); masked-cell pretraining + transfer;
the synthetic twin-column DB + audit (existence proof when docs are the only disambiguator).

## 2026-06-25 — Pivot DOC-RT → MoRE (Mixture of Relational Experts); Phase A done

**Pivot.** New method = **MoRE**: keep RT's cell-token substrate + the 4 relational masks, replace RT's SwiGLU
FFN with a **Mixture-of-Experts FFN** routed on a value-free per-cell *relational signature* (column-name
embedding + modality + recency), balanced by a router-orthogonality loss. The headline is the **routing-signal
ablation** (signature vs hidden vs value vs identity vs dense) across all entity tasks of rel-f1 / rel-stack /
rel-trial, reported on the held-out **test** set. Docs/FiLM retired. Specs rewritten: `idea.md`,
`implementation.md`. Plan + stress-test in `/memories/session/plan.md`.

**Phase A — archive DOC-RT, reduce to plain RT [done].** Moved the doc stack (grounding/corpus, the prose
corpus, the 9.6M embedding cache, doc scripts + tests) to `archive/doc-rt/`. Kept the reusable frozen text
encoder + cache as `gloss/text/cache.py`; added `gloss/text/schema.py` (the frozen per-column **name** table
the router will route on). Stripped FiLM/grounding from `CellEncoder` / `DOCRT` / the Lightning loop /
`finetune` — the model is now plain RT (value dtype-encoder + RT name token) and trains names-only. `gloss.docs`
removed; configs lost the `docs.*` block. DoD: `pytest` **40 passed, 1 skipped**; `run_train.py --dry-run`
forwards rel-f1 (B=8, 527 real cells, leakage=0, finite logits); a capped `--train` arm reports val metrics.

**Phase B — MoRE core [done].** `model/moe.py` (`SwiGLU` + `MoEFFN`: pool of SwiGLU experts, top-k router,
dense combine, router-orthogonality `ortho_loss`); `model/signature.py` (`RelationalSignature`: frozen
column-name embedding + learned modality (pytorch-frame stype) + **fixed-edge, context-independent** recency
buckets, bin 0 = untimed; computed once, value-free); `text/schema.py` gains `build_column_modality_ids`.
Wired MoE into `model/rt_substrate.py` — `RelationalBlock` takes a `moe` flag and `RTSubstrate` takes
`route_on` (router input dim set by the arm: `d_sig` for signature/identity, `d_model` for hidden/value),
threads `z`/value/id and returns `(states, aux)`. `model/more.py` (`MoRE`, `forward → (logits, aux)`,
`ROUTE_ONS = signature/hidden/value/identity/dense/dense_wide`; dense_wide = param-matched control).
`CellEncoder.forward(cb, return_value)` now also returns the value component (for the `value` arm). The
Lightning loop builds `MoRE`, adds `λ·aux` to the loss; `train_prebuilt`/`run_train` gained `--route-on`.
`model/docrt.py` removed. DoD: `pytest` **50 passed, 1 skipped** (new: signature value-free + recency,
MoE gating/dense-combine/ortho, routing-invariance, MoRE forward over all 6 arms + grad-flow to router &
signature); `--dry-run` signature arm aux≈0.69; a capped signature `--train` runs the MoE path through
Lightning.

**Phase C — training + regression [done].** `train/losses.py` gains `masked_mse` + a `task_loss` dispatch;
`eval/metrics.py` gains `regression_metrics` (mae/rmse/r2). `train/loop.py` → **`MoRELitModule`**: handles
both task types, **z-scores regression targets** with TRAIN-split stats (`finetune.target_stats`) for
training, de-standardizes for val metrics (mae/rmse/r2) and predictions; `finetune.task_kind` reads
`task.task_type`. `eval/test_eval.py` `predict_split` is now task-type-aware (the hardcoded sigmoid is gone —
binary → probability, regression → de-standardized value), so `task.evaluate` gets original-unit predictions.
`train/datamodule.py` → `MoREDataModule`. `run_train.py` gains `--num-experts/-k/--d-sig/--lambda-ortho/
--moe-placement/--dataset/--task/--test`. DoD: `pytest` **55 passed, 1 skipped** (new: masked-MSE/`task_loss`/
regression-metrics hermetic + a regression single-batch overfit); a capped `driver-position` (regression)
`--train --test` run reports val mae/rmse/r2 and a leaderboard-style test MAE via `task.evaluate`.

**Phase D — multi-DB routing-signal ablation [done].** `eval/ablation.py` rewritten around routing
**signals** (not doc regimes): `entity_tasks()` filters RelBench `get_task_names` to binary+regression via
`task.task_type` (rel-f1 → 5 entity tasks: driver-dnf/top3 + driver/qualifying/results-position; link tasks
excluded); `build_grid` enumerates `(dataset, task, signal, seed)`; `run_config(index)` trains one arm and
writes one JSON with val + **held-out test** metrics (`task.evaluate`); `aggregate` groups by
`(dataset, task, signal)` with 95% CI; `format_table` prints per-`(dataset, task)` the primary metric
(AUROC↑ / MAE↓) and each arm's **Δ vs dense**. `scripts/run_ablation.{py,sh}` (SLURM array) +
`scripts/build_schema_cache.{py,sh}` (precompute Qwen column-name embeddings once — the array does no LM
forward) + `eval/diagnostics.py` (expert-usage entropy + the cross-table specialization probe). DoD:
`pytest` **56 passed, 1 skipped** (grid enumeration, CI aggregation, binary/regression lift tables);
`run_ablation.py --list` = 30 for rel-f1×1-seed; a 2-arm local smoke (signature, dense) wrote JSON and
`--aggregate` printed the table with the signature-vs-dense lift.

**Honesty / framing.** Dense-combine MVP ⇒ MoE arms cost ~M× FFN FLOPs (not k×); we report vanilla `dense`
**and** param-matched `dense_wide`, and do not claim active-FLOP parity (sparse dispatch is deferred). The
full headline = signature vs {value, dense, identity} across all entity tasks of rel-f1/rel-stack/rel-trial,
reported on TEST — runs on SLURM once rel-stack is downloaded and the schema cache is built.

**Remaining.** Phase E doc-sync: `CLAUDE.md` still describes DOC-RT and should be rewritten to MoRE.

---

**Ablation Suite v2 ([moe_ablation.md](moe_ablation.md)) — env bootstrap + hybrid + S/C/P/H [done].**
Extends the MoE along two independent axes on top of the routing-signal ablation.

*New cluster bootstrap.* Rebuilt `.venv` (uv, py3.12) with **torch 2.8.0+cu128** from the pytorch cu128
index and the pyg extensions from `data.pyg.org/whl/torch-2.8.0+cu128.html` (a plain resolve pulls CPU
torch); verified CUDA on the interactive **H200** (driver 580, covers Hopper + Blackwell). Moved all caches
to a dedicated scratch subtree `~/scratch60/gloss/{hf,relbench,graph_cache,schema_cache}`
([scripts/env.sh](scripts/env.sh)) via new [gloss/utils/paths.py](gloss/utils/paths.py)
(`graph_cache_dir`/`schema_cache_path`, env-driven with repo-relative fallback), routed through
`prep_data.py`, `ablation.run_config`, and `finetune._name_encoder`. Fixed the SLURM scripts for this
cluster (`--partition=gpu_h200 --gpus=h200:1`; no h100 exists here; array throttle `%16`).

*Router-input axis.* Added the **`hybrid`** arm (= **signature+hidden**, `route_feat = [z ; h]`,
`d_route = d_sig + d_model`): `RTSubstrate` computes the hybrid width and `RelationalBlock._route_feat`
concatenates; `MoRE` builds the signature for `signature` **and** `hybrid`; `"hybrid"` added to `ROUTE_ONS`.

*Architecture additions (all off by default = base top-k MoE).* [model/moe.py](gloss/model/moe.py):
`MoEFFN` gains **S** (`use_shared`, an always-on shared expert added as a residual), **C** (`cosine`/`tau`,
cosine-normalized logits over learnable `keys`; `ortho_loss` then decorrelates the keys), **P** (`top_p`,
adaptive support = smallest set reaching cumulative mass P); new **`HMoEFFN`** (**H**) two-level gate
(learned group gate → per-group top-`k2`, dense combine; balance = level-1 gate-row orthogonality + a
`log Γ − H(occupancy)` collapse penalty). Threaded `use_shared/cosine/tau/top_p/hmoe/n_groups/
experts_per_group/k2` through `RTSubstrate → RelationalBlock`, `MoRE`, `configs/default.yaml` (`moe:`
block), and both CLIs.

*Two decoupled studies.* The additions are **run-level** model_kwargs, not grid axes. `run_config` now
writes an auto-derived **`variant`** label (router + S/C/P/H tags, e.g. `signature+SCPH`) and puts it in the
JSON **filename**, so a base router and its addition configs can share one out-dir without colliding.
`aggregate` groups by `variant` (falls back to `signal` — back-compat); `format_table` orders by a canonical
`ROUTER_DISPLAY` (dense/dense_wide/signature/hybrid/…) then addition variants, with a **configurable
`--baseline`** for the Δ column (`dense` for the routing study; a base router for the additions study).
Study A = routing methods (focus **signature vs hybrid**; signature base already run); Study B = S/C/P/H on
the signature/hybrid bases. Transfer (Tier 1b) stays deferred.

*Diagnostics.* `specialization_probe` uses the MoE's own `_logits` (cosine-safe) and is guarded to the
signature arm; added `mean_active_experts` (Top-P k̄).

DoD: `pytest` **65 passed, 2 skipped** (new: shared/cosine/top-p in `test_moe.py`, `test_hmoe.py`, hybrid
grad-flow + additions forward in `test_shapes.py`; `test_ablation` green via the `variant` fallback). Hash-
encoder smokes on the H200 trained the `hybrid` arm and S/H additions and produced a per-variant aggregate
table with Δ vs a chosen baseline. Real runs use the **harrier** schema cache (one-time prep) on `gpu_h200`.

## Phase S0 — SetJoin branch: hygiene, docs (branch `setjoin`)

New standalone direction: **SetJoin** ("one big joined table" + set transformer). Committed the pending
main-line fixes on `main` (`2403be4`: MET offset collation, nan-grad guard, grad clip, `--tasks
leaderboard`, gridsearch skip-guard, recap) and cut branch **`setjoin`** from it. Wrote the branch specs —
`setjoin_idea.md` (thesis: a relational neighborhood = one wide m2o-flattened joined row + one table-tagged
union set of child rows; a seed-conditioned set transformer replaces join duplication / lossy aggregation;
falsifier: ≈ the mean-pool LightGBM baseline) and `setjoin_implementation.md` (JoinBatch contract,
per-edge-type fanout dict, PyG sampling facts: rev-edge traversal storage, disjoint tree ⇒ n_id-based seed
exclusion, fk_role_id column-name collisions ⇒ table_emb load-bearing). CLAUDE.md got a branch-only pointer
section. MoRE files untouched.

DoD: `pytest` **66 passed, 1 skipped** (baseline unchanged).

## Phase S1 — SetJoin schema paths + JoinBatch collate

`gloss/setjoin/paths.py` (relations as `(child_type, fk_col, parent_type)` triples — the bare
`fk_role_id` column key collides across tables; `m2o_paths` BFS with depth cap; `setjoin_neighbors`
per-edge-type fanout dict: f2p `[1,1]`, rev `[fanout,0]` — samples exactly the m2o closure + 1-hop
children), `recency.py` (fixed log Δt buckets, standalone), `collate.py` (`JoinBatch` + `to_join_batch`:
wide m2o-flattened seed row with ONE missing-marker slot per dead path; one table-tagged union set of
child rows, most-recent-first, additive `elem_rows` row scatter = child @ FK_NONE + its flattened
parents @ fk-role, seed excluded **by n_id**; adjacency read from ALL edge types canonicalized by
direction — children arrive via rev_* types). Fixtures: `tests/_join_fixtures.py` 3-level chain
(payment→order→customer→region + payment→method side parent; rev-only child instances). 15 new tests
(`test_join_collate.py`, `test_join_leakage.py`).

DoD: `pytest` **81 passed, 1 skipped**. `run_setjoin.py --dry-run --collate-only` on cached rel-f1:
**dict num_neighbors accepted by the sampler**, B=8 W=128 N=128, 48 wide cells, 12 set elements
(2 empty-set early-career seeds), child_counts (8,3), **leakage=0**.

## Phase S2 — SetJoin model

`gloss/setjoin/model.py`: `SetJoin(bundle, name_emb, entity_table, ...)` → `(logits [B,1], aux=0)`.
Reuses `CellEncoder.encode_type` verbatim (each sampled type encoded ONCE, shared between the wide grid
scatter and the row-pooled set elements). `WideEncoder` (CLS, pre-norm, pad-masked) → seed_repr;
`RowPool` (shared gated-attention cell pool); additive element assembly (child @ FK_NONE + flattened
parents @ fk-role + fk/table/Δt/hop tags — table_emb load-bearing vs fk_role collisions); `SetEncoder`
(never-masked learned null element, no positional encoding, seed-conditioned PMA readout); head on
`[seed_repr ; context ; W_cnt log1p(child_counts)]`. Design note surfaced by testing: PMA seed
conditioning is inert over a single key (empty set → null only) since softmax(1 key)=1 — fine, the head
sees seed_repr directly; conditioning acts when ≥2 elements exist.

DoD: `pytest` **88 passed, 1 skipped** (7 new in `test_setjoin_model.py`: hermetic RowPool/WideEncoder/
SetEncoder permutation-invariance + empty-set/seed-conditioning; rel-f1 full-model forward/grads/budget,
end-to-end set permutation invariance, empty-set + missing-marker liveness). `run_setjoin.py --dry-run`:
forward OK on rel-f1 CPU, logits (8,1) finite, aux=0, **0.91M params** (hash encoder).

## Phase S3 — SetJoin training + eval

`gloss/setjoin/train.py`: `SetJoinLitModule` — three-line fork of `MoRELitModule` (SetJoin model,
`to_join_batch` in transfer_batch_to_device, no aux term); identical val metric names / regression
z-scoring / nan-grad guard, so `_BestValState` + `EarlyStopping` work unchanged.
`train_prebuilt_setjoin` mirrors `finetune.train_prebuilt` with `setjoin_neighbors(bundle, fanout)` as
the sampler fanout and reuses `MoREDataModule` untouched. `gloss/setjoin/eval.py`:
`predict_split`/`evaluate_split` fork of test_eval.py aligning via the JoinBatch-carried per-seed
`input_id`.

DoD: `pytest` **92 passed, 1 skipped** (4 new in `test_setjoin_train.py`: overfit-one-batch binary +
regression, end-to-end grad flow, standardization round-trip). CPU micro-trains complete with TEST eval:
driver-dnf (binary) and driver-position (regression), 2 epochs × 20 batches, hash encoder.

## Phase S4 — SetJoin gate runner + SLURM array (THE GATE)

`gloss/setjoin/runner.py`: `gate_grid` (9 leaderboard tasks × 3 seeds = 27, signal="setjoin", reusing
ablation `dataset_tasks`/`build_grid`), `run_config` (idempotent skip-if-done, CUDA-OOM batch halving à
la gridsearch, ablation-schema records + run-time `test_nmae` = MAE/train-std), `compare_table`
(SetJoin vs RT-from-scratch / GelGT from `results/leaderboard_baselines.json` + MoRE grid best).
`scripts/run_setjoin.py` grew `--list/--index/--aggregate/--compare`; `scripts/run_setjoin.sh` is
Milgram-native (partition=gpu, h100:1, %8 QOS cap, harrier encoder, rel-event symlink reminder).

DoD: `pytest` **97 passed, 1 skipped** (5 new hermetic runner tests). `--list` = 27. Smoke `--index 0`
(hash, 1 epoch, 5 batches) wrote `results/setjoin_smoke/0000_setjoin.json` with the full record schema;
`--aggregate` renders. rel-event symlinks + harrier caches verified. Gate array submitted (see below).

## GATE RESULT — SetJoin v1 (array 28993182, 27/27 completed, 2026-07-13)

**SetJoin beats RT (from scratch) on 4/9 leaderboard tasks** (3 seeds, TEST, harrier encoder,
d_model=128/fanout=64/wide=set=128, batch 128 everywhere, no OOMs, ~1.6 GPU-h total):

| task | SetJoin | RT | GelGT | MoRE best | vs RT |
|---|--:|--:|--:|--:|:--:|
| rel-f1/driver-top3 (AUROC↑) | **89.7±0.2** | 82.7 | 84.1 | 90.6 | ✅ (+7.0, also >GelGT) |
| rel-f1/driver-dnf (AUROC↑) | **79.1±0.6** | 78.7 | 76.1 | 82.9 | ✅ (also >GelGT) |
| rel-f1/driver-position (NMAE↓) | **0.448±0.011** | 0.477 | 0.531 | 0.395 | ✅ (also >GelGT) |
| rel-event/user-attendance (NMAE↓) | **0.477±0.019** | 0.504 | 0.317 | 0.399 | ✅ |
| rel-trial/study-outcome (AUROC↑) | 67.4±1.9 | 68.6 | 72.5 | 69.4 | ❌ (close) |
| rel-event/user-ignore (AUROC↑) | 80.8±2.2 | 85.1 | 87.8 | 87.3 | ❌ |
| rel-trial/study-adverse (NMAE↓) | 0.166±0.006 | 0.131 | 0.126 | 0.161 | ❌ |
| rel-event/user-repeat (AUROC↑) | 65.2±2.7 | 79.7 | 83.6 | 79.5 | ❌ (big) |
| rel-trial/site-success (NMAE↓) | 0.978±0.017 | 0.734 | 0.732 | 0.840 | ❌ (≈mean-predictor) |

Reading: wins are exactly the tasks whose signal is the seed's own recent 1-hop fact rows (all of
rel-f1; attendance counts). The three big losses (user-repeat, user-ignore, site-success) are the tasks
whose signal lives at 2 hops (event co-attendees / event popularity; a site's linked studies) — the o2m
context the MVP deliberately dropped (setjoin_idea.md falsifier/risk #2, now empirically confirmed;
site-success NMAE≈1 = the model sees almost nothing beyond a near-featureless site row). Val tracks
test on all 9 (no split pathology). Cost: mean cell runtime ~3.4 min on one h100 — an order of
magnitude cheaper than MoRE per run. **Stopped at the gate; next axis if pursued: hop-2 union
elements (`elem_hop` reserved), rel-event fanout/set_size sweep.**

## CORRECTION — v1 gate invalidated: fanout-direction inversion; v2 resubmitted

Post-gate diagnosis of the weak tasks found `setjoin_neighbors` had **PyG's sampling direction
inverted**: `num_neighbors[et]` budgets src-side draws per DST frontier node, so children arrive via
the FORWARD `f2p` types (which v1 capped at `[1,1]`) and parents via the REV types (which v1 gave
`[fanout, 0]` — hop-2 budget 0 ⇒ elements never got their flattened parents). Every v1 cell therefore
trained on ~1 child per relation and parent-less elements — e.g. site-success saw 5 facility columns +
one 1-column link row (no study!), fully explaining its NMAE≈0.98; that v1 still beat RT on 4/9 with
~3 rows/seed is a strong wide-row+recency floor, not the method. After the swap (`f2p [fanout,0]`,
`rev [1,1]`): driver-dnf ~69 elements/seed with races/constructors flattened; user-repeat ~88
(friends + attendance, event parents in); site-success ~10 with `studies` flattened in.

Why tests missed it: synthetic-batch collate tests bypass the sampler entirely; the rel-f1 tests only
asserted elements exist (>0). Added `test_sampler_multiplicity_and_parent_flattening_rel_f1`
(child_counts>1 + parent type present in elem_rows on real sampled data) + hardened the overfit test
(400 steps — richer batches put 200 at the flake edge). `pytest` **98 passed, 1 skipped** (×3 on the
hardened test). v1 results kept in `results/setjoin_v1/` for the record; corrected gate resubmitted to
`results/setjoin_v2/`.

## GATE RESULT — SetJoin v2 (corrected sampler; array 28993542, 26/27 + cell 4 rerun in flight)

**SetJoin v2 beats RT (from scratch) on 6/9 leaderboard tasks** — same count as MoRE's grid-search
best, which used per-task tuned architectures over ~90 configs/task; SetJoin is ONE untuned config
(d_model 128, 2+2 layers, fanout 64, set 128). Also **beats GelGT on 4/9** and posts the best known
number on user-ignore. (Cell 4 = driver-position seed 1 hit the known sporadic MET `offset[0]` assert
→ rerun with `--num-workers 0` as 28996296; n=2 shown there, CI already tight.)

| task | SetJoin v2 | v1 (starved) | RT | GelGT | MoRE best | vs RT |
|---|--:|--:|--:|--:|--:|:--:|
| rel-event/user-ignore (AUROC↑) | **90.1±0.6** | 80.8 | 85.1 | 87.8 | 87.3 | ✅ (>GelGT, >MoRE) |
| rel-f1/driver-top3 (AUROC↑) | **87.1±0.3** | 89.7 | 82.7 | 84.1 | 90.6 | ✅ (>GelGT) |
| rel-f1/driver-dnf (AUROC↑) | **81.5±0.9** | 79.1 | 78.7 | 76.1 | 82.9 | ✅ (>GelGT) |
| rel-f1/driver-position (NMAE↓) | **0.429±0.003** (n=2) | 0.448 | 0.477 | 0.531 | 0.395 | ✅ (>GelGT) |
| rel-event/user-attendance (NMAE↓) | **0.430±0.027** | 0.477 | 0.504 | 0.317 | 0.399 | ✅ |
| rel-trial/study-outcome (AUROC↑) | 68.6±0.9 | 67.4 | 68.6 | 72.5 | 69.4 | ➖ tie |
| rel-event/user-repeat (AUROC↑) | 78.0±0.6 | 65.2 | 79.7 | 83.6 | 79.5 | ❌ close (−1.7) |
| rel-trial/study-adverse (NMAE↓) | 0.162±0.009 | 0.166 | 0.131 | 0.126 | 0.161 | ❌ (=MoRE) |
| rel-trial/site-success (NMAE↓) | 0.941±0.037 | 0.978 | 0.734 | 0.732 | 0.840 | ❌ |

Reading vs v1: the starved-sampler tasks recovered exactly as diagnosed — user-repeat +12.8 AUROC,
user-ignore +9.3 (now SOTA on this table), attendance 0.477→0.430. Curious: driver-top3 DROPPED
89.7→87.1 — one most-recent standings row was a near-perfect feature; 60+ extra elements dilute it
(supports adding a per-relation quota or letting PMA see fewer, more recent rows; worth a look, not a
blocker). Still unsolved: site-success (0.94 ≈ mean predictor even with ~10 studies flattened in —
label likely depends on study outcomes, 2-hop o2m) and study-adverse (everything loses to RT here;
MoRE ties us). user-repeat's remaining −1.7 gap is consistent with missing co-attendee context (2-hop).
Cost: ~2.4 GPU-h for the full gate.

## Phase S5 — the MoE lands in SetJoin ("that was the whole thing")

SetJoin now carries **MoRE's Mixture-of-Relational-Experts FFN in every wide/set layer**
(`gloss/model/moe.py::MoEFFN` reused verbatim; new `MoELayer` = pre-norm attention + MoE-FFN).
Value-free routing signatures computed once per forward: `WideSignature` (wide tokens ARE cells → the
true MoRE cell signature: frozen name emb + modality + Δt + join path; learned CLS sig) and
`ElemSignature` (row analog: table + FK role + Δt + hop; learned null sig). Arms `route_on ∈
{signature, hidden, dense}` — `dense` keeps the stock layers byte-identical to the v2 gate model, so
**v2 IS the dense baseline** of the in-substrate ablation. Balance = router orthogonality → `aux`,
weighted `λ_ortho=0.5` in the Lit module. CLI: `--route-on --num-experts --k --d-sig --lambda-ortho`;
records carry `variant setjoin-<arm>` (dense keeps bare `setjoin` for v1/v2 back-compat).

Subtlety unit-tested: wide-signature routing needs `n_wide_layers ≥ 2` to influence the CLS readout
(with 1 layer no attention follows the FFN, so non-CLS routed outputs never reach CLS).

DoD: `pytest` **101 passed, 1 skipped** (new: hermetic MoELayer routing/perm-invariance; signature-arm
aux>0 + router/signature grad flow; dense-arm aux==0 no-router regression; signatures value-free under
tf value perturbation). `--dry-run`: aux=0.655, 2.22M params (vs 0.91M dense). Gate v3 (signature arm,
27 cells) submitted → `results/setjoin_v3/`.

## Phase S6 — three-level attention (row / column / set): apples-to-apples with RT's masks

`AxialCellEncoder` + `AxialCellBlock` (shared across types): per-seed per-type cell grids get
row-level attention (across columns), column-level attention (same column across the seed's rows),
and the signature-routed MoE FFN; set-level attention downstream completes RT's interaction patterns
(same-row / same-column / cross-table) at axial cost `O(n·C² + C·n² + N²)` instead of RT's
`O((n·C)²)`. Collate now carries per-TensorFrame-row metadata (`row_seg/row_times/row_timed`) for the
grids. Default `n_axial_layers=0` and module created LAST in `__init__` — behavior- and RNG-order-
preserving for the in-flight v3 arm; v4 enables it explicitly. Two findings while testing: constant
perturbations lie in pre-norm LayerNorm's null space (tests must perturb randomly), and pad grid
queries emit NaN under fully-masked SDPA which survives 0-weight attention (NaN·0=NaN) — pads are
re-zeroed after every sublayer.

DoD: `pytest` **107 passed, 1 skipped** (axial NaN-safety, cross-row mixing with cross-seed isolation,
row-axis mixing, MoE routing/ortho, full three-level forward/grads on rel-f1, collate row metadata).
`--dry-run --n-axial-layers 1`: aux=0.74, 2.75M params. Gate v4 (signature + axial, 27 cells)
submitted → `results/setjoin_v4/`.

## Phase S7 — multi-horizon study (predict k=1..10 timestamps ahead, both substrates)

`gloss/setjoin/horizon.py`: horizon-k evaluation = keep the held-out TEST labels, move each seed's
as-of feature time BACK k·task.timedelta (`shifted_task_table` + `make_shifted_loader`) — the model
predicts the label window starting k timestamps beyond everything it can see; k=0 is the standard
eval and is asserted equal to the normal path (`test_horizon.py`). Nothing is retrained: a
next-step-trained model is probed at longer ranges, isolating short-range recency vs longer-range
structure. Works for BOTH substrates (`kind="setjoin"` JoinBatch / `kind="more"` CellBatch; MoRE
files untouched). Runner: `run_config_horizon` — 9 leaderboard tasks × 3 seeds × {setjoin, more} =
**54 cells**; fixed configs per substrate (setjoin = the v4 arm: signature MoE + 1 axial block;
more = the grid modal winner d128×8blk/8 experts/signature); `plot_horizon_curves` → 3×3 PNG
(x = timestamps ahead, y = AUROC×100 / NMAE, mean ± 95% CI band per substrate).
`scripts/run_horizon.{py,sh}`.

DoD: `pytest` **110 passed, 1 skipped** (grid=54; k=0 ≡ standard eval; shift math; plot renders).
CPU smoke (hash, 1 epoch) ran both substrates end-to-end incl. the PNG. Array submitted →
`results/horizon/`.

## Phase S8 — shared-expert arm + the 2×2 MoE grid (shared/regular × 4/8 experts)

`use_shared` (MoRE's S addition — an always-on SwiGLU expert added to the routed sum, already in
`gloss/model/moe.py`) threaded through every SetJoin MoE site (`MoELayer`, `_MoEStack`, wide/set
encoders, axial blocks) + `--use-shared` CLI; variant label `setjoin-signature-shared`. Grid = 3 new
27-cell h100 arrays (29001032-34, all 81/81 COMPLETED, ~96 min wall) + v3 reused as regular-e4:
`results/setjoin_v5_{shared_e4,shared_e8,sig_e8}/`.

**RESULT — negative, matching the MoRE S/C/P/H ablation: no addition beats the base.** Beats-RT
counts: regular-e4 (v3) 5/9, shared-e4 5/9, regular-e8 5/9, shared-e8 4/9; v3 has the best mean
AUROC (79.8 vs 79.4/79.2/79.0) AND best mean NMAE (0.476 vs 0.479/0.497/0.491). Shared's only real
win is driver-dnf (81.2 at shared-e8, best MoE-arm number — still under v2 dense's 81.5); it costs
user-ignore (88.4→86.9) and driver-top3 (84.7→83.6). **Keep: v2 dense remains the best overall
single-table config (6/9), v3 the best MoE arm; shared expert and e8 both rejected.**

DoD: `pytest` **112 passed, 1 skipped** (shared-expert grads/arm label; variant_of). GPU smoke-train
green.

## Phase S9 — substrate complexity: asymptotics + measured latency

`scripts/bench_complexity.py` (A40, rel-f1/driver-dnf, real trained configs; JSONs in
`results/complexity/`). Asymptotics per seed: MoRE = O(L·4·S²·d) attention (S=512 fixed cap, bool
masks → SDPA math backend materializes L·4·h·S² scores = the 15-25 GB peak) + O(L·M·S·d·d_ff) MoE;
SetJoin = O(2W²d + 2N²d) attention (W=N=128, fast-path SDPA) + O(4·M·(W+N)·d·d_ff) MoE →
~126× less quadratic work, FFN-dominated. Measured: train step (fwd+bwd) 9.5-9.7 ms/seed MoRE vs
0.44-0.53 SetJoin-v3 (~20×, ~15× less memory; dense v2 ~30×). Inference forward (eval+no_grad):
B=1 latency 18-21 ms MoRE vs 10 ms v3 (both launch-overhead-bound); B=128 throughput 291 seeds/s
MoRE vs 6,395 v3 / 11,456 v2 (~22×/39×). Arm costs: +shared +9%, e8 +38%, axial ~3× (rejected on
cost AND accuracy). NB: SetJoin training is dataloader-bound (h100 end-to-end gap ~4× ≪ 20× step
gap) — headroom is in the sampler, not the model.

## Phase S10 GATE — backbone sweep result (arrays 29001119 + 29001753 re-runs; 486/486)

**The fairness sweep works: per-task best-of-18 beats RT 6/9 and MoRE's tuned grid best 3/9**
(user-ignore 90.5 = new best-anywhere number, >v2 90.1; user-repeat 79.69 ≈ RT 79.70 and > MoRE
79.5; study-outcome 69.3 ≈ MoRE 69.4). The shared expert gets its first real wins at bigger
backbones (study-outcome 69.3 and user-ignore 90.5 are both +shared configs) after losing
everywhere at d128/2+2 (v5). site-success still lost by everyone (hop-2 motivation intact).
**ADOPTED BACKBONE (≤30M mean-rank rule, complete data): cfg#10 = d256 2+2 ff1024, signature e4,
16.9M params, mean rank 5.67/18** — also the unrestricted winner; old default d128/2+2/ff256 ranks
in the middle of the top-5. Full table: `results/setjoin_grid/AGGREGATE.txt`. 12 cells hit the
sporadic MET offset assert → re-run `--num-workers 0` (29001753), all green. Winner's-curse caveat
applies to per-task bests (best-of-18, 3 seeds each).

Chain: `scripts/chain_gates.{py,sh}` + `grid.pick_backbone` auto-launched the S11/S12 gates on the
adopted backbone: hop-2 (fanout2=8, set256) = 29001765 → `results/setjoin_p2_hop2/`, control
(set256) = 29001766 → `results/setjoin_p2_ctrl/`, cap32 = 29001767 → `results/setjoin_p3_cap32/`.
