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
