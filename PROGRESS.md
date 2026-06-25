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

**Next — Phase D (multi-DB ablation).** `entity_tasks()` (binary+regression only), the routing-signal
`ablation.py` (signature/hidden/value/identity/dense/dense_wide × dataset × task × seed) + `run_ablation.{py,sh}`,
`diagnostics.py` (expert usage + specialization), `build_schema_cache.{py,sh}`.
