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

---

**Masked-cell pretraining, Phase 0 — a real cell-text encoder [done].** `build_gloss_graph` had a
`text_embedder` argument that **no call site anywhere ever passed** (18 checked), so every graph cache
on scratch held `EMB_DIM=32` `HashTextEmbedder` noise for free-text **cell values**. That is not a
corner: rel-trial is 47/102 columns free text, rel-stack 15/32, rel-f1 13/45 — roughly half of every
schema was blank input. (The frozen **qwen column-NAME** table, which is what the router routes on, was
always real and is untouched. Two text tables, two jobs; see `prep_data.py`'s header.)

Added `minilm` (`all-MiniLM-L12-v2`, d=384 — RT's own cell-text encoder) to `ENCODER_MODELS` plus
`text/cache.py:value_encoder`, routed through sentence-transformers rather than `HFLastTokenEmbedder`
(MiniLM is mean-pooled; last-token pooling would return a padding position). `--text-encoder` threads
through `build_gloss_graph` → `prep_data.py` → `prep.sh`, and `paths.graph_cache_dir(ds, enc)` keys the
cache by encoder — `hash` keeps the historical un-suffixed layout, so **every previously reported
grid/leaderboard number stays readable at the path it was written to**. Measured corpus cost at d=384:
138 GB / ~96.3M free-text cells (qwen-2560 would have been 918 GB and ~430 GB resident just to load
rel-amazon). 6 of 7 DBs rebuilt, all reporting `EMB_DIM=384`; rel-amazon still running.

Two bugs found by doing it. (1) sentence-transformers 5.x rejects `float('nan')` outright where
`HashTextEmbedder`'s `str(s)` swallowed it, so missing text cells killed the pass — `_as_texts` maps
them to `""`. (2) `prep.sh` ended with `echo`, so its six failed jobs reported `COMPLETED` / rc=0;
it now propagates the exit code.

**Phase 1 — true sparse MoE dispatch [done].** `MoEFFN`/`RowMoE` evaluated **every** expert on **every**
token and multiplied the non-top-k ones by a zero gate: the gate was sparse, the compute was not
(`M×` the FFN FLOPs and activations of a plain SwiGLU). `dispatch ∈ {dense, sparse}` — sparse buckets
tokens per expert via `moe.sparse_combine` and honours a `valid` mask, which at the cell level is the
larger win, since a sampled sequence runs 6–20% full at `seq_len=1024` on every DB measured.
`num_shared` adds ungated always-on experts at either level (`RowMoE.use_shared` still means exactly 1;
`shared` is now `None`-or-`ModuleList`). `TwoLevelSubstrate` gains `grad_checkpoint` and `mean_aux`
(aux is a sum over blocks, so `lambda_ortho=0.5` tuned at E=4/8-blocks is a different objective at
E=8/10-blocks/two-levels). **Dense stays the default and is byte-identical**, so nothing measured moves.
`tests/test_sparse_moe.py`: sparse ≡ dense in **output and gradients**, shared experts ungated,
`valid` zeroes exactly the excluded tokens, dense ignores `valid`, grad-checkpointed ≡ plain.

*Two flakes fixed on the way, neither caused by the above.* `HashTextEmbedder` seeded its per-string
RNG with Python's builtin `hash()` — **randomized per process** — so every cache-less
`build_gloss_graph` materialized a *different* rel-f1 (its sibling `HashEncoder` always used
`hashlib`). Now `hashlib`; cached bundles are unaffected because they never call an embedder.
`test_regression_overfits_single_batch` additionally could not survive CUDA nondeterminism: with the
fixture fixed and `init` identical to 4 dp, three sequential CUDA trials hit 0.5431 / 0.0000 / 1.1412
at step 200 and the third was still 0.9655 at step 600 — different basins, not last-bit noise, out of
`RowPool`'s atomic `index_add`. Pinned to CPU, where it is bit-reproducible (0.000004 at step 200,
three trials) and *faster*. DoD: `pytest` **281 passed** (258 + 23 new), twice, in 39 s.

**Phase 2 — the masked-cell objective [done].** `gloss/data/masking.py` (`ColumnTargetSpec`,
`maskable_cells`, `sample_cell_mask`, `gather_masked_targets`), a per-stype mask token in
`CellEncoder`, `gloss/model/mlm_head.py`, and `MoRE.forward(cell_mask=, return_cells=)` — the
substrate already computed the cell states and `more.py:80` was throwing them away.

*Objective.* Hybrid: one masked cell on the seed root row (RT's shape — `rt-v1` masks exactly one cell
per sequence, the seed's target column) **plus** Bernoulli-`p` over the remaining candidates.
`p_random=0` is the RT-faithful arm, `seed_target=False` is plain BERT. Maskable = numerical
(Huber on the col-stats z-score, the same normalization `LinearEncoder` applies to the input) +
categorical (CE). Excluded, each for a reason: **timestamp** (relbench republishes the time column as
`row_time` on every cell of the row, so predicting it is reading it), **text/embedding** (RT:
"masking text not supported"), multicategorical, zero-variance and `__const__` columns. NaN cells are
removed from the pool *before* the draw, so a masked position always has a label.

*The categorical head is tied to the encoder's own category table.* torch_frame keeps one `nn.Embedding`
per table for all its categorical columns with a per-column `offset`; the head projects to
`enc_channels` and scores against `emb.weight[cat_base : cat_base+n_cat]`. So the head has **no**
vocabulary-shaped parameter and loads onto an unseen schema unchanged (§0) — asserted, along with the
encoder-offset-vs-`col_stats`-cumsum cross-check.

*Three things measured while building it, all of which change how the loader must work.*
(1) **rel-f1's entity table `drivers` has zero maskable columns** (all 6 are text/timestamp), so the
seed-target half is structurally silent on every rel-f1 driver task; 7 of rel-trial's 15 tables are
the same, including `facilities_studies` at 1.87M rows (~half that DB). This is why Phase 3 weights
seed tables by *maskable* rows rather than raw rows. Pinned as a test.
(2) **`nn.ModuleDict` has no `.get`** — a `.get("categorical")` silently returned None for every
table, so the tied head fell back to no categorical branch and the offset cross-check never ran.
(3) **RT's zero-init numerical decoder passes no gradient into the trunk at step 0** (`dloss/dh =
err*W = 0`), so a naive "gradients reach the encoder" check fails against a healthy model. The
categorical branch is not zero-init and does feed the trunk immediately; both are asserted separately.

DoD: `pytest` **304 passed**, twice (`test_masking.py` 14, `test_pretrain_head.py` 9). Masks never land
on padding / unmaskable / NaN cells; targets match a hand gather through `cell_placement`; a masked
cell's routing signature `z` is **bit-identical** to the unmasked one (the property the method rests
on); the objective overfits one batch on CPU.

**Phase 3 — label-free loader, pretrain loop, checkpoints, wandb [done; GATE PASSED].**
`gloss/data/pretrain_loader.py`, `gloss/train/pretrain.py`, `scripts/run_pretrain.py`.

*Loader.* Same sampler as finetuning (`time_attr`, `temporal_strategy='last'`, `disjoint=True`) — RT
also uses an identical sampler at both stages — but seeded from ordinary table rows with an explicit
`input_time` and no transform. PyG writes `seed_time` from `input_time` with no task machinery
involved, verified on both timed and **untimed** seed tables. One table per batch (so `entity_table`
and `row_is_root` stay well defined), interleaved by `RoundRobinLoader` with a **deterministic**
schedule — every DDP rank must pick the same table at the same step or the per-table encoders leave
different parameters unused on different ranks.

*Seed tables are weighted by MASKABLE rows.* The payoff is measured: on the task-driven loader rel-f1
yields **zero** seed targets (entity table `drivers` has no maskable column); on this one every seed
gets one, because `drivers`/`constructors` are excluded as seed sources and still appear as
neighbours. Train seed times are capped at `val_timestamp`, so with `row_time <= seed_time` no
val/test-period cell can enter a pretraining batch — asserted, since the graph is materialized
`upto_test_timestamp=False` and nothing else would catch it.

*Loop.* First LR schedule in the repo (linear warmup 20% -> linear decay, written out rather than
`OneCycleLR`, whose defaults silently start at `max_lr/25` and cycle beta1). First disk checkpoints:
`PretrainCheckpoint` writes a **portable trunk** and **per-DB adapters** separately, plus
optimizer/scheduler/step for the 16 h SLURM wall, atomically renamed. `load_trunk` reports
`missing_adapter` and `missing_portable` apart — a trunk file never contains adapter keys, so
conflating them makes the report useless exactly on a LODO target. bf16-mixed; aux weighted once
(`loop.py:86`'s lambda-squared is left alone there so finetune baselines do not move).

*Verified portability.* Diffed a model built on rel-f1 (9 tables / 13 roles / 45 columns) against
rel-trial (15 / 15 / 103): of the keys in both, **zero differ in shape**; every non-shared key is under
`encoder.cell_encoders`. A separate scan for any learned tensor dimensioned by `C` / `n_tables` / `K`
found **0 hits** outside `cell_encoders`. A rel-f1 trunk loads onto a rel-trial model and produces a
finite forward on a rel-trial batch.

*Two bugs this found.* `split_state_dict` matched `startswith("encoder.cell_encoders.")` against
LightningModule keys prefixed `model.`, classifying every adapter tensor as portable — now a substring
match, with a regression test. And under autocast the sparse path's `index_add` raises where the dense
path's `+` promotes (RMSNorm is not an autocast op, so `x` is fp32 while the experts return bf16).

*GATE (rel-f1, 300 steps, flex + sparse + 8 routed/2 shared, qwen names, minilm cell text):*
`train/mlm` 0.195, `val/loss` **0.170**, 1.74 it/s, wandb run `ustlrwcm`. DoD: `pytest` **320 passed**,
twice.

*Measured model scale* at 10 blocks / d512 / ff2048 / 8+2 cell / 4+1 row: **513.0M total, 261.3M
active per token** (dense combine would activate all 513M). With `--num-shared 0` at both levels:
418.6M / 166.9M. All 10 blocks are structurally identical and carry the full six sublayers.

**Schema-free cell encoder [done].** `gloss/model/schema_free_encoder.py`, selected by
`MoRE(cell_encoder='schema_free')`; `per_column` stays the default because every reported result was
measured on it.

The RT cell token `x = W_v·Enc_dtype(v) + W_name·name_c` is unchanged — only the FIRST term moves.
torch_frame's stype encoders keep **per-column** weights, and at `d_text=384` that is 10.6M on
rel-trial (2.07% of a 514M model), 96% of it the 47 separate `Linear(384,512)` in
`LinearEmbeddingEncoder`. None of it can transfer. Now each stype gets ONE shared projection, RT-style
(`W_d` is datatype-specific, not column-specific), and everything that must stay per-column becomes
frozen DATA regenerable without gradients: numerical `(v-mu_c)/sigma_c` -> shared `Linear(1,e)`;
categorical -> frozen **category-LABEL embedding** -> shared `Linear(d_text,e)`; text -> one shared
`Linear(d_val,e)`; timestamp -> **one GLOBAL year mean/std** for the whole DB (per-column `YEAR_RANGE`
would feed inconsistent scales through a shared weight) + calendar-cyclic fields.

`text/schema.py` gains `category_index` / `build_category_name_embeddings`: `phase = "Phase 3"` used to
become the integer 4 and index an `nn.Embedding` learned per database, so the string never reached the
model and index 4 meant nothing elsewhere. Now it embeds `"table studies, column phase, value Phase 3"`
with the frozen encoder — the value-side counterpart of the column-name table. **383 distinct category
values across the six built DBs, ~0.6 MB.** The masked-cell head reties to it (`MaskedCellHead(d_cat=)`)
and `build_column_target_spec` reads the encoder's global `cat_base` when present.

**Result: a rel-f1 model and a rel-trial model have IDENTICAL state_dicts** — 143 keys each, zero
non-shared, zero differing shapes — and rel-f1's weights load onto rel-trial with `strict=True`. That
is stronger than the trunk/adapter split it replaces: LODO re-initializes **nothing**, and Phase 4's
`SchemaAdapter` is no longer needed. Multi-DB training becomes "same weights, swap the frozen tables".

*Caveat to measure, not assume:* all numerical columns now share one `Linear(1,e)`, so the only thing
separating `price of product` from `age of customer` is `W_name·name_c`. That is RT's bet and it is the
same signal the router already routes on, but it is a real capacity reduction and is why `per_column`
is kept as the comparison arm.

*Found on the way:* the CACHED rel-f1 hash bundle and a freshly built one disagree on `races.year`
(numerical vs categorical) — `get_stype_proposal` infers from a 1000-row sample, so a cache built at a
different time can carry a different schema. Stable across repeated calls today; the cache is
authoritative and `run_pretrain.py` always uses it.

DoD: `pytest` **327 passed**.
