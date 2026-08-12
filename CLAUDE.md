# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo builds **MoRE** (Mixture of Relational Experts) — a **method paper** on top of **RT** (Relational
Transformer, arXiv:2510.06377). We keep RT's **cell-token substrate and relational attention masks unchanged**
and add **one new mechanism**: a **Mixture-of-Experts FFN** inside each relational block whose **router
conditions only on a cell's value-free relational signature** — its frozen-LM column-name embedding, its
modality (pytorch-frame stype), and its causal recency — while the **experts transform the cell's evolving
content**. *"Route on semantics, transform the content."* Balance is a **router-orthogonality loss** (not a
uniform load-balancing aux loss), so expert usage may follow the long tail of relation frequencies.

**The load-bearing experiment is the routing-signal ablation, in this one codebase**: `signature` (route on
the relational signature) vs `dense` (plain RT FFN, no cell MoE). It is the **default result**, run on all
entity tasks of the three smallest RelBench DBs with **TEST-set** accuracy. The headline claim:
`signature ≥ dense` in-distribution. If `signature ≈ dense`, the honest output is a **negative result** — the
RGCN null: RT's frozen-LM token may already let a single shared FFN absorb every column. The `value` /
`identity` / `hidden` / `dense_wide` arms (and the deferred `signature ≫ identity` transfer claim) are
archived under `archive/multi-level/`; reinstating them means restoring that code first.

> **The `multi-level` branch was reduced to the adopted model on 2026-08-12.** The two-level
> encoder with the **flex** cell backend, the additive `Broadcast` and `route_on: signature` is the
> only configuration in the live tree. The single-level `arch: rt` substrate, the phase-0a/0b
> ablation ladder, `RowToCellAttention`, the recency x-channel, the S/C/P/H MoE additions and the
> four extra routing arms are under **`archive/multi-level/`** — read its README before assuming a
> switch still exists. **Do not reintroduce them.**

> **The earlier DOC-RT design is RETIRED.** Per-column documentation, the prose corpus, grounding/retrieval,
> the four doc regimes (full/null/shuffled/name_only), and FiLM conditioning of the cell encoder no longer
> exist here — that whole stack lives under `archive/doc-rt/`. **Do not reintroduce it** (no `docs.md` corpus,
> no grounding, no FiLM `γ/β`, no `d_null`). The even-earlier **HALOS** design (Gaussian-in-τ kernels, geometry
> generator, scale-equivariance) is under `archive/halos/` — also retired.

> **On the `multi-level` branch, read [changes.md](changes.md) AND [amendments.md](amendments.md).**
> `changes.md` is the plan for the two-level (cell, row) encoder; `amendments.md` records where that
> plan was **measurably wrong** — six falsified claims and five bugs, each with its measurement. Note
> `report.md`, which `changes.md` cites throughout, is **absent from the repo**, so every `report X##`
> citation is unverifiable. Never trust a `changes.md` figure that `amendments.md` supersedes.

**The spec files are normative — read them before doing anything:**
- [idea.md](idea.md) — method design / rationale: the thesis (route on metadata, transform content), the one
  observation it's built on (RT's *additive* cell token = value component + frozen-LM schema component), the
  method (relational signature + MoE FFN + orthogonality balance), positioning (RGCN / GMoE / HER / HOPE /
  Switch), and what would falsify it.
- [implementation.md](implementation.md) — build spec: where the MoE goes, the signature / `MoEFFN` code, the
  routing-signal ablation, the experiment plan, metrics & diagnostics. **Contracts (shapes, semantics) are
  normative; exact library function names are not** — `relbench`/`pytorch-frame` APIs drift, so adapt.

> implementation.md is written against the *upstream* `snap-stanford/relational-transformer` repo by name, but
> **we build on this repo's own faithful RT reimplementation** (now `archive/multi-level/model/rt_substrate.py`,
> superseded by the two-level `gloss/model/two_level.py`) — same
> cell-token + relational-mask substrate, SDPA attention, **no upstream clone**.

## Naming: MoRE = the method, `gloss` = the package

The *method* is **MoRE** (Mixture of Relational Experts); the *package/repo here* is **`gloss`**. The model
class is `MoRE` (`gloss/model/more.py`). Retired stacks: DOC-RT under `archive/doc-rt/`, HALOS under
`archive/halos/`. Rename freely if a better handle appears.

## Environment (provisioned)

`.venv` (uv, py3.12): **torch 2.8.0+cu128**, torch_geometric 2.8.0, torch_frame 0.3.0, relbench,
sentence-transformers, transformers, lightgbm, pytorch-lightning, hydra, wandb. Use `.venv/bin/python`. The dev
node has an **A40 (46 GB)**; bigger jobs go to SLURM (`gpu`/`priority_gpu`: h100:4 / a40:4; the ablation gate is
`scripts/run_gridsearch.sh`, a SLURM job array on **h100**). **flash-attn is installed (2.8.3) but unused** — MoRE's
relational attention uses **SDPA**, and its four boolean masks force the math backend (flash kernels don't take
arbitrary masks); the only other would-be consumer is the frozen Qwen encoder, which runs offline.

## Working agreement (these govern how you work here)

- **Build on RT, don't reinvent it.** Keep RT's cell-token + relational-mask substrate (same-column / same-row
  / parent-FK / child-FK masks, no positional encoding, names-as-strings). RT / GNN / LightGBM are baselines.
- **One mechanism, now at two granularities.** On `main` the MoE enters at exactly one point: RT's
  `SwiGLU` FFN inside each `RelationalBlock` becomes a `MoEFFN`, routed on the **value-free** cell
  signature. The `multi-level` branch **adds** a second, coarser MoE on the **row** FFN, routed on a
  value-free *row* signature (table name ⊕ in-role ⊕ hop ⊕ recency); the cell MoE is unchanged. The
  invariant that matters — *route on a value-free signature, transform the content* — holds at both
  levels; "exactly one point" no longer does, and that is a deliberate decision, not drift.
  Cell experts are **routed-only** (`use_shared=False`); row experts are **shared + routed**. That
  asymmetry is intentional: the cell `+S` arm measured only mildly positive, on regression alone.
- **Frozen text encoder, cached.** Column-name embeddings are computed offline with **`Qwen/Qwen3-Embedding-4B`**
  (`d_text`≈2560; config-swappable; `HashEncoder` for dev/tests) and cached to `$GLOSS_SCHEMA_CACHE` (see
  *Cache locations* below — **not** the repo's `data/schema_cache/`, which is only the unset-env fallback).
  **No LM forward passes in training** — gather the frozen `[C, d_text]` name table by column id. Note
  `EmbeddingCache` computes-and-writes on a miss rather than failing, so a wrong cache root degrades to a
  silent LM load at train time instead of an error.
- **Value-free routing = leak-free by construction.** The signature is a pure function of a cell's own
  `(column, modality, recency)`; no neighborhood/global statistic enters the router. Recency is the
  fixed-frequency ladder (`model/time_encoding.py`) whose `ω` is a non-persistent buffer, never
  learned — context-independent, so the leak-free claim holds. Unit-tested (`test_routing_invariance.py`).
- **Two routing arms:** `signature` (the method) and `dense` (a single shared SwiGLU = RT's FFN).
  `signature` vs `dense` is the headline. `value`/`identity`/`hidden`/`hybrid`/`dense_wide` are archived.
- **Entity tasks, both types.** Binary classification (AUROC) **and** regression (MAE) across all entity tasks
  of the 3 smallest DBs; report **TEST-set** accuracy. Recommendation/link tasks deferred. Regression targets
  are z-scored with TRAIN stats and de-standardized for metrics/eval.
- ≤ ~30M params; **global seed everywhere**. After each phase append `PROGRESS.md` and commit
  `feat(phase-N): <summary>`. **At a gate, stop and report.**
- **Test everything you implement** and run the tests; every phase ends green on its tests AND its DoD command.

## Build status — done in order; the gate is the ablation (Phase D)

- **A. Archive DOC-RT, reduce to plain RT.** **[done]** — doc stack → `archive/doc-rt/`; `gloss/text/` holds the
  frozen encoder + cache (`cache.py`) and the per-column name table (`schema.py`); FiLM/grounding stripped; the
  model is plain RT (value dtype-encoder + RT name token).
- **B. MoRE core.** **[done]** — `model/moe.py` (`SwiGLU` + `MoEFFN` + `ortho_loss`), `model/signature.py`
  (`RelationalSignature`), MoE wired into `model/two_level.py` at BOTH levels, `model/more.py`
  (`MoRE`, `forward → (logits, aux)`).
- **C. Training + regression.** **[done]** — `train/loop.py` `MoRELitModule` (task-type dispatch, regression
  standardization), `train/losses.py` (`masked_mse`, `task_loss`), `eval/metrics.py` (`regression_metrics`),
  task-type-aware `eval/test_eval.py`.
- **D. Multi-DB routing-signal ablation (THE GATE).** **[done]** — `eval/ablation.py` (`entity_tasks`,
  `build_grid`, `run_config`, `aggregate`, `format_table`), `scripts/run_gridsearch.{py,sh}`,
  `scripts/build_schema_cache.{py,sh}`, `eval/diagnostics.py`. **Stop & report at the gate.**
- **E. Docs sync.** **[this file].**
- **F. Reduce to the adopted model.** **[done, 2026-08-12]** — every non-adopted arm moved to
  `archive/multi-level/` (see its README for the full table and the collapsed switches); 258 tests green.

**Deferred** — leave-one-DB-out zero-shot transfer (`signature ≫ identity`, the idea's headline) + masked-cell
pretraining; true **sparse** MoE dispatch (for active-FLOP parity); recommendation/link tasks; the recency-axis
ablation; the self-label P0b (seed's past task-table rows as cells).

## Architecture cheat-sheet (parts that span multiple files)

**Pipeline:** RelBench DB → hetero temporal graph (rows=nodes, table=node type, PK→FK typed edges labeled by
`fk_role_id`) → leakage-safe temporal neighbor sample per seed → **`collate.py` → dense RT cell-token batch**
(`CellBatch`): per-cell `[B,S]` `node_idxs / col_idxs / table_idxs / is_padding / is_seed_cell / row_time /
is_timed`; per-cell `[B,S,max_fk]` `f2p_nbr_idxs`; per-seed `[B]` `seed_time / target / has_target`; plus
`tf_dict` (TensorFrame per node type) + `cell_placement` (scatter map). `seq_len` is a **fixed** cap (seed-row
cells emitted first so they survive truncation). rel-f1 ≈ 353 cells/seed → `seq_len=512`.

**Key equations / mechanism:**
- **Cell token (RT, unchanged — `column_encoder.py`):** `x_{u,c} = W_v Enc_dtype(v_{u,c}) + W_name name_c`
  (pytorch-frame dtype value embedding + frozen-LM column-name token). No FiLM.
- **Relational signature (the router input — `signature.py`):**
  `z_{u,c} = RMSNorm(W_s name_c + ψ(modality_c) + W_τ[sin θ; cos θ])` — **value-free**, computed **once**,
  shared across blocks. `θ` comes from the fixed `TimeLadder` over `seed_time − row_time`; untimed cells
  get an all-zero feature pair, so "unknown Δ" stays separable from "Δ = 0".
- **MoE FFN (the new mechanism — `moe.py`, at BOTH levels in `two_level.py`):** the cell FFN is
  `MoEFFN` = `M` SwiGLU experts + a top-`k` router on `z` (`d_route = d_sig`), routed-only; the row FFN
  is `RowMoE` on the row signature, shared + routed. `y = Σ_{top-k} softmax(W_g · z) · E_j(h)`.
  Balance: `‖ŴŴᵀ − I‖²_F` summed over blocks → `aux`, returned **split by level**.
- **Two-level block (`two_level.py`), six sublayers:** cell attention (temporal RoPE, **padding-only**
  mask, `flex` or `sdpa` backend) → cell FFN → `RowPool` (cell→row) → row attention (RoPE +
  name-derived role bias γ) → `RowMoE` → `Broadcast` (row→cell, additive). Pre-norm RMSNorm and a
  residual on every sublayer except `RowPool`/`Broadcast`, which are residual but unnormed.
- **Head (`heads.py`):** `RowTokenHead` reads the seed **root row token** (`row_is_root`) → logits
  `[B, out_dim]` (`out_dim=1` for binary and regression).

**Memory note (load-bearing for batch sizing):** cell attention is dense `O(S²)`, one attention per block
(the four RT masks are retired). Under `sdpa` a bool mask forces the math backend, which materializes the
score matrices for backward; under `flex` the padding block-mask skips them — that is what makes
`seq_len=3456` on rel-event fit (19.8 GiB vs 53.0 at B=64). The MoE **dense-combine MVP runs all `M` experts
on all tokens** → ~`M×` FFN compute/activations per block, at BOTH levels. Keep `seq_len` tight, batch
modest, **default `M=4`**. True sparse dispatch (and active-FLOP parity) is deferred — don't claim
active-FLOP parity. Measured: 256/4 OOMs above batch 256, 128/2 fits 512.

**Tested invariants / contracts (the test matrix):**
- **Temporal leakage** — no cell with `row_time > seed_time` (`test_leakage.py`).
- **Routing is leak-free** — a seed cell's signature is identical across two different neighbor samples
  (`test_routing_invariance.py`).
- **Signature is value-free** — changing cell *values* doesn't change `z` (`test_signature.py`).
- **MoE** — top-k gates sum to 1, dense combine = weighted expert sum, `ortho_loss` finite > 0 (`test_moe.py`).
- **FK-role disambiguation** — two FKs into one table get distinct `fk_role_id` (`test_graph.py`; synthetic
  dual-FK fixture in `conftest.py`).
- **Cell batch / shapes / model** — RT contracts (`test_shapes.py` = `MoRE` forward over both arms + grad-flow
  to BOTH levels' routers & the signature; `test_collate.py`, `test_column_encoder.py`); overfit one batch binary **and**
  regression (`test_train.py`); losses + regression metrics (`test_losses.py`); ablation bookkeeping —
  grid enumeration + CI aggregation + lift tables (`test_ablation.py`).

**Tests are hermetic** on tiny synthetic dual-FK fixtures (`tests/conftest.py`); only rel-f1-guarded tests touch
the cached real dataset (MoRE-level tests need real stype stats → rel-f1-guarded).

**Config-driven switches:** `moe.route_on ∈ {signature, dense}`; `moe.{num_experts, k, d_sig,
lambda_ortho}`; `model.two_level.cell.backend ∈ {sdpa, flex}`; `model.two_level.row.{pool_slots,
num_experts, top_k, use_shared, lambda_ortho, lambda_balance}`; `data.collate.{seq_len, max_fk}`.
CLI: `run_train.py --route-on / --num-experts / -k / --lambda-ortho / --dataset / --task / --test`;
`run_gridsearch.py --grid-set {default,small,large} / --cell-attn-backend {sdpa,flex}`.

## The datasets

First DB = **rel-f1** (**5 entity tasks**: `driver-dnf`/`driver-top3` binary,
`driver-position`/`qualifying-position`/`results-position` regression). The headline spans the **three
smallest** by rows — **rel-f1, rel-stack, rel-trial** — all entity (classification + regression) tasks,
TEST-set reported. `entity_tasks()` enumerates them and excludes link/recommendation.

### Cache locations — there are TWO roots, and only one of them is real for SLURM

`scripts/env.sh` redirects every cache to scratch, and **every SLURM script sources it**:

```
RELBENCH_CACHE_DIR=$HOME/scratch60/gloss/relbench      # NOT ~/.cache/relbench
GLOSS_SCHEMA_CACHE=$HOME/scratch60/gloss/schema_cache  # NOT <repo>/data/schema_cache
GLOSS_GRAPH_CACHE=$HOME/scratch60/gloss/graph_cache
HF_HOME=$HOME/scratch60/gloss/hf
```

`gloss/utils/paths.py` falls back to the **repo-relative** `data/schema_cache/` when the env var is unset —
which is what a plain `.venv/bin/python …` in a login shell gets. So the two roots disagree, and the repo one
is the misleading one: `data/schema_cache/rel-stack/name_emb_qwen.pt` exists but **no job will ever read it**,
while `data/schema_cache/rel-trial/` is missing its qwen file even though the qwen grid ran rel-trial fine.
**Always check `$HOME/scratch60/gloss/schema_cache` before concluding a dataset is uncached**, and `source
scripts/env.sh` before any local run meant to share state with SLURM.

Some entries under the scratch roots are **symlinks into `~/.cache/relbench`** (rel-avito, rel-stack) or into
`~/scratch60/relbench` (rel-event) rather than copies — done to avoid re-downloading data already on disk.
Home is the tight quota (~125 GiB) and is where `~/.cache/relbench` lives; scratch60 has ~20 TB. **New
downloads must go to the scratch root**, never home.

Leaderboard coverage (7 DBs): rel-f1 / rel-trial / rel-event are fully cached; rel-avito / rel-stack /
rel-hm / rel-amazon were prepped later via `sbatch scripts/prep.sh --datasets <ds> --encoder qwen`
(`prep.sh` pins `h100:1`; override with `--gpus=1` so a schema-cache build can take an a40 instead of
competing with training).

## Commands

Use `.venv/bin/python`.

```bash
.venv/bin/python scripts/run_train.py --dry-run                              # sample a cell batch, forward MoRE(signature)
.venv/bin/python scripts/run_train.py --train --route-on signature --test    # one arm + TEST eval (--encoder hash|qwen)
source scripts/env.sh                                                        # FIRST, for any local run sharing state with SLURM
.venv/bin/python scripts/build_schema_cache.py                               # cache Qwen column-name embeddings (prereq)
sbatch --gpus=1 scripts/prep.sh --datasets rel-hm --encoder qwen             # download + task tables + graph + schema, one DB
.venv/bin/python scripts/run_gridsearch.py --list --arch two_level --grid-set large --lr-set single --tasks all --seeds 3
.venv/bin/python scripts/run_gridsearch.py --index N ...                     # one (config,dataset,task,seed) arm
.venv/bin/python scripts/aggregate_gridsearch.py --dir results/<out-dir>     # per-task table vs RT / GelGT
sbatch scripts/build_schema_cache.sh                                         # then: N=$(... --list); sbatch --array=0-$((N-1))%8 scripts/run_gridsearch.sh <same flags>
.venv/bin/python -m pytest tests/                                            # the test matrix must stay green
```

## Non-goals / guardrails

Billion-row scale; training the text encoder; beating SOTA accuracy. **No docs / FiLM / grounding** (retired →
`archive/doc-rt/`); **no τ / geometry generator / scale-equivariance** (HALOS → `archive/halos/`). The headline
is **`signature` vs `dense` in this one codebase**; `signature ≈ dense` is a valid, honest negative. Keep **one
mechanism** (the MoE FFN). The router routes on the **value-free signature** only — `value` / `identity` /
`hidden` are *ablation arms*, never the default, and no neighborhood/global statistic ever enters the router.
