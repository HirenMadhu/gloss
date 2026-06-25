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
the relational signature) vs `value` vs `identity` vs `dense` (plain RT, no MoE), plus `hidden` and a
param-matched `dense_wide`. It is the **default result**, run on all entity tasks of the three smallest
RelBench DBs with **TEST-set** accuracy. The headline claim: `signature ≥ value/dense` in-distribution (and,
in a deferred phase, `signature ≫ identity` on held-out schemas — `identity` references dataset-specific ids
and *cannot* transfer). If `signature ≈ dense`, the honest output is a **negative result** — the RGCN null:
RT's frozen-LM token may already let a single shared FFN absorb every column.

> **The earlier DOC-RT design is RETIRED.** Per-column documentation, the prose corpus, grounding/retrieval,
> the four doc regimes (full/null/shuffled/name_only), and FiLM conditioning of the cell encoder no longer
> exist here — that whole stack lives under `archive/doc-rt/`. **Do not reintroduce it** (no `docs.md` corpus,
> no grounding, no FiLM `γ/β`, no `d_null`). The even-earlier **HALOS** design (Gaussian-in-τ kernels, geometry
> generator, scale-equivariance) is under `archive/halos/` — also retired.

**The spec files are normative — read them before doing anything:**
- [idea.md](idea.md) — method design / rationale: the thesis (route on metadata, transform content), the one
  observation it's built on (RT's *additive* cell token = value component + frozen-LM schema component), the
  method (relational signature + MoE FFN + orthogonality balance), positioning (RGCN / GMoE / HER / HOPE /
  Switch), and what would falsify it.
- [implementation.md](implementation.md) — build spec: where the MoE goes, the signature / `MoEFFN` code, the
  routing-signal ablation, the experiment plan, metrics & diagnostics. **Contracts (shapes, semantics) are
  normative; exact library function names are not** — `relbench`/`pytorch-frame` APIs drift, so adapt.

> implementation.md is written against the *upstream* `snap-stanford/relational-transformer` repo by name, but
> **we build on this repo's own faithful RT reimplementation** (`gloss/model/rt_substrate.py`) — same
> cell-token + relational-mask substrate, SDPA attention, **no upstream clone**.

## Naming: MoRE = the method, `gloss` = the package

The *method* is **MoRE** (Mixture of Relational Experts); the *package/repo here* is **`gloss`**. The model
class is `MoRE` (`gloss/model/more.py`). Retired stacks: DOC-RT under `archive/doc-rt/`, HALOS under
`archive/halos/`. Rename freely if a better handle appears.

## Environment (provisioned)

`.venv` (uv, py3.12): **torch 2.8.0+cu128**, torch_geometric 2.8.0, torch_frame 0.3.0, relbench,
sentence-transformers, transformers, lightgbm, pytorch-lightning, hydra, wandb. Use `.venv/bin/python`. The dev
node has an **A40 (46 GB)**; bigger jobs go to SLURM (`gpu`/`priority_gpu`: h100:4 / a40:4; the ablation gate is
`scripts/run_ablation.sh`, a SLURM job array on **h100**). **flash-attn is installed (2.8.3) but unused** — MoRE's
relational attention uses **SDPA**, and its four boolean masks force the math backend (flash kernels don't take
arbitrary masks); the only other would-be consumer is the frozen Qwen encoder, which runs offline.

## Working agreement (these govern how you work here)

- **Build on RT, don't reinvent it.** Keep RT's cell-token + relational-mask substrate (same-column / same-row
  / parent-FK / child-FK masks, no positional encoding, names-as-strings). RT / GNN / LightGBM are baselines.
- **One mechanism.** The MoE enters at exactly one point: RT's `SwiGLU` FFN inside each `RelationalBlock`
  becomes a `MoEFFN`. The router reads the **value-free signature**; the experts transform the hidden state.
- **Frozen text encoder, cached.** Column-name embeddings are computed offline with **`Qwen/Qwen3-Embedding-4B`**
  (`d_text`≈2560; config-swappable; `HashEncoder` for dev/tests) and cached to `data/schema_cache/`. **No LM
  forward passes in training** — gather the frozen `[C, d_text]` name table by column id.
- **Value-free routing = leak-free by construction.** The signature is a pure function of a cell's own
  `(column, modality, recency)`; no neighborhood/global statistic enters the router. Recency uses **fixed,
  context-independent** buckets. This is a unit-tested invariance (`test_routing_invariance.py`).
- **Six routing arms, one trained encoder:** `signature` (the method) | `value` | `identity` | `hidden` |
  `dense` (plain RT) | `dense_wide` (param-matched dense, `d_ff×k`). `signature` vs `dense` is the headline.
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
  (`RelationalSignature`), MoE wired into `model/rt_substrate.py`, `model/more.py` (`MoRE`, `forward →
  (logits, aux)`).
- **C. Training + regression.** **[done]** — `train/loop.py` `MoRELitModule` (task-type dispatch, regression
  standardization), `train/losses.py` (`masked_mse`, `task_loss`), `eval/metrics.py` (`regression_metrics`),
  task-type-aware `eval/test_eval.py`.
- **D. Multi-DB routing-signal ablation (THE GATE).** **[done]** — `eval/ablation.py` (`entity_tasks`,
  `build_grid`, `run_config`, `aggregate`, `format_table`), `scripts/run_ablation.{py,sh}`,
  `scripts/build_schema_cache.{py,sh}`, `eval/diagnostics.py`. **Stop & report at the gate.**
- **E. Docs sync.** **[this file].**

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
  (pytorch-frame dtype value embedding + frozen-LM column-name token). No FiLM. `forward(cb, return_value)` also
  returns the value component for the `value` arm.
- **Relational signature (the router input — `signature.py`):**
  `z_{u,c} = RMSNorm(W_s name_c + ψ(modality_c) + φ(recency_{u,c}))` — **value-free**, computed **once**, shared
  across blocks. `recency` = fixed log-spaced bucket of `seed_time − row_time` (bin 0 = untimed/pad).
- **MoE FFN (the only new mechanism — `moe.py` + `rt_substrate.py`):** each `RelationalBlock`'s FFN becomes
  `MoEFFN` = `M` SwiGLU experts + a top-`k` router. `y = Σ_{top-k} softmax(W_g · route_feat) · E_j(h)`.
  `route_feat` is set by `route_on`: `signature→z`, `hidden→normed h`, `value→value component`,
  `identity→learned per-column embedding`. **Router input dim is set by the arm** (`d_sig` for
  signature/identity, `d_model` for hidden/value). Balance: `‖ŴŴᵀ − I‖²_F` summed over blocks → `aux`.
- **Relational attention (`rt_substrate.py`, unchanged otherwise):** four boolean `[B,S,S]` masks
  (col/feat/nbr/full) rebuilt each forward; `col→feat→nbr→full` masked **SDPA** attention + the FFN, pre-norm
  RMSNorm; `n_blocks` blocks; `forward → (states, aux)`.
- **Head (`heads.py`):** `EntityHead` mean-pools the seed-row cells → logits `[B, out_dim]` (`out_dim=1` for
  binary and regression).

**Memory note (load-bearing for batch sizing):** attention is dense `O(S²)` (4 masks/block; SDPA bool-mask →
math backend materializes the score matrices for backward). The MoE **dense-combine MVP runs all `M` experts on
all tokens** → ~`M×` FFN compute/activations per block. Keep `seq_len` tight (512 for rel-f1), batch modest,
**default `M=4`**; placement configurable (`all` vs `upper_half`). True sparse dispatch (and active-FLOP parity)
is deferred — so report vanilla `dense` **and** `dense_wide`, and don't claim active-FLOP parity.

**Tested invariants / contracts (the test matrix):**
- **Temporal leakage** — no cell with `row_time > seed_time` (`test_leakage.py`).
- **Routing is leak-free** — a seed cell's signature is identical across two different neighbor samples
  (`test_routing_invariance.py`).
- **Signature is value-free** — changing cell *values* doesn't change `z` (`test_signature.py`).
- **MoE** — top-k gates sum to 1, dense combine = weighted expert sum, `ortho_loss` finite > 0 (`test_moe.py`).
- **FK-role disambiguation** — two FKs into one table get distinct `fk_role_id` (`test_graph.py`; synthetic
  dual-FK fixture in `conftest.py`).
- **Cell batch / shapes / model** — RT contracts (`test_shapes.py` = `MoRE` forward over all 6 arms + grad-flow
  to router & signature; `test_collate.py`, `test_column_encoder.py`); overfit one batch binary **and**
  regression (`test_train.py`); losses + regression metrics (`test_losses.py`); ablation bookkeeping —
  grid enumeration + CI aggregation + lift tables (`test_ablation.py`).

**Tests are hermetic** on tiny synthetic dual-FK fixtures (`tests/conftest.py`); only rel-f1-guarded tests touch
the cached real dataset (MoRE-level tests need real stype stats → rel-f1-guarded).

**Config-driven switches:** `moe.route_on ∈ {signature, hidden, value, identity, dense, dense_wide}`;
`moe.{num_experts, k, d_sig, lambda_ortho, placement}`; `data.collate.{seq_len, max_fk}`. CLI:
`run_train.py --route-on / --num-experts / -k / --lambda-ortho / --moe-placement / --dataset / --task / --test`.

## The datasets

First DB = **rel-f1** (cached at `~/.cache/relbench/rel-f1`; **5 entity tasks**: `driver-dnf`/`driver-top3`
binary, `driver-position`/`qualifying-position`/`results-position` regression). The headline spans the **three
smallest** by rows — **rel-f1, rel-stack, rel-trial** — all entity (classification + regression) tasks,
TEST-set reported. `entity_tasks()` enumerates them and excludes link/recommendation. **rel-stack is not cached
yet** (download + `build_schema_cache.sh` before the SLURM array). rel-trial and rel-event are cached.

## Commands

Use `.venv/bin/python`.

```bash
.venv/bin/python scripts/run_train.py --dry-run                              # sample a cell batch, forward MoRE(signature)
.venv/bin/python scripts/run_train.py --train --route-on signature --test    # one arm + TEST eval (--encoder hash|qwen)
.venv/bin/python scripts/build_schema_cache.py                               # cache Qwen column-name embeddings (prereq)
.venv/bin/python scripts/run_ablation.py --list                              # ablation grid size
.venv/bin/python scripts/run_ablation.py --index N                           # one (dataset,task,signal,seed) arm
.venv/bin/python scripts/run_ablation.py --aggregate                         # per-(dataset,task) table, Δ vs dense
sbatch scripts/build_schema_cache.sh                                         # then: N=$(... run_ablation.py --list); sbatch --array=0-$((N-1))%8 scripts/run_ablation.sh
.venv/bin/python -m pytest tests/                                            # the test matrix must stay green
```

## Non-goals / guardrails

Billion-row scale; training the text encoder; beating SOTA accuracy. **No docs / FiLM / grounding** (retired →
`archive/doc-rt/`); **no τ / geometry generator / scale-equivariance** (HALOS → `archive/halos/`). The headline
is **`signature` vs `dense` in this one codebase**; `signature ≈ dense` is a valid, honest negative. Keep **one
mechanism** (the MoE FFN). The router routes on the **value-free signature** only — `value` / `identity` /
`hidden` are *ablation arms*, never the default, and no neighborhood/global statistic ever enters the router.
