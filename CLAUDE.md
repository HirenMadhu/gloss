# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo builds **DOC-RT** — a **method paper** on top of **RT** (Relational Transformer, arXiv:2510.06377).
We keep RT's **cell-token substrate and relational attention masks unchanged** and add **one new signal**:
per-column **documentation**, grounded from realistic prose and injected by **FiLM** into the cell encoder, so
*the interpretation of a value is conditioned on its column's documented meaning* (what an opaque code means,
what a sentinel means, what unit a number is in, whether a table is a fact or a dimension).

**The load-bearing experiment is docs-on vs docs-off, in this one codebase** (`full` grounded `d_c` vs the
learned `null` embedding for every column). It is the **default Phase-4 result**, not a later ablation —
"DOC-RT beats RT-from-the-paper" is a different, confounded comparison. If docs-on does not beat docs-off,
the honest output is a negative result.

> **The earlier HALOS design is RETIRED.** Gaussian-in-τ "halo" kernels, the doc-generated geometry operator
> (`g_θ`), the dimensionless-time coordinate `τ`, and exact scale-equivariance no longer exist here — that whole
> stack lives under `archive/halos/`. **Do not reintroduce it** (no τ, no geometry generator, no temporal
> kernel, no scale-equivariance test). `DOC-RT` is a working handle until we pick a real name; rename freely.

**The spec files are normative — read them before doing anything:**
- [idea.md](idea.md) — method design / rationale: the thesis, the honest status of each claim, the architecture
  (RT substrate, prose grounding, FiLM cell conditioning), positioning (RT, RELATE, ConTextTab), the doc corpus.
- [implementation.md](implementation.md) — build spec: phases, data contracts, equations, config, test matrix.
  **Contracts (shapes, semantics) are normative; exact library function names are not** — `relbench`/
  `pytorch-frame` APIs drift, so adapt to the installed version.
- [doc_authoring.md](doc_authoring.md) — the blind doc-authoring protocol for the corpus.

Keep the spec files in sync (implementation.md = build plan; idea.md = rationale).

## Naming: DOC-RT = the method, `gloss` = the package

The spec calls the *method* **DOC-RT** and writes the package as `docrt/`. The *package/repo here is* **`gloss`**.
Wherever the spec writes `docrt/`, that maps to **`gloss/`** — keep the `gloss` name. The retired HALOS modules
live under `archive/halos/`.

## Environment (provisioned)

`.venv` (uv, py3.12) holds the full stack: **torch 2.8.0+cu128** (cxx11abi=True), torch_geometric 2.8.0,
torch_frame 0.3.0, relbench, sentence-transformers, transformers, lightgbm, shap, pytorch-lightning, hydra,
wandb. Use `.venv/bin/python`. The dev node has an **A40 (46 GB)**; bigger jobs go to SLURM
(`gpu`/`priority_gpu`: h100:4 / a40:4; the headline gate is `scripts/run_headline.sh`, a SLURM job array).
**flash-attn is not yet built** (build needs `wheel`+`ninja` in the venv, then `sbatch scripts/build_flash_attn.sh`,
archs `86;90`); its only consumer would be the **frozen Qwen encoder** — DOC-RT's own attention uses **SDPA**.

## Working agreement (implementation.md §0 — these govern how you work here)

- **Build on RT, don't reinvent it.** Keep RT's cell-token + relational-mask substrate
  (same-column / same-row / parent-FK / child-FK masks, no positional encoding, names-as-strings). RT / RelGT /
  GNN / LightGBM are **baselines only**.
- **One signal, one mechanism.** Documentation enters at exactly one point: **FiLM on the cell encoder**
  (`γ(d_c), β(d_c)`). The optional same-column attention-bias injection is a flag, **default OFF**.
- **Frozen text encoder, cached.** All span/query embeddings computed offline with
  **`Qwen/Qwen3-Embedding-4B`** (`d_text`≈2560; replaces the spec's MiniLM default; config-swappable) and
  cached to `data/doc_cache/`. **No LM forward passes in training** — gather `d_c` by id.
- **Four doc regimes, one trained encoder:** `full` (grounded `d_c`) | `null` (docs off — every `d_c = d_null`,
  the baseline) | `shuffled` (placebo: spans permuted, length-matched) | `name_only` (column-name string —
  RELATE-style control). `full` vs `null` is the headline.
- **Self-labels stay, in every arm.** The seed entity's past task-table rows enter the subgraph as cells (the
  dominant transfer lever per RT); held in every regime so they never confound the docs comparison. Never
  include the current target row (leakage).
- **Leakage is a hard test item.** ≤ ~30M params; global seed everywhere. After each phase append `PROGRESS.md`
  and commit `feat(phaseN): <summary>`. **At a gate, stop and report.**
- **Test everything you implement** and run the tests; every phase ends green on its tests AND its DoD command.

## Phased build plan — build in order; the headline gate is Phase 4

0. **Substrate** — `data/graph.py` (RelBench DB → hetero temporal graph; `fk_role_id` from the FK edge;
   leakage-safe temporal sampler), `data/collate.py` (subgraph → RT **cell-token** batch: per-cell
   `node_idxs/col_idxs/table_idxs`, `f2p_nbr_idxs`, `is_seed_cell`, fixed `seq_len`; the 4 relational masks are
   rebuilt in the model). *Reuse `relbench.modeling` — don't hand-roll.* **[done]**
1. **Doc corpus + grounding** — author `doc_corpus/<db>/docs.md` (blind); `docs/{corpus,grounding,cache}.py`
   with regimes `full | null | shuffled | name_only`, `d_null` fallback, coverage report. **[done]**
2. **Cell encoder (CORE)** — `model/column_encoder.py` (`CellEncoder`: pytorch-frame dtype encoders + **FiLM by
   `d_c`** + RT name token + learned `d_null`). **[done]**
3. **Substrate + model** — `model/rt_substrate.py` (RT `RelationalBlock`: col/feat/nbr/full masked attention +
   SwiGLU, SDPA), `model/docrt.py` (`DOCRT` = CellEncoder → RTSubstrate → head), `model/heads.py`. **[done]**
4. **Training + HEADLINE GATE** — `train/*`; `eval/ablation.py` + `scripts/run_headline.{py,sh}`:
   **`full` vs `null` vs `shuffled` vs `name_only`** on the same encoder, multi-seed, with CIs. docs-on vs
   docs-off is THE result. `full > null` (signal) `> shuffled` (meaning) `> name_only` (beats names);
   `full ≈ null` ⇒ honest negative. Stop & report. **[running]**
5–8. **Deferred** — P0b self-labels (task-table node type + seed's past task rows); coverage-vs-gain curve;
   more DBs; optional same-column attention-bias (v2 ablation); masked-cell pretraining + transfer; the
   synthetic twin-column DB + audit (existence proof when docs are the only disambiguator).

## Architecture cheat-sheet (parts that span multiple files)

**Pipeline:** RelBench DB → hetero temporal graph (rows=nodes, table=node type, PK→FK typed edges labeled by
`fk_role_id`) → leakage-safe temporal neighbor sample per seed → **`collate.py` → dense RT cell-token batch**
(`CellBatch`): per-cell `[B,S]` `node_idxs / col_idxs / table_idxs / is_padding / is_seed_cell / row_time /
is_timed`; per-cell `[B,S,max_fk]` `f2p_nbr_idxs`; per-seed `[B]` `seed_time / target / has_target`; plus
`tf_dict` (TensorFrame per node type) + `cell_placement` (scatter map) for the cell encoder. `seq_len` is a
**fixed** cap (pad/truncate; seed-row cells emitted first so they survive). rel-f1 ≈ 353 cells/seed max →
`seq_len=512`.

**Key equations / mechanism:**
- **Doc-conditioned cell encoding (the only new mechanism, `column_encoder.py`):**
  `e_{u,c} = W_v Enc_dtype(v_{u,c})` (pytorch-frame dtype cell embedding);
  `x_{u,c} = γ(d_c) ⊙ e_{u,c} + β(d_c) + W_name name_c` — FiLM by the grounded column doc `d_c`, plus the RT
  column-name token (constant across regimes, so it never confounds). `null` regime: `d_c = d_null` ⇒ a
  names-only RT cell (the baseline). `γ, β`: shared MLPs `d_text → d_model`.
- **Relational attention (`rt_substrate.py`):** four boolean `[B,S,S]` masks rebuilt each forward from the
  index tensors — `col` (same column), `feat` (same row OR forward-FK parent), `nbr` (reverse-FK child),
  `full` (all non-pad); identity OR-ed in so no row is fully masked. A `RelationalBlock` runs col→feat→nbr→full
  masked attention (**SDPA** with the bool mask) + a SwiGLU FFN, pre-norm RMSNorm; `n_blocks` of them.
- **Head (`heads.py`):** `EntityHead` mean-pools the seed-row cells (`is_seed_cell`) → logits `[B, out_dim]`.
- **Grounding (`grounding.py`):** `q_c = E_text("table <t>, column <c>")`; `d_c = softmax-topK(cos(q_c, span)/T)·span`
  if `max cos > thresh` else `d_null`; `rel_c = max cos`. Regimes select/permute the same cached embeddings.

**Memory note (load-bearing for batch sizing):** the attention is dense `O(S²)` and SDPA with a boolean mask
falls to the math backend (materializes all `4·n_blocks` score matrices for backward). Keep `seq_len` tight
(512 for rel-f1) and `batch_size` modest (64 fits A40/H100); `seq_len=1024 × batch=64` OOMs a 46 GB A40.

**Optional input behind a flag:** same-column attention-bias `bias_samecol(c,c') += MLP([d_c;d_{c'}])`
(`--docbias`, default false; a labeled ablation that keeps the mechanism single-point).

**Tested invariants / contracts (test matrix):**
- **Temporal leakage** — no context/self-label cell with `row_time > seed_time`; timeless rows always valid
  (`test_leakage.py`).
- **FiLM responds** — switching `full ↔ null` changes the cell vectors, i.e. the mechanism is wired
  (`test_film_responds.py`).
- **Grounding** — `d_null` fallback below threshold; `shuffled` placebo decorrelated + length-matched;
  `name_emb` regime-independent; cache deterministic (`test_grounding.py`).
- **FK-role disambiguation** — two FKs into one table get distinct `fk_role_id` (`test_graph.py`; rel-f1 has no
  dual-FK-into-one-table → hand-built synthetic fixture in `conftest.py`).
- **Cell batch / shapes / training** — RT cell-token contracts (`test_shapes.py`, `test_collate.py`,
  `test_column_encoder.py`); overfit one batch + grads reach FiLM/name/head (`test_train.py`); gate
  bookkeeping = config enumeration + seed aggregation (`test_ablation.py`).
- **(Deferred, P0b)** self-label cells identical across doc regimes (`test_selflabels_constant.py`).

**Tests are hermetic:** unit tests run on tiny synthetic dual-FK cell fixtures in `tests/conftest.py` (no
network); only rel-f1-guarded tests touch the cached real dataset.

**Config-driven switches (not forks):** `docs.regime ∈ {full, null, shuffled, name_only}`; `--docbias` (off);
`data.collate.seq_len / max_fk`.

## The doc corpus (on the critical path)

RelBench DBs ship without documentation, so the corpus is a deliverable. **We author it with Claude Code under
the blind protocol** ([doc_authoring.md](doc_authoring.md)): the author sees only **schema + sample rows —
never task definitions, labels, or splits** — writes senior-dev-style prose (table overviews, FK rationale,
inline units/nulls/codes, ~20–40% of columns deliberately unmentioned, mixed granularity), and records the
blindness attestation + coverage in `<db>/meta.yaml`. This keeps the blind-authoring control real and forces
the grounding module to retrieve rather than template-match. We start with **rel-f1**.

## Commands

Use `.venv/bin/python`.

```bash
.venv/bin/python scripts/run_train.py --dry-run                 # P0 DoD: sample a rel-f1 cell batch, forward
.venv/bin/python scripts/build_doc_cache.py                    # P1: embed + cache prose-doc grounding (Qwen)
.venv/bin/python scripts/run_train.py --train --regime full    # train one arm (full|null|shuffled|name_only)
.venv/bin/python scripts/run_headline.py --aggregate           # P4 headline table (full/null/shuffled/name_only)
sbatch scripts/run_headline.sh                                 # P4 gate: SLURM array (5 seeds × 4 regimes)
.venv/bin/python -m pytest tests/                              # the test matrix must stay green
```

First dataset = **rel-f1** (cached at `~/.cache/relbench/rel-f1`; task `driver-dnf`). rel-trial/rel-stack later.

## Non-goals / guardrails

Billion-row scale; training the text encoder; beating SOTA accuracy. **No τ / temporal kernel / geometry
generator / scale-equivariance** — that retired HALOS stack lives in `archive/halos/`; don't reintroduce it.
The headline is **docs-on vs docs-off in this one codebase** (not "DOC-RT beats RT-from-the-paper");
`full ≈ null` is a valid, honest negative. Keep **one signal, one mechanism** (FiLM); the same-column
attention-bias is an off-by-default flag. Self-labels in **every** arm; keep `shuffled` + `name_only` as
first-class controls. Frame the unit as **documentation conditioning the cell encoder** (codes/units/sentinels/
roles that names don't carry) — distinct from RELATE's name/metadata conditioning.
