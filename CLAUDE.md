# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo builds **HALOS** — and as of the **v3 spec rewrite** the deliverable is a **method paper**, not a
measurement paper. (The earlier v2 framing — a measurement paper with structured DocCards, a headline audit,
an RT-style cell-token prototype, a proxy-first gate, and the temporal kernel deferred — is **DEAD**; do not
follow it.)

**v3 thesis:** *A relational schema has accidental parts (column names, absolute timescales, node identities)
and invariant parts (documented meaning, dimensionless time, typed relations). Only invariants transfer.*
**HALOS is a node-level, geometry-aware temporal graph transformer whose attention geometry — which past rows
matter, at what time lag, through which typed link — is *generated from human-style prose documentation*.**
Build the geometric encoder **directly**: no prototype, no proxy gate.

The novel claims:
1. **Geometry generated from grounded prose docs** (`bias_generator.py`) — per-head **Gaussian-in-τ** attention
   biases whose parameters are produced by a small MLP `g_θ` over documentation + typed structure. **This is
   the core contribution** (it was the deferred "C2 kernel" in v2; it is now the headline).
2. **Exact scale-equivariance** via the dimensionless temporal coordinate `τ = log(Δt / T_ctx)` — a hard unit
   test, not a hope.
3. **Grounding from realistic prose** (chunk→embed→retrieve→pool→`d_e`), not curated per-column cards.
4. **The audit proves the mechanism routes meaning** — demoted from "the product" to a **validation section**.

**The two spec files are normative — read them before doing anything:**
- [idea.md](idea.md) — rationale: the v3 thesis, the architecture (graph substrate, grounding, doc-conditioned
  features, the doc-generated geometry operator, transfer, interpretability), positioning (RT, RelGT, GelGT,
  RELATE, RGP, KumoRFM-2, ConTextTab), hypotheses H1–H6, the doc corpus, risks.
- [implementation.md](implementation.md) — build spec: phases, data contracts, equations, config, test matrix,
  the doc-authoring protocol (Appendix A). **Contracts (shapes, semantics) are normative; exact library
  function names are not** — `relbench`/`pytorch-frame` APIs drift, so adapt to the installed version.

Keep the two files in sync (implementation.md = build plan; idea.md = rationale).

## Naming: HALOS = the method, `gloss` = the package

The spec calls the *method* **HALOS** and writes the package as `halos/`. The *package/repo here is* **`gloss`**.
Wherever the spec writes `halos/`, that maps to **`gloss/`** — keep the `gloss` name; do not rename to `halos`.

## Environment (provisioned)

`.venv` (uv, py3.12) holds the full stack: **torch 2.8.0+cu128** (cxx11abi=True), torch_geometric 2.8.0,
torch_frame 0.3.0, relbench, sentence-transformers, transformers, lightgbm, shap, pytorch-lightning, hydra,
wandb. Use `.venv/bin/python`. The dev node has an **A40 (46 GB)**; bigger jobs go to SLURM
(`gpu`/`priority_gpu`: h100:4 / a40:4; see `scripts/train.sh` for the torchrun launcher). **flash-attn is not
yet built** (build needs `wheel`+`ninja` in the venv, then `sbatch scripts/build_flash_attn.sh`, archs
`86;90`); its only consumer is the **frozen Qwen encoder** — HALOS's own attention uses **SDPA**.

## Working agreement (implementation.md §0 — these govern how you work here)

- **Method-first.** Deliverable = the HALOS encoder + evidence its doc-generated geometry works. RT/RelGT/GNNs
  are **baselines only**; no throwaway prototypes.
- **Docs are realistic prose**, not structured cards: per-DB markdown that reads like a senior dev's README
  (partial coverage ~60–80%, mixed granularity, FK rationale, inline units/codes). The corpus is a deliverable
  on the critical path (authoring protocol in implementation.md Appendix A). **We author docs with Claude Code
  here under the blind protocol** (see below) — no external documented DB.
- **Frozen text encoder, cached.** All span/query embeddings computed offline with **`Qwen/Qwen3-Embedding-4B`**
  (`d_text`≈2560; replaces the spec's MiniLM default; config-swappable). **No LM forward passes in training** —
  gather `d_e` by id.
- **Geometry generator is schema-compiled (v1):** consumes docs + typed structure only, so it runs once per DB
  into a per-(relation/metapath, head) table; recompute it each forward so gradients flow (it is not a frozen
  cache). Content-modulated (v2) is an ablation flag, off by default.
- **Exact invariances get unit tests:** time-rescale invariance and leakage are hard test-matrix items.
- **Self-labels stay.** The seed entity's past task-table rows enter the subgraph as nodes (the dominant
  transfer lever, per RT). Never include the current target row (leakage).
- **Small & reproducible.** ≤ ~30M params; global seed everywhere. After each phase append `PROGRESS.md` and
  commit `feat(phaseN): <summary>`. **At a gate, stop and report.**
- **Test everything you implement** and run the tests; every phase ends green on its tests AND its DoD command.

## Phased build plan (implementation.md §5) — build in order; two explicit gates

0. **Substrate** — `data/graph.py` (RelBench DB → hetero temporal graph; `fk_role_id` from the FK edge),
   leakage-safe temporal sampler, `collate.py` (dense pairwise batch: `Δt`, `T_ctx`, `τ`, metapath ids,
   `temporal_valid` mask). *Reuse `relbench.modeling` — don't hand-roll.*
1. **Doc corpus + grounding** — author `doc_corpus/<db>/docs.md` (Tier-2, blind); `docs/{corpus,grounding,cache}.py`
   with regimes `full | shuffled_spans (placebo) | null`, `d_null` fallback, coverage report.
2. **Node encoder** — `model/column_encoder.py` (dtype encoders + **FiLM by `d_c`**), `model/time_encoding.py`
   (`τ = log(Δt/T_ctx)`, Bochner `φ(τ)`).
3. **Core operator** — `model/bias_generator.py` (**`g_θ`**), `attention.py` (Gaussian-in-τ bias + SDPA),
   `halos.py`, `heads.py`, `eval/geometry_report.py`.
4. **Training** — `train/*`, supervised on 2–3 tasks; baselines hetero-GNN / LightGBM-flattened.
5. **GATE 1 — H1 (mechanism feeds signal):** `full` > `shuffled_spans` > `null` on the same encoder; plus H2
   (doc-generated biases vs free-learned per-relation biases at matched params). Stop & report.
6. **GATE 2 — synthetic ground truth + audit:** `data/synthetic.py` (twin columns disambiguated only in prose;
   `planted_truth`), `audit/*` (CMI, placebo, blind-authoring, paraphrase, Shapley vs attention). Stop & report.
7. **Transfer** — pretrain (masked-attribute + autocomplete); leave-one-DB-out, time-rescale transfer,
   name-shuffle survival.
8. **Paper assets** — factorial ablations, dual-FK case study, the **geometry exhibit** (compiled kernels +
   doc snippets per FK role), coverage-vs-gain curve.

## Architecture cheat-sheet (parts that span multiple files)

**Pipeline:** RelBench DB → hetero temporal graph (rows=nodes, table=node type, PK→FK typed edges labeled by
`fk_role_id`) → leakage-safe temporal neighbor sample per seed → **custom collate → dense per-subgraph batch**
(node-level `[B,N_max,d]` + pairwise `[B,N_max,N_max]` `τ`/metapath/mask/`temporal_valid`).

**Key equations (implementation.md §4):**
- **Doc-conditioned node features:** `x_c = γ(d_c) ⊙ W_v Enc_dtype(v_c) + β(d_c)`;
  `h_u = AttnPool_c(x_c; keys=d_c) + E_type(t) + W_t φ(τ_u)`.
- **Dimensionless time:** `τ_uw = log((|t_u − t_w| + ε)/T_ctx)`, `T_ctx = median nonzero gap in the subgraph`.
  For **exact** scale-equivariance use `eps_mode: relative` → `τ = log(Δt/T_ctx)` for `Δt>0`; zero-gap /
  timeless / no-path pairs route to a learned **structural bias bucket** (`b_head` only), gated by
  `temporal_valid`.
- **Geometry generator (`g_θ`, the core):** `ctx(p) = [pooled d_fkrole(p); d_col of linking keys;
  E_metapath(p); rel(p)]`; `(a_h, μ_h, σ_h, b_h) = g_θ(ctx(p))`, `σ = softplus(·)+sigma_floor`; compile once
  per DB over the path set → `GeometryTable[p,h]`.
- **Attention:** `B_h(u,w) = a_h·exp(−(τ_uw−μ_h)²/(2σ_h²)) + b_h`; `logits_h = (Q_h h_u·K_h h_w)/√d +
  B_h(u,w)`, masked to the sampled subgraph; **SDPA** mem-efficient kernel with additive bias.
- **Audit proxy (validation):** `Î(Y;Doc|V,S) ≈ E_heldout[logloss(model_null_docs) − logloss(model_full_docs)]`.

**Optional inputs behind flags:** `absolute_anchor` (append `log T_ctx` to `ctx` — **breaks** strict
invariance; default false, and the scale-equivariance test must *bite* when it's on); `content_modulated`
(residual `Δμ,Δσ` from `[h_u;h_w]`; default false, v2 ablation).

**Tested invariants / contracts (test matrix, implementation.md §6):**
- **Temporal leakage** — no context/self-label row with `row_time > seed_time`; timeless rows always valid
  (`test_leakage.py`).
- **Scale-equivariance** — logits identical under `t→c·t` (1e-5, float64, default config) and *changed* when
  `absolute_anchor=true` (`test_scale_equivariance.py`).
- **FK-role disambiguation** — two FKs into one table get distinct compiled geometry; role-swap changes preds
  (`test_fk_role.py`). **rel-f1 has no dual-FK-into-one-table → this test uses a hand-built fixture.**
- **Grounding** — `d_null` fallback below threshold; `shuffled_spans` placebo decorrelated + length-matched;
  cache deterministic (`test_grounding.py`).
- **Synthetic separation / audit** — no-doc model bounded below planted ceiling, doc model exceeds it; CMI>0
  planted / ≈0 placebo (`test_synthetic_separation.py`, `test_audit_recovers_planted.py`; Phase 6).

**Tests are hermetic:** unit/invariance tests run on tiny synthetic fixtures in `tests/conftest.py` (no
network); only DoD scripts touch the cached real rel-f1.

**The synthetic generator (Phase 6, implementation.md §5/§7):** `entity`/`event` tables with **twin columns**
(identical distribution + topological role, differing only in one documented fact, e.g. a coded sign),
disambiguated **only in the prose docs**; expose `planted_truth` so the audit scores attribution.

**Config-driven switches (not forks):** `docs.regime ∈ {full, shuffled_spans, null}`;
`geometry.absolute_anchor`, `geometry.content_modulated`; `eps_mode ∈ {relative, ...}`.

## The doc corpus (on the critical path)

RelBench DBs ship without documentation, so the corpus is a deliverable. **We author it with Claude Code under
the blind protocol** (implementation.md Appendix A): the author sees only **schema + sample rows — never task
definitions, labels, or splits** — writes senior-dev-style prose (table overviews, FK rationale, inline
units/nulls/codes, ~20–40% of columns deliberately unmentioned, mixed granularity), and records the blindness
attestation + coverage in `<db>/meta.yaml`. This makes the audit's blind-authoring control real. Tier-0 (adapt
genuine upstream docs) and Tier-2-at-scale are future work; we start Tier-2 for **rel-f1**.

## Commands

Use `.venv/bin/python`. Target entry points (built phase by phase):

```bash
.venv/bin/python scripts/run_finetune.py --dry-run    # Phase 0 DoD: sample a rel-f1 batch, print shapes
.venv/bin/python scripts/build_doc_cache.py           # Phase 1: embed + cache prose-doc grounding (Qwen)
.venv/bin/python scripts/run_geometry_report.py       # Phase 3: render compiled per-FK-role kernels
.venv/bin/python -m pytest tests/                      # the test matrix must stay green
sbatch scripts/build_flash_attn.sh                    # build flash-attn (SLURM; only for the Qwen encoder)
```

First dataset = **rel-f1** (cached at `~/.cache/relbench/rel-f1`; task `driver-dnf`). rel-trial/rel-stack later.

## Non-goals / guardrails

Billion-row scale; training the text encoder; beating SOTA accuracy; framing as "text-as-modality" (RELATE got
there — frame the unit as **documentation-generated *geometry*** + the invariances + the audit). Keep
`shuffled_spans` placebo and blind-authoring as first-class controls. The geometry generator must stay
schema-compiled in v1 (docs + structure only); the content-modulated residual is an ablation, off by default.
Don't reintroduce the dead v2 framing (measurement paper / proxy gate / structured cards).
