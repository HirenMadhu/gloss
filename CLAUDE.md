# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo builds **HALOS** (*Heterogeneous Attention, Language Of Schema*) — but as of the latest spec
revision the deliverable is a **measurement paper**, not a foundation model. The thesis:
**"Names lie, meaning transfers."** Across heterogeneous databases, column *names* are a brittle proxy;
the transferable signal is documented column *meaning* (units, null semantics, coded-value dictionaries,
FK-role descriptions) — and the contribution is to **prove**, not just assert, that the model uses it.

There are three orthogonal contributions; the spec **decouples** them and headlines the audit:
- **C1 — Structured DocCards.** Per-column structured documentation as a first-class, frozen-LM-encoded
  modality, FiLM-fused into cell representations; also fixes RT's dual-foreign-key ambiguity via FK-role
  edges. *Headline mechanism.*
- **C3 — Documentation Sufficiency Audit (DSA).** A model-agnostic information-theoretic + faithfulness
  audit (`Î(Y;Doc|Values,Structure)` + placebo-doc + blind-authoring + Shapley/sufficiency) that
  certifies non-redundant, non-leaking signal. **This is the product and the moat** — build it rigorously.
- **C2 — Scale-equivariant content-addressed temporal kernel.** **DEFERRED to Paper #2 / Appendix E.**
  **Do NOT build this in cycle 1.** It is the most scoop-exposed piece; Paper 1 uses *simple* temporal
  handling (monotone log-Δt decay or RT's normalized scalar).

**The two spec files are normative — read them before doing anything:**
- [idea.md](idea.md) — the rationale: the measurement-paper reframe, prior-work positioning (RT, RelGT,
  RGP, KumoRFM-2, GelGT, **RELATE, ConTextTab**), the calibrated effect-size expectations, the scorecard,
  and why C3 is the durable differentiator.
- [implementation.md](implementation.md) — the build spec: phased plan, data contracts, equations, config,
  test matrix. **Contracts (shapes, semantics) are normative; exact library function names are not** —
  `relbench` v2 / `pytorch-frame` APIs may drift, so adapt to the installed version.

Keep the two files in sync (implementation.md = build plan; idea.md = rationale).

## Naming: HALOS = the method, `gloss` = the package

The spec calls the *method* **HALOS** and writes the package as `halos/`. The *package/repo here is*
**`gloss`**. Wherever the spec writes `halos/` (layout §3, project name §2), that maps to **`gloss/`** —
keep the `gloss` name; do not rename to `halos`.

## Current repo state vs. target (IMPORTANT)

The repo currently contains a **generic PyTorch-Lightning + Hydra + W&B MNIST scaffold** (`train.py`,
`gloss/`, `configs/train.yaml`, `scripts/train.sh`, `tests/test_smoke.py`). This is **leftover starter
harness, not HALOS.** It does **not** match the structure `implementation.md` §3 mandates:

- Spec layout `halos/{data,proxy,model,audit,train,eval,utils,ext}/` → build it under **`gloss/`** (keep
  the package name; the internal subpackages from §3 are new). Note `audit/` is the product and `ext/` is
  the deferred Paper-#2 kernel.
- Spec deps include `relbench`, `pytorch-frame`, `torch_geometric`, `sentence-transformers`, **`lightgbm`**
  (the Phase-2 proxy probe), `shap`; current `pyproject.toml`/`requirements.txt` list torchvision/MNIST
  deps — realign to §2.
- The MNIST `train.py` / `MNISTModule` / `MNISTDataModule` are placeholders to be replaced.

Do not silently build HALOS on top of the MNIST scaffold — replace the placeholders and align deps as you
reach each phase.

## Working agreement (implementation.md §0 — these govern how you work here)

- **Measurement-first, not SOTA.** Success is a clean audit and a *gradient* ("when does documentation
  matter, and how much?"), not a leaderboard win. Do not optimize for raw accuracy.
- **Proxy before transformer.** Phase 2 answers the core question with embeddings + a GBM in ~a day. **Do
  not build the model (Phase 3) until the proxy gate is green or its result is recorded.**
- **The audit is the deliverable.** It must run **model-agnostically** on RT, RelGT, and HALOS. Build it
  early (Phase 5) and keep it rigorous — a sloppy CMI estimator sinks the paper.
- **Two legs.** Synthetic planted-doc data = **existence proof** (validates the estimator). Real RelBench
  DBs = **prevalence** (the weight-bearing result). Never let synthetic carry the headline.
- **Geometry is substrate, not headline.** Minimal RT-style relational backbone + FK-role edges + optional
  typed-metapath hop bias. **No content-addressed temporal kernel in Paper 1** (Appendix E only).
- **Small & reproducible.** ≤ ~30M params; global seed everywhere; **freeze the text encoder and cache its
  outputs to disk** — no LM forward passes in the training loop (gather DocCard embeddings by `col_global_id`).
- **After each phase:** append to `PROGRESS.md` and commit `feat(phaseN): <summary>`. (Not a git repo yet —
  `git init` first.) **At a gate, stop and report** the numbers and the recommended pivot.

## Phased build plan (implementation.md §6) — build in order; two explicit gates

0. **Scaffolding + leakage-safe data** — DB → PK-FK heterogeneous temporal graph + temporal neighbor
   sampler (`row_time <= seed_time`).
1. **DocCards + frozen text cache + synthetic generator** — the three doc regimes (`full`/`name_only`/
   `placebo`) + `blind` flag; planted-ground-truth generator (§7) with a `planted_truth` object.
2. **GATE 1 — the proxy test** (≈1 day, **before any transformer**): flat per-seed features + the column's
   embedding under `full`/`name_only`/`placebo`, fed to a **GBM/MLP** probe, on synthetic + 2 real tasks,
   each also under **name-shuffle**. *Go* if there's a measurable, leakage-controlled doc-over-name effect
   somewhere and synthetic separation holds; *no-go* → pivot to the deferred temporal direction or seek a
   genuinely-documented dataset.
3. **HALOS-minimal model** — RT-style relational attention + FiLM doc fusion + FK-role edges + *simple*
   temporal. FK-role test: swapping `fk_role_id`s changes preds and `name_only` cannot distinguish them.
4. **Training loop (supervised + transfer)** — MTP pretraining with **task-table self-labels** retained;
   name-shuffle survival test.
5. **GATE 2 — the Documentation Sufficiency Audit (THE PRODUCT)** — `cmi.py` (predictive-proxy `Î` + seed
   CIs), `controls.py` (placebo + blind-authoring), `shapley.py` (column/key-path KernelSHAP), `faithfulness.py`
   (deletion/insertion, comprehensiveness/sufficiency, polarity), `runner.py` (run over RT/RelGT/HALOS).
   The audit must be *clean* — estimator validated on synthetic ground truth, CIs reported, value-function
   documented — or fix it before Phase 6.
6. **The measurement (headline result)** — `eval/gradient.py`: effect-size **vs schema-nameability** map,
   plus the FK-role qualitative win. *This is the paper's main figure.*
7. **Ablations** — factorial doc-field toggles, placebo/blind arms, full-attention on/off.
- **Appendix-E phase (DEFERRED, Paper #2):** the scale-equivariant content-addressed kernel — do not start
  in cycle 1.

## Architecture cheat-sheet (the parts that require reading multiple files)

**Cell tokenization** packs value + DocCard + (simple) time + structural-role streams. DocCards are
structured per-column passages (table/column desc, unit, null semantics, coded values, FK-role) encoded
**once** by a frozen `all-MiniLM-L6-v2` and gathered by `col_global_id`. See implementation.md §4.

**Key equations (Paper 1 — kept deliberately simple, §5):**
- FiLM doc→value fusion: `x = γ(doc)⊙Wv(value) + β(doc) + Wd(doc)`.
- Simple temporal bias: `B_time(i,j) = -α_head·log(1 + dt_ij)` (or RT's normalized datetime scalar). **The
  content-addressed Hawkes mixture / Bochner log-Δt features are Appendix E — not here.**
- Optional typed-metapath hop bias `B_hop = HopTable[metapath_id, hop]`.
- Attention logits `= QK/√d + B_time + B_hop`, masked by RT's {column, feature, neighbor, full} masks
  (full attention **off by default**).
- Audit sufficiency proxy: `Î(Y;Doc|V,S) ≈ E_heldout[logloss(model_nodoc) − logloss(model_full)]` with seed CIs.

**Tested invariants / contracts:**
- **Temporal leakage** — a context cell is valid iff `row_time <= seed_time` (`test_leakage.py`).
- **FK-role disambiguation** — two FKs into one table get distinct `fk_role_id`; swapping changes preds and
  `name_only` docs cannot tell them apart (`test_shapes.py` FK-role check / Phase 3).
- **Synthetic separation (DPI)** — a values+structure-only model is bounded below the planted no-doc Bayes
  rate; a doc-using model can exceed it (`test_synthetic_separation.py`). This is what makes the thesis
  *falsifiable*, and it validates the CMI estimator against known ground truth.
- **Audit recovers planted / blind-authoring** — `CMI>0` on planted-doc, `≈0` on placebo; blind cards don't
  leak labels (`test_audit_recovers_planted.py`, `test_blind_authoring.py`).

**The synthetic generator is the crux** (implementation.md §7): `entity`/`event` tables with **twin
columns** (identical distribution + topological role, differing only in one documented fact such as a coded
sign), and a label that depends on the documented sign — not inferable from values or topology. Build it
carefully and expose `planted_truth` so the audit can score attribution precision/recall.

**Ablation switches are config-driven** (not forks): docs `regime ∈ {full, name_only, placebo}` + `blind`;
`temporal.mode ∈ {log_decay, rt_scalar}` (NOT the deferred kernel); `struct_bias.mode ∈ {none, scalar_hop,
typed_metapath}`; `audit.run_on ∈ subset of {rt, relgt, halos}`; `ext_temporal: true` only for the deferred
Paper-#2 kernel.

## The dataset question (answer before Phase 3 — it sets the ceiling)

Per idea.md §11 / implementation.md §10: *do you have a real DB with genuine, messy documentation (data
dictionary, coded-value glossaries, FK notes) not already in RelBench?* If **yes**, the prevalence audit
has a killer testbed. If **no**, you're auditing self-written docs, so the **blind-authoring control
(`audit/controls.py`) becomes the single most important component** — war-game authoring access,
inter-annotator agreement, and pre-registration.

## Commands

The HALOS commands below are **target** entry points from implementation.md §3 — they do not exist yet.
The scaffold commands are what runs today.

```bash
# --- current scaffold (placeholder, to be replaced) ---
python train.py                                  # MNIST scaffold; Hydra overrides via dotlist
pytest tests/                                    # smoke tests (CPU; needs torch installed)
pytest tests/test_smoke.py::test_module_step     # run a single test

# --- target HALOS entry points (implementation.md §3, build these) ---
python scripts/build_text_cache.py               # Phase 1: embed + cache DocCards
python scripts/run_proxy_gate.py                 # Phase 2: GATE 1 — the week-one proxy test
python scripts/run_audit.py                      # Phase 5: GATE 2 — the Documentation Sufficiency Audit
python scripts/run_gradient.py                   # Phase 6: the "when does documentation matter" map
python scripts/run_finetune.py --dry-run         # Phase 0 DoD: sample a batch, print shapes
pytest tests/                                    # the §9 test matrix must stay green
```

Environment: Python ≥3.10 (current `.python-version` pins 3.12), one CUDA GPU (24–48 GB) suffices. **This
shell has no `torch` installed** — the user develops on the Milgram HPC cluster (SLURM; `scripts/train.sh`
shows the `torchrun --nproc_per_node=N` launcher pattern), so run/verify training there, not here.

## Non-goals / guardrails

Billion-row scale; training the text encoder; beating SOTA accuracy; "text-as-modality" framing (RELATE
got there — frame the unit as **documentation *beyond names*** + **the audit**). Default to placebo-doc and
blind-authoring controls as first-class arms — reviewers will suspect documentation leakage, and pre-built
controls turn that objection into the moat. And do **not** be tempted to build the C2 temporal kernel early.
