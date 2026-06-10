# HALOS — Implementation Spec for Claude Code (revised: measurement-first)

*"Names lie, meaning transfers."* This build produces a **measurement paper**: structured schema
**documentation (DocCards)** as a modality, plus a model-agnostic **Documentation Sufficiency Audit (DSA)**
that proves the model uses meaning rather than leaked names/labels. The model (HALOS-minimal) is the
supporting act; **the audit is the product**.

Place this at the repo root (reference it from `CLAUDE.md`). Build **phase by phase**. The cheapest, highest-
leverage step (Phase 2, the proxy gate) comes *before* any transformer. The fancy temporal kernel is **deferred
to an Extension appendix** — do not build it in the first cycle.

---

## 0. Working agreement (read first)
- **Measurement-first, not SOTA.** Success is a clean audit and a *gradient* ("when does documentation matter,
  and how much?"), not a leaderboard win. Do not optimize for raw accuracy.
- **Proxy before transformer.** Phase 2 answers the core question with embeddings + a GBM in ~a day. Do not
  build the model (Phase 3) until the proxy gate is green or its result is recorded.
- **The audit is the deliverable.** It must run **model-agnostically** on RT, RelGT, and HALOS. Build it early
  (Phase 5), keep it rigorous; a sloppy CMI estimator sinks the paper.
- **Two legs.** Synthetic planted-doc data = **existence proof** (validates the estimator). Real RelBench DBs =
  **prevalence** (the weight-bearing result). Never let synthetic carry the headline.
- **Geometry is substrate.** A minimal RT-style relational backbone + FK-role edges + optional typed-metapath
  hop bias. **No content-addressed temporal kernel in Paper 1** (see Appendix E).
- **Keep it small/reproducible.** ≤ ~30M params; global seed everywhere; freeze the text encoder and **cache
  its outputs to disk** (no LM forward passes in the training loop).
- **After each phase**, append to `PROGRESS.md` and commit `feat(phaseN): …`. **At a gate, stop and report.**
- **APIs may drift** (`relbench` v2, `pytorch-frame`): the §4 *contracts* are normative; exact function names
  are not — adapt to the installed version.

---

## 1. What we build (scope)

**In (Paper 1):**
- **DocCards** — structured per-column documentation (units, null semantics, coded values, FK-role
  descriptions), frozen-LM-encoded, FiLM-fused.
- **DSA** — `Î(Y; Doc | Values, Structure)` + placebo + blind-authoring + faithfulness (Shapley/sufficiency),
  run on RT / RelGT / HALOS.
- **HALOS-minimal** — RT-style relational transformer, FK-role disambiguation, *simple* temporal handling.
- **Transfer regime** — MTP pretraining + task-table self-labels; the name-shuffle-survival test.

**Deferred (Paper #2, Appendix E):** scale-equivariant, content-addressed Hawkes temporal kernel ("C2").

**Out:** billion-row scale; training the text encoder; beating SOTA.

---

## 2. Environment & dependencies
Python ≥ 3.10, one CUDA GPU (24–48 GB plenty).

```toml
# pyproject.toml (core)
dependencies = [
  "torch>=2.4", "torch_geometric>=2.5", "pytorch-frame>=0.2",
  "relbench>=1.0", "sentence-transformers>=3.0",
  "lightgbm>=4.0",            # the Phase-2 proxy probe
  "shap>=0.45",               # or implement KernelSHAP (Phase 5)
  "scikit-learn", "numpy", "pandas", "pyyaml", "tqdm",
]
[project.optional-dependencies]
dev = ["pytest", "wandb", "matplotlib"]
```
Default frozen encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — matches RT and ConTextTab/RELATE;
swappable via config.

---

## 3. Repository layout

```
halos/
  CLAUDE.md
  HALOS_IMPLEMENTATION.md          # this file
  HALOS_method_design.md           # rationale (companion)
  PROGRESS.md
  pyproject.toml
  configs/{default,rel-stack,synthetic}.yaml
  halos/
    data/
      relbench_graph.py    # DB -> heterogeneous temporal graph + leakage-safe temporal sampler
      doccards.py          # DocCard schema + template renderer + regimes (full/name_only/placebo, blind)
      text_cache.py        # frozen text-embedding cache (per-DB, per-column)
      synthetic.py         # planted-ground-truth generator (existence proof; validates the audit)
      collate.py           # subgraph -> TokenBatch (+ pairwise geometry)
    proxy/
      embed_probe.py       # PHASE 2: DocCard-emb vs name-emb vs placebo -> GBM/MLP; the week-one gate
    model/
      tokenizer.py         # cell tokenization (value/doc/time/struct)
      fusion.py            # DocFiLMFusion (documentation modulates value)
      time_simple.py       # SIMPLE temporal handling for Paper 1 (monotone log-Δt decay / scalar)
      biases.py            # TypedMetapathHopBias (optional); NO content-addressed kernel here
      attention.py         # RelationalAttention (masks + additive bias)
      halos.py             # HALOS-minimal
      heads.py             # task heads + masked-token-prediction head
    audit/                 # ***THE PRODUCT***
      cmi.py               # Î(Y; Doc | Values, Structure) predictive-proxy estimator + CIs
      controls.py          # placebo-doc + blind-authoring orchestration
      shapley.py           # column / key-path KernelSHAP with relational masking
      faithfulness.py      # deletion/insertion (temporally masked), comprehensiveness/sufficiency, polarity
      runner.py            # run the audit model-agnostically over {RT, RelGT, HALOS}
      readback.py          # render top attributions through DocCards
    train/{loop,finetune,pretrain,losses}.py
    eval/
      gradient.py          # PHASE 6: effect-size vs schema-nameability "map" (the headline result)
      nameshuffle.py       # name-shuffle survival test
      metrics.py
    utils/{seeding,config,logging,flops}.py
    ext/                   # DEFERRED (Paper #2)
      time_scaleequiv.py   # Appendix E: Bochner log-Δt + content-addressed Hawkes mixture
  scripts/
    build_text_cache.py
    run_proxy_gate.py      # PHASE 2 entry point (the first thing to run)
    run_audit.py           # PHASE 5 entry point
    run_gradient.py        # PHASE 6 entry point
    run_finetune.py
    run_pretrain.py
  tests/
    test_leakage.py  test_shapes.py
    test_synthetic_separation.py     # DPI: no-doc bounded; doc exceeds
    test_audit_recovers_planted.py   # CMI>0 on planted; ~0 on placebo
    test_blind_authoring.py          # blind cards cannot leak labels
    test_faithfulness.py             # Shapley beats attention on planted-cause recovery
```

---

## 4. Core data contracts (normative)

### 4.1 DocCard (the text modality)
Structured per-column passage — **not** RT's `"<col> of <table>"` string.
```python
@dataclass
class DocCard:
    table: str; table_desc: str
    column: str; column_desc: str
    dtype: str                  # numeric|categorical|text|datetime|bool|id
    unit: str | None            # "USD","days","count"
    null_semantics: str | None  # "NULL = not yet shipped"
    coded_values: dict | None   # {0:"active",1:"churned",2:"suspended"}
    fk_role: str | None         # "buyer_id -> users.id: the user who PLACED this order"
    fk_target: str | None       # "users.id"
```
`doccards.py` renders one card per (table,column) via a fixed template. `text_cache.py` embeds each card **once**
with the frozen encoder → `doc_emb[col_global_id] : [d_text]`, cached to disk; train-time only gathers by id.

**Regimes (selectable per run):** `full` | `name_only` (RT-style) | `placebo` (length-matched, semantically
null). **Flag:** `blind` (authored without seeing labels/task). These power the audit.

### 4.2 TokenBatch (model input)
Pack `B` sampled subgraphs (one per labeled seed); block-diagonal with a `seg_id` so attention never crosses
subgraphs (padded `[B,T_max]` acceptable for the first prototype). Per-cell tensors (T = total cells):
```
value_input    # type-specific (pytorch-frame) or precomputed [T, d_val]
col_global_id  # [T] -> doc_emb + per-column stats
node_type_id   # [T]
fk_role_id     # [T]  (0 = not an FK cell)   <-- fixes RT dual-FK ambiguity
row_id row_time seed_time table_id seg_id is_self_label
```
Pairwise (only where cells can attend): `hop_ij`, `metapath_id_ij`, `dt_ij = |row_time_i - row_time_j|`,
`attn_mask_kind ∈ {column,feature,neighbor,full}`.

**Leakage rule:** a cell is valid in context iff `row_time <= seed_time`.

### 4.3 Attention masks (RT semantics, reused)
- **column**: same column across rows. **feature**: same row + F→P parent rows. **neighbor**: P→F child rows.
- **full**: optional, **off by default** (RT: full attention dispensable).

---

## 5. Key equations (Paper 1 — keep simple)
**FiLM doc-value fusion** (`fusion.py`):
```
g = gamma(doc_emb); b = beta(doc_emb)            # small MLPs -> [d_model]
x_cell = g * Wv(value_emb) + b + Wd(doc_emb)
```
**Simple temporal handling** (`time_simple.py`) — a monotone decay, *not* the deferred kernel:
```
B_time(i,j) = -alpha_head * log(1 + dt_ij)        # alpha_head >= 0 (softplus), per head
# (or just feed RT's normalized datetime scalar; both are fine for Paper 1)
```
**Typed-metapath hop bias** (optional, `biases.py`):
```
B_hop(i,j) = HopTable[metapath_id_ij, hop_ij]     # learned scalar per (typed-path, distance), per head
```
**Attention logits** (`attention.py`):
```
logits(i,j) = (Q_i·K_j)/sqrt(d) + B_time(i,j) + B_hop(i,j)   # masked by attn_mask_kind
```
**Audit — sufficiency proxy** (`audit/cmi.py`):
```
Î(Y; Doc | Values, Structure) ≈ E_heldout[ logloss(model_nodoc) - logloss(model_full) ]   # report with seed CIs
# model_nodoc uses regime=name_only or placebo; model_full uses regime=full
```
> The content-addressed Hawkes mixture and scale-equivariant log-Δt features are **Appendix E**, not here.

---

## 6. Phased build plan
Each phase: **Objective → Build → Tests → Definition of Done (DoD)**. Two gates are explicit.

### Phase 0 — Scaffolding + leakage-safe data
- **Build.** `utils/*`; `data/relbench_graph.py` (load a RelBench dataset+task; build PK-FK temporal graph;
  temporal neighbor sampler returning only rows with `row_time<=seed_time`).
- **Tests.** `test_leakage.py`, `test_shapes.py`.
- **DoD.** `run_finetune.py --dry-run` samples a batch and prints shapes.

### Phase 1 — DocCards + frozen text cache + synthetic generator
- **Build.** `data/doccards.py` (dataclass, template, regimes `full/name_only/placebo`, `blind` flag);
  `data/text_cache.py` (embed once, cache, idempotent); `data/synthetic.py` (planted-doc generator, §7);
  `scripts/build_text_cache.py`.
- **Tests.** Cache deterministic; placebo length-matched but uncorrelated; gather → `[T, d_text]`.
- **DoD.** Text cache builds once; synthetic DB + `planted_truth` object generate.

### Phase 2 — GATE 1: the proxy test (≈ one day, **before any transformer**)
- **Objective.** Answer the core question cheaply: does documented meaning beat names, by how much, and where?
- **Build.** `proxy/embed_probe.py` + `scripts/run_proxy_gate.py`: for each task, build a flat feature matrix
  per seed (aggregated cell values + structure features), then **append the column's embedding** under three
  regimes — `doc_emb(full)`, `name_emb(name_only)`, `placebo` — and train a **GBM/MLP** probe. Run on:
  (a) the **synthetic planted-doc** task, (b) **2 real tasks** (e.g. rel-stack `user-engagement`, rel-f1
  `driver-dnf`), each also under **name-shuffle** (shuffle names; keep DocCards).
- **Decision (record either way):**
  - On synthetic: `full` ≫ `placebo` (meaning recoverable) — sanity that the pipeline can detect signal.
  - On real: report the *gradient* — Δ(full − name_only) and survival under name-shuffle.
  - **Go** (build the model) if there is a measurable, leakage-controlled doc-over-name effect somewhere and
    the synthetic separation holds. **No-go / pivot** if `full ≈ name_only ≈ placebo` everywhere → the
    documentation thesis is weak on available data; **pivot to the deferred temporal direction (Appendix E)**
    or seek a genuinely-documented dataset (§ dataset question) before investing in the transformer.
- **DoD.** A one-page proxy result table (regime × task × shuffle) committed to `PROGRESS.md`.

### Phase 3 — HALOS-minimal model
- **Build.** `model/{time_simple,biases,attention,tokenizer,fusion,halos,heads}.py`. RT-style relational
  attention + FiLM doc fusion + FK-role edges + simple temporal. **No content-addressed kernel.**
- **Tests.** `test_shapes.py` end-to-end; FK-role: in a synthetic 2-FK schema, swapping `fk_role_id`s changes
  predictions and `name_only` cannot distinguish the two FKs.
- **DoD.** Forward pass runs; overfits a 256-seed subset to ~0 train loss.

### Phase 4 — Training loop (supervised + transfer)
- **Build.** `train/{loop,finetune,pretrain,losses}.py`; `eval/metrics.py`; `model/heads.py` MTP head;
  `eval/nameshuffle.py`. Pretrain via masked-token prediction with **task-table self-labels** retained.
- **Tests.** Pretraining loss decreases; zero-shot beats a no-pretraining control on ≥1 held-out DB.
- **DoD.** A fine-tuned single-task number vs a LightGBM-on-flattened baseline; a first name-shuffle number.

### Phase 5 — GATE 2: the Documentation Sufficiency Audit (***the product***)
- **Build.** `audit/cmi.py` (proxy estimator + seed CIs); `audit/controls.py` (placebo + blind-authoring runs);
  `audit/shapley.py` (column/key-path KernelSHAP; relational masking value function = column-mean/`[MASK]`
  baseline; amortize with a learned head, validated vs exact Shapley on small contexts);
  `audit/faithfulness.py` (deletion/insertion AUC respecting temporal masks; comprehensiveness/sufficiency;
  polarity); `audit/runner.py` (run over {RT, RelGT, HALOS}); `audit/readback.py`.
- **Tests.** `test_audit_recovers_planted.py` (CMI>0 on planted-doc; ≈0 on placebo); `test_blind_authoring.py`
  (blind cards cannot leak labels — CMI under blind ≈ CMI under informed for genuinely-semantic signal, and
  placebo stays ≈0); `test_faithfulness.py` (Shapley beats attention on planted-cause recovery; attention
  violates polarity on ≥1 task).
- **GATE.** The audit must be *clean*: estimator validated on synthetic ground truth, CIs reported, masking/
  value-function documented. If it cannot be made clean, fix it before Phase 6 — a sloppy audit is fatal.
- **DoD.** For any prediction: a faithful top-k attribution + a DocCard read-back; `Î` with CIs on synthetic
  (separated) and on ≥2 real DBs.

### Phase 6 — The measurement (the headline result)
- **Build.** `eval/gradient.py` + `scripts/run_gradient.py`: characterize **effect size vs schema-nameability**.
  Order DBs/tasks by a nameability proxy (e.g., header informativeness / coded-column fraction / language), and
  plot Δ(full − name_only) and name-shuffle survival across them. Include the **FK-role** qualitative win.
- **DoD.** The "when does documentation matter?" map across RelBench v2 (+ any private documented DB), with the
  synthetic existence proof and the real-DB prevalence audit side by side. **This is the paper's main figure.**

### Phase 7 — Ablations & (optional) efficiency
- **Build.** `eval/` factorial toggles {names, descriptions, units, null-semantics, coded-values, FK-roles};
  placebo + blind arms; full-attention on/off. Optional accuracy-vs-FLOPs.
- **DoD.** Factorial table with seed variance.

### Appendix-E phase (DEFERRED — Paper #2, do **not** start in cycle 1)
The scale-equivariant content-addressed temporal kernel. See Appendix E for equations and tests. ~45–60%
scoop-exposed; build only after the measurement paper plants the flag.

---

## 7. Synthetic generator contract (`data/synthetic.py`) — existence proof + audit validation
Normative, because it both proves the mechanism *can* work and validates the CMI estimator against known truth.

- Tables `entity` and `event` (FK `event.entity_id -> entity.id`), timestamped; inter-event times from a known
  distribution (planted lag structure).
- **Twin columns** `col_A`, `col_B` in `event`: identical distribution, identical topological role; their
  DocCards differ in **one documented fact** (e.g., a coded sign: `col_A` "higher = worse", `col_B` "higher =
  better").
- **Label:** `y = sigmoid( alpha * s_A * f(history of col_A) + noise )`, where `s_A ∈ {+1,-1}` is documented in
  `col_A`'s card and **not** inferable from values or topology.
- **Expose `planted_truth`** (which column, sign, lag) so `audit/` can score attribution precision/recall.
- Knobs: #entities/#events, twin noise, `alpha`, lag window, `s_A`, and an `event_driven` flag (label depends
  on time-since-last-trigger) for later use by the Appendix-E kernel.

By the **data-processing inequality**, any values+structure-only model is bounded below the no-doc Bayes rate;
a doc-using model can exceed it ⇒ a provable separation (existence). Real-DB CMI then answers prevalence.

---

## 8. Config (example)
```yaml
# configs/rel-stack.yaml
seed: 0
data: {dataset: rel-stack, task: user-engagement,
       sampler: {num_neighbors: [12,12], max_cells: 4096, time_attr: row_time}}
docs:  {regime: full, blind: false, encoder: sentence-transformers/all-MiniLM-L6-v2}
model: {d_model: 256, n_heads: 8, n_layers: 12,
        masks: [column, feature, neighbor],          # full off
        struct_bias: {mode: typed_metapath},         # none|scalar_hop|typed_metapath
        temporal: {mode: log_decay},                 # log_decay|rt_scalar   (NOT the deferred kernel)
        fusion: film}
audit: {estimator: predictive_cmi, seeds: 5,
        controls: [placebo, blind],
        faithfulness: [deletion, insertion, comprehensiveness, sufficiency, polarity],
        run_on: [rt, relgt, halos]}
train: {lr: 3.0e-4, weight_decay: 0.01, batch_seeds: 64, max_steps: 50000, amp: true}
ext_temporal: false       # set true ONLY for the deferred Paper-#2 kernel (Appendix E)
```

---

## 9. Test matrix (must stay green)
| Test | Asserts |
|---|---|
| `test_leakage.py` | no context cell has `row_time > seed_time` |
| `test_shapes.py` | tokenizer/model produce correct shapes end-to-end |
| `test_synthetic_separation.py` | no-doc model bounded by planted no-doc ceiling; doc model exceeds it |
| `test_audit_recovers_planted.py` | CMI > 0 on planted-doc; ≈ 0 on placebo |
| `test_blind_authoring.py` | blind cards don't leak labels (placebo stays ≈0; semantic signal survives) |
| `test_faithfulness.py` | Shapley/sufficiency beats attention on planted-cause recovery; polarity violation shown |

---

## 10. The dataset question (decides the ceiling — answer before Phase 3)
**Do you have a real DB with genuine, messy documentation (data dictionary, coded-value glossaries, FK notes)
not already in RelBench?**
- **Yes** → the prevalence audit has a killer testbed; cross-lingual/coded regimes are real.
- **No** → you're auditing self-written docs; **blind-authoring (`audit/controls.py`) becomes the most
  important component** — war-game authoring access, inter-annotator agreement, pre-registration.

---

## 11. Risks & fallbacks
1. **Docs redundant on real DBs (`Î≈0`).** The measurement is the result; lead with the gradient; FK-role win
   stands regardless.
2. **CMI estimator fragile.** Validate on synthetic first; report CIs; document masking/value-function. *Fatal
   if sloppy* — Phase 5 is a hard gate.
3. **RELATE/ConTextTab pre-emption.** Frame the unit as *documentation beyond names* + the *audit*, never as
   "text-as-modality."
4. **`relbench`/`pytorch-frame` drift.** §4 contracts are normative; adapt calls.
5. **Shapley too slow.** Group into columns/key-paths; amortize; validate vs exact on small contexts.
6. **Tempted to build C2 early.** Don't — it's the most scoop-exposed and a make-or-break surface; Appendix E,
   Paper #2.

---

## 12. Definition of done (Paper 1)
Green test matrix; a recorded Phase-2 proxy gate; HALOS-minimal trained with a name-shuffle number; a **clean
DSA** (estimator validated on synthetic, CIs, placebo + blind arms) run on RT/RelGT/HALOS; the Phase-6
**"when does documentation matter" map** (synthetic existence + real prevalence); FK-role qualitative win; a
factorial ablation table. **SOTA accuracy is explicitly not required.**

---

## Appendix E — DEFERRED temporal kernel (Paper #2; do not build in cycle 1)
Scale-equivariant, content-addressed temporal bias. Kept here for completeness only.
```
# Bochner functional-time features of log-Δt (scale-equivariant: dt->c*dt is a constant shift)
phi(dt)_k = sqrt(2/D) * cos(w_k * log(dt + eps) + b_k)            # w_k, b_k learnable

# Content-addressed mixture of Gaussians over log-Δt (Hawkes-style, per-entity), per head:
(w_m, mu_m, raw_sigma_m) = MLP([h_i ; h_j]); sigma_m = softplus(raw_sigma_m) + sigma_floor
B_time(i,j) = sum_m w_m * exp( -(log(dt_ij+eps) - mu_m)^2 / (2 sigma_m^2) ) + b_head
```
Deferred-phase tests (only if/when pursued): `test_scale_equivariance.py` (rescale all timestamps by random
`c>0`; with log-Δt the model's logits are invariant within tolerance; with raw-Δt they change). Compare against
`single_gaussian_global` (GelGT-like) and `scalar_decay` baselines. Scoop risk ~45–60%; ship after Paper 1.