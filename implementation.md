# HALOS — Implementation Spec for Claude Code (v3: documentation-conditioned geometry)

Build the **node-level geometric encoder directly** — no RT-style cell-token prototype, no interim
measurement study. The method: a heterogeneous temporal graph transformer whose attention geometry
(temporal kernels + structural weights) is **generated from human-style prose documentation** via a
grounding module. Companion rationale: `HALOS_method_design.md`.

Place this at the repo root; reference from `CLAUDE.md`. Build phase by phase; do not skip a phase's
tests or Definition of Done. **Two gates** (Phases 5 and 6): stop and report there.

---

## 0. Working agreement
- **Method-first.** Deliverable = the HALOS encoder + evidence its doc-generated geometry works. No
  throwaway prototypes; RT/RelGT/GNNs are *baselines only*.
- **Docs are realistic prose**, not structured cards: per-DB markdown that reads like a senior dev's
  README (partial coverage, mixed granularity, FK rationale, inline units/codes). The corpus is a
  deliverable on the critical path (Phase 1; authoring protocol in Appendix A).
- **Frozen text encoder, cached.** All span/query embeddings computed offline
  (`sentence-transformers/all-MiniLM-L6-v2` default, swappable). No LM forward passes in training.
- **Geometry generator is schema-compiled in v1**: it consumes docs + typed structure only, so it runs
  once per DB into a per-(relation/metapath, head) parameter table. Content-modulated v2 is an
  ablation flag, off by default.
- **Exact invariances get unit tests**: time-rescale invariance and leakage are hard test-matrix items.
- **Self-labels stay.** Past task-table rows of the seed entity enter the subgraph as nodes.
- **First evidence ablation is H1** (full vs shuffled-span vs null docs) on our own encoder — Phase 5
  gate. ≤ ~30M params; global seeds; log configs. After each phase append `PROGRESS.md`, commit
  `feat(phaseN): …`. `relbench`/`pytorch-frame` APIs may drift: §3 contracts are normative, exact
  call signatures are not.

## 1. Dependencies
Python ≥ 3.10, one 24–48 GB GPU for the prototype.
```toml
dependencies = [
  "torch>=2.4", "torch_geometric>=2.5", "pytorch-frame>=0.2", "relbench>=1.0",
  "sentence-transformers>=3.0", "shap>=0.45",
  "numpy", "pandas", "scikit-learn", "pyyaml", "tqdm", "lightgbm>=4.0",  # lightgbm = baseline only
]
[project.optional-dependencies]
dev = ["pytest", "wandb", "matplotlib"]
```

## 2. Repository layout
```
halos/
  CLAUDE.md  HALOS_IMPLEMENTATION.md  HALOS_method_design.md  PROGRESS.md  pyproject.toml
  doc_corpus/                  # THE CORPUS (deliverable)
    rel-trial/docs.md          # Tier 0: adapted from AACT data dictionaries
    rel-stack/docs.md          # Tier 0: adapted from SEDE community schema docs
    rel-f1/docs.md             # Tier 0/1: Ergast guide + blind-authored gaps
    <db>/meta.yaml             # tier, author, blindness attestation, coverage stats
  configs/{default,rel-trial,synthetic}.yaml
  halos/
    data/
      graph.py                 # RelBench DB -> hetero temporal graph; leakage-safe sampler; self-label nodes
      collate.py               # subgraph -> batch (+ Δt matrix, T_ctx, metapath ids)
      synthetic.py             # planted-truth generator (twin columns; doc-only disambiguation; event_driven flag)
    docs/
      corpus.py                # load/validate doc_corpus; coverage report
      grounding.py             # chunk -> embed -> retrieve -> pool; d_e, rel_e, d_null; placebo regimes
      cache.py                 # offline embedding cache (idempotent)
    model/
      column_encoder.py        # dtype encoders + FiLM(d_col) -> cell vecs -> AttnPool -> node h_u
      time_encoding.py         # tau = log((Δt+eps)/T_ctx); Bochner features of tau
      bias_generator.py        # *** CORE ***  g_theta: ctx(p) -> (a, mu, sigma, b) per head; compile per DB
      attention.py             # hetero graph attention; logits = QK/sqrt(d) + B(tau; compiled params)
      halos.py  heads.py       # encoder stack; task heads + masked-attribute/autocomplete heads
    train/{loop,finetune,pretrain,losses}.py
    audit/                     # validation section (not headline)
      cmi.py                   # Î(Y; Doc|V,S) predictive proxy + seed CIs
      controls.py              # shuffled-span placebo; blind-authoring arms; paraphrase control
      shapley.py  faithfulness.py  readback.py
    eval/{metrics,nameshuffle,transfer,geometry_report}.py
      # geometry_report: render compiled kernels per FK-role with doc snippets (the paper's exhibit)
  scripts/
    build_doc_cache.py  run_finetune.py  run_h1_gate.py  run_audit.py  run_pretrain.py  run_geometry_report.py
  tests/
    test_leakage.py  test_shapes.py
    test_scale_equivariance.py        # EXACT invariance under t -> c*t (default config)
    test_fk_role.py                   # dual-FK schemas get distinct compiled geometry; preds respond to role swap
    test_grounding.py                 # null fallback; placebo decorrelation; cache determinism
    test_synthetic_separation.py      # DPI: no-doc model bounded; doc model exceeds planted ceiling
    test_audit_recovers_planted.py    # CMI>0 planted, ~0 placebo; Shapley beats attention on planted causes
```

## 3. Data contracts (normative)

**3.1 Doc corpus.** One `docs.md` per DB: free prose, senior-dev style (Appendix A). `meta.yaml`:
`{tier: 0|1|2, author, blind: bool, coverage_target: ~0.6-0.8}`. No per-column templates.

**3.2 Grounding outputs** (offline, cached):
```
spans      : chunk(docs.md, 2–4 sentences);  s_k = E_text(span_k)
queries    : q_e = E_text(minimal descriptor)        # "table orders, column ship_date" / FK-role descriptor
d_e        : softmax-topK(cos(q_e, s_k)/T) · s_k   if max cos > thresh else d_null (learned)
rel_e      : max_k cos(q_e, s_k)                     # relevance scalar, kept as feature
regimes    : full | shuffled_spans (placebo: spans permuted across elements/DBs, length-matched) | null
```
Emit a per-DB coverage report (fraction of elements grounded).

**3.3 Graph & batch.** Nodes = rows (node type = table); typed edges = FK links **labeled by FK-role id**
(distinct ids for two FKs into one table); self-label nodes from past task rows. Sampler: per seed
(entity, seed_time) return only rows with `row_time ≤ seed_time` (hard rule). Collate computes:
`Δt_uw` for attendable pairs, `T_ctx = median nonzero Δt in subgraph`, `tau_uw = log((Δt+eps)/T_ctx)`,
`metapath_id(p)` for ≤2-hop typed paths, segment ids for block-diagonal packing.

## 4. Key equations (implement exactly)

**Node features** (`column_encoder.py`):
```
x_c = gamma(d_c) ⊙ W_v Enc_dtype(v_c) + beta(d_c)          # FiLM by grounded column doc
h_u = AttnPool_c(x_c; keys=d_c) + E_type(t) + W_t phi(tau_u)
```

**Geometry generator** (`bias_generator.py`) — the core; per typed relation path p, per head h:
```
ctx(p) = [ pooled d_fkrole(p) ; d_col of linking keys ; E_metapath(p) ; rel(p) ]
(a_h, mu_h, sigma_h, b_h) = g_theta(ctx(p))                # sigma = softplus(.) + sigma_floor
COMPILE: run g_theta once per DB over all relation paths -> GeometryTable[p, h]
```
Optional inputs behind flags: `absolute_anchor: log T_ctx` appended to ctx (breaks strict invariance —
default **false**); `content_modulated: true` adds residual Δmu, Δsigma from [h_u; h_w] (v2 ablation).

**Attention** (`attention.py`):
```
B_h(u,w)      = a_h · exp(−(tau_uw − mu_h)² / (2 sigma_h²)) + b_h        # params from GeometryTable
logits_h(u,w) = (Q_h h_u · K_h h_w)/sqrt(d) + B_h(u,w)                   # masked to sampled subgraph
```

**Audit proxy** (`audit/cmi.py`):
```
Î(Y; Doc | V, S) ≈ E_heldout[ logloss(model_null_docs) − logloss(model_full_docs) ]   # matched seeds, CIs
```

## 5. Phases

**Phase 0 — Substrate.** `data/graph.py`, sampler, collate (Δt, T_ctx, tau, metapaths, self-label
nodes). *Tests:* leakage, shapes. *DoD:* `run_finetune.py --dry-run` samples and prints a batch on
rel-trial.

**Phase 1 — Doc corpus v0 + grounding.** Adapt Tier-0 docs for rel-trial (AACT) and rel-stack (SEDE);
blind-author gaps per Appendix A. Build `docs/{corpus,grounding,cache}.py` + `build_doc_cache.py` with
all three regimes. *Tests:* `test_grounding.py`. *DoD:* coverage reports for 2–3 DBs; cached `d_e`,
`rel_e` load instantly.

**Phase 2 — Node encoder.** `column_encoder.py`, `time_encoding.py`. *DoD:* finite `[N, d_model]`
node states from a real batch; FiLM responds to regime switch (full vs null changes features).

**Phase 3 — Core operator.** `bias_generator.py` (+ per-DB compile step), `attention.py`, `halos.py`.
*Tests:* `test_scale_equivariance.py` — multiply every timestamp by random c>0: logits **identical**
(tolerance 1e-5) under default config, and *changed* when `absolute_anchor=true` (test must bite);
`test_fk_role.py`. *DoD:* end-to-end forward; both invariance tests green; `run_geometry_report.py`
renders compiled kernels per FK role (even untrained — pipeline check).

**Phase 4 — Training.** `train/*`, `heads.py`, `eval/metrics.py`. Supervised on 2–3 tasks
(rel-trial + rel-stack + one event-driven, e.g. rel-f1). Baselines: hetero-GNN, LightGBM-flattened;
RT/RelGT numbers from papers or released code. *Tests:* overfit 256 seeds to ~0 loss. *DoD:*
validation metrics in a sane range vs baselines.

**Phase 5 — GATE 1: H1, mechanism feeds signal.** `run_h1_gate.py`: same architecture, three doc
regimes — `full` vs `shuffled_spans` vs `null` — on the Phase-4 tasks; plus H2 (doc-generated biases
vs free-learned per-relation biases at matched parameter count). *Go* if full > shuffled with CIs and
doc-generated ≥ free-learned anywhere meaningful. *No-go* → grounding is feeding noise: debug
retrieval (chunking, thresholds, queries) before touching the model; if H2 fails in-DB, the
transfer claim (H2-OOD) becomes the load-bearing test in Phase 7. Record either way in `PROGRESS.md`.

**Phase 6 — GATE 2: synthetic ground truth + audit.** `data/synthetic.py`: twin columns identical in
values/topology, disambiguated **only in prose docs** (documented sign/code); planted lag; `event_driven`
flag; expose `planted_truth`. Run `audit/*`: CMI (planted > 0; placebo ≈ 0), blind-authoring arm,
paraphrase control, Shapley vs attention on planted causes, deletion/insertion + polarity. *Gate:* the
audit must be clean (estimator validated on planted truth, CIs, masking documented) — this section is
what makes the method paper defensible. *DoD:* audit table + per-prediction read-back demo.

**Phase 7 — Transfer.** `pretrain.py` (masked-attribute + autocomplete across RelBench v2 + ReDeLEx +
PluRel synthetic; mix synthetic with real), `eval/transfer.py`: leave-one-DB-out; **time-rescale
transfer** (rescale a held-out DB's clock — τ-geometry invariant, dimensioned ablation degrades);
**name-shuffle survival** (H4) vs RT. *DoD:* zero-shot numbers on ≥3 held-out DBs with H2-OOD,
H3, H4 reported.

**Phase 8 — Paper assets.** Factorial ablations {doc regime × generator inputs × content_modulated ×
absolute_anchor × free-learned-bias}; dual-FK case study (H5); the **geometry exhibit** (compiled
kernels + doc snippets per FK role); coverage-vs-gain curve across DBs. *DoD:* tables/figures with
seed variance.

## 6. Test matrix (always green)
| Test | Asserts |
|---|---|
| `test_leakage.py` | no context row with `row_time > seed_time` |
| `test_shapes.py` | end-to-end shapes |
| `test_scale_equivariance.py` | logits exactly invariant under t→ct (default); variant configs change |
| `test_fk_role.py` | dual FKs → distinct compiled geometry; role-swap changes predictions |
| `test_grounding.py` | null fallback works; placebo decorrelated; cache deterministic |
| `test_synthetic_separation.py` | no-doc bounded below planted ceiling; doc model exceeds it |
| `test_audit_recovers_planted.py` | CMI>0 planted / ≈0 placebo; Shapley beats attention |

## 7. Risks & fallbacks
1. **Grounding noise** → relevance gating + d_null; Phase-5 gate catches early; tune chunking/queries
   before model changes.
2. **Generator collapse / dead kernels** (all mass at one τ) → sigma floor, a_h init small, monitor
   compiled-kernel diversity in `geometry_report`.
3. **Compiled (v1) too coarse** → enable `content_modulated` residual; report both.
4. **Corpus authoring leakage** → blind protocol (Appendix A) + placebo + audit; never let an author
   see tasks/labels/splits.
5. **Library drift** → §3 contracts normative; adapt calls.
6. **Pairwise bias memory at scale** → biases only over sampled subgraph pairs; cap `max_nodes`;
   block-diagonal packing.

## 8. Definition of done (project)
Green test matrix; Phase-5 and Phase-6 gates recorded; transfer numbers (leave-one-DB-out,
time-rescale, name-shuffle) on ≥3 held-out DBs; dual-FK case study; the geometry exhibit; clean audit
section; doc corpus published with tiers + coverage. SOTA accuracy explicitly not required — the
claims are mechanism, invariance, and transfer.

---

## Appendix A — Doc authoring protocol (Tier 1)
Author sees: schema (tables, columns, keys, dtypes) + ~50 sample rows per table. Author never sees:
task definitions, labels, splits, or any model output. Write `docs.md` as a senior dev would: an
overview paragraph per table (what it records, when rows are created); FK rationale in prose
("orders reference users twice — the buyer and the assigned courier"); units, null meanings, and coded
values mentioned inline only where a dev naturally would; **leave ~20–40% of columns unmentioned**;
mixed granularity is good; one or two mild staleness notes are realistic. No bullet-per-column
templates. Record author + attestation in `meta.yaml`; a second blind author covers a 20% subset for
agreement stats. Tier-2 (LLM-drafted) follows the same blindness rules and is flagged in `meta.yaml`.