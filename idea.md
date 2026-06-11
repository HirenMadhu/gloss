# HALOS — Method Design (v3: documentation-conditioned geometry)

**This is a method paper.** HALOS is a node-level, geometry-aware temporal graph transformer for
relational databases whose attention geometry — *which past rows matter, at what time lag, through
which typed link* — is **generated from human-written schema documentation**. The audit machinery from
earlier drafts survives as a *validation section*, not the headline. There is no RT-prototype stage:
we build the geometric encoder directly.

---

## 1. Thesis

> A relational schema has **accidental** parts (column names, absolute timescales, node identities)
> and **invariant** parts (documented meaning, dimensionless time, typed relations). Only invariants
> transfer. HALOS encodes only invariants — and uses one signal, documented meaning, to do double
> duty: it conditions the **node features** and it **generates the attention geometry**.

Concretely: a column whose documentation says *"renewal_date — when the customer's annual
subscription renews"* should induce temporal attention centered around a year-scale lag; *"last_login"*
should induce recency; a foreign key documented as *"the user who placed the order"* should induce a
different structural weight than *"the user who fulfilled it."* Today every relational model either
ignores this (RT reduces semantics to a name string; RelGT/GelGT learn per-dataset geometric constants
that cannot transfer) or treats text as a mere node feature (RELATE, RGP). HALOS makes meaning the
*generator* of geometry. Because both the text-embedding space and the dimensionless temporal
coordinate are schema-agnostic, the generated geometry transfers to unseen databases.

**Documentation is realistic, not curated.** The text modality is what a senior developer actually
writes — README/wiki prose with partial coverage, mixed granularity, table-level overviews, inline
mentions of units and coded values, occasional staleness. This forces a *grounding* module (retrieve
relevant spans per schema element) rather than a template, makes "works with documentation as it
exists" the claim, and turns doc coverage/quality into evaluation axes instead of assumptions.

---

## 2. Architecture

### 2.1 Graph substrate (node-level, geometric)
- Rows = nodes; tables = node types; PK→FK links = typed edges; every row timestamped.
- Per prediction seed (entity, seed_time): temporal neighbor sampling returns a bounded heterogeneous
  subgraph containing only rows with `row_time ≤ seed_time` (hard leakage rule).
- **Self-labels retained.** The seed entity's past task-table rows are included as nodes linked to the
  seed (task-table prompting). RT showed this is the dominant transfer lever (zero-shot AUROC
  70.1→53.8 without it); a node-level design must not lose it.

### 2.2 Schema grounding from prose (the text pipeline)
Input: one markdown document per database (per-DB doc corpus, §5). Offline, cached:
1. **Chunk** the prose into spans (~2–4 sentences).
2. **Embed** spans with a frozen sentence encoder; embed a minimal query per schema element
   (each table, column, and FK role): e.g. `"table orders, column ship_date"`.
3. **Retrieve & pool**: for element *e*, take top-K spans by cosine, softmax-weight, and pool to a
   doc embedding `d_e`; keep the max-similarity as a **relevance scalar** `rel_e`.
4. **Null embedding**: if no span clears a threshold, `d_e ← d_null` (learned). Undocumented elements
   degrade gracefully toward a name-only (RT-like) regime instead of breaking.

Output: `{d_table, d_col, d_fkrole}` per schema element + relevance/coverage statistics. All static
per database — computed once, cached, gathered by id at train time.

### 2.3 Documentation-conditioned node features
For row u of table t, each cell (column c, value v) is encoded by a dtype-specific encoder and
**FiLM-modulated by its grounded doc embedding**:
```
x_c = γ(d_c) ⊙ W_v Enc_dtype(v) + β(d_c)
h_u = AttnPool_c( x_c ; keys = d_c )  +  E_type(t)  +  W_t φ(τ_u)
```
so the *interpretation* of a value is conditional on its documented meaning (a "3" in a severity
column coded high reads differently from a "3" in quantity).

### 2.4 The core operator: documentation-generated attention geometry
Between query node u and key node w, connected by typed relation path p (FK role r at one hop), the
pre-softmax bias is **generated, per head, from meaning + typed structure + dimensionless time**:
```
ctx(p)            = [ pooled d_fkrole(p) ; d_col anchors of the linking keys ; E_metapath(p) ; rel(p) ]
(a_h, μ_h, σ_h, b_h) = g_θ( ctx(p) )                       # small MLP; σ = softplus + floor
B_h(u,w)          = a_h · exp( −(τ_uw − μ_h)² / 2σ_h² ) + b_h
logits_h(u,w)     = (Q_h h_u · K_h h_w)/√d  +  B_h(u,w)
```
with the **dimensionless temporal coordinate**
```
τ_uw = log( (|t_u − t_w| + ε) / T_ctx ),   T_ctx = median nonzero gap in the sampled subgraph.
```

Three properties make this the contribution:
- **Schema-compiled geometry (v1).** `ctx(p)` depends only on docs + typed structure — not node
  content — so the generator runs **once per database**, compiling its documentation into a geometry
  table (per relation/metapath, per head). Zero per-token cost; gradients still flow into g_θ.
  A content-modulated residual (μ, σ shifted by [h_u; h_w]) is the v2 extension/ablation.
- **Exact scale-equivariance.** Rescaling all timestamps t→ct cancels in Δt/T_ctx, so τ — and the
  logits — are *invariant by construction* (Buckingham-π: relate dimensionless groups only; Bochner
  features of τ for the node-level time encoding). This is a unit test, not a hope. (Design tension,
  flagged: docs may express *absolute* intents — "annual" — which a strictly relative coordinate can't
  pin; an optional `log T_ctx` input to g_θ restores absolute anchoring at the cost of strict
  invariance. Default off; ablate.)
- **FK-role disambiguation for free.** Two FKs into the same table get different `d_fkrole`, hence
  different generated geometry — fixing the limitation RT explicitly states it cannot handle.

### 2.5 Transfer
Masked-attribute + temporal-autocomplete pretraining across many databases (RelBench v2 + ReDeLEx +
PluRel synthetic; mix synthetic with real to avoid the lazy-kernel regime), self-labels retained.
The transfer argument is structural: every input to the geometry generator (text embeddings, metapath
types, τ) lives in a schema-agnostic space; nothing dimensioned or identity-bound is encoded.

### 2.6 Interpretability (two layers, validation not headline)
- **The geometry is itself a readable artifact.** Because geometry is compiled from docs per relation,
  you can *plot* what the model decided: "links through `placed_by` matter at τ≈recency; links through
  `renewal` matter at τ≈+5.9 (~1 year for this DB)." No attention-faithfulness leap needed — these are
  the actual biases applied.
- **Faithful attribution + sufficiency audit** (kept from v2 as a rigor section): column/key-path
  Shapley with deletion/insertion and polarity checks (never raw attention, per Jain & Wallace /
  Wiegreffe & Pinter), and the conditional-information test Î(Y; Doc | Values, Structure) > 0 with
  **shuffled-span placebo** and **blind-authoring** controls — proving the generator routes *meaning*,
  not leaked names or annotator foresight.

---

## 3. Four properties, one mechanism
| Property | Where it lives |
|---|---|
| **Geometric** | node-level temporal graph transformer; typed-metapath + Gaussian-in-τ biases — *generated*, not hand-set |
| **Text** | realistic prose docs → grounding module → conditions features **and** generates geometry |
| **Transferable** | only invariants encoded (meaning-space, τ, types); self-labels; multi-DB pretraining |
| **Interpretable** | compiled geometry is directly readable; Shapley + CMI audit validates it |

---

## 4. Positioning (what is genuinely new)
- **RT (2510.06377)**: cell-level, no PE, time = normalized scalar, semantics = name string, cannot
  disambiguate dual FKs. HALOS: node-level geometry generated from prose docs; dual-FK solved; time
  dimensionless. RT becomes a *baseline*, not a base.
- **RelGT (2505.10960) / GelGT (2605.15575, partly unverified)**: hop/time/type biases exist but are
  per-dataset learned constants (GelGT: a single dimensioned Gaussian center) — they do not transfer
  and are not semantic. HALOS's biases are functions of documentation in a shared text space and of a
  dimensionless coordinate.
- **RELATE (2510.19954, SAP)**: closest prior — conditions shared *feature encoders* on column-metadata
  text. HALOS conditions the **attention geometry** (temporal kernels + structural weights) on
  *documentation beyond names*, grounded from realistic prose. Different object being conditioned;
  state this contrast explicitly in the paper.
- **RGP (2511.04557)**: text-encoded *labels* in the decoder; time-as-signal via sampling. No
  doc-conditioned geometry.
- **KumoRFM-2 (2604.12596)**: no documentation axis at all.
- **ConTextTab (2506.10707)**: single-table header semantics; adjacent, pre-empts "name semantics are
  huge" claims (~1–2% there) — which is precisely why HALOS routes meaning into geometry, where the
  effect is structural rather than a feature-level nudge.

**The novel claims:** (i) geometry *generated from* grounded prose documentation (schema-compiled,
content-extensible); (ii) exact scale-equivariant relational time via the dimensionless τ; (iii)
grounding-from-realistic-docs as the text interface (no one requires curated cards); (iv) the audit
proving the mechanism routes meaning. Recombined with credit: relational graph substrate (RDL/RelGT),
masked pretraining + self-labels (RT), Bochner time (Xu et al. 2019), Shapley/ERASER machinery.

---

## 5. Doc corpus (now on the critical path)
RelBench databases ship without documentation; the corpus is a deliverable.
- **Tier 0 — genuine upstream docs (best evidence):** rel-trial ← AACT/ClinicalTrials.gov data
  dictionaries (real, messy, external); rel-stack ← community-documented Stack Exchange schema;
  rel-f1 ← Ergast user guide; rel-hm / rel-avito ← Kaggle data descriptions. Adapt, don't rewrite.
- **Tier 1 — blind-authored:** for undocumented DBs, a human writes senior-dev-style markdown seeing
  only schema + sample rows, **never task definitions, labels, or splits**; pre-registered protocol,
  second annotator for agreement on a subset.
- **Tier 2 — LLM-drafted, human-edited** (flagged, same blindness rules), for scale.
- **Style guide:** prose paragraphs; table-level overviews; FK rationale ("orders reference users twice:
  the buyer and the assigned courier"); units/null semantics/coded values mentioned inline where a dev
  would; deliberately partial coverage (~60–80% of columns); no per-column bullet templates.
- Report per-DB **coverage** (fraction of elements grounded above threshold) and use it as an axis.

---

## 6. Experimental plan & hypotheses
Baselines: heterogeneous GNN (RDL), RelGT, RGP, RT (zero-shot + fine-tuned), LightGBM-flattened;
GelGT if reproducible. Datasets: RelBench v2 (start rel-trial + rel-stack + rel-f1 — two with Tier-0
docs, one event-driven), ReDeLEx/PluRel for pretraining, planted synthetic for ground truth.

- **H1 (mechanism feeds signal — first ablation, gate):** full docs > shuffled-span placebo > null-doc
  on the *same trained encoder*; if full ≈ shuffled, the grounding pipeline is feeding noise.
- **H2 (geometry from meaning beats learned constants):** doc-generated biases > free-learned
  per-relation biases (same parameter count) in-DB, and ≫ them on leave-one-DB-out.
- **H3 (scale-equivariance):** logits exactly invariant under global time rescale (unit test); on
  cross-DB transfer with mismatched timescales, τ-based geometry > dimensioned variants.
- **H4 (name-shuffle survival):** shuffle column names, keep docs: HALOS retains performance where RT
  loses; docs grounded by meaning, not lexical overlap (paraphrase control).
- **H5 (dual-FK):** on schemas with two FKs into one table, FK-role-conditioned geometry wins; no
  effect elsewhere.
- **H6 (audit):** Î(Y; Doc | V, S) > 0 on planted synthetic and on Tier-0 DBs; ≈ 0 under placebo;
  survives blind authoring. Shapley recovers planted causes better than attention.
- **Readable-geometry exhibit:** the compiled kernels per FK role, with doc snippets — the figure that
  sells the method.

**Calibrated expectations:** modest deltas on well-named, weakly-temporal benchmarks; the wins should
concentrate where the mechanism has leverage — poorly-named/coded/cross-lingual schemas, dual-FK
structure, event-driven timing, and cross-DB transfer with mismatched timescales. Say so in the paper.

## 7. Risks
1. **Grounding noise** (prose retrieval mis-binds spans → garbage geometry). Mitigate: relevance
   gating + null fallback; H1 gate catches it early; report coverage.
2. **Generator memorizes pretraining timescales** via the optional absolute anchor. Default strict
   invariance; ablate the anchor; Lipschitz-bound g_θ if needed.
3. **Docs add little on clean benchmarks.** Expected; the claim is structural + transfer + regimes
   (§6 calibration), with Tier-0 rel-trial as the flagship.
4. **Corpus authoring leakage.** Blind protocol + placebo + the audit; this is why the audit section
   stays in a method paper.
5. **GelGT/KumoRFM-2/RelBench-v2 details postdate indexes** — verify numbers at writing time.

---
*Companion:* `implementation.md` (build plan). Keep in sync.