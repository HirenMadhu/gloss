# DOC-RT — Method Design (v1: documentation-conditioned cell encoding)

**This is a method paper built on top of RT (Relational Transformer, 2510.06377).** We keep RT's
cell-token substrate and relational attention masks unchanged, and add **one new signal**: per-column
**documentation**, grounded from prose and injected by **FiLM** into the cell encoders, so the
*interpretation of a value is conditioned on its column's documented meaning*. No temporal kernels, no
geometry generator, no dimensionless-time coordinate — those belonged to the earlier (GelGT-based)
design and are retired. One signal, one mechanism, one question.

> **Naming note.** The old name (HALOS) referred to Gaussian-in-time "halo" kernels that no longer
> exist here. Working handle is **DOC-RT** until we pick a real name; rename freely.

---

## 1. Thesis

> RT reduces a column's semantics to its **name string**. But a schema column carries meaning a name
> cannot: what an opaque **code** means (`statusId = 4` → "mechanical retirement"), what a sentinel
> value means (`grid = 0` → "pit-lane start, not pole"), what **unit** a number is in, whether a table
> is a time-varying **fact** or a static **dimension**. This meaning lives in **documentation**, not in
> the schema. DOC-RT grounds that documentation per column and uses it to **condition how each cell is
> encoded** — so a value is read in light of what its column *means*, not just what it's *named*.

**Why this is the right place to inject text (and not a feature nudge).** Names + FK topology — which
RT already has — tell the model the wiring and the lexical hint. Documentation earns its place only by
supplying what those don't: **the meaning of opaque codes, sentinels, units, and entity-vs-event
roles.** That information is genuinely absent from the schema, so conditioning the cell encoder on it
is not re-deriving something RT has — it is adding a signal RT structurally cannot access.

**Documentation is realistic, not curated.** A coding agent authors per-DB markdown that reads like a
senior developer's README — partial coverage, mixed granularity, inline mentions of units and coded
values, FK rationale, occasional staleness (authoring protocol: `DOC_AUTHORING.md`). This forces a
**grounding** module (retrieve relevant spans per column) rather than a rigid template, and makes
"works with documentation as a developer would write it" the claim. Uncovered columns fall back to a
learned null embedding — i.e. **degrade gracefully to RT's name-only regime** rather than break.

---

## 2. The honest status of each claim (read before building)

This conversation established a hard prior: in prior work, **schema-documentation prose gave no
measurable improvement** once names were kept. DOC-RT is a bet that *routing* documentation into the
cell encoder — where it can disambiguate codes/units/sentinels — beats names-only, even though
documentation-as-a-plain-feature did not. **That bet is unproven and partially contradicted.** So the
contribution is not "we added documentation"; it is a **finding**, and it stands or falls on one
comparison:

> **docs-on vs docs-off, same codebase, same everything** (full grounded doc embedding vs the learned
> null embedding for every column). This is the load-bearing experiment. It is baked into the first
> training run (`IMPLEMENTATION.md`, Phase 4) as the **default result**, not a later ablation, because
> "DOC-RT beats RT-from-the-paper" is *not* the same comparison — that delta could be our hierarchy
> implementation, sampler, or hyperparameters rather than documentation. Only docs-on vs docs-off, in
> one codebase, isolates the signal.

If docs-on does not beat docs-off, there is no documentation paper; the honest output is a negative
result (and the hierarchy/operator directions become the fallback). We find that out in the **second
training run**, before writing a word.

---

## 3. Architecture

### 3.1 Substrate (RT, unchanged)
- Cell-level tokens; relational attention masks (same-column / same-row / parent-FK / child-FK); no
  positional encoding; names embedded as strings (kept — we *add* to RT, we don't remove its inputs).
- Rows = graph nodes for sampling; per seed `(entity, seed_time)`, temporal sampling returns only rows
  with `row_time ≤ seed_time` (**hard leakage rule**).
- **Self-labels retained.** The seed entity's past task-table rows enter the subgraph as cells/rows.
  RT showed this is the dominant transfer lever (zero-shot AUROC 70.1 → 53.8 without it); we must not
  lose it. It is held in **every** arm so it never confounds the docs comparison.

### 3.2 Grounding from prose (the text pipeline, offline + cached)
Input: one `docs.md` per DB (authored by a coding agent; `DOC_AUTHORING.md`). Offline, cached:
1. **Chunk** prose into spans (~2–4 sentences).
2. **Embed** spans with a **frozen** sentence encoder; embed a minimal query per **column**:
   `"table <t>, column <c>"` (and per FK-role: `"FK <c> of <t> referencing <t'>"`).
3. **Retrieve & pool**: top-K spans by cosine, softmax-weight, pool → column-doc embedding `d_c`; keep
   max cosine as a **relevance scalar** `rel_c`.
4. **Null fallback**: if no span clears a threshold, `d_c ← d_null` (learned). This is exactly the
   docs-off regime, applied per-column.

Output: `{d_c}` per column (+ `rel_c`, coverage stats). **Static per DB** — computed once, cached,
gathered by id at train time. **No LM forward passes during training.**

Three regimes (for the ablation, all on the same trained encoder):
`full` (grounded `d_c`) | `null` (every `d_c ← d_null` — the docs-off baseline) | `shuffled` (placebo:
spans permuted across columns/DBs, length-matched — catches "any text vector helps" artifacts).

### 3.3 Documentation-conditioned cell encoding (the mechanism — FiLM)
For row `u`, column `c`, value `v`, dtype encoder `Enc_dtype`:
```
x_{u,c} = γ(d_c) ⊙ W_v Enc_dtype(v)  +  β(d_c)          # FiLM: column-doc modulates the cell
```
- `γ, β` are small shared MLPs `R^{dim(d)} → R^{d_model}` (FiLM scale/shift).
- A "3" in a severity column documented as coded-high is read differently from a "3" in a quantity
  column — this is the entire point. Coded columns (`statusId`) and sentinels (`grid = 0`) are where
  this should bite hardest.

Cell tokens `{x_{u,c}}` then flow through RT's transformer with its relational masks, unchanged. Row
representation is pooled from its cells as in RT. **Hierarchy = RT's cell→row structure; documentation
conditions the bottom (cell) level.**

**Optional second injection (flag, default OFF):** add `d_c` as an additive feature to the
same-column attention bias, so cell↔cell attention can use documented meaning. Keep OFF by default so
the mechanism stays single-point and ablatable; turn on only as a labeled ablation.

### 3.4 What is deliberately NOT here
- **No τ / no temporal kernel / no scale-equivariance.** Removed. Time enters only as whatever RT
  already uses (it is not part of the documentation claim).
- **No geometry generator / no Gaussian-in-time bias.** Removed.
- **No counting/cardinality operator yet.** Explicitly deferred to a v2 axis (`§6`), to be added
  *after* the documentation finding is settled — not entangled with it now.

---

## 4. Positioning (what is and isn't new)

- **RT (2510.06377):** cell tokens, relational masks, names-as-strings, self-labels, masked-cell
  pretraining. RT is the **base** here (we keep its substrate) *and* the **baseline** (names-only =
  our `null` regime). The hierarchy is RT's — **not claimed as novel.**
- **RELATE (2510.19954):** conditions shared *feature encoders* on column-metadata **text** — the
  closest prior, and the one to separate from explicitly. RELATE conditions on **names/metadata**;
  DOC-RT conditions on **grounded prose documentation beyond names** (coded-value/unit/sentinel/role
  meaning), retrieved from realistic docs. Same *kind* of mechanism (text→encoder conditioning),
  **different signal** (documentation, not metadata). This contrast is the paper's main novelty
  boundary — state it sharply, and benchmark against a RELATE-style names-only conditioning arm.
- **ConTextTab (2506.10707):** single-table header semantics; pre-empts "name semantics are huge"
  (~1–2%). This is *why* DOC-RT targets documented meaning (codes/units), not header names — the
  signal must come from what names *don't* carry, or there is no paper.
- **RT's stated limitation — dual FKs into one table:** documentation gives each FK-role a distinct
  `d`, hence distinct cell-encoding conditioning. **Audit whether any RelBench DB actually has
  same-table dual FKs before claiming this** — rel-f1's driver vs constructor are *different tables*
  (type-distinguished for free), so they are **not** an example of this case. If no RelBench DB has
  it, drop the claim; it has no test bed.

**The candidate novel claim (singular, contingent):** *grounded prose documentation, injected into
cell encoding, recovers signal that names cannot — measured as docs-on > docs-off on coded/unit-heavy
RelBench tasks, with a shuffled-span placebo and a names-only (RELATE-style) control.* Everything else
(hierarchy, masks, self-labels, masked pretraining) is RT, credited.

---

## 5. Experimental plan & hypotheses (RelBench only, always)

Start **single-DB on rel-f1** (it has genuine upstream docs via Ergast, and coded columns —
`statusId` — plus sentinels — `grid = 0` — which is exactly where documentation should help). Add
rel-trial / rel-stack once their docs are authored.

Baselines/arms (all share substrate, self-labels, hyperparameters — only the doc signal changes):
- **RT / names-only** = the `null` regime (docs off).
- **DOC-RT** = the `full` regime (docs on).
- **placebo** = `shuffled` regime.
- **RELATE-style control** = condition on column *name* embedding instead of grounded doc.

- **H1 — docs feed signal (THE gate, first result):** `full` > `null` with seed CIs on rel-f1. If
  `full ≈ null`, stop — documentation adds nothing here.
- **H1b — it's meaning, not any-text:** `full` > `shuffled`. If `full ≈ shuffled`, the win is a text-
  vector artifact, not documented meaning.
- **H1c — beyond names:** `full` > **RELATE-style names-only**. This is the claim's real bar — docs
  must beat *names*, not just beat *nothing*.
- **H2 — coded-column locality:** the gain concentrates on tasks/columns where docs encode hidden
  meaning (`statusId`, `grid`), and is ~flat elsewhere. A *uniform* gain is a red flag (it suggests a
  generic effect, not documentation) — report the per-task breakdown, not just the average.
- **H3 — coverage curve:** performance vs documentation coverage (fraction of columns grounded);
  graceful degradation toward `null` as coverage drops.
- **H4 (defer):** transfer / leave-one-DB-out — only meaningful once multiple DBs have docs; not in
  the first build.

**Calibrated expectation (say it in the paper):** modest or no delta on well-named, code-free tasks;
wins concentrate where columns are coded/unit-bearing/sentinel-laden. rel-f1's `statusId` is the
flagship probe.

---

## 6. Deferred axis — the operator (v2, do not build yet)

The cardinality-conditioned multiset aggregator (counting/sum — the provable representational edge RT
lacks, since attention pooling is a convex combination and cannot recover counts) is a **separate**
axis. Add it **after** the documentation finding is settled, as a second contribution, with its own
isolated ablation. Do **not** entangle it with documentation now — two unproven axes built together
produce an unablatable system and a "two incremental ideas stapled together" reject. One axis, proven,
first.

---

## 7. Risks
1. **Docs add nothing beyond names** (the prior says this is likely). Caught in run 2 (H1/H1c). Output
   = honest negative result; fall back to hierarchy/operator axes. This is *why* the ablation is the
   default, not an afterthought.
2. **Grounding noise** (retrieval mis-binds spans → garbage `d_c`). Relevance gating + null fallback;
   H1b placebo catches "any vector helps."
3. **Uniform gain** (docs help everywhere equally) → suspect a confound, not documentation; H2 per-
   task breakdown adjudicates.
4. **Dual-FK claim with no test bed** → audit RelBench for same-table dual FKs before claiming §4's
   FK-role point.
5. **Authoring leakage** (agent sees tasks/labels) → blind authoring protocol (`DOC_AUTHORING.md`):
   the doc author sees schema + sample rows only, never tasks/labels/splits.

---
*Companion:* `IMPLEMENTATION.md` (build plan), `DOC_AUTHORING.md` (coding-agent doc protocol).