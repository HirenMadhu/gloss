# HALOS / "Names Lie, Meaning Transfers" — Method Design (revised)

> **Framing change (read first).** Three independent reviews converged on the same verdict: the
> all-in-one "HALOS bundle" is a *kill* for a single-cycle paper, and the strong paper is the
> **decoupled documentation + audit** result, written as a **measurement paper**, not a foundation-model
> breakthrough. This document is rewritten accordingly. The fancy temporal-geometry kernel (the
> scale-equivariant, content-addressed Hawkes bias — "C2") is **demoted to a scoped extension / Paper #2**,
> not a co-headline.

---

## TL;DR (revised)
- **The paper is a measurement, framed by one falsifiable claim:** *across heterogeneous databases, column
  names are a brittle proxy; the transferable signal is documented column meaning — units, null semantics,
  coded-value dictionaries, and FK-role descriptions — and we can **prove**, not just assert, that the model
  uses it.*
- **Two co-equal contributions, both defensible:** (C1) **structured DocCards** as a first-class, frozen-LM
  modality that survives name-shuffling and uniquely fixes RT's dual-foreign-key ambiguity; (C3) the
  **Documentation Sufficiency Audit (DSA)** — an information-theoretic + faithfulness audit, with placebo-doc
  and blind-authoring controls, that certifies non-redundant, non-leaking signal. **C3 is the moat.**
- **The headline result is a *gradient*, not a single number.** Documentation helps little on well-named
  benchmark schemas (the field already shows name-semantics is worth only ~1–3% R²/AUROC) and a lot on
  poorly-named / coded / cross-lingual schemas. *Characterizing where and how much is the contribution* —
  which turns the small-effect risk into the scientific result.
- **HALOS-the-model is the supporting act.** It is a deliberately minimal relational transformer (geometry as
  substrate, not headline). The temporal kernel C2 is deferred.
- **Reproducible:** ~22M params, public data (RelBench v2, ReDeLEx, PluRel synthetic), frozen cached text
  embeddings, single-GPU prototype. **Highest-leverage first step is a one-day proxy test, not a model.**

---

## 1. The decision, and why
The original HALOS bundled three orthogonal deltas: documentation (C1), a scale-equivariant content-addressed
temporal kernel (C2), and a faithful attribution/audit layer (C3). Independent critiques agreed:

- **C2 is the most scoopable thing in the project** (~45–60% over ~6 months): GelGT already occupies the
  "learnable Gaussian temporal bias for relational data" lane, and "Hawkes → attention" work is converging.
  Headlining it is strategically wrong.
- **C1 is defensible *as executed* but exposed *as a headline*** (~25–35% scoop): "we bolted descriptions onto
  RT" is a weekend project for any well-resourced lab. What makes C1 safe is the *falsification machinery*
  (structured DocCards + the audit), not the descriptions themselves.
- **C3 is the durable differentiator** (~10–15% scoop): industrial labs racing on leaderboards are
  *structurally disincentivized* to build a faithfulness/sufficiency audit, because it can only make their
  headline numbers look smaller. That asymmetry is the moat.

So the bundling instinct was backwards for defense: the component one is most attached to (C2) is the most
exposed; the unglamorous audit (C3) is the most durable. **Verdict: kill the bundle; write C1 + C3 as a
measurement paper led by C3; defer C2.**

A useful tell: the bundle's natural conclusion ("we improved relational FMs via documentation, a temporal
kernel, and an attribution layer, achieving X%") is the hollow "we-improved-numbers" pattern. The decoupled
conclusion ("names are a brittle proxy; documented meaning is the transferable signal, and here is the audit
that proves it") is sharp. The fact that decoupling sharpens the conclusion *is* the verdict in miniature.

---

## 2. The reframe: a measurement paper
Stop selling "a better foundation model." Sell **"a measurement: when does schema documentation matter for
relational transfer, by how much, and can we prove the model uses meaning rather than leaked names/labels?"**

- The **contrived-vs-real gradient becomes the result**, not a weakness. Synthetic schemas where documentation
  is the *only* disambiguating cue are the **existence proof**; real databases are the **prevalence**
  question. Reporting both — "here is that it *can* work, and here is *how often / where* it does" — is a
  two-legged stool far sturdier than betting the paper on a single effect size.
- This reframe also neutralizes the strongest reviewer objection ("you planted the signal, of course you
  recover it"): the synthetic result is explicitly labeled an existence proof, and the **real-DB audit carries
  the scientific weight**.

---

## 3. Scope: first paper vs deferred extension

| Component | Role in Paper 1 | Status | 6-mo scoop risk |
|---|---|---|---|
| **C1 — Structured DocCards** (units, null-semantics, coded values, FK-role descriptions; FiLM fusion; FK-disambiguation) | **Headline mechanism** | In | 25–35% (safe *as executed*) |
| **C3 — Documentation Sufficiency Audit** (CMI + placebo + blind-authoring + faithfulness, model-agnostic) | **Headline / moat** | In | 10–15% (safest) |
| **Transfer** (MTP pretraining, task-table self-labels, name-shuffle survival) | Evaluation regime for C1/C3 | In | — |
| **Geometry** (relational-attention backbone over PK-FK graph, typed-metapath hop bias, FK-role edges) | **Substrate, not headline** | In (minimal) | — |
| **C2 — Scale-equivariant content-addressed temporal kernel** (log-Δt Bochner features + Hawkes mixture) | — | **Deferred to Paper #2** | 45–60% (most exposed) |

---

## 4. The nugget, abstract, conclusion (calibrated)

**Nugget (one sentence).** Across heterogeneous databases, column names are a brittle proxy; the transferable
signal is documented column *meaning* — units, null semantics, coded-value dictionaries, and FK-role
descriptions — and we can prove, not just assert, that the model uses it.

**Draft abstract.**
1. Relational foundation models transfer to unseen databases by tokenizing schema metadata, but they reduce a
   column's semantics to its name string.
2. Names are brittle: they shuffle, abbreviate, collide, and fail to disambiguate two foreign keys into the
   same table — so name-based transfer degrades exactly where schemas differ most.
3. We introduce structured documentation as a first-class, frozen-LM-encoded modality — DocCards carrying
   units, null semantics, coded-value dictionaries, and foreign-key role descriptions — fused into cell
   representations, and we measure when it recovers transfer that name-shuffling destroys, while uniquely
   resolving the dual-foreign-key ambiguity.
4. Crucially, we validate the mechanism: an information-theoretic sufficiency audit,
   I(Y; Doc | Values, Structure) > 0, with placebo-doc and blind-authoring controls, certifying that
   documentation contributes non-redundant signal rather than leaking correlated names or labels.
5. The result is a map, not a trophy: it tells practitioners *which* databases benefit from documentation
   (poorly-named, coded, cross-lingual) and which do not (well-named, semantically redundant), and gives the
   community an auditable definition of "uses meaning."

> Note the deliberately lowered rhetorical ceiling vs. an earlier draft: not "reframes how all relational FMs
> should be built" (the evidence won't support that on well-named benchmarks), but "names the conditions under
> which documentation is a genuine, auditable modality." The measurement framing survives the small-effect
> reality; the foundation-model framing does not.

**Draft conclusion.** Documented meaning transfers where names do not, and the gain survives controls that rule
out leakage — making documentation a genuine, auditable modality for relational foundation models rather than a
cosmetic feature. The same audit names the conditions under which it does *not* help, turning an empirical risk
into a scientific contribution. The labs racing on benchmarks won't build this audit, because it can only make
their numbers look smaller — which is exactly why it is the flag worth planting.

---

## 5. State of the field (grounding, with the new adjacencies)
- **RT — Relational Transformer (Ranjan, Hudovernik, … Leskovec, arXiv:2510.06377, ICLR 2026).** Cell = token
  (value, column name, table name); frozen LM embeds only `"<column> of <table>"`; relational attention
  (column/feature/neighbor/full); masked-token pretraining; ~22M params. Zero-shot ≈ **90.3% of fully
  supervised AUROC, → 93.1% with continued task-free pretraining**, vs. 83.7% for a 27B LLM. **Ablations that
  define our baseline:** removing **self-labels** drops zero-shot AUROC **70.1%→53.8%** (the dominant transfer
  lever); **shuffling column/table names reduces zero-shot R² only ~2.3%**; **full attention is "surprisingly
  dispensable."** Stated limitations DocCards target: no PK/FK name semantics; cannot disambiguate two FKs into
  the same table.
- **RelGT (Dwivedi et al., arXiv:2505.10960).** Schema-specific, node-level, 5-element tokenization; up to 18%
  over GNNs; already biases attention by hop/time/type.
- **RGP — Relational Graph Perceiver (arXiv:2511.04557).** Time-as-signal; temporal subgraph sampler; Perceiver
  bottleneck; a **multi-task decoder that compares queries to text-encoded labels** — i.e. text-as-modality is
  already entering RDL decoders.
- **KumoRFM-2 (Hudovernik et al., arXiv:2604.12596).** Billion-scale relational FM; in-context learning;
  pretrains across rows/columns/foreign-keys/cross-sample. **Documentation / column meaning is explicitly not
  one of its axes** — our opening.
- **PluRel (Kothapalli et al., arXiv:2602.04029, ICML 2026).** Synthetic multi-table DBs unlock power-law
  scaling; synthetic→continued-pretraining beats real-only; caution re a synthetic-only "lazy-kernel" regime.
- **GelGT (Ding, Li, Wang, Xie, arXiv:2605.15575, May 2026 — *not* the SNAP group).** Learnable Gaussian
  temporal attention bias on sampled subgraphs; "up to 13.8% improvement." Occupies the C2 lane. Its specific
  lemmas/theorems and "beats ALiBi" claims **could not be independently verified**; treat as reported.
- **NEW — ConTextTab (Spinaci, Polewczyk, Schambach, Thelin; SAP Business AI; arXiv:2506.10707, NeurIPS 2025).**
  A **single-table** semantics-aware ICL model that embeds **column headers and categorical cells** with
  `all-MiniLM-L6-v2` and ablates the semantic contribution (their §5.1; SOTA on the semantically-rich CARTE
  benchmark). *Why it matters:* a strong "name/header semantics help, but modestly" baseline already exists.
  The strategy review cites ~1% acc / ~2% R² for dropping name semantics — **verify against their §5.1 before
  citing the number.** It is **single-table, not relational**, so it is an adjacency, not a direct scoop, but
  it pre-empts any claim that name-shuffling is catastrophic on well-curated data.
- **NEW — RELATE (Meyer et al.; SAP; arXiv:2510.19954).** A **schema-agnostic, plug-and-play encoder for
  multimodal *relational* graphs**: shared modality-specific encoders **conditioned on column metadata via a
  shared text embedding model**, Perceiver cross-attention, **within 3% of schema-specific encoders** on
  RelBench. *Why it matters most:* this is the closest prior work — relational *and* text-conditioned — so
  "metadata-as-text for relational data" is **already done**. Our distinction must be explicit: RELATE
  conditions on the **short column-metadata string** (engineering: parameter sharing, schema-agnosticism);
  HALOS conditions on **structured documentation beyond names** (units/nulls/codes/FK-roles) and, decisively,
  ships the **audit that proves the meaning is used**. RELATE strengthens our case that the unclaimed
  contribution is *documentation-beyond-names + the audit*, not "text as a feature."

**Net positioning:** the "text-as-modality for relational data" camp now has at least RGP (decoder), RELATE
(encoder), and the broader tabular-semantics line (ConTextTab/CARTE/TabSTAR). HALOS does **not** claim to
invent text-as-modality. It claims (a) *structured documentation beyond names* as the right unit, (b) the
*FK-role disambiguation* RT explicitly cannot do, and (c) the *audit* nobody is incentivized to build.

---

## 6. The model: HALOS-minimal (supporting act)
A deliberately small relational transformer — just enough to be a credible object to audit. **No C2.**
- **Geometry (substrate).** RT-style relational attention over the PK-FK graph (column/feature/neighbor masks;
  full attention off by default per RT). Optional **typed-metapath hop bias** keyed on relation-type sequences
  (schema-portable). **FK-role edge features** so two FKs into the same table are distinct — the RT-stated
  limitation, fixed here as a structural feature.
- **Text (headline mechanism).** **DocCards**: per-column structured passages (table/column descriptions,
  units, null semantics, coded-value dictionaries, FK-role descriptions), rendered by a fixed template, encoded
  **once** by a frozen `all-MiniLM-L6-v2`, cached, and gathered by column id. Fused into the cell via **FiLM
  gating** (documentation modulates value): `x = γ(doc)⊙Wv(value) + β(doc) + Wd(doc)`. Three regimes for the
  audit: `full`, `name_only` (RT-style), `placebo` (length-matched, semantically null); plus a `blind` flag.
- **Transfer.** Masked-token pretraining across schemas with **task-table prompting retained** so self-labels
  (the dominant lever) are present.
- **Temporal handling (intentionally simple in Paper 1).** A plain monotonic log-Δt decay (or RT's normalized
  scalar). The expressive scale-equivariant content-addressed kernel is **Paper #2** (§12). Keeping it simple
  removes a make-or-break surface from the first paper.

---

## 7. The audit (DSA) — the centerpiece
A **model-agnostic** procedure run on RT, RelGT, and HALOS alike. This is the product.

- **Information-theoretic sufficiency.** Estimate `Î(Y; Doc | Values, Structure)` via a predictive proxy:
  train matched models with and without documentation and report
  `Î ≈ E[ logloss_nodoc − logloss_full ]` on held-out data, with seed CIs. `Î > 0` ⇒ documentation adds
  non-redundant signal; `Î ≈ 0` on placebo docs by the data-processing inequality.
- **Leakage controls (what makes it credible).** **Placebo-doc** (length-matched, irrelevant) and
  **blind-authoring** (cards written without access to labels/task), reported as first-class arms. If the
  effect survives placebo and blind authoring, it is meaning, not leaked names or annotator foresight.
- **Faithfulness (not attention).** Predictions are attributed by **Shapley over columns / key-paths** (not
  attention weights, per Jain & Wallace / Serrano & Smith / Wiegreffe & Pinter), validated by
  deletion/insertion AUC (temporally masked), ERASER comprehensiveness/sufficiency, and polarity consistency.
- **Two legs.** **Existence** on synthetic schemas with planted documented meaning (DPI gives a provable
  separation: any values+structure-only model is bounded below Bayes; a doc model can exceed it). **Prevalence**
  on real RelBench v2 DBs (the honest, weight-bearing measurement).

---

## 8. The four required properties — all still present
The reframe does not drop any of the four desiderata; it changes which ones *headline*.
- **Geometric** — yes: relational-attention backbone over the PK-FK graph + typed-metapath hop bias + FK-role
  edges. (Substrate. The *novel* geometric contribution, C2, is deferred — but the property is satisfied.)
- **Text** — yes, and headlining: structured DocCards beyond names, FiLM-fused.
- **Transferable** — yes, and headlining: name-shuffle-survival + MTP/self-labels; the "meaning transfers where
  names don't" claim is exactly a transfer claim.
- **Interpretable** — yes, and the moat: the faithful Shapley/sufficiency audit, not attention heatmaps.

---

## 9. Calibrated expectations (be honest in the paper)
- On **well-named** RelBench DBs: documentation likely buys **~1–3% R²/AUROC** (consistent with RT's ~2.3%
  name-shuffle and ConTextTab's name-semantics ablation). Do not over-claim here.
- On **poorly-named / abbreviated / coded / cross-lingual** schemas, and on **dual-FK** tasks: expected to be
  **large**. This is where the story lives.
- Therefore the deliverable is a **gradient / map** ("effect size vs schema-nameability"), with the FK-role
  result as a clean qualitative win independent of effect size.

---

## 10. Competitive landscape & the moat
- **Who is nearby:** Stanford/Kumo (RT, RelGT, RGP, KumoRFM-2); the GelGT camp (temporal bias); the SAP camp
  (ConTextTab, RELATE — text-conditioned encoders).
- **Where you cannot win:** raw scale or speed (Kumo); generic "text-as-modality for relational" (RELATE got
  there).
- **Your unfair advantage (structural, not technical):** the **audit**. A measurement/benchmark artifact that
  the racing labs are disincentivized to produce, plus the **FK-role disambiguation** RT explicitly cannot do.
- **The real ceiling-setter (answer honestly, §11):** access to genuinely-documented relational data.

---

## 11. The dataset question that sets the ceiling
**Do you have a real database with genuine, messy documentation — a data dictionary, coded-value glossaries,
FK notes — that is *not* already in RelBench?**
- **If yes:** the prevalence audit has a killer testbed, the cross-lingual/coded regime is real, and several of
  the scores below move up a point.
- **If no:** you are auditing documentation you wrote yourself, and the **blind-authoring control becomes the
  single most important thing in the paper** — war-game it before committing (who authors, what they can see,
  inter-annotator agreement, pre-registration).

---

## 12. Deferred extension (Paper #2): the temporal kernel C2
Keep, but do not headline now. The idea: a **scale-equivariant** temporal attention bias over **log Δt**
(Bochner functional-time features; invariant to global time rescaling by Buckingham-π reasoning), with a
**content-addressed mixture of Gaussians** (Hawkes-style, per-entity, event-driven) replacing GelGT's single
dimensioned center. Strong as a standalone follow-up; **45–60% scoop-exposed**, so ship it *after* the
measurement paper establishes the documentation/audit flag. (Full equations live in the implementation spec's
Extension appendix.)

---

## 13. Scorecard (consensus rescore)
For the **decoupled C1+C3 measurement paper** (the bundle scores lower — ~5.5 — because three nuggets is no
nugget). Three independent scorings were reconciled; these are deliberately base-rate-calibrated, not
optimistic.

| # | Criterion | Score | One-line |
|---|---|:---:|---|
| 1 | Novelty | 6.5 | Idea is mediumish (RELATE/RGP nearby); the *audit + FK-role fix + falsification* is the unclaimed wedge. |
| 2 | Impact | 6 | Small on well-named benchmarks; Medium-High as a *measurement* ("when does meaning beat names?"). |
| 3 | Timing | 8 | RT just landed naming these gaps; KumoRFM-2 omits docs; RelBench v2 leaves names un-harmonized. ~6-mo window. |
| 4 | Feasibility | 7 | Riskiest assumption shifted from "do docs help" to "can the real-DB CMI audit be executed cleanly." Week-one testable. |
| 5 | Competition | 4.5 | Crowded (Kumo + GelGT + SAP/RELATE). Moat is structural (audit), not technical. Private documented data would lift this. |
| 6 | Nugget | 8 | "Names lie, meaning transfers" — sharp, falsifiable. |
| 7 | Narrative | 7 | "Forgotten semantic layer / Rosetta Stone" — strong, but hostage to effect size; the audit rescues it. |

**Average ≈ 6.7.** Every soft score traces to one fact: documentation's effect on standard benchmarks is
small. That is why the week-one proxy test is the highest-leverage action, and why the audit (which converts
"small effect" into "a map of when the effect exists") carries the paper.

---

## 14. Risks & caveats
1. **Docs redundant on real DBs (`Î ≈ 0`).** Mitigation: the measurement *is* the result; lead with the
   gradient; FK-role win stands regardless.
2. **Real-DB CMI audit is fragile.** A sloppy estimator sinks the paper. Mitigation: validate the estimator
   against synthetic planted ground truth first; report CIs and the masking/value-function explicitly.
3. **RELATE/ConTextTab pre-emption.** Mitigation: be explicit that the unit is *documentation beyond names* and
   the contribution is the *audit*; do not frame as "text-as-modality."
4. **Synthetic "you planted it" objection.** Mitigation: synthetic = existence only; real-DB audit = weight.
5. **No private documented dataset.** Mitigation: blind-authoring control becomes load-bearing (§11).
6. **GelGT specifics unverified; RelBench v2 / KumoRFM-2 / PluRel postdate indexes.** Treat fine-grained numbers
   as provisional; the ConTextTab ~1–2% figure is from the strategy review — verify against §5.1.

---

### Companion
The build plan (phases, interfaces, tests, gates) is in `HALOS_IMPLEMENTATION.md`. This file is the rationale;
that file is how to build it. Keep them in sync.