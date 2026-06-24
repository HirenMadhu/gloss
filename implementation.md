# DOC-RT — Implementation Spec for Claude Code (v1)

Build **RT's cell-token substrate + documentation-conditioned cell encoding (FiLM)**. No temporal
kernel, no geometry generator, no τ. The method, the rationale, and the honest-status caveats are in
`METHOD_DESIGN.md`; the doc-authoring protocol is in `DOC_AUTHORING.md`. Place this at repo root;
reference from `CLAUDE.md`. Build phase by phase; do not skip a phase's tests or Definition of Done.

**The one rule that overrides convenience:** the first training result (Phase 4) is **docs-on vs
docs-off in this codebase**, reported side by side, before any other experiment. This is two flags on
one run, costs one extra (unsupervised) training pass, and is the only comparison that isolates
documentation. Everything else waits behind it.

---

## 0. Working agreement
- **Build on RT, don't reinvent it.** If a released RT implementation is usable, wrap it and inject
  documentation at the cell encoder. If not, implement RT's cell-token + relational-mask substrate
  faithfully (same-column / same-row / parent-FK / child-FK masks, no PE, names-as-strings,
  self-labels). RT/RelGT/GNN/LightGBM are **baselines only**.
- **One signal, one mechanism.** Documentation enters at exactly one point by default: FiLM on the
  cell encoder (`γ(d_c), β(d_c)`). The optional same-column attention-bias injection is a flag,
  **default OFF**.
- **Frozen text encoder, cached.** All span/query embeddings offline
  (`sentence-transformers/all-MiniLM-L6-v2` default, swappable). No LM forward passes in training.
- **Docs grounding has three regimes**: `full` | `null` (docs off — every `d_c = d_null`) |
  `shuffled` (placebo). `null` is the docs-off baseline and must be a single config flag.
- **Self-labels stay, in every arm**, so they never confound the docs comparison.
- **Leakage is a hard test-matrix item.** ≤ ~30M params; global seeds; log every config. Append
  `PROGRESS.md` after each phase; commit `feat(phaseN): …`. `relbench` / `pytorch-frame` APIs may
  drift — §3 contracts are normative, exact call signatures are not.
- **RelBench only, always.** Start rel-f1.

---

## 1. Dependencies
Python ≥ 3.10, one 24–48 GB GPU.
```toml
dependencies = [
  "torch>=2.4", "torch_geometric>=2.5", "pytorch-frame>=0.2", "relbench>=1.0",
  "sentence-transformers>=3.0",
  "numpy", "pandas", "scikit-learn", "pyyaml", "tqdm", "lightgbm>=4.0",  # lightgbm = baseline only
]
[project.optional-dependencies]
dev = ["pytest", "wandb", "matplotlib"]
```

## 2. Repository layout
```
docrt/
  CLAUDE.md  IMPLEMENTATION.md  METHOD_DESIGN.md  DOC_AUTHORING.md  PROGRESS.md  pyproject.toml
  doc_corpus/
    rel-f1/docs.md             # authored by the doc agent (DOC_AUTHORING.md); Ergast-derived
    rel-f1/meta.yaml           # tier, author=agent, blind: true, coverage stats
    <db>/docs.md               # add more DBs here later — drop-in
  configs/{default,rel-f1}.yaml
  docrt/
    data/
      graph.py                 # RelBench DB -> hetero temporal graph; leakage-safe sampler; self-label cells
      collate.py               # subgraph -> RT batch (cell tokens, relational masks, segment ids)
    docs/
      corpus.py                # load/validate doc_corpus; coverage report
      grounding.py             # chunk -> embed -> retrieve -> pool; d_c, rel_c, d_null; full/null/shuffled regimes
      cache.py                 # offline embedding cache (idempotent)
    model/
      column_encoder.py        # *** CORE ***  dtype encoders + FiLM(d_c) -> cell vectors
      rt_substrate.py          # RT cell-token transformer + relational masks (wrap released RT if available)
      docrt.py  heads.py       # encoder stack; task heads (+ masked-cell head for later pretraining)
    train/{loop,losses}.py
    eval/
      metrics.py
      ablation.py              # *** the headline runner: full vs null vs shuffled vs name-only, one call ***
      coverage_curve.py        # performance vs doc coverage
  scripts/
    build_doc_cache.py         # author-independent: embeds whatever docs.md exists, all regimes
    run_train.py               # trains ONE arm; --doc_regime {full,null,shuffled,name_only}
    run_headline.py            # *** trains full AND null (AND shuffled, name_only) and prints the table ***
  tests/
    test_leakage.py  test_shapes.py
    test_grounding.py          # null fallback; placebo decorrelation; cache determinism
    test_film_responds.py      # switching regime full<->null changes cell vectors (mechanism is wired)
    test_selflabels_constant.py# self-label nodes identical across doc regimes (no confound)
```

## 3. Data contracts (normative)

**3.1 Doc corpus.** One `docs.md` per DB, free prose, senior-dev style (`DOC_AUTHORING.md`).
`meta.yaml`: `{tier, author, blind: bool, coverage_target: ~0.6-0.8}`. No per-column templates.

**3.2 Grounding outputs** (offline, cached):
```
spans     : chunk(docs.md, 2–4 sentences);  s_k = E_text(span_k)
queries   : q_c = E_text("table <t>, column <c>")            # + FK-role descriptors
d_c       : softmax-topK(cos(q_c, s_k)/T) · s_k    if max cos > thresh   else d_null (learned)
rel_c     : max_k cos(q_c, s_k)                               # relevance scalar, kept as feature
regimes   : full | null (ALL d_c := d_null) | shuffled (spans permuted across cols/DBs, length-matched)
```
Emit a per-DB coverage report. **`null` regime = docs off**, the baseline.

**3.3 Graph & batch.** Cells = tokens (RT style); rows carry node type = table; relational masks =
same-column / same-row / parent-FK / child-FK; self-label cells from past task rows. Sampler: per seed
`(entity, seed_time)` return only rows with `row_time ≤ seed_time` (**hard rule**).

## 4. Key equations (implement exactly)

**Documentation-conditioned cell encoding** (`column_encoder.py`) — the only new mechanism:
```
e_{u,c}  = W_v Enc_dtype(v_{u,c})                            # RT-style dtype cell embedding
x_{u,c}  = γ(d_c) ⊙ e_{u,c} + β(d_c)                         # FiLM by grounded column doc
           # null regime: d_c = d_null  =>  recovers a names-only RT cell (the baseline)
```
`γ, β`: shared MLPs `dim(d) → d_model`. Cells `{x_{u,c}}` → `rt_substrate.py` (masks unchanged) →
row/seed representation → `heads.py`.

**Optional (flag, default false)** same-column attention-bias injection:
```
bias_samecol(c, c') += MLP([d_c ; d_{c'}])                  # only when --docbias=true ; labeled ablation
```

## 5. Phases

**Phase 0 — Substrate.** `data/graph.py`, leakage-safe sampler, `collate.py`, `rt_substrate.py`
(wrap released RT if available, else implement masks). *Tests:* `test_leakage.py`, `test_shapes.py`.
*DoD:* `run_train.py --dry-run` prints an RT batch on rel-f1 and a forward pass runs.

**Phase 1 — Doc corpus + grounding.** Ensure `doc_corpus/rel-f1/docs.md` exists (the doc agent
produces it per `DOC_AUTHORING.md`; if absent, Phase 1 still builds the pipeline and runs in `null`).
Build `docs/{corpus,grounding,cache}.py` + `build_doc_cache.py` with all three regimes. *Tests:*
`test_grounding.py`. *DoD:* coverage report for rel-f1; cached `d_c`, `rel_c` load instantly; `null`
and `shuffled` regimes produce the expected degenerate/permuted embeddings.

**Phase 2 — Cell encoder (core).** `column_encoder.py` (FiLM), wire into `docrt.py`. *Tests:*
`test_film_responds.py` (full vs null **changes** cell vectors — mechanism wired),
`test_selflabels_constant.py` (self-label cells **identical** across regimes — no confound). *DoD:*
finite `[N_cells, d_model]` from a real rel-f1 batch in all three regimes.

**Phase 3 — Training.** `train/*`, `heads.py`, `eval/metrics.py`. Supervised on rel-f1 entity tasks.
Baselines wired: LightGBM-flattened; RT numbers from paper/released code; **names-only = our `null`
regime** (so the strongest baseline is in-codebase and identically trained). *Tests:* overfit 256
seeds to ~0 loss. *DoD:* validation metrics in a sane range.

**Phase 4 — THE HEADLINE RESULT (do this before anything else).** `run_headline.py` trains the
**same architecture** under `full`, `null`, `shuffled`, and `name_only`, identical seeds/hparams/
self-labels, and prints one table: metric per regime with seed CIs, plus a **per-task breakdown**
(esp. tasks touching `statusId` / `grid`). Read it:
- `full > null` (with CIs) → documentation feeds signal. Proceed.
- `full > shuffled` → it's meaning, not any-text.
- `full > name_only` → it beats **names** (the real bar; this is the claim).
- gain **concentrated** on coded/sentinel tasks, ~flat elsewhere → consistent with the mechanism.
  *Uniform* gain → suspect a confound; investigate before believing the docs story.
- `full ≈ null` → **STOP.** No documentation paper on rel-f1. Record in `PROGRESS.md`; fall back to
  the hierarchy/operator axes (`METHOD_DESIGN.md` §6). Do not proceed to Phase 5+.
*DoD:* the four-regime table + per-task breakdown committed to `PROGRESS.md`. **This is the gate.**

**Phase 5 — Locality & coverage (only if Phase 4 passes).** `coverage_curve.py`: performance vs
documentation coverage; column-ablation showing the gain tracks coded/unit columns (H2). *DoD:*
coverage curve + per-column attribution.

**Phase 6 — More DBs (only if Phase 4 passes).** Author docs for rel-trial / rel-stack
(`DOC_AUTHORING.md`), rerun the four-regime headline per DB, report which DBs show the effect and how
it tracks coverage. *DoD:* headline table per DB.

**Phase 7 — (deferred) operator axis.** Only after documentation is settled: add the cardinality-
conditioned aggregator as a **separate, isolated** contribution (`METHOD_DESIGN.md` §6). Not now.

**Phase 8 — (deferred) transfer.** Leave-one-DB-out + masked-cell pretraining across the DBs that have
docs. Only meaningful with ≥3 documented DBs.

## 6. Test matrix (always green)
| Test | Asserts |
|---|---|
| `test_leakage.py` | no context row with `row_time > seed_time` |
| `test_shapes.py` | end-to-end shapes through RT substrate |
| `test_grounding.py` | null fallback works; placebo decorrelated; cache deterministic |
| `test_film_responds.py` | full vs null changes cell vectors (mechanism wired) |
| `test_selflabels_constant.py` | self-label cells identical across doc regimes (no confound) |

## 7. Risks & fallbacks
1. **`full ≈ null`** (docs add nothing — the prior says likely). The Phase-4 gate is designed to surface
   exactly this in run 2. Output = honest negative result; pivot to operator/hierarchy axes. Not a
   failure of the build — the point of the build.
2. **`full ≈ name_only`** (docs don't beat names). Same gate (H1c). The claim needs docs > names, or
   it collapses to RELATE. Surface it early.
3. **Grounding noise** → relevance gating + `d_null`; placebo (`shuffled`) catches "any vector helps."
4. **Uniform gain** → confound suspected; per-task breakdown (Phase 4) and column attribution
   (Phase 5) adjudicate before any claim.
5. **Library drift** → §3 contracts normative; adapt calls.

## 8. Definition of done (project, v1)
Green test matrix; **Phase-4 four-regime headline table with per-task breakdown and seed CIs recorded
in `PROGRESS.md`** (this is the deliverable that decides whether there is a paper); coverage curve if
Phase 4 passes; doc corpus published with tier + coverage. SOTA accuracy not required — the claim is
the *finding* (docs-on > docs-off > placebo, and docs-on > names-only), localized to where
documentation carries meaning names don't.