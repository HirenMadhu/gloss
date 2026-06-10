"""gloss — HALOS (*Heterogeneous Attention, Language Of Schema*).

A measurement-paper codebase: structured per-column **DocCards** as a frozen-LM modality (C1) and a
model-agnostic **Documentation Sufficiency Audit / DSA** (C3). Thesis: *"names lie, meaning transfers."*

Subpackages
-----------
- ``gloss.data``   leakage-safe relational graph + sampler, DocCards, text cache, synthetic generator.
- ``gloss.proxy``  Phase-2 GATE-1 embedding probe (the week-one proxy test).
- ``gloss.model``  HALOS-minimal relational transformer (Phase 3 — stubs until the gate is green).
- ``gloss.audit``  the product: CMI estimator, controls, Shapley, faithfulness (Phase 5).
- ``gloss.train``  Lightning training/pretraining loops.
- ``gloss.eval``   the gradient map, name-shuffle test, metrics.
- ``gloss.utils``  seeding, config, logging, flops.
- ``gloss.ext``    DEFERRED Paper-#2 temporal kernel (do not build in cycle 1).

See ``idea.md`` (rationale) and ``implementation.md`` (build spec) — both normative.
"""

__version__ = "0.1.0"
