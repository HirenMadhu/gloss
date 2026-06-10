"""Documentation Sufficiency: predictive-proxy CMI estimator (implementation.md §5, §7). Phase-5 STUB.

    I_hat(Y; Doc | Values, Structure) ~ E_heldout[ logloss(model_nodoc) - logloss(model_full) ]   (+ seed CIs)

**Capacity confound (plan stress-test #2):** model_nodoc MUST be the SAME architecture with the doc pathway
neutralized (doc->zeros), NOT a smaller model. **Placebo is the capacity control** — only full >> placebo~0
counts as information. Model-agnostic: the 'model' can be the Phase-2 proxy probe (validate the estimator
early, before any transformer).
"""
from __future__ import annotations


def estimate_cmi(*args, **kwargs):
    raise NotImplementedError("Phase 5 (GATE 2) — the product. Validate on synthetic planted truth first.")
