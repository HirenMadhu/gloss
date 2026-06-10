"""DocFiLMFusion — documentation modulates value (implementation.md §5). Phase-3 STUB.

    g = gamma(doc_emb); b = beta(doc_emb)          # small MLPs -> [d_model]
    x_cell = g * Wv(value_emb) + b + Wd(doc_emb)

Do NOT implement before GATE 1 is green ('proxy before transformer', §0). Signature is fixed so the
tokenizer/attention wiring is stable.
"""
from __future__ import annotations


class DocFiLMFusion:
    """FiLM fusion of DocCard embedding into the cell value representation.

    Args (Phase 3): ``d_value``, ``d_doc`` (=2560 for Qwen3-Embedding-4B), ``d_model``.
    forward(value_emb ``[T, d_value]``, doc_emb ``[T, d_doc]``) -> ``[T, d_model]``.
    """

    def __init__(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("Phase 3 — build after GATE 1 (see PROGRESS.md).")
