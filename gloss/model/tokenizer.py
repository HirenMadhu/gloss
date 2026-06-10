"""Cell tokenization: value/doc/time/struct streams (implementation.md §4.2). Phase-3 STUB.

Packs per-cell value_input (pytorch-frame type-specific or precomputed), gathers doc_emb by col_global_id
from the frozen text cache, adds fk_role / node_type / time. Consumes a gloss.data.collate.TokenBatch.
"""
from __future__ import annotations


class CellTokenizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — build after GATE 1.")
