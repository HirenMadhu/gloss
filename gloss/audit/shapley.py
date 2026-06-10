"""Column / key-path KernelSHAP with relational masking (implementation.md §7). Phase-5 STUB.

Faithfulness via Shapley over columns/key-paths (NOT attention weights). Value function = column-mean /
[MASK] baseline respecting temporal masks; amortize with a learned head, validated vs exact Shapley on
small contexts.
"""
from __future__ import annotations


def column_shapley(*args, **kwargs):
    raise NotImplementedError("Phase 5 (GATE 2).")
