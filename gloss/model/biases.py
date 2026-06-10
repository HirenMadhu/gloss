"""TypedMetapathHopBias (optional, implementation.md §5). Phase-3 STUB.

    B_hop(i,j) = HopTable[metapath_id_ij, hop_ij]   # learned scalar per (typed-path, distance), per head

struct_bias.mode in {none, scalar_hop, typed_metapath}. NO content-addressed kernel here.
"""
from __future__ import annotations


class TypedMetapathHopBias:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — build after GATE 1.")
