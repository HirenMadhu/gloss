"""Signature routing is leak-free: a seed cell's signature is invariant to which neighbors were sampled.

With ``route_on='signature'`` a cell's gate is a pure function of its own (column, modality, recency); it
cannot depend on the sampled neighborhood. We verify the upstream signature ``z`` (which drives the gate)
is identical for the seed cells across two batches of the same seeds that differ only in their neighbors.
"""
from __future__ import annotations

import torch

from gloss.data.collate import to_cell_batch
from gloss.model.signature import RelationalSignature
from gloss.text.cache import HashEncoder
from gloss.text.schema import build_column_modality_ids, build_column_name_embeddings

from .conftest import ENTITY, make_synth_batch, synthetic_bundle


def test_seed_signature_invariant_to_neighbor_sampling():
    bundle = synthetic_bundle()
    name_emb = build_column_name_embeddings(bundle, HashEncoder(dim=16))
    modality_id, n_stypes = build_column_modality_ids(bundle)
    sig = RelationalSignature(name_emb, modality_id, n_stypes, d_sig=8)

    # same two seeds, different sampled event neighbors (different times/values)
    cb_a = to_cell_batch(make_synth_batch(event_times=(10., 20., 30., 40.)), bundle, ENTITY,
                         seq_len=32, max_fk=2)
    cb_b = to_cell_batch(make_synth_batch(event_times=(13., 27., 31., 44.)), bundle, ENTITY,
                         seq_len=32, max_fk=2)

    za = sig(cb_a)[cb_a.is_seed_cell]
    zb = sig(cb_b)[cb_b.is_seed_cell]
    assert za.numel() > 0 and za.shape == zb.shape
    assert torch.allclose(za, zb)
