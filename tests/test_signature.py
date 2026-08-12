"""The value-free relational signature (hermetic, synthetic dual-FK fixture)."""
from __future__ import annotations

import pytest
import torch

from gloss.data.collate import to_cell_batch
from gloss.model.signature import RelationalSignature
from gloss.text.cache import HashEncoder
from gloss.text.schema import build_column_modality_ids, build_column_name_embeddings

from .conftest import ENTITY, make_synth_batch, synthetic_bundle


def _signature(bundle, *, d_text=16, d_sig=8):
    name_emb = build_column_name_embeddings(bundle, HashEncoder(dim=d_text))
    modality_id, n_stypes = build_column_modality_ids(bundle)
    return RelationalSignature(name_emb, modality_id, n_stypes, d_sig=d_sig)


def _batch(bundle, **kw):
    return to_cell_batch(make_synth_batch(**kw), bundle, ENTITY, seq_len=32, max_fk=2)


def test_signature_shape_and_finite():
    bundle = synthetic_bundle()
    cb = _batch(bundle)
    z = _signature(bundle)(cb)
    assert z.shape == (cb.num_seeds, cb.seq_len, 8)
    assert torch.isfinite(z).all()


def test_signature_is_value_free():
    """z must depend only on (column, modality, recency) — never on cell values."""
    bundle = synthetic_bundle()
    cb = _batch(bundle)
    sig = _signature(bundle)
    z1 = sig(cb)
    # perturb the raw cell VALUES in every TensorFrame; z must not move
    for tf in cb.tf_dict.values():
        for st in list(tf.feat_dict):
            tf.feat_dict[st] = tf.feat_dict[st] + 7.0
    z2 = sig(cb)
    assert torch.allclose(z1, z2)


def test_signature_changes_with_column():
    bundle = synthetic_bundle()
    cb = _batch(bundle)
    z = _signature(bundle)(cb)
    real = ~cb.is_padding
    cols = cb.col_idxs[real]
    zr = z[real]
    a = zr[cols == int(cols.min())][0]
    b = zr[cols == int(cols.max())][0]
    assert not torch.allclose(a, b)


def test_column_signature_is_value_free():
    """`column_signature()` is the diagnostics' way to ask "which expert does column c route to?"
    without a batch. It must exist under BOTH time modes: the old inline version reached for
    `recency_emb`, which only exists under `buckets`, so `specialization_probe` raised
    AttributeError under `rope` — the mode the two-level runs use — and the method's central claim
    went unmeasured because of it.
    """
    torch.manual_seed(0)
    C, d_text, d_sig = 7, 16, 8
    name = torch.randn(C, d_text)
    modality = torch.randint(0, 3, (C,))
    sig = RelationalSignature(name, modality, n_stypes=3, d_sig=d_sig, )
    z = sig.column_signature()
    assert z.shape == (C, d_sig)
    assert torch.isfinite(z).all()
    # one row per column, and distinct column names must not collapse to one point
    assert len({tuple(r.round(decimals=4).tolist()) for r in z}) > 1
