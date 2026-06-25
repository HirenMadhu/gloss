"""The value-free relational signature (hermetic, synthetic dual-FK fixture)."""
from __future__ import annotations

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


def test_recency_bins_untimed_zero_timed_positive_and_resolved():
    bundle = synthetic_bundle()
    # seed_time large; events spread across orders of magnitude -> distinct recency buckets
    cb = _batch(bundle, seed_time=1_000_000.0,
                event_times=(1_000_000.0 - 5, 1_000_000.0 - 50,
                             1_000_000.0 - 5000, 1_000_000.0 - 500_000))
    bins = _signature(bundle).recency_bins(cb)
    untimed = ~cb.is_timed                       # users + pad
    timed = cb.is_timed & ~cb.is_padding
    assert int(bins[untimed].max()) == 0
    assert int(bins[timed].min()) >= 1
    assert len(set(bins[timed].tolist())) >= 2   # binning resolves different recencies


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
