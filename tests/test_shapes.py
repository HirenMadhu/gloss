"""End-to-end DOC-RT forward shapes (rel-f1 guarded)."""
from __future__ import annotations

import torch

from gloss.model.docrt import DOCRT

from ._relf1 import groundings, sample_cell_batch
from .conftest import rel_f1_available


@rel_f1_available
def test_docrt_forward_shapes():
    bundle, _task, cb = sample_cell_batch(seq_len=384, batch_size=8)
    g_full, _g_null, _g_name = groundings()
    model = DOCRT(bundle, d_model=64, d_text=g_full.d_text, n_blocks=2, n_heads=4,
                  d_ff=128, enc_channels=64)
    with torch.no_grad():
        logits = model(cb, g_full)
    assert logits.shape == (cb.num_seeds, 1)
    assert torch.isfinite(logits).all()
