"""Phase 2 — the documentation-conditioned cell encoder (rel-f1 guarded; HashEncoder grounding)."""
from __future__ import annotations

import torch

from gloss.model.column_encoder import CellEncoder

from ._relf1 import groundings, sample_cell_batch
from .conftest import rel_f1_available


@rel_f1_available
def test_cell_encoder_shapes_and_pad_zero():
    bundle, _task, cb = sample_cell_batch()
    g_full, _g_null, _g_name = groundings()
    enc = CellEncoder(bundle, d_model=64, d_text=g_full.d_text, enc_channels=64)
    h = enc(cb, g_full)
    assert h.shape == (cb.num_seeds, cb.seq_len, 64)
    assert torch.isfinite(h).all()
    assert torch.allclose(h[cb.is_padding], torch.zeros_like(h[cb.is_padding]))
