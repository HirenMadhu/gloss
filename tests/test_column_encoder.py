"""The RT cell encoder (rel-f1 guarded; HashEncoder name table)."""
from __future__ import annotations

import torch

from gloss.model.column_encoder import CellEncoder

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available


@rel_f1_available
def test_cell_encoder_shapes_and_pad_zero():
    bundle, _task, cb = sample_cell_batch()
    name_emb = name_table()
    enc = CellEncoder(bundle, name_emb, d_model=64, enc_channels=64)
    h = enc(cb)
    assert h.shape == (cb.num_seeds, cb.seq_len, 64)
    assert torch.isfinite(h).all()
    assert torch.allclose(h[cb.is_padding], torch.zeros_like(h[cb.is_padding]))
