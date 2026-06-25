"""Phase 2 — the FiLM mechanism is wired: switching the doc regime changes the cell vectors."""
from __future__ import annotations

import torch

from gloss.model.column_encoder import CellEncoder

from ._relf1 import groundings, sample_cell_batch
from .conftest import rel_f1_available


@rel_f1_available
def test_full_vs_null_changes_cell_vectors():
    bundle, _task, cb = sample_cell_batch()
    g_full, g_null, _g_name = groundings()
    torch.manual_seed(0)
    enc = CellEncoder(bundle, d_model=64, d_text=g_full.d_text, enc_channels=64)
    enc.eval()
    with torch.no_grad():
        h_full = enc(cb, g_full)
        h_null = enc(cb, g_null)
    real = ~cb.is_padding
    # the documentation regime (full vs null=d_null) must move the cell vectors
    assert not torch.allclose(h_full[real], h_null[real])


@rel_f1_available
def test_full_vs_name_only_differ():
    bundle, _task, cb = sample_cell_batch()
    g_full, _g_null, g_name = groundings()
    torch.manual_seed(0)
    enc = CellEncoder(bundle, d_model=64, d_text=g_full.d_text, enc_channels=64)
    enc.eval()
    with torch.no_grad():
        h_full = enc(cb, g_full)
        h_name = enc(cb, g_name)
    real = ~cb.is_padding
    # conditioning FiLM on docs vs on names yields different cell vectors (the H1c contrast)
    assert not torch.allclose(h_full[real], h_name[real])
