"""The schema-free cell encoder: no learned weight sized by any database.

The claim being pinned is stronger than the one in `test_pretrain_ckpt.py`. There, a *trunk* built on
rel-f1 loaded onto rel-trial and the per-DB `cell_encoders` were legitimately missing. Here the
**entire** state_dict is identical in shape across the two schemas, so `strict=True` load works and
LODO transfer re-initializes nothing at all.
"""
from __future__ import annotations

import pytest
import torch

from gloss.data.graph import build_gloss_graph
from gloss.data.masking import CATEGORICAL_ID, build_column_target_spec
from gloss.model.more import CELL_ENCODERS, MoRE
from gloss.model.schema_free_encoder import SchemaFreeCellEncoder
from gloss.text.cache import HashEncoder
from gloss.text.schema import (build_category_name_embeddings, build_column_name_embeddings,
                               build_table_name_embeddings, category_index,
                               role_name_embeddings_with_none)

from ._relf1 import bundle_and_task
from .conftest import rel_f1_available

ENC = HashEncoder(dim=64)
D_MODEL = 64


def _model(bundle, kind="schema_free"):
    return MoRE(bundle, build_column_name_embeddings(bundle, ENC), d_model=D_MODEL, d_sig=32,
                n_blocks=2, n_heads=4, d_ff=128, enc_channels=D_MODEL, num_experts=4, k=2,
                cell_encoder=kind, cat_name_emb=build_category_name_embeddings(bundle, ENC),
                table_name_emb=build_table_name_embeddings(bundle, ENC),
                role_name_emb=role_name_embeddings_with_none(bundle, ENC))


@rel_f1_available
def test_no_learned_tensor_is_sized_by_the_schema():
    bundle, _t = bundle_and_task()
    enc = SchemaFreeCellEncoder(bundle, build_column_name_embeddings(bundle, ENC),
                                build_category_name_embeddings(bundle, ENC),
                                d_model=D_MODEL, enc_channels=D_MODEL)
    n_cat = enc.cat_emb.shape[0]
    forbidden = {enc.name_emb.shape[0], len(bundle.node_types), bundle.num_roles, n_cat}
    forbidden -= {D_MODEL, enc.enc_channels, enc.d_text}
    for name, p in enc.named_parameters():
        assert not (set(p.shape) & forbidden), (name, tuple(p.shape), forbidden)


@rel_f1_available
def test_frozen_tables_stay_out_of_the_state_dict():
    bundle, _t = bundle_and_task()
    enc = SchemaFreeCellEncoder(bundle, build_column_name_embeddings(bundle, ENC),
                                build_category_name_embeddings(bundle, ENC), d_model=D_MODEL)
    keys = set(enc.state_dict())
    for banned in ("name_emb", "cat_emb", "num_mean", "num_std", "cat_base", "modality_id",
                   "year_stats", "cyclic_periods"):
        assert not any(k.endswith(banned) for k in keys), banned


@rel_f1_available
def test_forward_matches_the_per_column_encoder_in_shape_and_padding():
    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import make_loader
    from relbench.tasks import get_task

    bundle, _t = bundle_and_task()
    task = get_task("rel-f1", "driver-dnf", download=False)
    raw = next(iter(make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=8)))
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=256, max_fk=5)
    for kind in CELL_ENCODERS:
        m = _model(bundle, kind)
        with torch.no_grad():
            h = m.encoder(cb)
        assert h.shape == (cb.num_seeds, cb.seq_len, D_MODEL), kind
        assert torch.isfinite(h).all(), kind
        assert bool((h[cb.is_padding] == 0).all()), kind


@rel_f1_available
def test_masking_still_replaces_only_the_value_half():
    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import make_loader
    from relbench.tasks import get_task

    from gloss.data.masking import sample_cell_mask

    bundle, _t = bundle_and_task()
    task = get_task("rel-f1", "driver-dnf", download=False)
    raw = next(iter(make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=8)))
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=256, max_fk=5)
    m = _model(bundle)
    spec = build_column_target_spec(bundle, m.encoder)
    mask, _seed = sample_cell_mask(cb, spec, p_random=0.4,
                                   generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        plain, masked, token = m.encoder(cb), m.encoder(cb, mask), m.encoder.masked_token(cb)
    real = ~cb.is_padding
    assert torch.allclose(masked[mask & real], token[mask & real], atol=1e-6)
    assert torch.allclose(masked[real & ~mask], plain[real & ~mask], atol=0)


@rel_f1_available
def test_category_base_comes_from_the_global_frozen_index():
    """The head must score against the frozen table's slice, not torch_frame's per-table offsets."""
    bundle, _t = bundle_and_task()
    m = _model(bundle)
    spec = build_column_target_spec(bundle, m.encoder)
    index, _texts = category_index(bundle)
    from gloss.data.collate import column_vocab

    vocab = column_vocab(bundle)
    checked = 0
    for (nt, col), (base, n) in index.items():
        gid = vocab[(nt, col)]
        assert int(spec.cat_base[gid]) == base, (nt, col)
        assert int(spec.n_cat[gid]) == n
        checked += 1
    assert checked > 0
    assert int(spec.cat_base.max()) < m.encoder.cat_emb.shape[0]


@rel_f1_available
def test_the_whole_model_is_identical_in_shape_across_two_schemas():
    """The payoff: not "the trunk transfers" but "everything transfers", `strict=True`."""
    from gloss.utils.paths import graph_cache_dir

    if not graph_cache_dir("rel-trial", "hash").exists():
        pytest.skip("rel-trial graph cache not built")
    a = _model(bundle_and_task()[0])
    b = _model(build_gloss_graph("rel-trial", cache_dir=str(graph_cache_dir("rel-trial", "hash"))))
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), set(sa) ^ set(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)
    b.load_state_dict(sa, strict=True)          # raises if anything at all fails to transfer


@rel_f1_available
def test_timestamp_year_uses_one_global_normalizer():
    """Per-column `YEAR_RANGE` would feed inconsistent scales through a single shared projection."""
    bundle, _t = bundle_and_task()
    enc = SchemaFreeCellEncoder(bundle, build_column_name_embeddings(bundle, ENC),
                                build_category_name_embeddings(bundle, ENC), d_model=D_MODEL)
    assert enc.year_stats.shape == (2,), "one mean and one std for the entire database"
    assert float(enc.year_stats[1]) > 0
