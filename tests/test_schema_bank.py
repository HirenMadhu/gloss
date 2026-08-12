"""Binding one set of shared weights to different databases' frozen tables.

The whole multi-database design rests on two claims, and both are asserted here: binding swaps only
**non-persistent buffers** (so it cannot touch a parameter, an optimizer state, or a checkpoint), and
a model bound to database B behaves exactly as a model *built* on B would.
"""
from __future__ import annotations

import pytest
import torch

from gloss.data.collate import to_cell_batch
from gloss.data.graph import build_gloss_graph
from gloss.data.masking import build_column_target_spec
from gloss.data.pretrain_loader import build_multi_pretrain_stream, split_key
from gloss.model.more import MoRE
from gloss.model.schema_bank import SchemaBank, SchemaTables
from gloss.text.cache import HashEncoder
from gloss.text.schema import (build_category_name_embeddings, build_column_name_embeddings,
                               build_table_name_embeddings, role_name_embeddings_with_none)
from gloss.utils.paths import graph_cache_dir

from ._relf1 import bundle_and_task
from .conftest import rel_f1_available

ENC = HashEncoder(dim=64)
D = 64


def _model(bundle):
    return MoRE(bundle, build_column_name_embeddings(bundle, ENC), d_model=D, d_sig=32, n_blocks=2,
                n_heads=4, d_ff=128, enc_channels=D, num_experts=4, k=2,
                cell_encoder="schema_free", cat_name_emb=build_category_name_embeddings(bundle, ENC),
                table_name_emb=build_table_name_embeddings(bundle, ENC),
                role_name_emb=role_name_embeddings_with_none(bundle, ENC))


def _two_bundles():
    cache = graph_cache_dir("rel-trial", "hash")
    if not cache.exists():
        pytest.skip("rel-trial graph cache not built")
    return bundle_and_task()[0], build_gloss_graph("rel-trial", cache_dir=str(cache))


def _batch(bundle, spec, seed=0):
    stream = build_multi_pretrain_stream({bundle.dataset_name: bundle},
                                         {bundle.dataset_name: spec},
                                         split="train", steps=1, batch_size=4, seed=seed)
    key, raw = next(iter(stream))
    _ds, nt = split_key(key)
    return to_cell_batch(raw, bundle, nt, seq_len=256, max_fk=5)


@rel_f1_available
def test_binding_touches_only_non_persistent_buffers():
    a, b = _two_bundles()
    ma, mb = _model(a), _model(b)
    before = {k: v.clone() for k, v in ma.state_dict().items()}
    SchemaBank({"rel-trial": SchemaTables.from_model(mb, "rel-trial")}).bind(ma, "rel-trial")
    after = ma.state_dict()
    assert set(before) == set(after)
    for k in before:
        assert torch.equal(before[k], after[k]), f"binding changed a state_dict entry: {k}"


@rel_f1_available
def test_a_bound_model_matches_a_natively_built_one():
    """The claim that makes multi-DB training legitimate rather than merely runnable."""
    a, b = _two_bundles()
    ma, mb = _model(a), _model(b)
    mb.load_state_dict(ma.state_dict(), strict=True)      # identical weights, different schema
    spec_b = build_column_target_spec(b, mb.encoder)
    cb = _batch(b, spec_b)

    bank = SchemaBank({"rel-trial": SchemaTables.from_model(mb, "rel-trial")})
    bank.bind(ma, "rel-trial")
    ma.eval(), mb.eval()
    with torch.no_grad():
        la, _aa, ca = ma(cb, return_cells=True)
        lb, _ab, cb_out = mb(cb, return_cells=True)
    assert torch.allclose(la, lb, atol=1e-5), (la - lb).abs().max()
    assert torch.allclose(ca, cb_out, atol=1e-5)


@rel_f1_available
def test_rebinding_is_reversible():
    a, b = _two_bundles()
    ma, mb = _model(a), _model(b)
    spec_a = build_column_target_spec(a, ma.encoder)
    cb_a = _batch(a, spec_a)
    ma.eval()
    with torch.no_grad():
        first = ma(cb_a)[0]
    bank = SchemaBank({"rel-f1": SchemaTables.from_model(ma, "rel-f1"),
                       "rel-trial": SchemaTables.from_model(mb, "rel-trial")})
    bank.bind(ma, "rel-trial")
    bank.bind(ma, "rel-f1")
    with torch.no_grad():
        again = ma(cb_a)[0]
    assert torch.equal(first, again), "a round trip through another schema changed the output"


@rel_f1_available
def test_bank_rejects_disagreeing_name_widths():
    """`schema_proj`/`name_proj` are shared weights of width d_text, so a mismatch cannot be bound."""
    a, b = _two_bundles()
    ta = SchemaTables.from_model(_model(a), "rel-f1")
    tb = SchemaTables.from_model(_model(b), "rel-trial")
    tb.name_emb = tb.name_emb[:, :8]
    with pytest.raises(ValueError, match="d_text"):
        SchemaBank({"rel-f1": ta, "rel-trial": tb})


@rel_f1_available
def test_multi_stream_keys_are_namespaced_and_weighted_per_dataset():
    a, b = _two_bundles()
    specs = {a.dataset_name: build_column_target_spec(a), b.dataset_name: build_column_target_spec(b)}
    stream = build_multi_pretrain_stream({a.dataset_name: a, b.dataset_name: b}, specs,
                                         split="train", steps=40, batch_size=4, seed=0)
    assert all("/" in n for n in stream.names)
    seen = {split_key(k)[0] for k in stream.table_schedule()}
    assert seen == {a.dataset_name, b.dataset_name}, seen
    # equal dataset weights by default, so neither DB should own the schedule
    counts = {d: sum(split_key(k)[0] == d for k in stream.table_schedule()) for d in seen}
    assert min(counts.values()) > 0, counts
