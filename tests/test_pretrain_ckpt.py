"""Checkpointing: the trunk/adapter split, an exact round-trip, and the cross-schema load.

The split is the whole point of the checkpoint format. `encoder.cell_encoders.*` is sized by one
database's columns and category vocabularies; everything else is not. If a per-DB encoder leaks into
the trunk, a LODO load either fails on a shape mismatch or — worse — silently succeeds against a
different database's columns. That happened once already during development: `split_state_dict` used
`startswith("encoder.cell_encoders.")` against LightningModule keys that are prefixed `model.`, so
every adapter tensor was classified as portable.
"""
from __future__ import annotations

import torch

from gloss.model.mlm_head import MaskedCellHead
from gloss.model.more import MoRE
from gloss.text.cache import HashEncoder
from gloss.text.schema import (build_column_name_embeddings, build_table_name_embeddings,
                               role_name_embeddings_with_none)
from gloss.train.pretrain import is_adapter_key, load_trunk, split_state_dict

from ._relf1 import bundle_and_task
from .conftest import rel_f1_available

D_MODEL, D_SIG, ENC = 64, 32, 64


def _model(bundle):
    e = HashEncoder(dim=64)
    return MoRE(bundle, build_column_name_embeddings(bundle, e), d_model=D_MODEL, d_sig=D_SIG,
                n_blocks=2, n_heads=4, d_ff=128, enc_channels=ENC, num_experts=4, k=2,
                table_name_emb=build_table_name_embeddings(bundle, e),
                role_name_emb=role_name_embeddings_with_none(bundle, e))


def _lit_style_state(model, head) -> dict:
    """Keys exactly as a LightningModule would emit them — the case the prefix bug got wrong."""
    return {**{f"model.{k}": v for k, v in model.state_dict().items()},
            **{f"head.{k}": v for k, v in head.state_dict().items()}}


@rel_f1_available
def test_split_puts_every_per_db_encoder_in_the_adapter_and_nothing_else():
    bundle, _task = bundle_and_task()
    model, head = _model(bundle), MaskedCellHead(D_MODEL, ENC)
    trunk, adapter = split_state_dict(_lit_style_state(model, head))

    assert adapter, "the rel-f1 encoders must land in the adapter, not vanish"
    assert all("cell_encoders" in k for k in adapter)
    assert not any("cell_encoders" in k for k in trunk), "an adapter tensor leaked into the trunk"
    assert any(k.startswith("head.") for k in trunk), "the decode head is portable and belongs here"
    assert any("substrate" in k for k in trunk)


def test_prefixed_and_unprefixed_keys_are_both_recognised():
    """The regression guard for the prefix bug: a LightningModule prefixes with `model.`."""
    assert is_adapter_key("encoder.cell_encoders.races.encoder_dict.numerical.weight")
    assert is_adapter_key("model.encoder.cell_encoders.races.encoder_dict.numerical.weight")
    assert not is_adapter_key("model.substrate.blocks.0.cell_ffn.router.weight")
    assert not is_adapter_key("head.num_head.weight")


@rel_f1_available
def test_frozen_name_tables_are_absent_from_the_state_dict_entirely():
    """`persistent=False` is what keeps `C`/`K`/`n_tables` out of every saved shape (SS0)."""
    bundle, _task = bundle_and_task()
    keys = _model(bundle).state_dict().keys()
    for banned in ("name_emb", "table_name_emb", "role_name_emb", "col_name_emb", "modality_id"):
        assert not any(k.endswith(banned) for k in keys), banned


@rel_f1_available
def test_trunk_round_trips_into_a_freshly_built_model(tmp_path):
    bundle, _task = bundle_and_task()
    torch.manual_seed(0)
    src_model, src_head = _model(bundle), MaskedCellHead(D_MODEL, ENC)
    torch.manual_seed(1)                                     # deliberately different init
    dst_model, dst_head = _model(bundle), MaskedCellHead(D_MODEL, ENC)

    trunk, adapter = split_state_dict(_lit_style_state(src_model, src_head))
    path = tmp_path / "trunk.pt"
    torch.save(trunk, path)
    dst_model.load_state_dict({k[6:]: v for k, v in adapter.items()}, strict=False)
    report = load_trunk(dst_model, dst_head, path)

    # The adapter keys are missing from a trunk file BY DESIGN; only a portable key going missing is
    # a failure. Here the adapter was loaded separately just above, so the model is complete.
    assert report["missing_portable"] == [] and report["shape_mismatch"] == []
    assert report["unknown"] == 0 and report["missing_adapter"] > 0
    for (n, a), (_, b) in zip(src_model.named_parameters(), dst_model.named_parameters()):
        assert torch.equal(a, b), n
    for (n, a), (_, b) in zip(src_head.named_parameters(), dst_head.named_parameters()):
        assert torch.equal(a, b), n


@rel_f1_available
def test_a_rel_f1_trunk_loads_onto_a_rel_trial_model_with_no_shape_mismatch(tmp_path):
    """The LODO proxy for the pretraining checkpoint. rel-f1 is 9 tables / 13 roles / 45 columns,
    rel-trial is 15 / 15 / 103, so every schema-bound shape differs — yet the trunk must load."""
    import pytest

    from gloss.data.graph import build_gloss_graph
    from gloss.utils.paths import graph_cache_dir

    cache = graph_cache_dir("rel-trial", "hash")
    if not cache.exists():
        pytest.skip("rel-trial graph cache not built")
    a_bundle, _task = bundle_and_task()
    b_bundle = build_gloss_graph("rel-trial", cache_dir=str(cache))

    a_model, a_head = _model(a_bundle), MaskedCellHead(D_MODEL, ENC)
    b_model, b_head = _model(b_bundle), MaskedCellHead(D_MODEL, ENC)
    trunk, _adapter = split_state_dict(_lit_style_state(a_model, a_head))
    path = tmp_path / "trunk.pt"
    torch.save(trunk, path)

    report = load_trunk(b_model, b_head, path)
    assert report["shape_mismatch"] == [], report["shape_mismatch"]
    assert report["unknown"] == 0, "the trunk carried a key rel-trial's model has no slot for"
    assert report["missing_portable"] == [], report["missing_portable"]
    assert report["loaded"] > 0 and report["missing_adapter"] > 0
    # What is missing must be exactly the per-DB encoders rel-trial needs and rel-f1 could not supply.
    own = {f"model.{k}" for k in b_model.state_dict()} | {f"head.{k}" for k in b_head.state_dict()}
    assert all(is_adapter_key(k) for k in own - set(trunk)), "a portable key failed to transfer"


@rel_f1_available
def test_the_transferred_trunk_actually_runs_on_the_other_schema(tmp_path):
    """Loading is necessary but not sufficient — the transferred weights must produce a finite
    forward on a batch from the schema they never saw."""
    import pytest

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph
    from gloss.data.masking import build_column_target_spec
    from gloss.data.pretrain_loader import build_pretrain_stream
    from gloss.utils.paths import graph_cache_dir

    cache = graph_cache_dir("rel-trial", "hash")
    if not cache.exists():
        pytest.skip("rel-trial graph cache not built")
    a_bundle, _task = bundle_and_task()
    b_bundle = build_gloss_graph("rel-trial", cache_dir=str(cache))

    a_model, a_head = _model(a_bundle), MaskedCellHead(D_MODEL, ENC)
    trunk, _ = split_state_dict(_lit_style_state(a_model, a_head))
    path = tmp_path / "trunk.pt"
    torch.save(trunk, path)

    b_model, b_head = _model(b_bundle), MaskedCellHead(D_MODEL, ENC)
    load_trunk(b_model, b_head, path)

    spec = build_column_target_spec(b_bundle, b_model.encoder)
    stream = build_pretrain_stream(b_bundle, spec, split="train", steps=1, batch_size=4, seed=0)
    nt, raw = next(iter(stream))
    cb = to_cell_batch(raw, b_bundle, nt, seq_len=256, max_fk=5)
    with torch.no_grad():
        logits, aux, cells = b_model(cb, return_cells=True)
    assert torch.isfinite(logits).all() and torch.isfinite(aux) and torch.isfinite(cells).all()
