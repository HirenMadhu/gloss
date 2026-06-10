"""Phase-1 tests: DocCard regimes, programmatic synthetic cards, idempotent text cache, gather shapes."""
from __future__ import annotations

import numpy as np

from gloss.data.doccards import DocCard, cards_for_database, render
from gloss.data.synthetic import make_synthetic_bundle
from gloss.data.text_cache import DummyEncoder, build_text_cache


def _card():
    return DocCard(table="event", column="col_X", dtype="numeric",
                   table_desc="event log", column_desc="a measured signal",
                   unit="z-score", coded_values={"sign": "+1 (higher = more likely)"})


def test_regimes_render_distinctly():
    c = _card()
    full, name, placebo = render(c, "full"), render(c, "name_only"), render(c, "placebo")
    assert name == "col_X of event"
    assert "z-score" in full and "sign" in full
    # placebo is length-matched to full (word count) but shares no content words with it
    assert abs(len(placebo.split()) - len(full.split())) <= 1
    assert "z-score" not in placebo and "sign" not in placebo


def test_placebo_deterministic():
    c = _card()
    assert render(c, "placebo") == render(c, "placebo")


def test_synthetic_cards_encode_sign():
    bundle, planted = make_synthetic_bundle(seed=0)
    cards = cards_for_database(bundle.db, bundle.registry, planted=planted)
    causal_cg = bundle.registry.col_global_id[("event", planted.causal_col)]
    decoy_cg = bundle.registry.col_global_id[("event", planted.decoy_col)]
    full_causal = render(cards[causal_cg], "full")
    full_decoy = render(cards[decoy_cg], "full")
    # the causal card documents a sign; the decoy is explicitly inert
    assert "sign" in full_causal.lower() or "more likely" in full_causal or "less likely" in full_causal
    assert "inert" in full_decoy.lower()
    # FK column gets an fk_role filled from schema
    fk_cg = bundle.registry.col_global_id[("event", "entity_id")]
    assert cards[fk_cg].fk_target is not None


def test_text_cache_idempotent_and_gather():
    bundle, planted = make_synthetic_bundle(seed=1)
    cards = cards_for_database(bundle.db, bundle.registry, planted=planted)
    enc = DummyEncoder(dim=48)
    n = bundle.registry.num_cols
    tc1 = build_text_cache(cards, n, "full", enc, dataset="synthetic_test", force=True)
    tc2 = build_text_cache(cards, n, "full", enc, dataset="synthetic_test", force=False)  # loads cache
    assert tc1.emb.shape == (n, 48)
    assert np.allclose(tc1.emb, tc2.emb), "cache not deterministic / not reloaded identically"
    # gather by col_global_id -> [T, d]
    ids = np.array([0, 3, 3, n - 1])
    g = tc1.gather(ids)
    assert g.shape == (4, 48)
    assert np.allclose(g[1], g[2])  # same id -> same vector


def test_full_vs_placebo_embeddings_differ():
    bundle, planted = make_synthetic_bundle(seed=2)
    cards = cards_for_database(bundle.db, bundle.registry, planted=planted)
    enc = DummyEncoder(dim=48)
    n = bundle.registry.num_cols
    full = build_text_cache(cards, n, "full", enc, dataset="syn_fp", force=True)
    plac = build_text_cache(cards, n, "placebo", enc, dataset="syn_fp", force=True)
    causal_cg = bundle.registry.col_global_id[("event", planted.causal_col)]
    assert not np.allclose(full.emb[causal_cg], plac.emb[causal_cg])
