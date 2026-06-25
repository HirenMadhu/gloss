"""Phase 1 — grounding: retrieval, regimes (full/null/shuffled/name_only), name tokens, cache."""
from __future__ import annotations

import torch

from gloss.docs.cache import EmbeddingCache, HashEncoder
from gloss.docs.corpus import SchemaElement
from gloss.docs.grounding import GroundingConfig, ground


class BoWEncoder:
    """Deterministic bag-of-words encoder so retrieval is controllable (query/doc share tokens)."""

    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(sorted(vocab))}
        self.dim = len(self.vocab)

    def __call__(self, texts, kind="document"):
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                if w in self.vocab:
                    out[i, self.vocab[w]] += 1.0
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


SPANS = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
VOCAB = {w for s in SPANS for w in s.split()} | {"alpha", "delta", "eta"}
ELEMENTS = [
    SchemaElement("e0", "column", "t", "a", "alpha beta"),     # -> span 0
    SchemaElement("e1", "column", "t", "b", "delta epsilon"),  # -> span 1
    SchemaElement("e2", "column", "t", "c", "eta theta"),      # -> span 2
]


def _enc():
    return BoWEncoder(VOCAB)


def test_full_retrieves_matching_span():
    r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(top_k=2, sim_threshold=0.3))
    assert r.grounded.all()
    span_emb = _enc()(SPANS)
    for i in range(3):
        assert int((r.emb[i] @ span_emb.T).argmax()) == i


def test_name_emb_is_regime_independent():
    queries = _enc()([e.query for e in ELEMENTS], kind="query")
    for regime in ("full", "null", "shuffled", "name_only"):
        r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0), regime=regime)
        assert torch.allclose(r.name_emb, queries)        # RT name token never changes across regimes


def test_null_regime_all_ungrounded_but_keeps_names():
    r = ground(ELEMENTS, SPANS, _enc(), regime="null")
    assert not r.grounded.any()
    assert torch.count_nonzero(r.emb) == 0                # FiLM falls back to d_null
    assert torch.count_nonzero(r.name_emb) > 0            # but names survive (RT baseline)


def test_name_only_conditions_on_names():
    r = ground(ELEMENTS, SPANS, _enc(), regime="name_only")
    assert r.grounded.all()
    assert torch.allclose(r.emb, r.name_emb)              # FiLM conditioning == the name embedding


def test_high_threshold_falls_back_to_ungrounded():
    r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=1.01))
    assert not r.grounded.any()
    assert torch.count_nonzero(r.emb) == 0


def test_placebo_is_decorrelated_but_same_shape():
    full = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0), regime="full")
    plac = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0), regime="shuffled", seed=1)
    assert plac.emb.shape == full.emb.shape
    assert not torch.allclose(plac.emb, full.emb)         # element->doc assignment permuted


def test_grounding_is_deterministic():
    a = ground(ELEMENTS, SPANS, _enc(), regime="shuffled", seed=7)
    b = ground(ELEMENTS, SPANS, _enc(), regime="shuffled", seed=7)
    assert torch.equal(a.emb, b.emb) and torch.equal(a.grounded, b.grounded)


def test_gather_unknown_key_returns_zeros():
    r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0))
    emb, name, rel, grd = r.gather(["e0", "does-not-exist"])
    assert emb.shape == (2, r.d_text) and name.shape == (2, r.d_text)
    assert torch.count_nonzero(emb[1]) == 0 and not bool(grd[1])
    assert bool(grd[0])


def test_embedding_cache_is_idempotent(tmp_path):
    path = tmp_path / "cache.pt"
    c1 = EmbeddingCache(HashEncoder(dim=32), path)
    out1 = c1(["table races", "table drivers"], kind="document")
    assert path.exists()
    c2 = EmbeddingCache(HashEncoder(dim=32), path)
    out2 = c2(["table races", "table drivers"], kind="document")
    assert torch.equal(out1, out2)
    q = c2(["table races"], kind="query")
    d = c2(["table races"], kind="document")
    assert not torch.allclose(q, d)
