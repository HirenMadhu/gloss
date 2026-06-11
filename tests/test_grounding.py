"""Phase 1 — grounding: retrieval correctness, null fallback, placebo decorrelation, cache determinism."""
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
    # each element's pooled embedding should be closest to its matching span
    enc = _enc()
    span_emb = enc(SPANS)
    for i in range(3):
        sims = r.emb[i] @ span_emb.T
        assert int(sims.argmax()) == i


def test_null_regime_all_ungrounded():
    r = ground(ELEMENTS, SPANS, _enc(), regime="null")
    assert not r.grounded.any()
    assert torch.count_nonzero(r.emb) == 0
    assert r.d_text == _enc().dim


def test_high_threshold_falls_back_to_ungrounded():
    r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=1.01))
    assert not r.grounded.any()
    assert torch.count_nonzero(r.emb) == 0       # model would substitute learned d_null


def test_placebo_is_decorrelated_but_same_shape():
    full = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0), regime="full")
    plac = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0),
                  regime="shuffled_spans", seed=1)
    assert plac.emb.shape == full.emb.shape
    assert not torch.allclose(plac.emb, full.emb)   # spans permuted -> different pooling


def test_grounding_is_deterministic():
    a = ground(ELEMENTS, SPANS, _enc(), regime="shuffled_spans", seed=7)
    b = ground(ELEMENTS, SPANS, _enc(), regime="shuffled_spans", seed=7)
    assert torch.equal(a.emb, b.emb) and torch.equal(a.grounded, b.grounded)


def test_gather_unknown_key_returns_zeros():
    r = ground(ELEMENTS, SPANS, _enc(), GroundingConfig(sim_threshold=0.0))
    emb, rel, grd = r.gather(["e0", "does-not-exist"])
    assert emb.shape == (2, r.d_text)
    assert torch.count_nonzero(emb[1]) == 0 and not bool(grd[1])
    assert bool(grd[0])


def test_embedding_cache_is_idempotent(tmp_path):
    path = tmp_path / "cache.pt"
    enc = HashEncoder(dim=32)
    c1 = EmbeddingCache(enc, path)
    out1 = c1(["table races", "table drivers"], kind="document")
    assert path.exists()
    # fresh cache loads from disk and returns identical vectors without re-encoding
    c2 = EmbeddingCache(HashEncoder(dim=32), path)
    out2 = c2(["table races", "table drivers"], kind="document")
    assert torch.equal(out1, out2)
    # query vs document of the same text differ (instruction-aware)
    q = c2(["table races"], kind="query")
    d = c2(["table races"], kind="document")
    assert not torch.allclose(q, d)
