"""Phase 1 — frozen text encoders + an idempotent on-disk embedding cache.

The grounding module is encoder-agnostic: it takes any ``encode(texts, kind) -> Tensor[N, d]`` callable
(L2-normalized rows; ``kind`` is 'query' or 'document'). Two implementations:

* ``QwenEncoder`` — the real frozen ``Qwen/Qwen3-Embedding-4B`` via sentence-transformers, with an
  instruction prompt on *queries* only (the model is instruction-aware).
* ``HashEncoder`` — deterministic, dependency-free, for dev/tests (no model download). Distinct,
  stable vectors per (text, kind); semantically meaningless but enough to exercise grounding.

``EmbeddingCache`` wraps an encoder and memoizes per (kind, text) to disk so the (expensive) Qwen pass
runs once. It satisfies the same ``encode`` protocol, so it is a drop-in for grounding.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

QUERY_INSTRUCTION = (
    "Given a database schema element (a table, column, or foreign-key role), retrieve documentation "
    "spans that describe its meaning."
)


def _l2(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class HashEncoder:
    """Deterministic pseudo-embedding for dev/tests. Stable per (text, kind); unit-normalized."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        out = torch.empty(len(texts), self.dim, dtype=torch.float32)
        for i, t in enumerate(texts):
            h = int(hashlib.sha256(f"{kind}{t}".encode()).hexdigest()[:16], 16)
            g = torch.Generator().manual_seed(h & 0x7FFFFFFFFFFFFFFF)
            out[i] = torch.randn(self.dim, generator=g)
        return _l2(out)


class QwenEncoder:
    """Frozen Qwen3-Embedding-4B via sentence-transformers (lazy-loaded). Queries get the instruction."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        *,
        device: str | None = None,
        query_instruction: str = QUERY_INSTRUCTION,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        prompt = self.query_instruction if kind == "query" else None
        emb = self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.float().cpu()


class EmbeddingCache:
    """Memoize an encoder's outputs per (kind, text) to ``path`` (idempotent, content-hash keyed)."""

    def __init__(self, encoder, path: str | Path | None = None):
        self.encoder = encoder
        self.path = Path(path) if path is not None else None
        self._store: dict[str, Tensor] = {}
        if self.path is not None and self.path.exists():
            self._store = torch.load(self.path, map_location="cpu")

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}{text}".encode()).hexdigest()

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        missing = [t for t in texts if self._key(kind, t) not in self._store]
        if missing:
            embs = self.encoder(missing, kind=kind)
            for t, e in zip(missing, embs):
                self._store[self._key(kind, t)] = e.clone()
            self._flush()
        return torch.stack([self._store[self._key(kind, t)] for t in texts], dim=0)

    def _flush(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._store, self.path)

    @property
    def dim(self) -> int:
        if self._store:
            return next(iter(self._store.values())).numel()
        return getattr(self.encoder, "dim", -1)
