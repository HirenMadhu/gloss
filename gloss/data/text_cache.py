"""Frozen text-embedding cache (implementation.md §4.1, §0).

DocCards are encoded **once** by a frozen encoder (default `Qwen/Qwen3-Embedding-4B`, 2560-dim) and cached
to disk, indexed by `col_global_id`. The training loop never runs the LM — it only `gather`s cached
vectors. The cache is **idempotent**: keyed by (dataset, encoder, regime, hash of rendered card texts), so
re-running is a no-op load.

The encoder is injectable: real runs use :class:`QwenEmbedder`; tests use :class:`DummyEncoder`
(deterministic, offline) so cache/gather logic is verified without an 8 GB model download.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gloss.data.doccards import DocCard, Regime, render
from gloss.utils.config import scratch_dir

DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-4B"


# --------------------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------------------
class DummyEncoder:
    """Deterministic hash-based embeddings — for tests / offline dev (no model download)."""

    def __init__(self, dim: int = 64, name: str = "dummy"):
        self.dim = dim
        self.name = name

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.blake2b(t.encode("utf-8"), digest_size=32).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out


class QwenEmbedder:
    """Frozen sentence-transformers encoder (lazy-loaded). Last-token pooling + left padding are handled
    by the model's own sentence-transformers config. DocCards are encoded as *documents* (no query
    instruction). Runs in bf16 on GPU; cache to scratch."""

    def __init__(self, name: str = DEFAULT_ENCODER, device: str | None = None,
                 batch_size: int = 32, normalize: bool = True):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _ensure(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(
                self.name, device=dev,
                model_kwargs={"torch_dtype": torch.bfloat16} if dev == "cuda" else None,
            )
        return self._model

    @property
    def dim(self) -> int:
        return self._ensure().get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure()
        emb = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                            convert_to_numpy=True, show_progress_bar=len(texts) > 256)
        return emb.astype(np.float32)


# --------------------------------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------------------------------
@dataclass
class TextCache:
    """``emb[col_global_id] -> [d_text]`` for one (dataset, encoder, regime)."""

    emb: np.ndarray                # [num_cols, d_text]
    regime: str
    encoder: str
    dataset: str

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    def gather(self, col_global_ids) -> np.ndarray:
        """[T] ids -> [T, d_text]. Accepts numpy / torch / list."""
        ids = np.asarray(col_global_ids)
        return self.emb[ids]

    def gather_torch(self, col_global_ids):
        import torch

        ids = col_global_ids.detach().cpu().numpy() if hasattr(col_global_ids, "detach") else np.asarray(col_global_ids)
        return torch.from_numpy(self.emb[ids])


def _texts_for_cards(cards: dict[int, DocCard], num_cols: int, regime: Regime) -> list[str]:
    """Rendered text per col_global_id (0..num_cols-1); empty string for columns without a card."""
    return [render(cards[i], regime) if i in cards else "" for i in range(num_cols)]


def _cache_key(dataset: str, encoder: str, regime: str, texts: list[str]) -> str:
    h = hashlib.blake2b("␟".join(texts).encode("utf-8"), digest_size=16).hexdigest()
    enc = encoder.replace("/", "_")
    return f"{dataset}__{enc}__{regime}__{h}"


def build_text_cache(
    cards: dict[int, DocCard],
    num_cols: int,
    regime: Regime,
    encoder,
    dataset: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> TextCache:
    """Render cards under ``regime``, embed once, cache idempotently. Returns a :class:`TextCache`."""
    cache_dir = Path(cache_dir) if cache_dir else scratch_dir("gloss_text_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    enc_name = getattr(encoder, "name", encoder.__class__.__name__)
    texts = _texts_for_cards(cards, num_cols, regime)
    key = _cache_key(dataset, enc_name, regime, texts)
    npy = cache_dir / f"{key}.npy"
    manifest = cache_dir / f"{key}.json"

    if npy.exists() and manifest.exists() and not force:
        emb = np.load(npy)
        return TextCache(emb=emb, regime=regime, encoder=enc_name, dataset=dataset)

    emb = encoder.encode(texts)
    np.save(npy, emb)
    manifest.write_text(json.dumps(
        {"dataset": dataset, "encoder": enc_name, "regime": regime, "num_cols": num_cols,
         "dim": int(emb.shape[1])}, indent=2))
    return TextCache(emb=emb, regime=regime, encoder=enc_name, dataset=dataset)
