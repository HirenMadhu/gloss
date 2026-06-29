"""Frozen text encoders + an idempotent on-disk embedding cache.

Encoder-agnostic: any ``encode(texts, kind) -> Tensor[N, d]`` callable (L2-normalized rows; ``kind`` is
a free label such as 'query' or 'document', used only for the cache key and the optional instruction
prompt). MoRE uses these to embed **schema text** (column names) once, offline, into the frozen table
its router routes on. Three implementations:

* ``QwenEncoder`` — the real frozen ``Qwen/Qwen3-Embedding-4B`` via sentence-transformers, with an
  optional instruction prompt on *queries* only (the model is instruction-aware).
* ``HFLastTokenEmbedder`` — a raw-transformers decoder-only embedder (last-token pool) for big models.
* ``HashEncoder`` — deterministic, dependency-free, for dev/tests (no model download). Distinct,
  stable vectors per (text, kind); semantically meaningless but enough to exercise the wiring.

``EmbeddingCache`` wraps an encoder and memoizes per (kind, text) to disk so the (expensive) Qwen pass
runs once. It satisfies the same ``encode`` protocol, so it is a drop-in anywhere an encoder is taken.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

QUERY_INSTRUCTION = (
    "Given the name of a database table and column, represent its relational role and semantic meaning."
)

# Short labels -> HuggingFace model ids for the frozen sentence-embedding encoders we ground with.
# The cache file is keyed by the label (``emb_cache_<label>.pt``), so swapping encoders never clobbers
# another's cache. Add new encoders here; ``make_text_encoder`` loads any label *or* a raw HF id.
ENCODER_MODELS = {
    "qwen": "Qwen/Qwen3-Embedding-4B",          # d_text 2560
    "harrier": "microsoft/harrier-oss-v1-27b",   # d_text 5376 (Gemma-3-27B backbone; needs bf16)
}


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
    """Frozen sentence-embedding encoder via sentence-transformers (lazy-loaded). Despite the name it
    wraps *any* SentenceTransformer model id; queries get the instruction prompt. ``model_kwargs`` are
    forwarded to ``from_pretrained`` (e.g. ``{'torch_dtype': torch.bfloat16}`` so a 27B model fits on
    one GPU)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        *,
        device: str | None = None,
        query_instruction: str = QUERY_INSTRUCTION,
        batch_size: int = 16,
        model_kwargs: dict | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kw = {"model_kwargs": self.model_kwargs} if self.model_kwargs else {}
            self._model = SentenceTransformer(self.model_name, device=self.device, **kw)
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


class HFLastTokenEmbedder:
    """Frozen decoder-only embedding model via *raw* transformers (lazy-loaded).

    Bypasses sentence-transformers, whose 5.x ``Transformer`` module unconditionally calls
    ``AutoProcessor`` and so crashes on multimodal backbones like Gemma-3 (it demands an image
    processor a text embedder never ships). Implements the harrier recipe directly: **last-token
    pooling + L2 normalize**, with an ``Instruct: …\\nQuery:`` prefix on *queries* only (documents are
    raw). Loaded in bf16 so a 27B model fits one GPU.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        query_instruction: str = QUERY_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 512,
        dtype=None,
    ):
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token  # decoder tokenizers often lack a pad token
            self._tok.padding_side = "right"  # last real token at index sum(mask)-1
            try:
                model = AutoModel.from_pretrained(self.model_name, dtype=self.dtype)
            except TypeError:  # older transformers
                model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
            self._model = model.to(self.device).eval()
        return self._model, self._tok

    def _format(self, texts: list[str], kind: str) -> list[str]:
        if kind == "query":
            return [f"Instruct: {self.query_instruction}\nQuery: {t}" for t in texts]
        return list(texts)

    @torch.no_grad()
    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        model, tok = self._load()
        inputs = self._format(texts, kind)
        out: list[Tensor] = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i : i + self.batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=self.max_length,
                      return_tensors="pt").to(self.device)
            hidden = model(**enc).last_hidden_state                      # [B, T, H]
            last = enc["attention_mask"].sum(dim=1) - 1                  # right padding -> last real tok
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
            out.append(_l2(pooled.float()).cpu())
        return torch.cat(out, dim=0)

    @property
    def dim(self) -> int:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_name)
        return int(getattr(cfg, "hidden_size", getattr(getattr(cfg, "text_config", cfg), "hidden_size", -1)))


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


def make_text_encoder(label_or_id: str, *, device: str | None = None, batch_size: int = 8):
    """Build a frozen decoder-only embedding encoder for a registry label (``ENCODER_MODELS``) or a raw
    HF model id, via raw transformers (``HFLastTokenEmbedder``): last-token pooling + L2 norm in bf16.

    We deliberately avoid sentence-transformers here — its 5.x ``Transformer`` module forces
    ``AutoProcessor`` and crashes on multimodal backbones like Gemma-3 (the 27B ``harrier``). bf16 also
    keeps such a model off the fp32 path that would OOM even an 80 GB H100. (The ``qwen`` path still
    uses sentence-transformers, where it works and is already cached.)
    """
    model_id = ENCODER_MODELS.get(label_or_id, label_or_id)
    return HFLastTokenEmbedder(model_id, device=device, batch_size=batch_size)
