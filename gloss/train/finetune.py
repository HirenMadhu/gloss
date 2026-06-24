"""Phase 3 — supervised fine-tuning entry point + the regime -> grounding plumbing reused by the
four-regime headline runner (``eval/ablation.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl

from ..data.graph import build_gloss_graph
from ..docs.cache import EmbeddingCache, HashEncoder, QwenEncoder
from ..docs.corpus import DocCorpus, schema_elements_from_db
from ..docs.grounding import GroundingConfig, GroundingResult, ground
from .datamodule import DOCRTDataModule
from .loop import DOCRTLitModule

REPO = Path(__file__).resolve().parents[2]


def make_grounding(
    dataset: str,
    *,
    regime: str = "full",
    encoder: str = "qwen",
    d_text: int = 64,
    sim_threshold: float = 0.60,
    chunk_sentences: int = 3,
    top_k: int = 4,
) -> GroundingResult:
    """Build a GroundingResult for ``regime`` from the authored doc corpus + the cached text encoder."""
    from relbench.datasets import get_dataset

    corpus = DocCorpus.load(REPO / "doc_corpus", dataset)
    db = get_dataset(dataset, download=False).get_db(upto_test_timestamp=False)
    elements = schema_elements_from_db(db)
    spans = corpus.spans(chunk_sentences)
    if encoder == "qwen":
        cache_path = REPO / "data" / "doc_cache" / dataset / "emb_cache_qwen.pt"
        enc = EmbeddingCache(QwenEncoder("Qwen/Qwen3-Embedding-4B"), cache_path)
    else:
        enc = HashEncoder(dim=d_text)
    cfg = GroundingConfig(chunk_sentences=chunk_sentences, top_k=top_k, sim_threshold=sim_threshold)
    return ground(elements, spans, enc, cfg, regime=regime)


def docs_for_regime(dataset: str, regime: str, *, encoder: str = "qwen", d_text: int = 64, **kw):
    """-> GroundingResult for ``regime``. ``null`` keeps the (regime-independent) RT name tokens but
    turns the FiLM doc conditioning off (d_null everywhere)."""
    return make_grounding(dataset, regime=regime, encoder=encoder, d_text=d_text, **kw)


def train_prebuilt(
    bundle,
    task,
    grounding,
    *,
    model_kwargs: dict | None = None,
    num_neighbors: list[int] | None = None,
    batch_size: int = 64,
    seq_len: int = 1024,
    max_fk: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 5,
    accelerator: str = "auto",
    logger=False,
    seed: int = 0,
    num_workers: int = 0,
    limit_train_batches: float | int | None = None,
    limit_val_batches: float | int | None = None,
):
    """Train one DOC-RT run on a PREBUILT bundle + grounding (the headline runner reuses these across
    regimes so the graph is built once)."""
    from ..utils.seeding import seed_everything

    seed_everything(seed)
    module = DOCRTLitModule(
        bundle, grounding, task.entity_table,
        model_kwargs=model_kwargs, lr=lr, weight_decay=weight_decay, seq_len=seq_len, max_fk=max_fk,
    )
    dm = DOCRTDataModule(bundle, task, num_neighbors=num_neighbors, batch_size=batch_size,
                         num_workers=num_workers)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices=1,
        logger=logger, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, log_every_n_steps=20,
        limit_train_batches=limit_train_batches or 1.0,
        limit_val_batches=limit_val_batches or 1.0,
    )
    trainer.fit(module, dm)
    return module, {k: float(v) for k, v in trainer.callback_metrics.items()}


def train(
    *,
    dataset: str = "rel-f1",
    task_name: str = "driver-dnf",
    regime: str = "full",
    encoder: str = "qwen",
    model_kwargs: dict | None = None,
    sim_threshold: float = 0.60,
    **kw,
):
    from relbench.tasks import get_task

    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g = docs_for_regime(dataset, regime, encoder=encoder,
                        d_text=(model_kwargs or {}).get("d_text", 64), sim_threshold=sim_threshold)
    return train_prebuilt(bundle, task, g, model_kwargs=model_kwargs, **kw)
