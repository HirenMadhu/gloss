"""Phase 4 — supervised fine-tuning entry point + the regime->(grounding, doc_per_metapath) plumbing
reused by the H1 gate (Phase 5).
"""
from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch

from ..data.graph import build_gloss_graph
from ..docs.cache import EmbeddingCache, HashEncoder, QwenEncoder
from ..docs.corpus import DocCorpus, schema_elements_from_db
from ..docs.grounding import GroundingConfig, GroundingResult, ground
from ..model.halos import build_doc_per_metapath
from .datamodule import HALOSDataModule
from .loop import HALOSLitModule

REPO = Path(__file__).resolve().parents[2]


def make_grounding(
    bundle,
    dataset: str,
    *,
    regime: str = "full",
    encoder: str = "qwen",
    d_text: int = 64,
    sim_threshold: float = 0.60,
    chunk_sentences: int = 3,
    top_k: int = 4,
) -> GroundingResult:
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


def docs_for_regime(bundle, dataset: str, regime: str, **kw):
    """-> (grounding, doc_per_metapath). null => no doc-conditioned geometry (doc_mp=None)."""
    g = make_grounding(bundle, dataset, regime=regime, **kw)
    doc_mp = None if regime == "null" else build_doc_per_metapath(bundle, g)
    return g, doc_mp


def train_prebuilt(
    bundle,
    task,
    grounding,
    doc_per_metapath,
    *,
    model_kwargs: dict | None = None,
    num_neighbors: list[int] | None = None,
    batch_size: int = 64,
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
    """Train one HALOS run on a PREBUILT bundle + grounding (the H1 gate reuses these across configs)."""
    from ..utils.seeding import seed_everything

    seed_everything(seed)
    module = HALOSLitModule(
        bundle, grounding, doc_per_metapath, task.entity_table,
        model_kwargs=model_kwargs, lr=lr, weight_decay=weight_decay,
    )
    dm = HALOSDataModule(bundle, task, num_neighbors=num_neighbors, batch_size=batch_size,
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
    **kw,
):
    from relbench.tasks import get_task

    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    g, doc_mp = docs_for_regime(bundle, dataset, regime, encoder=encoder,
                                d_text=(model_kwargs or {}).get("d_text", 64))
    return train_prebuilt(bundle, task, g, doc_mp, model_kwargs=model_kwargs, **kw)
