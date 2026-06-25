"""Supervised fine-tuning entry point + the per-column name-embedding plumbing.

Builds the frozen schema-name table (``gloss.text.schema``) once and trains the RT model on a
(dataset, task). The routing-signal ablation (``eval/ablation.py``, Phase D) reuses ``train_prebuilt``.
"""
from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl

from ..data.graph import build_gloss_graph
from ..text.cache import EmbeddingCache, HashEncoder, QwenEncoder, make_text_encoder
from ..text.schema import build_column_name_embeddings
from .datamodule import MoREDataModule
from .loop import MoRELitModule

REPO = Path(__file__).resolve().parents[2]


def _name_encoder(dataset: str, *, encoder: str = "hash", d_text: int = 64):
    """An ``encode(texts, kind) -> Tensor`` callable for the column-name table (cached for real models).

    ``hash`` is the dependency-free dev/test encoder (``d_text`` sets its width). ``qwen`` / a registry
    label / a raw HF id wrap the frozen model in an on-disk :class:`EmbeddingCache` so its single pass
    is idempotent across runs.
    """
    if encoder == "hash":
        return HashEncoder(dim=d_text)
    safe = encoder.replace("/", "__")
    cache_path = REPO / "data" / "schema_cache" / dataset / f"name_emb_{safe}.pt"
    if encoder == "qwen":
        return EmbeddingCache(QwenEncoder("Qwen/Qwen3-Embedding-4B"), cache_path)
    return EmbeddingCache(make_text_encoder(encoder), cache_path)


def name_embeddings(bundle, dataset: str, *, encoder: str = "hash", d_text: int = 64):
    """Frozen ``[C, d_text]`` column-name table for ``bundle`` (built once; cached for real encoders)."""
    enc = _name_encoder(dataset, encoder=encoder, d_text=d_text)
    return build_column_name_embeddings(bundle, enc)


def task_kind(task) -> str:
    """-> 'binary' | 'regression' from the RelBench task type (entity tasks only)."""
    from relbench.base import TaskType

    if task.task_type == TaskType.REGRESSION:
        return "regression"
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "binary"
    raise ValueError(f"unsupported task_type {task.task_type} (entity binary/regression only)")


def target_stats(task) -> tuple[float, float]:
    """TRAIN-split target mean/std for regression standardization (std floored at 1e-6)."""
    import numpy as np

    y = task.get_table("train").df[task.target_col].to_numpy(dtype="float64")
    y = y[~np.isnan(y)]
    return float(y.mean()), float(max(y.std(), 1e-6))


def train_prebuilt(
    bundle,
    task,
    name_emb,
    *,
    model_kwargs: dict | None = None,
    route_on: str = "dense",
    lambda_ortho: float = 0.5,
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
    """Train one run on a PREBUILT bundle + name table (the ablation reuses these across arms so the
    graph and the frozen name embeddings are built once)."""
    from ..utils.seeding import seed_everything

    seed_everything(seed)
    kind = task_kind(task)
    mean, std = target_stats(task) if kind == "regression" else (0.0, 1.0)
    module = MoRELitModule(
        bundle, name_emb, task.entity_table,
        task_type=kind, target_mean=mean, target_std=std,
        model_kwargs=model_kwargs, route_on=route_on, lambda_ortho=lambda_ortho,
        lr=lr, weight_decay=weight_decay, seq_len=seq_len, max_fk=max_fk,
    )
    dm = MoREDataModule(bundle, task, num_neighbors=num_neighbors, batch_size=batch_size,
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
    encoder: str = "hash",
    model_kwargs: dict | None = None,
    **kw,
):
    from relbench.tasks import get_task

    bundle = build_gloss_graph(dataset)
    task = get_task(dataset, task_name, download=False)
    d_text = 2560 if encoder == "qwen" else int((model_kwargs or {}).get("d_text", 64))
    name_emb = name_embeddings(bundle, dataset, encoder=encoder, d_text=d_text)
    return train_prebuilt(bundle, task, name_emb, model_kwargs=model_kwargs, **kw)
