"""Supervised fine-tuning entry point + the per-column name-embedding plumbing.

Builds the frozen schema-name table (``gloss.text.schema``) once and trains the RT model on a
(dataset, task). The routing-signal ablation (``eval/ablation.py``, Phase D) reuses ``train_prebuilt``.
"""
from __future__ import annotations

import pytorch_lightning as pl

from ..data.graph import build_gloss_graph
from ..text.cache import EmbeddingCache, HashEncoder, QwenEncoder, make_text_encoder
from ..text.schema import build_column_name_embeddings
from ..utils.paths import schema_cache_path
from .datamodule import MoREDataModule
from .loop import MoRELitModule


def _name_encoder(dataset: str, *, encoder: str = "hash", d_text: int = 64):
    """An ``encode(texts, kind) -> Tensor`` callable for the column-name table (cached for real models).

    ``hash`` is the dependency-free dev/test encoder (``d_text`` sets its width). ``qwen`` / a registry
    label / a raw HF id wrap the frozen model in an on-disk :class:`EmbeddingCache` so its single pass
    is idempotent across runs.
    """
    if encoder == "hash":
        return HashEncoder(dim=d_text)
    safe = encoder.replace("/", "__")
    cache_path = schema_cache_path(dataset, safe)
    if encoder == "qwen":
        return EmbeddingCache(QwenEncoder("Qwen/Qwen3-Embedding-4B"), cache_path)
    return EmbeddingCache(make_text_encoder(encoder), cache_path)


def name_embeddings(bundle, dataset: str, *, encoder: str = "hash", d_text: int = 64):
    """Frozen ``[C, d_text]`` column-name table for ``bundle`` (built once; cached for real encoders).

    Column names are embedded as instruction **queries** (``cache.QUERY_INSTRUCTION``): instruction-tuned
    encoders (qwen / harrier) are trained to take a one-sentence task instruction on the *query* side, so
    routing on a value-free schema representation needs it here (documents get no instruction)."""
    enc = _name_encoder(dataset, encoder=encoder, d_text=d_text)
    return build_column_name_embeddings(bundle, enc, kind="query")


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


def target_clamp(task, lo_pct: float = 2.0, hi_pct: float = 98.0) -> tuple[float, float]:
    """TRAIN-split target percentiles used to clip regression PREDICTIONS at eval time.

    Copied from GelGT's reference implementation (`main_node_ddp.py`), which clips predictions to the
    train target's 2nd/98th percentile before scoring. It is an eval-time transform only — training
    never sees it — and it is aimed squarely at MAE on heavy-tailed targets, where a handful of
    runaway predictions dominate the mean absolute error. rel-trial/study-adverse has train skew
    39.1, so this is the failure mode it targets.

    Returns raw-unit bounds (NOT standardized), so callers must clip after de-standardizing.
    """
    import numpy as np

    y = task.get_table("train").df[task.target_col].to_numpy(dtype="float64")
    y = y[~np.isnan(y)]
    lo, hi = np.percentile(y, [lo_pct, hi_pct])
    return float(lo), float(hi)


class _BestValState(pl.Callback):
    """Keep the best-val model weights (in memory, on CPU — no disk checkpoint) and the val metrics at
    that epoch, so a run reports / evaluates its best-val model rather than the last epoch."""

    def __init__(self, monitor: str, mode: str):
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self.best_state: dict | None = None
        self.best_metrics: dict = {}

    def on_validation_end(self, trainer, pl_module) -> None:
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        score = float(score)
        if score != score:                       # NaN guard (e.g. single-class val subsample)
            return
        better = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if better:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}
            self.best_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                                 if k.startswith("val/")}


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
    early_stop: bool = True,
    patience: int = 3,
    regression_loss: str = "mse",
    binary_loss: str = "bce",
    optimizer: str = "adamw",
    target_scaling: str = "zscore",
    clamp_pct: float | None = None,
    accumulate_grad_batches: int = 1,
):
    """Train one run on a PREBUILT bundle + name table (the ablation reuses these across arms so the
    graph and the frozen name embeddings are built once)."""
    from ..utils.seeding import seed_everything

    seed_everything(seed)
    kind = task_kind(task)
    # `raw` trains on the unstandardized target (GelGT does this). Under L1 that is only a per-task
    # rescale of the loss — i.e. a different EFFECTIVE lr per task — not a different optimum; under
    # MSE it also changes the curvature. Either way it is not free of the lr sweep, so read the two
    # together.
    if kind != "regression" or target_scaling == "raw":
        mean, std = 0.0, 1.0
    elif target_scaling == "zscore":
        mean, std = target_stats(task)
    else:
        raise ValueError(f"unknown target_scaling {target_scaling!r}; expected 'zscore' or 'raw'")
    clamp = target_clamp(task, clamp_pct, 100.0 - clamp_pct) if (
        kind == "regression" and clamp_pct) else None
    module = MoRELitModule(
        bundle, name_emb, task.entity_table,
        task_type=kind, target_mean=mean, target_std=std,
        model_kwargs=model_kwargs, route_on=route_on, lambda_ortho=lambda_ortho,
        lr=lr, weight_decay=weight_decay, seq_len=seq_len, max_fk=max_fk,
        regression_loss=regression_loss, binary_loss=binary_loss,
        optimizer=optimizer, clamp=clamp,
    )
    dm = MoREDataModule(bundle, task, num_neighbors=num_neighbors, batch_size=batch_size,
                        num_workers=num_workers)
    # Best-val model selection: validate every epoch, keep the best-val weights in memory, and (optionally)
    # early-stop on the primary metric. The held-out TEST eval (eval/test_eval.py) then scores the best-val
    # model. No disk checkpoint — MoRELitModule.__init__ takes the (unserializable) bundle + name table.
    monitor, mode = ("val/auroc", "max") if kind == "binary" else ("val/mae", "min")
    best = _BestValState(monitor, mode)
    callbacks: list = [best]
    if early_stop:
        callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                                                    strict=False))
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices=1,
        logger=logger, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, log_every_n_steps=20, callbacks=callbacks,
        num_sanity_val_steps=0,
        # Equivalent to a larger batch for SUM-decomposable losses (BCE, MSE, L1) but NOT for the
        # pairwise AUC surrogate, whose pairs are formed within a batch — accumulating 8x64 gives the
        # gradient noise of 64, not 512. Only a real batch increase helps there.
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,                     # stabilizes training (hidden-router + HMoE explodes to
                                                   # NaN without it); a no-op for arms whose grads stay < 1
        check_val_every_n_epoch=1,                 # best-val selection + early stopping need per-epoch val
        limit_train_batches=limit_train_batches or 1.0,
        limit_val_batches=limit_val_batches or 1.0,
    )
    trainer.fit(module, dm)
    module.clamp = clamp          # carried to the TEST eval (eval/test_eval.py)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    if best.best_state is not None:
        module.load_state_dict(best.best_state)      # restore best-val weights for downstream TEST eval
        metrics.update(best.best_metrics)            # report best-val (not last-epoch) val metrics
    return module, metrics


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
