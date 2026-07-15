"""SetJoin supervised training: a thin fork of the MoRE Lightning scaffolding.

`SetJoinLitModule` differs from `MoRELitModule` in exactly three ways: it instantiates `SetJoin`
(no routing arm / no ortho loss), converts raw sampled minibatches with `to_join_batch` instead of
`to_cell_batch`, and drops the aux term from the loss (SetJoin's aux is always 0). Metric names
(`val/auroc`, `val/mae`, ...), regression standardization (TRAIN-stats z-scoring, de-standardized
metrics), the nan-grad guard, and best-val/early-stop monitors are IDENTICAL — `_BestValState` and
`EarlyStopping` read the same keys. `train_prebuilt_setjoin` mirrors `finetune.train_prebuilt` with the
SetJoin per-edge-type fanout dict as the sampler's `num_neighbors`.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from ..data.graph import GraphBundle
from ..eval.metrics import binary_metrics, regression_metrics
from ..train.datamodule import MoREDataModule
from ..train.finetune import _BestValState, target_stats, task_kind
from ..train.losses import task_loss
from .collate import to_join_batch
from .model import SetJoin
from .paths import setjoin_neighbors


class SetJoinLitModule(pl.LightningModule):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb,
        entity_table: str,
        *,
        task_type: str = "binary",
        target_mean: float = 0.0,
        target_std: float = 1.0,
        model_kwargs: dict | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        wide_len: int = 128,
        set_size: int = 128,
        lambda_ortho: float = 0.5,
    ):
        super().__init__()
        self.bundle = bundle
        self.entity_table = entity_table
        self.task_type = task_type
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) > 0 else 1.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.wide_len = wide_len
        self.set_size = set_size
        self.lambda_ortho = lambda_ortho
        mk = dict(model_kwargs or {})
        mk.pop("d_text", None)               # d_text is inferred from name_emb inside the encoder
        self.model = SetJoin(bundle, name_emb, entity_table, **mk)
        self._val: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, jb):
        logits, _aux = self.model(jb)
        return logits.squeeze(-1)            # [B] raw output (logit for binary, std-space for regression)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        jb = to_join_batch(batch, self.bundle, self.entity_table,
                           wide_len=self.wide_len, set_size=self.set_size)
        return jb.to(device)

    def _standardize(self, target):
        if self.task_type == "regression":
            return (target - self.target_mean) / self.target_std
        return target

    def training_step(self, jb, _idx):
        logits, aux = self.model(jb)
        y = self._standardize(jb.target)
        loss = task_loss(logits.squeeze(-1), y, jb.has_target, self.task_type) \
            + self.lambda_ortho * aux                       # router orthogonality (0 on the dense arm)
        self.log("train/loss", loss, prog_bar=True, batch_size=int(jb.num_seeds))
        return loss

    def on_before_optimizer_step(self, optimizer):
        # zero any non-finite gradient so an occasional NaN/inf can't poison the weights (a no-op when
        # all grads are finite; same guard as the MoRE module).
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def validation_step(self, jb, _idx):
        out = self(jb)
        m = jb.has_target
        if m.any():
            self._val.append((out[m].detach().cpu(), jb.target[m].detach().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        out = torch.cat([a for a, _ in self._val])
        target = torch.cat([b for _, b in self._val])
        if self.task_type == "regression":
            pred = out * self.target_std + self.target_mean        # de-standardize to original units
            metrics = regression_metrics(pred, target)
            prog = ("mae",)
        else:
            metrics = binary_metrics(out, target)
            prog = ("ap", "auroc")
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k in prog))
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def train_prebuilt_setjoin(
    bundle,
    task,
    name_emb,
    *,
    model_kwargs: dict | None = None,
    fanout: int = 64,
    wide_len: int = 128,
    set_size: int = 128,
    lambda_ortho: float = 0.5,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 30,
    accelerator: str = "auto",
    logger=False,
    seed: int = 0,
    num_workers: int = 0,
    limit_train_batches: float | int | None = None,
    limit_val_batches: float | int | None = None,
    early_stop: bool = True,
    patience: int = 3,
):
    """Train one SetJoin run on a PREBUILT bundle + name table (mirror of finetune.train_prebuilt)."""
    from ..utils.seeding import seed_everything

    seed_everything(seed)
    kind = task_kind(task)
    mean, std = target_stats(task) if kind == "regression" else (0.0, 1.0)
    module = SetJoinLitModule(
        bundle, name_emb, task.entity_table,
        task_type=kind, target_mean=mean, target_std=std,
        model_kwargs=model_kwargs, lr=lr, weight_decay=weight_decay,
        wide_len=wide_len, set_size=set_size, lambda_ortho=lambda_ortho,
    )
    dm = MoREDataModule(bundle, task, num_neighbors=setjoin_neighbors(bundle, fanout),
                        batch_size=batch_size, num_workers=num_workers)
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
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        limit_train_batches=limit_train_batches or 1.0,
        limit_val_batches=limit_val_batches or 1.0,
    )
    trainer.fit(module, dm)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    if best.best_state is not None:
        module.load_state_dict(best.best_state)      # restore best-val weights for downstream TEST eval
        metrics.update(best.best_metrics)            # report best-val (not last-epoch) val metrics
    return module, metrics
