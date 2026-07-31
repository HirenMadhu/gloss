"""The supervised training LightningModule wrapping MoRE.

Converts each raw sampled minibatch into a dense ``CellBatch`` in ``transfer_batch_to_device``. Handles
both binary classification (BCE; val AP/AUROC/logloss) and regression (MSE on a **standardized** target;
val MAE/RMSE/R²). Regression targets are z-scored with TRAIN-split stats (``target_mean``/``target_std``)
for training stability and the metrics/predictions are de-standardized back to original units. Leaderboard
TEST eval is a separate concern in ``eval/test_eval.py`` (which reads these same stats off the module).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from ..data.collate import to_cell_batch
from ..data.graph import GraphBundle
from ..eval.metrics import binary_metrics, regression_metrics
from ..model.more import MoRE
from .losses import task_loss


class MoRELitModule(pl.LightningModule):
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
        route_on: str = "dense",
        lambda_ortho: float = 0.5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        seq_len: int = 1024,
        max_fk: int = 5,
        max_rows: int | None = None,
        regression_loss: str = "mse",
        binary_loss: str = "bce",
        optimizer: str = "adamw",
        clamp: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.entity_table = entity_table
        self.task_type = task_type
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) > 0 else 1.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.max_fk = max_fk
        self.max_rows = max_rows          # None = fit the row axis R to each batch (see collate)
        self.lambda_ortho = lambda_ortho
        self.regression_loss = regression_loss   # each is ignored for the other task
        self.binary_loss = binary_loss           # kind; see losses.task_loss
        self.optimizer_name = optimizer
        # Raw-unit prediction bounds (GelGT-style). Applied to VAL metrics so best-val
        # selection scores the model the same way the test eval will.
        self.clamp = clamp
        mk = dict(model_kwargs or {})
        mk.pop("d_text", None)               # d_text is inferred from name_emb inside the encoder
        self.model = MoRE(bundle, name_emb, route_on=route_on, **mk)
        self._val: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, cb):
        logits, _aux = self.model(cb)
        return logits.squeeze(-1)            # [B] raw output (logit for binary, std-space for regression)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        cb = to_cell_batch(batch, self.bundle, self.entity_table, seq_len=self.seq_len,
                           max_fk=self.max_fk, max_rows=self.max_rows)
        return cb.to(device)

    def _standardize(self, target):
        if self.task_type == "regression":
            return (target - self.target_mean) / self.target_std
        return target

    def training_step(self, cb, _idx):
        logits, aux = self.model(cb)
        y = self._standardize(cb.target)
        loss = task_loss(logits.squeeze(-1), y, cb.has_target, self.task_type,
                         regression_loss=self.regression_loss,
                         binary_loss=self.binary_loss) + self.lambda_ortho * aux
        self.log("train/loss", loss, prog_bar=True, batch_size=int(cb.num_seeds))
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Zero any non-finite gradient so an occasional NaN/inf can't poison the weights (the step just
        # skips those params). Some arms — notably HMoE with hidden routing — produce NaN grads while the
        # forward stays finite, which gradient clipping can't fix (clipping a NaN grad stays NaN). A no-op
        # when all grads are finite. If an arm needs this constantly, its numbers reflect an unstable arm.
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def validation_step(self, cb, _idx):
        out = self(cb)
        m = cb.has_target
        if m.any():
            self._val.append((out[m].detach().cpu(), cb.target[m].detach().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        out = torch.cat([a for a, _ in self._val])
        target = torch.cat([b for _, b in self._val])
        if self.task_type == "regression":
            pred = out * self.target_std + self.target_mean        # de-standardize to original units
            if self.clamp is not None:
                pred = pred.clamp(self.clamp[0], self.clamp[1])   # score val as test will be scored
            metrics = regression_metrics(pred, target)
            prog = ("mae",)
        else:
            metrics = binary_metrics(out, target)
            prog = ("ap", "auroc")
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k in prog))
        self._val.clear()

    def configure_optimizers(self):
        # GelGT uses plain Adam (wd 1e-5); ours has always been AdamW (wd 0.01). AdamW decouples the
        # decay from the gradient, so at equal `weight_decay` the two are NOT the same optimizer.
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name != "adamw":
            raise ValueError(f"unknown optimizer {self.optimizer_name!r}; expected 'adamw' or 'adam'")
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
