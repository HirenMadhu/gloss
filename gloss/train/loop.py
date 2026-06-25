"""The supervised training LightningModule wrapping the RT model.

Converts each raw sampled minibatch into a dense ``CellBatch`` in ``transfer_batch_to_device``. Val
metrics = AP/AUROC/log_loss (relbench masks test labels, so we only validate here; leaderboard TEST
eval is a separate concern in ``eval/test_eval.py``).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from ..data.collate import to_cell_batch
from ..data.graph import GraphBundle
from ..eval.metrics import binary_metrics
from ..model.docrt import DOCRT
from .losses import masked_bce


class DOCRTLitModule(pl.LightningModule):
    def __init__(
        self,
        bundle: GraphBundle,
        name_emb,
        entity_table: str,
        *,
        model_kwargs: dict | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        seq_len: int = 1024,
        max_fk: int = 5,
    ):
        super().__init__()
        self.bundle = bundle
        self.entity_table = entity_table
        self.lr = lr
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.max_fk = max_fk
        mk = dict(model_kwargs or {})
        mk.pop("d_text", None)               # d_text is inferred from name_emb inside the encoder
        self.model = DOCRT(bundle, name_emb, **mk)
        self._val: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, cb):
        return self.model(cb).squeeze(-1)   # [B]

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        cb = to_cell_batch(batch, self.bundle, self.entity_table, seq_len=self.seq_len, max_fk=self.max_fk)
        return cb.to(device)

    def training_step(self, cb, _idx):
        logits = self(cb)
        loss = masked_bce(logits, cb.target, cb.has_target)
        self.log("train/loss", loss, prog_bar=True, batch_size=int(cb.num_seeds))
        return loss

    def validation_step(self, cb, _idx):
        logits = self(cb)
        m = cb.has_target
        if m.any():
            self._val.append((logits[m].detach().cpu(), cb.target[m].detach().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        logits = torch.cat([a for a, _ in self._val])
        target = torch.cat([b for _, b in self._val])
        for k, v in binary_metrics(logits, target).items():
            self.log(f"val/{k}", v, prog_bar=(k in ("ap", "auroc")))
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
