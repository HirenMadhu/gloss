"""Phase 4 — the supervised training LightningModule wrapping HALOS.

Holds the (frozen, cached) grounding + the precomputed FK-doc `doc_per_metapath`; converts each raw
sampled minibatch into a dense `GlossBatch` in `transfer_batch_to_device`. Val metrics = AP/AUROC
(test labels are masked by relbench, so we only validate).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from ..data.collate import to_gloss_batch
from ..data.graph import GraphBundle
from ..docs.grounding import GroundingResult
from ..eval.metrics import binary_metrics
from ..model.halos import HALOS
from .losses import masked_bce


class HALOSLitModule(pl.LightningModule):
    def __init__(
        self,
        bundle: GraphBundle,
        grounding: GroundingResult,
        doc_per_metapath: torch.Tensor | None,
        entity_table: str,
        *,
        model_kwargs: dict | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.bundle = bundle
        self.entity_table = entity_table
        self.grounding = grounding              # plain attribute (not a Parameter); gathered per-batch
        self._doc_mp = doc_per_metapath         # moved to device lazily in forward
        self.lr = lr
        self.weight_decay = weight_decay
        mk = dict(model_kwargs or {})
        mk.setdefault("d_text", grounding.d_text)
        self.model = HALOS(bundle, **mk)
        self._val: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, gb):
        doc_mp = self._doc_mp.to(self.device) if self._doc_mp is not None else None
        return self.model(gb, self.grounding, doc_per_metapath=doc_mp).squeeze(-1)   # [B]

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        gb = to_gloss_batch(batch, self.bundle, self.entity_table)
        return gb.to(device)

    def training_step(self, gb, _idx):
        logits = self(gb)
        loss = masked_bce(logits, gb.target, gb.has_target)
        self.log("train/loss", loss, prog_bar=True, batch_size=int(gb.num_seeds))
        return loss

    def validation_step(self, gb, _idx):
        logits = self(gb)
        m = gb.has_target
        if m.any():
            self._val.append((logits[m].detach().cpu(), gb.target[m].detach().cpu()))

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
