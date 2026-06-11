"""Phase 4 — Lightning DataModule over the leakage-safe disjoint temporal loaders.

The loaders yield raw PyG `HeteroData` minibatches; conversion to the dense `GlossBatch` happens in the
LightningModule's `transfer_batch_to_device` (it needs the bundle + entity table). num_workers=0 keeps
the sampler/transform simple to pickle.
"""
from __future__ import annotations

import pytorch_lightning as pl

from ..data.graph import GraphBundle, make_loader


class HALOSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        bundle: GraphBundle,
        task,
        *,
        num_neighbors: list[int] | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.bundle = bundle
        self.task = task
        self.num_neighbors = num_neighbors or [12, 12]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _loader(self, split: str, shuffle: bool):
        return make_loader(
            self.bundle, self.task, split,
            num_neighbors=self.num_neighbors, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
