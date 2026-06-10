"""LightningModule wrapping HALOS-minimal + LightningDataModule over the temporal sampler. Phase-4 STUB.

Harness kept = Lightning + Hydra + W&B (build HALOS on top). The DataModule yields collated TokenBatches
from gloss.data; the module fuses cached DocCard embeddings (gathered by col_global_id) — no LM in the loop.
"""
from __future__ import annotations


class HALOSLitModule:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 4 — build after Phase 3 model.")


class RelationalDataModule:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 4.")
