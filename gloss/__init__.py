"""gloss — MoRE: RT's cell-token relational-transformer substrate + a Mixture-of-Experts FFN routed on
each cell's value-free relational signature (column-name embedding + modality + recency). Route on
semantics, transform the content.

Package layout:
  data/   relational graph substrate + leakage-safe temporal sampler + cell-token collate
  text/   frozen text encoders + cache + the per-column schema-name embedding table (router input)
  model/  RT cell encoder, relational-attention substrate (+ MoE FFN), signature, heads
  train/  Lightning loop, datamodule, losses
  eval/   metrics, LightGBM floor, the routing-signal ablation runner
  utils/  seeding, config, logging

The retired DOC-RT (doc FiLM) and HALOS (doc-generated temporal geometry) designs live under archive/.
"""

__version__ = "0.3.0"
