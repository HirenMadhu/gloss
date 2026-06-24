"""gloss — DOC-RT: RT's cell-token relational-transformer substrate + documentation-conditioned
cell encoding (FiLM). A value is read in light of what its column's documentation *means*, not just
what it is *named*.

Package layout:
  data/   relational graph substrate + leakage-safe temporal sampler + cell-token collate
  docs/   prose doc corpus loader + grounding (chunk/embed/retrieve/pool) + frozen embedding cache
  model/  documentation-conditioned cell encoder (FiLM), RT relational-attention substrate, heads
  eval/   metrics, LightGBM floor, the four-regime headline runner
  utils/  seeding, config, logging

The retired HALOS design (doc-generated temporal geometry, tau kernels) lives under archive/halos/.
"""

__version__ = "0.2.0"
