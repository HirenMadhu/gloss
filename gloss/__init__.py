"""gloss — HALOS: a node-level temporal graph transformer whose attention geometry is
generated from human-style prose schema documentation (v3: documentation-conditioned geometry).

Package layout (Phases 0-3 implemented this session):
  data/   relational graph substrate + leakage-safe temporal sampler + dense collate
  docs/   prose doc corpus loader + grounding (chunk/embed/retrieve/pool) + frozen embedding cache
  model/  doc-conditioned node encoder, dimensionless-time encoding, the doc-generated bias
          generator (the core operator), attention, encoder stack, heads
  eval/   geometry report (the readable-geometry exhibit)
  utils/  seeding, config, logging
"""

__version__ = "0.1.0"
