"""SetJoin — "one big joined table" + set transformer (standalone direction, branch `setjoin`).

Per seed: a **wide m2o-flattened row** (seed ⋈ parents ⋈ grandparents — exact, duplication-free) plus
**one table-tagged union set** of 1-hop child rows (the rows a SQL join would have duplicated the seed
against). Specs: setjoin_idea.md / setjoin_implementation.md at the repo root. Reuses MoRE's data
pipeline, `CellEncoder.encode_type`, and train/eval scaffolding; modifies no MoRE file.
"""
