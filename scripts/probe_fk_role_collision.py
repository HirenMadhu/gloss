"""probe_fk_role_collision.py — is changes.md P0.1 fixing a LIVE bug or a latent one?

The retired vocabulary keyed `fk_role_id` on the FK **column name alone**
(``cols = sorted({canonical_relation(rel) for (_, rel, _) in edge_types})``). Its docstring's
claim that "two FKs into one table stay distinct" was true but answered a different question: two
FK columns that happen to *share a name* in *different child tables* collided onto one role id.

changes.md P0.1 specifies role ids over the triple ``(child_table, fk_column, parent_table)``, which
is what `graph.py:_build_vocabs` now builds. This script reports, per database, the role count under
each keying — the gap is the size of the bug that was fixed — and then **checks the live vocabulary**:
it rebuilds the PyG edge types the way ``make_pkey_fkey_graph`` names them and asserts
``_build_vocabs`` yields one id per FK edge.

Reads only the RelBench schema (no graph build, no sampling).

    .venv/bin/python scripts/probe_fk_role_collision.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def probe(dataset: str) -> None:
    from relbench.datasets import get_dataset

    from gloss.data.graph import _build_vocabs

    db = get_dataset(dataset, download=True).get_db(upto_test_timestamp=False)

    by_col: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    triples: set[tuple[str, str, str]] = set()
    node_types: set[str] = set(db.table_dict)
    edge_types: list[tuple[str, str, str]] = []
    for child, table in db.table_dict.items():
        for fk_col, parent in table.fkey_col_to_pkey_table.items():
            by_col[fk_col].append((child, fk_col, parent))
            triples.add((child, fk_col, parent))
            # exactly how relbench.modeling.graph.make_pkey_fkey_graph names the two directions
            edge_types.append((child, f"f2p_{fk_col}", parent))
            edge_types.append((parent, f"rev_f2p_{fk_col}", child))

    n_col_keyed = len(by_col)
    n_triple_keyed = len(triples)
    collisions = {c: v for c, v in by_col.items() if len(v) > 1}

    _, fk_role_id, _ = _build_vocabs(sorted(node_types), edge_types)
    n_built = len(fk_role_id)

    print(f"=== {dataset}")
    print(f"  FK edges (child, col, parent)      {n_triple_keyed}")
    print(f"  distinct role ids, RETIRED (col)   {n_col_keyed}")
    print(f"  distinct role ids, P0.1 (triple)   {n_triple_keyed}")
    print(f"  distinct role ids, _build_vocabs   {n_built}"
          f"   {'OK' if n_built == n_triple_keyed else 'MISMATCH'}")
    assert n_built == n_triple_keyed, "graph.py role vocabulary does not match the schema triples"
    assert set(fk_role_id) == triples, "role keys are not the (child, col, parent) triples"
    assert sorted(fk_role_id.values()) == list(range(1, n_built + 1)), "role ids are not 1..K"
    if collisions:
        print(f"  LIVE COLLISION — {len(collisions)} column name(s) shared across child tables:")
        for c, v in sorted(collisions.items()):
            joined = ", ".join(f"{ch}.{co}->{pa}" for ch, co, pa in sorted(v))
            print(f"    {c!r}: {joined}")
        print(f"  VERDICT: P0.1 is a REAL behaviour change here "
              f"({n_col_keyed} -> {n_triple_keyed} roles)")
    else:
        print("  VERDICT: no collision on this DB; P0.1 tightens the definition but the ids "
              "are already 1:1 (still required — it is a correctness guarantee, not a no-op)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    for ds in args.datasets:
        try:
            probe(ds)
        except Exception as e:
            print(f"=== {ds}: FAILED — {type(e).__name__}: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
