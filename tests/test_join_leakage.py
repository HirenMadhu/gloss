"""SetJoin leakage: no wide cell or set element may carry a timestamp after its seed's time."""
from __future__ import annotations

from gloss.setjoin.collate import to_join_batch

from ._join_fixtures import ENTITY, chain_bundle, make_chain_batch, make_hop2_batch
from .conftest import rel_f1_available


def _n_leaks(jb) -> int:
    st = jb.seed_time.unsqueeze(1)
    wide = ((jb.wide_row_time > st) & jb.wide_is_timed & ~jb.wide_is_pad).sum()
    elem = ((jb.elem_row_time > st) & jb.elem_is_timed & jb.elem_mask).sum()
    return int(wide) + int(elem)


def test_no_leakage_synth():
    jb = to_join_batch(make_chain_batch(seed_time=100.0), chain_bundle(), ENTITY,
                       wide_len=16, set_size=8)
    assert _n_leaks(jb) == 0


def test_planted_leak_is_detectable():
    # a payment at t=150 > seed 100 must be flagged by the element predicate
    jb = to_join_batch(make_chain_batch(seed_time=100.0, payment_times=(10.0, 150.0, 30.0)),
                       chain_bundle(), ENTITY, wide_len=16, set_size=8)
    assert _n_leaks(jb) >= 1


def test_no_leakage_hop2():
    jb = to_join_batch(make_hop2_batch(seed_time=100.0), chain_bundle(), ENTITY,
                       wide_len=16, set_size=8, hop2=True)
    assert int((jb.elem_hop[jb.elem_mask] == 2).sum()) > 0     # hop-2 elements are actually present
    assert _n_leaks(jb) == 0


def test_planted_hop2_leak_is_detectable():
    # a GRANDCHILD at t=150 > seed 100 must be flagged by the same element predicate
    jb = to_join_batch(make_hop2_batch(seed_time=100.0, refund_time=150.0), chain_bundle(), ENTITY,
                       wide_len=16, set_size=8, hop2=True)
    assert _n_leaks(jb) >= 1


@rel_f1_available
def test_no_leakage_rel_f1():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.setjoin.paths import setjoin_neighbors

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout=16),
                         batch_size=16)
    jb = to_join_batch(next(iter(loader)), bundle, task.entity_table, wide_len=128, set_size=64)
    assert _n_leaks(jb) == 0
    assert int(jb.elem_mask.sum()) > 0                     # drivers do have children (results etc.)
