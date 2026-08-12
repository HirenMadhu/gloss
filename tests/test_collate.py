"""Hermetic tests for the cell-token collate + the RT relational masks/substrate."""
from __future__ import annotations

import torch

from gloss.data.collate import column_vocab, feature_col_names, to_cell_batch

from .conftest import ENTITY, make_synth_batch, synthetic_bundle


def _cb(seq_len=16, max_fk=2):
    return to_cell_batch(make_synth_batch(), synthetic_bundle(), ENTITY, seq_len=seq_len, max_fk=max_fk)


def test_cell_batch_contract():
    cb = _cb()
    assert cb.num_seeds == 2 and cb.seq_len == 16
    real = ~cb.is_padding
    # per seed: 1 user cell (1 col) + 2 events * 2 cols = 5 real cells
    assert real.sum(1).tolist() == [5, 5]
    # exactly one seed (user) row per seed -> 1 seed cell each (user has 1 feature col)
    assert cb.is_seed_cell.sum().item() == 2
    # column ids valid on real cells, -1 on pad
    assert (cb.col_idxs[real] >= 0).all()
    assert (cb.col_idxs[~real] == -1).all()
    # events are timed and <= seed_time; users untimed
    assert (cb.row_time[cb.is_timed] <= 100.0).all()


def test_feature_col_order_is_sorted():
    cb = _cb()
    tf = cb.tf_dict["event"]
    assert feature_col_names(tf) == sorted(feature_col_names(tf))
    vocab = column_vocab(synthetic_bundle())
    # same column of same table -> one id; distinct columns -> distinct ids
    assert vocab[("event", "e_x")] != vocab[("event", "e_y")]
    assert vocab[("event", "e_x")] != vocab[("user", "u_attr")]


def test_f2p_neighbors_point_to_parent_user():
    cb = _cb()
    # a cell of an event row must list its parent user's local node idx as an f2p neighbor
    event_tid = synthetic_bundle().node_type_id["event"]
    user_tid = synthetic_bundle().node_type_id["user"]
    b = 0
    user_nodes = set(cb.node_idxs[b][cb.table_idxs[b] == user_tid].tolist())
    ev_cells = (cb.table_idxs[b] == event_tid).nonzero(as_tuple=True)[0]
    assert len(ev_cells) > 0
    for s in ev_cells.tolist():
        nbrs = [x for x in cb.f2p_nbr_idxs[b, s].tolist() if x >= 0]
        assert nbrs and set(nbrs).issubset(user_nodes)


