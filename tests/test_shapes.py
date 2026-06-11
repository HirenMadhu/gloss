"""Phase 0 — dense batch shape/dtype contracts."""
from __future__ import annotations

import torch

from gloss.data.collate import to_gloss_batch
from tests.conftest import rel_f1_available


def test_dense_shapes_and_dtypes(dualfk_batch, dualfk_bundle):
    gb = to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)
    B, N = gb.num_seeds, gb.n_max
    # node-level [B, N]
    for t in (gb.node_type_id, gb.pad_mask, gb.is_seed, gb.is_timed, gb.row_time, gb.n_id):
        assert t.shape == (B, N)
    # pairwise [B, N, N]
    for t in (gb.attend_mask, gb.metapath_id, gb.fk_role_id, gb.dt, gb.tau, gb.temporal_valid):
        assert t.shape == (B, N, N)
    # per-seed [B]
    assert gb.seed_time.shape == (B,) and gb.t_ctx.shape == (B,)
    # dtypes: time math is float64 (exact scale-equivariance), ids long, masks bool
    assert gb.row_time.dtype == torch.float64 and gb.tau.dtype == torch.float64
    assert gb.dt.dtype == torch.float64 and gb.seed_time.dtype == torch.float64
    assert gb.node_type_id.dtype == torch.long and gb.metapath_id.dtype == torch.long
    assert gb.attend_mask.dtype == torch.bool and gb.temporal_valid.dtype == torch.bool


def test_placement_covers_all_real_nodes(dualfk_batch, dualfk_bundle):
    gb = to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)
    # placement maps every node-type row to a (seg, localpos) inside the grid; together they tile pad_mask
    grid = torch.zeros(gb.num_seeds, gb.n_max, dtype=torch.long)
    for nt, (seg, pos) in gb.placement.items():
        grid[seg, pos] += 1
    assert torch.equal(grid.bool(), gb.pad_mask)
    assert grid.max().item() == 1  # no double placement


@rel_f1_available
def test_real_rel_f1_shapes():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=[8, 8], batch_size=8, shuffle=False)
    gb = to_gloss_batch(next(iter(loader)), bundle, task.entity_table, max_nodes=4096)
    B, N = gb.num_seeds, gb.n_max
    assert B == 8 and N >= 1
    assert gb.attend_mask.shape == (B, N, N)
    # at least the seeds are present and flagged
    assert gb.is_seed.sum().item() == B
