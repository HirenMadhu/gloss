"""Multi-horizon evaluation contracts: time-shift math, grid bookkeeping, k=0 equivalence, plot."""
from __future__ import annotations

import numpy as np

import gloss.eval.ablation as ablation
from gloss.eval.ablation import LEADERBOARD_TASKS
from gloss.setjoin.horizon import horizon_grid, plot_horizon_curves

from .conftest import rel_f1_available


def test_horizon_grid_is_9_tasks_x_seeds_x_2_models(monkeypatch):
    monkeypatch.setattr(ablation, "entity_tasks",
                        lambda ds: sorted(LEADERBOARD_TASKS.get(ds, ())))
    grid = horizon_grid(seeds=3)
    assert len(grid) == 54                                  # 9 tasks x 3 seeds x {setjoin, more}
    assert {c["signal"] for c in grid} == {"setjoin", "more"}
    assert len({tuple(sorted(c.items())) for c in grid}) == 54


def _rec(ds, task, model, seed, ttype, base, slope):
    metric = "roc_auc" if ttype == "binary" else "nmae"
    return {"dataset": ds, "task": task, "model": model, "seed": seed, "task_type": ttype,
            "horizons": {str(k): {metric: base + slope * k, "mae": 1.0} for k in range(10)}}


def test_plot_horizon_curves(tmp_path):
    records = [_rec("rel-f1", "driver-dnf", m, s, "binary", 0.8 - 0.02 * (m == "more"), -0.01)
               for m in ("setjoin", "more") for s in range(3)]
    records += [_rec("rel-f1", "driver-position", m, s, "regression", 0.45, 0.02)
                for m in ("setjoin", "more") for s in range(3)]
    png = plot_horizon_curves(records, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").exists() and (tmp_path / "curves.png").stat().st_size > 10_000


@rel_f1_available
def test_shifted_table_and_k0_equivalence():
    import torch
    from relbench.tasks import get_task

    from gloss.setjoin.eval import predict_split
    from gloss.setjoin.horizon import predict_shifted, shifted_task_table
    from gloss.setjoin.model import SetJoin
    from gloss.setjoin.paths import setjoin_neighbors
    from gloss.setjoin.train import SetJoinLitModule
    from gloss.utils.seeding import seed_everything

    from ._relf1 import bundle_and_task, name_table

    bundle, task = bundle_and_task()
    # shift math: times move back exactly k * timedelta; rows/labels/order untouched
    t0 = task.get_table("test").df[task.get_table("test").time_col]
    for k in (0, 3):
        sh = shifted_task_table(task, "test", k)
        assert len(sh.df) == len(t0)
        assert ((t0 - sh.df[sh.time_col]) == k * task.timedelta).all()

    # k=0 must reproduce the standard eval path exactly (same loader recipe, same alignment)
    seed_everything(0)
    module = SetJoinLitModule(bundle, name_table(), task.entity_table,
                              model_kwargs=dict(d_model=32, n_heads=2, n_wide_layers=1,
                                                n_set_layers=1, n_pma=2, dropout=0.0,
                                                route_on="dense"),
                              wide_len=64, set_size=32)
    cfg = dict(num_neighbors=setjoin_neighbors(bundle, 8), batch_size=256, num_workers=0,
               wide_len=64, set_size=32)
    p_std = predict_split(module, bundle, task, "test", fanout=8, wide_len=64, set_size=32,
                          batch_size=256, num_workers=0)
    p_h0 = predict_shifted(module, bundle, task, "test", 0, kind="setjoin", **cfg)
    assert np.allclose(p_std, p_h0, atol=1e-6)

    # longer horizons stay well-defined and (generically) differ from k=0
    p_h5 = predict_shifted(module, bundle, task, "test", 5, kind="setjoin", **cfg)
    assert np.isfinite(p_h5).all() and not np.allclose(p_h0, p_h5, atol=1e-6)