"""SetJoin backbone sweep bookkeeping (hermetic: grid shape, indexing, batch heuristic,
aggregation/adoption logic, idempotent resume — no training)."""
from __future__ import annotations

import json

from gloss.setjoin.grid import (D512_PROBES, PARAM_BUDGET, TASKS, aggregate_grid, arch_grid,
                                init_batch, jobs, run_index)


def test_arch_grid_is_18_configs_axes_and_dff():
    grid = arch_grid()
    assert len(grid) == 18
    assert sum(1 for c in grid if c["use_shared"]) == 9              # 8 product + 1 probe
    assert sum(1 for c in grid if c["d_model"] == 512) == len(D512_PROBES) == 2
    for c in grid:
        assert c["d_ff"] in (2 * c["d_model"], 4 * c["d_model"])
        assert c["n_wide_layers"] == c["n_set_layers"]
        assert c["enc_channels"] == min(c["d_model"], 256)           # stype encoders never past 256
        assert c["route_on"] == "signature" and c["num_experts"] == 4 and c["k"] == 2
        assert c["d_sig"] == 64 and c["n_heads"] == 4 and c["n_pma"] == 4
    # d512 probes are depth-2 ff×2 only
    for c in grid:
        if c["d_model"] == 512:
            assert c["n_wide_layers"] == 2 and c["d_ff"] == 1024
    assert len({tuple(sorted(c.items())) for c in grid}) == 18       # all distinct


def test_jobs_indexing_486_stable_and_unique():
    J = jobs(seeds=3)
    assert len(J) == 18 * 9 * 3 == 486
    assert len({(ci, ds, tk, s) for ci, _, ds, tk, s in J}) == 486
    # stable: config-major, then TASKS order, then seed
    assert J[0][0] == 0 and J[0][2:] == (*TASKS[0], 0)
    assert J[27][0] == 1                                             # 9 tasks × 3 seeds per config
    assert jobs(seeds=3) == J                                        # deterministic


def test_init_batch_heuristic():
    grid = arch_grid()
    def cfg_of(dm, dep, mult, sh):
        return next(c for c in grid if (c["d_model"], c["n_wide_layers"],
                                        c["d_ff"] // c["d_model"], c["use_shared"]) == (dm, dep, mult, sh))
    assert init_batch(cfg_of(128, 2, 2, False)) == 128
    assert init_batch(cfg_of(128, 4, 4, True)) == 128
    assert init_batch(cfg_of(256, 4, 4, False)) == 64                # 30.2M borderline config
    assert init_batch(cfg_of(256, 4, 4, True)) == 64
    assert init_batch(cfg_of(512, 2, 2, False)) == 64                # d512 probe
    for c in grid:
        assert init_batch(c) in (64, 128)


def _rec(ci, ds, tk, seed, ttype, n_params, **m):
    cfg = arch_grid()[ci]
    return {"config_idx": ci, "dataset": ds, "task": tk, "seed": seed, "task_type": ttype,
            "batch_size": 128, "n_params": n_params, **cfg, **m}


def test_aggregate_grid_best_baselines_and_adoption(tmp_path):
    # cfg#0 (small, under budget) wins driver-dnf strongly; cfg#16 (d512 probe, over budget)
    # wins driver-top3 — the adopted line must skip it.
    recs = []
    for s in range(3):
        recs.append(_rec(0, "rel-f1", "driver-dnf", s, "binary", 3_200_000, test_roc_auc=0.85))
        recs.append(_rec(1, "rel-f1", "driver-dnf", s, "binary", 9_000_000, test_roc_auc=0.80))
        recs.append(_rec(16, "rel-f1", "driver-top3", s, "binary", 36_300_000, test_roc_auc=0.95))
        recs.append(_rec(0, "rel-f1", "driver-top3", s, "binary", 3_200_000, test_roc_auc=0.84))
        recs.append(_rec(0, "rel-f1", "driver-position", s, "regression", 3_200_000,
                         test_mae=1.0, test_nmae=0.40))
        recs.append(_rec(1, "rel-f1", "driver-position", s, "regression", 9_000_000,
                         test_mae=1.2, test_nmae=0.48))
    for i, r in enumerate(recs):
        (tmp_path / f"{i:05d}.json").write_text(json.dumps(r))
    out = aggregate_grid(tmp_path)
    assert "driver-dnf" in out and "best cfg#0" in out
    assert "BEAT✅" in out                                           # 85 > RT 78.7 on driver-dnf
    assert "MoRE-best" in out
    assert "[>30M]" in out                                           # the d512 probe is flagged
    assert "winner (unrestricted): cfg#16" not in out.split("ADOPTED")[0].split("winner")[0]
    # cfg#16 wins mean rank (1.0 on its only task... ) — adoption must respect the budget
    assert "ADOPTED BACKBONE" in out
    adopted_line = next(l for l in out.splitlines() if l.startswith("ADOPTED"))
    assert "cfg#16" not in adopted_line
    assert "winner's curse" in out


def test_run_index_is_idempotent(tmp_path):
    # a pre-existing JSON must short-circuit before any torch/relbench import cost
    (tmp_path / "00003.json").write_text(json.dumps({"config_idx": 0, "done": True}))
    rec = run_index(3, seeds=3, epochs=1, num_workers=0, out_dir=tmp_path)
    assert rec == {"config_idx": 0, "done": True}
    # out-of-range index is a no-op
    assert run_index(10_000, seeds=3, epochs=1, num_workers=0, out_dir=tmp_path) == {}
