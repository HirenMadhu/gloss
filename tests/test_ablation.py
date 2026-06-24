"""Phase 4 — headline-gate bookkeeping (hermetic: config enumeration + seed aggregation, no training)."""
from __future__ import annotations

import math

from gloss.eval.ablation import REGIMES, aggregate, enumerate_configs, format_table


def test_enumerate_configs_covers_every_regime_and_seed():
    cfgs = enumerate_configs(seeds=5)
    assert len(cfgs) == 5 * len(REGIMES)
    assert {c["regime"] for c in cfgs} == set(REGIMES)
    assert {c["seed"] for c in cfgs} == set(range(5))
    # each (regime, seed) pair is unique
    assert len({(c["regime"], c["seed"]) for c in cfgs}) == len(cfgs)


def test_aggregate_computes_seed_mean_std_ci():
    records = [
        {"regime": "full", "seed": 0, "ap": 0.80, "auroc": 0.90, "logloss": 0.30},
        {"regime": "full", "seed": 1, "ap": 0.90, "auroc": 0.92, "logloss": 0.28},
        {"regime": "null", "seed": 0, "ap": 0.70, "auroc": 0.85, "logloss": 0.40},
    ]
    agg = aggregate(records, regimes=("full", "null"))
    mean, sd, ci, n = agg["full"]["ap"]
    assert n == 2
    assert math.isclose(mean, 0.85)
    assert math.isclose(sd, 0.0707106781, rel_tol=1e-6)        # sample stdev of {0.80, 0.90}
    assert math.isclose(ci, 1.96 * sd / math.sqrt(2), rel_tol=1e-9)
    # single-sample regime -> zero spread
    _m, sd1, ci1, n1 = agg["null"]["ap"]
    assert n1 == 1 and sd1 == 0.0 and ci1 == 0.0


def test_aggregate_ignores_missing_and_nan_metrics():
    records = [
        {"regime": "full", "seed": 0, "ap": 0.80},                 # no auroc key
        {"regime": "full", "seed": 1, "ap": float("nan")},         # NaN dropped
        {"regime": "full", "seed": 2, "ap": 0.60},
    ]
    mean, _sd, _ci, n = aggregate(records, regimes=("full",))["full"]["ap"]
    assert n == 2 and math.isclose(mean, 0.70)
    assert aggregate(records, regimes=("full",))["full"]["auroc"][3] == 0   # n==0, all missing


def test_format_table_reports_regimes_and_lift_vs_null():
    records = [
        {"regime": "full", "seed": 0, "ap": 0.80, "auroc": 0.90},
        {"regime": "null", "seed": 0, "ap": 0.70, "auroc": 0.85},
    ]
    out = format_table(records, regimes=("full", "null"))
    assert "HEADLINE" in out and "full" in out and "null" in out
    assert "Δ vs null" in out
    assert "ΔAUROC=+0.0500" in out      # 0.90 - 0.85
