"""probe_relevent_time.py — diagnose the two rel-event time anomalies.

Two findings need explaining before rel-event results can be trusted:

1. tau = log1p(seed_time - row_time) reaches 23.08 on rel-event, i.e. Delta ~ 336 years.
   changes.md 3.1 asserts every database lands in tau in [0, 22] and 6 makes that a test.
2. 97 sampled rows had row_time > seed_time, which is the leakage assert.

Both are decidable from the raw RelBench tables plus the task tables — no graph build, no
sampling. Reports per-table timestamp ranges, sentinel/outlier counts, and (for 2) whether the
offending rows are the SEED entity rows themselves, which RelBench includes regardless of their
own timestamp.

    .venv/bin/python scripts/probe_relevent_time.py --dataset rel-event
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEC_PER_YEAR = 365.25 * 86400


def as_unix(series):
    """``to_unix_time`` returns a Tensor on some relbench versions and an ndarray on others."""
    from relbench.modeling.utils import to_unix_time

    u = to_unix_time(series)
    u = u.numpy() if hasattr(u, "numpy") else u
    return u.astype("float64")


def fmt_unix(v: float) -> str:
    import datetime as dt

    try:
        return dt.datetime.utcfromtimestamp(float(v)).strftime("%Y-%m-%d")
    except Exception:
        return f"<unrepresentable {v:g}>"


def probe(dataset: str, task_name: str) -> None:
    import numpy as np
    import pandas as pd
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    db = get_dataset(dataset, download=True).get_db(upto_test_timestamp=False)

    print(f"=== {dataset}: per-table timestamp ranges (as stored, then as UNIX seconds)")
    per_table = {}
    for name, table in db.table_dict.items():
        tc = table.time_col
        if tc is None:
            print(f"  {name:24} (untimed table)")
            continue
        s = table.df[tc]
        u = as_unix(s)
        per_table[name] = u
        n_nat = int(pd.isna(s).sum())
        print(f"  {name:24} n={len(u):>9}  min {fmt_unix(u.min())}  max {fmt_unix(u.max())}"
              f"  NaT={n_nat}")
        # sentinel / outlier detection: epoch-0-ish, negative, or far outside the bulk
        n_zero = int((u == 0).sum())
        n_neg = int((u < 0).sum())
        p01, p99 = np.percentile(u, [0.1, 99.9])
        n_far_lo = int((u < p01 - 50 * SEC_PER_YEAR).sum())
        if n_zero or n_neg or n_far_lo:
            print(f"    {'':22} SENTINELS: ==0 {n_zero}   <0 {n_neg}   "
                  f">50y below p0.1 {n_far_lo}")

    # ---- finding 1: what produces tau > 22? ----
    task = get_task(dataset, task_name, download=True)
    tt = task.get_table("train").df
    seed_u = as_unix(tt[task.time_col])
    print(f"\n=== finding 1: tau > 22 needs Delta > {np.expm1(22.0):.3e} s "
          f"({np.expm1(22.0)/SEC_PER_YEAR:.0f} y)")
    print(f"  task '{task_name}' seed times: min {fmt_unix(seed_u.min())}  "
          f"max {fmt_unix(seed_u.max())}  n={len(seed_u)}")
    worst_seed = seed_u.max()
    for name, u in sorted(per_table.items(), key=lambda kv: kv[1].min()):
        tau_max = float(np.log1p(max(0.0, worst_seed - u.min())))
        flag = "  <-- EXCEEDS 22" if tau_max > 22.0 else ""
        print(f"  {name:24} oldest row {fmt_unix(u.min())}  -> max tau {tau_max:6.2f}{flag}")

    # ---- finding 2: are the row_time > seed_time rows the SEED rows themselves? ----
    ent = task.entity_table
    print(f"\n=== finding 2: entity_table = {ent!r}")
    et = db.table_dict[ent]
    if et.time_col is None:
        print(f"  {ent} is UNTIMED -> its rows carry the untimed sentinel, not a real time.")
        print("  So sampled seed rows cannot themselves violate row_time <= seed_time.")
    else:
        eu = as_unix(et.df[et.time_col])
        ekey = et.df[et.pkey_col].to_numpy() if et.pkey_col else None
        ecol = task.entity_col
        # for each (entity, seed_time) pair in the task table, compare the entity row's OWN time
        m = pd.DataFrame({"e": tt[ecol].to_numpy(), "seed": seed_u})
        lut = pd.DataFrame({"e": ekey, "etime": eu})
        j = m.merge(lut, on="e", how="left")
        ok = j["etime"].notna()
        later = int((j.loc[ok, "etime"] > j.loc[ok, "seed"]).sum())
        print(f"  entity rows: min {fmt_unix(eu.min())}  max {fmt_unix(eu.max())}")
        print(f"  task rows joined to an entity row: {int(ok.sum())} / {len(j)}")
        print(f"  SEED rows whose OWN time > their seed_time: {later} "
              f"({100.0*later/max(int(ok.sum()),1):.2f}%)")
        if later:
            d = (j.loc[ok, "etime"] - j.loc[ok, "seed"])
            d = d[d > 0]
            print(f"    worst overshoot {d.max()/86400:.1f} days; median {d.median()/86400:.1f} days")
            print("  => the 97 flagged rows are almost certainly these: RelBench includes the QUERY")
            print("     entity's own row regardless of its timestamp. Benign for leakage, but it")
            print("     means Delta<0 -> clamped to 0 -> tau=0 for those seed cells.")
        else:
            print("  => seed rows do NOT explain the violations; investigate the sampler.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-event")
    ap.add_argument("--task", default="user-attendance")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    probe(args.dataset, args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
