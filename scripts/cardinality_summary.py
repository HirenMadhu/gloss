#!/usr/bin/env python3
"""Read the cardinality_census.py JSONs together and print the cross-database tables.

    .venv/bin/python scripts/cardinality_summary.py results/cardinality/*.json
"""

import json
import sys
from collections import defaultdict


def load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def per_db_fanout(reports):
    """Fanout is a property of the DB, not the task — dedupe by dataset."""
    seen = {}
    for r in reports:
        seen.setdefault(r["dataset"], r)
    return seen


def main(paths):
    reports = load(paths)
    widths = reports[0]["widths"]

    print("=" * 100)
    print("FANOUT — worst FK roles per database (sorted by rows lost at w=32)")
    print("=" * 100)
    for ds, r in sorted(per_db_fanout(reports).items()):
        roles = sorted(r["fanout"].items(),
                       key=lambda kv: -kv[1]["frac_rows_lost_w32"])
        print(f"\n{ds}  ({len(r['fanout'])} FK roles)")
        print(f"  {'role':50s} {'med':>5} {'p99':>8} {'max':>9}  " +
              "  ".join(f"lost@{w:<4d}" for w in widths))
        for role, v in roles[:8]:
            print(f"  {role:50s} {v['median']:5.0f} {v['p99']:8.0f} {v['max']:9.0f}  " +
                  "  ".join(f"{v[f'frac_rows_lost_w{w}']:9.3f}" for w in widths))
        lost32 = [v["frac_rows_lost_w32"] for v in r["fanout"].values()]
        print(f"  -> {sum(1 for x in lost32 if x > 0.5)}/{len(lost32)} roles lose "
              f">50% of child rows at w=32")

    print("\n" + "=" * 100)
    print("SEED-SIDE FANOUT — per-seed counts on the entity table's roles (val split)")
    print("=" * 100)
    for r in reports:
        key = f"{r['dataset']}/{r['task']}"
        print(f"\n{key}")
        for role, v in sorted(r["seed_counts"]["val"].items()):
            if not v:
                continue
            print(f"  {role:34s} mean={v['mean']:9.1f} med={v['median']:7.0f} "
                  f"p99={v['p99']:9.0f} max={v['max']:9.0f}")

    print("\n" + "=" * 100)
    print("DRIFT — mean log1p fanout, first vs last equal-row time bucket")
    print("=" * 100)
    for ds, r in sorted(per_db_fanout(reports).items()):
        print(f"\n{ds}")
        rows = []
        for role, v in r["drift"].items():
            b = v["buckets"]
            if len(b) < 2:
                continue
            rows.append((b[-1]["mean_log1p"] - b[0]["mean_log1p"], role, b))
        for delta, role, b in sorted(rows, reverse=True)[:8]:
            print(f"  {role:50s} {b[0]['mean_log1p']:5.2f} -> {b[-1]['mean_log1p']:5.2f}"
                  f"  (delta {delta:+5.2f}; parents {b[0]['n_parents']} -> {b[-1]['n_parents']})")

    print("\n" + "=" * 100)
    print("DAMAGE — censoring cost, per feature basis (val)")
    print("  damage > 0 means censoring HURT (sign already flipped for MAE)")
    print("  UNINFORMATIVE = the uncensored probe cannot beat a constant predictor,")
    print("  so its damage number measures nothing regardless of magnitude")
    print("=" * 100)
    for r in reports:
        d = r["damage"]
        key = f"{r['dataset']}/{r['task']}"
        base, metric = d["baseline"], d["metric"]
        print(f"\n{key}   metric={metric}  baseline={base:.4f} ({d['baseline_kind']})")
        print(f"  {'basis':7s} " + "".join(f"{'w'+str(w):>22s}" for w in widths))
        print(f"  {'':7s} " + "".join(f"{'cens/true (damage)':>22s}" for _ in widths))
        for probe in d.get("probes", []):
            if probe not in d:
                continue
            cells, flags = [], []
            for w in widths:
                e = d[probe][f"w{w}"]
                cells.append(f"{e['censored_counts']:.3f}/{e['true_counts']:.3f}"
                             f"({e['damage']:+.3f})")
                flags.append(e["informative"])
            tag = "" if all(flags) else "   <- UNINFORMATIVE" if not any(flags) else \
                  "   <- partly uninformative"
            print(f"  {probe:7s} " + "".join(f"{c:>22s}" for c in cells) + tag)
            if "censored_plus_age" in d[probe][f"w{widths[0]}"]:
                cells = []
                for w in widths:
                    e = d[probe][f"w{w}"]
                    cells.append(f"{e['censored_plus_age']:.3f}"
                                 f"(dmg {e['damage_after_age']:+.3f})")
                print(f"  {'  +age':7s} " + "".join(f"{c:>22s}" for c in cells))
        cf = [d[d["probes"][0]][f"w{w}"]["frac_val_seeds_with_any_censored_role"]
              for w in widths]
        print(f"  {'censd':7s} " + "".join(f"{x:>22.3f}" for x in cf))

    print("\n" + "=" * 100)
    print("PROBE STRENGTH — does the uncensored probe beat a constant predictor?")
    print("=" * 100)
    for r in reports:
        d = r["damage"]
        key = f"{r['dataset']}/{r['task']}"
        w = widths[-1]
        best = max(d.get("probes", []),
                   key=lambda p: d[p][f"w{w}"]["true_headroom_over_baseline"])
        e = d[best][f"w{w}"]
        verdict = "informative" if e["informative"] else "UNINFORMATIVE"
        print(f"  {key:32s} best basis={best:7s} true={e['true_counts']:8.4f} "
              f"baseline={d['baseline']:8.4f} headroom={e['true_headroom_over_baseline']:+8.4f}"
              f"  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
