#!/usr/bin/env python3
"""
LODO probe: does per-database fanout centering buy transfer?

Trains nothing but a logistic regression. numpy / pandas / sklearn. CPU.

    python lodo_scale_probe.py \
        --spec rel-f1:driver-top3,driver-dnf \
               rel-event:user-repeat,user-ignore \
               rel-trial:study-outcome \
        --out lodo_scale.json

The question this answers
------------------------
A per-database summary token holds a constant. Within one database a constant
cannot change the ranking of seeds, so it can only matter ACROSS databases: it
recenters a shared count->label mapping onto each database's own fanout scale.
This probe measures whether that recentering helps, by fitting a schema-agnostic
count probe on some databases and scoring it on a held-out one.

Every seed is reduced to five features that mean the same thing in any schema,
so a single probe can be fit across databases with different roles and arities:

    mean_l       mean over incident o2m roles of log1p(child count)
    max_l        max over roles of log1p(child count)
    frac_active  fraction of roles with at least one child
    hist         log1p(self-label history length)
    std_l        spread of log1p counts across roles

Four arms, differing only in what is subtracted from l_r = log1p(count_r)
before the five features are built. Each arm corresponds to a different amount
of per-database state:

    raw     nothing.                    No state. Note that the StandardScaler
                                        fit on the training databases already
                                        applies a fixed global offset, so this
                                        arm IS the "one learned constant is
                                        enough" control.
    db      mu_d, one scalar per DB.    State = 1 float per database.
    role    mu_{d,r}, per (DB, role).   State = 1 float per (database, role).
    role_z  (l - mu_{d,r}) / sd_{d,r}.  State = 2 floats per (database, role).
                                        This is PNA's degree scaler, learned
                                        per role rather than fixed.

All centering statistics come from that database's own TRAIN-split seed counts
with labels never touched, which is exactly the unlabeled calibration pass a
held-out database would get at test time. The held-out database's statistics are
therefore legitimately available and are used.

How to read it
--------------
role or db beating raw on held-out AUROC means the recentering has a measured
job and the number is its size. All arms equal means a model without per-database
state recovers the scale from its other inputs, and the summary token is
redundant. The within-DB oracle row is the ceiling for these five features and
tells you whether the probe has any strength at all on that database before you
read anything into the transfer numbers.
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NEG_INF = np.iinfo(np.int64).min
ARMS = ["raw", "db", "role", "role_z"]
FEATS = ["mean_l", "max_l", "frac_active", "hist", "std_l"]
AGE_FEATS = ["mean_age_l", "min_age_l"]   # added by the counts+age featureset
FEATSETS = ["counts", "counts+age"]


def to_ns(series):
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype(np.int64)


class ChildIndex:
    """One FK edge: parent key -> sorted child row times. (Same as the census script.)"""

    def __init__(self, db, child_table, fk_col):
        t = db.table_dict[child_table]
        df = t.df
        keep = df[fk_col].notna().to_numpy()
        keys = df.loc[keep, fk_col].to_numpy()
        if t.time_col is not None and t.time_col in df.columns:
            times = to_ns(df.loc[keep, t.time_col])
            self.timed = True
        else:
            times = np.full(keys.shape[0], NEG_INF, dtype=np.int64)
            self.timed = False
        order = np.lexsort((times, keys))
        self.keys, self.times = keys[order], times[order]
        uniq, starts = np.unique(self.keys, return_index=True)
        ends = np.append(starts[1:], self.keys.shape[0])
        self.span = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))

    def count_upto(self, key, t_ns):
        sp = self.span.get(key)
        if sp is None:
            return 0
        s, e = sp
        if not self.timed:
            return e - s
        return int(np.searchsorted(self.times[s:e], t_ns, side="right"))

    def count_and_oldest(self, key, t_ns, w):
        """-> (true count, age in ns of the oldest child a last-w sampler keeps).

        NaN age where undefined (no children, or an untimed child table). This is
        the quantity MoRE actually observes: temporal_strategy="last" keeps the w
        most recent children, so the oldest survivor's row_time is in the batch.
        """
        sp = self.span.get(key)
        if sp is None:
            return 0, np.nan
        s, e = sp
        if not self.timed:
            return e - s, np.nan
        m = int(np.searchsorted(self.times[s:e], t_ns, side="right"))
        if m == 0:
            return 0, np.nan
        return m, float(t_ns - self.times[s + max(0, m - w)])


def load_db(name):
    from relbench.datasets import get_dataset

    ds = get_dataset(name, download=True)
    try:
        db = ds.get_db(upto_test_timestamp=False)
    except TypeError:
        db = ds.get_db()
    return ds, db


def entity_roles(db, entity_table):
    """ChildIndex for every o2m role whose parent is the entity table."""
    out = {}
    for tname, t in db.table_dict.items():
        for fk_col, parent in (t.fkey_col_to_pkey_table or {}).items():
            if parent == entity_table:
                out[f"{tname}.{fk_col}"] = ChildIndex(db, tname, fk_col)
    return out


def raw_counts(task, split, roles, max_seeds, age_width):
    """-> L [n,K] log1p counts, A [n,K] log1p sampled-age (days), H [n], y [n]."""
    df = task.get_table(split).df
    if max_seeds and len(df) > max_seeds:
        df = df.sample(max_seeds, random_state=0).reset_index(drop=True)
    ents, ts = df[task.entity_col].to_numpy(), to_ns(df[task.time_col])
    n, names = len(df), sorted(roles)

    C = np.zeros((n, len(names)), dtype=np.int64)
    G = np.full((n, len(names)), np.nan)
    for j, name in enumerate(names):
        idx = roles[name]
        for i in range(n):
            C[i, j], G[i, j] = idx.count_and_oldest(ents[i], ts[i], age_width)

    tr = task.get_table("train").df
    hk, ht = tr[task.entity_col].to_numpy(), to_ns(tr[task.time_col])
    o = np.lexsort((ht, hk))
    hk, ht = hk[o], ht[o]
    uniq, starts = np.unique(hk, return_index=True)
    ends = np.append(starts[1:], hk.shape[0])
    span = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))
    hist = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sp = span.get(ents[i])
        if sp is not None:
            s, e = sp
            hist[i] = int(np.searchsorted(ht[s:e], ts[i], side="left"))

    y = df[task.target_col].to_numpy().astype(float)
    A = np.log1p(np.clip(G / 86_400e9, 0.0, None))   # log1p(age in days)
    return np.log1p(C.astype(float)), A, np.log1p(hist.astype(float)), y, names


def build(L, A, H, arm, stats, featureset="counts"):
    """Schema-agnostic features from log-counts (+ sampled age), under one arm.

    The age columns get the SAME centering treatment as the counts. That matters
    more for age than for counts: an "age in days" scale is wildly database
    specific (rel-f1 spans 70 years, rel-event a few weeks), so an uncentered age
    feature cannot mean the same thing across databases.
    """
    active = (L > 0).astype(float)
    Ai = np.where(np.isfinite(A), A, stats["med_a"][None, :]) if A.shape[1] else A

    if arm == "raw":
        Lc, Hc, Ac = L, H, Ai
    elif arm == "db":
        Lc, Hc, Ac = L - stats["mu_d"], H - stats["mu_h"], Ai - stats["mu_ad"]
    elif arm == "role":
        Lc, Hc = L - stats["mu_r"][None, :], H - stats["mu_h"]
        Ac = Ai - stats["mu_ar"][None, :]
        # note: max over (l_r - mu_r) is "which role most exceeds its own norm",
        # a different and more transferable quantity than max over raw l_r.
    elif arm == "role_z":
        Lc = (L - stats["mu_r"][None, :]) / stats["sd_r"][None, :]
        Hc = (H - stats["mu_h"]) / stats["sd_h"]
        Ac = (Ai - stats["mu_ar"][None, :]) / stats["sd_ar"][None, :]
    else:
        raise ValueError(arm)

    cols = [
        Lc.mean(axis=1),
        Lc.max(axis=1) if Lc.shape[1] else np.zeros(len(Lc)),
        active.mean(axis=1),
        Hc,
        Lc.std(axis=1) if Lc.shape[1] > 1 else np.zeros(len(Lc)),
    ]
    if featureset == "counts+age":
        # short sampled span = the last-w window saturated = dense parent
        cols.append(Ac.mean(axis=1) if Ac.shape[1] else np.zeros(len(Lc)))
        cols.append(Ac.min(axis=1) if Ac.shape[1] else np.zeros(len(Lc)))
    return np.stack(cols, axis=1)


def calib(L_train, A_train, H_train):
    """Unlabeled per-database calibration statistics, from train-split counts only."""
    sd_r = L_train.std(axis=0)
    sd_r[sd_r < 1e-6] = 1.0
    sd_h = float(H_train.std()) or 1.0

    K = A_train.shape[1]
    med_a = np.array([
        np.nanmedian(A_train[:, j]) if np.isfinite(A_train[:, j]).any() else 0.0
        for j in range(K)
    ])
    Ai = np.where(np.isfinite(A_train), A_train, med_a[None, :]) if K else A_train
    sd_ar = Ai.std(axis=0) if K else np.zeros(0)
    if K:
        sd_ar[sd_ar < 1e-6] = 1.0
    return {
        "mu_d": float(L_train.mean()),
        "mu_r": L_train.mean(axis=0),
        "sd_r": sd_r,
        "mu_h": float(H_train.mean()),
        "sd_h": sd_h,
        "med_a": med_a,
        "mu_ad": float(Ai.mean()) if K else 0.0,
        "mu_ar": Ai.mean(axis=0) if K else np.zeros(0),
        "sd_ar": sd_ar,
    }


def fit_score(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(yva)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yva, m.predict_proba(sc.transform(Xva))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="dataset:task1,task2 ... (binary classification only)")
    ap.add_argument("--max-seeds", type=int, default=40000)
    ap.add_argument("--age-width", type=int, default=12,
                    help="sampler width the age feature is computed under; 12 matches "
                         "MoRE's num_neighbors=[12,12]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from relbench.tasks import get_task

    spec = {}
    for s in args.spec:
        d, ts = s.split(":")
        spec[d] = ts.split(",")

    data = {}      # (db, task) -> {"train": (X per arm, y), "val": ...}
    role_names = {}
    for dbname, tasknames in spec.items():
        print(f"[{dbname}] loading", file=sys.stderr)
        _, db = load_db(dbname)
        cache, per_db = {}, {}
        for tn in tasknames:
            task = get_task(dbname, tn, download=True)
            if "class" not in str(task.task_type).lower():
                print(f"  skip {tn}: not binary classification", file=sys.stderr)
                continue
            if task.entity_table not in cache:
                cache[task.entity_table] = entity_roles(db, task.entity_table)
            roles = cache[task.entity_table]
            Ltr, Atr, Htr, ytr, names = raw_counts(task, "train", roles,
                                                   args.max_seeds, args.age_width)
            Lva, Ava, Hva, yva, _ = raw_counts(task, "val", roles,
                                               args.max_seeds, args.age_width)
            per_db[tn] = (Ltr, Atr, Htr, ytr, Lva, Ava, Hva, yva)
            role_names[(dbname, tn)] = names
            print(f"  {tn}: {len(ytr)} train / {len(yva)} val, {len(names)} roles",
                  file=sys.stderr)

        if not per_db:
            continue
        # one calibration per database, pooled over its tasks' train seeds
        Lpool = np.concatenate([v[0] for v in per_db.values()], axis=0)
        Apool = np.concatenate([v[1] for v in per_db.values()], axis=0)
        Hpool = np.concatenate([v[2] for v in per_db.values()], axis=0)
        st = calib(Lpool, Apool, Hpool)
        for tn, (Ltr, Atr, Htr, ytr, Lva, Ava, Hva, yva) in per_db.items():
            data[(dbname, tn)] = {
                "train": ({(a, f): build(Ltr, Atr, Htr, a, st, f)
                           for a in ARMS for f in FEATSETS}, ytr),
                "val": ({(a, f): build(Lva, Ava, Hva, a, st, f)
                         for a in ARMS for f in FEATSETS}, yva),
            }

    dbs = sorted({k[0] for k in data})
    report = {"spec": {d: sorted(t for (dd, t) in data if dd == d) for d in dbs},
              "roles": {f"{d}/{t}": v for (d, t), v in role_names.items()},
              "features": FEATS, "lodo": {}, "within_db": {}}

    if len(dbs) < 2:
        print(f"LODO needs >=2 databases, got {dbs} — within-DB rows only",
              file=sys.stderr)

    for fs in FEATSETS:
        report["lodo"][fs], report["within_db"][fs] = {}, {}
        for held in dbs:
            tr_keys = [k for k in data if k[0] != held]
            va_keys = [k for k in data if k[0] == held]
            if not tr_keys:
                report["lodo"][fs][held] = {a: {"per_task": {}, "mean": None}
                                            for a in ARMS}
                continue
            fold = {}
            for arm in ARMS:
                Xtr = np.concatenate([data[k]["train"][0][(arm, fs)]
                                      for k in tr_keys], axis=0)
                ytr = np.concatenate([data[k]["train"][1] for k in tr_keys], axis=0)
                per_task = {}
                for k in va_keys:
                    per_task[k[1]] = fit_score(Xtr, ytr, data[k]["val"][0][(arm, fs)],
                                               data[k]["val"][1])
                vals = [v for v in per_task.values() if v is not None]
                fold[arm] = {"per_task": per_task,
                             "mean": float(np.mean(vals)) if vals else None}
            report["lodo"][fs][held] = fold

        for k in data:
            report["within_db"][fs][f"{k[0]}/{k[1]}"] = {
                arm: fit_score(data[k]["train"][0][(arm, fs)], data[k]["train"][1],
                               data[k]["val"][0][(arm, fs)], data[k]["val"][1])
                for arm in ARMS
            }

    report["lodo_overall_mean"] = {}
    for fs in FEATSETS:
        report["lodo_overall_mean"][fs] = {}
        for arm in ARMS:
            vals = [report["lodo"][fs][d][arm]["mean"] for d in dbs
                    if report["lodo"][fs][d][arm]["mean"] is not None]
            report["lodo_overall_mean"][fs][arm] = (float(np.mean(vals))
                                                    if vals else None)

    def cell(x):
        return (f"{x:.4f}" if x is not None else "  n/a").rjust(9)

    w = max(max(len(d) for d in dbs), 14) + 2
    for fs in FEATSETS:
        print(f"\nheld-out AUROC (LODO)  featureset={fs}\n{'db'.ljust(w)}" +
              "".join(a.rjust(9) for a in ARMS))
        for d in dbs:
            print(d.ljust(w) + "".join(
                cell(report["lodo"][fs][d][a]["mean"]) for a in ARMS))
        print("MEAN".ljust(w) + "".join(
            cell(report["lodo_overall_mean"][fs][a]) for a in ARMS))

    print(f"\nage effect on LODO (counts+age minus counts)\n{'db'.ljust(w)}" +
          "".join(a.rjust(9) for a in ARMS))
    for d in dbs:
        row = []
        for a in ARMS:
            x = report["lodo"]["counts+age"][d][a]["mean"]
            y = report["lodo"]["counts"][d][a]["mean"]
            row.append(f"{x - y:+.4f}".rjust(9) if x is not None and y is not None
                       else "  n/a".rjust(9))
        print(d.ljust(w) + "".join(row))
    print("MEAN".ljust(w) + "".join(
        f"{report['lodo_overall_mean']['counts+age'][a] - report['lodo_overall_mean']['counts'][a]:+.4f}".rjust(9)
        for a in ARMS))

    print(f"\nwithin-DB oracle (ceiling for these features)\n"
          f"{'task'.ljust(28)}{'featureset':13s}" + "".join(a.rjust(9) for a in ARMS))
    for k in report["within_db"]["counts"]:
        for fs in FEATSETS:
            print(k.ljust(28) + fs.ljust(13) + "".join(
                cell(report["within_db"][fs][k][a]) for a in ARMS))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
