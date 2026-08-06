#!/usr/bin/env python3
"""
Cardinality census on RelBench + how much of it a width-bounded sampler destroys.

Trains nothing. numpy / pandas / sklearn only. CPU.

    python cardinality_census.py --dataset rel-f1 --task driver-top3 \
        --widths 8 16 32 64 128 --out rel-f1_driver-top3.json

Run it on several (dataset, task) pairs and keep the JSON files; the cross-database
comparison is done by reading them together.

Outputs four blocks:
  1. fanout       per FK role: child-count distribution + censoring rate at each w
  2. drift        mean log fanout per time bucket, train vs val vs test window
  3. seed_counts  per-seed true vs censored counts for the entity table's FK roles
  4. damage       val AUROC (or MAE) from censored counts vs true counts
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NEG_INF = np.iinfo(np.int64).min


# ----------------------------------------------------------------------------- utils

def to_ns(series):
    """datetime64 column -> int64 nanoseconds."""
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype(np.int64)


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "mean_log1p": float(np.log1p(x).mean()),
        "std_log1p": float(np.log1p(x).std()),
    }


# ----------------------------------------------------------------------------- load

def load(dataset_name, task_name):
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    ds = get_dataset(dataset_name, download=True)
    # full DB (not truncated at test time): per-seed counts are time-filtered anyway,
    # and the drift block needs the test window.
    try:
        db = ds.get_db(upto_test_timestamp=False)
    except TypeError:
        db = ds.get_db()
    task = get_task(dataset_name, task_name, download=True)
    return ds, db, task


def fk_edges(db):
    """[(child_table, fk_col, parent_table), ...]"""
    out = []
    for tname, t in db.table_dict.items():
        for fk_col, parent in (t.fkey_col_to_pkey_table or {}).items():
            out.append((tname, fk_col, parent))
    return out


# ------------------------------------------------------------------- child-time index

class ChildIndex:
    """For one FK edge: parent key -> sorted array of child row times."""

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
        self.keys = keys[order]
        self.times = times[order]
        uniq, starts = np.unique(self.keys, return_index=True)
        ends = np.append(starts[1:], self.keys.shape[0])
        self.span = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))

    def total_counts(self):
        """Child count per parent over all time (parents with >=1 child only)."""
        return np.array([e - s for s, e in self.span.values()], dtype=np.int64)

    def count_upto(self, key, t_ns):
        span = self.span.get(key)
        if span is None:
            return 0
        s, e = span
        if not self.timed:
            return e - s
        return int(np.searchsorted(self.times[s:e], t_ns, side="right"))

    def count_and_oldest(self, key, t_ns, widths):
        """-> (true count, {w: age in ns of the oldest child a last-w sampler keeps}).

        This is the quantity a ``temporal_strategy="last"`` sampler exposes for free:
        of the w most recent children, how far back does the oldest one reach. Dense
        parents saturate the window and get a SHORT span; sparse ones reach further
        back. So it is a censoring-survivable proxy for the count the cap destroyed.

        NaN where undefined: no children at all, or an untimed child table (no
        ordering, so "the last w" is not a temporal notion).
        """
        nan = {w: np.nan for w in widths}
        span = self.span.get(key)
        if span is None or not self.timed:
            return (0 if span is None else span[1] - span[0]), nan
        s, e = span
        m = int(np.searchsorted(self.times[s:e], t_ns, side="right"))
        if m == 0:
            return 0, nan
        return m, {w: float(t_ns - self.times[s + max(0, m - w)]) for w in widths}


# ------------------------------------------------------------------------- block 1+2

def fanout_census(db, edges, widths):
    out = {}
    for child, fk_col, parent in edges:
        idx = ChildIndex(db, child, fk_col)
        counts = idx.total_counts()
        if counts.size == 0:
            continue
        role = f"{child}.{fk_col}->{parent}"
        rec = describe(counts)
        rec["timed_child"] = idx.timed
        rec["n_parents_with_children"] = int(counts.size)
        # fraction of parents censored, and fraction of child rows discarded
        for w in widths:
            rec[f"frac_parents_over_w{w}"] = float((counts > w).mean())
            rec[f"frac_rows_lost_w{w}"] = float(
                np.clip(counts - w, 0, None).sum() / counts.sum()
            )
        out[role] = rec
    return out


def drift(db, edges, val_ts, test_ts, n_buckets=8):
    """Mean log1p child-count per parent, by time bucket, and per split window."""
    out = {}
    for child, fk_col, parent in edges:
        t = db.table_dict[child]
        df = t.df
        if t.time_col is None or t.time_col not in df.columns:
            continue
        keep = df[fk_col].notna().to_numpy()
        keys = df.loc[keep, fk_col].to_numpy()
        times = to_ns(df.loc[keep, t.time_col])
        if times.size == 0:
            continue

        role = f"{child}.{fk_col}->{parent}"
        edges_ns = np.quantile(times, np.linspace(0, 1, n_buckets + 1))
        bucket_stats = []
        for i in range(n_buckets):
            lo, hi = edges_ns[i], edges_ns[i + 1]
            m = (times >= lo) & (times < hi if i < n_buckets - 1 else times <= hi)
            if m.sum() == 0:
                continue
            _, c = np.unique(keys[m], return_counts=True)
            bucket_stats.append({
                "t_start": str(pd.Timestamp(int(lo))),
                "mean_log1p": float(np.log1p(c).mean()),
                "n_rows": int(m.sum()),
                "n_parents": int(c.size),
            })

        def window(lo, hi):
            m = (times >= lo) & (times < hi)
            if m.sum() == 0:
                return None
            _, c = np.unique(keys[m], return_counts=True)
            return {"mean_log1p": float(np.log1p(c).mean()), "n_rows": int(m.sum())}

        v, s = to_ns(pd.Series([val_ts]))[0], to_ns(pd.Series([test_ts]))[0]
        out[role] = {
            "buckets": bucket_stats,
            "train_window": window(times.min(), v),
            "val_window": window(v, s),
            "test_window": window(s, times.max() + 1),
        }
    return out


# --------------------------------------------------------------------------- block 3

def seed_features(db, task, split, widths, entity_edges, max_seeds=None):
    """True and censored counts per FK role, per seed, plus self-label history length."""
    tbl = task.get_table(split)
    df = tbl.df
    if max_seeds is not None and len(df) > max_seeds:
        df = df.sample(max_seeds, random_state=0).reset_index(drop=True)

    ents = df[task.entity_col].to_numpy()
    ts = to_ns(df[task.time_col])
    n = len(df)

    feats = {}
    ages = {w: {} for w in widths}
    for role, idx in entity_edges.items():
        true = np.empty(n, dtype=np.int64)
        per_w = {w: np.full(n, np.nan) for w in widths}
        for i in range(n):
            cnt, ag = idx.count_and_oldest(ents[i], ts[i], widths)
            true[i] = cnt
            for w in widths:
                per_w[w][i] = ag[w]
        feats[role] = true
        for w in widths:
            ages[w][role] = per_w[w]

    # self-label history: prior rows for this entity in the TRAIN task table
    train_df = task.get_table("train").df
    h_keys = train_df[task.entity_col].to_numpy()
    h_times = to_ns(train_df[task.time_col])
    order = np.lexsort((h_times, h_keys))
    h_keys, h_times = h_keys[order], h_times[order]
    uniq, starts = np.unique(h_keys, return_index=True)
    ends = np.append(starts[1:], h_keys.shape[0])
    span = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))
    hist = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sp = span.get(ents[i])
        if sp is None:
            continue
        s, e = sp
        hist[i] = int(np.searchsorted(h_times[s:e], ts[i], side="left"))
    feats["__self_label_history__"] = hist

    y = df[task.target_col].to_numpy()
    return feats, ages, y


def build_age(ages_w, roles, med=None):
    """log1p(age in days) of the oldest child a last-w sampler keeps, per role.

    NaN (no children / untimed role) is imputed with the TRAIN median for that role,
    passed in via ``med`` so val never sees its own statistics.
    """
    if not roles:
        return np.zeros((0, 0)), np.zeros(0)
    A = np.stack([ages_w[r] for r in roles], axis=1).astype(float) / 86_400e9
    A = np.log1p(np.clip(A, 0.0, None))
    if med is None:
        med = np.array([
            np.nanmedian(A[:, j]) if np.isfinite(A[:, j]).any() else 0.0
            for j in range(A.shape[1])
        ])
    bad = ~np.isfinite(A)
    if bad.any():
        A[bad] = np.take(med, np.where(bad)[1])
    return A, med


def stack(feats, w):
    """log1p of true counts, and log1p of counts censored at w."""
    names = sorted(feats)
    true = np.stack([np.log1p(feats[k]) for k in names], axis=1)
    cens = np.stack([np.log1p(np.minimum(feats[k], w)) for k in names], axis=1)
    return names, true, cens


PROBES = ("linear", "bins", "spline")


def featurize(kind, Xtr, Xva, n_bins=8, n_knots=6):
    """Expand log1p-count columns into the basis the probe actually gets to use.

    ``linear`` is one monotone log1p term per role, which structurally cannot
    express the tail — the exact region censoring operates on. ``bins`` and
    ``spline`` add a per-role nonlinear basis on top of that term, so a probe can
    distinguish "50 children" from "5000" instead of only their log ratio.

    Constant columns are dropped first: they carry no signal, and quantile knots /
    bin edges on a constant are degenerate.
    """
    from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer

    if kind == "linear":
        return Xtr, Xva

    keep = Xtr.std(axis=0) > 0
    if not keep.any():
        return Xtr, Xva
    Xtr_k, Xva_k = Xtr[:, keep], Xva[:, keep]

    if kind == "bins":
        try:
            enc = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense",
                                   strategy="quantile", subsample=None,
                                   quantile_method="averaged_inverted_cdf")
        except TypeError:  # sklearn < 1.5 has no quantile_method
            enc = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense",
                                   strategy="quantile", subsample=None)
    elif kind == "spline":
        enc = SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                                include_bias=False)
    else:
        raise ValueError(f"unknown probe basis {kind!r}")

    B_tr = np.asarray(enc.fit_transform(Xtr_k), dtype=np.float64)
    B_va = np.asarray(enc.transform(Xva_k), dtype=np.float64)
    # keep the linear term alongside the basis: monotone signal stays available
    return np.hstack([Xtr, B_tr]), np.hstack([Xva, B_va])


def damage(feats_tr, ages_tr, y_tr, feats_va, ages_va, y_va, widths, task_type,
           probes=PROBES):
    """val score from censored vs true counts, per width, per feature basis.

    Sign convention: ``damage`` is always "how much censoring COST you", positive
    = censoring hurt. AUROC is higher-is-better and MAE is lower-is-better, so the
    subtraction flips; reporting a raw ``s_true - s_cens`` for both (as an earlier
    version did) inverts the meaning of every regression row.

    ``baseline`` is reported on every row so an uninformative probe is visible as
    such: a probe that cannot beat a constant predictor is not measuring censoring
    damage, whatever its ``damage`` number says.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    is_clf = "class" in str(task_type).lower()
    metric = "auroc" if is_clf else "mae"
    higher_is_better = is_clf

    # a constant predictor: chance AUROC for binary, train-median for MAE
    baseline = 0.5 if is_clf else float(np.mean(np.abs(y_va - np.median(y_tr))))

    def score(Xtr, Xva):
        if is_clf:
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), y_tr)
            return float(roc_auc_score(y_va, m.predict_proba(sc.transform(Xva))[:, 1]))
        # MAE-consistent learner: Ridge minimizes squared error and loses to the
        # median baseline on these heavy-tailed targets.
        m = GradientBoostingRegressor(loss="absolute_error", random_state=0).fit(Xtr, y_tr)
        return float(mean_absolute_error(y_va, m.predict(Xva)))

    out = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "baseline": baseline,
        "baseline_kind": "chance (AUROC 0.5)" if is_clf else "train-median MAE",
        "features": sorted(feats_tr),
        "damage_convention": "positive = censoring cost you; sign flipped for MAE",
        "probes": list(probes),
    }

    # roles with a real timestamp axis; the self-label history has no age analogue
    age_roles = sorted(r for r in ages_tr[widths[0]] if r in feats_tr)
    out["age_roles"] = age_roles

    for probe in probes:
        rec = {}
        for w in widths:
            names, true_tr, cens_tr = stack(feats_tr, w)
            _, true_va, cens_va = stack(feats_va, w)
            A_tr, med = build_age(ages_tr[w], age_roles)
            A_va, _ = build_age(ages_va[w], age_roles, med)

            s_true = score(*featurize(probe, true_tr, true_va))
            s_cens = score(*featurize(probe, cens_tr, cens_va))
            if A_tr.size:
                s_age = score(*featurize(probe, np.hstack([cens_tr, A_tr]),
                                         np.hstack([cens_va, A_va])))
            else:
                s_age = s_cens

            sgn = 1.0 if higher_is_better else -1.0
            dmg = sgn * (s_true - s_cens)
            dmg_age = sgn * (s_true - s_age)
            head = sgn * (s_true - baseline)
            frac_cens = float(
                np.mean(np.any(np.stack([feats_va[k] > w for k in names], 1), axis=1))
            )
            rec[f"w{w}"] = {
                "censored_counts": s_cens,
                "censored_plus_age": s_age,
                "true_counts": s_true,
                "baseline": baseline,
                "damage": dmg,
                "damage_after_age": dmg_age,
                "age_recovers": dmg - dmg_age,
                "age_recovers_frac": (float((dmg - dmg_age) / dmg)
                                      if abs(dmg) > 1e-9 else None),
                "true_headroom_over_baseline": head,
                "informative": bool(head > 0),
                "frac_val_seeds_with_any_censored_role": frac_cens,
                "metric": metric,
            }
        out[probe] = rec
    return out


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--max-seeds", type=int, default=40000)
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES),
                    help="feature bases for the damage probe")
    ap.add_argument("--damage-only", action="store_true",
                    help="recompute only the damage block, merging into an existing "
                         "--out JSON (keeps its fanout/drift/seed_counts blocks)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.damage_only and not args.out:
        ap.error("--damage-only needs --out (the JSON to merge into)")

    ds, db, task = load(args.dataset, args.task)
    edges = fk_edges(db)
    print(f"{args.dataset}: {len(db.table_dict)} tables, {len(edges)} FK roles",
          file=sys.stderr)

    if args.damage_only:
        with open(args.out) as fh:
            report = json.load(fh)
        report["widths"] = args.widths
    else:
        report = {"dataset": args.dataset, "task": args.task, "widths": args.widths}
        report["fanout"] = fanout_census(db, edges, args.widths)
        report["drift"] = drift(db, edges, ds.val_timestamp, ds.test_timestamp)

    # FK roles whose parent is the task's entity table: these are the o2m fanouts
    # the seed actually aggregates over.
    entity_edges = {}
    for child, fk_col, parent in edges:
        if parent == task.entity_table:
            entity_edges[f"{child}.{fk_col}"] = ChildIndex(db, child, fk_col)
    print(f"entity-incident roles: {sorted(entity_edges)}", file=sys.stderr)

    f_tr, a_tr, y_tr = seed_features(db, task, "train", args.widths, entity_edges,
                                     args.max_seeds)
    f_va, a_va, y_va = seed_features(db, task, "val", args.widths, entity_edges,
                                     args.max_seeds)
    report["seed_counts"] = {
        "train": {k: describe(v) for k, v in f_tr.items()},
        "val": {k: describe(v) for k, v in f_va.items()},
    }
    report["damage"] = damage(f_tr, a_tr, y_tr, f_va, a_va, y_va, args.widths,
                              task.task_type, probes=args.probes)

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
