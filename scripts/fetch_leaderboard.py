#!/usr/bin/env python
"""Scrape the RelBench leaderboard into ``results/leaderboard_baselines.json``.

The leaderboard is a HF Space whose whole payload is a static ``index.html`` holding three tables
(Classification / Regression / Recommendation). We keep only the methods we compare against and only
the classification + regression tables (recommendation tasks are deferred — see CLAUDE.md).

Because the Space is live, every scrape records the retrieval date: ``fetched`` is the first time a
value set was seen, ``last_verified`` the last time a scrape agreed with what is already stored. A
later discrepancy is then traceable to a leaderboard update rather than looking like a model
regression.

    python scripts/fetch_leaderboard.py            # scrape + write
    python scripts/fetch_leaderboard.py --check    # scrape + diff vs stored, write nothing (exit 1 on drift)
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "leaderboard_baselines.json"

SPACE = "https://huggingface.co/spaces/relbench/leaderboard"
RAW = f"{SPACE}/raw/main/index.html"

# The methods we compare against. Keys in the JSON are the identifier-safe forms.
METHODS = {"RT (from scratch)": "RT_from_scratch", "GelGT": "GelGT"}
SECTIONS = ("Classification", "Regression")


def _cells(row: str) -> list[str]:
    """Text of every th/td in a row; ``<br>`` (the dataset/task split) becomes a space."""
    out = []
    for m in re.finditer(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S):
        cell = re.sub(r"<br\s*/?>", " ", m.group(1))
        out.append(html.unescape(re.sub(r"<[^>]+>", "", cell)).strip())
    return out


def parse(page: str) -> dict[str, dict[str, dict[str, str]]]:
    """``{"classification": {"rel-f1 driver-dnf": {"RT_from_scratch": "78.7", ...}}, ...}``.

    Column headers carry the ``dataset``/``task`` pair, so task keys come straight from the page
    rather than from a hardcoded list — a leaderboard that adds a task shows up as a new key.
    """
    out: dict[str, dict[str, dict[str, str]]] = {}
    for section in SECTIONS:
        head = re.search(rf"<h\d[^>]*>\s*{section}\s*</h\d>", page)
        if head is None:
            raise SystemExit(f"leaderboard layout changed: no <h*> heading {section!r}")
        start = page.index("<table", head.end())
        rows = re.findall(r"<tr>(.*?)</tr>", page[start : page.index("</table>", start)], re.S)
        # Header: #, Method, Regime, Mean, then one column per task.
        tasks = _cells(rows[0])[4:]
        table: dict[str, dict[str, str]] = {t: {} for t in tasks}
        seen = set()
        for row in rows[1:]:
            cell = _cells(row)
            if len(cell) < 5 or cell[1] not in METHODS:
                continue
            seen.add(cell[1])
            for task, value in zip(tasks, cell[4:]):
                table[task][METHODS[cell[1]]] = value
        missing = set(METHODS) - seen
        if missing:
            raise SystemExit(f"{section}: methods not found on the leaderboard: {sorted(missing)}")
        out[section.lower()] = table
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff the live leaderboard against the stored file; write nothing")
    args = ap.parse_args()

    with urllib.request.urlopen(RAW, timeout=60) as fh:
        page = fh.read().decode()
    live = parse(page)
    today = dt.date.today().isoformat()

    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    drift = [
        f"  {sec}/{task}/{meth}: stored={stored[sec][task][meth]} live={val}"
        for sec, table in live.items()
        for task, row in table.items()
        for meth, val in row.items()
        if sec in stored and task in stored[sec] and meth in stored[sec][task]
        and float(stored[sec][task][meth]) != float(val)
    ]
    added = [f"  {sec}/{task}" for sec, table in live.items()
             for task in table if sec not in stored or task not in stored[sec]]

    n = sum(len(t) for t in live.values())
    print(f"{SPACE}\nparsed {n} task columns over {list(live)} for {sorted(METHODS)}")
    for line in drift:
        print("CHANGED" + line)
    for line in added:
        print("NEW    " + line)
    if not drift and not added and stored:
        print(f"identical to stored {OUT.name} (fetched {stored.get('fetched')})")

    if args.check:
        return 1 if (drift or added) else 0

    doc = {
        "source": SPACE,
        # First sighting of this value set is preserved; only re-dated when a value actually moves.
        "fetched": today if (drift or added or not stored) else stored.get("fetched", today),
        "last_verified": today,
        "metrics": {
            "classification": "AUROC (percent, higher better)",
            "regression": "NMAE = MAE/train-std (lower better)",
        },
        "methods": list(METHODS),
        **live,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
