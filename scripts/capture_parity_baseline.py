#!/usr/bin/env python
"""Capture the §6 bit-for-bit parity baseline for the CURRENT (pre-refactor) `arch: rt` model.

    .venv/bin/python scripts/capture_parity_baseline.py            # write tests/fixtures/parity_baseline.json
    .venv/bin/python scripts/capture_parity_baseline.py --check    # recompute and diff, write nothing
    .venv/bin/python scripts/capture_parity_baseline.py --force    # overwrite an existing baseline

changes.md §6 makes a bit-for-bit reproduction of the pre-change numbers *the* regression guard for the
whole two-level refactor. changes.md §9.7 makes capturing it **urgent and ordered**: P0.5 pins the
pytorch-frame stype id space to a fixed enum, which resizes
``RelationalSignature.stype_emb.num_embeddings``, which changes how many RNG draws module construction
consumes, which changes every parameter initialised after it. After P0.5 lands there is no "before" left
to capture. Hence: run this **first**.

The fixture is frozen and self-contained (``tests/fixtures/parity_fixture.py``) — synthetic tables,
``HashEncoder`` name table, CPU, single-threaded, no relbench download and no schema cache. The routing
arm is ``signature``; ``dense``/``dense_wide`` are never run (standing project rule).

**Re-capturing is a deliberate act.** If ``tests/test_parity.py`` fails you must first decide *which*
happened: (a) an unintended regression — fix the code; or (b) a legitimate, intended change to init
order or numerics (P0.5 is exactly this) — then re-run with ``--force`` and commit the new baseline in
the *same* commit as the change, with the reason in the message. Never re-capture to silence a failure
you have not explained.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.fixtures.parity_fixture import (  # noqa: E402
    ARTIFACT_PATH,
    compute_fingerprint,
    env_info,
    load_baseline,
)


def _flat(d, prefix=""):
    """Flatten a nested fingerprint to ``dotted.path -> leaf`` for a readable diff."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d)
    else:
        out[prefix] = d
    return out


def diff(a: dict, b: dict) -> list[str]:
    """Human-readable differences between two fingerprints (``a`` = baseline, ``b`` = current)."""
    fa, fb = _flat(a), _flat(b)
    lines = []
    for k in sorted(set(fa) | set(fb)):
        if k not in fa:
            lines.append(f"  + {k} = {fb[k]!r}   (new)")
        elif k not in fb:
            lines.append(f"  - {k} = {fa[k]!r}   (gone)")
        elif fa[k] != fb[k]:
            lines.append(f"  ! {k}: baseline={fa[k]!r}  current={fb[k]!r}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    ap.add_argument("--force", action="store_true", help="overwrite an existing baseline (deliberate re-capture)")
    ap.add_argument("--check", action="store_true", help="recompute and diff against the stored baseline; write nothing")
    args = ap.parse_args()

    fp = compute_fingerprint()
    env = env_info()

    if args.check:
        if not args.out.exists():
            print(f"no baseline at {args.out}", file=sys.stderr)
            return 2
        base = load_baseline(args.out)
        d = diff(base["fingerprint"], fp)
        if d:
            print(f"PARITY DRIFT vs {args.out} ({len(d)} difference(s)):")
            print("\n".join(d[:60]))
            if len(d) > 60:
                print(f"  ... and {len(d) - 60} more")
            return 1
        print(f"identical to {args.out}")
        return 0

    if args.out.exists() and not args.force:
        base = load_baseline(args.out)
        d = diff(base["fingerprint"], fp)
        if not d:
            print(f"baseline already present and identical: {args.out}")
            return 0
        print(
            f"REFUSING to overwrite {args.out}: it differs from the current model in {len(d)} place(s).\n"
            "Re-capturing is a deliberate act (changes.md §9.7). Decide first whether this is a\n"
            "regression or an intended init-order change (e.g. P0.5), then re-run with --force.\n",
            file=sys.stderr,
        )
        print("\n".join(d[:40]), file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_doc": (
            "Bit-for-bit parity baseline of the pre-refactor `arch: rt` MoRE (changes.md §6 / §9.7). "
            "Captured on a frozen, self-contained synthetic fixture (tests/fixtures/parity_fixture.py) "
            "that deliberately does NOT depend on tests/conftest.py, because the fingerprint is exact "
            "and any unrelated fixture edit would otherwise read as a parity regression. "
            "Regenerate ONLY deliberately: scripts/capture_parity_baseline.py --force."
        ),
        "captured_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "env": env,
        "fingerprint": fp,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")

    init = fp["init"]
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.1f} KB)")
    print(f"  commit            {env['git_commit'][:12]}   seed {env['seed']}   torch {env['torch']}")
    print(f"  arm               {init['config']['route_on']}   params {init['n_params']}")
    print(f"  stype_emb.num_emb {init['stype_emb_num_embeddings']}   <- the §9.7 tripwire (P0.5 changes this)")
    print(f"  rng after init    {init['rng']['state_after_init']}")
    print(f"  logits            {fp['numerics']['forward']['logits_exact']}")
    print(f"  aux               {fp['numerics']['forward']['aux']}")
    print(f"  train losses      {fp['numerics']['train_losses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
