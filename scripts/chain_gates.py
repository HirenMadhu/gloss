"""chain_gates.py — auto-launch the S11/S12 gate arrays when the S10 sweep drains.

Run by a SLURM dependency job (`scripts/chain_gates.sh`, --dependency=afterany:<sweep array>):
1. aggregate the sweep -> results/setjoin_grid/AGGREGATE.txt (the S10 gate report body);
2. pick the ADOPTED backbone (best mean-rank config ≤ 30M params — the approved adoption rule;
   the sweep contains the v3 default, so "keep the default" is the automatic fallback);
3. sbatch the three 27-cell gate arrays on that backbone:
   hop-2 (fanout2=8, set_size 256) / control (set_size 256) / cap32.
The conditional P2×P3 combo arm is NOT auto-launched — it needs both gate verdicts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]

from gloss.setjoin.grid import aggregate_grid, init_batch, pick_backbone  # noqa: E402


def main() -> int:
    out_dir = REPO / "results" / "setjoin_grid"
    report = aggregate_grid(out_dir)
    (out_dir / "AGGREGATE.txt").write_text(report)
    print(report)

    ci, cfg = pick_backbone(out_dir)
    flags = ["--route-on", "signature",
             "--d-model", str(cfg["d_model"]),
             "--n-wide-layers", str(cfg["n_wide_layers"]),
             "--n-set-layers", str(cfg["n_set_layers"]),
             "--d-ff", str(cfg["d_ff"]),
             "--batch-size", str(init_batch(cfg))]
    if cfg["use_shared"]:
        flags.append("--use-shared")
    print(f"\nADOPTED cfg#{ci}: {cfg}\nbackbone flags: {' '.join(flags)}")

    arms = [
        ("setjoin_p2_hop2", ["--fanout2", "8", "--set-size", "256"]),
        ("setjoin_p2_ctrl", ["--set-size", "256"]),
        ("setjoin_p3_cap32", ["--per-relation-cap", "32"]),
    ]
    for name, extra in arms:
        cmd = ["sbatch", "--array=0-26%8", str(REPO / "scripts" / "run_setjoin.sh"),
               "--out-dir", f"results/{name}", *flags, *extra]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(f"{name}: {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
