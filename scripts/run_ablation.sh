#!/bin/bash
#SBATCH --job-name=gloss_abl
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Routing-signal ablation as a SLURM job ARRAY: one (dataset, task, signal, seed) config per task.
#
# Partition: `gpu` with h100:1. An earlier version targeted `gpu_h200`/`h200:1` and claimed "this
# cluster has no h100" — both wrong: `sinfo` shows NO gpu_h200 partition at all, and `gpu` carries
# a40:4 and h100:4 (plus priority_gpu h100:4, scavenge). As written it would never have scheduled.
#
# Concurrency is capped at %8 — the account's simultaneous-GPU limit. Raising it just queues.
#
# Prereqs: the graph cache and the schema cache for the encoder you pass. Size the array to `--list`:
#
#   N=$(.venv/bin/python scripts/run_ablation.py --list --datasets rel-f1 rel-trial rel-event \
#         --tasks leaderboard --signals signature --seeds 3)
#   sbatch --array=0-$((N-1))%8 scripts/run_ablation.sh \
#       --datasets rel-f1 rel-trial rel-event --tasks leaderboard --signals signature --seeds 3 \
#       --encoder qwen --out-dir results/baseline_qwen
#   # when all tasks finish:
#   .venv/bin/python scripts/run_ablation.py --aggregate --datasets rel-f1 rel-trial rel-event \
#       --tasks leaderboard --signals signature --seeds 3 --out-dir results/baseline_qwen

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}  HF_HOME=$HF_HOME RELBENCH_CACHE_DIR=$RELBENCH_CACHE_DIR"

# NOTE `--encoder` is NOT hardcoded here. It used to be pinned to harrier *before* "$@", so a
# caller passing --encoder qwen only won by argparse last-wins — easy to get silently wrong.
# run_ablation.py defaults to qwen; pass --encoder explicitly to be unambiguous.
python scripts/run_ablation.py --index "${SLURM_ARRAY_TASK_ID}" \
    --epochs 10 --batch-size 64 --num-workers 8 "$@"
rc=$?
echo "ABLATION_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
