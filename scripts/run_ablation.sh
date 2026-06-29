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
# Prep once first (graph + harrier schema caches on scratch60), then size the array to `--list`:
#
#   sbatch scripts/prep.sh                                    # graph + harrier schema caches
#   N=$(.venv/bin/python scripts/run_ablation.py --list --datasets rel-f1 rel-trial rel-event --signals signature --seeds 5)
#   sbatch --dependency=afterok:<prep> --array=0-$((N-1))%8 scripts/run_ablation.sh \
#       --datasets rel-f1 rel-trial rel-event --signals signature --seeds 5 --out-dir results/ablation_sig_harrier
#   # when all tasks finish:
#   .venv/bin/python scripts/run_ablation.py --aggregate --datasets rel-f1 rel-trial rel-event --signals signature --seeds 5 --out-dir results/ablation_sig_harrier

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}  HF_HOME=$HF_HOME RELBENCH_CACHE_DIR=$RELBENCH_CACHE_DIR"

python scripts/run_ablation.py --index "${SLURM_ARRAY_TASK_ID}" \
    --encoder harrier --epochs 10 --batch-size 64 --num-workers 8 "$@"
rc=$?
echo "ABLATION_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
