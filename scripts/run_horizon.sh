#!/bin/bash
#SBATCH --job-name=gloss_horizon
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Multi-horizon study as a SLURM job ARRAY (MILGRAM-native: gpu partition, h100:1, %8 QOS cap):
# one (dataset, task, model in {setjoin, more}, seed) cell per task; each trains its substrate's
# fixed config and evaluates the FULL horizon curve k=0..9 on TEST. rel-event needs the scratch
# symlinks (recap.md §2).
#
#   N=$(.venv/bin/python scripts/run_horizon.py --list)        # 54
#   sbatch --array=0-$((N-1))%8 scripts/run_horizon.sh --out-dir results/horizon
#   # when all tasks finish:
#   .venv/bin/python scripts/run_horizon.py --plot --out-dir results/horizon

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/run_horizon.py --index "${SLURM_ARRAY_TASK_ID}" \
    --encoder harrier --num-workers 8 "$@"
rc=$?
echo "HORIZON_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
