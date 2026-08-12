#!/bin/bash
#SBATCH --job-name=more_pretrain
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Masked-cell pretraining. `gpu` is the only usable partition on this cluster (priority_gpu rejects
# the ying_rex account, gpu_h200 does not exist); 12 H100s cluster-wide.
#
#   sbatch scripts/run_pretrain.sh --datasets all --max-steps 50000 --wandb
#   sbatch scripts/run_pretrain.sh --holdout rel-f1 --max-steps 50000 --wandb   # one LODO fold
#   bash   scripts/launch_lodo.sh                                              # all 7 folds, chained
#
# --cpus-per-task=40, not the 8 the other scripts use, and that is measured rather than cautious:
# `to_cell_batch` costs 121-353 ms per batch against the neighbour sampler's 50-60 ms, so the loader
# is the bottleneck and 4 GPUs need ~10 workers each. Nodes have 48 CPUs.
#
# --mem: the MiniLM graph caches are large (rel-amazon 66G, rel-trial 24G, rel-event 21G on disk) and
# a bundle is loaded resident. 256G covers everything except an all-7 run that includes rel-amazon;
# override with --mem=500G for that.

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NGPU=$(nvidia-smi -L | wc -l)
echo "launching on ${NGPU} GPU(s)"

# --resume is always passed: the 16 h wall means a real run is a chain of jobs, and resuming from a
# missing train_state.pt is a no-op, so it is safe on the first link too.
python scripts/run_pretrain.py --devices "${NGPU}" --num-workers 10 --resume "$@"
rc=$?
echo "PRETRAIN_JOB_DONE (rc=$rc)"
exit $rc
