#!/bin/bash
#SBATCH --job-name=gloss_bsprobe
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname

# Probe the two extremes of the grid on the widest dataset we lose on (rel-trial/site-success),
# plus rel-f1 for comparison. The binding config is the largest one: d_model=256, n_blocks=4.
for spec in "rel-f1 driver-position 128 2" "rel-f1 driver-position 256 4" \
            "rel-trial site-success 128 2" "rel-trial site-success 256 4"; do
    set -- $spec
    echo "=================================================================="
    python scripts/probe_batch_size.py --dataset "$1" --task "$2" --d-model "$3" --n-blocks "$4"
done
echo "BSPROBE_DONE"
