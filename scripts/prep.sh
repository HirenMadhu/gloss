#!/bin/bash
#SBATCH --job-name=gloss_prep
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=256G   # harrier (Gemma-3-27B) is ~51GB. NB: gpu_h200 does not exist on this
                     # cluster; `gpu` carries a40:4 / h100:4 (see sinfo).
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# One-time prep before the ablation array: materialize task tables, populate the graph cache, and build
# the frozen harrier schema cache for rel-f1 / rel-trial / rel-event. RelBench + HF + graph + schema
# caches live on scratch under ~/scratch60/gloss (see scripts/env.sh).
#   sbatch scripts/prep.sh

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "HF_HOME=$HF_HOME RELBENCH_CACHE_DIR=$RELBENCH_CACHE_DIR"

# Default to the current trio only when the caller didn't pass --datasets (so
# `sbatch scripts/prep.sh --datasets rel-event` preps just that one, no surprising override).
case " $* " in
  *" --datasets "*) DS=() ;;
  *)                DS=(--datasets rel-f1 rel-trial rel-event) ;;
esac
python scripts/prep_data.py --encoder harrier "${DS[@]}" "$@"
echo "PREP_DONE"
