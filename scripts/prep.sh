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

# One-time prep: materialize task tables, populate the graph cache, and build the frozen schema cache.
# RelBench + HF + graph + schema caches live on scratch under ~/scratch60/gloss (see scripts/env.sh).
#
# TWO encoders, two jobs (see prep_data.py's header):
#   --encoder       column/table/role NAMES  -> $GLOSS_SCHEMA_CACHE   (the router's table)
#   --text-encoder  free-text CELL VALUES    -> $GLOSS_GRAPH_CACHE/<enc>/<ds>
#
#   sbatch scripts/prep.sh                                                   # names only, hash values
#   sbatch --mem=128G scripts/prep.sh --datasets rel-stack \
#       --text-encoder minilm --no-download                                  # real cell text, d=384
#
# `--mem` is the knob that matters for the value pass: the embedded text columns are materialized
# densely at n_cells x d_text x 4 bytes, so rel-amazon at d=384 is ~64 GB resident and needs ~500G.

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
# qwen, not harrier: harrier is a 27B Gemma-3 that only ever got name caches for 3 of the 7 DBs, and
# a stale default here silently loads 51GB of weights for a job that only needs cache hits.
python scripts/prep_data.py --encoder qwen "${DS[@]}" "$@"
rc=$?
# Propagate the exit code. Without this the job reports COMPLETED / ExitCode 0 even when the Python
# raised, because `echo` is the last command and its status is the script's — which is exactly how
# six failed cell-text prep jobs looked green in `sacct` on 2026-08-12.
echo "PREP_DONE (rc=$rc)"
exit $rc
