#!/bin/bash
# Queue the production pretraining runs: 7 leave-one-database-out folds and/or the all-7 model.
#
#   bash scripts/launch_lodo.sh --lodo                     # 7 folds
#   bash scripts/launch_lodo.sh --all                      # the single all-7 model
#   bash scripts/launch_lodo.sh --lodo --chain 3           # 3 chained links per fold (48 h total)
#   bash scripts/launch_lodo.sh --all --dry-run
#
# Anything after `--` is forwarded to run_pretrain.py, e.g.
#   bash scripts/launch_lodo.sh --lodo -- --lambda-cat 0.2 --num-neighbors 24 24
#
# CHAINING exists because the `gpu` partition caps at 16 h and a 50k-step run at ~2-4 steps/s is
# 4-7 h before the fanout is raised to fill the sequence. Each link `--resume`s from the previous
# one's train_state.pt and is submitted with `--dependency=afterany`, so a link that dies on the wall
# clock still hands off rather than breaking the chain.

set -euo pipefail
cd "$(dirname "$0")/.."

ALL_DBS=(rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial)
MODE=""; CHAIN=1; DRY=0; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lodo) MODE=lodo; shift ;;
    --all)  MODE=all;  shift ;;
    --chain) CHAIN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
done
[[ -z "$MODE" ]] && { echo "pass --lodo or --all" >&2; exit 2; }

# rel-amazon's MiniLM graph cache is 66G on disk and is loaded resident, so any run containing it
# needs far more than the script's 256G default.
submit_chain () {           # $1 = job name, rest = run_pretrain.py args
  local name="$1"; shift
  # A run needs the big allocation iff it LOADS rel-amazon, i.e. every run except the fold that holds
  # rel-amazon out. Stating it as "excludes" rather than "mentions": `--holdout rel-amazon` is
  # precisely the run that does NOT load it, so keying off the substring gets it exactly backwards.
  local mem=500G
  [[ "$*" == *"--holdout rel-amazon"* ]] && mem=256G
  local dep="" jid=""
  for ((i = 0; i < CHAIN; i++)); do
    local cmd=(sbatch --parsable --job-name="$name" --mem="$mem")
    [[ -n "$dep" ]] && cmd+=(--dependency=afterany:"$dep")
    cmd+=(scripts/run_pretrain.sh --run-name "$name" "$@")
    if [[ $DRY -eq 1 ]]; then
      echo "DRY: ${cmd[*]}"; jid="dry$i"
    else
      jid=$("${cmd[@]}")
      echo "  link $((i + 1))/$CHAIN -> job $jid"
    fi
    dep="$jid"
  done
}

if [[ "$MODE" == lodo ]]; then
  for ds in "${ALL_DBS[@]}"; do
    echo "LODO fold: hold out $ds"
    submit_chain "pt-hold-$ds" --holdout "$ds" --wandb "${EXTRA[@]+"${EXTRA[@]}"}"
  done
else
  echo "all-7 model"
  submit_chain "pt-all7" --datasets all --wandb "${EXTRA[@]+"${EXTRA[@]}"}"
fi
echo "queued. watch: squeue -u \$USER ; results: \$GLOSS_CKPT_DIR/<run>/"
