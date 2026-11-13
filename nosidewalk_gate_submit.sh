#!/bin/bash
# One command for the NoSidewalk 5-fold out-of-fold gate (issue 42, step (a); analysis/nosidewalk_gate.py), from the
# repo root on klone, in the activated environment:
#
#   DATA=/gscratch/scrubbed/$USER/nosidewalk-2026-11 GATE=$DATA/gate RUNS=/gscratch/scrubbed/$USER/nosidewalk-runs/gate \
#       ./nosidewalk_gate_submit.sh
#   DRY_RUN=1 DATA=... GATE=... RUNS=... ./nosidewalk_gate_submit.sh     # print the commands, submit nothing
#
# 1. folds (login node, seconds): K face-grouped folds over the pilot's whole-view train/val/test rows into $GATE,
#    unless $GATE/folds.json already exists (FOLDS and SEED set the count and seed for a fresh run).
# 2. One train job per fold (ckpt-all, one GPU): train, then score the fold's test and test_heldout rows.
# 3. One report job after every train job (ckpt-all, no GPU): the pooled leave-one-fold-out gate and the bar.
# Job ids are appended to nosidewalk_gate_jobs.txt.
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; GATE="${GATE:?set GATE (absolute)}"; RUNS="${RUNS:?set RUNS}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
[ -f .gitmodules ] || { echo "Run this from the repo root." >&2; exit 1; }
export DATA GATE RUNS

N=0
sb() {
  if [ "$DRY_RUN" = 1 ]; then
    N=$((N + 1))
    echo "sbatch --parsable --kill-on-invalid-dep=yes $*"
    JOB="dry$N"
  else
    JOB=$(sbatch --parsable --kill-on-invalid-dep=yes "$@")
  fi
}
run() {
  if [ "$DRY_RUN" = 1 ]; then echo "$*"; else "$@"; fi
}

if [ -f "$GATE/folds.json" ]; then
  echo "Using the folds in $GATE" >&2
else
  run python -u analysis/nosidewalk_gate.py folds --split "$DATA/whole/NoSidewalk/split.csv" --out "$GATE" \
      --folds "$FOLDS" --seed "$SEED"
fi

[ "$DRY_RUN" = 1 ] || mkdir -p logs
TRAIN=()
for ((K = 0; K < FOLDS; K++)); do
  sb --job-name="ns-gate-fold$K" --export=ALL,STEP=train,FOLD="$K" nosidewalk_gate.slurm
  TRAIN+=("$JOB")
done
sb --job-name=ns-gate-report --gpus=0 --constraint="" --cpus-per-task=4 --mem=32G --time=2:00:00 \
   --dependency="afterany:$(IFS=:; echo "${TRAIN[*]}")" --export=ALL,STEP=report nosidewalk_gate.slurm
echo "train=${TRAIN[*]} report=$JOB" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a nosidewalk_gate_jobs.txt; fi; }
