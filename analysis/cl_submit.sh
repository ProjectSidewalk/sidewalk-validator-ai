#!/bin/bash
# Submit the whole confident-learning chain for the given label types on klone (issues #76 and #75), from a checkout
# of this branch with logs/ present:
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/cl-2026-10 \
#       analysis/cl_submit.sh CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem
#
# Per label type: prep (folds + rule variants, CPU) -> k fold trainings with out-of-fold scoring (GPU, parallel) and
# the rule-variant trainings (GPU, parallel, need only prep) -> flag (CPU, after every fold) -> cl-drop and cl-flip
# trainings (GPU, after flag). Slurm dependencies carry the chain; a requeued (preempted) job keeps its id, so the
# dependencies survive preemption. Job ids are appended to cl_jobs.txt in the submit directory.
# Optional: PARTITION_ARGS for the GPU jobs (e.g. "--partition=gpu-l40s --constraint=l40s" to use the lab's own
# nodes), FOLDS (default 5), RAWLABELS (default $DATA/../rawlabels).
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
FOLDS="${FOLDS:-5}"
RAWLABELS="${RAWLABELS:-$(cd "$DATA/.." && pwd)/rawlabels}"
PARTITION_ARGS="${PARTITION_ARGS:-}"
[ $# -gt 0 ] || { echo "usage: $0 <LabelType> [...]" >&2; exit 2; }
[ -d logs ] || mkdir -p logs
sb() { sbatch --parsable "$@"; }
for LT in "$@"; do
  [ -f "$DATA/$LT/split.csv" ] || { echo "no $DATA/$LT/split.csv" >&2; exit 2; }
  prep=$(STEP=prep LT="$LT" DATA="$DATA" RAWLABELS="$RAWLABELS" FOLDS="$FOLDS" \
         sb --job-name="cl-prep-$LT" analysis/confident_learning_cpu.slurm)
  folds=()
  for k in $(seq 0 $((FOLDS - 1))); do
    folds+=("$(LT="$LT" VARIANT="fold$k" DATA="$DATA" RUNS="$RUNS" \
               sb $PARTITION_ARGS --job-name="cl-$LT-fold$k" --dependency="afterok:$prep" analysis/confident_learning.slurm)")
  done
  rules=()
  for rule in symmetric unanimous3; do
    rules+=("$(LT="$LT" VARIANT="$rule" DATA="$DATA" RUNS="$RUNS" \
               sb $PARTITION_ARGS --job-name="cl-$LT-$rule" --dependency="afterok:$prep" analysis/confident_learning.slurm)")
  done
  flag=$(STEP=flag LT="$LT" DATA="$DATA" \
         sb --job-name="cl-flag-$LT" --dependency="afterok:$(IFS=:; echo "${folds[*]}")" analysis/confident_learning_cpu.slurm)
  cls=()
  for v in cl-drop cl-flip; do
    cls+=("$(LT="$LT" VARIANT="$v" DATA="$DATA" RUNS="$RUNS" \
             sb $PARTITION_ARGS --job-name="cl-$LT-$v" --dependency="afterok:$flag" analysis/confident_learning.slurm)")
  done
  echo "$LT prep=$prep folds=${folds[*]} rules=${rules[*]} flag=$flag cl=${cls[*]}" | tee -a cl_jobs.txt
done
