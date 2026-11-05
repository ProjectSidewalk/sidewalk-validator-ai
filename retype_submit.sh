#!/bin/bash
# Submit the whole re-typing chain (issue #13) on klone, from the repo root of this branch:
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/retype-2026-11 \
#       PANO_STORE=/path/to/store ./retype_submit.sh
#
# extra x4 (GPU, parallel) and edits (CPU) first; crops (GPU) after edits; assemble (CPU) after crops and after every
# extra has ended (afterany: an extra type that fails, e.g. Other with too few labels, is left out, its class stays);
# then per leg train (GPU) -> eval (GPU). A requeued (preempted) job keeps its id, so the dependencies survive.
# Job ids are appended to retype_jobs.txt. Optional: LEGS (default "complement other"; "complement" runs one leg),
# EXTRAS (default "NoSidewalk Signal Other Occlusion"), SKIP_EXTRAS=1 (the extra types already exist under DATA),
# PARTITION_ARGS for the GPU jobs (e.g. "--partition=gpu-l40s --constraint=l40s"), and every variable retype.slurm
# reads (RAWLABELS, EDIT_RAWLABELS, HOLDOUT, SEED, EPOCHS, CORRECT_PROPORTION, PUBLIC_CITIES, EXPERTS).
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"; PANO_STORE="${PANO_STORE:?set PANO_STORE}"
export DATA RUNS PANO_STORE
LEGS="${LEGS:-complement other}"
EXTRAS="${EXTRAS:-NoSidewalk Signal Other Occlusion}"
PARTITION_ARGS="${PARTITION_ARGS:-}"
CPU_ARGS="--gpus=0 --cpus-per-task=4 --mem=48G --time=2:00:00"
[ -f retype.slurm ] || { echo "run from the repo root" >&2; exit 2; }
[ -d logs ] || mkdir -p logs
for T in CurbRamp NoCurbRamp Obstacle SurfaceProblem Crosswalk; do
  [ -f "$DATA/$T/split.csv" ] && [ -f "$DATA/$T/predictions.csv" ] || { echo "no $DATA/$T/split.csv or predictions.csv" >&2; exit 2; }
done
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
sb() { sbatch --parsable --kill-on-invalid-dep=yes "$@"; }
extras=()
if [ "${SKIP_EXTRAS:-0}" != 1 ]; then
  for LT in $EXTRAS; do
    extras+=("$(STEP=extra LT="$LT" sb $PARTITION_ARGS --job-name="retype-extra-$LT" retype.slurm)")
  done
fi
edits=$(STEP=edits sb $CPU_ARGS --job-name=retype-edits retype.slurm)
crops=$(STEP=crops sb $PARTITION_ARGS --job-name=retype-crops --dependency="afterok:$edits" retype.slurm)
dep="afterok:$crops"
if [ ${#extras[@]} -gt 0 ]; then dep="$dep,afterany:$(IFS=:; echo "${extras[*]}")"; fi
assemble=$(STEP=assemble sb $CPU_ARGS --job-name=retype-assemble --dependency="$dep" retype.slurm)
line="extras=${extras[*]:-none} edits=$edits crops=$crops assemble=$assemble"
for LEG in $LEGS; do
  train=$(STEP=train LEG="$LEG" sb $PARTITION_ARGS --job-name="retype-train-$LEG" --dependency="afterok:$assemble" \
          retype.slurm)
  ids="${extras[*]:-} $edits $crops $assemble $train"
  eval_id=$(STEP=eval LEG="$LEG" JOB_IDS="$(echo $ids | tr ' ' ',')" sb $PARTITION_ARGS \
            --job-name="retype-eval-$LEG" --dependency="afterok:$train" retype.slurm)
  line="$line $LEG: train=$train eval=$eval_id"
done
echo "$(date -u +%FT%TZ) $line" | tee -a retype_jobs.txt
