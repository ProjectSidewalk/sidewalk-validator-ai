#!/bin/bash
# Submit the whole re-typing chain (issue #13) on klone, from the repo root of this branch. Two steps:
#
#   # 1. login node (network): the labelEdits exports, from the live city list. The chain refuses to start without them,
#   #    because a fetch on a compute node without network would kill every job after it.
#   python retype.py fetch-edits --out /gscratch/scrubbed/$USER/retype-2026-11/retype-data/label_edits
#   # 2. submit
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/retype-2026-11 \
#       PANO_STORE=/path/to/store ./retype_submit.sh
#
# extra x4 (GPU, parallel) and edits (CPU) first; crops (GPU) after edits; assemble (CPU) after crops and after every
# extra has ended (afterany: an extra type that fails, e.g. Other with too few labels, is left out, its class stays);
# then per leg train (GPU, TRAIN_TIME, default 24 h) -> eval (GPU) -> report (CPU). A requeued (preempted) job keeps
# its id, so the dependencies survive; training resumes from its last checkpoint.
#
# report needs the network (public/private comes from the live /v3/api/cities). If the compute nodes cannot reach it,
# only the report job fails; rerun it on a login node (minutes of CPU):
#   STEP=report LEG=complement DATA=... RUNS=... bash retype.slurm
#
# Job ids are appended to retype_jobs.txt. Optional: LEGS (default "complement other"; "complement" runs one leg),
# EXTRAS (default "NoSidewalk Signal Other Occlusion"), SKIP_EXTRAS=1 (the extra types already exist under DATA),
# GPU_CONSTRAINT (default "a40|l40|l40s|a100|h200"), PARTITION_ARGS for the GPU jobs (e.g.
# "--partition=gpu-l40s --constraint=l40s"; later options win), TRAIN_TIME (default 24:00:00), ALLOW_NODE_FETCH=1
# (let the edits job fetch, for compute nodes with network), and every variable retype.slurm reads (OUT, RAWLABELS,
# EDIT_RAWLABELS, HOLDOUT, SEED, EPOCHS, CORRECT_PROPORTION, PUBLIC_CITIES, EXPERTS).
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"; PANO_STORE="${PANO_STORE:?set PANO_STORE}"
OUT="${OUT:-$RUNS/retype-data}"
export DATA RUNS PANO_STORE OUT
LEGS="${LEGS:-complement other}"
EXTRAS="${EXTRAS:-NoSidewalk Signal Other Occlusion}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-a40|l40|l40s|a100|h200}"
PARTITION_ARGS="${PARTITION_ARGS:-}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
# CPU steps: no GPU and no GPU constraint, so they do not queue for GPU nodes.
CPU_ARGS="--gpus=0 --cpus-per-task=4 --mem=48G --time=2:00:00"
GPU_ARGS="--gpus=1 --constraint=$GPU_CONSTRAINT"
[ -f retype.slurm ] || { echo "run from the repo root" >&2; exit 2; }
[ -d logs ] || mkdir -p logs
for T in CurbRamp NoCurbRamp Obstacle SurfaceProblem Crosswalk; do
  [ -f "$DATA/$T/split.csv" ] && [ -f "$DATA/$T/predictions.csv" ] || { echo "no $DATA/$T/split.csv or predictions.csv" >&2; exit 2; }
done
if [ ! -f "$OUT/label_edits/fetch.json" ] && [ "${ALLOW_NODE_FETCH:-0}" != 1 ]; then
  echo "no $OUT/label_edits/fetch.json: fetch the labelEdits exports here first (login node, network):" >&2
  echo "  python retype.py fetch-edits --out $OUT/label_edits" >&2
  echo "(or ALLOW_NODE_FETCH=1 if the compute nodes have network)" >&2
  exit 2
fi
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
sb() { sbatch --parsable --kill-on-invalid-dep=yes "$@"; }
extras=()
if [ "${SKIP_EXTRAS:-0}" != 1 ]; then
  for LT in $EXTRAS; do
    extras+=("$(STEP=extra LT="$LT" sb $GPU_ARGS --time=12:00:00 $PARTITION_ARGS --job-name="retype-extra-$LT" \
                retype.slurm)")
  done
fi
edits=$(STEP=edits sb $CPU_ARGS --job-name=retype-edits retype.slurm)
crops=$(STEP=crops sb $GPU_ARGS --time=2:00:00 $PARTITION_ARGS --job-name=retype-crops --dependency="afterok:$edits" \
        retype.slurm)
dep="afterok:$crops"
if [ ${#extras[@]} -gt 0 ]; then dep="$dep,afterany:$(IFS=:; echo "${extras[*]}")"; fi
assemble=$(STEP=assemble sb $CPU_ARGS --job-name=retype-assemble --dependency="$dep" retype.slurm)
line="extras=${extras[*]:-none} edits=$edits crops=$crops assemble=$assemble"
for LEG in $LEGS; do
  train=$(STEP=train LEG="$LEG" sb $GPU_ARGS --time="$TRAIN_TIME" $PARTITION_ARGS --job-name="retype-train-$LEG" \
          --dependency="afterok:$assemble" retype.slurm)
  eval_id=$(STEP=eval LEG="$LEG" sb $GPU_ARGS --time=6:00:00 $PARTITION_ARGS --job-name="retype-eval-$LEG" \
            --dependency="afterok:$train" retype.slurm)
  ids="${extras[*]:-} $edits $crops $assemble $train $eval_id"
  report=$(STEP=report LEG="$LEG" JOB_IDS="$(echo $ids | tr ' ' ',')" sb $CPU_ARGS --job-name="retype-report-$LEG" \
           --dependency="afterok:$eval_id" retype.slurm)
  line="$line $LEG: train=$train eval=$eval_id report=$report"
done
echo "$(date -u +%FT%TZ) $line" | tee -a retype_jobs.txt
