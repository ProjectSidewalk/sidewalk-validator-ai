#!/bin/bash
# One command for the NoSidewalk pilot (issue 42, phase B; docs/nosidewalk-2026-11.md), from the repo root on klone,
# in the activated environment:
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/nosidewalk-2026-11 RUNS=/gscratch/scrubbed/$USER/nosidewalk \
#       PANO_STORE=/path/to/Panoramas EXPERTS=... ./nosidewalk_submit.sh
#   DRY_RUN=1 DATA=... RUNS=... PANO_STORE=... ./nosidewalk_submit.sh     # print the commands, submit nothing
#
# 1. rawLabels: CACHE (default $DATA/rawlabels-ns) is used as it is when it holds a .complete marker; otherwise a
#    NoSidewalk-only pull of every public deployment is made into it (analysis/fetch_all.py --label-types NoSidewalk).
#    To include private deployments, as round 3 did, point CACHE at a complete full cache (e.g. round 3's rawlabels/).
# 2. Per arm (depth, whole): one data job (crops with --verdict-unit face, the face-size prior, the face-grouped
#    split holding out HOLDOUT, default tucson-az,chicago-il).
# 3. Per arm and seed (42 43 44): one train job (train, evaluate, calibrate Agree-only), after its arm's data job.
# 4. One report job after every train job: analysis/nosidewalk_pilot.py -> $DATA/report/pilot.{json,md}.
# Job ids are appended to nosidewalk_jobs.txt. EXPERTS stays in the environment (sbatch exports it); never in a file.
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"; PANO_STORE="${PANO_STORE:?set PANO_STORE}"
CACHE="${CACHE:-$DATA/rawlabels-ns}"
HOLDOUT="${HOLDOUT-tucson-az,chicago-il}"
ARMS="${ARMS:-depth whole}"
SEEDS="${SEEDS:-42 43 44}"
DRY_RUN="${DRY_RUN:-0}"
[ -f .gitmodules ] || { echo "Run this from the repo root." >&2; exit 1; }
[ -n "${EXPERTS:-}" ] || echo "WARNING: EXPERTS is not set: expert votes will not count (margin and face-margin rules only)." >&2
export DATA RUNS PANO_STORE CACHE HOLDOUT

N=0
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
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

if [ -f "$CACHE/.complete" ]; then
  echo "Using the rawLabels cache $CACHE" >&2
else
  run rm -rf "$CACHE.tmp"
  run mkdir -p "$CACHE.tmp"
  run python -u analysis/fetch_all.py "$CACHE.tmp" --label-types NoSidewalk
  run rm -rf "$CACHE"
  run mv "$CACHE.tmp" "$CACHE"
  run touch "$CACHE/.complete"
fi

[ "$DRY_RUN" = 1 ] || mkdir -p logs
TRAIN=()
for ARM in $ARMS; do
  sb --job-name="ns-data-$ARM" --export=ALL,STEP=data,ARM="$ARM" nosidewalk_pilot.slurm
  DATA_JOB=$JOB
  for SEED in $SEEDS; do
    sb --job-name="ns-train-$ARM-s$SEED" --dependency="afterok:$DATA_JOB" \
       --export=ALL,STEP=train,ARM="$ARM",SEED="$SEED" nosidewalk_pilot.slurm
    TRAIN+=("$JOB")
  done
  echo "$ARM data=$DATA_JOB" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a nosidewalk_jobs.txt; fi; }
done
# The report is CPU work: the free ckpt-all partition without a GPU (the #SBATCH lines are overridden).
sb --job-name=ns-report --partition=ckpt-all --gpus-per-node=0 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G \
   --time=2:00:00 --dependency="afterany:$(IFS=:; echo "${TRAIN[*]}")" --export=ALL,STEP=report nosidewalk_pilot.slurm
echo "train=${TRAIN[*]} report=$JOB" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a nosidewalk_jobs.txt; fi; }
