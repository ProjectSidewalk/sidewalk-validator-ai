#!/bin/bash
# Submit the issue #14 context-arms chain on klone (docs/context-arms-2026-11.md), from a checkout of this branch:
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/context-2026-11 \
#   PANO_STORE=/abs/Panoramas analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem CurbRamp Crosswalk
#
# Per label type: backfill (GPU; writes <DATA>/<LT>/context/) -> the three arm trainings (GPU, parallel, afterok), each
# scored on the round-3 rows with context_model.py predict + evaluate.py --tag <arm>. The round-3 baseline is not
# retrained: its metrics.json/predictions.csv (seed 42) and the confident-learning batch's baseline-s43/-s44 are in
# <DATA>/<LT>/ already. Job ids are appended to context_jobs.txt. Optional: PARTITION_ARGS for the GPU jobs (e.g.
# "--partition=gpu-l40s --constraint=l40s"), RAWLABELS (default $DATA/../rawlabels), ARMS (default all three),
# SKIP_BACKFILL=1 when <DATA>/<LT>/context/ is complete already.
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
RAWLABELS="${RAWLABELS:-$(cd "$DATA/.." && pwd)/rawlabels}"
PARTITION_ARGS="${PARTITION_ARGS:-}"
ARMS="${ARMS:-two_scale view_heatmap side_info}"
SKIP_BACKFILL="${SKIP_BACKFILL:-0}"
[ $# -gt 0 ] || { echo "usage: $0 <LabelType> [...]" >&2; exit 2; }
[ "$SKIP_BACKFILL" = 1 ] || PANO_STORE="${PANO_STORE:?set PANO_STORE (or SKIP_BACKFILL=1)}"
[ -d logs ] || mkdir -p logs
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
sb() { sbatch --parsable --kill-on-invalid-dep=yes "$@"; }
for LT in "$@"; do
  [ -f "$DATA/$LT/split.csv" ] || { echo "no $DATA/$LT/split.csv" >&2; exit 2; }
  dep=()
  bf=none
  if [ "$SKIP_BACKFILL" != 1 ]; then
    bf=$(STEP=backfill LT="$LT" DATA="$DATA" PANO_STORE="$PANO_STORE" RAWLABELS="$RAWLABELS" \
         sb $PARTITION_ARGS --job-name="ctx-bf-$LT" analysis/context_arms.slurm)
    dep=(--dependency="afterok:$bf")
  fi
  jobs=()
  for ARM in $ARMS; do
    # side_info reads only split.csv, so it needs no backfill and starts at once.
    if [ "$ARM" = side_info ]; then d=(); else d=(${dep[@]+"${dep[@]}"}); fi
    jobs+=("$(STEP=train LT="$LT" ARM="$ARM" DATA="$DATA" RUNS="$RUNS" \
              sb $PARTITION_ARGS --job-name="ctx-$LT-$ARM" ${d[@]+"${d[@]}"} analysis/context_arms.slurm)")
  done
  echo "$LT backfill=$bf arms=${jobs[*]}" | tee -a context_jobs.txt
done
