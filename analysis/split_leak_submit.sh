#!/bin/bash
# Submit one stage of the split-leak study (issue #3; docs/split-leak-2026-11.md) on klone, from the repo root of a
# checkout of this branch:
#
#   STAGE=0 DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/split-leak \
#       analysis/split_leak_submit.sh          # CPU, minutes: make-split, check, bound -> $RUNS/report
#   STAGE=1 DATA=... RUNS=... analysis/split_leak_submit.sh    # L arm, seed 42, five types (~30 GPU-h)
#   STAGE=2 DATA=... RUNS=... analysis/split_leak_submit.sh    # L arm, seeds 43 and 44 (~60 GPU-h)
#   DRY_RUN=1 STAGE=1 DATA=... RUNS=... analysis/split_leak_submit.sh   # print the commands, submit and write nothing
#
# Stages 1 and 2 refuse until Stage 0's outputs exist (the leaky splits, their exposure tables, design.json and
# bound.json): the planning figures go into the doc before any GPU job. Each GPU stage is one array, one task per
# (type, arm, seed), at most MAX_PARALLEL (default 4) running at once, then one report job (afterany).
# FRESH_G=1 interleaves a fresh grouped arm (G) with L in the same array (~95 more GPU-h over three seeds; a Jon
# decision, J1); without it the report uses round 3's stored grouped predictions (descriptive inflation). Stage 2
# with FRESH_G=1 refuses unless Stage 1 also ran with FRESH_G=1 (every type's G-s42/predictions.csv), since its
# report reads a fresh grouped seed 42.
# Optional: RAWLABELS (Stage 0; default $DATA/rawlabels), TYPES (space list), PARTITION_ARGS for the GPU array.
# Job ids are appended to split_leak_jobs.txt in the submit directory.
set -euo pipefail
STAGE="${STAGE:?set STAGE (0, 1 or 2)}"; DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
RAWLABELS="${RAWLABELS:-$DATA/rawlabels}"
TYPES="${TYPES:-CurbRamp NoCurbRamp Obstacle SurfaceProblem Crosswalk}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"; FRESH_G="${FRESH_G:-0}"; PARTITION_ARGS="${PARTITION_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
REPORT="$RUNS/report"
case "$RUNS/" in "$DATA"/*) echo "RUNS must not be under DATA (nothing is written under round 3's data)" >&2; exit 2 ;; esac

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
log() { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a split_leak_jobs.txt; fi; }
[ "$DRY_RUN" = 1 ] || mkdir -p logs

case "$STAGE" in
  0)
    [ "$DRY_RUN" = 1 ] || [ -d "$RAWLABELS" ] || { echo "no $RAWLABELS (set RAWLABELS)" >&2; exit 2; }
    sb --job-name=leak-stage0 --export="ALL,STEP=stage0,DATA=$DATA,RUNS=$RUNS,RAWLABELS=$RAWLABELS,TYPES=${TYPES// /:}" \
       analysis/split_leak_cpu.slurm
    echo "stage0 $JOB" | log
    exit 0 ;;
  1) SEEDS="42"; ALL_SEEDS="42" ;;
  2) SEEDS="43 44"; ALL_SEEDS="42 43 44" ;;
  *) echo "STAGE must be 0, 1 or 2" >&2; exit 2 ;;
esac

for f in "$REPORT/design.json" "$REPORT/bound.json"; do
  [ -f "$f" ] || { echo "no $f: run Stage 0 (STAGE=0) and put its figures in the doc first" >&2; exit 2; }
done
ARMS="L"; [ "$FRESH_G" = 1 ] && ARMS="L G"
if [ "$FRESH_G" = 1 ] && [ "$STAGE" = 2 ]; then
  for LT in $TYPES; do
    [ -f "$RUNS/$LT/G-s42/predictions.csv" ] || { echo "no $RUNS/$LT/G-s42/predictions.csv: FRESH_G=1 at Stage 2" \
      "needs the fresh grouped seed 42 (run Stage 1 with FRESH_G=1 first)" >&2; exit 2; }
  done
fi
TASKS="$RUNS/split_leak_tasks_stage$STAGE.txt"
LINES=()
for SEED in $SEEDS; do
  for LT in $TYPES; do
    for f in "$RUNS/$LT/splits/label_level_s$SEED.csv" "$RUNS/$LT/splits/exposure_s$SEED.csv"; do
      [ -f "$f" ] || { echo "no $f: run Stage 0 (STAGE=0) first" >&2; exit 2; }
    done
    for ARM in $ARMS; do LINES+=("$LT $ARM $SEED"); done
  done
done
for i in "${!LINES[@]}"; do echo "task $i: ${LINES[$i]}"; done
[ "$DRY_RUN" = 1 ] || printf '%s\n' "${LINES[@]}" > "$TASKS"
sb $PARTITION_ARGS --job-name="leak-s$STAGE" --array="0-$((${#LINES[@]} - 1))%${MAX_PARALLEL:-4}" \
   --export="ALL,DATA=$DATA,RUNS=$RUNS,TASKS=$TASKS" analysis/split_leak.slurm
ARRAY=$JOB
if [ "$FRESH_G" = 1 ]; then G_SOURCE=fresh; G_SEEDS="$ALL_SEEDS"; else G_SOURCE=stored; G_SEEDS="42 43 44"; fi
sb --job-name="leak-report-s$STAGE" --dependency="afterany:$ARRAY" \
   --export="ALL,STEP=report,DATA=$DATA,RUNS=$RUNS,TYPES=${TYPES// /:},SEEDS=${ALL_SEEDS// /:},G_SEEDS=${G_SEEDS// /:},G_SOURCE=$G_SOURCE" \
   analysis/split_leak_cpu.slurm
echo "stage$STAGE array=$ARRAY report=$JOB tasks=$TASKS" | log
