#!/bin/bash
# Submit the crop-constant sweep (issue #21; docs/crop-constant-2026-11.md) on klone, from the repo root of the clone
# the run uses ($W/sidewalk-validator-ai), with the conda env active. NOT RUN in the PR that adds it; the run waits
# for Jon's go.
#
#   R3=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 analysis/crop_constant_submit.sh
#   DRY_RUN=1 R3=... analysis/crop_constant_submit.sh        # print every command, submit nothing
#
# Chain, each step afterok on the one before, --kill-on-invalid-dep=yes so a failure cancels what follows:
#   1. stage (RESTAGE=1): analysis/ai_labels_stage.slurm, unchanged, pulls stage_files.txt into DEST. It opens one
#      ssh connection per 1,000-file batch, sequentially, and stops on any rsync error but 23 (missing files).
#   2. preflight (CPU): G0 and the store check.          3. recrop (GPU array): one task per panorama chunk.
#   4. gates (CPU): G2 and G3; writes c4000/, c6100/, c8000/ only when every gate passes.
#   5. trainings: analysis/confident_learning.slurm, unchanged, VARIANT=baseline, one per arm x type x seed, with
#      DATA=$W/c<C> and RUNS=$W/runs/c<C> (one RUNS directory per arm), TAG=s<seed>, KEEP_CKPT=0.
#
# Everything is written under W=/gscratch/scrubbed/$USER/crop-constant-2026-11 (fixed). Settings, all optional:
#   TYPES       label types (default "CurbRamp SurfaceProblem": stage 1)                     PROVISIONAL (decision 1)
# The seeds are not a setting: they are REGISTERED_SEEDS in analysis/crop_constant_sweep.py (42 to 46, PROVISIONAL,
# decision 2), read here through `registered-seeds`, and compare reads the same constant. Changing them is a commit.
#   ARMS        arms to train (default "4000 6100 8000"; stage 2: "<the passing arm> 6100"; must include 6100)
#   RESTAGE     1 (default) stages into DEST; 0 reads PANO_STORE as it is (a shared store)  PROVISIONAL (decision 3)
#   DEST        the stage destination (default $W/store); refused unless it is inside $W
#   PANO_STORE  the store the re-cut reads (default DEST)
#   STAGE_TIME  the stage job's --time (default 24:00:00; about 215 to 260 GB for stage 1)
#   RECROP_TIME the recrop array's --time (default 8:00:00); it also fixes the chunking (labels x 3.5 s x 1.5)
#   RAWLABELS   round 3's rawLabels cache (default $R3/rawlabels)
#   PARTITION_ARGS extra sbatch arguments for the trainings; PYTHON (default python)
set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"; PYTHON="${PYTHON:-python}"
[ -f .gitmodules ] && [ -f analysis/crop_constant_sweep.py ] || { echo "Run this from the repo root." >&2; exit 2; }
REPO=$(pwd -P)
W="/gscratch/scrubbed/${USER:?}/crop-constant-2026-11"
R3="${R3:?set R3 (round 3 data root: <type>/split.csv, generation_run.json, the stored crops)}"
RAWLABELS="${RAWLABELS:-$R3/rawlabels}"
[ -z "${SEEDS:-}" ] || { echo "refusing: SEEDS is not a setting; edit REGISTERED_SEEDS and commit" >&2; exit 2; }
TYPES="${TYPES:-CurbRamp SurfaceProblem}"; ARMS="${ARMS:-4000 6100 8000}"
RESTAGE="${RESTAGE:-1}"; DEST="${DEST:-$W/store}"; PANO_STORE="${PANO_STORE:-$DEST}"
STAGE_TIME="${STAGE_TIME:-24:00:00}"; RECROP_TIME="${RECROP_TIME:-8:00:00}"; PARTITION_ARGS="${PARTITION_ARGS:-}"
# The stage job's own default destination is the lab's durable store, which cannot take this volume: DEST is always
# passed, and only inside the working directory.
case "$DEST" in
  *..*) echo "refusing: DEST=$DEST contains '..'" >&2; exit 2 ;;
  "$W"/*) ;;
  *) echo "refusing: DEST=$DEST is outside $W" >&2; exit 2 ;;
esac
[[ " $ARMS " == *" 6100 "* ]] || { echo "refusing: ARMS=\"$ARMS\" must include the 6100 baseline" >&2; exit 2; }
TYPES_ARG="${TYPES// /:}"; TYPES_CSV="${TYPES// /,}"
N=$("$PYTHON" analysis/crop_constant_sweep.py chunks --root "$R3" --types "$TYPES_CSV" --time-limit "$RECROP_TIME")
SEEDS=$("$PYTHON" analysis/crop_constant_sweep.py registered-seeds)

J=0
SBATCH=(sbatch --parsable --kill-on-invalid-dep=yes)   # one array for both paths, so the dry run prints what runs
sb() {
  if [ "$DRY_RUN" = 1 ]; then
    J=$((J + 1)); echo "${SBATCH[*]} $*"; JOB="dry$J"
  else
    JOB=$("${SBATCH[@]}" "$@")
  fi
}
run() { if [ "$DRY_RUN" = 1 ]; then echo "$*"; else "$@"; fi; }
log() { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a "$W/crop_constant_jobs.txt"; fi; }
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$W/logs" logs
  # A real write before anything is submitted: the stage job (the largest writer) and the trainings do not test.
  # Scrubbed has refused data writes while a 0-byte touch succeeded (running-on-hyak.md).
  dd if=/dev/zero of="$W/.write_test.$$" bs=1M count=8 status=none \
    || { echo "refusing: cannot write 8 MiB under $W" >&2; exit 2; }
  rm -f "$W/.write_test.$$"
fi
ENVS="REPO=$REPO,R3=$R3,RAWLABELS=$RAWLABELS,TYPES=$TYPES_ARG,PANO_STORE=$PANO_STORE"
CPU=(--gpus=0 --constraint= --cpus-per-task=2 --mem=16G --time=2:00:00)

DEP=()  # empty without a stage job; expanded as ${DEP[@]+...} for bash 3.2's set -u
if [ "$RESTAGE" = 1 ]; then
  run "$PYTHON" analysis/crop_constant_sweep.py stage-list --root "$R3" --types "$TYPES_CSV" --out "$W/stage_files.txt" \
      --exclude-store "$DEST"
  sb --chdir="$W" --job-name=cc-stage --time="$STAGE_TIME" --export="ALL,RUN=$W,DEST=$DEST" \
     "$REPO/analysis/ai_labels_stage.slurm"
  echo "stage=$JOB" | log
  DEP=(--dependency="afterok:$JOB")
fi
sb --chdir="$W" --job-name=cc-preflight "${CPU[@]}" ${DEP[@]+"${DEP[@]}"} --export="ALL,STEP=preflight,$ENVS" \
   "$REPO/analysis/crop_constant_recrop.slurm"
PRE=$JOB
sb --chdir="$W" --job-name=cc-recrop --array="0-$((N - 1))" --time="$RECROP_TIME" --dependency="afterok:$PRE" \
   --export="ALL,STEP=recrop,$ENVS,TIME_LIMIT=$RECROP_TIME" "$REPO/analysis/crop_constant_recrop.slurm"
REC=$JOB
sb --chdir="$W" --job-name=cc-gates "${CPU[@]}" --dependency="afterok:$REC" --export="ALL,STEP=gates,$ENVS" \
   "$REPO/analysis/crop_constant_recrop.slurm"
GATES=$JOB
echo "preflight=$PRE recrop=$REC ($N tasks) gates=$GATES" | log
for C in $ARMS; do
  for LT in $TYPES; do
    for S in ${SEEDS//,/ }; do
      sb $PARTITION_ARGS --job-name="cc-$LT-c$C-s$S" --dependency="afterok:$GATES" \
         --export="ALL,LT=$LT,VARIANT=baseline,DATA=$W/c$C,RUNS=$W/runs/c$C,SEED=$S,TAG=s$S,KEEP_CKPT=0" \
         analysis/confident_learning.slurm
      echo "train $LT c$C s$S $JOB" | log
    done
  done
done
echo "compare (on the Mac, from a copy of $W): python analysis/crop_constant_sweep.py compare --work <copy> \\"
echo "    --types $TYPES_CSV --control-root <round 3 root> --logs-dir <copy of logs> --out compare.json"
