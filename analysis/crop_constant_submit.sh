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
#   TYPES       label types (default "CurbRamp SurfaceProblem": stage 1)                decided 2026-11-25 (decision 1)
# The seeds are not a setting: they are REGISTERED_SEEDS in analysis/crop_constant_sweep.py (42 to 46, decided
# 2026-11-25, decision 2), read here through `registered-seeds`, and compare reads the same constant. Changing them is a commit.
#   ARMS        arms to train (default "4000 6100 8000"; stage 2: "<the passing arm> 6100"; must include 6100)
#   RESTAGE     1 (default) stages into DEST; 0 reads PANO_STORE as it is (nothing is pulled)
#   DEST        the stage destination (default $W/store); refused unless it is inside $W
#   SHARED_STORE a store another study staged, read in place and never written (decision 3, 2026-11-25: the
#               context-arms store, /gscratch/scrubbed/$USER/pano-stage/context-arms). When set: before anything is
#               submitted, `shared-store-check` refuses a store without its .pano-store marker, one that would be
#               more than 20 days old by the end of the re-cut (its age plus 2 days of jobs; scrubbed purges at 21),
#               and any SHARED_TYPES panorama in neither DEST nor the shared store (listed in
#               $W/shared_store_missing.txt), with RESTAGE=1 too. With RESTAGE=1 the stage list leaves out every file
#               the shared store holds, so DEST receives only the rest (CurbRamp's). PANO_STORE defaults to
#               DEST:SHARED_STORE; preflight requires every SHARED_TYPES panorama in one of them, and preflight and
#               each recrop task re-check the store's marker and age when they start. Paths are compared with
#               symlinks resolved.
#   SHARED_TYPES the types the shared store is meant to cover (default SurfaceProblem); only with SHARED_STORE
#   PANO_STORE  the store the re-cut reads (default DEST, or DEST:SHARED_STORE); a colon-separated list is searched
#               in order, each panorama read from the first store that holds it
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
RESTAGE="${RESTAGE:-1}"; DEST="${DEST:-$W/store}"; SHARED_STORE="${SHARED_STORE:-}"
SHARED_TYPES="${SHARED_TYPES:-SurfaceProblem}"
strip_slashes() { local p="$1"; while [ "${#p}" -gt 1 ] && [ "${p%/}" != "$p" ]; do p="${p%/}"; done; echo "$p"; }
DEST=$(strip_slashes "$DEST"); SHARED_STORE=$(strip_slashes "$SHARED_STORE")   # before PANO_STORE is built from them
if [ -n "$SHARED_STORE" ]; then PANO_STORE="${PANO_STORE:-$DEST:$SHARED_STORE}"; else PANO_STORE="${PANO_STORE:-$DEST}"; fi
STAGE_TIME="${STAGE_TIME:-24:00:00}"; RECROP_TIME="${RECROP_TIME:-8:00:00}"; PARTITION_ARGS="${PARTITION_ARGS:-}"
# A path with every symlink resolved, without klone's /mmfs1 prefix (/gscratch and /mmfs1/gscratch are one tree), so
# an alias, a doubled slash or a symlink cannot get past the containment guards below. Missing parts are kept as
# written (os.path.realpath is not strict), so this also works before DEST exists.
canon() {
  "$PYTHON" -c 'import os, sys; p = os.path.realpath(sys.argv[1]); print(p[6:] if p.startswith("/mmfs1/") else p)' "$1"
}
# The stage job's own default destination is the lab's durable store, which cannot take this volume: DEST is always
# passed, and only inside the working directory.
case "$DEST" in
  *..*|*:*) echo "refusing: DEST=$DEST contains '..' or ':'" >&2; exit 2 ;;
  "$W"/*) ;;
  *) echo "refusing: DEST=$DEST is outside $W" >&2; exit 2 ;;
esac
[ ! -L "$DEST" ] || { echo "refusing: DEST=$DEST is a symlink (the stage job would write wherever it points)" >&2; exit 2; }
# A shared store belongs to another study: it is read in place and never staged into (the stage job removes DEST's
# marker while it pulls), so DEST may be neither the shared store nor inside it, nor the other way round, after every
# symlink is resolved.
if [ -n "$SHARED_STORE" ]; then
  case "$SHARED_STORE" in
    /*) ;;
    *) echo "refusing: SHARED_STORE=$SHARED_STORE must be an absolute path" >&2; exit 2 ;;
  esac
  case "$SHARED_STORE" in
    *..*|*:*|*,*|*" "*) echo "refusing: SHARED_STORE=$SHARED_STORE contains '..', ':', ',' or a space" >&2; exit 2 ;;
  esac
  CDEST=$(canon "$DEST"); CSHARED=$(canon "$SHARED_STORE")
  case "$CDEST/" in
    "$CSHARED"/*) echo "refusing: DEST=$DEST ($CDEST) is the shared store ($CSHARED) or inside it" >&2; exit 2 ;;
  esac
  case "$CSHARED/" in
    "$CDEST"/*) echo "refusing: SHARED_STORE=$SHARED_STORE ($CSHARED) is inside DEST ($CDEST)" >&2; exit 2 ;;
  esac
  [[ ":$PANO_STORE:" == *":$SHARED_STORE:"* ]] \
    || { echo "refusing: PANO_STORE=$PANO_STORE does not list SHARED_STORE=$SHARED_STORE" >&2; exit 2; }
fi
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

SKIP=()  # the stage list's --skip-store, only with a shared store
if [ -n "$SHARED_STORE" ]; then
  # Read-only, so it runs under DRY_RUN=1 too: a stale or unmarked shared store refuses the dry run as well. A
  # SHARED_TYPES panorama in neither DEST nor the shared store refuses the launch here, before any pull, with RESTAGE=1
  # as well: the context-arms pull already asked makelab2 for it, so it is most likely missing there too, and pulling
  # the rest (about 107 GB) first would only end in preflight's refusal. The age check adds CHAIN_DAYS (2) of jobs.
  CHECK=(analysis/crop_constant_sweep.py shared-store-check --root "$R3" --types "$TYPES_CSV"
         --shared-store "$SHARED_STORE" --shared-types "${SHARED_TYPES// /,}" --search-store "$PANO_STORE"
         --require-complete)
  [ "$DRY_RUN" = 1 ] || CHECK+=(--missing-out "$W/shared_store_missing.txt" --out "$W/shared_store_check.json")
  echo "$PYTHON ${CHECK[*]}"
  CHECK_OUT=$("$PYTHON" "${CHECK[@]}") || {
    [ -z "$CHECK_OUT" ] || echo "$CHECK_OUT"
    echo "refusing: the shared store check failed; nothing submitted. A missing $SHARED_TYPES panorama has to be" \
         "found on makelab2 by hand and staged into DEST first (or its rows dropped by Jon's decision)" >&2
    exit 2
  }
  echo "$CHECK_OUT"
  SHARED_STAGED=$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^stage_date=//p')
  # preflight and every recrop task check the store again (marker, age) and stop with PanoStoreUnavailable.
  ENVS="$ENVS,COMPLETE_TYPES=${SHARED_TYPES// /:},SHARED_STORE=$SHARED_STORE,SHARED_STAGED=$SHARED_STAGED"
  SKIP=(--skip-store "$SHARED_STORE")
fi

DEP=()  # empty without a stage job; expanded as ${DEP[@]+...} for bash 3.2's set -u
if [ "$RESTAGE" = 1 ]; then
  run "$PYTHON" analysis/crop_constant_sweep.py stage-list --root "$R3" --types "$TYPES_CSV" --out "$W/stage_files.txt" \
      --exclude-store "$DEST" ${SKIP[@]+"${SKIP[@]}"}
  if [ "$DRY_RUN" != 1 ] && [ ! -s "$W/stage_files.txt" ]; then
    # The stage job cannot take an empty list (no batch to pull, and no marker written for DEST).
    if [ -n "$SHARED_STORE" ]; then
      echo "refusing: nothing to stage (the shared store holds every panorama); use RESTAGE=0 PANO_STORE=$SHARED_STORE" >&2
    else
      echo "refusing: the stage list $W/stage_files.txt is empty" >&2
    fi
    exit 2
  fi
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
