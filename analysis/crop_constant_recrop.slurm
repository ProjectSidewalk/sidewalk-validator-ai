#!/bin/bash
#SBATCH --job-name=cc-recrop
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Issue #21, the crop-constant sweep (docs/crop-constant-2026-11.md). NOT RUN in the PR that adds it; the run waits
# for Jon's go. One job file, four steps (STEP):
#
#   preflight  CPU   G0 (round 3's pins, the panorama-tools commit, versions) and the store check -> preflight.json
#   recrop     GPU   one array task per panorama chunk: one read and one torch render per label, three crops
#                    (4000 / 6100 / 8000) into c<C>/<type>/; resumable, so a ckpt-all requeue is harmless
#   gates      CPU   G2 (row loss) and G3 (re-cut parity); writes the arm directories only when both pass
#   SMOKE=1          20 labels of chunk 0 into smoke/, printing seconds per label (measure before the full submit)
#
# analysis/crop_constant_submit.sh submits the chain with the right dependencies, --array, --time and the CPU
# overrides; submitting by hand:
#
#   W=/gscratch/scrubbed/$USER/crop-constant-2026-11
#   cd $W && SMOKE=1 REPO=$W/sidewalk-validator-ai R3=<round 3 root> PANO_STORE=$W/store \
#       sbatch --array=0 $W/sidewalk-validator-ai/analysis/crop_constant_recrop.slurm
#
# PANO_STORE may be a colon-separated search path (decision 3: $W/store:/gscratch/scrubbed/$USER/pano-stage/
# context-arms); each panorama is read from the first store that holds it. COMPLETE_TYPES (colon-separated, set by the
# submit script with SHARED_STORE) makes preflight fail when any panorama of those types is in no store.
#
# The array's chunking depends on TIME_LIMIT, which must equal this job's --time: a chunk holds at most
# --time / (3.5 s x 1.5) labels, so a slow node (3.5 s per label, the decode-bound seg-prior rate) still finishes a
# chunk inside --time and a preempted task does not time out in a requeue loop.
set -euo pipefail
W=$(pwd -P)
# Only ever run in this study's scrubbed directory: never /gscratch/makelab, never another study's directory.
[[ "$W" =~ ^(/mmfs1)?/gscratch/scrubbed/${USER:?}/crop-constant-2026-11$ ]] || {
  echo "refusing to run in $W: submit from /gscratch/scrubbed/$USER/crop-constant-2026-11 (sbatch --chdir)"; exit 2; }
STEP="${STEP:-recrop}"
REPO="${REPO:?set REPO (the clone this run uses)}"; R3="${R3:?set R3 (round 3 data root)}"
RAWLABELS="${RAWLABELS:-$R3/rawlabels}"; TYPES="${TYPES:-CurbRamp:SurfaceProblem}"
PANO_STORE="${PANO_STORE:-$W/store}"; TIME_LIMIT="${TIME_LIMIT:-8:00:00}"
# A real write, not touch: scrubbed has refused data writes while a 0-byte touch succeeded (running-on-hyak.md).
if ! dd if=/dev/zero of="$W/.write_test.$$" bs=1M count=8 status=none; then echo "refusing to run: cannot write 8 MiB under $W"; exit 2; fi
rm -f "$W/.write_test.$$"
# sbatch --export=ALL carries the submitting shell's CONDA_PREFIX (the submit script runs with the env active), and
# Lmod then refuses `module load conda` ("Conda is already active"), which set -e makes fatal: job 43577059 stopped
# here on 2026-11-27 and took the whole chain with it. Clear it; `conda activate` below sets it again.
unset CONDA_PREFIX
module load conda
eval "$(conda shell.bash hook)"
conda activate "${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
export PYTHONDONTWRITEBYTECODE=1
echo "commit $(git -C "$REPO" rev-parse HEAD) step $STEP smoke ${SMOKE:-0} task ${SLURM_ARRAY_TASK_ID:-none}"
CC=(python -u "$REPO/analysis/crop_constant_sweep.py")
COMMON=(--root "$R3" --rawlabels-dir "$RAWLABELS" --types "$TYPES")
# With a shared store (decision 3): preflight and each recrop task check its marker and age again when they start,
# and a recrop task stops (PanoStoreUnavailable) on a no_pano of a COMPLETE_TYPES type rather than record it.
SHARED=()
[ -z "${SHARED_STORE:-}" ] || SHARED=(--shared-store "$SHARED_STORE" --shared-staged "${SHARED_STAGED:?set by the submit script}")
[ -z "${COMPLETE_TYPES:-}" ] || SHARED+=(--complete-types "$COMPLETE_TYPES")
case "$STEP" in
  preflight)
    "${CC[@]}" preflight "${COMMON[@]}" --pano-store "$PANO_STORE" --out "$W/preflight.json" ${SHARED[@]+"${SHARED[@]}"} ;;
  recrop)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "${SMOKE:-0}" = 1 ]; then
      "${CC[@]}" recrop "${COMMON[@]}" --work "$W/smoke" --chunk 0 --time-limit "$TIME_LIMIT" \
          --pano-store "$PANO_STORE" --limit 20 ${SHARED[@]+"${SHARED[@]}"}
    else
      "${CC[@]}" recrop "${COMMON[@]}" --work "$W" --chunk "${SLURM_ARRAY_TASK_ID:?submit as an array}" \
          --time-limit "$TIME_LIMIT" --pano-store "$PANO_STORE" ${SHARED[@]+"${SHARED[@]}"}
    fi ;;
  gates)
    "${CC[@]}" gates "${COMMON[@]}" --work "$W" ;;
  *) echo "unknown STEP $STEP (preflight, recrop, gates)"; exit 2 ;;
esac
echo "$(date +%T) done $STEP"
