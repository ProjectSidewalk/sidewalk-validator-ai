#!/bin/bash
# Issue #74, the CPU steps of the round-4 recipe on klone (analysis/round4_submit.sh submits them):
#   STEP=prep           the gold-free mirror R4 (round4.py prep, --round3-ref), cl-drop for CL_TYPES (variants),
#                       per-variant slice training counts (slice-train), the r3crops gold manifests (gold-manifest)
#   STEP=gold-manifest  only the gold manifests (CROPS=r3crops|uniform; uniform needs CROPS_ROOT)
#   STEP=variants       only the variants (WHICH, SEEDS, CL_TYPES)
#   STEP=collect        round4.py collect: run manifest, untagged baseline copies, small artifacts to KEEP
# Deterministic and idempotent: a rerun (or a requeue after preemption) rewrites the same files.
# Lists (TYPES, CL_TYPES, SEEDS) are comma- or colon-separated; EXPECT items are joined by '+' (no spaces).
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=round4-disagree-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STEP="${STEP:?set STEP (prep|gold-manifest|variants|collect)}"
DATA="${DATA:?set DATA (round 3, read-only)}"; R4="${R4:?set R4}"; GOLD_UIDS="${GOLD_UIDS:?set GOLD_UIDS}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
TYPES="${TYPES:-Crosswalk:CurbRamp:NoCurbRamp:Obstacle:SurfaceProblem}"
CL_TYPES="${CL_TYPES:-Crosswalk:CurbRamp:NoCurbRamp:Obstacle}"
WHICH="${WHICH:-cl-drop}"; SEEDS="${SEEDS:-42:43:44}"; CROPS="${CROPS:-r3crops}"
# Never /tmp (node-local on klone) and never the makelab fileset for the mirror (no room): scrubbed only.
case "$R4" in /tmp*|/gscratch/makelab*) echo "refusing R4=$R4 (use /gscratch/scrubbed)" >&2; exit 2 ;; esac
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
python -c "import pandas; assert pandas.__version__.startswith('2.'), pandas.__version__"
R="python -u analysis/round4.py"
manifests() {
  if [ "$CROPS" = uniform ]; then
    $R gold-manifest --gold-uids "$GOLD_UIDS" --crops-root "${CROPS_ROOT:?set CROPS_ROOT}" --mode uniform \
        --types "${TYPES//:/,}" --out "$R4/gold/uniform"
  else
    $R gold-manifest --gold-uids "$GOLD_UIDS" --split-root "$DATA" --mode r3crops --types "${TYPES//:/,}" \
        --out "$R4/gold/r3crops"
  fi
}
case "$STEP" in
  prep)
    RAWLABELS="${RAWLABELS:?set RAWLABELS (the cached rawLabels exports, for the 3 m scene merge)}"
    $R prep --data "$DATA" --r4 "$R4" --gold-uids "$GOLD_UIDS" --types "${TYPES//:/,}" --rawlabels "$RAWLABELS" \
        --round3-ref
    $R variants --r4 "$R4" --data "$DATA" --types "${CL_TYPES//:/,}" --which cl-drop
    $R slice-train --r4 "$R4" --types "${TYPES//:/,}" --variants round3,baseline,cl-drop --out "$R4/slice_train"
    CROPS=r3crops manifests
    ;;
  gold-manifest) manifests ;;
  variants)
    $R variants --r4 "$R4" --data "$DATA" --types "${CL_TYPES//:/,}" --which "$WHICH" --seeds "${SEEDS//:/,}"
    ;;
  collect)
    RUNS="${RUNS:?set RUNS}"; KEEP="${KEEP:?set KEEP}"; EXPECT="${EXPECT:?set EXPECT}"
    # EXPECT: round4.py collect --expect items joined by '+', e.g. baseline:42,43,44+cl-drop:42,43,44:CurbRamp,Crosswalk
    read -r -a EXP <<< "${EXPECT//+/ }"
    $R collect --r4 "$R4" --runs "$RUNS" --keep "$KEEP" --types "${TYPES//:/,}" --expect "${EXP[@]}" \
        --jobs "${JOBS:-}" --gold-uids "$GOLD_UIDS" --extra ${SUBMITTED:+$SUBMITTED/*}
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "$(date) done $STEP"
