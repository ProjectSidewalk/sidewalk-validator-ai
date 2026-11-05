#!/bin/bash
# Issue #13, validation as re-typing: every step of the chain on klone, one STEP per job (retype_submit.sh submits
# them all with dependencies). Restartable: ckpt-all may preempt and requeue; training resumes from its last complete
# checkpoint (run_key.json guards the split hash and the leg's settings) and every other step reruns idempotently.
#
#   STEP=extra LT=NoSidewalk   crops + split for a type with no round-3 crops (GPU; data_generation.py, split_data.py)
#   STEP=edits                 labelEdits per public deployment, type changes, audit.json (CPU; needs the network
#                              unless $OUT/label_edits/fetch.json exists: run `python retype.py fetch-edits` on a
#                              login node first, or copy the folder from a Mac)
#   STEP=crops                 crops for the type-edit labels (GPU, minutes)
#   STEP=assemble              $OUT/retype_split.csv + assemble.json (CPU)
#   STEP=train LEG=complement  fine_tune_retype.py --negatives $LEG (GPU, about 16 h at 12 epochs)
#   STEP=eval LEG=complement   predict, per-type export, evaluate.py, calibrate, report (GPU, about 1 h)
#
#   STEP=train LEG=complement DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 \
#       RUNS=/gscratch/scrubbed/$USER/retype-2026-11 sbatch retype.slurm
#
# DATA is the round-3 data root (<Type>/split.csv, crops, predictions.csv, deployment_priors.json); the extra types
# are written into it too. RUNS holds the checkpoints (2 x 3.5 GB per leg) and OUT ($RUNS/retype-data) everything
# with label ids. Nothing here writes into the repository; copy $OUT/docs-$LEG into docs/retype-2026-11/ by hand.
# CPU steps are submitted with --gpus=0 by retype_submit.sh. Submit from the repo root (logs/ must exist).
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-retype
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
STEP="${STEP:?set STEP (extra|edits|crops|assemble|train|eval)}"
DATA="${DATA:?set DATA (absolute round-3 data root)}"; RUNS="${RUNS:?set RUNS (scratch)}"
OUT="${OUT:-$RUNS/retype-data}"
RAWLABELS="${RAWLABELS:-$DATA/rawlabels}"
# rawLabels for the type-edit join: the label's CURRENT type must equal the edit's new type, so edits made after the
# round-3 snapshot need a fresh pull here (analysis/fetch_all.py); the round-3 cache by default.
EDIT_RAWLABELS="${EDIT_RAWLABELS:-$RAWLABELS}"
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-12}"; LEG="${LEG:-complement}"
CORRECT_PROPORTION="${CORRECT_PROPORTION:-0.9}"
TYPES="${TYPES:-CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem,Crosswalk,NoSidewalk,Signal,Other,Occlusion}"
# A saved /v3/api/cities response for the public/private split of the outputs; "live" asks the API (needs network).
if [ -z "${PUBLIC_CITIES:-}" ]; then
  if [ -f "$RAWLABELS/cities.json" ]; then PUBLIC_CITIES="$RAWLABELS/cities.json"; else PUBLIC_CITIES=live; fi
fi
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
[ -f retype.py ] || { echo "submit from the repo root" >&2; exit 2; }
mkdir -p "$OUT"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Round 3's own settings, so the extra types and the type-edit crops are rendered the same way.
R3RUN="$DATA/CurbRamp/generation_run.json"
r3() { python -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print(','.join(v) if isinstance(v, list) else ('' if v is None else v))" "$R3RUN" "$1"; }
# Held-out cities: round 3's own (the cities of CurbRamp's test_heldout rows), so the extra types hold out the same.
HOLDOUT="${HOLDOUT:-$(python -c "import sys, pandas as pd; d = pd.read_csv(sys.argv[1], usecols=['city', 'split'], dtype=str); print(','.join(sorted(set(d[d.split == 'test_heldout'].city))))" "$DATA/CurbRamp/split.csv")}"
RUN="$RUNS/retype/$LEG"
echo "$(date) STEP=$STEP LEG=$LEG DATA=$DATA OUT=$OUT"

case "$STEP" in
  extra)
    LT="${LT:?set LT (NoSidewalk|Signal|Other|Occlusion)}"
    PANO_STORE="${PANO_STORE:?set PANO_STORE}"
    [ -n "${EXPERTS:-}" ] || echo "WARNING: EXPERTS is not set: crowd margin rule only (set it as for round 3)." >&2
    AI_FLAG=""; [ "$(r3 include_ai_labels)" = True ] && AI_FLAG="--include-ai-labels"
    python -u data_generation.py --label_type "$LT" --out "$DATA" --pano-store "$PANO_STORE" \
        --rawlabels-cache "$RAWLABELS" --cities "$(r3 cities)" --tilt-beta "$(r3 tilt_beta)" $AI_FLAG \
        --correct-proportion "$CORRECT_PROPORTION" || { echo "data_generation.py failed for $LT" >&2; exit 1; }
    python -u split_data.py --label_type "$LT" --out "$DATA" --rawlabels "$RAWLABELS" --holdout-cities "$HOLDOUT" \
        --seed "$SEED" --hard-examples analysis/hard_examples_2026-10-17.csv \
        || { echo "split_data.py failed for $LT" >&2; exit 1; }
    ;;
  edits)
    if [ ! -f "$OUT/label_edits/fetch.json" ]; then
      python -u retype.py fetch-edits --out "$OUT/label_edits" --cities-cache "$RAWLABELS" \
        || { echo "fetch-edits failed (no network on this node?): run it on a login node, then resubmit" >&2; exit 1; }
    fi
    python -u retype.py type-edits --edits "$OUT/label_edits" --rawlabels "$EDIT_RAWLABELS" \
        --out "$OUT/type_edits.csv" || exit 1
    python -u retype.py audit --rawlabels "$RAWLABELS" --splits "$DATA" --edits "$OUT/label_edits" \
        --public-cities "$PUBLIC_CITIES" --out "$OUT/audit.json" || echo "WARNING: audit failed (not needed later)" >&2
    ;;
  crops)
    PANO_STORE="${PANO_STORE:?set PANO_STORE}"
    python -u retype.py crops --labels "$OUT/type_edits.csv" --pano-store "$PANO_STORE" --out "$OUT/edit_crops" \
        --tilt-beta "$(r3 tilt_beta)" || exit 1
    ;;
  assemble)
    EDIT_ARGS=""
    if [ -s "$OUT/type_edits.csv" ] && [ -f "$OUT/edit_crops/manifest.csv" ]; then
      EDIT_ARGS="--edits $OUT/type_edits.csv --edit-crops $OUT/edit_crops"
    else
      echo "WARNING: no type-edit crops; the split has no test_retype rows" >&2
    fi
    python -u retype.py assemble --splits "$DATA" --types "$TYPES" $EDIT_ARGS --rawlabels "$RAWLABELS" \
        --out "$OUT/retype_split.csv" || exit 1
    ;;
  train)
    mkdir -p "$RUN"
    python -u fine_tune_retype.py --split "$OUT/retype_split.csv" --crop-root "$DATA" --output-dir "$RUN" \
        --negatives "$LEG" --epochs "$EPOCHS" --seed "$SEED" --num-workers 7 || { echo "training failed" >&2; exit 1; }
    ;;
  eval)
    TAG="retype-$LEG"
    PRED="$OUT/$LEG/retype_predictions.csv"
    if [ ! -f "$PRED" ] || [ "$RUN/training_run.json" -nt "$PRED" ]; then
      python -u retype.py predict --run-dir "$RUN" --split "$OUT/retype_split.csv" --crop-root "$DATA" \
          --out "$PRED" || { echo "predict failed" >&2; exit 1; }
    fi
    python -u retype.py export-binary --predictions "$PRED" --split "$OUT/retype_split.csv" --out-root "$DATA" \
        --tag "$TAG" || exit 1
    for T in CurbRamp NoCurbRamp Obstacle SurfaceProblem Crosswalk; do
      python -u evaluate.py --label_type "$T" --data-dir "$DATA" --skip-inference --tag "$TAG" --per-city || exit 1
      t=$(echo "$T" | tr '[:upper:]' '[:lower:]')
      python -u -m calibration.calibrate --predictions "$DATA/$T/predictions_$TAG.csv" \
          --priors "$DATA/deployment_priors.json" --label-type "$T" --out "$DATA/$T/calibration_$TAG/$t.json" || exit 1
    done
    python -u retype.py report --predictions "$PRED" --split "$OUT/retype_split.csv" --root "$DATA" \
        --run-dir "$RUN" --docs-dir "$OUT/docs-$LEG" --public-cities "$PUBLIC_CITIES" --rawlabels "$RAWLABELS" \
        --job-ids "${JOB_IDS:-${SLURM_JOB_ID:-}}" || exit 1
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "$(date) done $STEP ${LT:-} ${LEG}"
