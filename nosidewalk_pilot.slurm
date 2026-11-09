#!/bin/bash
#SBATCH --job-name=nosidewalk-pilot
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# The NoSidewalk pilot (issue 42, phase B; docs/nosidewalk-2026-11.md). One step per job, chosen by STEP;
# nosidewalk_submit.sh submits them all with the right dependencies, so this file is rarely submitted by hand.
#
#   STEP=data   ARM=depth|whole        crops with --verdict-unit face, the face-size prior, the face-grouped split
#   STEP=train  ARM=depth|whole SEED=n train, evaluate (--tag s<n>, --per-city), calibrate (Agree only)
#   STEP=report                        analysis/nosidewalk_pilot.py over every arm and seed (CPU only:
#                                      nosidewalk_submit.sh submits it on ckpt-all with --gpus-per-node=0)
#
# ARM=depth is the default depth-sized crop; ARM=whole is --fixed-crop-half-px 1024, the whole 90 degree view
# (a NoSidewalk label marks an absence along a stretch, and issue 42 asked whether a wider crop helps).
# Every path is relative to the repo root, so submit from there, in the activated environment (as
# data_generation.slurm). EXPERTS="<name>=<user_id>,..." comes from the submitting shell (sbatch exports it); it is
# never written into this file or the repo.
set -uo pipefail
# Compute nodes cannot reach Hugging Face's Xet server: read the DINOv2 backbone from a cache pre-downloaded on a login
# node (docs/running-on-hyak.md, "NoSidewalk pilot"); the first pilot run failed in training without this.
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
STEP="${STEP:?set STEP=data|train|report}"
DATA="${DATA:?set DATA (the pilot root, absolute)}"
LT=NoSidewalk
CACHE="${CACHE:-$DATA/rawlabels-ns}"
HOLDOUT="${HOLDOUT-tucson-az,chicago-il}"
MAX_PER_FACE="${MAX_PER_FACE:-5}"
EPOCHS="${EPOCHS:-25}"
[ -f .gitmodules ] || { echo "Submit this job from the repo root." >&2; exit 1; }
[ -n "${EXPERTS:-}" ] || echo "WARNING: EXPERTS is not set: expert votes will not count (margin and face-margin rules only)." >&2

case "$STEP" in
  data)
    ARM="${ARM:?set ARM=depth|whole}"
    PANO_STORE="${PANO_STORE:?set PANO_STORE to the pano store root}"
    case "$ARM" in
      depth) CROP=() ;;
      whole) CROP=(--fixed-crop-half-px 1024) ;;
      *) echo "ARM must be depth or whole, got $ARM" >&2; exit 2 ;;
    esac
    [ -f "$CACHE/.complete" ] || { echo "no complete rawLabels cache at $CACHE (nosidewalk_submit.sh pulls it)" >&2; exit 2; }
    OUT="$DATA/$ARM"
    python -u data_generation.py --label_type "$LT" --out "$OUT" --rawlabels-cache "$CACHE" \
        --pano-store "$PANO_STORE" --depth-encoder vitb --tilt-beta 0.95 --seed 42 \
        --verdict-unit face --max-labels-per-face "$MAX_PER_FACE" "${CROP[@]}" \
        || { echo "data_generation.py failed" >&2; exit 1; }
    # data_generation.py wrote a margin-rule placeholder for the prior; the face-size method replaces it (issue 42).
    python -u -m calibration.nosidewalk_prior --cache "$CACHE" --priors "$OUT/deployment_priors.json" \
        || { echo "nosidewalk_prior failed" >&2; exit 1; }
    python -u split_data.py --label_type "$LT" --out "$OUT" --rawlabels "$CACHE" --group-col face_id \
        --holdout-cities "$HOLDOUT" --seed 42 || { echo "split_data.py failed" >&2; exit 1; }
    ;;
  train)
    ARM="${ARM:?set ARM=depth|whole}"; SEED="${SEED:?set SEED}"
    RUNS="${RUNS:?set RUNS (checkpoints, on scratch)}"
    OUT="$DATA/$ARM"
    RUN="$RUNS/$ARM/s$SEED"
    python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$OUT" --output-dir "$RUN" --seed "$SEED" \
        --epochs "$EPOCHS" --num-workers 7 || { echo "training failed" >&2; exit 1; }
    BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN") \
        || { echo "no best checkpoint in $RUN" >&2; exit 1; }
    python -u evaluate.py --label_type "$LT" --data-dir "$OUT" --checkpoint "$BEST" --tag "s$SEED" --per-city \
        || { echo "evaluate.py failed" >&2; exit 1; }
    # Agree only, as for the five shipped types (issue 57): the Disagree verdict is held back from the gate.
    python -u -m calibration.calibrate --predictions "$OUT/$LT/predictions_s$SEED.csv" \
        --priors "$OUT/deployment_priors.json" --label-type "$LT" \
        --out "$OUT/$LT/calibration/s$SEED/nosidewalk.json" --hold-back incorrect \
        --hold-back-reason "NoSidewalk pilot (issue 42): Disagree held back as for the five shipped types (issue 57)" \
        || { echo "calibration failed" >&2; exit 1; }
    ;;
  report)
    python -u analysis/nosidewalk_pilot.py --data "$DATA" --out-json "$DATA/report/pilot.json" \
        --out-md "$DATA/report/pilot.md" || { echo "report failed" >&2; exit 1; }
    ;;
  *)
    echo "STEP must be data, train or report, got $STEP" >&2; exit 2 ;;
esac
echo "$(date) done STEP=$STEP ${ARM:-} ${SEED:-}"
