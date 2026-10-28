#!/bin/bash
# Issues #76 / #75, the CPU steps of the confident-learning chain on klone (see analysis/cl_submit.sh):
#   STEP=prep  LT=<type> DATA=... RAWLABELS=...  : folds (k panorama-grouped folds, 3 m merge, as the split) + rules
#   STEP=flag  LT=<type> DATA=...                : confident joint over oof_fold*.csv -> noisy_targets.csv, cl-drop, cl-flip
# Both are deterministic and idempotent; a rerun rewrites the same files.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-cl-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STEP="${STEP:?set STEP (prep|flag)}"; LT="${LT:?set LT}"; DATA="${DATA:?set DATA (absolute)}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
FOLDS="${FOLDS:-5}"; SEED="${SEED:-42}"; RADIUS="${RADIUS:-3}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
CL="$DATA/$LT/cl"
case "$STEP" in
  prep)
    RAWLABELS="${RAWLABELS:?set RAWLABELS (the cached rawLabels exports, for coordinates)}"
    python -u analysis/confident_learning.py folds --split "$DATA/$LT/split.csv" --rawlabels "$RAWLABELS" \
        --folds "$FOLDS" --seed "$SEED" --merge-radius-m "$RADIUS" --out "$CL"
    python -u analysis/confident_learning.py rules --split "$DATA/$LT/split.csv" --out "$CL"
    ;;
  flag)
    python -u analysis/confident_learning.py flag --split "$DATA/$LT/split.csv" --oof "$CL"/oof_fold*.csv --out "$CL"
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "$(date) done $STEP $LT"
