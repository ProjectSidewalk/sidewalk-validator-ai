#!/bin/bash
# Issues #33 / #52 / #15, the CPU steps of the cross-region chain on klone (see analysis/region_submit.sh):
#   STEP=variants  LT=<type> DATA=... CITIES_JSON=... [RAWLABELS=$DATA/rawlabels] : LOCO and fine-tune split files
#                  (the launcher runs this itself on the login node; here for a manual rerun)
#   STEP=calibrate LT=<type> DATA=... CITIES_JSON=... [SEEDS=42] : arm (d), pooled vs local maps -> region/calibration.json
#   STEP=cleanup   LT=<type> RUNS=... COUNTRY=... SEED=...       : remove the global run's weights once its ft jobs are done
# All three are deterministic and idempotent; a rerun rewrites the same files.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-region-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STEP="${STEP:?set STEP (variants|calibrate|cleanup)}"; LT="${LT:?set LT}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
case "$STEP" in
  variants)
    DATA="${DATA:?set DATA}"; CITIES_JSON="${CITIES_JSON:?set CITIES_JSON}"
    RAWLABELS="${RAWLABELS:-$DATA/rawlabels}"   # split.csv has no coordinates; the 3 m merge reads them here
    python -u analysis/region_transfer.py variants --split "$DATA/$LT/split.csv" --cities-json "$CITIES_JSON" \
        --rawlabels "$RAWLABELS" --seed "${SPLIT_SEED:-42}" --out "$DATA/$LT/region"
    mkdir -p "$DATA/$LT/region/eval/$LT"
    [ -e "$DATA/$LT/region/eval/$LT/split.csv" ] || ln -s "$DATA/$LT/split.csv" "$DATA/$LT/region/eval/$LT/split.csv"
    ;;
  calibrate)
    DATA="${DATA:?set DATA}"; CITIES_JSON="${CITIES_JSON:?set CITIES_JSON}"
    python -u analysis/region_transfer.py calibrate --data-dir "$DATA" --label-type "$LT" \
        --priors "$DATA/deployment_priors.json" --cities-json "$CITIES_JSON" --seeds "${SEEDS:-42}"
    ;;
  cleanup)
    RUNS="${RUNS:?set RUNS}"; COUNTRY="${COUNTRY:?set COUNTRY}"; SEED="${SEED:?set SEED}"
    RUN="$RUNS/$LT/global-$COUNTRY-s$SEED"
    BEST=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'])" "$RUN/training_run.json")
    cp "$RUN/$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
    rm -rf "$RUN"/checkpoint-*
    echo "removed $RUN/checkpoint-*"
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "$(date) done $STEP $LT"
