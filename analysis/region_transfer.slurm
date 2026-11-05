#!/bin/bash
# Issues #33 / #52 / #15: one GPU job per (label type, arm, country, fraction, seed) of the cross-region study on klone.
# Train with fine_tune_classifier.py on the arm's split, score it with analysis/region_transfer.py predict, then
# evaluate.py --skip-inference --tag <tag> --per-city with region/eval/ as its --data-dir (so nothing lands next to
# the round-3 outputs). Restartable: ckpt may preempt and requeue; training resumes from run_key.json, predict skips a
# complete predictions file, and a job whose metrics are written exits 0 at once.
#
#   ARM=global  LT=CurbRamp COUNTRY=mexico SEED=42 DATA=/abs/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/region-2026-11 \
#       CITIES_JSON=$DATA/rawlabels/cities.json sbatch analysis/region_transfer.slurm
#   ARM=ft      ... COUNTRY=mexico FRAC=25        (needs the global run of the same country and seed)
#   ARM=cond    ... COUNTRY=mexico                (LEVEL=country; the held-out country scores as <unk>)
#   ARM=condall ... LEVEL=country|region          (the round-3 split itself)
#
# analysis/region_submit.sh submits the whole chain. Arms and their split files:
#   global, cond : $DATA/$LT/region/$COUNTRY/split.csv          ft : $DATA/$LT/region/$COUNTRY/ft-$FRAC/split.csv
#   condall      : $DATA/$LT/split.csv
# Checkpoints: global keeps its weights (ft starts from them; region_transfer_cpu.slurm STEP=cleanup removes them once
# every ft job of that country and seed succeeded); ft, cond and condall remove theirs after scoring (KEEP_CKPT=0).
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-region
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
ARM="${ARM:?set ARM (global|ft|cond|condall)}"; LT="${LT:?set LT}"; DATA="${DATA:?set DATA (absolute)}"
RUNS="${RUNS:?set RUNS}"; CITIES_JSON="${CITIES_JSON:?set CITIES_JSON (a saved /v3/api/cities listing)}"
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-25}"; FT_EPOCHS="${FT_EPOCHS:-10}"; LEVEL="${LEVEL:-country}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
REGION="$DATA/$LT/region"
EVAL="$REGION/eval"
KEEP_CKPT=0
COND_ARGS=()
case "$ARM" in
  global)  COUNTRY="${COUNTRY:?set COUNTRY}"; TAG="global-$COUNTRY-s$SEED"; SPLIT="$REGION/$COUNTRY/split.csv"
           SPLITS="val,test,test_heldout,test_heldout_country,pool"; KEEP_CKPT=1 ;;
  ft)      COUNTRY="${COUNTRY:?set COUNTRY}"; FRAC="${FRAC:?set FRAC (percent)}"; TAG="ft-$COUNTRY-f$FRAC-s$SEED"
           SPLIT="$REGION/$COUNTRY/ft-$FRAC/split.csv"; SPLITS="test,test_heldout,test_heldout_country"; EPOCHS="$FT_EPOCHS" ;;
  cond)    COUNTRY="${COUNTRY:?set COUNTRY}"; TAG="cond-$COUNTRY-s$SEED"; SPLIT="$REGION/$COUNTRY/split.csv"
           SPLITS="val,test,test_heldout,test_heldout_country,pool"; COND_ARGS=(--conditioned --cities-json "$CITIES_JSON") ;;
  condall) TAG="condall-$LEVEL-s$SEED"; SPLIT="$DATA/$LT/split.csv"; SPLITS="val,test,test_heldout"
           COND_ARGS=(--conditioned --cities-json "$CITIES_JSON") ;;
  *) echo "unknown ARM $ARM" >&2; exit 2 ;;
esac
RUN="$RUNS/$LT/$TAG"
PRED="$EVAL/$LT/predictions_$TAG.csv"
[ -f "$SPLIT" ] || { echo "no $SPLIT (run analysis/region_transfer.py variants first)" >&2; exit 2; }
if [ -f "$RUN/training_run.json" ] && [ "$EVAL/$LT/metrics_$TAG.json" -nt "$RUN/training_run.json" ] \
   && { [ "$KEEP_CKPT" = 1 ] || ! ls -d "$RUN"/checkpoint-* >/dev/null 2>&1; }; then
  echo "$(date) $LT $TAG already evaluated ($EVAL/$LT/metrics_$TAG.json); nothing to do"; exit 0
fi
mkdir -p "$RUN" "$EVAL/$LT"
# evaluate.py reads <data-dir>/<LT>/split.csv (every row, one-to-one): region_submit.sh links it to the round-3 file
# once, before any job starts. Not relinked here: parallel jobs would race on the unlink/relink window.
# Side effect: evaluate.py's slices_<tag>.md counts "training rows" from that round-3 split, not from this arm's split,
# so for global/ft/cond the column is wrong. Ignore that file; the report stage recomputes slices from each arm's own
# train rows (region_transfer.py slice_rows / train_frames).
[ -e "$EVAL/$LT/split.csv" ] || { echo "no $EVAL/$LT/split.csv (run region_submit.sh, or STEP=variants)" >&2; exit 2; }
TRAIN_ARGS=(--seed "$SEED" --epochs "$EPOCHS")
case "$ARM" in
  ft)
    GLOBAL="$RUNS/$LT/global-$COUNTRY-s$SEED"
    INIT=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$GLOBAL") \
      || { echo "no global checkpoint under $GLOBAL" >&2; exit 1; }
    TRAIN_ARGS+=(--init-from "$INIT") ;;
  cond|condall)
    TRAIN_ARGS+=(--region-conditioning "$LEVEL" --cities-json "$CITIES_JSON") ;;
esac
if [ ! -f "$RUN/training_run.json" ]; then
  echo "$(date) train $LT $TAG on $SPLIT -> $RUN"
  python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --split "$SPLIT" --output-dir "$RUN" \
      "${TRAIN_ARGS[@]}" --num-workers 7 || { echo "training failed" >&2; exit 1; }
fi
BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN") \
  || { echo "no checkpoint under $RUN" >&2; exit 1; }
python -u analysis/region_transfer.py predict --split-file "$SPLIT" --data-dir "$DATA" --label-type "$LT" \
    --checkpoint "$BEST" --splits "$SPLITS" --out "$PRED" "${COND_ARGS[@]}" || { echo "predict failed" >&2; exit 1; }
# The embedding ablation (the same checkpoint, every row <unk>), condall only: for LOCO cond the held-out country is
# <unk> already, so its rows would score identically, and the report reads -unk files only for condall.
if [ "$ARM" = condall ]; then
  python -u analysis/region_transfer.py predict --split-file "$SPLIT" --data-dir "$DATA" --label-type "$LT" \
      --checkpoint "$BEST" --splits "$SPLITS" --out "$EVAL/$LT/predictions_$TAG-unk.csv" "${COND_ARGS[@]}" --force-unk \
      || { echo "predict --force-unk failed" >&2; exit 1; }
  python -u evaluate.py --label_type "$LT" --data-dir "$EVAL" --skip-inference --tag "$TAG-unk" --per-city \
      || { echo "evaluate (unk) failed" >&2; exit 1; }
fi
python -u evaluate.py --label_type "$LT" --data-dir "$EVAL" --skip-inference --tag "$TAG" --per-city \
    || { echo "evaluate failed" >&2; exit 1; }
cat "$EVAL/$LT/report_$TAG.txt"
if [ "$KEEP_CKPT" = 0 ]; then
  cp "$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
  rm -rf "$RUN"/checkpoint-*; echo "removed $RUN/checkpoint-*"
fi
echo "$(date) done $LT $TAG"
