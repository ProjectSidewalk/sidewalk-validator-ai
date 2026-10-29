#!/bin/bash
# Issues #76 / #75: one GPU job per (label type, split variant) on klone: train on the variant split, then either
# score the fold's held-out train rows (VARIANT=fold<k>, out-of-fold predictions for the confident joint) or score
# the variant model on the production split's val/test/test_heldout rows (VARIANT=symmetric|unanimous3|cl-drop|cl-flip,
# evaluate.py --tag <variant>, written next to the baseline's outputs). Restartable: ckpt may preempt and requeue, and
# fine_tune_classifier.py resumes from its last complete checkpoint (run_key.json guards the split hash); the
# prediction / evaluation step just reruns.
#
#   LT=CurbRamp VARIANT=fold0 DATA=/abs/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/cl-2026-10 \
#       sbatch analysis/confident_learning.slurm
#   LT=CurbRamp VARIANT=symmetric ... sbatch --partition=gpu-l40s --constraint=l40s analysis/confident_learning.slurm
#
# analysis/cl_submit.sh submits the whole chain (prep -> folds -> flag -> cl-* variants, rules in parallel).
# DATA is the data_generation.py --out directory holding <LT>/split.csv and <LT>/cl/<VARIANT>/split.csv
# (confident_learning.py folds / rules / flag); RUNS holds the checkpoints (3.5 GB each, two per run), so it belongs
# on scratch. Fold checkpoints are deleted once their out-of-fold scores are written; variant checkpoints are kept.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-cl
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
LT="${LT:?set LT}"; VARIANT="${VARIANT:?set VARIANT}"; DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-25}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SPLIT="$DATA/$LT/cl/$VARIANT/split.csv"
RUN="$RUNS/$LT/$VARIANT"
[ -f "$SPLIT" ] || { echo "no $SPLIT (run confident_learning.py folds / rules / flag first)" >&2; exit 2; }
mkdir -p "$RUN"
case "$VARIANT" in
  fold*)  # a requeue after the fold's weights were deleted must not retrain: the out-of-fold scores are the product
    if [ -f "$DATA/$LT/cl/oof_${VARIANT}.csv" ] && [ "$DATA/$LT/cl/oof_${VARIANT}.csv" -nt "$DATA/$LT/cl/folds.csv" ] \
       && [ -f "$RUN/training_run.json" ] && ! ls -d "$RUN"/checkpoint-* >/dev/null 2>&1; then
      echo "$(date) $LT $VARIANT already scored ($DATA/$LT/cl/oof_${VARIANT}.csv); nothing to do"; exit 0
    fi ;;
esac
echo "$(date) train $LT $VARIANT on $SPLIT -> $RUN"
python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --split "$SPLIT" --output-dir "$RUN" \
    --seed "$SEED" --epochs "$EPOCHS" --num-workers 7 || { echo "training failed" >&2; exit 1; }
case "$VARIANT" in
  fold*)
    K="${VARIANT#fold}"
    OUT="$DATA/$LT/cl/oof_fold$K.csv"
    python -u analysis/confident_learning.py predict --split "$DATA/$LT/split.csv" \
        --folds-file "$DATA/$LT/cl/folds.csv" --fold "$K" --run-dir "$RUN" --out "$OUT" \
        || { echo "predict failed" >&2; exit 1; }
    # Keep the training record, drop the weights: the fold models exist only for their out-of-fold scores.
    BEST=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'])" "$RUN/training_run.json")
    cp "$RUN/$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
    rm -rf "$RUN"/checkpoint-*
    echo "$(date) wrote $OUT; removed $RUN/checkpoint-*"
    ;;
  *)
    BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN")
    python -u evaluate.py --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$VARIANT" --per-city \
        || { echo "evaluate failed" >&2; exit 1; }
    cat "$DATA/$LT/report_$VARIANT.txt"
    ;;
esac
echo "$(date) done $LT $VARIANT"
