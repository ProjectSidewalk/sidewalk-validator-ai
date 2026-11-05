#!/bin/bash
# Issue #14: one GPU job per step of the context-arms experiment on klone (docs/context-arms-2026-11.md).
#
#   STEP=backfill: write <DATA>/<LT>/context/ (the wide crop and the 448 px view of every ok row) from the pano store,
#                  re-rendered with the pose the crops were cut under; --verify 50 re-cuts 50 tight crops and fails the
#                  job if they drift from the stored WebPs. Crops, manifest and split.csv are not touched. Resumable.
#   STEP=train:    train one arm (ARM=two_scale|view_heatmap|side_info) on the round-3 split with the round-3 recipe,
#                  score it with context_model.py predict, then evaluate.py --skip-inference --tag $TAG --per-city
#                  (outputs next to the baseline's in <DATA>/<LT>/). Restartable: ckpt may preempt and requeue, and
#                  fine_tune_classifier.py resumes from its last complete checkpoint (run_key.json guards the split,
#                  the arm, the view resolution and the batch size).
#
#   STEP=backfill LT=NoCurbRamp DATA=/abs/retrain-2026-10 PANO_STORE=/abs/Panoramas sbatch analysis/context_arms.slurm
#   STEP=train LT=NoCurbRamp ARM=two_scale DATA=/abs/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/context-2026-11 \
#       sbatch analysis/context_arms.slurm
#
# Variables (${VAR:?} = required):
#   STEP        backfill | train
#   LT          label type
#   DATA        data_generation.py --out holding <LT>/manifest.csv, <LT>/split.csv and the baseline's metrics.json
#   PANO_STORE  (backfill) the pano store the crops were cut from, with its .pano-store marker
#   RAWLABELS   (backfill) the rawLabels snapshot the dataset was built from (default $DATA/../rawlabels)
#   ARM         (train) two_scale | view_heatmap | side_info
#   RUNS        (train) checkpoint root on scratch (3.5 GB per checkpoint, two kept per run)
#   SEED        (train) default 42; a replicate sets SEED and TAG=<arm>-s<seed>
#   TAG         (train) evaluate.py tag, default $ARM (view_heatmap at 448: TAG=view_heatmap-r448 RES=448)
#   RES         (train) --context-view-res, default 224 (only view_heatmap reads it)
#   BATCH       (train) per-device batch, default 8 (gradient accumulation keeps the effective batch at 32)
#   KEEP_CKPT   (train) 1 (default) keeps the checkpoints; 0 deletes them after evaluation (seed replicates)
#   ENV, EPOCHS, HF_HOME as in analysis/confident_learning.slurm
#
# analysis/context_submit.sh submits backfill -> three arm trainings per label type.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-ctx
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
STEP="${STEP:?set STEP (backfill|train)}"; LT="${LT:?set LT}"; DATA="${DATA:?set DATA (absolute)}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$STEP" in
  backfill)
    PANO_STORE="${PANO_STORE:?set PANO_STORE}"
    RAWLABELS="${RAWLABELS:-$(cd "$DATA/.." && pwd)/rawlabels}"
    echo "$(date) backfill $LT in $DATA/$LT/context from $PANO_STORE"
    python -u context_crops.py backfill --data-dir "$DATA" --label_type "$LT" --pano-store "$PANO_STORE" \
        --rawlabels-cache "$RAWLABELS" --verify 50 || { echo "backfill failed" >&2; exit 1; }
    ;;
  train)
    ARM="${ARM:?set ARM (two_scale|view_heatmap|side_info)}"; RUNS="${RUNS:?set RUNS}"
    SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-25}"; TAG="${TAG:-$ARM}"; RES="${RES:-224}"; BATCH="${BATCH:-8}"
    KEEP_CKPT="${KEEP_CKPT:-1}"
    RUN="$RUNS/$LT/$TAG"
    mkdir -p "$RUN"
    # A requeue after the weights were deleted must not retrain: the evaluation report is the product.
    if [ "$KEEP_CKPT" = 0 ] && [ -f "$RUN/training_run.json" ] \
       && [ "$DATA/$LT/report_$TAG.txt" -nt "$RUN/training_run.json" ] && ! ls -d "$RUN"/checkpoint-* >/dev/null 2>&1; then
      echo "$(date) $LT $TAG already evaluated ($DATA/$LT/report_$TAG.txt); nothing to do"; exit 0
    fi
    echo "$(date) train $LT $ARM (tag $TAG, seed $SEED, res $RES, batch $BATCH) -> $RUN"
    python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --output-dir "$RUN" --seed "$SEED" \
        --epochs "$EPOCHS" --num-workers 7 --context-arm "$ARM" --context-view-res "$RES" --batch-size "$BATCH" \
        || { echo "training failed" >&2; exit 1; }
    BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN")
    python -u context_model.py predict --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$TAG" \
        || { echo "predict failed" >&2; exit 1; }
    python -u evaluate.py --label_type "$LT" --data-dir "$DATA" --skip-inference --tag "$TAG" --per-city \
        || { echo "evaluate failed" >&2; exit 1; }
    cat "$DATA/$LT/report_$TAG.txt"
    if [ "$KEEP_CKPT" = 0 ]; then
      cp "$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
      rm -rf "$RUN"/checkpoint-*; echo "removed $RUN/checkpoint-*"
    fi
    ;;
  *) echo "unknown STEP=$STEP (backfill|train)" >&2; exit 2 ;;
esac
echo "$(date) done $STEP $LT ${TAG:-}"
