#!/bin/bash
# Issue #3, the panorama leak in isolation (docs/split-leak-2026-11.md): one GPU array task per (label type, arm,
# seed), read from line SLURM_ARRAY_TASK_ID + 1 of $TASKS ("<Type> <L|G> <seed>"; analysis/split_leak_submit.sh
# writes it). Train round 3's recipe (DINOv2-L, 25 epochs, best by val macro-F1, hard targets) on the arm's own split,
# score that split's val/test/test_heldout rows, then delete the weights:
#
#   arm L: $RUNS/<Type>/splits/label_level_s<seed>.csv (the coupled label-level split, scatter seed = training seed)
#   arm G: $DATA/<Type>/split.csv (round 3's production split; only for a fresh grouped arm)
#
# Everything is written under RUN=$RUNS/<Type>/<arm>-s<seed> (training_run.json, trainer_state.best.json,
# predictions.csv); nothing under $DATA, which is only read. Restartable: ckpt-all may preempt and requeue, and
# fine_tune_classifier.py resumes from its last complete checkpoint (run_key.json guards the split hash); once
# predictions.csv is written and the weights are gone, a requeue does nothing.
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/split-leak \
#       TASKS=$RUNS/split_leak_tasks_stage1.txt sbatch --array=0-4%4 analysis/split_leak.slurm
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-leak
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"; TASKS="${TASKS:?set TASKS}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
EPOCHS="${EPOCHS:-25}"
read -r LT ARM SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[ -n "${SEED:-}" ] || { echo "no task $SLURM_ARRAY_TASK_ID in $TASKS" >&2; exit 2; }
case "$ARM" in
  L) SPLIT="$RUNS/$LT/splits/label_level_s$SEED.csv" ;;
  G) SPLIT="$DATA/$LT/split.csv" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
RUN="$RUNS/$LT/$ARM-s$SEED"
[ -f "$SPLIT" ] || { echo "no $SPLIT (STAGE=0 first)" >&2; exit 2; }
if [ -f "$RUN/predictions.csv" ] && [ -f "$RUN/training_run.json" ] && [ "$RUN/predictions.csv" -nt "$RUN/training_run.json" ] \
   && ! ls -d "$RUN"/checkpoint-* >/dev/null 2>&1; then
  echo "$(date) $LT $ARM s$SEED already scored ($RUN/predictions.csv); nothing to do"; exit 0
fi
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$RUN"
if [ -f "$RUN/training_run.json" ] && ls -d "$RUN"/checkpoint-* >/dev/null 2>&1; then
  echo "$(date) $LT $ARM s$SEED trained already (requeued before scoring); scoring"
else
  echo "$(date) train $LT $ARM s$SEED on $SPLIT -> $RUN"
  python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --split "$SPLIT" --output-dir "$RUN" \
      --seed "$SEED" --epochs "$EPOCHS" --num-workers 7 --target hard || { echo "training failed" >&2; exit 1; }
fi
python -u analysis/split_leak.py predict --data "$DATA" --label_type "$LT" --split "$SPLIT" --run-dir "$RUN" \
    --out "$RUN/predictions.csv" || { echo "predict failed" >&2; exit 1; }
# Keep the training record and the best epoch's trainer state, drop the weights (3.5 GB per checkpoint): the
# predictions are the product, and a leaky model must not outlive its scoring.
BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN")
cp "$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
rm -rf "$RUN"/checkpoint-*
touch "$RUN/predictions.csv"  # newer than training_run.json, so a requeue sees the task as done
echo "$(date) done $LT $ARM s$SEED; removed $RUN/checkpoint-*"
