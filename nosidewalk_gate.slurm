#!/bin/bash
# The NoSidewalk 5-fold out-of-fold gate (issue 42, step (a) of docs/nosidewalk-2026-11.md; analysis/nosidewalk_gate.py).
# One step per job, chosen by STEP; nosidewalk_gate_submit.sh submits the chain with the right dependencies.
#
#   STEP=train  FOLD=k   train fold k's model on the fold's split file (same recipe as the pilot: 25 epochs, seed 42),
#                        then score its test and test_heldout rows (nosidewalk_gate.py predict)
#   STEP=report          nosidewalk_gate.py report over every fold (CPU only: the launcher submits it without a GPU)
#
# DATA is the pilot root (whole/NoSidewalk/split.csv, whole/deployment_priors.json), GATE the gate directory that
# `nosidewalk_gate.py folds` wrote (the launcher runs that step on the login node), RUNS holds the fold checkpoints
# (3.5 GB each, so scratch). Fold checkpoints are deleted once the fold's predictions are written: they exist only for
# their out-of-fold scores. Restartable: ckpt may preempt and requeue; fine_tune_classifier.py resumes from its last
# complete checkpoint, and a requeue after the weights were deleted skips straight to "nothing to do".
# KEEP_BEST=1 keeps each fold's best checkpoint (and deletes the others), for a run whose models go on to score
# other labels (step (b), analysis/nosidewalk_round.py); the default deletes every checkpoint as before.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=ns-gate
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
STEP="${STEP:?set STEP=train|report}"
DATA="${DATA:?set DATA (the pilot root, absolute)}"
GATE="${GATE:?set GATE (the gate directory, absolute)}"
RUNS="${RUNS:?set RUNS (checkpoints, on scratch)}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
LT=NoSidewalk
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-42}"
KEEP_BEST="${KEEP_BEST:-0}"
[ -f .gitmodules ] || { echo "Submit this job from the repo root." >&2; exit 1; }
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
# Compute nodes cannot reach Hugging Face's Xet server: read the DINOv2 backbone from the pre-downloaded cache
# (docs/running-on-hyak.md, "NoSidewalk pilot").
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

case "$STEP" in
  train)
    FOLD="${FOLD:?set FOLD}"
    SPLIT="$GATE/fold$FOLD/split.csv"
    RUN="$RUNS/fold$FOLD"
    PRED="$GATE/fold$FOLD/predictions.csv"
    [ -f "$SPLIT" ] || { echo "no $SPLIT (run nosidewalk_gate.py folds first)" >&2; exit 2; }
    # Done = scored, and the checkpoints already tidied: none left, or with KEEP_BEST=1 exactly the best one.
    if [ -f "$PRED" ] && [ "$PRED" -nt "$SPLIT" ] && [ -f "$RUN/training_run.json" ]; then
      LEFT=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | wc -l)
      if [ "$LEFT" -eq 0 ] || { [ "$KEEP_BEST" = 1 ] && [ "$LEFT" -eq 1 ] && [ -f "$RUN/trainer_state.best.json" ]; }; then
        echo "$(date) fold $FOLD already scored ($PRED); nothing to do"; exit 0
      fi
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "$(date) train fold $FOLD on $SPLIT -> $RUN"
    python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA/whole" --split "$SPLIT" \
        --output-dir "$RUN" --seed "$SEED" --epochs "$EPOCHS" --num-workers 7 \
        || { echo "training failed" >&2; exit 1; }
    python -u analysis/nosidewalk_gate.py predict --gate "$GATE" --fold "$FOLD" --run-dir "$RUN" \
        || { echo "predict failed" >&2; exit 1; }
    # Keep the training record, drop the weights: the fold models exist only for their out-of-fold scores.
    BEST=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'])" "$RUN/training_run.json")
    cp "$RUN/$BEST/trainer_state.json" "$RUN/trainer_state.best.json" 2>/dev/null || true
    if [ "$KEEP_BEST" = 1 ]; then
      for C in "$RUN"/checkpoint-*; do [ "$C" = "$RUN/$BEST" ] || rm -rf "$C"; done
      echo "$(date) wrote $PRED; kept $RUN/$BEST"
    else
      rm -rf "$RUN"/checkpoint-*
      echo "$(date) wrote $PRED; removed $RUN/checkpoint-*"
    fi
    ;;
  report)
    python -u analysis/nosidewalk_gate.py report --gate "$GATE" --priors "$DATA/whole/deployment_priors.json" \
        --out-json "$GATE/report/gate.json" --out-md "$GATE/report/gate.md" \
        || { echo "report failed" >&2; exit 1; }
    ;;
  *)
    echo "STEP must be train or report, got $STEP" >&2; exit 2 ;;
esac
echo "$(date) done STEP=$STEP ${FOLD:-}"
