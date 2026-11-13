#!/bin/bash
# NoSidewalk step (b), the masked expert round (issue 42; analysis/nosidewalk_round.py): cut the whole-view crop of
# every phase-1 label from the staged store copy, then score the crops with each fold model of a gate run that kept
# its best checkpoints (KEEP_BEST=1 ./nosidewalk_gate_submit.sh). One job, resumable: a requeue keeps the ok crops
# (manifest.csv) and skips a fold whose predictions file is newer than the manifest.
#
#   RUN=/gscratch/scrubbed/$USER/nosidewalk-2026-11/round-b RUNS=/gscratch/scrubbed/$USER/nosidewalk-runs/gate2 \
#     STORE=/gscratch/scrubbed/$USER/pano-stage/NoSidewalk sbatch nosidewalk_round.slurm
#
# RUN holds labels.csv (nosidewalk_round.py pool); the panoramas of its stage_files.txt are staged first with
# analysis/ai_labels_stage.slurm (RUN=... DEST=$STORE). Writes RUN/manifest.csv, RUN/crops/ and
# RUN/predictions_fold<k>.csv, the files `nosidewalk_round.py build` reads.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=ns-round
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
RUN="${RUN:?set RUN (the directory holding labels.csv, absolute)}"
RUNS="${RUNS:?set RUNS (the fold<k> checkpoint directories of the gate run)}"
STORE="${STORE:?set STORE (the staged pano store copy)}"
FOLDS="${FOLDS:-5}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
[ -f .gitmodules ] || { echo "Submit this job from the repo root." >&2; exit 1; }
[ -f "$STORE/.pano-store" ] || { echo "no marker in $STORE: staging not verified" >&2; exit 1; }
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u analysis/ai_labels_eval.py crops --labels "$RUN/labels.csv" --pano-store "$STORE" --out "$RUN" \
    --fixed-crop-half-px 1024 || { echo "crops failed" >&2; exit 1; }
best_of() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'])" "$1"
}
for ((K = 0; K < FOLDS; K++)); do
  PRED="$RUN/predictions_fold$K.csv"
  if [ -f "$PRED" ] && [ "$PRED" -nt "$RUN/manifest.csv" ]; then echo "fold $K already scored"; continue; fi
  BEST=$(best_of "$RUNS/fold$K/training_run.json") || { echo "no training_run.json for fold $K" >&2; exit 1; }
  [ -d "$RUNS/fold$K/$BEST" ] || { echo "fold $K: $RUNS/fold$K/$BEST is gone (was the gate run made with KEEP_BEST=1?)" >&2; exit 1; }
  python -u analysis/ai_labels_eval.py score --manifest "$RUN/manifest.csv" --checkpoint "$RUNS/fold$K/$BEST" \
      --out "$PRED" || { echo "score fold $K failed" >&2; exit 1; }
done
echo "$(date) done: $(ls "$RUN"/predictions_fold*.csv | wc -l) prediction files"
