#!/bin/bash
#SBATCH --job-name=sva-ailabels-gen
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
# AI-labels evaluation (analysis/ai_labels_eval.py): cut the crops from the staged store copy exactly as
# data_generation.py does, then score them with the round-3 CurbRamp checkpoint. Resumable: a preemption reruns
# `crops` from manifest.csv (ok rows kept) and `score` from scratch. Submit from the repo root of a checkout with the
# Depth-Anything-V2 submodule and depth_anything_v2_metric_vkitti_vitb.pth in place.
#   RUN=/gscratch/makelab/jfroehli/ai-labels-2026-10 CKPT=/gscratch/makelab/jfroehli/sidewalk-validator-ai/retrain-2026-10/CurbRamp/dinov2/checkpoint-4340 \
#     sbatch analysis/ai_labels_gen.slurm
set -uo pipefail
RUN="${RUN:?set RUN to the run directory holding labels.csv}"
CKPT="${CKPT:?set CKPT to the round-3 CurbRamp checkpoint directory}"
STORE="${STORE:-/gscratch/makelab/jfroehli/pano-stage/ai-labels}"
module load conda
eval "$(conda shell.bash hook)"
conda activate /gscratch/makelab/jfroehli/envs/sidewalk-validator-ai
export HF_HOME=/gscratch/makelab/jfroehli/hf_cache HF_HUB_OFFLINE=1
[ -f "$STORE/.pano-store" ] || { echo "no marker in $STORE: staging not verified" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u analysis/ai_labels_eval.py crops --labels "$RUN/labels.csv" --pano-store "$STORE" --out "$RUN" || exit 1
python -u analysis/ai_labels_eval.py score --manifest "$RUN/manifest.csv" --checkpoint "$CKPT" --out "$RUN/predictions.csv" || exit 1
echo "done: $(wc -l < "$RUN/predictions.csv") prediction rows"
