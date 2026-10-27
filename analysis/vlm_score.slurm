#!/bin/bash
# Issue #63: score the round-3 held-out crops with one open-weight VLM (vlm_baseline.py score-torch) on klone.
#
# Defaults are the free checkpoint partition with --requeue: a preempted job is requeued and resumes, because
# score-torch appends to its per-type CSV in chunks and skips rows already there. Override GPU type/count on the
# command line, from a checkout of this branch (logs/ must exist in the submit directory):
#
#   MODEL=qwen3vl-8b-bf16 LIMIT=40 TYPES=CurbRamp sbatch analysis/vlm_score.slurm               # smoke, L40S/A40
#   MODEL=gemma4-31b BATCH=8 sbatch --constraint=h200 analysis/vlm_score.slurm                    # one H200
#   MODEL=nemotron3.5-super-vl-120b sbatch --constraint=h200 --gpus-per-node=2 --mem=400G analysis/vlm_score.slurm
#   MODEL=molmo2-8b ENV=/gscratch/scrubbed/$USER/envs/molmo2 sbatch analysis/vlm_score.slurm
#   MODEL=gemma4-31b FEWSHOT=8 BATCH=2 sbatch --constraint=h200 analysis/vlm_score.slurm         # 8-shot arm
#   MODEL=gemma4-12b-bf16 MODE=timing sbatch analysis/vlm_score.slurm                              # batch-1 timing
#   MODE=production ENV=/gscratch/makelab/$USER/envs/sidewalk-validator-ai sbatch analysis/vlm_score.slurm
#
# --nodes=1 matters for multi-GPU legs: without it Slurm may give one GPU on each of two nodes, and accelerate then
# silently offloads to CPU (see RampNet's run_qwen.slurm). Weights are pre-downloaded on the login node
# (HF_HUB_DISABLE_XET=1); the job runs offline. Outputs carry private-city label ids, so OUT is outside the repo.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=vlm-score
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=l40s|a40
#SBATCH --output=logs/vlm_score_%j.out
#SBATCH --open-mode=append

set -euo pipefail

MODEL="${MODEL:-}"
MODE="${MODE:-score}"                     # score | timing | production
TYPES="${TYPES:-CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem,Crosswalk}"
VARIANT="${VARIANT:-definition}"
BATCH="${BATCH:-8}"
FEWSHOT="${FEWSHOT:-0}"
LIMIT="${LIMIT:-0}"
MAX_BATCH_DIFF="${MAX_BATCH_DIFF:-0.25}"     # abort threshold of the batch-of-N vs batch-of-1 check (p_incorrect)
ENV="${ENV:-/gscratch/scrubbed/$USER/envs/vlm-scoreboard}"
SPLIT_ROOT="${SPLIT_ROOT:-/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10}"
OUT="${OUT:-/gscratch/scrubbed/$USER/vlm-scoreboard-2026-10}"
[ "$LIMIT" != "0" ] && OUT="${OUT}/smoke"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Home is over quota on klone: keep compile caches on scrubbed.
export TRITON_CACHE_DIR="/gscratch/scrubbed/$USER/cache/triton" TORCHINDUCTOR_CACHE_DIR="/gscratch/scrubbed/$USER/cache/inductor"
# The envs use the system python3.12, which ships no Python.h, and Triton compiles a small C helper at first use
# (Molmo2's rotary embedding goes through torch.compile). Borrow the headers of a uv-managed CPython 3.12.
PY_INC="${PY_INC:-$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/include/python3.12 2>/dev/null | head -1)}"
[ -n "$PY_INC" ] && export CPATH="$PY_INC${CPATH:+:$CPATH}"
PY="$ENV/bin/python"

echo "--- vlm_score: mode=$MODE model=$MODEL types=$TYPES variant=$VARIANT batch=$BATCH fewshot=$FEWSHOT limit=$LIMIT"
echo "job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on ${SLURMD_NODENAME:-?}; env $ENV; out $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
"$PY" -c 'import sys, torch, transformers; print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)'

case "$MODE" in
  score)
    "$PY" -u analysis/vlm_baseline.py score-torch --model "$MODEL" --split-root "$SPLIT_ROOT" --out-dir "$OUT" \
        --label-types "$TYPES" --variant "$VARIANT" --batch "$BATCH" --fewshot "$FEWSHOT" --limit "$LIMIT" \
        --max-batch-diff "$MAX_BATCH_DIFF" ;;
  timing)
    "$PY" -u analysis/vlm_baseline.py timing-torch --model "$MODEL" --split-root "$SPLIT_ROOT" --out-dir "$OUT" ;;
  production)
    "$PY" -u analysis/vlm_baseline.py baseline --split-root "$SPLIT_ROOT" --out-dir "$OUT" --device cuda ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "--- done"
