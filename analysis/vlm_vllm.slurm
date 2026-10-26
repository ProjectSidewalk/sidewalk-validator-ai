#!/bin/bash
# Issue #63, the vLLM leg: serve a VLM with vLLM inside the vllm/vllm-openai Apptainer image, wait for /health,
# score the round-3 held-out crops with analysis/vlm_vllm.py against it, then stop the server.
#
# Defaults are the GLM-5.3-Flash run (FP8, tensor parallel 4 on four H200s, recipe at
# https://recipes.vllm.ai/zai-org/GLM-5.3-Flash: vLLM >= 0.29.0; on Hopper the KV cache must stay BF16, so no
# --kv-cache-dtype fp8; no MTP speculative decoding, since only one token is generated). Thinking cannot be switched
# off in GLM-5.3-Flash's template, so the client closes it with an assistant prefill '<think></think>' (see
# vlm_vllm.py). Everything is overridable from the environment (sbatch exports the caller's environment; do not use
# --export=ALL,K=V for TEMPLATE_KWARGS, its commas split the list), e.g. the small-model smoke on one H200:
#
#   MODEL=qwen3vl-8b-bf16 REPO=Qwen/Qwen3-VL-8B-Instruct TP=1 TYPES=CurbRamp LIMIT=40 CONCURRENCY=8 PREFILL= \
#     TEMPLATE_KWARGS='{"enable_thinking":false}' THINKING_NOTE='enable_thinking=False (no-op for Instruct)' \
#     TIMING_PER_TYPE=0 sbatch --gpus=h200:1 --mem=100G --cpus-per-task=8 --time=1:00:00 analysis/vlm_vllm.slurm
#
#   LIMIT=40 TIMING_PER_TYPE=0 sbatch analysis/vlm_vllm.slurm       # GLM smoke: 40 crops per type
#   sbatch analysis/vlm_vllm.slurm                                  # GLM full run, all 13,144 crops
#
# Resumable: on preemption Slurm requeues the job and the client skips crops already in the CSVs.
# Never writes under /tmp: outputs, caches and the container home live on /gscratch/scrubbed; the server's IPC
# temp dir is in /dev/shm (node-local RAM, removed at exit).
#SBATCH -J vlm-vllm
#SBATCH -A makelab
#SBATCH -p ckpt-all
#SBATCH --gpus=h200:4
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH -o /gscratch/scrubbed/jfroehli/logs/vlm_vllm.%j.log

set -uo pipefail

MODEL=${MODEL:-glm5.3-flash}
REPO=${REPO:-zai-org/GLM-5.3-Flash}
TP=${TP:-4}
TYPES=${TYPES:-CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem,Crosswalk}
LIMIT=${LIMIT:-0}
CONCURRENCY=${CONCURRENCY:-32}
TIMING_PER_TYPE=${TIMING_PER_TYPE:-10}   # 0 skips the concurrency-1 timing run
PREFILL=${PREFILL-<think></think>}       # empty string = no prefill (models with an enable_thinking switch)
TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{"reasoning_effort":"low"}'}
THINKING_NOTE=${THINKING_NOTE:-"template has no off switch; closed by assistant prefill <think></think> (continue_final_message); reasoning_effort=low"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
IMAGE_TAG=${IMAGE_TAG:-vllm/vllm-openai:v0.32.0-x86_64}   # CUDA 13.0; the -cu129 variant dies at import (torchcodec wants libnvrtc.so.13)
SIF=${SIF:-/gscratch/scrubbed/jfroehli/containers/vllm-openai_v0.32.0-x86_64.sif}
SCRUB=/gscratch/scrubbed/jfroehli
CODE=${CODE:-/gscratch/makelab/jfroehli/sva-wt-vlm}
DATA=${DATA:-/gscratch/makelab/jfroehli/sidewalk-validator-ai/retrain-2026-10}
OUT=${OUT:-$SCRUB/vlm-scoreboard-2026-10/scores}
PY=${PY:-$SCRUB/envs/vllm-client/bin/python}
PORT=${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 1000))}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-5400}

export HF_HOME=$SCRUB/hf HF_HUB_OFFLINE=1
# klone sets http_proxy for every login; its no_proxy lists localhost but not 127.0.0.1, so local calls hit the proxy
export no_proxy="127.0.0.1,${no_proxy:-}" NO_PROXY="127.0.0.1,${NO_PROXY:-}"
case "$OUT" in /tmp*) echo "refusing to write under /tmp"; exit 2 ;; esac
mkdir -p "$OUT" "$SCRUB/vllm_home" "$SCRUB/logs"
SHM=/dev/shm/vllm_${SLURM_JOB_ID:-$$}
mkdir -p "$SHM"

hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NGPU=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/ .*//')
GPU_LABEL=${GPU_LABEL:-${NGPU}x${GPU_NAME}}
echo "model=$MODEL repo=$REPO tp=$TP gpus=$GPU_LABEL types=$TYPES limit=$LIMIT concurrency=$CONCURRENCY"
echo "restart count: ${SLURM_RESTART_COUNT:-0}"

module load apptainer 2>/dev/null || true
SERVE_ARGS="--tensor-parallel-size $TP --port $PORT --host 127.0.0.1 --served-model-name $REPO \
  --max-model-len $MAX_MODEL_LEN --limit-mm-per-prompt {\"image\":1,\"video\":0} --max-logprobs 20 \
  --gpu-memory-utilization 0.90 $EXTRA_SERVE_ARGS"
export VLLM_SERVE_ARGS="vllm serve $REPO $SERVE_ARGS"
echo "$VLLM_SERVE_ARGS"

# --home keeps vLLM/torch/flashinfer caches off the over-quota home; HF cache bound at the same path.
# shellcheck disable=SC2086
apptainer exec --nv --cleanenv --home "$SCRUB/vllm_home" --bind /gscratch --bind "$SHM" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}",HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1,TMPDIR="$SHM",VLLM_CACHE_ROOT="$SCRUB/vllm_home/.cache/vllm" \
  "$SIF" vllm serve "$REPO" $SERVE_ARGS > "$SCRUB/logs/vlm_vllm.${SLURM_JOB_ID:-x}.server.log" 2>&1 &
SERVER=$!

cleanup() { kill "$SERVER" 2>/dev/null; sleep 5; kill -9 "$SERVER" 2>/dev/null; rm -rf "$SHM"; }
trap cleanup EXIT

t0=$(date +%s)
until curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited before becoming healthy"; tail -50 "$SCRUB/logs/vlm_vllm.${SLURM_JOB_ID:-x}.server.log"; exit 3
  fi
  if (( $(date +%s) - t0 > HEALTH_TIMEOUT_S )); then echo "server not healthy after ${HEALTH_TIMEOUT_S}s"; exit 4; fi
  sleep 15
done
echo "server healthy after $(( $(date +%s) - t0 ))s"; curl -s --noproxy "*" "http://127.0.0.1:$PORT/version"; echo

COMMON=(--server "http://127.0.0.1:$PORT" --model "$MODEL" --repo "$REPO" --data-root "$DATA" --out-dir "$OUT"
        --gpu "$GPU_LABEL" --tp "$TP" --image-tag "$IMAGE_TAG" --template-kwargs "$TEMPLATE_KWARGS"
        --thinking-note "$THINKING_NOTE" --hf-home "$HF_HOME")
[ -n "$PREFILL" ] && COMMON+=(--prefill "$PREFILL")

cd "$CODE" || exit 5
git log -1 --format='code %H %s'
if [ "$TIMING_PER_TYPE" -gt 0 ] && [ ! -f "$OUT/$MODEL/timing_c1.csv" ]; then
  "$PY" analysis/vlm_vllm.py timing "${COMMON[@]}" --label-types "$TYPES" --per-type "$TIMING_PER_TYPE" || exit 6
fi
"$PY" analysis/vlm_vllm.py score "${COMMON[@]}" --label-types "$TYPES" --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" || exit 7
echo "done after $(( $(date +%s) - t0 ))s"; date
