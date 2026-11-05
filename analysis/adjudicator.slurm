#!/bin/bash
# Issues #77 / #74: the VLM adjudicator on klone. One job per model and stage:
#   STAGE=crops    cut the gold items' crops that round 3 does not have (analysis/ai_labels_eval.py crops, unchanged;
#                  one GPU for the depth model; no vLLM)
#   STAGE=score    serve MODEL with vLLM (vllm/vllm-openai Apptainer image, as analysis/vlm_vllm.slurm), wait for
#                  /health, score the gold set with analysis/adjudicator.py score (verdict + one-sentence reason)
#   STAGE=relabel  serve MODEL, score the contested train rows (ROWS=train) or the pool (ROWS=pool) of TYPES with
#                  analysis/adjudicator.py relabel (refuses a pair that does not clear the bar in $OUT/agree/bar.json)
#
# analysis/adjudicator_submit.sh sets the resources per preset; by hand, e.g. the Gemma smoke:
#   MODEL=gemma4-31b STAGE=score LIMIT=8 sbatch --gpus=l40s:2 --mem=200G --time=1:00:00 analysis/adjudicator.slurm
#
# Presets (MODEL): glm5.3-flash, gemma4-31b, qwen3vl-32b; repo, revision, tensor parallel size and how thinking is
# switched off are set below and recorded in every meta file. Zero-shot only: the scoreboard's 8-shot examples are
# train rows whose targets are what is under test.
#
# Inputs on klone (copied by Jon; private, never in the repo): $OUT/gold.csv (adjudicator.py gold), the crops leg's
# $OUT/crops (labels.csv + the staged panos for STAGE=crops), and for relabel $POOL (relabel_pool.csv) and
# $CITYPARAMS (SidewalkWebpage's conf/cityparams.conf).
#
# Resumable: on preemption Slurm requeues the job and the client skips items already in the CSVs. Never writes under
# /tmp: outputs, caches and the container home live on /gscratch/scrubbed; the server's IPC dir is in /dev/shm.
#SBATCH -J sva-adjudicator
#SBATCH -A makelab
#SBATCH -p ckpt-all
#SBATCH --gpus=l40s:2
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH -o /gscratch/scrubbed/%u/logs/adjudicator.%j.log
#SBATCH --open-mode=append

set -uo pipefail

STAGE=${STAGE:-score}
MODEL=${MODEL:-gemma4-31b}
case "$MODEL" in
  glm5.3-flash)
    REPO=${REPO:-zai-org/GLM-5.3-Flash}; REVISION=${REVISION:-46a050bec13002ad29ee795c16e61cf4fa3c4a9c}; TP=${TP:-4}
    PREFILL=${PREFILL-<think></think>}; TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{"reasoning_effort":"low"}'}
    THINKING_NOTE=${THINKING_NOTE:-"template has no off switch; closed by assistant prefill <think></think> (continue_final_message), in the verdict and the reason requests; reasoning_effort=low"}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}; CONCURRENCY=${CONCURRENCY:-32} ;;
  gemma4-31b)
    REPO=${REPO:-google/gemma-4-31B-it}; REVISION=${REVISION:-842da3794eaa0b77d5f08bae87a17459d91ff475}; TP=${TP:-2}
    PREFILL=${PREFILL-}; TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
    THINKING_NOTE=${THINKING_NOTE:-"enable_thinking=False"}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}; CONCURRENCY=${CONCURRENCY:-16} ;;
  qwen3vl-32b)
    # revision not recorded by the scoreboard run: empty = the cache's refs/main, resolved and recorded at run time
    REPO=${REPO:-Qwen/Qwen3-VL-32B-Instruct}; REVISION=${REVISION:-}; TP=${TP:-2}
    PREFILL=${PREFILL-}; TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
    THINKING_NOTE=${THINKING_NOTE:-"Instruct model (no thinking); enable_thinking=False is a no-op"}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}; CONCURRENCY=${CONCURRENCY:-16} ;;
  *)
    : "${REPO:?unknown MODEL preset $MODEL: set REPO, REVISION, TP, PREFILL, TEMPLATE_KWARGS}"
    REVISION=${REVISION:-}; TP=${TP:-1}; PREFILL=${PREFILL-}; TEMPLATE_KWARGS=${TEMPLATE_KWARGS:-'{}'}
    THINKING_NOTE=${THINKING_NOTE:-}; MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}; CONCURRENCY=${CONCURRENCY:-16} ;;
esac
LIMIT=${LIMIT:-0}
ROWS=${ROWS:-train}
TYPES=${TYPES:-CurbRamp,Crosswalk}
REASON=${REASON:-1}                      # 0: verdict only (--no-reason)
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
IMAGE_TAG=${IMAGE_TAG:-vllm/vllm-openai:v0.32.0-x86_64}
SIF=${SIF:-/gscratch/scrubbed/$USER/containers/vllm-openai_v0.32.0-x86_64.sif}
SCRUB=/gscratch/scrubbed/$USER
CODE=${CODE:-/gscratch/makelab/$USER/sidewalk-validator-ai}
DATA=${DATA:-/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10}
OUT=${OUT:-$SCRUB/adjudicator-2026-11}
POOL=${POOL:-$OUT/relabel_pool.csv}
CITYPARAMS=${CITYPARAMS:-$OUT/cityparams.conf}
PANO_STORE=${PANO_STORE:-/gscratch/makelab/$USER/pano-stage/adjudicator}
PY=${PY:-$SCRUB/envs/vllm-client/bin/python}
ENV=${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}
PORT=${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 1000))}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-5400}

case "$OUT" in /tmp*) echo "refusing to write under /tmp"; exit 2 ;; esac
mkdir -p "$OUT" "$SCRUB/vllm_home" "$SCRUB/logs"
hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "stage=$STAGE model=$MODEL repo=$REPO revision=${REVISION:-refs/main} tp=$TP restart=${SLURM_RESTART_COUNT:-0}"
cd "$CODE" || exit 5
git log -1 --format='code %H %s'

if [ "$STAGE" = crops ]; then
  module load conda; eval "$(conda shell.bash hook)"; conda activate "$ENV"
  export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
  [ -f "$OUT/crops/labels.csv" ] || { echo "no $OUT/crops/labels.csv (adjudicator.py crops)"; exit 2; }
  python -u analysis/ai_labels_eval.py crops --labels "$OUT/crops/labels.csv" --pano-store "$PANO_STORE" \
      --out "$OUT/crops" || exit 6
  echo "done crops"; date; exit 0
fi

export HF_HOME=$SCRUB/hf HF_HUB_OFFLINE=1
export no_proxy="127.0.0.1,${no_proxy:-}" NO_PROXY="127.0.0.1,${NO_PROXY:-}"
SHM=/dev/shm/vllm_${SLURM_JOB_ID:-$$}
mkdir -p "$SHM"
NGPU=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/ .*//')
GPU_LABEL=${GPU_LABEL:-${NGPU}x${GPU_NAME}}
module load apptainer 2>/dev/null || true
REV_ARG=""; [ -n "$REVISION" ] && REV_ARG="--revision $REVISION"
SERVE_ARGS="--tensor-parallel-size $TP --port $PORT --host 127.0.0.1 --served-model-name $REPO $REV_ARG \
  --max-model-len $MAX_MODEL_LEN --limit-mm-per-prompt {\"image\":1,\"video\":0} --max-logprobs 20 \
  --gpu-memory-utilization 0.90 $EXTRA_SERVE_ARGS"
export VLLM_SERVE_ARGS="vllm serve $REPO $SERVE_ARGS"
echo "$VLLM_SERVE_ARGS"
# shellcheck disable=SC2086
apptainer exec --nv --cleanenv --home "$SCRUB/vllm_home" --bind /gscratch --bind "$SHM" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env TMPDIR="$SHM" --env VLLM_CACHE_ROOT="$SCRUB/vllm_home/.cache/vllm" \
  "$SIF" vllm serve "$REPO" $SERVE_ARGS > "$SCRUB/logs/adjudicator.${SLURM_JOB_ID:-x}.server.log" 2>&1 &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null; sleep 5; kill -9 "$SERVER" 2>/dev/null; rm -rf "$SHM"; }
trap cleanup EXIT

t0=$(date +%s)
until curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited before becoming healthy"; tail -50 "$SCRUB/logs/adjudicator.${SLURM_JOB_ID:-x}.server.log"; exit 3
  fi
  if (( $(date +%s) - t0 > HEALTH_TIMEOUT_S )); then echo "server not healthy after ${HEALTH_TIMEOUT_S}s"; exit 4; fi
  sleep 15
done
echo "server healthy after $(( $(date +%s) - t0 ))s"; curl -s --noproxy "*" "http://127.0.0.1:$PORT/version"; echo

COMMON=(--server "http://127.0.0.1:$PORT" --model "$MODEL" --repo "$REPO" --gpu "$GPU_LABEL" --tp "$TP"
        --image-tag "$IMAGE_TAG" --template-kwargs "$TEMPLATE_KWARGS" --thinking-note "$THINKING_NOTE"
        --hf-home "$HF_HOME" --concurrency "$CONCURRENCY" --crops-root "$OUT/crops" --split-root "$DATA")
[ -n "$PREFILL" ] && COMMON+=(--prefill "$PREFILL")
[ -n "$REVISION" ] && COMMON+=(--revision "$REVISION")
case "$STAGE" in
  score)
    R=(); [ "$REASON" = 0 ] && R=(--no-reason)
    "$PY" analysis/adjudicator.py score "${COMMON[@]}" --gold "$OUT/gold.csv" --out-dir "$OUT/scores" \
        --limit "$LIMIT" "${R[@]}" || exit 7 ;;
  relabel)
    "$PY" analysis/adjudicator.py relabel "${COMMON[@]}" --bar "$OUT/agree/bar.json" --rows "$ROWS" \
        --types "$TYPES" --pool "$POOL" --cityparams "$CITYPARAMS" --out-dir "$OUT/relabel" || exit 7 ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "done after $(( $(date +%s) - t0 ))s"; date
