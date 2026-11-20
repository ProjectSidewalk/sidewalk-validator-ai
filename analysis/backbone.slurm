#!/bin/bash
# Issue #119: one GPU job of the backbone comparison on klone (docs/backbone-2026-11.md). Submitted by
# analysis/backbone_submit.sh, which sets every variable below; run from the repo root of a checkout of this branch.
#
#   STEP=train:   array mode: line $SLURM_ARRAY_TASK_ID of $TASKS (columns LT ARM SEED TAG LR, tab separated) goes
#                 to run_one. LOOP=1 runs every line in turn instead (the smoke job).
#   STEP=convert: backbones.py convert-rampnet --verify-with-timm into $OUT/backbones/ (skipped when its
#                 conversion.json already says verified).
#   STEP=smoke:   convert, then LOOP=1 over $TASKS, then every line again (each must stop at the done guard), the
#                 torch tests on the GPU node, and backbone_comparison.py report on the smoke records.
#   STEP=onnx:    export_onnx.py export for the run $TAG of $LT (a later stage; not part of the pilot).
#
# run_one LT ARM SEED TAG LR (requeue-safe; ckpt may preempt at any point):
#   1. returns 0 at once when $KEEP/<LT>/runs/<TAG>/done exists;
#   2. fine_tune_classifier.py --backbone ARM --seed SEED --learning-rate LR into $RUNS/<LT>/<TAG> (resumes through
#      run_key.json), then evaluate.py --checkpoint <best> --tag TAG --per-city on $DATA (the view on scrubbed, so the
#      tagged outputs land there); marker <run>/evaluated;
#   3. copies the run's small records to $KEEP/<LT>/runs/<TAG>/;
#   4. copies the best checkpoint's model.safetensors, config.json, preprocessor_config.json and trainer_state.json to
#      <run>/best/, checks the sizes, and only then deletes every checkpoint-* (marker <run>/slimmed);
#   5. touches $KEEP/<LT>/runs/<TAG>/done.
#   An empty TAG is refused before anything runs. SVA_BACKBONE_RANDOM_INIT=1 only for tags smoke-rand-*.
#
# Variables (${VAR:?} = required):
#   STEP     train | convert | smoke | onnx
#   OUT      the study's scrubbed root (runs, views, converted weights)
#   TASKS    (train, smoke) the task file
#   DATA     (train, smoke) the data view (backbone_comparison.py view)
#   RUNS     (train, smoke) checkpoint root
#   KEEP     (train, smoke) small run records (makelab for the pilot; $OUT/smoke/keep for the smoke job)
#   R        (smoke) the round-3 directory, only listed to show nothing was written under it
#   TAG, LT  (onnx) the run to export
#   EPOCHS   [25]; the smoke job sets 1
#   BASE_ENV, VENV, HF_HOME as in analysis/backbone_env.sh and analysis/backbone_prefetch.sh
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=backbone-comparison
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -uo pipefail
PY="${PY:-python}"

setup_env() {
  BASE_ENV="${BASE_ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
  VENV="${VENV:-/gscratch/scrubbed/$USER/envs/sva-bb}"
  module load conda
  eval "$(conda shell.bash hook)"
  conda activate "$BASE_ENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1 SVA_BACKBONE_DIR="$OUT/backbones"
  "$PY" -c "import torch, transformers, pandas; print('torch', torch.__version__, 'transformers', transformers.__version__, 'pandas', pandas.__version__)"
  echo "commit $(git rev-parse HEAD 2>/dev/null)"
}

# run_one LT ARM SEED TAG LR
run_one() {
  local LT="${1:-}" ARM="${2:-}" SEED="${3:-}" TAG="${4:-}" LR="${5:-}"
  if [ -z "$TAG" ] || [ -z "$LT" ] || [ -z "$ARM" ] || [ -z "$SEED" ] || [ -z "$LR" ]; then
    echo "run_one: empty field (LT='$LT' ARM='$ARM' SEED='$SEED' TAG='$TAG' LR='$LR'); refusing" >&2
    return 2
  fi
  local RUN="$RUNS/$LT/$TAG" K="$KEEP/$LT/runs/$TAG"
  if [ -f "$K/done" ]; then
    echo "$(date) $LT $TAG: done already ($K/done)"
    return 0
  fi
  mkdir -p "$RUN" "$K"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$RUN/gpu.txt"
  if [ ! -f "$RUN/evaluated" ]; then
    local RAND=()
    case "$TAG" in smoke-rand-*) RAND=(env SVA_BACKBONE_RANDOM_INIT=1) ;; esac
    echo "$(date) train $LT $ARM seed $SEED lr $LR -> $RUN"
    ${RAND[@]+"${RAND[@]}"} "$PY" -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --output-dir "$RUN" \
        --backbone "$ARM" --seed "$SEED" --epochs "${EPOCHS:-25}" --learning-rate "$LR" --num-workers 7 \
        || { echo "training failed: $LT $TAG" >&2; return 1; }
    local BEST
    BEST=$("$PY" -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN") \
        || { echo "no best checkpoint in $RUN" >&2; return 1; }
    [ -n "$BEST" ] && [ -d "$BEST" ] || { echo "no best checkpoint in $RUN" >&2; return 1; }
    "$PY" -u evaluate.py --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$TAG" \
        --per-city || { echo "evaluate failed: $LT $TAG" >&2; return 1; }
    echo "$BEST" > "$RUN/best_checkpoint.txt"
    touch "$RUN/evaluated"
  fi
  local BEST
  BEST="$(cat "$RUN/best_checkpoint.txt")"
  local f
  for f in "metrics_$TAG.json" "predictions_$TAG.csv" "per_city_$TAG.md" "slices_$TAG.md" "report_$TAG.txt"; do
    if [ -f "$DATA/$LT/$f" ]; then cp "$DATA/$LT/$f" "$K/" || return 1; fi
  done
  for f in training_run.json run_key.json gpu.txt; do
    if [ -f "$RUN/$f" ]; then cp "$RUN/$f" "$K/" || return 1; fi
  done
  if [ ! -f "$RUN/slimmed" ]; then
    mkdir -p "$RUN/best"
    for f in model.safetensors config.json preprocessor_config.json trainer_state.json; do
      [ -f "$BEST/$f" ] || { echo "$BEST/$f is missing" >&2; return 1; }
      cp "$BEST/$f" "$RUN/best/$f" || return 1
      [ "$(wc -c < "$BEST/$f")" = "$(wc -c < "$RUN/best/$f")" ] || { echo "copy of $f differs in size" >&2; return 1; }
    done
    cp "$RUN/best/trainer_state.json" "$K/trainer_state.json" || return 1
    rm -rf "$RUN"/checkpoint-*
    touch "$RUN/slimmed"
    echo "kept $RUN/best/; removed $RUN/checkpoint-*"
  fi
  [ -f "$K/trainer_state.json" ] || cp "$RUN/best/trainer_state.json" "$K/trainer_state.json" || return 1
  touch "$K/done"
  echo "$(date) $LT $TAG done -> $K"
}

# task_line N: line N of $TASKS (comment lines are not counted)
task_line() {
  grep -v '^#' "$TASKS" | sed -n "${1}p"
}

run_line() {
  local LT ARM SEED TAG LR
  IFS=$'\t' read -r LT ARM SEED TAG LR <<< "$1"
  run_one "$LT" "$ARM" "$SEED" "$TAG" "$LR"
}

convert() {
  "$PY" -u backbones.py convert-rampnet --out "$OUT/backbones/rampnet-convnextv2-base" --verify-with-timm \
      || { echo "conversion failed" >&2; return 1; }
}

main() {
  STEP="${STEP:?set STEP (train|convert|smoke|onnx)}"; OUT="${OUT:?set OUT}"
  setup_env
  case "$STEP" in
    train)
      TASKS="${TASKS:?set TASKS}"; DATA="${DATA:?set DATA}"; RUNS="${RUNS:?set RUNS}"; KEEP="${KEEP:?set KEEP}"
      if [ "${LOOP:-0}" = 1 ]; then
        local n line
        n=$(grep -cv '^#' "$TASKS")
        for i in $(seq 1 "$n"); do line=$(task_line "$i"); run_line "$line" || exit 1; done
      else
        local line
        line=$(task_line "${SLURM_ARRAY_TASK_ID:?not an array job; set LOOP=1}")
        [ -n "$line" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $TASKS" >&2; exit 2; }
        run_line "$line" || exit 1
      fi
      ;;
    convert) convert || exit 1 ;;
    smoke)
      TASKS="${TASKS:?set TASKS}"; DATA="${DATA:?set DATA}"; RUNS="${RUNS:?set RUNS}"; KEEP="${KEEP:?set KEEP}"
      R="${R:?set R}"
      export EPOCHS="${EPOCHS:-1}"
      local fails=0 n line
      convert || fails=$((fails + 1))
      cat "$OUT/backbones/rampnet-convnextv2-base/conversion.json" 2>/dev/null
      n=$(grep -cv '^#' "$TASKS")
      for i in $(seq 1 "$n"); do
        line=$(task_line "$i")
        if run_line "$line"; then echo "SMOKE PASS run $line"; else echo "SMOKE FAIL run $line"; fails=$((fails + 1)); fi
      done
      for i in $(seq 1 "$n"); do
        line=$(task_line "$i")
        if run_line "$line" | grep -q "done already"; then echo "SMOKE PASS guard $line"; else
          echo "SMOKE FAIL guard $line"; fails=$((fails + 1)); fi
      done
      "$PY" -m pytest -q -p no:cacheprovider tests/test_backbones.py tests/test_backbone_cli.py \
          && echo "SMOKE PASS pytest" || { echo "SMOKE FAIL pytest"; fails=$((fails + 1)); }
      "$PY" -u analysis/backbone_comparison.py report --split-dir "$DATA" --runs-dir "$KEEP" \
          --inputs "$OUT/smoke/backbone_inputs_smoke.json" --shares docs/deployment-prior-2026-11/shares.json \
          --public-cities "$OUT/cities.json" --label-types CurbRamp --tag-prefix smoke- --n-boot 100 \
          --manifest "$OUT/backbone_manifest.json" --pip-freeze "$OUT/pip_freeze.txt" --out "$OUT/smoke/report" \
          && echo "SMOKE PASS report" || { echo "SMOKE FAIL report"; fails=$((fails + 1)); }
      echo "written under the round-3 directory since the smoke job was submitted (expected: nothing of ours):"
      find "$R" -maxdepth 2 -newer "$OUT/smoke/.start" | head -20
      echo "SMOKE FAILURES $fails"
      [ "$fails" = 0 ] || exit 1
      ;;
    onnx)
      LT="${LT:?set LT}"; TAG="${TAG:?set TAG}"; RUNS="${RUNS:?set RUNS}"
      # Export only; quantize and bench (export_onnx.py quantize / bench) follow on the laptop or a later job.
      "$PY" -u export_onnx.py export --model "$RUNS/$LT/$TAG/best" --out "$OUT/onnx/$LT/$TAG" \
          || { echo "onnx export failed" >&2; exit 1; }
      ;;
    *) echo "unknown STEP=$STEP" >&2; exit 2 ;;
  esac
  echo "$(date) done $STEP"
}

# sbatch runs this file as a script; the tests source it for run_one.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
