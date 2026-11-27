#!/bin/bash
# The input-resolution study (docs/resolution-2026-11.md): one GPU job. Submitted by analysis/resolution_submit.sh,
# which sets every variable below; run from the repo root of a checkout of this branch. The pattern, the environment
# and the requeue-safe run_one are the backbone study's (analysis/backbone.slurm).
#
#   STEP=train:  array mode: line $SLURM_ARRAY_TASK_ID of $TASKS (columns LT RES SEED TAG, tab separated) goes to
#                run_one. LOOP=1 runs every line in turn instead (the smoke job).
#   STEP=smoke:  LOOP=1 over $TASKS (one epoch each), then every line again (each must stop at the done guard), the
#                torch tests on the GPU node, the latency bench, and resolution_study.py report on the smoke records.
#   STEP=bench:  resolution_study.py bench on this node's GPU into $KEEP/bench_<gpu>.json (weights random: no hub).
#
# run_one LT RES SEED TAG (requeue-safe; ckpt may preempt at any point, and a run longer than the QOS time limit is
# continued by the launcher's `resume` stage from its last epoch checkpoint):
#   1. returns 0 at once when $KEEP/<LT>/runs/<TAG>/done exists;
#   2. fine_tune_classifier.py --input-res RES --seed SEED into $RUNS/<LT>/<TAG> (resumes through run_key.json;
#      marker <run>/trained, so a requeue after training never re-runs the Trainer); checks that the best
#      checkpoint's saved processor crops RES and copies it to $KEEP; then evaluate.py --checkpoint <best> --tag TAG
#      --per-city on $DATA (the view, so the tagged outputs land there; that processor gives evaluate.py the run's
#      input size); marker <run>/evaluated;
#   3. copies the run's small records to $KEEP/<LT>/runs/<TAG>/;
#   4. copies the best checkpoint's model.safetensors, config.json, preprocessor_config.json and trainer_state.json to
#      <run>/best/, checks the sizes, and only then deletes every checkpoint-* (marker <run>/slimmed);
#   5. touches $KEEP/<LT>/runs/<TAG>/done.
#   An empty field is refused before anything runs.
#
# Variables (${VAR:?} = required):
#   STEP     train | smoke | bench
#   OUT      the study's scrubbed root (runs, views, task files)
#   TASKS    (train, smoke) the task file
#   DATA     (train, smoke) the data view (backbone_comparison.py view --inputs analysis/resolution_inputs.json)
#   RUNS     (train, smoke) checkpoint root
#   KEEP     small run records (makelab for the study; $OUT/smoke/keep for the smoke job)
#   R        (smoke) the round-3 directory, only listed to show nothing was written under it
#   EPOCHS   [25]; the smoke job sets 1
#   BATCH    [8] per-device batch (gradient accumulation 32 / BATCH keeps the effective batch); a value other than 8
#            is a deviation the doc records, and the report counts it only with --allow-batch
#   GC       [0] 1 adds --gradient-checkpointing (memory fallback; in the run key and training_run.json; the
#            report counts it only with --allow-gc)
#   MEM_CEILING_GB [36] (smoke) the peak GPU memory every smoke run must stay under
#   BASE_ENV, VENV, HF_HOME as in analysis/backbone_env.sh and analysis/backbone_prefetch.sh
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=resolution-study
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
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
  export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
  # The backbone study's env (analysis/backbone_env.sh): the 224 rerun is in-env against the same versions.
  "$PY" - <<'PYCHECK' || { echo "the study environment did not resolve (see above); not training" >&2; exit 3; }
import sys, torch, transformers, pandas
v = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__, "pandas": pandas.__version__}
print(" ".join(f"{k} {x}" for k, x in v.items()), "python", sys.executable)
want = {"torch": "2.7.1", "transformers": "4.56.2", "pandas": "2.3.1"}
bad = {k: v[k] for k in want if v[k] != want[k]}
sys.exit(f"unexpected versions {bad}, want {want}" if bad else 0)
PYCHECK
  echo "commit $(git rev-parse HEAD 2>/dev/null)"
}

# run_one LT RES SEED TAG
run_one() {
  local LT="${1:-}" RES="${2:-}" SEED="${3:-}" TAG="${4:-}"
  if [ -z "$TAG" ] || [ -z "$LT" ] || [ -z "$RES" ] || [ -z "$SEED" ]; then
    echo "run_one: empty field (LT='$LT' RES='$RES' SEED='$SEED' TAG='$TAG'); refusing" >&2
    return 2
  fi
  case "$RES" in 224|448|518) ;; *) echo "run_one: RES '$RES' is not a study size; refusing" >&2; return 2 ;; esac
  local RUN="$RUNS/$LT/$TAG" K="$KEEP/$LT/runs/$TAG"
  if [ -f "$K/done" ]; then
    echo "$(date) $LT $TAG: done already ($K/done)"
    return 0
  fi
  mkdir -p "$RUN" "$K"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$RUN/gpu.txt"
  if [ ! -f "$RUN/evaluated" ]; then
    if [ ! -f "$RUN/trained" ]; then
      local EXTRA=()
      [ "${BATCH:-8}" = 8 ] || EXTRA+=(--batch-size "$BATCH")
      [ "${GC:-0}" = 1 ] && EXTRA+=(--gradient-checkpointing)
      echo "$(date) train $LT res $RES seed $SEED -> $RUN"
      "$PY" -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --output-dir "$RUN" \
          --input-res "$RES" --seed "$SEED" --epochs "${EPOCHS:-25}" --num-workers 7 ${EXTRA[@]+"${EXTRA[@]}"} \
          || { echo "training failed: $LT $TAG" >&2; return 1; }
      # A requeue from here on scores the finished run; re-running the Trainer would be a zero-step attempt that
      # rewrites training_run.json with an inference-only peak memory and timestamps (PR #133 review S6).
      touch "$RUN/trained"
    fi
    local BEST
    BEST=$("$PY" -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN") \
        || { echo "no best checkpoint in $RUN" >&2; return 1; }
    [ -n "$BEST" ] && [ -d "$BEST" ] || { echo "no best checkpoint in $RUN" >&2; return 1; }
    # The processor evaluate.py will score with (and export_onnx.py and serving read): checked before scoring, and
    # kept beside the records so the report checks it too (PR #133 review S2).
    "$PY" - "$BEST/preprocessor_config.json" "$RES" <<'PYRES' || { echo "saved processor does not crop $RES" >&2; return 1; }
import json, sys
c = json.load(open(sys.argv[1]))["crop_size"]
sys.exit(0 if c.get("height") == int(sys.argv[2]) == c.get("width") else f"crop_size {c}")
PYRES
    cp "$BEST/preprocessor_config.json" "$K/preprocessor_config.json" || return 1
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
  if [ ! -f "$RUN/slimmed" ] && [ -f "$RUN/best_copied" ]; then
    rm -rf "$RUN"/checkpoint-*   # preempted after best/ was checked but before the marker: finish the slimming
    touch "$RUN/slimmed"
  fi
  if [ ! -f "$RUN/slimmed" ]; then
    mkdir -p "$RUN/best"
    for f in model.safetensors config.json preprocessor_config.json trainer_state.json; do
      [ -f "$BEST/$f" ] || { echo "$BEST/$f is missing" >&2; return 1; }
      cp "$BEST/$f" "$RUN/best/$f" || return 1
      [ "$(wc -c < "$BEST/$f")" = "$(wc -c < "$RUN/best/$f")" ] || { echo "copy of $f differs in size" >&2; return 1; }
    done
    cp "$RUN/best/trainer_state.json" "$K/trainer_state.json" || return 1
    touch "$RUN/best_copied"
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
  local LT RES SEED TAG
  IFS=$'\t' read -r LT RES SEED TAG <<< "$1"
  run_one "$LT" "$RES" "$SEED" "$TAG"
}

# memory_check KEEP: every smoke run trained with SDPA attention (eager attention stores the attention maps, about
# 26 GB more at 448 and batch 8) and peaked under MEM_CEILING_GB (default 36, so a 40 GB A100 in the constraint
# fits too); prints each run's size, attention and peak (PR #133 review S5).
memory_check() {
  local f bad=0
  echo "peak GPU memory per smoke run (GB; batch ${BATCH:-8}; ceiling ${MEM_CEILING_GB:-36}):"
  for f in "$1"/*/runs/*/training_run.json; do
    [ -f "$f" ] || continue
    "$PY" - "$f" "${MEM_CEILING_GB:-36}" <<'PYMEM' || bad=1
import json, sys
r = json.load(open(sys.argv[1]))
gb = (r.get("peak_gpu_memory_bytes") or 0) / 2 ** 30
print(r.get("input_res"), r.get("attn_implementation"), round(gb, 1))
bad = [m for m, ok in (("attention is not sdpa", r.get("attn_implementation") == "sdpa"),
                       (f"peak {gb:.1f} GB is over {sys.argv[2]} GB", gb < float(sys.argv[2]))) if not ok]
sys.exit("; ".join(bad) if bad else 0)
PYMEM
  done
  return "$bad"
}

bench() {
  local name
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_')
  mkdir -p "$KEEP"
  "$PY" -u analysis/resolution_study.py bench --out "$KEEP/bench_${name:-gpu}.json" \
      || { echo "bench failed" >&2; return 1; }
}

main() {
  STEP="${STEP:?set STEP (train|smoke|bench)}"; OUT="${OUT:?set OUT}"
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
    bench) KEEP="${KEEP:?set KEEP}"; bench || exit 1 ;;
    smoke)
      TASKS="${TASKS:?set TASKS}"; DATA="${DATA:?set DATA}"; RUNS="${RUNS:?set RUNS}"; KEEP="${KEEP:?set KEEP}"
      R="${R:?set R}"
      export EPOCHS="${EPOCHS:-1}"
      local fails=0 n line
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
      "$PY" -m pytest -q -p no:cacheprovider tests/test_input_resolution.py tests/test_backbones.py \
          && echo "SMOKE PASS pytest" || { echo "SMOKE FAIL pytest"; fails=$((fails + 1)); }
      bench && echo "SMOKE PASS bench" || { echo "SMOKE FAIL bench"; fails=$((fails + 1)); }
      if [ ! -f "$OUT/cities.json" ]; then
        echo "SMOKE FAIL report: $OUT/cities.json is missing (the curl in 'view' failed); save it and rerun"
        fails=$((fails + 1))
      else
        "$PY" -u analysis/resolution_study.py report --split-dir "$DATA" --runs-dir "$KEEP" \
            --inputs "$OUT/smoke/resolution_inputs_smoke.json" --public-cities "$OUT/cities.json" \
            --label-types CurbRamp --tag-prefix smoke-res- --n-boot 50 --min-boot 20 --epochs "$EPOCHS" --arms 448 \
            $(for b in "$KEEP"/bench_*.json; do [ -f "$b" ] && printf -- '--bench %s ' "$b"; done) \
            --out "$OUT/smoke/report" \
            && echo "SMOKE PASS report" || { echo "SMOKE FAIL report"; fails=$((fails + 1)); }
      fi
      memory_check "$KEEP" && echo "SMOKE PASS memory" || { echo "SMOKE FAIL memory"; fails=$((fails + 1)); }
      echo "written under the round-3 directory since the smoke job was submitted (expected: nothing of ours):"
      find "$R" -maxdepth 2 -newer "$OUT/smoke/.start" | head -20
      echo "SMOKE FAILURES $fails"
      [ "$fails" = 0 ] || exit 1
      ;;
    *) echo "unknown STEP=$STEP" >&2; exit 2 ;;
  esac
  echo "$(date) done $STEP"
}

# sbatch runs this file as a script; the tests source it for run_one.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
