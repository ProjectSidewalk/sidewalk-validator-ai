#!/bin/bash
# Issue #119: build the data views and submit the backbone comparison's jobs on klone (docs/backbone-2026-11.md).
# Run on the login node from the repo root of a checkout of this branch, after analysis/backbone_env.sh and
# analysis/backbone_prefetch.sh:
#
#   analysis/backbone_submit.sh view           # $OUT/data: verified copies of round 3's small files, crop dirs linked
#   analysis/backbone_submit.sh view smoke     # $OUT/smoke/data: a seeded class-balanced CurbRamp subsample
#   analysis/backbone_submit.sh smoke          # ONE job: convert RampNet, 5 one-epoch runs, guards, torch tests, report
#   MAX_PARALLEL=8 analysis/backbone_submit.sh pilot   # one array, 30 elements (24 main + 6 learning-rate rows)
#   analysis/backbone_submit.sh dinov3         # the 18 DINOv3 runs (3 arms x 2 pilot types x 3 seeds), after prefetch
#   analysis/backbone_submit.sh rest | seeds   # REFUSED: the report cannot read them yet (needs the follow-up PR)
#   LT=<LT> TAG=<tag> analysis/backbone_submit.sh onnx   # later; Jon's call; needs onnx + onnxruntime in $PY
#   analysis/backbone_submit.sh resubmit <task file>   # rows to run again (failed elements); counts against the cap
#
# Before any submit: a real 8 MiB write to $OUT; scrubbed has MAX_PARALLEL x 12 GB + 1.3 GB per element free;
# /gscratch/makelab has 5 GB and 1,000 files free for $KEEP; every arm is available in $OUT/backbone_manifest.json
# (an unavailable one is skipped with a message) and the converted RampNet weights are verified; the stream's
# elements so far (from $OUT/backbone_jobs.txt) plus the new ones stay within JOB_CAP; no smoke- tag outside the
# smoke job; no row whose run is done ($KEEP/<LT>/runs/<TAG>/done) or in flight. "In flight" is read from squeue, not
# from a lock file: every element of this stream that squeue lists (pending, running or requeued; job name
# $SLUG-*) is mapped back to its task-file row through the ledger's job id -> task file column, so a second element
# never trains in the same run directory, and nothing is written into run directories (the running arrays' own
# done/slimmed logic is untouched). The launcher refuses to submit when squeue cannot be read, or when squeue lists a
# job of this stream that the ledger does not know. Each submit appends date, job id, element count, stage, commit,
# task file and its sha256 to $OUT/backbone_jobs.txt and copies the ledger and the task file to $KEEP.
#
# Variables (defaults in brackets):
#   OUT       [/gscratch/scrubbed/$USER/backbone-comparison-2026-11] runs, views, converted weights, task files
#   KEEP      [/gscratch/makelab/$USER/backbone-comparison-2026-11-keep] small run records only
#   R         [/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10] round-3 data, read only
#   VENV, BASE_ENV as in analysis/backbone_env.sh; PY [$VENV/bin/python]
#   MAX_PARALLEL [8] array throttle; ARMS, TYPES, SEEDS override a stage's rows; SBATCH [sbatch]; SQUEUE [squeue]
#   CITIES_URL [the Project Sidewalk /v3/api/cities listing] saved by `view` for the offline report
set -euo pipefail
SLUG=backbone-comparison
OUT="${OUT:-/gscratch/scrubbed/$USER/$SLUG-2026-11}"
KEEP="${KEEP:-/gscratch/makelab/$USER/$SLUG-2026-11-keep}"
R="${R:-/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10}"
VENV="${VENV:-/gscratch/scrubbed/$USER/envs/sva-bb}"
PY="${PY:-$VENV/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SBATCH="${SBATCH:-sbatch}"
SQUEUE="${SQUEUE:-squeue}"
CITIES_URL="${CITIES_URL:-https://sidewalk-columbus.cs.washington.edu/v3/api/cities}"
# Jon's cap for this stream: at most 40 GPU jobs in all, array elements counted (2026-11-19), raised to 55 the same
# night for the 18 DINOv3 runs plus resubmits. Raise it only on his word.
JOB_CAP=55
EPOCHS_MAIN=25
DINOV3_ARMS="dinov3-vitl16 dinov3-vitl16-256 dinov3-vitb16"
LEDGER="$OUT/backbone_jobs.txt"
PILOT_TYPES="CurbRamp SurfaceProblem"
PILOT_ARMS="dinov2-large dinov2-reg-large convnextv2-base rampnet-convnextv2-base"
LR_ARMS="dinov2-large convnextv2-base rampnet-convnextv2-base"
SMOKE_ARMS="dinov2-large dinov2-reg-large convnextv2-base rampnet-convnextv2-base"
SMOKE_RAND_ARMS="dinov3-vitl16"
RAMPNET_DIR="$OUT/backbones/rampnet-convnextv2-base"

die() { echo "$*" >&2; exit 2; }

# tag PREFIX ARM SEED LR -- the same tags as backbone_comparison.make_tag
tag() {
  local t="$1$2"
  [ "$4" = 1e-5 ] || t="$t-lr$(echo "$4" | tr -d -- '-')"
  [ "$3" = 42 ] || t="$t-s$3"
  echo "$t"
}

write_test() {
  mkdir -p "$OUT"
  dd if=/dev/zero of="$OUT/.bb_write_test" bs=1M count=8 status=none \
    || die "an 8 MiB write to $OUT failed; not submitting"
  rm -f "$OUT/.bb_write_test"
}

free_gb() { df -Pk "$1" | awk 'NR == 2 { print int($4 / 1048576) }'; }

space_check() {   # space_check N_ELEMENTS
  local need have
  need=$(awk -v p="$MAX_PARALLEL" -v n="$1" 'BEGIN { printf "%d", p * 12 + 1.3 * n + 0.999 }')
  have=$(free_gb "$OUT")
  [ "$have" -ge "$need" ] || die "scrubbed has ${have} GB free under $OUT; $1 elements at $MAX_PARALLEL in parallel need $need"
  hyakstorage /gscratch/makelab | "$PY" backbones.py storage-free --min-gb 5 --min-files 1000 \
    || die "too little free space on /gscratch/makelab for $KEEP"
}

used() { [ -f "$LEDGER" ] && awk -F'\t' '{ s += $3 } END { print s + 0 }' "$LEDGER" || echo 0; }

cap_check() {   # cap_check N_ELEMENTS
  local u
  u=$(used)
  [ $((u + $1)) -le "$JOB_CAP" ] || die "the stream has used $u of $JOB_CAP GPU jobs; $1 more would exceed the cap"
}

available() {   # available ARM -> 0 when its weights are in the manifest (and, for RampNet, converted and verified)
  local arm="$1"
  "$PY" - "$OUT/backbone_manifest.json" "$arm" <<'PY' || return 1
import json, sys
m = json.load(open(sys.argv[1]))["backbones"].get(sys.argv[2], {})
sys.exit(0 if m.get("available") else 1)
PY
  if [ "$arm" = rampnet-convnextv2-base ] && [ "${STAGE:-}" != smoke ]; then
    grep -q '"verified_against_timm": true' "$RAMPNET_DIR/conversion.json" 2>/dev/null || return 1
  fi
  return 0
}

ledger() {   # ledger JOBID N STAGE TASKFILE
  local sha=-
  [ "$4" = - ] || sha=$(sha256sum "$4" | cut -d' ' -f1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    "$(git rev-parse HEAD 2>/dev/null || echo unknown)" "$4" "$sha" >> "$LEDGER"
  mkdir -p "$KEEP"
  cp "$LEDGER" "$KEEP/"
  [ "$4" = - ] || cp "$4" "$KEEP/"
  # Provenance the report needs, kept past scrubbed's purge (about 30 KB).
  local p
  for p in backbone_manifest.json pip_freeze.txt constraints.txt cities.json; do
    [ -f "$OUT/$p" ] && cp "$OUT/$p" "$KEEP/"
  done
  echo "job $1: $2 element(s), stage $3, task file $4 ($sha)"
}

# in_flight: a file of "LT<TAB>TAG" lines, one per element of this stream that squeue lists (pending, running,
# requeued), mapped through the ledger (job id -> task file, the $KEEP copy when scrubbed has purged it). Computed once.
INFLIGHT=""
in_flight() {
  [ -z "$INFLIGHT" ] || { echo "$INFLIGHT"; return 0; }
  mkdir -p "$OUT"
  local q="$OUT/.squeue.$$"
  "$SQUEUE" -h -r -u "${USER:-$(id -un)}" -o '%i %j' > "$q" \
    || { rm -f "$q"; die "could not read $SQUEUE; not submitting (a queued or running element could be doubled)"; }
  "$PY" - "$LEDGER" "$KEEP" "$SLUG" "$q" > "$q.tags" <<'PY' \
    || { rm -f "$q" "$q.tags"; die "could not map squeue to task rows (above); not submitting"; }
import os, sys
ledger, keep, slug, qfile = sys.argv[1:5]
jobs = {}
if os.path.isfile(ledger):
    for ln in open(ledger):
        c = ln.rstrip("\n").split("\t")
        if len(c) >= 6:
            jobs[c[1]] = c[5]
def indices(spec):   # "7" or "[3-5,9%8]" (a pending range squeue did not expand)
    out = []
    for part in spec.strip("[]").split("%")[0].split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out
for ln in open(qfile):
    parts = ln.split()
    if len(parts) < 2 or not parts[1].startswith(slug + "-"):
        continue
    jid, _, idx = parts[0].partition("_")
    if jid not in jobs:
        sys.exit(f"squeue lists {parts[0]} ({parts[1]}), which is not in the ledger {ledger}")
    if jobs[jid] == "-":   # the onnx stage has no task file and trains nothing
        continue
    f = jobs[jid] if os.path.isfile(jobs[jid]) else os.path.join(keep, os.path.basename(jobs[jid]))
    if not os.path.isfile(f):
        sys.exit(f"job {jid}: its task file {jobs[jid]} is gone, and there is no copy in {keep}")
    rows = [r.split("\t") for r in open(f).read().splitlines() if r and not r.startswith("#")]
    for i in indices(idx or "1"):
        if 1 <= i <= len(rows):
            print(f"{rows[i - 1][0]}\t{rows[i - 1][3]}")
PY
  rm -f "$q"
  INFLIGHT="$q.tags"
  trap 'rm -f "$INFLIGHT"' EXIT
  echo "$INFLIGHT"
}

add_row() {   # add_row FILE LT ARM SEED LR PREFIX -- skips an unavailable arm once, with a message
  local f="$1" lt="$2" arm="$3" seed="$4" lr="$5" prefix="${6:-bb-}" t
  if ! available "$arm"; then
    grep -qx "$arm" "$f.skipped" 2>/dev/null || { echo "skipping $arm: not available (manifest or conversion)"; echo "$arm" >> "$f.skipped"; }
    return 0
  fi
  [ -f "$OUT/data/$lt/split.csv" ] || [ "$prefix" = smoke- ] || die "no view for $lt ($OUT/data/$lt); run 'view' with TYPES"
  t=$(tag "$prefix" "$arm" "$seed" "$lr")
  if [ "$prefix" != smoke- ] && [ -f "$KEEP/$lt/runs/$t/done" ]; then
    echo "skipping $lt $t: done already"; return 0
  fi
  [ "$prefix" = smoke- ] || [ -n "$INFLIGHT" ] || in_flight > /dev/null
  if [ "$prefix" != smoke- ] && grep -qxF "$lt"$'\t'"$t" "$INFLIGHT"; then
    echo "skipping $lt $t: queued or running (squeue)"; return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$lt" "$arm" "$seed" "$t" "$lr" >> "$f"
}

submit_array() {   # submit_array STAGE TASKFILE
  local stage="$1" f="$2" n id
  n=$(grep -cv '^#' "$f" || true)
  [ "$n" -gt 0 ] || die "no runnable rows for $stage"
  if cut -f4 "$f" | grep -q '^smoke-'; then die "a smoke- tag in stage $stage; smoke tags belong to the smoke job only"; fi
  cap_check "$n"
  write_test
  space_check "$n"
  mkdir -p logs "$KEEP"
  id=$(env -u CONDA_PREFIX -u CONDA_SHLVL -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_EXE \
       STEP=train OUT="$OUT" TASKS="$f" DATA="$OUT/data" RUNS="$OUT/runs" KEEP="$KEEP" EPOCHS="$EPOCHS_MAIN" \
       "$SBATCH" --parsable --job-name="$SLUG-$stage" --array="1-$n%$MAX_PARALLEL" analysis/backbone.slurm)
  ledger "$id" "$n" "$stage" "$f"
}

taskfile() { mkdir -p "$OUT"; local f="$OUT/tasks_$1_$(date -u +%Y%m%dT%H%M%SZ).tsv"; : > "$f"; echo "$f"; }

cmd="${1:-}"
case "$cmd" in
  view)
    if [ "${2:-}" = smoke ]; then
      "$PY" analysis/backbone_comparison.py view --src "$R" --dst "$OUT/smoke/data" --smoke \
          --smoke-inputs-out "$OUT/smoke/backbone_inputs_smoke.json" --scratch "$OUT/smoke"
    else
      "$PY" analysis/backbone_comparison.py view --src "$R" --dst "$OUT/data" \
          --label-types "$(echo "${TYPES:-$PILOT_TYPES}" | tr ' ' ',')"
    fi
    curl -sSf --max-time 60 "$CITIES_URL" -o "$OUT/cities.json.tmp" && mv "$OUT/cities.json.tmp" "$OUT/cities.json" \
      || echo "could not save $CITIES_URL; the report will need --public-cities" >&2
    ;;
  smoke)
    STAGE=smoke
    [ -f "$OUT/smoke/backbone_inputs_smoke.json" ] || die "run 'view smoke' first"
    f="$OUT/smoke/tasks_smoke.tsv"
    : > "$f"
    for arm in $SMOKE_ARMS; do add_row "$f" CurbRamp "$arm" 42 1e-5 smoke-; done
    for arm in $SMOKE_RAND_ARMS; do
      printf 'CurbRamp\t%s\t42\t%s\t1e-5\n' "$arm" "$(tag smoke-rand- "$arm" 42 1e-5)" >> "$f"
    done
    cap_check 1
    write_test
    mkdir -p logs "$OUT/smoke/keep"
    touch "$OUT/smoke/.start"
    id=$(STEP=smoke OUT="$OUT" TASKS="$f" DATA="$OUT/smoke/data" RUNS="$OUT/smoke/runs" KEEP="$OUT/smoke/keep" R="$R" \
         EPOCHS=1 "$SBATCH" --parsable --job-name="$SLUG-smoke" --time=02:00:00 --array=1 analysis/backbone.slurm)
    ledger "$id" 1 smoke "$f"
    ;;
  pilot)
    in_flight > /dev/null
    f=$(taskfile pilot)
    for seed in ${SEEDS:-42 43 44}; do
      for lt in ${TYPES:-$PILOT_TYPES}; do
        for arm in ${ARMS:-$PILOT_ARMS}; do add_row "$f" "$lt" "$arm" "$seed" 1e-5; done
      done
    done
    for lt in ${TYPES:-$PILOT_TYPES}; do
      for arm in ${LR_ARMS_OVERRIDE:-$LR_ARMS}; do add_row "$f" "$lt" "$arm" 42 3e-5; done
    done
    submit_array pilot "$f"
    ;;
  dinov3)
    # The DINOv3 arms (gated on HF; analysis/backbone_prefetch.sh with a token first), three seeds on the pilot types,
    # compared with the pilot's dinov2-large runs. 18 elements by default.
    in_flight > /dev/null
    f=$(taskfile dinov3)
    for seed in ${SEEDS:-42 43 44}; do
      for lt in ${TYPES:-$PILOT_TYPES}; do
        for arm in ${ARMS:-$DINOV3_ARMS}; do
          case "$arm" in dinov3-*) ;; *) die "the dinov3 stage takes DINOv3 arms only (got $arm)" ;; esac
          add_row "$f" "$lt" "$arm" "$seed" 1e-5
        done
      done
    done
    submit_array dinov3 "$f"
    ;;
  seeds|rest)
    # Both would spend GPU jobs the report cannot read yet: it takes the pilot types only, has no confirmation block,
    # and the other three types' round-3 files are not pinned (PR #121 review S8).
    die "the $cmd stage needs the follow-up PR (report support for other types and extra seeds); not submitting"
    ;;
  resubmit)
    # A hand-made task file of rows to run again (e.g. failed pilot elements, copied from the pilot's task file).
    # Each row goes through add_row, so a done, queued or running row is skipped and an unavailable arm refused; a tag
    # that is not the planned one for its arm, seed and rate is refused (the report would never read it).
    src="${2:?usage: $0 resubmit <task file>}"
    [ -f "$src" ] || die "no $src"
    in_flight > /dev/null
    f=$(taskfile resubmit)
    while IFS=$'\t' read -r lt arm seed t lr _; do
      [ -n "$lt" ] || continue
      case "$t" in smoke-*) die "a smoke- tag in stage resubmit; smoke tags belong to the smoke job only" ;; esac
      [ "$(tag bb- "$arm" "$seed" "$lr")" = "$t" ] \
        || die "row $lt $t: not the planned tag for $arm seed $seed lr $lr (the report reads planned tags only)"
      add_row "$f" "$lt" "$arm" "$seed" "$lr"
    done < <(grep -v '^#' "$src" || true)
    submit_array resubmit "$f"
    ;;
  onnx)
    : "${LT:?set LT}" "${TAG:?set TAG}"
    # Refused up front rather than after spending a cap job (PR #121 review S9).
    case "$TAG" in *dinov2-reg-large*)
      die "$TAG: Dinov2WithRegisters does not export at opset 17 (docs/backbone-2026-11.md); not submitting" ;; esac
    "$PY" -c "import onnx, onnxruntime" 2>/dev/null \
      || die "onnx and onnxruntime do not import with $PY (the study env does not install them); add them to \$VENV first"
    [ -s "$OUT/runs/$LT/$TAG/best/model.safetensors" ] \
      || die "no $OUT/runs/$LT/$TAG/best/model.safetensors (not finished, or purged from scrubbed); not submitting"
    cap_check 1
    write_test
    mkdir -p logs
    id=$(STEP=onnx OUT="$OUT" LT="$LT" TAG="$TAG" RUNS="$OUT/runs" "$SBATCH" --parsable --job-name="$SLUG-onnx" \
         --time=02:00:00 --array=1 analysis/backbone.slurm)
    ledger "$id" 1 onnx -
    ;;
  *) die "usage: $0 view [smoke] | smoke | pilot | dinov3 | seeds | rest | resubmit <task file> | onnx" ;;
esac
