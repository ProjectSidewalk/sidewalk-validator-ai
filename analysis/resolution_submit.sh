#!/bin/bash
# The input-resolution study (docs/resolution-2026-11.md): build the data views and submit its jobs on klone. The
# backbone study's launcher pattern (analysis/backbone_submit.sh): a job cap counted from a ledger, a real write test,
# free-space checks, a done marker per run, and squeue-based skipping of rows already queued or running. Run on the
# login node from the repo root of a checkout of this branch, after analysis/backbone_env.sh (the same overlay env):
#
#   analysis/resolution_submit.sh view          # $OUT/data: verified copies of the five types' small files, crops linked
#   analysis/resolution_submit.sh view smoke    # $OUT/smoke/data: a seeded class-balanced CurbRamp subsample
#   analysis/resolution_submit.sh smoke         # ONE job: 224/448/518 one-epoch runs, guards, torch tests, bench, report
#   analysis/resolution_submit.sh bench         # ONE job on an A40 (the serving GPU class): latency per crop per size
#   MAX_PARALLEL=8 analysis/resolution_submit.sh main     # one array: 5 types x {224, 448} x 5 seeds = 50 elements
#   ALLOW_518=1 analysis/resolution_submit.sh r518        # 25 elements; only on Jon's word (cost; doc, J3)
#   analysis/resolution_submit.sh resume        # every planned row with no done marker that is not queued or running
#                                               # (a run past the QOS time limit, or a failed one): continues from its
#                                               # last epoch checkpoint; counts against the cap
#
# Before any submit: a real 8 MiB write to $OUT; scrubbed has MAX_PARALLEL x 12 GB + 1.3 GB per element free;
# /gscratch/makelab has 5 GB and 1,000 files free for $KEEP; the stream's elements so far (from $OUT/resolution_jobs.txt)
# plus the new ones stay within JOB_CAP; no smoke- tag outside the smoke job; no row whose run is done
# ($KEEP/<LT>/runs/<TAG>/done) or that squeue lists as pending, running or requeued (job name $SLUG-<stage>, mapped back
# to its task-file row through the ledger). Each submit appends date, job id, element count, stage, commit, task file
# and its sha256 to the ledger and copies the ledger and the task file to $KEEP.
#
# Variables (defaults in brackets):
#   OUT       [/gscratch/scrubbed/$USER/resolution-study-2026-11] runs, views, task files
#   KEEP      [/gscratch/makelab/$USER/resolution-study-2026-11-keep] small run records only
#   R         [/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10] round-3 data, read only
#   VENV, BASE_ENV as in analysis/backbone_env.sh; PY [$VENV/bin/python]
#   MAX_PARALLEL [8] array throttle; TYPES, SEEDS override a stage's rows; SBATCH [sbatch]; SQUEUE [squeue]
#   BATCH [8], GC [0] passed to the job (memory fallbacks; any other value is a recorded deviation)
#   CITIES_URL [the Project Sidewalk /v3/api/cities listing] saved by `view` for the offline report
set -euo pipefail
SLUG=resolution-study
OUT="${OUT:-/gscratch/scrubbed/$USER/$SLUG-2026-11}"
KEEP="${KEEP:-/gscratch/makelab/$USER/$SLUG-2026-11-keep}"
R="${R:-/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10}"
VENV="${VENV:-/gscratch/scrubbed/$USER/envs/sva-bb}"
PY="${PY:-$VENV/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SBATCH="${SBATCH:-sbatch}"
SQUEUE="${SQUEUE:-squeue}"
CITIES_URL="${CITIES_URL:-https://sidewalk-columbus.cs.washington.edu/v3/api/cities}"
# Cap for this stream (doc, J2; 2026-11-27, five seeds): smoke 1 + bench 1 + main 50 + 35 resumes (the fifteen 448
# rows of the three large types may need two each past the QOS's 9:05 limit, CurbRamp's five one each) + 5 spare.
# The 518 stage needs a raise (25 rows and their resumes). Raise it only on Jon's word.
JOB_CAP=92
EPOCHS_MAIN=25
ALL_TYPES="Obstacle SurfaceProblem NoCurbRamp CurbRamp Crosswalk"
MAIN_RES="224 448"
LATER_RES="518"
SMOKE_RES="224 448 518"
LEDGER="$OUT/resolution_jobs.txt"
INPUTS=analysis/resolution_inputs.json

die() { echo "$*" >&2; exit 2; }

# tag PREFIX RES SEED -- the same tags as resolution_study.make_tag
tag() {
  local t="$1r$2"
  [ "$3" = 42 ] || t="$t-s$3"
  echo "$t"
}

write_test() {
  mkdir -p "$OUT"
  dd if=/dev/zero of="$OUT/.rs_write_test" bs=1M count=8 status=none \
    || die "an 8 MiB write to $OUT failed; not submitting"
  rm -f "$OUT/.rs_write_test"
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

ledger() {   # ledger JOBID N STAGE TASKFILE
  local sha=-
  [ "$4" = - ] || sha=$(sha256sum "$4" | cut -d' ' -f1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    "$(git rev-parse HEAD 2>/dev/null || echo unknown)" "$4" "$sha" >> "$LEDGER"
  mkdir -p "$KEEP"
  cp "$LEDGER" "$KEEP/"
  [ "$4" = - ] || cp "$4" "$KEEP/"
  local p
  for p in pip_freeze.txt cities.json; do
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
STAGES = {"smoke", "main", "r518", "resume", "bench"}
def indices(spec):   # "7" or "[3-5,9%8]" / "[14-30:2%8]" (a pending range squeue did not expand)
    out = []
    for part in spec.strip("[]").split("%")[0].split(","):
        rng, _, step = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(step or 1)))
    return out
for ln in open(qfile):
    parts = ln.split()
    if len(parts) < 2 or parts[1] not in {f"{slug}-{st}" for st in STAGES}:
        continue
    jid, _, idx = parts[0].partition("_")
    if jid not in jobs:
        sys.exit(f"squeue lists {parts[0]} ({parts[1]}), which is not in the ledger {ledger}")
    if jobs[jid] == "-":   # the bench job has no task file and trains nothing
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

add_row() {   # add_row FILE LT RES SEED PREFIX
  local f="$1" lt="$2" res="$3" seed="$4" prefix="${5:-res-}" t
  [ -f "$OUT/data/$lt/split.csv" ] || [ "$prefix" = smoke-res- ] || die "no view for $lt ($OUT/data/$lt); run 'view'"
  t=$(tag "$prefix" "$res" "$seed")
  if [ "$prefix" != smoke-res- ] && [ -f "$KEEP/$lt/runs/$t/done" ]; then
    echo "skipping $lt $t: done already"; return 0
  fi
  [ "$prefix" = smoke-res- ] || [ -n "$INFLIGHT" ] || in_flight > /dev/null
  if [ "$prefix" != smoke-res- ] && grep -qxF "$lt"$'\t'"$t" "$INFLIGHT"; then
    echo "skipping $lt $t: queued or running (squeue)"; return 0
  fi
  printf '%s\t%s\t%s\t%s\n' "$lt" "$res" "$seed" "$t" >> "$f"
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
       BATCH="${BATCH:-8}" GC="${GC:-0}" \
       "$SBATCH" --parsable --job-name="$SLUG-$stage" --array="1-$n%$MAX_PARALLEL" analysis/resolution.slurm)
  ledger "$id" "$n" "$stage" "$f"
}

taskfile() { mkdir -p "$OUT"; local f="$OUT/tasks_$1_$(date -u +%Y%m%dT%H%M%SZ).tsv"; : > "$f"; echo "$f"; }

planned() {   # planned RES_LIST -> rows "LT RES SEED" of a stage, seed-major so seed 42 of every row starts first
  local seed lt res
  for seed in ${SEEDS:-42 43 44 45 46}; do
    for lt in ${TYPES:-$ALL_TYPES}; do
      for res in $1; do printf '%s\t%s\t%s\n' "$lt" "$res" "$seed"; done
    done
  done
}

cmd="${1:-}"
case "$cmd" in
  view)
    if [ "${2:-}" = smoke ]; then
      "$PY" analysis/backbone_comparison.py view --src "$R" --dst "$OUT/smoke/data" --inputs "$INPUTS" --smoke \
          --label-types CurbRamp --smoke-inputs-out "$OUT/smoke/resolution_inputs_smoke.json" --scratch "$OUT/smoke"
    else
      "$PY" analysis/backbone_comparison.py view --src "$R" --dst "$OUT/data" --inputs "$INPUTS" \
          --label-types "$(echo "${TYPES:-$ALL_TYPES}" | tr ' ' ',')"
    fi
    curl -sSf --max-time 60 "$CITIES_URL" -o "$OUT/cities.json.tmp" && mv "$OUT/cities.json.tmp" "$OUT/cities.json" \
      || echo "could not save $CITIES_URL; the report will need --public-cities" >&2
    [ -f "$VENV/bin/pip" ] && "$VENV/bin/pip" freeze > "$OUT/pip_freeze.txt" || true
    ;;
  smoke)
    [ -f "$OUT/smoke/resolution_inputs_smoke.json" ] || die "run 'view smoke' first"
    f="$OUT/smoke/tasks_smoke.tsv"
    : > "$f"
    for res in $SMOKE_RES; do add_row "$f" CurbRamp "$res" 42 smoke-res-; done
    cap_check 1
    write_test
    mkdir -p logs "$OUT/smoke/keep"
    touch "$OUT/smoke/.start"
    id=$(env -u CONDA_PREFIX -u CONDA_SHLVL -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_EXE \
         STEP=smoke OUT="$OUT" TASKS="$f" DATA="$OUT/smoke/data" RUNS="$OUT/smoke/runs" KEEP="$OUT/smoke/keep" R="$R" \
         EPOCHS=1 BATCH="${BATCH:-8}" GC="${GC:-0}" \
         "$SBATCH" --parsable --job-name="$SLUG-smoke" --time=02:00:00 --array=1 analysis/resolution.slurm)
    ledger "$id" 1 smoke "$f"
    ;;
  bench)
    cap_check 1
    write_test
    mkdir -p logs "$KEEP"
    id=$(env -u CONDA_PREFIX -u CONDA_SHLVL -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_EXE \
         STEP=bench OUT="$OUT" KEEP="$KEEP" "$SBATCH" --parsable --job-name="$SLUG-bench" --time=01:00:00 \
         --constraint=a40 --array=1 analysis/resolution.slurm)
    ledger "$id" 1 bench -
    ;;
  main|r518)
    if [ "$cmd" = r518 ]; then
      [ "${ALLOW_518:-0}" = 1 ] || die "the 518 arm runs only on Jon's word (docs/resolution-2026-11.md, J3); set ALLOW_518=1"
      res_list="$LATER_RES"
    else
      res_list="$MAIN_RES"
    fi
    in_flight > /dev/null
    f=$(taskfile "$cmd")
    while IFS=$'\t' read -r lt res seed; do add_row "$f" "$lt" "$res" "$seed"; done < <(planned "$res_list")
    submit_array "$cmd" "$f"
    ;;
  resume)
    # Every planned row (main, and r518 when ALLOW_518=1) not done and not in flight. fine_tune_classifier.py resumes
    # from the run's last complete epoch checkpoint under the same run_key.json; a finished-but-unslimmed run only
    # finishes its copy and slimming.
    res_list="$MAIN_RES"
    [ "${ALLOW_518:-0}" = 1 ] && res_list="$MAIN_RES $LATER_RES"
    in_flight > /dev/null
    f=$(taskfile resume)
    while IFS=$'\t' read -r lt res seed; do
      [ -d "$OUT/runs/$lt/$(tag res- "$res" "$seed")" ] || continue   # never started: belongs to its stage
      add_row "$f" "$lt" "$res" "$seed"
    done < <(planned "$res_list")
    submit_array resume "$f"
    ;;
  *) die "usage: $0 view [smoke] | smoke | bench | main | r518 | resume" ;;
esac
