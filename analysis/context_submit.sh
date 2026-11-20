#!/bin/bash
# Submit one stage of the issue #14 context-arms experiment on klone (docs/context-arms-2026-11.md), from a clone of
# this branch (the jobs run the code of this clone):
#
#   STAGE=pilot DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem
#   STAGE=smoke DATA=$WORK/smoke/data RUNS=$WORK/smoke/runs PANO_STORE=/abs/store analysis/context_submit.sh NoCurbRamp
#   STAGE=A DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP PANO_STORE=/abs/store DRY_RUN=1 analysis/context_submit.sh ...
#
# Stage presets (the job list of each stage lives here and is tested, not typed by hand). A run spec is
# variant[@res]:seed[,seed...]; variants: baseline-rerun (ARM=none, the tight-crop model on today's code), two_scale,
# view_heatmap, side_info, side_info_nohq; @448 only for view_heatmap. Tags: seed 42 -> <variant>[-r448], any other seed
# -> <variant>[-r448]-s<seed> (analysis/confident_learning.SEED_RUN_RE reads them back).
#   smoke  baseline-rerun:42 two_scale:42 view_heatmap:42 view_heatmap@448:42 side_info:42 side_info_nohq:42, EPOCHS=1,
#          plus the backfill: 7 jobs per type (every code path once)
#   pilot  baseline-rerun:42,43,44,45,46 side_info:42 side_info_nohq:42: 7 jobs per type, no backfill, no panoramas
#   A      two_scale:42,43,44 view_heatmap:42,43,44 view_heatmap@448:42 plus one backfill: 8 jobs per type (A5=1: the
#          two confirmatory arms at seeds 42 to 46, 12 jobs per type)
#   B      baseline-rerun:42,43,44,45,46 $WINNER:42,43,44 (A5=1: 42 to 46) plus one backfill when $WINNER reads the
#          context images: the rule reads the same five baseline seeds on every type, guard types included
#
# The presets' seed lists must equal DECISION_RULE's (analysis/context_arms.py): the launcher passes them to the
# preflight, which refuses a mismatch, so A5=1 needs seeds_confirmatory changed in the rule and the doc's json block
# first. A repeated label type or a repeated tag is refused, and (where squeue exists) so is a job whose type and tag
# is already queued or running under any stage: two trainers must never share a run directory (the job also holds a
# lock on it).
#
# Per type: at most one backfill; the image arms (two_scale, view_heatmap) wait for it (afterok), the baseline rerun and
# the side arms start at once and never die with a failed backfill. Before any sbatch the stage's preflight runs on this
# node (analysis/context_arms.py preflight, with $ENV's python: no torch) and the job count is checked against
# MAX_JOBS and against the session cap SESSION_MAX_JOBS, which counts every job line already in context_jobs_*.txt
# (not under DRY_RUN=1, which only prints; the pilot and smoke logs alone hold 29 lines, so stage A needs
# ALLOW_OVER_SESSION_CAP=1 once Jon has approved it).
# Job ids, with each job's full command, are appended to $JOBS_DIR/context_jobs_<stage>.txt. The script stops at the
# first sbatch that fails and never retries (a socket timeout may still have queued the job: check squeue).
#
# Variables (${VAR:?} = required):
#   STAGE         smoke | pilot | A | B
#   DATA          the shadow data directory (analysis/context_arms.py shadow; refused without shadow_run.json unless
#                 ALLOW_SOURCE_DATA=1)
#   RUNS          checkpoint root on scratch
#   KEEP          optional durable directory for each run's small artifacts (passed to the jobs)
#   PANO_STORE    the pano store, needed when the stage submits a backfill
#   RUNS_SPEC     replaces the preset's run list (e.g. "side_info_nohq:43" to rerun one job)
#   WINNER        stage B: the arm that won stage A
#   A5            stage A: 1 runs the confirmatory arms at five seeds
#   BACKFILL_JOB  attach the image arms to this already submitted backfill (one label type per invocation)
#   SKIP_BACKFILL 1: the shadow's context/index.csv already covers the manifest (preflight checks it)
#   EPOCHS        passed to the jobs (smoke default 1)
#   TIME          passed as sbatch --time (e.g. stage A's r448 runs, re-timed from the smoke throughput)
#   PARTITION_ARGS extra sbatch arguments
#   MAX_JOBS      refuse an invocation above this many jobs (default 40)
#   SESSION_MAX_JOBS refuse when this invocation would take the jobs logged in context_jobs_*.txt past this (default
#                 40); ALLOW_OVER_SESSION_CAP=1 overrides, deliberately
#   JOBS_DIR      where context_jobs_<stage>.txt live (default: this clone's root)
#   DRY_RUN       1: print the sbatch lines, submit nothing, log nothing
#   PREFLIGHT     0 skips the preflight (tests only; never on klone)
#   ENV, HF_HOME  as in analysis/context_arms.slurm (preflight runs "$ENV/bin/python")
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${STAGE:?set STAGE (smoke|pilot|A|B)}"; DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
KEEP="${KEEP:-}"; PANO_STORE="${PANO_STORE:-}"; RUNS_SPEC="${RUNS_SPEC:-}"; WINNER="${WINNER:-}"; A5="${A5:-0}"
BACKFILL_JOB="${BACKFILL_JOB:-}"; SKIP_BACKFILL="${SKIP_BACKFILL:-0}"; TIME="${TIME:-}"
PARTITION_ARGS="${PARTITION_ARGS:-}"; MAX_JOBS="${MAX_JOBS:-40}"; SESSION_MAX_JOBS="${SESSION_MAX_JOBS:-40}"
ALLOW_OVER_SESSION_CAP="${ALLOW_OVER_SESSION_CAP:-0}"; JOBS_DIR="${JOBS_DIR:-$PWD}"; DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
U="${USER:-$(id -un)}"
ENV="${ENV:-/gscratch/makelab/$U/envs/sidewalk-validator-ai}"; HF_HOME="${HF_HOME:-/gscratch/makelab/$U/hf_cache}"
[ $# -gt 0 ] || { echo "usage: STAGE=... DATA=... RUNS=... $0 <LabelType> [...]" >&2; exit 2; }
die() { echo "context_submit: $*" >&2; exit 2; }
for v in DATA RUNS KEEP PANO_STORE; do
  [ -z "${!v}" ] || [[ "${!v}" = /* ]] || die "$v must be an absolute path (the jobs run from the clone root)"
done

if [ ! -f "$DATA/shadow_run.json" ] && [ "${ALLOW_SOURCE_DATA:-0}" != 1 ]; then
  die "$DATA has no shadow_run.json: jobs write their outputs into DATA, so point it at a shadow" \
      "(analysis/context_arms.py shadow), not at the round-3 directory"
fi
if [ "$STAGE" = smoke ]; then EPOCHS="${EPOCHS:-1}"; fi
EPOCHS="${EPOCHS:-}"

needs_context() { case "$1" in two_scale|view_heatmap) return 0 ;; *) return 1 ;; esac; }

preset() {
  local s="42,43,44"
  case "$1" in
    smoke) echo "baseline-rerun:42 two_scale:42 view_heatmap:42 view_heatmap@448:42 side_info:42 side_info_nohq:42" ;;
    pilot) echo "baseline-rerun:42,43,44,45,46 side_info:42 side_info_nohq:42" ;;
    A) [ "$A5" = 1 ] && s="42,43,44,45,46"; echo "two_scale:$s view_heatmap:$s view_heatmap@448:42" ;;
    B) [ -n "$WINNER" ] || die "stage B needs WINNER=<arm>"; [ "$A5" = 1 ] && s="42,43,44,45,46"
       echo "baseline-rerun:42,43,44,45,46 $WINNER:$s" ;;
    *) die "unknown STAGE=$1 (smoke|pilot|A|B)" ;;
  esac
}
if [ -n "$RUNS_SPEC" ]; then SPEC="$RUNS_SPEC"; else SPEC=$(preset "$STAGE") || exit 2; fi

# Expand the spec into one line per training: "<variant> <arm> <res> <seed> <tag>".
RUN_LINES=()
for item in $SPEC; do
  [[ "$item" == *:* ]] || die "bad run spec '$item' (variant[@res]:seed[,seed...])"
  vr="${item%%:*}"; seeds="${item#*:}"; variant="${vr%@*}"; res=224
  [[ "$vr" == *@* ]] && res="${vr#*@}"
  case "$variant" in baseline-rerun|two_scale|view_heatmap|side_info|side_info_nohq) ;; *) die "unknown variant '$variant'" ;; esac
  case "$res" in 224) ;; 448) [ "$variant" = view_heatmap ] || die "@448 is for view_heatmap only" ;; *) die "res must be 224 or 448" ;; esac
  arm="$variant"; [ "$variant" = baseline-rerun ] && arm=none
  for seed in ${seeds//,/ }; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "bad seed '$seed' in '$item'"
    tag="$variant"; [ "$res" = 448 ] && tag="$tag-r448"; [ "$seed" != 42 ] && tag="$tag-s$seed"
    RUN_LINES+=("$variant $arm $res $seed $tag")
  done
done
[ ${#RUN_LINES[@]} -gt 0 ] || die "empty run list"
DUP_TAGS=$(for l in "${RUN_LINES[@]}"; do echo "$l" | cut -d' ' -f5; done | sort | uniq -d | paste -sd' ' -)
[ -z "$DUP_TAGS" ] || die "tag(s) listed twice: $DUP_TAGS (two jobs would train into one run directory)"
DUP_TYPES=$(printf '%s\n' "$@" | sort | uniq -d | paste -sd' ' -)
[ -z "$DUP_TYPES" ] || die "label type(s) passed twice: $DUP_TYPES"
# The seed lists of a preset, for the preflight's comparison with DECISION_RULE (a RUNS_SPEC rerun is not compared).
C_SEEDS=""; B_SEEDS=""
if [ -z "$RUNS_SPEC" ] && [ "$STAGE" != smoke ]; then
  for l in "${RUN_LINES[@]}"; do
    read -r variant arm res seed tag <<< "$l"
    if [ "$res" = 224 ] && { [ "$variant" = two_scale ] || [ "$variant" = view_heatmap ]; }; then
      # each arm repeats the list; keep each seed once so the printed preflight command reads as the rule's list
      [[ ",$C_SEEDS," == *",$seed,"* ]] || C_SEEDS="${C_SEEDS:+$C_SEEDS,}$seed"
    elif [ "$variant" = baseline-rerun ]; then
      [[ ",$B_SEEDS," == *",$seed,"* ]] || B_SEEDS="${B_SEEDS:+$B_SEEDS,}$seed"
    fi
  done
fi

ARMS_IN_STAGE=$(for l in "${RUN_LINES[@]}"; do echo "$l" | cut -d' ' -f2; done | sort -u | paste -sd, -)
NEED_CTX=0
for l in "${RUN_LINES[@]}"; do needs_context "$(echo "$l" | cut -d' ' -f2)" && NEED_CTX=1; done
SUBMIT_BF=0
if [ "$NEED_CTX" = 1 ] && [ -z "$BACKFILL_JOB" ] && [ "$SKIP_BACKFILL" != 1 ]; then
  SUBMIT_BF=1
  [ -n "$PANO_STORE" ] || die "the image arms need a backfill: set PANO_STORE (or BACKFILL_JOB / SKIP_BACKFILL=1)"
fi
[ -z "$BACKFILL_JOB" ] || [ $# -eq 1 ] || die "BACKFILL_JOB attaches to one type's backfill: pass one label type"

PER_TYPE=$(( ${#RUN_LINES[@]} + SUBMIT_BF ))
TOTAL=$(( PER_TYPE * $# ))
ALREADY=$( (cat "$JOBS_DIR"/context_jobs_*.txt 2>/dev/null || true) | grep -cv '^#' || true)
echo "stage $STAGE: ${#RUN_LINES[@]} training(s)$([ "$SUBMIT_BF" = 1 ] && echo ' + 1 backfill') per type x $# type(s)" \
     "= $TOTAL job(s); $ALREADY already logged in $JOBS_DIR/context_jobs_*.txt"
for l in "${RUN_LINES[@]}"; do echo "  $l"; done
[ "$TOTAL" -le "$MAX_JOBS" ] || die "$TOTAL jobs is above MAX_JOBS=$MAX_JOBS"
if [ "$DRY_RUN" != 1 ] && [ $(( ALREADY + TOTAL )) -gt "$SESSION_MAX_JOBS" ] && [ "$ALLOW_OVER_SESSION_CAP" != 1 ]; then
  die "$ALREADY logged + $TOTAL now = $(( ALREADY + TOTAL )) jobs, above the session cap SESSION_MAX_JOBS=$SESSION_MAX_JOBS" \
      "(ALLOW_OVER_SESSION_CAP=1 overrides; ask first)"
fi

if [ "$DRY_RUN" = 1 ] && [ $(( ALREADY + TOTAL )) -gt "$SESSION_MAX_JOBS" ] && [ "$ALLOW_OVER_SESSION_CAP" != 1 ]; then
  echo "note: $ALREADY logged + $TOTAL now is above SESSION_MAX_JOBS=$SESSION_MAX_JOBS; the real submission needs" \
       "ALLOW_OVER_SESSION_CAP=1 (ask first)"
fi

# A job of the same type and tag already queued or running (any stage) would share its run directory.
if command -v squeue >/dev/null 2>&1; then
  QUEUED=$(squeue -h -u "$U" -o %j 2>/dev/null || true)
  for LT in "$@"; do
    for l in "${RUN_LINES[@]}"; do
      tag=$(echo "$l" | cut -d' ' -f5)
      if printf '%s\n' "$QUEUED" | grep -qxE "context-arms-launch-[^-]+-$LT-$tag"; then
        die "a job for $LT $tag is already queued or running (squeue); not submitting a second"
      fi
    done
  done
fi

RAWLABELS="${RAWLABELS:-}"
if [ -z "$RAWLABELS" ] && [ -f "$DATA/shadow_run.json" ]; then
  RAWLABELS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("rawlabels") or "")' "$DATA/shadow_run.json")
fi

if [ "$PREFLIGHT" = 1 ]; then
  pf=(--data "$DATA" --runs "$RUNS" --types "$(IFS=,; echo "$*")" --arms "$ARMS_IN_STAGE" --stage "$STAGE"
      --hf-home "$HF_HOME")
  [ -n "$KEEP" ] && pf+=(--keep "$KEEP")
  [ "$SUBMIT_BF" = 1 ] && pf+=(--backfill --pano-store "$PANO_STORE")
  [ "$NEED_CTX" = 1 ] && [ "$SKIP_BACKFILL" = 1 ] && pf+=(--skip-backfill)
  [ "$STAGE" != smoke ] && pf+=(--pair-october)
  [ -n "$C_SEEDS" ] && pf+=(--confirmatory-seeds "$C_SEEDS")
  [ -n "$B_SEEDS" ] && pf+=(--baseline-seeds "$B_SEEDS")
  echo "preflight: $ENV/bin/python analysis/context_arms.py preflight ${pf[*]}"
  "$ENV/bin/python" analysis/context_arms.py preflight "${pf[@]}" || die "preflight failed; nothing submitted"
fi

mkdir -p logs
SB_EXTRA=(--parsable --kill-on-invalid-dep=yes)
[ -n "$TIME" ] && SB_EXTRA+=(--time="$TIME")
# shellcheck disable=SC2206
[ -n "$PARTITION_ARGS" ] && SB_EXTRA+=($PARTITION_ARGS)
submit() {  # submit <tag-for-log> <LT> <env assignments...> -- <sbatch args...>; prints the job id
  local logtag="$1" lt="$2"; shift 2
  local vars=()
  while [ "$1" != -- ]; do vars+=("$1"); shift; done
  shift
  local cmd="${vars[*]} sbatch ${SB_EXTRA[*]} $*"
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY-RUN: $cmd" >&2
    echo "dry-$lt-$logtag"
    return 0
  fi
  local id
  id=$(env "${vars[@]}" sbatch "${SB_EXTRA[@]}" "$@") || {
    echo "context_submit: sbatch failed for $lt $logtag; STOPPING. Check squeue before resubmitting (a socket" \
         "timeout may still have queued it); nothing is retried." >&2
    return 1
  }
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$STAGE" "$lt" "$logtag" "$cmd" >> "$JOBS_DIR/context_jobs_$STAGE.txt"
  echo "$id"
}

common=(DATA="$DATA" KEEP="$KEEP" ENV="$ENV" HF_HOME="$HF_HOME")
[ -n "$EPOCHS" ] && common+=(EPOCHS="$EPOCHS")
for LT in "$@"; do
  [ -f "$DATA/$LT/split.csv" ] || die "no $DATA/$LT/split.csv"
  bf="$BACKFILL_JOB"
  if [ "$SUBMIT_BF" = 1 ]; then
    bf=$(submit backfill "$LT" STEP=backfill LT="$LT" "${common[@]}" PANO_STORE="$PANO_STORE" RAWLABELS="$RAWLABELS" \
         -- --job-name="context-arms-launch-$STAGE-bf-$LT" analysis/context_arms.slurm) || exit 1
  fi
  ids=()
  for l in "${RUN_LINES[@]}"; do
    read -r variant arm res seed tag <<< "$l"
    dep=()
    if needs_context "$arm" && [ -n "$bf" ]; then dep=(--dependency="afterok:$bf"); fi
    id=$(submit "$tag" "$LT" STEP=train LT="$LT" ARM="$arm" SEED="$seed" TAG="$tag" RES="$res" RUNS="$RUNS" \
         "${common[@]}" -- --job-name="context-arms-launch-$STAGE-$LT-$tag" ${dep[@]+"${dep[@]}"} \
         analysis/context_arms.slurm) || exit 1
    ids+=("$tag=$id")
  done
  echo "$LT backfill=${bf:-none} ${ids[*]}"
done
[ "$DRY_RUN" = 1 ] && echo "DRY_RUN=1: nothing submitted, nothing logged"
exit 0
