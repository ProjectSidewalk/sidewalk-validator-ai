#!/bin/bash
# Issue #74: submit the round-4 recipe on round-3 data on klone, from this stream's own clone of the pushed branch
# (never the shared clone), with logs/ present. Three stages (docs/round4-2026-11.md):
#
#   STAGE=smoke  analysis/round4_submit.sh        # 1 CPU + 2 GPU jobs: prep via the CPU job file, a 1-epoch cl-drop
#                                                 # training with KEEP_BEST, gold scoring of it and of round 3
#   STAGE=smoke-rerun analysis/round4_submit.sh   # 1 GPU job: the smoke training again; must exit "already evaluated"
#   STAGE=pilot  analysis/round4_submit.sh        # 1 CPU prep, 27 trainings (gold-free baseline x 5 types and
#                                                 # cl-drop x 4 types, seeds 42/43/44), 2 scoring jobs, 1 CPU collect
#   STAGE=stage2 CONFIRM_STAGE2=1 analysis/round4_submit.sh   # Jon's call: uniform gold crops (makelab2 pull), uniform
#                                                 # scoring, the rebalanced rule test, the VLM adjudicator
#
# Every job id is appended to $WORK/round4_jobs.txt with its kind (gpu or cpu), the time, the job file's sha256 and
# the full command line (the job's variables travel in the environment, as in analysis/cl_submit.sh, and are logged
# in front of the sbatch call). MAX_GPU_JOBS (default 40) caps this stream's GPU submissions across all stages: the
# launcher counts the gpu lines already logged and refuses a batch that would pass the cap. PREFLIGHT=0 skips the
# login-node checks (for the stub test only) and says so.
set -euo pipefail
STAGE="${STAGE:?set STAGE (smoke|smoke-rerun|pilot|stage2)}"
SCRUB="${SCRUB:-/gscratch/scrubbed/$USER}"
WORK="${WORK:-$SCRUB/round4-disagree-2026-11}"
DATA="${DATA:-/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10}"        # round 3, read-only
RAWLABELS="${RAWLABELS:-/gscratch/makelab/$USER/sidewalk-validator-ai/rawlabels}"    # read-only
KEEP="${KEEP:-/gscratch/makelab/$USER/round4-disagree-2026-11-keep}"                 # small artifacts only
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"                     # used read-only
HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}"
R4="$WORK/data"; RUNS="$WORK/runs"; SMOKE="$WORK/smoke"; GOLD_UIDS="$WORK/gold_uids.csv"
JOBS="$WORK/round4_jobs.txt"; SUBMITTED="$WORK/submitted"
TYPES="${TYPES:-Crosswalk,CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem}"
CL_TYPES="${CL_TYPES:-Crosswalk,CurbRamp,NoCurbRamp,Obstacle}"
SEEDS="${SEEDS:-42 43 44}"
MAX_GPU_JOBS="${MAX_GPU_JOBS:-40}"; MAX_PARALLEL="${MAX_PARALLEL:-16}"
PILOT_TAGS="baseline,baseline-s43,baseline-s44,cl-drop,cl-drop-s43,cl-drop-s44"
die() { echo "round4_submit: $*" >&2; exit 2; }
# WORK (and so R4, RUNS, SMOKE) never under /tmp (node-local on klone) or the makelab fileset (no room). The list is
# overridable only so the stub test can run where its temporary directory lives; the default is what klone uses.
FORBIDDEN_PREFIXES="${ROUND4_FORBIDDEN_PREFIXES:-/tmp /gscratch/makelab}"
for f in $FORBIDDEN_PREFIXES; do
  case "$WORK/" in "$f"/*) die "refusing WORK=$WORK: under $f (/tmp is node-local, makelab has no room)" ;; esac
done
[ -d logs ] || mkdir -p logs
mkdir -p "$WORK" "$SUBMITTED"

gpu_used() { [ -f "$JOBS" ] && awk -F'\t' '$2 == "gpu"' "$JOBS" | wc -l | tr -d ' ' || echo 0; }
need_gpu() {  # need_gpu <n>: refuse a batch that would pass MAX_GPU_JOBS
  local used; used=$(gpu_used)
  if [ $((used + $1)) -gt "$MAX_GPU_JOBS" ]; then
    die "$STAGE needs $1 GPU jobs; $used already submitted (round4_jobs.txt), cap MAX_GPU_JOBS=$MAX_GPU_JOBS"
  fi
}
# submit <gpu|cpu> "<VAR=value ...>" <sbatch args ...> <job file>; prints the job id.
submit() {
  local kind=$1 envs=$2 id file sha; shift 2
  file="${*: -1}"
  sha=$({ sha256sum "$file" 2>/dev/null || shasum -a 256 "$file"; } | cut -d' ' -f1)
  cp -n "$file" "$SUBMITTED/$(basename "$file").$sha" 2>/dev/null || true
  # shellcheck disable=SC2086
  id=$(env $envs sbatch --parsable --kill-on-invalid-dep=yes "$@") || die "sbatch failed: $envs sbatch $*"
  id="${id%%;*}"
  printf '%s\t%s\t%s\tsha256=%s\t%s sbatch --parsable --kill-on-invalid-dep=yes %s\n' \
    "$id" "$kind" "$(date -u +%FT%TZ)" "$sha" "$envs" "$*" >> "$JOBS"
  echo "$id"
}

preflight() {
  if [ "${PREFLIGHT:-1}" = 0 ]; then echo "preflight skipped (PREFLIGHT=0)"; return 0; fi
  module load conda >/dev/null 2>&1 || true
  eval "$(conda shell.bash hook)"; conda activate "$ENV"
  echo "== git: clean checkout at a pushed commit"
  [ -z "$(timeout 60 git status --porcelain --untracked-files=no)" ] || die "uncommitted changes in $PWD"
  timeout 120 git fetch -q origin || die "git fetch failed"
  local br; br=$(git rev-parse --abbrev-ref HEAD)
  timeout 30 git merge-base --is-ancestor HEAD "origin/$br" || die "HEAD $(git rev-parse --short HEAD) is not pushed"
  echo "HEAD $(git rev-parse HEAD) on $br"
  echo "== inputs"
  [ -f "$GOLD_UIDS" ] || die "no $GOLD_UIDS (round4.py gold-uids on the Mac, then scp)"
  [ -d "$RAWLABELS" ] || die "no $RAWLABELS"
  local T
  for T in ${TYPES//,/ }; do
    [ -f "$DATA/$T/split.csv" ] || die "no $DATA/$T/split.csv"
    timeout 60 python analysis/round4.py round3-ckpt --dinov2 "$DATA/$T/dinov2" || die "round-3 checkpoint of $T"
  done
  for T in ${CL_TYPES//,/ }; do
    for f in noisy_targets.csv flag_summary.json; do [ -f "$DATA/$T/cl/$f" ] || die "no $DATA/$T/cl/$f"; done
  done
  echo "== a real 8 MiB write (a touch passes on a full quota)"
  local d
  for d in "$WORK" "$KEEP"; do
    mkdir -p "$d"
    timeout 120 dd if=/dev/zero of="$d/.round4-ddtest" bs=1M count=8 status=none || die "cannot write 8 MiB to $d"
    rm -f "$d/.round4-ddtest"
  done
  echo "== storage"
  timeout 120 hyakstorage /gscratch/makelab | tee "$WORK/hyakstorage_makelab.txt"
  python analysis/round4.py storage --name /gscratch/makelab --min-gb 5 --min-files 2000 \
      < "$WORK/hyakstorage_makelab.txt" || die "makelab nearly full"
  timeout 120 hyakstorage /gscratch/scrubbed || true
  echo "== Hugging Face cache (offline)"
  HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 timeout 120 python -c \
    "from huggingface_hub import snapshot_download as s; print(s('facebook/dinov2-large', local_files_only=True))" \
    || { echo "on the login node: HF_HUB_DISABLE_XET=1 HF_HOME=$HF_HOME huggingface-cli download facebook/dinov2-large"
         die "facebook/dinov2-large is not in $HF_HOME"; }
  timeout 60 python -c "import pandas; assert pandas.__version__.startswith('2.'), pandas.__version__" \
    || die "the env must keep pandas 2.x"
  timeout 30 scontrol show reservation || true
}

cpu_env() { echo "DATA=$DATA GOLD_UIDS=$GOLD_UIDS RAWLABELS=$RAWLABELS ENV=$ENV"; }
gpu_env() { echo "ENV=$ENV HF_HOME=$HF_HOME"; }

case "$STAGE" in
  smoke)
    need_gpu 2
    preflight
    prep=$(submit cpu "STEP=prep R4=$SMOKE/data TYPES=Crosswalk CL_TYPES=Crosswalk $(cpu_env)" \
           --job-name=round4-disagree-smoke-prep analysis/round4_cpu.slurm)
    train=$(submit gpu "LT=Crosswalk VARIANT=cl-drop TAG=smoke EPOCHS=1 KEEP_CKPT=0 KEEP_BEST=1 DATA=$SMOKE/data RUNS=$SMOKE/runs $(gpu_env)" \
            --job-name=round4-disagree-smoke-train --time=01:00:00 --dependency="afterok:$prep" \
            analysis/confident_learning.slurm)
    score=$(submit gpu "MODE=round3,runs TYPES=Crosswalk TAGS=smoke R4=$SMOKE/data RUNS=$SMOKE/runs DATA=$DATA $(gpu_env)" \
            --job-name=round4-disagree-smoke-score --dependency="afterok:$train" analysis/round4_score.slurm)
    echo "smoke prep=$prep train=$train score=$score; then: python analysis/round4.py verify-smoke --smoke $SMOKE"
    ;;
  smoke-rerun)
    need_gpu 1
    preflight
    rerun=$(submit gpu "LT=Crosswalk VARIANT=cl-drop TAG=smoke EPOCHS=1 KEEP_CKPT=0 KEEP_BEST=1 DATA=$SMOKE/data RUNS=$SMOKE/runs $(gpu_env)" \
            --job-name=round4-disagree-smoke-train --time=01:00:00 analysis/confident_learning.slurm)
    echo "smoke rerun=$rerun (expect 'already evaluated' within two minutes)"
    ;;
  pilot)
    n_cl=$(echo "${CL_TYPES//,/ }" | wc -w); n_t=$(echo "${TYPES//,/ }" | wc -w); n_s=$(echo "$SEEDS" | wc -w)
    need_gpu $(( (n_t + n_cl) * n_s + 2 ))
    preflight
    prep=$(submit cpu "STEP=prep R4=$R4 TYPES=$TYPES CL_TYPES=$CL_TYPES $(cpu_env)" \
           --job-name=round4-disagree-prep analysis/round4_cpu.slurm)
    trains=()
    for seed in $SEEDS; do                       # seed 42 first
      for v in baseline cl-drop; do
        for T in ${TYPES//,/ }; do
          if [ "$v" = cl-drop ]; then case ",$CL_TYPES," in *",$T,"*) ;; *) continue ;; esac; fi
          tag="$v"; [ "$seed" = 42 ] || tag="$v-s$seed"
          dep="afterok:$prep"
          i=${#trains[@]}
          [ "$i" -lt "$MAX_PARALLEL" ] || dep="$dep,afterany:${trains[$((i - MAX_PARALLEL))]}"   # throttle
          trains+=("$(submit gpu "LT=$T VARIANT=$v TAG=$tag SEED=$seed KEEP_CKPT=0 KEEP_BEST=1 DATA=$R4 RUNS=$RUNS $(gpu_env)" \
                      --job-name="round4-disagree-$T-$tag" --dependency="$dep" analysis/confident_learning.slurm)")
        done
      done
    done
    all=$(IFS=:; echo "${trains[*]}")
    r3=$(submit gpu "MODE=round3 CROPS=r3crops TYPES=$TYPES R4=$R4 DATA=$DATA $(gpu_env)" \
         --job-name=round4-disagree-score-round3 --dependency="afterok:$prep" analysis/round4_score.slurm)
    sr=$(submit gpu "MODE=runs CROPS=r3crops TYPES=$TYPES CL_TYPES=$CL_TYPES TAGS=$PILOT_TAGS R4=$R4 RUNS=$RUNS DATA=$DATA $(gpu_env)" \
         --job-name=round4-disagree-score-runs --dependency="afterany:$all" analysis/round4_score.slurm)
    col=$(submit cpu "STEP=collect R4=$R4 RUNS=$RUNS KEEP=$KEEP TYPES=$TYPES JOBS=$JOBS SUBMITTED=$SUBMITTED EXPECT=baseline:42,43,44+cl-drop:42,43,44:$CL_TYPES $(cpu_env)" \
          --job-name=round4-disagree-collect --dependency="afterany:$all:$r3:$sr" analysis/round4_cpu.slurm)
    echo "pilot prep=$prep trainings=${trains[*]} score-round3=$r3 score-runs=$sr collect=$col"
    ;;
  stage2)
    [ "${CONFIRM_STAGE2:-0}" = 1 ] || die "stage 2 is Jon's call (docs/round4-2026-11.md, J4); set CONFIRM_STAGE2=1"
    RB_TYPES="${RB_TYPES:-CurbRamp,Crosswalk}"
    ADJ="${ADJ:-$WORK/adj}"; SIF="${SIF:-$SCRUB/containers/vllm-openai_v0.32.0-x86_64.sif}"
    n_rb=$(echo "${RB_TYPES//,/ }" | wc -w); n_s=$(echo "$SEEDS" | wc -w)
    need_gpu $(( 2 * n_rb * n_s + 2 + 4 ))
    preflight
    if [ "${PREFLIGHT:-1}" != 0 ]; then
      echo "== stage-2 inputs"
      [ -f "$ADJ/gold.csv" ] || die "no $ADJ/gold.csv (J14: Jon copies it, chmod 700 on $ADJ)"
      [ -f depth_anything_v2_metric_vkitti_vitb.pth ] || die "no depth weights in $PWD (the loader reads the working directory)"
      [ -d Depth-Anything-V2/depth_anything_v2 ] || die "no Depth-Anything-V2 submodule in $PWD (git submodule update --init)"
      [ -f "$SIF" ] || die "no vLLM image $SIF (build it as docs/adjudicator-2026-11.md says)"
      [ -x "$SCRUB/envs/vllm-client/bin/python" ] || die "no $SCRUB/envs/vllm-client"
      for repo in zai-org/GLM-5.3-Flash google/gemma-4-31B-it Qwen/Qwen3-VL-32B-Instruct; do
        HF_HOME="$SCRUB/hf" HF_HUB_OFFLINE=1 timeout 120 python -c \
          "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], local_files_only=True)" "$repo" \
          || { echo "on the login node: HF_HUB_DISABLE_XET=1 HF_HOME=$SCRUB/hf huggingface-cli download $repo"
               die "$repo is not in $SCRUB/hf"; }
      done
      for T in ${TYPES//,/ }; do for tag in ${PILOT_TAGS//,/ }; do
        case "$tag" in cl-*) case ",$CL_TYPES," in *",$T,"*) ;; *) continue ;; esac ;; esac
        [ -f "$RUNS/$T/$tag/best/model.safetensors" ] || die "no $RUNS/$T/$tag/best (purged from scrubbed?)"
      done; done
    fi
    timeout 600 python analysis/adjudicator.py crops --all --gold "$ADJ/gold.csv" --rawlabels "$RAWLABELS" \
        --out-dir "$ADJ/crops" || die "adjudicator.py crops failed"
    # The panoramas go to scrubbed (about 10 MB each), not to the makelab fileset, which has no room.
    STORE="$SCRUB/pano-stage/round4-disagree"
    pull=$(submit cpu "RUN=$ADJ/crops DEST=$STORE" --job-name=round4-disagree-pano-pull analysis/adjudicator_stage.slurm)
    crops=$(submit gpu "STAGE=crops CODE=$PWD OUT=$ADJ DATA=$DATA PANO_STORE=$STORE" --job-name=round4-disagree-crops --gpus=1 \
            --constraint="a40|l40|l40s|a100|h200" --mem=64G --cpus-per-task=8 --time=8:00:00 \
            --dependency="afterok:$pull" analysis/adjudicator.slurm)
    man=$(submit cpu "STEP=gold-manifest CROPS=uniform CROPS_ROOT=$ADJ/crops R4=$R4 TYPES=$TYPES $(cpu_env)" \
          --job-name=round4-disagree-gold-manifest --dependency="afterok:$crops" analysis/round4_cpu.slurm)
    su=$(submit gpu "MODE=round3,runs CROPS=uniform TYPES=$TYPES CL_TYPES=$CL_TYPES TAGS=$PILOT_TAGS R4=$R4 RUNS=$RUNS DATA=$DATA $(gpu_env)" \
         --job-name=round4-disagree-score-uniform --dependency="afterok:$man" analysis/round4_score.slurm)
    var=$(submit cpu "STEP=variants WHICH=symmetric-rb,matched-rb SEEDS=${SEEDS// /,} CL_TYPES=$RB_TYPES R4=$R4 $(cpu_env)" \
          --job-name=round4-disagree-rb-variants analysis/round4_cpu.slurm)
    rb=()
    for seed in $SEEDS; do for v in symmetric-rb matched-rb; do for T in ${RB_TYPES//,/ }; do
      name="$v"; [ "$seed" = 42 ] || name="$v-s$seed"
      rb+=("$(submit gpu "LT=$T VARIANT=$name TAG=$name SEED=$seed KEEP_CKPT=0 KEEP_BEST=1 DATA=$R4 RUNS=$RUNS $(gpu_env)" \
              --job-name="round4-disagree-$T-$name" --dependency="afterok:$var" analysis/confident_learning.slurm)")
    done; done; done
    smoke=$(submit gpu "STAGE=score MODEL=gemma4-31b LIMIT=8 CODE=$PWD OUT=$ADJ DATA=$DATA" \
            --job-name=round4-disagree-adj-smoke --gpus=l40s:2 --mem=200G --cpus-per-task=16 --time=1:00:00 \
            --dependency="afterok:$crops" analysis/adjudicator.slurm)
    adj=()
    for M in glm5.3-flash gemma4-31b qwen3vl-32b; do
      case "$M" in
        glm5.3-flash) RES=(--gpus=h200:4 --mem=400G --cpus-per-task=32 --time=8:00:00) ;;
        *)            RES=(--gpus=l40s:2 --mem=200G --cpus-per-task=16 --time=6:00:00) ;;
      esac
      adj+=("$(submit gpu "STAGE=score MODEL=$M CODE=$PWD OUT=$ADJ DATA=$DATA" --job-name="round4-disagree-adj-$M" \
               "${RES[@]}" --dependency="afterok:$smoke" analysis/adjudicator.slurm)")
    done
    echo "stage2 pull=$pull crops=$crops manifest=$man score=$su variants=$var rb=${rb[*]} adj-smoke=$smoke adj=${adj[*]}"
    ;;
  *) die "unknown STAGE $STAGE (smoke|smoke-rerun|pilot|stage2)" ;;
esac
