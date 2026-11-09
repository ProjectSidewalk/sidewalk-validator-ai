#!/bin/bash
# Submit the adjudicator's vLLM jobs on klone (issue #77), one per model, from a checkout with logs/ present:
#
#   STAGE=score analysis/adjudicator_submit.sh glm5.3-flash gemma4-31b qwen3vl-32b
#   STAGE=score LIMIT=8 analysis/adjudicator_submit.sh gemma4-31b                     # the smoke run first (Q4)
#   STAGE=relabel ROWS=train TYPES=CurbRamp,Crosswalk analysis/adjudicator_submit.sh glm5.3-flash
#   STAGE=crops analysis/adjudicator_submit.sh                                        # the crop leg (1 GPU)
#
# Optional AFTER=<jobid> (e.g. the crops job) makes every job wait for it (afterok). Resources follow the preset
# table in docs/adjudicator-2026-11.md; everything else (DATA, OUT, LIMIT, REASON, ...) passes through the
# environment to analysis/adjudicator.slurm. Job ids are appended to adjudicator_jobs.txt in the submit directory.
set -euo pipefail
STAGE="${STAGE:-score}"
[ -d logs ] || mkdir -p logs
DEP=(); [ -n "${AFTER:-}" ] && DEP=(--dependency="afterok:$AFTER")
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
sb() { sbatch --parsable --kill-on-invalid-dep=yes "${DEP[@]}" "$@"; }
if [ "$STAGE" = crops ]; then
  id=$(STAGE=crops sb --job-name=adj-crops --gpus=1 --constraint="a40|l40|l40s|a100|h200" --mem=64G \
       --cpus-per-task=8 --time=8:00:00 analysis/adjudicator.slurm)
  echo "crops $id" | tee -a adjudicator_jobs.txt
  exit 0
fi
[ $# -gt 0 ] || { echo "usage: STAGE=score|relabel $0 <model preset> [...]" >&2; exit 2; }
for M in "$@"; do
  case "$M" in
    glm5.3-flash) RES=(--gpus=h200:4 --mem=400G --cpus-per-task=32 --time=8:00:00) ;;
    gemma4-31b)   RES=(--gpus=l40s:2 --mem=200G --cpus-per-task=16 --time=6:00:00) ;;
    qwen3vl-32b)  RES=(--gpus=l40s:2 --mem=200G --cpus-per-task=16 --time=6:00:00) ;;
    *) echo "unknown preset $M (glm5.3-flash, gemma4-31b, qwen3vl-32b)" >&2; exit 2 ;;
  esac
  id=$(MODEL="$M" STAGE="$STAGE" sb --job-name="adj-$STAGE-$M" "${RES[@]}" analysis/adjudicator.slurm)
  echo "$STAGE $M $id" | tee -a adjudicator_jobs.txt
done
