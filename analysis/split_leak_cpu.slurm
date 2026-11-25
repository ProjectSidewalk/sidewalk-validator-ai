#!/bin/bash
# Issue #3, the CPU steps of the split-leak study (docs/split-leak-2026-11.md; analysis/split_leak_submit.sh submits):
#   STEP=stage0  DATA=... RUNS=... RAWLABELS=...  : make-split (one leaky split per seed and type, after proving the
#                grouped split reproduces round 3's byte for byte), check (coupling, exposure census, design.json), bound
#                (the three planning figures) -> $RUNS/report/{design,bound}.json, bound.md
#   STEP=report  DATA=... RUNS=... SEEDS=42:43:44 G_SEEDS=42:43:44 G_SOURCE=stored|fresh
#                -> $RUNS/report/{tables,seeds,slices}.md, results.json
# Lists are colon-separated (sbatch --export splits on commas). HOLDOUT (optional, comma list, environment only, never
# a file) overrides the held-out deployments make-split otherwise reads off the production split.
# Both steps are deterministic; a rerun rewrites the same files.
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-leak-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STEP="${STEP:?set STEP (stage0|report)}"; DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
TYPES="${TYPES:-CurbRamp:NoCurbRamp:Obstacle:SurfaceProblem:Crosswalk}"
SEEDS="${SEEDS:-42:43:44}"; G_SEEDS="${G_SEEDS:-42:43:44}"; G_SOURCE="${G_SOURCE:-stored}"
REPORT="$RUNS/report"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
case "$STEP" in
  stage0)
    RAWLABELS="${RAWLABELS:?set RAWLABELS (the rawLabels cache round 3 was split with)}"
    HOLD=(); [ -n "${HOLDOUT:-}" ] && HOLD=(--holdout-cities "$HOLDOUT")
    python -u analysis/split_leak.py make-split --data "$DATA" --runs "$RUNS" --types ${TYPES//:/ } \
        --seeds ${SEEDS//:/ } --rawlabels "$RAWLABELS" ${HOLD[@]+"${HOLD[@]}"}
    python -u analysis/split_leak.py check --data "$DATA" --runs "$RUNS" --types ${TYPES//:/ } \
        --seeds ${SEEDS//:/ } --rawlabels "$RAWLABELS" --out "$REPORT"
    python -u analysis/split_leak.py bound --data "$DATA" --runs "$RUNS" --types ${TYPES//:/ } \
        --rawlabels "$RAWLABELS" --g-seeds ${G_SEEDS//:/ } --out "$REPORT"
    ;;
  report)
    python -u analysis/split_leak.py report --data "$DATA" --runs "$RUNS" --types ${TYPES//:/ } \
        --seeds ${SEEDS//:/ } --g-seeds ${G_SEEDS//:/ } --g-source "$G_SOURCE" --out "$REPORT"
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "$(date) done $STEP -> $REPORT"
