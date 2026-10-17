#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=200:00:00

# Steps 1-5 for all five label types. Writes <OUT>/<label_type>/{correct,incorrect}/*.webp, manifest.csv,
# generation_run.json and split.csv (hard joined from $HARD), plus the shared <OUT>/deployment_priors.json.
# Needs CUDA (no CPU fallback) and depth_anything_v2_metric_vkitti_vitb.pth (the checkpoint the API serves) in the
# repo root. Resumable: resubmitting skips labels whose manifest row is "ok", whose crop exists and which are still
# selected with the same class under the same settings. If the selection or a setting changed since the first run,
# the job stops and reports the stale rows; add --force-reselect to accept.
set -uo pipefail

OUT=.
CACHE="$OUT/rawlabels"            # one rawLabels pull, shared by the five label-type processes and by resubmissions
FRESH="${FRESH:-0}"               # FRESH=1 sbatch data_generation.slurm  re-pulls rawLabels (a new selection snapshot)
HARD=analysis/hard_examples_2026-10-17.csv
# Optional held-out cities for split_data.py (whole cities to test_heldout), e.g. HOLDOUT="teaneck-nj,taipei"
HOLDOUT=""
LABEL_TYPES=(Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem)

# 1. Pull rawLabels once, only if there is no cache yet (or FRESH=1). Reusing it on resubmission keeps the
#    selection identical, so a resumed run is not refused for new votes that arrived in between.
if [ "$FRESH" = "1" ] || [ ! -d "$CACHE" ]; then
    rm -rf "$CACHE.tmp" && mkdir -p "$CACHE.tmp"
    python -u - "$CACHE.tmp" <<'PY' || { echo "rawLabels pull failed" >&2; exit 1; }
import sys, data_generation as dg
for city_id, sub in dg.CITIES.items():
    dg.load_city(city_id, sub).drop(columns="city").to_csv(f"{sys.argv[1]}/{city_id}.csv", index=False)
    print("pulled", city_id)
PY
    rm -rf "$CACHE" && mv "$CACHE.tmp" "$CACHE"   # only a complete pull replaces the cache
else
    echo "Using the existing rawLabels cache $CACHE (FRESH=1 to re-pull)."
fi

# 2. Five label types in parallel, reading the cache; wait on every PID and stop on any failure.
declare -A PIDS
for LT in "${LABEL_TYPES[@]}"; do
    python -u data_generation.py --label_type="$LT" --out "$OUT" --rawlabels-cache "$CACHE" \
        --depth-encoder vitb --tilt-beta 0.95 --seed 42 &
    PIDS[$LT]=$!
done
FAILED=0
for LT in "${LABEL_TYPES[@]}"; do
    if ! wait "${PIDS[$LT]}"; then
        echo "data_generation.py failed for $LT" >&2
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "Not running split_data.py: fix the failure and resubmit (the run resumes)." >&2
    exit 1
fi

# 3. Split (cheap; never touches images).
for LT in "${LABEL_TYPES[@]}"; do
    python -u split_data.py --label_type="$LT" --out "$OUT" --seed 42 --holdout-cities "$HOLDOUT" \
        --hard-examples "$HARD" || exit 1
done
