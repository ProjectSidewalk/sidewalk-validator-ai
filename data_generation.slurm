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
# Needs CUDA (no CPU fallback), depth_anything_v2_metric_vkitti_vitb.pth (the checkpoint the API serves) in the
# repo root, and PANO_STORE: either the pano store itself or a staged copy of it that keeps the store's layout,
# <PANO_STORE>/<city_id>/<pano_id[:2]>/<pano_id>.jpg (what sidewalk-ai-api reads as SIDEWALK_SCRAPES_DIR), with a
# .pano-store marker file at its root (a staged copy writes its own only after a verified pull). Panoramas are read
# only from there; none is downloaded. Every selected city needs a directory (data_generation.py
# --allow-missing-cities to run without some). Resumable: resubmitting skips labels whose manifest row is "ok", whose
# crop exists and which are still selected with the same class under the same settings. If the selection or a
# setting changed since the first run, the job stops and reports the stale rows; add --force-reselect to accept.
set -uo pipefail

OUT=.
# PANO_STORE=/path/to/Panoramas sbatch data_generation.slurm
PANO_STORE="${PANO_STORE:?set PANO_STORE to the pano store root}"
# One rawLabels pull, shared by the five label-type processes and by resubmissions.
CACHE="$OUT/rawlabels"
# FRESH=1 sbatch data_generation.slurm  re-pulls rawLabels (a new selection snapshot).
FRESH="${FRESH:-0}"
HARD=analysis/hard_examples_2026-10-17.csv
# Optional held-out cities for split_data.py (whole cities to test_heldout), e.g. HOLDOUT="teaneck-nj,taipei"
HOLDOUT=""
# Panorama groups with labels within this many metres are merged before splitting (issue 27); 0 = panorama only.
# MERGE_RADIUS_M=0 sbatch data_generation.slurm
MERGE_RADIUS_M="${MERGE_RADIUS_M:-3}"
LABEL_TYPES=(Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem)

# 0. Check the store before spending time on the pull (data_generation.py checks it again, with the city directories).
[ -d "$PANO_STORE" ] && [ -f "$PANO_STORE/.pano-store" ] || {
    echo "PANO_STORE=$PANO_STORE is not a pano store (no directory, or no .pano-store marker; is it mounted?)" >&2
    exit 1
}

# 1. Pull rawLabels once, only if there is no cache yet (or FRESH=1). Reusing it on resubmission keeps the
#    selection identical, so a resumed run is not refused for new votes that arrived in between. The pull first
#    saves the /v3/api/cities listing as $CACHE/cities.json and then pulls every public deployment it lists; the
#    five processes read that file, so the city list is fixed by the pull too and step 2 makes no API call.
# A cache without its .complete marker (an interrupted pull from before the atomic move) counts as absent.
if [ "$FRESH" != "1" ] && [ -f "$CACHE/.complete" ] && [ ! -f "$CACHE/cities.json" ]; then
    echo "$CACHE predates city discovery (issue 8): it has no cities.json, so its city list is the old hardcoded" \
         "one. FRESH=1 re-pulls every public deployment (a new selection); to resume a run made from that cache," \
         "use the commit that made it." >&2
    exit 1
fi
if [ "$FRESH" = "1" ] || [ ! -f "$CACHE/.complete" ]; then
    rm -rf "$CACHE.tmp" && mkdir -p "$CACHE.tmp"
    python -u - "$CACHE.tmp" <<'PY' || { echo "rawLabels pull failed" >&2; exit 1; }
import sys, data_generation as dg
listing, source = dg.load_city_listing(sys.argv[1], refresh=True)  # writes cities.json into the new cache
cities = dg.discover_cities(listing)
print(f"{len(cities)} public deployments in /v3/api/cities ({source['fetched_on']})")
for city_id, sub in cities.items():
    dg.load_city(city_id, sub).drop(columns="city").to_csv(f"{sys.argv[1]}/{city_id}.csv", index=False)
    print("pulled", city_id)
PY
    rm -rf "$CACHE" && mv "$CACHE.tmp" "$CACHE" && touch "$CACHE/.complete"   # only a complete pull replaces it
else
    echo "Using the existing rawLabels cache $CACHE (FRESH=1 to re-pull)."
fi

# 2. Five label types in parallel, reading the cache; wait on every PID and stop on any failure.
declare -A PIDS
for LT in "${LABEL_TYPES[@]}"; do
    python -u data_generation.py --label_type="$LT" --out "$OUT" --rawlabels-cache "$CACHE" \
        --pano-store "$PANO_STORE" --depth-encoder vitb --tilt-beta 0.95 --seed 42 &
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

# 3. Split (cheap; never touches images). Label coordinates for the merge come from the rawLabels cache.
for LT in "${LABEL_TYPES[@]}"; do
    python -u split_data.py --label_type="$LT" --out "$OUT" --seed 42 --holdout-cities "$HOLDOUT" \
        --hard-examples "$HARD" --rawlabels "$CACHE" --merge-radius-m "$MERGE_RADIUS_M" || exit 1
done
