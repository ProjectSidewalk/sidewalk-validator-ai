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
# selected with the same class. If the selection changed since the first run (new votes), the job stops and reports
# the stale rows; add --force-reselect to accept.
OUT=.
HARD=analysis/hard_examples_2026-10-17.csv
# Optional held-out cities for split_data.py (whole cities to test_heldout), e.g. HOLDOUT="teaneck-nj,taipei"
HOLDOUT=""

for LT in Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem; do
    python -u data_generation.py --label_type="$LT" --out "$OUT" --depth-encoder vitb --tilt-beta 0.95 --seed 42 &
done

wait

for LT in Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem; do
    python -u split_data.py --label_type="$LT" --out "$OUT" --seed 42 --holdout-cities "$HOLDOUT" --hard-examples "$HARD"
done
