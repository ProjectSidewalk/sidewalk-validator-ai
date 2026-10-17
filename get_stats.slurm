#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

# Step 7 and 7b, from the data root (evaluate.py reads ./<label_type>/split.csv and ./<label_type>/dinov2/).
# 7: predictions.csv + metrics for test / test_heldout / hard.  7b: the served map, fitted on val under the
# deployment prior that data_generation.py wrote to ./deployment_priors.json. The job fails if any step fails.
set -uo pipefail
DATA=.

run_batch() {
    declare -A pids
    for LT in "$@"; do
        python -u evaluate.py --label_type="$LT" --data-dir "$DATA" &
        pids[$LT]=$!
    done
    local failed=0
    for LT in "$@"; do
        wait "${pids[$LT]}" || { echo "evaluate.py failed for $LT" >&2; failed=1; }
    done
    return $failed
}

run_batch Crosswalk CurbRamp Obstacle || exit 1
run_batch NoCurbRamp SurfaceProblem || exit 1

for LT in Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem; do
    lower=$(echo "$LT" | tr '[:upper:]' '[:lower:]')
    python -u -m calibration.calibrate --predictions "$DATA/$LT/predictions.csv" --priors "$DATA/deployment_priors.json" \
        --label-type "$LT" --out "$DATA/$LT/calibration/$lower.json" || { echo "calibrate failed for $LT" >&2; exit 1; }
done
