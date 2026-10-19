#!/bin/bash
#SBATCH --job-name=fine_tune_classifier
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

# Step 6. Reads <DATA>/<label_type>/split.csv; writes <DATA>/<label_type>/dinov2/ with checkpoints, split.csv,
# deployment_priors.json and training_run.json. Three label types run concurrently, then two; the job fails if any
# of them fails.
set -uo pipefail
DATA=.
SEED=42
EPOCHS=25
# --num-workers 4: three label types run at once on 16 CPUs, so 3 x (4 workers + 1 main) fit.

run_batch() {
    declare -A pids
    for LT in "$@"; do
        python -u fine_tune_classifier.py --label_type="$LT" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS --num-workers 4 &
        pids[$LT]=$!
    done
    local failed=0
    for LT in "$@"; do
        wait "${pids[$LT]}" || { echo "fine_tune_classifier.py failed for $LT" >&2; failed=1; }
    done
    return $failed
}

run_batch Crosswalk CurbRamp Obstacle || exit 1
run_batch NoCurbRamp SurfaceProblem || exit 1
