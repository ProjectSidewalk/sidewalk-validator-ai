#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

# Step 7 and 7b, run from the data root (evaluate.py reads ./<label_type>/split.csv and ./<label_type>/dinov2/).
# 7: predictions.csv + metrics for test / test_heldout / hard.  7b: the served map, fitted on val under the
# deployment prior that data_generation.py wrote to ./deployment_priors.json.

python -u evaluate.py --label_type="Crosswalk" &
python -u evaluate.py --label_type="CurbRamp" &
python -u evaluate.py --label_type="Obstacle" &

wait

python -u evaluate.py --label_type="NoCurbRamp" &
python -u evaluate.py --label_type="SurfaceProblem" &

wait

for LT in Crosswalk CurbRamp NoCurbRamp Obstacle SurfaceProblem; do
    lower=$(echo "$LT" | tr '[:upper:]' '[:lower:]')
    python -u -m calibration.calibrate --predictions "$LT/predictions.csv" --priors deployment_priors.json \
        --label-type "$LT" --out "$LT/calibration/$lower.json"
done
