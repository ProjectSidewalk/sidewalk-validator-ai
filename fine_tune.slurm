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
# deployment_priors.json and training_run.json.
DATA=.
SEED=42
EPOCHS=25

python -u fine_tune_classifier.py --label_type="Crosswalk" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS &
python -u fine_tune_classifier.py --label_type="CurbRamp" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS &
python -u fine_tune_classifier.py --label_type="Obstacle" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS &

wait

python -u fine_tune_classifier.py --label_type="NoCurbRamp" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS &
python -u fine_tune_classifier.py --label_type="SurfaceProblem" --data-dir "$DATA" --seed $SEED --epochs $EPOCHS &

wait
