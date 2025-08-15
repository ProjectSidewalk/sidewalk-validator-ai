#!/bin/bash
#SBATCH --job-name=fine_tune_classifier
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

python -u fine_tune_classifier.py --label_type="Crosswalk" & 
python -u fine_tune_classifier.py --label_type="CurbRamp" &
python -u fine_tune_classifier.py --label_type="Obstacle" &

wait

python -u fine_tune_classifier.py --label_type="NoCurbRamp" &
python -u fine_tune_classifier.py --label_type="SurfaceProblem" &

wait
