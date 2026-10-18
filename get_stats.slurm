#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

python -u evaluate.py --label_type="Crosswalk" & 
python -u evaluate.py --label_type="CurbRamp" &
python -u evaluate.py --label_type="Obstacle" &

wait

python -u evaluate.py --label_type="NoCurbRamp" &
python -u evaluate.py --label_type="SurfaceProblem" &

wait
