#!/bin/bash
#SBATCH --job-name=get_stats
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

python -u get_stats.py --label_type="Crosswalk" & 
python -u get_stats.py --label_type="CurbRamp" &
python -u get_stats.py --label_type="Obstacle" &

wait

python -u get_stats.py --label_type="NoCurbRamp" &
python -u get_stats.py --label_type="SurfaceProblem" &

wait
