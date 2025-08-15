#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=200:00:00

python -u data_generation.py --label_type="Crosswalk" & 
python -u data_generation.py --label_type="CurbRamp" &
python -u data_generation.py --label_type="NoCurbRamp" &
python -u data_generation.py --label_type="Obstacle" &
python -u data_generation.py --label_type="SurfaceProblem" &

wait

python -u split_data.py --label_type="Crosswalk" 
python -u split_data.py --label_type="CurbRamp"
python -u split_data.py --label_type="NoCurbRamp"
python -u split_data.py --label_type="Obstacle"
python -u split_data.py --label_type="SurfaceProblem"