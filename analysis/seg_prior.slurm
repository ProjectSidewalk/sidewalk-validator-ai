#!/bin/bash
#SBATCH --job-name=seg-prior-render
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
# Issue #44, step 2 of 2 on klone: render every sample label's 2048x2048 view exactly as data_generation.py did,
# check the re-cut crop against the training crop, and segment the view (analysis/sidewalk_seg.py render-segment).
#
# Run directory layout ($W, scrubbed scratch; nothing goes under /gscratch/makelab):
#   $W/sidewalk-validator-ai/   clone of the seg-prior branch (data_generation importable; Depth-Anything not needed)
#   $W/labels.csv               from `sidewalk_seg.py filelist`
#   $W/panos/                   from seg_prior_stage.slurm
#   $W/crops/<LabelType>/...    the sample's training crops, rsynced up from the Mac (for the check only)
#   $W/hf/                      HF_HOME; the segmenter is downloaded there on the LOGIN node first, because compute
#                               nodes cannot resolve the HF Xet CDN:
#       HF_HUB_DISABLE_XET=1 HF_HOME=$W/hf python -c "from huggingface_hub import snapshot_download as s; \
#           s('facebook/mask2former-swin-large-mapillary-vistas-semantic', revision='$SEG_REV', \
#           allow_patterns=['*.json', 'model.safetensors'])"
#
#   cd $W && sbatch $W/sidewalk-validator-ai/analysis/seg_prior.slurm
#
# Reads (never writes) the retrain's rawLabels exports for pano_x/pano_y and the pano-extra fallback store.
set -euo pipefail
W=$(pwd)
# Only ever write under the scrubbed scratch dir: refuse any other working directory (never /gscratch/makelab).
case "$W/" in /gscratch/scrubbed/*/seg-prior-2026-10/*|/mmfs1/gscratch/scrubbed/*/seg-prior-2026-10/*) ;; *) echo "refusing to write under $W"; exit 2;; esac
SEG_REV=4772b6bf101d91f2534c106dc524d906aeb3c68a
RAWLABELS=${RAWLABELS:-/gscratch/makelab/$USER/sidewalk-validator-ai/rawlabels}
EXTRA=${EXTRA:-/gscratch/makelab/$USER/pano-extra}
module load conda
eval "$(conda shell.bash hook)"
conda activate /gscratch/makelab/$USER/envs/sidewalk-validator-ai
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $W/sidewalk-validator-ai
echo "commit $(git rev-parse HEAD)"
python analysis/sidewalk_seg.py render-segment --labels $W/labels.csv --store $W/panos --extra-store $EXTRA \
    --rawlabels-dir $RAWLABELS --crops-dir $W/crops --out-dir $W/out --seg-revision $SEG_REV
