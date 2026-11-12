#!/bin/bash
#SBATCH --job-name=depth-agreement
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue
# Issue #21, step 3 of the klone recipe (docs/depth-ablation-2026-11.md, "Reproduce"): ViT-B (re-run, the R0 parity
# control) and ViT-L depth at every sampled label, rendered exactly as round 3 rendered it (read_store_panorama's
# 8192 resize, the torch renderer, crop_pose's pose), then Google's depth along each crop's centre ray.
# NOT RUN in the PR that adds it. Both steps are resumable, so a ckpt-all requeue is harmless.
#
# Run directory layout ($W, scrubbed scratch; nothing is written under /gscratch/makelab):
#   $W/sidewalk-validator-ai/   clone of the depth-agreement branch, with the Depth-Anything-V2 submodule, and both
#                               checkpoints downloaded into it on the LOGIN node (compute nodes cannot resolve the
#                               HF Xet CDN):
#     wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/e96fd0c8cedf1825a55b9c61309f754982948df0/depth_anything_v2_metric_vkitti_vitb.pth
#     wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/$VITL_REV/depth_anything_v2_metric_vkitti_vitl.pth
#     sha256sum depth_anything_v2_metric_vkitti_vit{b,l}.pth      # record both in the doc
#   $W/sample.csv               from `depth_agreement.py sample` (copied up from ~/Data)
#   $W/store/                   from depth_agreement_stage.slurm (the depth artifacts)
#
#   cd $W && sbatch --export=ALL,VITL_REV=<40-hex commit>,VITL_SHA256=<64-hex> \
#       $W/sidewalk-validator-ai/analysis/depth_agreement.slurm
#
# The job REFUSES to start until the ViT-L checkpoint is pinned: VITL_REV must be the 40-hex commit of the HF repo the
# file was downloaded from (never a branch name such as "main") and VITL_SHA256 the checkpoint's sha256, which the
# job verifies against the file. Once both are known, write them into this file as the defaults (TODO below).
#
# Round 3's panoramas are read through each row's own pano_path store root (read only). If they are gone, stage
# them with `sample --stage-panos` and set PANO_STORE=$W/store when submitting.
set -euo pipefail
W=$(pwd -P)
# Only ever write under this study's scrubbed directory: never /gscratch/makelab, never another study's directory.
# The physical path (symlinks resolved) must be <scrubbed>/<one user dir>/depth-ablation-2026-11 or a child of it.
[[ "$W" =~ ^(/mmfs1)?/gscratch/scrubbed/[^/]+/depth-ablation-2026-11(/.*)?$ ]] || { echo "refusing to write under $W"; exit 2; }
# TODO(#21): pin the ViT-L revision. Read it from the HF model API on the login node
# (https://huggingface.co/api/models/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large), download the checkpoint
# at that commit, then replace the two empty defaults below with the commit and the file's sha256, put the commit in
# docs/running-on-hyak.md, and record both in docs/depth-ablation-2026-11.md. Until then the job stops here.
VITL_REV=${VITL_REV:-}
VITL_SHA256=${VITL_SHA256:-}
[[ "$VITL_REV" =~ ^[0-9a-f]{40}$ ]] || { echo "refusing to run: VITL_REV is not pinned to a 40-hex commit (got '${VITL_REV:-unset}'); see the TODO(#21) in this file"; exit 2; }
[[ "$VITL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "refusing to run: VITL_SHA256 is not set to the ViT-L checkpoint's sha256; see the TODO(#21) in this file"; exit 2; }
# A real write, not touch: scrubbed has refused data writes while a 0-byte touch succeeded (running-on-hyak.md).
if ! dd if=/dev/zero of="$W/.write_test.$$" bs=1M count=8 status=none; then echo "refusing to run: cannot write 8 MiB under $W"; exit 2; fi
rm -f "$W/.write_test.$$"
module load conda
eval "$(conda shell.bash hook)"
conda activate "/gscratch/makelab/$USER/envs/sidewalk-validator-ai"
export HF_HOME="/gscratch/makelab/$USER/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$W/sidewalk-validator-ai"
echo "commit $(git rev-parse HEAD) vitl_rev $VITL_REV"
sha256sum depth_anything_v2_metric_vkitti_vitb.pth depth_anything_v2_metric_vkitti_vitl.pth
echo "$VITL_SHA256  depth_anything_v2_metric_vkitti_vitl.pth" | sha256sum --check --status || { echo "refusing to run: depth_anything_v2_metric_vkitti_vitl.pth does not have sha256 $VITL_SHA256"; exit 2; }
mkdir -p "$W/out"
python analysis/depth_agreement.py depth-pass --sample "$W/sample.csv" --out "$W/out/depths.csv" --encoders vitb,vitl \
    ${PANO_STORE:+--pano-store "$PANO_STORE"}
python analysis/depth_agreement.py google-depth --sample "$W/sample.csv" --store "$W/store" --out "$W/out/google.csv"
echo "$(date +%T) done: $(($(wc -l < "$W/out/depths.csv") - 1)) depth rows, $(($(wc -l < "$W/out/google.csv") - 1)) google rows"
