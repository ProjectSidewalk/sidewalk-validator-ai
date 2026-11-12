#!/bin/bash
#SBATCH --job-name=depth-agreement-stage
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
# Issue #21, step 2 of the klone recipe (docs/depth-ablation-2026-11.md, "Reproduce"): stage the sample's Google
# depth artifacts (<city>/<id[:2]>/<id>.depth.npz) from the makelab2 pano store. NOT RUN in the PR that adds it.
#
# Submit from the run directory ($W), which holds stage_files.txt from `depth_agreement.py sample` (one
# store-relative path per line; with --stage-panos the panorama JPEGs too):
#
#   W=/gscratch/scrubbed/$USER/depth-ablation-2026-11
#   cd $W && sbatch $W/sidewalk-validator-ai/analysis/depth_agreement_stage.slurm
#
# The recipe is seg_prior_stage.slurm's: a read-only rrsync key and ONE rsync connection for the whole list (the
# makelab hosts ban an IP that reconnects quickly, and that ban hits every session on the network). Exit 23 (some
# listed files missing on the store) is EXPECTED: an absent artifact is the coverage measurement. Any other failure
# or a timeout stops the job: do NOT resubmit without checking why, and tell Jon.
# --no-requeue: ckpt-all requeues preempted jobs, and a requeue would open a second rsync connection.
# The job writes its own .pano-store marker after the pull, saying it is a staged copy.
set -uo pipefail
W=$(pwd)
# Only ever write under this study's scrubbed directory: never /gscratch/makelab, never another study's directory.
case "$W/" in /gscratch/scrubbed/*/depth-ablation-2026-11/*|/mmfs1/gscratch/scrubbed/*/depth-ablation-2026-11/*) ;; *) echo "refusing to write under $W"; exit 2;; esac
LIST=$W/stage_files.txt
DEST=$W/store
SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $HOME/.ssh/pano_store_ro -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
test -s "$LIST" || { echo "no $LIST"; exit 2; }
mkdir -p "$DEST"
rm -f "$DEST/.pano-store"
echo "$(date +%T) pulling $(wc -l < "$LIST") files"
timeout 3000 rsync -e "$SSH" -a --timeout=300 --files-from="$LIST" jonf@makelab2.cs.washington.edu:/ "$DEST/"
rc=$?
if [ $rc -ne 0 ] && [ $rc -ne 23 ]; then echo "STOP: rsync exit $rc; not retrying"; exit $rc; fi
comm -23 <(sort "$LIST") <(cd "$DEST" && find . -type f \( -name "*.depth.npz" -o -name "*.jpg" \) | sed "s|^\./||" | sort) > "$W/stage_missing.txt"
echo "$(date +%T) rc=$rc; want $(wc -l < "$LIST"), have $(find "$DEST" -type f \( -name "*.depth.npz" -o -name "*.jpg" \) | wc -l), missing $(wc -l < "$W/stage_missing.txt"); $(du -sh "$DEST" | cut -f1)"
echo "staged copy of the Project Sidewalk pano store for depth-agreement (issue 21), pulled $(date -Iseconds) job ${SLURM_JOB_ID:-?}" > "$DEST/.pano-store"
