#!/bin/bash
#SBATCH --job-name=seg-prior-stage
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
# Issue #44, step 1 of 2 on klone: stage the sample's panoramas from the makelab2 pano store.
#
# Submit from the run directory ($W below), which holds stage_files.txt (from `sidewalk_seg.py filelist`: one
# store-relative path <city>/<pano_id[:2]>/<pano_id>.jpg per line, 768 for the 2026-10-21 sample):
#
#   W=/gscratch/scrubbed/jfroehli/seg-prior-2026-10
#   cd $W && sbatch $W/sidewalk-validator-ai/analysis/seg_prior_stage.slurm
#
# The recipe is the retrain's (full_stage.slurm): a read-only rrsync key, ONE rsync connection for the whole list
# (the makelab hosts ban an IP that reconnects quickly, and that ban hits every session on the network). Exit 23
# (some listed files missing on the store) is expected for a few panos; the render job falls back to the read-only
# pano-extra copy for those. Any other failure or a timeout stops the job: do NOT resubmit without checking why.
# --no-requeue: ckpt-all requeues preempted jobs, and a requeue would open a second rsync connection.
# No second connection is made for the store's marker: the job writes its own .pano-store marker after the pull,
# saying it is a staged copy (data_generation.read_store_panorama only needs the marker to exist).
set -uo pipefail
W=$(pwd)
# Only ever write under the scrubbed scratch dir: refuse any other working directory (never /gscratch/makelab).
case "$W/" in /gscratch/scrubbed/jfroehli/*|/mmfs1/gscratch/scrubbed/jfroehli/*) ;; *) echo "refusing to write under $W"; exit 2;; esac
LIST=$W/stage_files.txt
DEST=$W/panos
SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $HOME/.ssh/pano_store_ro -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
test -s "$LIST" || { echo "no $LIST"; exit 2; }
mkdir -p "$DEST"
rm -f "$DEST/.pano-store"
echo "$(date +%T) pulling $(wc -l < "$LIST") files"
timeout 3600 rsync -e "$SSH" -a --timeout=300 --files-from="$LIST" jonf@makelab2.cs.washington.edu:/ "$DEST/"
rc=$?
if [ $rc -ne 0 ] && [ $rc -ne 23 ]; then echo "STOP: rsync exit $rc; not retrying"; exit $rc; fi
comm -23 <(sort "$LIST") <(cd "$DEST" && find . -name "*.jpg" | sed "s|^\./||" | sort) > "$W/stage_missing.txt"
echo "$(date +%T) rc=$rc; want $(wc -l < "$LIST"), have $(find "$DEST" -name "*.jpg" | wc -l), missing $(wc -l < "$W/stage_missing.txt"); $(du -sh "$DEST" | cut -f1)"
echo "staged copy of the Project Sidewalk pano store for seg-prior (issue 44), pulled $(date -Iseconds) job ${SLURM_JOB_ID:-?}" > "$DEST/.pano-store"
