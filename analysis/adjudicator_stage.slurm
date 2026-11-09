#!/bin/bash
#SBATCH --job-name=sva-adjudicator-stage
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
# VLM adjudicator (analysis/adjudicator.py crops, issue #77): pull the panos of RUN/stage_files.txt from the makelab2 pano store
# (read-only rrsync key) into a staged store copy, one ssh connection per 1000-file batch, sequentially: the makelab
# hosts ban a source that reconnects too fast, so a connection error or hang STOPS the job with no retry. rsync skips
# files already staged, so a resubmit resumes. The .pano-store marker is written only after the pull.
#   RUN=/gscratch/scrubbed/$USER/adjudicator-2026-11/crops sbatch analysis/adjudicator_stage.slurm
# (RUN holds stage_files.txt from `adjudicator.py crops`; for a pool run, size it with `stage-list` first.)
set -uo pipefail
RUN="${RUN:-/gscratch/scrubbed/$USER/adjudicator-2026-11/crops}"
DEST="${DEST:-/gscratch/makelab/$USER/pano-stage/adjudicator}"
SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $HOME/.ssh/pano_store_ro -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
LIST=$RUN/stage_files.txt
B=$RUN/stage_batches
mkdir -p "$DEST" "$B"
rm -f "$DEST/.pano-store" "$B"/b*
split -l 1000 -d -a 3 "$LIST" "$B/b"
for b in "$B"/b*; do
  echo "$(date +%T) $b ($(wc -l < "$b") files)"
  timeout 3600 rsync -e "$SSH" -a --timeout=300 --files-from="$b" jonf@makelab2.cs.washington.edu:/ "$DEST/"
  rc=$?
  # 23 = some listed files missing on the store (recorded as no_pano by the crops stage). Anything else stops.
  if [ $rc -ne 0 ] && [ $rc -ne 23 ]; then echo "STOP: rsync exit $rc on $b; not retrying"; exit $rc; fi
  echo "$(date +%T) rc=$rc; staged $(find "$DEST" -name "*.jpg" | wc -l); $(du -sh "$DEST" | cut -f1)"
done
comm -23 <(sort "$LIST") <(cd "$DEST" && find . -name "*.jpg" | sed "s|^\./||" | sort) > "$RUN/stage_missing.txt"
echo "want $(wc -l < "$LIST"), have $(find "$DEST" -name "*.jpg" | wc -l), missing $(wc -l < "$RUN/stage_missing.txt")"
timeout 120 rsync -e "$SSH" -a jonf@makelab2.cs.washington.edu:/.pano-store "$DEST/" && echo "marker written"
