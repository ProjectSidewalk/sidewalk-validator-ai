#!/bin/bash
# Submit the cross-region chain (issues #33, #52 option 3, #15) for the given label types on klone, from a checkout of
# this branch with the conda env active (the variants step runs here, on the login node: CPU, seconds) and logs/
# present:
#
#   DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/region-2026-11 \
#     CITIES_JSON=$DATA/rawlabels/cities.json SEEDS=42 \
#     analysis/region_submit.sh CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem
#
# Per label type: variants (foreground) -> per eligible country and seed: global -> ft at each fraction (first seed
# only, FT_SEEDS) -> cleanup of the global weights (after every ft succeeded); cond (independent); per seed condall at
# the country and the region level; calibrate (CPU, after every global and ft of the first seed). Slurm dependencies
# cannot branch on file content, which is why variants runs here and its eligible list drives the submissions.
# A requeued (preempted) job keeps its id, so the dependencies survive preemption. Job ids are appended to
# region_jobs.txt in the submit directory. Seeds 43 and 44 (SEEDS=42,43,44) add global, cond and condall only.
# Optional: PARTITION_ARGS for the GPU jobs (e.g. "--partition=gpu-l40s --constraint=l40s"), FT_SEEDS (default the
# first of SEEDS), LEVELS (default "country region" for condall), FT_EPOCHS (default 10), REGION_OVERRIDES (a private
# JSON kept outside the repo: {"_regions": {"<country>": "<region>"}} for countries whose deployments are all
# private, which regions.py does not list; exported to every job; without it, at the region level, they share the
# <unk> row with every unseen country, in training too; country-level conditioning is unaffected). Optional:
# RAWLABELS (default $DATA/rawlabels: the cached rawLabels exports <city>.csv, for the 3 m merge of the fine-tune
# pools, because split.csv has no coordinates).
set -euo pipefail
DATA="${DATA:?set DATA (absolute)}"; RUNS="${RUNS:?set RUNS}"; CITIES_JSON="${CITIES_JSON:?set CITIES_JSON}"
SEEDS="${SEEDS:-42}"; FT_SEEDS="${FT_SEEDS:-${SEEDS%%,*}}"; LEVELS="${LEVELS:-country region}"
FT_EPOCHS="${FT_EPOCHS:-10}"; PARTITION_ARGS="${PARTITION_ARGS:-}"; RAWLABELS="${RAWLABELS:-$DATA/rawlabels}"
[ $# -gt 0 ] || { echo "usage: $0 <LabelType> [...]" >&2; exit 2; }
[ -f "$CITIES_JSON" ] || { echo "no $CITIES_JSON (compute nodes have no network; save the listing first)" >&2; exit 2; }
[ -d "$RAWLABELS" ] || { echo "no $RAWLABELS (the cached rawLabels exports; set RAWLABELS)" >&2; exit 2; }
python -c "import pandas" 2>/dev/null || { echo "activate the conda env first (python with pandas)" >&2; exit 2; }
[ -d logs ] || mkdir -p logs
# --kill-on-invalid-dep: a dependent whose prerequisite failed is cancelled instead of pending forever.
sb() { sbatch --parsable --kill-on-invalid-dep=yes "$@"; }
common() { echo "LT=$LT,DATA=$DATA,RUNS=$RUNS,CITIES_JSON=$CITIES_JSON,FT_EPOCHS=$FT_EPOCHS"; }
for LT in "$@"; do
  [ -f "$DATA/$LT/split.csv" ] || { echo "no $DATA/$LT/split.csv" >&2; exit 2; }
  python -u analysis/region_transfer.py variants --split "$DATA/$LT/split.csv" --cities-json "$CITIES_JSON" \
      --rawlabels "$RAWLABELS" --out "$DATA/$LT/region"
  # evaluate.py reads <data-dir>/<LT>/split.csv (every row, one-to-one). Linked here, once, before any GPU job starts,
  # so no job relinks it while another job's evaluate.py is reading it.
  mkdir -p "$DATA/$LT/region/eval/$LT"
  ln -sfn "$DATA/$LT/split.csv" "$DATA/$LT/region/eval/$LT/split.csv"
  calib_deps=()
  while read -r COUNTRY PCTS; do
    [ -n "$COUNTRY" ] || continue
    for SEED in ${SEEDS//,/ }; do
      g=$(sb $PARTITION_ARGS --job-name="rg-$LT-global-$COUNTRY-s$SEED" \
            --export="ALL,$(common),ARM=global,COUNTRY=$COUNTRY,SEED=$SEED" analysis/region_transfer.slurm)
      c=$(sb $PARTITION_ARGS --job-name="rg-$LT-cond-$COUNTRY-s$SEED" \
            --export="ALL,$(common),ARM=cond,COUNTRY=$COUNTRY,SEED=$SEED,LEVEL=country" analysis/region_transfer.slurm)
      fts=()
      if [[ ",$FT_SEEDS," == *",$SEED,"* ]]; then
        calib_deps+=("$g")
        for PCT in $PCTS; do
          fts+=("$(sb $PARTITION_ARGS --job-name="rg-$LT-ft-$COUNTRY-f$PCT-s$SEED" --dependency="afterok:$g" \
                     --export="ALL,$(common),ARM=ft,COUNTRY=$COUNTRY,FRAC=$PCT,SEED=$SEED" analysis/region_transfer.slurm)")
        done
        calib_deps+=("${fts[@]}")
      fi
      dep="afterok:$g"; [ ${#fts[@]} -gt 0 ] && dep="afterok:$(IFS=:; echo "${fts[*]}")"
      cl=$(sb --job-name="rg-$LT-cleanup-$COUNTRY-s$SEED" --dependency="$dep" \
             --export="ALL,STEP=cleanup,LT=$LT,RUNS=$RUNS,COUNTRY=$COUNTRY,SEED=$SEED" analysis/region_transfer_cpu.slurm)
      echo "$LT $COUNTRY s$SEED global=$g cond=$c ft=${fts[*]:-} cleanup=$cl" | tee -a region_jobs.txt
    done
  done < <(python analysis/region_transfer.py list --variants "$DATA/$LT/region/variants.json")
  for SEED in ${SEEDS//,/ }; do
    for LEVEL in $LEVELS; do
      j=$(sb $PARTITION_ARGS --job-name="rg-$LT-condall-$LEVEL-s$SEED" \
            --export="ALL,$(common),ARM=condall,LEVEL=$LEVEL,SEED=$SEED" analysis/region_transfer.slurm)
      echo "$LT condall-$LEVEL s$SEED $j" | tee -a region_jobs.txt
    done
  done
  if [ ${#calib_deps[@]} -gt 0 ]; then
    k=$(sb --job-name="rg-$LT-calibrate" --dependency="afterany:$(IFS=:; echo "${calib_deps[*]}")" \
          --export="ALL,STEP=calibrate,LT=$LT,DATA=$DATA,CITIES_JSON=$CITIES_JSON,SEEDS=$FT_SEEDS" \
          analysis/region_transfer_cpu.slurm)
    echo "$LT calibrate $k" | tee -a region_jobs.txt
  fi
done
echo "report (on the Mac, from a pulled copy of \$DATA): python analysis/region_transfer.py report --data-dir <copy> \\"
echo "    --cities-json <copy>/rawlabels/cities.json --seeds $SEEDS --jobs-file region_jobs.txt"
