#!/bin/bash
# Issue #53, NOT RUN YET (Jon launches): the CurbRamp pilot of training on SidewalkWebpage's crop-job windows.
#
#   1. data_generation.py --job-window-crops on round 3's rawLabels cache, seed, EXPERTS and --include-ai-labels, so
#      the selection should be round 3's. It can still differ: the selection code moved since round 3 (#72 keeps
#      Mapillary/Panoramax labels; on the round-3 cache none of its 15k Mapillary CurbRamps meets the crowd vote rule,
#      measured 2026-11-05, so that one should not bite), so a differing selection_sha256 is printed as a warning,
#      not a stop; the pairing is guaranteed by step 2 instead;
#   2. analysis/job_crop_compare.py restrict: round 3's split restricted to the rows that got a job crop, each row's
#      split inherited (split_data.py is not re-run); restrict_summary.json says how many round-3 rows were lost and
#      why (not_selected, dims_mismatch, out_of_frame);
#   3. fine_tune_classifier.py (seed 42, 25 epochs, round 3's recipe) and evaluate.py on that split, then
#      analysis/job_crop_compare.py compare against round 3's predictions on the same rows.
#
#   ROUND3=/gscratch/makelab/$USER/retrain-2026-10 CACHE=<round 3 rawlabels cache> \
#   PANO_STORE=<the store or staged copy round 3 read> OUT=/gscratch/scrubbed/$USER/job-crops-2026-11 \
#   EXPERTS="<name>=<user_id>,..." sbatch job_crop.slurm
#
# EXPERTS must be round 3's value (it is part of the selection); it is never written into the repo. Restartable: ckpt
# may preempt and requeue; step 1 resumes (ok rows are skipped), step 2 rewrites the same split, step 3 resumes from
# the last complete checkpoint. Outputs stay under $OUT (private: crops and per-label rows); only the aggregate tables
# that compare writes to $OUT/report are meant for docs/job-window-crops-2026-11/ (check them for deployment names
# first: they are per provider and pooled, so there should be none).
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=sva-job-crops
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
ROUND3="${ROUND3:?set ROUND3 (round 3 data_generation --out, holding CurbRamp/split.csv and predictions.csv)}"
CACHE="${CACHE:?set CACHE (round 3 rawlabels cache)}"; PANO_STORE="${PANO_STORE:?set PANO_STORE}"
OUT="${OUT:?set OUT (absolute, on scratch)}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
LT=CurbRamp; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-25}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
[ -n "${EXPERTS:-}" ] || echo "WARNING: EXPERTS is not set; round 3 used it (2 accounts), so fewer round-3 rows will be selected." >&2
[ -f "$PANO_STORE/.pano-store" ] || { echo "PANO_STORE=$PANO_STORE has no .pano-store marker (mounted?)" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. Job-window crops (no depth model; the GPU is for step 3).
echo "$(date) data_generation --job-window-crops -> $OUT"
python -u data_generation.py --label_type "$LT" --out "$OUT" --rawlabels-cache "$CACHE" --pano-store "$PANO_STORE" \
    --job-window-crops --include-ai-labels --tilt-beta 0.95 --seed "$SEED" || { echo "data_generation failed" >&2; exit 1; }
python - "$ROUND3/$LT/generation_run.json" "$OUT/$LT/generation_run.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["selection_sha256"] for p in sys.argv[1:])
print(f"selection {'matches' if a == b else 'DIFFERS FROM'} round 3 ({a[:12]} vs {b[:12]}); restrict pairs the rows")
PY

# 2. Round 3's split, restricted to rows with a job crop.
python -u analysis/job_crop_compare.py restrict --split "$ROUND3/$LT/split.csv" --manifest "$OUT/$LT/manifest.csv" \
    --out "$OUT/$LT/split.csv" || { echo "restrict failed" >&2; exit 1; }

# 3. Train, evaluate, compare (round 3's model is not retrained; it is scored on the same rows).
python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$OUT" --seed "$SEED" --epochs "$EPOCHS" \
    --num-workers 7 || { echo "training failed" >&2; exit 1; }
BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$OUT/$LT/dinov2")
python -u evaluate.py --label_type "$LT" --data-dir "$OUT" --checkpoint "$BEST" --per-city \
    || { echo "evaluate failed" >&2; exit 1; }
python -u analysis/job_crop_compare.py compare --job-pred "$OUT/$LT/predictions.csv" \
    --depth-pred "$ROUND3/$LT/predictions.csv" --split "$OUT/$LT/split.csv" --out-dir "$OUT/report" \
    || { echo "compare failed" >&2; exit 1; }
echo "$(date) done: $OUT/report/tables.md"
