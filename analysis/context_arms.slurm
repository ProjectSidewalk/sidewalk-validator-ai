#!/bin/bash
# Issue #14: one GPU job per step of the context-arms experiment on klone (docs/context-arms-2026-11.md).
# analysis/context_submit.sh submits whole stages (smoke, pilot, A, B); this file runs one job.
#
#   STEP=backfill: write <DATA>/<LT>/context/ (the wide crop and the 448 px view of every ok row) from the pano store,
#                  re-rendered with the pose the crops were cut under; --verify 50 re-cuts 50 tight crops and fails the
#                  job if they drift from the stored WebPs. Crops, manifest and split.csv are not touched. Resumable.
#                  Holds an exclusive flock on <DATA>/<LT>/context/.backfill.lock for the whole step: a second backfill
#                  of the same type exits at once instead of rewriting the index under the first.
#   STEP=train:    train one run on the round-3 split with the round-3 recipe (--target hard --best-metric macro_f1,
#                  passed explicitly), score it, then evaluate.py --tag $TAG --per-city (outputs in <DATA>/<LT>/).
#                  ARM=none is the tight-crop baseline rerun (scored by evaluate.py --checkpoint); any other ARM is a
#                  context arm (context_model.py predict, then evaluate.py --skip-inference). After the evaluation and
#                  BEFORE any checkpoint is deleted, the run record (training_run.json, run_key.json, the best
#                  checkpoint's trainer_state.json and config.json, job.txt) is copied to <DATA>/<LT>/runs/$TAG/.
#                  Restartable: ckpt may preempt and requeue, and fine_tune_classifier.py resumes from its last complete
#                  checkpoint (run_key.json guards the split, the arm, the view resolution and the batch size). A job
#                  that finished (marker $RUNS/$LT/$TAG/.evaluated) exits at once when requeued or resubmitted.
#                  A TIMEOUT is NOT requeued by Slurm: resubmit the same command and it resumes from the last checkpoint.
#
#   STEP=train LT=NoCurbRamp ARM=none TAG=baseline-rerun DATA=$WORK/data RUNS=$WORK/runs sbatch analysis/context_arms.slurm
#   STEP=train LT=NoCurbRamp ARM=side_info_nohq SEED=43 TAG=side_info_nohq-s43 DATA=$WORK/data RUNS=$WORK/runs \
#       sbatch analysis/context_arms.slurm
#   STEP=backfill LT=NoCurbRamp DATA=$WORK/data PANO_STORE=/abs/store sbatch analysis/context_arms.slurm
#
# Variables (${VAR:?} = required):
#   STEP        backfill | train
#   LT          label type
#   DATA        the shadow data directory (analysis/context_arms.py shadow): <LT>/split.csv, <LT>/manifest.csv,
#               symlinked crops; every output of this job lands under it, never in the round-3 directory
#   PANO_STORE  (backfill) the pano store the crops were cut from, with its .pano-store marker
#   RAWLABELS   (backfill) the rawLabels snapshot the dataset was built from (default: "rawlabels" in
#               $DATA/shadow_run.json, else $DATA/../rawlabels)
#   ARM         (train) none (the baseline rerun) | two_scale | view_heatmap | side_info | side_info_nohq
#   RUNS        (train) checkpoint root on scratch (about 3.5 GB per checkpoint, two kept while training)
#   SEED        (train) default 42
#   TAG         (train) evaluate.py tag and run name; default $ARM, required for ARM=none. Refused: empty, none,
#               baseline and baseline-s<seed> (those are October's files in the shadow and must not be overwritten)
#   RES         (train) --context-view-res, default 224 (only view_heatmap reads it; 448 with TAG=view_heatmap-r448)
#   BATCH       (train) per-device batch, default 8 (gradient accumulation keeps the effective batch at 32)
#   KEEP_CKPT   (train) 0 (default) deletes every checkpoint after the run record is copied; best keeps only the best
#               checkpoint's model.safetensors, config.json and preprocessor_config.json; 1 keeps everything (debugging)
#   KEEP        (train, optional) a durable directory: the run's small artifacts (metrics, predictions, per-city,
#               report, slices, run record; no weights, no figures) are copied to $KEEP/$LT/. A failed copy warns only.
#   EPOCHS      (train) default 25 (the smoke stage uses 1)
#   ENV         conda env (default /gscratch/makelab/$USER/envs/sidewalk-validator-ai; read-only, never installed into)
#   HF_HOME     HF cache holding facebook/dinov2-large (default /gscratch/makelab/$USER/hf_cache); jobs run offline
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=context-arms-launch
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
STEP="${STEP:?set STEP (backfill|train)}"; LT="${LT:?set LT}"; DATA="${DATA:?set DATA (absolute)}"
# Argument checks come first, before any module or GPU work, so a bad submission fails in a second.
if [ "$STEP" = train ]; then
  ARM="${ARM:?set ARM (none|two_scale|view_heatmap|side_info|side_info_nohq)}"; RUNS="${RUNS:?set RUNS}"
  case "$ARM" in none|two_scale|view_heatmap|side_info|side_info_nohq) ;; *) echo "unknown ARM=$ARM" >&2; exit 2 ;; esac
  TAG="${TAG:-$ARM}"
  if [ -z "$TAG" ] || [ "$TAG" = none ] || [[ "$TAG" =~ ^baseline(-s[0-9]+)?$ ]]; then
    echo "refusing TAG='$TAG': set a run name (e.g. baseline-rerun); 'baseline' and 'baseline-s<seed>' are" \
         "October's outputs in the shadow and must not be overwritten" >&2
    exit 2
  fi
  [[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "TAG must be letters, digits, '-' or '_'" >&2; exit 2; }
  KEEP_CKPT="${KEEP_CKPT:-0}"
  case "$KEEP_CKPT" in 0|best|1) ;; *) echo "KEEP_CKPT must be 0, best or 1" >&2; exit 2 ;; esac
fi
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU"
START=$(date -Is)
case "$STEP" in
  backfill)
    PANO_STORE="${PANO_STORE:?set PANO_STORE}"
    if [ -z "${RAWLABELS:-}" ] && [ -f "$DATA/shadow_run.json" ]; then
      RAWLABELS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["rawlabels"])' "$DATA/shadow_run.json")
    fi
    RAWLABELS="${RAWLABELS:-$(cd "$DATA/.." && pwd)/rawlabels}"
    mkdir -p "$DATA/$LT/context"
    exec 9>"$DATA/$LT/context/.backfill.lock"
    flock -n 9 || { echo "another backfill of $LT holds $DATA/$LT/context/.backfill.lock; not starting a second" >&2; exit 1; }
    echo "$(date) backfill $LT in $DATA/$LT/context from $PANO_STORE (rawLabels $RAWLABELS)"
    python -u context_crops.py backfill --data-dir "$DATA" --label_type "$LT" --pano-store "$PANO_STORE" \
        --rawlabels-cache "$RAWLABELS" --verify 50 || { echo "backfill failed" >&2; exit 1; }
    ;;
  train)
    SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-25}"; RES="${RES:-224}"; BATCH="${BATCH:-8}"
    RUN="$RUNS/$LT/$TAG"
    REC="$DATA/$LT/runs/$TAG"
    mkdir -p "$RUN"
    # A finished job (evaluated, record copied, checkpoints cleaned) must not retrain on a requeue or resubmit.
    if [ -f "$RUN/.evaluated" ] && [ -f "$DATA/$LT/report_$TAG.txt" ] && [ -f "$REC/run_key.json" ]; then
      echo "$(date) $LT $TAG already evaluated ($DATA/$LT/report_$TAG.txt, record $REC); nothing to do"; exit 0
    fi
    if [ "$ARM" = none ]; then CTX=(); else CTX=(--context-arm "$ARM" --context-view-res "$RES"); fi
    echo "$(date) train $LT arm $ARM (tag $TAG, seed $SEED, res $RES, batch $BATCH, epochs $EPOCHS) -> $RUN"
    python -u fine_tune_classifier.py --label_type "$LT" --data-dir "$DATA" --output-dir "$RUN" --seed "$SEED" \
        --epochs "$EPOCHS" --num-workers 7 --target hard --best-metric macro_f1 --batch-size "$BATCH" ${CTX[@]+"${CTX[@]}"} \
        || { echo "training failed" >&2; exit 1; }
    BEST=$(python -c "import evaluate,sys; print(evaluate.best_checkpoint_in(sys.argv[1]))" "$RUN") \
        || { echo "no best checkpoint under $RUN" >&2; exit 1; }
    case "$BEST" in /*) ;; *) BEST="$RUN/$(basename "$BEST")" ;; esac
    if [ "$ARM" = none ]; then
      python -u evaluate.py --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$TAG" --per-city \
          || { echo "evaluate failed" >&2; exit 1; }
    else
      python -u context_model.py predict --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$TAG" \
          || { echo "predict failed" >&2; exit 1; }
      python -u evaluate.py --label_type "$LT" --data-dir "$DATA" --skip-inference --tag "$TAG" --per-city \
          || { echo "evaluate failed" >&2; exit 1; }
    fi
    cat "$DATA/$LT/report_$TAG.txt"
    # The durable run record, before any delete: the report's provenance reads it, and scratch is purged.
    mkdir -p "$REC"
    cp "$RUN/training_run.json" "$RUN/run_key.json" "$REC/" && cp "$BEST/trainer_state.json" "$REC/trainer_state.json" \
        && cp "$BEST/config.json" "$REC/config.json" \
        || { echo "run record copy to $REC failed; checkpoints kept" >&2; exit 1; }
    printf 'slurm_job_id=%s\ngpu=%s\nstarted=%s\nfinished=%s\nbest=%s\n' "${SLURM_JOB_ID:-}" "$GPU" "$START" \
        "$(date -Is)" "$(basename "$BEST")" > "$REC/job.txt"
    case "$KEEP_CKPT" in
      0) rm -rf "$RUN"/checkpoint-*; echo "removed $RUN/checkpoint-*" ;;
      best)
        for d in "$RUN"/checkpoint-*; do [ "$(basename "$d")" = "$(basename "$BEST")" ] || rm -rf "$d"; done
        rm -f "$BEST/optimizer.pt" "$BEST/scheduler.pt" "$BEST/rng_state.pth" "$BEST/training_args.bin"
        echo "kept the weights of $BEST only" ;;
      1) echo "KEEP_CKPT=1: every checkpoint kept" ;;
    esac
    if [ -n "${KEEP:-}" ]; then
      if mkdir -p "$KEEP/$LT/runs" && cp -r "$REC" "$KEEP/$LT/runs/"; then
        for f in "metrics_$TAG.json" "predictions_$TAG.csv" "per_city_$TAG.md" "report_$TAG.txt" "slices_$TAG.md"; do
          if [ -f "$DATA/$LT/$f" ]; then cp "$DATA/$LT/$f" "$KEEP/$LT/"; fi
        done
        echo "copied the small artifacts to $KEEP/$LT/"
      else
        echo "WARNING: copy to $KEEP/$LT failed; the outputs are only in $DATA/$LT (scratch, purged after 21 days)" >&2
      fi
    fi
    touch "$RUN/.evaluated"
    ;;
  *) echo "unknown STEP=$STEP (backfill|train)" >&2; exit 2 ;;
esac
echo "$(date) done $STEP $LT ${TAG:-}"
