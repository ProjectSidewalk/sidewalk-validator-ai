#!/bin/bash
# Issue #74: score the gold manifests ($R4/gold/$CROPS/<Type>.csv, from round4.py gold-manifest) with kept models, in
# one GPU job. Each checkpoint costs a model load plus inference: measured 2.6 to 10.9 min per checkpoint on klone
# (job 43405831, 5 round-3 checkpoints in 26 min), so 27 pilot checkpoints need 1 to 5 hours and stage 2's uniform
# scoring (32 checkpoints) more; --time is 4 h and a TIMEOUT is resumed by resubmitting the same line (finished
# outputs are skipped by their sha256). collect records which gold prediction files exist per run. Outputs go next
# to evaluate.py's: $R4/<Type>/gold_predictions_<CROPS>_<TAG>.csv and its _provenance.json (weights and manifest
# sha256).
#   MODE=round3  the round-3 checkpoint of each type ($DATA/<Type>/dinov2, resolved by round4.py round3-ckpt)
#   MODE=runs    $RUNS/<Type>/<TAG>/best for every TAG in TAGS (a tag starting "cl-" only for the types in CL_TYPES:
#                SurfaceProblem has no cl-drop arm)
#   MODE=one     CKPT and TAG for TYPES (one type)
# MODE may list several, comma-separated (the smoke scores MODE=round3,runs). A missing best/ or a failed scoring is
# recorded and the job exits non-zero after trying the rest. Requeue-safe: score-gold skips an output whose
# provenance records the same weights and manifest.
#   MODE=runs TYPES=Crosswalk TAGS=baseline,cl-drop R4=... RUNS=... DATA=... sbatch analysis/round4_score.slurm
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --job-name=round4-disagree-score
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
MODE="${MODE:?set MODE (round3|runs|one, or a comma list)}"; R4="${R4:?set R4}"; DATA="${DATA:?set DATA}"
CROPS="${CROPS:-r3crops}"; TYPES="${TYPES:-Crosswalk,CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem}"
ENV="${ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
case "$R4" in /tmp*|/gscratch/makelab*) echo "refusing R4=$R4 (use /gscratch/scrubbed)" >&2; exit 2 ;; esac
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
FAILED=()
score() {  # score <Type> <checkpoint dir> <tag>
  local man="$R4/gold/$CROPS/$1.csv"
  [ -f "$man" ] || { echo "no manifest $man" >&2; FAILED+=("$1/$3:no-manifest"); return; }
  python -u analysis/round4.py score-gold --manifest "$man" --checkpoint "$2" --tag "$3" --label-type "$1" \
      --out "$R4/$1/gold_predictions_${CROPS}_$3.csv" || FAILED+=("$1/$3:score-failed")
}
for M in ${MODE//,/ }; do
  for T in ${TYPES//[,:]/ }; do
    case "$M" in
      round3)
        # The path trainer_state.json records is the training job's; resolve it next to that file (A3).
        if CKPT=$(python analysis/round4.py round3-ckpt --dinov2 "$DATA/$T/dinov2"); then score "$T" "$CKPT" round3
        else FAILED+=("$T/round3:no-checkpoint"); fi ;;
      runs)
        RUNS="${RUNS:?set RUNS}"; CL_TYPES="${CL_TYPES:-Crosswalk,CurbRamp,NoCurbRamp,Obstacle}"
        for TAG in ${TAGS//[,:]/ }; do
          case "$TAG" in cl-*) case ",${CL_TYPES//:/,}," in *",$T,"*) ;; *) continue ;; esac ;; esac
          if [ -f "$RUNS/$T/$TAG/best/model.safetensors" ]; then score "$T" "$RUNS/$T/$TAG/best" "$TAG"
          else echo "missing $RUNS/$T/$TAG/best" >&2; FAILED+=("$T/$TAG:no-best"); fi
        done ;;
      one) score "$T" "${CKPT:?set CKPT}" "${TAG:?set TAG}" ;;
      *) echo "unknown MODE $M" >&2; exit 2 ;;
    esac
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "$(date) failed: ${FAILED[*]}" >&2; exit 1; fi
echo "$(date) done $MODE $CROPS"
