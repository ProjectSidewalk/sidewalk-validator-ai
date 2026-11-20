#!/bin/bash
# Issue #119 (docs/backbone-2026-11.md): download the comparison's backbone weights into the HF cache on klone's
# login node (network; download only, nothing is converted or trained here), at the registry's pinned revisions.
#
#   analysis/backbone_prefetch.sh       # from the repo root, after analysis/backbone_env.sh
#
# Checks /gscratch/makelab first (hyakstorage; at least 10 GB and 2,000 files free, since the cache lives there),
# then runs `backbones.py prefetch` for the four non-gated rows, plus the gated rows only when HF_TOKEN is set
# (weights are gated on HF; a token with access is required). Writes $OUT/backbone_manifest.json: per row
# available or not (and why), file sha256s, and for dinov2-large whether the cache's refs/main equals the pin.
# Exit 3 when a requested row is unavailable (the submit script then skips that arm).
#
# Variables (defaults in brackets):
#   OUT       [/gscratch/scrubbed/$USER/backbone-comparison-2026-11]
#   HF_HOME   [/gscratch/makelab/$USER/hf_cache]
#   BASE_ENV, VENV as in analysis/backbone_env.sh
#   KEYS      [dinov2-large,dinov2-reg-large,convnextv2-base,rampnet-convnextv2-base]
#   GATED_KEYS [dinov3-vitl16,dinov3-vitb16] added only when HF_TOKEN is set
#   PY        [$VENV/bin/python]
set -euo pipefail
OUT="${OUT:-/gscratch/scrubbed/$USER/backbone-comparison-2026-11}"
VENV="${VENV:-/gscratch/scrubbed/$USER/envs/sva-bb}"
PY="${PY:-$VENV/bin/python}"
export HF_HOME="${HF_HOME:-/gscratch/makelab/$USER/hf_cache}" HF_HUB_DISABLE_XET=1
KEYS="${KEYS:-dinov2-large,dinov2-reg-large,convnextv2-base,rampnet-convnextv2-base}"
GATED_KEYS="${GATED_KEYS:-dinov3-vitl16,dinov3-vitb16}"
mkdir -p "$OUT"
hyakstorage /gscratch/makelab | "$PY" backbones.py storage-free --min-gb 10 --min-files 2000 \
  || { echo "too little free space on /gscratch/makelab for the HF cache; not downloading" >&2; exit 4; }
if [ -n "${HF_TOKEN:-}" ]; then
  KEYS="$KEYS,$GATED_KEYS"
else
  echo "HF_TOKEN is not set: skipping $GATED_KEYS (weights are gated on HF; a token with access is required)"
fi
"$PY" backbones.py prefetch --backbones "$KEYS" --out "$OUT/backbone_manifest.json"
