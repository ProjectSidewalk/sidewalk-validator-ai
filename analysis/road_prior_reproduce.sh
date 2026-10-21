#!/usr/bin/env bash
# Rerun every stage of analysis/road_prior.py behind docs/road-prior-2026-10.md and check the result against the
# recorded digests in docs/road-prior-2026-10/provenance.json.
#
# Usage:
#   EXPERT_COLUMNS=<comma list> analysis/road_prior_reproduce.sh <pull dir> <output root> [--record]
#
#   <pull dir>     the private pull, e.g. ~/Data/sidewalk-validator-ai-prod-2026-10-17, holding cities.json,
#                  rawlabels_cache/, phase0/labels_oof.csv, ai_vs_humans.csv, streets_cache/<date>/, osm_cache/<date>/
#   <output root>  a new directory outside any git checkout; one subdirectory per stage is written there
#   --record       write a new provenance.json instead of verifying against the recorded one
#
# Environment:
#   EXPERT_COLUMNS  expert vote column names in ai_vs_humans.csv (required; kept out of the repo)
#   STREETS_DATE    streets cache date (default 2026-10-20)
#   OSM_DATE        OSM cache date (default 2026-10-20)
#   PACKET_SEED_FILE  file holding the packet's secret seed; if set, the packet is rebuilt into
#                     <output root>/packet and its key into <output root>/packet/private-key-do-not-share
#   PYTHON          interpreter (default: .venv/bin/python if present, else python)
#
# The network stages (fetch-streets, fetch-osm) are not rerun: the dated caches are the inputs, pinned by digest.
# To refresh them, run those two stages into a new dated directory and rerun with --record; the numbers will move.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,24p' "$0"
  exit 2
fi
D=$1
OUT=$2
MODE=${3:-verify}
: "${EXPERT_COLUMNS:?set EXPERT_COLUMNS to the expert vote columns of ai_vs_humans.csv}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$([[ -x "$REPO/.venv/bin/python" ]] && echo "$REPO/.venv/bin/python" || echo python)}
RP="$PY $REPO/analysis/road_prior.py"
STREETS=$D/streets_cache/${STREETS_DATE:-2026-10-20}
OSM=$D/osm_cache/${OSM_DATE:-2026-10-20}
PROV=$REPO/docs/road-prior-2026-10/provenance.json

mkdir -p "$OUT"
$RP features --cities-json "$D/cities.json" --rawlabels "$D/rawlabels_cache" --streets-dir "$STREETS" \
  --osm-dir "$OSM" --labels-oof "$D/phase0/labels_oof.csv" --ai-vs-humans "$D/ai_vs_humans.csv" \
  --experts "$EXPERT_COLUMNS" --out "$OUT/features/features.csv" > "$OUT/features.log"
F=$OUT/features/features.csv
$RP evaluate --features "$F" --cities-json "$D/cities.json" --out-dir "$OUT/evaluate" \
  --fig-dir "$REPO/docs/road-prior-2026-10" > "$OUT/evaluate.log"
$RP stratify --features "$F" --cities-json "$D/cities.json" --out-dir "$OUT/stratify" > "$OUT/stratify.log"
$RP routing --features "$F" --cities-json "$D/cities.json" --out-dir "$OUT/routing" > "$OUT/routing.log"
RUNS=(--run-dir "$OUT/features" --run-dir "$OUT/evaluate" --run-dir "$OUT/stratify" --run-dir "$OUT/routing")
if [[ -n "${PACKET_SEED_FILE:-}" ]]; then
  KEY=$OUT/packet/private-key-do-not-share
  $RP packet --features "$F" --cities-json "$D/cities.json" --out-dir "$OUT/packet" --key-dir "$KEY" \
    --seed "$(cat "$PACKET_SEED_FILE")" > "$OUT/packet.log"
  RUNS+=(--run-dir "$KEY")
fi

if [[ $MODE == --record ]]; then
  $RP provenance "${RUNS[@]}" --out "$PROV"
else
  PARTIAL=$([[ -z "${PACKET_SEED_FILE:-}" ]] && echo --partial || true)
  $RP verify --provenance "$PROV" "${RUNS[@]}" $PARTIAL
fi
