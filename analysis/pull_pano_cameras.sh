#!/usr/bin/env bash
# On the database host, read-only: camera positions of labelled panoramas (MODE=panos, the default), or each label's
# position estimator (MODE=methods), for temporal validation Stage 0 (issue #17, docs/temporal-validation-2026-11.md).
# Input to `analysis/temporal_pairs.py positions --db-panos ... [--db-methods ...]`; only database positions can give a
# Gate 0 verdict (positions inverted from public rawLabels are a preview).
#
#   ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' bash -s" < analysis/pull_pano_cameras.sh \
#       > ~/Data/sidewalk-validator-ai-temporal-2026-11/db/panos.csv
#   ssh makelab1 "MODE=methods SCHEMAS='sidewalk_seattle sidewalk_chicago' bash -s" < analysis/pull_pano_cameras.sh \
#       > ~/Data/sidewalk-validator-ai-temporal-2026-11/db/methods.csv
#
# SCHEMAS has no default: pass the PUBLIC deployments' schemas only (today's /v3/api/cities listing, visibility
# "public", mapped to schemas through SidewalkWebpage conf/cityparams.conf). This script cannot check visibility.
# makelab rules apply: reuse a live control master, put both modes through one connection if you can, and on one
# timeout stop (no retries).
#
# Output, MODE=panos: schema,pano_id,lat,lng,camera_heading,camera_pitch,camera_roll,capture_date,width,height,source,
#   expired,has_backup,n_labels -- one row per panorama that holds at least one non-deleted, non-tutorial label.
# Output, MODE=methods: schema,label_id,computation_method -- one row per such label.
# No user ids. The output has pano and label ids: keep it under ~/Data/sidewalk-validator-ai-temporal-2026-11/, never
# in the repo.
#
# Column dependencies (a schema that lags these evolutions fails its query, is reported on stderr, and makes the
# script exit 1 after the other schemas, so check the exit status: the CSV then lacks that schema's rows):
#   298 renamed gsv_data -> pano_data and gsv_panorama_id -> pano_id; 306 added pano_data.camera_roll;
#   319 added pano_data.has_backup; 352 gave label_point.computation_method its three values
#   (depth / approximation2 / approximation3).
# UNVERIFIED: that pano_data.lat/lng is the provider-reported camera position, unchanged since the label was placed
#   (temporal_pairs.py positions compares it with the inverted position per city; centimetres expected).
# UNVERIFIED: that every schema has pano_data.source (the PanoSource enum: gsv, mapillary, infra3d, panoramax, ...);
#   the Scala model reads it (PanoDataTable.scala), the evolution that added it was not looked up.
# UNVERIFIED: that capture_date is text in YYYY, YYYY-MM or YYYY-MM-DD (evolution 326's comment says so).
#
# A schema name that is not sidewalk_[a-z0-9_]* stops the script before any query (exit 2), as does an unknown MODE
# or an empty SCHEMAS.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=900000'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
MODE="${MODE:-panos}"
SCHEMAS="${SCHEMAS:-}"
case "$MODE" in panos|methods) ;; *) echo "MODE must be panos or methods, got '$MODE'" >&2; exit 2 ;; esac
if [ -z "$SCHEMAS" ]; then echo "SCHEMAS is required (the public deployments' schemas)" >&2; exit 2; fi
for S in $SCHEMAS; do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) echo "$S: not a city schema name" >&2; exit 2 ;; esac
  case "$S" in *[!a-z0-9_]*) echo "$S: not a city schema name" >&2; exit 2 ;; esac
done
FAILED=0
if [ "$MODE" = panos ]; then
  echo "schema,pano_id,lat,lng,camera_heading,camera_pitch,camera_roll,capture_date,width,height,source,expired,has_backup,n_labels"
else
  echo "schema,label_id,computation_method"
fi
for S in $SCHEMAS; do
  if [ "$MODE" = panos ]; then
    SQL="COPY (
      SELECT '$S', p.pano_id, p.lat, p.lng, p.camera_heading, p.camera_pitch, p.camera_roll, p.capture_date,
             p.width, p.height, p.source, p.expired, p.has_backup, l.n_labels
      FROM $S.pano_data p
      JOIN (SELECT pano_id, COUNT(*) AS n_labels FROM $S.label
            WHERE NOT deleted AND NOT tutorial GROUP BY pano_id) l USING (pano_id)
      ORDER BY p.pano_id) TO STDOUT WITH CSV"
  else
    SQL="COPY (
      SELECT '$S', l.label_id, lp.computation_method
      FROM $S.label l
      JOIN $S.label_point lp ON lp.label_id = l.label_id
      WHERE NOT l.deleted AND NOT l.tutorial
      ORDER BY l.label_id) TO STDOUT WITH CSV"
  fi
  $P -c "$SQL" || { echo "$S: query failed" >&2; FAILED=1; }
done
exit "$FAILED"
