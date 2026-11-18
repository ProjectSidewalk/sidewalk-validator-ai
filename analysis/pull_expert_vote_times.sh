#!/usr/bin/env bash
# On the database host, read-only: every vote the two expert accounts cast, with its time and the page it was cast
# on, for issue #102 item 1 (analysis/nosidewalk_runs.py agreement --vote-times). The public rawLabels carry each
# vote's value but not its time, and "do the labels of a run get the same verdict" cannot be told apart from "the
# expert judged the run in one sitting" without the time between the two votes.
#
#   ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' EXPERT1_ID=<uuid> EXPERT2_ID=<uuid> bash -s" \
#       < analysis/pull_expert_vote_times.sh > ~/Data/sidewalk-validator-ai-nosidewalk-2026-11/runs/expert_vote_times.csv
#
# Output: schema,label_id,label_type,expert_idx,validation_result,start_timestamp,end_timestamp,source,mission_id,
# label_author_is_expert; one row per expert vote (every label type, non-deleted and non-tutorial labels), times in
# UTC. The experts are written as 1 and 2 (EXPERT1_ID, EXPERT2_ID), never as ids, and no other user id is read out.
# Keep the file outside the repo (it has label ids). A schema name that is not sidewalk_[a-z0-9_]*, or an expert id
# that is not a UUID, stops the script before any query (exit 2). A failed query is reported on stderr and makes the
# script exit 1 after the other schemas, so check the exit status: the CSV then lacks that schema's rows.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=900000 -c timezone=UTC'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
SCHEMAS="${SCHEMAS:?set SCHEMAS (space-separated city schemas)}"
E1="${EXPERT1_ID:-}"; E2="${EXPERT2_ID:-}"
for E in "$E1" "$E2"; do  # checked, never echoed: the value is a user id
  case "$E" in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]-*) ;;
    *) echo "EXPERT1_ID and EXPERT2_ID must both be user_id UUIDs" >&2; exit 2 ;;
  esac
  case "$E" in *[!0-9a-f-]*) echo "EXPERT1_ID and EXPERT2_ID must both be user_id UUIDs" >&2; exit 2 ;; esac
  [ "${#E}" = 36 ] || { echo "EXPERT1_ID and EXPERT2_ID must both be user_id UUIDs" >&2; exit 2; }
done
for S in $SCHEMAS; do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) echo "$S: not a city schema name" >&2; exit 2 ;; esac
  case "$S" in *[!a-z0-9_]*) echo "$S: not a city schema name" >&2; exit 2 ;; esac
done
FAILED=0
echo "schema,label_id,label_type,expert_idx,validation_result,start_timestamp,end_timestamp,source,mission_id,label_author_is_expert"
for S in $SCHEMAS; do
  $P -c "COPY (
    SELECT '$S', l.label_id, l.label_type::text,
      CASE v.user_id WHEN '$E1' THEN 1 ELSE 2 END,
      CASE v.validation_result::text WHEN '1' THEN 'Agree' WHEN '2' THEN 'Disagree' WHEN '3' THEN 'Unsure'
        ELSE v.validation_result::text END,
      v.start_timestamp, v.end_timestamp, v.source::text, v.mission_id, (l.user_id IN ('$E1', '$E2'))
    FROM $S.label_validation v
    JOIN $S.label l ON l.label_id = v.label_id
    WHERE v.user_id IN ('$E1', '$E2') AND NOT l.deleted AND NOT l.tutorial
    ORDER BY l.label_id, v.end_timestamp, v.label_validation_id) TO STDOUT WITH CSV" \
    || { echo "$S: query failed" >&2; FAILED=1; }
done
exit "$FAILED"
