#!/usr/bin/env bash
# On the database host, read-only: the crop geometry of named labels, for gold items the public rawLabels cache lacks
# (private deployments, excluded authors). Input to `analysis/adjudicator.py crops --label-points`. `crops` writes
# the exact command, with the label list, to label_points_request.txt:
#
#   ssh makelab1 "LABELS='sidewalk_a:12 sidewalk_b:34' bash -s" < analysis/pull_label_points.sh > label_points.csv
#
# Output: schema,label_id,pano_id,pano_source,author_is_ai,pano_x,pano_y,pano_width,pano_height,camera_pitch,camera_roll
# (the rawLabels columns of the same name: label_point.pano_x/pano_y, pano_data.width/height/camera_pitch/camera_roll/
# source). No user ids. It has private-deployment rows and label ids: keep it outside the repo.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=300000'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'  # the SidewalkAI account (role AI); public, see the API's validator_type
LABELS="${LABELS:?set LABELS to space-separated schema:label_id items}"
echo "schema,label_id,pano_id,pano_source,author_is_ai,pano_x,pano_y,pano_width,pano_height,camera_pitch,camera_roll"
for S in $(printf '%s\n' $LABELS | cut -d: -f1 | sort -u); do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) echo "$S: not a city schema name" >&2; continue ;; esac
  IDS=$(printf '%s\n' $LABELS | awk -F: -v s="$S" '$1 == s && $2 ~ /^[0-9]+$/ {print $2}' | paste -sd, -)
  [ -n "$IDS" ] || continue
  $P -c "COPY (
    SELECT '$S', l.label_id, l.pano_id, pd.source, (l.user_id = '$AI'), lp.pano_x, lp.pano_y, pd.width, pd.height,
           pd.camera_pitch, pd.camera_roll
    FROM $S.label l
    JOIN $S.label_point lp ON lp.label_id = l.label_id
    JOIN $S.pano_data pd ON pd.pano_id = l.pano_id
    WHERE l.label_id IN ($IDS)
    ORDER BY l.label_id) TO STDOUT WITH CSV" || echo "$S: query failed" >&2
done
