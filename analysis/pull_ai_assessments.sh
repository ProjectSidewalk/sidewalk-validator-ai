#!/bin/bash
# One row per AI-assessed label in every city schema: the AI's latest raw verdict, served accuracy and confidence,
# plus the HUMAN votes on that label (the AI's own vote excluded). Accuracy and confidence are exported unrounded:
# most confidences sit within 1e-4 of 1.0, where the map's top grid cells are. Read-only; run on the database host.
#   EXPERT1_ID=<user_id> EXPERT2_ID=<user_id> ./pull_ai_assessments.sh > ai_assessments.csv
# EXPERT1_ID / EXPERT2_ID are optional sidewalk_login user_ids whose individual votes get their own columns
# (kept out of the repo on purpose). The output contains private-city data: never commit it.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000'
P="${PSQL:-psql -X -q -d sidewalk_prod -p 5434 -At}"
AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the SidewalkAI validator account
E1="${EXPERT1_ID:-none}"; E2="${EXPERT2_ID:-none}"
echo "schema,label_id,label_type,deleted,severity,ai_result,ai_accuracy,ai_confidence,ai_date,model_id,h_agree,h_dis,h_unsure,expert1,expert2"
for S in $($P -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'sidewalk_%' AND schema_name NOT IN ('sidewalk_login') AND schema_name NOT LIKE '%backup%' ORDER BY 1"); do
  has=$($P -c "SELECT to_regclass('$S.label_ai_assessment') IS NOT NULL")
  [ "$has" = "t" ] || { echo "$S: no label_ai_assessment" >&2; continue; }
  $P -c "COPY (
    WITH a AS (
      SELECT DISTINCT ON (label_id) label_id,
        CASE validation_result::text WHEN '1' THEN 'Agree' WHEN '2' THEN 'Disagree' WHEN '3' THEN 'Unsure' ELSE validation_result::text END AS r, validation_accuracy, validation_confidence, timestamp::date AS d, validator_model_id
      FROM $S.label_ai_assessment ORDER BY label_id, timestamp DESC)
    SELECT '$S', l.label_id, l.label_type::text, l.deleted, l.severity, a.r, a.validation_accuracy, a.validation_confidence, a.d, a.validator_model_id,
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('1','Agree') THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('2','Disagree') THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('3','Unsure') THEN 1 ELSE 0 END),0),
      MAX(CASE WHEN v.user_id='$E1' THEN v.validation_result::text END),
      MAX(CASE WHEN v.user_id='$E2' THEN v.validation_result::text END)
    FROM a JOIN $S.label l ON l.label_id=a.label_id
    LEFT JOIN $S.label_validation v ON v.label_id=l.label_id AND v.user_id <> '$AI'
    GROUP BY 1,2,3,4,5,6,7,8,9,10) TO STDOUT WITH CSV" || echo "$S: query failed" >&2
done
