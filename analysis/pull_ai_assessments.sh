#!/bin/bash
# One row per AI-assessed label in every city schema: the AI's latest raw verdict, served accuracy and confidence,
# plus the HUMAN votes on that label (the AI's own vote excluded). Accuracy and confidence are exported unrounded:
# most confidences sit within 1e-4 of 1.0, where the map's top grid cells are. Read-only; run on the database host.
#   EXPERT1_ID=<user_id> EXPERT2_ID=<user_id> ./pull_ai_assessments.sh > ai_assessments.csv
# EXPERT1_ID / EXPERT2_ID are optional sidewalk_login user_ids whose individual votes get their own columns
# (kept out of the repo on purpose). The output contains private-city data: never commit it.
#
# Columns after expert2 (added for issue #52, analysis/deployment_gate.py; readers that select columns by name are
# unaffected):
#   validator_revision       the model's hub commit (SidewalkWebpage evolution 420); empty before API 1.0.2, and
#                            empty in a schema that predates the evolution. The deployment gate keys its era on it.
#   validator_training_date  the model's training date as the API reported it.
#   api_version              the sidewalk-ai-api release that scored the label.
#   ai_vote_cast             t when this assessment cast a vote (its label_validation_id is set), f when it was
#                            scored without a vote (shadow mode). Shadow and vote evidence are never pooled.
#   h_agree_excl, h_dis_excl the human Agree / Disagree votes (already in h_agree / h_dis) cast by validators whose
#                            user_stat.excluded is true, for the excluded-validator sensitivity.
#   vote_type_join           t when the human counts are restricted to votes on the label's CURRENT type
#                            (label_validation.label_type, evolution 395), f in a schema without that column.
#                            Before this change the counts included votes cast on a label's earlier type.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000'
P="${PSQL:-psql -X -q -d sidewalk_prod -p 5434 -At}"
AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the SidewalkAI validator account
E1="${EXPERT1_ID:-none}"; E2="${EXPERT2_ID:-none}"
echo "schema,label_id,label_type,deleted,severity,ai_result,ai_accuracy,ai_confidence,ai_date,model_id,h_agree,h_dis,h_unsure,expert1,expert2,validator_revision,validator_training_date,api_version,ai_vote_cast,h_agree_excl,h_dis_excl,vote_type_join"
has_col() {  # schema table column -> t/f
  $P -c "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='$1' AND table_name='$2' AND column_name='$3')"
}
for S in $($P -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'sidewalk_%' AND schema_name NOT IN ('sidewalk_login') AND schema_name NOT LIKE '%backup%' ORDER BY 1"); do
  has=$($P -c "SELECT to_regclass('$S.label_ai_assessment') IS NOT NULL")
  [ "$has" = "t" ] || { echo "$S: no label_ai_assessment" >&2; continue; }
  REV="NULL::text"; [ "$(has_col "$S" label_ai_assessment validator_revision)" = "t" ] && REV="validator_revision"
  TYPEJOIN=""; TJ="f"
  [ "$(has_col "$S" label_validation label_type)" = "t" ] && { TYPEJOIN="AND v.label_type = l.label_type"; TJ="t"; }
  $P -c "COPY (
    WITH a AS (
      SELECT DISTINCT ON (label_id) label_id,
        CASE validation_result::text WHEN '1' THEN 'Agree' WHEN '2' THEN 'Disagree' WHEN '3' THEN 'Unsure' ELSE validation_result::text END AS r, validation_accuracy, validation_confidence, timestamp::date AS d, validator_model_id,
        $REV AS rev, validator_training_date AS trained, api_version AS api, (label_validation_id IS NOT NULL) AS cast
      FROM $S.label_ai_assessment ORDER BY label_id, timestamp DESC, label_ai_assessment_id DESC)
    SELECT '$S', l.label_id, l.label_type::text, l.deleted, l.severity, a.r, a.validation_accuracy, a.validation_confidence, a.d, a.validator_model_id,
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('1','Agree') THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('2','Disagree') THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN v.validation_result::text IN ('3','Unsure') THEN 1 ELSE 0 END),0),
      MAX(CASE WHEN v.user_id='$E1' THEN CASE v.validation_result::text WHEN '1' THEN 'Agree' WHEN '2' THEN 'Disagree' WHEN '3' THEN 'Unsure' ELSE v.validation_result::text END END),
      MAX(CASE WHEN v.user_id='$E2' THEN CASE v.validation_result::text WHEN '1' THEN 'Agree' WHEN '2' THEN 'Disagree' WHEN '3' THEN 'Unsure' ELSE v.validation_result::text END END),
      a.rev, a.trained, a.api, a.cast,
      COALESCE(SUM(CASE WHEN us.excluded AND v.validation_result::text IN ('1','Agree') THEN 1 ELSE 0 END),0),
      COALESCE(SUM(CASE WHEN us.excluded AND v.validation_result::text IN ('2','Disagree') THEN 1 ELSE 0 END),0),
      '$TJ'
    FROM a JOIN $S.label l ON l.label_id=a.label_id
    LEFT JOIN $S.label_validation v ON v.label_id=l.label_id AND v.user_id <> '$AI' $TYPEJOIN
    LEFT JOIN $S.user_stat us ON us.user_id = v.user_id
    GROUP BY 1,2,3,4,5,6,7,8,9,10,16,17,18,19) TO STDOUT WITH CSV" || echo "$S: query failed" >&2
done
