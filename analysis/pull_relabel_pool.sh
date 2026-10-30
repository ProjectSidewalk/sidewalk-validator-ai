#!/usr/bin/env bash
# On the database host, read-only: one row per label the AI validator has voted on, across all city schemas, with
# the label's severity, tags and pano view, the AI's vote and latest served accuracy, and the human votes (AI
# excluded). Input to ai_disagree_relabel.py (the masked expert re-label) and its severity/tag breakdown.
#
#   ssh makelab1 'bash -s' < analysis/pull_relabel_pool.sh > ~/Data/.../relabel_pool.csv
#
# Optional EXPERT1_ID / EXPERT2_ID (user_id UUIDs) add those validators' own votes as role-keyed columns, and
# expert1_labeled / expert2_labeled (t/f) mark labels each of them placed; the human tallies are every non-AI vote,
# unfiltered. Pass them through the environment, never commit them. Optional SCHEMAS ("sidewalk_a sidewalk_b")
# restricts the pull to those schemas (a GT round in named cities, analysis/gt_rounds.py); the default is every city
# schema. author_excluded (t/f) marks labels whose author is excluded in user_stat; Gallery never shows those, so GT
# rounds drop them (analysis/gt_rounds.py). Output has private-city rows: keep it outside the repo.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=900000'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'  # the SidewalkAI account (role AI); public, see the API's validator_type
E1="${EXPERT1_ID:-00000000-0000-0000-0000-000000000000}"
E2="${EXPERT2_ID:-00000000-0000-0000-0000-000000000000}"
echo "schema,label_id,label_type,severity,tags,ai_vote,ai_accuracy,h_agree,h_dis,h_unsure,expert1,expert2,expert1_labeled,expert2_labeled,pano_id,heading,pitch,author_excluded"
for S in ${SCHEMAS:-$($P -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'sidewalk_%' AND schema_name NOT IN ('sidewalk_login','sidewalk_validation') AND schema_name NOT LIKE '%backup%' ORDER BY 1")}; do
  has=$($P -c "SELECT to_regclass('$S.label_ai_assessment') IS NOT NULL")
  [ "$has" = "t" ] || { echo "$S: no label_ai_assessment" >&2; continue; }
  $P -c "COPY (
    WITH a AS (
      SELECT DISTINCT ON (label_id) label_id, validation_accuracy
      FROM $S.label_ai_assessment ORDER BY label_id, timestamp DESC),
    ai AS (SELECT DISTINCT ON (label_id) label_id, validation_result::text AS r FROM $S.label_validation
           WHERE user_id = '$AI' ORDER BY label_id, label_validation_id DESC)
    SELECT '$S', l.label_id, l.label_type::text, l.severity, array_to_string(l.tags, ';'), ai.r,
      round(a.validation_accuracy::numeric, 4),
      COALESCE(SUM(CASE WHEN v.validation_result::text = 'Agree' THEN 1 ELSE 0 END), 0),
      COALESCE(SUM(CASE WHEN v.validation_result::text = 'Disagree' THEN 1 ELSE 0 END), 0),
      COALESCE(SUM(CASE WHEN v.validation_result::text = 'Unsure' THEN 1 ELSE 0 END), 0),
      MAX(CASE WHEN v.user_id = '$E1' THEN v.validation_result::text END),
      MAX(CASE WHEN v.user_id = '$E2' THEN v.validation_result::text END),
      (l.user_id = '$E1'), (l.user_id = '$E2'),
      l.pano_id, round(lp.heading::numeric, 2), round(lp.pitch::numeric, 2), COALESCE(us.excluded, false)
    FROM ai
    JOIN $S.label l ON l.label_id = ai.label_id AND NOT l.deleted AND NOT l.tutorial
    JOIN $S.label_point lp ON lp.label_id = l.label_id
    LEFT JOIN a ON a.label_id = l.label_id
    LEFT JOIN $S.user_stat us ON us.user_id = l.user_id
    LEFT JOIN $S.label_validation v ON v.label_id = l.label_id AND v.user_id <> '$AI'
    GROUP BY 1, 2, 3, 4, 5, 6, 7, l.user_id, l.pano_id, lp.heading, lp.pitch, us.excluded) TO STDOUT WITH CSV" || echo "$S: query failed" >&2
done
