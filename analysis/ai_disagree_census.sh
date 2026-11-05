#!/bin/bash
# Census of the SidewalkAI validator's Disagree votes in every city schema (SidewalkWebpage #5793): how many there
# are, how many of their labels are hidden (correct = FALSE), how many the AI decided alone or by breaking a human tie,
# and how often the human majority agrees with the AI where humans voted. Read-only; run on the database host.
#   ./ai_disagree_census.sh > ai_disagree_census.csv
# The output names private cities: never commit it.
#
# One row per city and label type, then ALL rows per label type and overall. Only labels that count are included:
# not deleted, not tutorial, and only an AI vote on the label's current type (FilteredTables.isVerdictVote, which is
# what lets it decide `correct`). Human votes are the other isVerdictVote votes (not the labeler's own, current type,
# validator not excluded), the AI's excluded.
#   lone_ai                 no human vote at all
#   ai_broke_tie            humans voted, agrees = disagrees, and the label is hidden
#   human_majority_*        humans voted and more of them disagreed / agreed; human_tied is neither
#   human_majority_sides_with_ai = disagree / (disagree + agree), ties left out
# The 2026-10-24 census posted on #5793 used this description; its SQL was not saved, so small differences from the
# posted figures can come from the definitions (the current-type rule, how ties count), not only from new votes.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000'
DB_NAME=${DB_NAME:-sidewalk_prod}
DB_PORT=${DB_PORT:-5434}
P="${PSQL:-psql} -X -q -d $DB_NAME -p $DB_PORT -At"
AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the SidewalkAI validator account
MIN_EVOLUTION=395                            # label_validation.label_type

HEADER="schema,label_type,ai_disagree_votes,hidden,lone_ai,ai_broke_tie,humans_voted,human_majority_disagree,\
human_majority_agree,human_tied,human_majority_sides_with_ai"
echo "$HEADER"
rows=''
for S in $($P -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'sidewalk_%' AND schema_name NOT IN ('sidewalk_login') AND schema_name NOT LIKE '%backup%' ORDER BY 1"); do
  evo=$($P -c "SELECT CASE WHEN to_regclass('$S.play_evolutions') IS NULL OR to_regclass('$S.label_validation') IS NULL
                 THEN -1 ELSE (SELECT COALESCE(max(id), 0) FROM $S.play_evolutions WHERE state = 'applied') END") \
    || { echo "$S: query failed" >&2; continue; }
  (( evo >= MIN_EVOLUTION )) || { echo "$S: skipped (evolution $evo)" >&2; continue; }
  out=$($P -F, -c "
    WITH ai AS (
      SELECT label.label_id, label.label_type, label.correct, label.user_id
      FROM $S.label_validation
      INNER JOIN $S.label ON label_validation.label_id = label.label_id
      WHERE label_validation.user_id = '$AI' AND label_validation.validation_result = 'Disagree'
        AND label_validation.label_type = label.label_type AND label_validation.user_id <> label.user_id
        AND NOT label.deleted AND NOT label.tutorial
    ), h AS (
      SELECT ai.label_id, ai.label_type, ai.correct,
             COUNT(label_validation.label_validation_id) AS h_votes,
             COUNT(*) FILTER (WHERE label_validation.validation_result = 'Agree') AS h_agree,
             COUNT(*) FILTER (WHERE label_validation.validation_result = 'Disagree') AS h_disagree
      FROM ai
      LEFT JOIN $S.label_validation ON label_validation.label_id = ai.label_id
        AND label_validation.user_id <> '$AI' AND label_validation.user_id <> ai.user_id
        AND label_validation.label_type = ai.label_type
        AND NOT EXISTS (SELECT 1 FROM $S.user_stat
                        WHERE user_stat.user_id = label_validation.user_id AND user_stat.excluded)
      GROUP BY ai.label_id, ai.label_type, ai.correct
    )
    SELECT '$S', label_type::text, COUNT(*),
           COUNT(*) FILTER (WHERE correct IS FALSE),
           COUNT(*) FILTER (WHERE h_votes = 0),
           COUNT(*) FILTER (WHERE h_votes > 0 AND h_agree = h_disagree AND correct IS FALSE),
           COUNT(*) FILTER (WHERE h_votes > 0),
           COUNT(*) FILTER (WHERE h_votes > 0 AND h_disagree > h_agree),
           COUNT(*) FILTER (WHERE h_votes > 0 AND h_agree > h_disagree),
           COUNT(*) FILTER (WHERE h_votes > 0 AND h_agree = h_disagree)
    FROM h GROUP BY label_type ORDER BY label_type") || { echo "$S: query failed" >&2; continue; }
  [ -n "$out" ] || continue
  rows+="$out"$'\n'
done

# Per-row agreement rate, then ALL rows per label type and overall.
printf '%s' "$rows" | awk -F, 'BEGIN { OFS = "," }
  function rate(d, a) { return (d + a) ? sprintf("%.3f", d / (d + a)) : "" }
  NF >= 10 {
    print $0, rate($8, $9)
    for (i = 3; i <= 10; i++) { t[$2, i] += $i; all[i] += $i }
    types[$2] = 1
  }
  END {
    n = 0; for (k in types) list[++n] = k
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (list[j] < list[i]) { x = list[i]; list[i] = list[j]; list[j] = x }
    for (i = 1; i <= n; i++) {
      line = "ALL," list[i]; for (c = 3; c <= 10; c++) line = line "," t[list[i], c]
      print line, rate(t[list[i], 8], t[list[i], 9])
    }
    if (n) { line = "ALL,ALL"; for (c = 3; c <= 10; c++) line = line "," all[c]; print line, rate(all[8], all[9]) }
  }'
