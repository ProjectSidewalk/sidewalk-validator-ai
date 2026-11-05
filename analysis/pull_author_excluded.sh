#!/usr/bin/env bash
# On the database host, read-only: whether each label's author is excluded in user_stat, for the schemas of a GT
# round. Input to `analysis/gt_rounds.py recount` and `score --author-excluded` (issue #85), which join it onto the
# round's ORIGINAL pinned pool, so eligibility stays as of the round. Gallery hides excluded authors' labels, so a
# rater could never have voted on them; the round-2 pool predates the pool query's author_excluded column.
#
#   ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' bash -s" < analysis/pull_author_excluded.sh \
#       > ~/Data/sidewalk-validator-ai-relabel-2026-10/round2/author_excluded.csv
#
# Output: schema,label_id,author_excluded (t/f), one row per non-deleted, non-tutorial label. No user ids. Keep it
# outside the repo with the pool (it has label ids).
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=900000'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
echo "schema,label_id,author_excluded"
for S in ${SCHEMAS:-sidewalk_seattle sidewalk_chicago}; do
  $P -c "COPY (
    SELECT '$S', l.label_id, COALESCE(us.excluded, false)
    FROM $S.label l
    LEFT JOIN $S.user_stat us USING (user_id)
    WHERE NOT l.deleted AND NOT l.tutorial
    ORDER BY l.label_id) TO STDOUT WITH CSV" || echo "$S: query failed" >&2
done
