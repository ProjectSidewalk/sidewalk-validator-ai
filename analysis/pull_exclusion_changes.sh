#!/usr/bin/env bash
# On the database host, read-only, COUNTS ONLY: when were authors' `user_stat.excluded` flags toggled, and how many
# labels do those authors hold? Companion to pull_author_excluded.sh for the round-2 recount (issue #85): the flags
# that script reads are as of the pull date, not the rating window, and this log dates every change in between.
#
#   ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' SINCE=2026-10-18 bash -s" \
#       < analysis/pull_exclusion_changes.sh > ~/Data/.../exclusion_changes.csv
#
# Normally run through analysis/round2_flags_pull.sh, which bundles it with pull_author_excluded.sh into one ssh call.
#
# Output (no header; each row starts with its record type so both kinds share one stream):
#   change,<schema>,<pacific_day>,<new_value>,<n_users>,<n_labels>
#       exclusion toggles logged since SINCE, per Pacific day and new value (true = excluded); n_labels = the
#       non-deleted, non-tutorial labels those users hold NOW.
#   author_mismatch,<schema>,<n>
#       non-deleted, non-tutorial labels whose label.user_id differs from their audit_task.user_id. Prod joins
#       user_stat through audit_task; the pull scripts join through label, so this should be 0.
# The activity string embeds a user_id: the queries never select it, the user_id or the activity text into the output.
#
# What the log cannot see: toggles before 2026-09-14 (SidewalkWebpage 0e9974eda added the logging in
# AdminController's exclusion endpoint) and any change made by direct SQL.
#
# Inputs: SCHEMAS (default sidewalk_seattle sidewalk_chicago; each must match sidewalk_[a-z0-9_]*), SINCE (default
# 2026-10-18, the rawLabels export date; YYYY-MM-DD). A bad input stops the script before any query (exit 2). A
# failed query is reported on stderr ("<schema>: ... query failed") and makes the script exit 1 after the rest.
# Writes nothing on the host: every statement runs under default_transaction_read_only.
#
# UNVERIFIED: both queries were written from the Slick table definitions (WebpageActivityTable.scala, LabelTable.scala,
# AuditTaskTable.scala at SidewalkWebpage f741590d7) and run only against a synthetic schema in a local postgres:16
# container, never against production. Not known: that the jonf role can read webpage_activity, that the queries
# finish inside the 900 s statement timeout. round2_flags_pull.sh therefore treats a failure here as non-fatal for
# the flags.
set -u
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=900000'
P="psql -X -q -d sidewalk_prod -p 5434 -At"
SCHEMAS="${SCHEMAS:-sidewalk_seattle sidewalk_chicago}"
SINCE="${SINCE:-2026-10-18}"
for S in $SCHEMAS; do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) echo "$S: not a city schema name" >&2; exit 2 ;; esac
  case "$S" in *[!a-z0-9_]*) echo "$S: not a city schema name" >&2; exit 2 ;; esac
done
case "$SINCE" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) echo "SINCE=$SINCE: not YYYY-MM-DD" >&2; exit 2 ;;
esac
FAILED=0
for S in $SCHEMAS; do
  # UNVERIFIED against production (see header).
  $P -c "COPY (
    WITH ev AS (
      SELECT (a.timestamp AT TIME ZONE 'America/Los_Angeles')::date AS day,
             substring(a.activity from '_New=(true|false)\$') AS new_value,
             substring(a.activity from '^UpdateUserExcluded_User=(.*)_New=(?:true|false)\$') AS uid
      FROM $S.webpage_activity a
      WHERE a.activity LIKE 'UpdateUserExcluded%' AND a.timestamp >= '$SINCE')
    SELECT 'change', '$S', e.day, e.new_value, count(DISTINCT e.uid),
           (SELECT count(*) FROM $S.label l
             WHERE NOT l.deleted AND NOT l.tutorial
               AND l.user_id IN (SELECT x.uid FROM ev x WHERE x.day = e.day AND x.new_value = e.new_value))
    FROM ev e GROUP BY e.day, e.new_value ORDER BY e.day, e.new_value) TO STDOUT WITH CSV" \
    || { echo "$S: exclusion-log query failed" >&2; FAILED=1; }
  # UNVERIFIED against production (see header).
  $P -c "COPY (
    SELECT 'author_mismatch', '$S', count(*)
    FROM $S.label l JOIN $S.audit_task t ON t.audit_task_id = l.audit_task_id
    WHERE NOT l.deleted AND NOT l.tutorial AND l.user_id IS DISTINCT FROM t.user_id) TO STDOUT WITH CSV" \
    || { echo "$S: author-mismatch query failed" >&2; FAILED=1; }
done
exit "$FAILED"
