#!/bin/bash
# Turns the SidewalkAI validator's Disagree votes into Unsure votes and recomputes what they decided (SidewalkWebpage
# #5793): each affected label's agree/disagree/unsure counts and `correct`, then its labeler's own_labels_validated,
# accuracy and high_quality. Runs ON the database host, as your own peer-auth role. Runbook and rationale:
# docs/ai-disagree-reversal-2026-10.md.
#
# The default is a dry run that writes nothing: one CSV row per city of what the conversion would change. --apply
# makes the change, one transaction per city, after writing a backup of every vote row it is about to touch.
#
# The recount is the app's own SQL, ported from SidewalkWebpage (7215f10e0): LabelTable.recalculateValidationCountsWhere
# with FilteredTables.isVerdictVote, UserStatTable.updateAccuracyWhere over FilteredTables.accuracyLabels, and
# UserStatTable.computedHighQuality as evolution 347 writes it in SQL. The dry run reads the same CTEs with the AI's
# Disagree votes taken as Unsure, so its numbers are what --apply changes. label_ai_assessment is left alone: it keeps
# the AI's raw verdict and points at the converted vote through label_validation_id.
#
# Usage: ai_disagree_to_unsure.sh [-p | -t] [--cities a,b,...] [--apply --backup-dir <dir> [--yes]]
#   -p / -t        The prod (default: sidewalk_prod, port 5434) or test (sidewalk_test, port 6432) database. DB_NAME,
#                  DB_PORT and PSQL (default `psql`) override them, e.g. for a rehearsal on a dev database.
#   --cities       Comma-separated schema suffixes (seattle,newport_ky) or schema names. Default: every sidewalk_*
#                  schema except sidewalk_login and *backup*, as analysis/pull_ai_assessments.sh reads them.
#   --apply        Write. Requires --backup-dir. Without it nothing is written, whatever else is given.
#   --backup-dir   Where --apply puts a run directory (run-<UTC time>) holding, per city, the vote rows it changed
#                  (<schema>_ai_disagree_votes.csv) and the before/after report rows (<schema>_report.csv).
#   --yes          Skip the confirmation prompt.
# Output: CSV on stdout, one row per city and phase plus TOTAL rows; progress, warnings and errors on stderr. Warns
# and continues past a city that fails, then exits 1 naming the failures. The output names private cities: keep it
# out of the repo.
set -o pipefail
set -u

AI='51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the SidewalkAI validator account

# The newest column the ported SQL reads is label.deleted_source (evolution 400, used by accuracyLabels). A schema
# behind that runs an older release whose rules this port does not describe, so it is skipped rather than guessed at.
MIN_EVOLUTION=400

PSQL=${PSQL:-psql}
db_name=${DB_NAME:-sidewalk_prod}
port=${DB_PORT:-5434}

# Every connection is read-only (the guard the rest of Project Sidewalk's server scripts run under), except the one
# write transaction per city, which lifts it with SET TRANSACTION READ WRITE. libpq's `options=` in a connection
# string REPLACES PGOPTIONS rather than adding to it, so every connection that sets its search_path that way restates
# the guard (RO_OPTION). Timeouts as in pull_ai_assessments.sh, plus a lock timeout so the write never queues for long
# behind the app.
RO_OPTION="-c default_transaction_read_only=on -c statement_timeout=600000"
export PGOPTIONS="$RO_OPTION"
WRITE_OPTIONS="$RO_OPTION -c lock_timeout=10000"

usage() {
  sed -n '2,/^set -o/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' | sed -n '/^Usage/,$p'
}

apply=0; yes=0; backup_dir=''; cities_arg=''
while (( $# > 0 )); do
  case $1 in
    -p) db_name=sidewalk_prod; port=5434 ;;
    -t) db_name=sidewalk_test; port=6432 ;;
    --apply) apply=1 ;;
    --yes) yes=1 ;;
    --backup-dir) [[ $# -ge 2 ]] || { echo "ERROR: --backup-dir needs a directory." >&2; exit 1; }
                  backup_dir=$2; shift ;;
    --cities) [[ $# -ge 2 ]] || { echo "ERROR: --cities needs a list." >&2; exit 1; }
              cities_arg=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'." >&2; usage >&2; exit 1 ;;
  esac
  shift
done
if (( apply )) && [[ -z "$backup_dir" ]]; then
  echo "ERROR: --apply needs --backup-dir <dir>: the vote rows are backed up there before anything is written." >&2
  exit 1
fi
if (( ! apply )) && { (( yes )) || [[ -n "$backup_dir" ]]; }; then
  echo "NOTE: --yes / --backup-dir only matter with --apply; this is a dry run and writes nothing." >&2
fi

# psql on a read-only connection to one schema. -X keeps a ~/.psqlrc (a SET search_path, \timing) out of it.
# $PSQL is split on purpose, so it can be a command with arguments (e.g. docker exec ... psql).
# shellcheck disable=SC2086
city_psql() {
  local schema=$1; shift
  $PSQL -X -q -At -v ON_ERROR_STOP=1 -p "$port" "dbname=$db_name options='$RO_OPTION -c search_path=$schema,public'" "$@"
}
# shellcheck disable=SC2086
db_psql() {
  $PSQL -X -q -At -v ON_ERROR_STOP=1 -d "$db_name" -p "$port" "$@"
}

# ---------------------------------------------------------------------------------------------------------------------
# The shared CTEs. With $1 = 1 the AI's Disagree votes are read as Unsure (the dry run, and the expected figures for
# --apply); with $1 = 0 they are read as stored, which after the conversion is the app's own recount, verbatim.
ctes() {
  local remap=$1 dis_filter unsure_filter
  if (( remap )); then
    dis_filter="label_validation.validation_result = 'Disagree' AND label_validation.user_id <> '$AI'"
    unsure_filter="label_validation.validation_result = 'Unsure'
                       OR (label_validation.validation_result = 'Disagree' AND label_validation.user_id = '$AI')"
  else
    dis_filter="label_validation.validation_result = 'Disagree'"
    unsure_filter="label_validation.validation_result = 'Unsure'"
  fi
  cat <<EOSQL
WITH
-- Every label the AI has voted on, whatever the vote: LabelTable.recalculateValidationCounts(Some(validatorId)).
scope AS (
    SELECT DISTINCT label_id FROM label_validation WHERE user_id = '$AI'
),
-- LabelTable.recalculateValidationCountsWhere. A vote counts when FilteredTables.isVerdictVote says so: not by the
-- label's author, on the label's current type, and not by an excluded user.
counts AS (
    SELECT label.label_id,
           COUNT(*) FILTER (WHERE label_validation.validation_result = 'Agree') AS n_agree,
           COUNT(*) FILTER (WHERE $dis_filter) AS n_disagree,
           COUNT(*) FILTER (WHERE $unsure_filter) AS n_unsure
    FROM label
    LEFT JOIN label_validation ON label.label_id = label_validation.label_id
        AND label_validation.user_id <> label.user_id AND label_validation.label_type = label.label_type
        AND NOT EXISTS (SELECT 1 FROM user_stat
                        WHERE user_stat.user_id = label_validation.user_id AND user_stat.excluded)
    WHERE label.label_id IN (SELECT label_id FROM scope)
    GROUP BY label.label_id
),
-- LabelTable.majorityCorrect: TRUE if agrees outnumber disagrees, FALSE if the reverse, NULL on a tie.
recount AS (
    SELECT label.label_id, label.user_id, label.deleted, label.tutorial,
           label.agree_count AS old_agree, label.disagree_count AS old_disagree, label.unsure_count AS old_unsure,
           label.correct AS old_correct,
           counts.n_agree::INT AS new_agree, counts.n_disagree::INT AS new_disagree, counts.n_unsure::INT AS new_unsure,
           CASE WHEN counts.n_agree > counts.n_disagree THEN TRUE
                WHEN counts.n_disagree > counts.n_agree THEN FALSE END AS new_correct
    FROM label
    INNER JOIN counts ON label.label_id = counts.label_id
),
recount_changed AS (
    SELECT * FROM recount
    WHERE (old_agree, old_disagree, old_unsure, old_correct)
          IS DISTINCT FROM (new_agree, new_disagree, new_unsure, new_correct)
),
-- UserStatTable.updateAccuracyForLabelersValidatedBy: the labelers of every label the AI voted on.
labelers AS (
    SELECT DISTINCT user_id FROM recount
),
-- UserStatTable.updateAccuracyWhere over FilteredTables.accuracyLabels (LabelTable.countsTowardAccuracySql, not
-- tutorial, neither the label's nor its audit's street the tutorial street), reading each recounted label's new
-- \`correct\` in place of the stored one.
accuracy AS (
    SELECT label.user_id,
           CAST(SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS FLOAT)
               / NULLIF(SUM(CASE WHEN correct THEN 1 ELSE 0 END) + SUM(CASE WHEN NOT correct THEN 1 ELSE 0 END), 0)
               AS new_accuracy,
           COUNT(CASE WHEN correct IS NOT NULL THEN 1 END) AS new_validated_count
    FROM (
        SELECT label.user_id, label.deleted, label.deleted_source, label.tutorial, label.street_edge_id,
               label.audit_task_id,
               CASE WHEN recount.label_id IS NULL THEN label.correct ELSE recount.new_correct END AS correct
        FROM label
        LEFT JOIN recount ON label.label_id = recount.label_id
        WHERE label.user_id IN (SELECT user_id FROM labelers)
    ) AS label
    INNER JOIN audit_task ON label.audit_task_id = audit_task.audit_task_id
    WHERE (NOT label.deleted OR (label.deleted_source <> 'Explore' AND label.correct = FALSE))
        AND label.tutorial = FALSE
        AND label.street_edge_id <> (SELECT tutorial_street_edge_id FROM config)
        AND audit_task.street_edge_id <> (SELECT tutorial_street_edge_id FROM config)
    GROUP BY label.user_id
),
-- updateAccuracyWhere's filter picks the user_stat rows whose stored values differ (accuracy compared at 3 decimals)
-- and then rewrites every row of those users (user_id is not unique in user_stat, #4776). Users it skips keep their
-- stored own_labels_validated and accuracy, even if the unrounded accuracy moved.
stat AS (
    SELECT user_stat.user_stat_id, user_stat.user_id, user_stat.excluded, user_stat.high_quality AS old_hq,
           user_stat.high_quality_manual, user_stat.meters_audited, user_stat.labels_per_meter,
           user_stat.accuracy AS old_accuracy, user_stat.own_labels_validated AS old_validated,
           COALESCE(accuracy.new_validated_count, 0)::INT AS calc_validated, accuracy.new_accuracy AS calc_accuracy,
           (user_stat.own_labels_validated <> COALESCE(accuracy.new_validated_count, 0)
               OR (user_stat.accuracy IS NULL AND accuracy.new_accuracy IS NOT NULL)
               OR (user_stat.accuracy IS NOT NULL AND accuracy.new_accuracy IS NULL)
               OR (user_stat.accuracy IS NOT NULL AND accuracy.new_accuracy IS NOT NULL
                   AND ROUND(user_stat.accuracy::NUMERIC, 3) <> ROUND(accuracy.new_accuracy::NUMERIC, 3)))
               AS row_differs
    FROM user_stat
    LEFT JOIN accuracy ON user_stat.user_id = accuracy.user_id
    WHERE user_stat.user_id IN (SELECT user_id FROM labelers)
),
stat_new AS (
    SELECT s.*,
           CASE WHEN s.rewritten THEN s.calc_validated ELSE s.old_validated END AS new_validated,
           CASE WHEN s.rewritten THEN s.calc_accuracy ELSE s.old_accuracy END AS new_accuracy
    FROM (SELECT stat.*, bool_or(stat.row_differs) OVER (PARTITION BY stat.user_id) AS rewritten FROM stat) AS s
),
-- UserStatTable.computedHighQuality, in evolution 347's SQL (0.6 compares as a double, as the Scala requires).
-- updateUserQualityForLabelersValidatedBy never rewrites an excluded row, so those keep their flag.
quality AS (
    SELECT stat_new.*,
           CASE WHEN excluded THEN old_hq
                ELSE COALESCE(high_quality_manual, TRUE)
                     AND (COALESCE(high_quality_manual, FALSE)
                          OR ((meters_audited = 0 OR COALESCE(labels_per_meter, 5) > 0.0375)
                              AND (COALESCE(new_accuracy, 1.0) > 0.6 OR new_validated < 50)))
           END AS new_hq
    FROM stat_new
),
labeler_quality AS (
    SELECT user_id, bool_or(old_hq) AS old_hq, bool_or(new_hq) AS new_hq, bool_or(excluded) AS excluded
    FROM quality GROUP BY user_id
)
EOSQL
}

# The map/Gallery visibility rule (LabelTable.getGalleryLabelsQuery): the labeler is high quality or the label is
# validated correct, the label is not validated incorrect, and it passes the disagree gate.
visible() {  # $1 = high-quality expression, $2 = prefix of the count/correct columns (old_ or new_)
  echo "(($1 OR $2correct IS TRUE) AND $2correct IS DISTINCT FROM FALSE
         AND ($2disagree < 3 OR $2disagree < 2 * $2agree))"
}

REPORT_HEADER="phase,schema,ai_disagree_rows,on_deleted_or_tutorial,labels_recounted,false_to_null,null_to_true,\
correct_unchanged,other_transition,becomes_visible_now,becomes_visible_after_quality_flip,stays_hidden_low_quality,\
becomes_hidden,labelers_touched,accuracy_changed,quality_low_to_high,quality_high_to_low,no_longer_judged,\
stat_rows_rewritten,quality_rows_changed,quality_flip_unhides_all,quality_flip_hides_all"

# One CSV row for the current schema (no phase column), preceded by the connection's transaction_read_only, which the
# caller checks: the dry run must prove it ran read-only, not assume it.
report_sql() {
  ctes 1
  cat <<EOSQL
, judged AS (  -- labels the map can show at all: not deleted, not tutorial, labeler not excluded
    SELECT recount_changed.*, COALESCE(labeler_quality.old_hq, FALSE) AS old_hq,
           COALESCE(labeler_quality.new_hq, FALSE) AS new_hq
    FROM recount_changed
    LEFT JOIN labeler_quality ON recount_changed.user_id = labeler_quality.user_id
    WHERE NOT recount_changed.deleted AND NOT recount_changed.tutorial
        AND NOT COALESCE(labeler_quality.excluded, FALSE)
), vis AS (
    SELECT judged.*,
           $(visible old_hq old_) AS before,
           $(visible old_hq new_) AS after_now,
           $(visible new_hq new_) AS after_flip
    FROM judged
), per_user AS (
    SELECT user_id, bool_or(rewritten) AS rewritten,
           bool_or(new_validated IS DISTINCT FROM old_validated OR new_accuracy IS DISTINCT FROM old_accuracy)
               AS changed,
           MAX(old_validated) AS old_validated, MAX(new_validated) AS new_validated
    FROM quality GROUP BY user_id
), flip_labels AS (  -- every map-eligible label of a labeler whose high_quality flips, AI-voted or not
    SELECT labeler_quality.old_hq, labeler_quality.new_hq,
           label.agree_count AS old_agree, label.disagree_count AS old_disagree, label.correct AS old_correct,
           COALESCE(recount.new_agree, label.agree_count) AS new_agree,
           COALESCE(recount.new_disagree, label.disagree_count) AS new_disagree,
           CASE WHEN recount.label_id IS NULL THEN label.correct ELSE recount.new_correct END AS new_correct
    FROM label
    INNER JOIN labeler_quality ON label.user_id = labeler_quality.user_id
        AND NOT labeler_quality.excluded AND labeler_quality.old_hq <> labeler_quality.new_hq
    LEFT JOIN recount ON label.label_id = recount.label_id
    WHERE NOT label.deleted AND NOT label.tutorial
), flip_vis AS (
    SELECT old_hq, new_hq, $(visible old_hq old_) AS before, $(visible new_hq new_) AS after_flip FROM flip_labels
)
SELECT current_setting('transaction_read_only'),
       current_schema(),
       (SELECT COUNT(*) FROM label_validation WHERE user_id = '$AI' AND validation_result = 'Disagree'),
       (SELECT COUNT(*) FROM label_validation INNER JOIN label ON label_validation.label_id = label.label_id
        WHERE label_validation.user_id = '$AI' AND label_validation.validation_result = 'Disagree'
            AND (label.deleted OR label.tutorial)),
       (SELECT COUNT(*) FROM recount_changed),
       (SELECT COUNT(*) FROM recount_changed WHERE old_correct IS FALSE AND new_correct IS NULL),
       (SELECT COUNT(*) FROM recount_changed WHERE old_correct IS NULL AND new_correct IS TRUE),
       (SELECT COUNT(*) FROM recount_changed WHERE old_correct IS NOT DISTINCT FROM new_correct),
       (SELECT COUNT(*) FROM recount_changed
        WHERE old_correct IS DISTINCT FROM new_correct
            AND NOT (old_correct IS FALSE AND new_correct IS NULL)
            AND NOT (old_correct IS NULL AND new_correct IS TRUE)),
       (SELECT COUNT(*) FROM vis WHERE NOT before AND after_now),
       (SELECT COUNT(*) FROM vis WHERE NOT before AND NOT after_now AND after_flip),
       (SELECT COUNT(*) FROM vis
        WHERE NOT after_flip AND NOT new_hq AND old_correct IS FALSE AND new_correct IS DISTINCT FROM FALSE),
       (SELECT COUNT(*) FROM vis WHERE before AND NOT after_flip),
       (SELECT COUNT(*) FROM labelers),
       (SELECT COUNT(*) FROM per_user WHERE rewritten AND changed),
       (SELECT COUNT(*) FROM labeler_quality WHERE NOT excluded AND NOT old_hq AND new_hq),
       (SELECT COUNT(*) FROM labeler_quality WHERE NOT excluded AND old_hq AND NOT new_hq),
       (SELECT COUNT(*) FROM per_user WHERE old_validated >= 50 AND new_validated < 50),
       (SELECT COUNT(*) FROM stat_new WHERE rewritten),
       (SELECT COUNT(*) FROM quality WHERE NOT excluded AND new_hq IS DISTINCT FROM old_hq),
       (SELECT COUNT(*) FROM flip_vis WHERE NOT old_hq AND new_hq AND NOT before AND after_flip),
       (SELECT COUNT(*) FROM flip_vis WHERE old_hq AND NOT new_hq AND before AND NOT after_flip);
EOSQL
}

# Runs the dry-run report on one schema; prints "<schema>,<counts...>" (no phase) on success.
report_city() {
  local schema=$1 out ro
  out=$(city_psql "$schema" -F, -c "$(report_sql)") || return 1
  ro=${out%%,*}
  if [[ "$ro" != "on" ]]; then
    echo "  REFUSING: the report connection for $schema was not read-only (transaction_read_only=$ro)." >&2
    return 1
  fi
  echo "${out#*,}"
}

# Checks the dry-run invariants on a report row; warnings only (a label already off its votes before the AI's vote
# is touched shows up as other_transition or as labels_recounted > ai_disagree_rows, which the recount then fixes).
check_invariants() {
  local row=$1 schema f
  IFS=, read -r -a f <<< "$row"
  schema=${f[0]}
  # f: 1 ai_disagree_rows, 3 labels_recounted, 4 false_to_null, 5 null_to_true, 6 correct_unchanged, 7 other
  if (( f[4] + f[5] + f[6] + f[7] != f[3] )); then
    echo "  WARNING: $schema transitions do not add up to labels_recounted ($row)." >&2
    return 1
  fi
  if (( f[1] < f[3] )); then
    echo "  WARNING: $schema would recount ${f[3]} labels from ${f[1]} AI Disagree rows: some labels' stored counts" \
         "were already off their votes, and the recount fixes those too." >&2
  fi
  if (( f[7] > 0 )); then
    echo "  WARNING: $schema has ${f[7]} label(s) whose correct would change in a way removing one Disagree cannot" \
         "explain (stored counts out of date)." >&2
  fi
  return 0
}

# ---------------------------------------------------------------------------------------------------------------------
# The write, for one schema. $2 = the backup CSV, $3 = its row count, $4 = the dry-run row taken just before.
apply_city() {
  local schema=$1 csv=$2 n_backup=$3 row=$4 f
  IFS=, read -r -a f <<< "$row"
  # f: 3 labels_recounted, 17 stat_rows_rewritten, 18 quality_rows_changed
  # shellcheck disable=SC2086
  $PSQL -X -q -v ON_ERROR_STOP=1 -p "$port" \
    "dbname=$db_name options='$WRITE_OPTIONS -c search_path=$schema,public'" <<EOSQL
BEGIN;
SET TRANSACTION READ WRITE;

CREATE TEMP TABLE ai_disagree_backup (
    schema_name         TEXT,
    label_validation_id INTEGER,
    label_id            INTEGER,
    label_type          TEXT,
    validation_result   TEXT,
    end_timestamp       TEXT
) ON COMMIT DROP;
\\copy ai_disagree_backup FROM '$csv' WITH (FORMAT csv, HEADER true)

DO \$\$
DECLARE
    n INTEGER;
    -- "Wildly off": more than 10 rows and 2% from the dry run taken moments before. Votes cast in between move the
    -- counts a little; a different number than that means the SQL is not doing what the dry run said.
    min_slack CONSTANT INTEGER := 10;
BEGIN
    IF (SELECT COUNT(*) FROM ai_disagree_backup WHERE schema_name <> current_schema()) > 0 THEN
        RAISE EXCEPTION 'the backup file is for another schema';
    END IF;
    IF (SELECT COUNT(*) FROM ai_disagree_backup) <> $n_backup THEN
        RAISE EXCEPTION 'the backup file did not load whole';
    END IF;

    -- Step 1: the AI's Disagree votes, exactly the rows in the backup, become Unsure.
    UPDATE label_validation SET validation_result = 'Unsure'
    FROM ai_disagree_backup
    WHERE label_validation.label_validation_id = ai_disagree_backup.label_validation_id
        AND label_validation.user_id = '$AI' AND label_validation.validation_result = 'Disagree';
    GET DIAGNOSTICS n = ROW_COUNT;
    RAISE NOTICE 'step 1: % AI vote row(s) Disagree -> Unsure (backup has $n_backup)', n;
    IF n <> $n_backup THEN
        RAISE EXCEPTION 'step 1 changed % rows, not the % in the backup', n, $n_backup;
    END IF;

    -- Step 2: LabelTable.recalculateValidationCountsWhere on every label the AI voted on.
    $(ctes 0)
    UPDATE label
    SET (agree_count, disagree_count, unsure_count, correct) = (
        recount_changed.new_agree, recount_changed.new_disagree, recount_changed.new_unsure,
        recount_changed.new_correct)
    FROM recount_changed
    WHERE label.label_id = recount_changed.label_id;
    GET DIAGNOSTICS n = ROW_COUNT;
    RAISE NOTICE 'step 2: % label(s) recounted (dry run: ${f[3]})', n;
    IF abs(n - ${f[3]}) > GREATEST(min_slack, ${f[3]} / 50) THEN
        RAISE EXCEPTION 'step 2 recounted % labels, the dry run said ${f[3]}', n;
    END IF;

    -- Step 3: UserStatTable.updateAccuracyWhere for those labels' labelers. The labels are recounted now, so the
    -- CTEs read the stored \`correct\` (no remapping) exactly as the app would.
    $(ctes 0)
    UPDATE user_stat
    SET own_labels_validated = stat_new.calc_validated, accuracy = stat_new.calc_accuracy
    FROM stat_new
    WHERE user_stat.user_stat_id = stat_new.user_stat_id AND stat_new.rewritten;
    GET DIAGNOSTICS n = ROW_COUNT;
    RAISE NOTICE 'step 3: % user_stat row(s) given a new accuracy (dry run: ${f[17]})', n;
    IF abs(n - ${f[17]}) > GREATEST(min_slack, ${f[17]} / 50) THEN
        RAISE EXCEPTION 'step 3 rewrote % user_stat rows, the dry run said ${f[17]}', n;
    END IF;

    -- Step 4: high_quality for the same labelers, from the values step 3 stored.
    $(ctes 0)
    UPDATE user_stat SET high_quality = quality.new_hq
    FROM quality
    WHERE user_stat.user_stat_id = quality.user_stat_id AND quality.new_hq IS DISTINCT FROM quality.old_hq;
    GET DIAGNOSTICS n = ROW_COUNT;
    RAISE NOTICE 'step 4: % user_stat row(s) with a new high_quality (dry run: ${f[18]})', n;
    IF abs(n - ${f[18]}) > GREATEST(min_slack, ${f[18]} / 50) THEN
        RAISE EXCEPTION 'step 4 changed % high_quality flags, the dry run said ${f[18]}', n;
    END IF;
END
\$\$;

COMMIT;
EOSQL
}

# ---------------------------------------------------------------------------------------------------------------------
ro=$(db_psql -c "SHOW default_transaction_read_only") || { echo "ERROR: cannot reach $db_name on port $port." >&2; exit 1; }
if [[ "$ro" != "on" ]]; then
  echo "ERROR: connections are not read-only by default (got '$ro'); refusing to run." >&2
  exit 1
fi

all_schemas=$(db_psql -c "SELECT schema_name FROM information_schema.schemata
  WHERE schema_name LIKE 'sidewalk_%' AND schema_name NOT IN ('sidewalk_login') AND schema_name NOT LIKE '%backup%'
  ORDER BY 1") || { echo "ERROR: could not list schemas." >&2; exit 1; }

schemas=()
if [[ -n "$cities_arg" ]]; then
  IFS=, read -r -a wanted <<< "$cities_arg"
  for c in "${wanted[@]}"; do
    [[ "$c" == sidewalk_* ]] || c=sidewalk_$c
    if ! grep -qx -- "$c" <<< "$all_schemas"; then
      echo "ERROR: no schema $c in $db_name (or it is excluded)." >&2
      exit 1
    fi
    schemas+=("$c")
  done
else
  while IFS= read -r s; do [[ -n "$s" ]] && schemas+=("$s"); done <<< "$all_schemas"
fi

failed=()
eligible() {  # a schema with the tables and an evolution this port describes
  local schema=$1 evo
  evo=$(city_psql "$schema" -c "SELECT CASE WHEN to_regclass('play_evolutions') IS NULL
                                              OR to_regclass('label_validation') IS NULL THEN -1
                                         ELSE (SELECT COALESCE(max(id), 0) FROM play_evolutions
                                               WHERE state = 'applied') END") || return 1
  if (( evo < MIN_EVOLUTION )); then
    echo "  skipping $schema: evolution $evo (needs $MIN_EVOLUTION or later, or no label_validation)." >&2
    return 2
  fi
}

sum_rows() {  # TOTAL row over "phase,schema,..." rows of one phase on stdin
  awk -F, -v phase="$1" 'BEGIN { OFS = "," }
    $1 == phase { n = NF; for (i = 3; i <= NF; i++) s[i] += $i }
    END { if (n) { line = phase ",TOTAL"; for (i = 3; i <= n; i++) line = line "," s[i]; print line } }'
}

echo "Database: $db_name (port $port); ${#schemas[@]} schema(s); mode: $( ((apply)) && echo APPLY || echo dry run)" >&2
echo "$REPORT_HEADER"

if (( ! apply )); then
  rows=''
  for s in "${schemas[@]}"; do
    echo "== $s" >&2
    eligible "$s"; rc=$?
    (( rc == 2 )) && continue
    if (( rc != 0 )) || ! row=$(report_city "$s"); then
      echo "WARNING: dry run failed for $s." >&2; failed+=("$s"); continue
    fi
    check_invariants "$row" || failed+=("$s")
    echo "dry_run,$row"
    rows+="dry_run,$row"$'\n'
  done
  printf '%s' "$rows" | sum_rows dry_run
else
  if [[ "$backup_dir" == *"'"* ]]; then
    echo "ERROR: the backup path may not contain a single quote." >&2; exit 1
  fi
  run_dir=$backup_dir/run-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$run_dir" || { echo "ERROR: cannot create $run_dir." >&2; exit 1; }
  run_dir=$(cd "$run_dir" && pwd)
  if (( ! yes )); then
    if [[ ! -t 0 ]]; then
      echo "ERROR: --apply without --yes needs a terminal to confirm on." >&2; exit 1
    fi
    echo "About to WRITE to ${#schemas[@]} schema(s) in $db_name on port $port; backups go to $run_dir." >&2
    read -r -p "Type the database name ($db_name) to go on: " answer
    [[ "$answer" == "$db_name" ]] || { echo "Not confirmed; nothing written." >&2; exit 1; }
  fi
  before_rows=''; after_rows=''
  for s in "${schemas[@]}"; do
    echo "== $s" >&2
    eligible "$s"; rc=$?
    (( rc == 2 )) && continue
    if (( rc != 0 )) || ! row=$(report_city "$s"); then
      echo "WARNING: dry run failed for $s; nothing written there." >&2; failed+=("$s"); continue
    fi
    if ! check_invariants "$row"; then
      echo "WARNING: $s fails the dry-run invariants; nothing written there." >&2; failed+=("$s"); continue
    fi
    echo "before,$row"
    before_rows+="before,$row"$'\n'
    report_file=$run_dir/${s}_report.csv
    { echo "$REPORT_HEADER"; echo "before,$row"; } > "$report_file"
    n_disagree=$(cut -d, -f2 <<< "$row")
    if (( n_disagree == 0 )); then
      echo "  nothing to convert in $s." >&2; continue
    fi
    csv=$run_dir/${s}_ai_disagree_votes.csv
    if ! city_psql "$s" -c "COPY (
          SELECT current_schema() AS schema_name, label_validation_id, label_id, label_type, validation_result,
                 end_timestamp
          FROM label_validation WHERE user_id = '$AI' AND validation_result = 'Disagree'
          ORDER BY label_validation_id) TO STDOUT WITH (FORMAT csv, HEADER true)" > "$csv"; then
      echo "WARNING: backup failed for $s; nothing written there." >&2; failed+=("$s"); continue
    fi
    n_backup=$(( $(wc -l < "$csv") - 1 ))
    echo "  backed up $n_backup vote row(s) to $csv" >&2
    if ! apply_city "$s" "$csv" "$n_backup" "$row" >&2; then
      echo "WARNING: apply failed for $s; its transaction rolled back, nothing was written there." >&2
      failed+=("$s"); continue
    fi
    if ! after=$(report_city "$s"); then
      echo "WARNING: $s was written, but the after-report failed: rerun the dry run for it." >&2
      failed+=("$s"); continue
    fi
    echo "after,$after"
    after_rows+="after,$after"$'\n'
    echo "after,$after" >> "$report_file"
    if [[ "$(cut -d, -f2,4 <<< "$after")" != "0,0" ]]; then
      echo "  NOTE: $s still has AI Disagree rows or recounts after the apply (votes cast since the backup?):" \
           "rerun for it once the AI has stopped voting Disagree." >&2
    fi
  done
  printf '%s' "$before_rows" | sum_rows before
  printf '%s' "$after_rows" | sum_rows after
  echo "Backups and per-city reports: $run_dir" >&2
fi

if (( ${#failed[@]} > 0 )); then
  echo "Done, but these schemas FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "Done." >&2
