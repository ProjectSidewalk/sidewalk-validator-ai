# Reversing the AI's Disagree votes: runbook (October 2026)

[SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793) agreed to undo the SidewalkAI
validator's Disagree votes. Where the AI was a label's only voter, or broke a tie between humans, its vote alone set
`label.correct = FALSE`. That hides the label from the map and counts against the labeler's accuracy, and humans
who voted on the same labels sided with those Disagrees only about half the time. This doc is the runbook for the
database side. The scripts are `analysis/ai_disagree_to_unsure.sh` (dry run and update) and
`analysis/ai_disagree_census.sh` (the census). The test is `tests/test_ai_disagree_to_unsure.py`.

**Status:** the scripts have been tested only on a throwaway Postgres 16 with a minimal schema (below). Nobody has
run them on the test instance (6432) or on production yet. Until someone does, the only numbers here are the
2026-10-24 census.

## The numbers so far (census of 2026-10-24, before any change)

From a read-only query on production, all city schemas, non-deleted, non-tutorial labels, posted
[on #5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793#issuecomment-6389544178):

| | labels |
|---|---|
| AI Disagree votes (52 cities, 2025-08-13 to 2026-10-24) | 91,506 |
| of those, label hidden (`correct = FALSE`) | 67,668 |
| no human vote: the AI decided alone | 29,800 |
| humans tied: the AI broke the tie | 7,371 |
| human majority agreed with the AI, where humans voted | 56% |

The census SQL from that day was not saved. `ai_disagree_census.sh` recreates it from the posted description, and
its header comment gives each definition. It counts only an AI vote on the label's current type, and it leaves ties
out of the agreement rate. A rerun can therefore differ a little from the table even before counting new votes.
This doc will name cities only after the dry run has been run and each city has been checked as public at that time.

## Three steps, and who does each

1. **Stop new AI Disagree votes** (SidewalkWebpage; a code or config change, not in this repo). The AI's vote is
   set in `AiService.scala:181-186`. The vote is the model's verdict when its served accuracy is at least
   `city-params.ai-validation-min-accuracy.<city>`, and Unsure otherwise. `city-params.ai-validation-enabled.<city>`
   (both in `conf/cityparams.conf`) turns AI votes off altogether. Raising the minimum above 1.0 makes every AI vote
   Unsure. That includes its Agrees, so stopping only the Disagrees needs a small code change. **Do this first.**
   Otherwise the AI keeps casting Disagrees after step 2, and the script's `after` row shows them.
2. **Convert the AI's existing Disagree votes to Unsure** (this script, `--apply`). This keeps the record that the AI
   looked at the label, as Mikey suggested on the issue.
3. **Recompute what those votes decided**: each label's counts and `correct`, then each labeler's
   `own_labels_validated`, `accuracy` and `high_quality`. The same script does this in the same transaction as step 2.

Steps 2 and 3 need write access on makelab1 (Mikey, or Jon). The scripts run there as your own peer-auth role.

## What the script does

`ai_disagree_to_unsure.sh` handles one city schema at a time. It reads every `sidewalk_*` schema except
`sidewalk_login` and `*backup*`, the same rule as `pull_ai_assessments.sh`, unless `--cities` narrows the list. It
skips any schema before evolution 400 (`label.deleted_source`), because this port does not describe an older
release.

**The dry run (default)** writes nothing. Every connection opens with `default_transaction_read_only=on`. The guard
is restated in each connection string that sets a `search_path`, because libpq's `options=` replaces PGOPTIONS.
Each report row also carries the connection's own `transaction_read_only`, and the script refuses the row unless it
reads `on`. The dry run issues no UPDATE. It computes the new state with CTEs in which the AI's Disagree votes are
read as Unsure.

**`--apply`** does this for each city:

1. It reruns the dry run and stores that row as `before`.
2. It writes the AI's Disagree rows to the backup CSV, using a read-only connection: `schema_name,
   label_validation_id, label_id, label_type, validation_result, end_timestamp`.
3. It opens one transaction (`SET TRANSACTION READ WRITE` lifts the read-only default for that transaction alone,
   with a 10 s lock timeout) and runs four steps. Each step prints its row count to stderr and is checked against
   the dry run:
   1. `label_validation`: exactly the rows in the backup that are still AI Disagree become `Unsure`. The city aborts
      if the count differs from the backup's row count.
   2. Labels: the port of `LabelTable.recalculateValidationCountsWhere` on every label the AI voted on. This is what
      `recalculateValidationCounts(Some(<AI user_id>))` would do. A vote counts under `FilteredTables.isVerdictVote`:
      it is not by the label's author, it is on the label's current type, and the validator is not excluded.
      `correct` is TRUE when agrees outnumber disagrees, FALSE for the reverse, and NULL on a tie.
   3. `user_stat.own_labels_validated` and `accuracy` for those labels' labelers: the port of
      `UserStatTable.updateAccuracyWhere` over `FilteredTables.accuracyLabels`. A label counts if it is not deleted,
      or was deleted outside Explore while incorrect. It must not be a tutorial label, and neither the label's
      street nor its audit's street may be the tutorial street (`config.tutorial_street_edge_id`). As in the app, a
      user is rewritten only when the stored count differs or the accuracy differs at 3 decimals. All of that
      user's `user_stat` rows are rewritten, because `user_id` is not unique there
      ([SidewalkWebpage#4776](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4776)).
   4. `user_stat.high_quality` for the same labelers: `UserStatTable.computedHighQuality`, as evolution 347 writes
      it. `high_quality_manual` wins when set. Otherwise a labeler is high quality when
      `(meters_audited = 0 OR labels_per_meter > 0.0375) AND (accuracy > 0.6 OR own_labels_validated < 50)`.
      Excluded rows are not touched (`updateUserQualityForLabelersValidatedBy`).

   Steps 2 to 4 abort the city (ROLLBACK) if their count is off the dry run by more than 10 rows and 2%. Votes cast
   between the dry run and the write can move the counts a little. A bigger difference means the SQL is not doing
   what the dry run said.
4. After COMMIT it reruns the dry run as the `after` row. If everything went through, `ai_disagree_rows`,
   `labels_recounted`, `accuracy_changed`, the quality flips, `stat_rows_rewritten` and `quality_rows_changed` are
   all 0.

**Why the script recomputes `high_quality` itself:** the nightly `UserStatActor` (01:30, `ScheduledJobs.scala:36`)
recomputes `accuracy` for everyone, but `high_quality` only for users active in the last 36 hours
(`UserStatActor.scala:70-74`). A labeler whom the AI pushed below 60% and who has since stopped contributing would
otherwise stay low quality, and their labels would stay hidden.

**Left alone:** `label_ai_assessment` keeps the AI's raw verdict. It links to the converted vote through
`label_validation_id`, and that link is the record of the conversion. The AI never edits labels, so there are no
`label_history` or `label_edit` rows to change. No AI Agree votes are added and nothing is re-gated under the new
calibration maps. Those can come later as a separate pass.

### Report columns (one row per city and phase, plus a TOTAL row per phase)

| column | meaning |
|---|---|
| `phase` | `dry_run` (default mode), or `before` / `after` (`--apply`) |
| `schema` | city schema; `label_id` is a per-city serial, so every row carries it |
| `ai_disagree_rows` | AI Disagree vote rows to convert |
| `on_deleted_or_tutorial` | of those, rows on deleted or tutorial labels (converted too, so "the AI says Disagree nowhere" holds everywhere) |
| `labels_recounted` | labels whose `(agree, disagree, unsure, correct)` changes |
| `false_to_null` | `correct` FALSE → NULL: the AI alone, or the AI breaking a tie |
| `null_to_true` | NULL → TRUE: humans ahead by one, and the AI had tied it |
| `correct_unchanged` | counts change but `correct` stays (e.g. the humans also voted it down) |
| `other_transition` | any other change of `correct`. Removing one Disagree cannot cause one, so non-zero means the stored counts were already out of date. The recount fixes them, and the script warns |
| `becomes_visible_now` | non-deleted, non-tutorial labels hidden before and shown after under the visibility rule below, with today's `high_quality` |
| `becomes_visible_after_quality_flip` | shown only once step 4 makes the labeler high quality |
| `stays_hidden_low_quality` | no longer incorrect, but still hidden because the labeler stays low quality |
| `becomes_hidden` | shown before, hidden after (expected 0) |
| `labelers_touched` | labelers of labels the AI voted on |
| `accuracy_changed` | of those, users whose `own_labels_validated` or `accuracy` step 3 rewrites |
| `quality_low_to_high`, `quality_high_to_low` | `high_quality` flips in step 4. high→low should be 0. Either can include a flag that was already stale for another reason, which step 4 corrects as the app would |
| `no_longer_judged` | `own_labels_validated` drops below 50, so accuracy stops counting against the labeler |
| `stat_rows_rewritten`, `quality_rows_changed` | `user_stat` rows steps 3 and 4 write (the figures `--apply` checks itself against) |

The script checks two invariants on every row and warns if either fails. First, `false_to_null + null_to_true +
correct_unchanged + other_transition = labels_recounted`. Second, `ai_disagree_rows ≥ labels_recounted`. The
second can fail only when stored counts were already off their votes. `--apply` refuses a city that fails the
first.

### Which labels the map and Gallery show

The visibility columns use the Gallery's server-side rule (`LabelTable.getGalleryLabelsQuery`, lines 2024-2050):
`(labeler high_quality OR correct = TRUE) AND correct IS DISTINCT FROM FALSE AND (disagree_count < 3 OR
disagree_count < 2 * agree_count)`. The pages' default filters add the following on top of that rule:

- **LabelMap**: the default validation filter shows correct, unsure and unvalidated labels, and hides incorrect
  ones (`psMapUtilities.js`, `filterSidebar.scala.html`). Its high-quality filter is on by default
  (`MapSidebarFilter`), and it hides a low-quality labeler's labels even when they are validated correct. So on the
  map, `becomes_visible_now` is about right, and a label in `becomes_visible_after_quality_flip` appears only after
  step 4.
- **Gallery**: the default is `validationOptions=correct,unvalidated` (`conf/routes`, `GalleryFilter.js`), so
  "unsure" is off. A label whose AI vote becomes Unsure has `correct` NULL with at least one vote, so it counts as
  unsure. It **stays out of the default Gallery** until a human votes it correct. This includes every lone-AI
  label. The dry-run columns do not model this default.
- The public API applies neither page filter.

## Commands

All of these run on makelab1 from a checkout of this repo (or a copy of the one file). Output goes outside the repo,
because it names private cities.

```bash
# 0. Which database on the test instance holds the copy? (Expected: sidewalk_test; check before relying on it.)
psql -X -p 6432 -l

# 1. Rehearse on the test instance: dry run, then apply, then dry run again (expect zeros).
analysis/ai_disagree_to_unsure.sh -t > ~/ai-disagree/test_dry_run.csv
analysis/ai_disagree_to_unsure.sh -t --apply --backup-dir ~/ai-disagree/test > ~/ai-disagree/test_apply.csv
analysis/ai_disagree_to_unsure.sh -t > ~/ai-disagree/test_after.csv
#    If the copy is in another database there: DB_NAME=<name> DB_PORT=6432 analysis/ai_disagree_to_unsure.sh ...

# 2. Dry run on production (read-only; safe at any time).
analysis/ai_disagree_to_unsure.sh > ~/ai-disagree/prod_dry_run.csv
analysis/ai_disagree_census.sh > ~/ai-disagree/prod_census.csv

# 3. Once step 1 (stop new Disagrees) is deployed, apply off-peak. It asks you to type the database name first.
#    Try one city first, e.g. --cities seattle.
analysis/ai_disagree_to_unsure.sh --apply --backup-dir ~/ai-disagree/prod > ~/ai-disagree/prod_apply.csv
```

`--apply` creates `<backup-dir>/run-<UTC time>/` and never overwrites an earlier run. For each city it writes
`<schema>_ai_disagree_votes.csv` and `<schema>_report.csv` (the before and after rows). Each city's transaction locks
the labels and `user_stat` rows it updates, typically for a few seconds. Run it off-peak anyway. A city that fails
rolls back whole. The script then goes on to the next city and exits 1 at the end, naming the cities that failed.

### Checks afterwards

- Rerun the dry run. Every city's `ai_disagree_rows` and `labels_recounted` should be 0. A non-zero
  `ai_disagree_rows` means the AI voted Disagree after the backup was taken, so step 1 is not in place.
- In each city's admin API, `POST /adminapi/recalculateValidationCounts` recounts every label in that city. It
  should return `{"labels_updated": 0}`, or only labels the AI never voted on whose counts had drifted.
- `GET /adminapi/updateUserStats` (with no `hoursCutoff`, it covers every user since the beginning) recomputes
  accuracy and `high_quality` for everyone. It also recomputes audited distance and labels-per-meter, so its
  "updated for N users" can be non-zero for reasons unrelated to this change. Compare it with a run of the same
  call before the apply if you want a clean signal.

### Undoing it

The backup CSV holds every vote row that was converted. To put them back for one city, load the CSV and reverse
step 1, then recount with the endpoints above:

```sql
-- psql on the city schema, READ WRITE; the CSV is the run's <schema>_ai_disagree_votes.csv
CREATE TEMP TABLE b (schema_name TEXT, label_validation_id INT, label_id INT, label_type TEXT,
                     validation_result TEXT, end_timestamp TEXT);
\copy b FROM '<csv>' WITH (FORMAT csv, HEADER true)
UPDATE label_validation SET validation_result = 'Disagree'
FROM b WHERE label_validation.label_validation_id = b.label_validation_id
  AND b.schema_name = current_schema() AND label_validation.validation_result = 'Unsure';
```

Then `POST /adminapi/recalculateValidationCounts` and `GET /adminapi/updateUserStats` restore the counts, accuracy
and quality flags.

## Where the port differs from the Scala, and why

- **The scope is every label the AI voted on.** That is `recalculateValidationCounts(Some(<AI user_id>))`, not just
  the labels with an AI Disagree. Labels with only an AI Agree are recounted too. They change only if their stored
  counts had already drifted, and those changes show up in `other_transition` or `correct_unchanged`.
- **Step 1 converts the rows in the backup, not "every AI Disagree now".** A Disagree cast between the backup and the
  write is left as it is and shows up in the `after` row, so the backup always matches exactly what changed.
- **The four app operations run in one transaction per city.** The app runs them as separate actions. Here one
  failure rolls back the vote change and the recount together.
- **Step 4 applies `computedHighQuality` to every non-excluded `user_stat` row of the labelers**, as
  `updateUserQualityForLabelersValidatedBy` does, so a `high_quality` that was already stale for another reason is
  corrected too. `labels_per_meter` and `meters_audited` are read as stored. They are not recomputed, because the
  conversion does not change them.
- **The dry run adds a column the plan did not list**, `correct_unchanged`. Labels whose counts change while
  `correct` does not need their own column for the transition counts to add up to `labels_recounted`. The plan's
  "`other` should be ~0" holds for the remaining column, `other_transition`.

## How this was tested

`tests/test_ai_disagree_to_unsure.py` starts a `postgres:16` container, or uses a local database named in
`AI_DISAGREE_TEST_DB`, and builds the tables the scripts read, with the column types from the evolutions. It seeds
two cities that reuse the same `label_id`s, plus one city at evolution 399 and the schemas the loop must skip. The
seed covers a lone AI Disagree, a tie the AI broke, humans ahead by one, a label humans also voted down, labels
deleted in Explore and from the LabelMap, a tutorial label, an excluded validator, an AI vote on a label's old type,
labelers at exactly 50 validations and exactly 0.6 accuracy, a manually low-quality labeler and a duplicated
`user_stat` row. The test checks:

- every dry-run column against hand-computed values;
- that the dry run leaves the data byte-identical;
- that `--apply` refuses without `--backup-dir` and without confirmation;
- the label counts and `user_stat` values after `--apply`;
- the backup files, and a zero `after` row;
- that a second apply finds nothing to do;
- that a step count that disagrees with the backup rolls the city back whole;
- the census's counts.

What it cannot check is the real schema and data. Indexes, triggers, the row counts' effect on run time and lock
duration, and anything about the 6432 copy show up only in the rehearsal (step 1 under Commands).
