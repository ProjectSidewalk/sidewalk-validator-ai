# Round 2: the population recounted without excluded authors' labels (#85)

<!-- recount-status: pending -->
**Status: pending the `pull_author_excluded.sh` pull.** The pull has not been made yet; the tooling and the
acceptance checks below are in place. The tables below are filled by `gt_rounds.py recount` once the pull is run;
until then the round-2 numbers in [`../ai-disagree-relabel-2026-10.md`](../ai-disagree-relabel-2026-10.md) (Round 2)
are the counts before the exclusion.

## Why

Round 2's pool (`data/relabel-2026-10/round2/pool.csv.gz`, pulled 2026-10-28) predates the pool query's
`author_excluded` column. Gallery hides excluded authors' labels, so a rater could never draw or vote on them, yet
they sit in the stratum populations that `score` weights by and in the eligible-Disagree denominators the doc quotes
(Seattle 17,284, Chicago 10,769). Eleven Seattle items from one excluded author were rated unmasked for exactly this
reason.

## Method

The flags are joined onto the **original pinned pool** (`apply_author_excluded`, one-to-one on `(schema, label_id)`),
not a fresh pull: the raters' own round-2 votes are in prod now, and a fresh pool would drop the labels they voted on
from the population they were sampled from. A pool label missing from the flags (deleted since) counts as not
excluded, and the count is reported. The public rawLabels export also omits excluded authors' labels, so
`--rawlabels-proxy` prints how many pool AI Disagrees are absent from it as a cross-check (an upper bound: deleted and
newer labels are absent too). The proxy is never used for weights.

- **Flag date.** The flags are `user_stat.excluded` **as read on the pull date**, not as of the round. An author
  excluded or re-included between the round and the pull moves with the flag.
- **Acceptance checks.** `analysis/check_author_excluded.py` must pass before the flags are used
  (`analysis/round2_flags_pull.sh` runs it and accepts nothing otherwise): exact header and three fields per row;
  only `t` / `f`; exactly the two schemas, and no schema of the pool left out; `(schema, label_id)` unique; per
  city, under 1% of the pool's labels missing from the flags; and the **whole key** joined onto the flags, where
  the key items flagged excluded must be exactly the 11 rated unmasked, in both directions (an item whose author
  was excluded after it was rated masked would otherwise stay in the sample while its label left the population).
  The check's counts are committed as `round2_flags_check.json`.
- **Exclusion log.** SidewalkWebpage logs every exclusion toggle made on the admin page to the city's
  `webpage_activity` table. `analysis/pull_exclusion_changes.sh` reads it in the same connection, counts only (toggles
  per Pacific day and new value since the rawLabels export of 2026-10-18, with the labels those authors hold), so
  changes can be placed before, inside or after the rating window (2026-10-28 to 2026-11-02) and the pull. Its rows
  are a dated list of changes, not a net change: an author toggled more than once is counted in each row, with the
  same labels, so the rows must not be summed. Its two limits: logging began on 2026-09-14, and a change made by
  direct SQL is not logged. It also counts labels whose `label.user_id` differs from their `audit_task.user_id`
  (prod joins `user_stat` through the audit task; the pull joins through the label).
- **Where the recount bites.** Excluded authors' labels are hidden from Validate as well as Gallery, so from the
  day an author is excluded the crowd cannot vote on their labels. Labels placed after that day, or not yet voted
  on by then, have no human votes and fall in the `lone` stratum. That is where the recount is expected to change
  the population most, and the items rated unmasked were all `lone`. Labels that were voted on before their author
  was excluded sit in the other strata and leave those.

## Reproducing

The pull is one ssh call to the database host, run once; if it fails in any way, stop and do not retry. Everything
else runs locally. Run the block from the repository root, in bash or zsh (`py` is a function because zsh does not
word-split `$VAR`).

```bash
D=~/Data/sidewalk-validator-ai-relabel-2026-10/round2
R=data/relabel-2026-10/round2
DOC=docs/ai-disagree-relabel-2026-10
py() { uv run --no-project --python 3.10 --with 'pandas<3' --with numpy --with scikit-learn --with matplotlib python "$@"; }

# 1. pull + accept (the only remote step): writes $D/remote_bundle.sh, $D/author_excluded.csv, $D/exclusion_changes.csv
#    (or $D/exclusion_changes.csv.incomplete if the log could not be read in full: say so below, do not use it)
PYTHON="uv run --no-project --python 3.10 --with pandas<3 python" bash analysis/round2_flags_pull.sh --connect \
    --pool $R/pool.csv.gz --key $R/key/key.csv --unmasked $R/pre-consensus-2026-11-02/unmasked.csv \
    --check-out $DOC/round2_flags_check.json

# 2. recount
py analysis/gt_rounds.py recount --pool $R/pool.csv.gz --author-excluded $D/author_excluded.csv \
    --rawlabels-proxy ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels \
    --out $DOC/round2_recount.json --markdown $D/round2_recount_tables.md

# 3. score, per rater, before and after the exclusion
for r in jon mikey; do
  py analysis/gt_rounds.py score --key $R/key/key.csv --rater $r \
      --sheet $R/pre-consensus-2026-11-02/votes_$r.csv --unmasked $R/pre-consensus-2026-11-02/unmasked.csv \
      --pool $R/pool.csv.gz > $D/score_${r}_before_exclusion.md
  py analysis/gt_rounds.py score --key $R/key/key.csv --rater $r \
      --sheet $R/pre-consensus-2026-11-02/votes_$r.csv --unmasked $R/pre-consensus-2026-11-02/unmasked.csv \
      --pool $R/pool.csv.gz --author-excluded $D/author_excluded.csv > $D/score_${r}_recount.md
  diff $D/score_${r}_before_exclusion.md $D/score_${r}_recount.md
done
```

`round2_recount.json` holds counts only (per city and per stratum x type), with the sha256 of the pool, the flags and
the design. Its `pool_sha256` is of `pool.csv.gz` as committed (the value in `data/relabel-2026-10/round2/SHA256SUMS`);
the pin `a85a573c...` in the main doc is of the decompressed CSV. The flags file has label ids and stays in `~/Data`.
`--rawlabels-proxy` also reads `--cityparams` (default `~/Git/SidewalkWebpage/conf/cityparams.conf`), pinned below.

## Results

<!-- Filled from `recount` output after the pull. Before the exclusion (from the committed pool, reproduced by
`tests/test_gt_rounds.py::test_recount_population_reproduces_the_committed_round2_manifest`): Seattle 17,284 eligible
AI Disagrees, 8,611 in no stratum; Chicago 10,769, 3,853 in no stratum. -->

| City (schema) | Excluded-author labels | Eligible AI Disagrees, before -> after | Unsampled, before -> after |
|---|---:|---|---|
| sidewalk_seattle | pending | 17,284 -> pending | 8,611 -> pending |
| sidewalk_chicago | pending | 10,769 -> pending | 3,853 -> pending |

### Pins

Pending: `remote_bundle.sh` (the script that ran), `author_excluded.csv` (rows per schema), the two rawLabels files,
`cityparams.conf` (sha256 and SidewalkWebpage commit), the design and the pool.

### Acceptance

Pending: the result of the whole-key check, and the pool labels missing from the flags per city.

### Exclusion log

Pending: toggles before, inside and after the rating window, with the labels they touch, and the
`author_mismatch` counts.

### Effect of the recount on `score`'s weighted rows

Pending: per rater, the four Disagree strata's population-weighted "all" rows before and after the recount, with
decided n and the cells that get no weight. This is not the round-2 per-rater result.

### Cells under the minimum n

Pending: the (stratum, type) cells with fewer than 5 decided items (`--min-cell 5`) after the recount, which carry no
weight.

### Unsampled partition

Pending: the AI Disagrees in no stratum after the recount (one decisive vote / only Unsure / tie), which no round-2
rate covers.
