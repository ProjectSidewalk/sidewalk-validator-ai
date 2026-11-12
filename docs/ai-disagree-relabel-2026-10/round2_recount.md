# Round 2: the population recounted without excluded authors' labels (#85)

**Status: done.** The flags were pulled once, on 2026-11-12 at 18:34:48–18:34:53 UTC (10:34 Pacific), and passed
every acceptance check. The tables below are `gt_rounds.py recount` and `score` output on that pull; the round-2
counts in [`../ai-disagree-relabel-2026-10.md`](../ai-disagree-relabel-2026-10.md) (Round 2) quote them, with the
counts before the exclusion in parentheses.

## Why

Round 2's pool (`data/relabel-2026-10/round2/pool.csv.gz`, pulled 2026-10-28) predates the pool query's
`author_excluded` column. Gallery hides excluded authors' labels, so a rater cannot draw or vote on them, yet
they sit in the stratum populations that `score` weights by and in the eligible-Disagree denominators the doc quotes
(Seattle 17,284, Chicago 10,769). Eleven Seattle items from one excluded author were rated unmasked for exactly this
reason. The flags are read after the round, so an author excluded after it is removed too, although their labels
were visible during it: that happened for two Chicago authors (see the exclusion log below).

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

`recount` output, verbatim (`$D/round2_recount_tables.md`; the same counts are in
[`round2_recount.json`](round2_recount.json)). The authors' exclusion is `user_stat.excluded` as read on 2026-11-12.
"Before" is the committed pool as built, reproduced by
`tests/test_gt_rounds.py::test_recount_population_reproduces_the_committed_round2_manifest`.

| City (schema) | Excluded-author labels | Eligible AI Disagrees, before -> after | Unsampled, before -> after (one decisive / Unsure only / tie, after) |
|---|---:|---|---|
| sidewalk_seattle | 3,042 | 17,284 -> 16,840 | 8,611 -> 8,608 (6,698 / 371 / 1,539) |
| sidewalk_chicago | 70 | 10,769 -> 10,751 | 3,853 -> 3,847 (2,996 / 145 / 706) |

Population of rater jon's city (before -> after):

| Stratum | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | all |
|---|---:|---:|---:|---:|---:|---:|
| lone | 2,753 -> 2,737 | 404 -> 396 | 417 -> 63 | 507 -> 445 | 289 -> 288 | 4,370 -> 3,929 |
| light_voted | 669 -> 669 | 557 -> 557 | 160 -> 160 | 565 -> 565 | 15 -> 15 | 1,966 -> 1,966 |
| margin_incorrect | 627 -> 627 | 580 -> 580 | 261 -> 261 | 607 -> 607 | 48 -> 48 | 2,123 -> 2,123 |
| margin_correct | 66 -> 66 | 65 -> 65 | 21 -> 21 | 53 -> 53 | 9 -> 9 | 214 -> 214 |

Population of rater mikey's city (before -> after):

| Stratum | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | all |
|---|---:|---:|---:|---:|---:|---:|
| lone | 1,143 -> 1,142 | 530 -> 527 | 34 -> 34 | 474 -> 474 | 1,131 -> 1,131 | 3,312 -> 3,308 |
| light_voted | 405 -> 405 | 361 -> 360 | 84 -> 84 | 246 -> 246 | 201 -> 201 | 1,297 -> 1,296 |
| margin_incorrect | 267 -> 267 | 1,247 -> 1,242 | 278 -> 276 | 216 -> 216 | 57 -> 57 | 2,065 -> 2,058 |
| margin_correct | 61 -> 61 | 103 -> 103 | 18 -> 18 | 27 -> 27 | 33 -> 33 | 242 -> 242 |

Cross-check against the public rawLabels export (absent labels are an upper bound on excluded authors' labels; never used for weights):

| City (schema) | AI Disagrees absent from the export | of which flagged excluded | flagged excluded but present in the export |
|---|---:|---:|---:|
| sidewalk_seattle | 476 | 444 | 0 |
| sidewalk_chicago | 78 | 9 | 13 |

0 pool labels are not in the flags and count as not excluded.

Where the removed labels sit. Of Seattle's 3,042 excluded-author pool labels, 444 are eligible AI Disagrees: 441 in
`lone` (Obstacle 354, SurfaceProblem 62, CurbRamp 16, NoCurbRamp 8, Crosswalk 1) and 3 unsampled (one decisive
vote). No Seattle label leaves `light_voted`, `margin_incorrect` or `margin_correct`. Seattle's lone Obstacle cell
goes from 417 to 63, and lone SurfaceProblem from 507 to 445. Of Chicago's 70, 18 are eligible AI Disagrees: 7
`margin_incorrect`, 4 `lone`, 1 `light_voted` and 6 unsampled. All 444 removed Seattle Disagrees are among the 476
absent from the public export; in Chicago, 9 of the 18 are absent and 13 Disagrees flagged excluded are present in
the export of 2026-10-18 (see the exclusion log below).

Every count above was recomputed a second way, from `pool.csv.gz` and the flags with a separate pandas script that
re-implements eligibility and the strata from the pool columns (merge on `(schema, label_id)`,
`validate="one_to_one"`): the eligible counts, the unsampled partition, all 40 (stratum, type) population cells of
both halves, and the excluded-label counts agree exactly. The review of this page repeated that with a third,
separately written script, with the same result, and also recomputed the proxy table and the weighted rows below.

### Pins

| File | sha256 |
|---|---|
| `remote_bundle.sh` (the script that ran on the host; built by `round2_flags_pull.sh` at 1f84317) | `fd90605257d78da161e52f4b241d4b7d34ebfa86309a54505a0bdc736760b620` |
| `author_excluded.csv` (pulled 2026-11-12; Seattle 270,408 rows, 6,383 `t`; Chicago 173,023 rows, 134 `t`; not committed, label ids) | `767c3d5b0fca35b57deb09e28995549bd382c383fd4f1634b55e14e6c469bfd7` |
| `exclusion_changes.csv` (same connection; counts only, kept in `~/Data`) | `731685e523d2510c3b776210557ccbba71ab83acf09e093aff247edde3ab07f7` |
| rawLabels `seattle-wa.csv` (fetched 2026-10-18; `--rawlabels-proxy` only) | `1a2d575cbaabc1d247ce29492c99e1939d6b64d3d619f3e2872152be4d9d53af` |
| rawLabels `chicago-il.csv` (fetched 2026-10-18; `--rawlabels-proxy` only) | `b7c1783dcca10c8ef87e621cf9ac0a2ddc10011dbc19020aaf808c399ed0f6d9` |
| `cityparams.conf` (SidewalkWebpage f741590d7 plus 17 local added lines, none for `seattle-wa` or `chicago-il`; `--rawlabels-proxy` only) | `eed92cfd7658762f981c082f135fa1474041d688c7b3ef63f21278263026a899` |
| `analysis/gt_rounds/round2.json` (design) | `499fb9dfe5d976651f08d932d9a0c47477386e119080ae2107c91fc5c7404e57` |
| `data/relabel-2026-10/round2/pool.csv.gz` (as committed; the JSON's `pool_sha256`) | `8dd0c178ba88e2000886a42aaf1157f12ec4e48e3995f57d3a6753dec7f74959` |
| [`round2_flags_check.json`](round2_flags_check.json) (the checker's counts) | `b51ccbabb4ebac53e7cdf609ce178a610aad964f209d508a5dafe71d22673719` |

### Acceptance

`check_author_excluded.py` passed every check ([`round2_flags_check.json`](round2_flags_check.json)). The whole key
(500 items) joined onto the flags: the key items flagged excluded are exactly the 11 in `unmasked.csv`, no other key
item is flagged, and no key item is missing from the flags. No pool label is missing from the flags (0 of 200,362 in
Seattle, 0 of 146,373 in Chicago), so no deleted label had to be counted as not excluded.

### Exclusion log

The exclusion-log query ran in the same connection without an error (its stderr section is empty) and returned one
change row and both `author_mismatch` rows. Toggles logged since 2026-10-18, by Pacific day:

| Schema | Day | New value | Authors | Their labels at the pull (whole schema) |
|---|---|---|---:|---:|
| sidewalk_seattle | none logged | | | |
| sidewalk_chicago | 2026-11-03 | excluded | 2 | 31 |

- **Before the rating window (2026-10-18 to 2026-10-27):** nothing logged in either city.
- **Inside the rating window (2026-10-28 to 2026-11-02):** nothing logged in either city.
- **After it (2026-11-03 to the pull):** two Chicago authors were excluded on 2026-11-03, the day after the last
  rating sheet was pinned. So nothing logged changed what Gallery showed during the round, but the Chicago recount
  removes labels that were visible and eligible during it. The log has no label ids; the public export of
  2026-10-18 identifies them, because it omits excluded authors' labels and carries each label's author:
  - *Measured.* Of Chicago's 70 excluded-author pool labels, 47 are absent from the export and 23 are in it. The
    23 belong to exactly 2 authors, who hold 31 labels in the whole export: the log's 2 authors and 31 labels. (The
    export's author ids were only counted for this; none is recorded here.) **12 of the 18 removed eligible AI
    Disagrees are theirs**: 7 `margin_incorrect` (NoCurbRamp 5, Obstacle 2), 1 `light_voted` (NoCurbRamp) and 4
    unsampled (one decisive vote). They are 12 of the 13 Disagrees "flagged excluded but present in the export"
    above; the 13th is not eligible. The other 6 removed Disagrees (4 `lone`, 2 unsampled) are absent from the
    export, and none is newer than the export's newest label.
  - *Measured.* No rated item is affected: none of the 250 Chicago key items is by either author (0 by the flags,
    0 by the export's author field).
  - *What follows.* With exclusion as of the round, Chicago would lose 6 eligible AI Disagrees, not 18: 10,763
    eligible, 3,849 unsampled, `lone` 3,308, and `light_voted` (1,297) and `margin_incorrect` (2,065) unchanged.
    The tables on this page use the flags as read on 2026-11-12, so they remove the 12 as well. In `score` below,
    Chicago's `light_voted` and `margin_incorrect` changes come from those 12 labels alone; its `lone` change
    (4 labels) does not.
  - *Not known.* Whether the 6 absent labels' authors were excluded for the whole round. Their absence from the
    export and the empty log say so, as far as the log can see (limits below).
- **Seattle:** no toggle is logged since 2026-10-18, and all 3,042 excluded-author pool labels are absent from the
  2026-10-18 export, so they come from exclusions made before the export. That agrees with the 11 unmasked items
  found excluded on 2026-11-02.
- **Limits.** Logging began on 2026-09-14; a change made by direct SQL is not logged; and the rows are a dated list,
  not a net state (an author toggled twice is in two rows). The labels column counts each author's non-deleted,
  non-tutorial labels in the whole schema at the pull, not only pool labels.
- **`author_mismatch`** (labels whose `label.user_id` differs from their `audit_task.user_id`; prod joins
  `user_stat` through the audit task, the pull through the label): Seattle 0, Chicago 68, whole schema. Every flag
  on this page is the exclusion of the label's own author (`label.user_id`). Seattle's 0 means no Seattle number
  depends on that choice. In Chicago it is a bound, not a measurement: at most 68 of the schema's 173,023 labels
  could carry a different flag than Gallery and Validate apply, and only where the two accounts differ in
  exclusion. Which of the 68 are pool labels was not pulled.

### Effect of the recount on `score`'s weighted rows

This is the effect of the new denominators on `score`'s population-weighted "all" rows, before and after, from
`score --pool` and `score --pool --author-excluded` on the committed sheets. **It is not the round-2 per-rater
result**: that analysis, with type x severity and type x tag slices, is a separate step. The unweighted columns
(rate, n, can't tell) are identical before and after; only the population and the weighted rate can move.

| Rater (own city) | Stratum | Decided n | Population, before -> after | In cells that carry weight, before -> after | Weighted, before | Weighted, after |
|---|---|---:|---|---|---|---|
| jon (Seattle, 239 items) | lone | 48 | 4,370 -> 3,929 | 3,953 -> 3,866 | 57.2% [38.7, 75.8] | 57.8% [39.0, 76.5] |
| jon (Seattle) | light_voted | 22 | 1,966 -> 1,966 | 1,241 -> 1,241 | 73.1% [45.4, 100.0] | 73.1% [45.4, 100.0] |
| jon (Seattle) | margin_incorrect | 52 | 2,123 -> 2,123 | 2,123 -> 2,123 | 74.9% [61.0, 88.8] | 74.9% [61.0, 88.8] |
| jon (Seattle) | margin_correct | 30 | 214 -> 214 | 161 -> 161 | 22.7% [1.3, 44.1] | 22.7% [1.3, 44.1] |
| mikey (Chicago, 250 items) | lone | 55 | 3,312 -> 3,308 | 3,312 -> 3,308 | 43.7% [29.8, 57.6] | 43.7% [29.8, 57.6] |
| mikey (Chicago) | light_voted | 24 | 1,297 -> 1,296 | 967 -> 966 | 43.8% [20.1, 67.6] | 43.8% [20.0, 67.6] |
| mikey (Chicago) | margin_incorrect | 53 | 2,065 -> 2,058 | 2,065 -> 2,058 | 89.4% [74.2, 100.0] | 89.4% [74.2, 100.0] |
| mikey (Chicago) | margin_correct | 30 | 242 -> 242 | 215 -> 215 | 29.0% [7.7, 50.3] | 29.0% [7.7, 50.3] |

**A weighted rate is an estimate for the cells that carry weight, not for the whole stratum.** A (stratum, type)
cell with fewer than 5 decided items is left out and the weights are renormalised over the rest, so the rate says
nothing about the labels in that cell. Seattle's `lone` rate of 57.8% covers 3,866 of the stratum's 3,929 labels
(98.4%): CurbRamp, NoCurbRamp, SurfaceProblem and Crosswalk. Lone Obstacle, with a population of 63 and 1 decided
item, carries no weight and is not in it. Before the recount the same rate covered 3,953 of 4,370 (90.5%), with 417
lone Obstacle labels outside it. The widest gap is `light_voted`: Seattle's rate covers 1,241 of 1,966 labels
(63.1%) and Chicago's 966 of 1,296 (74.5%). The cells left out, with their populations, are listed in the next
section.

Two printed values move: Seattle `lone` from 57.2% [38.7, 75.8] to 57.8% [39.0, 76.5], and the lower bound of
Chicago `light_voted` from 20.1 to 20.0. Seattle's lone population falls by 441 (10.1%), but nearly all of that is
lone Obstacle (417 -> 63), a cell with 1 decided item that already carried no weight; the cells that carry weight
lose 87 labels (CurbRamp 16, NoCurbRamp 8, SurfaceProblem 62, Crosswalk 1). Both Chicago changes (`light_voted`,
and `margin_incorrect` below the printed precision) come from labels of the two authors excluded on 2026-11-03,
which were visible during the round; the 4 labels Chicago `lone` loses are absent from the export.

The point estimates and the intervals were recomputed at review with separately written code, from the committed
key and sheets and the populations in `round2_recount.json` (the interval is the normal approximation on the
stratified variance, each cell's rate adjusted to (k + 2) / (n + 4), as `weighted_rate` documents). All eight rows
agree before and after. To two decimals: Seattle `lone` 57.24% [38.70, 75.78] -> 57.77% [39.02, 76.52]; Chicago
`light_voted` 43.83% [20.06, 67.59] -> 43.79% [20.03, 67.55]; Chicago `lone` 43.72% -> 43.68% and
`margin_incorrect` 89.44% -> 89.43%, both below the printed precision.
`tests/test_round2_recount.py` recomputes the table's rows from the committed files.

### Cells under the minimum n

With `--min-cell 5`, the cells with fewer than 5 decided items carry no weight. After the recount they are the same
cells as before, with the same n, and every one still has a population above zero, so each is still named:

- jon (Seattle): lone Obstacle (n=1; population 417 -> 63); light_voted Obstacle (n=4; population 160) and
  SurfaceProblem (n=3; population 565); margin_correct SurfaceProblem (n=4; population 53).
- mikey (Chicago): light_voted Obstacle (n=4; population 84) and SurfaceProblem (n=4; population 246);
  margin_correct SurfaceProblem (n=4; population 27).

The populations of these seven cells, 1,198 labels after the recount, are in no weighted rate.

No stratum is unestimable: every stratum keeps at least one cell with 5 or more decided items.

### Unsampled partition

The AI Disagrees in no stratum after the recount, which no round-2 rate covers (one decisive vote / only Unsure /
tie):

- Seattle: 8,608 of 16,840 eligible (6,698 / 371 / 1,539), before 8,611 of 17,284 (6,701 / 371 / 1,539).
- Chicago: 3,847 of 10,751 eligible (2,996 / 145 / 706), before 3,853 of 10,769 (3,002 / 145 / 706).
