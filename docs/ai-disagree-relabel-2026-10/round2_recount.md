# Round 2: the population recounted without excluded authors' labels (#85)

**Status: pending the `pull_author_excluded.sh` pull.** The tables below are filled by `gt_rounds.py recount` once Jon
runs the pull; until then the round-2 numbers in [`../ai-disagree-relabel-2026-10.md`](../ai-disagree-relabel-2026-10.md)
(Round 2) are the counts before the exclusion.

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

## Reproducing

```bash
D=~/Data/sidewalk-validator-ai-relabel-2026-10/round2
ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' bash -s" < analysis/pull_author_excluded.sh \
    > $D/author_excluded.csv
python analysis/gt_rounds.py recount --pool data/relabel-2026-10/round2/pool.csv.gz \
    --author-excluded $D/author_excluded.csv \
    --rawlabels-proxy ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels \
    --out docs/ai-disagree-relabel-2026-10/round2_recount.json \
    --markdown $D/round2_recount_tables.md
# then paste the tables below and, per rater, the weighted scores:
python analysis/gt_rounds.py score --key $K --rater jon --sheet $D/votes_jon.csv --unmasked $D/unmasked.csv \
    --pool data/relabel-2026-10/round2/pool.csv.gz --author-excluded $D/author_excluded.csv
```

`round2_recount.json` holds counts only (per city and per stratum x type), with the sha256 of the pool, the flags and
the design. The flags file has label ids and stays in `~/Data`.

## Results

<!-- Filled from `recount` output after the pull. Before the exclusion (from the committed pool, reproduced by
`tests/test_gt_rounds.py::test_recount_population_reproduces_the_committed_round2_manifest`): Seattle 17,284 eligible
AI Disagrees, 8,611 in no stratum; Chicago 10,769, 3,853 in no stratum. -->

| City (schema) | Excluded-author labels | Eligible AI Disagrees, before -> after | Unsampled, before -> after |
|---|---:|---|---|
| sidewalk_seattle | pending | 17,284 -> pending | 8,611 -> pending |
| sidewalk_chicago | pending | 10,769 -> pending | 3,853 -> pending |
