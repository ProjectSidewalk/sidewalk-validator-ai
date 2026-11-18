# A round for the AI Disagrees no round has sampled (#106)

**Status: DRAFT, not approved.** Nothing here has been built, crop-checked or rated. The design
(`analysis/gt_rounds/unsampled_draft.json`) carries `"draft": true`, so `gt_rounds.py build` refuses it; every
size, the cities and the raters are proposals for Jon to change (section 9). The counts below size the design on
round 2's pinned pool; they are not the population a build would weight by, which comes from a fresh pull at build
time.

Round 2 ([#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78)) sampled four AI-Disagree strata
(`lone`, `light_voted`, `margin_incorrect`, `margin_correct`). The AI Disagrees in none of them were never drawn, in
round 1 either (it sampled only `lone_disagree` / `voted_disagree`), so no expert verdict covers them:

- **`one_decisive`**: exactly one decisive human vote (Agree or Disagree), with or without Unsure votes;
- **`unsure_only`**: only Unsure human votes;
- **`tie`**: two or more decisive human votes, split evenly (short of the margin rule).

`gt_rounds.py` now samples them as three strata of their own (a design names all three in `shares`), and
`check-design` sizes a design before a build. This page is that check on a draft design, with the analysis plan
fixed before any rating.

## Pins and the command

| Input / output | Path | sha256 |
|---|---|---|
| Round-2 pool, as committed (pulled 2026-10-28) | `data/relabel-2026-10/round2/pool.csv.gz` | `8dd0c178ba88e2000886a42aaf1157f12ec4e48e3995f57d3a6753dec7f74959` |
| The same, decompressed (local only) | `~/Data/sidewalk-validator-ai-relabel-2026-10/round2/pool.csv` | `a85a573c303fe6538dba5821cbb7f3dc4667e244aa0c548354aa0e3bdcc23720` |
| Author-exclusion flags of 2026-11-12 (label ids; never committed) | `~/Data/sidewalk-validator-ai-relabel-2026-10/round2/author_excluded.csv` | `767c3d5b0fca35b57deb09e28995549bd382c383fd4f1634b55e14e6c469bfd7` |
| Draft design | `analysis/gt_rounds/unsampled_draft.json` | `76071e768803d70ef0bc54fbd7b232f60522877a201633f93397d9e620469df6` |
| `check-design` output, committed (aggregates only) | [`unsampled_strata_design.json`](unsampled_strata_design.json) | `abb07a90e66875d542662ee996897799215a351f1decca2ff43d4c001d0b9b93` |
| Its printout (local) | `~/Data/sidewalk-validator-ai-relabel-2026-10/round2/unsampled_strata_design_tables.md` | `9a1294e6ae8f1c47aac63cf5c6626c6894de93093ea9ca8ec20a5df5cef1cbf5` |

From the repo root (nothing is sampled, so there is no seed; Python 3.10 and 3.12 give the same JSON byte for
byte):

```bash
python analysis/gt_rounds.py check-design --design analysis/gt_rounds/unsampled_draft.json \
    --pool data/relabel-2026-10/round2/pool.csv.gz \
    --author-excluded ~/Data/sidewalk-validator-ai-relabel-2026-10/round2/author_excluded.csv \
    --out docs/ai-disagree-relabel-2026-10/unsampled_strata_design.json \
    --markdown ~/Data/sidewalk-validator-ai-relabel-2026-10/round2/unsampled_strata_design_tables.md
```

The JSON has two halves: `before` (the pool as committed, excluded authors' labels counted) and `after` (the flags
joined, excluded authors' labels out). `tests/test_gt_rounds.py` recomputes `before` from the committed pool in CI
and checks `after`'s totals against [`round2_recount.json`](round2_recount.json); a local test re-runs `after` cell
by cell when the flags are present (CI skips it). The design is sized on `after`. Running with `--cant-tell 0.05`
gives the 5% figures quoted in the precision section.

## The population (round-2 pool, excluded authors out)

Eligible AI Disagrees in the raters' own cities (neither expert has voted on or placed the label, author not
excluded). These equal the issue's table and `round2_recount.json` (`after.unsampled`); before the exclusion they
were 6,701 / 371 / 1,539 (Seattle) and 3,002 / 145 / 706 (Chicago).

| City | Stratum | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | all |
|---|---|---:|---:|---:|---:|---:|---:|
| Seattle (16,840 eligible AI Disagrees) | one_decisive | 4,794 | 881 | 103 | 878 | 42 | 6,698 |
| | unsure_only | 192 | 113 | 12 | 51 | 3 | 371 |
| | tie | 574 | 390 | 91 | 471 | 13 | 1,539 |
| Chicago (10,751) | one_decisive | 1,227 | 312 | 80 | 560 | 817 | 2,996 |
| | unsure_only | 56 | 34 | 2 | 25 | 28 | 145 |
| | tie | 214 | 195 | 26 | 176 | 95 | 706 |

- Ties are mostly 1-1: 1,398 of 1,539 in Seattle (the largest 5-5), 530 of 706 in Chicago (up to 12-12).
- In `one_decisive` the single decisive vote is an Agree for 3,165 of 6,698 (Seattle) and 1,467 of 2,996 (Chicago).
- `unsure_only` is mostly one Unsure vote: 347 of 371 and 132 of 145.

## The draft design

| Setting | Draft value |
|---|---|
| Cities and raters | Seattle (jon) and Chicago (mikey), the round-2 pairing; rater user_ids only on the command line |
| Strata | only the three new ones: `shares` tie 0.45, one_decisive 0.33, unsure_only 0.22; `lone`, `light_voted`, `margin` 0 |
| Per type and city | 36 AI-Disagree items: tie 16, one_decisive 12, unsure_only 8 (largest remainder of 36) |
| Decoys | 14 per type (AI Agree, Unsure where a type has no Agree), so the rater cannot tell the AI's verdict; only labels with at least one human vote (below) |
| Shared | 50 of each city's items rated by the other rater |
| Load | 250 items per city as designed, 300 per rater with the shared items (round 2's load) |

**Decoys have a human vote, as every item of these strata does.** Every `tie`, `one_decisive` and `unsure_only`
label has at least one human vote, while 30% of Seattle's eligible decoys and 36% of Chicago's have none (79% of
Seattle's Crosswalk decoys). Drawn from all of them, "nobody has validated this label" would mark an item as an AI
Agree decoy through any cue of prior validation the masking misses. Round 2's `lone` stratum (no human vote either)
balanced that; this design has none, so `gt_rounds.py` draws the decoys of any design with a zero `lone` share from
labels with at least one human vote (`voted_decoys`). Every type keeps at least 820 eligible decoys (Seattle
Crosswalk), so all 14-item decoy quotas still fill. The number of votes is not matched, so a weaker cue remains.

### Cells

From `check-design` (status precedence `no_weight` > `short` > `under_min` > `ok`; at a 25% can't-tell allowance a
cell needs a quota of 7 for 5 decided items, and every quota here is at least 8):

| City | Stratum | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk |
|---|---|---|---|---|---|---|
| Seattle | one_decisive (12) | ok | ok | ok | ok | ok |
| | unsure_only (8) | ok | ok | ok | ok | **no_weight**: 3 labels |
| | tie (16) | ok | ok | ok | ok | **short**: 13 of 16, about 9 decided |
| Chicago | one_decisive (12) | ok | ok | ok | ok | ok |
| | unsure_only (8) | ok | ok | **no_weight**: 2 labels | ok | ok |
| | tie (16) | ok | ok | ok | ok | ok |

The two `no_weight` cells take every label they have (3 and 2) and carry no population weight in `score`
(`--min-cell 5`), as round 2's thin cells did; they hold 3 of Seattle's 371 and 2 of Chicago's 145 `unsure_only`
labels. The
certain shortfall is 8 items in Seattle and 6 in Chicago, so a build gives 242 and 244 own-city items (292 and 294
rated per rater). Crop availability is not checked here; round 2 built with no crop shortfall in these two cities,
and `build` reports any.

### Expected precision

The 95% half-width of each stratum's population-weighted rate at p = 0.5 (the widest case), from `check-design`
(`half_width`, `kish_n`): the variance is `sum_t W_t^2 * 0.25 / n_t` over the types expected to carry weight, with
`W_t` the type's population share renormalised over them, as `score` weights.

| City | Stratum | Expected decided (25% can't tell) | Half-width, 25% can't tell | Kish effective n | Half-width, 5% can't tell |
|---|---|---:|---:|---:|---:|
| Seattle | tie | 57 | ±15.5 points | 40 | ±13.9 |
| | one_decisive | 45 | ±24.2 | 16 | ±21.9 |
| | unsure_only | 26 | ±24.9 | 16 | ±23.0 |
| Chicago | tie | 60 | ±14.1 | 48 | ±12.6 |
| | one_decisive | 45 | ±17.5 | 31 | ±15.9 |
| | unsure_only | 25 | ±21.1 | 22 | ±19.6 |

Round 2's can't-tell share on own-city, masked AI-Disagree items was 7 of 159 (4.4%, jon) and 8 of 170 (4.7%,
mikey), highest for SurfaceProblem (14%, mikey), so 25% is a deliberately cautious allowance and 5% the likely case.
A single type cell is wider still: a Wilson interval at p = 0.5 is about ±25 points on 12 decided items (a tie cell
at 25% can't tell), ±27 on 9 (one_decisive) and ±31 on 6 (unsure_only).

The half-widths have no finite-population correction, so they are conservative: with one on the sampled share of
each cell (1 - items / population), Chicago `unsure_only` is ±18.9 instead of ±21.1, and every other stratum moves by
under one point.

**What this round can and cannot say.** It can tell whether the AI's Disagree in each stratum is mostly right or
mostly wrong; it cannot place a rate within ten points. The quotas are equal per type, while Seattle's populations
are dominated by one type (CurbRamp is 72% of its `one_decisive`), so the weighted estimate rests on few items
(Kish n 16 of 45 expected decided). Two alternatives for Jon, neither taken in the draft:

- **Quotas in proportion to the type populations.** The same expected decided items spread in proportion to each
  type's share of the stratum give a Kish n equal to the item count: Seattle `one_decisive` about ±14.6 points
  instead of ±24.2, `tie` ±13.0 instead of ±15.5, `unsure_only` ±19.2 instead of ±24.9; Chicago ±14.6, ±12.7 and
  ±19.6 (1.96 x sqrt(0.25 / n) at the expected decided n above: an ideal that ignores the minimum cell and
  rounding). `per_type` can already lean toward CurbRamp without a schema change, but it is shared by both cities
  (Chicago's `one_decisive` is 27% Crosswalk), and the per-type rows (secondary 1) get thinner for the small types.
- **Ties only, 30 per type** (share 1 for `tie`, 0 for the others): Seattle's tie rate ±11.5 points at 25% can't
  tell (97 expected decided, Kish n 73) and Chicago's ±10.4 (107, Kish n 88), from `check-design`'s computation on
  the same pool and flags with that design; no estimate for the other two strata.

## Analysis plan, fixed before rating

Written before any item is drawn, so the secondary rows are not chosen after seeing the data.

**Primary estimand.** Per stratum (`tie`, `one_decisive`, `unsure_only`) and city: the population-weighted share
of the AI's Disagrees the expert confirms (the expert calls the label incorrect), from the rater's own-city
independent sheet (`votes_<rater>`, before any adjudication). Can't tell and unanswered items are left out of the
rate and counted; items rated with the AI's vote visible (`unmasked.csv`) are left out and counted. A (stratum,
type) cell with fewer than 5 decided items carries no weight (`score --min-cell 5`) and is named. Weights are the
eligible population of the build-time pool with excluded authors' labels out (`score --pool <build pool>`; the
design comes from the manifest).

**Secondary, pre-registered:**

1. Per type rates in each stratum (unweighted, Wilson intervals).
2. `one_decisive` split by the single human vote's direction (`h_agree == 1` vs `h_dis == 1`), per type and
   overall, with each direction's population share (`score` prints this table for any key with `one_decisive`
   items). With a single human Agree the AI's Disagree turns a 1-0 label into a 1-1; with a single human Disagree
   it concurs.
3. The single-vote baseline on `one_decisive`: how often the single human vote matches the expert on the same
   decided items. `tie` and `unsure_only` have no crowd verdict to compare.
4. Type x severity and type x tag slices with round-3 training counts (`score --round3-splits`), under the standing
   slice rule. At this size most slice rates will be blank (under 5 decided items) or very wide; the population
   counts below are this PR's slice result.
5. Inter-rater agreement on the shared items and, after the adjudication meeting, the gold labels per stratum
   (`adjudicate-report`).
6. `tie` split by whether the AI's vote decides the stored `correct` (FALSE) or not (empty), per city, with each
   part's population share (the caveat under "Why each stratum matters"). It needs each label's stored `correct`,
   `agree_count` and `disagree_count` in the build-time pool, which `pull_relabel_pool.sh` does not pull yet (step
   2 under "What a real round still needs"); without them this row is not computed and the `tie` rate is read with
   the caveat.

Anything else is exploratory and labelled so.

## Why each stratum matters

In SidewalkWebpage `label.correct` is TRUE when Agree votes outnumber Disagree votes, FALSE in the reverse case and
empty on a tie, counting every validator other than the label's author whose account is not excluded, the AI
included (`conf/evolutions/default/355.sql` and `LabelTable.recalculateValidationCounts`, which must match
`ValidationService`; votes filtered by `FilteredTables.isVerdictVote`; read in a local clone at `f741590d7`,
2026-11-11). `correct = FALSE` hides the label from the map and counts against its labeler
([`../ai-disagree-reversal-2026-10.md`](../ai-disagree-reversal-2026-10.md)). So, while the AI's Disagree votes
stand:

- **`tie`**: humans 1-1 plus the AI's Disagree is 1-2, so the AI's vote alone sets `correct = FALSE`. These are,
  mostly, the labels where the AI breaks the tie (see the caveat below).
- **`unsure_only`**: the AI's Disagree is the only decisive vote (0-1), as in `lone`, and alone hides the label.
- **`one_decisive`**: with a single human Agree, the AI's Disagree turns 1-0 (shown as correct) into 1-1
  (`correct` empty); with a single human Disagree it concurs (0-2). These one-vote labels are also where the
  `majority` and `majority2` training targets differ ([`../target-scoreboard-2026-11.md`](../target-scoreboard-2026-11.md)).

**Caveat: the strata use the pool's tallies, not those behind `label.correct`.** The strata come from the pool's
human tallies, which `pull_relabel_pool.sh` takes as every non-AI vote, unfiltered. `label.correct` counts only
verdict votes (`FilteredTables.isVerdictVote`: on the label's current type, not by its author, not by an excluded
user), so a pool tie is not always a tie in production. One read-only aggregate on production (2026-11-17, on the
drift query's basis below: the five types, excluded authors out, experts' labels in) gives the stored `correct` per
stratum:

| | Seattle | Chicago |
|---|---:|---:|
| `tie`: `correct` FALSE (the AI's vote decides) | 1,617 of 1,916 (84%) | 837 of 838 |
| `tie`: `correct` empty (the AI's vote hides nothing) | 299 (16%) | 1 |
| `tie`: stored Agree / Disagree counts differ from the unfiltered tallies plus the AI's vote | 646 (34%) | 22 |
| `one_decisive`, single human Agree: `correct` FALSE rather than empty | 206 of 3,384 (6%) | 8 of 1,525 |
| `one_decisive`, single human Disagree: `correct` FALSE | 4,196 of 4,196 | 1,759 of 1,759 |
| `unsure_only`: `correct` FALSE | 407 of 407 | 166 of 166 |

No label in these strata has `correct` TRUE. So Seattle's `tie` rate is not purely "how often the AI is right when
it breaks a tie": about one Seattle tie in six is a label where its vote decides nothing. Chicago's ties are almost
all decided by the AI's vote. Secondary analysis 6 splits `tie` by this.

<details><summary>The query (read-only; the same joins as the drift query below; the AI validator's account id goes on the command line)</summary>

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 makelab1 "AI=$AI_ID bash -s" <<'EOF'
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000'
P="psql -X -q -d sidewalk_prod -p 5434 -At -F,"
for S in sidewalk_seattle sidewalk_chicago; do
$P -c "
WITH ai AS (SELECT DISTINCT ON (label_id) label_id, validation_result::text AS r FROM $S.label_validation
            WHERE user_id = '$AI' ORDER BY label_id, label_validation_id DESC),
h AS (SELECT l.label_id, ai.r, l.correct, l.agree_count ac, l.disagree_count dc,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Agree' THEN 1 ELSE 0 END), 0) a,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Disagree' THEN 1 ELSE 0 END), 0) d,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Unsure' THEN 1 ELSE 0 END), 0) u
      FROM ai JOIN $S.label l ON l.label_id = ai.label_id AND NOT l.deleted AND NOT l.tutorial
      JOIN $S.label_point lp ON lp.label_id = l.label_id
      LEFT JOIN $S.user_stat us ON us.user_id = l.user_id
      LEFT JOIN $S.label_validation v ON v.label_id = l.label_id AND v.user_id <> '$AI'
      WHERE NOT COALESCE(us.excluded, false) AND ai.r = 'Disagree'
        AND l.label_type::text IN ('CurbRamp','NoCurbRamp','Obstacle','SurfaceProblem','Crosswalk')
      GROUP BY l.label_id, ai.r, l.correct, l.agree_count, l.disagree_count),
s AS (SELECT *, CASE WHEN a+d+u = 0 THEN 'lone' WHEN d-a > 1 THEN 'margin_incorrect'
  WHEN a-d > 2 THEN 'margin_correct' WHEN a+d >= 2 AND a <> d THEN 'light_voted' WHEN a+d = 1 AND a = 1 THEN 'one_decisive_agree'
  WHEN a+d = 1 THEN 'one_decisive_disagree' WHEN a+d = 0 THEN 'unsure_only' ELSE 'tie' END st FROM h)
SELECT '$S', st, count(*), sum((correct IS TRUE)::int), sum((correct IS FALSE)::int), sum((correct IS NULL)::int),
  sum((ac <> a OR dc <> d + 1)::int)
FROM s WHERE st IN ('tie','one_decisive_agree','one_decisive_disagree','unsure_only') GROUP BY 1,2 ORDER BY 1,2" || echo "$S failed"
done
EOF
```

Columns: schema, stratum, labels, `correct` TRUE, FALSE, empty, and labels whose stored counts differ from the
tallies plus the AI's vote. Run by the review (Fable 5.2) and again by the fix round (Opus 5.5) on 2026-11-17, through
the existing SSH master; both gave the counts above.

</details>

Whether the reversal of the AI's Disagree votes
([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)) has run is not checked
here; the drift query below, on 2026-11-17, still counts the AI's Disagree votes in both cities.

## Known costs of the allocation

- **`per_type` is shared by every city** in the design schema. Seattle spends 36 Crosswalk items on 58 labels (0.7%
  of its 8,608 unsampled AI Disagrees) while Chicago's Crosswalk is 27% of its `one_decisive`. Per-city quotas
  need a schema change (follow-up).
- **`unsure_only` takes 22% of the Disagree budget for 4% of the population** (371 of 8,608 and 145 of 3,847).
  Defensible, since the AI's vote is the only decisive one there, as in `lone`, but it is the costliest stratum per
  label.
- **Two cells cannot reach n = 5** (Seattle `unsure_only` Crosswalk, Chicago `unsure_only` Obstacle): they take
  every label and carry no weight.

## Population slices of the three strata

The standing slice rule asks for type x severity and type x tag, against a baseline, with training counts. A round of
this size cannot fill those slices with rates, so this PR reports the population side: the eligible labels of the
three strata per type x severity and per type x tag (a label counts once per tag; tags with fewer than 30 labels in
the type are folded into `(other tags)`, chosen on counts only). `score` prints the sample side (rates, the
single-vote baseline on `one_decisive`, training counts) once a round is rated.

<details><summary>Seattle, type x severity (20 rows)</summary>

| Type | severity | one_decisive | unsure_only | tie | all |
|---|---|---:|---:|---:|---:|
| CurbRamp | (none) | 330 | 18 | 17 | 365 |
| CurbRamp | 1 | 3,688 | 144 | 451 | 4,283 |
| CurbRamp | 2 | 522 | 16 | 76 | 614 |
| CurbRamp | 3 | 254 | 14 | 30 | 298 |
| NoCurbRamp | (none) | 45 | 6 | 22 | 73 |
| NoCurbRamp | 1 | 128 | 15 | 53 | 196 |
| NoCurbRamp | 2 | 416 | 54 | 181 | 651 |
| NoCurbRamp | 3 | 292 | 38 | 134 | 464 |
| Obstacle | (none) | 15 | 1 | 13 | 29 |
| Obstacle | 1 | 30 | 3 | 26 | 59 |
| Obstacle | 2 | 23 | 3 | 23 | 49 |
| Obstacle | 3 | 35 | 5 | 29 | 69 |
| SurfaceProblem | (none) | 25 | 3 | 21 | 49 |
| SurfaceProblem | 1 | 617 | 17 | 329 | 963 |
| SurfaceProblem | 2 | 154 | 20 | 76 | 250 |
| SurfaceProblem | 3 | 82 | 11 | 45 | 138 |
| Crosswalk | (none) | 2 | 0 | 2 | 4 |
| Crosswalk | 1 | 21 | 1 | 7 | 29 |
| Crosswalk | 2 | 9 | 1 | 2 | 12 |
| Crosswalk | 3 | 10 | 1 | 2 | 13 |

</details>

<details><summary>Seattle, type x tag (27 rows)</summary>

| Type | tag | one_decisive | unsure_only | tie | all |
|---|---|---:|---:|---:|---:|
| CurbRamp | (none) | 3,118 | 116 | 375 | 3,609 |
| CurbRamp | (other tags) | 2 | 0 | 0 | 2 |
| CurbRamp | missing tactile warning | 933 | 38 | 107 | 1,078 |
| CurbRamp | narrow | 238 | 12 | 23 | 273 |
| CurbRamp | not enough landing space | 131 | 3 | 11 | 145 |
| CurbRamp | not level with street | 420 | 12 | 65 | 497 |
| CurbRamp | points into traffic | 488 | 25 | 48 | 561 |
| CurbRamp | steep | 102 | 5 | 10 | 117 |
| CurbRamp | surface problem | 57 | 4 | 8 | 69 |
| NoCurbRamp | (none) | 417 | 48 | 182 | 647 |
| NoCurbRamp | alternate route present | 254 | 37 | 108 | 399 |
| NoCurbRamp | no alternate route | 178 | 21 | 78 | 277 |
| NoCurbRamp | unclear if needed | 85 | 15 | 39 | 139 |
| Obstacle | (none) | 19 | 2 | 16 | 37 |
| Obstacle | (other tags) | 44 | 6 | 31 | 81 |
| Obstacle | parked car | 15 | 2 | 13 | 30 |
| Obstacle | pole | 18 | 3 | 17 | 38 |
| Obstacle | trash/recycling can | 15 | 0 | 22 | 37 |
| SurfaceProblem | (none) | 116 | 19 | 69 | 204 |
| SurfaceProblem | (other tags) | 49 | 3 | 19 | 71 |
| SurfaceProblem | bumpy | 71 | 8 | 40 | 119 |
| SurfaceProblem | cracks | 352 | 13 | 114 | 479 |
| SurfaceProblem | grass | 325 | 11 | 230 | 566 |
| SurfaceProblem | narrow sidewalk | 70 | 7 | 29 | 106 |
| SurfaceProblem | uneven/slanted | 54 | 7 | 38 | 99 |
| Crosswalk | (none) | 33 | 2 | 7 | 42 |
| Crosswalk | (other tags) | 9 | 1 | 10 | 20 |

</details>

<details><summary>Chicago, type x severity (20 rows)</summary>

| Type | severity | one_decisive | unsure_only | tie | all |
|---|---|---:|---:|---:|---:|
| CurbRamp | (none) | 63 | 1 | 9 | 73 |
| CurbRamp | 1 | 826 | 41 | 131 | 998 |
| CurbRamp | 2 | 230 | 9 | 46 | 285 |
| CurbRamp | 3 | 108 | 5 | 28 | 141 |
| NoCurbRamp | (none) | 127 | 13 | 28 | 168 |
| NoCurbRamp | 1 | 84 | 6 | 45 | 135 |
| NoCurbRamp | 2 | 55 | 10 | 50 | 115 |
| NoCurbRamp | 3 | 46 | 5 | 72 | 123 |
| Obstacle | (none) | 13 | 2 | 5 | 20 |
| Obstacle | 1 | 41 | 0 | 12 | 53 |
| Obstacle | 2 | 19 | 0 | 8 | 27 |
| Obstacle | 3 | 7 | 0 | 1 | 8 |
| SurfaceProblem | (none) | 56 | 1 | 14 | 71 |
| SurfaceProblem | 1 | 244 | 16 | 49 | 309 |
| SurfaceProblem | 2 | 200 | 8 | 91 | 299 |
| SurfaceProblem | 3 | 60 | 0 | 22 | 82 |
| Crosswalk | (none) | 41 | 3 | 2 | 46 |
| Crosswalk | 1 | 466 | 21 | 48 | 535 |
| Crosswalk | 2 | 147 | 0 | 18 | 165 |
| Crosswalk | 3 | 163 | 4 | 27 | 194 |

</details>

<details><summary>Chicago, type x tag (30 rows)</summary>

| Type | tag | one_decisive | unsure_only | tie | all |
|---|---|---:|---:|---:|---:|
| CurbRamp | (none) | 508 | 25 | 74 | 607 |
| CurbRamp | (other tags) | 23 | 3 | 1 | 27 |
| CurbRamp | debris / pooled water | 42 | 1 | 12 | 55 |
| CurbRamp | missing tactile warning | 561 | 26 | 110 | 697 |
| CurbRamp | narrow | 78 | 5 | 10 | 93 |
| CurbRamp | not enough landing space | 28 | 1 | 6 | 35 |
| CurbRamp | not level with street | 86 | 3 | 22 | 111 |
| CurbRamp | points into traffic | 59 | 3 | 7 | 69 |
| CurbRamp | surface problem | 151 | 10 | 36 | 197 |
| NoCurbRamp | (none) | 162 | 16 | 101 | 279 |
| NoCurbRamp | alternate route present | 41 | 5 | 34 | 80 |
| NoCurbRamp | no alternate route | 80 | 10 | 48 | 138 |
| NoCurbRamp | unclear if needed | 38 | 4 | 18 | 60 |
| Obstacle | (none) | 15 | 1 | 4 | 20 |
| Obstacle | (other tags) | 44 | 1 | 17 | 62 |
| Obstacle | pole | 28 | 0 | 8 | 36 |
| SurfaceProblem | (none) | 46 | 2 | 9 | 57 |
| SurfaceProblem | (other tags) | 92 | 1 | 20 | 113 |
| SurfaceProblem | bumpy | 49 | 3 | 8 | 60 |
| SurfaceProblem | cracks | 134 | 10 | 24 | 168 |
| SurfaceProblem | grass | 70 | 7 | 10 | 87 |
| SurfaceProblem | narrow sidewalk | 231 | 4 | 106 | 341 |
| SurfaceProblem | uneven/slanted | 49 | 4 | 10 | 63 |
| Crosswalk | (none) | 342 | 18 | 30 | 390 |
| Crosswalk | (other tags) | 19 | 0 | 1 | 20 |
| Crosswalk | brick/cobblestone | 43 | 1 | 4 | 48 |
| Crosswalk | broken surface | 68 | 1 | 18 | 87 |
| Crosswalk | bumpy | 49 | 2 | 11 | 62 |
| Crosswalk | paint fading | 403 | 8 | 55 | 466 |
| Crosswalk | uneven surface | 45 | 1 | 9 | 55 |

</details>

## Drift since the pool (2026-11-17)

One read-only aggregate on the production database (makelab1, port 5434, through the existing SSH master), run by
the plan review on 2026-11-17: the AI-Disagree strata today against the pinned pool on the same basis (author
exclusion as read on 2026-11-12 for the pool side and on 2026-11-17 for the query; neither side drops the experts'
labels, so these totals are larger than the eligible counts above). The pool side is `pool.csv.gz` with the flags
joined, the five types, excluded authors out, strata from `assign_stratum(thin="split")`.

<details><summary>The query (read-only; the AI validator's account id goes on the command line)</summary>

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 makelab1 "AI=$AI_ID bash -s" <<'EOF'
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000'
P="psql -X -q -d sidewalk_prod -p 5434 -At -F,"
for S in sidewalk_seattle sidewalk_chicago; do
$P -c "
WITH ai AS (SELECT DISTINCT ON (label_id) label_id, validation_result::text AS r FROM $S.label_validation
            WHERE user_id = '$AI' ORDER BY label_id, label_validation_id DESC),
h AS (SELECT l.label_id, l.label_type::text AS t, ai.r,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Agree' THEN 1 ELSE 0 END), 0) a,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Disagree' THEN 1 ELSE 0 END), 0) d,
        COALESCE(SUM(CASE WHEN v.validation_result::text = 'Unsure' THEN 1 ELSE 0 END), 0) u
      FROM ai JOIN $S.label l ON l.label_id = ai.label_id AND NOT l.deleted AND NOT l.tutorial
      JOIN $S.label_point lp ON lp.label_id = l.label_id
      LEFT JOIN $S.user_stat us ON us.user_id = l.user_id
      LEFT JOIN $S.label_validation v ON v.label_id = l.label_id AND v.user_id <> '$AI'
      WHERE NOT COALESCE(us.excluded, false)
        AND l.label_type::text IN ('CurbRamp','NoCurbRamp','Obstacle','SurfaceProblem','Crosswalk')
      GROUP BY l.label_id, l.label_type, ai.r)
SELECT '$S', r, CASE WHEN r <> 'Disagree' THEN '-' WHEN a+d+u = 0 THEN 'lone' WHEN d-a > 1 THEN 'margin_incorrect'
  WHEN a-d > 2 THEN 'margin_correct' WHEN a+d >= 2 AND a <> d THEN 'light_voted' WHEN a+d = 1 THEN 'one_decisive'
  WHEN a+d = 0 THEN 'unsure_only' ELSE 'tie' END, count(*) FROM h GROUP BY 1,2,3 ORDER BY 1,2,3" || echo "$S failed"
done
EOF
```

</details>

| City | Stratum | Pool, 2026-10-28 | Production, 2026-11-17 |
|---|---|---:|---:|
| Seattle | one_decisive | 7,562 | 7,580 |
| | unsure_only | 405 | 407 |
| | tie | 1,893 | 1,916 |
| | all AI Disagrees | 19,642 | 19,642 |
| Chicago | one_decisive | 3,196 | 3,284 |
| | unsure_only | 164 | 166 |
| | tie | 818 | 838 |
| | all AI Disagrees | 12,817 | 12,808 |

At most 3% drift per stratum: the sizing holds. The build still needs a fresh pull (below), because eligibility must
drop the labels the raters have voted on in prod since 2026-10-28.

## What a real round still needs

1. Jon's decisions on the defaults below, then a copy of the design without `"draft"` (and a new name).
2. A fresh read-only pool: `pull_relabel_pool.sh` with `SCHEMAS='sidewalk_seattle sidewalk_chicago'` and both
   experts' ids on the command line only, into a new `~/Data/` round directory, sha256 pinned and `pool.err` checked
   empty (the script reports a failed schema only on stderr). The current script emits `author_excluded`, and
   `build` now refuses a pool without it (`--no-author-excluded` exists only to replay an older pool). For
   secondary analysis 6 the pull must also carry each label's stored `correct`, `agree_count` and `disagree_count`,
   and `score` must split `tie` by them: a small change to both, not made in this PR.
3. `check-design` on that pool, then `build` (crop check and fetch), `gallery`, rating, `votes`, `score`,
   `agreement` and the adjudication commands, as in round 2.

## Defaults taken (Jon can override)

- **J1.** Its own round: only the three new strata plus decoys (shares 0 for `lone` / `light_voted` / `margin`).
- **J2.** Seattle and Chicago, the same two raters as round 2, 50 shared items per city.
- **J3.** 250 per city / 300 per rater; per type tie 16, one_decisive 12, unsure_only 8; 14 decoys per type.
  Precision as above (about ±14 to ±25 points per stratum). Alternatives: quotas in proportion to the type
  populations (Seattle `one_decisive` about ±14.6 instead of ±24.2, fewer items for the small types), or ties only,
  30 per type (±11.5 for Seattle ties, ±10.4 for Chicago's).
- **J4.** No top-up of round 2's seven cells under n = 5 in this round.
- **J5.** `one_decisive` stays one stratum; the split by the single vote's direction is the pre-registered secondary
  analysis above.
- **J6.** Cells that cannot reach n = 5 take every available label and carry no population weight.
- **J7.** A 25% can't-tell allowance for sizing (a quota of 7 for 5 decided items; the draft has at least 8).
- **Remote.** No fresh pool pull, crop check or klone job in this PR; the one dated drift query above is cited.
