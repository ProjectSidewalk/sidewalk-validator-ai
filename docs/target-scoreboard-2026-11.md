# Target-cleaning methods against the expert gold set, rounds 1 + 2 (November 2026)

Which automatic way of cleaning the crowd's Disagree targets agrees with an expert, before any of them rewrites a
training target? This scores the methods that run without a GPU, the stricter vote rules of
[#75](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75) and confident learning of
[#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76), against the masked expert verdicts of
rounds 1 and 2 ([#66](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66),
[#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78)). Part of
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74).

**Status:** run 2026-11-13 on the Mac from files on disk (`analysis/target_scoreboard.py`). The VLM adjudicator of
[#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77) is not in this scoreboard: it has not
been scored yet (that needs klone), and `adjudicator.py agree` already scores it against the same gold set with a
paired bar. Its rows join this scoreboard in the PR that brings its klone scores back.

## Short answer

Both primaries are the pooled rows (rounds 1 + 2), the pre-stated estimate; the per-round rows and a heterogeneity
check are in Results.

- **`symmetric` against the training margin rule (P1).** On the AI Disagrees whose margin target is `incorrect`,
  the expert says incorrect on 54/65 rows `symmetric` keeps and 68/85 rows it drops: kept minus dropped
  +3.1 pts [-9.4, +15.9]. Applying the rule (keeping the 65, dropping the 85) changes the share of `incorrect`
  targets the expert confirms from 122/150 to 54/65: +1.7 pts [-5.2, +8.6]. The data are compatible with a gain of
  up to about 9 points, with no change, and with a loss of about 5. The dropped two-vote Disagrees are not mostly
  wrong: the expert confirms them at least 70.3% of the time (Wilson lower bound of 68/85). This is not evidence
  that stricter rules do not help; it is a difference this gold set is too small to resolve.
- **Confident learning (P2).** On the 131 decided round-3 train rows, 4/13 flagged rows have a wrong target against
  19/118 unflagged: +14.7 pts [-10.1, +43.2], Fisher p 0.242. The flags catch 4 of the 23 wrong targets. Flipping
  the flagged targets unread pays off only if more than half of them are wrong. The point estimate is well under
  half (4 of 13, 30.8%), but its interval [12.7, 57.6] does not exclude a majority, so the gold set neither
  licenses nor rules out acting on the flags unread; sending them to an expert is a judgement on the point
  estimate. (Dropping is a different trade: a dropped right target loses a row rather than being broken.) About
  30 rated flagged rows would settle it if the share stays near 31% (Wilson's upper bound then falls under 50%).
  The data also cannot tell whether flagged rows are more often wrong than the rest. 11 of the 13 flagged rows are
  from round 2, so the pooled P2 is essentially round 2's.
- **Coverage is a property of the pool, not of the gold set.** Of 91,626 AI Disagrees in the pool, 28.0% have one
  decisive vote and 6.6% are ties. No gold item comes from either, so every rate below is silent on them, and they
  are 47% of what an any-majority rule decides.

## Why

The training margin rule (`incorrect` at net Disagree >= 2, `correct` at net Agree >= 3) matches an expert about
three times in four on contested labels. [#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74)
asks which method should replace or filter it, scored on one gold set
by one protocol. The gold set it waited for has been complete since 2026-11-02; three docs each promised the
comparison and none had it ([`confident-learning-2026-10.md`](confident-learning-2026-10.md),
[`adjudicator-2026-11.md`](adjudicator-2026-11.md), [`ai-disagree-relabel-2026-10.md`](ai-disagree-relabel-2026-10.md)).

## The gold set

`adjudicator.py gold` builds it (rounds 1 and 2 in one table keyed on `city_id:label_id`; definition and counts in
[`adjudicator-2026-11.md`](adjudicator-2026-11.md#the-gold-set)). It was rebuilt for this run with that doc's
Reproducing step 1 and its sha256 matches the pin there. With the meeting verdicts left out it has 488 decided AI
Disagrees (184 from round 1, 304 from round 2) and 131 decided round-3 train rows.

<!-- BEGIN GENERATED: gold_counts (analysis/target_scoreboard.py) -->

| Count | Value |
|---|---:|
| Gold items (rounds 1 + 2) | 800 |
| Decided (a correct or incorrect verdict, as `adjudicator.py gold` defines it) | 741 |
| of which the round-2 meeting reached (`r2_consensus`) | 10 |
| Decided, masked-only gold (the headline) | 731 |
| AI Disagrees, masked-only: round 1 / round 2 | 184 / 304 (488) |
| AI Disagrees, masked-only, per type | Crosswalk 110, CurbRamp 119, NoCurbRamp 101, Obstacle 79, SurfaceProblem 79 |
| Round-3 train rows: round 1 / round 2 | 37 / 94 (131) |
| of which flagged by confident learning / wrong target | 13 / 23 |
| Items whose pool tallies hold a rater's earlier vote: round 1 / round 2 | 2 / 0 |
| Train rows whose round-3 target differs from the margin target of the pool tallies | 5 |

<!-- END GENERATED: gold_counts -->

## Method

### Pre-stated primary contrasts

The adversarial review of the plan (2026-11-13) fixed two primary contrasts before this module existed, and
computed the pooled value of each by hand while doing so; this doc adopts them unchanged, with their populations and
outcomes as below. The implementer's first run of the module came before this text was written, so the reading
rule is stated only in terms of the intervals, not of any threshold chosen after seeing them.

- **P1. `symmetric` kept against dropped.** Rows: decided AI Disagrees whose margin target (from the 2026-10-27
  pool tallies, a rater's earlier production vote removed) is `incorrect`. Groups: the rows `symmetric` keeps (net
  Disagree >= 3) and the rows it drops (net Disagree 2). Outcome: the expert says incorrect. Statistic: kept minus
  dropped, 95% percentile interval from 2,000 label-level resamples, and the two-sided Fisher exact p.
- **P2. Confident-learning flag.** Rows: decided gold items that are round-3 train rows. Groups: flagged by
  confident learning (the rows `cl-drop` drops and `cl-flip` flips) and unflagged. Outcome: the round-3 target
  differs from the expert. Statistic as P1.
- **P3, later.** The VLM adjudicator against the crowd majority: the paired bar of
  [`adjudicator-2026-11.md`](adjudicator-2026-11.md#the-bar), scored there once klone has run.

**Which row is primary.** For each primary the pooled row (rounds 1 + 2, a sample rate) is the pre-stated estimate:
it is the value the review fixed and computed by hand, and the one the Short answer reports. The per-round rows are
printed above it as its check, with a heterogeneity check (round 1's difference minus round 2's, resampled within
round). For P2 that check cannot be computed: round 1 has 2 flagged rows, so the pooled P2 is essentially round 2
(11 of its 13 flagged rows). The tables label the primary pooled rows "pooled (primary estimate)" and every other
pooled row "descriptive".

**Reading rule:** an interval that excludes 0 is a difference; one that spans 0 is reported as what it rules out at
each end, never as "no effect". No correction for multiplicity: there are two primaries, and every other table is
labelled descriptive. A difference interval is not printed when either group has fewer than 5 items.

**The bootstrap** resamples labels iid over the pooled set, ignoring round and stratum. That is slightly
conservative here: a bootstrap stratified by round x stratum gives P1 about [-9.0, +15.4] and P2 about
[-10.2, +41.4], against [-9.4, +15.9] and [-10.1, +43.2]. Panorama clustering does not matter: P1's 150 rows sit on
149 panoramas and P2's 131 on 130.

### Choices

- **Masked-only gold leads.** The 10 decided round-2 verdicts reached in the meeting (unmasked to each other's
  votes, `expert_source == r2_consensus`) are left out of every headline; the sensitivity table puts them back.
  None of them is a train row, so P2 is the same either way.
- **A rater's own earlier vote is not crowd evidence.** Round 1 excluded labels one rater had voted on; the pool
  records the other rater's vote as a value with no id (`expert2`), and it is subtracted from the tallies of the
  items that carry one. Round 2 excluded both raters' labels and votes.
- **Coverage comes from the pool.** The strata were drawn on the same tallies the rules read, so the share of gold
  items a rule decides restates the sampling design. Coverage and drop shares are exact counts over every AI
  Disagree of `relabel_pool.csv`; the gold set is used only for how often a verdict is right.
- **`majority` and `majority2`.** They differ only on single-vote labels, and the gold set has none (no stratum
  draws them). With the tallies as pulled they give the same verdict on every gold AI Disagree; after the expert
  votes are removed two round-1 items fall to a single decisive vote, so the rules table keeps both rows.
- **Population weighting.** Each round is weighted on its own: round 1 by its manifest's population per (group,
  type), round 2 by the [#85](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/85) recount per
  (city, stratum, type). A domain inside a cell (all margin-`incorrect` rows, or the kept or dropped ones) is
  weighted by the cell's population times its share of the cell's decided items. A cell with fewer than 5 domain
  items gets no weight (the rule of `gt_rounds.score`). Kept minus dropped is weighted only over the cells where
  both groups reach 5, so both shares average the same cells, with an interval from the stratified variance (not
  clipped at 0 or 100%); it is not estimable with fewer than 2 such cells. The rounds are never weighted together:
  round 2's two cities are part of round 1's population.
- **Slices.** The standing type x severity and type x tag rule is met on the method-independent quantity (the
  share of AI Disagrees the expert calls incorrect, with the round-3 train rows of each slice) at a minimum of 20
  items per cell, and P1 is sliced by severity pooled over types. Method-by-slice tables would mostly measure which
  strata each method covers. P2 is not sliced: with 13 flagged rows no severity or tag cell can reach 20.

### What the numbers are not

- Rates in a stratified sample of contested labels. Every gold item is an AI Disagree or a decoy, so "target
  accuracy" here is on labels the production AI contested, not on the training set as a whole.
- Silent on single-vote and tied AI Disagrees (34.6% of the pool), the labels where `majority` and `majority2`
  differ. Issue [#106](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/106) proposes the stratum
  that would cover them.
- Seattle and Chicago: every round-2 item is from one of them; round 1 spreads over 28 deployments. Per-deployment
  rows are not shown (private deployments are pooled everywhere).
- A test of the target only. Whether a stricter rule makes a better *training set* (with the class rebalanced) is
  the question of [`confident-learning-2026-10.md`](confident-learning-2026-10.md), Recommendation 2.
- Three tally sources: the 2026-10-27 pool (view of the rules), the 2026-10-19 manifests (round-3 targets), and
  for 21 round-2 decoys missing from that pool, the round-2 key. 5 train rows have a round-3 target that differs
  from the margin target of the pool tallies.

## Results

### Coverage on the whole AI-Disagree pool

<!-- BEGIN GENERATED: pool (analysis/target_scoreboard.py) -->

| Type | AI Disagrees | no decisive vote | 1 vote | tie | light | margin incorrect | margin correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 91,626 | 33.9% | 28.0% | 6.6% | 11.7% | 16.3% | 3.5% |
| CurbRamp | 32,098 | 34.4% | 34.4% | 6.1% | 11.5% | 9.9% | 3.8% |
| NoCurbRamp | 19,534 | 23.9% | 21.2% | 7.2% | 14.0% | 29.2% | 4.5% |
| Obstacle | 14,150 | 46.5% | 24.9% | 5.2% | 7.9% | 13.3% | 2.1% |
| SurfaceProblem | 15,070 | 24.0% | 27.4% | 10.5% | 15.1% | 20.2% | 2.9% |
| Crosswalk | 10,773 | 47.6% | 26.4% | 3.5% | 8.5% | 10.0% | 4.0% |

| Type | `majority` | `majority2` | `symmetric2` | `margin` | `symmetric3` | `unanimous3` |
|---|---:|---:|---:|---:|---:|---:|
| all | 59.5% (33.3%) | 31.5% (19.7%) | 25.7% (16.3%) | 19.8% (16.3%) | 11.7% (8.2%) | 6.4% (4.1%) |
| CurbRamp | 59.5% (29.3%) | 25.2% (12.8%) | 20.1% (9.9%) | 13.7% (9.9%) | 7.2% (3.4%) | 5.4% (2.6%) |
| NoCurbRamp | 68.9% (43.5%) | 47.7% (33.9%) | 39.7% (29.2%) | 33.7% (29.2%) | 24.5% (20.0%) | 8.8% (6.8%) |
| Obstacle | 48.3% (30.2%) | 23.4% (16.0%) | 19.2% (13.3%) | 15.5% (13.3%) | 7.6% (5.5%) | 4.9% (3.5%) |
| SurfaceProblem | 65.5% (39.3%) | 38.2% (25.1%) | 29.8% (20.2%) | 23.1% (20.2%) | 9.9% (7.0%) | 6.7% (5.0%) |
| Crosswalk | 49.0% (22.7%) | 22.5% (11.4%) | 20.2% (10.0%) | 14.0% (10.0%) | 10.2% (6.2%) | 6.9% (3.5%) |

Cells: share of the type's AI Disagrees the rule gives any target (in brackets: an `incorrect` target). The rest is dropped. Exact counts over the whole pool, so no interval.

<!-- END GENERATED: pool -->

A third of the pool has no decisive vote and is decided by no vote rule; the gold set's `lone` stratum scores the AI
on those. One-vote labels (28.0%) and ties (6.6%) are what separates `majority` from `majority2`, and no gold round
has sampled them. `symmetric` gives a target to 11.7% of the AI Disagrees against 19.8% for the margin rule; nearly
all of the difference is two-vote Disagrees (16.3% of the pool get an `incorrect` margin target, 8.2% under
`symmetric`).

### The primary contrasts

<!-- BEGIN GENERATED: primary (analysis/target_scoreboard.py) -->

**P1. `symmetric` kept against dropped** (decided AI Disagrees whose margin target is `incorrect`; outcome: the expert says incorrect; pooled row: primary):

| Rows | kept: net Disagree >= 3 | dropped: net Disagree = 2 | kept minus dropped [95%] | Fisher p |
|---|---|---|---|---:|
| round 1 | 12/17 (70.6% [46.9, 86.7]) | 22/29 (75.9% [57.9, 87.8]) | -5.3 pts [-31.3, +21.1] | 0.737 |
| round 2 | 42/48 (87.5% [75.3, 94.1]) | 46/56 (82.1% [70.2, 90.0]) | +5.4 pts [-8.4, +18.9] | 0.588 |
| **pooled (primary estimate)** | 54/65 (83.1% [72.2, 90.3]) | 68/85 (80.0% [70.3, 87.1]) | +3.1 pts [-9.4, +15.9] | 0.677 |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): -10.6 pts [-40.7, +21.2].

Kept minus all rows of the margin rule (what applying the stricter rule changes in the share of `incorrect` targets the expert confirms; P1 rescaled by the dropped share): round 1 -3.3 pts [-20.1, +14.1]; round 2 +2.9 pts [-4.5, +10.5]; pooled +1.7 pts [-5.2, +8.6].

**P2. Confident-learning flag** (decided round-3 train rows, both target classes; outcome: the target differs from the expert; pooled row: primary):

| Rows | flagged | unflagged | flagged minus unflagged [95%] | Fisher p | Recall | False alarms |
|---|---|---|---|---:|---|---|
| round 1 | 0/2 (0.0% [0.0, 65.8]) | 9/35 (25.7% [14.2, 42.1]) | -25.7 pts | 1 | 0/9 (0.0% [0.0, 29.9]) | 2/28 (7.1% [2.0, 22.6]) |
| round 2 | 4/11 (36.4% [15.2, 64.6]) | 10/83 (12.0% [6.7, 20.8]) | +24.3 pts [-4.1, +56.4] | 0.0559 | 4/14 (28.6% [11.7, 54.6]) | 7/80 (8.8% [4.3, 17.0]) |
| **pooled (primary estimate)** | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.1, +43.2] | 0.242 | 4/23 (17.4% [7.0, 37.1]) | 9/108 (8.3% [4.4, 15.1]) |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): not computable (a group in one round has 2 rows; it needs 5).

The pooled P2 is mostly round 2: 11 of its 13 flagged rows are from round 2, and with 2 flagged round-1 rows the rounds cannot be compared.

Descriptive twin of P1, `unanimous3` (same rows; pooled row: descriptive):

| Rows | kept: unanimous | dropped | kept minus dropped [95%] | Fisher p |
|---|---|---|---|---:|
| round 1 | 10/13 (76.9% [49.7, 91.8]) | 24/33 (72.7% [55.8, 84.9]) | +4.2 pts [-26.3, +29.6] | 1 |
| round 2 | 25/30 (83.3% [66.4, 92.7]) | 63/74 (85.1% [75.3, 91.5]) | -1.8 pts [-18.4, +13.3] | 0.773 |
| pooled (sample rate; descriptive) | 35/43 (81.4% [67.4, 90.3]) | 87/107 (81.3% [72.9, 87.6]) | +0.1 pts [-14.5, +13.2] | 1 |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): +6.0 pts [-26.5, +35.9].

Kept minus all rows of the margin rule (what applying the stricter rule changes in the share of `incorrect` targets the expert confirms; P1 rescaled by the dropped share): round 1 +3.0 pts [-18.3, +22.1]; round 2 -1.3 pts [-13.3, +8.9]; pooled +0.1 pts [-10.2, +10.6].

<!-- END GENERATED: primary -->

**P1.** Pooled (the primary estimate), kept 54/65 against dropped 68/85: kept minus dropped +3.1 pts
[-9.4, +15.9]. Neither round's interval excludes 0 (round 1 -5.3, round 2 +5.4), and the heterogeneity check
(-10.6 pts [-40.7, +21.2]) is too wide to say whether the rounds agree. Kept minus dropped is not the gain from the
rule: applying `symmetric` keeps the 65 rows and drops the 85, which moves the expert-confirmed share of the
margin rule's `incorrect` targets from 122/150 to 54/65, +1.7 pts [-5.2, +8.6] (kept minus all rows, the line under
the table). What the data rule out: that the two-vote Disagrees `symmetric` drops are mostly wrong targets (the
expert confirms 68/85, Wilson lower bound 70.3%), and a gain from applying the rule larger than about 9 points. What
they cannot rule out: any gain from 0 to about 9 points, or a loss of up to about 5. At 200 rows per group a
10-point kept-minus-dropped difference would be detected with about 80% power at 80% against 90%, and about 70% at
75% against 85%; this set has 65 and 85. The `unanimous3` twin's interval also spans 0 (+0.1 pts [-14.5, +13.2]).

**P2.** Pooled (the primary estimate), flagged 4/13 against unflagged 19/118, +14.7 pts [-10.1, +43.2], Fisher p
0.242. Round 1 has 2 flagged rows, so no round-1 interval and no heterogeneity check, and the pooled P2 is
essentially round 2's (4/11 against 10/83 there). Recall is 4/23 wrong targets, and 9 of the 13 flagged targets are
right. Flipping a flagged target unread helps only when it is wrong, so flipping all of them pays off only if more
than half are wrong: the point estimate (30.8%) is well under half, but the interval [12.7, 57.6] does not exclude
a majority, so the gold set neither licenses nor rules out acting on the flags unread. Dropping is a different
trade (a dropped right target loses a row, it is not broken) and is not tested here. Recommending an expert look
at the flags is a judgement on the point estimate. About 30 rated flagged rows would settle it if the share stays
near 31% (Wilson's upper bound falls under 50% from about 30); at this gold set's flag rate (13 of 131) that is
about 300 rated train rows, or fewer from a round that oversamples flags. What the data also cannot rule out: that
flagged rows are enriched for wrong targets, which the point estimate suggests and the interval does not settle.

### Population-weighted P1, each round on its own

<!-- BEGIN GENERATED: weighted (analysis/target_scoreboard.py) -->

| Round | margin-`incorrect` AI Disagrees, expert says incorrect (cells without weight) | cells with >= 5 kept / >= 5 dropped / both, of all | kept minus dropped over the shared cells [95%, unclipped normal approx.] |
|---|---|---|---|
| 1 | 73.1% [58.3, 87.9] (n 43; 1 without weight) | 2 / 2 / 0, of 5 | not estimable: 0 shared cells (needs 2) |
| 2 | 82.0% [71.7, 92.4] (n 104; 0 without weight) | 5 / 5 / 1, of 10 | not estimable: 1 shared cell (needs 2) |

A kept-minus-dropped difference is computed only over cells where both groups hold at least 5 rows, so the two shares average the same cells; with fewer than 2 such cells it is not estimable. Its interval uses the stratified variance of both groups, not clipped at 0 or 100%.

<!-- END GENERATED: weighted -->

What the weighted analysis can say: the margin rule's `incorrect` targets on the AI Disagrees are confirmed by the
expert 73.1% [58.3, 87.9] of the time in round 1's population and 82.0% [71.7, 92.4] in round 2's. What it cannot
say: a population-weighted kept-minus-dropped difference. Kept and dropped rows fall in different cells (round 1
drew 20 voted Disagrees per type, so no type holds 5 of both; round 2 has one city-type cell with 5 of both), so
no common set of cells exists to average over, and P1 rests on the unweighted sample rates above. An earlier
version of this table printed weighted kept and dropped shares averaged over different cells, and an interval from
clipped bounds; both are gone.

### The vote rules on the gold set (descriptive)

<!-- BEGIN GENERATED: rules (analysis/target_scoreboard.py) -->

**Decided AI Disagrees**

| Rule | Round | `incorrect` target: expert agrees | `correct` target: expert agrees |
|---|---|---|---|
| reference: expert says incorrect | round 1 | 106/184 (57.6% [50.4, 64.5]) | |
| reference: expert says incorrect | round 2 | 182/304 (59.9% [54.3, 65.2]) | |
| reference: expert says incorrect | pooled (sample rate; descriptive) | 288/488 (59.0% [54.6, 63.3]) | |
| `majority` | round 1 | 41/55 (74.5% [61.7, 84.2]) | 29/39 (74.4% [58.9, 85.4]) |
| `majority` | round 2 | 101/122 (82.8% [75.1, 88.5]) | 54/80 (67.5% [56.6, 76.8]) |
| `majority` | pooled (sample rate; descriptive) | 142/177 (80.2% [73.7, 85.4]) | 83/119 (69.7% [61.0, 77.3]) |
| `majority2` | round 1 | 40/54 (74.1% [61.1, 83.9]) | 28/38 (73.7% [58.0, 85.0]) |
| `majority2` | round 2 | 101/122 (82.8% [75.1, 88.5]) | 54/80 (67.5% [56.6, 76.8]) |
| `majority2` | pooled (sample rate; descriptive) | 141/176 (80.1% [73.6, 85.3]) | 82/118 (69.5% [60.7, 77.1]) |
| `symmetric2` | round 1 | 34/46 (73.9% [59.7, 84.4]) | 26/34 (76.5% [60.0, 87.6]) |
| `symmetric2` | round 2 | 88/104 (84.6% [76.5, 90.3]) | 53/73 (72.6% [61.4, 81.5]) |
| `symmetric2` | pooled (sample rate; descriptive) | 122/150 (81.3% [74.3, 86.8]) | 79/107 (73.8% [64.8, 81.2]) |
| `margin` | round 1 | 34/46 (73.9% [59.7, 84.4]) | 9/10 (90.0% [59.6, 98.2]) |
| `margin` | round 2 | 88/104 (84.6% [76.5, 90.3]) | 43/55 (78.2% [65.6, 87.1]) |
| `margin` | pooled (sample rate; descriptive) | 122/150 (81.3% [74.3, 86.8]) | 52/65 (80.0% [68.7, 87.9]) |
| `symmetric3` | round 1 | 12/17 (70.6% [46.9, 86.7]) | 9/10 (90.0% [59.6, 98.2]) |
| `symmetric3` | round 2 | 42/48 (87.5% [75.3, 94.1]) | 43/55 (78.2% [65.6, 87.1]) |
| `symmetric3` | pooled (sample rate; descriptive) | 54/65 (83.1% [72.2, 90.3]) | 52/65 (80.0% [68.7, 87.9]) |
| `unanimous3` | round 1 | 10/13 (76.9% [49.7, 91.8]) | 7/8 (87.5% [52.9, 97.8]) |
| `unanimous3` | round 2 | 25/30 (83.3% [66.4, 92.7]) | 36/46 (78.3% [64.4, 87.7]) |
| `unanimous3` | pooled (sample rate; descriptive) | 35/43 (81.4% [67.4, 90.3]) | 43/54 (79.6% [67.1, 88.2]) |

**Decoys (AI Agree or Unsure)**

| Rule | Round | `incorrect` target: expert agrees | `correct` target: expert agrees |
|---|---|---|---|
| reference: expert says incorrect | round 1 | 8/92 (8.7% [4.5, 16.2]) | |
| reference: expert says incorrect | round 2 | 17/151 (11.3% [7.1, 17.3]) | |
| reference: expert says incorrect | pooled (sample rate; descriptive) | 25/243 (10.3% [7.1, 14.7]) | |
| `majority` | round 1 | 2/5 (40.0% [11.8, 76.9]) | 46/49 (93.9% [83.5, 97.9]) |
| `majority` | round 2 | 5/7 (71.4% [35.9, 91.8]) | 83/88 (94.3% [87.4, 97.5]) |
| `majority` | pooled (sample rate; descriptive) | 7/12 (58.3% [32.0, 80.7]) | 129/137 (94.2% [88.9, 97.0]) |
| `majority2` | round 1 | 2/5 (40.0% [11.8, 76.9]) | 29/31 (93.5% [79.3, 98.2]) |
| `majority2` | round 2 | 3/3 (100.0% [43.8, 100.0]) | 41/45 (91.1% [79.3, 96.5]) |
| `majority2` | pooled (sample rate; descriptive) | 5/8 (62.5% [30.6, 86.3]) | 70/76 (92.1% [83.8, 96.3]) |
| `symmetric2` | round 1 | 2/4 (50.0% [15.0, 85.0]) | 29/31 (93.5% [79.3, 98.2]) |
| `symmetric2` | round 2 | 3/3 (100.0% [43.8, 100.0]) | 40/44 (90.9% [78.8, 96.4]) |
| `symmetric2` | pooled (sample rate; descriptive) | 5/7 (71.4% [35.9, 91.8]) | 69/75 (92.0% [83.6, 96.3]) |
| `margin` | round 1 | 2/4 (50.0% [15.0, 85.0]) | 16/18 (88.9% [67.2, 96.9]) |
| `margin` | round 2 | 3/3 (100.0% [43.8, 100.0]) | 20/20 (100.0% [83.9, 100.0]) |
| `margin` | pooled (sample rate; descriptive) | 5/7 (71.4% [35.9, 91.8]) | 36/38 (94.7% [82.7, 98.5]) |
| `symmetric3` | round 1 | 1/2 (50.0% [9.5, 90.5]) | 16/18 (88.9% [67.2, 96.9]) |
| `symmetric3` | round 2 | 2/2 (100.0% [34.2, 100.0]) | 20/20 (100.0% [83.9, 100.0]) |
| `symmetric3` | pooled (sample rate; descriptive) | 3/4 (75.0% [30.1, 95.4]) | 36/38 (94.7% [82.7, 98.5]) |
| `unanimous3` | round 1 | 1/1 (100.0% [20.7, 100.0]) | 10/11 (90.9% [62.3, 98.4]) |
| `unanimous3` | round 2 | 0/0 | 14/14 (100.0% [78.5, 100.0]) |
| `unanimous3` | pooled (sample rate; descriptive) | 1/1 (100.0% [20.7, 100.0]) | 24/25 (96.0% [80.5, 99.3]) |

<!-- END GENERATED: rules -->

Coverage is left out on purpose (see Choices). The margin rule's `incorrect` targets agree with the expert 73.9% of
the time in round 1 and 84.6% in round 2; the point estimates of the rules nested inside it (`symmetric3`,
`unanimous3`) are within about 3 points of it in each round, with intervals 20 to 40 points wide. That is P1 seen
from the other side, and the gain line under the P1 table is its test. Every rule's `correct` target on an AI Disagree
is right 67 to 90% of the time: those are the AI's errors the crowd caught. On the decoys every rule's `correct`
target is right at least 88% of the time.

### Every split variant as a detector of wrong train targets (descriptive)

<!-- BEGIN GENERATED: detector (analysis/target_scoreboard.py) -->

| Variant | Target class | Rows | Touched: wrong target | Untouched: wrong target | touched minus untouched [95%] | Fisher p |
|---|---|---|---|---|---|---:|
| `symmetric` | incorrect | 105 | 14/61 (23.0% [14.2, 34.9]) | 6/44 (13.6% [6.4, 26.7]) | +9.3 pts [-5.6, +23.6] | 0.315 |
| `symmetric` | correct | 26 | 0/0 | 3/26 (11.5% [4.0, 29.0]) | n/a | 1 |
| `symmetric` | all (descriptive pool) | 131 | 14/61 (23.0% [14.2, 34.9]) | 9/70 (12.9% [6.9, 22.7]) | +10.1 pts [-3.3, +23.0] | 0.168 |
| `unanimous3` | incorrect | 105 | 18/81 (22.2% [14.5, 32.4]) | 2/24 (8.3% [2.3, 25.8]) | +13.9 pts [-1.1, +26.8] | 0.152 |
| `unanimous3` | correct | 26 | 1/9 (11.1% [2.0, 43.5]) | 2/17 (11.8% [3.3, 34.3]) | -0.7 pts [-23.5, +30.0] | 1 |
| `unanimous3` | all (descriptive pool) | 131 | 19/90 (21.1% [14.0, 30.6]) | 4/41 (9.8% [3.9, 22.5]) | +11.4 pts [-1.9, +23.3] | 0.141 |
| `cl-drop` | incorrect | 105 | 3/5 (60.0% [23.1, 88.2]) | 17/100 (17.0% [10.9, 25.5]) | +43.0 pts [-16.7, +86.3] | 0.0466 |
| `cl-drop` | correct | 26 | 1/8 (12.5% [2.2, 47.1]) | 2/18 (11.1% [3.1, 32.8]) | +1.4 pts [-22.2, +33.3] | 1 |
| `cl-drop` | all (descriptive pool) | 131 | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.1, +43.2] | 0.242 |
| `cl-flip` | incorrect | 105 | 3/5 (60.0% [23.1, 88.2]) | 17/100 (17.0% [10.9, 25.5]) | +43.0 pts [-14.6, +86.2] | 0.0466 |
| `cl-flip` | correct | 26 | 1/8 (12.5% [2.2, 47.1]) | 2/18 (11.1% [3.1, 32.8]) | +1.4 pts [-23.5, +33.3] | 1 |
| `cl-flip` | all (descriptive pool) | 131 | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.8, +43.5] | 0.242 |

Rounds pooled (round 1 has 37 train rows); per-round rows of every cell are in `scoreboard.json`. Touched = dropped (`symmetric`, `unanimous3`, `cl-drop`) or flipped (`cl-flip`).

<!-- END GENERATED: detector -->

Split by the target's class, because the strata oversample `incorrect` targets (round 2 capped `margin_correct`),
so a pooled recall or false-alarm rate reflects the sample's mix. Inside `incorrect` targets, `cl-drop` touches 5
rows and 3 are wrong (17/100 among the rest), Fisher p 0.047, the only Fisher p under 0.05 in the table (`cl-flip`
touches the same rows); it is a descriptive cell of 5 rows among 12 rows of comparisons, not a finding. The two
rule variants touch rows with a wrong target about twice as often as the rows they leave (`symmetric` 23.0% against
12.9%, +10.1 pts [-3.3, +23.0]; `unanimous3` +11.4 pts [-1.9, +23.3]); neither interval excludes 0, and neither
excludes a difference of about 23 points. These are the counts the confident-learning report's `--gold` rerun
promised (131 rated train rows instead of 37); they equal `confident_learning.expert_agreement` summed over types (check B,
asserted in the code).

### Does a score rank wrong targets? (descriptive)

<!-- BEGIN GENERATED: ranking (analysis/target_scoreboard.py) -->

| Score | Target class | Items | Wrong targets | AUC [95%, panorama bootstrap] |
|---|---|---:|---:|---|
| oof (train rows) | correct | 26 | 3 | 0.630 |
| oof (train rows) | incorrect | 105 | 20 | 0.769 [0.642, 0.890] |
| vote margin (AI Disagrees) | correct | 65 | 13 | 0.477 [0.337, 0.607] |
| vote margin (AI Disagrees) | incorrect | 150 | 28 | 0.543 [0.441, 0.640] |

No interval when the smaller class has fewer than 8 items.

<!-- END GENERATED: ranking -->

Within `incorrect` targets the out-of-fold probability of the opposite class ranks the 20 wrong targets above the
85 right ones with AUC 0.769 [0.642, 0.890], while the binary flag (a threshold on the same score) catches 3 of
them. The vote lead's AUC intervals span 0.5 (0.543 [0.441, 0.640] within `incorrect`, 0.477 [0.337, 0.607]
within `correct`): this set cannot tell whether a larger lead marks a more reliable target, and does not exclude a
modest ranking (up to 0.64), which matches P1's unresolved reading. The out-of-fold score is a candidate for a
pre-stated test on the next gold round, not a result here: this AUC was not a primary.

### Slices (descriptive)

<!-- BEGIN GENERATED: slices (analysis/target_scoreboard.py) -->

**Decided AI Disagrees by type x severity: the expert says incorrect** (rounds pooled; rate blank under 20 items; train = round-3 train rows of the slice; 10 of 20 cells reach 20)

| Type | Slice | Expert says incorrect | Train rows |
|---|---|---|---:|
| CurbRamp | 1 | 51/84 (60.7% [50.0, 70.5]) | 5,083 |
| CurbRamp | 2 | 8/22 (36.4% [19.7, 57.0]) | 837 |
| NoCurbRamp | 1 | 16/21 (76.2% [54.9, 89.4]) | 1,839 |
| NoCurbRamp | 2 | 17/25 (68.0% [48.4, 82.8]) | 3,595 |
| NoCurbRamp | 3 | 16/36 (44.4% [29.5, 60.4]) | 6,074 |
| Obstacle | 1 | 32/38 (84.2% [69.6, 92.6]) | 3,142 |
| SurfaceProblem | 1 | 20/45 (44.4% [30.9, 58.8]) | 5,050 |
| Crosswalk | 1 | 17/51 (33.3% [22.0, 47.0]) | 1,394 |
| Crosswalk | 2 | 10/22 (45.5% [26.9, 65.3]) | 187 |
| Crosswalk | 3 | 15/24 (62.5% [42.7, 78.8]) | 369 |

Below 20 items: 10 cells holding 120 items, not shown (no rate is read from them; their counts are in `scoreboard.json`).

**Decided AI Disagrees by type x tag: the expert says incorrect** (rounds pooled; rate blank under 20 items; train = round-3 train rows of the slice; 6 of 48 cells reach 20)

| Type | Slice | Expert says incorrect | Train rows |
|---|---|---|---:|
| CurbRamp | missing tactile warning | 10/32 (31.2% [18.0, 48.6]) | 1,482 |
| CurbRamp | (none) | 45/66 (68.2% [56.2, 78.2]) | 3,933 |
| NoCurbRamp | no alternate route | 14/27 (51.9% [34.0, 69.3]) | 4,670 |
| NoCurbRamp | (none) | 34/47 (72.3% [58.2, 83.1]) | 4,631 |
| Crosswalk | paint fading | 11/40 (27.5% [16.1, 42.8]) | 339 |
| Crosswalk | (none) | 34/56 (60.7% [47.6, 72.4]) | 1,557 |

Below 20 items: 42 cells holding 306 items, not shown (no rate is read from them; their counts are in `scoreboard.json`).

**P1 by severity** (types pooled; a group under 20 items shows its size only)

| Severity | kept by `symmetric` | dropped | kept minus dropped (Fisher p) |
|---|---|---|---|
| 1 | 27/29 (93.1% [78.0, 98.1]) | 30/41 (73.2% [58.1, 84.3]) | +19.9 pts (0.0587) |
| 2 | n 12 | 18/20 (90.0% [69.9, 97.2]) |  |
| 3 | n 17 | n 16 |  |
| (none) | n 7 | n 8 |  |

<!-- END GENERATED: slices -->

Among the 16 cells that reach 20, the AI's Disagrees are least often right where the feature is present but
degraded: CurbRamp tagged `missing tactile warning` (31.2% [18.0, 48.6]), Crosswalk tagged `paint fading` (27.5%
[16.1, 42.8]), low-severity Crosswalk (33.3% [22.0, 47.0]) and SurfaceProblem (44.4% [30.9, 58.8]). These cells
were picked by looking, so they are a pattern to test, not a finding; it is the rating-rule question of
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74)'s first decision box, seen in the gold
set, and the pattern round 1 reported. P1 by severity has one row where both groups reach 20 (severity 1: kept
27/29 against dropped 30/41, +19.9 pts, Fisher p 0.059). That is a descriptive slice among four, not pre-stated,
and it does not reach 0.05; it is a reason to slice P1 by severity in the next round's pre-stated analysis, not a
result.

### Sensitivity

<!-- BEGIN GENERATED: sensitivity (analysis/target_scoreboard.py) -->

| Contrast | Gold | Tallies | kept / flagged | dropped / unflagged | difference [95%] |
|---|---|---|---|---|---|
| P1 | masked-only | expert votes removed | 54/65 (83.1% [72.2, 90.3]) | 68/85 (80.0% [70.3, 87.1]) | +3.1 pts [-9.4, +15.9] |
| P2 | masked-only | n/a (train targets) | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.1, +43.2] |
| P1 | masked-only | tallies as pulled | 54/65 (83.1% [72.2, 90.3]) | 69/86 (80.2% [70.6, 87.3]) | +2.8 pts [-9.9, +15.3] |
| P1 | with meeting verdicts | expert votes removed | 55/66 (83.3% [72.6, 90.4]) | 68/85 (80.0% [70.3, 87.1]) | +3.3 pts [-9.5, +15.9] |
| P2 | with meeting verdicts | n/a (train targets) | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.1, +43.2] |
| P1 | with meeting verdicts | tallies as pulled | 55/66 (83.3% [72.6, 90.4]) | 69/86 (80.2% [70.6, 87.3]) | +3.1 pts [-9.4, +15.9] |

Pooled rows. Masked-only gold leaves out the decided items reached in the round-2 meeting; `expert votes removed` takes a rater's earlier production vote out of the pool tallies.

<!-- END GENERATED: sensitivity -->

Neither choice moves a primary: putting the 10 meeting verdicts back changes P1 by 0.2 points and P2 not at all,
and leaving the rater's earlier vote in the tallies adds one row to P1's dropped group (with the vote taken out,
that item has a single Disagree and no margin target).

### Cross-check against the numbers computed by hand

<!-- BEGIN GENERATED: crosscheck (analysis/target_scoreboard.py) -->

Adjudicated gold, tallies as pulled (the configuration the plan and its review used by hand):

- decided 741, decided AI Disagrees 497, decided train rows 131, flagged 13;
- `margin` covers 219, `incorrect` 152 at 81.6%; `symmetric3` covers 133, `incorrect` 66 at 83.3%; `unanimous3` covers 99, `incorrect` 44 at 81.8%; `majority` covers 303 at 75.2%, and `majority` gives the same verdict as `majority2` on every one of these items;
- confident learning: 4 of 13 flagged rows have a wrong target, 19 of 118 unflagged;
- pool (91,626 AI Disagrees) given any target: `majority` 59.5%, `majority2` 31.5%, `margin` 19.8%, `symmetric2` 25.7%, `symmetric3` 11.7%, `unanimous3` 6.4%.

<!-- END GENERATED: crosscheck -->

The plan and its review each computed these from the same gold table with throwaway code before this module
existed, and the module reproduces every one (`tests/test_target_scoreboard.py` binds them to the committed JSON).
The review's bootstrap interval for P1 in that configuration was [-9.2, +15.8]; this module's is [-9.4, +15.9],
the same point with a different resampling stream.

## What this means

- **[#75](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75), the stricter rules.** No
  target-accuracy reason to adopt or reject them: P1 cannot separate them from the margin rule, and the gain from
  applying `symmetric` is +1.7 pts [-5.2, +8.6]. The training-set result already argues against filtering with them
  without rebalancing ([`confident-learning-2026-10.md`](confident-learning-2026-10.md), Recommendation 2). Settling
  the target question needs about three times the margin-`incorrect` rows of rounds 1 + 2 (about 200 per group for
  70 to 80% power against a 10-point kept-minus-dropped difference).
- **[#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76), confident learning.** Recommendation 3
  of the confident-learning doc (send flags to an expert rather than flip or drop them unread) is kept, with 131
  rated rows instead of 37, as a judgement on the point estimate: 4 of 13 flagged targets are wrong (30.8%), but
  the interval [12.7, 57.6] does not exclude a majority, so the data neither license nor rule out acting on the
  flags unread. About 30 rated flagged rows would settle it. The continuous out-of-fold score looks more useful
  than the flag as a ranking for that expert packet (descriptive).
- **[#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12), reliability weighting.** Not re-scored
  here. [`adjudicator-2026-11.md`](adjudicator-2026-11.md) records
  that the weights reverse 0 of 142,275 Seattle majorities and only break ties, and the gold set has no tied AI
  Disagree, so it cannot score the one thing the weights change.
- **[#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77), the VLM adjudicator.** The only
  remaining candidate that could change round 4's targets, and the only one that gives a verdict on the 80% of the
  pool the margin rule leaves without a target. It needs the klone run.
- **[#106](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/106), the next stratum.** One-vote labels and ties are 34.6% of the AI-Disagree pool and 47% of what an
  any-majority rule decides; no rate here covers them.

## Provenance

<!-- BEGIN GENERATED: provenance (analysis/target_scoreboard.py) -->

| Item | Pin |
|---|---|
| gold.csv | `2b472651cb01011a8640b86eb4103e1794ab482186ba3a3471e7d019273edcae` (matches the pin in `docs/adjudicator-2026-11.md`: yes) |
| relabel_pool.csv | `dd5a6d363ce7f6c05ac6faf250f2526090bd2199541e6164b76029cd35483d09` |
| round-1 manifest (population) | `d5c2011386c253283e27fdbd77931a52f7f233cfd092b7343f9cbcb617bc055d` |
| round-2 recount / design | `5e9a18e193b970ad79809b4ce034001a79009fc75a11454a964a846e74c9de47` / `499fb9dfe5d976651f08d932d9a0c47477386e119080ae2107c91fc5c7404e57` |
| round-3 splits, noisy targets and variant splits (sha256 of the sorted per-file list) | `761fb89088deea286859cfbe7c600a92f2894ed4e5fb5bd2aebcd9cf66e1dcff` (30 files) |
| verdicts.csv (private) | `e17a408a5db6c49dbe7e7ee2deef83c230a84068f383a32be033f5af6382e4f1` |
| code | commit `0f1a324698473d632b1ce35e01e51f36d714f769`; Python 3.10.19, pandas 2.3.3, numpy 2.2.6 |
| intervals | 2000 label-level resamples (AUC: 1,000 panorama resamples), seed 0 |
| public-deployment list | live `/v3/api/cities`, read 2026-11-13T15:39:45+00:00 (46 public) |

<!-- END GENERATED: provenance -->

The per-label verdicts (`verdicts.csv`), `run.json` (argv, every input path and sha256, the public-deployment list
used) and a copy of `scoreboard.json` are in `~/Data/sidewalk-validator-ai-target-scoreboard-2026-11/`, outside the
repo.

## Reproducing

```bash
R=~/Data/sidewalk-validator-ai-relabel-2026-10; ADJ=~/Data/sidewalk-validator-ai-adjudicator-2026-11
DATA=~/Data/sidewalk-validator-ai-cl-2026-10/retrain-2026-10
# 1. the gold table: docs/adjudicator-2026-11.md, Reproducing step 1 (gold.sha256 must be 2b472651...edcae)
# 2. the scoreboard (one HTTPS read of /v3/api/cities for the privacy check; no klone, no database)
python analysis/target_scoreboard.py score --gold $ADJ/gold.csv --pool $R/relabel_pool.csv --split-root $DATA \
    --r1-manifest $R/private-key-do-not-share/manifest.before-rating.json \
    --out-dir ~/Data/sidewalk-validator-ai-target-scoreboard-2026-11
python analysis/target_scoreboard.py report            # splice the blocks; --check verifies without writing
```

The run on 2026-11-13 used `uv run --no-project --python 3.10 --with 'pandas<3' --with numpy --with scikit-learn
--with matplotlib --with requests`. Seed 0 and 2,000 resamples are the defaults.
