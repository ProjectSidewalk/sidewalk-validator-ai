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

- **`symmetric` against the training margin rule (P1).** On the AI Disagrees whose margin target is `incorrect`,
  the expert says incorrect on 54/65 rows `symmetric` keeps and 68/85 rows it drops: kept minus dropped
  +3.1 pts [-9.4, +15.9]. The data rule out that the dropped two-vote Disagrees are mostly wrong: the expert
  confirms them at least 70.3% of the time (Wilson lower bound). They cannot rule out a gain of up to about 16
  points from the stricter rule, or a loss of about 9. This is not evidence that stricter rules do not help; it is
  a difference this gold set is too small to resolve.
- **Confident learning (P2).** On the 131 decided round-3 train rows, 4/13 flagged rows have a wrong target against
  19/118 unflagged: +14.7 pts [-10.1, +43.2], Fisher p 0.242. The flags catch 4 of the 23 wrong targets. The data
  rule out using the flags as an automatic relabeller (flipping all 13 would fix 4 targets and break 9). They
  cannot tell whether flagged rows are more often wrong than the rest.
- **Coverage is a property of the pool, not of the gold set.** Of 91,626 AI Disagrees in the pool, 28.0% have one
  decisive vote and 6.6% are ties. No gold item comes from either, so every rate below is silent on them, and they
  are 47% of what an any-majority rule decides.

## Why

The training margin rule (`incorrect` at net Disagree >= 2, `correct` at net Agree >= 3) matches an expert about
three times in four on contested labels. #74 asks which method should replace or filter it, scored on one gold set
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

Each primary is shown per round first; the pooled row is a sample rate printed beside a heterogeneity check (round
1's difference minus round 2's, resampled within round). **Reading rule:** an interval that excludes 0 is a
difference; one that spans 0 is reported as what it rules out at each end, never as "no effect". No correction for
multiplicity: there are two primaries, and every other table is labelled descriptive. A difference interval is not
printed when either group has fewer than 5 items.

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
  type), round 2 by the #85 recount per (city, stratum, type). A domain inside a cell (kept or dropped rows) is
  weighted by the cell's population times its share of the cell's decided items. A cell with fewer than 5 domain
  items gets no weight (the rule of `gt_rounds.score`). The rounds are never weighted together: round 2's two
  cities are part of round 1's population.
- **Slices.** The standing type x severity and type x tag rule is met on the method-independent quantity (the
  share of AI Disagrees the expert calls incorrect, with the round-3 train rows of each slice) at a minimum of 20
  items per cell, and P1 is sliced by severity pooled over types. Method-by-slice tables would mostly measure which
  strata each method covers.

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

**P1. `symmetric` kept against dropped** (decided AI Disagrees whose margin target is `incorrect`; outcome: the expert says incorrect):

| Rows | kept: net Disagree >= 3 | dropped: net Disagree = 2 | kept minus dropped [95%] | Fisher p |
|---|---|---|---|---:|
| round 1 | 12/17 (70.6% [46.9, 86.7]) | 22/29 (75.9% [57.9, 87.8]) | -5.3 pts [-31.3, +21.1] | 0.737 |
| round 2 | 42/48 (87.5% [75.3, 94.1]) | 46/56 (82.1% [70.2, 90.0]) | +5.4 pts [-8.4, +18.9] | 0.588 |
| pooled (sample rate; descriptive) | 54/65 (83.1% [72.2, 90.3]) | 68/85 (80.0% [70.3, 87.1]) | +3.1 pts [-9.4, +15.9] | 0.677 |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): -10.6 pts [-40.7, +21.2].

**P2. Confident-learning flag** (decided round-3 train rows, both target classes; outcome: the target differs from the expert):

| Rows | flagged | unflagged | flagged minus unflagged [95%] | Fisher p | Recall | False alarms |
|---|---|---|---|---:|---|---|
| round 1 | 0/2 (0.0% [0.0, 65.8]) | 9/35 (25.7% [14.2, 42.1]) | -25.7 pts | 1 | 0/9 (0.0% [0.0, 29.9]) | 2/28 (7.1% [2.0, 22.6]) |
| round 2 | 4/11 (36.4% [15.2, 64.6]) | 10/83 (12.0% [6.7, 20.8]) | +24.3 pts [-4.1, +56.4] | 0.0559 | 4/14 (28.6% [11.7, 54.6]) | 7/80 (8.8% [4.3, 17.0]) |
| pooled (sample rate; descriptive) | 4/13 (30.8% [12.7, 57.6]) | 19/118 (16.1% [10.6, 23.8]) | +14.7 pts [-10.1, +43.2] | 0.242 | 4/23 (17.4% [7.0, 37.1]) | 9/108 (8.3% [4.4, 15.1]) |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): n/a.

Descriptive twin of P1, `unanimous3` (same rows):

| Rows | kept: unanimous | dropped | kept minus dropped [95%] | Fisher p |
|---|---|---|---|---:|
| round 1 | 10/13 (76.9% [49.7, 91.8]) | 24/33 (72.7% [55.8, 84.9]) | +4.2 pts [-26.3, +29.6] | 1 |
| round 2 | 25/30 (83.3% [66.4, 92.7]) | 63/74 (85.1% [75.3, 91.5]) | -1.8 pts [-18.4, +13.3] | 0.773 |
| pooled (sample rate; descriptive) | 35/43 (81.4% [67.4, 90.3]) | 87/107 (81.3% [72.9, 87.6]) | +0.1 pts [-14.5, +13.2] | 1 |

Heterogeneity check (round 1's difference minus round 2's, resampled within round): +6.0 pts [-26.5, +35.9].

<!-- END GENERATED: primary -->

**P1.** Pooled, kept 54/65 against dropped 68/85, +3.1 pts [-9.4, +15.9]; neither round's interval excludes 0
(round 1 -5.3, round 2 +5.4), and the rounds do not differ detectably (heterogeneity -10.6 pts [-40.7, +21.2]).
What this rules out: that the two-vote Disagrees `symmetric` drops are mostly wrong targets (the expert confirms
68/85, lower bound 70.3%), and a gain from the stricter rule larger than about 16 points. What it cannot rule out:
a real gain of 5 to 15 points, or a loss of up to 9. About 200 rows per group would resolve a 10-point difference
at these rates; this set has 65 and 85. `unanimous3` gives the same picture (+0.1 pts [-14.5, +13.2]).

**P2.** Pooled, flagged 4/13 against unflagged 19/118, +14.7 pts [-10.1, +43.2], Fisher p 0.242. Round 1 has 2
flagged rows, so no round-1 interval and no heterogeneity check; round 2 alone is 4/11 against 10/83. Recall is
4/23 wrong targets, and 9 of the 13 flagged targets are right. What this rules out: flipping or dropping the flags
unread (it would remove or break more right targets than wrong ones). What it cannot rule out: that flagged rows
are enriched for wrong targets, which the point estimate suggests and the interval does not settle.

### Population-weighted P1, each round on its own

<!-- BEGIN GENERATED: weighted (analysis/target_scoreboard.py) -->

| Round | margin-`incorrect` AI Disagrees | kept by `symmetric` | dropped | kept minus dropped [95%, normal approx.] | cells left without weight (n < 5) |
|---|---|---|---|---|---|
| 1 | 73.1% [58.3, 87.9] (n 43) | 72.8% [42.3, 100.0] (n 12) | 70.3% [48.9, 91.6] (n 20) | +2.6 pts [-33.3, +38.5] | all: 1; kept: 3; dropped: 3 |
| 2 | 82.0% [71.7, 92.4] (n 104) | 90.3% [71.9, 100.0] (n 34) | 84.4% [69.6, 99.3] (n 42) | +5.8 pts [-14.6, +26.3] | all: 0; kept: 5; dropped: 5 |

<!-- END GENERATED: weighted -->

Weighting does not change the reading: round 2 is +5.8 pts [-14.6, +26.3]. Round 1's estimate rests on few cells
(round 1 drew 20 voted Disagrees per type, so most type cells hold under 5 kept or dropped rows) and is not
informative.

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
the time in round 1 and 84.6% in round 2; the rules nested inside it (`symmetric3`, `unanimous3`) stay within a few
points of it in each round, which is P1 seen from the other side. Every rule's `correct` target on an AI Disagree
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
touches the same rows); it is a descriptive cell of 5 rows among 12 rows of comparisons, not a finding. These are the counts the confident-learning report's `--gold` rerun promised
(131 rated train rows instead of 37); they equal `confident_learning.expert_agreement` summed over types (check B,
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
them. The size of the vote lead does not rank wrong targets (0.543 [0.441, 0.640] within `incorrect`, 0.477
within `correct`), which is P1's reading in continuous form. The out-of-fold score is a candidate for a
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

Below 20 items (10 cells, counts only): CurbRamp 3 3/8; CurbRamp (none) 4/5; NoCurbRamp (none) 17/19; Obstacle 2 13/17; Obstacle 3 15/18; Obstacle (none) 5/6; SurfaceProblem 2 13/18; SurfaceProblem 3 3/10; SurfaceProblem (none) 4/6; Crosswalk (none) 9/13.

**Decided AI Disagrees by type x tag: the expert says incorrect** (rounds pooled; rate blank under 20 items; train = round-3 train rows of the slice; 6 of 48 cells reach 20)

| Type | Slice | Expert says incorrect | Train rows |
|---|---|---|---:|
| CurbRamp | missing tactile warning | 10/32 (31.2% [18.0, 48.6]) | 1,482 |
| CurbRamp | (none) | 45/66 (68.2% [56.2, 78.2]) | 3,933 |
| NoCurbRamp | no alternate route | 14/27 (51.9% [34.0, 69.3]) | 4,670 |
| NoCurbRamp | (none) | 34/47 (72.3% [58.2, 83.1]) | 4,631 |
| Crosswalk | paint fading | 11/40 (27.5% [16.1, 42.8]) | 339 |
| Crosswalk | (none) | 34/56 (60.7% [47.6, 72.4]) | 1,557 |

Below 20 items (42 cells, counts only): CurbRamp not level with street 9/13; CurbRamp points into traffic 3/10; CurbRamp surface problem 2/10; CurbRamp narrow 1/5; CurbRamp not enough landing space 1/4; CurbRamp steep 1/3; NoCurbRamp alternate route present 10/18; NoCurbRamp unclear if needed 9/11; Obstacle pole 18/19; Obstacle tree 14/17; Obstacle vegetation 7/12; Obstacle narrow 8/9; Obstacle sign 6/7; Obstacle parked car 4/4; Obstacle construction 3/3; Obstacle parked scooter/motorcycle 1/3; Obstacle trash/recycling can 2/3; Obstacle garage entrance 2/2; Obstacle fire hydrant 1/1; Obstacle height difference 1/1; Obstacle litter/garbage 1/1; Obstacle stairs 0/1; Obstacle (none) 9/10; SurfaceProblem grass 6/19; SurfaceProblem cracks 7/17; SurfaceProblem narrow sidewalk 11/17; SurfaceProblem bumpy 4/9; SurfaceProblem uneven/slanted 2/8; SurfaceProblem brick/cobblestone 1/6; SurfaceProblem utility panel 4/6; SurfaceProblem debris 0/3; SurfaceProblem height difference 0/2; SurfaceProblem sand/gravel 0/2; SurfaceProblem very broken 1/2; SurfaceProblem construction 1/1; SurfaceProblem (none) 8/8; Crosswalk broken surface 4/10; Crosswalk brick/cobblestone 3/9; Crosswalk bumpy 4/9; Crosswalk uneven surface 2/6; Crosswalk very long crossing 1/4; Crosswalk rail/tram track 0/1.

**P1 by severity** (types pooled; rates blank under 20 items in a group)

| Severity | kept by `symmetric` | dropped |
|---|---|---|
| 1 | 27/29 (93.1% [78.0, 98.1]) | 30/41 (73.2% [58.1, 84.3]) |
| 2 | 7/12 | 18/20 (90.0% [69.9, 97.2]) |
| 3 | 14/17 | 13/16 |
| (none) | 6/7 | 7/8 |

<!-- END GENERATED: slices -->

The AI's Disagrees are least often right where the feature is present but degraded: CurbRamp tagged `missing
tactile warning` (31.2%), Crosswalk tagged `paint fading` (27.5%), low-severity Crosswalk (33.3%) and
SurfaceProblem (44.4%). That is the rating-rule question of #74's first decision box, seen in the gold set, and the
pattern round 1 reported. P1 by severity has one row where both groups reach 20 (severity 1: kept 27/29 against
dropped 30/41); the rows are descriptive and their intervals overlap.

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

- **#75, the stricter rules.** No target-accuracy reason to adopt or reject them: P1 cannot separate them from the
  margin rule. The training-set result already argues against filtering with them without rebalancing
  ([`confident-learning-2026-10.md`](confident-learning-2026-10.md), Recommendation 2). Settling the target question
  needs about three times the margin-`incorrect` rows of rounds 1 + 2.
- **#76, confident learning.** Recommendation 3 of the confident-learning doc stands with 131 rated rows instead of
  37: send flags to an expert, do not flip or drop them unread. The continuous out-of-fold score looks more useful
  than the flag as a ranking for that expert packet (descriptive).
- **#12, reliability weighting.** Not re-scored here. [`adjudicator-2026-11.md`](adjudicator-2026-11.md) records
  that the weights reverse 0 of 142,275 Seattle majorities and only break ties, and the gold set has no tied AI
  Disagree, so it cannot score the one thing the weights change.
- **#77, the VLM adjudicator.** The only remaining candidate that could change round 4's targets, and the only one
  that gives a verdict on the 80% of the pool the margin rule leaves without a target. It needs the klone run.
- **#106, the next stratum.** One-vote labels and ties are 34.6% of the AI-Disagree pool and 47% of what an
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
| code | commit `ebf10bde7ff20ad9c69204d8422298f369cd6a7f`; Python 3.10.19, pandas 2.3.3, numpy 2.2.6 |
| intervals | 2000 label-level resamples (AUC: 1,000 panorama resamples), seed 0 |
| public-deployment list | live `/v3/api/cities`, read 2026-11-13T14:56:33+00:00 (46 public) |

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
