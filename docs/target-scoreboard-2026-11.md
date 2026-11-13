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
Reproducing step 1 and its sha256 matches the pin there.

<!-- BEGIN GENERATED: gold_counts (analysis/target_scoreboard.py) -->
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
<!-- END GENERATED: pool -->

A third of the pool has no decisive vote and is decided by no vote rule; the gold set's `lone` stratum scores the AI
on those. One-vote labels (28.0%) and ties (6.6%) are what separates `majority` from `majority2`, and no gold round
has sampled them. `symmetric` gives a target to 11.7% of the AI Disagrees against 19.8% for the margin rule; nearly
all of the difference is two-vote Disagrees (16.3% of the pool get an `incorrect` margin target, 8.2% under
`symmetric`).

### The primary contrasts

<!-- BEGIN GENERATED: primary (analysis/target_scoreboard.py) -->
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
<!-- END GENERATED: weighted -->

Weighting does not change the reading: round 2 is +5.8 pts [-14.6, +26.3]. Round 1's estimate rests on few cells
(round 1 drew 20 voted Disagrees per type, so most type cells hold under 5 kept or dropped rows) and is not
informative.

### The vote rules on the gold set (descriptive)

<!-- BEGIN GENERATED: rules (analysis/target_scoreboard.py) -->
<!-- END GENERATED: rules -->

Coverage is left out on purpose (see Choices). The margin rule's `incorrect` targets agree with the expert 73.9% of
the time in round 1 and 84.6% in round 2; the rules nested inside it (`symmetric3`, `unanimous3`) stay within a few
points of it in each round, which is P1 seen from the other side. Every rule's `correct` target on an AI Disagree
is right 67 to 90% of the time: those are the AI's errors the crowd caught. On the decoys every rule's `correct`
target is right at least 88% of the time.

### Every split variant as a detector of wrong train targets (descriptive)

<!-- BEGIN GENERATED: detector (analysis/target_scoreboard.py) -->
<!-- END GENERATED: detector -->

Split by the target's class, because the strata oversample `incorrect` targets (round 2 capped `margin_correct`),
so a pooled recall or false-alarm rate reflects the sample's mix. Inside `incorrect` targets, `cl-drop` touches 5
rows and 3 are wrong (17/100 among the rest), the only cell whose Fisher p is under 0.05; it is a descriptive row
of 5, one of 12, and not a finding. These are the counts the confident-learning report's `--gold` rerun promised
(131 rated train rows instead of 37); they equal `confident_learning.expert_agreement` summed over types (check B,
asserted in the code).

### Does a score rank wrong targets? (descriptive)

<!-- BEGIN GENERATED: ranking (analysis/target_scoreboard.py) -->
<!-- END GENERATED: ranking -->

Within `incorrect` targets the out-of-fold probability of the opposite class ranks the 20 wrong targets above the
85 right ones with AUC 0.769 [0.642, 0.890], while the binary flag (a threshold on the same score) catches 3 of
them. The size of the vote lead does not rank wrong targets (0.543 [0.441, 0.640] within `incorrect`, 0.477
within `correct`), which is P1's reading in continuous form. The out-of-fold score is a candidate for a
pre-stated test on the next gold round, not a result here: this AUC was not a primary.

### Slices (descriptive)

<!-- BEGIN GENERATED: slices (analysis/target_scoreboard.py) -->
<!-- END GENERATED: slices -->

The AI's Disagrees are least often right where the feature is present but degraded: CurbRamp tagged `missing
tactile warning` (31.2%), Crosswalk tagged `paint fading` (27.5%), low-severity Crosswalk (33.3%) and
SurfaceProblem (44.4%). That is the rating-rule question of #74's first decision box, seen in the gold set, and the
pattern round 1 reported. P1 by severity has one row where both groups reach 20 (severity 1: kept 27/29 against
dropped 30/41); the rows are descriptive and their intervals overlap.

### Sensitivity

<!-- BEGIN GENERATED: sensitivity (analysis/target_scoreboard.py) -->
<!-- END GENERATED: sensitivity -->

Neither choice moves a primary: putting the 10 meeting verdicts back changes P1 by 0.2 points and P2 not at all,
and leaving the rater's earlier vote in the tallies adds one row to P1's dropped group (with the vote taken out,
that item has a single Disagree and no margin target).

### Cross-check against the numbers computed by hand

<!-- BEGIN GENERATED: crosscheck (analysis/target_scoreboard.py) -->
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
