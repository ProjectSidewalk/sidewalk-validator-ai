# A latent-truth model of the crowd's votes, against the expert gold set (November 2026)

Does an asymmetric Dawid–Skene model of who voted what agree with an expert more often than the `majority2` vote
rule? This is the proposed successor of [#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12)
(scalar per-validator reliability weighting), scored on the gold set and by the conventions of
[`target-scoreboard-2026-11.md`](target-scoreboard-2026-11.md), under the umbrella of
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74).

**Status:** pre-registered 2026-11-26. The sections up to and including "The rule" were written and committed, with
the code (`analysis/joint_vote_model.py`) and its synthetic-data tests, before any number involving a gold verdict
was computed. Before that commit the only real data read were the votes themselves (counts, the Unsure share and the
EM's run time, none of them gold); the gold table was read for its deployment list only. Results are spliced in
from `docs/joint-vote-model-2026-11/results.json` by `analysis/joint_vote_model.py report`.

## Why

#12's reliability weighting gives each validator one smoothed hit rate. Its weighted targets correlate 0.999 with
plain soft targets ([`training-data.md`](training-data.md#training-targets)), and on Seattle it reverses 0 of
142,275 majorities and only breaks ties ([`adjudicator-2026-11.md`](adjudicator-2026-11.md)). One rate cannot tell a
validator who agrees with everything from one who is careful, and the targets go wrong on the Disagree side: on the
contested AI Disagrees `majority2` matches the expert about three times in four
([`target-scoreboard-2026-11.md`](target-scoreboard-2026-11.md), the rules table). A model with two rates per
validator, sensitivity and specificity, can weigh someone's Agree and Disagree votes differently. If it does not
beat `majority2` on the same gold items, no training arm should be built on it.

## Pre-registration

### Data

- **Votes.** The `validations` column of the 2026-10-18 rawLabels snapshot that round 3 trained on
  (`round3/rawlabels`). Every file predates the first rating day, 2026-10-27, and the code refuses one that does
  not, so no vote cast after an expert had seen an item is in the fit.
- **Deployments.** The 28 deployments with at least one gold item; **all** labels of the five gold types
  (CurbRamp, NoCurbRamp, Obstacle, SurfaceProblem, Crosswalk) there, not only the gold items, enter the fit.
- **Which votes.** `validator_type == "Human"` only, so the AI account's votes never enter. Agree and Disagree
  only. A validator listed twice on a label keeps the last vote (`validator_reliability.human_votes`).
- **The expert accounts are left out of every fit, in every arm.** The gold raters are the accounts in `EXPERTS`
  ([`training-data.md`](training-data.md#training-signal)). Their votes are dropped before anything else, on every
  label, not only the gold items: a rater's production votes on other labels would otherwise shape the rates of
  everyone who voted alongside them, and through those rates the verdicts on gold items. The code exits when
  `EXPERTS` does not name both accounts, and asserts that no expert vote reached the fit. A label an expert placed
  keeps its crowd votes but gets no labeler parameter in the `labeler` arm.
- **Unsure votes are dropped, not modelled.** Three reasons. (1) Every rule on the scoreboard ignores Unsure, so
  dropping it keeps this a test of per-validator reliability rather than of extra evidence the baseline never
  sees. (2) Unsure is 3.3% of the non-expert human votes in the snapshot (read before this pre-registration from
  the votes alone), so it carries little information. (3) A third response doubles each validator's parameters
  (a 2 x 3 confusion matrix), and most validators have few votes. Modelling it is left for a later arm if this one
  passes.

### The model

One fit per label type. Each label `i` has a latent truth `z_i`, correct or incorrect. Each validator `u` (keyed on
`user_id`, the cross-city key) has a sensitivity `s_u = P(Agree | correct)` and a specificity
`p_u = P(Disagree | incorrect)`. Each has a Beta prior of strength `kappa` pseudo-votes, centred on the type's
population rate (`mu_s`, `mu_p`, re-estimated every iteration), so a validator with few votes sits near the
population and the hierarchy shrinks the sparse ones. Each deployment has a class prior `pi_c = P(correct)`,
shrunk toward the type's rate with 10 pseudo-labels (fixed, not tuned). Votes are conditionally independent given
`z_i`.

Fitting is EM with pseudo-count M-steps (the MAP estimate under those priors), started from the smoothed vote share
`(agree + 0.5) / (votes + 1)`, to a tolerance of 1e-7 on every posterior. A fit whose `mu_s + mu_p <= 1` (the
label-switched solution) is refused. The EM never reads a gold verdict.

- **Arm `votes` (primary).** The model above.
- **Arm `labeler` (secondary, descriptive).** The same, but a label's prior is its labeler's precision `lambda_l`
  (P(correct) of that person's labels of the type), shrunk toward the deployment priors of their labels with 10
  pseudo-labels. The labeler is an annotator who always says "correct", so their precision is exactly a per-labeler
  prior. This arm cannot license a training arm on its own.

### Choosing `kappa`: nested cross-validation

`kappa` is the only quantity chosen on gold. The grid is 0.5, 1, 2, 5, 10, 20, 50, 100 and 1000; at 1000 every
validator is close to the population, which makes the model nearly a weighted vote with one pair of rates per type.
Each grid value is fitted once per arm (the fit uses no gold). The decided gold items are split into 5 outer folds
by panorama, `sha256("joint-vote:<pano_id>")` mod 5, so a panorama's items share a fold. Each fold's items are
scored with the `kappa` that minimises the Brier score against the expert over the primary-population items of the
other four folds (ties go to the larger `kappa`), with all types pooled into one choice. No gold item ever helps
choose the `kappa` it is scored with. Each arm is selected on its own.

### The baseline

`majority2`, recomputed on the **same snapshot's tallies with the expert votes removed**: at least two decisive
human votes, majority wins, ties left out (`data_generation.prior_class`). The model and the baseline therefore see
exactly the same votes. As a check, the scoreboard's own `majority2` (on the 2026-10-27 pool tallies, the ~75%
figure) is reported on the same items, along with how many items' tallies differ between the two snapshots.

### Populations

Gold items are those `adjudicator.py gold` marks decided, masked-only (the round-2 meeting verdicts left out, as in
the scoreboard's headline), with the label present in the snapshot.

- **Primary population:** the AI Disagrees (`ai_vote == "Disagree"`) on which `majority2` gives a verdict.
- **Decisive items:** every gold item, AI Disagree or decoy, on which the training margin rule gives a target (net
  Disagree >= 2 or net Agree >= 3, snapshot tallies, experts removed). On these `majority2` gives the margin
  rule's verdict.

### The primary contrast

On the primary population, arm `votes`: the share of items whose posterior verdict (correct if the out-of-fold
posterior is above 0.5, incorrect below, no verdict at exactly 0.5, which counts as a miss) matches the expert,
minus the share whose `majority2` verdict does. Paired, item by item. Two 95% percentile intervals from 2,000
resamples (seed 0): one resampling labels, one resampling panoramas (clusters). In both, the nested `kappa`
selection is redone inside every resample, so the interval carries the selection's own variability. The exact
McNemar p on the discordant pairs is reported beside them.

### The rule

**Proceed to a training arm only if all three hold for arm `votes`:**

1. the primary contrast is at least **+4.0 points**;
2. **both** 95% intervals (label-level and panorama-cluster) exclude 0;
3. it is **not worse on decisive items**: the paired model-minus-`majority2` agreement on the decisive items has a
   point estimate of at least 0.

Anything short of that is reported as what the intervals rule out at each end, never as "no effect". If arm
`labeler` meets the rule and arm `votes` does not, that is a hypothesis to pre-register for the next gold round,
not a licence.

### Secondary (descriptive)

- **Brier against the expert** on the primary population: the model's out-of-fold posterior against the soft
  target `(agree + 1) / (agree + disagree + 2)` (the `--target soft` training target, same tallies), paired, both
  intervals (selection not redone; these are descriptive). `majority2`'s 0/1 error rate is printed beside them.
- **Per round, per type**, and **type x severity / type x tag** slices on the primary population, with counts on
  every cell and rates at 20 items or more (the standing slice rule).
- **The AI Disagrees `majority2` leaves undecided** (one vote, a tie, no vote): the model's agreement there. These
  are outside the primary population; the scoreboard notes that no gold stratum draws one-vote labels or ties.
- **Coverage from the pool**, as the scoreboard does: over the AI Disagrees of `relabel_pool.csv` in the 28 fitted
  deployments, exact shares that `majority2` decides against the shares the model decides at posterior confidence
  0.8, 0.9 and 0.95, how many `majority2` verdicts the model reverses, and how many ties and single votes it
  decides at 0.8. The `kappa` here is the one the selection picks on all primary items, the value a training arm
  would use; no gold rate is read with it.
- **The fit:** per type, `mu_s`, `mu_p` and the spread of the per-validator rates (validators with at least 20
  votes), with the number of EM iterations.
- **Every `kappa` on the primary population**, labelled as the grid the selection chooses among, not as estimates.

### What the numbers are not

- Rates on a stratified sample of contested labels, as on the scoreboard: the gold strata oversample vote patterns
  and types, so the primary contrast carries the sample's mix. Per-deployment rows are never shown; private
  deployments are pooled with the rest and none is named.
- Silent on labels an expert voted on or placed (the draws left them out) and, as a primary, on the one-vote and
  tied AI Disagrees, which no gold stratum samples.
- Under conditional independence. Validators who see the same label in the same queue, or who learn from each
  other, are not independent; the model then over-counts agreeing votes.
- A test of the target, not of a model trained on it. That is the training arm the rule gates.

## Results

### Counts

<!-- BEGIN GENERATED: counts (analysis/joint_vote_model.py) -->

| Count | Value |
|---|---:|
| Gold items / decided / decided masked-only | 800 / 741 / 731 |
| Decided masked-only items found in the snapshot | 720 |
| of which AI Disagrees | 483 |
| **Primary population** (AI Disagrees that majority2 decides, snapshot tallies, experts removed) | **282** on 281 panoramas |
| primary, round 1 / round 2 | 90 / 192 |
| primary, per type | Crosswalk 63, CurbRamp 63, NoCurbRamp 60, Obstacle 51, SurfaceProblem 45 |
| Decisive items (any decided item the margin rule decides) | 242 |
| Deployments fitted / labels of the five types / with a kept vote | 28 / 1,138,175 / 524,284 |
| Kept votes (human Agree / Disagree, experts removed) | 1,107,723 |
| Left out: AI votes / expert-account votes / human Unsure / duplicate | 924,620 / 66,018 / 37,544 (3.3% of non-expert human votes) / 0 |
| Check: majority2 on the 2026-10-27 pool tallies (the scoreboard's), same items | 218/282 (77.3% [72.1, 81.8]); tallies differ from the snapshot's on 47 |

<!-- END GENERATED: counts -->

### The primary contrast and the rule

<!-- BEGIN GENERATED: primary (analysis/joint_vote_model.py) -->

**Arm `votes`** (primary; kappa chosen per outer fold: fold 0 10, fold 1 10, fold 2 10, fold 3 100, fold 4 20)

| Rows | n | model agrees with expert | majority2 agrees | model only / majority2 only right | model minus majority2 [95%, label-level] | [95%, panorama-cluster] | McNemar p |
|---|---:|---|---|---|---|---|---:|
| round 1 (check) | 90 | 66/90 (73.3% [63.4, 81.4]) | 67/90 (74.4% [64.6, 82.3]) | 1 / 2 | -1.1 pts [-5.6, +2.2] | -1.1 pts [-5.6, +2.2] | 1 |
| round 2 (check) | 192 | 153/192 (79.7% [73.4, 84.8]) | 151/192 (78.6% [72.3, 83.9]) | 4 / 2 | +1.0 pts [-1.6, +3.6] | +1.0 pts [-1.6, +3.6] | 0.688 |
| **pooled, primary population** | 282 | 219/282 (77.7% [72.4, 82.1]) | 218/282 (77.3% [72.1, 81.8]) | 5 / 4 | +0.4 pts [-1.5, +3.5] | +0.4 pts [-1.5, +3.5] | 1 |
| decisive items (the not-worse part of the rule) | 242 | 201/242 (83.1% [77.8, 87.3]) | 201/242 (83.1% [77.8, 87.3]) | 3 / 3 | +0.0 pts [-1.8, +3.0] | +0.0 pts [-1.9, +3.1] | 1 |

Rule (applies): **not met** (gain >= +4 pts: no; label interval excludes 0: no; cluster interval excludes 0: no; not worse on decisive: yes).

**Arm `labeler`** (secondary, descriptive; kappa chosen per outer fold: fold 0 5, fold 1 5, fold 2 5, fold 3 5, fold 4 5)

| Rows | n | model agrees with expert | majority2 agrees | model only / majority2 only right | model minus majority2 [95%, label-level] | [95%, panorama-cluster] | McNemar p |
|---|---:|---|---|---|---|---|---:|
| round 1 (check) | 90 | 68/90 (75.6% [65.8, 83.3]) | 67/90 (74.4% [64.6, 82.3]) | 3 / 2 | +1.1 pts [-3.3, +6.7] | +1.1 pts [-3.3, +6.7] | 1 |
| round 2 (check) | 192 | 156/192 (81.2% [75.1, 86.1]) | 151/192 (78.6% [72.3, 83.9]) | 6 / 1 | +2.6 pts [+0.0, +5.7] | +2.6 pts [+0.0, +5.2] | 0.125 |
| pooled, primary population | 282 | 224/282 (79.4% [74.3, 83.7]) | 218/282 (77.3% [72.1, 81.8]) | 9 / 3 | +2.1 pts [-0.4, +4.5] | +2.1 pts [-0.4, +4.6] | 0.146 |
| decisive items (the not-worse part of the rule) | 242 | 204/242 (84.3% [79.2, 88.3]) | 201/242 (83.1% [77.8, 87.3]) | 5 / 2 | +1.2 pts [-1.2, +3.1] | +1.2 pts [-1.3, +3.1] | 0.453 |

Rule (reported only; this arm cannot license a training arm): **not met** (gain >= +4 pts: no; label interval excludes 0: no; cluster interval excludes 0: no; not worse on decisive: yes).

<!-- END GENERATED: primary -->

### Brier, the undecided items and every `kappa` (descriptive)

<!-- BEGIN GENERATED: secondary (analysis/joint_vote_model.py) -->

| Arm | n | Brier, model | Brier, soft target | model minus soft [95%, label] | [95%, cluster] | majority2 error rate | model on AI Disagrees majority2 leaves undecided |
|---|---:|---:|---:|---|---|---:|---|
| `votes` | 282 | 0.1791 | 0.1711 | +0.0080 [-0.0140, +0.0311] | [-0.0131, +0.0319] | 0.2270 | 107/201 (53.2% [46.3, 60.0]) |
| `labeler` | 282 | 0.1754 | 0.1711 | +0.0043 [-0.0176, +0.0275] | [-0.0176, +0.0278] | 0.2270 | 119/201 (59.2% [52.3, 65.8]) |

Brier against the expert verdict (1 = correct), lower is better; the soft target is `(agree + 1) / (agree + disagree + 2)` on the same snapshot tallies, the `--target soft` training target. majority2's error rate is its Brier score as a 0/1 forecast. The last column is descriptive: those items are outside the primary population.

Every kappa on the primary population (descriptive: this is the grid the nested selection chooses from, read on all its items, so no row here is an estimate of anything):

| kappa | `votes`: agrees with expert | `votes`: Brier | `labeler`: agrees | `labeler`: Brier |
|---:|---|---:|---|---:|
| 0.5 | 220/282 (78.0% [72.8, 82.5]) | 0.1869 | 225/282 (79.8% [74.7, 84.1]) | 0.1783 |
| 1 | 219/282 (77.7% [72.4, 82.1]) | 0.1831 | 224/282 (79.4% [74.3, 83.7]) | 0.1775 |
| 2 | 221/282 (78.4% [73.2, 82.8]) | 0.1776 | 223/282 (79.1% [74.0, 83.4]) | 0.1766 |
| 5 | 222/282 (78.7% [73.6, 83.1]) | 0.1717 | 224/282 (79.4% [74.3, 83.7]) | 0.1754 |
| 10 | 222/282 (78.7% [73.6, 83.1]) | 0.1698 | 223/282 (79.1% [74.0, 83.4]) | 0.1764 |
| 20 | 222/282 (78.7% [73.6, 83.1]) | 0.1713 | 222/282 (78.7% [73.6, 83.1]) | 0.1782 |
| 50 | 221/282 (78.4% [73.2, 82.8]) | 0.1751 | 221/282 (78.4% [73.2, 82.8]) | 0.1816 |
| 100 | 222/282 (78.7% [73.6, 83.1]) | 0.1748 | 221/282 (78.4% [73.2, 82.8]) | 0.1841 |
| 1000 | 216/282 (76.6% [71.3, 81.2]) | 0.2065 | 220/282 (78.0% [72.8, 82.5]) | 0.1963 |

<!-- END GENERATED: secondary -->

### Per type (descriptive)

<!-- BEGIN GENERATED: types (analysis/joint_vote_model.py) -->

| Arm, type | n | model agrees with expert | majority2 agrees | model only / majority2 only right | model minus majority2 [95%, label-level] | [95%, panorama-cluster] | McNemar p |
|---|---:|---|---|---|---|---|---:|
| `votes`, CurbRamp | 63 | 49/63 (77.8% [66.1, 86.3]) | 51/63 (81.0% [69.6, 88.8]) | 1 / 3 | -3.2 pts [-9.5, +3.2] | -3.2 pts [-9.5, +3.2] | 0.625 |
| `votes`, NoCurbRamp | 60 | 47/60 (78.3% [66.4, 86.9]) | 46/60 (76.7% [64.6, 85.6]) | 2 / 1 | +1.7 pts [-3.3, +6.7] | +1.7 pts [-3.3, +6.7] | 1 |
| `votes`, Obstacle | 51 | 39/51 (76.5% [63.2, 86.0]) | 37/51 (72.5% [59.1, 82.9]) | 2 / 0 | +3.9 pts [+0.0, +9.8] | +3.9 pts [+0.0, +9.8] | 0.5 |
| `votes`, SurfaceProblem | 45 | 28/45 (62.2% [47.6, 74.9]) | 28/45 (62.2% [47.6, 74.9]) | 0 / 0 | +0.0 pts [+0.0, +0.0] | +0.0 pts [+0.0, +0.0] | 1 |
| `votes`, Crosswalk | 63 | 56/63 (88.9% [78.8, 94.5]) | 56/63 (88.9% [78.8, 94.5]) | 0 / 0 | +0.0 pts [+0.0, +0.0] | +0.0 pts [+0.0, +0.0] | 1 |
| `labeler`, CurbRamp | 63 | 51/63 (81.0% [69.6, 88.8]) | 51/63 (81.0% [69.6, 88.8]) | 2 / 2 | +0.0 pts [-6.3, +6.3] | +0.0 pts [-6.3, +6.3] | 1 |
| `labeler`, NoCurbRamp | 60 | 47/60 (78.3% [66.4, 86.9]) | 46/60 (76.7% [64.6, 85.6]) | 2 / 1 | +1.7 pts [-3.3, +6.7] | +1.7 pts [-3.3, +6.7] | 1 |
| `labeler`, Obstacle | 51 | 38/51 (74.5% [61.1, 84.5]) | 37/51 (72.5% [59.1, 82.9]) | 1 / 0 | +2.0 pts [+0.0, +5.9] | +2.0 pts [+0.0, +5.9] | 1 |
| `labeler`, SurfaceProblem | 45 | 32/45 (71.1% [56.6, 82.3]) | 28/45 (62.2% [47.6, 74.9]) | 4 / 0 | +8.9 pts [+2.2, +17.8] | +8.9 pts [+2.2, +17.8] | 0.125 |
| `labeler`, Crosswalk | 63 | 56/63 (88.9% [78.8, 94.5]) | 56/63 (88.9% [78.8, 94.5]) | 0 / 0 | +0.0 pts [+0.0, +0.0] | +0.0 pts [+0.0, +0.0] | 1 |

Descriptive. Each type is fitted on its own; a per-type interval is wide at these sizes.

<!-- END GENERATED: types -->

### Slices (descriptive)

<!-- BEGIN GENERATED: slices (analysis/joint_vote_model.py) -->

**Arm `votes`, type x severity** (primary population; rates at >= 20 items; 5 of 20 cells reach it)

| Type | Slice | n | model agrees | majority2 agrees | model minus majority2 |
|---|---|---:|---|---|---|
| CurbRamp | 1 | 43 | 35/43 (81.4% [67.4, 90.3]) | 36/43 (83.7% [70.0, 91.9]) | -2.3 pts |
| NoCurbRamp | 3 | 26 | 17/26 (65.4% [46.2, 80.6]) | 18/26 (69.2% [50.0, 83.5]) | -3.8 pts |
| Obstacle | 1 | 27 | 21/27 (77.8% [59.2, 89.4]) | 20/27 (74.1% [55.3, 86.8]) | +3.7 pts |
| SurfaceProblem | 1 | 25 | 15/25 (60.0% [40.7, 76.6]) | 15/25 (60.0% [40.7, 76.6]) | +0.0 pts |
| Crosswalk | 1 | 33 | 29/33 (87.9% [72.7, 95.2]) | 29/33 (87.9% [72.7, 95.2]) | +0.0 pts |

Below 20 items: 15 cells holding 128 item-slices (counts in `results.json`).

**Arm `votes`, type x tag** (primary population; rates at >= 20 items; 3 of 43 cells reach it)

| Type | Slice | n | model agrees | majority2 agrees | model minus majority2 |
|---|---|---:|---|---|---|
| CurbRamp | (none) | 34 | 27/34 (79.4% [63.2, 89.7]) | 27/34 (79.4% [63.2, 89.7]) | +0.0 pts |
| NoCurbRamp | (none) | 28 | 21/28 (75.0% [56.6, 87.3]) | 20/28 (71.4% [52.9, 84.7]) | +3.6 pts |
| Crosswalk | (none) | 35 | 32/35 (91.4% [77.6, 97.0]) | 32/35 (91.4% [77.6, 97.0]) | +0.0 pts |

Below 20 items: 40 cells holding 229 item-slices (counts in `results.json`).

<!-- END GENERATED: slices -->

### Coverage on the AI-Disagree pool

<!-- BEGIN GENERATED: coverage (analysis/joint_vote_model.py) -->

**Arm `votes`** (kappa 10, the value the selection picks on all primary items; 87,669 of the pool's 88,418 AI Disagrees in the fitted deployments are in the snapshot)

| Type | AI Disagrees | majority2 decides (incorrect) | model >= 0.8 (incorrect) | model >= 0.9 | model >= 0.95 | model reverses majority2 | ties / model >= 0.8 on them | one vote / model >= 0.8 |
|---|---:|---|---|---|---|---:|---|---|
| all | 87,669 | 28.7% (18.1%) | 73.0% (24.0%) | 55.1% | 44.2% | 2,196 | 5,435 / 2,930 | 25,190 / 16,271 |
| CurbRamp | 31,129 | 21.2% (10.5%) | 77.5% (16.0%) | 50.2% | 35.8% | 754 | 1,677 / 961 | 10,949 / 7,069 |
| NoCurbRamp | 18,410 | 44.7% (32.8%) | 85.1% (38.7%) | 71.5% | 60.6% | 511 | 1,276 / 742 | 4,079 / 3,101 |
| Obstacle | 13,353 | 22.1% (14.6%) | 55.2% (23.7%) | 49.2% | 42.9% | 278 | 691 / 374 | 3,346 / 1,828 |
| SurfaceProblem | 14,568 | 36.1% (24.1%) | 60.7% (29.4%) | 39.8% | 30.9% | 603 | 1,463 / 637 | 4,190 / 2,300 |
| Crosswalk | 10,209 | 20.5% (10.8%) | 78.9% (14.7%) | 70.3% | 60.9% | 50 | 328 / 216 | 2,626 / 1,973 |

**Arm `labeler`** (kappa 5, the value the selection picks on all primary items; 87,669 of the pool's 88,418 AI Disagrees in the fitted deployments are in the snapshot)

| Type | AI Disagrees | majority2 decides (incorrect) | model >= 0.8 (incorrect) | model >= 0.9 | model >= 0.95 | model reverses majority2 | ties / model >= 0.8 on them | one vote / model >= 0.8 |
|---|---:|---|---|---|---|---:|---|---|
| all | 87,669 | 28.7% (18.1%) | 74.8% (32.7%) | 59.6% | 48.4% | 2,545 | 5,435 / 3,647 | 25,190 / 18,916 |
| CurbRamp | 31,129 | 21.2% (10.5%) | 74.8% (26.3%) | 58.2% | 44.4% | 826 | 1,677 / 1,199 | 10,949 / 8,363 |
| NoCurbRamp | 18,410 | 44.7% (32.8%) | 85.7% (42.5%) | 70.2% | 62.3% | 519 | 1,276 / 809 | 4,079 / 3,265 |
| Obstacle | 13,353 | 22.1% (14.6%) | 59.4% (29.0%) | 44.4% | 35.0% | 332 | 691 / 443 | 3,346 / 2,091 |
| SurfaceProblem | 14,568 | 36.1% (24.1%) | 71.4% (46.6%) | 53.9% | 44.0% | 806 | 1,463 / 941 | 4,190 / 3,030 |
| Crosswalk | 10,209 | 20.5% (10.8%) | 79.7% (19.3%) | 72.9% | 59.2% | 62 | 328 / 255 | 2,626 / 2,167 |

Shares of the type's AI Disagrees; model >= t means max(q, 1 - q) >= t. Exact counts over the pool, no interval, no gold verdict.

<!-- END GENERATED: coverage -->

### The fitted rates

<!-- BEGIN GENERATED: fit (analysis/joint_vote_model.py) -->

| Arm | Type | kappa | validators (>= 20 votes) | mu_s | mu_p | s p10 / p50 / p90 | p p10 / p50 / p90 | EM iterations |
|---|---|---:|---|---:|---:|---|---|---:|
| `votes` | CurbRamp | 10 | 6,352 (2,154) | 0.944 | 0.544 | 0.81 / 0.96 / 0.99 | 0.44 / 0.54 / 0.64 | 358 |
| `votes` | NoCurbRamp | 10 | 6,013 (1,716) | 0.868 | 0.686 | 0.69 / 0.89 / 0.96 | 0.49 / 0.69 / 0.84 | 251 |
| `votes` | Obstacle | 10 | 4,755 (1,079) | 0.864 | 0.535 | 0.65 / 0.89 / 0.96 | 0.30 / 0.54 / 0.77 | 743 |
| `votes` | SurfaceProblem | 10 | 5,146 (1,441) | 0.852 | 0.584 | 0.64 / 0.87 / 0.96 | 0.44 / 0.58 / 0.74 | 528 |
| `votes` | Crosswalk | 10 | 2,183 (719) | 0.976 | 0.756 | 0.90 / 0.99 / 1.00 | 0.66 / 0.76 / 0.86 | 154 |
| `labeler` | CurbRamp | 5 | 6,352 (2,154) | 0.952 | 0.566 | 0.81 / 0.98 / 0.99 | 0.38 / 0.57 / 0.75 | 612 |
| `labeler` | NoCurbRamp | 5 | 6,013 (1,716) | 0.875 | 0.687 | 0.65 / 0.90 / 0.98 | 0.42 / 0.69 / 0.88 | 301 |
| `labeler` | Obstacle | 5 | 4,755 (1,079) | 0.860 | 0.553 | 0.60 / 0.89 / 0.97 | 0.25 / 0.57 / 0.84 | 1000 (not converged) |
| `labeler` | SurfaceProblem | 5 | 5,146 (1,441) | 0.869 | 0.556 | 0.63 / 0.90 / 0.98 | 0.31 / 0.56 / 0.79 | 1000 (not converged) |
| `labeler` | Crosswalk | 5 | 2,183 (719) | 0.978 | 0.747 | 0.91 / 1.00 / 1.00 | 0.57 / 0.77 / 0.90 | 198 |

mu_s = population P(Agree | correct), mu_p = population P(Disagree | incorrect); the quantiles are over validators with at least 20 votes of the type.

<!-- END GENERATED: fit -->

## Reading

Written after the run (2026-11-26, code at the pre-registration commit `9e932dc`). Nothing in the pre-registration
was changed. Two additions came after the numbers: the convergence check below, and a rendering fix that prints the
rule's parts in their pre-registered order.

**The rule is not met.** On the 282 AI Disagrees `majority2` decides, the model's verdict matches the expert on 219
and `majority2`'s on 218. The difference is +0.4 pts, [-1.5, +3.5] resampling labels and the same resampling
panoramas (the 282 items sit on 281 panoramas). McNemar p is 1 on 5 against 4 discordant items. Both upper ends are
below the +4 points the rule asked for, so at the 95% level the data rule out the required gain rather than merely
failing to show it. The rule's last part holds: on the 242 decisive items the two are tied, 201 each (3 against 3
discordant). Round 1 (-1.1 pts) and round 2 (+1.0 pts) agree within their intervals.

**Why nothing moves: the model almost never overrules the majority on these items.** It differs from `majority2`
on 9 of 282 primary items (3.2%). Over the pool it reverses 2,196 of the 25,134 `majority2` verdicts on AI
Disagrees (8.7%). So the gold strata, drawn on vote margins, sample the labels where reversals are rare. The fitted
rates are asymmetric in the direction the model was built for. In every type the population specificity (0.54 to
0.76) is well below the sensitivity (0.85 to 0.98), and validators differ (CurbRamp specificity 0.44 to 0.64,
p10 to p90). But on a label with two or more decisive votes that spread rarely outweighs a majority. Under the model
an average validator misses about half the wrong CurbRamp and Obstacle labels; that holds only under the model's
conditional-independence assumption, and no gold number checks it.

**Brier is no better than the soft target.** The model scores 0.179 against 0.171 for
`(agree + 1) / (agree + disagree + 2)`, +0.008 [-0.014, +0.031]. The nested selection picked `kappa` 10 to 100
(mostly 10). At `kappa` 1000, where every validator is nearly the population, Brier is worst (0.207). So
per-validator rates do help the posterior over a one-pair-per-type model; they just do not help it beat the soft
target or the majority.

**The coverage gain is not validated.** At posterior confidence 0.8 or more, the model gives a verdict to 73.0% of
the pool's AI Disagrees, against 28.7% for `majority2`. Most of the gain is single-vote labels (16,271 of 25,190
reach 0.8) and ties (2,930 of 5,435). The gold set cannot score those. Of the 201 gold AI Disagrees `majority2`
leaves undecided, 188 have no non-expert human vote (there the model's verdict is its deployment prior: it says
correct on 163, the expert on 80), 12 have one vote and 1 is a tie. The model matches the expert on 107 of the 201
(53.2%). A training arm that used the model's confident verdicts on one-vote labels would rest on no gold evidence;
the stratum of [#106](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/106) is what could provide
it.

**Arm `labeler` (secondary, descriptive).** It reaches 224 of 282, +2.1 pts [-0.4, +4.5] (label-level; [-0.4, +4.6]
by panorama), 9 against 3 discordant, and +1.2 pts on decisive items. It fails the rule's first two parts too. Its
upper end does not exclude +4, and most of its gain is SurfaceProblem (+8.9 pts, 4 against 0 discordant, 45 items).
Pre-registration said this can only be a hypothesis for the next gold round, and that is how it stands. A labeler's
precision used as a prior is what helps there, not the per-validator rates. It also makes the most confident `incorrect` calls
(32.7% of the pool's AI Disagrees at 0.8, against 24.0% for arm `votes` and 18.1% `incorrect` targets from
`majority2`). A next-round test of it is therefore worth pre-registering with a stratum that oversamples labels
where it reverses the majority.

**Convergence.** Several fits stopped at the 1,000-iteration cap (`fit` table; in `results.json`, 8 of 45
(type, `kappa`) cells in arm `votes` and 11 in arm `labeler`, among them Obstacle at `kappa` 100, which fold 3 used).
A check after the run continued every capped fit to the 1e-7 tolerance (1,014 to 3,279 iterations). No verdict on
any primary gold item changed. The largest posterior change on a primary item was 0.011, at a `kappa` never
selected; elsewhere it was at most 2e-4. The coverage fits (`votes` at 10, `labeler` at 5) flip no pool label. Two
unselected fits do flip pool labels (192 at `votes` 0.5 Obstacle, 2,996 at `labeler` 0.5 SurfaceProblem); they are
in no published table. The check is a scratch script, not part of the module, so a rerun should raise
`max_iter` rather than rely on it.

**Leakage and inputs, as checked.** 66,018 votes by the two expert accounts were left out before fitting, and no
vote by them reached the fit (asserted). 11 decided gold items are not in the 2026-10-18 snapshot, most likely
labels placed after it. On the 282 primary items, the scoreboard's `majority2` on the 2026-10-27 pool tallies also
reaches 218 (77.3%), although 47 items' tallies differ between the two pulls, so the baseline here is the
scoreboard's.

### What this means

- **[#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12).** No training arm. Asymmetric
  per-validator rates, like #12's scalar weights, do not move the contested targets on this gold set, and the
  interval rules out a gain of +4 points over `majority2`. Neither weighting scheme should become a training target
  on target-accuracy grounds.
- **[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74).** The joint vote model joins the
  stricter rules and confident learning: none beats the crowd majority on the gold set. The VLM adjudicator
  ([#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77)) is the remaining candidate.
- **Next gold round.** If one is drawn, two questions are worth pre-registering: the `labeler` arm on SurfaceProblem,
  and the model's verdicts on one-vote labels and ties
  ([#106](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/106)), with items sampled where the model
  reverses or decides what `majority2` cannot.

## Provenance

<!-- BEGIN GENERATED: provenance (analysis/joint_vote_model.py) -->

| Item | Pin |
|---|---|
| gold.csv | `2b472651cb01011a8640b86eb4103e1794ab482186ba3a3471e7d019273edcae` (matches the pin in `docs/adjudicator-2026-11.md`: yes) |
| rawLabels snapshot (round 3's, sha256 of the sorted per-file list) | `ed6b78ff718e35e7d390db5ece95480132554e0922a966fa8f64dafea744622d` (28 files, every one written before 2026-10-27) |
| relabel_pool.csv | `dd5a6d363ce7f6c05ac6faf250f2526090bd2199541e6164b76029cd35483d09` |
| scoreboard verdicts.csv (private) | `e17a408a5db6c49dbe7e7ee2deef83c230a84068f383a32be033f5af6382e4f1` |
| gold_posteriors.csv (private) | `6392960390224c28a66aa004fd0cf4f584cdf3155eb3843a8bd2cd95e100e0d5` |
| expert accounts left out | 2 (from `EXPERTS`; ids not recorded) |
| code | commit `9e932dc92518afa286606b47797edad37ca67b01`; Python 3.11.15, pandas 2.3.3, numpy 2.4.6 |
| intervals | 2000 resamples (labels, and panorama clusters), seed 0 |
| public-deployment list | live `/v3/api/cities`, read 2026-11-27T03:50:35+00:00 (47 public) |

<!-- END GENERATED: provenance -->

The per-label posteriors of the gold items (`gold_posteriors.csv`), `run.json` and a copy of `results.json` are in
`~/Data/sidewalk-validator-ai-joint-vote-2026-11/`, outside the repository. No user id, rater id or label id is in
the repository.

## Reproducing

```bash
# EXPERTS names both expert accounts, in the environment only (never in the repository)
EXPERTS="<name>=<user_id>,<name>=<user_id>" python analysis/joint_vote_model.py score \
    --rawlabels ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels \
    --gold ~/Data/sidewalk-validator-ai-adjudicator-2026-11/gold.csv \
    --pool ~/Data/sidewalk-validator-ai-relabel-2026-10/relabel_pool.csv \
    --scoreboard-verdicts ~/Data/sidewalk-validator-ai-target-scoreboard-2026-11/verdicts.csv
python analysis/joint_vote_model.py report            # --check verifies without writing
```

CPU only, about ten minutes on a laptop (18 fits of five types over about 1.1M votes); one HTTPS read of
`/v3/api/cities` for the privacy check.
