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

_Not yet run._

<!-- END GENERATED: counts -->

### The primary contrast and the rule

<!-- BEGIN GENERATED: primary (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: primary -->

### Brier, the undecided items and every `kappa` (descriptive)

<!-- BEGIN GENERATED: secondary (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: secondary -->

### Per type (descriptive)

<!-- BEGIN GENERATED: types (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: types -->

### Slices (descriptive)

<!-- BEGIN GENERATED: slices (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: slices -->

### Coverage on the AI-Disagree pool

<!-- BEGIN GENERATED: coverage (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: coverage -->

### The fitted rates

<!-- BEGIN GENERATED: fit (analysis/joint_vote_model.py) -->

_Not yet run._

<!-- END GENERATED: fit -->

## Provenance

<!-- BEGIN GENERATED: provenance (analysis/joint_vote_model.py) -->

_Not yet run._

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
