# Wrong crowd targets: confident learning and stricter Disagree rules (October 2026)

Experiment for [#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76) and
[#75](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75), two arms of the umbrella
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74): can the training targets that the
crowd gets wrong be found without rating labor, and does removing or correcting them give a validator whose
Disagree verdicts hold up? Everything here ran on the round-3 crops and splits
([`retrain-2026-10.md`](retrain-2026-10.md)); only the **train** rows change between variants, so every number
below is on the same val / test / test_heldout rows as the round-3 baseline.

**Status: done.** The chain ran on klone on 2026-10-28 (55 jobs plus the prior job, none failed). The result
tables come from the checked-in [`confident-learning-2026-10/variants.json`](confident-learning-2026-10/variants.json),
the rule and prior tables from the chain's `rule.json` files and [`priors.json`](confident-learning-2026-10/priors.json).
One training seed per variant, as in round 3, then two more seeds of the baseline and of `cl-drop` for every
type on 2026-10-28/29 (22 jobs, none failed; [`seeds.md`](confident-learning-2026-10/seeds.md)). Short version:
dropping the rows the fold models confidently disagree with (`cl-drop`, 4–18% of the train rows) costs about half
a hundredth on the test split and raises the AUC on the three unseen deployments for four of the five types; across
three seeds the direction holds in 11 of 12 same-seed pairs, but the size is about half of what the first seed
showed (mean +0.009 to +0.020) and of the same order as the seed noise of the held-out AUC itself. The stricter
vote rules, run without rebalancing the classes, cost AUC in distribution for every type and out of distribution
clearly for SurfaceProblem and for Obstacle under `symmetric`.

## Why

The production validators cannot vote Disagree: every `incorrect` map is held back because no confidence level
reaches 0.92 precision ([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)). The masked
expert re-label ([#66](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66)) says why calibration
alone cannot fix it: an expert agrees with a gated AI Disagree about 58% of the time, the AI's confidence does
not separate its right Disagrees from its wrong ones, and the crowd majority itself matches the expert only
74.5% of the time on those contested labels. The training target `incorrect` is the crowd margin rule (net
Disagree > 1), so the model learns the crowd's Disagree habits, and more crowd data of the same kind reinforces
them.

Two zero-labor ways to clean the targets are tested here:

- **Stricter crowd rules (#75).** `symmetric` requires net Disagree > 2 for `incorrect` (as `correct` already
  requires net Agree > 2); `unanimous3` requires at least three votes with none the other way. Rows that fail
  the rule are left out of training. The crowd's Disagree confirmation rises from 56% at one human vote to 71%
  at five ([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)), so the
  contested two-vote Disagrees are the targets most likely to be wrong.
- **Confident learning (#76).** Train the round-3 recipe five ways on the train rows (panorama-grouped folds),
  score every train row with the fold model that did not see it, and apply the confident joint of Northcutt,
  Jiang and Chuang (JAIR 2021, [doi:10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125)): a row counts
  as "likely class j" when its out-of-fold p(j) is at least the mean p(j) of the rows whose target is j; the
  off-diagonal cells of that 2 × 2 count estimate how many targets are wrong per (given, likely) pair, and the
  rows flagged are, per cell, the ones with the highest p(likely). Two treatments: drop the flagged rows
  (`cl-drop`) or flip their target (`cl-flip`).

## Setup

| Setting | Value |
|---|---|
| Crops, splits, held-out deployments | round 3 (`retrain-2026-10/<Type>/split.csv`, seed 42, tucson-az and two private deployments held out) |
| Training recipe | round 3: DINOv2-L, 25 epochs, best checkpoint by val macro-F1, seed 42, hard targets |
| Folds | 5, grouped by panorama with the split's 3 m merge, class-ratio stratified (`confident_learning.py folds`); the same object seen from panoramas more than 3 m apart can sit in two folds, which pulls out-of-fold scores toward the given target, so the flags err on the low side |
| Flag rule | off-diagonal of the confident joint, uncalibrated, rows ranked by p(likely) as in cleanlab's `prune_by_noise_rate` (`confident_learning.py flag`); cleanlab's row calibration of the joint would flag 1–9% more rows |
| Variants per type | `symmetric`, `unanimous3`, `cl-drop`, `cl-flip`; baseline = round 3 |
| Evaluation | `evaluate.py --per-city` on the round-3 split with each variant's best checkpoint (`--tag <variant>`): test and held-out AUC, the hard slice, recall at precision 0.92 per verdict; then `confident_learning.py report`: within-city AUC as defined in the round-3 report (per-deployment AUCs weighted by n_incorrect × n_correct, deployments with ≥ 10 labels), AUC per held-out deployment, and every variant against the baseline on the same rows with a paired panorama bootstrap (1,000 resamples, `analysis/slices.py`), overall and by severity and tag |
| Compute | klone `ckpt-all`, one GPU per run (A40 or better); 25 fold runs + 20 variant runs + 22 seed replicates; fold and replicate checkpoints deleted once their scores are written |

Reproduce from a checkout of this branch on klone (`DATA` is the round-3 output directory, `RUNS` a scratch
directory for checkpoints):

```
DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/cl-2026-10 \
    analysis/cl_submit.sh CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem
# seed replicates of a variant and of the baseline (the seed noise), named by TAG, checkpoints deleted after evaluation:
for LT in CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem; do for S in 43 44; do for V in baseline cl-drop; do
    LT=$LT VARIANT=$V SEED=$S TAG=$V-s$S KEEP_CKPT=0 DATA=$DATA RUNS=$RUNS \
        sbatch --job-name=cl-$LT-$V-s$S analysis/confident_learning.slurm; done; done; done
for S in 43 44; do LT=SurfaceProblem VARIANT=cl-flip SEED=$S TAG=cl-flip-s$S KEEP_CKPT=0 DATA=$DATA RUNS=$RUNS \
    sbatch --job-name=cl-SurfaceProblem-cl-flip-s$S analysis/confident_learning.slurm; done
# when every job is done (add --gold <private expert file> for the agreement counts; the file never enters the repo):
python analysis/confident_learning.py report --data-dir $DATA --out docs/confident-learning-2026-10 \
    --public-cities "$(curl -sL https://sidewalk-sea.cs.washington.edu/v3/api/cities | python -c 'import json,sys; print(",".join(c["city_id"] for c in json.load(sys.stdin)["cities"] if c.get("visibility") == "public"))')"
```

The report's outputs are checked in under [`confident-learning-2026-10/`](confident-learning-2026-10/):
`variants.md` and `variants.json` (the result tables, plus the flag accounting per deployment with the private
deployments pooled), `slices.md` (severity × tag, every variant against the baseline), `seeds.md` (the seed
replicates, each against the baseline of the same seed) and `priors.json` (the prior counts under each rule,
deployment names replaced by a count and a hash). The chain's own
outputs (the fold assignments, out-of-fold scores, flagged rows, variant split files, checkpoints' trainer
states, predictions) stay on klone under `$DATA/<Type>/cl/`; the flag summary records their sha256s.

## Train-row accounting

**The stricter rules** (from each type's `cl/<rule>/rule.json`, 2026-10-28). The round-3 train rows per type and
what each rule keeps; `symmetric` touches only `incorrect` rows (it tightens the Disagree side to match the
Agree side), `unanimous3` also drops `correct` rows that carry any Disagree vote. Expert-sourced targets
are never dropped. **The rows are dropped and nothing is rebalanced**: round 3's train split is about 55:45
correct:incorrect, and under the rules it becomes as skewed as 79:21 (SurfaceProblem `symmetric`) or 81:19
(NoCurbRamp and SurfaceProblem `unanimous3`), whereas a retrain under the rule would downsample `correct` back to 55:45
(`data_generation.py`). So the rule variants measure the rule *and* the class mix, and read as "fewer, cleaner
Disagree examples, more imbalance", not as the rule alone.

| Type | train rows (correct / incorrect) | symmetric keeps | dropped | unanimous3 keeps | dropped |
|---|---:|---:|---:|---:|---:|
| CurbRamp | 3,795 / 3,130 | 3,795 / 1,430 | 1,700 (54% of incorrect) | 3,534 / 1,213 | 2,178 |
| Crosswalk | 1,156 / 933 | 1,156 / 609 | 324 (35%) | 1,106 / 373 | 610 |
| NoCurbRamp | 6,738 / 5,461 | 6,738 / 3,498 | 1,963 (36%) | 5,553 / 1,338 | 5,308 |
| Obstacle | 6,097 / 4,457 | 6,097 / 1,810 | 2,647 (59%) | 4,862 / 1,417 | 4,275 |
| SurfaceProblem | 5,839 / 4,661 | 5,839 / 1,580 | 3,081 (66%) | 5,140 / 1,171 | 4,189 |

So between a third and two thirds of the `incorrect` targets rest on a net Disagree of exactly two, and under
`unanimous3` the `incorrect` class shrinks to a quarter to two fifths of its size. Whatever these variants do
to the held-out numbers, they do it with far fewer Disagree examples and a different class mix, which is part
of what is being measured.

**The confident-joint flags** (from each type's `cl/flag_summary.json`; the per-row `noisy_targets.csv` stays on
klone). A row is flagged when its out-of-fold score for the other class clears that class's threshold (the mean
out-of-fold p(j) over the rows whose target is j), so a flag is the five fold models disagreeing with the target,
not a verdict on it. The table counts the flags by target class, by the vote margin behind the target, for the
expert-sourced targets, for the hard slice, and how many of them the `symmetric` rule drops anyway (it drops
exactly the net-Disagree-2 `incorrect` rows, so the last column is the net-Disagree-2 flags).

| Type | train rows | flagged | `correct` flagged | `incorrect` flagged | of net Disagree 2 | of net Disagree ≥ 3 | of expert-sourced | of the hard slice | also dropped by `symmetric` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | 6,925 | 545 (7.9%) | 239 (6.3%) | 306 (9.8%) | 212 (12.5%) | 54 (6.7%) | 119 (6.0%) | 46 of 85 | 212 (39% of flags) |
| Crosswalk | 2,089 | 81 (3.9%) | 26 (2.3%) | 55 (5.9%) | 40 (12.3%) | 14 (2.6%) | 7 (4.0%) | 10 of 23 | 40 (49%) |
| NoCurbRamp | 12,199 | 1,290 (10.6%) | 683 (10.1%) | 607 (11.1%) | 355 (18.1%) | 205 (6.4%) | 197 (8.3%) | 141 of 347 | 355 (28%) |
| Obstacle | 10,554 | 1,946 (18.4%) | 679 (11.1%) | 1,267 (28.4%) | 846 (32.0%) | 306 (24.5%) | 245 (15.4%) | 20 of 37 | 846 (43%) |
| SurfaceProblem | 10,500 | 1,502 (14.3%) | 731 (12.5%) | 771 (16.5%) | 567 (18.4%) | 104 (9.2%) | 361 (13.3%) | 60 of 128 | 567 (38%) |

Three things stand out. The flag rate orders the types as the round-3 test AUC does (Crosswalk 4%, CurbRamp
8%, NoCurbRamp 11%, SurfaceProblem 14%, Obstacle 18%): the models disagree with their targets most where they
are weakest, and the method cannot tell a wrong target from a hard crop (41–54% of the hard slice, the labels the
production AI got wrong against a unanimous crowd or an expert, is flagged). The two-vote Disagrees are flagged
about twice to nearly five times as often as the Disagrees with three or more net votes (CurbRamp 12.5% against
6.7%, Crosswalk 12.3 against 2.6, NoCurbRamp 18.1 against 6.4, SurfaceProblem 18.4 against 9.2; Obstacle 32.0
against 24.5), which is the premise of #75 seen from the model's side; but the method flags only 12–32% of those
two-vote rows and, unlike any vote rule, also flags 2–13% of the `correct` targets. And the expert-sourced
targets are flagged only a little less often than the crowd's (6.0% against 8.6% for CurbRamp, 8.3 against 11.1
NoCurbRamp, 15.4 against 19.0 Obstacle, 13.3 against 14.7 SurfaceProblem; Crosswalk 4.0 against 3.9), with the
expert-sourced `incorrect` rows of NoCurbRamp, Obstacle and SurfaceProblem at 16.6%, 20.6% and 22.3% (CurbRamp
6.5%, Crosswalk 1.4%): the fold models disagree with the expert votes at nearly the rate they disagree with the
crowd's, so the flags are not a clean measure of crowd error.

## Results

AUC is of the `incorrect` score. Δ is the variant minus the baseline on the same rows, with a 95% interval from
1,000 resamples of whole panoramas scored by both models (the paired bootstrap of `analysis/slices.py`); it
captures the sampling noise of the evaluation rows, not the training seed (the seed replicates further down do that). "Within-city"
is the round-3 definition (per-deployment AUCs weighted by n_incorrect × n_correct over deployments with at
least 10 labels; the baseline rows reproduce the round-3 report). The last column is the raw-score recall of
`incorrect` at precision 0.92 on the held-out rows, the production gate's level but not its calibrated map, so
read it as direction only; it is a step function of the threshold, which is why it can sit at 0.000 for a model
whose precision peaks just under 0.92. The full tables, with the test-split recall at 0.92 for both verdicts and
the hard slice, are in [`variants.md`](confident-learning-2026-10/variants.md).

**CurbRamp**

| variant | train rows | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | Disagree R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6,925 | 0.968 |  | 0.973 (19) | 0.907 |  | 0.838 (3) | 0.000 |
| symmetric | 5,225 (−1,700) | 0.959 | -0.009 [-0.015, -0.002] | 0.964 (19) | 0.889 | -0.018 [-0.038, +0.000] | 0.814 (3) | 0.522 |
| unanimous3 | 4,747 (−2,178) | 0.962 | -0.005 [-0.012, +0.001] | 0.967 (19) | 0.900 | -0.007 [-0.027, +0.011] | 0.832 (3) | 0.610 |
| cl-drop | 6,380 (−545) | 0.962 | -0.006 [-0.012, -0.000] | 0.962 (19) | 0.920 | +0.012 [-0.002, +0.028] | 0.876 (3) | 0.671 |
| cl-flip | 6,925 (545 flipped) | 0.958 | -0.009 [-0.015, -0.003] | 0.962 (19) | 0.916 | +0.009 [-0.005, +0.026] | 0.874 (3) | 0.618 |

**Crosswalk**

| variant | train rows | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | Disagree R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2,089 | 0.980 |  | 0.983 (8) | 0.941 |  | 0.946 (3) | 0.890 |
| symmetric | 1,765 (−324) | 0.956 | -0.024 [-0.048, -0.006] | 0.973 (8) | 0.964 | +0.023 [+0.001, +0.050] | 0.959 (3) | 0.945 |
| unanimous3 | 1,479 (−610) | 0.963 | -0.017 [-0.046, +0.006] | 0.963 (8) | 0.948 | +0.007 [-0.022, +0.037] | 0.950 (3) | 0.877 |
| cl-drop | 2,008 (−81) | 0.972 | -0.008 [-0.027, +0.005] | 0.970 (8) | 0.973 | +0.032 [+0.004, +0.067] | 0.980 (3) | 0.945 |
| cl-flip | 2,089 (81 flipped) | 0.977 | -0.003 [-0.009, +0.002] | 0.976 (8) | 0.972 | +0.031 [+0.003, +0.069] | 0.974 (3) | 0.904 |

**NoCurbRamp**

| variant | train rows | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | Disagree R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 12,199 | 0.934 |  | 0.903 (18) | 0.881 |  | 0.861 (3) | 0.562 |
| symmetric | 10,236 (−1,963) | 0.926 | -0.008 [-0.016, -0.000] | 0.866 (18) | 0.887 | +0.005 [-0.016, +0.028] | 0.871 (3) | 0.578 |
| unanimous3 | 6,891 (−5,308) | 0.925 | -0.009 [-0.016, -0.002] | 0.878 (18) | 0.884 | +0.003 [-0.021, +0.027] | 0.880 (3) | 0.359 |
| cl-drop | 10,909 (−1,290) | 0.928 | -0.005 [-0.012, +0.002] | 0.908 (18) | 0.923 | +0.041 [+0.016, +0.063] | 0.904 (3) | 0.792 |
| cl-flip | 12,199 (1,290 flipped) | 0.930 | -0.003 [-0.011, +0.004] | 0.909 (18) | 0.908 | +0.027 [+0.007, +0.049] | 0.896 (3) | 0.487 |

**Obstacle**

| variant | train rows | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | Disagree R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10,554 | 0.883 |  | 0.870 (18) | 0.868 |  | 0.788 (3) | 0.529 |
| symmetric | 7,907 (−2,647) | 0.869 | -0.014 [-0.024, -0.004] | 0.848 (18) | 0.844 | -0.024 [-0.039, -0.009] | 0.761 (3) | 0.434 |
| unanimous3 | 6,279 (−4,275) | 0.856 | -0.027 [-0.039, -0.016] | 0.827 (18) | 0.860 | -0.008 [-0.023, +0.008] | 0.805 (3) | 0.442 |
| cl-drop | 8,608 (−1,946) | 0.891 | +0.009 [-0.001, +0.019] | 0.871 (18) | 0.890 | +0.022 [+0.009, +0.036] | 0.816 (3) | 0.704 |
| cl-flip | 10,554 (1,946 flipped) | 0.878 | -0.005 [-0.016, +0.006] | 0.855 (18) | 0.877 | +0.009 [-0.006, +0.024] | 0.806 (3) | 0.682 |

**SurfaceProblem**

| variant | train rows | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | Disagree R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10,500 | 0.896 |  | 0.858 (21) | 0.710 |  | 0.682 (3) | 0.000 |
| symmetric | 7,419 (−3,081) | 0.863 | -0.032 [-0.045, -0.020] | 0.812 (21) | 0.624 | -0.087 [-0.115, -0.059] | 0.628 (3) | 0.000 |
| unanimous3 | 6,311 (−4,189) | 0.866 | -0.030 [-0.043, -0.017] | 0.823 (21) | 0.655 | -0.055 [-0.085, -0.027] | 0.621 (3) | 0.015 |
| cl-drop | 8,998 (−1,502) | 0.901 | +0.005 [-0.003, +0.015] | 0.866 (21) | 0.686 | -0.024 [-0.047, +0.002] | 0.658 (3) | 0.000 |
| cl-flip | 10,500 (1,502 flipped) | 0.897 | +0.001 [-0.008, +0.011] | 0.862 (21) | 0.726 | +0.016 [-0.009, +0.040] | 0.706 (3) | 0.000 |

**Held-out deployments** (AUC per deployment; blank under 10 labels or with one class)

| type | deployment | n (incorrect) | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---|---:|---:|---:|---:|---:|---:|
| CurbRamp | private city A (outside the US) | 284 (168) | 0.821 | 0.792 | 0.813 | 0.871 | 0.877 |
| CurbRamp | private city B (US) | 202 (30) | 0.881 | 0.876 | 0.882 | 0.878 | 0.846 |
| CurbRamp | tucson-az | 60 (30) | 0.958 | 0.943 | 0.968 | 0.960 | 0.954 |
| Crosswalk | private city A (outside the US) | 45 (16) | 0.888 | 0.871 | 0.851 | 0.957 | 0.940 |
| Crosswalk | private city B (US) | 77 (47) | 0.972 | 0.990 | 0.984 | 0.991 | 0.984 |
| Crosswalk | tucson-az | 16 (10) | 0.767 | 0.917 | 0.917 | 0.908 | 1.000 |
| NoCurbRamp | private city A (outside the US) | 125 (121) | 0.800 | 0.769 | 0.800 | 0.866 | 0.741 |
| NoCurbRamp | private city B (US) | 572 (234) | 0.860 | 0.872 | 0.881 | 0.905 | 0.898 |
| NoCurbRamp | tucson-az | 46 (29) | 0.955 | 0.826 | 0.795 | 0.851 | 0.850 |
| Obstacle | private city A (outside the US) | 993 (910) | 0.666 | 0.646 | 0.723 | 0.699 | 0.707 |
| Obstacle | private city B (US) | 628 (196) | 0.896 | 0.865 | 0.882 | 0.921 | 0.898 |
| Obstacle | tucson-az | 191 (119) | 0.792 | 0.746 | 0.769 | 0.809 | 0.782 |
| SurfaceProblem | private city A (outside the US) | 356 (214) | 0.612 | 0.596 | 0.538 | 0.600 | 0.653 |
| SurfaceProblem | private city B (US) | 262 (77) | 0.798 | 0.735 | 0.794 | 0.790 | 0.831 |
| SurfaceProblem | tucson-az | 229 (185) | 0.743 | 0.559 | 0.630 | 0.644 | 0.683 |

**What the tables say.**

- **The stricter rules cost.** `symmetric` and `unanimous3` lower the test AUC for every type (−0.005 to
  −0.032; the interval excludes zero for eight of the ten cells) and the within-city test AUC with it, so this is
  not the deployment mix shifting. On the unseen deployments the picture is mixed: a clear loss for SurfaceProblem
  under both rules (`symmetric` −0.087 [−0.115, −0.059], `unanimous3` −0.055 [−0.085, −0.027]) and for Obstacle
  under `symmetric` (−0.024 [−0.039, −0.009]); within noise for CurbRamp (−0.018 [−0.038, +0.000] and −0.007),
  NoCurbRamp (+0.005 and +0.003) and Obstacle `unanimous3` (−0.008); and one gain, Crosswalk under `symmetric`
  (+0.023 [+0.001, +0.050], on 138 held-out rows). Of the 57 test slices with an interval, 16 (`symmetric`) and
  14 (`unanimous3`) fall below it and one rises (under `symmetric`); of the 42 held-out slices, 10 and 8 fall and
  one rises (under `unanimous3`). The rules remove a third to two thirds of the `incorrect` examples and skew the
  class mix (see the accounting above), so what they measure is cleaner targets *at the cost of* fewer examples
  and more imbalance, and as run here the cost wins. They say nothing about the rule as a definition with more
  data behind it, or with the classes rebalanced.
- **Confident learning helps on the held-out deployments (seed 42; the replicates below temper the size).** `cl-drop` removes 4–18% of the train rows and leaves the test split within
  a hundredth of the baseline (−0.008 to +0.009; the within-city test AUC moves by −0.013 to +0.008), while
  the held-out AUC rises for Crosswalk (+0.032 [+0.004, +0.067]), NoCurbRamp (+0.041 [+0.016, +0.063]) and
  Obstacle (+0.022 [+0.009, +0.036]), rises short of the interval for CurbRamp (+0.012 [−0.002, +0.028]), and
  falls short of it for SurfaceProblem (−0.024 [−0.047, +0.002]). Per deployment it is at or above the baseline
  in 10 of the 15 cells; the exceptions are all three SurfaceProblem deployments, tucson-az NoCurbRamp (46
  labels) and CurbRamp's private city B by 0.003. Eight held-out slices rise past their interval and one falls (SurfaceProblem `uneven/slanted`,
  94 labels); on test, one rises and four fall, none by more than 0.031. The raw-score Disagree recall at 0.92
  precision on the held-out rows goes from 0.00 / 0.89 / 0.56 / 0.53 to 0.67 / 0.95 / 0.79 / 0.70 for
  CurbRamp / Crosswalk / NoCurbRamp / Obstacle, and stays at zero for SurfaceProblem.
- **Flipping is the gentler version of dropping.** `cl-flip` keeps every row and moves the held-out AUC the
  same way for four types, by less (NoCurbRamp +0.027 [+0.007, +0.049], Crosswalk +0.031 [+0.003, +0.069],
  Obstacle +0.009, CurbRamp +0.009), and the other way for SurfaceProblem (+0.016 against `cl-drop`'s −0.024,
  both within noise); it is the only variant with no held-out slice below its interval (six above). Its test cost is
  CurbRamp's −0.009 [−0.015, −0.003] and otherwise noise.
- **How much to believe it.** The test rows come from the training deployments and the same crowds; the
  held-out rows come from three deployments the model never saw, and their targets are crowd-margin labels too.
  A change that leaves the first alone and lifts the second *could* be removing something deployment- or
  crowd-specific that the test split still pays for, which is what wrong targets absorbed as habits would look
  like; it could as well be that a training set without its hardest crops generalizes better, since 41–54% of
  the hard slice is flagged and `cl-drop` drops those rows too (the hard-slice AUCs move both ways on 5–90
  labels, too few to read). The run cannot tell these apart. Three more cautions: the intervals resample
  panoramas, not deployments, and there are only three of those (CurbRamp's gain is private city A alone); the
  seed-42 run turned out to be the most favorable of three seeds for Crosswalk, NoCurbRamp and Obstacle (next
  section); and the held-out split has now been used to pick among four variants for five types, so about one
  of those twenty comparisons is expected to clear its interval by chance.

### Seed replicates

The baseline and `cl-drop` were retrained with seeds 43 and 44 for every type (and `cl-flip` for SurfaceProblem,
the one type where it beat `cl-drop`), everything else fixed. Each replicate is compared with the baseline of the
*same* seed; the full per-seed table with intervals is [`seeds.md`](confident-learning-2026-10/seeds.md). Held-out
AUC as mean ± sample sd over the three seeds; Δ as the mean of the three same-seed differences with their range
and the number of seeds where `cl-drop` was ahead.

| type | baseline held-out AUC | `cl-drop` held-out AUC | Δ held-out, mean [min, max], seeds ahead | Δ test, mean | incorrect R@P0.92 held-out, baseline → `cl-drop` |
|---|---:|---:|---:|---:|---:|
| CurbRamp | 0.908 ± 0.006 | 0.917 ± 0.010 | +0.009 [+0.005, +0.012], 3/3 | −0.004 | 0.21 ± 0.36 → 0.63 ± 0.09 |
| Crosswalk | 0.949 ± 0.014 | 0.970 ± 0.003 | +0.020 [+0.002, +0.032], 3/3 | −0.004 | 0.90 → 0.92 |
| NoCurbRamp | 0.905 ± 0.021 | 0.921 ± 0.006 | +0.017 [+0.002, +0.041], 3/3 | −0.005 | 0.68 → 0.77 |
| Obstacle | 0.869 ± 0.008 | 0.878 ± 0.013 | +0.009 [−0.014, +0.022], 2/3 | +0.006 | 0.51 → 0.61 |
| SurfaceProblem | 0.720 ± 0.010 | 0.701 ± 0.016 | −0.019 [−0.024, −0.011], 0/3 | +0.001 | 0.03 → 0.00 |
| SurfaceProblem, `cl-flip` | 0.720 ± 0.010 | 0.721 ± 0.009 | +0.000 [−0.020, +0.016], 2/3 | −0.006 | 0.03 → 0.00 |

- **The direction survives, the size shrinks.** On the four types where seed 42 showed a held-out gain, `cl-drop`
  is ahead of its same-seed baseline in 11 of 12 pairs; the exception is Obstacle with seed 44 (−0.014
  [−0.028, −0.001]). But only four of the twelve pairs clear their own interval (Crosswalk, NoCurbRamp and
  Obstacle with seed 42, Obstacle with seed 43), and the seed-42 run was the most favorable of the three for
  Crosswalk, NoCurbRamp and Obstacle. The mean gains are about half of the seed-42 figures: NoCurbRamp +0.017
  against +0.041, Crosswalk +0.020 against +0.032. Part of that is selection: seed 42's numbers are the ones that
  picked `cl-drop` as the variant to replicate.
- **The seed moves the held-out AUC as much as the variant does.** The baseline's held-out AUC has a seed sd of
  0.006–0.021 (NoCurbRamp 0.881 / 0.912 / 0.921 across the three seeds), the size of the `cl-drop` effect. For
  Crosswalk and NoCurbRamp `cl-drop` is also the *less* seed-sensitive model (sd 0.003 and 0.006 against 0.014
  and 0.021), which is what removing a few hundred contradictory targets would be expected to do; for the other
  three types the spreads are alike.
- **The test cost is real and small.** CurbRamp, Crosswalk and NoCurbRamp lose 0.004–0.005 test AUC under
  `cl-drop` with every seed (0 of 3 ahead for CurbRamp and NoCurbRamp); Obstacle gains 0.006 with every seed;
  SurfaceProblem is unchanged.
- **The raw Disagree recall at 0.92 precision on the held-out rows is higher and steadier under `cl-drop`** for the
  four types (means 0.63 / 0.92 / 0.77 / 0.61 against 0.21 / 0.90 / 0.68 / 0.51). The baseline's CurbRamp figure is
  0.00 for two seeds and 0.61 for the third, so the round-3 number at that operating point is itself a matter of
  seed; `cl-drop`'s three are 0.53–0.70.
- **SurfaceProblem is not helped.** `cl-drop` lowers its held-out AUC with every seed (−0.011 to −0.024) and
  `cl-flip`, which looked like a gain with seed 42 (+0.016), averages to zero over three. The type's held-out AUC
  stays at 0.70–0.73 whatever is done to the train rows; its problem is not the targets this method can find.

Severity × tag slices for every variant, test and held-out, are in [`slices.md`](confident-learning-2026-10/slices.md),
with the baseline AUC and each variant's Δ and interval per slice; the training rows per slice are the round-3
counts in [`retrain-2026-10/slices.csv`](retrain-2026-10/slices.csv) (`cl-drop` removes 4–18% of a type's train
rows and the rules 16–44%, not evenly across slices: both fall mostly on `incorrect` rows).
The flag share by deployment, for the deployments with at least 1,000 train rows (private deployments pooled,
from `variants.json`): seattle-wa 12.4%, chicago-il 12.1%, spgg 10.3%, amsterdam 13.0%, newberg-or 10.8%,
columbus-oh 18.2%, mendota-il 8.5%, pittsburgh-pa 15.6%, teaneck-nj 7.6%, other (private deployments) 14.2%;
within a type the spread is wider (CurbRamp: seattle-wa 4.1%, chicago-il 8.7%, columbus-oh 15.5%, cdmx 18.8%).

## Agreement with the expert-rated rows

Only 65 of the 300 items rated in #66 are rows of the round-3 manifests at all (CurbRamp 5, Crosswalk 9,
NoCurbRamp 19, Obstacle 11, SurfaceProblem 21); the other 235 are lone or lightly voted AI Disagrees that never
met the margin rule, so they have no training target to check. On the 65, the training target matches the
expert 48 times of 63 (two were rated "can't tell"), and 37 of them are **train** rows (CurbRamp 3, Crosswalk 5,
NoCurbRamp 13, Obstacle 6, SurfaceProblem 10), the only ones a variant can touch; there the target matches the
expert 28 times. The report counts, per variant, the rated train rows it dropped or flipped and how many of
those the expert overturned, against the rows it left alone (`variants.md`, per type):

| variant | rated train rows touched | of which the expert overturns | rows left alone | of which the expert overturns |
|---|---:|---:|---:|---:|
| `symmetric` | 20 | 4 | 17 | 5 |
| `unanimous3` | 28 | 7 | 9 | 2 |
| `cl-drop` / `cl-flip` | 2 | 0 | 35 | 9 |

The rules drop rated rows the expert overturns at about the rate of the rows they keep (one in four either way),
and confident learning touched two of the 37 and both targets held. With 37 rows this is a sanity check, not a
measurement; the round-2 packets of [#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78)
are stratified to add rated items that are also training rows (95 in the two raters' train-split strata), and
the report reruns on the combined private file with `--gold` once they are rated; an item both raters saw
counts only when they gave the same verdict (`adjudicate_gold`), otherwise it is unrated.

## The deployment prior under each rule

`deployment_priors.json`, which `calibration/calibrate.py` reweights the val split to before fitting the
confidence maps, counts the human-authored labels that meet the training margin rule before the downsample.
[#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56) found that this prior is an artifact
of the rule's asymmetry and asked, as its second question, whether the rule itself should change. This is what
the file becomes under each rule (`confident_learning.py prior` over the cached rawLabels exports of the 54
deployments in the 2026-10-19 cities list, 1,667,379 labels; the margin column reproduces the shares #56 quotes
from the served file):

| Type | margin rule (served today) | `symmetric` | `unanimous3` |
|---|---:|---:|---:|
| CurbRamp | 28,470 / 3,797 (11.8% incorrect) | 28,470 / 1,255 (4.2%) | 25,545 / 939 (3.5%) |
| Crosswalk | 19,671 / 1,281 (6.1%) | 19,671 / 771 (3.8%) | 18,714 / 441 (2.3%) |
| NoCurbRamp | 13,030 / 7,774 (37.4%) | 13,030 / 4,790 (26.9%) | 9,692 / 1,646 (14.5%) |
| Obstacle | 12,247 / 6,822 (35.8%) | 12,247 / 2,304 (15.8%) | 9,300 / 1,694 (15.4%) |
| SurfaceProblem | 13,790 / 6,484 (32.0%) | 13,790 / 1,792 (11.5%) | 11,120 / 1,148 (9.4%) |

Against the ≥ 2-vote majority shares #56 measured on the production rows (4 / 6 / 20 / 22 / 15% for Crosswalk /
CurbRamp / NoCurbRamp / Obstacle / SurfaceProblem; a different population, so not the same number), the
symmetric rule lands at or below them for four types (3.8 / 4.2 / 15.8 / 11.5%) and covers about three fifths of
the gap for NoCurbRamp (26.9%): the served prior's distance from the crowd majority is the two-vote Disagrees,
and for three types removing them goes past it. Whatever prior calibration adopts, the counts above are what a
retrain under either rule would write to the file.

## Recommendation

1. **Take `cl-drop` into the next retrain, not into production from here.** The gain is where the Disagree
   gate fails, on deployments the model has not seen, and it costs about half a hundredth in distribution.
   Across three seeds the gain holds in direction for CurbRamp, Crosswalk and NoCurbRamp (3 of 3 seeds each) and
   mostly for Obstacle (2 of 3), at about half the size the first seed showed: +0.01 to +0.02 held-out AUC, the
   same order as the seed noise. That is enough to carry `cl-drop` into the round-4 recipe (one extra CPU step
   and five fold trainings per type, all scripted here) but not enough to call the gain settled: what is still
   missing is calibrated maps (`calibration/calibrate.py` on the `cl-drop` runs, read through the gate table the
   way [`retrain-2026-10.md`](retrain-2026-10.md) does) and deployments this experiment never selected on, which
   the next data pull will have. Leave SurfaceProblem's train rows alone: neither `cl-drop` nor `cl-flip` moves
   its held-out AUC up over three seeds.
2. **Do not filter the training set with the stricter vote rules as run here**, without rebalancing: they
   lose in distribution for every type and on the unseen deployments for SurfaceProblem and for Obstacle under
   `symmetric`, and gain only Crosswalk's. A fair test of the rule as a *definition* would downsample `correct` back to round 3's
   55:45 after the filter, which the chain does not do; that is one more batch of ten trainings if the question
   matters for training. The rule question of #56 stays a prior question: the table above is what the served
   prior becomes under either rule, and making the rule symmetric lands at or below the crowd-majority shares
   #56 measured for four types. That is a calibration decision, separate from training, and this experiment
   does not argue against it.
3. **Send the flags to an expert, not to the trash.** The flagged rows (`noisy_targets.csv` on klone, with the
   out-of-fold score, vote margin and target source) concentrate on the two-vote Disagrees but also mark 2–13%
   of the `correct` targets and half the hard slice; they are the right packet for the VLM adjudicator of
   [#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77) and for a round-3 re-label, where a
   rated flag becomes either a fixed target or a hard example to keep. The 37 rated train rows available today
   cannot score them; the round-2 packets of #78 add 95 and the report reruns with `--gold` when they are in.
4. **A variant worth one more batch:** `cl-drop` that keeps the expert-sourced and hard-slice rows whatever
   the fold models say (the rules already spare expert-sourced rows), to find out whether the transfer gain
   comes from dropping crowd targets or from dropping hard crops.
