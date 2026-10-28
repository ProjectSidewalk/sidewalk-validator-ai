# Wrong crowd targets: confident learning and stricter Disagree rules (October 2026)

Experiment for [#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76) and
[#75](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75), two arms of the umbrella
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74): can the training targets that the
crowd gets wrong be found without rating labor, and does removing or correcting them give a validator whose
Disagree verdicts hold up? Everything here ran on the round-3 crops and splits
([`retrain-2026-10.md`](retrain-2026-10.md)); only the **train** rows change between variants, so every number
below is on the same val / test / test_heldout rows as the round-3 baseline.

**Status: running.** The chain was submitted on klone on 2026-10-28; this note fills in as the outputs land.

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
| Folds | 5, grouped by panorama with the split's 3 m merge, class-ratio stratified (`confident_learning.py folds`) |
| Flag rule | confident joint, `prune_by_noise_rate` (`confident_learning.py flag`) |
| Variants per type | `symmetric`, `unanimous3`, `cl-drop`, `cl-flip`; baseline = round 3 |
| Evaluation | `evaluate.py --per-city` on the round-3 split with each variant's best checkpoint (`--tag <variant>`): test and held-out AUC, within-city AUC, the hard slice, recall at precision 0.92 per verdict, severity × tag slices |
| Compute | klone `ckpt-all`, one GPU per run (A40 or better); 25 fold runs + 20 variant runs; fold checkpoints deleted after their out-of-fold scores are written |

Reproduce from a checkout of this branch on klone (`DATA` is the round-3 output directory, `RUNS` a scratch
directory for checkpoints):

```
DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/cl-2026-10 \
    analysis/cl_submit.sh CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem
# when every job is done:
python analysis/confident_learning.py report --data-dir $DATA --out docs/confident-learning-2026-10 \
    --public-cities "$(curl -sL https://sidewalk-sea.cs.washington.edu/v3/api/cities | python -c 'import json,sys; print(",".join(c["city_id"] for c in json.load(sys.stdin)["cities"] if c.get("visibility") == "public"))')"
```

## Train-row accounting

_To be filled from `rule.json` and `flag_summary.json` per type: rows dropped by each rule, rows flagged by
the confident joint (by class, by vote margin, by target source, by author, by deployment with private
deployments pooled), and the overlap between the rule drops and the flags._

## Results

_One table per type from `confident_learning.py report` (`confident-learning-2026-10/variants.md`): baseline
and the four variants on test, held-out, within-city and hard-slice AUC, and `incorrect` recall at precision
0.92. Severity × tag slices per variant are in the `slices_<variant>.md` files next to the predictions._

## Agreement with the expert-rated rows

Only 65 of the 300 items rated in #66 are rows of the round-3 training manifests (CurbRamp 5, Crosswalk 9,
NoCurbRamp 19, Obstacle 11, SurfaceProblem 21); the other 235 are lone or lightly voted AI Disagrees that never
met the margin rule, so they have no training target to check. On the 65, the training target matches the
expert 48 times of 63 (two were rated "can't tell"). The report step counts, per variant, how many of the rated
rows it touched and how many of those the expert overturned, against the untouched rows; with so few rows this
is a sanity check, not a measurement, until the round-2 packets of
[#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78) add rated items that are also
training rows.

## The deployment prior under each rule

_From `confident_learning.py prior`: the pre-downsampling class counts of human-authored labels meeting each
rule, per type, i.e. what `deployment_priors.json` becomes if a rule is adopted (the second question of #56)._

## Recommendation

_Pending results._
