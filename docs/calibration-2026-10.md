# Calibration of the deployed validators against production outcomes (October 2026)

Data pulled 2026-10-17 from every city schema's `label_ai_assessment` joined to its human votes
(`analysis/pull_ai_assessments.sh`, with accuracy and confidence exported unrounded). Maps fitted with
`python -m calibration.calibrate_from_production --min-human-votes 2 --holdout-fraction 0.5 --seed 42 --since 2025-08-16`.
Raw rows and per-label outputs stay outside the repo (they include private cities); only aggregates are here.
Related: [#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4) (prior), the audit's
[finding 5](audit-2026-10-14.md#5-the-confidence-to-precision-map-is-computed-under-the-wrong-prior).

**Scope.** These maps are for the model revisions in production today (sidewalk-ai-api `model_map.py`,
training_date 08-16-2025): curbramp `969ad5a`, nocurbramp `9f3f3bb`, obstacle `8f0fe8f`, surfaceproblem `0e1439d`,
crosswalk `b7936e5`. A retrained model needs its own map from `calibration/calibrate.py`.

## What production shows

996,774 Agree/Disagree assessments in 52 schemas (2025-08-16 .. 2026-10-17; the internal `sidewalk_validation`
schema and 163 assessments from before the current models were dropped). For each, the raw verdict, the raw softmax
confidence, the accuracy the current maps served, and the human votes on that label with the AI's own vote excluded.
Outcome = the strict majority of >= 2 human Agree/Disagree votes agrees with the AI's raw verdict (243,649 labels).

- **Raw Agree verdicts are right 93–99%** against the human majority (CurbRamp 98.8%, Crosswalk 99.3%, NoCurbRamp
  93.5%, SurfaceProblem 92.6%); Obstacle Agree is the exception at 85.2%.
- **Raw Disagree verdicts that passed the 0.92 gate are right only 50–72%**: CurbRamp 50%, Crosswalk 50%,
  SurfaceProblem 66%, Obstacle 70%, NoCurbRamp 72% (held-out half). Across all types the served accuracy of gated
  votes averages 0.956 against 0.919 observed, and the gap is almost entirely in the Disagree class.
- **The Obstacle `correct` map never reaches 0.92**, so the Obstacle model can never vote Agree, although its
  highest-confidence Agree verdicts are right about 94% of the time.
- **The Crosswalk `incorrect` map reaches 0.92 at confidence 0.5**, so every Crosswalk Disagree verdict becomes a vote;
  they are right 50% of the time.
- The API's lookup loop, applied to the current maps, reproduces the recorded served accuracy for 100.0% of
  assessments of every type, so the served numbers above are what the maps say, not a logging artefact.
- Half of all verdicts have confidence above 0.99905 and 29% above 0.9999, spread down to the sixth decimal. The
  map grid therefore has extra points between 0.999 and 1.0 (down to 0.999999); an earlier run on an export rounded
  to 4 decimals put most verdicts into the 0.9999/1.0 cells and mis-stated the top of the curve.

Why the old maps are off: they were computed on a 55:45 test split (production is roughly 90:10 on human votes), on
a file-level split that leaks across panoramas
([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)), with preprocessing that differs between
training and serving ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5)), and on a few hundred
test images per class. Production outcomes sidestep all four for the models as deployed.

## New maps: coverage and precision at the 0.92 gate

The shipped maps are isotonic fits on **all** labelled rows. A second fit on a hash-selected half is used only to
check: it supplies the held-out gate precision, the per-bin acceptance table and the hold-back rule below.

| type | verdict | assessed | labelled | observed precision | old coverage | old gate precision (held-out) | new coverage | new gate precision (held-out) | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CurbRamp | Agree | 301,716 | 65,769 | 98.8% | 91.5% | 99.1% (n=29,904) | 100.0% | 98.8% (n=32,586) | clears |
| CurbRamp | Disagree | 59,315 | 13,332 | 35.6% | 56.0% | 50.4% (n=4,005) | 0.0% | — (n=0) | below gate |
| NoCurbRamp | Agree | 82,327 | 30,304 | 93.5% | 89.3% | 95.4% (n=13,795) | 65.4% | 97.6% (n=10,495) | clears |
| NoCurbRamp | Disagree | 29,860 | 12,155 | 64.1% | 68.4% | 71.6% (n=4,525) | 0.0% | — (n=0) | held back |
| Obstacle | Agree | 114,646 | 22,467 | 85.2% | 0.0% | — (n=0) | 17.0% | 94.5% (n=2,814) | clears |
| Obstacle | Disagree | 72,857 | 13,722 | 44.9% | 20.1% | 70.0% (n=1,622) | 0.0% | 66.7% (n=3) | below gate |
| SurfaceProblem | Agree | 160,626 | 37,881 | 92.6% | 59.3% | 96.7% (n=11,291) | 64.8% | 96.4% (n=11,978) | clears |
| SurfaceProblem | Disagree | 52,509 | 14,909 | 45.7% | 29.8% | 65.6% (n=2,826) | 0.0% | — (n=0) | held back |
| Crosswalk | Agree | 110,644 | 30,783 | 99.3% | 73.7% | 99.6% (n=11,292) | 100.0% | 99.4% (n=15,196) | clears |
| Crosswalk | Disagree | 12,274 | 2,327 | 51.1% | 100.0% | 50.0% (n=1,145) | 0.0% | — (n=0) | held back |

Coverage is the share of all assessed labels of that type and verdict whose served accuracy (shipped map, after the
hold-back rule) is >= 0.92, i.e. how many AI votes would be cast. Gate precision is measured on the held-out half
of the labelled subset, old and new on the same labels.

Applied retrospectively to the 996,774 assessments, the new maps would cast about 590k votes instead of 623k, and
the estimated number of votes that contradict the human majority falls from about 47k to about 10k. Agree votes go
from 526k to 590k (Obstacle Agree is switched on; CurbRamp and Crosswalk Agree cover nearly everything) and
**Disagree votes go from 96k to zero**. The deployed models' Disagree verdicts still route labels to the Triage queue,
which is where they are useful today; a retrained model has to earn its Disagree votes.

### Hold-back rule

A (label type, verdict) class may clear 0.92 only if the held-out precision of the verdicts its check map lets
through, minus 1.96 standard errors, is >= 0.92. Otherwise its shipped map is capped at 0.9199 everywhere. Numbers
below are for the uncapped check map.

| type | verdict | gate n | gate precision | SE | precision − 1.96·SE | decision |
|---|---|---:|---:|---:|---:|---|
| CurbRamp | Agree | 32,586 | 0.9883 | 0.0006 | 0.9871 | **clears** |
| CurbRamp | Disagree | 0 | — | — | — | **below gate** |
| NoCurbRamp | Agree | 10,495 | 0.9758 | 0.0015 | 0.9729 | **clears** |
| NoCurbRamp | Disagree | 413 | 0.9080 | 0.0142 | 0.8801 | **held back** |
| Obstacle | Agree | 2,814 | 0.9446 | 0.0043 | 0.9361 | **clears** |
| Obstacle | Disagree | 3 | 0.6667 | 0.2722 | 0.1332 | **below gate** |
| SurfaceProblem | Agree | 11,978 | 0.9642 | 0.0017 | 0.9609 | **clears** |
| SurfaceProblem | Disagree | 0 | — | — | — | **held back** |
| Crosswalk | Agree | 15,196 | 0.9942 | 0.0006 | 0.9930 | **clears** |
| Crosswalk | Disagree | 126 | 0.9444 | 0.0204 | 0.9044 | **held back** |

- **Clears:** every Agree class. Obstacle Agree is the closest (0.945 on 2,814 held-out labels, lower bound 0.936).
- **Held back:** NoCurbRamp Disagree (0.908 on 413, lower bound 0.880), Crosswalk Disagree (0.944 on 126, lower
  bound 0.904: good, but too few labels to be sure), and SurfaceProblem Disagree (the map fitted on all rows reaches
  0.92 at the very top, but the check fit lets no held-out label through, so there is no evidence for it).
- **Below gate:** CurbRamp and Obstacle Disagree never reach 0.92, so there is nothing to hold back.

Reliability on the held-out half (|served − observed| averaged over confidence bins, weighted by n):

| type | Agree: old | Agree: new | Disagree: old | Disagree: new |
|---|---:|---:|---:|---:|
| CurbRamp | 0.044 | 0.002 | 0.576 | 0.011 |
| NoCurbRamp | 0.032 | 0.009 | 0.313 | 0.027 |
| Obstacle | 0.026 | 0.013 | 0.431 | 0.028 |
| SurfaceProblem | 0.021 | 0.010 | 0.446 | 0.028 |
| Crosswalk | 0.068 | 0.002 | 0.478 | 0.033 |

Pooled over all types and both verdicts: 0.134 for the old maps, 0.010 for the new.

## Held-out acceptance check

Check map (fitted on the other half) vs held-out observed precision, on confidence bins with n >= 200:

| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | median 1.96·SE |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | Agree | 13 | 10 | 12 | 0.048 | 0.013 |
| CurbRamp | Disagree | 12 | 5 | 12 | 0.032 | 0.036 |
| NoCurbRamp | Agree | 13 | 6 | 10 | 0.091 | 0.029 |
| NoCurbRamp | Disagree | 13 | 2 | 10 | 0.059 | 0.049 |
| Obstacle | Agree | 13 | 4 | 13 | 0.048 | 0.038 |
| Obstacle | Disagree | 13 | 2 | 11 | 0.066 | 0.044 |
| SurfaceProblem | Agree | 13 | 3 | 10 | 0.044 | 0.029 |
| SurfaceProblem | Disagree | 13 | 1 | 8 | 0.055 | 0.043 |
| Crosswalk | Agree | 7 | 6 | 6 | 0.018 | 0.008 |
| Crosswalk | Disagree | 2 | 1 | 2 | 0.036 | 0.050 |
| **all** | | 112 | 40 | 94 | 0.091 | 0.034 |

**The plan's ±0.01 criterion is met on 40 of 112 bins.** It is tighter than the data can resolve: the median 95%
interval on a bin's observed precision is ±0.034, and the fit half carries noise of its own. **94 of 112 bins are
within 1.96 SE** of the held-out observation (about 106 would be expected if the curve were exact and the fit half
noise-free). The largest misses are below the gate: NoCurbRamp Agree at confidence 0.5–0.6 (map 0.675 vs observed
0.766, n=273) and Obstacle Disagree near 0.999 (0.586 vs 0.652). Near the gate, SurfaceProblem Agree at 0.999–0.9999
(0.923 vs 0.938) and NoCurbRamp Agree at 0.99–0.995 (0.910 vs 0.929) are under-estimates, the safe direction.

## Caveats

- **Selection bias.** The fit uses production outcomes under the Triage queue's selection: SidewalkWebpage sends
  labels the AI contested to humans first, so the >= 2-vote subset over-represents hard Disagree cases. The Disagree
  curves are therefore pessimistic, which is the safe direction for a gate that votes against crowd labels.
- **The hold-out is split per label, not per panorama.** The assessment pull has no `pano_id`, so labels on the same
  panorama (often near-duplicate crops) can land on both sides, and the held-out numbers are somewhat optimistic.
  The hold-back rule's 1.96·SE margin does not cover this.
- The human majority is itself noisy (expert-vs-crowd agreement 86–91% in `analysis/analyze.py`), which caps
  observed precision below 1.0.
- These maps describe the current model revisions and the current mix of cities. A retrain, or a large new
  deployment, needs a new fit; `calibrate_from_production.py` re-runs on a fresh pull in under a minute.
