# Calibration of the deployed validators against production outcomes (October 2026)

Data pulled 2026-10-17 from every city schema's `label_ai_assessment` joined to its human votes
(`analysis/pull_ai_assessments.sh`). Maps fitted with
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

- **Raw Agree verdicts are right 93–99%** against the human majority at nearly every confidence (CurbRamp 98.8%,
  Crosswalk 99.3%, NoCurbRamp 93.5%, SurfaceProblem 92.6%); Obstacle Agree is the exception at 85.2%.
- **Raw Disagree verdicts that passed the 0.92 gate are right only 50–72%**: CurbRamp 50–51%, Crosswalk 50–51%,
  SurfaceProblem 66%, Obstacle 69–70%, NoCurbRamp 72%. Across all types the served accuracy of gated votes averages
  0.956 against 0.919 observed (ECE 0.037), and the gap is almost entirely in the Disagree class.
- **The Obstacle `correct` map never reaches 0.92**, so the Obstacle model can never vote Agree, although its
  highest-confidence Agree verdicts are right about 94% of the time.
- **The Crosswalk `incorrect` map reaches 0.92 at confidence 0.5**, so every Crosswalk Disagree verdict becomes a vote;
  they are right 51% of the time.
- The current maps are reproduced exactly by the API's lookup loop for every assessment once the export's
  4-decimal rounding of confidence is allowed for (100.0% of rows per type), so the served numbers above are what the
  maps say, not a logging artefact.

Why the old maps are off: they were computed on a 55:45 test split (production is roughly 90:10 on human votes), on
a file-level split that leaks across panoramas
([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)), with preprocessing that differs between
training and serving ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5)), and on a few hundred
test images per class. Production outcomes sidestep all four for the models as deployed.

## New maps: coverage and precision at the 0.92 gate

Labels with >= 2 human votes and a strict majority; coverage is over all assessed labels of that type and verdict; gate precision is measured on the held-out labelled subset.

| type | verdict | assessed | labelled | observed precision | old coverage | old gate precision | new coverage | new gate precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | Agree | 301,716 | 65,769 | 98.8% | 91.5% | 99.1% (n=29,904) | 99.4% | 98.8% (n=32,586) |
| CurbRamp | Disagree | 59,315 | 13,332 | 35.6% | 56.0% | 50.4% (n=4,005) | 0.0% | — (n=0) |
| NoCurbRamp | Agree | 82,327 | 30,304 | 93.5% | 89.3% | 95.4% (n=13,795) | 61.4% | 97.6% (n=10,504) |
| NoCurbRamp | Disagree | 29,860 | 12,155 | 64.1% | 68.4% | 71.6% (n=4,525) | 2.3% | 92.0% (n=225) |
| Obstacle | Agree | 114,646 | 22,467 | 85.2% | 0.0% | — (n=0) | 19.9% | 94.1% (n=2,973) |
| Obstacle | Disagree | 72,857 | 13,722 | 44.9% | 20.1% | 70.0% (n=1,622) | 0.0% | — (n=0) |
| SurfaceProblem | Agree | 160,626 | 37,881 | 92.6% | 59.3% | 96.7% (n=11,291) | 60.9% | 96.5% (n=11,649) |
| SurfaceProblem | Disagree | 52,509 | 14,909 | 45.7% | 29.8% | 65.6% (n=2,826) | 0.0% | — (n=0) |
| Crosswalk | Agree | 110,644 | 30,783 | 99.3% | 73.7% | 99.6% (n=11,292) | 98.5% | 99.4% (n=15,196) |
| Crosswalk | Disagree | 12,274 | 2,327 | 51.1% | 100.0% | 50.0% (n=1,145) | 0.0% | — (n=0) |

Coverage is the share of all assessed labels of that type and verdict whose served accuracy is >= 0.92, i.e. how
many AI votes would be cast. Gate precision is measured on the held-out half of the labelled subset (old and new on
the same labels).

Applied retrospectively to the 996,774 assessments, the new maps would cast about 581k votes instead of 623k, and
the estimated number of votes that contradict the human majority falls from about 47k to about 10k: Agree votes go
from 526k to 580k (Obstacle Agree is switched on, CurbRamp and Crosswalk Agree cover nearly everything), and
Disagree votes from 96k to under 1k (only NoCurbRamp's highest-confidence Disagree verdicts still clear 0.92). In
effect **the deployed models stop casting Disagree votes** until a retrained model earns them; their Disagree
verdicts still route labels to the Triage queue, which is where they are useful today.

Reliability on the held-out half (|served − observed| averaged over confidence bins, weighted by n):

| type | Agree: old | Agree: new | Disagree: old | Disagree: new |
|---|---:|---:|---:|---:|
| CurbRamp | 0.044 | 0.002 | 0.576 | 0.019 |
| NoCurbRamp | 0.032 | 0.010 | 0.312 | 0.029 |
| Obstacle | 0.025 | 0.012 | 0.431 | 0.027 |
| SurfaceProblem | 0.021 | 0.010 | 0.446 | 0.026 |
| Crosswalk | 0.068 | 0.002 | 0.478 | 0.038 |

Pooled over all types and both verdicts: 0.133 for the old maps, 0.011 for the new.

## Held-out acceptance check

Held-out check: |new map − observed| <= 0.01 on bins with n >= 200.

| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | median 1.96·SE |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | Agree | 13 | 10 | 13 | 0.023 | 0.013 |
| CurbRamp | Disagree | 12 | 5 | 11 | 0.056 | 0.036 |
| NoCurbRamp | Agree | 13 | 7 | 8 | 0.092 | 0.029 |
| NoCurbRamp | Disagree | 13 | 2 | 10 | 0.066 | 0.049 |
| Obstacle | Agree | 13 | 4 | 12 | 0.039 | 0.038 |
| Obstacle | Disagree | 13 | 2 | 11 | 0.062 | 0.044 |
| SurfaceProblem | Agree | 13 | 3 | 10 | 0.042 | 0.029 |
| SurfaceProblem | Disagree | 13 | 2 | 9 | 0.054 | 0.044 |
| Crosswalk | Agree | 8 | 7 | 7 | 0.018 | 0.007 |
| Crosswalk | Disagree | 1 | 0 | 1 | 0.017 | 0.034 |
| **all** | | 112 | 42 | 92 | 0.092 | 0.034 |

**The plan's ±0.01 criterion is met on 42 of 112 bins.** It is tighter than the data can resolve: the median
95% interval on a bin's observed precision is ±0.034, and the fit half carries its own noise of similar size. 92 of
112 bins are within 1.96 SE of the held-out observation (about 106 would be expected if the curve were exact and the
fit half noise-free). The largest misses are NoCurbRamp Agree at confidence 0.5–0.6 (map 0.674 vs observed 0.766,
n=273, far below the gate either way) and NoCurbRamp Disagree near 0.99. The one miss that matters for voting is
**NoCurbRamp Disagree at confidence >= 0.9999**: the map serves 0.930 and held-out precision is 0.907 (n=580),
so the 225 held-out labels that clear the gate are right 92.0% of the time, right at the threshold. That class is the
one to watch, or to hold back, when these maps ship.

## Caveats

- **Selection bias.** SidewalkWebpage's Triage queue sends labels the AI contested to humans first, so the >= 2-vote
  subset over-represents hard Disagree cases. The Disagree curves are therefore pessimistic, which is the safe
  direction for a gate that votes against crowd labels.
- The human majority is itself noisy (expert-vs-crowd agreement 86–91% in `analysis/analyze.py`), which caps
  observed precision below 1.0.
- These maps describe the current model revisions and the current mix of cities. A retrain, or a large new
  deployment, needs a new fit; `calibrate_from_production.py` can be re-run on a fresh pull in about 20 seconds.
