# Calibration of the deployed validators against production outcomes (October 2026)

Data pulled 2026-10-17 from every city schema's `label_ai_assessment` joined to its human votes
(`analysis/pull_ai_assessments.sh`, accuracy and confidence exported unrounded). Maps fitted with
`python -m calibration.calibrate_from_production --min-human-votes 2 --folds 5 --seed 42 --since 2025-08-16`
(outcome `tie_half`, the default). Raw rows and per-label outputs stay outside the repo because they include private
cities; only aggregates are here. Related: [#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)
(prior), and the audit's
[finding 5](audit-2026-10-14.md#5-the-confidence-to-precision-map-is-computed-under-the-wrong-prior).

**Scope.** These maps are for the model revisions in production today (sidewalk-ai-api `model_map.py`,
training_date 08-16-2025): curbramp `969ad5a`, nocurbramp `9f3f3bb`, obstacle `8f0fe8f`, surfaceproblem `0e1439d`,
crosswalk `b7936e5`. A retrained model needs its own map from `calibration/calibrate.py`.

**Model window.** The API pinned revisions only on 2026-10-15. Before that, a process started before 2025-08-16 could
have served the previous weights for a few days after the new models were published (late-August volume is about
2.4k rows), and the assessment table records `model_id` but not the revision. Recording `validator_revision` per
assessment is now tracked in
[ProjectSidewalk/SidewalkWebpage#5794](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5794).

## Data

The fit uses **996,774 Agree/Disagree assessments in 52 schemas, 2025-08-16 .. 2026-10-17**. The internal
`sidewalk_validation` schema and 163 assessments from before 2025-08-16 were dropped. Each row has the raw verdict,
the raw softmax confidence, the accuracy the current maps served, and the human votes on that label with the AI's own
vote excluded. "Labelled" means at least 2 human Agree/Disagree votes (273,169 labels).

**Outcome (`tie_half`).** 1 if the strict human majority matches the AI's verdict, 0 if it opposes it, 0.5 for a
1–1 (or 2–2 …) tie. Dropping ties instead (`strict`) keeps only unanimous pairs at two votes. Tie shares are 5–14%
for the Agree classes other than Crosswalk and 13–23% for Disagree, so dropping them flatters the marginal classes.

**Why not the vote share.** The review proposed `share`, the fraction of human votes that match the verdict
(2–1 → 0.667). That measures agreement with one randomly chosen human voter rather than with the crowd's verdict, so
it is capped by how often humans disagree with each other. A model that always sided with the majority would score
0.85 on NoCurbRamp, SurfaceProblem and Obstacle, 0.93 on CurbRamp and 0.97 on Crosswalk. Under `share` the 0.92 gate
is unreachable for three of the five types whatever the model does. Its results are reported below as a sensitivity
row, but it does not drive the maps.

## What production shows

Precision of each raw verdict against the human majority (`tie_half`, all labelled rows) varies by type and by
confidence:

- **Agree:** CurbRamp 96.2%, Crosswalk 98.5%, NoCurbRamp 89.2%, SurfaceProblem 86.5%, Obstacle 80.8%. It rises with
  confidence. Obstacle Agree stays at 0.78–0.84 even at confidence 0.99–0.9999 and reaches 0.90 only above 0.9999.
  NoCurbRamp Agree is about 0.70 at confidence 0.5–0.7. So "Agree is right 93–99% at every confidence" holds only for CurbRamp and Crosswalk.
- **Disagree:** 38–62% overall. Among the Disagree verdicts that passed the current 0.92 gate:

  | type | gate precision (old maps) |
  |---|---:|
  | CurbRamp | 50.8% |
  | Crosswalk | 51.0% |
  | SurfaceProblem | 62.4% |
  | Obstacle | 65.4% |
  | NoCurbRamp | 69.3% |

  Under the strict outcome on the earlier held-out half the figures were CurbRamp 50.4% and Crosswalk 50.0%.
- **The Obstacle `correct` map never reaches 0.92** today, so Obstacle never votes Agree. **The Crosswalk
  `incorrect` map reaches 0.92 at confidence 0.5**, so every Crosswalk Disagree verdict becomes a vote.
- The API's lookup loop on the current maps reproduces the recorded served accuracy for 100.0% of assessments of
  every type.
- Half of all verdicts have confidence above 0.99905 and 29% above 0.9999, spread down to the sixth decimal. The grid
  therefore has extra points from 0.999 to 0.999999.

## Method

- **Shipped maps:** an isotonic fit of the outcome against raw confidence on all labelled rows. Each grid value is
  the **mean of its isotonic block**, a calibrated point estimate. The exception is the **thin-block guard**: a block
  with fewer than 300 labels is served at the lower end of its 90% Wilson interval, with a running maximum to keep
  the map monotone. That stops a thin end of the curve from voting on a lucky estimate; the 0.5–0.6 CurbRamp and
  Crosswalk bins are the cases that matter, at n ≈ 100–640. The guard moves served values by −0.0002 to −0.0065
  depending on the class, except Crosswalk Disagree, where all 36 blocks are thin (−0.064):

| type | verdict | blocks | thin blocks | mean shift |
|---|---|---:|---:|---:|
| CurbRamp | Agree | 32 | 8 | -0.0002 |
| CurbRamp | Disagree | 41 | 20 | -0.0021 |
| NoCurbRamp | Agree | 41 | 14 | -0.0005 |
| NoCurbRamp | Disagree | 46 | 28 | -0.0065 |
| Obstacle | Agree | 35 | 14 | -0.0011 |
| Obstacle | Disagree | 34 | 17 | -0.0038 |
| SurfaceProblem | Agree | 48 | 13 | -0.0004 |
| SurfaceProblem | Disagree | 46 | 22 | -0.0036 |
| Crosswalk | Agree | 27 | 7 | -0.0005 |
| Crosswalk | Disagree | 36 | 36 | -0.0641 |

- **Hold-back rule (5-fold cross-fitting):** folds come from a hash of `schema:label_id`. Each labelled label gets an
  out-of-fold served value from a map fitted without its fold. The labels that clear 0.92 out-of-fold are pooled, and
  the class may clear the gate only if the **Wilson 95% lower bound** of their mean outcome is ≥ 0.92. Otherwise the
  shipped map is capped at 0.9199. This replaces the single seed-42 split, whose decisions flipped with the seed.

## Results: coverage and precision at the 0.92 gate

| type | verdict | assessed | labelled (ties) | precision | old coverage | old gate precision | new coverage | new gate precision (out-of-fold) | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CurbRamp | Agree | 301,716 | 69,456 (5.3%) | 96.2% | 91.5% | 96.8% (n=63,111) | 95.3% | 96.6% (n=65,698) | clears |
| CurbRamp | Disagree | 59,315 | 16,061 (17.0%) | 38.0% | 56.0% | 50.8% (n=9,883) | 0.0% | — (n=0) | below gate |
| NoCurbRamp | Agree | 82,327 | 33,634 (9.9%) | 89.2% | 89.3% | 91.1% (n=30,663) | 43.3% | 94.8% (n=16,576) | clears |
| NoCurbRamp | Disagree | 29,860 | 14,119 (13.9%) | 62.2% | 68.4% | 69.3% (n=10,485) | 0.0% | 88.9% (n=9) | below gate |
| Obstacle | Agree | 114,646 | 25,637 (12.4%) | 80.8% | 0.0% | — (n=0) | 5.5% | 94.2% (n=2,098) | clears |
| Obstacle | Disagree | 72,857 | 16,771 (18.2%) | 45.8% | 20.1% | 65.4% (n=3,943) | 0.0% | — (n=0) | below gate |
| SurfaceProblem | Agree | 160,626 | 44,204 (14.3%) | 86.5% | 59.3% | 91.6% (n=25,432) | 26.6% | 94.5% (n=9,813) | clears |
| SurfaceProblem | Disagree | 52,509 | 19,264 (22.6%) | 46.6% | 29.8% | 62.4% (n=7,242) | 0.0% | — (n=0) | below gate |
| Crosswalk | Agree | 110,644 | 31,339 (1.8%) | 98.5% | 73.7% | 98.9% (n=23,104) | 97.3% | 98.6% (n=30,548) | clears |
| Crosswalk | Disagree | 12,274 | 2,684 (13.3%) | 51.0% | 100.0% | 51.0% (n=2,684) | 0.0% | — (n=0) | below gate |

Coverage is the share of all assessed labels of that type and verdict whose served accuracy under the shipped map is
≥ 0.92. New gate precision is out-of-fold. Where the shipped maps first reach 0.92:

| type | Agree from confidence |
|---|---|
| CurbRamp | 0.855 |
| Crosswalk | 0.8825 |
| NoCurbRamp | 0.999 |
| SurfaceProblem | 0.99999 |
| Obstacle | 0.999998 |

No Disagree map reaches 0.92.

### Hold-back decisions

| type | verdict | pooled n | precision | Wilson 95% lower | decision | at 0.90 |
|---|---|---:|---:|---:|---|---|
| CurbRamp | Agree | 65,698 | 0.9660 | 0.9646 | **clears** | clears |
| CurbRamp | Disagree | 0 | — | — | **below gate** | below gate |
| NoCurbRamp | Agree | 16,576 | 0.9485 | 0.9450 | **clears** | clears |
| NoCurbRamp | Disagree | 9 | 0.8889 | 0.5650 | **below gate** | below gate |
| Obstacle | Agree | 2,098 | 0.9418 | 0.9310 | **clears** | clears |
| Obstacle | Disagree | 0 | — | — | **below gate** | below gate |
| SurfaceProblem | Agree | 9,813 | 0.9446 | 0.9399 | **clears** | clears |
| SurfaceProblem | Disagree | 0 | — | — | **below gate** | below gate |
| Crosswalk | Agree | 30,548 | 0.9863 | 0.9849 | **clears** | clears |
| Crosswalk | Disagree | 0 | — | — | **below gate** | below gate |

- **Every Agree class clears.**
  - Obstacle Agree is the most marginal: 0.942 on 2,098 out-of-fold labels, lower bound 0.931. It votes only at
    confidence ≥ 0.999998, which comes to about 6.3k votes over the period (5.5% coverage), against none today.
  - SurfaceProblem Agree is next: 0.945 on 9,813, lower bound 0.940.
- **No Disagree class reaches the gate** under `tie_half`, so the hold-back cap never fires. Under `strict`,
  NoCurbRamp Disagree reaches 0.953 on 149 labels but its lower bound of 0.906 is borderline, so it is held back
  conservatively. No Disagree class demonstrates ≥ 0.92 with confidence under any of the three outcomes.

Sensitivity. These rows are not used for the maps.

`strict` (ties dropped):

| type | verdict | pooled n | precision | Wilson 95% lower | decision |
|---|---|---:|---:|---:|---|
| CurbRamp | Agree | 65,502 | 0.9884 | 0.9875 | clears |
| CurbRamp | Disagree | 0 | — | — | below gate |
| NoCurbRamp | Agree | 21,116 | 0.9749 | 0.9727 | clears |
| NoCurbRamp | Disagree | 149 | 0.9530 | 0.9062 | held back |
| Obstacle | Agree | 4,951 | 0.9463 | 0.9396 | clears |
| Obstacle | Disagree | 0 | — | — | below gate |
| SurfaceProblem | Agree | 23,562 | 0.9643 | 0.9618 | clears |
| SurfaceProblem | Disagree | 0 | — | — | below gate |
| Crosswalk | Agree | 30,356 | 0.9942 | 0.9932 | clears |
| Crosswalk | Disagree | 0 | — | — | below gate |

`share` (vote share, capped by human disagreement as explained above):

| type | verdict | pooled n | precision | Wilson 95% lower | decision |
|---|---|---:|---:|---:|---|
| CurbRamp | Agree | 54,617 | 0.9552 | 0.9534 | clears |
| CurbRamp | Disagree | 0 | — | — | below gate |
| NoCurbRamp | Agree | 2,825 | 0.9220 | 0.9116 | held back |
| NoCurbRamp | Disagree | 0 | — | — | below gate |
| Obstacle | Agree | 0 | — | — | below gate |
| Obstacle | Disagree | 0 | — | — | below gate |
| SurfaceProblem | Agree | 2,810 | 0.9260 | 0.9157 | held back |
| SurfaceProblem | Disagree | 0 | — | — | below gate |
| Crosswalk | Agree | 30,720 | 0.9763 | 0.9745 | clears |
| Crosswalk | Disagree | 0 | — | — | below gate |

### What the website will see

| type | verdict | old votes | new votes | lost (→ Unsure) | gained |
|---|---|---:|---:|---:|---:|
| CurbRamp | Agree | 276,019 | 287,678 | 0 | 11,659 |
| CurbRamp | Disagree | 33,229 | 0 | 33,229 | 0 |
| NoCurbRamp | Agree | 73,558 | 35,645 | 37,913 | 0 |
| NoCurbRamp | Disagree | 20,435 | 0 | 20,435 | 0 |
| Obstacle | Agree | 0 | 6,343 | 0 | 6,343 |
| Obstacle | Disagree | 14,634 | 0 | 14,634 | 0 |
| SurfaceProblem | Agree | 95,178 | 42,655 | 52,523 | 0 |
| SurfaceProblem | Disagree | 15,638 | 0 | 15,638 | 0 |
| Crosswalk | Agree | 81,563 | 107,634 | 0 | 26,071 |
| Crosswalk | Disagree | 12,274 | 0 | 12,274 | 0 |
| **all** | Agree | 526,318 | 479,955 | 90,436 | 44,073 |
| **all** | Disagree | 96,210 | 0 | 96,210 | 0 |

- **Agree:** about 480k Agree votes over the period instead of 526k. 90k are lost (NoCurbRamp 38k, SurfaceProblem
  53k) and 44k gained (Crosswalk 26k, CurbRamp 12k, Obstacle 6k).
- **Disagree:** all 96k Disagree votes are lost.
- **What a lost vote becomes.** A verdict that no longer clears the gate is submitted as an **Unsure** vote by the AI
  account. It counts toward the label's `unsure_count` and uses one of the five crowd-vote slots.
- **Triage stops seeing AI Disagree verdicts.** `ValidationQueuePolicy.aiContested` reads the AI's *vote*, so
  Disagree verdicts would no longer trigger the Triage queue. Keeping Disagree-based triage would need SidewalkWebpage
  to key on the raw `ai_result` instead.
- **The NoCurbRamp Agree drop is per-label semantics, not miscalibration.** The old gate let through NoCurbRamp Agree
  verdicts whose pooled precision is 0.911. The new map keeps a subset at 0.948. The 14,087 labelled labels it drops
  (confidence mostly 0.90–0.999) are right 0.866 of the time individually: each of those votes is below 0.92 on its
  own, even though the pooled old set looked acceptable. On the website that is about 38k fewer NoCurbRamp Agree votes
  and as many more Unsure.

### Reliability (out-of-fold)

| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | mean diff |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | Agree | 13 | 12 | 12 | 0.046 | -0.005 |
| CurbRamp | Disagree | 13 | 4 | 10 | 0.037 | -0.011 |
| NoCurbRamp | Agree | 13 | 11 | 12 | 0.047 | -0.006 |
| NoCurbRamp | Disagree | 13 | 3 | 7 | 0.055 | -0.016 |
| Obstacle | Agree | 13 | 10 | 11 | 0.054 | -0.006 |
| Obstacle | Disagree | 13 | 5 | 9 | 0.055 | -0.010 |
| SurfaceProblem | Agree | 13 | 10 | 12 | 0.029 | -0.005 |
| SurfaceProblem | Disagree | 13 | 6 | 9 | 0.031 | -0.009 |
| Crosswalk | Agree | 8 | 7 | 7 | 0.033 | -0.006 |
| Crosswalk | Disagree | 3 | 0 | 0 | 0.106 | -0.092 |
| **all** | | 115 | 68 | 89 | 0.106 | -0.011 |

**89 of 115 bins are within 1.96 binomial SE**, and the mean signed gap is −0.011. ±0.01 is met on 68 bins.
- **Bias outside Crosswalk Disagree:** −0.005 to −0.016 per class. That is the thin-block guard plus out-of-fold fits
  being noisier than the all-rows fit.
- **Crosswalk Disagree:** −0.092. All its blocks are thin, and it is far below the gate.
- **Bins served above observed:** 35, by at most +0.021.
  - The only ones above 0.92 are two CurbRamp Agree bins: 0.925 vs 0.923, and 0.929 vs 0.926. Both are still above
    the gate when observed.
- **Pooled calibration error** (|served − observed| by bin, weighted by n): 0.144 for the old maps, 0.008 for the
  new.

### City mix

The labelled sample is not the assessed population:

- **Over-represented:** Seattle (25% of labelled vs 19% of assessed), Chicago (17% vs 15%), Kaohsiung (15% vs 9%) and
  Taipei.
- **Under-represented:** spgg (2.8% vs 9.7%), cdmx (2.2% vs 7.2%) and St. Louis (1.1% vs 6.1%). Coverage numbers for
  these cities are extrapolated from the others.
- **Per city (n ≥ 500 labelled):** the out-of-fold precision of the new votes is ≥ 0.92 in 27 of the 28 cities. The
  exception is **spgg at 0.909** (n = 1,745 new votes); cdmx is next at 0.922.
- **City × type strata below 0.92 with n ≥ 300:** cdmx NoCurbRamp (0.851, n = 397), spgg CurbRamp (0.902), Seattle
  SurfaceProblem (0.901) and Seattle Obstacle (0.917).

A per-city gate is a SidewalkWebpage decision; these maps are global.

## Caveats

- **Selection.** Humans do not vote on a random sample of AI-assessed labels: the Triage queue routes AI-contested
  labels to them, and labels collect votes at different rates. Disagree precision rises with the number of human votes:
  NoCurbRamp 0.37 / 0.50 / 0.56 at 1 / 2 / 3 votes (`tie_half`), with the same trend for the other types. The routing
  effect looks small. The direction of the bias on the ≥ 2-vote subset cannot be identified from vote counts alone,
  because the order in which votes arrived is not exported. Earlier versions of this note called the subset
  "pessimistic for Disagree, the safe direction"; that claim is withdrawn.
- **Folds are split per label, not per panorama.** The assessment pull has no `pano_id`, so labels on the same
  panorama can fall in different folds and the out-of-fold numbers are somewhat optimistic.
- The human majority is itself noisy (expert-vs-crowd agreement 86–91% in `analysis/analyze.py`).
- These maps describe the current model revisions and city mix. A retrain or a large new deployment needs a new fit.
  `calibrate_from_production.py` re-runs on a fresh pull in under a minute.
