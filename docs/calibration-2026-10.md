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
assessment is now tracked in a SidewalkWebpage issue.

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
  the **lower end of a 90% Wilson interval for its isotonic block**, so the served accuracy is a lower confidence
  bound on precision, not a point estimate. Blocks hold a median of 210–1,160 labels, depending on the class (68 for
  Crosswalk Disagree), so this lowers served values by 0.016 on average, from 0.006 for CurbRamp/Crosswalk Agree up to
  0.064 for Crosswalk Disagree. At the thin low-confidence end a single lucky block can no longer vote.
- **Hold-back rule (5-fold cross-fitting):** folds come from a hash of `schema:label_id`. Each labelled label gets an
  out-of-fold served value from a map fitted without its fold. The labels that clear 0.92 out-of-fold are pooled, and
  the class may clear the gate only if the **Wilson 95% lower bound** of their mean outcome is ≥ 0.92. Otherwise the
  shipped map is capped at 0.9199. This replaces the single seed-42 split, whose decisions flipped with the seed.

## Results: coverage and precision at the 0.92 gate

| type | verdict | assessed | labelled (ties) | precision | old coverage | old gate precision | new coverage | new gate precision (out-of-fold) | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CurbRamp | Agree | 301,716 | 69,456 (5.3%) | 96.2% | 91.5% | 96.8% (n=63,111) | 88.9% | 97.1% (n=58,263) | clears |
| CurbRamp | Disagree | 59,315 | 16,061 (17.0%) | 38.0% | 56.0% | 50.8% (n=9,883) | 0.0% | — (n=0) | below gate |
| NoCurbRamp | Agree | 82,327 | 33,634 (9.9%) | 89.2% | 89.3% | 91.1% (n=30,663) | 34.5% | 95.1% (n=14,230) | clears |
| NoCurbRamp | Disagree | 29,860 | 14,119 (13.9%) | 62.2% | 68.4% | 69.3% (n=10,485) | 0.0% | 88.9% (n=9) | below gate |
| Obstacle | Agree | 114,646 | 25,637 (12.4%) | 80.8% | 0.0% | — (n=0) | 0.2% | 94.8% (n=781) | clears |
| Obstacle | Disagree | 72,857 | 16,771 (18.2%) | 45.8% | 20.1% | 65.4% (n=3,943) | 0.0% | — (n=0) | below gate |
| SurfaceProblem | Agree | 160,626 | 44,204 (14.3%) | 86.5% | 59.3% | 91.6% (n=25,432) | 19.3% | 94.7% (n=7,957) | clears |
| SurfaceProblem | Disagree | 52,509 | 19,264 (22.6%) | 46.6% | 29.8% | 62.4% (n=7,242) | 0.0% | — (n=0) | below gate |
| Crosswalk | Agree | 110,644 | 31,339 (1.8%) | 98.5% | 73.7% | 98.9% (n=23,104) | 97.3% | 98.6% (n=30,548) | clears |
| Crosswalk | Disagree | 12,274 | 2,684 (13.3%) | 51.0% | 100.0% | 51.0% (n=2,684) | 0.0% | — (n=0) | below gate |

Coverage is the share of all assessed labels of that type and verdict whose served accuracy under the shipped map is
≥ 0.92. New gate precision is out-of-fold.

### Hold-back decisions

| type | verdict | pooled n | precision | Wilson 95% lower | decision | at 0.90 |
|---|---|---:|---:|---:|---|---|
| CurbRamp | Agree | 58,263 | 0.9707 | 0.9693 | **clears** | clears |
| CurbRamp | Disagree | 0 | — | — | **below gate** | below gate |
| NoCurbRamp | Agree | 14,230 | 0.9511 | 0.9474 | **clears** | clears |
| NoCurbRamp | Disagree | 9 | 0.8889 | 0.5650 | **below gate** | below gate |
| Obstacle | Agree | 781 | 0.9481 | 0.9303 | **clears** | clears |
| Obstacle | Disagree | 0 | — | — | **below gate** | below gate |
| SurfaceProblem | Agree | 7,957 | 0.9471 | 0.9420 | **clears** | clears |
| SurfaceProblem | Disagree | 0 | — | — | **below gate** | below gate |
| Crosswalk | Agree | 30,548 | 0.9863 | 0.9849 | **clears** | clears |
| Crosswalk | Disagree | 0 | — | — | **below gate** | below gate |

- **Every Agree class clears.** Obstacle Agree clears narrowly: 0.948 on 781 out-of-fold labels, lower bound 0.930.
  After the Wilson lowering only its highest-confidence verdicts reach 0.92, so it casts about 275 votes over the
  whole period (0.2% coverage), against none today. SurfaceProblem Agree is the next most marginal: 0.947 on 7,957,
  lower bound 0.942.
- **No Disagree class reaches the gate** under `tie_half`. All five fall below it, so the hold-back cap is not even
  needed: the cross-fitted, tie-aware curves never reach 0.92. Under `strict`, NoCurbRamp Disagree reaches 0.953 on
  149 labels but its lower bound of 0.906 is borderline, so it is held back conservatively. No Disagree class
  demonstrates ≥ 0.92 with confidence under any of the three outcomes.

Sensitivity. These rows are not used for the maps.

`strict` (ties dropped):

| type | verdict | pooled n | precision | Wilson 95% lower | decision |
|---|---|---:|---:|---:|---|
| CurbRamp | Agree | 64,700 | 0.9889 | 0.9880 | clears |
| CurbRamp | Disagree | 0 | — | — | below gate |
| NoCurbRamp | Agree | 21,116 | 0.9749 | 0.9727 | clears |
| NoCurbRamp | Disagree | 149 | 0.9530 | 0.9062 | held back |
| Obstacle | Agree | 1,974 | 0.9696 | 0.9611 | clears |
| Obstacle | Disagree | 0 | — | — | below gate |
| SurfaceProblem | Agree | 21,260 | 0.9679 | 0.9655 | clears |
| SurfaceProblem | Disagree | 0 | — | — | below gate |
| Crosswalk | Agree | 30,292 | 0.9943 | 0.9934 | clears |
| Crosswalk | Disagree | 0 | — | — | below gate |

`share` (vote share, capped by human disagreement as explained above):

| type | verdict | pooled n | precision | Wilson 95% lower | decision |
|---|---|---:|---:|---:|---|
| CurbRamp | Agree | 50,434 | 0.9573 | 0.9555 | clears |
| CurbRamp | Disagree | 0 | — | — | below gate |
| NoCurbRamp | Agree | 0 | — | — | below gate |
| NoCurbRamp | Disagree | 0 | — | — | below gate |
| Obstacle | Agree | 0 | — | — | below gate |
| Obstacle | Disagree | 0 | — | — | below gate |
| SurfaceProblem | Agree | 0 | — | — | below gate |
| SurfaceProblem | Disagree | 0 | — | — | below gate |
| Crosswalk | Agree | 29,730 | 0.9775 | 0.9757 | clears |
| Crosswalk | Disagree | 0 | — | — | below gate |

### What the website will see

| type | verdict | old votes | new votes | lost (→ Unsure) | gained |
|---|---|---:|---:|---:|---:|
| CurbRamp | Agree | 276,019 | 268,367 | 7,652 | 0 |
| CurbRamp | Disagree | 33,229 | 0 | 33,229 | 0 |
| NoCurbRamp | Agree | 73,558 | 28,409 | 45,149 | 0 |
| NoCurbRamp | Disagree | 20,435 | 0 | 20,435 | 0 |
| Obstacle | Agree | 0 | 275 | 0 | 275 |
| Obstacle | Disagree | 14,634 | 0 | 14,634 | 0 |
| SurfaceProblem | Agree | 95,178 | 30,960 | 64,218 | 0 |
| SurfaceProblem | Disagree | 15,638 | 0 | 15,638 | 0 |
| Crosswalk | Agree | 81,563 | 107,634 | 0 | 26,071 |
| Crosswalk | Disagree | 12,274 | 0 | 12,274 | 0 |
| **all** | Agree | 526,318 | 435,645 | 117,019 | 26,346 |
| **all** | Disagree | 96,210 | 0 | 96,210 | 0 |

- **Agree:** about 436k Agree votes over the period instead of 526k. 117k are lost and 26k gained: Crosswalk gains
  26k, and Obstacle gains 275.
- **Disagree:** all 96k Disagree votes are lost.
- **What a lost vote becomes.** A verdict that no longer clears the gate is submitted as an **Unsure** vote by the AI
  account. It counts toward the label's `unsure_count` and uses one of the five crowd-vote slots.
- **Triage stops seeing AI Disagree verdicts.** `ValidationQueuePolicy.aiContested` reads the AI's *vote*, so
  Disagree verdicts would no longer trigger the Triage queue. Keeping Disagree-based triage would need SidewalkWebpage
  to key on the raw `ai_result` instead.
- **The NoCurbRamp Agree drop is per-label semantics, not miscalibration.** The old gate let through NoCurbRamp Agree
  verdicts whose pooled precision is 0.911. The new map keeps a subset at 0.951. The 16,433 labelled labels it drops
  (confidence mostly 0.90–0.999) are right 0.876 of the time individually: each of those votes is below 0.92 on its
  own, even though the pooled old set looked acceptable. On the website that is about 45k fewer NoCurbRamp Agree votes
  and as many more Unsure.

### Reliability (out-of-fold)

| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | mean diff |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | Agree | 13 | 4 | 6 | 0.063 | -0.016 |
| CurbRamp | Disagree | 13 | 1 | 5 | 0.064 | -0.034 |
| NoCurbRamp | Agree | 13 | 0 | 5 | 0.071 | -0.027 |
| NoCurbRamp | Disagree | 13 | 1 | 3 | 0.067 | -0.042 |
| Obstacle | Agree | 13 | 1 | 5 | 0.070 | -0.025 |
| Obstacle | Disagree | 13 | 2 | 7 | 0.086 | -0.032 |
| SurfaceProblem | Agree | 13 | 0 | 6 | 0.059 | -0.025 |
| SurfaceProblem | Disagree | 13 | 0 | 4 | 0.059 | -0.036 |
| Crosswalk | Agree | 8 | 4 | 1 | 0.045 | -0.017 |
| Crosswalk | Disagree | 3 | 0 | 0 | 0.106 | -0.092 |
| **all** | | 115 | 13 | 42 | 0.106 | -0.030 |

Served values sit on average 0.030 below the observed mean outcome per bin. About half of that is the deliberate
Wilson lowering, and the rest is the out-of-fold fits being noisier than the all-rows fit. So the ±0.01 criterion is
met on only 13 of 115 bins, and 42 are within 1.96 binomial SE. The misses are all in the conservative direction
except one: Obstacle Disagree at 0.8–0.85, where the served value is 0.002 above observed and far below the gate.

### City mix

The labelled sample is not the assessed population:

- **Over-represented:** Seattle (25% of labelled vs 19% of assessed), Chicago (17% vs 15%), Kaohsiung (15% vs 9%) and
  Taipei.
- **Under-represented:** spgg (2.8% vs 9.7%), cdmx (2.2% vs 7.2%) and St. Louis (1.1% vs 6.1%). Coverage numbers for
  these cities are extrapolated from the others.
- **Per city:** the out-of-fold precision of the new votes is ≥ 0.92 in all 28 cities with ≥ 500 labelled labels.
  The lowest are spgg at 0.920 and cdmx at 0.930.
- **Below 0.92:** the only city × type strata with n ≥ 300 are Seattle SurfaceProblem (0.904, n = 1,689) and spgg
  CurbRamp (0.914, n = 1,033).

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
