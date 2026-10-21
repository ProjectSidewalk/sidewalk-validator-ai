# Road-network prior for CurbRamp: first measurement (October 2026)

Issue #45 asks whether road-network context can tell driveway aprons mislabelled as CurbRamp from real curb ramps,
and whether it adds anything over the production confidence. This is the first measurement. It uses no GPU, no
imagery and no retraining. The code is `analysis/road_prior.py`; every number below comes from its `evaluate` and
`stratify` stages.

## Recommendation

**Do not add road context as a model feature. Keep it as a candidate routing signal for cast CurbRamp Disagree
votes, and decide after the hand-coding packet is coded.**

- **As a feature it adds nothing.** Distance to an intersection predicts the human verdict on its own (AUC 0.69),
  but the production confidence already carries it. Leave-one-city-out, adding the road features to the
  recalibrated confidence changes AUC by -0.0001 (streets) and +0.0019 (streets + OSM), with CIs that include 0.
- **On cast Disagree votes it separates, but does not yet transfer as a model.** At the same confidence, a cast
  Disagree near both an intersection and a crossing is confirmed by humans 10 to 27 points less often than one far
  from both, in three of the four confidence quartiles. Across the 24 public cities with such votes, a
  leave-one-city-out logistic model does not turn that into a significant gain.
- **The OSM driveway layer is too sparse to use.** Only 9.5% of CurbRamp labels have any mapped `service=driveway`
  way within 50 m, and the driveway features sit at AUC 0.50.

The routing use would be: a cast CurbRamp Disagree near both an intersection and a crossing is a weak Disagree, so
it should not hide a label on its own and should go to a human first. That is a rule on top of the routing in
PR #39, not a change to the classifier. It covers about 28% of cast CurbRamp Disagrees.

## Data

| | |
|---|---|
| Labels | 2026-10-17 private pull, public cities only in every table (41 public cities) |
| Streets | `/v3/api/streets?filetype=csv`, pulled 2026-10-20 |
| OSM | Overpass, `highway=crossing` nodes and `highway=service` ways, pulled 2026-10-20, all 41 cities ok |
| Production verdict | `labels_oof.csv` (recalibrated `oof_accuracy`, Phase 0) |
| Human verdict | strict majority of human votes, AI excluded, ties dropped |
| CurbRamp analysis set | 22,197 labels with >= 3 human votes, 34 cities, human-majority Disagree rate 8.6% |
| Crosswalk analysis set | 6,790 labels, 28 cities, Disagree rate 14.0% |

Intersection nodes are street-edge endpoints shared by two or more edges, because Project Sidewalk splits edges at
intersections. `d_junction3_m` uses three or more edges and behaves the same.

## Results

### Each feature alone (CurbRamp, pooled)

| Feature | AUC | 95% CI |
|---|---|---|
| Distance to nearest intersection node | 0.691 | 0.676–0.705 |
| Distance to nearest OSM crossing | 0.644 | 0.634–0.655 |
| Driveways within 20 m | 0.503 | 0.498–0.507 |
| Driveways within 50 m | 0.493 | 0.486–0.499 |
| Own street is a service road | 0.500 | flat: the streets API has almost no service edges |
| Own OSM way is a driveway | 0.500 | flat: labels sit on street ways, never on driveways |

The intersection AUC is 0.78 in Chicago and 0.80 in Seattle, and lower elsewhere. Walla Walla is the one city below
0.5. The relation is not monotone. The Disagree rate is 4–5% from 5 to 15 m, rises to 31–36% from 25 to 100 m,
and falls to 19% beyond 100 m.

![Disagree rate by distance to the nearest intersection](road-prior-2026-10/disagree_vs_intersection_pooled.png)

![Disagree rate by distance to the nearest OSM crossing](road-prior-2026-10/disagree_vs_crossing_pooled.png)

### Lift over the production confidence (leave-one-city-out)

Baseline is a logistic model on the logit of the production P(Agree). The full model adds the road features, with
distances as bins. Each city is predicted by a model fitted on the others. ΔAUC and Δlog-loss have paired bootstrap
95% CIs.

| Subset | Features | n | ΔAUC (CI) | Δlog-loss (CI) |
|---|---|---|---|---|
| All verdicts | streets | 21,840 | -0.0001 (-0.0013, 0.0010) | -0.0002 (-0.0010, 0.0006) |
| All verdicts | streets + OSM | 21,840 | +0.0019 (-0.0002, 0.0038) | -0.0019 (-0.0032, -0.0007) |
| AI Agree | streets + OSM | 18,087 | -0.0122 (-0.0284, 0.0061) | +0.0001 (-0.0009, 0.0011) |
| AI Disagree assessments | streets + OSM | 3,753 | +0.0093 (0.0034, 0.0154) | -0.0042 (-0.0104, 0.0012) |
| Cast AI Disagree votes | streets + OSM | 2,359 | +0.0060 (-0.0052, 0.0161) | +0.0031 (-0.0052, 0.0111) |

The one CI that clears zero on log-loss is all verdicts with OSM, and it is about 1% of the baseline log-loss. On
AI Agrees there is no gain, so this does not help catch the contested Agrees.

### Cast Disagree votes by road context

"Far from both" means at least 25 m from the nearest intersection node and the nearest OSM crossing. The rate is how
often the human majority confirms the Disagree. Quartiles are of `oof_accuracy` among cast Disagrees with >= 3 human
votes.

| oof_accuracy quartile | Far from both | Far from one | Near both |
|---|---|---|---|
| 0.23–0.35 | 0.40 (n 136) | 0.38 (335) | 0.24 (125) |
| 0.35–0.55 | 0.65 (177) | 0.46 (309) | 0.38 (116) |
| 0.55–0.67 | 0.74 (309) | 0.65 (321) | 0.63 (103) |
| 0.67–0.72 | 0.80 (233) | 0.74 (147) | 0.83 (48) |
| All | 0.68 (855) | 0.52 (1,112) | 0.46 (392) |

By city, near-both versus far-from-both is 0.33 vs 0.68 in Chicago, 0.51 vs 0.85 in Seattle, 0.51 vs 0.69 in SPGG
and 0.56 vs 0.56 in Amsterdam. The Wilson CIs are in `stratify.md`.

Of all 16,263 cast CurbRamp Disagrees in the public cities, 28% are near both, 38% far from one and 33% far from
both. The same pattern holds for all AI Disagree assessments, including the ones cast as Unsure: near-both is 0.27
and far-from-both 0.61.

### Crosswalk sanity check

Distance to an OSM crossing predicts Crosswalk Disagrees with AUC 0.71, the strongest single signal in the study,
as it should be. Distance to an intersection node is slightly reversed for Crosswalk (AUC 0.46), plausibly because
a crosswalk spans the roadway away from the centreline node. Leave-one-city-out on top of the confidence, the
streets + OSM features improve Crosswalk log-loss by 0.0058 (CI 0.0016–0.0100), about 5% of the baseline, with
no significant AUC change. Crosswalk is already at AUC 0.98, so this is not worth pursuing either.

## Caveats

- **Selection.** Only 15% of cast CurbRamp Disagrees have three or more human votes. The labels that collect votes
  are not a random sample, so the confirm rates describe the validated subset.
- **The human majority is the reference, not ground truth.** The hand-coding packet exists to check whether
  near-intersection Disagrees that humans overturn are real ramps, and whether far ones are driveways.
- **OSM coverage varies by city.** Crossing mapping is dense in some cities and sparse in others, which may explain
  why Amsterdam is flat. Driveway mapping is sparse almost everywhere.
- **The per-quartile pattern did not survive as a transferable model.** The binned logistic may be too coarse, or
  the effect may vary too much by city. Either way, a routing rule should be validated per city before it ships.

## Hand-coding packet

200 contested public CurbRamp labels from 36 cities: 115 AI Disagrees and 85 Agrees overturned by the crowd or an
expert, stratified by distance to the nearest intersection (0–15, 15–25, 25–40 and 40+ m). The packet holds only
the label, its share-page link and a Street View link. The key, the verdicts and the seed are kept outside the repo.

- **Code from the Street View link.** The share page shows earlier validations to admins and offers a vote, so it
  unblinds the item and adds a vote if used carelessly.
- **Codes:** `driveway`, `real_ramp`, `other`, `cant_tell`.
- Key sha256: `5f5a10f89d8a53f5e99489f8ae0f4f408e61e1f58238b7fcc57ae4fb01a79433`

## Reproduce

The commands are in the module docstring of `analysis/road_prior.py`. Inputs and the label-level features CSV
are private and stay outside the repo. The figures in `docs/road-prior-2026-10/` show public cities only.
