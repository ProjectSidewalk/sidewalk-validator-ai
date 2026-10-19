# Per-city calibration of the deployed validators (October 2026)

Does the 0.92 gate mean the same thing in every city? This note answers that for the models in production today,
from the same production pull as [`calibration-2026-10.md`](calibration-2026-10.md) (2026-10-17, labels with at least
two human votes, outcome `tie_half`). It is the production half of
[#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15); the held-out-city benchmark for the
retrained models is the README's [Held-out-city benchmark](../README.md#held-out-city-benchmark) recipe and has not
been run yet.

**Reproduce.** Both sets of tables and figures below are the unedited output of `calibration/per_city.py` on the
`labels_oof.csv` that the calibration note's `calibrate_from_production` command wrote:

```
python -m calibration.per_city --labels ~/Data/<pull>/phase0/labels_oof.csv --cities-json ~/Data/<pull>/cities.json --served-column oof_accuracy --out-dir ~/Data/<pull>/per_city_oof
python -m calibration.per_city --labels ~/Data/<pull>/phase0/labels_oof.csv --cities-json ~/Data/<pull>/cities.json --out-dir ~/Data/<pull>/per_city_served
```

`cities.json` is the `/v3/api/cities` response saved with the pull. Figures are copied from `per_city_<type>.png` as
`per-city-calibration-2026-10/current_<type>.png` (first command) and `served_<type>.png` (second).

## Method

- **Rows.** Every AI-assessed label with at least two human Agree/Disagree votes (273,169 labels from 49 of the 52
  assessed schemas; the internal test schema dropped), with the AI's own vote excluded from the human count.
- **Gated vote.** A row whose served accuracy is at least 0.92, SidewalkWebpage's `ai-validation-min-accuracy`
  (`GATE` in `calibration/_isotonic_map.py`). Two served accuracies are measured:
  - **Current maps:** `oof_accuracy`, the out-of-fold value of the Phase 0 maps live since sidewalk-ai-api
    [PR #16](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/16) (API 1.0.5). Each label is scored by a
    map fitted without its fold, so this is an honest estimate of what the live maps do in each city.
  - **As served:** `ai_accuracy`, what production served when the vote was cast, under the maps before PR #16.
- **Per city × label type × verdict:** `n` gated votes, their mean served accuracy, and their **observed
  precision**: the mean outcome (1 if the strict human majority matches the verdict, 0.5 for a tie, 0 otherwise),
  with a Wilson 95% interval. **Calibration error** is served minus observed, pooled over both verdicts' gated
  votes in that city and type (positive: the map over-promised there).
- **Threshold reaching the gate.** For an Agree stratum below 0.92, the lowest served-accuracy threshold at which
  that city's votes of that type reach 0.92 precision on at least 50 votes, and the share of its gated votes that
  threshold keeps. It is chosen on the rows it is measured on, so it is optimistic.
- **Naming.** A city is named by its `city_id` only if `cities.json` lists it as public. The 14 private
  deployments (counted over all 52 assessed schemas) are pooled into one row, and only that pooled row is reported
  for them. Strata with fewer than 50 gated votes are shown as —, and cities with no such stratum are counted only
  in **all cities**.

## The live question is Agree votes

The Phase 0 maps hold back every Disagree class: its out-of-fold votes either never clear 0.92 or fail the Wilson
lower-bound rule, so its map is capped at 0.9199 and **no Disagree verdict becomes a vote anywhere**. Whether the
gate should vary by city is therefore a question about Agree votes today, and about the retrained models later.
The Disagree columns of the current-maps tables are empty for that reason (the dozen out-of-fold Disagree rows in
NoCurbRamp are a cross-fitting artefact; the shipped map is capped).

## Results under the current maps (Agree votes)

- **83 Agree city × type strata** have at least 50 gated votes. **77 are at or above 0.92**. The 6 below hold
  4,816 of the 126,367 gated Agree votes in those strata (3.8%), and 4 of them have their whole 95% interval below
  the gate.
- Among the 41 strata with at least 300 votes, served minus observed runs from −0.039 to +0.086, and only 3 are
  below the gate.
- Per type, the all-cities row is calibrated to within 0.01 for every type (the out-of-fold construction makes
  that close to automatic); the per-city spread is what the maps cannot see.

**Agree strata below the gate (current maps):**

| city | type | n | served | observed (95% CI) | below the gate by | whole CI below | threshold reaching the gate (votes kept) |
|---|---|---:|---:|---:|---:|---|---:|
| tucson-az | SurfaceProblem | 57 | 0.936 | 0.816 (0.696–0.895) | 0.104 | yes | none with n >= 50 |
| cdmx | NoCurbRamp | 397 | 0.937 | 0.851 (0.813–0.883) | 0.069 | yes | none with n >= 50 |
| paterson-nj | SurfaceProblem | 65 | 0.940 | 0.877 (0.775–0.936) | 0.043 | no | none with n >= 50 |
| seattle-wa | SurfaceProblem | 2,766 | 0.938 | 0.894 (0.882–0.905) | 0.026 | yes | none with n >= 50 |
| spgg | CurbRamp | 1,395 | 0.953 | 0.902 (0.885–0.916) | 0.018 | yes | 0.9726 (14%) |
| spgg | NoCurbRamp | 136 | 0.937 | 0.908 (0.848–0.946) | 0.012 | no | 0.9354 (54%) |

- **Seattle SurfaceProblem** (n = 2,766): 0.894, about 2.6 points short, with a tight interval. Precision stays
  below 0.92 at every served threshold that keeps at least 50 votes, so a stricter gate would not fix it.
- **cdmx NoCurbRamp** (n = 397): 0.851, 6.9 points short, also with no threshold that reaches the gate.
- **spgg CurbRamp** (n = 1,395): 0.902, 1.8 points short. A threshold of 0.9726 reaches 0.92, but it keeps 14% of
  spgg's CurbRamp votes.
- **tucson-az SurfaceProblem** (n = 57) and **paterson-nj SurfaceProblem** (n = 65) are small strata; the
  paterson-nj interval reaches the gate. **spgg NoCurbRamp** (n = 136) is 1.2 points short, and its interval also
  reaches the gate.
- **Seattle Obstacle** is exactly at the gate (0.920, n = 1,004; its interval spans 0.902–0.936).

These are the same three city × type strata the calibration note flagged (cdmx NoCurbRamp, Seattle SurfaceProblem,
spgg CurbRamp), now with intervals and the full table.

### Per-city tables, current maps

#### CurbRamp

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chicago-il | 18,736 | 0.969 | 0.968 (0.966–0.971) | 0 | — | — | +0.000 |
| seattle-wa | 12,560 | 0.965 | 0.950 (0.946–0.953) | 0 | — | — | +0.015 |
| teaneck-nj | 3,826 | 0.970 | 0.985 (0.981–0.989) | 0 | — | — | -0.016 |
| columbus-oh | 2,257 | 0.966 | 0.959 (0.950–0.966) | 0 | — | — | +0.007 |
| spgg | 1,395 | 0.953 | 0.902 (0.885–0.916) | 0 | — | — | +0.051 |
| oradell-nj | 1,282 | 0.970 | 0.988 (0.981–0.993) | 0 | — | — | -0.018 |
| newberg-or | 1,275 | 0.969 | 0.987 (0.980–0.992) | 0 | — | — | -0.019 |
| pittsburgh-pa | 1,256 | 0.962 | 0.966 (0.954–0.974) | 0 | — | — | -0.003 |
| mendota-il | 1,188 | 0.967 | 0.983 (0.974–0.989) | 0 | — | — | -0.016 |
| amsterdam | 1,100 | 0.951 | 0.958 (0.944–0.968) | 0 | — | — | -0.007 |
| cdmx | 1,086 | 0.957 | 0.936 (0.920–0.949) | 0 | — | — | +0.021 |
| walla-walla-wa | 946 | 0.969 | 0.987 (0.978–0.993) | 0 | — | — | -0.018 |
| st-louis-mo | 540 | 0.963 | 0.964 (0.945–0.977) | 0 | — | — | -0.001 |
| knox-oh | 399 | 0.965 | 0.982 (0.964–0.991) | 0 | — | — | -0.018 |
| tucson-az | 397 | 0.959 | 0.977 (0.957–0.988) | 0 | — | — | -0.018 |
| burnaby | 372 | 0.966 | 0.997 (0.985–1.000) | 0 | — | — | -0.032 |
| maywood-nj | 276 | 0.972 | 0.995 (0.977–0.999) | 0 | — | — | -0.023 |
| detroit-mi | 205 | 0.965 | 0.968 (0.934–0.985) | 0 | — | — | -0.004 |
| cliffside-park-nj | 164 | 0.968 | 0.963 (0.922–0.983) | 0 | — | — | +0.005 |
| paterson-nj | 148 | 0.970 | 0.966 (0.923–0.985) | 0 | — | — | +0.004 |
| madison-wi | 79 | 0.967 | 0.975 (0.912–0.993) | 0 | — | — | -0.008 |
| waltham-ma | 65 | 0.969 | 1.000 (0.944–1.000) | 0 | — | — | -0.031 |
| clifton-nj | 56 | 0.967 | 0.938 (0.842–0.977) | 0 | — | — | +0.030 |
| niagara-falls-ny | 51 | 0.963 | 0.941 (0.841–0.980) | 0 | — | — | +0.021 |
| other (14 private deployments) | 15,830 | 0.961 | 0.973 (0.970–0.975) | 0 | — | — | -0.012 |
| **all cities** | 65,698 | 0.965 | 0.966 (0.965–0.967) | 0 | — | — | -0.001 |

12 more cities have fewer than 50 gated votes of either verdict (209 votes); they are counted in **all cities**.

#### NoCurbRamp

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 13,164 | 0.948 | 0.947 (0.943–0.951) | 2 | — | — | +0.001 |
| newberg-or | 781 | 0.946 | 0.983 (0.971–0.990) | 0 | — | — | -0.037 |
| teaneck-nj | 465 | 0.948 | 0.984 (0.968–0.992) | 0 | — | — | -0.036 |
| cdmx | 397 | 0.937 | 0.851 (0.813–0.883) | 0 | — | — | +0.086 |
| st-louis-mo | 180 | 0.944 | 0.944 (0.901–0.970) | 0 | — | — | -0.000 |
| spgg | 136 | 0.937 | 0.908 (0.848–0.946) | 0 | — | — | +0.029 |
| walla-walla-wa | 136 | 0.943 | 1.000 (0.973–1.000) | 0 | — | — | -0.057 |
| columbus-oh | 124 | 0.940 | 0.931 (0.873–0.964) | 0 | — | — | +0.009 |
| pittsburgh-pa | 105 | 0.942 | 0.981 (0.933–0.995) | 0 | — | — | -0.039 |
| chicago-il | 100 | 0.937 | 0.940 (0.875–0.972) | 0 | — | — | -0.003 |
| oradell-nj | 89 | 0.940 | 0.927 (0.854–0.965) | 0 | — | — | +0.013 |
| burnaby | 75 | 0.944 | 0.993 (0.939–0.999) | 0 | — | — | -0.050 |
| mendota-il | 54 | 0.938 | 0.981 (0.902–0.997) | 0 | — | — | -0.044 |
| other (14 private deployments) | 663 | 0.940 | 0.960 (0.942–0.972) | 10 | — | — | -0.019 |
| **all cities** | 16,576 | 0.947 | 0.948 (0.945–0.952) | 12 | — | — | -0.001 |

14 more cities have fewer than 50 gated votes of either verdict (107 votes); they are counted in **all cities**.

#### Obstacle

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 1,004 | 0.933 | 0.920 (0.902–0.936) | 0 | — | — | +0.012 |
| chicago-il | 899 | 0.934 | 0.963 (0.949–0.974) | 0 | — | — | -0.030 |
| newberg-or | 103 | 0.935 | 0.966 (0.911–0.987) | 0 | — | — | -0.031 |
| st-louis-mo | 63 | 0.931 | 0.992 (0.928–0.999) | 0 | — | — | -0.061 |
| columbus-oh | 53 | 0.932 | 0.925 (0.821–0.970) | 0 | — | — | +0.008 |
| other (14 private deployments) | 385 | 0.933 | 0.953 (0.927–0.970) | 0 | — | — | -0.021 |
| **all cities** | 2,834 | 0.933 | 0.942 (0.933–0.950) | 0 | — | — | -0.009 |

20 more cities have fewer than 50 gated votes of either verdict (327 votes); they are counted in **all cities**.

#### SurfaceProblem

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 2,766 | 0.938 | 0.894 (0.882–0.905) | 0 | — | — | +0.044 |
| chicago-il | 2,147 | 0.939 | 0.942 (0.931–0.951) | 0 | — | — | -0.003 |
| oradell-nj | 1,739 | 0.941 | 0.976 (0.968–0.982) | 0 | — | — | -0.034 |
| teaneck-nj | 659 | 0.940 | 0.956 (0.938–0.969) | 0 | — | — | -0.016 |
| columbus-oh | 573 | 0.939 | 0.959 (0.939–0.972) | 0 | — | — | -0.020 |
| newberg-or | 497 | 0.940 | 0.953 (0.930–0.968) | 0 | — | — | -0.013 |
| pittsburgh-pa | 470 | 0.939 | 0.956 (0.934–0.971) | 0 | — | — | -0.017 |
| mendota-il | 389 | 0.941 | 0.981 (0.962–0.990) | 0 | — | — | -0.039 |
| st-louis-mo | 281 | 0.940 | 0.943 (0.910–0.965) | 0 | — | — | -0.003 |
| spgg | 226 | 0.937 | 0.938 (0.899–0.963) | 0 | — | — | -0.001 |
| cdmx | 207 | 0.939 | 0.983 (0.955–0.994) | 0 | — | — | -0.045 |
| knox-oh | 133 | 0.941 | 0.940 (0.886–0.969) | 0 | — | — | +0.001 |
| maywood-nj | 113 | 0.941 | 0.960 (0.906–0.984) | 0 | — | — | -0.019 |
| walla-walla-wa | 99 | 0.939 | 0.955 (0.894–0.981) | 0 | — | — | -0.016 |
| paterson-nj | 65 | 0.940 | 0.877 (0.775–0.936) | 0 | — | — | +0.063 |
| tucson-az | 57 | 0.936 | 0.816 (0.696–0.895) | 0 | — | — | +0.120 |
| waltham-ma | 57 | 0.937 | 0.982 (0.907–0.997) | 0 | — | — | -0.045 |
| danville-il | 54 | 0.939 | 0.972 (0.888–0.994) | 0 | — | — | -0.033 |
| burnaby | 50 | 0.939 | 0.960 (0.865–0.989) | 0 | — | — | -0.021 |
| other (14 private deployments) | 1,010 | 0.938 | 0.951 (0.936–0.963) | 0 | — | — | -0.013 |
| **all cities** | 11,780 | 0.939 | 0.941 (0.937–0.945) | 0 | — | — | -0.002 |

14 more cities have fewer than 50 gated votes of either verdict (188 votes); they are counted in **all cities**.

#### Crosswalk

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chicago-il | 3,717 | 0.985 | 0.978 (0.972–0.982) | 0 | — | — | +0.008 |
| teaneck-nj | 1,513 | 0.986 | 0.995 (0.990–0.998) | 0 | — | — | -0.010 |
| amsterdam | 857 | 0.983 | 0.987 (0.976–0.992) | 0 | — | — | -0.004 |
| oradell-nj | 479 | 0.988 | 0.982 (0.966–0.991) | 0 | — | — | +0.005 |
| mendota-il | 453 | 0.985 | 0.990 (0.976–0.996) | 0 | — | — | -0.005 |
| seattle-wa | 437 | 0.987 | 0.974 (0.954–0.985) | 0 | — | — | +0.013 |
| burnaby | 258 | 0.985 | 0.996 (0.978–0.999) | 0 | — | — | -0.011 |
| walla-walla-wa | 233 | 0.986 | 0.985 (0.960–0.994) | 0 | — | — | +0.001 |
| pittsburgh-pa | 202 | 0.987 | 0.980 (0.950–0.992) | 0 | — | — | +0.007 |
| st-louis-mo | 169 | 0.983 | 0.970 (0.933–0.987) | 0 | — | — | +0.013 |
| tucson-az | 148 | 0.984 | 1.000 (0.975–1.000) | 0 | — | — | -0.016 |
| columbus-oh | 141 | 0.986 | 0.989 (0.955–0.998) | 0 | — | — | -0.004 |
| maywood-nj | 123 | 0.986 | 1.000 (0.970–1.000) | 0 | — | — | -0.014 |
| knox-oh | 78 | 0.984 | 0.968 (0.902–0.990) | 0 | — | — | +0.016 |
| cdmx | 75 | 0.979 | 0.940 (0.862–0.975) | 0 | — | — | +0.039 |
| cliffside-park-nj | 74 | 0.988 | 0.946 (0.869–0.979) | 0 | — | — | +0.042 |
| waltham-ma | 59 | 0.989 | 1.000 (0.939–1.000) | 0 | — | — | -0.011 |
| other (14 private deployments) | 21,294 | 0.986 | 0.988 (0.986–0.989) | 0 | — | — | -0.002 |
| **all cities** | 30,548 | 0.986 | 0.986 (0.985–0.988) | 0 | — | — | -0.001 |

17 more cities have fewer than 50 gated votes of either verdict (238 votes); they are counted in **all cities**.

### Figures, current maps

Each row is one city: observed precision of its gated votes with the Wilson 95% interval (blue), and the mean
value the map served them (green). A city to the left of the dashed gate line voted below 0.92 precision.

![CurbRamp, current maps](per-city-calibration-2026-10/current_curbramp.png)
![NoCurbRamp, current maps](per-city-calibration-2026-10/current_nocurbramp.png)
![Obstacle, current maps](per-city-calibration-2026-10/current_obstacle.png)
![SurfaceProblem, current maps](per-city-calibration-2026-10/current_surfaceproblem.png)
![Crosswalk, current maps](per-city-calibration-2026-10/current_crosswalk.png)

## Results as served (the maps before PR #16)

This is what the platform recorded before the Phase 0 maps: both verdicts voted, and Obstacle never voted Agree.

- **Agree:** 17 of 82 strata with at least 50 votes were below 0.92 (21,749 of 141,384 votes; 13 with the whole
  interval below). Most are NoCurbRamp, the type the Phase 0 maps then cut to 43% coverage.
- **Disagree:** gated precision ranged from **0.147 to 0.940** across the 52 strata with at least 50 votes, and only
  2 of them reached the gate. The variation between cities is far larger for Disagree than for Agree, which matters
  if a retrained model's Disagree class is ever allowed to vote again.

**Agree strata below the gate (as served):**

| city | type | n | served | observed (95% CI) | below the gate by | whole CI below | threshold reaching the gate (votes kept) |
|---|---|---:|---:|---:|---:|---|---:|
| paterson-nj | NoCurbRamp | 51 | 0.953 | 0.667 (0.530–0.780) | 0.253 | yes | none with n >= 50 |
| tucson-az | SurfaceProblem | 177 | 0.941 | 0.698 (0.626–0.761) | 0.222 | yes | none with n >= 50 |
| chicago-il | NoCurbRamp | 769 | 0.956 | 0.773 (0.742–0.801) | 0.147 | yes | 0.9748 (26%) |
| amsterdam | NoCurbRamp | 202 | 0.952 | 0.792 (0.731–0.842) | 0.128 | yes | 0.9570 (36%) |
| paterson-nj | SurfaceProblem | 101 | 0.949 | 0.832 (0.747–0.892) | 0.088 | yes | none with n >= 50 |
| oradell-nj | NoCurbRamp | 297 | 0.964 | 0.845 (0.800–0.882) | 0.075 | yes | 0.9767 (38%) |
| cdmx | NoCurbRamp | 1,645 | 0.965 | 0.862 (0.844–0.878) | 0.058 | yes | none with n >= 50 |
| seattle-wa | SurfaceProblem | 6,842 | 0.944 | 0.870 (0.862–0.878) | 0.050 | yes | none with n >= 50 |
| st-louis-mo | NoCurbRamp | 412 | 0.970 | 0.876 (0.841–0.905) | 0.044 | yes | 0.9531 (74%) |
| spgg | NoCurbRamp | 884 | 0.961 | 0.879 (0.856–0.899) | 0.041 | yes | 0.9828 (14%) |
| columbus-oh | NoCurbRamp | 589 | 0.960 | 0.892 (0.865–0.915) | 0.028 | yes | 0.9534 (55%) |
| mendota-il | NoCurbRamp | 254 | 0.963 | 0.894 (0.850–0.926) | 0.026 | no | 0.9356 (85%) |
| detroit-mi | SurfaceProblem | 60 | 0.946 | 0.900 (0.799–0.953) | 0.020 | no | 0.9248 (97%) |
| other (14 private deployments) | NoCurbRamp | 3,078 | 0.961 | 0.902 (0.891–0.912) | 0.018 | yes | 0.9479 (77%) |
| pittsburgh-pa | NoCurbRamp | 469 | 0.962 | 0.902 (0.872–0.926) | 0.018 | no | 0.9413 (81%) |
| spgg | CurbRamp | 1,243 | 0.950 | 0.906 (0.889–0.921) | 0.014 | no | 0.9558 (36%) |
| chicago-il | SurfaceProblem | 4,676 | 0.945 | 0.911 (0.902–0.918) | 0.009 | yes | 0.9309 (85%) |

### Per-city tables, as served

#### CurbRamp

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chicago-il | 18,268 | 0.947 | 0.970 (0.967–0.972) | 1,147 | 0.952 | 0.466 (0.438–0.495) | +0.007 |
| seattle-wa | 12,193 | 0.948 | 0.951 (0.947–0.955) | 2,527 | 0.955 | 0.599 (0.580–0.618) | +0.059 |
| teaneck-nj | 3,767 | 0.948 | 0.986 (0.982–0.989) | 77 | 0.953 | 0.591 (0.479–0.694) | -0.030 |
| spgg | 1,243 | 0.950 | 0.906 (0.889–0.921) | 1,715 | 0.955 | 0.559 (0.536–0.583) | +0.248 |
| columbus-oh | 2,192 | 0.949 | 0.960 (0.951–0.967) | 114 | 0.949 | 0.303 (0.226–0.392) | +0.021 |
| cdmx | 1,008 | 0.949 | 0.938 (0.922–0.952) | 369 | 0.951 | 0.354 (0.307–0.404) | +0.168 |
| amsterdam | 983 | 0.950 | 0.962 (0.948–0.972) | 388 | 0.952 | 0.563 (0.513–0.612) | +0.102 |
| oradell-nj | 1,253 | 0.946 | 0.991 (0.984–0.995) | 59 | 0.954 | 0.712 (0.586–0.812) | -0.032 |
| pittsburgh-pa | 1,192 | 0.948 | 0.965 (0.953–0.974) | 110 | 0.950 | 0.373 (0.288–0.466) | +0.034 |
| newberg-or | 1,243 | 0.946 | 0.988 (0.980–0.992) | 51 | 0.951 | 0.627 (0.490–0.747) | -0.027 |
| mendota-il | 1,160 | 0.947 | 0.985 (0.976–0.990) | 104 | 0.953 | 0.587 (0.490–0.676) | -0.004 |
| walla-walla-wa | 916 | 0.946 | 0.988 (0.979–0.993) | 29 | — | — | -0.017 |
| st-louis-mo | 519 | 0.949 | 0.963 (0.944–0.976) | 44 | — | — | +0.052 |
| knox-oh | 383 | 0.947 | 0.986 (0.968–0.994) | 48 | — | — | +0.018 |
| tucson-az | 370 | 0.949 | 0.976 (0.954–0.987) | 58 | 0.953 | 0.491 (0.367–0.617) | +0.040 |
| burnaby | 362 | 0.949 | 0.997 (0.985–1.000) | 18 | — | — | -0.008 |
| maywood-nj | 272 | 0.945 | 0.994 (0.976–0.999) | 6 | — | — | -0.028 |
| detroit-mi | 193 | 0.948 | 0.979 (0.948–0.992) | 6 | — | — | -0.006 |
| cliffside-park-nj | 160 | 0.948 | 0.963 (0.921–0.983) | 9 | — | — | -0.013 |
| paterson-nj | 146 | 0.947 | 0.969 (0.927–0.987) | 2 | — | — | -0.016 |
| madison-wi | 75 | 0.949 | 0.987 (0.928–0.998) | 6 | — | — | +0.023 |
| waltham-ma | 63 | 0.946 | 1.000 (0.943–1.000) | 7 | — | — | +0.032 |
| clifton-nj | 55 | 0.950 | 0.936 (0.839–0.976) | 1 | — | — | +0.030 |
| other (14 private deployments) | 14,849 | 0.948 | 0.975 (0.972–0.977) | 2,944 | 0.952 | 0.446 (0.428–0.464) | +0.061 |
| **all cities** | 63,111 | 0.948 | 0.968 (0.966–0.969) | 9,883 | 0.953 | 0.508 (0.499–0.518) | +0.043 |

13 more cities have fewer than 50 gated votes of either verdict (290 votes); they are counted in **all cities**.

#### NoCurbRamp

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 18,936 | 0.980 | 0.922 (0.918–0.926) | 1,857 | 0.956 | 0.605 (0.583–0.627) | +0.084 |
| chicago-il | 769 | 0.956 | 0.773 (0.742–0.801) | 3,119 | 0.966 | 0.841 (0.827–0.853) | +0.137 |
| cdmx | 1,645 | 0.965 | 0.862 (0.844–0.878) | 176 | 0.949 | 0.372 (0.304–0.446) | +0.149 |
| newberg-or | 1,353 | 0.976 | 0.955 (0.943–0.965) | 230 | 0.958 | 0.602 (0.538–0.663) | +0.070 |
| teaneck-nj | 814 | 0.976 | 0.952 (0.935–0.965) | 273 | 0.960 | 0.703 (0.647–0.754) | +0.083 |
| spgg | 884 | 0.961 | 0.879 (0.856–0.899) | 169 | 0.948 | 0.287 (0.224–0.359) | +0.175 |
| columbus-oh | 589 | 0.960 | 0.892 (0.865–0.915) | 384 | 0.954 | 0.570 (0.520–0.619) | +0.193 |
| mendota-il | 254 | 0.963 | 0.894 (0.850–0.926) | 665 | 0.972 | 0.898 (0.873–0.919) | +0.072 |
| amsterdam | 202 | 0.952 | 0.792 (0.731–0.842) | 607 | 0.960 | 0.698 (0.660–0.733) | +0.237 |
| pittsburgh-pa | 469 | 0.962 | 0.902 (0.872–0.926) | 319 | 0.955 | 0.542 (0.487–0.596) | +0.202 |
| st-louis-mo | 412 | 0.970 | 0.876 (0.841–0.905) | 166 | 0.949 | 0.190 (0.137–0.256) | +0.285 |
| oradell-nj | 297 | 0.964 | 0.845 (0.800–0.882) | 185 | 0.956 | 0.505 (0.434–0.577) | +0.246 |
| walla-walla-wa | 364 | 0.969 | 0.979 (0.959–0.990) | 116 | 0.961 | 0.509 (0.419–0.598) | +0.102 |
| burnaby | 159 | 0.972 | 0.950 (0.904–0.974) | 45 | — | — | +0.098 |
| knox-oh | 93 | 0.961 | 0.973 (0.917–0.992) | 26 | — | — | +0.101 |
| paterson-nj | 51 | 0.953 | 0.667 (0.530–0.780) | 57 | 0.953 | 0.772 (0.648–0.862) | +0.231 |
| other (14 private deployments) | 3,078 | 0.961 | 0.902 (0.891–0.912) | 1,905 | 0.965 | 0.689 (0.668–0.709) | +0.142 |
| **all cities** | 30,663 | 0.974 | 0.911 (0.908–0.914) | 10,485 | 0.962 | 0.693 (0.685–0.702) | +0.115 |

17 more cities have fewer than 50 gated votes of either verdict (480 votes); they are counted in **all cities**.

#### Obstacle

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 0 | — | — | 749 | 0.981 | 0.755 (0.723–0.784) | +0.226 |
| chicago-il | 0 | — | — | 504 | 0.985 | 0.846 (0.812–0.875) | +0.139 |
| amsterdam | 0 | — | — | 200 | 0.985 | 0.720 (0.654–0.778) | +0.265 |
| mendota-il | 0 | — | — | 111 | 0.990 | 0.928 (0.864–0.963) | +0.062 |
| columbus-oh | 0 | — | — | 98 | 0.987 | 0.740 (0.645–0.816) | +0.247 |
| pittsburgh-pa | 0 | — | — | 62 | 0.983 | 0.669 (0.545–0.774) | +0.314 |
| other (14 private deployments) | 0 | — | — | 1,909 | 0.982 | 0.543 (0.520–0.565) | +0.439 |
| **all cities** | 0 | — | — | 3,943 | 0.983 | 0.654 (0.639–0.668) | +0.329 |

21 more cities have fewer than 50 gated votes of either verdict (310 votes); they are counted in **all cities**.

#### SurfaceProblem

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| seattle-wa | 6,842 | 0.944 | 0.870 (0.862–0.878) | 2,041 | 0.961 | 0.570 (0.548–0.591) | +0.147 |
| chicago-il | 4,676 | 0.945 | 0.911 (0.902–0.918) | 821 | 0.961 | 0.617 (0.583–0.650) | +0.081 |
| oradell-nj | 2,542 | 0.950 | 0.969 (0.962–0.975) | 12 | — | — | -0.019 |
| teaneck-nj | 1,141 | 0.948 | 0.948 (0.934–0.960) | 44 | — | — | +0.005 |
| columbus-oh | 1,114 | 0.946 | 0.951 (0.937–0.962) | 37 | — | — | +0.007 |
| newberg-or | 960 | 0.946 | 0.927 (0.908–0.941) | 65 | 0.958 | 0.746 (0.629–0.836) | +0.032 |
| pittsburgh-pa | 953 | 0.946 | 0.938 (0.921–0.952) | 56 | 0.958 | 0.732 (0.604–0.830) | +0.020 |
| spgg | 656 | 0.942 | 0.928 (0.905–0.945) | 39 | — | — | +0.047 |
| mendota-il | 566 | 0.950 | 0.978 (0.962–0.987) | 21 | — | — | -0.023 |
| cdmx | 533 | 0.943 | 0.961 (0.941–0.974) | 24 | — | — | +0.007 |
| amsterdam | 63 | 0.937 | 0.952 (0.869–0.984) | 470 | 0.962 | 0.803 (0.765–0.837) | +0.138 |
| st-louis-mo | 500 | 0.947 | 0.939 (0.914–0.957) | 19 | — | — | +0.033 |
| walla-walla-wa | 269 | 0.943 | 0.952 (0.919–0.972) | 54 | 0.955 | 0.306 (0.199–0.438) | +0.101 |
| knox-oh | 260 | 0.946 | 0.946 (0.912–0.968) | 4 | — | — | +0.003 |
| tucson-az | 177 | 0.941 | 0.698 (0.626–0.761) | 16 | — | — | +0.245 |
| maywood-nj | 174 | 0.949 | 0.948 (0.905–0.973) | 1 | — | — | +0.007 |
| waltham-ma | 121 | 0.944 | 0.975 (0.930–0.992) | 6 | — | — | +0.003 |
| burnaby | 101 | 0.945 | 0.921 (0.851–0.959) | 15 | — | — | +0.089 |
| danville-il | 109 | 0.945 | 0.968 (0.916–0.988) | 2 | — | — | -0.014 |
| paterson-nj | 101 | 0.949 | 0.832 (0.747–0.892) | 0 | — | — | +0.117 |
| virden-il | 76 | 0.946 | 0.934 (0.855–0.972) | 2 | — | — | +0.011 |
| madison-wi | 59 | 0.943 | 0.975 (0.897–0.994) | 6 | — | — | +0.046 |
| detroit-mi | 60 | 0.946 | 0.900 (0.799–0.953) | 2 | — | — | +0.051 |
| other (14 private deployments) | 3,193 | 0.941 | 0.925 (0.915–0.933) | 3,465 | 0.967 | 0.637 (0.621–0.653) | +0.179 |
| **all cities** | 25,432 | 0.945 | 0.916 (0.912–0.919) | 7,242 | 0.964 | 0.624 (0.612–0.635) | +0.098 |

11 more cities have fewer than 50 gated votes of either verdict (206 votes); they are counted in **all cities**.

#### Crosswalk

| city | Agree n | Agree served | Agree observed (95% CI) | Disagree n | Disagree served | Disagree observed (95% CI) | calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chicago-il | 2,818 | 0.957 | 0.981 (0.975–0.985) | 547 | 0.979 | 0.410 (0.370–0.452) | +0.072 |
| teaneck-nj | 1,123 | 0.959 | 0.998 (0.994–1.000) | 242 | 0.980 | 0.620 (0.557–0.679) | +0.032 |
| amsterdam | 580 | 0.952 | 0.986 (0.973–0.993) | 166 | 0.978 | 0.732 (0.660–0.794) | +0.028 |
| mendota-il | 333 | 0.956 | 0.994 (0.978–0.998) | 298 | 0.979 | 0.940 (0.907–0.961) | -0.001 |
| seattle-wa | 344 | 0.955 | 0.980 (0.959–0.990) | 143 | 0.979 | 0.843 (0.774–0.893) | +0.023 |
| oradell-nj | 370 | 0.962 | 0.986 (0.969–0.994) | 48 | — | — | +0.011 |
| walla-walla-wa | 183 | 0.956 | 0.989 (0.961–0.997) | 27 | — | — | +0.002 |
| pittsburgh-pa | 155 | 0.964 | 0.990 (0.959–0.998) | 54 | 0.977 | 0.630 (0.496–0.746) | +0.070 |
| burnaby | 184 | 0.964 | 1.000 (0.980–1.000) | 18 | — | — | +0.034 |
| cdmx | 43 | — | — | 140 | 0.980 | 0.489 (0.408–0.571) | +0.368 |
| st-louis-mo | 117 | 0.962 | 0.970 (0.921–0.989) | 58 | 0.978 | 0.147 (0.078–0.259) | +0.270 |
| columbus-oh | 104 | 0.959 | 0.990 (0.948–0.998) | 46 | — | — | +0.209 |
| tucson-az | 95 | 0.961 | 1.000 (0.961–1.000) | 47 | — | — | +0.216 |
| maywood-nj | 91 | 0.959 | 1.000 (0.959–1.000) | 5 | — | — | -0.009 |
| knox-oh | 55 | 0.960 | 0.982 (0.904–0.997) | 19 | — | — | +0.053 |
| spgg | 11 | — | — | 58 | 0.978 | 0.845 (0.731–0.916) | +0.101 |
| cliffside-park-nj | 60 | 0.968 | 0.958 (0.875–0.987) | 4 | — | — | +0.039 |
| waltham-ma | 51 | 0.962 | 1.000 (0.930–1.000) | 3 | — | — | -0.027 |
| other (14 private deployments) | 16,241 | 0.956 | 0.990 (0.989–0.992) | 692 | 0.975 | 0.267 (0.235–0.301) | -0.004 |
| **all cities** | 23,104 | 0.957 | 0.989 (0.988–0.990) | 2,684 | 0.978 | 0.510 (0.491–0.529) | +0.020 |

16 more cities have fewer than 50 gated votes of either verdict (215 votes); they are counted in **all cities**.

### Figures, as served

![CurbRamp, as served](per-city-calibration-2026-10/served_curbramp.png)
![NoCurbRamp, as served](per-city-calibration-2026-10/served_nocurbramp.png)
![Obstacle, as served](per-city-calibration-2026-10/served_obstacle.png)
![SurfaceProblem, as served](per-city-calibration-2026-10/served_surfaceproblem.png)
![Crosswalk, as served](per-city-calibration-2026-10/served_crosswalk.png)

## Recommendation: keep `ai-validation-min-accuracy` at 0.92 everywhere

**Do not make the gate a per-city setting now.**

1. **It is already met almost everywhere.** Under the live maps, 77 of 83 Agree strata are at or above 0.92, and
   the 6 that are not hold 3.8% of gated Agree votes.
2. **The failures are per city *and type*, and a per-city scalar cannot target them.** Seattle's
   SurfaceProblem votes fall short, but its CurbRamp, NoCurbRamp and Crosswalk votes clear the gate. Raising
   Seattle's gate would throw away good votes of those types to fix one type.
3. **Raising the gate does not fix the strata that fail.** For Seattle SurfaceProblem and cdmx NoCurbRamp, no
   threshold reaches 0.92. For spgg CurbRamp the threshold that does keeps 14% of its votes. The problem is how the
   model does in those cities, not where the gate sits.
4. **The evidence per stratum is thin.** About 80 strata are tested here, so a few 95% intervals will miss the gate
   by chance, and the human majority is itself noisy (expert-vs-crowd agreement 86–91%).

What to do instead:

- **If the three strata with a tight interval below the gate need action before a retrain** (Seattle
  SurfaceProblem, cdmx NoCurbRamp, spgg CurbRamp), the right tool is a per city × type hold-back of Agree votes.
  SidewalkWebpage would need a new setting for that, since today the gate is one number. Holding all three back would
  remove 4,558 of the 127,436 gated Agree votes in this sample. This is a judgement call for SidewalkWebpage,
  not something this note settles.
- **For the retrained models**, run the held-out-city benchmark (README) with Seattle, cdmx and spgg among the
  held-out cities. All three are training cities today, so a per-city failure there points at the data as well as
  the gate. Then re-run `calibration.per_city` on the first production pull after the retrain, before any held-back
  class is re-enabled.
- **Before a retrained model's Disagree class votes again**, check it per city. As served, Disagree precision
  ranged from 0.15 to 0.94 between cities. A global map could clear the gate on the pooled data while voting badly in
  individual cities.

## Caveats

- **Selection.** Humans do not vote on a random sample of AI-assessed labels (Triage routes AI-contested labels to
  them), and the labelled share varies by city: the calibration note found spgg, cdmx and St. Louis
  under-represented in the labelled sample. Per-city precision is measured on the labelled subset only.
- **Ties count 0.5.** The Wilson interval treats the outcome as a Bernoulli rate. With ties it is approximate.
- **Folds are split per label, not per panorama**, so the out-of-fold values are somewhat optimistic (see the
  calibration note).
- **The threshold that reaches the gate is in-sample.** It shows how large a change a per-city gate would need. It
  is not a value to ship.
- These numbers describe the model revisions in production on 2026-10-17. A retrained model needs a new run.
