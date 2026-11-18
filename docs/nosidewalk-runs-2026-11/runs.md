**Not yet computed: item 2 (pooled scores), item 4 (the recommendation); item 3 (neighbour features).** They need the step (a) labels re-scored with the kept gate2 weights on klone; until those scores are copied back and pinned, this page reports item 1 and the run census only, and no statement below is about pooling or neighbour features.

#### Item 1: do the labels of a run get the same expert verdict?

Descriptive only. NoSidewalk on the 2026-11-05 pull; a label's verdict is its **own** expert votes (every expert vote on it the same way, crowd ignored): 2,400 labels (19,835 placed by an expert and 0 by SidewalkAI left out, 1 where the experts split, 119 without a face). Pair agreement is the share of within-run label pairs with the same verdict, with a 95 % CI from resampling runs within city. Null 1 permutes verdicts within the city, null 2 within the labeler (across their faces); p is one-sided over 1,000 permutations.

| city | labels | runs of 2+ | pairs | pair agreement [95 % CI] | all-same runs | ICC | null 1: city (p) | null 2: labeler (p) | same labeler, other face | same face, other labeler | any two in the city |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seattle-wa | 700 | 77 | 368 | 0.984 [0.955, 1.000] | 0.961 | 0.955 | 0.636 (p 0.001) | 0.944 (p 0.002) | 0.941 | 0.960 | 0.635 |
| chicago-il | 215 | 22 | 300 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.532 (p 0.001) | 0.804 (p 0.001) | 0.726 | 1.000 | 0.531 |
| oradell-nj | 1,076 | 142 | 6,570 | 0.989 [0.969, 0.999] | 0.965 | 0.641 | 0.959 (p 0.001) | 0.970 (p 0.001) | 0.926 | 0.942 | 0.958 |
| other (24 cities, pooled) | 290 | 40 | 84 | 0.952 [0.868, 1.000] | 0.975 | n/a | 0.915 (p 0.157) | 0.934 (p 0.299) | 0.925 | 0.889 | 0.928 |
| **macro mean**, 3 cities with >= 10 runs of 2+ (chicago-il, oradell-nj, seattle-wa) | 1,991 | | 7,238 | 0.991 [0.979, 0.998] | 0.975 | 0.865 | 0.709 (p 0.001) | 0.906 (p 0.001) | 0.864 | 0.967 | 0.708 |

Within-run pairs by the time between the expert's two votes, and by the page they were cast on (`pull_expert_vote_times.sh`, each expert's latest vote before the pull; cells under 30 pairs blank). Of the pull's expert votes on these labels 2,289 match a timed vote, 0 were changed around the pull and 2 have no timed vote; a pair with either is "time unknown".

| scope | different expert | same expert, < 1 h | same expert, 1 h - 1 day | same expert, >= 1 day | same expert, time unknown |
|---|---|---|---|---|---|
| all cities (pair-weighted pool, not a macro mean; 90% of them oradell-nj) | 7: n/a | 2,188: 0.984 | 176: 1.000 | 4,950: 0.991 | 1: n/a |
| seattle-wa | 0 | 360: 0.983 | 0 | 8: n/a | 0 |
| chicago-il | 0 | 299: 1.000 | 0 | 0 | 1: n/a |
| oradell-nj | 1: n/a | 1,458: 0.983 | 176: 1.000 | 4,935: 0.990 | 0 |

| scope (same-expert pairs) | AdminUserDashboard | Gallery | LabelMap | Old data | Validate | mixed | other | unknown |
|---|---|---|---|---|---|---|---|---|
| all cities (pair-weighted pool, not a macro mean; 90% of them oradell-nj) | 77: 0.922 | 11: n/a | 275: 1.000 | 364: 0.989 | 6,554: 0.989 | 32: 1.000 | 1: n/a | 1: n/a |
| seattle-wa | 19: n/a | 0 | 0 | 340: 0.988 | 4: n/a | 5: n/a | 0 | 0 |
| chicago-il | 19: n/a | 1: n/a | 275: 1.000 | 3: n/a | 0 | 0 | 1: n/a | 1: n/a |
| oradell-nj | 0 | 0 | 0 | 12: n/a | 6,530: 0.989 | 27: n/a | 0 | 0 |

By tag (pairs where both labels carry it, pair-weighted over all cities): street has no sidewalks 1,915 pairs, 0.966; street has a sidewalk 200 pairs, 1.000; gravel/dirt road 67 pairs, 1.000. NoSidewalk has no severity.

Sensitivities of the macro mean: `data_generation.expert_verdict` (a contradicting crowd vote drops the label): 0.992 over 3 cities (null 1 0.776 (p 0.001), null 2 0.925 (p 0.001)); without Oradell: 0.992 over 2 cities (null 1 0.584 (p 0.001), null 2 0.874 (p 0.001)); the 2026-10-18 pull: 0.992 over 3 cities (null 1 0.699 (p 0.001), null 2 0.908 (p 0.001)).

All seven types on the 2026-10-18 pull (macro mean over cities with >= 10 runs of 2+; same rules):

| type | labels with own verdict | pairs | cities | pair agreement [95 % CI] | null 1: city (p) | null 2: labeler (p) | same labeler, other face | same face, other labeler | share of same-expert pairs voted < 1 h apart |
|---|---|---|---|---|---|---|---|---|---|
| NoSidewalk | 2,325 | 6,917 | 3 | 0.992 [0.981, 0.999] | 0.699 (p 0.001) | 0.908 (p 0.001) | 0.860 | 0.981 | 29% |
| CurbRamp | 22,398 | 3,937 | 16 | 0.949 [0.938, 0.961] | 0.901 (p 0.001) | 0.928 (p 0.001) | 0.947 | 0.947 | 43% |
| NoCurbRamp | 12,960 | 2,343 | 13 | 0.915 [0.891, 0.935] | 0.695 (p 0.001) | 0.803 (p 0.001) | 0.770 | 0.926 | 28% |
| Obstacle | 8,291 | 4,450 | 14 | 0.895 [0.858, 0.917] | 0.623 (p 0.001) | 0.798 (p 0.001) | 0.757 | 0.890 | 51% |
| SurfaceProblem | 14,899 | 14,838 | 18 | 0.943 [0.926, 0.955] | 0.850 (p 0.001) | 0.901 (p 0.001) | 0.911 | 0.924 | 24% |
| Crosswalk | 4,659 | 261 | 6 | 0.982 [0.960, 0.998] | 0.818 (p 0.001) | 0.919 (p 0.001) | 0.942 | 0.967 | 43% |
| Signal | 1,321 | 110 | 3 | 0.986 [0.958, 1.000] | 0.926 (p 0.037) | 0.987 (p 0.728) | 0.977 | 0.958 | 60% |

Type x severity and type x tag (within-run pairs where both labels carry the same severity or the tag, a pair-weighted pool over cities, not a macro mean, so the cities with the most pairs set it; blank under 30 pairs; top three tags by pairs):

| type | severity 1 | severity 2 | severity 3 | tags |
|---|---|---|---|---|
| NoSidewalk | no severity | |  | street has no sidewalks 1,656: 0.963; street has a sidewalk 199: 1.000; gravel/dirt road 67: 1.000 |
| CurbRamp | 2,722: 0.928 | 206: 0.913 | 93: 0.935 | missing tactile warning 689: 0.940; points into traffic 295: 0.912; narrow 82: 0.915 |
| NoCurbRamp | 158: 0.905 | 320: 0.869 | 1,219: 0.931 | no alternate route 825: 0.908; unclear if needed 202: 0.975; alternate route present 138: 0.877 |
| Obstacle | 1,335: 0.928 | 553: 0.837 | 1,106: 0.910 | pole 764: 0.846; trash/recycling can 521: 0.985; parked car 452: 0.927 |
| SurfaceProblem | 6,675: 0.905 | 1,552: 0.907 | 1,178: 0.914 | cracks 3,586: 0.895; grass 3,079: 0.976; uneven/slanted 1,391: 0.940 |
| Crosswalk | 176: 0.994 | 7: n/a | 24: n/a | paint fading 17: n/a; broken surface 13: n/a; rail/tram track 11: n/a |
| Signal | no severity | |  | one button 50: 1.000; button waist height 29: n/a; two buttons 9: n/a |

#### Runs of every type (the issue's table)

A run is one labeler's labels of one type on one block face; a label without a face is a run of one. Public cities only.

| type (pull) | cities | labels | in a run of 2+ | of 3+ | run length of the median label | multi-label faces with one labeler | seattle-wa: labels, in 2+, median run |
|---|---|---|---|---|---|---|---|
| NoSidewalk 2026-11-05 | 45 | 237,421 | 86% | 78% | 7 | 58% | 51,646, 82%, 6 |
| NoSidewalk 2026-10-18 | 45 | 232,258 | 85% | 76% | 7 | 59% | 51,642, 81%, 6 |
| CurbRamp 2026-10-18 | 46 | 541,936 | 66% | 41% | 2 | 35% | 100,933, 54%, 2 |
| NoCurbRamp 2026-10-18 | 46 | 122,675 | 43% | 19% | 1 | 27% | 44,663, 44%, 1 |
| Obstacle 2026-10-18 | 46 | 158,013 | 72% | 56% | 3 | 52% | 17,982, 61%, 2 |
| SurfaceProblem 2026-10-18 | 46 | 192,017 | 76% | 61% | 3 | 68% | 37,545, 71%, 3 |
| Crosswalk 2026-10-18 | 45 | 67,325 | 15% | 4% | 1 | 33% | 6,280, 14%, 1 |
| Signal 2026-10-18 | 39 | 15,168 | 31% | 8% | 1 | 42% | 1,671, 25%, 1 |

NoSidewalk on the 2026-11-05 pull, cities with >= 1,000 labels:

| city | labels | in a run of 2+ | of 3+ | run length of the median label | multi-label faces with one labeler |
|---|---|---|---|---|---|
| seattle-wa | 51,646 | 82% | 74% | 6 | 50% |
| washington-dc | 45,213 | 88% | 82% | 10 | 39% |
| chicago-il | 14,733 | 82% | 72% | 6 | 84% |
| st-louis-mo | 14,535 | 95% | 90% | 9 | 95% |
| knox-oh | 12,617 | 94% | 90% | 48 | 92% |
| mendota-il | 12,412 | 91% | 85% | 11 | 13% |
| columbus-oh | 9,371 | 88% | 79% | 6 | 43% |
| gainesville-fl | 7,962 | 94% | 81% | 5 | 99% |
| spgg | 7,470 | 65% | 50% | 3 | 33% |
| teaneck-nj | 7,226 | 85% | 75% | 6 | 31% |
| newberg-or | 7,037 | 92% | 84% | 9 | 29% |
| burnaby | 6,916 | 93% | 89% | 9 | 39% |
| walla-walla-wa | 6,420 | 91% | 81% | 6 | 67% |
| danville-il | 5,890 | 94% | 90% | 8 | 86% |
| cdmx | 5,744 | 58% | 43% | 2 | 40% |
| oradell-nj | 5,679 | 91% | 84% | 11 | 14% |
| amsterdam | 4,002 | 78% | 71% | 7 | 74% |
| pittsburgh-pa | 3,509 | 76% | 63% | 4 | 78% |
| maywood-nj | 1,524 | 90% | 81% | 5 | 47% |
| paterson-nj | 1,268 | 75% | 59% | 4 | 72% |
| laurens-ia | 1,250 | 89% | 85% | 11 | 91% |
| tucson-az | 1,215 | 88% | 80% | 8 | 43% |
