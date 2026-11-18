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

#### Item 2 on step (b), the masked round (primary; exploratory, post hoc)

Exploratory, post hoc; the step (b) bar is as published above and is not re-read here. 235 decided ratings (31 incorrect), each label's inclusion weight from the key. A pooled score is the mean logit of one fold model's `p_correct` over the group (the rated label included), served through a map fitted on step (a)'s pooled out-of-fold scores of the **same** group (all five folds; Disagree held back), then 3 of 5 models at >= 0.92 as in the round. Vote pooling: Agree when at least half of the group's scored labels pass the round's own rule; strict also needs the label itself to pass. The single rule is the round's pass stratum: 142 / 144 = 0.986 (0.951). 58 decided labels have no other scored label in their run, so a run pool cannot change them. Nothing is refitted on step (b).

| group | gate | passing: correct / decided (Wilson 95 % lower) | added outside the pass stratum (incorrect) | newly admitted (incorrect) | dropped (incorrect) | weighted coverage | weighted precision [95 % CI]; Wilson at the Kish n | weighted AUC | within-city weighted AUC |
|---|---|---|---|---|---|---|---|---|---|
| single (the round's rule) | as served in round b | 142 / 144 = 0.986 (0.951) | 0 (0) | 0 (0) | 0 (0) | 0.405 | 0.998 [0.995, 1.000]; Kish n 84.5, 0.953 | 0.929 | 0.949 |
| single, step (a) re-scored map | score | 142 / 144 = 0.986 (0.951) | 0 (0) | 0 (0) | 0 (0) | 0.405 | 0.998 [0.995, 1.000]; Kish n 84.5, 0.953 | 0.929 | 0.949 |
| the labeler's run on the face (R, primary) | score | 118 / 119 = 0.992 (0.954) | 7 (0) | 7 (0) | 32 (1) | 0.401 | 0.999 [0.997, 1.000]; Kish n 57.7, 0.936 | 0.965 | 0.986 |
| the labeler's run on the face (R, primary) | vote | 143 / 144 = 0.993 (0.962) | 15 (0) | 15 (0) | 15 (1) | 0.477 | 0.999 [0.997, 1.000]; Kish n 72.3, 0.948 | | |
| the labeler's run on the face (R, primary) | vote strict | 128 / 129 = 0.992 (0.957) | 0 (0) | 0 (0) | 15 (1) | 0.384 | 0.999 [0.997, 1.000]; Kish n 79.3, 0.952 | | |
| the whole face (F) | score | 162 / 163 = 0.994 (0.966) | 30 (0) | 30 (0) | 11 (1) | 0.652 | 0.999 [0.998, 1.000]; Kish n 69.6, 0.947 | 0.965 | 0.979 |
| the whole face (F) | vote | 143 / 144 = 0.993 (0.962) | 19 (0) | 19 (0) | 19 (1) | 0.467 | 0.999 [0.997, 1.000]; Kish n 73.6, 0.949 | | |
| the whole face (F) | vote strict | 124 / 125 = 0.992 (0.956) | 0 (0) | 0 (0) | 19 (1) | 0.385 | 0.999 [0.997, 1.000]; Kish n 78.6, 0.951 | | |
| the face without the label itself | score | 115 / 116 = 0.991 (0.953) | 13 (0) | 13 (0) | 41 (1) | 0.400 | 0.999 [0.997, 1.000]; Kish n 57.3, 0.935 | 0.939 | 0.960 |
| the face without the label itself | vote | 134 / 135 = 0.993 (0.959) | 23 (0) | 23 (0) | 32 (1) | 0.469 | 0.999 [0.997, 1.000]; Kish n 65.5, 0.943 | | |
| the face without the label itself | vote strict | 111 / 112 = 0.991 (0.951) | 0 (0) | 0 (0) | 32 (1) | 0.353 | 0.999 [0.997, 1.000]; Kish n 71.6, 0.947 | | |
| the run, one vote per panorama (P) | score | 118 / 119 = 0.992 (0.954) | 7 (0) | 7 (0) | 32 (1) | 0.401 | 0.999 [0.997, 1.000]; Kish n 57.7, 0.936 | 0.964 | 0.981 |
| the run, one vote per panorama (P) | vote | 144 / 145 = 0.993 (0.962) | 15 (0) | 15 (0) | 14 (1) | 0.477 | 0.999 [0.997, 1.000]; Kish n 72.4, 0.948 | | |
| the run, one vote per panorama (P) | vote strict | 129 / 130 = 0.992 (0.958) | 0 (0) | 0 (0) | 14 (1) | 0.385 | 0.999 [0.997, 1.000]; Kish n 79.5, 0.952 | | |
| labels within 50 m on the face (D) | score | 141 / 142 = 0.993 (0.961) | 14 (0) | 14 (0) | 16 (1) | 0.512 | 0.999 [0.998, 1.000]; Kish n 66.3, 0.944 | 0.955 | 0.974 |
| labels within 50 m on the face (D) | vote | 139 / 140 = 0.993 (0.961) | 13 (0) | 13 (0) | 17 (1) | 0.425 | 0.999 [0.997, 1.000]; Kish n 78.9, 0.952 | | |
| labels within 50 m on the face (D) | vote strict | 126 / 127 = 0.992 (0.957) | 0 (0) | 0 (0) | 17 (1) | 0.391 | 0.999 [0.997, 1.000]; Kish n 79.5, 0.952 | | |

**Rule (fixed before any pooled number was computed):** >= 10 added outside the pass stratum, <= 1 newly admitted incorrect, raw precision within 0.01 of the single rule, on the run, score pooling. Added outside the pass stratum: 7 (not met); newly admitted incorrect: 0 (met); raw precision 0.992 against 0.986 (met). **Do not pursue a per-run gate on this evidence.** The rule compares the run pool under its refitted map with the round's own rule, so part of any change is the change of map. Without pooling, the single label under its own refitted map: added outside the pass stratum 0, newly admitted incorrect 0, raw precision 0.986 (it does not meet the rule on its own). **What pooling adds (run minus single, both refitted):** +7 added outside the pass stratum, +0 newly admitted incorrect, raw precision +0.005.

Transitions of the primary gate (stratum, single pass, pooled pass, verdict: labels, summed weight):

| stratum | single passes | pooled passes | verdict | labels | weight |
|---|---|---|---|---|---|
| agree_below | no | no | correct | 35 | 6892.2 |
| agree_below | no | no | incorrect | 6 | 1298.0 |
| agree_below | no | yes | correct | 7 | 1462.0 |
| model_incorrect | no | no | correct | 20 | 1125.0 |
| model_incorrect | no | no | incorrect | 23 | 1135.2 |
| pass | yes | no | correct | 31 | 1535.3 |
| pass | yes | no | incorrect | 1 | 8.2 |
| pass | yes | yes | correct | 111 | 6546.7 |
| pass | yes | yes | incorrect | 1 | 8.2 |

Sensitivities of the primary gate (run, score pooling):

| group | gate | passing: correct / decided (Wilson 95 % lower) | added outside the pass stratum (incorrect) | newly admitted (incorrect) | dropped (incorrect) | weighted coverage | weighted precision [95 % CI]; Wilson at the Kish n | weighted AUC | within-city weighted AUC |
|---|---|---|---|---|---|---|---|---|---|
| without neighbours the rater placed or voted on | score | 118 / 119 = 0.992 (0.954) | 7 (0) | 7 (0) | 32 (1) | 0.401 | 0.999; Kish n 57.7, 0.936 | 0.965 | 0.986 |
| without neighbours on trained panoramas or on step (a) held-out faces | score | 118 / 119 = 0.992 (0.954) | 7 (0) | 7 (0) | 32 (1) | 0.401 | 0.999; Kish n 57.7, 0.936 | 0.965 | 0.986 |
| without the 42 faces the rater had touched | score | 92 / 93 = 0.989 (0.942) | 6 (0) | 6 (0) | 26 (1) | 0.381 | 0.999; Kish n 47.1, 0.922 | 0.963 | 0.986 |
| cdmx | score | 27 / 27 = 1.000 (0.875) | 1 (0) | 1 (0) | 3 (0) | 0.195 | 1.000; Kish n 7.3, 0.656 | 0.981 | 0.981 |
| chicago-il | score | 44 / 44 = 1.000 (0.920) | 3 (0) | 3 (0) | 10 (0) | 0.443 | 1.000; Kish n 36.4, 0.904 | 0.989 | 0.989 |
| oradell-nj | score | 30 / 30 = 1.000 (0.886) | 2 (0) | 2 (0) | 5 (0) | 0.731 | 1.000; Kish n 29.6, 0.885 | 0.988 | 0.988 |
| tucson-az | score | 17 / 18 = 0.944 (0.742) | 1 (0) | 1 (0) | 14 (1) | 0.214 | 0.954; Kish n 12.3, 0.697 | 0.944 | 0.944 |

By tag, run, score pooling (>= 30 decided; raw precision of passers, Wilson 95 % lower; NoSidewalk has no severity): (none) (48 decided, 9 incorrect): pooled 19 / 19 = 1.000 (0.832), single 23 / 23 = 1.000 (0.857); street has no sidewalks (127 decided, 4 incorrect): pooled 79 / 79 = 1.000 (0.954), single 95 / 95 = 1.000 (0.961).

By tag, run, vote pooling (>= 30 decided; raw precision of passers, Wilson 95 % lower; NoSidewalk has no severity): (none) (48 decided, 9 incorrect): pooled 23 / 23 = 1.000 (0.857), single 23 / 23 = 1.000 (0.857); street has no sidewalks (127 decided, 4 incorrect): pooled 95 / 95 = 1.000 (0.961), single 95 / 95 = 1.000 (0.961).

By tag, run, strict vote pooling (>= 30 decided; raw precision of passers, Wilson 95 % lower; NoSidewalk has no severity): (none) (48 decided, 9 incorrect): pooled 19 / 19 = 1.000 (0.832), single 23 / 23 = 1.000 (0.857); street has no sidewalks (127 decided, 4 incorrect): pooled 86 / 86 = 1.000 (0.957), single 95 / 95 = 1.000 (0.961).

#### Item 2 on step (a), out of fold (secondary; exploratory)

Every NoSidewalk label on the 440 out-of-fold faces and the held-out faces was re-cropped and scored with the kept gate2 weights (7,736 of 7,759 labels scored, 99.7 %; the plan's threshold is 90 %: met; seattle-wa 2,484 of 2,507 (every other city fully scored)), each out-of-fold face by its own fold's model only. Both arms use these re-scored values; against gate2's own predictions for the 746 split rows (other crop path) the median |delta logit| is 0.000 (90th percentile 0.002, max 0.034), 0 verdicts flip at 0.5 and 0 gate passes flip under the gate2 map; for the 126 held-out split rows (five-model mean) the median is 0.001 (max 0.048), 0 verdicts and 0 gate passes flip. The 13 split rows without a face are on no step (a) face, so they are neither re-scored nor pooled. Each pooled score's map is refitted leave-one-fold-out on the split rows' pooled scores of the same group. **The label-level expert truth on one face comes from the same sittings as item 1, so this comparison favours pooling; step (b) is the primary test.**

| set | group | rows (incorrect) | AUC [95 % CI, faces] | within-city AUC | delta AUC vs single [95 % CI] | Agree gate: passing (precision, unweighted) |
|---|---|---|---|---|---|---|
| (ii) label-level truth, all | single | 1125 (242) | 0.968 [0.951, 0.981] | 0.962 |  | 610 (0.992) |
| (ii) label-level truth, all | run | 1125 (242) | 0.971 [0.953, 0.984] | 0.958 | 0.003 [-0.007, 0.012] | 603 (0.985) |
| (ii) label-level truth, all | face | 1125 (242) | 0.973 [0.955, 0.986] | 0.962 | 0.005 [-0.006, 0.015] | 556 (0.987) |
| (ii) label-level truth, all | face_loo | 1125 (242) | 0.971 [0.950, 0.984] | 0.958 | 0.003 [-0.011, 0.013] | 553 (0.987) |
| (ii) label-level truth, all | run_pano | 1125 (242) | 0.971 [0.953, 0.984] | 0.959 | 0.003 [-0.006, 0.012] | 720 (0.988) |
| (ii) label-level truth, all | near50 | 1125 (242) | 0.976 [0.960, 0.988] | 0.973 | 0.008 [0.002, 0.015] | 606 (0.988) |
| (ii) split rows (margin / expert) | single | 283 (197) | 0.959 [0.933, 0.980] | 0.954 |  | 59 (0.915) |
| (ii) split rows (margin / expert) | run | 283 (197) | 0.958 [0.933, 0.980] | 0.944 | -0.000 [-0.013, 0.012] | 61 (0.869) |
| (ii) split rows (margin / expert) | face | 283 (197) | 0.966 [0.943, 0.984] | 0.957 | 0.007 [-0.010, 0.025] | 55 (0.891) |
| (ii) split rows (margin / expert) | face_loo | 283 (197) | 0.963 [0.939, 0.982] | 0.957 | 0.005 [-0.017, 0.025] | 54 (0.889) |
| (ii) split rows (margin / expert) | run_pano | 283 (197) | 0.958 [0.933, 0.981] | 0.945 | -0.000 [-0.013, 0.012] | 72 (0.889) |
| (ii) split rows (margin / expert) | near50 | 283 (197) | 0.970 [0.949, 0.986] | 0.968 | 0.012 [-0.001, 0.027] | 61 (0.885) |
| (ii) other labels' own expert verdicts | single | 842 (45) | 0.976 [0.953, 0.998] | 0.969 |  | 551 (1.000) |
| (ii) other labels' own expert verdicts | run | 842 (45) | 0.980 [0.954, 0.998] | 0.983 | 0.004 [-0.021, 0.020] | 542 (0.998) |
| (ii) other labels' own expert verdicts | face | 842 (45) | 0.982 [0.950, 0.997] | 0.984 | 0.005 [-0.025, 0.024] | 501 (0.998) |
| (ii) other labels' own expert verdicts | face_loo | 842 (45) | 0.976 [0.934, 0.997] | 0.979 | -0.000 [-0.040, 0.019] | 499 (0.998) |
| (ii) other labels' own expert verdicts | run_pano | 842 (45) | 0.980 [0.954, 0.998] | 0.983 | 0.004 [-0.021, 0.020] | 648 (0.998) |
| (ii) other labels' own expert verdicts | near50 | 842 (45) | 0.985 [0.968, 0.999] | 0.987 | 0.009 [-0.008, 0.024] | 545 (1.000) |
| (iii) one label-level row per face | single | 365 (120) | 0.938 [0.907, 0.964] | 0.936 |  | 135 (0.970) |
| (iii) one label-level row per face | run | 365 (120) | 0.936 [0.901, 0.966] | 0.935 | -0.002 [-0.022, 0.016] | 139 (0.942) |
| (iii) one label-level row per face | face | 365 (120) | 0.944 [0.911, 0.971] | 0.946 | 0.006 [-0.016, 0.026] | 134 (0.955) |
| (iii) one label-level row per face | face_loo | 365 (120) | 0.938 [0.901, 0.966] | 0.941 | -0.000 [-0.026, 0.023] | 129 (0.953) |
| (iii) one label-level row per face | run_pano | 365 (120) | 0.936 [0.901, 0.966] | 0.936 | -0.002 [-0.021, 0.016] | 169 (0.953) |
| (iii) one label-level row per face | near50 | 365 (120) | 0.950 [0.920, 0.974] | 0.953 | 0.012 [-0.001, 0.025] | 150 (0.960) |
| (i) all split rows (circular) | single | 746 (309) | 0.937 [0.906, 0.962] | 0.939 |  | 262 (0.950) |
| (i) all split rows (circular) | run | 746 (309) | 0.944 [0.916, 0.968] | 0.947 | 0.007 [-0.003, 0.017] | 249 (0.932) |
| (i) all split rows (circular) | face | 746 (309) | 0.950 [0.921, 0.973] | 0.956 | 0.013 [0.000, 0.026] | 242 (0.942) |
| (i) all split rows (circular) | face_loo | 746 (309) | 0.949 [0.920, 0.972] | 0.956 | 0.012 [-0.002, 0.025] | 223 (0.955) |
| (i) all split rows (circular) | run_pano | 746 (309) | 0.944 [0.916, 0.968] | 0.947 | 0.007 [-0.003, 0.017] | 314 (0.943) |
| (i) all split rows (circular) | near50 | 746 (309) | 0.948 [0.919, 0.972] | 0.954 | 0.011 [0.003, 0.020] | 269 (0.933) |
| (iii) one split row per face (circular) | single | 440 (123) | 0.935 [0.904, 0.963] | 0.935 |  | 184 (0.973) |
| (iii) one split row per face (circular) | run | 440 (123) | 0.939 [0.908, 0.966] | 0.941 | 0.004 [-0.013, 0.022] | 181 (0.956) |
| (iii) one split row per face (circular) | face | 440 (123) | 0.941 [0.910, 0.970] | 0.945 | 0.006 [-0.014, 0.027] | 174 (0.966) |
| (iii) one split row per face (circular) | face_loo | 440 (123) | 0.938 [0.906, 0.966] | 0.946 | 0.003 [-0.020, 0.026] | 164 (0.970) |
| (iii) one split row per face (circular) | run_pano | 440 (123) | 0.939 [0.908, 0.966] | 0.942 | 0.004 [-0.014, 0.022] | 225 (0.964) |
| (iii) one split row per face (circular) | near50 | 440 (123) | 0.943 [0.914, 0.969] | 0.948 | 0.008 [-0.006, 0.022] | 196 (0.959) |

Per city, set (ii) label-level truth, cities with >= 10 rows (AUC; n/a with one class):

| city | rows (incorrect) | AUC single | AUC run | AUC face |
|---|---|---|---|---|
| amsterdam | 38 (36) | 0.403 | 0.528 | 0.653 |
| newberg-or | 17 (0) | n/a | n/a | n/a |
| oradell-nj | 661 (21) | 0.962 | 0.946 | 0.941 |
| seattle-wa | 330 (169) | 0.963 | 0.965 | 0.973 |
| walla-walla-wa | 20 (2) | 1.000 | 1.000 | 1.000 |

Held-out faces (descriptive; five-model mean logit, map fitted on all split rows):

| group | rows (incorrect) | AUC | within-city AUC | passing (precision) |
|---|---|---|---|---|
| single | 77 (39) | 0.952 | 0.945 | 27 (0.926) |
| run | 77 (39) | 0.954 | 0.942 | 19 (1.000) |
| face | 77 (39) | 0.960 | 0.949 | 36 (0.861) |
| face_loo | 77 (39) | 0.937 | 0.919 | 25 (0.920) |
| run_pano | 77 (39) | 0.955 | 0.943 | 19 (1.000) |
| near50 | 77 (39) | 0.951 | 0.942 | 30 (0.933) |

Sensitivity, without expert-placed neighbours: single AUC 0.968 (one per face 0.938); run AUC 0.971 (one per face 0.935); face AUC 0.974 (one per face 0.946); face_loo AUC 0.972 (one per face 0.942); run_pano AUC 0.971 (one per face 0.936); near50 AUC 0.978 (one per face 0.955).

Sensitivity, without neighbours on the scoring model's trained panoramas: single AUC 0.968 (one per face 0.938); run AUC 0.970 (one per face 0.936); face AUC 0.973 (one per face 0.944); face_loo AUC 0.970 (one per face 0.936); run_pano AUC 0.971 (one per face 0.937); near50 AUC 0.976 (one per face 0.950).

Stability, the step (a) `gate/` run (its own predictions, split rows only): on 283 label-level rows single AUC 0.956, face mean over split rows 0.956.

Tag slices on (ii), run pooled against single (training rows: the split's train, val and test rows carrying the tag, the pool the fold models draw from; NoSidewalk has no severity):

| tag | training rows | rows (incorrect) | single AUC | run AUC | delta [95 % CI] |
|---|---|---|---|---|---|
| (none) | 325 | 631 (137) | 0.984 | 0.984 | 0.001 [-0.006, 0.005] |
| street has no sidewalks | 278 | 377 (51) | 0.938 | 0.948 | 0.009 [-0.043, 0.046] |
| street has a sidewalk | 93 | 64 (27) | 0.951 | 0.960 | 0.009 [-0.022, 0.047] |
| ends abruptly | 47 | 53 (23) | 0.854 | 0.917 | 0.064 [0.013, 0.130] |
| shared pedestrian/car space | 40 | 34 (15) | 0.895 | 0.926 | 0.032 [-0.087, 0.159] |

#### Item 3: neighbour features without the image

Logistic regression (standardised, C = 1, no tuning): a = image logit, b = image logit + features, c = features only. Fitted on step (a) label-level truth (faces weighted equally), leave-one-fold-out; then fitted on all of it, frozen, and applied to each fold model's logit of the step (b) ratings (averaged over the five). Contradicting labels come from a fresh public pull; labels whose own row moved face or vanished there are dropped (step (a): 1,125 on the same face, 0 moved, 0 missing; step (b): 234 on the same face, 0 moved, 1 missing). The face-size prior already uses the face's label count at calibration, so `n_ns_face` is not new to the served map.

| variant | step (a) set | rows | AUC a | AUC b | AUC c | delta b - a [95 % CI] | within-city a / b |
|---|---|---|---|---|---|---|---|
| with expert placed | ii_all | 1125 | 0.967 | 0.985 | 0.949 | 0.018 [0.007, 0.030] | 0.961 / 0.982 |
| with expert placed | iii | 365 | 0.936 | 0.965 | 0.897 | 0.029 [0.009, 0.051] | 0.934 / 0.974 |
| with expert placed | ii_split | 283 | 0.958 | 0.965 | 0.872 | 0.006 [-0.011, 0.025] | 0.953 / 0.964 |
| with expert placed | ii_extra | 842 | 0.977 | 0.983 | 0.938 | 0.006 [-0.031, 0.031] | 0.971 / 0.985 |
| without expert placed | ii_all | 1125 | 0.967 | 0.985 | 0.948 | 0.017 [0.005, 0.032] | 0.961 / 0.977 |
| without expert placed | iii | 365 | 0.936 | 0.969 | 0.898 | 0.033 [0.013, 0.055] | 0.934 / 0.977 |
| without expert placed | ii_split | 283 | 0.958 | 0.962 | 0.857 | 0.004 [-0.014, 0.024] | 0.953 / 0.955 |
| without expert placed | ii_extra | 842 | 0.977 | 0.977 | 0.931 | 0.000 [-0.052, 0.028] | 0.971 / 0.972 |

| variant | step (b) decided | weighted AUC a / b / c | within-city weighted AUC a / b / c | delta b - a, within-city weighted [95 % CI] |
|---|---|---|---|---|
| with expert placed | 234 | 0.909 / 0.861 / 0.746 | 0.931 / 0.898 / 0.788 | -0.033 [-0.110, 0.046] |
| without expert placed | 234 | 0.909 / 0.854 / 0.715 | 0.931 / 0.905 / 0.752 | -0.026 [-0.106, 0.052] |

By tag, with expert placed (>= 30 rows; AUC a / b / c; NoSidewalk has no severity): step (a) (ii) (none) 631 (137): 0.983 / 0.996 / 0.975; ends abruptly 53 (23): 0.849 / 0.919 / 0.827; shared pedestrian/car space 34 (15): 0.902 / 0.965 / 0.944; street has a sidewalk 64 (27): 0.953 / 0.949 / 0.836; street has no sidewalks 377 (51): 0.938 / 0.958 / 0.907; step (b), weighted (none) 47 (9): 0.984 / 0.954 / 0.722; street has no sidewalks 127 (4): 0.870 / 0.568 / 0.373.

By tag, without expert placed (>= 30 rows; AUC a / b / c; NoSidewalk has no severity): step (a) (ii) (none) 631 (137): 0.983 / 0.998 / 0.974; ends abruptly 53 (23): 0.849 / 0.923 / 0.832; shared pedestrian/car space 34 (15): 0.902 / 0.958 / 0.933; street has a sidewalk 64 (27): 0.953 / 0.968 / 0.858; street has no sidewalks 377 (51): 0.938 / 0.946 / 0.890; step (b), weighted (none) 47 (9): 0.984 / 0.960 / 0.696; street has no sidewalks 127 (4): 0.870 / 0.497 / 0.332.

**Rule:** the features count as adding only if the step (b) within-city weighted delta (b - a) without expert-placed labels has a 95 % CI above 0: **they do not add**.

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
