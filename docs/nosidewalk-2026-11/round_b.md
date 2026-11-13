250 items rated (13 Unsure, 2 unanswered). Precision cells are correct / decided = rate (Wilson 95 % lower bound); Unsure votes are outside the denominator.

**Bar:** in the pass stratum, 142 / 144 = 0.986 (0.951); a lower bound >= 0.92 on >= 100 decided labels is met. **PASSES.** Reweighted to the cities' population: 0.998 [0.995, 1.000] (the interval holds the weights fixed, so it is somewhat narrow). With every Unsure in the pass stratum counted as incorrect: 142 / 149 = 0.953 (0.906).

Share of correct NoSidewalk labels in these cities' eligible population: 0.878 [0.819, 0.932] (the deployment prior assumes 0.831 over every label with a face; this population leaves out the rater's own labels and trained faces).

| stratum | items | Unsure | correct / decided (Wilson 95 % lower) |
|---|---|---|---|
| pass | 150 | 5 | 142 / 144 = 0.986 (0.951) |
| agree_below | 50 | 2 | 42 / 48 = 0.875 (0.753) |
| model_incorrect | 50 | 6 | 20 / 43 = 0.465 (0.325) |

| city | decided (incorrect) | pass stratum (Wilson 95 % lower) | population share correct |
|---|---|---|---|
| chicago-il | 84 (8) | 51 / 51 = 1.000 (0.930) | 0.925 |
| tucson-az | 54 (12) | 29 / 31 = 0.935 (0.793) | 0.788 |
| oradell-nj | 49 (2) | 33 / 33 = 1.000 (0.896) | 0.981 |
| cdmx | 48 (9) | 29 / 29 = 1.000 (0.883) | 0.713 |

Per fold model, on its own gate verdicts (weights are the items' inverse inclusion probabilities):

| model | gate passes, rated items unweighted (Wilson 95 % lower) | precision, weighted | coverage, weighted | AUC, weighted | within-city AUC, weighted (cities) |
|---|---|---|---|---|---|
| fold 0 | 115 / 117 = 0.983 (0.940) | 0.997 | 0.318 | 0.883 | 0.926 (4) |
| fold 1 | 83 / 84 = 0.988 (0.936) | 0.998 | 0.264 | 0.910 | 0.943 (4) |
| fold 2 | 155 / 158 = 0.981 (0.946) | 0.975 | 0.622 | 0.871 | 0.941 (4) |
| fold 3 | 121 / 121 = 1.000 (0.969) | 1.000 | 0.372 | 0.923 | 0.942 (4) |
| fold 4 | 153 / 157 = 0.975 (0.936) | 0.963 | 0.536 | 0.819 | 0.753 (4) |

| severity | decided | pass stratum (Wilson 95 % lower) | population share correct | training rows |
|---|---|---|---|---|
| (none) | 235 | 142 / 144 = 0.986 (0.951) | 0.878 | 758 |

| tag | decided | pass stratum (Wilson 95 % lower) | population share correct | training rows |
|---|---|---|---|---|
| (none) | 48 | 23 / 23 = 1.000 (0.857) | 0.892 | 325 |
| ends abruptly | 22 | 4 / 5 = 0.800 (0.376) | 0.812 | 47 |
| gravel/dirt road | 20 | 14 / 15 = 0.933 (0.702) | 0.857 | 38 |
| shared pedestrian/car space | 20 | 5 / 6 = 0.833 (0.436) | 0.328 | 40 |
| street has a sidewalk | 29 | 14 / 14 = 1.000 (0.785) | 0.849 | 93 |
| street has no sidewalks | 127 | 95 / 95 = 1.000 (0.961) | 0.949 | 278 |

Crowd majority (human votes at the pool pull) against the expert, decided items:

| crowd | expert correct | expert incorrect |
|---|---|---|
| correct | 34 | 2 |
| incorrect | 2 | 12 |
| none or tie | 168 | 17 |
