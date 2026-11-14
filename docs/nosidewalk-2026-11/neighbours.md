| City | Scored labels | Faces | Pass the gate | Share | Mean share per face |
|---|---|---|---|---|---|
| cdmx | 272 | 55 | 121 | 0.445 | 0.463 |
| chicago-il | 895 | 90 | 517 | 0.578 | 0.533 |
| oradell-nj | 817 | 50 | 665 | 0.814 | 0.712 |
| tucson-az | 651 | 55 | 256 | 0.393 | 0.410 |
| all | 2635 | 250 | 1559 | 0.592 | 0.527 |

| Group | Scored labels | Faces | Pass the gate | Share | Mean share per face |
|---|---|---|---|---|---|
| rated | 250 | 250 | 150 | 0.600 | 0.600 |
| not rated | 2385 | 222 | 1409 | 0.591 | 0.522 |
| same labeler | 1280 | 190 | 680 | 0.531 | 0.507 |
| other labeler | 1105 | 124 | 729 | 0.660 | 0.536 |
| rater placed | 480 | 26 | 310 | 0.646 | 0.676 |
| rater voted | 112 | 22 | 109 | 0.973 | 0.937 |
| author excluded | 0 | 0 | 0 |  |  |
| trained pano | 26 | 12 | 22 | 0.846 | 0.854 |
| pilot heldout | 16 | 7 | 3 | 0.188 | 0.429 |

Labels beside the rated one, by the stratum of the face's rated label (the faces were chosen through that label, 60 % of them from the pass stratum, so every share on this page describes this sample only):

| Rated label's stratum | Scored labels | Faces | Pass the gate | Share | Mean share per face |
|---|---|---|---|---|---|
| pass | 1579 | 139 | 1151 | 0.729 | 0.673 |
| agree below | 482 | 44 | 196 | 0.407 | 0.357 |
| model incorrect | 324 | 39 | 62 | 0.191 | 0.170 |

Faces with at least two scored labels: 222. The model's gate verdict is pass for every label on 38, for none on 38, and mixed on 146; verdicts drawn independently at each city's pass share (over the scored labels of these faces, rated ones included) would leave about 192 mixed, so the verdict clusters by face. Mixed faces are longer (median 9 labels against 5 on the others); on 42 of them at most a fifth of the labels differ from the rest. Of the 27 two-label faces, 11 are mixed. 640 of the 2635 scored labels sit at 2 or 3 of the 5 models, one model away from the other verdict.

42 of the 250 rated faces carry a label the rater placed or voted on. Correct / decided = rate (Wilson 95 % lower bound) on the rated labels:

| Stratum | Faces the rater had touched | Other faces |
|---|---|---|
| pass | 31 / 31 = 1.000 (0.890) | 111 / 113 = 0.982 (0.938) |
| agree below | 7 / 7 = 1.000 (0.646) | 35 / 41 = 0.854 (0.716) |
| model incorrect | 4 / 4 = 1.000 (0.510) | 16 / 39 = 0.410 (0.271) |

Without the touched faces the bar on the pass stratum is still met.

| Severity | Scored labels | Faces | Pass the gate | Share | Mean share per face |
|---|---|---|---|---|---|
| (none) | 2635 | 250 | 1559 | 0.592 | 0.527 |

| Tag | Scored labels | Faces | Pass the gate | Share | Mean share per face |
|---|---|---|---|---|---|
| street has no sidewalks | 1220 | 166 | 815 | 0.668 | 0.641 |
| street has a sidewalk | 245 | 56 | 126 | 0.514 | 0.449 |
| gravel/dirt road | 209 | 50 | 107 | 0.512 | 0.494 |
| shared pedestrian/car space | 190 | 39 | 59 | 0.311 | 0.237 |
| ends abruptly | 146 | 72 | 38 | 0.260 | 0.269 |
| (none) | 907 | 117 | 530 | 0.584 | 0.569 |

