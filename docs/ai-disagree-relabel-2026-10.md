# Blind expert re-label of the AI's Disagree votes, and its severity / tag breakdown (October 2026)

Production's AI Disagree votes agree with human majorities only 56–71% of the time, depending on how strict the
human reference is, against 96–99.7% for its Agree votes
([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793#issuecomment-6442681734)).
Those figures rest on crowd votes, which are noisy. This note sets up the one measurement that does not: an expert
re-labels a random sample blind. It also breaks the crowd comparison down by label severity and by tag.

Script: [`analysis/ai_disagree_relabel.py`](../analysis/ai_disagree_relabel.py) (design and usage in its
docstring); pull: [`analysis/pull_relabel_pool.sh`](../analysis/pull_relabel_pool.sh). Inputs and outputs have
label-level rows from private deployments and stay outside the repo.

## The pool (pulled 2026-10-27)

Every non-deleted, non-tutorial label the `SidewalkAI` account has voted on, across all city schemas: 987,571
labels, of which 529,298 Agree, 91,626 Disagree and 366,647 Unsure (an abstention below the 0.92 gate).

## The re-label packet

300 labels, 20 per label type in each of three groups:

| Group | What it is | Why |
|---|---|---|
| `lone_disagree` | AI Disagree, no human vote | Hidden on the AI's vote alone; their precision decides whether to restore them |
| `voted_disagree` | AI Disagree, ≥ 2 decisive human votes with a majority | Ties the re-label to the crowd numbers and measures the crowd's own error |
| decoys | AI Agree (Obstacle: 7 Agree + 13 Unsure) | Without them every item is an AI Disagree and the rater knows the verdict |

Obstacle has only 7 AI Agree votes in production (its old map never lets an Agree clear the gate), so its decoys
are topped up with labels the AI abstained on; those are scored apart and give no AI precision. Labels the rater has
already voted on are excluded. The rater sees the site's own crop of the label (what Validate shows), a ring at the
label's position from `label_crop.marker_x/y`, the label type and a Street View link to the same pano; nothing about
the AI, the crowd, the city or the labeler. Ids are random and the order is shuffled with a secret seed; the key's
sha256 is `28a36aefe4347b2110eab5bf3d0429dd4b26440591cc5489708d59bb6d419601`.

At 100 items per group, a precision near 65% carries a 95% interval of about ±9 points, enough to tell it from the
0.92 gate.

**Result:** pending the rating ([#58](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/58)).

## Severity and tags, against the crowd

How often a human majority of at least two decisive votes confirms the AI's Agree and Disagree votes, per label type
and severity, and per label type and tag (a label with several tags counts once per tag). Wilson 95% intervals;
cells under 100 votes of a verdict are blank. The last column counts labels currently hidden by an AI Disagree with
no human vote. Same caveat as the production numbers: Triage routes AI-contested labels to humans, so the
human-voted labels lean toward hard cases.

What stands out:

- **Agree holds up everywhere**: 88% or more in every cell, 96–99.7% in most.
- **Crosswalk Disagree on severity-1 crosswalks is right 33% of the time**, rising to 78% at severity 3. By tag,
  `paint fading` (36.5%) and `brick/cobblestone` (37%) are the worst, and `paint fading` alone accounts for 2,711
  lone-hidden crosswalks.
- **CurbRamp Disagree is below 50% at severities 1 and 2**, and worst on `surface problem` (38%), `missing tactile
  warning` (40%) and `narrow` (42%). One reading, not tested here: the AI may take a flawed ramp for no ramp.
- **Obstacle Disagree falls with severity** (73% → 59%), so the AI is most often wrong on the obstacles that matter
  most; `parked scooter/motorcycle` is the worst tag (54%).
- **SurfaceProblem Disagree is right 84% on untagged labels** and 54–68% on tagged ones.

### By severity (human majority of at least 2; cells under 100 votes left blank)

| Type | Severity | AI Agree confirmed | n | AI Disagree confirmed | n | hidden by a lone AI Disagree |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | 1 | 99.3% [99, 99] | 53,156 | 48.9% [48, 50] | 5,509 | 7,297 |
| CurbRamp | 2 | 98.6% [98, 99] | 4,838 | 48.9% [46, 52] | 1,318 | 1,773 |
| CurbRamp | 3 | 97.2% [96, 98] | 1,510 | 57.6% [54, 61] | 879 | 815 |
| NoCurbRamp | 1 | 91.2% [90, 92] | 2,226 | 77.6% [76, 79] | 1,980 | 628 |
| NoCurbRamp | 2 | 95.0% [95, 95] | 9,174 | 68.6% [67, 70] | 2,633 | 1,234 |
| NoCurbRamp | 3 | 96.3% [96, 97] | 16,059 | 68.1% [67, 70] | 4,016 | 1,651 |
| Obstacle | 1 |  | 0 | 73.4% [71, 76] | 1,511 | 1,836 |
| Obstacle | 2 |  | 3 | 65.0% [62, 68] | 781 | 1,698 |
| Obstacle | 3 |  | 1 | 58.7% [55, 62] | 772 | 2,357 |
| SurfaceProblem | 1 | 96.1% [96, 96] | 11,005 | 65.4% [64, 67] | 3,397 | 2,274 |
| SurfaceProblem | 2 | 96.4% [96, 97] | 7,151 | 64.2% [62, 67] | 1,412 | 558 |
| SurfaceProblem | 3 | 98.0% [98, 98] | 3,968 | 64.2% [60, 68] | 625 | 382 |
| Crosswalk | 1 | 99.7% [100, 100] | 21,646 | 33.4% [31, 36] | 1,210 | 2,341 |
| Crosswalk | 2 | 98.7% [98, 99] | 745 | 53.7% [49, 59] | 395 | 1,155 |
| Crosswalk | 3 | 96.3% [92, 98] | 163 | 78.1% [75, 81] | 603 | 1,267 |

### By tag (human majority of at least 2; cells under 100 votes left blank)

| Type | Tag | AI Agree confirmed | n | AI Disagree confirmed | n | hidden by a lone AI Disagree |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | (none) | 99.4% [99, 99] | 41,532 | 53.5% [52, 55] | 3,832 | 5,511 |
| CurbRamp | debris / pooled water | 99.2% [98, 100] | 733 |  | 93 | 160 |
| CurbRamp | missing tactile warning | 98.6% [98, 99] | 10,332 | 40.0% [38, 42] | 2,248 | 2,481 |
| CurbRamp | narrow | 98.5% [98, 99] | 2,841 | 41.7% [38, 45] | 741 | 743 |
| CurbRamp | not aligned with crosswalk | 100.0% [99, 100] | 268 |  | 38 | 80 |
| CurbRamp | not enough landing space | 98.7% [98, 99] | 2,509 | 53.7% [49, 58] | 471 | 379 |
| CurbRamp | not level with street | 96.5% [95, 97] | 1,531 | 61.9% [59, 65] | 879 | 1,280 |
| CurbRamp | parallel lines | 100.0% [97, 100] | 121 |  | 5 | 7 |
| CurbRamp | points into traffic | 99.2% [99, 99] | 5,259 | 54.9% [52, 58] | 849 | 793 |
| CurbRamp | steep | 97.5% [96, 98] | 550 | 59.2% [55, 64] | 453 | 335 |
| CurbRamp | surface problem | 98.9% [98, 99] | 2,611 | 37.8% [34, 42] | 619 | 1,011 |
| NoCurbRamp | (none) | 95.3% [95, 96] | 9,165 | 78.1% [77, 79] | 3,898 | 2,064 |
| NoCurbRamp | alternate route present | 92.0% [91, 93] | 4,551 | 67.4% [65, 69] | 2,082 | 462 |
| NoCurbRamp | no alternate route | 96.9% [97, 97] | 14,012 | 61.9% [60, 64] | 2,625 | 1,589 |
| NoCurbRamp | unclear if needed | 88.1% [86, 90] | 916 | 76.4% [74, 79] | 979 | 273 |
| Obstacle | (none) |  | 3 | 72.1% [68, 76] | 566 | 936 |
| Obstacle | fire hydrant |  | 0 | 73.3% [64, 81] | 105 | 47 |
| Obstacle | narrow |  | 0 | 61.5% [55, 68] | 213 | 340 |
| Obstacle | parked car |  | 1 | 73.5% [68, 78] | 260 | 308 |
| Obstacle | parked scooter/motorcycle |  | 0 | 53.8% [46, 62] | 143 | 629 |
| Obstacle | pole |  | 0 | 68.2% [65, 71] | 817 | 1,415 |
| Obstacle | sign |  | 0 | 80.0% [73, 86] | 145 | 370 |
| Obstacle | trash/recycling can |  | 0 | 80.6% [74, 86] | 170 | 65 |
| Obstacle | tree |  | 0 | 64.7% [61, 68] | 649 | 1,506 |
| Obstacle | vegetation |  | 0 | 62.0% [57, 67] | 316 | 430 |
| SurfaceProblem | (none) | 93.6% [92, 95] | 1,681 | 84.3% [82, 86] | 948 | 329 |
| SurfaceProblem | brick/cobblestone | 91.5% [87, 95] | 200 | 74.6% [70, 79] | 378 | 143 |
| SurfaceProblem | bumpy | 96.8% [96, 97] | 3,792 | 60.9% [58, 64] | 1,145 | 709 |
| SurfaceProblem | cracks | 97.4% [97, 98] | 9,787 | 55.9% [53, 59] | 870 | 715 |
| SurfaceProblem | debris | 97.1% [94, 99] | 207 |  | 23 | 38 |
| SurfaceProblem | grass | 97.4% [97, 98] | 8,159 | 54.5% [51, 58] | 924 | 471 |
| SurfaceProblem | height difference | 98.4% [98, 99] | 2,389 | 56.6% [49, 64] | 159 | 163 |
| SurfaceProblem | narrow sidewalk | 94.3% [93, 95] | 2,198 | 58.0% [54, 62] | 665 | 442 |
| SurfaceProblem | sand/gravel | 97.5% [96, 99] | 526 |  | 72 | 99 |
| SurfaceProblem | uneven/slanted | 97.2% [97, 98] | 4,164 | 54.0% [50, 58] | 504 | 300 |
| SurfaceProblem | utility panel | 95.5% [94, 97] | 1,018 | 67.9% [65, 71] | 1,067 | 923 |
| SurfaceProblem | very broken | 98.5% [98, 99] | 1,623 | 58.1% [50, 66] | 129 | 81 |
| Crosswalk | (none) | 99.7% [100, 100] | 18,812 | 59.1% [56, 62] | 1,329 | 1,682 |
| Crosswalk | brick/cobblestone |  | 62 | 36.8% [30, 45] | 155 | 410 |
| Crosswalk | broken surface | 99.7% [99, 100] | 689 |  | 91 | 194 |
| Crosswalk | bumpy | 99.4% [98, 100] | 328 | 47.7% [39, 57] | 109 | 382 |
| Crosswalk | no pedestrian priority | 99.1% [97, 100] | 215 |  | 49 | 50 |
| Crosswalk | paint fading | 99.5% [99, 100] | 2,767 | 36.5% [33, 40] | 758 | 2,711 |
| Crosswalk | too close to traffic | 98.9% [97, 100] | 261 |  | 5 | 5 |
| Crosswalk | uneven surface | 98.8% [97, 100] | 340 | 60.0% [50, 69] | 105 | 227 |
| Crosswalk | very long crossing | 98.5% [97, 99] | 734 |  | 71 | 146 |
