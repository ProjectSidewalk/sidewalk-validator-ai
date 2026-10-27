# Masked expert re-label of the AI's Disagree votes, and its severity / tag breakdown (October 2026)

Production's AI Disagree votes agree with human majorities only 56–71% of the time, depending on how strict the
human reference is, against 96–99.7% for its Agree votes
([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793#issuecomment-6442681734)).
Those figures rest on crowd votes, which are noisy. This note sets up the one measurement that does not: an expert
re-labels a random sample with the AI and crowd verdicts masked. It also breaks the crowd comparison down by label severity and by tag.

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
| `lone_disagree` | AI Disagree, no human vote of any kind | Hidden on the AI's vote alone; their precision decides whether to restore them |
| `voted_disagree` | AI Disagree, ≥ 2 decisive human votes with a strict majority | Ties the re-label to the crowd numbers and measures the crowd's own error |
| decoys | AI Agree (Obstacle: 1 Agree + 19 AI Unsure) | Without them every item is an AI Disagree and the rater knows the verdict |

- **Exclusions:** labels the rater has voted on or placed.
- **Obstacle decoys are mostly abstentions.** Production has only 7 Obstacle Agree votes, because its old map never
  lets one clear the gate. Agree decoys get first pass at every city slot, but only one of the seven is in a city
  with an Obstacle Disagree item and has a crop. The other 19 Obstacle decoys are labels the AI abstained on. They
  mask the verdict as well, but they are scored apart and give no AI precision.
- **Decoys are matched on city.** They are drawn city by city in proportion to the Disagree items of the same type,
  so the place, which the imagery and the Street View link show, does not separate decoys from Disagree items. It
  can still hint lone from crowd-voted, which does not reveal the AI's verdict.
- **What the rater sees:** the site's own crop of the label (what Validate shows), a ring at the label's position
  from `label_crop.marker_x/y`, the label type, and a Street View link to the same pano. Nothing about the AI, the
  crowd or the labeler.
- **Masking:** random ids, and the order is shuffled with a secret seed. The key's sha256 is
  `ae3468a6409150c61e45ef358d44b579dd21cae6d01e15db720a3f447aabb860`.

**Precision is reported two ways.** The sample has 20 per type, so its plain rate weights the five types equally.
`score` also gives the population-weighted estimate for each Disagree group. The weights are the eligible
labels per type, for example 10,505 CurbRamp and 3,384 SurfaceProblem among the lone Disagrees, with a stratified
interval. The interval uses the Agresti-Coull adjustment, so a type rated 20 of 20 still carries variance. That
weighted figure is what the restore decision rests on. "Eligible" leaves out labels the rater voted on or placed.
That removes under 1% of the lone Disagrees but 14–25% of the crowd-voted ones per type, so the voted group's
weights describe the labels the rater has not seen. Each per-type rate rests on 20 items, less
any rated can't tell.

**Result:** pending the rating ([#58](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/58)).

## Severity and tags, against the crowd

How often a human majority of at least two decisive votes confirms the AI's Agree and Disagree votes:
- per label type and severity;
- per label type and tag (a label with several tags counts once per tag).

**How to read the tables:**
- Intervals are Wilson 95%.
- Rates on fewer than 100 votes of a verdict are left blank.
- A row is shown when any column reaches 100.
- The last column counts labels currently hidden by an AI Disagree with no human vote.

**Caveats:**
- Same as the production numbers: Triage routes AI-contested labels to humans, so the human-voted labels lean toward
  hard cases.
- The human tallies are every non-AI vote, unfiltered.

These tables were the first instance of a standing analysis: every performance study is to be sliced by severity
and tag alongside its baseline comparison. That is being built as a shared module, separately from this note.

What stands out:

- **Agree holds up everywhere**: 88% or more in every cell, 96–99.7% in most.
- **Crosswalk Disagree on severity-1 crosswalks is right 33% of the time**, rising to 78% at severity 3.
  - By tag, `paint fading` (36.5%) and `brick/cobblestone` (37%) are the worst.
  - `paint fading` alone accounts for 2,711 lone-hidden crosswalks.
- **CurbRamp Disagree is below 50% at severities 1 and 2.** It is worst on `surface problem` (38%), `missing
  tactile warning` (40%) and `narrow` (42%). One reading, not tested here: the AI may take a flawed ramp for no
  ramp.
- **Obstacle Disagree falls with severity** (73% → 59%), so the AI is most often wrong on the obstacles that matter
  most. `parked scooter/motorcycle` is the worst tag (54%).
- **SurfaceProblem Disagree is right 84% on untagged labels** and 54–75% on tagged ones.
- **Labels with no severity are where Disagree does best for four of the five types** (69–81%). Crosswalk is the
  exception: 66% with no severity, against 78% at severity 3.

### By severity (human majority of at least 2; rates with under 100 votes left blank; rows shown when any column reaches 100)

| Type | Severity | AI Agree confirmed | n | AI Disagree confirmed | n | hidden by a lone AI Disagree |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | (none) | 98.3% [97.5, 98.9] | 1,372 | 69.2% [64.3, 73.6] | 373 | 641 |
| CurbRamp | 1 | 99.3% [99.2, 99.4] | 53,156 | 48.9% [47.6, 50.2] | 5,509 | 7,296 |
| CurbRamp | 2 | 98.6% [98.2, 98.9] | 4,838 | 48.9% [46.2, 51.6] | 1,318 | 1,772 |
| CurbRamp | 3 | 97.2% [96.3, 97.9] | 1,510 | 57.6% [54.3, 60.8] | 879 | 815 |
| NoCurbRamp | (none) | 93.5% [91.6, 95.0] | 832 | 80.6% [77.5, 83.4] | 686 | 767 |
| NoCurbRamp | 1 | 91.2% [89.9, 92.3] | 2,226 | 77.6% [75.7, 79.4] | 1,980 | 628 |
| NoCurbRamp | 2 | 95.0% [94.5, 95.4] | 9,174 | 68.6% [66.8, 70.4] | 2,633 | 1,234 |
| NoCurbRamp | 3 | 96.3% [96.0, 96.6] | 16,059 | 68.1% [66.6, 69.5] | 4,016 | 1,651 |
| Obstacle | (none) |  | 0 | 78.3% [72.8, 83.0] | 249 | 571 |
| Obstacle | 1 |  | 0 | 73.4% [71.1, 75.6] | 1,511 | 1,836 |
| Obstacle | 2 |  | 3 | 65.0% [61.6, 68.3] | 781 | 1,698 |
| Obstacle | 3 |  | 1 | 58.7% [55.2, 62.1] | 772 | 2,357 |
| SurfaceProblem | (none) | 94.4% [92.6, 95.8] | 789 | 77.7% [72.8, 81.9] | 318 | 199 |
| SurfaceProblem | 1 | 96.1% [95.8, 96.5] | 11,005 | 65.4% [63.8, 67.0] | 3,397 | 2,274 |
| SurfaceProblem | 2 | 96.4% [95.9, 96.8] | 7,151 | 64.2% [61.7, 66.7] | 1,412 | 558 |
| SurfaceProblem | 3 | 98.0% [97.5, 98.4] | 3,968 | 64.2% [60.3, 67.8] | 625 | 382 |
| Crosswalk | (none) | 98.6% [97.3, 99.3] | 582 | 66.1% [59.5, 72.0] | 218 | 261 |
| Crosswalk | 1 | 99.7% [99.6, 99.8] | 21,646 | 33.4% [30.8, 36.1] | 1,210 | 2,341 |
| Crosswalk | 2 | 98.7% [97.5, 99.3] | 745 | 53.7% [48.7, 58.5] | 395 | 1,155 |
| Crosswalk | 3 | 96.3% [92.2, 98.3] | 163 | 78.1% [74.6, 81.2] | 603 | 1,267 |

### By tag (human majority of at least 2; rates with under 100 votes left blank; rows shown when any column reaches 100)

| Type | Tag | AI Agree confirmed | n | AI Disagree confirmed | n | hidden by a lone AI Disagree |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | (none) | 99.4% [99.3, 99.5] | 41,532 | 53.5% [51.9, 55.1] | 3,832 | 5,510 |
| CurbRamp | debris / pooled water | 99.2% [98.2, 99.6] | 733 |  | 93 | 160 |
| CurbRamp | missing tactile warning | 98.6% [98.4, 98.8] | 10,332 | 40.0% [38.0, 42.1] | 2,248 | 2,480 |
| CurbRamp | narrow | 98.5% [97.9, 98.8] | 2,841 | 41.7% [38.2, 45.3] | 741 | 743 |
| CurbRamp | not aligned with crosswalk | 100.0% [98.6, 100.0] | 268 |  | 38 | 80 |
| CurbRamp | not enough landing space | 98.7% [98.2, 99.1] | 2,509 | 53.7% [49.2, 58.2] | 471 | 379 |
| CurbRamp | not level with street | 96.5% [95.4, 97.3] | 1,531 | 61.9% [58.6, 65.0] | 879 | 1,280 |
| CurbRamp | parallel lines | 100.0% [96.9, 100.0] | 121 |  | 5 | 7 |
| CurbRamp | points into traffic | 99.2% [98.9, 99.4] | 5,259 | 54.9% [51.5, 58.2] | 849 | 792 |
| CurbRamp | steep | 97.5% [95.8, 98.5] | 550 | 59.2% [54.6, 63.6] | 453 | 335 |
| CurbRamp | surface problem | 98.9% [98.4, 99.2] | 2,611 | 37.8% [34.1, 41.7] | 619 | 1,011 |
| NoCurbRamp | (none) | 95.3% [94.8, 95.7] | 9,165 | 78.1% [76.8, 79.4] | 3,898 | 2,064 |
| NoCurbRamp | alternate route present | 92.0% [91.2, 92.8] | 4,551 | 67.4% [65.4, 69.4] | 2,082 | 462 |
| NoCurbRamp | no alternate route | 96.9% [96.6, 97.1] | 14,012 | 61.9% [60.1, 63.8] | 2,625 | 1,589 |
| NoCurbRamp | unclear if needed | 88.1% [85.8, 90.0] | 916 | 76.4% [73.6, 79.0] | 979 | 273 |
| Obstacle | (none) |  | 3 | 72.1% [68.2, 75.6] | 566 | 936 |
| Obstacle | fire hydrant |  | 0 | 73.3% [64.2, 80.9] | 105 | 47 |
| Obstacle | garage entrance |  | 0 |  | 0 | 715 |
| Obstacle | height difference |  | 0 |  | 0 | 126 |
| Obstacle | narrow |  | 0 | 61.5% [54.8, 67.8] | 213 | 340 |
| Obstacle | parked bike |  | 0 |  | 0 | 153 |
| Obstacle | parked car |  | 1 | 73.5% [67.8, 78.5] | 260 | 308 |
| Obstacle | parked scooter/motorcycle |  | 0 | 53.8% [45.7, 61.8] | 143 | 629 |
| Obstacle | pole |  | 0 | 68.2% [64.9, 71.3] | 817 | 1,415 |
| Obstacle | sign |  | 0 | 80.0% [72.8, 85.7] | 145 | 370 |
| Obstacle | trash/recycling can |  | 0 | 80.6% [74.0, 85.8] | 170 | 65 |
| Obstacle | tree |  | 0 | 64.7% [61.0, 68.3] | 649 | 1,506 |
| Obstacle | vegetation |  | 0 | 62.0% [56.6, 67.2] | 316 | 430 |
| SurfaceProblem | (none) | 93.6% [92.3, 94.7] | 1,681 | 84.3% [81.8, 86.5] | 948 | 329 |
| SurfaceProblem | brick/cobblestone | 91.5% [86.8, 94.6] | 200 | 74.6% [70.0, 78.7] | 378 | 143 |
| SurfaceProblem | bumpy | 96.8% [96.2, 97.3] | 3,792 | 60.9% [58.0, 63.7] | 1,145 | 709 |
| SurfaceProblem | cracks | 97.4% [97.0, 97.7] | 9,787 | 55.9% [52.5, 59.1] | 870 | 715 |
| SurfaceProblem | debris | 97.1% [93.8, 98.7] | 207 |  | 23 | 38 |
| SurfaceProblem | grass | 97.4% [97.0, 97.7] | 8,159 | 54.5% [51.3, 57.7] | 924 | 471 |
| SurfaceProblem | height difference | 98.4% [97.8, 98.8] | 2,389 | 56.6% [48.8, 64.1] | 159 | 163 |
| SurfaceProblem | narrow sidewalk | 94.3% [93.2, 95.2] | 2,198 | 58.0% [54.3, 61.7] | 665 | 442 |
| SurfaceProblem | sand/gravel | 97.5% [95.8, 98.6] | 526 |  | 72 | 99 |
| SurfaceProblem | uneven/slanted | 97.2% [96.7, 97.7] | 4,164 | 54.0% [49.6, 58.3] | 504 | 300 |
| SurfaceProblem | utility panel | 95.5% [94.0, 96.6] | 1,018 | 67.9% [65.1, 70.7] | 1,067 | 923 |
| SurfaceProblem | very broken | 98.5% [97.8, 99.0] | 1,623 | 58.1% [49.5, 66.3] | 129 | 81 |
| Crosswalk | (none) | 99.7% [99.6, 99.8] | 18,812 | 59.1% [56.5, 61.8] | 1,329 | 1,682 |
| Crosswalk | brick/cobblestone |  | 62 | 36.8% [29.6, 44.6] | 155 | 410 |
| Crosswalk | broken surface | 99.7% [98.9, 99.9] | 689 |  | 91 | 194 |
| Crosswalk | bumpy | 99.4% [97.8, 99.8] | 328 | 47.7% [38.6, 57.0] | 109 | 382 |
| Crosswalk | no pedestrian priority | 99.1% [96.7, 99.7] | 215 |  | 49 | 50 |
| Crosswalk | paint fading | 99.5% [99.2, 99.7] | 2,767 | 36.5% [33.2, 40.0] | 758 | 2,711 |
| Crosswalk | too close to traffic | 98.9% [96.7, 99.6] | 261 |  | 5 | 5 |
| Crosswalk | uneven surface | 98.8% [97.0, 99.5] | 340 | 60.0% [50.4, 68.9] | 105 | 227 |
| Crosswalk | very long crossing | 98.5% [97.3, 99.2] | 734 |  | 71 | 146 |
