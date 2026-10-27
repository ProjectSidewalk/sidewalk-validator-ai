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

**Rated on the live site.** Telling whether a missing curb ramp is truly missing often means panning around the
intersection, which a fixed crop cannot do. So the items are rated in Project Sidewalk's Gallery, one review list
per deployment (`/gallery?labelIds=…`, SidewalkWebpage#5444), with a user stylesheet that hides the vote counts,
the AI's vote, the vote tooltips, earlier validations and comments (`GALLERY_MASK_CSS` in the script).

- **The rater's prod vote is the record:** Agree = correct, Disagree = incorrect, Unsure = can't tell. `votes`
  pulls the votes back as a sheet for `score`.
- **Rating changes prod as it goes.** A vote joins the crowd for later analyses, and a lone label stops being lone.
  Scoring uses the groups as they were at sampling time.
- **`rate.html` remains the offline fallback.**

**Replicability.** The run directory keeps everything needed to rebuild and check the lists:
- the 28 review lists in the order shown, with label ids and URLs (`gallery/gallery_urls.csv`, sha256
  `c3bf124966c2a4155398e58a1f0ee32308de9052aab1abf2fb2c147c38cb6c71`);
- the `cityparams.conf` snapshot that mapped schemas to deployments (SidewalkWebpage 38599ca5, sha256
  `9731618a43f90d1b3dde757f19f4420a6dc26c42ec0e69c0eae7572c2e0d246a`);
- the masking CSS (sha256 `904fbf76e99943fb74a017226c59308513b44616b8e7d4d9b2a96c99b52eaec5`);
- the pool, the key and the seed.

The lists name non-public deployments, so they stay outside the repo; the hashes here pin them.

**Precision is reported two ways.** The sample has 20 per type, so its plain rate weights the five types equally.
`score` also gives the population-weighted estimate for each Disagree group. The weights are the eligible
labels per type, for example 10,505 CurbRamp and 3,384 SurfaceProblem among the lone Disagrees, with a stratified
interval. The interval uses the Agresti-Coull adjustment, so a type rated 20 of 20 still carries variance. That
weighted figure is what the restore decision rests on. "Eligible" leaves out labels the rater voted on or placed.
That removes under 1% of the lone Disagrees but 14–25% of the crowd-voted ones per type, so the voted group's
weights describe the labels the rater has not seen. Each per-type rate rests on 20 items, less
any rated can't tell.

**Result:** see [Results](#results-rated-2026-10-27) below.

## Results (rated 2026-10-27)

The rater voted on all 300 items on the live site, with the votes masked while voting, and left comments on 184.
Comments were often added afterwards, with the page unmasked, so they can mention other validators; the votes
were cast masked. The AI votes scored here were cast by the 2025-08 production models, before sidewalk-ai-api
1.2.0 was deployed on 2026-10-27.

**Headline:**
- **AI Disagree, no human votes** (the label is hidden on the AI's vote alone): the expert agrees with the AI
  **61.1%** [50.8, 70.5], **62.9%** [52.4, 73.4] weighted to the population.
- **AI Disagree, crowd-voted:** **54.3%** [44.2, 64.0].
- **AI Agree:** **94.8%** [87.4, 98.0].

So about 4 in 10 labels hidden on a lone AI Disagree are correct labels, and the Disagree precision is far below
the 0.92 the gate targets. On the crowd-voted items the human majority matches the expert 74.5% of the time,
against 54% for the AI.

**Confidence separates right from wrong only weakly.** The expert agrees 50%, 57% and 65% of the time across
served accuracy bands (AUC of `ai_accuracy` for a right Disagree 0.574, Mann-Whitney p = 0.087; both in the report). Within each type, the AI's mean served accuracy differs by at most 0.01 between right and wrong. No
band comes near 0.92.

**Where the AI is wrong:** real features in poor condition. The verdicts show it in the slices (Crosswalk severity
1 at 19%, `paint fading` 33%; CurbRamp `missing tactile warning` 12%, `narrow` 20%; SurfaceProblem
`brick/cobblestone` 20%), and so do the comments: "an incredibly faded crosswalk that took me a long time to
determine if the lines were actually even there. They are!", "not a good curb ramp but it is, I think, a curb
ramp", "an extremely narrow and poorly designed curb ramp". Blurry or washed-out imagery, and labels placed off
the feature, also cause wrong Disagrees. This matches the crowd breakdown below: the AI takes a degraded feature
for a missing one.

**Where it is right:** context errors. The rater's canned comment reasons (the reason-code table in the
report) show it: `driveway` on 11 CurbRamp items, `no-visible-crosswalk` on 14 Crosswalk items, `ample-space` on
14 Obstacle items and `not-pedestrian-path` on 14 Obstacle and SurfaceProblem items, all judged incorrect labels.
Free-text comments add a few more of each. Obstacle Disagree is right 76.5% in both groups.

**Training counts.** Each slice shows its training rows (both classes) for two training sets:
- **2025-08 production:** the models that cast these votes, from the published Hugging Face datasets. CurbRamp is
  blank: its published dataset is a copy of Crosswalk's (audit finding 13), so its rows cannot be recovered.
- **Round 3:** the retrain's `split == "train"` rows.

The weak slices are not thin in training. CurbRamp `missing tactile warning` has 1,482 round-3 rows, and Crosswalk
`paint fading` had 198 in production and has 339 in round 3. Whether these slices improve in round 3 is a
question for the round-3 votes once they accumulate. The production-side slices in
[PR #67](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/67) cover them on the test splits.

**Caveats:**
- One rater. Expert 2 has votes on only 2 of the 300 labels, so there is no inter-rater figure.
- Slices have 1–28 items each and wide intervals; read them as a screen, not estimates.
- Comments skew toward `incorrect` verdicts (110 of 114 commented, against 50 of 162 `correct`), so the failure
  themes rest more on the verdicts than on the comments.
- Obstacle has one Agree decoy, so there is no Obstacle Agree precision.
- The votes are taken as of the pinned pull below. Later changes on prod are not reflected.
- Production training rows carry their severity and tags as of the 2026-10-27 pull, not as of training in
  2025-08; edits since then can move a row between slices. Their label types all still match the datasets.

### Report (generated by `report`; aggregates only)

| Group | Type | Expert agrees with the AI (sample) | n | can't tell | population-weighted | crowd majority agrees with the expert |
|---|---|---:|---:|---:|---:|---:|
| lone_disagree | all | 61.1% [50.8, 70.5] | 90 | 10 | 62.9% [52.4, 73.4] |  |
| voted_disagree | all | 54.3% [44.2, 64.0] | 94 | 6 | 52.5% [41.7, 63.3] | 74.5% [64.8, 82.2] |
| decoy_agree | all | 94.8% [87.4, 98.0] | 77 | 4 |  |  |
| decoy_unsure | all | 73.3% [48.0, 89.1] | 15 | 4 |  |  |
| lone_disagree | CurbRamp | 65.0% [43.3, 81.9] | 20 | 0 |  |  |
| lone_disagree | NoCurbRamp | 64.7% [41.3, 82.7] | 17 | 3 |  |  |
| lone_disagree | Obstacle | 76.5% [52.7, 90.4] | 17 | 3 |  |  |
| lone_disagree | SurfaceProblem | 62.5% [38.6, 81.5] | 16 | 4 |  |  |
| lone_disagree | Crosswalk | 40.0% [21.9, 61.3] | 20 | 0 |  |  |
| voted_disagree | CurbRamp | 42.1% [23.1, 63.7] | 19 | 1 |  | 68.4% [46.0, 84.6] |
| voted_disagree | NoCurbRamp | 55.0% [34.2, 74.2] | 20 | 0 |  | 80.0% [58.4, 91.9] |
| voted_disagree | Obstacle | 76.5% [52.7, 90.4] | 17 | 3 |  | 70.6% [46.9, 86.7] |
| voted_disagree | SurfaceProblem | 50.0% [29.0, 71.0] | 18 | 2 |  | 66.7% [43.7, 83.7] |
| voted_disagree | Crosswalk | 50.0% [29.9, 70.1] | 20 | 0 |  | 85.0% [64.0, 94.8] |
| decoy_agree | CurbRamp | 95.0% [76.4, 99.1] | 20 | 0 |  |  |
| decoy_agree | NoCurbRamp | 89.5% [68.6, 97.1] | 19 | 1 |  |  |
| decoy_agree | Obstacle | 100.0% [20.7, 100.0] | 1 | 0 |  |  |
| decoy_agree | SurfaceProblem | 94.1% [73.0, 99.0] | 17 | 3 |  |  |
| decoy_agree | Crosswalk | 100.0% [83.9, 100.0] | 20 | 0 |  |  |
| decoy_unsure | Obstacle | 73.3% [48.0, 89.1] | 15 | 4 |  |  |

Both Disagree groups pooled: 57.6% [50.4, 64.5] (n = 184). `decoy_unsure` rows are the share the expert called correct (the AI abstained), not an AI precision.

#### AI Disagree precision by type x severity (both Disagree groups, decided items)

| Type | Severity | Expert agrees with the AI | n | training rows: 2025-08 production | training rows: round 3 |
|---|---|---:|---:|---:|---:|
| CurbRamp | 1 | 60.7% [42.4, 76.4] | 28 |  | 5,083 |
| CurbRamp | 2 | 28.6% [8.2, 64.1] | 7 |  | 837 |
| CurbRamp | 3 | 50.0% [15.0, 85.0] | 4 |  | 551 |
| NoCurbRamp | (none) | 85.7% [48.7, 97.4] | 7 | 253 | 691 |
| NoCurbRamp | 1 | 100.0% [43.8, 100.0] | 3 | 918 | 1,839 |
| NoCurbRamp | 2 | 63.6% [35.4, 84.8] | 11 | 1,857 | 3,595 |
| NoCurbRamp | 3 | 37.5% [18.5, 61.4] | 16 | 3,376 | 6,074 |
| Obstacle | (none) | 100.0% [20.7, 100.0] | 1 | 226 | 669 |
| Obstacle | 1 | 70.0% [39.7, 89.2] | 10 | 1,059 | 3,142 |
| Obstacle | 2 | 81.8% [52.3, 94.9] | 11 | 1,025 | 2,844 |
| Obstacle | 3 | 75.0% [46.8, 91.1] | 12 | 1,266 | 3,899 |
| SurfaceProblem | (none) | 75.0% [30.1, 95.4] | 4 | 211 | 658 |
| SurfaceProblem | 1 | 50.0% [29.9, 70.1] | 20 | 2,431 | 5,050 |
| SurfaceProblem | 2 | 71.4% [35.9, 91.8] | 7 | 1,349 | 2,959 |
| SurfaceProblem | 3 | 33.3% [6.1, 79.2] | 3 | 775 | 1,833 |
| Crosswalk | (none) | 80.0% [37.6, 96.4] | 5 | 79 | 139 |
| Crosswalk | 1 | 18.8% [6.6, 43.0] | 16 | 722 | 1,394 |
| Crosswalk | 2 | 40.0% [16.8, 68.7] | 10 | 103 | 187 |
| Crosswalk | 3 | 77.8% [45.3, 93.7] | 9 | 204 | 369 |

#### AI Disagree precision by type x tag (both Disagree groups, decided items)

| Type | Tag | Expert agrees with the AI | n | training rows: 2025-08 production | training rows: round 3 |
|---|---|---:|---:|---:|---:|
| CurbRamp | (none) | 69.6% [49.1, 84.4] | 23 |  | 3,933 |
| CurbRamp | missing tactile warning | 12.5% [2.2, 47.1] | 8 |  | 1,482 |
| CurbRamp | narrow | 20.0% [3.6, 62.4] | 5 |  | 451 |
| CurbRamp | not enough landing space | 33.3% [6.1, 79.2] | 3 |  | 348 |
| CurbRamp | not level with street | 50.0% [18.8, 81.2] | 6 |  | 513 |
| CurbRamp | points into traffic | 0.0% [0.0, 65.8] | 2 |  | 770 |
| CurbRamp | steep | 0.0% [0.0, 79.3] | 1 |  | 226 |
| CurbRamp | surface problem | 33.3% [9.7, 70.0] | 6 |  | 351 |
| NoCurbRamp | (none) | 66.7% [45.4, 82.8] | 21 | 2,568 | 4,631 |
| NoCurbRamp | alternate route present | 75.0% [30.1, 95.4] | 4 | 1,135 | 2,300 |
| NoCurbRamp | no alternate route | 36.4% [15.2, 64.6] | 11 | 2,390 | 4,670 |
| NoCurbRamp | unclear if needed | 50.0% [9.5, 90.5] | 2 | 427 | 860 |
| Obstacle | (none) | 100.0% [43.8, 100.0] | 3 | 459 | 1,933 |
| Obstacle | construction | 100.0% [20.7, 100.0] | 1 | 205 | 416 |
| Obstacle | garage entrance | 100.0% [34.2, 100.0] | 2 | 22 | 95 |
| Obstacle | height difference | 100.0% [20.7, 100.0] | 1 | 16 | 74 |
| Obstacle | parked car | 100.0% [20.7, 100.0] | 1 | 395 | 1,293 |
| Obstacle | parked scooter/motorcycle | 33.3% [6.1, 79.2] | 3 | 81 | 527 |
| Obstacle | pole | 83.3% [43.6, 97.0] | 6 | 767 | 2,087 |
| Obstacle | sign | 0.0% [0.0, 79.3] | 1 | 211 | 486 |
| Obstacle | stairs | 0.0% [0.0, 79.3] | 1 | 14 | 35 |
| Obstacle | trash/recycling can | 100.0% [20.7, 100.0] | 1 | 543 | 1,046 |
| Obstacle | tree | 84.6% [57.8, 95.7] | 13 | 264 | 854 |
| Obstacle | vegetation | 60.0% [23.1, 88.2] | 5 | 392 | 1,136 |
| SurfaceProblem | (none) | 100.0% [56.6, 100.0] | 5 | 500 | 1,490 |
| SurfaceProblem | brick/cobblestone | 20.0% [3.6, 62.4] | 5 | 202 | 337 |
| SurfaceProblem | bumpy | 60.0% [23.1, 88.2] | 5 | 984 | 1,982 |
| SurfaceProblem | cracks | 0.0% [0.0, 79.3] | 1 | 1,435 | 3,031 |
| SurfaceProblem | debris | 0.0% [0.0, 79.3] | 1 | 13 | 66 |
| SurfaceProblem | grass | 33.3% [9.7, 70.0] | 6 | 1,272 | 2,636 |
| SurfaceProblem | narrow sidewalk | 55.6% [26.7, 81.1] | 9 | 526 | 1,160 |
| SurfaceProblem | uneven/slanted | 33.3% [6.1, 79.2] | 3 | 647 | 1,574 |
| SurfaceProblem | utility panel | 60.0% [23.1, 88.2] | 5 | 496 | 904 |
| Crosswalk | (none) | 57.1% [36.5, 75.5] | 21 | 810 | 1,557 |
| Crosswalk | brick/cobblestone | 0.0% [0.0, 56.2] | 3 | 26 | 48 |
| Crosswalk | broken surface | 50.0% [15.0, 85.0] | 4 | 46 | 58 |
| Crosswalk | bumpy | 50.0% [18.8, 81.2] | 6 | 34 | 58 |
| Crosswalk | paint fading | 33.3% [15.2, 58.3] | 15 | 198 | 339 |
| Crosswalk | uneven surface | 100.0% [20.7, 100.0] | 1 | 36 | 64 |
| Crosswalk | very long crossing | 50.0% [9.5, 90.5] | 2 | 31 | 68 |

#### AI Disagree precision by the AI's served accuracy

| ai_accuracy | Expert agrees with the AI | n |
|---|---:|---:|
| (0.9, 0.95] | 50.0% [34.8, 65.2] | 38 |
| (0.95, 0.98] | 57.1% [47.3, 66.5] | 98 |
| (0.98, 1.0] | 64.6% [50.4, 76.6] | 48 |

Pooled, the served accuracy ranks right Disagrees above wrong ones with AUC 0.574 (two-sided Mann-Whitney p = 0.087, n = 184).

| Type | mean ai_accuracy, AI right | mean ai_accuracy, AI wrong | n |
|---|---:|---:|---:|
| Crosswalk | 0.979 | 0.977 | 40 |
| CurbRamp | 0.953 | 0.953 | 39 |
| NoCurbRamp | 0.962 | 0.953 | 37 |
| Obstacle | 0.982 | 0.981 | 34 |
| SurfaceProblem | 0.965 | 0.955 | 34 |

#### Crowd majority against the expert (`voted_disagree`, decided items)

| Crowd says | Expert: correct | Expert: incorrect |
|---|---:|---:|
| correct | 29 | 10 |
| incorrect | 14 | 41 |

#### Comment reason codes on AI Disagree items

| Type | Reason | Expert: correct | Expert: incorrect | Expert: can't tell |
|---|---|---:|---:|---:|
| Crosswalk | no-visible-crosswalk | 0 | 14 | 0 |
| Crosswalk | stop-line | 0 | 1 | 0 |
| CurbRamp | driveway | 0 | 11 | 0 |
| NoCurbRamp | no-sidewalk-here | 0 | 4 | 0 |
| NoCurbRamp | residential-walkway | 0 | 2 | 0 |
| NoCurbRamp | unsafe-crossing | 0 | 2 | 0 |
| Obstacle | ample-space | 0 | 14 | 0 |
| Obstacle | not-pedestrian-path | 0 | 3 | 0 |
| Obstacle | space-to-avoid-unsure | 0 | 0 | 2 |
| SurfaceProblem | better-image | 0 | 0 | 1 |
| SurfaceProblem | not-pedestrian-path | 0 | 11 | 0 |

#### Comments

| Expert verdict | items | with a comment |
|---|---:|---:|
| cant_tell | 24 | 24 |
| correct | 162 | 50 |
| incorrect | 114 | 110 |

### Reproducing

**Rebuild the posted numbers** from the pinned files (nothing is pulled from prod):

```bash
D=~/Data/sidewalk-validator-ai-relabel-2026-10
K=$D/private-key-do-not-share/key.csv
python analysis/ai_disagree_relabel.py score  --key $K --sheet $D/packet/relabel_sheet__jon.csv
python analysis/ai_disagree_relabel.py report --key $K --sheet $D/packet/relabel_sheet__jon.csv \
    --comments $D/packet/jon_comments.csv --prod-train $D/prod_train_labels.csv \
    --train-split ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 \
    --out $D/results_report.md --private-out $D/packet/joined_scored.csv
```

`score` and `report` refuse any input whose sha256 differs from its pin in the key's `manifest.json`: the key
(`key_sha256`), the votes, the comments, the production training rows and the five round-3 `split.csv` files
(`rating`). `--allow-unpinned` downgrades that to a warning and leaves the joined file unpinned.

**How the pinned files were made** (2026-10-27; do not re-run to rebuild, since prod has moved on):

```bash
ssh -fNM makelab1                       # one master; every pull goes through it
python analysis/ai_disagree_relabel.py votes      --key $K --rater-id <rater uuid> --out $D/packet/relabel_sheet__jon.csv
python analysis/ai_disagree_relabel.py comments   --key $K --rater-id <rater uuid> --out $D/packet/jon_comments.csv
python analysis/ai_disagree_relabel.py prod-train --key $K --out $D/prod_train_labels.csv
python analysis/ai_disagree_relabel.py pin --key $K --name round3_splits --file ~/Data/sidewalk-validator-ai-retrain-2026-10/round3
```

Each pull pins its output. A pull refuses to overwrite an existing file, and `pin` refuses to replace a
different pin, unless `--repin` is given. `prod-train` reads the Hugging Face datasets at the revisions already
pinned, so a later push to a dataset does not change it. (The votes and comments were first pulled by hand with
the same queries and pinned afterwards with `pin`; their hashes are the ones below.)

| File (in the run directory) | sha256 |
|---|---|
| `relabel_pool.csv` | `dd5a6d363ce7f6c05ac6faf250f2526090bd2199541e6164b76029cd35483d09` |
| `private-key-do-not-share/key.csv` (v3) | `ae3468a6409150c61e45ef358d44b579dd21cae6d01e15db720a3f447aabb860` |
| the build seed (hash of the seed string) | `9fdd983d978568ef9bb4cb2208b029489ea7e69d2680ece79f436a914b5834a3` |
| `packet/relabel_sheet__jon.csv` (votes, pulled 2026-10-27 22:26 UTC) | `7a758981cffd30abe10bb360dccb5fc1c8321eb5c31710a01571f01d7ff87383` |
| `packet/jon_comments.csv` (184 comments, pulled 22:26 UTC) | `23eeb2da130e7ddfa1964a310341813d4981e67dd2d4a47060020471551d4373` |
| `prod_train_labels.csv` (15,854 rows; none missing or retyped) | `89ba27c6e9f8f33a2f243283ea87fe2f854dadbc5506485a9cf2fc46bcf7d9e0` |
| `packet/joined_scored.csv` (key, verdicts and comments; private) | `145aac54b3e5c219fbd32a085471495c62fbc01e4baa8612ecc465cec21ebc0e` |
| `results_report.md` (the report above) | `e07f1654e4397812aa003bcfacd390b082bf344c347aba1fe1ae1f966c236436` |
| round-3 `Crosswalk/split.csv` | `91001f64b778ec3d59b381d50911c6bd8ea35449052574d97eb4c9380d37f00f` |
| round-3 `CurbRamp/split.csv` | `f2ec89f8032c9b6b7c1533bea7ac82d82b7c139847b14286dd53d932a2195610` |
| round-3 `NoCurbRamp/split.csv` | `fbb1e53bd121bdd5eaca8b12d8fcf5a6ec3f6102920f1f68735eb78c9703ce33` |
| round-3 `Obstacle/split.csv` | `d41e865f45b5f0b91d3bc81c1a60dae5e3effea2d30dd92c7883207feb87e8aa` |
| round-3 `SurfaceProblem/split.csv` | `c48cac6c889b1df2c125238401f8f886904ae2a28c8e43628f5e2e799abd8fdd` |

The gallery lists, `cityparams.conf` snapshot and masking CSS are pinned above under the packet. The Hugging Face
dataset revisions the production training rows were read from:

| Dataset | Revision |
|---|---|
| `sidewalk-validator-ai-dataset-crosswalk` | `1be513a0f8ae34b40f646566b808b6fc048fd635` |
| `sidewalk-validator-ai-dataset-nocurbramp` | `e737c59a8ed7163588bd870016188704e6edb037` |
| `sidewalk-validator-ai-dataset-obstacle` | `9fdc3c30c950455a4978fc991d2757b4f942d4d6` |
| `sidewalk-validator-ai-dataset-surfaceproblem` | `23f32c0758861db075146e861ddd447bb674fc23` |

The joined per-label file (`packet/joined_scored.csv`: key, verdict and comments) is pinned in the manifest; it
holds comment text and label ids from private deployments and stays out of the repo.

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
