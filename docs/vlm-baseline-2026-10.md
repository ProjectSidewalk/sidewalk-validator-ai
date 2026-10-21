# Open-weight VLMs as zero-shot label validators: first baseline (October 2026)

Issue [#16](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/16) asks how well current open-weight
vision-language models (VLMs) validate Project Sidewalk labels with no training, and whether their explanations are
worth showing. This is the first measurement. Three MLX models ran on a Mac on the same crops the production and
retrained validators see. The five production DINOv2 validators ran on the same crops for comparison. The code is
`analysis/vlm_baseline.py`. Every number below comes from its `report` stage, and
`vlm-baseline-2026-10/provenance.json` pins the inputs, the model revisions and the output tables.

## Short answer

**Zero-shot open-weight VLMs are not a substitute for the production validators, and their explanations
should not go into the API response yet.**

- **Production wins on three of five types.** On the same 500 crops, the best VLM (`gemma4-12b`, 4-bit) reaches
  AUC 0.94 on Crosswalk, 0.83 on CurbRamp, 0.70 on SurfaceProblem, 0.62 on NoCurbRamp and 0.50 on Obstacle.
  Production reaches 0.96, 0.93, 0.72, 0.96 and 0.87. The paired bootstrap puts production ahead on CurbRamp,
  NoCurbRamp and Obstacle. On Crosswalk and SurfaceProblem, 100 crops cannot separate the two. Pooled over types,
  the AUCs are 0.72 and 0.89.
- **Scale helps a little, the definition helps a lot.** The mean per-type AUC is 0.63 for the 2B model, 0.66 for
  the 8B and 0.72 for the 12B. Dropping the type definition from the prompt takes the 8B model's CurbRamp AUC
  from 0.76 to 0.56.
- **The VLMs do not drop on held-out deployments; production does.** Pooled AUC on the three held-out deployments
  is 0.84 for production against 0.91 on `test`, and 0.71 against 0.73 for `gemma4-12b`. The production number on
  `test` is probably inflated by training overlap (Caveats).
- **A VLM costs 25 to 30 times production's time per label on this Mac.** Median time per crop is 3.9 s for
  `gemma4-12b` and 0.16 s for production.
- **The explanations read fluently but are not yet trustworthy.** 86 reasons from `gemma4-12b` (median 27 words)
  are in a blinded packet for rating. Some argue against the model's own answer: one says a utility pole is "an
  obstacle because it sits directly in the pedestrian path" next to the answer No. Whether such reasons help a
  validator is the rating's question.

## Coverage

| Model | Crops scored | Types | Notes |
|---|---|---|---|
| production (DINOv2-L, 2025-08-16 revisions) | 1,988 (all) | all five | also reported on the 500-crop headline subset |
| `qwen3vl-2b` | 500 headline subset (+171 more from an interrupted full run, unused) | all five | |
| `qwen3vl-8b` | 500 headline subset | all five | + the no-definition prompt on the 100 CurbRamp crops |
| `gemma4-12b` | 500 headline subset | all five | |
| Few-shot prompting | not run | — | needs in-context crops from the round-3 `train` split, which are not on this machine (they would have to be pulled from klone) |
| Frontier models (Claude, Gemini) | not run | — | out of scope here; designed in [#47](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/47) |

The VLMs scored a subset because the two larger ones took 4 to 6 seconds of wall time per crop on this Mac
(median latency 3.9 s and 4.8 s), so the full 1,988 crops would have taken about three hours each. The subset is `balanced_subset(per_type=100)`: per label type, 30
correct + 30 incorrect from `test` and 20 + 20 from `test_heldout`, sampled with seed 20261021 after sorting by
`label_uid`. Every VLM scored the same 500 crops.

## How the numbers were made

**Data.** The held-out round-3 sample (`~/Data/sidewalk-validator-ai-retrain-2026-10/round3/sample.csv`, 1,988
rows, 41 deployments): per type up to 125 + 125 `test` and 75 + 75 `test_heldout` rows, class-balanced by
construction. `test_heldout` comes from three deployments that were never in the round-3 training data. One is
public (`tucson-az`) and two are private. The crops are the round-3 training inputs: WebP q95, 640 x 640,
label-centred, tilt-corrected and depth-sized. The label point is the crop centre. The target is `label_class` from
the human-vote margin rule, with `incorrect` as the positive class, as in `evaluate.py`.

**VLM score.** One prompt per crop, batch size 1, greedy decoding, no sampling. The score is read from the
next-token distribution after the prompt: `p_incorrect = P(no) / (P(yes) + P(no))`. P(yes) sums the probability of
every single-token spelling of yes the tokenizer has (`yes`, `Yes`, `YES`, ` yes`, ` Yes`, ` YES`; all six are
single tokens in both tokenizers), and likewise for no. It is computed from the log-odds, so it stays resolved when
one answer holds almost all the mass. mlx-vlm returns bfloat16 log-probabilities, so scores are quantised at about
three significant digits. That creates some ties, which AUC counts as half. The verdict at the argmax is
`incorrect` when `p_incorrect > 0.5`.

**Greedy check.** On the first 10 crops of each run, greedy generation's first word was compared with the score's
argmax:

| Model | Crops | Greedy word = score argmax |
|---|---|---|
| `qwen3vl-2b` | 10 (20-crop pilot) | 10 |
| `qwen3vl-8b` | 10 | 9 |
| `gemma4-12b` | 10 | 10 |

The one disagreement is not a bug. The single most likely token was a lower-case `no`, but `Yes` + `yes` together
outweighed `No` + `no`. Summing the spellings can flip the verdict against the single top token. A unit test
(`test_summed_variants_can_disagree_with_greedy`) pins that case, and another
(`test_greedy_agrees_with_score_argmax_on_a_stub`) checks agreement on a one-token-per-answer stub.

**Prompts.** Each prompt is a definition of the type, condensed from the Project Sidewalk labeling guide to three
sentences or fewer, then "The labelled point is at the centre of the image.", then "Is this a correct <type> label?
Answer yes or no." The chat template puts the image first. Thinking is switched off where the template has a switch.

| Type | Prompt (the `definition` variant; `no-definition` drops the first part) |
|---|---|
| CurbRamp | A curb ramp is a short ramp cut into or built up to a curb so that people using wheelchairs, walkers or strollers can move between the sidewalk and the street, often with yellow tactile paving at the bottom. Driveways, alleys and flush building entrances are not curb ramps. The labelled point is at the centre of the image. Is this a correct curb ramp label? Answer yes or no. |
| NoCurbRamp | A missing curb ramp label marks a place where a curb ramp is needed but absent: a sidewalk meets a street crossing or corner at a raised curb, with no ramp for wheelchair users. The label is wrong if a curb ramp is actually present, or if no pedestrian route crosses there. The labelled point is at the centre of the image. Is this a correct missing curb ramp label? Answer yes or no. |
| Obstacle | An obstacle label marks an object in the sidewalk's pedestrian path that blocks it or narrows it so a wheelchair user could not easily pass, such as a pole, sign, fire hydrant, tree, bin, parked vehicle or overgrown vegetation. An object beside the path that leaves it clear is not an obstacle. The labelled point is at the centre of the image. Is this a correct obstacle label? Answer yes or no. |
| SurfaceProblem | A surface problem label marks damage to the sidewalk's walking surface that makes it hard to roll or walk on, such as large cracks, uplifted or sunken slabs (often from tree roots), bumps, missing or broken pavement, loose gravel or dirt, or grass growing over the sidewalk. The labelled point is at the centre of the image. Is this a correct surface problem label? Answer yes or no. |
| Crosswalk | A crosswalk label marks a designated place for pedestrians to cross a road, usually painted stripes or parallel lines on the roadway surface, sometimes raised or textured paving. The label is wrong if there is no marked crossing at the labelled point. The labelled point is at the centre of the image. Is this a correct crosswalk label? Answer yes or no. |

The explanation is a second turn after the model's own answer (`Yes.` or `No.`, its argmax): "In at most 30 words, why?", greedy, at most 60 new tokens.

**Production baseline.** The five validators at the revisions sidewalk-ai-api pins (`model_map.py`, trained
2025-08-16), preprocessed as `evaluate.py` does (`AutoImageProcessor.from_pretrained(..., use_fast=False)`,
`id2label = {0: correct, 1: incorrect}`). They ran on MPS with transformers 4.55.0 and torch 2.14.1, in a separate
venv, because mlx-vlm upgrades transformers.

**Metrics.** `evaluate.slice_metrics` for each slice: AUC, balanced accuracy at the argmax, and recall at
precision 0.95 and 0.92 for each class (`R@P95(incorrect)` is the share of incorrect labels a model could flag at
95% precision). Per-city tables pool all five types, name a deployment only if `calibration.per_city.public_name`
lists it as public, require n >= 30, and pool every private deployment into one row. The AUC intervals are a paired
bootstrap by label, stratified by class, with 1,000 resamples and seed 20261021. "Best VLM" is the one with the
highest mean per-type AUC on the same 500 crops, so it was picked on the data it is reported on.

**Hardware and versions.** Apple M1 Max, 32 GB, macOS 15.5, Python 3.12.14; mlx 0.32.4, mlx-vlm 0.7.10,
transformers 5.20.2 (VLM venv); torch 2.14.1, transformers 4.55.0 (production venv). The exact repo revisions and
on-disk sizes are in `provenance.json`.

## Results

All tables except the last are on the same 500-crop headline subset. They come from `headline.csv`,
`bootstrap_auc.csv`, `by_split.csv`, `per_city.csv`, `hard.csv`, `latency.csv`, `ablation.csv` and `full_sample.csv`
in `vlm-baseline-2026-10/`.

### AUC per type (incorrect = positive)

| Model | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | All (pooled) |
|---|---|---|---|---|---|---|
| production | 0.93 | 0.96 | 0.87 | 0.72 | 0.96 | 0.89 |
| gemma4-12b | 0.83 | 0.62 | 0.50 | 0.70 | 0.94 | 0.72 |
| qwen3vl-8b | 0.76 | 0.46 | 0.51 | 0.66 | 0.90 | 0.68 |
| qwen3vl-2b | 0.57 | 0.55 | 0.49 | 0.62 | 0.92 | 0.61 |

![ROC per label type: production (solid) and the three VLMs (dashed), on the 500-crop subset](vlm-baseline-2026-10/roc.png)

### Balanced accuracy at the argmax

| Model | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | All (pooled) |
|---|---|---|---|---|---|---|
| production | 0.90 | 0.89 | 0.72 | 0.64 | 0.87 | 0.80 |
| gemma4-12b | 0.61 | 0.51 | 0.48 | 0.65 | 0.85 | 0.62 |
| qwen3vl-8b | 0.66 | 0.46 | 0.52 | 0.60 | 0.83 | 0.61 |
| qwen3vl-2b | 0.50 | 0.50 | 0.49 | 0.51 | 0.85 | 0.57 |

The VLMs' answers are lopsided by type, so their balanced accuracy at the argmax is near 0.5 even where the score
ranks crops better than chance. On the subset, `gemma4-12b` answers "no" to 89% of CurbRamp crops and 93% of
NoCurbRamp crops, and `qwen3vl-8b` answers "yes" to 96% of Obstacle crops. Production's share of `incorrect`
verdicts is 38–52% on every type. A per-type threshold would fix the bias, but it would have to be fitted on
labelled data, so the result would no longer be zero-shot.

### Best VLM vs production, paired bootstrap

The best VLM is `gemma4-12b`, which has the highest mean per-type AUC (0.72, against 0.66 for `qwen3vl-8b` and
0.63 for `qwen3vl-2b`).

| Type | gemma4-12b AUC (95% CI) | production AUC (95% CI) | Difference (95% CI) |
|---|---|---|---|
| CurbRamp | 0.83 (0.74–0.90) | 0.93 (0.87–0.98) | -0.10 (-0.19 to -0.02) |
| NoCurbRamp | 0.62 (0.51–0.73) | 0.96 (0.92–0.99) | -0.34 (-0.45 to -0.23) |
| Obstacle | 0.50 (0.38–0.62) | 0.87 (0.79–0.93) | -0.37 (-0.48 to -0.26) |
| SurfaceProblem | 0.70 (0.59–0.79) | 0.72 (0.62–0.82) | -0.02 (-0.15 to +0.09) |
| Crosswalk | 0.94 (0.89–0.98) | 0.96 (0.93–0.99) | -0.03 (-0.07 to +0.01) |

Production is ahead on CurbRamp, NoCurbRamp and Obstacle, with intervals that exclude zero. On SurfaceProblem and
Crosswalk the intervals include zero. On those two types, 100 crops cannot separate `gemma4-12b` from production.

### Recall at fixed precision (both classes)

| Model | Type | R@P95 incorrect | R@P92 incorrect | R@P95 correct | R@P92 correct |
|---|---|---|---|---|---|
| production | CurbRamp | 0.66 | 0.76 | 0.00 | 0.54 |
| production | NoCurbRamp | 0.84 | 0.86 | 0.68 | 0.76 |
| production | Obstacle | 0.28 | 0.32 | 0.64 | 0.64 |
| production | SurfaceProblem | 0.20 | 0.24 | 0.12 | 0.12 |
| production | Crosswalk | 0.80 | 0.82 | 0.82 | 0.88 |
| production | all (pooled) | 0.46 | 0.55 | 0.03 | 0.45 |
| gemma4-12b | CurbRamp | 0.22 | 0.28 | 0.42 | 0.42 |
| gemma4-12b | NoCurbRamp | 0.06 | 0.06 | 0.02 | 0.02 |
| gemma4-12b | Obstacle | 0.04 | 0.04 | 0.02 | 0.02 |
| gemma4-12b | SurfaceProblem | 0.18 | 0.24 | 0.04 | 0.04 |
| gemma4-12b | Crosswalk | 0.58 | 0.84 | 0.50 | 0.60 |
| gemma4-12b | all (pooled) | 0.17 | 0.20 | 0.04 | 0.06 |

The 50/50 prior makes precision easier to reach than it would be in production. Even so, `gemma4-12b` cannot flag
more than about a fifth of incorrect labels at 95% precision on any type except Crosswalk (0.58).

### By split

| Model | Split | n | AUC | Balanced accuracy | R@P95 incorrect |
|---|---|---|---|---|---|
| production | test | 300 | 0.91 | 0.83 | 0.57 |
| production | test_heldout | 200 | 0.84 | 0.77 | 0.27 |
| gemma4-12b | test | 300 | 0.73 | 0.65 | 0.17 |
| gemma4-12b | test_heldout | 200 | 0.71 | 0.58 | 0.24 |
| qwen3vl-8b | test | 300 | 0.69 | 0.64 | 0.03 |
| qwen3vl-8b | test_heldout | 200 | 0.65 | 0.57 | 0.07 |
| qwen3vl-2b | test | 300 | 0.61 | 0.58 | 0.02 |
| qwen3vl-2b | test_heldout | 200 | 0.61 | 0.56 | 0.01 |

`test_heldout` comes from three deployments that were never in the round-3 training data. Production loses 0.07 AUC
there. The VLMs lose 0.02 to 0.04, or nothing, which is expected since they were not trained on any of these
deployments.

### Per deployment (all types pooled, n >= 30)

| Model | Deployment | n | AUC | Balanced accuracy |
|---|---|---|---|---|
| production | seattle-wa | 71 | 0.94 | 0.86 |
| production | chicago-il | 52 | 0.91 | 0.82 |
| production | other (8 private deployments) | 259 | 0.89 | 0.82 |
| gemma4-12b | seattle-wa | 71 | 0.73 | 0.62 |
| gemma4-12b | chicago-il | 52 | 0.66 | 0.66 |
| gemma4-12b | other (8 private deployments) | 259 | 0.74 | 0.63 |
| qwen3vl-8b | seattle-wa | 71 | 0.65 | 0.60 |
| qwen3vl-8b | chicago-il | 52 | 0.71 | 0.71 |
| qwen3vl-8b | other (8 private deployments) | 259 | 0.68 | 0.59 |
| qwen3vl-2b | seattle-wa | 71 | 0.66 | 0.59 |
| qwen3vl-2b | chicago-il | 52 | 0.46 | 0.48 |
| qwen3vl-2b | other (8 private deployments) | 259 | 0.65 | 0.61 |

Only two public deployments have 30 or more rows in the subset. The other public deployments are omitted, and the
private ones are pooled into one row.

### The `hard` slice

| Slice | Model | n (incorrect) | Balanced accuracy | AUC |
|---|---|---|---|---|
| hard == 1, all types, subset | gemma4-12b | 6 (6) | 0.67 |  |
| hard == 1, all types, subset | production | 6 (6) | 0.33 |  |
| hard == 1, all types, subset | qwen3vl-2b | 6 (6) | 0.50 |  |
| hard == 1, all types, subset | qwen3vl-8b | 6 (6) | 0.67 |  |
| hard == 1, all types, full sample | production | 20 (10) | 0.20 | 0.22 |

`hard` marks labels where the deployed AI was wrong against a unanimous crowd or an expert, which explains
production's 0.20 on the full sample. Six rows in the subset are too few to read anything into.

### Prompt ablation: the definition matters

| Model | Type | Prompt | n | AUC | Balanced accuracy |
|---|---|---|---|---|---|
| qwen3vl-8b | CurbRamp | definition | 100 | 0.76 | 0.66 |
| qwen3vl-8b | CurbRamp | no-definition | 100 | 0.56 | 0.55 |

Removing the definition from the CurbRamp prompt takes `qwen3vl-8b` from AUC 0.76 to 0.56 on the same 100 crops.
This was one model and one type, chosen before any results were seen.

### Latency and size (this Mac, batch size 1)

| Model | Prompt | n | Median s/crop | p90 s/crop | Size on disk (GB) | Median P(yes)+P(no) |
|---|---|---|---|---|---|---|
| gemma4-12b | definition | 500 | 3.94 | 5.62 | 11.0 | 0.98 |
| production | definition | 500 | 0.16 | 0.23 | 1.2 |  |
| qwen3vl-2b | definition | 500 | 1.36 | 1.49 | 1.8 | 0.98 |
| qwen3vl-8b | definition | 500 | 4.75 | 5.40 | 5.8 | 0.99 |
| qwen3vl-8b | no-definition | 100 | 2.45 | 2.59 | 5.8 | 0.99 |

The last column is the median share of the next-token distribution held by the yes and no spellings. The models
answer in the requested format almost always.

### Production on all 1,988 crops

| Type | n | AUC | Balanced accuracy | R@P95 incorrect | R@P95 correct |
|---|---|---|---|---|---|
| Crosswalk | 388 | 0.96 | 0.91 | 0.85 | 0.78 |
| CurbRamp | 400 | 0.93 | 0.87 | 0.71 | 0.20 |
| NoCurbRamp | 400 | 0.91 | 0.83 | 0.49 | 0.48 |
| Obstacle | 400 | 0.83 | 0.73 | 0.12 | 0.00 |
| SurfaceProblem | 400 | 0.82 | 0.72 | 0.15 | 0.26 |
| all (pooled) | 1988 | 0.89 | 0.81 | 0.35 | 0.12 |

Production's pooled AUC is 0.89 on all 1,988 crops and 0.89 on the subset (0.885), so the subset is
representative for production.

## Explanations packet (for rating)

The packet asks the best VLM, `gemma4-12b`, for a reason, and asks a human rater to judge it. Items come from
public deployments only, within the 500-crop subset. The aim was 20 per type: 10 where the VLM disagrees with the
crowd, and 10 where the VLM agrees with the crowd but production does not. A short pool was filled with any other
crop where either model disagrees with the crowd.

| Type | VLM disagrees with crowd | VLM agrees, production disagrees | Fallback (any disagreement) | Total |
|---|---|---|---|---|
| CurbRamp | 10 | 1 | 6 | 17 |
| NoCurbRamp | 10 | 3 | 7 | 20 |
| Obstacle | 10 | 4 | 6 | 20 |
| SurfaceProblem | 10 | 10 | 0 | 20 |
| Crosswalk | 6 | 3 | 0 | 9 |
| **All** | **46** | **21** | **19** | **86** |

There are 86 items, not 100. Only 48% of subset rows come from public deployments, and among those, Crosswalk and
CurbRamp did not have enough disagreements to fill their quotas. Reaching 100 would mean scoring more public crops
with `gemma4-12b`, about 4 s each. The second category is small for the same reason: production rarely gets a
public subset crop wrong while the VLM gets it right. In practice, every fallback item is one where the VLM is
wrong, so the packet leans toward the VLM's mistakes.

The packet follows the design of PR #40's packet. Each item has a random `p<6 hex>` id. Items are shuffled with a
secret seed that has no default. The packet holds only the crop, the label type, the VLM's answer and its reason.
The category, the crowd class, the scores, the deployment and the label id are all in the key, which is kept
outside the repo with the seed (`~/Data/sidewalk-validator-ai-vlm-2026-10/private-key-do-not-share/`). The packet
is `~/Data/sidewalk-validator-ai-vlm-2026-10/vlm-explanations-packet.zip`, which holds the 86 item images,
`coding_sheet.csv` and `RUBRIC.md`.

- Key sha256: `5f5f5f37b7bb0a79bbc84b4891d2dc71a33d95a009bea41d738ce0f75d7bfa3b` (`vlm-baseline-2026-10/packet_key.sha256`)
- Median reason length 27 words (range 19–48; the 30-word limit is not always obeyed). Median time to generate a
  reason: 4.5 s.

**Rubric** (also `RUBRIC.md` in the packet). Each item shows the crop with a cyan ring at the label point, the
label type, the model's yes/no answer and its reason. One row per item in `coding_sheet.csv`:

| Column | Values | Meaning |
|---|---|---|
| `reason_correct` | `yes` / `partly` / `no` | Does the reason truly describe what the crop shows, and does it bear on the question? `partly`: some of it is right, or it is right but misses the deciding feature |
| `would_change_my_vote` | `yes` / `no` | Had you been validating this label, would this answer and reason have changed your vote? |
| `my_verdict` | `correct` / `incorrect` / `cant_tell` | Your own verdict, from the crop alone |

Rate the reason, not the answer. A wrong answer can come with a correct description.
`python analysis/vlm_baseline.py agreement <filled sheet> --key <key> --key-sha256 docs/vlm-baseline-2026-10/packet_key.sha256`
unblinds the sheet and prints rater-vs-VLM and rater-vs-crowd agreement, and the reason-code rates, overall and per
category.

## Should explanations go into the API response? What would a VLM cost?

**Not yet.** There are three reasons.

1. **The verdicts behind the explanations are weak.** An explanation attached to an AI vote would come from a model
   that is near chance on NoCurbRamp and Obstacle. A fluent reason for a wrong verdict is worse than no reason,
   because validators may defer to it. The packet's `would_change_my_vote` and `reason_correct` columns measure
   exactly this risk. Ship nothing until they are coded.
2. **The reasons are post hoc.** The reason is generated after the answer and conditioned on it, so it is a
   rationalisation, not a trace of how the model decided. It also comes from a different model than the DINOv2
   validator that casts the vote. Putting it next to a DINOv2 verdict would explain a decision the VLM did not
   make. The honest version is "VLM verdict + VLM reason" as a separate, labelled signal.
3. **The cost is out of proportion to the API's volume.** SidewalkWebpage's nightly actor sends up to 800 labels
   per city. Measured on this Mac at batch size 1:

| Per label | production (DINOv2-L) | `gemma4-12b` verdict | `gemma4-12b` verdict + reason |
|---|---|---|---|
| Median seconds | 0.16 | 3.9 | 8.5 (3.9 + 4.5) |
| One city, 800 labels | 2.1 min | 53 min | 1.9 h |
| 41 public deployments at 800 each (32,800 labels) | 1.4 h | 36 h | 77 h |
| Model size on disk | 1.2 GB per type (5 models) | 11.0 GB, one model for all types | same |

These are Mac numbers, not serving numbers, and a GPU server with batching would cut every column. But the ratio of
about 25x for the verdict and 55x with a reason is a property of the model sizes and the token counts (about 350
image+prompt tokens for Gemma, 480 for Qwen, plus about 40 generated tokens for a reason), so it would carry over.
One VLM would replace five classifiers, which is an operational simplification, but not at this accuracy.

**What would change the answer:** a VLM (open or frontier) that matches production's AUC on these crops, and a
coded packet showing that its reasons are mostly correct (`reason_correct = yes`) and that they move validators
toward the crowd verdict rather than away from it. A cheaper use is offline: run the VLM only on the labels
production abstains on (below the 0.92 gate), where a second opinion has value and the volume is smaller.

## Caveats

- **The production numbers are optimistic.** The production models were trained in 2025-08 on a label set that was
  not recorded and may overlap these labels. The round-3 retrained checkpoints, which would give a clean
  comparison, do not exist yet. The VLMs have never seen these labels. So the size of the gap is an upper bound.
  The direction is not in doubt, because the VLMs are near chance on four of five types.
- **The class balance is 50/50 by construction.** In production, about 85–90% of labels are correct. Plain
  accuracy on this sample is therefore not a production number, and these tables do not report it. AUC and
  balanced accuracy do not depend on the prior. Recall at a fixed precision does: at the production prior,
  precision for `incorrect` would be lower at every threshold.
- **The subsets are small.** There are 100 crops per type in the headline set, 40 per type from the held-out
  deployments, and 6 `hard` rows in the subset (20 in the full sample). Per-split and per-city rows are noisy, and
  the bootstrap intervals show how noisy.
- **Zero-shot, one prompt.** One wording per type and no few-shot examples. The definition ablation shows the
  wording matters. A tuned prompt, few-shot examples, or a larger model could do better. This baseline measures
  what ships with no work.
- **The crowd verdict is the reference, not ground truth.** It is the human-vote margin rule. The packet's
  `my_verdict` column gives a second reference on 100 items.
- **Latency is on one Mac, at batch size 1.** Other processes shared the machine during some runs, so p90 is
  inflated more than the median. Production serving hardware differs. Treat the ratio between models as the
  finding, not the seconds.
- **Model choice.** The three models were picked by the 32 GB limit and by what mlx-community publishes:
  `Qwen3-VL-2B/8B-Instruct-4bit` and `gemma-4-12B-it-qat-4bit`. Larger open models (Qwen3-VL-32B, Gemma-4-31B) do
  not fit with headroom at 4 bits.

## Next steps

1. **Rate the packet** (`vlm-explanations-packet.zip`), then run `agreement` on the filled sheet. A second rater
   on the same sheet gives inter-rater agreement.
2. **Rerun against the round-3 retrained checkpoints** once they exist. That replaces the production column with a
   clean comparison, with no training overlap.
3. **Try the cheap improvements** on CurbRamp and Obstacle, the types with the largest gaps: few-shot prompting with
   4 `train` crops per type (the crops have to be pulled from klone first), a per-type threshold fitted on `val`,
   and a prompt that asks for the label's own failure modes (for example "is this a driveway?").
4. **Frontier-model legs** (Claude, Gemini) on the same crops, with a cost ledger, are designed in a follow-up
   issue: [#47](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/47).
5. **Score the full 1,988 crops with `gemma4-12b`** (about 2.2 h on this Mac) if the subset's per-split or per-city
   rows turn out to matter for a decision.

## Reproduce

```
S=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; O=~/Data/sidewalk-validator-ai-vlm-2026-10
C=~/Data/sidewalk-validator-ai-prod-2026-10-17/cities.json
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt mlx-vlm==0.7.10
uv venv --python 3.12 .venv-prod && VIRTUAL_ENV=.venv-prod uv pip install -r requirements.txt
.venv-prod/bin/python analysis/vlm_baseline.py baseline --sample-dir $S --out-dir $O --device mps
for m in qwen3vl-2b qwen3vl-8b gemma4-12b; do
  python analysis/vlm_baseline.py score --model $m --subset 100 --check-greedy 10 --sample-dir $S --out-dir $O
done
python analysis/vlm_baseline.py score --model qwen3vl-8b --subset 100 --label-types CurbRamp --variant no-definition \
    --sample-dir $S --out-dir $O
python analysis/vlm_baseline.py report --subset 100 --cities-json $C --sample-dir $S --out-dir $O \
    --doc-dir docs/vlm-baseline-2026-10
python analysis/vlm_baseline.py explain --model <best> --cities-json $C --seed 20261021 --sample-dir $S --out-dir $O
python analysis/vlm_baseline.py packet --packet-dir $O/packet --key $O/private-key-do-not-share/packet_key.csv \
    --seed <secret> --sample-dir $S --out-dir $O
```

The per-crop score CSVs and `explanations.csv` hold label-level rows from private deployments. They stay in
`~/Data/sidewalk-validator-ai-vlm-2026-10/` and their sha256 values are in `provenance.json`. The 10 crops of
`qwen3vl-2b`'s greedy check come from a 20-crop pilot run before the full run, because the resumed run skipped it.
