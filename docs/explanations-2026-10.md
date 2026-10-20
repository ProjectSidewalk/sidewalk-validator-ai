# What the validators look at: attention rollout and Grad-CAM overlays (October 2026)

Part of [#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19). This page ships the tool, the
sampled overlays, the rubric and the coding sheet for two human raters. The rating itself has **not** been done.
The per-type observations near the end are one model-generated first pass. The raters' coding replaces them.

**Where the numbers come from.** `explain.py` (this PR), run on 2026-10-20 on an M1 Max (macOS 15.5, torch 2.14.1,
transformers 4.55.0). Every count and median below is printed by the commands in [Reproduce](#reproduce).

## Inputs

- **Models.** The production revisions sidewalk-ai-api pins (see the Scope note in
  [calibration-2026-10.md](calibration-2026-10.md)): `projectsidewalk/sidewalk-validator-ai-<type>` at nocurbramp
  `9f3f3bb`, obstacle `8f0fe8f`, surfaceproblem `0e1439d`, crosswalk `b7936e5`. The preprocessing is the saved
  `AutoImageProcessor` with `use_fast=False`, as in `evaluate.py` and the API.
- **Crops.** The `test/` folder of `projectsidewalk/sidewalk-validator-ai-dataset-<type>` at the revisions below. The
  `correct`/`incorrect` folder is the crowd's class. Every file was checked against the repo's LFS sha256.

  | type | dataset revision | test crops |
  |---|---|---:|
  | Crosswalk | `1be513a0f8ae34b40f646566b808b6fc048fd635` | 240 |
  | NoCurbRamp | `e737c59a8ed7163588bd870016188704e6edb037` | 1,374 |
  | Obstacle | `9fdc3c30c950455a4978fc991d2757b4f942d4d6` | 769 |
  | SurfaceProblem | `23f32c0758861db075146e861ddd447bb674fc23` | 1,024 |

- **Caveats on the crops.**
  - This is the old file-level split, which leaks across panoramas
    ([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)). So the error *rates* below are
    optimistic. That does not matter for what the overlays show on a given crop.
  - **CurbRamp is skipped.** Its published dataset is a byte-for-byte copy of the Crosswalk one
    ([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)). Explaining the CurbRamp model on
    crosswalk crops would say nothing about curb ramps.
  - **Private deployments.** The 2025 datasets include crops from deployments that are private today, by
    `cities.json` from the 2026-10-17 pull. Overlays, `samples.csv` and the coding sheet use **public deployments
    only** (`--cities-json`), and their label ids carry the deployment's `city_id` (`seattle-wa:123`). The category
    counts cover every crop and name no city.


## Method

- **Categories.** The positive class is `correct` (the API's Agree). TP = crowd correct, AI correct. FP = crowd
  incorrect, AI correct, so the AI would Agree with a bad label. FN = crowd correct, AI incorrect, so the AI would
  Disagree with a good label. These are raw verdicts, with the API's tie rule (`predict_verdicts`) and **no 0.92
  gate**. Many of them would be recorded as Unsure in production.
- **What the overlays show.** The processor resizes the short side to 256 and centre-crops 224. So the model never
  sees the outer ~6% on each side of the published crop. Every panel is that 224 × 224 view, and a cyan cross marks
  its centre, which is the label point.
- **Attention rollout** ([Abnar & Zuidema 2020](https://arxiv.org/abs/2005.00928)):
  - Head-mean, no discarding, identity added for the residual, rows re-normalised, multiplied through all 24 layers.
  - The map is the CLS row over the 16 × 16 patches.
  - It is **class-agnostic**: TP and FN overlays answer "where does the network route information", not "why this
    verdict".
  - The model is loaded with `attn_implementation="eager"`, because SDPA returns no weights. On 16 Crosswalk crops
    the eager and SDPA probabilities differed by at most 6.1e-7 (CPU, smoke test).
- **Grad-CAM**, adapted to a ViT as [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) does it:
  - Activations are the output of the last block's `norm1`, the tokens entering its self-attention.
  - Weights are the gradient averaged over patches, and the map is the ReLU of the weighted sum.
  - The target is the **logit margin of the predicted class**, `z[pred] − z[other]`, i.e. the log-odds the
    confidence is a function of. So an FN map shows evidence *for Disagree*.
  - `Dinov2ForImageClassification` classifies `[CLS ‖ mean(patch tokens)]` of the final hidden state. The gradient
    reaches the hooked tokens through the last block's attention into CLS and through the mean-pool. The residual
    stream around `norm1` is not part of the hooked activation.
- **Sampling.** Up to 50 per category, seed 42, after sorting by `label_uid`, from public deployments only. Tiles are
  ordered by descending confidence.
- **Device.** The committed overlays were produced on **MPS**. The prediction pass was repeated on CPU for every crop
  (`--check-device cpu`): max |Δp(correct)| was 7.7e-6 (Crosswalk), 2.1e-5 (NoCurbRamp), 1.4e-5 (Obstacle) and 2.8e-5
  (SurfaceProblem), with **0 category flips** in every type. Each overlay's own forward pass reproduced the prediction
  pass within 1.2e-7. On one Crosswalk crop the rollout and Grad-CAM maps from MPS and CPU correlated at r > 0.99999.

## Counts per category (test split, all crops; raw verdicts)

From `docs/explanations-2026-10/<type>_counts.json` (`categories`; `categories_sampling_pool` is the public-only pool the overlays are drawn from).
These counts are on the leaky file-level split (#3), so treat the rates as optimistic.

| type | n | TP | FP | FN | TN | public pool TP / FP / FN | overlays TP / FP / FN |
|---|---:|---:|---:|---:|---:|---|---|
| Crosswalk | 240 | 126 | 16 | 3 | 95 | 66 / 1 / 3 | 50 / 1 / 3 |
| NoCurbRamp | 1,374 | 732 | 103 | 65 | 474 | 700 / 97 / 62 | 50 / 50 / 50 |
| Obstacle | 769 | 376 | 104 | 67 | 222 | 353 / 97 / 48 | 50 / 50 / 48 |
| SurfaceProblem | 1,024 | 546 | 115 | 74 | 289 | 486 / 98 / 50 | 50 / 50 / 50 |

**Crosswalk has 1 FP and 3 FN in the public pool**, so the issue's 50/50/50 is not reachable for it. 15 of its 16 FPs
come from deployments that are private today. Rating them would need Jon's sign-off and must stay out of the repo.

## Automatic map statistics (not a substitute for the raters)

Printed by `python explain.py summary docs/explanations-2026-10/*_samples.csv`. **Centre mass** is the share of a
map inside the central quarter of the image (the middle 8 × 8 patches). A uniform map scores 0.25. **Top-3 mass** is
the share held by the 3 strongest of 256 patches; a uniform map scores 0.012. These are crude: a feature need not be
centred or fill a quarter of the crop.

| type | category | n | median confidence | rollout centre | Grad-CAM centre | rollout top-3 | Grad-CAM top-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Crosswalk | TP | 50 | 0.999 | 0.204 | 0.411 | 0.129 | 0.079 |
| Crosswalk | FP | 1 | 0.999 | 0.180 | 0.480 | 0.143 | 0.078 |
| Crosswalk | FN | 3 | 0.997 | 0.180 | 0.184 | 0.130 | 0.046 |
| NoCurbRamp | TP | 50 | 1.000 | 0.218 | 0.325 | 0.113 | 0.087 |
| NoCurbRamp | FP | 50 | 0.871 | 0.235 | 0.315 | 0.114 | 0.072 |
| NoCurbRamp | FN | 50 | 0.921 | 0.227 | 0.274 | 0.115 | 0.061 |
| Obstacle | TP | 50 | 1.000 | 0.241 | 0.279 | 0.116 | 0.063 |
| Obstacle | FP | 50 | 0.990 | 0.235 | 0.283 | 0.115 | 0.067 |
| Obstacle | FN | 48 | 0.923 | 0.234 | 0.209 | 0.116 | 0.056 |
| SurfaceProblem | TP | 50 | 1.000 | 0.218 | 0.404 | 0.116 | 0.096 |
| SurfaceProblem | FP | 50 | 0.997 | 0.215 | 0.431 | 0.118 | 0.118 |
| SurfaceProblem | FN | 50 | 0.937 | 0.223 | 0.163 | 0.119 | 0.058 |

What these say, before any rating:

- **Rollout does not find the label point.** Its median centre mass is 0.18–0.24 in every cell, at or below a
  uniform map's 0.25. About 12% of its mass sits on 3 of 256 patches, ten times a uniform map, and that share barely
  changes across types and categories. In the sheets these peaks are isolated bright dots, often on background
  (sky, a car roof, a dark corner).
  - This matches the high-norm "artifact" tokens that large DINOv2 models put on low-information patches
    ([Darcet et al. 2024](https://arxiv.org/abs/2309.16588)). The production models are DINOv2-L without
    registers.
  - So rollout is a poor explanation for this model. That is a hypothesis for the raters to confirm, not a
    conclusion.
- **Grad-CAM concentrates on the centre when the verdict is Agree on Crosswalk and SurfaceProblem.** Median centre
  mass is 0.40–0.48 for TP/FP, against 0.16–0.18 for FN, where the Disagree evidence is spread over the background.
  - On NoCurbRamp and Obstacle the difference is small (0.21–0.33 throughout).
  - FP and TP maps look alike on these statistics. Nothing here would let a validator tell an FP from a TP by its
    map alone.

## Preliminary observations per type (model-generated first pass; to be replaced by the raters)

These come from one viewing of the contact sheets by the implementing model (Opus 5.5). They are hypotheses for the
raters to test, not findings. Tile numbers refer to the sheets.

- **Crosswalk.**
  - On TPs, Grad-CAM traces the painted stripes or transverse lines (TP 01–05, 19, 42) and often extends along the
    paint well beyond the label point.
  - The 3 FNs show little or no visible paint at the centre (worn or absent markings). Grad-CAM there is diffuse over
    the asphalt.
  - The one public FP (`chicago-il:64484`) and the private ones (not committed) mostly show a clearly painted
    crosswalk. The crowd's "incorrect" is likely about placement or duplication, which a crop-level saliency map
    cannot express.
- **NoCurbRamp.**
  - On TPs, Grad-CAM often follows the raised curb edge at a corner (TP 01, 05–09, 21).
  - On FPs it lands as often on vegetation, a planting strip or a driveway edge (FP 02, 03, 09, 15) as on a curb.
  - Several FNs appear to show a ramp-like or flush transition near the centre (FN 12, 17, 20, 30). There, a
    Disagree may be the model judging "a ramp is present" rather than a model failure; that needs the label's
    intent, which the crop does not carry.
- **Obstacle.**
  - High-confidence FPs nearly always contain a salient object near the centre: a hydrant, bin, parked car, sign or
    pole (FP 02, 03, 09, 12, 21, 24, 25).
  - The crowd's "incorrect" plausibly means "present but not in the path" (a hydrant on the grass, a car in the
    road). The model, and both maps, respond to *object presence*, not to *path obstruction*.
  - On FNs, Grad-CAM often covers sidewalk surface or edges rather than the narrow object (poles, posts; FN 01, 07,
    12).
- **SurfaceProblem.**
  - On TPs and FPs, Grad-CAM tends to sit on slab joints, cracks, heaving and grass-in-joints along the walkway (TP
    18, 25, 37; FP 04, 14, 20, 24).
  - FPs therefore look like ordinary joints or wear that the crowd judged not to be a problem. The map localises
    well; the decision threshold, not the attention, is what differs.
  - FN maps are off-centre and diffuse (centre mass 0.16), often on lawn, cars or walls.

Common to all four: neither map shows the *context* the crowd judges, such as whether the label is a duplicate,
whether an object blocks the path, or whether a ramp is needed here. That limits what an overlay could tell a
validator.

## Rubric (for two raters)

Each rater codes every row of `docs/explanations-2026-10/coding_sheet.csv` (502 rows: one per tile, identified by
`overlay_id` = `<type>-<category>-<tile>` on the contact sheet named in `contact_sheet`). The two raters work
independently and do not discuss rows until both are done.

1. Copy the template to `coding_rater1.csv` / `coding_rater2.csv`.
2. Put the role (`rater1` / `rater2`) in `rater`, never a name.
3. Code the columns below and commit both files under `docs/explanations-2026-10/`.

The feature to judge is **what the label claims at the cross**: a crosswalk, a missing curb ramp, an obstacle, a
surface problem. For FPs the crowd said the claim is wrong. Code where the claimed feature (or the thing the labeller
probably meant) is. If there is nothing there, use `no`/`unclear` as below.

| column | values | rule |
|---|---|---|
| `rollout_covers` | `yes` / `partly` / `no` / `unclear` | `yes`: the brightest region(s) of the rollout panel lie on the labelled feature. `partly`: the feature is among several bright regions, or only its edge is lit. `no`: bright regions are elsewhere. `unclear`: no identifiable feature at the cross, or the map is uniform |
| `gradcam_covers` | same | same, for the Grad-CAM panel |
| `other_highlight` | `none`, `sidewalk_surface`, `road_surface`, `curb_edge`, `crosswalk_paint`, `vehicle`, `pole_or_sign`, `vegetation`, `building_or_wall`, `person`, `sky`, `image_corner_or_border`, `scattered_background`, `other` | the **most prominent** off-feature region in either panel (one value). `scattered_background`: isolated dots with no object, the pattern rollout often shows |
| `informative` | `yes` / `no` | Would showing this Grad-CAM overlay help a validator decide Agree/Disagree on this label? `yes` only if it points at the evidence the verdict rests on |
| `notes` | free text | anything else, e.g. "label is on the far crosswalk", "verdict depends on context outside crop" |

The vocabulary is enforced: `explain.py agreement` rejects any other value.

**Agreement.** `python explain.py agreement coding_rater1.csv coding_rater2.csv --out agreement.csv` joins on
`overlay_id` (one-to-one). For each coded column, overall and per type, it reports n, percent agreement and Cohen's
kappa. For the two `*_covers` columns it also reports a linear-weighted kappa over no < partly < yes. Rows coded
`unclear` or left blank by either rater are excluded. **Report the per-type kappa next to every per-type
conclusion.** If kappa < 0.6 on `gradcam_covers`, revise the rubric before drawing conclusions.

**Go/no-go for surfacing the overlay** (the second box of #19). Suggested rule, to be agreed before rating:

- **Go, for one type**, if both raters code `gradcam_covers` = `yes`/`partly` on at least 70% of that type's TPs and
  FNs, `informative` = `yes` on at least half, and kappa on `gradcam_covers` is ≥ 0.6.
- **No-go** otherwise.

## Limits of the method on this classifier

- Both maps are coarse: 16 × 16 patches of 14 px on a 224 px view, so a thin pole or a crack is one patch wide.
- Rollout is class-agnostic. It ignores the MLPs and the classifier head, and it is dominated by DINOv2-L's
  artifact tokens (above). Discarding the weakest attention (`discard_ratio`) or head-max fusion are the usual
  mitigations. They are exposed in `attention_rollout` but were **not** used for the committed overlays.
- Grad-CAM at one layer sees only what flows through that layer's attention input. The head's mean-pool over all
  patches spreads gradient across the whole image, which favours diffuse maps.
- The ReLU hides evidence against the verdict.
- The target is the margin of the predicted class, so FP maps explain the (wrong) Agree and FN maps the (wrong)
  Disagree.
- Neither map is faithful in a causal sense. Deletion/insertion tests would measure that; they are a follow-up, not
  done here.
- The crops come from the leaky 2025 split and from the models trained on it (#3). A retrained model needs its own
  overlays.
- The categories use the crowd's folder class, which is itself noisy: some "FPs" and "FNs" will be crowd errors.

## Draft issue text for sidewalk-ai-api (not filed; Jon decides after the rating)

> **Optional Grad-CAM overlay in the `/process` response (behind a flag)**
>
> For [sidewalk-validator-ai#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19): add an opt-in
> request flag (e.g. `"explain": true`, default false). When set, the validator's response gains
> `validation_overlay_png`, a base64 PNG of the 224 × 224 model view blended with a Grad-CAM map (last block's
> `norm1`, target = logit margin of the returned verdict). It is computed with one extra backward pass on the same
> forward. The flag must not change `validation_result`, `validation_score` or `validation_estimated_accuracy`.
> `sidewalk-validator-ai/explain.py` (`explain_one`, `heat_overlay`) is the reference implementation.
>
> Cost to measure before enabling: one backward pass through DINOv2-L per label. On an M1 Max (MPS) the forward plus
> backward took about 1–3 s per crop. On the API's GPU it should be a fraction of that; latency on the serving
> hardware is not yet measured. Rollout is deliberately not proposed (see the doc: it does not localise on these
> models). Do not enable the flag in SidewalkWebpage's nightly actor; it is for a mock-up in the Validate label card,
> pending the raters' go/no-go.

## Reproduce

`$DATA/<type>/test/<class>/*.webp` must hold the dataset files at the revisions in [Inputs](#inputs) (downloaded with
`huggingface_hub`; scratch scripts not committed). `cities.json` is the v3 API's city list from the 2026-10-17 pull
(`curl https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > cities.json`). Then:

```
for T in Crosswalk Obstacle NoCurbRamp SurfaceProblem; do
  t=$(echo $T | tr A-Z a-z)
  python explain.py run --label-type $T --crops-dir $DATA/$t --out-dir $OUT/$t \
      --per-category 50 --seed 42 --device mps --check-device cpu --cities-json cities.json
done
python explain.py coding-sheet docs/explanations-2026-10/*_samples.csv \
    --out docs/explanations-2026-10/coding_sheet.csv
python explain.py summary docs/explanations-2026-10/*_samples.csv
```

Each run takes 2–10 minutes on the M1 Max. Committed from each `$OUT/<type>/`: the three category contact sheets
(`<type>_{TP,FP,FN}.jpg`, JPEG q82, tiles 112 px high), `samples.csv` and `counts.json`, renamed `<type>_samples.csv` and `<type>_counts.json` (the repo .gitignore ignores directories named after label types, so the folder is flat). The full-size per-overlay
PNGs (`overlay_path` in `samples.csv`) and `predictions.csv` (every crop, private deployments included) stay outside
the repo; rerun to regenerate them.
