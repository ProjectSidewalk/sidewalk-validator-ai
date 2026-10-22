# What the validators look at: saliency overlays and a blinded rating design (October 2026)

Part of [#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19). This page ships the tool, the
sampled maps, automatic statistics about them, a blinded packet and rubric for two human raters, and the rule that
turns their coding into a go/no-go. **The human rating has not been done.** The automatic statistics below already
rule out some readings of the maps. Whether any map shows the labelled feature is for the raters to decide.

**Where the numbers come from.** `explain.py` (this PR), run on 2026-10-20 on an M1 Max (macOS 15.5, torch 2.14.1,
transformers 4.55.0). Every count and median below is printed by the commands in [Reproduce](#reproduce) or read
from the committed `docs/explanations-2026-10/<type>_counts.json`.

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
    counts cover every crop and name no city. Prefixes map to deployments by exact subdomain only (the
    `/v3/api/cities` url), and a prefix that also names a non-public deployment's subdomain or id is dropped. A
    private deployment listed without a url cannot be checked by subdomain, only by its `city_id`; no such case
    occurs in these datasets.
  - **About 9% of crops are not label-centred.** The 2025 crops were clipped at the view border, which left 640 × h
    crops (h < 640) whose label point is off-centre by an unknown amount. They are flagged
    `label_point_uncertain`: no centre cross, a yellow "?" tag, and no centre/border statistics.

    | type | non-square crops (all) | non-square among sampled |
    |---|---:|---:|
    | Crosswalk | 24 of 240 | 13 of 104 |
    | NoCurbRamp | 125 of 1,374 | 15 of 200 |
    | Obstacle | 55 of 769 | 21 of 198 |
    | SurfaceProblem | 97 of 1,024 | 19 of 200 |


## Method

- **Categories.** The positive class is `correct` (the API's Agree):
  - TP = crowd correct, AI correct.
  - FP = crowd incorrect, AI correct, so the AI would Agree with a bad label.
  - FN = crowd correct, AI incorrect, so the AI would Disagree with a good label.
  - TN = both incorrect. **TN is the control category**: correct Disagree verdicts.
  - These are raw verdicts, with the API's tie rule and no 0.92 gate.
- **What the model sees.** The processor resizes the short side to 256 and centre-crops 224. On a square crop the
  label point is the centre of that 224 view, and 1/16 of each side is cut. On a non-square crop up to ~22% per side
  is also cut horizontally. Every panel shows the 224 view.
- **Maps.** All are per 14 px patch on the 16 × 16 grid.
  - **Attention rollout** ([Abnar & Zuidema 2020](https://arxiv.org/abs/2005.00928)): head-mean, no discarding,
    identity added for the residual, rows re-normalised, multiplied through 24 layers. It is the CLS row and is
    class-agnostic. The model is loaded with `attn_implementation="eager"`.
  - **Last-layer CLS attention**: head-mean. Class-agnostic.
  - **Grad-CAM**, as pytorch-grad-cam adapts it to a ViT:
    - Activations are the output of the last block's `norm1`.
    - One channel-weight vector (the gradient averaged over patches) serves the whole image; the map is the ReLU of
      the weighted sum.
    - The target is the logit margin `z[class] − z[other]`, computed for **both** classes. "Verdict" means the
      predicted class, passed explicitly from the prediction.
  - **grad × act**: per-token gradient times activation at the last block's output (the residual stream the final
    layernorm reads), summed over channels, ReLU. Same targets.
    - The head reads `[CLS ‖ mean(patch tokens)]`, so this attributes the mean-pool path exactly, token by token.
  - **Centre blob**: a fixed Gaussian at the grid centre (σ = 2.5 patches). It appears only in the rater packet, as
    a control.
- **Sampling.**
  - Up to 50 per category, seed 42, from public deployments only: 702 crops in all. Crosswalk has only 1 public FP
    and 3 public FNs.
  - Each type × category set is sorted by `label_uid` before sampling.
- **Device.**
  - Every overlay was computed on **MPS**. The prediction pass was repeated on CPU for every crop: max
    |Δp(correct)| was 7.7e-6 (Crosswalk), 2.1e-5 (NoCurbRamp), 1.4e-5 (Obstacle) and 2.8e-5 (SurfaceProblem), with
    0 category flips.
  - The explain pass reproduced each prediction within 2e-7.
  - It took a median of 0.17–0.20 s per crop on MPS for one forward pass and two backward passes
    (`explain_seconds_per_crop_median` in `counts.json`; wall-clock, including copying the maps to the CPU).

## Counts per category (test split, all crops; raw verdicts)

From `<type>_counts.json`: `categories` covers all crops, and `categories_sampling_pool` is the public-only pool
the overlays are drawn from. These counts are on the leaky file-level split
([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)), so treat the rates as optimistic.

| type | n | TP | FP | FN | TN | public pool TP / FP / FN / TN | sampled TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---|---|
| Crosswalk | 240 | 126 | 16 | 3 | 95 | 66 / 1 / 3 / 94 | 50 / 1 / 3 / 50 |
| NoCurbRamp | 1,374 | 732 | 103 | 65 | 474 | 700 / 97 / 62 / 423 | 50 / 50 / 50 / 50 |
| Obstacle | 769 | 376 | 104 | 67 | 222 | 353 / 97 / 48 / 149 | 50 / 50 / 48 / 50 |
| SurfaceProblem | 1,024 | 546 | 115 | 74 | 289 | 486 / 98 / 50 / 150 | 50 / 50 / 50 / 50 |

15 of Crosswalk's 16 test FPs are in deployments that are private today, so the issue's 50/50/50 is not reachable
for Crosswalk.

## Automatic map statistics (before any rating)

All are medians per type × category. **Centre** is the share of a map in the middle 8 × 8 patches; a uniform map
scores 0.25 and the centre blob 0.80. **Border** is the share on the outer ring of 60 patches; uniform scores 0.234.
**Top-3** is the share on the 3 strongest of 256 patches; uniform scores 0.012. Centre and border use certain crops
only (`n_certain`). Rows with n ≤ 3 are marked; do not read them as rates.

**Class-agnostic maps** (`python explain.py summary docs/explanations-2026-10/*_samples.csv --stats
rollout_centre,rollout_border,rollout_top3,rollout_top3_in_highnorm,cls_attention_centre,cls_attention_top3,cls_attention_top3_in_highnorm`):

| type | cat | n (certain) | rollout centre | rollout border | rollout top-3 | rollout top-3 ∩ high-norm | CLS-attn centre | CLS-attn top-3 | CLS-attn top-3 ∩ high-norm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Crosswalk | TP | 50 (46) | 0.202 | 0.304 | 0.129 | 0 | 0.287 | 0.400 | 0.667 |
| Crosswalk | FP | 1 (1) ⚠ | 0.180 | 0.348 | 0.143 | 0 | 0.221 | 0.445 | 0.333 |
| Crosswalk | FN | 3 (1) ⚠ | 0.151 | 0.341 | 0.130 | 0 | 0.586 | 0.577 | 0.333 |
| Crosswalk | TN | 50 (43) | 0.178 | 0.311 | 0.128 | 0 | 0.126 | 0.555 | 0.333 |
| NoCurbRamp | TP | 50 (45) | 0.219 | 0.270 | 0.113 | 0 | 0.273 | 0.532 | 0.333 |
| NoCurbRamp | FP | 50 (47) | 0.236 | 0.262 | 0.114 | 0 | 0.243 | 0.480 | 0.333 |
| NoCurbRamp | FN | 50 (47) | 0.223 | 0.277 | 0.115 | 0 | 0.263 | 0.451 | 0.333 |
| NoCurbRamp | TN | 50 (46) | 0.242 | 0.279 | 0.115 | 0 | 0.272 | 0.463 | 0.333 |
| Obstacle | TP | 50 (43) | 0.240 | 0.250 | 0.116 | 0 | 0.250 | 0.466 | 0.333 |
| Obstacle | FP | 50 (44) | 0.231 | 0.246 | 0.115 | 0 | 0.242 | 0.480 | 0.333 |
| Obstacle | FN | 48 (46) | 0.234 | 0.259 | 0.116 | 0 | 0.201 | 0.461 | 0.333 |
| Obstacle | TN | 50 (44) | 0.237 | 0.254 | 0.116 | 0 | 0.199 | 0.480 | 0.333 |
| SurfaceProblem | TP | 50 (48) | 0.218 | 0.263 | 0.116 | 0 | 0.268 | 0.430 | 0.333 |
| SurfaceProblem | FP | 50 (42) | 0.215 | 0.280 | 0.118 | 0 | 0.249 | 0.455 | 0.333 |
| SurfaceProblem | FN | 50 (44) | 0.221 | 0.272 | 0.119 | 0 | 0.233 | 0.450 | 0.333 |
| SurfaceProblem | TN | 50 (47) | 0.209 | 0.283 | 0.113 | 0 | 0.226 | 0.460 | 0.333 |

"∩ high-norm" is the share of a map's top-3 patches that are among the 5 highest-norm patch tokens. Those are taken
from the block where the max/median token norm is largest. That was block 20 of 24 for all 702 crops, with a median
max/median ratio of 25–28 per type (`highnorm_layer`, `highnorm_ratio` in `samples.csv`).

**Gradient maps, both targets** (`--stats
gradcam_correct_centre,gradcam_incorrect_centre,gradxact_correct_centre,gradxact_incorrect_centre`). "Agree" is the
`correct` target and "Disagree" the `incorrect` one, whatever the verdict was:

| type | cat | n (certain) | Grad-CAM Agree | Grad-CAM Disagree | grad×act Agree | grad×act Disagree |
|---|---|---|---:|---:|---:|---:|
| Crosswalk | TP | 50 (46) | 0.414 | 0.138 | 0.247 | 0.254 |
| Crosswalk | FP | 1 (1) ⚠ | 0.480 | 0.127 | 0.309 | 0.202 |
| Crosswalk | FN | 3 (1) ⚠ | 0.243 | 0.299 | 0.184 | 0.309 |
| Crosswalk | TN | 50 (43) | 0.262 | 0.202 | 0.189 | 0.269 |
| NoCurbRamp | TP | 50 (45) | 0.321 | 0.221 | 0.225 | 0.277 |
| NoCurbRamp | FP | 50 (47) | 0.310 | 0.222 | 0.245 | 0.260 |
| NoCurbRamp | FN | 50 (47) | 0.262 | 0.275 | 0.231 | 0.298 |
| NoCurbRamp | TN | 50 (46) | 0.240 | 0.282 | 0.244 | 0.313 |
| Obstacle | TP | 50 (43) | 0.271 | 0.215 | 0.276 | 0.224 |
| Obstacle | FP | 50 (44) | 0.283 | 0.213 | 0.250 | 0.246 |
| Obstacle | FN | 48 (46) | 0.300 | 0.206 | 0.257 | 0.260 |
| Obstacle | TN | 50 (44) | 0.251 | 0.240 | 0.220 | 0.256 |
| SurfaceProblem | TP | 50 (48) | 0.404 | 0.179 | 0.279 | 0.190 |
| SurfaceProblem | FP | 50 (42) | 0.418 | 0.192 | 0.240 | 0.231 |
| SurfaceProblem | FN | 50 (44) | 0.407 | 0.163 | 0.244 | 0.257 |
| SurfaceProblem | TN | 50 (47) | 0.319 | 0.181 | 0.247 | 0.265 |

What follows from these numbers:

- **Rollout is border-biased and does not find the label point.**
  - Its centre share is 0.15–0.24 in every cell, at or below uniform.
  - Its border share is 0.25–0.35, at or above uniform (0.234). This is most marked on Crosswalk (0.30–0.35).
  - About 12% of each map sits on 3 patches, and that barely changes across types and categories.
  - **None** of those peaks coincides with the high-norm tokens (median overlap 0 in every cell). So the earlier
    draft's attribution of the rollout peaks to DINOv2's artifact tokens was wrong. The border bias is unexplained.
  - Discarding weak attention (`discard_ratio`) or fusing heads by max were not evaluated here.
- **The last-layer CLS attention is where the artifact tokens show up.**
  - 40–58% of it sits on 3 patches.
  - In every cell with n ≥ 50, a median of 1 of those 3 is among the 5 highest-norm tokens (2 of 3 on Crosswalk
    TP).
  - Its centre share (0.13–0.29 outside the n ≤ 3 cells) is no better than uniform. These are the low-information
    "register-like" tokens of [Darcet et al. 2024](https://arxiv.org/abs/2309.16588); DINOv2-L without registers
    has them.
- **Grad-CAM's centre concentration is a property of the Agree target, not of right versus wrong.**
  - With the Agree target, SurfaceProblem maps put 0.40–0.42 in the centre for TP, FP and FN alike, and 0.32 for
    TN. With the Disagree target they put 0.16–0.19 there for every category.
  - Crosswalk TP: 0.41 (Agree) against 0.14 (Disagree).
  - NoCurbRamp and Obstacle are near uniform for both targets (0.21–0.32).
  - So the earlier "FNs are diffuse" observation was the Disagree target, not the crop. An Agree map that sits on
    the centre of any crop may simply reflect the training-time centre prior (every training crop is centred on a
    label), not the feature.
- **Grad-CAM at `norm1` does not localise a planted signal** (`python explain.py planted-check`: a tiny random
  Dinov2 whose head reads one patch token, 20 random inputs each for token 6 and token 15; chance ≈ 1.25 of 20).
  - Grad-CAM peaked on the read token in 2 and 0 of 20 for the predicted class.
  - grad × act at the block output peaked there in 18 and 20 of 20 for the predicted class, but in only 2 and 0 of
    20 for the other class.
  - `tests/test_explain.py` asserts the grad × act result (≥ 9 of 10, and ≥ 3 of 4 through `planted_check`).
  - The cause: Grad-CAM uses one channel-weight vector for the whole image, and `norm1`'s output does not include
    the residual path that carries a token's own signal.
- **grad × act is near uniform on the real models.** Its centre share is 0.18–0.31 for either target. A per-token
  attribution of the mean-pool path does not concentrate on the label point either.

**Conclusion before rating.**
- None of the four maps has been shown to localise the labelled feature on these models.
- Rollout is border-biased.
- CLS attention is dominated by artifact tokens.
- Grad-CAM at this hook does not localise even a planted signal, and its centre concentration follows the Agree
  target on any crop.
- grad × act is the one map with a demonstrated localising mechanism, but it is not concentrated on the label point
  in real crops.

The raters' coding against the centre-blob control is what can still show a map to be useful.

## Rater packet (blinded)

The labelled contact sheets and `*_samples.csv` in `docs/explanations-2026-10/` are for readers. **Raters must not
open them until both have finished.** They do not reveal the packet's order or panels, because those come from a
secret seed. The rating uses a separate packet built by `explain.py packet`, kept outside the repo:

- **One JPEG per crop, 702 items.** Each shows the crop at 224 px (with the label-point cross, or a yellow "?" when
  uncertain) and five panels A–E at 224 px.
- **What the panels hide.** They hold rollout, last-layer CLS attention, Grad-CAM (verdict), grad × act (verdict)
  and the centre blob, in a random order per item. Raters are told neither the method, nor the category, crowd
  class, verdict or confidence. The label type is shown, because the feature to judge depends on it.
- **Order and ids.** Items are in a shuffled order and named `<NNN>_<packet_id>.jpg`, where `packet_id` is random.
- **Coding sheet.** `coding_sheet.csv` in the packet, in the same order: `packet_id`, `label_point_uncertain` and
  blank coding columns.
- **Key.** `packet_key.csv` maps `packet_id` to `overlay_id`, type, category, label_uid, the
  `label_point_uncertain` flag and each panel's method.
  - It was built with a **secret seed** and is kept outside the repo, with Jon.
  - The repo holds only its hash, `docs/explanations-2026-10/packet_key.sha256`, so the key used for unblinding can
    be checked.
  - **After both raters finish**, the key (and the seed) are committed beside the hash, and anyone can then
    reproduce the unblinding.
  - The seed has no default; with it and the committed `*_samples.csv`, `packet` rebuilds the same key.
  - `packet` refuses a key path inside any checkout of this repo, including other git worktrees. It uses git's
    common directory, so a copy of the repo cloned elsewhere is not detected.

**How a rater gets it.** Jon shares `rater-packet-2026-10.zip` (62 MB: `items/` and `coding_sheet.csv`). It is
built by this PR's run and is not in the repo. A rater needs nothing else.

**Rater burden.** 702 items × 5 panels is a lot. If that is too much, code the items with
`label_point_uncertain = False` first, in packet order: the order is random, so any prefix is a random subset.

## Rubric

Each rater fills a copy of the packet's `coding_sheet.csv` (`coding_rater1.csv`, `coding_rater2.csv`). Put the role
in `rater`, never a name. Work independently.

**The feature to judge** is what the label type claims at the cross: a crosswalk, a missing curb ramp, an obstacle,
a surface problem. If nothing of the kind is visible near the cross, or the label point is uncertain and nothing
stands out, code `unclear`.

| column | values | rule |
|---|---|---|
| `covers_A` … `covers_E` | `yes` / `partly` / `no` / `unclear` | Code each panel by where its brightest region lies. `yes`: on the feature. `partly`: the feature is one of several bright regions, or only its edge is lit. `no`: elsewhere. `unclear`: no identifiable feature, or a uniform map |
| `best_panel` | `A`–`E` / `none` | the one panel that would best help a validator decide Agree/Disagree on this label; `none` if no panel would |
| `notes` | free text | e.g. "label is on the far crosswalk", "feature outside the 224 view" |

`explain.py agreement` rejects any other value.

## Agreement and go/no-go

```
python explain.py agreement coding_rater1.csv coding_rater2.csv --key <path>/packet_key.csv \
    --key-sha256 docs/explanations-2026-10/packet_key.sha256 --out agreement
```

**How the codes are scored.**
- The command checks the key against the committed hash (`--key-sha256`), unblinds both sheets, and drops items whose label point is
  uncertain. The flag is taken from the key, not from the editable sheet.
- It reports, per method and per type × method, n, percent agreement, unweighted Cohen's kappa, and linear-weighted
  kappa over no < partly < yes.
- Rows where either rater wrote `unclear` or left the cell blank are excluded.
- Kappa is **NaN** when it is undefined (both raters used one value throughout); percent agreement then carries the
  result.

**Go/no-go rule.** This is a **proposal**, implemented in `go_no_go`. Jon fixes the thresholds before the rating
starts; changing them after seeing the codes would void the test. The rule is applied per label type, separately
for Grad-CAM and for grad × act (the verdict maps). A method is **go** for a type only if all of the following hold
on that type's **Agree-verdict items** (TP + FP) with a certain label point:

1. Each rater coded at least 20 of them `yes`/`partly`/`no`.
2. The covers rate, (yes + partly) / (yes + partly + no) with `unclear` and blank excluded, averaged over the two
   raters, is ≥ 0.70.
3. That rate exceeds the **centre blob's** rate on the same items by ≥ 0.20.
4. The raters are reliable on **this method's** codes (`method_reliability`, per type × method). The codes used
   are the type's Agree-verdict items that both raters coded `yes`/`partly`/`no`; `unclear`, blank, FN and TN are
   excluded.
   - If unweighted Cohen's kappa is defined and the most common code's share (prevalence) is ≤ 0.90, reliable means
     kappa ≥ 0.60.
   - Otherwise reliable means percent agreement ≥ 0.85.
   - The fallback exists because kappa fails exactly when a map is good: 30/30 `yes` from both raters gives an
     undefined kappa, and 29/30 against 30/30 gives 0. Both cases pass on percent agreement (1.0 and 0.97).
   - Reliability is not pooled across panels. Agreement on the uninformative panels would otherwise lend
     reliability to a map the raters disagree on. `agreement` prints the pooled figure for context only.

Any other outcome is **no-go**.

**Rule 3 may be close to unreachable.** The labels sit at the centre of most crops, so the centre blob will often
"cover" the feature.
- If the blob's covers rate is b, a map needs a rate of at least max(0.70, b + 0.20).
- With b ≥ 0.80 that is 1.0 or more, which no map can reach.
- This is deliberate: a map that does no better than a fixed centre blob adds nothing the cross does not already
  show.
- Jon may lower the margin before rating, not after.

Disagree-verdict items (FN, TN) are reported but do not enter the rule. Their verdict maps show evidence for "no
such feature", so "covers the labelled feature" is not the right test for them. TN items also serve as the control
for the observation that Agree maps centre on any crop.

## Limits of the method on this classifier

- **Resolution.** All maps are on 16 × 16 patches of 14 px, so a pole or a crack is one patch wide.
- **Rollout.** It is class-agnostic and ignores the MLPs and the head. Here it is also border-biased, for reasons
  not established.
- **CLS attention.** It is dominated by high-norm artifact tokens.
- **Grad-CAM at `norm1`.**
  - It shares one channel weighting across the image and misses the residual stream. On a planted signal it does no
    better than chance (above).
  - The ReLU hides evidence against the target.
- **grad × act.**
  - It attributes only the mean-pool half of the head at the last block, not CLS's path through the earlier layers.
  - It is a first-order (gradient) attribution.
  - The map for the class that was **not** predicted is ReLU-clipped counter-evidence, not an explanation. It failed
    the planted-signal check (2 and 0 of 20) while the predicted class's map passed it (18 and 20 of 20). So in the
    statistics table the "Disagree" columns for TP/FP and the "Agree" columns for FN/TN are clipped counter-evidence
    maps. The packet shows only the predicted class's map.
- **Faithfulness.** Neither deletion nor insertion tests were run, so no map is shown to be causally faithful.
- **Centre prior.** The training crops are all centred on a label, so a model can learn "centre matters"
  regardless of content. The centre blob control and the TN category exist to separate that from localisation.
- **The blob is recognisable.** It is a smooth Gaussian, so raters may recognise it. It bounds what "covers"
  means for a map that simply sits in the centre; it is not a blind placebo.
- **Data.** The crops come from the leaky 2025 split and from models trained on it
  ([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)). The crowd class is itself noisy.
- **Uncertain crops.** About 9% of crops have an uncertain label point and are left out of the centre statistics
  and the rule.

## Draft issue text for sidewalk-ai-api (not filed; only if the rating says go)

> **Optional saliency overlay in the `/process` response (behind a flag)**
>
> For [sidewalk-validator-ai#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19): add an opt-in
> request flag (e.g. `"explain": true`, default false). When set, the validator's response gains
> `validation_overlay_png`, a base64 PNG of the 224 × 224 model view blended with the map method that passed the
> raters' go/no-go for that label type. `sidewalk-validator-ai/explain.py` (`explain_one`, `heat_overlay`) is the
> reference implementation. The flag must not change `validation_result`, `validation_score` or
> `validation_estimated_accuracy`.
>
> Cost: one forward pass with attention weights and two backward passes through DINOv2-L. That took a median of
> 0.17–0.20 s per crop on an M1 Max (MPS); serving hardware is not measured. Do not enable the flag in
> SidewalkWebpage's nightly actor; it is for a mock-up in the Validate label card.

## Reproduce

`$DATA/<type>/test/<class>/*.webp` must hold the dataset files at the revisions in [Inputs](#inputs), downloaded with
`huggingface_hub`. `cities.json` is the v3 API's city list from the 2026-10-17 pull
(`curl https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > cities.json`).

```
for T in Crosswalk Obstacle NoCurbRamp SurfaceProblem; do
  t=$(echo $T | tr A-Z a-z)
  python explain.py run --label-type $T --crops-dir $DATA/$t --out-dir $OUT/$t --cities-json cities.json \
      --per-category 50 --seed 42 --device mps --check-device cpu
done
# the committed sheets were re-rendered smaller from the saved maps (no torch needed):
python explain.py sheets $OUT/crosswalk $OUT/nocurbramp $OUT/obstacle $OUT/surfaceproblem --tile-height 88 --quality 70
# copy what is committed:
for t in crosswalk nocurbramp obstacle surfaceproblem; do
  cp $OUT/$t/*.jpg docs/explanations-2026-10/
  cp $OUT/$t/samples.csv docs/explanations-2026-10/${t}_samples.csv
  cp $OUT/$t/counts.json docs/explanations-2026-10/${t}_counts.json
done
python explain.py summary docs/explanations-2026-10/*_samples.csv --stats <columns as above>
python explain.py planted-check
# blinded packet and key outside the repo (secret seed); only the key's hash goes into the repo:
python explain.py packet $OUT/crosswalk $OUT/nocurbramp $OUT/obstacle $OUT/surfaceproblem \
    --out-dir $PACKET --key $KEYDIR/packet_key.csv --seed $SECRET_SEED
cp $KEYDIR/packet_key.csv.sha256 docs/explanations-2026-10/packet_key.sha256
```

**Run time.** Measured from file timestamps on the M1 Max, Obstacle took 4.8 min, SurfaceProblem 6.1 and NoCurbRamp 8.1, CPU check included. Crosswalk was not timed.

**Committed.** 16 contact sheets (`<type>_{TP,FP,FN,TN}.jpg`, tiles 88 px high, JPEG q70, 5.9 MB), and per type
`<type>_samples.csv` and `<type>_counts.json`, plus `packet_key.sha256`. The folder is flat because the repo's
`.gitignore` ignores directories named after label types.

**Not committed.** `predictions.csv` (every crop, private deployments included), `maps/*.npz`, the packet, and
(until the rating is done) the key and its seed stay outside the repo. An earlier, superseded key built with seed
42 appears in this branch's history; it does not match the current packet.
