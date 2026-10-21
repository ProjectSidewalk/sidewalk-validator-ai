# Train/serve parity: do training and sidewalk-ai-api cut the same crop and tensor? (October 2026)

Issue [#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5) has one box left open: "Train, eval and
serving produce byte-identical crops and tensors for the same label." The retrained models will be served behind
[sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17), so this check has to pass before
that deploy. It runs both pipelines on the same panorama and label points, stage by stage, and measures every
difference and its effect on the five production validators. The code is `analysis/serving_parity.py`, the tests are
`tests/test_serving_parity.py`, and `serving-parity-2026-10/provenance.json` pins the inputs, versions and commands.

## Answer

**The code is at parity. Two settings are not.** With the same package versions, the training code and the API's
`/process` produce byte-identical views, depth maps and crops, and identical logits, in 30 of 30 cases (10 points ×
3 poses, all five models). Two things still put a different input in front of the model:

| Skew | Size on the input | Effect on the five production models (150 model × case pairs) | Fix |
|---|---|---|---|
| **opencv version**: `environment.yml` pins opencv-python 4.12.0.88, the API image 5.0.0.93. Their `cv2.resize(INTER_LINEAR)` differs (JPEG decode and `INTER_CUBIC` do not) | 1 level on 10.1% of panorama values; then 0/30 crops identical, crop half-size off by 1–6 px in 5/30 | max \|Δp\| 0.064, p90 0.011; 0 verdict flips, 0 vote changes | Same opencv in both. Check the klone env's actual version first |
| **WebP q95**: training models see a decoded q95 WebP, the API the raw crop | mean 1.16 levels per value, max 25 | max \|Δp\| 0.109, p90 0.026; 1 verdict flip, **2 vote changes** | The API encodes and decodes q95 WebP before the validator, or training stores lossless crops |

Everything else is exact or negligible: preprocessing (transformers 4.55 `use_fast=False` vs 5.20 `backend="pil"`:
identical tensors), the transformers/torch version on the model itself (max \|Δp\| 1.7e-5), the camera tilt (same
package, same frame rule for every pose a GSV label has), the depth model (same checkpoint, BGR on both sides,
identical maps given identical views) and the crop rule (identical except a float32 rounding that hits about 1 label
in 90,000). One frame-rule difference exists, a roll sent without a pitch, but no label in the round-3 data has a
roll, so it never fires today.

Applying both fixes makes the served input byte-identical to what training saw, on the CPU. GPU-to-GPU differences
(klone's CUDA vs the server's) were not measured; the CPU-vs-mps probe puts them near \|Δp\| 0.002.

## How the numbers were made

- **Training side**: `data_generation.py`'s own functions, unchanged: `read_store_panorama` on a one-pano shim store,
  `main`'s BGR→RGB, `render_label_view` (numpy, torch cpu, torch mps), `create_perspective_crop` with the real
  Depth Anything V2 metric vkitti ViT-B (a recorder wraps `infer_image`), `save_webp`, then `evaluate.run_inference`'s
  preprocessing (`AutoImageProcessor(use_fast=False)`, `Image.open(...).convert("RGB")`) and model call.
- **Serving side**: sidewalk-ai-api `tilt-fields` at 174a169, read-only. Its real `/process` handler runs through
  Flask's test client with the real `Panorama` cache path (`SIDEWALK_SCRAPES_DIR` = the same shim store), the real
  `DepthAnythingPredictor` and the real `ImageValidator` for each type. Recorders capture the image handed to depth
  and the PIL crop handed to the validator, so nothing is re-derived. The recorder's logits reproduce `validate()`'s
  verdict and confidence in 200/200 calls.
- **Environments** (each a separate venv; arrays cross between them as `.npy` / lossless `.png`):

| | python | numpy | opencv | pillow (libwebp) | torch | transformers |
|---|---|---|---|---|---|---|
| train (`environment.yml` pins, as on klone) | 3.10.19 | 2.2.6 | 4.12.0 | 11.3.0 (1.5.0) | 2.7.1 | 4.55.0 |
| train_other1 (`requirements.txt` venv) | 3.12.14 | 2.5.4 | 5.0.0 | 13.0.0 (1.6.0) | 2.14.1 | 4.55.0 |
| api (sidewalk-ai-api `requirements.txt`) | 3.12.14 | 2.5.4 | 5.0.0 | 12.3.0 (1.6.0) | 2.14.1 | 5.20.0 |

  The API pins the CUDA build of torch 2.14.1; the Mac gets the CPU/MPS wheel of the same version, and `triton` is
  left out. Both repos pin sidewalk-panorama-tools v1.0.0 (= 7ce25ff, the commit round-3's `generation_run.json`
  records) and Depth-Anything-V2 a561b84. The brief said transformers 5.19; the API's requirements.txt pins 5.20.0.
- **Inputs**: sidewalk-panorama-tools' `samples/sample_pano.jpg` (13312 × 6656 GSV). Its `labeldata.csv` rows belong
  to other panoramas (pano_id mismatch), so P01 is the curb ramp hand-picked in that repo's `assets/make_banner.py`
  and P02–P10 are synthetic: sidewalk-level points round the view, the horizon, both sides of the seam, near nadir and
  near zenith. Poses: level, pitch 3°, pitch −5° + roll 1°, and roll 1° alone; β = 0.95.
- **Device**: every reference stage on the CPU. The vendored `infer_image` sends its input to mps on a Mac whatever the
  model's device, so mps is hidden for the reference depth runs; mps renders and depth are probes only.
- **Held-out sample** (stage 5b): the 1,988 round-3 crops (five types, `test` + `test_heldout`, balanced), scored with
  each type's production validator on mps, as stored and after one more q95 encode (`webp-sample`).
- Hardware: Apple M1 Max, 32 GB, macOS 15.5; 2026-10-21. About 40 minutes for `run`, 15 for `webp-sample`.

```bash
python analysis/serving_parity.py run --train-python venv-klone/bin/python --other-train-python .venv/bin/python \
    --api-python venv-api/bin/python --api-dir ~/Git/sidewalk-ai-api-wt/tilt-fields \
    --pano ~/Git/sidewalk-panorama-tools/samples/sample_pano.jpg \
    --depth-checkpoint depth_anything_v2_metric_vkitti_vitb.pth --work WORK --out WORK/out
python analysis/serving_parity.py webp-sample --sample-csv round3/sample.csv --crops round3/crops \
    --api-dir ~/Git/sidewalk-ai-api-wt/tilt-fields --device mps --out-csv ~/Data/serving-parity-2026-10/webp_gen2.csv
python analysis/serving_parity.py crop-half --sample-csv round3/sample.csv
SIDEWALK_AI_API_DIR=~/Git/sidewalk-ai-api-wt/tilt-fields pytest -q tests/test_serving_parity.py   # live API checks
```

Full per-case tables: `serving-parity-2026-10/crops.csv` (stages 1–4), `models.csv` (stages 2, 4–7 per model) and
`summary.md`; `webp_gen2_summary.csv` and `crop_half.txt` for stages 5b and 4b.

## Results by stage

"Matched poses" are the three poses both sides read the same way (30 cases); `roll1_only` is reported apart.

### 1–4. Panorama, view, depth, crop

| Stage | train (opencv 4.12) vs API | train_other1 (opencv 5.0) vs API |
|---|---|---|
| 1. Pano read + resize to 8192 × 4096 | not identical: 1 level on 10.1% of values (17.5% of pixels) | identical |
| 2. View, torch cpu | 0/30 identical, max 2 levels, on 5–32% of pixels | 30/30 identical |
| 3. Depth map | 0/30 identical, max 2.25 m anywhere; centre within 0.05 m | 30/30 identical |
| 4. `crop_half_px` | 25/30 equal (P08–P10: off by 1, 1, 3, 6, 1 px) | 30/30 |
| 4. Crop array | 0/30 identical, mean abs ≤ 3.95 levels | 30/30 identical |

Isolated on the same file: JPEG decode is identical across opencv 4.12 and 5.0, `INTER_CUBIC` on an identical view is
identical, and only `INTER_LINEAR` (the panorama resize) differs.

| Check (all 40 cases) | Result |
|---|---|
| Renderer: numpy `render_view` vs torch cpu (training's `--allow-cpu` path vs the GPU path) | max 1 level, 0.17% of pixels (max 0.30%) |
| Renderer: torch cpu vs mps (torch 2.14) | max 1 level, ≤ 0.16% of pixels; torch 2.7's mps has no `padding_mode="border"` and fails |
| Depth input is the BGR view, training side (array equality with the recorder) | 40/40; the API passes its BGR render directly |
| Probe: the same view with channels swapped | centre depth moves median 0.26 m, max 3.2 m; `crop_half_px` changes in 35/40 |
| Probe: a channel-order bug in the comparison itself | would show as ≥ 3.7 levels mean abs on the view and ≥ 3.6 on the crop; none seen |
| Probe: depth on mps vs cpu | centre within 3.4e-5 m; `crop_half_px` unchanged 40/40 |
| API's float32 crop-half arithmetic on training's depths | equal 40/40 here; see 4b |

**4b. Crop-half rounding.** Training computes `int(6100 / float(depth))` in float64; the API `int(1 / depth * 6100)`
in float32. Over all 52,428,800 float32 depths in 1–80 m they disagree for 3,113 (5.9e-5): the API is 1 px larger in
3,050 and 1 px smaller in 63 (61.0 m gives 100 in training, 99 in the API; `test_round_depth_is_the_crop_half_skew`).
Weighted by the sample's depths (median 24.7 m) that is 1.1e-5 of labels. A one-pixel larger crop moves \|Δp\| by up
to 0.19 (median 0.0013) with no vote change here, so the rare case is not free, but it is rare.

![Crop half-size disagreement by depth](serving-parity-2026-10/crop_half_rounding.png)

### 5–7. Model outputs, matched poses, 150 model × case pairs

| Comparison | max \|Δlogit\| | median \|Δp\| | p90 \|Δp\| | max \|Δp\| | verdict flips | vote changes |
|---|---|---|---|---|---|---|
| 6+7. Same raw crop: train env (torch 2.7, tf 4.55) vs API env (torch 2.14, tf 5.20) | 2.9e-4 | 2.2e-8 | 9.7e-7 | 1.7e-5 | 0 | 0 |
| 7. Raw crop end to end: train_other1 env vs API | 0 | 0 | 0 | 0 | 0 | 0 |
| 7. Raw crop: train env (opencv 4.12) vs API (opencv 5.0) | 0.204 | 3.4e-4 | 0.011 | 0.064 | 0 | 0 |
| 5. WebP q95 vs raw, same crop | 0.427 | 4.3e-4 | 0.026 | 0.109 | 1 | 2 |
| 5. Second WebP generation vs first | 0.259 | 3.2e-4 | 0.0098 | 0.117 | 1 | 0 |
| **7. End to end: training as on klone (opencv 4.12, WebP) vs API** | 0.383 | 7.6e-4 | 0.026 | 0.081 | 1 | **2** |
| 2. Probe: numpy vs torch render | 0.099 | 1.6e-5 | 5.7e-4 | 0.0073 | 1 | 0 |
| 2. Probe: mps vs cpu render | 0.111 | 1.4e-5 | 4.5e-4 | 0.0016 | 0 | 0 |
| 4. Probe: crop half-size + 1 px | 0.630 | 0.0013 | 0.049 | 0.190 | 0 | 0 |

The verdict flips are at p(incorrect) ≈ 0.50 (SurfaceProblem P10, a zenith point). The two vote changes are the
same CurbRamp crops (P07, the left seam edge, level and pitch 3°), whose served accuracy sits on the 0.92 gate. The
points are mostly not curb ramps or obstacles, so the votes column is small (19 of 150 pairs vote); read the \|Δp\|
columns, not the counts.

**6. Preprocessing tensor.** On the same PIL crop, transformers 4.55 `BitImageProcessor` (`use_fast=False`, pillow
11.3) and 5.20 `BitImageProcessorPil` (`backend="pil"`, pillow 12.3) give **identical** `pixel_values` (0 difference,
200/200). The controls differ, as expected: 4.55 `use_fast=True` and 5.20's default torchvision backend both by up to
0.0175 in normalised units. So the API's `backend="pil"` and its class-name assert are load-bearing; keep them.

### 5b. WebP on the 1,988 sampled training crops

These crops are already WebP and their panoramas are gone, so the first-generation loss cannot be recomputed; a
second q95 generation stands in. On the real-panorama cases the second generation changes pixels 5× less than the
first (mean abs 0.22 vs 1.16) and p90 \|Δp\| 2.6× less (0.0098 vs 0.026), so **this table is a lower bound** on the
WebP skew. Plain accuracy on this 50/50 sample is not a production number; AUC and balanced accuracy are reported.

| Type | n | median \|Δp\| | p99 \|Δp\| | max \|Δp\| | verdict flips | vote changes | votes | AUC gen1 → gen2 | balanced acc gen1 → gen2 |
|---|---|---|---|---|---|---|---|---|---|
| CurbRamp | 400 | 1.6e-5 | 0.041 | 0.074 | 2 | 1 | 177 | 0.9302 → 0.9297 | 0.8725 → 0.8725 |
| NoCurbRamp | 400 | 1.2e-4 | 0.030 | 0.052 | 2 | 4 | 86 | 0.9078 → 0.9079 | 0.8325 → 0.8325 |
| Obstacle | 400 | 1.3e-4 | 0.028 | 0.049 | 1 | 0 | 30 | 0.8278 → 0.8270 | 0.7325 → 0.7350 |
| SurfaceProblem | 400 | 7.8e-5 | 0.065 | 0.216 | 2 | 0 | 56 | 0.8222 → 0.8230 | 0.7250 → 0.7250 |
| Crosswalk | 388 | 1.6e-5 | 0.022 | 0.054 | 0 | 1 | 190 | 0.9649 → 0.9648 | 0.9054 → 0.9054 |

6 of 1,988 served votes change (0.3%) and AUC moves by at most 0.0009. Production-model numbers on this sample are
optimistic: the 2025-08 models were trained on an undocumented label set that may overlap these labels, and the
retrained round-3 checkpoints were not available.

![Change in p(incorrect) from one more WebP generation, per model](serving-parity-2026-10/webp_gen2_dprob.png)

(Zero changes are drawn at 1e-8.)

## Verdict per skew

| Skew | Verdict | Measured | What to change, where |
|---|---|---|---|
| Preprocessing (issue #5) | **Closed** | identical `pixel_values`, 4.55 slow vs 5.20 pil | Nothing. Keep `use_fast=False` (training) and `backend="pil"` + the assert (API) |
| Camera pitch / tilt (issue #5) | **Closed** for every pose real labels have | 30/30 identical views with the same opencv | Nothing for GSV. Optional: the API ignores `camera_roll` without `camera_pitch`; training tilts by it. 0 of 64,465 round-3 rows carry a roll, so make the rule the same before Panoramax/Mapillary pose lands (`docs/decisions.md`) |
| Depth model (issue #5) | **Closed** | same ViT-B checkpoint (sha256 4dad67a7…) and submodule; BGR on both sides; identical maps given identical views | Nothing |
| Crop rule | **Closed, with a 1e-5 residual** | float32 vs float64 crop half-size, 1.1e-5 of labels | API: `crop_size_half = int(6100 / float(depth[center_y, center_x]))` |
| WebP (not in issue #5) | **Open** | p90 \|Δp\| 0.026, max 0.11; 2/150 vote changes on real-panorama crops; ≥ 6/1,988 on the sample | API: encode and decode q95 WebP in memory before the validator (`Image.save(buf, "WEBP", quality=95)`; the encoded bytes were identical with libwebp 1.5.0 and 1.6.0 here). Or, for the next dataset, lossless crops in training |
| transformers / torch version (not in issue #5) | **Closed** | max \|Δp\| 1.7e-5, no flip | Nothing |
| opencv version (found here) | **Open** | 1 level on 10% of the panorama → max \|Δp\| 0.064 | One opencv-python in both. `environment.yml` (4.12.0.88) vs the API image (5.0.0.93). Serving the retrained models should use the version the round-3 crops were cut with; check it on klone |
| GPU (not measured) | **Unknown, probably small** | cpu vs mps: \|Δp\| ≤ 0.0016 | One CUDA run of the train and API workers on the same GPU class, if wanted |

## Caveats

- One panorama, ten points, four poses. The byte-identity results are structural (same code, same pins), but the
  \|Δp\| sizes are from 150 model × case pairs on mostly non-feature points, and one sampled distribution for WebP.
- Training's environment is assumed to be `environment.yml`'s pins. `generation_run.json` records no package
  versions, so which opencv cut the round-3 crops is not recorded anywhere; the requirements.txt venv (opencv 5.0)
  matches the API exactly.
- Everything ran on an arm64 Mac. The opencv 4.12 vs 5.0 `INTER_LINEAR` difference was measured with the arm64 wheels;
  x86 wheels may differ differently. Training renders on klone's CUDA and the API on the server's; neither was run.
- The sample's WebP study is a second generation standing in for the first (a lower bound, above).
- The production validators were not trained on these crops (their 2025-08 data is undocumented and may overlap the
  sample), so model sensitivity may differ for the retrained ones; rerun `run` with the round-3 checkpoints when
  they exist (`model_map.py` is read from `--api-dir`).
- On a Mac the API's `DepthAnythingPredictor` crashes (`Input type (MPSFloatType) and weight type (torch.FloatTensor)
  should be the same`): the model goes to "cuda else cpu", the vendored `infer_image` input to "cuda else mps else
  cpu". Production (CUDA) is unaffected; a developer running `SIDEWALK_ALLOW_CPU=1` on Apple silicon hits it.

## What to do next

1. Read `python -c "import cv2; print(cv2.__version__)"` in the klone conda env that generated round 3. If it is
   4.12.0, pin `opencv-python==4.12.0.88` in sidewalk-ai-api for the retrained models (or regenerate with 5.0.0.93).
2. Add a q95 WebP round trip before the validator in sidewalk-ai-api (text below), in the same PR as the retrained
   model map.
3. Record package versions in `generation_run.json` (diff in the PR description), so (1) never has to be guessed.
4. Rerun `analysis/serving_parity.py run` against the API branch that will serve the retrained models; expect 30/30
   identical crops and 0 \|Δp\| end to end on the CPU.

## Proposed closing text for #5

> The last box ("byte-identical crops and tensors") is answered in docs/serving-parity-2026-10.md. With the same
> package versions, training's code and the API's `/process` produce byte-identical views, depth maps, crops and
> logits (30/30 cases, five models), and the preprocessing tensors are identical across transformers 4.55 slow and
> 5.20 pil. The three skews listed here (preprocessing, camera pitch, depth checkpoint) are closed. Two residuals
> remain, both in settings rather than code, and move to sidewalk-ai-api: the API image's opencv 5.0 resizes the
> panorama differently from training's 4.12 (max |Δp| 0.064), and training models see q95 WebP while the API feeds the
> raw crop (max |Δp| 0.11, 2 of 150 served votes changed). `tests/test_serving_parity.py` checks the live API code
> when `SIDEWALK_AI_API_DIR` is set.

## Proposed sidewalk-ai-api issue

> **Serve the validator the same input training uses: WebP q95 round trip and training's opencv**
>
> sidewalk-validator-ai's parity check (docs/serving-parity-2026-10.md there) found the API's crop is byte-identical
> to training's when both run the same opencv, with two differences in what reaches the model:
>
> 1. Training models see each crop after `save_webp` (PIL, `quality=95`) and decode; `/process` passes the raw array.
>    Measured: max |Δp| 0.11, p90 0.026, 2 of 150 served votes changed. Fix in `main.py`, after `cvtColor`:
>    `buf = io.BytesIO(); Image.fromarray(perspective_image).save(buf, format="WEBP", quality=95);
>    perspective_image = Image.open(buf).convert("RGB")` (the tagger should keep the raw crop unless its training
>    data was WebP too).
> 2. `requirements.txt` pins opencv-python 5.0.0.93; training's environment pins 4.12.0.88, and their
>    `cv2.resize(INTER_LINEAR)` of the panorama differs by one level on 10% of values (max |Δp| 0.064 downstream).
>    Pin the version the retrained models' crops were cut with.
>
> Smaller: `crop_size_half = int(inv_depth * 6100)` is float32 arithmetic; `int(6100 / float(depth[cy, cx]))`
> matches training exactly (differs for ~1e-5 of labels today). `camera_roll` without `camera_pitch` is ignored
> while training applies it (no GSV label has a roll). `DepthAnythingPredictor` fails on Apple silicon with
> `SIDEWALK_ALLOW_CPU=1` because the vendored `infer_image` moves its input to mps.
