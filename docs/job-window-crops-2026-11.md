# Job-window crops: the backend's crop rule as a training geometry (November 2026)

Experiment for [#53](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/53): would a CurbRamp validator
trained on the crops SidewalkWebpage's nightly crop job cuts do as well as the round-3 validator trained on
depth-sized perspective views? If it does, the API could classify the crop the website already stores instead of
rendering a view and running a depth model per label.

**Status: code, tests and the klone job are ready; nothing has been trained. Results: not yet run.** Jon launches the
pilot (`job_crop.slurm`, below). The pre-training counts (Jon's point 1 on the issue) were measured on 2026-11-05
from the local copy of round 3's split and are below.

## The question, and Jon's decisions (2026-10-26, on the issue)

- Keep the backend's **3:2 window as cut**; no square variant.
- Size the window from **each stored panorama's own dimensions**, exactly as `CropService.windowFor` and
  `CropSizingRule` do.
- **Pilot on CurbRamp**, seed 42, 25 epochs, the PR [#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51)
  protocol; decide on the other four types afterwards.
- **Break the comparison down by imagery provider** (GSV, Mapillary, Panoramax): first how many labels with a job
  crop exist per provider, so we know whether a per-provider AUC is measurable, then the held-out comparison per
  provider. If the non-GSV counts are too small, say so rather than pooling them away.

## What the backend cuts

`job_crop.py` ports these, one to one, from ProjectSidewalk/SidewalkWebpage at `5d89dc179`:

| backend | port | what it does |
|---|---|---|
| `CropGeometry.scala` `CropSizingRule` (lines 16-101), rule `v2` | `reference_crop_size`, `predict_crop_size`, `window_width` | The 2013 pano-y to distance to size regression, evaluated in its 6656-px reference space, scaled x2.5, clamped to 8-90 degrees of azimuth, back to pixels through the pano's width |
| `CropGeometry.scala` `CropGeometry` (111-226) | `compute_crop_box`, `label_position_in_crop`, `label_fraction_in_crop`, `segments` | A 3:2 window centred on the label's native pixel, capped at the pano width and at `1.5 x` its height; x wraps at the seam (two source runs), y shifts off the poles (`shifted`), never padded |
| `CropService.scala` `windowFor` (177-181), `storedSize` (171-174), `storedCrop` (164) | `window_for`, `stored_size`, `stored_crop` | The window for one label; stored at most 1440 px wide, never upscaled |
| `CropService.scala` `cutCrops` / `cutCrop` (493-560) | `cut_job_crop` (raises `JobCropSkipped`) | Skip a pano whose recorded `pano_width`/`pano_height` differ from the stored file's (`dims_mismatch`, "rather than mis-centring"; missing recorded dims are cut anyway), and a label with `pano_y` outside `[0, height)` (`out_of_frame`) |

So a job crop is: a 3:2 window, 8 to 90 degrees of azimuth wide, centred on the label, cut from the **native** stored
panorama, downscaled to at most 1440 px wide. No perspective projection, no tilt correction, no depth.

**Fidelity.** The geometry is exact. The Scala rounds half to even (`Math.rint`) and uses `floorMod` because it was
ported from panorama-tools' `CropRunner.py`, and the port does the same (`round`, `%`); `storedSize` alone uses Java's
half-up `Math.round`, reproduced as `floor(x + 0.5)`. The backend's golden fixtures, written by `CropRunner.py` itself
for `test/service/CropGeometrySpec.scala`, are copied verbatim into `tests/fixtures/job_crop/`, and
`tests/test_job_crop.py` reproduces every sizing row (66), every mechanics box and every cut window **pixel for
pixel**. The resampling is not byte-identical: the backend's `ImageUtils.scaleToMaxEdge` halves repeatedly and finishes
with a bilinear draw, the port uses one `cv2.resize(..., INTER_AREA)`. Windows at most 1440 px wide are not resampled
and are exact; on round 3's CurbRamp rows 52% of windows are wider and are downscaled, so their pixels differ from the
website's stored PNG in the low bits. If a future serve-from-store path needs byte parity, that is sidewalk-ai-api work.

## How it differs from what the model sees today

| | depth-sized (round 3, serving) | job window (this experiment) |
|---|---|---|
| source | pano resized to 8192 x 4096 | the native stored file (16384 x 8192 for 87% of round 3's CurbRamp rows) |
| projection | 2048² perspective view, 90° FOV, centred on the label | none: an equirectangular window |
| tilt correction | GSV beta 0.95 (`crop_pose.py`) | none (`tilt_applied = 0` on every row) |
| size | square, half-width `6100 / depth_m` view px (median 238 px, about 26° wide) | 3:2, `CropSizingRule` v2 (median 33° wide; 12% at the 90° ceiling, none at the 8° floor) |
| saved | long side 640 px, WebP q95 | stored size (at most 1440 x 960; median 1440 wide), WebP q95 |

**What the training recipe then does to a 3:2 image.** `fine_tune_classifier.py` trains with
`RandomResizedCrop(224, scale=(0.7, 1.0))` and evaluates with the DINOv2 processor (shortest edge to 256, centre crop
224), as `evaluate.py` does. A 3:2 crop resized to a 256-px short edge is 384 px wide, so evaluation keeps the
**central 224/384 = 58% of the width** and drops the side margins, and training samples sub-windows of it. This is
kept on purpose: the experiment asks what the backend's crop does **under the current recipe**. A 3:2-aware recipe
(pad or resize to a square) is a follow-up if the result is close.

`crop_half_px` in the manifest is `box.width // 2`, the **native** half-width of the 3:2 window (not a view
half-width, so not comparable with the depth path's number); `clipped` is the window's `shifted` flag; `depth_m` is
empty. The window facts the manifest has no column for go to `<out>/CurbRamp/job_crops.csv` (box, label fraction,
stored size, native pano size, rule version), appended per row and compacted to the manifest's ok rows after every
run. `--tilt-beta` and the pose lookup still run and are recorded; Panoramax's pitch-only latitude offset
(`click_lat_offset_deg`) is recorded but not applied, as the backend does not apply it either.

## Protocol

PR #51's, restricted to the rows that get a job crop:

1. `data_generation.py --job-window-crops` on round 3's rawLabels cache with round 3's seed, `EXPERTS` and
   `--include-ai-labels` (round 3's settings).
2. `analysis/job_crop_compare.py restrict`: round 3's `split.csv` restricted to the rows with an ok job crop, **every
   row keeping its round-3 split** (`split_data.py` is not re-run, so nothing drifts). `restrict_summary.json` says
   how many round-3 rows were lost and why.
3. `fine_tune_classifier.py` (seed 42, 25 epochs) and `evaluate.py` on the restricted split.
4. `analysis/job_crop_compare.py compare`: job minus depth AUC on identical rows, with a paired panorama bootstrap
   (1000 resamples, `retrain_report.cluster_boot`), pooled and per split (`test`, `test_heldout`), per provider where
   measurable (n >= 30, each class >= 10), within-city AUC for both models (`retrain_report.within_city_auc`), and the
   severity x tag slices (`slices.paired_delta_table`, the standing rule). No table names a deployment.

The depth model is **not retrained** on the restricted rows (#51 did not either): it is round 3's model, scored on the
subset. Its training set is about 1% larger than the job model's (the `dims_mismatch` rows below). A retrain-both-arms
variant is a follow-up only if that turns out to matter.

## Counts before training (Jon's point 1)

`python analysis/job_crop_compare.py counts`, run 2026-11-05 on the local copy of round 3's CurbRamp `split.csv` and
the rawLabels cache it was built from ([`job-window-crops-2026-11/counts.csv`](job-window-crops-2026-11/counts.csv),
per class there):

| provider | split | n | job crop | dims_mismatch | out_of_frame | not from the store | AUC |
|---|---|---|---|---|---|---|---|
| gsv | train | 6,925 | 6,849 | 76 | 0 | 0 | |
| gsv | val | 1,487 | 1,474 | 13 | 0 | 0 | |
| gsv | test | 1,479 | 1,463 | 16 | 0 | 0 | measurable |
| gsv | test_heldout | 546 | 546 | 0 | 0 | 0 | measurable |

- **99.0% of round 3's rows (10,332 of 10,437) get a job crop.** The 105 that do not are `dims_mismatch`: the
  label's recorded frame is not the stored file's size (the backend's comment blames the old scraper, which upscaled
  small panos to 13312 x 6656), and the backend skips them too.
- **Round 3's CurbRamp split is GSV only**, so the per-provider comparison Jon asked for **cannot be measured on it**:
  round 3 was generated before [#72](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/72) kept
  Mapillary and Panoramax labels. Nor would a regenerated split help yet: in the same rawLabels snapshot there are
  15,008 Mapillary and 13 Panoramax CurbRamps, 5,835 and 4 of them with at least one human vote, but **none with
  three human Agree votes and none meeting the crowd margin rule** (measured 2026-11-05 with
  `data_generation.eligible_labels`, without the expert rule). Whether `CropSizingRule` v2, a regression fit on GSV,
  transfers to other rigs therefore cannot be answered from validated labels today; it needs either more validation
  on non-GSV deployments or a rated sample. That is itself an input to whether the job cutter needs a
  provider-aware sizing rule.

## Results

**Not yet run.** `headline.csv`, `per_provider.csv`, `slices.csv` and `tables.md` will go in
`docs/job-window-crops-2026-11/` once the job has run (they name no deployment; check before committing).

## How to run (klone, when Jon says)

```bash
ROUND3=/gscratch/makelab/$USER/retrain-2026-10 CACHE=<round 3 rawlabels cache> \
PANO_STORE=<the store, or the staged copy round 3 read> OUT=/gscratch/scrubbed/$USER/job-crops-2026-11 \
EXPERTS="<name>=<user_id>,..." sbatch job_crop.slurm
```

One GPU job on `ckpt-all` with `--requeue`; every step resumes or is idempotent. The selection is compared with
round 3's `selection_sha256` and a difference is printed as a warning (the selection code moved since round 3), not a
stop: `restrict` guarantees the pairing. Then copy `$OUT/report/*.csv` and `tables.md` into
`docs/job-window-crops-2026-11/` and fill in Results.

Models trained this way are **experiment-only**: `generation_run.json` and `training_run.json` record
`job_window_crops: true`, and `publish_to_hf.py` (via `hf_cards.require_depth_sized`) refuses to stage or push them.
`job_window_crops` is also a resume setting of `data_generation.py`, compared whenever either side has it, so a
job-crop output directory is never resumed as, or reused for, a depth-sized one.

## Decision rule

Jon's: pilot CurbRamp, then decide on the other four types. Suggested reading of the pilot: if the paired held-out
delta's interval excludes a loss larger than 0.01 AUC and the severity x tag slices show no slice falling clearly
behind, extend to the other types (each needs its own counts first); if the job crop loses clearly, the 3:2-aware
recipe above is the cheap thing to try before giving up on serving from stored crops.

## Code

- `job_crop.py`: the port and the sidecar helpers; `tests/test_job_crop.py` with the backend's golden fixtures.
- `data_generation.py --job-window-crops`: hooks only (the flag, its exclusion with `--fixed-crop-half-px`, the resume
  setting, no depth model, one branch at the crop call site, the skip statuses, the sidecar after the loop).
- `fine_tune_classifier.generation_settings` copies `job_window_crops` into `training_run.json`;
  `hf_cards.job_window_crops` / `require_depth_sized` and `publish_to_hf.check_depth_sized` refuse it.
- `analysis/job_crop_compare.py` (`counts`, `restrict`, `compare`) and `job_crop.slurm`.
