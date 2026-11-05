# More context for the validator: two-scale crops, whole view with a point heatmap, side information (November 2026)

Issue [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14). Three ways to show the round-3
DINOv2-L validator more than its tight, depth-sized crop, each trained with the round-3 recipe on the round-3 split
and scored on the same rows as the round-3 model.

## Status

**Not yet run.** The code, the tests and the klone job files are in place; no arm has been trained and no number
below is a result. What exists:

- `context_crops.py`: cuts the two context images (a 4x wider crop and the whole view at 448 px) from the same
  label-centred view the training crop comes from; `context_crops.py backfill` writes them for an existing dataset;
  `data_generation.py --context-crops` writes them for a new one.
- `context_model.py`: the three arm models, their input transforms, and `predict` (writes
  `predictions_<tag>.csv` for `evaluate.py --skip-inference --tag`).
- `fine_tune_classifier.py --context-arm two_scale|view_heatmap|side_info [--context-view-res 224|448]
  [--batch-size N]`.
- `analysis/context_arms.py report`, `analysis/context_arms.slurm`, `analysis/context_submit.sh`.

The exact commands are in [How to run](#how-to-run).

## Why

The validator sees a square crop whose half-width is `6100 / depth` px of the 2048 px view (median about 238 px,
[#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51)). For some label types the question a
human validator answers is about the surroundings, not the crop: is this obstacle actually in the walking path,
is there a curb ramp a metre to the left of this NoCurbRamp, is this surface problem on the sidewalk or on the
road. Both earlier validators used more than one crop: Weld et al. (ASSETS 2019,
https://doi.org/10.1145/3308561.3353798) fed non-image features alongside the crop, and Duan et al. (ASSETS 2022,
https://doi.org/10.1145/3517428.3550381, section 4) argued that fixed crops cut off the context that missing curb
ramp and obstacle judgments need.

The types this targets are the three weakest in round 3 on the deployments never seen in training
([`retrain-2026-10.md`](retrain-2026-10.md)):

| type | test AUC | unseen-deployment AUC | within-city AUC, unseen |
|---|---:|---:|---:|
| NoCurbRamp | 0.934 | 0.881 | 0.861 |
| Obstacle | 0.883 | 0.868 | 0.788 |
| SurfaceProblem | 0.896 | 0.710 | 0.682 |

CurbRamp and Crosswalk are run as well (the same chain costs little more), but they are not where a gain is
expected.

## The three arms

| arm | input | head | what it tests |
|---|---|---|---|
| `two_scale` | the tight crop and a crop 4x wider around the same point; one shared backbone, `[CLS, mean(patches)]` of each, concatenated | 4 x 1024 | does nearby context ("is it blocking the path") help? |
| `view_heatmap` | the whole 90-degree label-centred view, downsampled (stored at 448 px, trained at 224 by default), plus a 4th channel: a Gaussian at the label point | 2 x 1024 | RampNet-style point conditioning on the full view |
| `side_info` | the tight crop plus 9 normalised non-image features from `split.csv` | 2 x 1024 + 9 | Weld et al.-style side information |
| baseline | round 3 as trained (seed 42) and its seed-43/44 replicates from the confident-learning batch | | not retrained |

Every arm pools like `Dinov2ForImageClassification` and trains with the round-3 recipe: DINOv2-L, 25 epochs,
seed 42, learning rate 1e-5, warm-up 10%, effective batch 32, hard targets, best checkpoint by validation macro-F1.
The models are never registered with `AutoModel` (`model_type` `context_dinov2`), so `evaluate.py` and
sidewalk-ai-api cannot load one by accident, and `--context-arm` refuses to run without `--output-dir`, so an arm
never lands in `<Type>/dinov2/` where `evaluate.py` and `publish_to_hf.py` look by default.

**Decisions** (the plan asked for these to be recorded):

- **Wide factor 4**, the issue's number, a parameter (`--factor` / `--context-wide-factor`). At 4x the wide crop is
  capped at the whole view (half-width 1024) whenever the tight half-width exceeds 256 px, which is roughly half of
  the labels, so for those labels `two_scale` sees the tight crop and the view. This is reported, not hidden:
  `clipping.md` gives the share per type from the backfilled index. If more than about 60% are capped, a 2.5x
  second pass is the next step.
- **Stored view 448 px, trained at 224 by default.** 448 is twice the model input (32 x 32 DINOv2 patches), so a
  448 run needs no upsampling. 224 keeps the token count and memory of the baseline (the memory-fair comparison);
  448 (`--context-view-res 448`, `TAG=view_heatmap-r448`) is the detail-fair one at 4x the tokens and about 3x the
  step time, planned for the types where 224 comes close. 2048 to 224 is a 9x reduction, so the 224 view loses
  detail the tight crop has.
- **The heatmap carries scale as well as position.** At evaluation the label is always at the view centre, so a
  fixed-width Gaussian would be the same image for every label: a constant channel. Its sigma is half of the tight
  crop's half-width mapped into the input (`heatmap_sigma`: `crop_half_px * res / 2048 / 2`, at least 4 px; about
  13 px for the median label at 224), so the channel says where the label is and how large the depth-sized crop
  around it is. This departs from the issue's "a Gaussian at the label position"; the plain point is
  `heatmap_sigma` with `crop_half_px = 0` and a fixed floor.
- **The view skips the processor's centre crop.** The checkpoint's processor resizes to 256 and centre-crops 224,
  dropping the outer 12.5% on each side, which for this arm is exactly the context under test. The view is resized
  to `res` x `res`; in training it goes through `RandomResizedCrop(scale=(0.7, 1.0))` and a flip like the tight
  crop, and the heatmap is computed analytically for the transformed point and sigma (`map_point`, checked against
  torchvision's own `resized_crop` in the tests).
- **4-channel patch embedding**: channels 0 to 2 are the pretrained filters, channel 3 starts at zero, so the model
  computes exactly the pretrained function at step 0 (tested) and learns to use the heatmap.
- **Side features** (`context_model.side_feature_frame`): severity, number of tags, imagery age in years (against
  the latest capture date in the train rows, saved in the checkpoint), log depth, high-quality user, clipped, plus
  missing-value flags for severity, age and depth. Missing values take the train median; every column is
  standardised on the train rows; the normaliser is saved in the checkpoint's `config.json` so inference applies
  the same transform.
- **The wide crop's augmentation is drawn independently of the tight crop's** (its own `RandomResizedCrop` and
  flip); the evaluation transform of the wide crop is the processor's own, like the tight crop's.

## Setup

| | |
|---|---|
| Data | the round-3 dataset on klone (`retrain-2026-10/<Type>/`): crops, manifest and `split.csv` unchanged; the backfill adds only `<Type>/context/` |
| Split | the round-3 `split.csv` (panorama-grouped, three deployments held out); every arm trains on its train rows and is scored on its val/test/test_heldout rows |
| Recipe | round 3 (above); `--batch-size` only moves the split between per-device batch and gradient accumulation, effective batch 32 either way |
| Seeds | 42 for every arm; 43 and 44 for any arm whose held-out Δ interval excludes zero (the baseline's seed-43/44 runs exist already as `metrics_baseline-s43.json` / `-s44`) |
| Evaluation | `context_model.py predict` then `evaluate.py --skip-inference --tag <arm> --per-city`; report by `analysis/context_arms.py` |
| Compute | 5 types x 3 arms at seed 42 = 15 GPU trainings on `ckpt-all` (A40/L40/L40S/A100/H200), plus 5 backfills; `two_scale` makes two backbone passes per step |
| Storage | about 100 KB per label (a 448 px view of about 40 KB and a 640 px wide crop of about 60 KB at WebP q95), about 6.5 GB for all five types beside the crops on `/gscratch/makelab` (87 GB free on 2026-10-28); checkpoints on `/gscratch/scrubbed` |

## How to run

On klone, from a checkout of this branch, with the round-3 data and its rawLabels snapshot beside it
(`$DATA/../rawlabels`, as `cl_submit.sh` assumes):

```
DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 \
RUNS=/gscratch/scrubbed/$USER/context-2026-11 \
PANO_STORE=/path/to/Panoramas \
    analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem CurbRamp Crosswalk
```

Per type this submits the backfill (GPU; `context_crops.py backfill --verify 50`) and, after it, the
`two_scale` and `view_heatmap` trainings; `side_info` reads only `split.csv` and starts at once. Each training job
scores its best checkpoint with `context_model.py predict` and runs `evaluate.py --skip-inference --tag <arm>
--per-city`, so `metrics_<arm>.json` and `predictions_<arm>.csv` land next to the baseline's. Job ids go to
`context_jobs.txt`. Single steps:

```
STEP=backfill LT=NoCurbRamp DATA=$DATA PANO_STORE=/path/to/Panoramas sbatch analysis/context_arms.slurm
STEP=train LT=NoCurbRamp ARM=view_heatmap DATA=$DATA RUNS=$RUNS sbatch analysis/context_arms.slurm
STEP=train LT=NoCurbRamp ARM=view_heatmap RES=448 TAG=view_heatmap-r448 DATA=$DATA RUNS=$RUNS \
    sbatch analysis/context_arms.slurm
STEP=train LT=NoCurbRamp ARM=two_scale SEED=43 TAG=two_scale-s43 KEEP_CKPT=0 DATA=$DATA RUNS=$RUNS \
    sbatch analysis/context_arms.slurm
```

The backfill re-renders each ok row's view from the pano store with the pose its crop was cut under (pano points
from the rawLabels snapshot, the pose from `crop_pose.pose_for_row` exactly as the generator builds it, checked
against the manifest's pose columns; round-3 rows predate those columns, so their `tilt_beta` and `tilt_applied` are
checked instead) and refuses on any mismatch. `--verify 50` re-cuts 50 tight crops from the re-rendered views and
fails above a mean absolute difference of 12 grey levels against the stored WebPs (WebP q95 alone accounts for a
few). It is resumable and writes `context/context_run.json` (factor, view size, renderer, a hash of the
generation settings); a later backfill under other settings refuses rather than mixing two geometries in one index.

Then, anywhere with the outputs (no GPU):

```
python analysis/context_arms.py report --data-dir $DATA --public-cities cities.json \
    --arms two_scale,view_heatmap,side_info --out docs/context-arms-2026-11
```

`--public-cities` takes a saved `/v3/api/cities` response (default: the live API at report time); only public
deployments are named, private held-out deployments are lettered as in the round-3 report.

## Results

Not yet run. The report writes, under [`context-arms-2026-11/`](context-arms-2026-11/):

- `arms.md` / `arms.json`: per type x arm, test and held-out AUC with the paired panorama-bootstrap Δ against
  the baseline on the same rows, within-city AUC, macro F1 at the model's verdict, incorrect and correct recall at
  precision 0.92 and 0.95, and the held-out deployments one by one.
- `seeds.md`: replicates paired with the same-seed baseline.
- `slices.md`: type x severity and type x tag, each arm against the baseline, with the slice's training rows and
  the thin flag.
- `clipping.md`: the share of wide crops capped at the whole view, per type, and `wide_half_px` quantiles.
- `provenance.json`: sha256 of every file read.

| type | arm | test AUC (Δ) | held-out AUC (Δ) | within-city held-out |
|---|---|---:|---:|---:|
| NoCurbRamp | two_scale / view_heatmap / side_info | not yet run | not yet run | not yet run |
| Obstacle | two_scale / view_heatmap / side_info | not yet run | not yet run | not yet run |
| SurfaceProblem | two_scale / view_heatmap / side_info | not yet run | not yet run | not yet run |
| CurbRamp | two_scale / view_heatmap / side_info | not yet run | not yet run | not yet run |
| Crosswalk | two_scale / view_heatmap / side_info | not yet run | not yet run | not yet run |

## What could ship

- **`two_scale`**: sidewalk-ai-api already renders the 2048 px view and computes the crop half-width; it would cut
  one more `centre_crop` and run the backbone twice, roughly doubling validator latency. Deployable with a small API
  change and a new model loader.
- **`view_heatmap`**: the API has the view and the depth; it would downsample the view and build the heatmap from
  the crop half-width. One backbone pass, but a 4-channel model is not loadable with
  `AutoModelForImageClassification`, so the API needs a loader for it.
- **`side_info`**: the `/process` request carries `label_type`, `panorama_id`, `x`, `y`, the camera pose fields,
  `coord_frame` and `city`. Depth and clipping the API computes itself, but severity, tags, the labeler's
  high-quality flag and the imagery capture date are not in the request today; SidewalkWebpage would have to send
  them. Least deployable; read its result as an upper bound on what side information adds, not as a candidate.

Any arm that wins needs its own serving-parity stage
([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5),
[`serving-parity-2026-10.md`](serving-parity-2026-10.md)) before a deploy. `publish_to_hf.py` copies checkpoint
files without loading them, so it does not refuse an arm checkpoint pointed at explicitly; a guard on
`training_run.json`'s `context_arm` (as it already refuses `fixed_crop_half_px`) belongs with any proposal to ship
one.

## Threats to validity

- **Seed noise.** The baseline's held-out AUC varies across seeds 42/43/44 with a sample sd of 0.006 (CurbRamp),
  0.008 (Obstacle), 0.010 (SurfaceProblem), 0.014 (Crosswalk) and 0.021 (NoCurbRamp)
  ([`confident-learning-2026-10/seeds.md`](confident-learning-2026-10/seeds.md)). A single-seed Δ of that size is
  not a finding; that is why replicates follow any arm whose interval excludes zero.
- **The wide crop is often the whole view** at 4x (see the decisions above), so `two_scale` partly tests "tight
  crop plus whole view at 224", not a graded context ring.
- **Detail at 224.** The view arm sees the scene at 1/9 of the rendered resolution; a null result at 224 does not
  rule out the arm, which is what the 448 run is for.
- **Batch size.** If `two_scale` runs with `--batch-size 4` (it should fit at 8 on a 48 GB card in fp32, untested),
  the effective batch is unchanged but the per-step noise and batch statistics are not identical to the baseline.
- **Backfill fidelity.** The context images are re-rendered, not saved at generation time. `--verify` bounds the
  difference on the tight crop, and on the synthetic test store the hook and the backfill write byte-identical
  files; a GPU backfill of a numpy-rendered dataset (or the reverse) differs by rounding and prints a warning.
- **Side features are what the crowd saw**: severity, tags and the high-quality flag come from the labeler, so
  `side_info` can learn labeler habits that do not transfer to new deployments; the held-out deployments are the
  check.
