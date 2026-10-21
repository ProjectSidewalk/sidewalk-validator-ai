# Sidewalk segmentation as a localisation prior for Obstacle and SurfaceProblem: first experiment (October 2026)

Issue [#44](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/44) asks two questions. Does an explicit
sidewalk mask (is the label point on the pedestrian path?) tell correct Obstacle and SurfaceProblem labels from
incorrect ones? And does it add anything to the production validators? This is the issue's first experiment: an
off-the-shelf segmenter run on the label-centred views that the training crops are cut from, thirteen
point-vs-sidewalk features, no retraining. The code is `analysis/sidewalk_seg.py`. Every number below comes from
its `report` stage. The full tables are in `segmentation-prior-2026-10/` (`report.md` and the CSVs), pinned by
`provenance.json`.

**The segmenter's sidewalk IoU on GSV has not been measured yet.** That needs Jon's hand-check of the contact sheets
(below). Until then every result here is provisional: the mask looks right on inspection, but nobody has verified it.

## Recommendation (provisional, pending the hand-check)

**Do not add it as a model feature. Do not route on it yet. Do not spend a GPU run on the fourth-channel experiment.**

- **No lift that this sample can detect.** Adding the features to a head on the frozen production features changes
  the out-of-fold AUC by +0.003 for both types. The CIs are -0.017 to 0.022 for Obstacle and -0.017 to 0.024 for
  SurfaceProblem. At n = 400 per type that rules out a lift above about +0.022 (Obstacle) and +0.024 (SurfaceProblem).
  It does not rule out a smaller one. On top of the production checkpoint's own score the change is +0.006 and
  +0.002. No pooled CI clears zero, whether rows or cities are resampled.
- **The signal does not carry to new cities.** The transfer test fits on the `test` split and predicts the three
  held-out cities. There, adding the features lowers AUC:
  - CLS + features is 0.045 below CLS alone. The CI is -0.081 to -0.010 with rows resampled and -0.071 to 0.068 with
    cities resampled, so the city-level CI includes zero.
  - The features alone score AUC 0.41, worse than chance (0.50 within city).
  - Several single features flip direction between the splits. Obstacle `sidewalk_frac_crop` is 0.44 on `test` and
    0.70 on `test_heldout` (0.41 and 0.65 within city).

  This result rests on three cities and 150 rows per type.
- **Alone, the features are weak.** Ranked by oriented AUC, max(AUC, 1 − AUC), with both splits pooled:
  - SurfaceProblem: the best is distance to the nearest sidewalk, AUC 0.617 (0.567–0.666; 0.615 within city). On
    `test` it is 0.675; on `test_heldout` it is 0.530. Being on a sidewalk is about as strong in the other direction:
    oriented 0.592, so labels on a sidewalk are more often correct.
  - Obstacle: the best is `curb_frac_crop`, AUC 0.414, oriented 0.586 (CI 0.360–0.470; 0.435 within city). More curb
    in the crop goes with a correct label. `near_sidewalk_1m` is next, at oriented 0.568. No other Obstacle feature
    reaches 0.55.
- **"Not on any sidewalk" separates the crowd's verdict only for SurfaceProblem, and only weakly.** Among AI
  Disagrees, 59% of the labels the crowd rejected are off the sidewalk, against 41% of those it accepted. The
  Wilson CIs (50–67% and 28–57%) overlap. For Obstacle the label sits on the obstacle, so 80–93% of every pool is
  "not on sidewalk" whatever the verdict. The "near the path" version (within 1 m of a sidewalk) does not separate
  either: 28% of rejected and 35% of accepted AI Disagrees are not near one.
- **Serving cost is not small.** Mask2Former Swin-L has 216M parameters (866 MB of weights) and segments 3.3 views/s
  on an L40 at 1024 px. That is a second model about as large as the validator, for no measured gain.

Two things would change this:
- the hand-check finds the mask is wrong often enough to explain the null (then try a better segmenter, not this
  one); or
- a production-proportioned sample of AI Disagrees shows that the SurfaceProblem gap above is real. This sample is
  class-balanced and held out, so it cannot show that.

## How the numbers were made

| Step | Where | What |
|---|---|---|
| Sample | Mac | 800 rows of the round-3 held-out sample: Obstacle and SurfaceProblem, 125 + 125 `test` and 75 + 75 `test_heldout` per type, correct and incorrect balanced; 32 deployments (28 per type), 768 panoramas. All human-authored, all tilt-corrected, none clipped |
| Stage | klone job 42112597 (`analysis/seg_prior_stage.slurm`) | 768 of 768 panoramas from the pano store in one rsync connection (read-only key), 7.4 GB, exit 0, 2.6 min. None needed the fallback copy |
| Render, check, segment | klone job 42113381 (`analysis/seg_prior.slurm`, commit b1a7761), NVIDIA L40, 51.8 min | `read_store_panorama` → `render_label_view` on the GPU torch path, with the label's pitch/roll, β 0.95 and tilt for human labels, in the conda env. That env has sidewalk-panorama-tools at 7ce25ff, the commit the round-3 generation recorded. The crop is re-cut at the manifest's `depth_m`. All 800 have the training crop's `crop_half_px`. Mean absolute difference from the training WebP crop: 1.33 grey levels (max 2.47; 99th-percentile pixel 5.4). For scale, on the same 800 training crops, a second q95 re-encode differs by 0.40 (median; max 0.97), and a 1-px horizontal shift by 4.7 (median; 5th–95th percentile 2.1–9.6). The freshly rendered crops were not kept, so their own encode-decode floor was not measured, and the 1.33 is reported without attributing it to WebP |
| Segmenter | same job | `facebook/mask2former-swin-large-mapillary-vistas-semantic` @ `4772b6bf`. The view is resized to 1024² (INTER_AREA), the processor size set to 1024×1024 (its default is 384), and semantic post-processing run at 1024². Class-id masks are saved at 1024 (features) and 512 (sheets). 243.5 s of segmentation for 800 views: 0.30 s/view, including pre- and post-processing |
| Coordinates | klone job 42113822 | `pano_y` for the view pitch, from the retrain's rawLabels exports. The render job's `views.csv` predates that column |
| Features | Mac | `mask_features` on the 1024 masks (below) |
| Production checkpoints re-scored on round-3 crops | Mac, M1 Max, MPS | The production validators at their serving revisions (Obstacle `8f0fe8fc`, SurfaceProblem `0e1439da`), preprocessed as `evaluate.py` does, applied to the round-3 tilt-corrected crops. Saved: `p_incorrect` and the 2048-d `[CLS ; mean(patch tokens)]` that DINOv2's classification head sees (checked: the head applied to it reproduces the logits). **This is not the served score.** Its verdict (`p_incorrect > 0.5` = Disagree) matches the served `ai_result` on 94.5% of the 783 joined labels (Obstacle 94.0%, SurfaceProblem 95.0%; `served_vs_rescored.csv`). Served verdicts date from 2025-08-17 to 2026-10-17, on crops cut without the tilt correction |
| Report | Mac, Python 3.12.14, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, 4 min | Single-feature AUC with 2,000-sample bootstrap CIs. 5-fold `StratifiedGroupKFold` by `pano_id` (seed 20261021). Logistic regression with C picked by inner 3-fold grouped CV. The 2048-d block is reduced to 64 PCA components inside each fold, so that 2048 coefficients under one penalty do not drown the 13 features. Paired bootstrap for ΔAUC, resampling rows and, separately, whole cities. Within-city AUC as defined below |

```
R=~/Data/sidewalk-validator-ai-retrain-2026-10/round3 O=~/Data/sidewalk-validator-ai-seg-2026-10 P=~/Data/sidewalk-validator-ai-prod-2026-10-17
python analysis/sidewalk_seg.py filelist --sample $R/sample.csv --out-dir $O
# klone, W=/gscratch/scrubbed/$USER/seg-prior-2026-10 holding labels.csv, stage_files.txt, crops/ and a clone:
#   cd $W && sbatch sidewalk-validator-ai/analysis/seg_prior_stage.slurm
#   (login node) HF_HUB_DISABLE_XET=1 HF_HOME=$W/hf python -c "...snapshot_download(...)"   # see seg_prior.slurm
#   cd $W && sbatch sidewalk-validator-ai/analysis/seg_prior.slurm
#   python analysis/sidewalk_seg.py coords --labels $W/labels.csv --rawlabels-dir <rawlabels> --out $W/coords.csv  # CPU sbatch
# pull $W/out -> $O/klone and $W/coords.csv -> $O
python analysis/sidewalk_seg.py features --labels $O/labels.csv --run-dir $O/klone --coords $O/coords.csv --out $O/features.csv
python analysis/sidewalk_seg.py cls --labels $O/labels.csv --crops-dir $R/crops --out $O/cls.npz
python analysis/sidewalk_seg.py sheets --features $O/features.csv --run-dir $O/klone --cities-json $P/cities.json \
    --out-dir $O/sheets --doc-fig $O/sample_views.png
python analysis/sidewalk_seg.py report --features $O/features.csv --cls $O/cls.npz \
    --assessments $P/ai_assessments_rounded4.csv --cities-json $P/cities.json --run-dir $O/klone \
    --out-dir docs/segmentation-prior-2026-10 --private-dir $O/private --extra-provenance $O/klone_provenance.json \
    --hash-extra sample_views_png=$O/sample_views.png
```

`provenance.json` records the following:
- the report's commit;
- the `features` stage's own commit, from its sidecar record;
- the `cls` stage's commit (b1a7761), hand-entered because that output predates the sidecar;
- the segmenter's parameter count, read from the cached weights;
- the klone job ids and timings, hand-entered from `sacct` under `hand_entered`.

### Features

The label point is the view centre. The class groups are:
- **sidewalk:** the Mapillary classes Sidewalk, Curb Cut and Pedestrian Area;
- **road:** Road, Service Lane, Bike Lane, Parking, Crosswalk - Plain and the two lane-marking classes;
- **curb:** Curb.

Ground objects (Manhole, Catch Basin, Pothole) are in none of the groups. A label on a manhole cover inside a
sidewalk therefore reads as "not on sidewalk". This applies to 4.5% of SurfaceProblem points and 0.5% of Obstacle
points.

| Feature | Definition |
|---|---|
| `point_on_sidewalk` | Centre pixel is sidewalk. `_disc`: the sidewalk share of a 15 px (view) disc is > 0.5. `sidewalk_frac_disc`: that share |
| `dist_px_to_sidewalk` | View pixels to the nearest sidewalk pixel (0 if on it). A view with no sidewalk is scored as the view diagonal |
| `dist_m_to_sidewalk` | `dist_px × R / 1024`, where f = 1024 px (90° over 2048 px) and R = 2.5 m / sin(−pitch) is the ground range implied by the view pitch, for a camera 2.5 m above flat ground (pitch clamped at −2°, so R ≤ 72 m). Small-angle, same-range approximation. It assumes the label point is on the ground: true for a surface problem or an obstacle's base, too short for a point on an obstacle's body |
| `dist_m_depth_to_sidewalk` | The same at the manifest's `depth_m` (Depth Anything V2 vkitti, same-depth approximation): the issue's literal recipe, kept for comparison |
| `near_sidewalk_1m` | `dist_m_to_sidewalk ≤ 1` (ground range), the issue's "near the path" reading for Obstacle |
| `sidewalk_frac_crop`, `road_frac_crop`, `curb_frac_crop` | Shares of each class group inside the depth-sized crop square |
| `sidewalk_frac_view` | Share of the whole view |
| `on_road` | Centre pixel is road |
| `connected_to_bottom` | The sidewalk component that holds the point (or is nearest to it) reaches the bottom 10% of the view: "the path continues to the camera" |

Side observation: the vkitti metric depth on these views is about 2.5–2.7 times the pitch-implied ground range
(median 2.50 for Obstacle, 2.70 for SurfaceProblem; interquartile range 2.25–2.93 pooled). Metres computed from
`depth_m` are therefore larger by that factor. A 1 m threshold on them means about 0.4 m of ground.

**Within-city AUC.** This is the row-weighted mean of the per-city AUCs, over cities that have both classes and at
least 10 rows. It qualifies 7 cities for Obstacle and 9 for SurfaceProblem with both splits pooled, 3 on
`test_heldout`.

It is reported next to every pooled AUC because the share of incorrect labels varies from 0 to 100% across cities
in this sample. On the Obstacle held-out cities it is 12% in one city and 82% in another. A score that only tells a
mostly-correct city from a mostly-incorrect one can therefore raise the pooled AUC without separating anything
within a city.

## Results

### Segmenter sanity (no ground truth yet)

| | All | Obstacle | SurfaceProblem |
|---|---|---|---|
| Views with any sidewalk | 98.1% | 97.5% | 98.8% |
| Median sidewalk share of the view | 3.9% | 3.8% | 4.2% |
| Views with > 5% of their sidewalk above the horizon | 0.5% | 0.3% | 0.7% |
| Sidewalk pixels above the horizon | 0.1% | 0.0% | 0.3% |
| Label point on sidewalk | 36.5% | 17.2% | 55.7% |
| Label point on road | 11.3% | 4.8% | 17.7% |
| Label point on a ground object (manhole, catch basin, pothole) | 2.5% | 0.5% | 4.5% |
| Within 1 m (ground range) of sidewalk | 80.1% | 78.3% | 82.0% |

At an Obstacle label point the segmenter's class is Sidewalk 16%, Vegetation 13%, Trash Can 12%, Pole 10% and Car
10%. It sees the obstacle, as the issue predicted. At a SurfaceProblem point it is Sidewalk 52%, Road 16% and
Terrain 8%.

**Hand-check packet (Jon).** There are four contact sheets of 40 views each, two per type, public deployments only,
in `~/Data/sidewalk-validator-ai-seg-2026-10/sheets/` (about 580 KB each). Each view shows the mask overlay, the crop
square, the centre mark, the `label_uid`, the class and the sidewalk flag. The masks behind them are in
`klone/masks_512/`. A six-view sample figure is at `~/Data/sidewalk-validator-ai-seg-2026-10/sample_views.png`
(sha256 in `provenance.json`). It is not committed: the repo has not held Street View–derived imagery before, so
that is the maintainer's decision.

### Each feature alone

AUC for `incorrect` as the positive class, both splits pooled (n 400 per type, 2,000 bootstrap resamples). Below
0.5 means a higher value goes with a correct label; "oriented" is max(AUC, 1 − AUC). Per-split values are in
`feature_auc.csv`. This table holds 26 AUCs, and `feature_auc.csv` holds 78 with no multiple-comparison correction.
Around one in twenty of the CIs is expected to miss 0.5 by chance, so a lone feature that clears 0.5 is weak
evidence on its own.

| Feature | Obstacle AUC (CI) | within city | SurfaceProblem AUC (CI) | within city |
|---|---|---|---|---|
| `point_on_sidewalk` | 0.528 (0.490–0.564) | 0.542 | 0.408 (0.362–0.457) | 0.410 |
| `sidewalk_frac_disc` | 0.549 (0.500–0.599) | 0.589 | 0.408 (0.358–0.463) | 0.413 |
| `dist_m_to_sidewalk` | 0.515 (0.458–0.571) | 0.479 | 0.617 (0.567–0.666) | 0.615 |
| `dist_m_depth_to_sidewalk` | 0.507 (0.448–0.562) | 0.471 | 0.615 (0.565–0.664) | 0.614 |
| `near_sidewalk_1m` | 0.432 (0.392–0.471) | 0.444 | 0.425 (0.388–0.461) | 0.431 |
| `sidewalk_frac_crop` | 0.541 (0.484–0.600) | 0.518 | 0.458 (0.400–0.516) | 0.410 |
| `sidewalk_frac_view` | 0.532 (0.478–0.589) | 0.491 | 0.443 (0.387–0.503) | 0.408 |
| `road_frac_crop` | 0.487 (0.430–0.545) | 0.546 | 0.541 (0.482–0.595) | 0.512 |
| `curb_frac_crop` | 0.414 (0.360–0.470) | 0.435 | 0.452 (0.395–0.507) | 0.462 |
| `on_road` | 0.507 (0.486–0.529) | 0.494 | 0.568 (0.530–0.603) | 0.552 |
| `connected_to_bottom` | 0.535 (0.505–0.566) | 0.513 | 0.495 (0.463–0.527) | 0.467 |

- **SurfaceProblem** behaves as the issue hoped on the `test` split: distance AUC 0.675 (CI 0.613–0.735), on
  sidewalk 0.372. It does not on the held-out cities: 0.530 and 0.467.
- **Obstacle** goes the "wrong" way on the held-out cities, where incorrect labels sit on more sidewalk than correct
  ones: `sidewalk_frac_crop` 0.695 (0.603–0.780). Within city that is still 0.645, so the city mix does not explain
  the flip.

### Lift over the frozen production features

Out-of-fold AUC. Each ΔAUC is paired against the model it adds to (its base): the CLS head for "CLS + seg", the
re-scored checkpoint for "Re-scored + seg". "Row CI" resamples labels; "city CI" resamples whole cities. With three
held-out cities the city CI is coarse, but it is the interval that speaks to unseen cities.

| Type | Slice | Re-scored checkpoint | CLS head | CLS + seg | ΔAUC vs CLS: row CI; city CI; within city | Seg only | Re-scored + seg | ΔAUC vs re-scored: row CI; city CI; within city |
|---|---|---|---|---|---|---|---|---|
| Obstacle | both splits, 5-fold | 0.828 | 0.845 | 0.847 | +0.003: (−0.017, 0.022); (−0.021, 0.022); −0.001 | 0.634 | 0.834 | +0.006: (−0.016, 0.026); (−0.034, 0.039); +0.021 |
| Obstacle | `test` rows | 0.821 | 0.813 | 0.823 | +0.010: (−0.016, 0.039); (−0.039, 0.032); +0.020 | 0.580 | 0.810 | −0.011: (−0.035, 0.014); (−0.065, 0.014); +0.006 |
| Obstacle | `test_heldout` rows | 0.838 | 0.890 | 0.876 | −0.014: (−0.052, 0.018); (−0.029, −0.007); −0.026 | 0.723 | 0.864 | +0.026: (−0.009, 0.066); (0.001, 0.071); +0.039 |
| Obstacle | fit `test` → `test_heldout` | 0.838 | 0.846 | 0.800 | **−0.045: (−0.081, −0.010)**; (−0.071, 0.068); −0.028 | 0.410 | 0.793 | **−0.044: (−0.081, −0.007)**; (−0.083, 0.006); −0.038 |
| SurfaceProblem | both splits, 5-fold | 0.822 | 0.810 | 0.813 | +0.003: (−0.017, 0.024); (−0.017, 0.027); +0.009 | 0.656 | 0.825 | +0.002: (−0.017, 0.022); (−0.015, 0.018); +0.030 |
| SurfaceProblem | `test` rows | 0.899 | 0.872 | 0.884 | +0.012: (−0.009, 0.032); (−0.021, 0.039); +0.024 | 0.698 | 0.892 | −0.007: (−0.026, 0.010); (−0.031, 0.008); +0.024 |
| SurfaceProblem | `test_heldout` rows | 0.659 | 0.681 | 0.669 | −0.012: (−0.061, 0.035); (−0.034, 0.016); −0.009 | 0.586 | 0.692 | +0.033: (−0.008, 0.076); (0.011, 0.083); +0.038 |
| SurfaceProblem | fit `test` → `test_heldout` | 0.659 | 0.560 | 0.554 | −0.006: (−0.055, 0.042); (−0.017, 0.018); +0.002 | 0.493 | 0.641 | −0.018: (−0.050, 0.015); (−0.054, 0.002); −0.027 |

Within-city AUCs of the models themselves (`lift.csv`):

| Type | Slice | Re-scored checkpoint | CLS head | CLS + seg | Seg only |
|---|---|---|---|---|---|
| Obstacle | both splits | 0.793 | 0.814 | 0.814 | 0.660 |
| Obstacle | fit `test` → `test_heldout` | 0.757 | 0.799 | 0.771 | 0.499 |
| SurfaceProblem | both splits | 0.744 | 0.740 | 0.749 | 0.628 |
| SurfaceProblem | fit `test` → `test_heldout` | 0.670 | 0.527 | 0.529 | 0.479 |

How to read the slices:
- **`test` rows and `test_heldout` rows** are subsets of the pooled 5-fold predictions. In those rows, the held-out
  cities were also in the training folds.
- **Fit `test` → `test_heldout`** is the only real new-city test, and it rests on three cities. Its two bold row
  CIs clear zero downward. Neither does with cities resampled.
- **The positive city CIs** are on in-sample `test_heldout` rows ("Re-scored + seg": Obstacle 0.001–0.071,
  SurfaceProblem 0.011–0.083). The same models lose on the true transfer. With three cities, a city CI has only a
  few distinct resamples, so treat it as coarse.
- **The hard slice** has 1 Obstacle row and 3 SurfaceProblem rows, all incorrect, so it has no AUC.
- **Within city** the pooled picture holds: no gain in-sample, a loss on transfer.

Per public city with n ≥ 30 (private deployments pooled), from the pooled 5-fold predictions:

| Type | City | n | incorrect | Re-scored checkpoint | CLS head | CLS + seg | Seg only | Re-scored + seg |
|---|---|---|---|---|---|---|---|---|
| Obstacle | chicago-il | 42 | 23 | 0.915 | 0.819 | 0.842 | 0.613 | 0.883 |
| Obstacle | seattle-wa | 76 | 36 | 0.832 | 0.819 | 0.872 | 0.666 | 0.846 |
| Obstacle | other (6 private deployments) | 196 | 91 | 0.800 | 0.879 | 0.869 | 0.666 | 0.837 |
| SurfaceProblem | chicago-il | 33 | 10 | 0.896 | 0.813 | 0.848 | 0.696 | 0.896 |
| SurfaceProblem | seattle-wa | 54 | 28 | 0.883 | 0.883 | 0.878 | 0.544 | 0.880 |
| SurfaceProblem | tucson-az | 33 | 23 | 0.535 | 0.526 | 0.526 | 0.522 | 0.617 |
| SurfaceProblem | other (6 private deployments) | 196 | 110 | 0.788 | 0.785 | 0.792 | 0.664 | 0.798 |

Per-city n is 33–196, so a single city's AUC moves by several points on a handful of labels.

The production checkpoints re-scored on this sample (`evaluate.slice_metrics`) are below. Plain accuracy is omitted
because the sample is 50/50. Among the labels eligible for training, the correct share is 64% for Obstacle and 68%
for SurfaceProblem (pre-downsampling counts in the round-3 `deployment_priors.json`).

| Type | Split | AUC | Balanced accuracy | Recall of incorrect at 90% precision |
|---|---|---|---|---|
| Obstacle | `test` | 0.821 | 0.740 | 0.384 |
| Obstacle | `test_heldout` | 0.838 | 0.720 | 0.053 |
| SurfaceProblem | `test` | 0.899 | 0.796 | 0.592 |
| SurfaceProblem | `test_heldout` | 0.659 | 0.607 | 0.040 |

### By production verdict (the routing question)

783 of the 800 labels have a production assessment, joined one-to-one on `label_uid` (built from `schema` +
`label_id`). The table gives the share of each pool whose point is not on a sidewalk, and not within 1 m (ground
range) of one, split by the sample's human class, with Wilson 95% CIs. The "other" pool (no or unusable
assessment, n 17) is in `strata.csv`.

| Type | Pool | Human class | n | Not on sidewalk (CI) | Not within 1 m (CI) |
|---|---|---|---|---|---|
| Obstacle | AI Disagree | correct | 40 | 0.93 (0.80–0.97) | 0.35 (0.22–0.51) |
| Obstacle | AI Disagree | incorrect | 119 | 0.80 (0.72–0.86) | 0.28 (0.21–0.36) |
| Obstacle | Contested Agree | correct | 29 | 0.86 (0.69–0.95) | 0.17 (0.08–0.34) |
| Obstacle | Contested Agree | incorrect | 74 | 0.80 (0.69–0.87) | 0.26 (0.17–0.37) |
| Obstacle | Uncontested Agree | correct | 124 | 0.82 (0.75–0.88) | 0.09 (0.05–0.15) |
| SurfaceProblem | AI Disagree | correct | 41 | 0.41 (0.28–0.57) | 0.20 (0.10–0.34) |
| SurfaceProblem | AI Disagree | incorrect | 130 | 0.59 (0.50–0.67) | 0.28 (0.21–0.37) |
| SurfaceProblem | Contested Agree | correct | 12 | 0.42 (0.19–0.68) | 0.08 (0.02–0.35) |
| SurfaceProblem | Contested Agree | incorrect | 68 | 0.46 (0.34–0.57) | 0.21 (0.13–0.32) |
| SurfaceProblem | Uncontested Agree | correct | 146 | 0.32 (0.25–0.40) | 0.08 (0.04–0.13) |

An uncontested Agree has no human Disagree, so in this sample it is always `correct`. Compare classes within a pool,
not pools with each other.

- **SurfaceProblem:** in the AI Disagree pool, rejected labels are off the sidewalk more often than accepted ones
  (+18 points, CIs overlapping). In the contested-Agree pool the gap is only +4.
- **Obstacle:** the AI Disagrees the crowd *accepted* are, if anything, farther from the path than the ones it
  rejected. A "far from the path" rule would route the wrong labels.

For scale, the AI disagrees with 38.8% of Obstacle and 24.8% of SurfaceProblem assessments in the 2026-10-17
production table.

## Caveats

- **No segmentation ground truth.** The IoU the issue asks for comes from the hand-check. The sanity numbers and
  the sheets look right: 98% of views have sidewalk, and 0.1% of sidewalk pixels are above the horizon. But they
  cannot show how often the boundary is off by the metre that matters here.
- **The baseline may have seen these labels.** The production ViT-L models were trained in 2025-08 on an
  undocumented label set that may overlap this sample. If they memorised some of these labels, the re-scored
  checkpoint and the CLS head are inflated, which leaves less room for the segmentation features (which cannot have
  seen them) to add anything. That bias runs toward a null. On truly unseen labels the lift could be larger than
  measured here. The retrained round-3 checkpoints were not available.
- **The heads are fitted on the sample itself.** The CLS head is a new logistic head fitted on 320 rows per fold. It
  is not the production head. It is the comparison the issue asked for, not a deployable model.
- **Row-level CIs are too narrow for city claims.** Labels within a city are not independent, so the single-feature
  CIs and the "row CI" columns overstate precision for anything about new cities. The city-resampled CIs are the
  ones to read for transfer, and with three held-out cities they are coarse.
- **City–class confound.** The incorrect share runs from 0 to 100% across cities. The within-city columns control
  for it, and the conclusions are the same.
- **Class balance.** The sample is 50/50 by construction, so plain accuracy and the pool shares are not production
  numbers. A routing decision would need the same measurement on a production-proportioned sample of AI Disagrees.
- **Small and private-heavy.** 400 labels per type. 392 of the 800 rows come from six private deployments, and two
  of the three held-out cities are private. Among public cities, only Seattle, Chicago and (for SurfaceProblem)
  Tucson reach n ≥ 30.
- **Metres are approximate.** The ground range assumes flat ground, a 2.5 m camera and a label on the ground. It also
  puts the nearest sidewalk pixel at the label's range. The 1 m threshold was fixed before any table existed and
  was not tuned. The first version of this note used the vkitti `depth_m` and so effectively thresholded at about
  0.4 m. The switch moved the "within 1 m" shares (70% → 80% overall) but left the conclusions unchanged.
- **One segmenter, one input size.** Mask2Former Swin-L (Mapillary Vistas v1.2) at 1024². Not tried: a Cityscapes
  SegFormer, a larger input, or a crop-level mask.

## What to do next

1. **Jon: hand-check the four sheets.** For each view, judge whether the sidewalk boundary near the label point is
   right or wrong.
   - If the mask is wrong near the point in more than about one view in five, the null result says little about the
     idea. The next step is then a better segmenter.
   - Otherwise, close the feature arm of [#44](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/44).
2. **Do not run the fourth-channel experiment** on this evidence. That is the mask as a fourth input channel, shaped
   like the point-heatmap arm of
   [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14). The issue made it conditional on a
   clear lift in step 3, and there is none.
3. **If routing is still wanted for SurfaceProblem**, measure "off sidewalk" on a production-proportioned sample of
   AI Disagrees, not a class-balanced one, and decide there. That is a short GPU rerun of `render-segment` on a new
   label list.
4. **Serving cost, if it ever ships:** 216M parameters, 866 MB of weights, 0.30 s/view on an L40 at 1024². A
   distilled or smaller segmenter
   ([#20](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/20)) would be needed before
   sidewalk-ai-api could run it per label. With no gain measured, that is not worth doing now.
