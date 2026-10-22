# Round-3 retrain of the validators (October 2026)

The five validator models were retrained on a panorama-grouped split with tilt-corrected, label-centred crops and
three deployments held out of training entirely. This note reports what came out: results on the training
deployments' test share and on the unseen deployments, the comparison with the models in production, the
calibration maps the serving API would use at its 0.92 gate, and a per-type recommendation on what to ship.
It is the first measurement, from one seed. The decisions behind the data are in
[`decisions.md`](decisions.md); the audit that led here is [`audit-2026-10-14.md`](audit-2026-10-14.md).

**In short.**

- **On the training deployments, every type ranks held-out labels well** (AUC 0.88 to 0.98). On the three
  unseen deployments, CurbRamp, NoCurbRamp and Crosswalk lose 0.04 to 0.06 of AUC, Obstacle 0.02, and
  **SurfaceProblem 0.19** (0.90 to 0.71). The unseen deployment outside the US is the weakest for every type but
  Crosswalk.
- **Against the production models on the same 1,988 crops, round 3 is level or ahead:** +0.07 AUC on Obstacle in
  the training deployments (the only difference whose interval excludes zero), small positive differences
  elsewhere, and −0.02 on SurfaceProblem in the unseen deployments. Production was trained in 2025-08 on an
  unrecorded label set that overlaps the training-deployment crops, so that half of the comparison flatters
  production; the unseen deployments were not in production's training cities either.
- **Pooled AUC on the unseen deployments overstates discrimination** because the deployments differ so much in
  their incorrect share (4% to 97%). The within-city AUC is what a deployment would see: 0.84 for CurbRamp,
  0.79 for Obstacle and 0.68 for SurfaceProblem on the unseen deployments.
- **At the 0.92 gate, Agree votes clear with precision 0.96 to 0.996 in the training deployments and
  0.89 (SurfaceProblem) to 0.996 (Crosswalk) in the unseen ones.** The shipped maps hold every Disagree verdict
  back below the gate. Whether a Disagree map could clear the gate depends on the deployment prior, which is
  under review (below).
- **Recommendation:** ship the CurbRamp, NoCurbRamp, Obstacle and Crosswalk models with Agree-only maps in the
  deployments they were trained on; serve SurfaceProblem in shadow mode (both verdicts held back); hold Disagree
  back for every type until the prior question is settled; and gate new deployments per deployment as
  [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52) proposes.

**Reproduce.** The tables and figures come from `analysis/retrain_report.py` on the five `evaluate.py` output
folders (`predictions.csv`, `metrics.json`, `split.csv`), with the production scores that
`analysis/vlm_baseline.py baseline` wrote for [`vlm-baseline-2026-10.md`](vlm-baseline-2026-10.md):

```
python analysis/retrain_report.py --run-dir ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 \
    --cities-json <saved /v3/api/cities> --sample .../round3/sample.csv \
    --production-scores ~/Data/sidewalk-validator-ai-vlm-2026-10/scores/production.csv --out-dir docs/retrain-2026-10
```

The calibration tables come from `python -m calibration.calibrate` on the same predictions (fit on `val`,
evaluated on `test` and on `test_heldout`), and the per-deployment gate table from applying those maps to the
predictions with `calibration/_isotonic_map.py`'s lookup (`gate_per_city.csv`). Intervals are 95% percentile
intervals from 2,000 resamples of whole panoramas, seed 20261022. Deployments are named only if the live
`/v3/api/cities` lists them as public; the private training deployments are pooled into one row, and the two
private unseen deployments are "private city A (outside the US)" and "private city B (US)".

## What round 3 is

Round 3 is the first full retrain since the audit, on the pipeline as it stands after
[#23](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/23) through
[#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51). Data generation ran on 2026-10-20 and
training on 2026-10-20/21 on klone; the eval outputs are dated 2026-10-22.

| Setting | Round 3 | Production models (2025-08) |
|---|---|---|
| Deployments | 54 (every deployment with GSV labels, private ones included; three held out) | 19, hard-coded |
| Labels | human votes only (`validations[].validator_type != "AI"`), margin rule plus two expert accounts' votes (`class_source` margin 7,453 / expert 3,000 for CurbRamp) | `agree_count` / `disagree_count`, which include the AI's own votes |
| View | label-centred, camera yaw and pitch to the label, 90° FOV, 2048² | pitch 0 |
| Tilt | rig-tilt correction from `camera_pitch` / `camera_roll`, β = 0.95 (`sidewalk-panorama-tools` v1.0.0) | none |
| Crop size | depth-sized (`6100 / depth`), Depth Anything V2 ViT-B metric (vKITTI) | depth-sized, ViT-L |
| Split | by panorama, panoramas within 3 m merged, 70 / 15 / 15, seed 42, three deployments held out (`test_heldout`) | by file |
| Targets | soft targets from the vote margin | hard |
| Training | DINOv2-L, 25 epochs, best by val macro-F1, seed 42 | DINOv2-L |
| Calibration | per-verdict isotonic map, reweighted to the deployment prior | map from the 55:45 test split |

Split sizes (train / val / test / unseen): CurbRamp 6,925 / 1,487 / 1,479 / 546; NoCurbRamp 12,199 / 2,607 /
2,613 / 743; Obstacle 10,554 / 2,262 / 2,266 / 1,812; SurfaceProblem 10,500 / 2,250 / 2,255 / 847; Crosswalk
2,089 / 448 / 445 / 138. The three unseen deployments are tucson-az, private city A (outside the US) and
private city B (US). None of the three was among the production models' 19 training cities either.

## Results: training deployments and unseen deployments

AUC is of the `incorrect` score. "Within-city AUC" is the per-deployment AUC averaged with weights
n_incorrect × n_correct over deployments with both classes and at least 10 labels; it cannot be raised by a
score that only separates deployments with many incorrect labels from deployments with few.

| Type | Split | n | correct / incorrect | panoramas | deployments | AUC [95% CI] | within-city AUC | accuracy at 0.5 | test − unseen [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | training cities (`test`) | 1479 | 812 / 667 | 1319 | 36 | 0.968 [0.958, 0.976] | 0.973 | 0.915 | +0.060 [0.034, 0.091] |
|  | unseen cities (`test_heldout`) | 546 | 318 / 228 | 469 | 3 | 0.907 [0.877, 0.933] | 0.838 | 0.826 |  |
| NoCurbRamp | training cities (`test`) | 2613 | 1443 / 1170 | 1974 | 34 | 0.934 [0.924, 0.943] | 0.903 | 0.879 | +0.052 [0.023, 0.082] |
|  | unseen cities (`test_heldout`) | 743 | 359 / 384 | 542 | 3 | 0.881 [0.854, 0.908] | 0.861 | 0.774 |  |
| Obstacle | training cities (`test`) | 2266 | 1305 / 961 | 1963 | 34 | 0.883 [0.869, 0.896] | 0.870 | 0.814 | +0.015 [-0.010, 0.040] |
|  | unseen cities (`test_heldout`) | 1812 | 587 / 1225 | 1261 | 3 | 0.868 [0.847, 0.888] | 0.788 | 0.724 |  |
| SurfaceProblem | training cities (`test`) | 2255 | 1256 / 999 | 1937 | 37 | 0.896 [0.881, 0.910] | 0.858 | 0.832 | +0.185 [0.147, 0.225] |
|  | unseen cities (`test_heldout`) | 847 | 371 / 476 | 757 | 3 | 0.710 [0.675, 0.745] | 0.682 | 0.658 |  |
| Crosswalk | training cities (`test`) | 445 | 247 / 198 | 374 | 27 | 0.980 [0.961, 0.993] | 0.983 | 0.957 | +0.039 [-0.004, 0.085] |
|  | unseen cities (`test_heldout`) | 138 | 65 / 73 | 130 | 3 | 0.941 [0.897, 0.980] | 0.946 | 0.906 |  |

![Pooled AUC per type with panorama-bootstrap intervals, and within-city AUC](retrain-2026-10/auc_by_type.png)

Three things stand out. The test-minus-unseen drop is real for every type but Obstacle and Crosswalk, whose
intervals include zero. SurfaceProblem's drop (0.19) is of a different size from the others (0.04 to 0.06).
And the within-city AUC on the unseen deployments is below the pooled figure for CurbRamp (0.84 against 0.91),
Obstacle (0.79 against 0.87) and SurfaceProblem (0.68 against 0.71), which is the confound discussed below.

### Per deployment, training deployments (test share, at least 50 labels)

**CurbRamp**

| Deployment | n | correct / incorrect | panoramas | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| seattle-wa | 367 | 141 / 226 | 346 | 0.991 [0.980, 0.998] | 0.962 |
| other (private deployments) | 340 | 201 / 139 | 271 | 0.966 [0.947, 0.982] | 0.900 |
| chicago-il | 220 | 131 / 89 | 207 | 0.960 [0.934, 0.982] | 0.905 |
| spgg | 105 | 22 / 83 | 100 | 0.906 [0.812, 0.979] | 0.886 |
| amsterdam | 68 | 35 / 33 | 48 | 0.943 [0.887, 0.986] | 0.897 |

**NoCurbRamp**

| Deployment | n | correct / incorrect | panoramas | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| seattle-wa | 964 | 740 / 224 | 734 | 0.914 [0.888, 0.939] | 0.897 |
| chicago-il | 501 | 75 / 426 | 381 | 0.859 [0.797, 0.911] | 0.898 |
| other (private deployments) | 302 | 177 / 125 | 250 | 0.942 [0.914, 0.967] | 0.874 |
| mendota-il | 173 | 17 / 156 | 82 | 0.880 [0.783, 0.960] | 0.913 |
| spgg | 111 | 104 / 7 | 96 | 0.499 [0.157, 0.819] | 0.865 |
| amsterdam | 74 | 11 / 63 | 57 | 0.791 [0.661, 0.908] | 0.824 |
| columbus-oh | 69 | 44 / 25 | 53 | 0.845 [0.730, 0.934] | 0.739 |
| teaneck-nj | 68 | 38 / 30 | 44 | 0.968 [0.909, 1.000] | 0.941 |
| cdmx | 62 | 48 / 14 | 49 | 0.749 [0.551, 0.923] | 0.839 |
| newberg-or | 60 | 35 / 25 | 43 | 0.968 [0.919, 0.997] | 0.900 |
| pittsburgh-pa | 53 | 36 / 17 | 48 | 0.719 [0.562, 0.853] | 0.698 |

**Obstacle**

| Deployment | n | correct / incorrect | panoramas | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| seattle-wa | 682 | 368 / 314 | 613 | 0.855 [0.825, 0.883] | 0.793 |
| other (private deployments) | 631 | 429 / 202 | 517 | 0.895 [0.867, 0.920] | 0.835 |
| chicago-il | 437 | 240 / 197 | 377 | 0.905 [0.874, 0.933] | 0.815 |
| amsterdam | 68 | 21 / 47 | 52 | 0.862 [0.741, 0.953] | 0.779 |
| st-louis-mo | 53 | 41 / 12 | 51 | 0.926 [0.826, 0.993] | 0.925 |
| spgg | 50 | 32 / 18 | 48 | 0.832 [0.694, 0.939] | 0.760 |

**SurfaceProblem**

| Deployment | n | correct / incorrect | panoramas | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| other (private deployments) | 584 | 220 / 364 | 406 | 0.926 [0.898, 0.949] | 0.872 |
| seattle-wa | 558 | 247 / 311 | 520 | 0.830 [0.794, 0.865] | 0.771 |
| chicago-il | 314 | 190 / 124 | 291 | 0.882 [0.840, 0.922] | 0.831 |
| amsterdam | 91 | 12 / 79 | 78 | 0.854 [0.730, 0.942] | 0.901 |
| oradell-nj | 82 | 79 / 3 | 59 | 0.869 [0.688, 1.000] | 0.915 |
| columbus-oh | 79 | 64 / 15 | 75 | 0.727 [0.556, 0.865] | 0.747 |
| newberg-or | 71 | 50 / 21 | 64 | 0.855 [0.728, 0.956] | 0.803 |
| pittsburgh-pa | 70 | 61 / 9 | 66 | 0.863 [0.735, 0.964] | 0.886 |
| spgg | 67 | 53 / 14 | 65 | 0.814 [0.669, 0.955] | 0.821 |
| st-louis-mo | 65 | 64 / 1 | 63 | 0.859 [0.769, 0.937] | 0.938 |
| cdmx | 51 | 50 / 1 | 44 | 0.360 [0.239, 0.481] | 0.922 |

**Crosswalk**

| Deployment | n | correct / incorrect | panoramas | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| other (private deployments) | 171 | 153 / 18 | 150 | 0.882 [0.766, 0.980] | 0.947 |
| seattle-wa | 59 | 25 / 34 | 52 | 1.000 [1.000, 1.000] | 0.983 |
| chicago-il | 57 | 20 / 37 | 47 | 0.946 [0.881, 0.991] | 0.860 |

The private training deployments pooled above are four deployments for most types, five for Crosswalk. Within
them the model is at or above the public deployments' level.

### Per deployment, unseen deployments

| Type | Deployment | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|
| CurbRamp | private city A (outside the US) | 284 | 116 / 168 | 0.821 [0.761, 0.872] | 0.729 |
| CurbRamp | private city B (US) | 202 | 172 / 30 | 0.881 [0.791, 0.951] | 0.921 |
| CurbRamp | tucson-az | 60 | 30 / 30 | 0.958 [0.888, 1.000] | 0.967 |
| NoCurbRamp | private city A (outside the US) | 125 | 4 / 121 | 0.800 [0.681, 0.903] | 0.704 |
| NoCurbRamp | private city B (US) | 572 | 338 / 234 | 0.860 [0.824, 0.893] | 0.781 |
| NoCurbRamp | tucson-az | 46 | 17 / 29 | 0.955 — | 0.870 |
| Obstacle | private city A (outside the US) | 993 | 83 / 910 | 0.666 [0.596, 0.727] | 0.655 |
| Obstacle | private city B (US) | 628 | 432 / 196 | 0.896 [0.866, 0.924] | 0.844 |
| Obstacle | tucson-az | 191 | 72 / 119 | 0.792 [0.724, 0.853] | 0.691 |
| SurfaceProblem | private city A (outside the US) | 356 | 142 / 214 | 0.612 [0.552, 0.669] | 0.618 |
| SurfaceProblem | private city B (US) | 262 | 185 / 77 | 0.798 [0.736, 0.859] | 0.767 |
| SurfaceProblem | tucson-az | 229 | 44 / 185 | 0.743 [0.668, 0.811] | 0.594 |
| Crosswalk | private city A (outside the US) | 45 | 29 / 16 | 0.888 — | 0.889 |
| Crosswalk | private city B (US) | 77 | 30 / 47 | 0.972 [0.929, 1.000] | 0.935 |
| Crosswalk | tucson-az | 16 | 6 / 10 | 0.767 — | 0.812 |

![AUC per unseen deployment and type](retrain-2026-10/heldout_per_city.png)

Private city A is the weakest deployment for every type but Crosswalk, and for Obstacle and SurfaceProblem it
is far below the rest (0.67 and 0.61). Private city B, a US deployment, is at the training deployments' level
for Obstacle and Crosswalk and 0.05 to 0.10 below it otherwise. tucson-az is small for three types; its
CurbRamp and NoCurbRamp are at the training level, its Obstacle and SurfaceProblem are not. With one unseen
deployment outside the US, "outside the US" and "this one deployment" cannot be told apart; that is
[#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)'s point.

### The hard slice

`split_data.py --hard-examples` tags the labels the production AI got wrong against a unanimous crowd or an
expert account ([`analysis/hard_examples_2026-10-17.csv`](../analysis/hard_examples_2026-10-17.csv)). Few of them
landed in the test splits, and round 3 does no better on them than chance:

| Type | n | correct / incorrect | AUC | accuracy at 0.5 |
|---|---:|---:|---:|---:|
| CurbRamp | 23 | 22 / 1 | 0.091 | 0.478 |
| NoCurbRamp | 90 | 48 / 42 | 0.644 | 0.633 |
| Obstacle | 11 | 9 / 2 | 0.556 | 0.455 |
| SurfaceProblem | 40 | 11 / 29 | 0.323 | 0.475 |
| Crosswalk | 5 | 4 / 1 | 0.750 | 0.600 |

These are labels whose human majority may itself be contested (the production AI disagreed with it), so a low
score here is not only a model failure. The slice is too small to say more.

## The city–class confound

`data_generation.py` keeps every incorrect label and downsamples correct labels globally to 55%, so each
deployment's incorrect share in training follows its own Disagree rate. A model can learn "this streetscape
means incorrect" and score well on a pooled AUC without judging the label. The unseen deployments are the
extreme case: private city A is 92% incorrect for Obstacle and 97% for NoCurbRamp, private city B 15% for
CurbRamp.

| Type | Split | Deployment | n | share of split | incorrect share |
|---|---:|---:|---:|---:|---:|
| CurbRamp | train | seattle-wa | 1722 | 25% | 63% |
| CurbRamp | train | other (private deployments) | 1559 | 23% | 41% |
| CurbRamp | train | chicago-il | 1095 | 16% | 36% |
| CurbRamp | train | spgg | 545 | 8% | 82% |
| CurbRamp | train | teaneck-nj | 258 | 4% | 12% |
| CurbRamp | unseen | private city A (outside the US) | 284 | 52% | 59% |
| CurbRamp | unseen | private city B (US) | 202 | 37% | 15% |
| CurbRamp | unseen | tucson-az | 60 | 11% | 50% |
| NoCurbRamp | train | seattle-wa | 4228 | 35% | 24% |
| NoCurbRamp | train | chicago-il | 2248 | 18% | 87% |
| NoCurbRamp | train | other (private deployments) | 1764 | 14% | 50% |
| NoCurbRamp | train | mendota-il | 500 | 4% | 81% |
| NoCurbRamp | train | newberg-or | 464 | 4% | 24% |
| NoCurbRamp | unseen | private city A (outside the US) | 125 | 17% | 97% |
| NoCurbRamp | unseen | private city B (US) | 572 | 77% | 41% |
| NoCurbRamp | unseen | tucson-az | 46 | 6% | 63% |
| Obstacle | train | other (private deployments) | 3398 | 32% | 37% |
| Obstacle | train | seattle-wa | 2990 | 28% | 44% |
| Obstacle | train | chicago-il | 1839 | 17% | 43% |
| Obstacle | train | amsterdam | 319 | 3% | 57% |
| Obstacle | train | st-louis-mo | 252 | 2% | 18% |
| Obstacle | unseen | private city A (outside the US) | 993 | 55% | 92% |
| Obstacle | unseen | private city B (US) | 628 | 35% | 31% |
| Obstacle | unseen | tucson-az | 191 | 11% | 62% |
| SurfaceProblem | train | seattle-wa | 2691 | 26% | 58% |
| SurfaceProblem | train | other (private deployments) | 2542 | 24% | 61% |
| SurfaceProblem | train | chicago-il | 1361 | 13% | 45% |
| SurfaceProblem | train | oradell-nj | 543 | 5% | 7% |
| SurfaceProblem | train | amsterdam | 438 | 4% | 85% |
| SurfaceProblem | unseen | private city A (outside the US) | 356 | 42% | 60% |
| SurfaceProblem | unseen | private city B (US) | 262 | 31% | 29% |
| SurfaceProblem | unseen | tucson-az | 229 | 27% | 81% |
| Crosswalk | train | other (private deployments) | 852 | 41% | 10% |
| Crosswalk | train | seattle-wa | 274 | 13% | 69% |
| Crosswalk | train | chicago-il | 222 | 11% | 60% |
| Crosswalk | train | mendota-il | 171 | 8% | 90% |
| Crosswalk | train | teaneck-nj | 142 | 7% | 61% |
| Crosswalk | unseen | private city A (outside the US) | 45 | 33% | 36% |
| Crosswalk | unseen | private city B (US) | 77 | 56% | 61% |
| Crosswalk | unseen | tucson-az | 16 | 12% | 62% |

This is why the tables above carry a within-city AUC next to the pooled one, and why the pooled AUC on the
unseen split is not the number to quote. The decision to run round 3 as is, rather than re-cut the correct
labels per deployment, was taken on 2026-10-21; per-deployment stratification is the first thing to try if the
within-city gap persists in the next round.

## Against the production models

[`vlm-baseline-2026-10.md`](vlm-baseline-2026-10.md) scored the five production validators (the 2025-08-16
revisions `sidewalk-ai-api` pins) on a 1,988-crop sample of the round-3 `test` and `test_heldout` splits,
balanced by class, built from the round-3 crops. Round 3's predictions on the same crops give a paired
comparison. Two intervals are shown: resampling labels within class, and resampling whole panoramas.

| Type | Cities | n | round 3 AUC | production AUC | Δ [label CI] | Δ [panorama CI] | within-city, round 3 | within-city, production | cities |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | training | 250 | 0.948 | 0.945 | +0.003 [-0.025, 0.031] | [-0.023, 0.030] | 0.968 | 0.950 | 8 |
|  | unseen | 150 | 0.919 | 0.890 | +0.029 [-0.003, 0.064] | [-0.003, 0.062] | 0.857 | 0.834 | 3 |
| NoCurbRamp | training | 250 | 0.903 | 0.899 | +0.005 [-0.027, 0.037] | [-0.028, 0.039] | 0.897 | 0.919 | 6 |
|  | unseen | 150 | 0.936 | 0.936 | -0.000 [-0.039, 0.037] | [-0.040, 0.037] | 0.917 | 0.933 | 2 |
| Obstacle | training | 250 | 0.895 | 0.821 | +0.074 [0.024, 0.124] | [0.028, 0.126] | 0.883 | 0.820 | 4 |
|  | unseen | 150 | 0.856 | 0.838 | +0.018 [-0.038, 0.077] | [-0.039, 0.073] | 0.705 | 0.718 | 3 |
| SurfaceProblem | training | 250 | 0.913 | 0.899 | +0.014 [-0.028, 0.057] | [-0.027, 0.055] | 0.889 | 0.845 | 6 |
|  | unseen | 150 | 0.639 | 0.659 | -0.020 [-0.081, 0.049] | [-0.083, 0.044] | 0.601 | 0.675 | 3 |
| Crosswalk | training | 250 | 0.980 | 0.979 | +0.000 [-0.013, 0.014] | [-0.013, 0.015] | 0.985 | 0.985 | 7 |
|  | unseen | 138 | 0.941 | 0.919 | +0.021 [-0.024, 0.067] | [-0.020, 0.065] | 0.946 | 0.942 | 3 |
| all (pooled) | training | 1250 | 0.926 | 0.912 | +0.014 [-0.002, 0.029] | [-0.002, 0.030] | 0.921 | 0.917 | 17 |
|  | unseen | 738 | 0.847 | 0.844 | +0.003 [-0.019, 0.025] | [-0.020, 0.027] | 0.844 | 0.844 | 3 |

Round 3 is ahead on Obstacle in the training deployments by 0.07, with both intervals excluding zero, and level
or slightly ahead everywhere else except SurfaceProblem in the unseen deployments (−0.02, interval includes
zero). Two caveats cut in opposite directions. **Production was trained in 2025-08 on an unrecorded label set**
drawn from the 19 cities that are most of round 3's training deployments, so many of the training-deployment
crops are in its training data; that flatters production on the `test` rows. On the unseen rows the comparison
is clean for both models, since neither trained on those deployments. And the crops are round 3's, tilt-corrected
and label-centred; production was trained on pitch-0, uncorrected crops, so it is being scored slightly off its
own distribution ([`serving-parity-2026-10.md`](serving-parity-2026-10.md) measures how much that matters for
serving). The sample's balance also puts the pooled held-out AUC of production here (0.84) above its figure on
the full held-out split in the VLM note.

## The fixed-crop ablation

[#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51) trained a second CurbRamp model on
fixed 238 px crops (the median depth-sized half-width) on the same split, to ask whether sizing crops by depth
helps. On identical rows, with a paired panorama bootstrap: depth-sized is ahead by +0.014 AUC
[+0.007, +0.021] in the training deployments and +0.009 [−0.006, +0.025] in the unseen ones, where one of the
three deployments goes the other way. Depth sizing stays; the depth-source arms of
[#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21) are not run.

## Calibration at the 0.92 gate

`calibration/calibrate.py` fits one isotonic curve per verdict on the `val` split, reweighted to the
deployment prior (`deployment_priors.json`, the class mix of the human votes in the 54 deployments'
`rawLabels` on 2026-10-20), and writes the map in `sidewalk-ai-api`'s format. The gate is SidewalkWebpage's
`ai-validation-min-accuracy`, 0.92 in every deployment. Coverage and precision below are prior-weighted, as the
map is; `n` is the unweighted count of gated verdicts. The shipped maps cap every Disagree value at 0.9199, so
no Disagree verdict votes; the "uncapped" columns show what the fitted Disagree curves would do without the
cap.

| Type | file prior, incorrect share | Agree clears at confidence | Agree, training cities: coverage / precision (n) | Agree, unseen cities: coverage / precision (n) | Disagree map maximum (uncapped) | Disagree, training cities (uncapped) | Disagree, unseen cities (uncapped) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | 11.8% | 0.99992 | 0.88 / 0.994 (692) | 0.75 / 0.991 (198) | 0.802 | 0 / — (0) | 0 / — (0) |
| NoCurbRamp | 37.4% | 0.9999 | 0.44 / 0.973 (627) | 0.37 / 0.933 (158) | 0.940 | 0.21 / 0.978 (249) | 0.06 / 1.000 (17) |
| Obstacle | 35.8% | 1.0 | 0.48 / 0.959 (631) | 0.49 / 0.970 (325) | 0.892 | 0 / — (0) | 0 / — (0) |
| SurfaceProblem | 32.0% | 0.999997 | 0.68 / 0.957 (819) | 0.53 / 0.890 (178) | 0.903 | 0 / — (0) | 0 / — (0) |
| Crosswalk | 6.1% | 0.9925 | 0.99 / 0.995 (255) | 0.86 / 0.996 (56) | 0.977 | 0.42 / 1.000 (98) | 0.32 / 0.643 (32) |

For comparison, the Phase 0 maps live in production since
[sidewalk-ai-api#16](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/16) also hold every Disagree back,
and let Agree vote at confidence 0.855 (CurbRamp), 0.8825 (Crosswalk), 0.999 (NoCurbRamp), 0.999996 (Obstacle)
and 0.999987 (SurfaceProblem). The thresholds are not comparable across models (round 3's soft targets make its
confidences less saturated); coverage is. On production outcomes the Phase 0 Agree coverage is 0.95 CurbRamp,
0.97 Crosswalk, 0.43 NoCurbRamp, 0.08 Obstacle and 0.29 SurfaceProblem
([`calibration-2026-10.md`](calibration-2026-10.md)); round 3's Agree coverage on its test split is 0.88, 0.99,
0.44, 0.48 and 0.68. The populations differ (production traffic against a 55:45 test split), so read this as
direction only: **Obstacle and SurfaceProblem Agree would vote far more often, CurbRamp Agree a little less.**

### The prior is under review

A review of the serving stack found that `deployment_priors.json`'s incorrect shares (6% Crosswalk, 12%
CurbRamp, 37% NoCurbRamp, 36% Obstacle, 32% SurfaceProblem) are well above an estimate from production
outcomes in the 2026-10-17 pull (about 4 / 7 / 20 / 22 / 15%). The two measure different things (the class mix
of all human-voted labels against the mix of labels the AI assessed and humans then voted on), and which one
serving should use is being recomputed; this note does not settle it. The dependence is not small, so the
same maps were refitted under the production-side estimate (scaled to counts; approximate):

| Type | approximate production prior, incorrect share | Agree, training cities: coverage / precision (n) | Agree, unseen cities | Disagree, training cities (uncapped) | Disagree, unseen cities (uncapped) |
|---|---:|---:|---:|---:|---:|
| CurbRamp | 7% | 0.89 / 0.997 (692) | 0.75 / 0.995 (198) | 0 / — (0) | 0 / — (0) |
| NoCurbRamp | 20% | 0.71 / 0.985 (984) | 0.63 / 0.957 (260) | 0 / — (0) | 0 / — (0) |
| Obstacle | 22% | 0.50 / 0.979 (631) | 0.53 / 0.984 (325) | 0 / — (0) | 0 / — (0) |
| SurfaceProblem | 15% | 0.70 / 0.983 (819) | 0.56 / 0.956 (178) | 0 / — (0) | 0 / — (0) |
| Crosswalk | 4% | 0.99 / 0.997 (255) | 0.86 / 0.997 (56) | 0.38 / 1.000 (98) | 0.29 / 0.535 (32) |

A lower incorrect share raises Agree precision and coverage (NoCurbRamp Agree coverage 0.44 to 0.71) and lowers
Disagree precision at every confidence: NoCurbRamp Disagree no longer clears the gate at all, and Crosswalk
Disagree on the unseen deployments is 0.54 precise on 32 votes. **Every Disagree map should stay held back until
the prior is settled**, whichever way it goes: under the file prior only NoCurbRamp and Crosswalk Disagree would
clear, on 17 and 32 unseen-deployment votes, and Crosswalk's at 0.64 precision.

### Agree votes per deployment

The same maps applied per deployment (file prior; training deployments with at least 50 Agree verdicts, the
private ones pooled; precision both prior-weighted, as the gate promises it, and raw):

| Type | Cities | Deployment | Agree verdicts | gated | coverage (weighted) | precision (weighted) | precision (raw) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | training | other (private deployments) | 195 | 168 | 0.90 | 0.996 | 0.976 |
| CurbRamp | training | seattle-wa | 149 | 135 | 0.94 | 0.995 | 0.970 |
| CurbRamp | training | chicago-il | 136 | 120 | 0.91 | 0.990 | 0.942 |
| CurbRamp | unseen | private city B (US) | 176 | 157 | 0.90 | 0.991 | 0.955 |
| CurbRamp | unseen | private city A (outside the US) | 71 | 17 | 0.27 | 0.976 | 0.882 |
| CurbRamp | unseen | tucson-az | 28 | 24 | 0.86 | 1.000 | 1.000 |
| NoCurbRamp | training | seattle-wa | 761 | 294 | 0.39 | 0.980 | 0.973 |
| NoCurbRamp | training | other (private deployments) | 193 | 123 | 0.65 | 0.970 | 0.959 |
| NoCurbRamp | training | spgg | 99 | 50 | 0.50 | 0.940 | 0.920 |
| NoCurbRamp | training | chicago-il | 66 | 14 | 0.23 | 0.946 | 0.929 |
| NoCurbRamp | training | cdmx | 54 | 32 | 0.60 | 0.929 | 0.906 |
| NoCurbRamp | unseen | private city B (US) | 407 | 150 | 0.39 | 0.942 | 0.900 |
| NoCurbRamp | unseen | private city A (outside the US) | 39 | 3 | 0.07 | 0.000 | 0.000 |
| NoCurbRamp | unseen | tucson-az | 17 | 5 | 0.32 | 1.000 | 1.000 |
| Obstacle | training | other (private deployments) | 437 | 204 | 0.48 | 0.978 | 0.971 |
| Obstacle | training | seattle-wa | 387 | 189 | 0.50 | 0.935 | 0.915 |
| Obstacle | training | chicago-il | 259 | 125 | 0.50 | 0.982 | 0.976 |
| Obstacle | unseen | private city B (US) | 460 | 269 | 0.63 | 0.989 | 0.959 |
| Obstacle | unseen | private city A (outside the US) | 350 | 21 | 0.10 | 0.738 | 0.429 |
| Obstacle | unseen | tucson-az | 109 | 35 | 0.36 | 0.891 | 0.686 |
| SurfaceProblem | training | seattle-wa | 247 | 128 | 0.54 | 0.907 | 0.852 |
| SurfaceProblem | training | other (private deployments) | 235 | 137 | 0.61 | 0.942 | 0.905 |
| SurfaceProblem | training | chicago-il | 187 | 133 | 0.73 | 0.954 | 0.925 |
| SurfaceProblem | training | oradell-nj | 76 | 62 | 0.82 | 0.990 | 0.984 |
| SurfaceProblem | training | st-louis-mo | 62 | 58 | 0.94 | 0.990 | 0.983 |
| SurfaceProblem | training | columbus-oh | 60 | 32 | 0.55 | 0.962 | 0.938 |
| SurfaceProblem | training | spgg | 59 | 42 | 0.74 | 0.957 | 0.929 |
| SurfaceProblem | training | pittsburgh-pa | 57 | 45 | 0.79 | 0.973 | 0.956 |
| SurfaceProblem | training | newberg-or | 52 | 36 | 0.72 | 0.966 | 0.944 |
| SurfaceProblem | unseen | private city B (US) | 190 | 119 | 0.67 | 0.971 | 0.924 |
| SurfaceProblem | unseen | tucson-az | 123 | 54 | 0.47 | 0.616 | 0.370 |
| SurfaceProblem | unseen | private city A (outside the US) | 64 | 5 | 0.08 | 0.804 | 0.600 |
| Crosswalk | training | other (private deployments) | 160 | 160 | 1.00 | 0.996 | 0.950 |
| Crosswalk | unseen | private city B (US) | 33 | 27 | 0.89 | 0.998 | 0.963 |
| Crosswalk | unseen | private city A (outside the US) | 32 | 26 | 0.82 | 0.992 | 0.885 |
| Crosswalk | unseen | tucson-az | 3 | 3 | 1.00 | 1.000 | 1.000 |

In the training deployments every stratum is at or above 0.92 weighted except seattle-wa SurfaceProblem
(0.907); cdmx NoCurbRamp (0.929) and spgg NoCurbRamp (0.940) are the closest to the line. In the unseen deployments, private city B is at or
above 0.94 for every type; tucson-az is below the gate for Obstacle (0.89) and far below for SurfaceProblem
(0.62); private city A clears for CurbRamp and Crosswalk, on 17 and 26 votes, and fails for the rest, with
NoCurbRamp's three gated Agree votes all wrong. The 0.92 promise holds in the deployments the model trained on
and does not transfer by itself.

## Ship recommendation

Per type, for the retrained model served with the maps above. "Ship Agree" means the capped map as fitted
(Disagree held back). "Shadow" means the API scores and `label_ai_assessment` records the raw verdict, but the
map caps both classes so nothing votes, as [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)
proposes for new deployments. Every row assumes the prior question is open; if the lower prior wins, the Agree
numbers improve and nothing below changes direction.

| Type | Training deployments | Unseen deployments and new deployments | Disagree | Why |
|---|---|---|---|---|
| CurbRamp | **Ship Agree** | Shadow private city A; ship Agree in private city B and tucson-az | Hold back | Agree 0.994 at 0.88 coverage; unseen 0.991 at 0.75. City A: 0.82 AUC, 17 gated votes. The Disagree curve tops out at 0.80. |
| NoCurbRamp | **Ship Agree** | Shadow private city A; ship Agree in private city B (0.94 weighted, 0.90 raw) | Hold back | Agree 0.973 at 0.44; unseen 0.933. City A's three gated Agree votes are all wrong. Disagree would clear under the file prior only, on 17 unseen votes. |
| Obstacle | **Ship Agree** | Shadow private city A and tucson-az; ship Agree in private city B (0.989) | Hold back | Agree 0.959 at 0.48, where Phase 0 lets 8% vote; unseen pooled 0.970 but 0.74 and 0.89 in two of three deployments. Disagree never clears. |
| SurfaceProblem | **Shadow** (both verdicts held back) | Shadow | Hold back | Unseen AUC 0.71, Agree precision 0.89 pooled and 0.62 in tucson-az; seattle-wa Agree is 0.907 even in training. Serving the model in shadow mode keeps the stack uniform (one tilt path for all types) and accumulates per-deployment outcomes for the check in #52. |
| Crosswalk | **Ship Agree** | Ship Agree (0.99 in all three; tucson-az has 3 votes) | Hold back | Agree 0.995 at 0.99 coverage; unseen 0.996 at 0.86. Disagree on the unseen deployments is 0.54 to 0.64 precise. |

Two serving conditions go with this. The API must apply the same tilt correction training used
([sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17), with
[sidewalk-ai-api#18](https://github.com/ProjectSidewalk/sidewalk-ai-api/issues/18)'s opencv pin and WebP round
trip), since the models were trained on corrected crops and
[`serving-parity-2026-10.md`](serving-parity-2026-10.md) shows the uncorrected path is a different input. And
the per-deployment gate is a SidewalkWebpage change that does not exist yet; until it does, "shadow in
deployment X" can only be approximated by not enabling AI validation there.

## What this does not establish

- One seed, one retrain. The fixed-crop ablation's two runs differ by 0.006 in best validation macro-F1 for a
  change that is itself small, so differences of that size between settings are within what one seed can show.
- One unseen deployment outside the US. Whether SurfaceProblem's and Obstacle's drop there is about that
  deployment, about its imagery, or about transfer in general needs leave-one-country-out retrains
  ([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52), option 3).
- The production comparison on the training deployments is contaminated in production's favour and off
  distribution for it in the other direction; only the unseen rows are clean.
- The calibration is evaluated on a 55:45 split reweighted to a prior, not on production traffic. The first
  weeks of production outcomes under the new maps are the real test, and
  `calibration/calibrate_from_production.py` plus `calibration/per_city.py` are the tools for it.
- AI-placed labels (`include_ai_labels: true` in `generation_run.json`) are in the training set under the same
  vote margins as human-placed ones; nothing here separates them.
