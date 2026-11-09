# Documentation index

The [README](../README.md) is the overview. Everything else lives here, in three groups. Study pages are dated
because each reports one measurement at one point in time; the pipeline pages are kept current.

## Using the pipeline

| page | what it covers |
|---|---|
| [`running-on-hyak.md`](running-on-hyak.md) | Setup (submodule, environment, depth checkpoint), the three SLURM jobs, resume and cache semantics, outputs, calibration, smoke tests, the held-out-city recipe |
| [`training-data.md`](training-data.md) | Which labels become training data and why: the vote-margin and expert rules, exclusions, cities, soft and reliability-weighted targets, the gold slice, crop geometry and tilt correction |
| [`publishing.md`](publishing.md) | `publish_to_hf.py`: staging, what it refuses, cards, pushing, the October 2026 publish |
| [`../analysis/README.md`](../analysis/README.md) | One line per analysis script |

## Results and studies (October–November 2026)

| page | question |
|---|---|
| [`retrain-2026-10.md`](retrain-2026-10.md) | **The round-3 retrain report**: test and unseen-deployment results per type, the comparison with the 2025-08 models, calibration at the 0.92 gate, the ship recommendation |
| [`vlm-scoreboard.md`](vlm-scoreboard.md) | **The scoreboard**: round 3, the 2025-08 models and every open-weight VLM leg on the round-3 held-out crops, 14 of 17 legs with full coverage (generated, drift-checked, [#63](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/63)) |
| [`vlm-baseline-2026-10.md`](vlm-baseline-2026-10.md) | The first VLM baseline: three MLX models on a Mac, zero-shot, and whether their explanations are worth showing ([#16](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/16)) |
| [`calibration-2026-10.md`](calibration-2026-10.md) | How the 2025-08 models' served accuracy compared with production outcomes, and the production-fitted maps proposed to replace theirs ([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)) |
| [`per-city-calibration-2026-10.md`](per-city-calibration-2026-10.md) | Does the 0.92 gate mean the same thing in every city? ([#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15), [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)) |
| [`deploy-checklist-2026-10.md`](deploy-checklist-2026-10.md) | The pre-deploy red team of the round-3 chain across four repositories: ranked findings, checklist, rollback |
| [`serving-parity-2026-10.md`](serving-parity-2026-10.md) | Do training and sidewalk-ai-api cut the same crop and tensor? Stage by stage ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5)) |
| [`routing-2026-10.md`](routing-2026-10.md) | Human–AI routing by calibrated accuracy: what each threshold costs and buys ([#18](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/18)) |
| [`explanations-2026-10.md`](explanations-2026-10.md) | Saliency overlays and a masked rating design; the rating has not been done ([#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19)) |
| [`distillation-2026-10.md`](distillation-2026-10.md) | Cheaper validators without training: ONNX, int8, the existing tiny models ([#20](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/20)) |
| [`road-prior-2026-10.md`](road-prior-2026-10.md) | A road-network prior for CurbRamp (driveway vs curb ramp): first measurement ([#45](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/45)) |
| [`adjudicator-2026-11.md`](adjudicator-2026-11.md) | VLM adjudicator of the AI-Disagree pool against the expert gold set; the reliability-weighted target scored the same way (not yet run; [#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77), [#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12), [#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74)) |
| [`segmentation-prior-2026-10.md`](segmentation-prior-2026-10.md) | Sidewalk segmentation as a localisation prior for Obstacle and SurfaceProblem ([#44](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/44)) |
| [`ai-disagree-relabel-2026-10.md`](ai-disagree-relabel-2026-10.md) | A masked expert re-label of the deployed AI's Disagree votes (300 items) and the severity × tag breakdown of AI vote agreement ([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)) |
| [`region-transfer-2026-11.md`](region-transfer-2026-11.md) | Does one global validator transfer to a new country, and what helps: local fine-tuning, a region-conditioned model or per-country calibration? Leave-one-country-out on the round-3 splits; not yet run ([#33](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/33), [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52), [#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15)) |
| [`confident-learning-2026-10.md`](confident-learning-2026-10.md) | Can wrong crowd targets be found without rating labor? Out-of-fold confident learning and the stricter Disagree rules as split variants, scored against round 3 on the same held-out rows ([#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76), [#75](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75)) |
| [`ai-disagree-reversal-2026-10.md`](ai-disagree-reversal-2026-10.md) | Runbook for turning the 2025-08 models' Disagree votes into Unsure in production: per-city dry run and the update script ([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)) |
| [`ai-labels-2026-10.md`](ai-labels-2026-10.md) | The round-3 CurbRamp validator on RampNet's AI-placed curb ramps in three public deployments, against the existing human votes: AUC per city, what the deployed gate would do, the round-3 leakage and its effect on the reference ([SidewalkWebpage#5790](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5790)) |
| [`ai-label-holdout-2026-11.md`](ai-label-holdout-2026-11.md) | AI-placed labels in training: the held-out evaluation slice, the with/without round-4 ablation (not yet run) and the full-pool ranked Disagree lists for the leads ([#79](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/79), [#80](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/80)) |
| [`job-window-crops-2026-11.md`](job-window-crops-2026-11.md) | Would a CurbRamp validator trained on the crops SidewalkWebpage's crop job cuts (3:2 windows, `CropSizingRule` v2, no projection, tilt or depth) match the depth-sized one? The port, the per-provider counts (round 3 is GSV only), the paired protocol; not yet run ([#53](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/53)) |

## Results and studies (November 2026)

| page | question |
|---|---|
| [`nosidewalk-2026-11.md`](nosidewalk-2026-11.md) | NoSidewalk in auto-validation: how much training data the public rawLabels hold, the block-face rule that replaces the label rule for this type, the face-size deployment prior, and the pilot (run 2026-11-05: whole-view crops beat depth-sized ones, within-city AUC 0.93 on test and 0.79 on held-out cities; the Agree gate is not evaluable at this size, not served; [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42)) |
| [`retype-2026-11.md`](retype-2026-11.md) | **Not yet run (2026-11).** Validation as re-typing: one 8-class model over label types, the verdict as p(claimed type), a suggested type for wrong labels, the round-3 split inherited by panorama ([#13](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/13)) |
| [`context-arms-2026-11.md`](context-arms-2026-11.md) | Does more context around the crop help? Two-scale crops, the whole view with a point heatmap, and side information, each against round 3 on the same held-out rows ([#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14); not yet run) |

## Decisions, audit and hygiene

| page | what it covers |
|---|---|
| [`decisions.md`](decisions.md) | Deliberate choices about the training data, with the evidence and what would change each |
| [`audit-2026-10-14.md`](audit-2026-10-14.md) | The code audit that opened issues [#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3) to [#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21); the evidence behind each |
| [`repo-hygiene.md`](repo-hygiene.md) | The old vendored Depth-Anything-V2 tree in history: blobless clones, and the opt-in purge |
