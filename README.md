<p align="center">
  <h1 align="center">sidewalk-validator-ai</h1>
  <p align="center">
    Training pipeline for the AI models that validate crowdsourced sidewalk accessibility labels in
    <a href="https://projectsidewalk.org">Project Sidewalk</a>
    <br>
    <a href="https://makeabilitylab.cs.washington.edu">Makeability Lab</a>, University of Washington
  </p>
  <h3 align="center">
    💻 <a href="https://github.com/ProjectSidewalk/sidewalk-validator-ai">Code</a> ·
    🤗 <a href="https://huggingface.co/collections/projectsidewalk/project-sidewalk-validator-models-6797bd158d5e385e464dbe45">Models</a> ·
    🗃️ <a href="https://huggingface.co/projectsidewalk">Datasets</a> ·
    🚀 <a href="https://github.com/ProjectSidewalk/sidewalk-ai-api">Serving API</a> ·
    🌐 <a href="https://github.com/ProjectSidewalk/SidewalkWebpage">Platform</a> ·
    📄 <a href="#lineage-and-related-work">Papers</a> ·
    📚 <a href="docs/README.md">Docs</a>
  </h3>
</p>

**sidewalk-validator-ai** trains the models that let Project Sidewalk check its own crowd. For each of five label
types (curb ramp, missing curb ramp, obstacle, surface problem, crosswalk) it pulls every label and its votes from
the public v3 API, keeps the labels the crowd clearly endorsed or clearly rejected, renders a label-centred,
tilt-corrected, depth-sized crop from the panorama, and fine-tunes a
[DINOv2](https://arxiv.org/abs/2304.07193)-Large classifier to predict **correct** or **incorrect**. A calibration
step maps model confidence to measured precision. [sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api)
serves the models, and [SidewalkWebpage](https://github.com/ProjectSidewalk/SidewalkWebpage) calls it nightly for up
to 800 labels per city with `{label_type, panorama_id, x, y, city}` and the camera pose; a verdict whose estimated
accuracy is at least 0.92 (`ai-validation-min-accuracy`) is cast as a vote by the `SidewalkAI` account, anything
below is recorded as Unsure, and every assessment is stored in `label_ai_assessment`
([`SidewalkWebpage/docs/ai-subsystems.md`](https://github.com/ProjectSidewalk/SidewalkWebpage/blob/develop/docs/ai-subsystems.md)).

**Since 2026-10-27 production runs the October 2026 models ("round 3")**, trained on a panorama-grouped split with
tilt-corrected crops and human votes only, published to Hugging Face on 2026-10-26 and served by sidewalk-ai-api
1.2.0. The models before them (2025-08) are the "production" row in the first table below.

## Results

All five validators and every open-weight vision-language model (VLM) leg with full coverage (14 of 17) are scored on
the same 13,144 held-out crops of the round-3 split (the training deployments' test share plus three deployments
held out of training entirely). AUC is of the `incorrect` score; macro-AUC gives each label type equal weight; the
interval is a 95% label bootstrap; Δ is the paired difference against round 3. The production row is optimistic:
those models were trained in 2025-08 on an unrecorded label set that overlaps the held-out crops. The block is
generated from
[`docs/vlm-scoreboard/board.csv`](docs/vlm-scoreboard/board.csv) and drift-checked in CI; the full board with
seventeen legs, within-city AUC, latency and per-deployment rows is [`docs/vlm-scoreboard.md`](docs/vlm-scoreboard.md).

<!-- BEGIN GENERATED: readme_scoreboard (analysis/vlm_scoreboard.py) -->

| model | class | macro-AUC [95% CI] | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | Δ vs round 3 [95% CI] | crops |
|---|---|---|---|---|---|---|---|---|---|
| **Round 3 (DINOv2-L, 2026-10)** | purpose-trained | 0.915 [0.910, 0.919] | 0.953 | 0.920 | 0.873 | 0.855 | 0.972 | — | 13,144 |
| Production (DINOv2-L, 2025-08) | purpose-trained | 0.899 [0.894, 0.905] | 0.942 | 0.925 | 0.814 | 0.846 | 0.970 | -0.015 [-0.020, -0.011] | 13,144 |
| Gemma 4 31B, 8-shot | chat VLM | 0.775 [0.768, 0.783] | 0.865 | 0.745 | 0.557 | 0.772 | 0.938 | -0.139 [-0.147, -0.132] | 13,144 |
| Gemma 4 31B | chat VLM | 0.756 [0.748, 0.763] | 0.825 | 0.659 | 0.584 | 0.756 | 0.954 | -0.159 [-0.167, -0.151] | 13,144 |
| GLM-5.3-Flash (FP8, vLLM) | chat VLM | 0.754 [0.746, 0.762] | 0.801 | 0.674 | 0.574 | 0.777 | 0.944 | -0.160 [-0.169, -0.153] | 13,144 |

<!-- END GENERATED: readme_scoreboard -->

The pooled AUC hides what a new deployment would see. Round 3 on the training deployments' test share against the
three unseen deployments, with the within-deployment AUC that cannot be raised by telling deployments apart
([`docs/retrain-2026-10.md`](docs/retrain-2026-10.md), one seed, panorama-bootstrap intervals):

| type | test AUC | unseen-deployment AUC | within-city AUC, unseen | n test / unseen |
|---|---|---|---|---|
| CurbRamp | 0.968 [0.958, 0.976] | 0.907 [0.877, 0.933] | 0.838 | 1,479 / 546 |
| NoCurbRamp | 0.934 [0.924, 0.943] | 0.881 [0.854, 0.908] | 0.861 | 2,613 / 743 |
| Obstacle | 0.883 [0.869, 0.896] | 0.868 [0.847, 0.888] | 0.788 | 2,266 / 1,812 |
| SurfaceProblem | 0.896 [0.881, 0.910] | 0.710 [0.675, 0.745] | 0.682 | 2,255 / 847 |
| Crosswalk | 0.980 [0.961, 0.993] | 0.941 [0.897, 0.980] | 0.946 | 445 / 138 |

Every type ranks labels well where it was trained; on unseen deployments CurbRamp, NoCurbRamp and Crosswalk lose
0.04 to 0.06 of AUC, Obstacle 0.02 and SurfaceProblem 0.19. The deployment outside the US is the weakest for every type but
Crosswalk, so the report recommends gating AI votes per deployment before enabling them in a new one
([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)).

**What is served.** Each model ships an isotonic confidence-to-accuracy map fitted on the validation split and
reweighted to the deployment class prior. The shipped maps hold every Disagree verdict below the 0.92 gate by
policy ([#57](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/57)) until the prior is settled
([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)) and production outcomes exist, so the
AI casts Agree votes only, for all five types. That includes SurfaceProblem, whose Agree map went live although the
retrain report recommended holding both of its verdicts back (shadow by map cap is this repo's call; the only
shadow SidewalkWebpage offers today is switching AI validation off per city). The per-deployment check the report
wants before Agree votes in an unseen deployment does not exist in SidewalkWebpage yet
([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)). Where the 2025-08 models stood
against production outcomes, per type and per city, is in [`docs/calibration-2026-10.md`](docs/calibration-2026-10.md)
and [`docs/per-city-calibration-2026-10.md`](docs/per-city-calibration-2026-10.md); the pre-deploy red team of the
round-3 chain is [`docs/deploy-checklist-2026-10.md`](docs/deploy-checklist-2026-10.md). Every other study (VLM
explanations, routing, distillation, road and segmentation priors, the AI Disagree reversal and re-label) is
indexed in [`docs/README.md`](docs/README.md).

## How it works

```
/v3/api/rawLabels (every deployment)   data_generation.py
        │                                     │
        ▼                                     ▼
 [1] select labels      human votes only · net ≥3 agree → correct · net ≥2 disagree → incorrect · expert votes break the rest
        │               (AI votes excluded; pre-downsampling counts → deployment_priors.json; correct downsampled to 55%)
        ▼
 [2] read panorama      pano store <city>/<pano_id[:2]>/<pano_id>.jpg → 8192×4096 equirectangular
        ▼
 [3] perspective view   90° FOV, 2048², centred on the label, rig-tilt corrected (sidewalk-panorama-tools, β = 0.95)
        ▼
 [4] depth-sized crop   Depth Anything V2 (metric, ViT-B) at the centre → crop_half = 6100 / depth_m px → 640 px WebP
        ▼
 <label_type>/{correct,incorrect}/<city>_<label_id>.webp  +  <label_type>/manifest.csv (one row per label)
        │
        ▼
 [5] split              split_data.py      grouped by panorama + labels ≤3 m apart, 70 / 15 / 15 (+ held-out cities)  →  split.csv
        ▼
 [6] fine-tune          fine_tune_classifier.py   facebook/dinov2-large on split.csv, best val macro-F1  →  <label_type>/dinov2/
        ▼
 [7] evaluate           evaluate.py        split.csv → predictions.csv, metrics on test / test_heldout / hard
        ▼
 [7b] calibrate         calibration/calibrate.py   val split reweighted to deployment_priors.json → accuracy map
        ▼
 [8] publish            publish_to_hf.py   cards + inference files + crops  →  Hugging Face
        ▼
        sidewalk-ai-api /process  →  SidewalkWebpage nightly actor (≤ 800 labels/city, gate 0.92)
```

The rules behind steps 1 and 3 to 5 (vote margins, expert accounts, exclusions, soft targets, crop geometry) are in
[`docs/training-data.md`](docs/training-data.md). One rule matters for anyone touching the data: the AI's own votes
sit in the API's `agree_count` / `disagree_count` like anyone else's, so training counts votes from the
`validations` column, by `validator_type`. Counting the aggregates would train the model on its own output.

| file | role |
|---|---|
| [`data_generation.py`](data_generation.py), [`data_generation.slurm`](data_generation.slurm) | Steps 1–5: label selection, deployment priors, panoramas from the pano store, views, depth-sized crops, a resumable `manifest.csv`; the job then runs the split |
| [`split_data.py`](split_data.py) | Step 5: train/val/test by panorama, nearby panoramas merged, optional held-out cities, the `hard` flag |
| [`fine_tune_classifier.py`](fine_tune_classifier.py), [`fine_tune.slurm`](fine_tune.slurm) | Step 6: Hugging Face `Trainer` on `facebook/dinov2-large`; hard, soft or reliability-weighted targets |
| [`evaluate.py`](evaluate.py), [`get_stats.slurm`](get_stats.slurm) | Step 7: predictions and metrics per split, per city and on the hard and gold slices; the job then calibrates |
| [`calibration/`](calibration) | Step 7b: isotonic confidence→accuracy maps in the API's format, from a retrained model's val split or from production outcomes; per-city checks |
| [`publish_to_hf.py`](publish_to_hf.py), [`hf_cards.py`](hf_cards.py) | Step 8: staged, checked uploads with generated cards ([`docs/publishing.md`](docs/publishing.md)) |
| [`hard_examples.py`](hard_examples.py), [`validator_reliability.py`](validator_reliability.py) | Labels the deployed AI got wrong against a unanimous crowd or an expert; per-validator reliability for weighted targets |
| [`export_onnx.py`](export_onnx.py), [`explain.py`](explain.py) | ONNX export, int8 quantisation and benchmarks; saliency maps and the masked rating packet |
| [`analysis/`](analysis) | The scripts behind every study in `docs/` ([`analysis/README.md`](analysis/README.md)); their data stays outside the repo |
| `Depth-Anything-V2/` | Git submodule of upstream [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), pinned to the commit sidewalk-ai-api uses |

## Published models and datasets

Everything lives under the [`projectsidewalk`](https://huggingface.co/projectsidewalk) organisation on Hugging
Face. Each model repo holds the inference files, `confidence_precision_map.json` with its calibration report,
`metrics.json`, `training_run.json` and a generated card. sidewalk-ai-api pins these revisions and serves its own
copy of each map, checked by sha256 against the hub file at startup
([sidewalk-ai-api#25](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/25)).

| label type | model (revision of 2026-10-26) | dataset |
|---|---|---|
| CurbRamp | [`sidewalk-validator-ai-curbramp`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-curbramp) `6791a64` | [2025-08 upload](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-curbramp) ⚠️ [a copy of the Crosswalk set](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7) |
| NoCurbRamp | [`sidewalk-validator-ai-nocurbramp`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-nocurbramp) `ca7167d` | [2025-08 upload](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-nocurbramp) |
| Obstacle | [`sidewalk-validator-ai-obstacle`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-obstacle) `64b082c` | [2025-08 upload](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-obstacle) |
| SurfaceProblem | [`sidewalk-validator-ai-surfaceproblem`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-surfaceproblem) `b5497b7` | [2025-08 upload](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-surfaceproblem) |
| Crosswalk | [`sidewalk-validator-ai-crosswalk`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk) `b8498b8` | [2025-08 upload](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-crosswalk) |

Each model is `Dinov2ForImageClassification` with `id2label = {0: correct, 1: incorrect}` and ships its
`preprocessor_config.json`. The dataset repos are still the 2025-08 uploads: the round-3 crops have not been
published (they include private deployments, which the publisher strips, and the crops live on the cluster), so the
datasets do not match the models until the dataset half of
[#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7) is done.

## Quick start

```bash
git clone --recurse-submodules https://github.com/ProjectSidewalk/sidewalk-validator-ai
conda env create -f environment.yml && conda activate sidewalk-validator-ai
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/e96fd0c8cedf1825a55b9c61309f754982948df0/depth_anything_v2_metric_vkitti_vitb.pth

PANO_STORE=/path/to/Panoramas sbatch data_generation.slurm   # steps 1–5, all five label types
sbatch fine_tune.slurm                                        # step 6
sbatch get_stats.slurm                                        # steps 7 and 7b
python publish_to_hf.py --label_type CurbRamp --out ROOT      # step 8: stage only; --push for a dry run, --push --yes to upload
```

Panoramas are read from the lab's pano store, never downloaded. Job files are written for klone's `gpu-l40s`
partition; the resume rules, outputs, smoke tests and the held-out-city recipe are in
[`docs/running-on-hyak.md`](docs/running-on-hyak.md). The CPU tests run with `pip install -r requirements.txt &&
pytest -q`.

## Status and open issues

The October 2026 audit ([`docs/audit-2026-10-14.md`](docs/audit-2026-10-14.md)) found that the published metrics
rested on a split that leaked across panoramas, a confidence map fitted under the wrong class prior, and a crop
pipeline that differed between training and serving. The round-3 retrain addressed those: the split is by panorama
([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3), open for the re-measurement items),
training and serving share one geometry package and depth checkpoint
([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5), closed), and the maps are reweighted to
the deployment prior ([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4), open until the
prior itself is settled). What is still open:

- **The deployment prior** the maps are reweighted to comes from an asymmetric training rule and is 15–17 points off
  the production estimate for three types; the Disagree hold-back stays until it is settled
  ([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)).
- **Transfer to unseen deployments** is not reliable (SurfaceProblem 0.71 AUC, the non-US deployment weakest), so
  AI votes in a new deployment need a per-deployment check that SidewalkWebpage does not have yet
  ([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52),
  [#33](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/33)).
- **Datasets on Hugging Face** are the 2025-08 uploads, with the CurbRamp set a copy of Crosswalk
  ([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)); cards, DOIs and licences are
  [#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2).
- **Non-GSV imagery** (Mapillary, Panoramax) is excluded from training and serving until per-provider pose
  handling exists ([#9](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/9),
  [`docs/decisions.md`](docs/decisions.md)).
- **AI Disagree votes** cast by the 2025-08 models hide labels on their own in production; the reversal and a
  masked expert re-label are in review ([SidewalkWebpage#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793),
  [#62](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/62),
  [#66](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66)).
- **Experiments** with a first run spelled out: re-typing ([#13](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/13)),
  more context ([#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14)), leave-one-city-out
  runs ([#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15)), temporal validation
  ([#17](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/17)), rating the saliency overlays
  ([#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19)), the depth-source ablation
  ([#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21)), region-specific models
  ([#33](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/33)), frontier-model baselines
  ([#47](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/47)) and merging the tagger pipeline
  into this repo ([#26](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/26)). Earlier ideas are
  in [#1](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/1); the history purge of the old
  vendored tree stays opt-in ([#11](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/11),
  [`docs/repo-hygiene.md`](docs/repo-hygiene.md)).

## Lineage and related work

Automatic validation of crowd labels is the easier half of the lab's Crowd+AI sidewalk work, and this repo is its
third generation:

| year | work | what it established |
|---|---|---|
| 2014 | Hara et al., **Tohme**, UIST'14 ([doi](https://doi.org/10.1145/2642918.2647403)) | Crowd + CV curb-ramp detection with a workflow controller that routes to humans only when the model is likely wrong. |
| 2019 | Saha et al., **Project Sidewalk**, CHI'19 ([doi](https://doi.org/10.1145/3290605.3300292)) | The platform and the label/validation data this repo trains on. |
| 2019 | Weld et al., ASSETS'19 ([doi](https://doi.org/10.1145/3308561.3353798), [code](https://github.com/ProjectSidewalk/sidewalk-cv-assets19)) | First deep-learning auto-validation of Project Sidewalk labels: ResNet on depth-sized crops plus geo features, panorama-level splits, DC → Seattle/Newberg transfer. |
| 2022 | Duan et al., ASSETS'22 ([doi](https://doi.org/10.1145/3517428.3550381), [code](https://github.com/ProjectSidewalk/sidewalk-cv-2021)) | Validated vs unvalidated training data (helps curb ramps, not the rest) and leave-one-city-out transfer; the direct ancestor of this pipeline. |
| 2022 | Hosseini et al., CVPR-W AVA'22 ([arXiv](https://arxiv.org/abs/2206.13677)) | The global-scale Crowd+AI agenda this work sits in. |
| 2024 | Liu, Wu et al., ASSETS'24 ([doi](https://doi.org/10.1145/3663548.3688531), [code](https://github.com/ProjectSidewalk/sidewalk-tagger-ai)) | The sibling **tagger** models (33 condition tags) served by the same API. |
| 2024 | Li et al., **LabelAId**, CHI'24 ([doi](https://doi.org/10.1145/3613904.3642089)) | Just-in-time AI feedback to labelers; the UX side of label quality. |
| 2025 | O'Meara et al., **RampNet**, ICCV'25 workshop ([arXiv](https://arxiv.org/abs/2508.09415), [code](https://github.com/ProjectSidewalk/RampNet)) | The harder half: finding curb ramps in whole panoramas, deployed through [sidewalk-auto-labeler](https://github.com/ProjectSidewalk/sidewalk-auto-labeler). |

Sibling repositories: [sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) (serves these models),
[sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai) (tag models),
[sidewalk-panorama-tools](https://github.com/ProjectSidewalk/sidewalk-panorama-tools) (the pano store and the shared
crop geometry), [SidewalkWebpage](https://github.com/ProjectSidewalk/SidewalkWebpage) (platform, orchestration,
dashboards). The org-wide map of AI repos is
[`SidewalkWebpage/docs/ai-subsystems.md`](https://github.com/ProjectSidewalk/SidewalkWebpage/blob/develop/docs/ai-subsystems.md).

## Citation

There is no paper for this repository yet. If you use the models, datasets or code, please cite Project
Sidewalk and link this repo:

```bibtex
@inproceedings{saha2019projectsidewalk,
  author    = {Saha, Manaswi and Saugstad, Michael and Maddali, Hanuma Teja and Zeng, Aileen and Holland, Ryan and Bower, Steven and Dash, Aditya and Chen, Sage and Li, Anthony and Hara, Kotaro and Froehlich, Jon},
  title     = {Project Sidewalk: A Web-based Crowdsourcing Tool for Collecting Sidewalk Accessibility Data at Scale},
  booktitle = {Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems},
  year      = {2019},
  doi       = {10.1145/3290605.3300292}
}
```

```
Project Sidewalk. sidewalk-validator-ai: training pipeline for AI validation of crowdsourced sidewalk
accessibility labels. https://github.com/ProjectSidewalk/sidewalk-validator-ai
```

## License

The code in this repository is released under the [MIT License](LICENSE). The `Depth-Anything-V2/` submodule
(upstream code) keeps its own Apache-2.0 license (see its `LICENSE`). Licenses for the released datasets and models
are tracked separately in [#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2).
