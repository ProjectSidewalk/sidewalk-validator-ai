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
    📄 <a href="#lineage-and-related-work">Papers</a>
  </h3>
</p>

**sidewalk-validator-ai** trains the models that let Project Sidewalk check its own crowd. For each of
five label types (curb ramp, missing curb ramp, obstacle, surface problem, crosswalk) it pulls every
label and its agree/disagree votes from the public v3 API across deployed cities, keeps the labels the
crowd clearly endorsed or clearly rejected, and turns each into a training image: it reassembles the
Google Street View panorama from tiles, renders a 90° perspective view toward the label, runs
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to estimate how far away the
labelled feature is, and cuts a crop whose size scales with distance so near and far features fill the
frame similarly. A [DINOv2](https://arxiv.org/abs/2304.07193)-Large classifier is then fine-tuned per
label type to predict **correct** or **incorrect**, selected by validation macro-F1. An evaluation script
maps model confidence to measured precision on a held-out split. The models and that confidence map ship
to Hugging Face and are served by [sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api),
which [SidewalkWebpage](https://github.com/ProjectSidewalk/SidewalkWebpage) calls nightly for up to 800
labels per city; verdicts with estimated accuracy ≥ 0.92 are recorded as real validations by the
`SidewalkAI` account.

> [!NOTE]
> **State of the repo (October 2026).** Production runs the five models' **2025-08-16** uploads: each
> Hugging Face model repo also holds an earlier 2025-01-26 upload with different weights, but
> sidewalk-ai-api's `model_map.py` pins the August revisions, and every metric in this README comes from
> those runs. They were trained around 2025-08-14, after AI validation merged
> ([SidewalkWebpage#3978](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/3978), 2025-08-01; earliest `label_ai_assessment` row 2025-08-13) and
> before the nightly `GetAiValidationsActor` sweep was added (2025-09-17,
> [`7b6e9d720b`](https://github.com/ProjectSidewalk/SidewalkWebpage/commit/7b6e9d720b)). So their training data was most likely free of AI votes, but
> AI votes cast between 2025-08-01 and the data pull are not ruled out. The code on
> `master` before [the audit PR](docs/audit-2026-10-14.md) could not run against the current API (a
> renamed column and a tile endpoint that now needs browser headers); it runs again, but several
> findings about the training signal, the split and calibration are open as issues and need a retrain
> to resolve. Read [Known limitations](#known-limitations-and-open-issues) before trusting a number.

## How it works

```
/v3/api/rawLabels (19 cities)          data_generation.py
        │                                     │
        ▼                                     ▼
 [1] select labels      net ≥3 human agree → correct · net ≥2 human disagree → incorrect
        │               (AI votes excluded; correct class downsampled to 55%)
        ▼
 [2] fetch panorama     GSV tiles → 8192×4096 equirectangular
        ▼
 [3] perspective view   90° FOV, 2048², camera yawed to the label, pitch 0
        ▼
 [4] depth-sized crop   Depth Anything V2 (metric, ViT-L) at the label → crop_half = 6100 / depth_m px → 640 px
        ▼
 <label_type>/{correct,incorrect}/<city>_<label_id>.webp
        │
        ▼
 [5] split              split_data.py      70 / 15 / 15 by file  →  <label_type>/ds/{train,val,test}
        ▼
 [6] fine-tune          fine_tune_classifier.py   facebook/dinov2-large, best val macro-F1  →  <label_type>/dinov2/
        ▼
 [7] evaluate           evaluate.py        predictions.csv, metrics on test / test_heldout / hard
        ▼
 [7b] calibrate         calibration/calibrate.py   val split reweighted to the deployment prior → accuracy map
                        (split.csv and deployment_priors.json: data-generation changes in progress, #3/#10)
        ▼
 [8] publish            Hugging Face  →  sidewalk-ai-api /process  →  SidewalkWebpage nightly actor (≤ 800 labels/city, gate 0.92)
```

### Components

| File | Role | Key constants |
|---|---|---|
| [`data_generation.py`](data_generation.py) | Steps 1–4. Pulls labels, selects classes, fetches panoramas, renders perspective views, sizes crops by depth. | `CITIES`, `CORRECT_PROPORTION = 0.55`, `SEED = 42`, FOV 90°, 2048² view, `6100 / depth` |
| [`split_data.py`](split_data.py) | Step 5. Shuffles each class folder into train/val/test image folders. | 70 / 15 / 15, `--seed` |
| [`fine_tune_classifier.py`](fine_tune_classifier.py) | Step 6. Hugging Face `Trainer` on `facebook/dinov2-large`. | lr 1e-5, batch 8 × 4 accumulation, 25 epochs, warm-up 10%, best by macro-F1 |
| [`evaluate.py`](evaluate.py) | Step 7. Predictions for every held-out row, then AUC, accuracy and balanced accuracy, per-class P/R/F1, recall at precision. (`get_stats.py` is a stub that says so.) | splits `val`, `test`, `test_heldout`; `hard` slice |
| [`calibration/`](calibration) | Step 7b. Isotonic confidence→accuracy maps in the API's format: `calibrate.py` for a retrained model (deployment-prior weighting), `calibrate_from_production.py` for the deployed models from production outcomes. | gate 0.92, grid 0.5–1.0 step 0.0025 |
| [`analysis/`](analysis) | Scripts behind the production measurements in [`docs/calibration-2026-10.md`](docs/calibration-2026-10.md). | outputs live outside the repo |
| `*.slurm` | Hyak job files for steps 1–5, 6 and 7 (five label types per job). | `gpu-l40s`, 100 GB, 1 GPU |
| `Depth-Anything-V2/` | Vendored upstream code (unmodified); the checkpoint is downloaded separately. | `depth_anything_v2_metric_vkitti_vitl.pth` |

### Where the models run in production

SidewalkWebpage's `GetAiValidationsActor` selects up to 800 not-yet-assessed GSV labels per city each
night and POSTs `{label_type, panorama_id, x, y, city}` to `sidewalk-ai-api`, which rebuilds the crop,
runs the validator (and the tagger), and returns `validation_result`, `validation_score` and
`validation_estimated_accuracy`. If the estimated accuracy is at least `ai-validation-min-accuracy`
(0.92 in every city) the result is cast as an Agree/Disagree vote by the `SidewalkAI` user; otherwise it
is recorded as Unsure. Every assessment is stored in `label_ai_assessment`. The full description, config
keys and database tables are in
[`SidewalkWebpage/docs/ai-subsystems.md`](https://github.com/ProjectSidewalk/SidewalkWebpage/blob/develop/docs/ai-subsystems.md).

Note that the AI's votes are counted in the API's `agree_count` / `disagree_count` columns like any
other user's. Training must therefore count votes from the `validations` column, where each vote
carries a `validator_type`; `data_generation.py` does this. Using the aggregate columns would train the
model on its own output.

## Training signal

A label is used only when the crowd was decisive, counting **human** votes only:

| class | rule | share of dataset |
|---|---|---|
| `incorrect` | disagree − agree > 1 | 45% |
| `correct` | agree − disagree > 2, then randomly downsampled | 55% |
| (dropped) | unvalidated, unsure, or contested | — |

Labels from non-GSV imagery providers and legacy labels with unconverted coordinates (negative `pano_y`:
Seattle 193, CDMX 14, Pittsburgh 8, SPGG 1) are excluded. The 55:45 ratio is a training-balance choice,
not the real prior: among labels the crowd decided in Teaneck the balance is roughly 90:10, a proxy for
production traffic rather than a measurement of it. That matters for how the confidence map is read; see
[#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4).

> [!WARNING]
> Counting human votes only shrinks the classes, most of all CurbRamp `incorrect`, because the AI's
> disagree votes no longer count. Across the 16 cached training cities (of the 19 in `CITIES`) the `incorrect` sets go from (API
> columns) to (human votes only): CurbRamp 7,957 → 2,626, NoCurbRamp 7,842 → 5,923, Obstacle
> 4,412 → 3,556, SurfaceProblem 4,688 → 3,623, Crosswalk 1,974 → 1,034. With the 55:45 rule the
> `correct` class shrinks with them, so a retrain has much less CurbRamp data than the counts in the
> API columns suggest.

## Crop geometry

The panorama is resized to 8192×4096. A 2048² perspective view with a 90° horizontal field of view is
rendered with the camera yawed to the label's longitude and pitch 0, so the label sits at the image's
horizontal centre and at whatever row its latitude projects to. Depth Anything V2 (metric, VKITTI, ViT-L,
`max_depth = 80`) runs on that view; the depth at the label's pixel sets the crop half-width as
`6100 / depth_m` pixels, clipped to the view, then the crop's long side is resized to 640 px. The serving
path in `sidewalk-ai-api` pitches the camera to the label and uses the ViT-B depth checkpoint; aligning
the two is [#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5).

## Published artifacts

Everything lives under the [`projectsidewalk`](https://huggingface.co/projectsidewalk) organisation on
Hugging Face. Metrics are the best validation macro-F1 recorded in each model's `trainer_state.json`
(file-level split, see limitations).

| label type | model | val macro-F1 | dataset (train / val / test) |
|---|---|---|---|
| Crosswalk | [`sidewalk-validator-ai-crosswalk`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk) | 0.949 | [1,108 / 237 / 240](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-crosswalk) |
| CurbRamp | [`sidewalk-validator-ai-curbramp`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-curbramp) | 0.860 | ⚠️ [the published dataset is a copy of the crosswalk one](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7) |
| NoCurbRamp | [`sidewalk-validator-ai-nocurbramp`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-nocurbramp) | 0.847 | [6,404 / 1,372 / 1,374](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-nocurbramp) |
| Obstacle | [`sidewalk-validator-ai-obstacle`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-obstacle) | 0.797 | [3,576 / 765 / 769](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-obstacle) |
| SurfaceProblem | [`sidewalk-validator-ai-surfaceproblem`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-surfaceproblem) | 0.808 | [4,766 / 1,021 / 1,024](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-surfaceproblem) |

Each model is `Dinov2ForImageClassification` with `id2label = {0: correct, 1: incorrect}` and ships its
`preprocessor_config.json` (resize 256, centre-crop 224, ImageNet normalisation). The per-model
`confidence_precision_map.json` is copied into `sidewalk-ai-api/accuracy_mappings/`.

## Setup

Create the conda environment (Linux with an NVIDIA GPU; the file is a frozen export of the Hyak
environment used for the published models):

```bash
conda env create -f environment.yml
conda activate sidewalk-validator-ai
```

Download the depth checkpoint into the repo root:

```bash
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
```

> [!IMPORTANT]
> Activate the conda environment before submitting any SLURM job or running any script; the job files
> do not activate it for you.

## Running on Hyak

The three job files each process all five label types. They are written for the `makelab` account on
klone's `gpu-l40s` partition; edit the `#SBATCH` lines for anything else.

**1. Generate datasets** (hours to days; fetches every panorama from Google):

```bash
sbatch data_generation.slurm
```

Creates `Crosswalk/`, `CurbRamp/`, `NoCurbRamp/`, `Obstacle/`, `SurfaceProblem/`, each with `correct/`
and `incorrect/` crops, then runs `split_data.py` so each ends up with a `ds/` folder of
`train/`, `val/`, `test/` and the class folders removed. Progress is in the SLURM log; the first lines
report how many labels qualified per class.

**2. Train** (several hours per label type; three run concurrently, then two):

```bash
sbatch fine_tune.slurm
```

Writes `<label_type>/dinov2/checkpoint-*/`; `save_total_limit = 2` keeps the best and the latest. The
best is named in the latest checkpoint's `trainer_state.json` under `best_model_checkpoint`.

**3. Evaluate**:

```bash
sbatch get_stats.slurm
```

Writes, per label type:

| file | contents |
|---|---|
| `predictions.csv` | `label_uid,split,true,p_correct,p_incorrect` for every `val`, `test` and `test_heldout` row |
| `metrics.json`, `report.txt` | accuracy, balanced accuracy, AUC, per-class precision/recall/F1 and recall at 90–99% precision, for `test`, `test_heldout` and the `hard` slice |
| `pvr.png`, `report.png` | precision–recall curve per class on the test split, and the curve with the text report |

Then fit the map the API serves, on the val split reweighted to the deployment class prior. Note that
`split.csv` (read by `evaluate.py`) and `deployment_priors.json` are written by data-generation changes still in
progress on another branch ([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3),
[#10](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/10)); today's `data_generation.py` writes
neither, and `evaluate.py` falls back to the old `ds/` image folders.

```bash
python -m calibration.calibrate --predictions CurbRamp/predictions.csv --priors deployment_priors.json \
    --label-type CurbRamp --out CurbRamp/calibration/curbramp.json
```

This writes the map (drop-in for `sidewalk-ai-api/accuracy_mappings/`), `calibration_report.json` (ECE, coverage and
precision at the 0.92 gate on the test split) and `reliability.png`. Unit tests for the metric and calibration code
run on a CPU: `pip install numpy pandas scikit-learn matplotlib pytest && pytest -q`.

To run a single label type outside SLURM: `python data_generation.py --label_type CurbRamp`, then
`split_data.py`, `fine_tune_classifier.py` and `evaluate.py` with the same flag.

## Publishing to Hugging Face

Today this is manual: upload the *best* checkpoint folder's `model.safetensors`, `config.json` and
`preprocessor_config.json` to `projectsidewalk/sidewalk-validator-ai-<labeltype>`, and the `ds/` folder
to `projectsidewalk/sidewalk-validator-ai-dataset-<labeltype>`; copy the map from `calibration/calibrate.py` to
`sidewalk-ai-api/accuracy_mappings/<labeltype>.json` and update `model_map.py` there with the training
date. Do not upload `optimizer.pt`, `scheduler.pt` or `rng_state.pth`. A script that does this, with a
content hash per dataset, is [#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7);
model and dataset cards are [#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2).

## Known limitations and open issues

The October 2026 audit is written up in [`docs/audit-2026-10-14.md`](docs/audit-2026-10-14.md) with
the evidence for each item. In order of how much they affect the numbers:

- **Split leaks across panoramas** ([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)). Labels average 2.8 per panorama across all types (Teaneck; 64% of panoramas carry more than one), and within one type's selected set CurbRamp averages 1.66 (62% of its labels share a panorama), SurfaceProblem 1.19 (27%), Obstacle 1.00; the file-level split puts overlapping crops on both sides. Published metrics are optimistic by an unknown amount. There is also no held-out-city evaluation.
- **Calibration under the wrong prior** ([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)). The confidence map is computed on a 55:45 test split and served against traffic that is closer to 90:10 (the decisive-crowd balance in Teaneck, a proxy). Platform-wide, when the AI votes "incorrect" the human majority agrees about 57% of the time; when it votes "correct", 96%. `calibration/calibrate.py` now reweights to the deployment prior, and production-calibrated maps for the deployed models are in [`docs/calibration-2026-10.md`](docs/calibration-2026-10.md) (not yet shipped to the API).
- **Train/serve skew** ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5), [sidewalk-ai-api#8](https://github.com/ProjectSidewalk/sidewalk-ai-api/issues/8)). Normalisation, camera pitch and depth checkpoint differ between training and serving.
- **Evaluation script bugs** ([#6](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/6)). Fixed in `evaluate.py`, which replaces `get_stats.py` (PR [#22](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/22), closed unmerged, had proposed a fix first); numbers in `report.png` files made by the old script are not reliable.
- **Publication** ([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)). The CurbRamp dataset on Hugging Face is a copy of the Crosswalk one; model repos carry 2.4 GB of optimizer state.
- **Imagery path** ([#8](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8), [#9](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/9)). Panoramas are re-downloaded from Google rather than read from the lab's pano store; 19 of 59 deployments are used (41 public, 18 private); crops are centred on stored coordinates that carry a small camera-tilt error ([sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54)).
- **Hygiene** ([#10](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/10), [#11](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/11)). No tests or CI; vendored 64 MB depth tree; frozen environment file.

Experiments we think are worth running next, each with a first experiment spelled out: soft targets and
validator reliability ([#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12)),
validation as re-typing ([#13](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/13)),
more context ([#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14)), held-out
cities ([#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15)), a VLM baseline
([#16](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/16)), temporal validation
([#17](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/17)), human–AI routing
([#18](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/18)), explanations
([#19](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/19)), distillation
([#20](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/20)) and the depth/crop-size
ablation ([#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21)). Earlier ideas are
in [#1](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/1).

## Lineage and related work

Automatic validation of crowd labels is the easier half of the lab's Crowd+AI sidewalk work, and this
repo is its third generation:

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

Sibling repositories: [sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) (serves
these models), [sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai) (tag
models), [sidewalk-panorama-tools](https://github.com/ProjectSidewalk/sidewalk-panorama-tools) (the
nightly panorama scrape and crop tools), [SidewalkWebpage](https://github.com/ProjectSidewalk/SidewalkWebpage)
(platform, orchestration, dashboards). The org-wide map of AI repos is
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

The repository does not yet carry a license file; adding one is tracked in
[#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2). The vendored
`Depth-Anything-V2/` code is Apache-2.0 (see its `LICENSE`).
