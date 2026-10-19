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
crowd clearly endorsed or clearly rejected, and turns each into a training image: it reads the
panorama from the lab's pano store, renders a 90° perspective view centred on the label (with rig-tilt correction), runs
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
> renamed column and a broken panorama download); it now reads panoramas from the lab's pano store
> instead and runs again, but several
> findings about the training signal, the split and calibration are open as issues and need a retrain
> to resolve. Read [Known limitations](#known-limitations-and-open-issues) before trusting a number.

## How it works

```
/v3/api/rawLabels (19 cities)          data_generation.py
        │                                     │
        ▼                                     ▼
 [1] select labels      human-authored only · net ≥3 human agree → correct · net ≥2 human disagree → incorrect
        │               (AI votes excluded; pre-downsampling counts → deployment_priors.json;
        │                correct class then downsampled to 55%)
        ▼
 [2] read panorama      pano store <city>/<pano_id[:2]>/<pano_id>.jpg → 8192×4096 equirectangular
        ▼
 [3] perspective view   90° FOV, 2048², centred on the label, rig-tilt corrected (sidewalk-panorama-tools)
        ▼
 [4] depth-sized crop   Depth Anything V2 (metric, ViT-B) at the centre → crop_half = 6100 / depth_m px → 640 px
        ▼
 <label_type>/{correct,incorrect}/<city>_<label_id>.webp  +  <label_type>/manifest.csv (one row per label)
        │
        ▼
 [5] split              split_data.py      grouped by panorama + labels ≤3 m apart, 70 / 15 / 15 (+ held-out cities)  →  <label_type>/split.csv
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

### Components

| File | Role | Key constants |
|---|---|---|
| [`data_generation.py`](data_generation.py) | Steps 1–4. Pulls labels, selects classes, writes `deployment_priors.json`, reads panoramas from the pano store, renders label-centred views, sizes crops by depth, appends to a resumable `manifest.csv`. | `CITIES`, `CORRECT_PROPORTION = 0.55`, `SEED = 42`, FOV 90°, 2048² view, `6100 / depth`, `--tilt-beta 0.95`, `--depth-encoder vitb` |
| [`split_data.py`](split_data.py) | Step 5. Assigns manifest rows to train/val/test by panorama (stratified by class), merging panoramas whose labels are within a few metres of each other (the same feature seen twice), optional held-out cities, joins the `hard` flag; prints the share of val/test labels with a train label within 2/5/10 m; writes `split.csv`, never touches images. | 70 / 15 / 15 of rows, `--merge-radius-m 3` (coordinates from `--rawlabels DIR`, default `<out>/rawlabels`), `--max-group-share 0.05`, `--seed`, `--holdout-cities`, `--hard-examples` |
| [`fine_tune_classifier.py`](fine_tune_classifier.py) | Step 6. Hugging Face `Trainer` on `facebook/dinov2-large`, from `split.csv`; val uses the checkpoint's image processor (as eval and serving do). | lr 1e-5, batch 8 × 4 accumulation, `--epochs 25`, warm-up 10%, `--seed`, best by macro-F1 (`--best-metric loss`), resumes from its last checkpoint, `--num-workers`, `id2label = {0: correct, 1: incorrect}` |
| [`hard_examples.py`](hard_examples.py) | Builds [`analysis/hard_examples_2026-10-17.csv`](analysis/hard_examples_2026-10-17.csv) (`label_uid,reason`) from a private production export: labels the deployed AI got wrong against a unanimous crowd (≥3 votes) or one of the project leads' accounts (configured via `EXPERTS`, not named). Feeds `split_data.py --hard-examples`. | |
| [`evaluate.py`](evaluate.py) | Step 7. Predictions for every held-out row, then AUC, accuracy and balanced accuracy, per-class P/R/F1, recall at precision. (`get_stats.py` is a stub that says so.) | splits `val`, `test`, `test_heldout`; `hard` slice |
| [`calibration/`](calibration) | Step 7b. Isotonic confidence→accuracy maps in the API's format: `calibrate.py` for a retrained model (deployment-prior weighting), `calibrate_from_production.py` for the deployed models from production outcomes. | gate 0.92; grid 0.5–1.0 at 0.0025, plus 1e-6 steps from 0.99998 to 1.0 |
| [`publish_to_hf.py`](publish_to_hf.py), [`hf_cards.py`](hf_cards.py) | Step 8. Stages a label type's model repo (inference files only) and dataset repo (crops by `split.csv`, `SHA256SUMS`, Croissant) with generated cards; `--audit` checks the live repos; `--push` / `--prune-remote` upload or clean up behind `--yes`. | refuses `optimizer.pt`, `scheduler.pt`, `rng_state*.pth`, `training_args.bin`, `scaler.pt`, `checkpoint-*` |
| [`analysis/`](analysis) | Scripts behind the production measurements in [`docs/calibration-2026-10.md`](docs/calibration-2026-10.md). | outputs live outside the repo |
| `*.slurm` | Hyak job files for steps 1–5, 6 and 7 (five label types per job). | `gpu-l40s`, 100 GB, 1 GPU |
| `Depth-Anything-V2/` | Vendored upstream code (unmodified); the checkpoint is downloaded separately. | `depth_anything_v2_metric_vkitti_vitb.pth` (`vitl` with `--depth-encoder vitl`) |

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

Labels from non-GSV imagery providers, rows with unusable pano geometry, and legacy labels with
unconverted coordinates (negative `pano_y`: Seattle 193, CDMX 14, Pittsburgh 8, SPGG 1) are excluded, and
so are labels placed by the `SidewalkAI` account (RampNet curb ramps in Vancouver WA, Richmond VA and
Laurens IA), because production never sends those to the validator; `--include-ai-labels` keeps them,
tagged `author = ai` in the manifest. The 55:45 ratio is a training-balance choice, not the real prior:
among labels the crowd decided in Teaneck the balance is roughly 90:10, a proxy for production traffic
rather than a measurement of it. `data_generation.py` therefore writes the pre-downsampling class counts
to `deployment_priors.json`, which `calibration/calibrate.py` reweights to
([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)).

> [!WARNING]
> Counting human votes only shrinks the classes, most of all CurbRamp `incorrect`, because the AI's
> disagree votes no longer count. Across the 16 cached training cities (of the 19 in `CITIES`) the
> `incorrect` sets go from (API columns) to (human votes only): CurbRamp 7,957 → 2,626, NoCurbRamp 7,842 → 5,923, Obstacle
> 4,412 → 3,556, SurfaceProblem 4,688 → 3,623, Crosswalk 1,974 → 1,034. With the 55:45 rule the
> `correct` class shrinks with them, so a retrain has much less CurbRamp data than the counts in the
> API columns suggest.

## Crop geometry

The panorama is resized to 8192×4096. A 2048² perspective view with a 90° horizontal field of view is
rendered **centred on the label** with `label_view` + `render_view` from
[`sidewalk-panorama-tools`](https://github.com/ProjectSidewalk/sidewalk-panorama-tools) (`sidewalk_panorama_tools.geometry`,
pinned to a commit in `requirements.txt`), the implementation the serving API is meant to share
([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5)). Human labels are clicked in the
gravity-levelled viewer frame while the stored panorama is in the camera-rig frame, so the view is rendered
with the rig-tilt correction from the label's `camera_pitch`/`camera_roll` at `beta = 0.95`
([sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54));
AI-authored labels are already rig-frame and are rendered without it. Depth Anything V2 (metric, VKITTI,
ViT-B as the API serves, `max_depth = 80`) runs on the view (BGR, as `infer_image` expects); the depth at
the centre sets the crop half-width as `6100 / depth_m` pixels, clipped to the view (`clipped = 1` in the
manifest), then the crop's long side is resized to 640 px and saved as WebP quality 95. The API applies the
same tilt correction with the same package in [sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17).

> [!IMPORTANT]
> **Deployment order.** Models retrained on these crops expect tilt-corrected input, so sidewalk-ai-api#17
> must be deployed before a retrained model is published; the current models keep working on the
> uncorrected API until then.

One residual difference remains by design: training reads crops back from WebP quality 95, while the API
classifies the in-memory PIL crop. On fresh crops the two differ by a mean |Δ| of 1.1 grey levels.

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

These are still the manual 2025-08-16 uploads: every model repo also holds `optimizer.pt`, `scheduler.pt`,
`rng_state.pth` and `training_args.bin`, and none of the ten repos has a real card (`python publish_to_hf.py --audit`
lists both problems). [`publish_to_hf.py`](#publishing-to-hugging-face) now builds the repos with cards, but nothing is
republished until the retrain on the panorama-grouped split.

## Setup

Create the conda environment (Linux with an NVIDIA GPU; the file is a frozen export of the Hyak
environment used for the published models):

```bash
conda env create -f environment.yml
conda activate sidewalk-validator-ai
```

Download the depth checkpoint into the repo root (ViT-B, the one the API serves; the Large one only for
`--depth-encoder vitl`):

```bash
# pinned to the same repo commit as sidewalk-ai-api's download_models.sh
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/e96fd0c8cedf1825a55b9c61309f754982948df0/depth_anything_v2_metric_vkitti_vitb.pth
```

> [!IMPORTANT]
> Activate the conda environment before submitting any SLURM job or running any script; the job files
> do not activate it for you.

## Running on Hyak

The three job files each process all five label types. They are written for the `makelab` account on
klone's `gpu-l40s` partition; edit the `#SBATCH` lines for anything else.

**1. Generate datasets** (hours to days; reads every panorama from the pano store):

```bash
PANO_STORE=/path/to/Panoramas sbatch data_generation.slurm
```

`PANO_STORE` (passed as `--pano-store`) is required. It points either at the lab's pano store itself or at a
staged copy of it on the cluster that keeps the store's layout, `<city_id>/<pano_id[:2]>/<pano_id>.jpg` with the
city_ids of `CITIES` (the layout sidewalk-ai-api reads through `SIDEWALK_SCRAPES_DIR`). The root must hold a
`.pano-store` marker file (any contents; its first line is recorded in `generation_run.json`); a staged copy
writes its own only after a verified pull. The staging step itself is not in this repo yet. Each panorama is read
with `cv2.imread` and resized to 8192×4096 (`INTER_LINEAR`), the same method as the API's cache path; identical
pixels also need the same opencv build, and the API and this environment currently pin different ones. Nothing is
downloaded. The job and the script check at startup that the store and its marker exist, and the script that
every selected city has a directory (`--allow-missing-cities` to run without some). A missing file is believed
only while the marker and the city directory are still there, and any other I/O error stops the run, so a
dropped mount cannot turn into silent `no_pano` rows.

Per label type this writes `<label_type>/{correct,incorrect}/<city>_<label_id>.webp`,
`<label_type>/manifest.csv` (one row per attempted label: `label_uid` = `<city>:<label_id>`, class, author,
human votes, the provider (`pano_source`), the store file read (`pano_path`, absolute) and its size before the
resize (`pano_native_w`/`pano_native_h`), pose, `tilt_applied`, `depth_m`, `crop_half_px`, `clipped`, `crop_path`, `status`) and
`generation_run.json` (settings, the store path, and a hash of the selection), plus the shared `deployment_priors.json` in the
output root. Views are rendered on the GPU with the package's torch recipe and depth runs on the same device;
the script stops if there is no CUDA device (`--allow-cpu` is for smoke tests). The manifest is written row by
row, so a killed job can simply be resubmitted: labels already `ok`, still selected and with the same class are
skipped. If the selection itself changed (new votes since the first run, other cities, or toggling `--include-ai-labels`), the job stops and says how many rows
are stale; `--force-reselect` drops them and carries on. The job then runs `split_data.py`, which writes
`<label_type>/split.csv` (the `ok` manifest rows plus `hard`, joined by `label_uid` from
`--hard-examples`, and `split` ∈ `train`/`val`/`test`/`test_heldout`), grouped by panorama so no panorama
appears in two splits, and leaves every image where it is. A new hard-example list only needs this step re-run.
Panorama grouping alone still let the same curb ramp or obstacle, labelled from an adjacent panorama or by another
user, land on both sides: on the 19 training cities 3.4% of CurbRamp, 8.2% of NoCurbRamp and 2.3% of Obstacle test
labels had a same-type train label within 2 m
([#27](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/27)). So the split step also merges
panorama groups whose labels are within `MERGE_RADIUS_M` metres (default 3; `MERGE_RADIUS_M=0 sbatch
data_generation.slurm` turns it off), reading label coordinates from the `rawlabels/` cache, and prints the share of
val/test labels with a train label within 2, 5 and 10 m either way. With merging on, it stops if one group holds
more than 5% (and more than 50) of a type's train/val/test rows (`--max-group-share`) or more than 1% of labels have
no coordinates, and it warns when the achieved shares drift more than 3 points from 70 / 15 / 15. The 3 m radius assumes label coordinates are accurate to
about a metre, which they are not always (a label's position is estimated from the panorama), so read the 5 m column
after the first real run before calling this leak fixed.

The job pulls `rawLabels` once into `rawlabels/` and reuses it on resubmission (`FRESH=1 sbatch
data_generation.slurm` re-pulls), so a resumed run sees the same selection. Even after a re-pull the selection
moves little: the kept `correct` labels are the lowest by a seeded hash of `label_uid`, so one label entering or
leaving the pool changes at most one kept label in and one out. A run stops rather than shrink the dataset
silently: after 50 panoramas in a row missing or unusable (usually a wrong `PANO_STORE` or city directory names that
are not the city_ids), after 50 consecutive per-label errors, or at the end if more than 20% of attempted labels
ended `no_pano`/`bad_pano` (`--max-missing-frac`). A panorama not in the store is `no_pano`; a file that cannot be
decoded, or is not 2:1, is `bad_pano`, never a stretched crop. Moving the store does not invalidate a resume: its
path is recorded for provenance, not compared. `ok` rows without a `pano_path` (made before the pano store) are
stale and redone, and the summary says how many. Set `HOLDOUT` in the job file to send whole cities to `test_heldout`. Progress is
in the SLURM log; the first lines report the deployment prior and how many labels were selected per class.

**2. Train** (several hours per label type; three run concurrently, then two):

```bash
sbatch fine_tune.slurm
```

Trains on the `train` rows of `split.csv` and selects on `val`. Writes `<label_type>/dinov2/checkpoint-*/`;
`save_total_limit = 2` keeps the best and the latest. The best is named in the latest checkpoint's
`trainer_state.json` under `best_model_checkpoint` and in `training_run.json`, which sits next to copies of
`split.csv` and `deployment_priors.json` in `<label_type>/dinov2/`.

**3. Evaluate**:

```bash
sbatch get_stats.slurm
```

Writes, per label type:

| file | contents |
|---|---|
| `predictions.csv` | `label_uid,split,true,p_correct,p_incorrect` for every `val`, `test` and `test_heldout` row |
| `metrics.json`, `report.txt` | accuracy, balanced accuracy, AUC, per-class precision/recall/F1 and recall at 90–99% precision, for `test`, `test_heldout` and the `hard` slice (small: e.g. 15 of 132 CurbRamp test rows, so read its numbers with wide error bars) |
| `pvr.png`, `report.png` | precision–recall curve per class on the test split, and the curve with the text report |

The same job then fits the map the API serves, on the val split reweighted to the deployment class prior
in `deployment_priors.json`; for one label type:

```bash
python -m calibration.calibrate --predictions CurbRamp/predictions.csv --priors deployment_priors.json \
    --label-type CurbRamp --out CurbRamp/calibration/curbramp.json
```

This writes the map (drop-in for `sidewalk-ai-api/accuracy_mappings/`), `calibration_report.json` (ECE, coverage and
precision at the 0.92 gate on the test split) and `reliability.png`. Unit tests for the metric and calibration code
run on a CPU: `pip install -r requirements.txt` (or just the CPU subset in `.github/workflows/tests.yml`), then
`pytest -q`.

To run a single label type outside SLURM: `python data_generation.py --label_type CurbRamp`, then
`split_data.py` (label coordinates from `<out>/rawlabels/`, or `--rawlabels DIR`, or `--merge-radius-m 0` without them), `fine_tune_classifier.py`
and `evaluate.py` with the same flag (the split's group-size guard needs a group of more than 50 rows, so it never
trips on a smoke test). For a smoke test add
`--limit 20 --out /some/scratch/dir --allow-cpu`; `--select-only` stops after selection and the priors (no GPU
needed), and `--rawlabels-cache DIR` reads saved `<city_id>.csv` exports instead of the API. Runs with `--limit`
or `--cities` write `deployment_priors.partial.json`, never the real `deployment_priors.json`.

## Publishing to Hugging Face

[`publish_to_hf.py`](publish_to_hf.py) replaces the old manual upload
([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7); cards are part of
[#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2)). It reads the same output root as every other
step and, by default, only stages, without network or token:

```bash
python publish_to_hf.py --label_type CurbRamp --out /path/to/output   # writes CurbRamp/publish/{model,dataset}/
```

| folder | contents |
|---|---|
| `publish/model/` | `model.safetensors`, `config.json`, `preprocessor_config.json` from the best checkpoint (the one `evaluate.py` scores: `best_model_checkpoint` in the latest checkpoint's `trainer_state.json`); that `trainer_state.json` trimmed to its log history and best-checkpoint fields; the map from `calibration/calibrate.py` renamed `confidence_precision_map.json`, with its `calibration_report.json`; `metrics.json`; `training_run.json`; the model card |
| `publish/dataset/` | crops as `<split>/<class>/<city>_<label_id>.webp`, placed by `split.csv` (never by listing a directory), hard-linked or copied (`--link-mode copy`); `split.csv` with `crop_path` rewritten to that layout; `manifest.csv`; this type's `deployment_priors.json` entry; `generation_run.json`; `dataset_info.json`; `SHA256SUMS`; `croissant.json`; the dataset card |

Staging stops instead of publishing the wrong thing. It refuses training-only files (`optimizer.pt`, `scheduler.pt`,
`rng_state*.pth`, `training_args.bin`, `scaler.pt`, any `checkpoint-*` directory) and checks every staged folder for them
again before upload. It also refuses:

- a manifest with any row whose `label_type` is not `--label_type`, a missing crop, or a `label_uid` that is not
  `<city>:<label_id>`;
- a map whose `calibration_report.json` names another label type or none, or a `run_key.json` for another type;
- a `split.csv` whose sha256 differs from the one `fine_tune_classifier.py` recorded in `dinov2/run_key.json`
  (the split changed after training);
- a `predictions.csv` with duplicate rows or that does not cover exactly the split's `val`/`test`/`test_heldout`
  rows (metrics and map from another split); this check runs only when `metrics.json` or a map exists, since
  without them the card has no evaluation to tie to the split;
- a dataset whose content hash (the sha256 of `SHA256SUMS`) equals another staged label type's.

Two weaker checks only warn: file times out of order (weights newer than predictions, predictions newer than
metrics or map), and the `train` split against the images per epoch implied by `trainer_state.json` (steps per
epoch × 8 × 4, which flags the CurbRamp/Crosswalk swap, 103 × 32 ≈ 3,300 images against 1,108 crops, but cannot
tell two similar-sized sets apart). The absolute pano-store paths (`pano_path`, `pano_store`) and the store marker
text are dropped from the published CSV and JSON files.

The cards follow the Hugging Face model and dataset templates and are filled from the files above: the evaluation
table from `metrics.json`; the prior, ECE, gate coverage and the confidence at which each verdict reaches 0.92
from the map and its report; class balance, split sizes, cities, capture years and content hash from `split.csv`.
What the pipeline does not record goes in `<out>/<label_type>/card_overrides.json`:

```json
{"failure_modes": ["..."], "split_seed": 42, "dataset_license": "<hub license id>", "doi": "...", "model_doi": "..."}
```

Anything still unknown is written as `TODO:` (the DOI lines just say none has been minted yet).
`croissant.json` is validated when `mlcroissant` is installed and written unvalidated otherwise.

```bash
python publish_to_hf.py --audit                                        # live repos: stray files, overlapping datasets
python publish_to_hf.py --label_type CurbRamp --out ROOT --push        # restage, then dry run
python publish_to_hf.py --label_type CurbRamp --out ROOT --push --yes
python publish_to_hf.py --label_type CurbRamp --out ROOT --push-staged --yes   # upload as staged, after editing a card
python publish_to_hf.py --label_type CurbRamp --out ROOT --push --yes --create-repo --repo-prefix <user>/test-validator
python publish_to_hf.py --prune-remote --label_type all [--yes]        # delete stray training files only
```

The dry run prints the files and sizes, the remote head, the remote files the commit would delete, and what share
of the staged crops (by sha256) already sits in each other label type's dataset repo. With `--yes`, which needs
`huggingface_hub>=0.34.4` and a token (`HF_TOKEN` or `huggingface-cli login`), each repo gets exactly one
`create_commit` holding every addition and deletion, based on the head the preview listed. A concurrent change
makes it fail instead of interleaving, an interrupted push leaves the repo as it was, `.gitattributes` stays, and
history is not rewritten, so the revisions sidewalk-ai-api pins keep resolving. (`upload_folder` is avoided: with
`hf_xet` installed, huggingface_hub 2.x splits a large folder into many commits.) `--yes` is refused, for both
repos before either is touched, while a card still contains `TODO:` (unless `--allow-todo`), when a remote cannot be
listed, when more than 5% of the crops are found in another type's dataset repo, or when another type's crops carry
no sha256 to compare. Repos under the `projectsidewalk` namespace (in any letter case) are never created; under
another namespace, `--create-repo` creates them private. `--push` restages, so hand
edits to the cards are lost; `--push-staged` uploads `publish/` as it is, after checking that nothing but the two
`README.md` cards changed since staging (`publish/staging.json`), that the crops still match `SHA256SUMS`, and that
the best checkpoint is unchanged. `--prune-remote` is the clean-up for the current repos before any retrain: it
deletes only the training-only files, one commit per repo, and the old blobs stay in history. Exit code 3 means any
Hub failure: unreachable, a listing that failed, or a `create_repo`/`create_commit` the Hub rejected. The dataset
and model repos are two separate commits. If the second fails, the first has landed; the error names it and prints
the `--push-staged --what model --yes` (or `dataset`) command that uploads the rest.

Two practical points for the first real upload. Make it a private test round with the full CurbRamp set
(`--push --yes --create-repo --repo-prefix <user>/test-validator`), since a commit of ~5,000 additions and
deletions is not exercised offline. Then push CurbRamp before Crosswalk: while the CurbRamp repo still holds the
Crosswalk crops, an unchanged Crosswalk set is refused at 100% overlap against it, as it should be. Copying the map to `sidewalk-ai-api/accuracy_mappings/<labeltype>.json` and updating
`model_map.py` there remain manual.

## Known limitations and open issues

The October 2026 audit is written up in [`docs/audit-2026-10-14.md`](docs/audit-2026-10-14.md) with
the evidence for each item. In order of how much they affect the numbers:

- **Split leaks across panoramas** ([#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)). Labels average 2.8 per panorama across all types (Teaneck; 64% of panoramas carry more than one), and within one type's selected set CurbRamp averages 1.66 (62% of its labels share a panorama), SurfaceProblem 1.19 (27%), Obstacle 1.00. The file-level split behind the published models put overlapping crops on both sides, so their metrics are optimistic by an unknown amount. `split_data.py` now splits by panorama, merges panoramas whose labels are within 3 m of each other (the same feature seen from two panoramas, [#27](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/27)) and can hold out whole cities; the published models have not been retrained on it yet.
- **Calibration under the wrong prior** ([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)). The confidence map is computed on a 55:45 test split and served against traffic that is closer to 90:10 (the decisive-crowd balance in Teaneck, a proxy). Platform-wide, when the AI votes "incorrect" the human majority agrees about 57% of the time; when it votes "correct", 96%. `calibration/calibrate.py` now reweights to the deployment prior, and production-calibrated maps for the deployed models are in [`docs/calibration-2026-10.md`](docs/calibration-2026-10.md) (not yet shipped to the API).
- **Train/serve skew** ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5), [sidewalk-ai-api#8](https://github.com/ProjectSidewalk/sidewalk-ai-api/issues/8)). Normalisation, camera pitch and depth checkpoint differed between training and serving. Training now uses the serving preprocessing for val, label-centred views from the shared geometry package and the ViT-B depth checkpoint; the API applies the same tilt correction in [sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17), which must be deployed before any retrained model (nothing has been retrained yet).
- **Evaluation script bugs** ([#6](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/6)). Fixed in `evaluate.py`, which replaces `get_stats.py` (PR [#22](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/22), closed unmerged, had proposed a fix first); numbers in `report.png` files made by the old script are not reliable.
- **Publication** ([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)). The CurbRamp dataset on Hugging Face is a copy of the Crosswalk one; model repos carry 2.4 GB of optimizer state. [`publish_to_hf.py`](#publishing-to-hugging-face) now stages clean repos with cards, content hashes and the checks that would have caught both, and `--prune-remote` can delete the optimizer files today; the datasets and models themselves are not republished until the retrain.
- **Imagery path** ([#8](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8), [#9](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/9)). Panoramas are read only from the lab's pano store (`--pano-store`), and only GSV labels are used, since serving is GSV-only and other providers need per-provider pose handling; 19 of 59 deployments are used (41 public, 18 private); the camera-tilt error in stored coordinates ([sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54)) is now corrected in training crops (beta 0.95).
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

The code in this repository is released under the [MIT License](LICENSE). The vendored
`Depth-Anything-V2/` code keeps its own Apache-2.0 license (see its `LICENSE`). Licenses for the
released datasets and models are tracked separately in
[#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2).
