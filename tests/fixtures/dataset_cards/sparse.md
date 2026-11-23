---
task_categories:
  - image-classification
size_categories:
  - "n<1K"
pretty_name: "Project Sidewalk CurbRamp label validation crops"
tags:
  - project-sidewalk
  - accessibility
  - sidewalks
  - street-view
  - crowdsourcing
  - label-validation
  - curbramp
configs:
  - config_name: default
    data_files:
      - split: train
        path: "train/*/*.webp"
      - split: validation
        path: "val/*/*.webp"
      - split: test
        path: "test/*/*.webp"
---

# sidewalk-validator-ai-dataset-curbramp

Street-level image crops of crowdsourced [Project Sidewalk](https://projectsidewalk.org) **CurbRamp** labels, each marked **correct** or **incorrect** by the crowd's validation votes. It is the training and evaluation data of [`projectsidewalk/sidewalk-validator-ai-curbramp`](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-curbramp), built by [sidewalk-validator-ai](https://github.com/ProjectSidewalk/sidewalk-validator-ai) (`data_generation.py`, `split_data.py`).

- **Crops:** 20 WebP images, one per label, laid out as `<split>/<correct|incorrect>/<city>_<label_id>.webp`.
- **Cities:** 3; **image capture years:** TODO: no image_capture_date values in the manifest.
- **Panoramas:** 14.
- **Content hash:** `ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff` (sha256 of `SHA256SUMS`, which lists the sha256 of every crop by path; recompute it to check an upload).

## How labels become correct or incorrect

Every label of this type and its validations are pulled from each deployment's public `/v3/api/rawLabels` export. Only **human** votes are counted: each vote carries a `validator_type`, and votes cast by the `SidewalkAI` account are excluded, so the model is never trained on its own output.

| class | rule (human votes) |
|---|---|
| `incorrect` | disagree − agree > 1 |
| `correct` | agree − disagree > 2, then randomly downsampled |
| (dropped) | unvalidated, unsure, or contested |

Labels placed by the `SidewalkAI` account are included (tagged `author = ai` in the manifest); labels with unusable panorama geometry or with legacy unconverted coordinates are excluded. The dataset includes 3 Mapillary, 1 Panoramax labels on non-GSV imagery. Mapillary clicks are stored in the rig frame, so those crops are not tilt-corrected; Panoramax crops are corrected only when both stored camera angles are non-zero (the only case its viewer levels the panorama). The manifest's `pose_source` says which pose each crop used. The `correct` class is then downsampled to 55% of the dataset by a seeded hash of `label_uid` (seed 42), a training-balance choice, not the real class mix. TODO: deployment_priors.json has no entry for CurbRamp.

## How a crop is made

The panorama is read from the lab's panorama store and resized to 8192×4096. A 2048² perspective view with a 90° field of view is rendered centred on the label with rig-tilt correction (beta = TODO: tilt beta for GSV). Depth Anything V2 (metric, VKITTI, TODO: encoder) estimates the depth at the view centre, the crop half-width is `6100 / depth_m` pixels (clipped to the view; `clipped = 1` in the manifest), and the crop's long side is resized to 640 px and saved as WebP quality 95.

## Splits

| split | correct | incorrect | total | share incorrect |
|---|---|---|---|---|
| train | 6 | 4 | 10 | 0.400 |
| val | 2 | 1 | 3 | 0.333 |
| test | 2 | 2 | 4 | 0.500 |
| **all** | 11 | 9 | 20 | 0.450 |

`split_data.py` groups labels by panorama so that no panorama appears in two splits (overlapping crops of one corner would otherwise leak), bins the groups by class ratio and fills train/val/test 70/15/15 of rows per bin with a seeded shuffle; no cities were held out. TODO: split seed (split_data.py does not record it; its default is 42; add `split_seed` to card_overrides.json). The Hub config maps `val` to the `validation` split.

## Cities

| city | crops | in test_heldout |
|---|---|---|
| city-a | 9 |  |
| city-b | 8 |  |
| city-c | 3 |  |

## Files

| file | contents |
|---|---|
| `<split>/<class>/<city>_<label_id>.webp` | the crops |
| `split.csv` | one row per crop: `label_uid` (`<city>:<label_id>`, the unique key; `label_id` alone repeats across cities), votes, pose, depth, crop size, `hard`, `split`, and `crop_path` within this repo |
| `manifest.csv` | every label the generator attempted, including failures (`status`) |
| `deployment_priors.json` | pre-downsampling class counts for this label type |
| `generation_run.json` | generator settings and the selection hash |
| `dataset_info.json` | the counts on this card, machine-readable |
| `SHA256SUMS` | sha256 of every crop |
| `croissant.json` | Croissant 1.1 metadata |

The absolute panorama-store paths recorded at generation time (`pano_path`, `pano_store`, `pano_store_marker`) are removed from the published copies.

## Intended use

Training and evaluating validators for Project Sidewalk CurbRamp labels, and research on crowd label quality. Not suitable for estimating how common a feature is in a city: labels exist where volunteers chose to label, and the classes are rebalanced.

## Versioning

Each upload is a commit of this repo; the content hash above identifies the crops exactly, and `generation_run.json` records the selection hash (`TODO: selection hash`), which changes when any label enters, leaves or changes class. A regenerated dataset is uploaded as a new commit with a new card. No DOI has been minted for this dataset yet.

## Citation

There is no paper for this dataset yet. Please cite Project Sidewalk and link the repository:

```bibtex
@inproceedings{saha2019projectsidewalk,
  author    = {Saha, Manaswi and Saugstad, Michael and Maddali, Hanuma Teja and Zeng, Aileen and Holland, Ryan and Bower, Steven and Dash, Aditya and Chen, Sage and Li, Anthony and Hara, Kotaro and Froehlich, Jon},
  title     = {Project Sidewalk: A Web-based Crowdsourcing Tool for Collecting Sidewalk Accessibility Data at Scale},
  booktitle = {Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems},
  year      = {2019},
  doi       = {10.1145/3290605.3300292}
}
```

Published from [sidewalk-validator-ai](https://github.com/ProjectSidewalk/sidewalk-validator-ai) with `publish_to_hf.py`.
