# Publishing models and datasets to Hugging Face

[`publish_to_hf.py`](../publish_to_hf.py) replaces the manual uploads of 2025
([#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7); cards are part of
[#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2)). It reads the same output root as every
other pipeline step ([`running-on-hyak.md`](running-on-hyak.md)) and, by default, only stages, without network or
token:

```bash
python publish_to_hf.py --label_type CurbRamp --out /path/to/output   # writes CurbRamp/publish/{model,dataset}/
```

| folder | contents |
|---|---|
| `publish/model/` | `model.safetensors`, `config.json`, `preprocessor_config.json` from the best checkpoint (the one `evaluate.py` scores: `best_model_checkpoint` in the latest checkpoint's `trainer_state.json`); that `trainer_state.json` trimmed to its log history and best-checkpoint fields; the map from `calibration/calibrate.py` renamed `confidence_precision_map.json`, with its `calibration_report.json`; `metrics.json`; `training_run.json`; the model card |
| `publish/dataset/` | crops as `<split>/<class>/<city>_<label_id>.webp`, placed by `split.csv` (never by listing a directory), hard-linked or copied (`--link-mode copy`); `split.csv` with `crop_path` rewritten to that layout; `manifest.csv`; this type's `deployment_priors.json` entry; `generation_run.json`; `dataset_info.json`; `SHA256SUMS`; `croissant.json`; the dataset card |

## The October 2026 publish

The round-3 models were published on 2026-10-26 with `--push-staged --what model --yes`, one commit per model repo,
from a machine without the crops (they live on klone). `--what model` stages and pushes the model repo alone; the
card's dataset content hash becomes a `TODO` until the dataset is published from the crop machine
(`--what dataset` stages both and uploads the dataset). The five dataset repos are therefore still the 2025-08-16
uploads, and the CurbRamp dataset is still a copy of the Crosswalk one; republishing them is the open half of #7.
sidewalk-ai-api pins the five model revisions and reads the maps from the hub
([sidewalk-ai-api#22](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/22),
[#25](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/25)).

## Private deployments

Training cities include private deployments. The city lists, read from `/v3/api/cities` or `--public-cities
<file>`, drive two independent checks: private deployments are *removed* from every list and source string ("and N
non-public deployments"), their per-city metrics are pooled into one `non_public` row, manifest rows from them are
dropped and a dataset staging is refused when `split.csv` holds one; then a survivor gate reads every staged text
file and refuses on any case-insensitive hit for any non-public id, display name or hostname the API lists.
`staging.json` records the lists used and `--push-staged` runs the gate again.

## What staging refuses

Staging stops instead of publishing the wrong thing. It refuses training-only files (`optimizer.pt`,
`scheduler.pt`, `rng_state*.pth`, `training_args.bin`, `scaler.pt`, any `checkpoint-*` directory) and checks every
staged folder for them again before upload. It also refuses:

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

## Cards

The cards follow the Hugging Face model and dataset templates and are filled from the files above: the evaluation
table from `metrics.json`; the prior, ECE, gate coverage and the confidence at which each verdict reaches 0.92
from the map and its report; class balance, split sizes, cities, capture years and content hash from `split.csv`.
What the pipeline does not record goes in `<out>/<label_type>/card_overrides.json`:

```json
{"failure_modes": ["..."], "split_seed": 42, "dataset_license": "<hub license id>", "doi": "...", "model_doi": "..."}
```

Anything still unknown is written as `TODO:` (the DOI lines just say none has been minted yet).
`croissant.json` is validated when `mlcroissant` is installed and written unvalidated otherwise.

## Pushing

```bash
python publish_to_hf.py --label_type CurbRamp --out ROOT --what model  # model only, no crops needed; private cities removed
python publish_to_hf.py --label_type CurbRamp --out ROOT --push-staged --what model --allow-todo --yes   # cards keep 2–3 TODOs
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
another namespace, `--create-repo` creates them private. `--push` restages, so hand edits to the cards are lost;
`--push-staged` uploads `publish/` as it is, after checking that nothing but the two `README.md` cards changed since
staging (`publish/staging.json`), that the crops still match `SHA256SUMS`, and that the best checkpoint is
unchanged. `--prune-remote` deletes only the training-only files, one commit per repo, and the old blobs stay in
history. Exit code 3 means any Hub failure: unreachable, a listing that failed, or a `create_repo`/`create_commit`
the Hub rejected. The dataset and model repos are two separate commits. If the second fails, the first has landed;
the error names it and prints the `--push-staged --what model --yes` (or `dataset`) command that uploads the rest.

Two practical points for a dataset upload. Make it a private test round with the full CurbRamp set
(`--push --yes --create-repo --repo-prefix <user>/test-validator`), since a commit of ~5,000 additions and
deletions is not exercised offline. Then push CurbRamp before Crosswalk: while the CurbRamp repo still holds the
Crosswalk crops, an unchanged Crosswalk set is refused at 100% overlap against it, as it should be. Updating
sidewalk-ai-api's pinned revisions remains manual.
