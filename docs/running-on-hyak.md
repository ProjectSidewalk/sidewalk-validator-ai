# Setup and running the pipeline on Hyak

How to set up a checkout and run the three SLURM jobs that produce a retrain: dataset generation, fine-tuning and
evaluation with calibration. The pipeline itself is described in the [README](../README.md#how-it-works); the data
rules are in [`training-data.md`](training-data.md); publishing the result is in [`publishing.md`](publishing.md).

## Setup

Clone with the Depth Anything V2 submodule (`Depth-Anything-V2/`, pinned to the commit `sidewalk-ai-api` serves
with):

```bash
git clone --recurse-submodules https://github.com/ProjectSidewalk/sidewalk-validator-ai
# or, in a clone made without --recurse-submodules:
git submodule update --init
```

Without it `Depth-Anything-V2/` is an empty directory and `data_generation.py` cannot import the depth model.
`data_generation.slurm` initialises the submodule itself and stops with a message if it is not checked out at the
pinned commit. Clones that want to skip the 45 MB of history from the old vendored copy can use
`--filter=blob:none` ([`repo-hygiene.md`](repo-hygiene.md)).

> [!IMPORTANT]
> A checkout from before `Depth-Anything-V2/` became a submodule (it was a vendored copy) needs a one-time
> migration. Wait until no job is running from that checkout, then:
>
> ```bash
> git checkout -- Depth-Anything-V2   # discard the .pyc files every run rewrites, or the pull refuses (don't git stash)
> git pull
> rm -rf Depth-Anything-V2            # ignored __pycache__/ leftovers block the submodule clone
> git submodule update --init
> ```

Create the conda environment (Linux with an NVIDIA GPU; the file is a frozen export of the Hyak environment):

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

Activate the conda environment before submitting any SLURM job or running any script; the job files do not activate
it for you. Unit tests for the metric and calibration code run on a CPU: `pip install -r requirements.txt` (or just
the CPU subset in `.github/workflows/tests.yml`), then `pytest -q`.

## The three jobs

Each job file processes all five label types. They are written for the `makelab` account on klone's `gpu-l40s`
partition; edit the `#SBATCH` lines for anything else.

### 1. Generate datasets

Hours to days; reads every panorama from the pano store.

```bash
PANO_STORE=/path/to/Panoramas sbatch data_generation.slurm
```

`PANO_STORE` (passed as `--pano-store`) is required. It points either at the lab's pano store itself or at a
staged copy of it on the cluster that keeps the store's layout, `<city_id>/<pano_id[:2]>/<pano_id>.jpg` with the
city_ids `/v3/api/cities` uses (the layout sidewalk-ai-api reads through `SIDEWALK_SCRAPES_DIR`). The root must hold a
`.pano-store` marker file (any contents; its first line is recorded in `generation_run.json`); a staged copy
writes its own only after a verified pull. The staging step itself is not in this repo yet. Each panorama is read
with `cv2.imread` and resized to 8192×4096 (`INTER_LINEAR`), the same method as the API's cache path; identical
pixels also need the same opencv build (see [`serving-parity-2026-10.md`](serving-parity-2026-10.md)). Nothing is
downloaded. The job and the script check at startup that the store and its marker exist, and the script, once
labels are selected, that every city with a selected label has a directory (`--allow-missing-cities` to run
without some; a listed city with no selected label, such as one whose labels are all non-GSV, needs none). A missing
file is believed only while the marker and the city directory are still there, and any other I/O error stops the
run, so a dropped mount cannot turn into silent `no_pano` rows.

**Outputs.** Per label type: `<label_type>/{correct,incorrect}/<city>_<label_id>.webp`, `<label_type>/manifest.csv`
(one row per attempted label: `label_uid` = `<city>:<label_id>`, class, `class_source`, author, human votes, the
provider (`pano_source`), the store file read (`pano_path`, absolute) and its size before the resize
(`pano_native_w`/`pano_native_h`), pose, `tilt_applied`, `depth_m`, `crop_half_px`, `clipped`, `crop_path`, `status`)
and `generation_run.json` (settings, the store path, and a hash of the selection), plus the shared
`deployment_priors.json` in the output root. Views are rendered on the GPU with the geometry package's torch recipe
and depth runs on the same device; the script stops if there is no CUDA device (`--allow-cpu` is for smoke tests).

**Resuming.** The manifest is written row by row, so a killed job can simply be resubmitted: labels already `ok`,
still selected and with the same class are skipped. If the selection itself changed (new votes since the first run,
a different city list, a different `EXPERTS`, or toggling `--include-ai-labels`), the job stops and says how many
rows are stale; `--force-reselect` drops them and carries on. `ok` rows without a `pano_path` (made before the pano
store) are stale and redone, and the summary says how many. Moving the store does not invalidate a resume: its path
is recorded for provenance, not compared.

**The rawLabels cache.** The job pulls `rawLabels` once into `rawlabels/` and reuses it on resubmission
(`FRESH=1 sbatch data_generation.slurm` re-pulls), so a resumed run sees the same selection. The pull first saves
the `/v3/api/cities` listing as `rawlabels/cities.json` and then fetches one `<city_id>.csv` per public deployment
in it. The five label-type processes read that file, so the city list is fixed by the pull too, and nothing calls
the API after it. A cache made before city discovery has no `cities.json`; the job stops on one rather than mix
the old 19-city cache with the new list. Re-pull with `FRESH=1`, or resume such a run with `--cities` and its old
list ([`training-data.md`](training-data.md#cities)). Even after a re-pull the selection moves little: the kept
`correct` labels are the lowest by a seeded hash of `label_uid`, so one label entering or leaving the pool changes
at most one kept label in and one out.

**Stopping rather than shrinking.** A run stops after 50 panoramas in a row missing or unusable (usually a wrong
`PANO_STORE` or city directory names that are not the city_ids), after 50 consecutive per-label errors, or at the
end if more than 20% of attempted labels ended `no_pano`/`bad_pano` (`--max-missing-frac`). A panorama not in the
store is `no_pano`; a file that cannot be decoded, or is not 2:1, is `bad_pano`, never a stretched crop.

**The split.** The job then runs `split_data.py`, which writes `<label_type>/split.csv` (the `ok` manifest rows
plus `hard`, joined by `label_uid` from `--hard-examples`, and `split` ∈ `train`/`val`/`test`/`test_heldout`),
grouped by panorama so no panorama appears in two splits, and leaves every image where it is. A new hard-example
list only needs this step re-run. Panorama grouping alone still let the same curb ramp or obstacle, labelled from an
adjacent panorama or by another user, land on both sides: on the 19 original training cities 3.4% of CurbRamp, 8.2%
of NoCurbRamp and 2.3% of Obstacle test labels had a same-type train label within 2 m
([#27](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/27)). So the split step also merges
panorama groups whose labels are within `MERGE_RADIUS_M` metres (default 3; `MERGE_RADIUS_M=0 sbatch
data_generation.slurm` turns it off), reading label coordinates from the `rawlabels/` cache, and prints the share
of val/test labels with a train label within 2, 5 and 10 m either way. With merging on, it stops if one group holds
more than 5% (and more than 50) of a type's train/val/test rows (`--max-group-share`) or more than 1% of labels
have no coordinates, and it warns when the achieved shares drift more than 3 points from 70 / 15 / 15. The 3 m
radius assumes label coordinates are accurate to about a metre, which they are not always (a label's position is
estimated from the panorama), so read the 5 m column after a real run before calling this leak fixed. Set `HOLDOUT`
in the job file to send whole cities to `test_heldout`. Progress is in the SLURM log; the first lines report the
deployment prior and how many labels were selected per class.

### 2. Train

Several hours per label type; three run concurrently, then two.

```bash
sbatch fine_tune.slurm
```

Trains on the `train` rows of `split.csv` and selects on `val`. Writes `<label_type>/dinov2/checkpoint-*/`;
`save_total_limit = 2` keeps the best and the latest. The best is named in the latest checkpoint's
`trainer_state.json` under `best_model_checkpoint` and in `training_run.json`, which sits next to copies of
`split.csv` and `deployment_priors.json` in `<label_type>/dinov2/`. Training targets other than the default hard
labels (`--target soft|weighted`) write to `<label_type>/dinov2-<target>` instead
([`training-data.md`](training-data.md#training-targets)).

### 3. Evaluate and calibrate

```bash
sbatch get_stats.slurm
```

Writes, per label type:

| file | contents |
|---|---|
| `predictions.csv` | `label_uid,split,true,p_correct,p_incorrect` for every `val`, `test` and `test_heldout` row |
| `metrics.json`, `report.txt` | accuracy, balanced accuracy, AUC, per-class precision/recall/F1 and recall at 90–99% precision, for `test`, `test_heldout` and the `hard` slice (small: e.g. 15 of 132 CurbRamp test rows, so read its numbers with wide error bars) |
| `pvr.png`, `report.png` | precision–recall curve per class on the test split, and the curve with the text report |
| `per_city.md` | with `--per-city` only: the same metrics per city of `test` and `test_heldout` (cities with at least `--min-city-n` rows, default 50), also under `per_city` in `metrics.json`. With `--vote-weights` too, a second table per split for the gold slice per city: metrics where a city's gold rows reach the same minimum, and every other city listed with its gold count, 0 included (`per_city.<split>.gold`; [#41](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/41)) |

The same job then fits the map the API serves, on the val split reweighted to the deployment class prior in
`deployment_priors.json`; for one label type:

```bash
python -m calibration.calibrate --predictions CurbRamp/predictions.csv --priors deployment_priors.json \
    --label-type CurbRamp --out CurbRamp/calibration/curbramp.json
```

This writes the map (drop-in for `sidewalk-ai-api/accuracy_mappings/`), `calibration_report.json` (ECE, coverage
and precision at the 0.92 gate on the test split) and `reliability.png`. `--hold-back incorrect
--hold-back-reason "..."` (either verdict or both; the reason is required and goes on the model card) caps that
verdict's map at 0.9199, the same hold-back `calibrate_from_production.py` applies to a class whose gate precision
is not established: thresholds above the cap serve 0.9199, so the verdict never clears the gate and never becomes a
vote. The report records the held-back verdicts, the reason and what the uncapped map would have done at the gate.
The October 2026 ship held `incorrect` back for every type
([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56),
[#57](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/57)).

## Outside SLURM, and smoke tests

To run a single label type: `python data_generation.py --label_type CurbRamp`, then `split_data.py` (label
coordinates from `<out>/rawlabels/`, or `--rawlabels DIR`, or `--merge-radius-m 0` without them),
`fine_tune_classifier.py` and `evaluate.py` with the same flag (the split's group-size guard needs a group of more
than 50 rows, so it never trips on a smoke test). For a smoke test add `--limit 20 --out /some/scratch/dir
--allow-cpu`; `--select-only` stops after selection and the priors (no GPU needed), and `--rawlabels-cache DIR`
reads saved `<city_id>.csv` exports and the `cities.json` listing instead of the API (a missing listing is fetched
once and saved there). Runs with `--limit` or `--cities` write `deployment_priors.partial.json`, never the real
`deployment_priors.json`.

## Held-out-city benchmark

The test split is drawn from the same cities the model trains on, so it says nothing about a city the model has
never seen ([#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15)). The steps above support
leave-one-city-out: `split_data.py --holdout-cities` sends every row of a city (and any panorama group that touches
it) to `test_heldout`, training never reads `test_heldout`, and `evaluate.py --per-city` reports each city of
`test` and `test_heldout` separately. The round-3 retrain held three deployments out this way
([`retrain-2026-10.md`](retrain-2026-10.md)). Give each held-out city its own run root so the main `split.csv` and
checkpoints are untouched; the crops are linked, not copied, because `crop_path` is relative to `<label_type>/`:

```bash
DATA=/path/to/data            # data_generation.py's --out: <label_type>/{correct,incorrect}/, manifest.csv, rawlabels/
LT=CurbRamp
CITY=amsterdam                # a city_id from the manifest
RUN=$DATA/loco/$CITY
mkdir -p $RUN/$LT
ln -s $DATA/$LT/correct $DATA/$LT/incorrect $DATA/$LT/generation_run.json $RUN/$LT/
cp $DATA/deployment_priors.json $RUN/
python split_data.py --label_type $LT --out $RUN --manifest $DATA/$LT/manifest.csv --rawlabels $DATA/rawlabels \
    --hard-examples analysis/hard_examples_2026-10-17.csv --holdout-cities $CITY --seed 42
python fine_tune_classifier.py --label_type $LT --data-dir $RUN --seed 42
python evaluate.py --label_type $LT --data-dir $RUN --per-city
```

`$RUN/$LT/per_city.md` then has the held-out city's accuracy, AUC and per-class precision/recall in its
`test_heldout` table, next to the seen cities in `test`. The comparison row is that city's row in the main run's
`test` table (`evaluate.py --per-city` on `$DATA`), where the city was trained on. That row covers only the city's
test share (about 15% of its rows), while `test_heldout` holds all of its rows, so the two differ in sample as well
as in training. `--min-city-n` applies to both splits. These per-city jobs are not in the SLURM files; on klone,
wrap the three commands per city in a copy of `fine_tune.slurm`. The production-side counterpart, whether the 0.92
gate holds per city for the deployed models, is [`per-city-calibration-2026-10.md`](per-city-calibration-2026-10.md)
(`python -m calibration.per_city`).
