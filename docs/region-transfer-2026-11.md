# Cross-region transfer: leave-one-country-out, local fine-tuning and a region-conditioned validator (November 2026)

Experiment for [#33](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/33),
[#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52) (option 3) and the retrained-model half of
[#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15). Does one global validator transfer to a
country it never saw? If not, what helps most: local fine-tuning, a model that is told the country, or
per-country calibration of the global model?

<!-- BEGIN GENERATED: status (analysis/region_transfer.py) -->

**Status: not yet run.** No GPU job has run; the report stage replaces this line.

<!-- END GENERATED: status -->

## In short

<!-- BEGIN GENERATED: summary (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: summary -->

## Why

Round 3 holds out three deployments ([`retrain-2026-10.md`](retrain-2026-10.md), "Per deployment, unseen
deployments"): tucson-az, private city A (outside the US) and private city B (US). That is one data point outside
the US. It is also the weak spot. On private city A the round-3 AUC is 0.82 for CurbRamp, 0.67 for Obstacle and
0.61 for SurfaceProblem, against 0.97, 0.88 and 0.90 on the training deployments' test share. One deployment
cannot say whether the gap is about the country, the deployment or its class mix (92% incorrect for Obstacle
there).

Three questions follow, one per issue:

- **#52 (option 3):** how much does accuracy drop for a country the model never saw? Leave-one-country-out (LOCO)
  gives one held-out country per run, for every country with enough labels, instead of one deployment.
- **#33:** does a model that knows the country do better, in distribution and on an unseen country? And how many
  local labels does a new deployment need before a fine-tuned model beats the global one?
- **#15:** does a per-country calibration map, with no retraining, fix the gate where the global map misses?

## Setup

| | |
|---|---|
| Crops and split | round 3 (`retrain-2026-10/<Type>/split.csv`), unchanged; variants are rewrites of it |
| Recipe | round 3: DINOv2-Large, 25 epochs, batch 8 x 4, lr 1e-5, best val macro-F1 |
| LOCO rule | a country's `train` rows become `pool`, its `val` and `test` rows `test_heldout_country`; every other row is byte-identical, so round-3 `test_heldout` stays as it was |
| Eligible country | at least 30 eval rows of each class; at most 50% of the train rows (holding out more would leave too little to train on; meant to keep the US in); not entirely round-3 `test_heldout` |
| Local fractions | 10, 25, 50 and 100% of the pool, nested, grouped by panorama with the 3 m merge of the split, seed 42; a fixed 20% of the pool is the local `val` |
| Fine-tune | from the global model's best checkpoint, 10 epochs, local rows only (no replay) |
| Region conditioning | a 16-dimensional country embedding concatenated to the pooled image features; unknown row trained by region dropout 0.15 |
| Seeds | 42 for every arm; 43 and 44 for global, cond and condall only |
| Country source | `country_id` of `/v3/api/cities` (saved with the run), via `regions.py`; the world region of a country with only private deployments comes from a private override file (`REGION_OVERRIDES`), never from the repository |

Minimum-n rules: a LOCO cell exists at 30 or more eval rows of each class. An AUC interval needs 8 or more of the
smaller class. The within-city AUC uses deployments with 10 or more labels of both classes. A gate precision is
shown at 50 or more gated votes. A country is judged in the summary only with a paired interval and at least 100
eval rows; otherwise its row says "n too small". A fine-tune fraction needs 20 or more train rows of each class.

## Method

**Arms.** Every arm is scored on the same rows: the held-out country's former val and test rows.

- **(a) global:** the round-3 recipe trained on the LOCO split, so it never sees the country.
- **(b) ft:** arm (a)'s best checkpoint fine-tuned on a nested fraction of the country's own former train rows
  (`fine_tune_classifier.py --init-from`). The curve shows how many local labels pay off. The AUC on `test` after
  fine-tuning shows how much it forgets.
- **(c) cond:** a region-conditioned model (`region_model.py`) trained on the LOCO split. The held-out country is
  outside its vocabulary, so it is scored as the unknown region, exactly as a new deployment would be served.
- **(d) per-country calibration:** arm (a)'s predictions with no retraining. The pooled map is fitted on the other
  countries' val rows. A local map is fitted on the same nested fractions as arm (b), so (b) and (d) share a curve
  axis. In distribution, a per-country map fitted on that country's round-3 val rows is compared with the pooled map
  on its test rows.
- **condall:** the region-conditioned model on the round-3 split itself, at the country and at the region level,
  against the round-3 baseline of the same seed. Scoring the same checkpoint with every row as unknown is the
  embedding ablation.

**Statistics.** AUC of the incorrect score. Every Δ is paired: both models on the same rows, with a 95% interval from
1,000 resamples of whole panoramas (`analysis/slices.py`). The within-city AUC is the round-3 definition
(`retrain_report.within_city_auc`): per-deployment AUCs weighted by n_incorrect x n_correct, so a score that only
separates deployments cannot raise it. Recall at precision 0.92 is reported for both verdicts. The gate statistic
is the raw precision of the gated votes under the eval rows' own class mix, with a 95% Wilson interval (the #52
statistic, as in `calibration/per_city.py`); the precision under the file prior is shown beside it. There is no
per-country prior, so every map is fitted under the file prior. Severity x tag slices compare each arm with the
same-seed global model, with training rows counted from the arm's own split.

**Naming.** A country is named only if at least one of its deployments is public at report time. A country whose
deployments are all private is "private country A", "B", and so on, the rule `publish_to_hf.py` enforces for model
cards. No deployment is named in this study; the report checks every committed string against the non-public names.

## The LOCO table

<!-- BEGIN GENERATED: loco_table (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: loco_table -->

## Results

### Leave one country out: global vs region-conditioned, zero-shot

<!-- BEGIN GENERATED: results_loco (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: results_loco -->

### Local data: fine-tuning and local calibration by fraction

<!-- BEGIN GENERATED: results_curves (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: results_curves -->

### Region conditioning in distribution

<!-- BEGIN GENERATED: results_conditioned (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: results_conditioned -->

### Calibration only (arm d)

<!-- BEGIN GENERATED: results_calibration (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: results_calibration -->

Severity and tag slices of every arm are in [`slices.md`](region-transfer-2026-11/slices.md); the seed replicates
in [`seeds.md`](region-transfer-2026-11/seeds.md).

## What this does not establish

- Where a country has one deployment (the Netherlands is Amsterdam, for example), "country" and "deployment"
  coincide. A drop there cannot be pinned on the country.
- The class mix differs widely between deployments (2% to 80% incorrect). Pooled AUC across deployments mixes that
  in, which is why the within-city AUC is reported next to it.
- No per-country deployment prior exists. The gate precision under the file prior is only a proxy for what a
  country would see; the raw precision under its own mix is the number to read.
- The conditioned model has no serving path yet. sidewalk-ai-api sends no region, and `evaluate.py` and
  `publish_to_hf.py` cannot load it by design.

## Serving implication for sidewalk-ai-api

Two ways to serve a regional effect, if one is found: per-region entries in the API's `MODEL_MAP`, or one model with
a region input. The decision is deferred to the results. A conditioned model that beats the global one in
distribution but not zero-shot would argue for a region input with the unknown row as the fallback. A fine-tune
curve that pays off early would argue for per-region models.

## Reproduce

On klone, from a checkout of this branch with the conda env active:

```bash
DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/region-2026-11 \
  CITIES_JSON=$DATA/rawlabels/cities.json SEEDS=42 \
  analysis/region_submit.sh CurbRamp Crosswalk NoCurbRamp Obstacle SurfaceProblem
```

Then, on the Mac, from a pulled copy of `$DATA` (per-crop files stay outside the repository):

```bash
python analysis/region_transfer.py report --data-dir <copy> --cities-json <copy>/rawlabels/cities.json \
    --seeds 42 --jobs-file region_jobs.txt
python analysis/region_transfer.py report --check
```

## Provenance

<!-- BEGIN GENERATED: provenance (analysis/region_transfer.py) -->

_Not yet run: the report stage fills this block._

<!-- END GENERATED: provenance -->
