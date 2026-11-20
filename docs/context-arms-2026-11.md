# More context for the validator: two-scale crops, whole view with a point heatmap, side information (November 2026)

Issue [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14). Four ways (three arms and one ablation) to show the round-3
DINOv2-L validator more than its tight, depth-sized crop, each trained with the round-3 recipe on the round-3 split
and scored on the same rows as the round-3 model.

## Status

**Smoke-tested on klone 2026-11-19; pilot submitted (running, job IDs below); the two image arms are not yet run.**
No number in this document is a result of the arms. The image arms (`two_scale`, `view_heatmap`) need their
context images re-rendered from the panoramas, and the round-3 panorama store on klone is gone: about 450 GB
(39,240 panoramas for the three target types) has to be restaged first (see [Launch checklist](#launch-checklist)).

| stage | what | jobs | state |
|---|---|---|---|
| smoke | NoCurbRamp on the 217 round-3 rows whose panorama is in the one surviving staged store; 1 epoch; every arm and the baseline rerun | 43403247 (backfill), 43403248 to 43403253 (six trainings), 43403800 (a resubmit of a finished run) | all completed, 2026-11-19 17:20 to 17:48 PST: `check-run` ok for all six tags, every checkpoint deleted after its record was copied, the backfill's `--verify 50` passed (below), the resubmit exited with "already evaluated" in 28 s, and `report` on the smoke shadow ran to the end with every row `INCOMPLETE` (subset shadow) |
| pilot | target types x (`baseline-rerun` seeds 42 to 46, `side_info` and `side_info_nohq` at seed 42); full round-3 recipe | 43404266, 43404267, 43404269 to 43404287 (21 jobs, ids in `context_jobs_pilot.txt`) | running (submitted 2026-11-19 17:50 PST); no number from it yet |
| A | confirmatory arms x target types x seeds 42 to 46 (`A5=1`, recommended; 42 to 44 without it), `view_heatmap-r448` at seed 42, 3 backfills | 36 (24) | needs the restage and Jon's go |
| B | only for an arm with a `WIN` in stage A: that arm and `baseline-rerun` on CurbRamp and Crosswalk | | needs a `WIN` |

What exists:

- `context_crops.py`: cuts the two context images (a 4x wider crop and the whole view at 448 px) from the same
  label-centred view the training crop comes from; `context_crops.py backfill` writes them for an existing dataset;
  `data_generation.py --context-crops` writes them for a new one.
- `context_model.py`: the arm models, their input transforms, and `predict` (writes `predictions_<tag>.csv` for
  `evaluate.py --skip-inference --tag`). It refuses any base checkpoint that is not DINOv2
  ([Rerunning an arm on another backbone](#rerunning-an-arm-on-another-backbone)).
- `fine_tune_classifier.py --context-arm two_scale|view_heatmap|side_info|side_info_nohq
  [--context-view-res 224|448] [--batch-size N]`.
- `analysis/context_arms.py`: `shadow` and `preflight` (login node, no torch), `check-run`, `power` and `report`
  (the pre-registered decision); `analysis/context_arms.slurm` (one job per backfill or training);
  `analysis/context_submit.sh` (the stage launcher).

## Why

The validator sees a square crop whose half-width is `6100 / depth` px of the 2048 px view (median about 238 px,
[#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51)). For some label types the question a
human validator answers is about the surroundings, not the crop: is this obstacle actually in the walking path,
is there a curb ramp a metre to the left of this NoCurbRamp, is this surface problem on the sidewalk or on the
road. Both earlier validators used more than one crop: Weld et al. (ASSETS 2019,
https://doi.org/10.1145/3308561.3353798) fed non-image features alongside the crop, and Duan et al. (ASSETS 2022,
https://doi.org/10.1145/3517428.3550381, section 4) argued that fixed crops cut off the context that missing curb
ramp and obstacle judgments need.

The types this targets are the three weakest in round 3 on the deployments never seen in training
([`retrain-2026-10.md`](retrain-2026-10.md)):

| type | test AUC, seed 42 | unseen-deployment AUC, seed 42 | within-city AUC unseen, seed 42 | within-city AUC unseen, mean ± sd over seeds 42/43/44 |
|---|---:|---:|---:|---:|
| NoCurbRamp | 0.934 | 0.881 | 0.861 | 0.885 ± 0.024 |
| Obstacle | 0.883 | 0.868 | 0.788 | 0.782 ± 0.009 |
| SurfaceProblem | 0.896 | 0.710 | 0.682 | 0.706 ± 0.021 |

The last column is from [`confident-learning-2026-10/seeds.md`](confident-learning-2026-10/seeds.md). Seed 42 is
the low draw for NoCurbRamp and SurfaceProblem, which is one reason the comparison below is against a seed mean,
not against the round-3 model.

CurbRamp and Crosswalk are the guard types: they are trained only for an arm that wins on a target type (stage B),
to check it does no harm there.

## The arms

| arm | input | head | what it tests |
|---|---|---|---|
| `two_scale` | the tight crop and a crop 4x wider around the same point; one shared backbone, `[CLS, mean(patches)]` of each, concatenated | 4 x 1024 | does nearby context ("is it blocking the path") help? |
| `view_heatmap` | the whole 90-degree label-centred view, downsampled (stored at 448 px, trained at 224 by default), plus a 4th channel: a Gaussian at the label point | 2 x 1024 | RampNet-style point conditioning on the full view |
| `side_info` | the tight crop plus 9 normalised non-image features from `split.csv` | 2 x 1024 + 9 | Weld et al.-style side information |
| `side_info_nohq` | `side_info` without `high_quality_user` (8 features) | 2 x 1024 + 8 | the ablation: how much of a `side_info` gain is the labeler's own validation record (see Threats) |
| `baseline-rerun` | the tight crop, `--context-arm none`, trained in this chain on the same code and environment as the arms, seeds 42 to 46 | 1024 (stock) | the baseline every arm is compared with |
| `baseline` | October's round-3 model (seed 42) and its seed-43/44 replicates, as trained | | the drift row only; not retrained |

Every arm pools like `Dinov2ForImageClassification` and trains with the round-3 recipe: DINOv2-L, 25 epochs,
learning rate 1e-5, warm-up 10%, effective batch 32, hard targets, best checkpoint by validation macro-F1.
The models are never registered with `AutoModel` (`model_type` `context_dinov2`), so `evaluate.py` and
sidewalk-ai-api cannot load one by accident, and `--context-arm` refuses to run without `--output-dir`, so an arm
never lands in `<Type>/dinov2/` where `evaluate.py` and `publish_to_hf.py` look by default.

**Decisions** (the plan asked for these to be recorded):

- **Wide factor 4**, the issue's number, a parameter (`--factor` / `--context-wide-factor`). At 4x the wide crop is
  capped at the whole view (half-width 1024) whenever the tight half-width exceeds 256 px, which is roughly half of
  the labels, so for those labels `two_scale` sees the tight crop and the view. This is reported, not hidden:
  `clipping.md` gives the share per type from the backfilled index. If more than about 60% are capped, a 2.5x
  second pass is the next step. At evaluation the wide crop goes through the processor's resize to 256 and centre
  crop to 224, which drops the outer 12.5% on each side, so the model sees about 3.5x the tight half-width, not 4x
  (the tight crop loses the same share, so the ratio between the two streams stays 4).
- **Stored view 448 px, trained at 224 by default.** 448 is twice the model input (32 x 32 DINOv2 patches), so a
  448 run needs no upsampling. 224 keeps the token count and memory of the baseline (the memory-fair comparison);
  448 (`--context-view-res 448`, `TAG=view_heatmap-r448`) is the detail-fair one at 4x the tokens and about 3x the
  step time, planned for the types where 224 comes close. 2048 to 224 is a 9x reduction, so the 224 view loses
  detail the tight crop has.
- **The heatmap carries scale as well as position.** At evaluation the label is always at the view centre, so a
  fixed-width Gaussian would be the same image for every label: a constant channel. Its sigma is half of the tight
  crop's half-width mapped into the input (`heatmap_sigma`: `crop_half_px * res / 2048 / 2`, at least 4 px; about
  13 px for the median label at 224), so the channel says where the label is and how large the depth-sized crop
  around it is. This departs from the issue's "a Gaussian at the label position"; the plain point is
  `heatmap_sigma` with `crop_half_px = 0` and a fixed floor.
- **The view skips the processor's centre crop.** The checkpoint's processor resizes to 256 and centre-crops 224,
  dropping the outer 12.5% on each side, which for this arm is exactly the context under test. The view is resized
  to `res` x `res`; in training it goes through `RandomResizedCrop(scale=(0.7, 1.0))` and a flip like the tight
  crop, and the heatmap is computed analytically for the transformed point and sigma (`map_point`, checked against
  torchvision's own `resized_crop` in the tests). The view is resampled bilinearly (torchvision's `resize` and
  `resized_crop`, antialiased), while the tight and wide crops go through the checkpoint's bicubic processor; the
  view arm is consistent within itself (train and eval both bilinear), so this is a difference between arms, not
  a train/eval mismatch.
- **4-channel patch embedding**: channels 0 to 2 are the pretrained filters, channel 3 starts at zero, so the model
  computes exactly the pretrained function at step 0 (tested) and learns to use the heatmap.
- **Side features** (`context_model.side_feature_frame`): severity, number of tags, imagery age in years (against
  the latest capture date in the train rows, saved in the checkpoint), log depth, high-quality user, clipped, plus
  missing-value flags for severity, age and depth. Missing values take the train median; every column is
  standardised on the train rows; the normaliser is saved in the checkpoint's `config.json` so inference applies
  the same transform.
- **`two_scale` augments the pair together**: in training the collate re-draws the tight crop's augmentation
  alongside the wide crop's, each with its own `RandomResizedCrop` but one shared horizontal flip, so the two
  streams are never mirrored against each other (they never are at evaluation; tested).

## Pre-registered design and decision rule (2026-11-19, proposed, awaiting Jon's confirmation)

Written before any arm was trained on the round-3 data. `analysis/context_arms.py report` applies it mechanically
and writes `decision.md` / `decision.json`; nothing below is changed after a result is seen without an entry under
[Deviations](#deviations).

**Data and recipe.** Round-3 dataset, split and recipe, unchanged (DINOv2-L, 25 epochs, lr 1e-5, effective batch 32,
hard targets, best checkpoint by validation macro-F1).

**Baseline.** `baseline-rerun`: the tight-crop model retrained in this chain on the same code and environment as the
arms, seeds 42 to 46. October's round-3 model and its seed-43/44 replicates are reported beside it as the drift row.
If the rerun's seed mean differs from October's by more than two of October's seed sds on the primary endpoint for
a type, that is stated at the top of the report, and the decision still uses the rerun.

**Arms.** Confirmatory: `two_scale` (4x), `view_heatmap` (224). Exploratory (reported; can never produce a `WIN`):
`view_heatmap-r448`, `side_info`, `side_info_nohq`.

**Types.** Target: NoCurbRamp, Obstacle, SurfaceProblem. Guard: CurbRamp, Crosswalk.

**Stages.**
- Pilot (submitted 2026-11-19): baseline reruns x 5 seeds and the two side arms at seed 42, target types.
- Stage A (needs the panorama restage): confirmatory arms x target types x seeds 42, 43, 44 (18 trainings),
  `view_heatmap-r448` at seed 42 (3 trainings), 3 backfills. If Jon takes the five-seed recommendation of the
  power table below (`A5=1`, seeds 42 to 46, 30 trainings), `seeds_confirmatory` becomes `[42, 43, 44, 45, 46]` here
  and in `DECISION_RULE` before stage A is launched.
- Stage B (only for an arm with a `WIN` in stage A): that arm and `baseline-rerun` x guard types x seeds 42, 43, 44,
  plus 2 backfills.
- No other run is added after looking at results without an entry under Deviations.

**Primary endpoint.** `D` = mean over the arm's seeds minus mean over the baseline's seeds of the **within-city AUC
on `test_heldout`**. Its 95% interval is `D ± t · se`, with
`se² = s_arm²/n_arm + s_base²/n_base + var_boot(D)`: the sample variance across seeds of each group, and the
variance of `D` over a joint bootstrap of panoramas within deployment applied to every run (the same resample for
every run; a deployment whose resample loses a class is redrawn up to 20 times, then the resample is dropped, and
the report counts both). `t` is the Welch t quantile for the two seed terms. Five baseline seeds against three arm
seeds is unbalanced on purpose (the baseline is shared by every arm); the Welch form handles it.

**Verdict per type x confirmatory arm.** `WIN` when all of:
1. `D >= +0.02`;
2. the interval's lower bound is above 0;
3. every arm seed is above the baseline's seed mean;
4. the seed-mean difference is positive in at least 2 of the 3 held-out deployments (those meeting the minimum-n
   rule; if fewer than 3 qualify, in all that qualify);
5. no harm in distribution: the lower bound of the same interval for the within-city AUC on `test` is above
   **-0.02**.

`REGRESSION` when the upper bound of the interval for `D` is below 0, or the upper bound of the `test` interval is
below -0.01. `INCOMPLETE` when a required run, record or gate block is missing, or the data is a subset shadow.
Otherwise `NO GAIN`, worded in every output as "no gain detected at this power", next to the power table.

**The gate is reported, not a criterion.** Agree and uncapped Disagree coverage, precision and gated n at 0.92 under
the **majority2** deployment prior ([`decisions.md`](decisions.md), 2026-11-12), with the margin prior beside it,
the confidence where each map first clears and the share of verdicts at exactly 1.0 (`gate.md`). Reason: for
Obstacle 46% of test Agree verdicts sit at exactly 1.0, so coverage at the gate tracks float saturation of a new
head, not discrimination. Any proposal to ship an arm must show this table, and Jon reads it then.

**What each outcome triggers.**
- `WIN` on a target type, and after stage B no `REGRESSION` for that arm on the guard types: the arm is **proposed
  for that type** (types are separate models). A `REGRESSION` on one target type does not veto another type. Next
  steps for a proposed arm: rerun on the [#119](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/119)
  backbone and the round-4 recipe, then serving parity, before any serving work.
- Both arms win on a type: prefer `view_heatmap` (one backbone pass) unless `two_scale`'s `D` is larger by at least
  0.01.
- No `WIN` anywhere: [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14) is closed as "no gain
  detected under the round-3 recipe at the stated power", and the arms are not carried into round 4. An exploratory
  arm that looks good earns a new pre-registered confirmatory run, not a verdict.

**Always reported** (per type x arm, beside the baseline rerun and October's round 3): test and held-out AUC;
within-city AUC on both with the interval above; the same-seed table (description only); per held-out deployment
(private ones lettered); macro F1; the gate block; type x severity and type x tag slices with training rows and the
thin flag, per seed and seed mean (`slices_seeds.md`, with October's round-3 AUC beside); clipping share (image
arms); analytic cost and measured training throughput.

**Known limits, stated up front.** Six confirmatory tests with no multiplicity correction; criteria 1, 3 and 4 are
the protection, and the simulated false-`WIN` rate is printed. With three seeds the power is low (table below).
Three held-out deployments, two private, 138 to 1,812 labels per type: the interval covers label sampling and
training seed, not "which deployments were held out". The held-out split is not a fresh test set. The Tier-A gold
set cannot score these models.

Constants (`DECISION_RULE` in `analysis/context_arms.py` must equal this block; a test compares them):

```json
{"all_seeds_above_baseline_mean": true, "baseline_variant": "baseline-rerun", "gate": "report_only", "level": 0.95, "min_delta": 0.02, "min_positive_deployments": 2, "prior_rule": "majority2", "regression_test_upper": -0.01, "seeds_baseline": [42, 43, 44, 45, 46], "seeds_confirmatory": [42, 43, 44], "test_no_harm_lower": -0.02}
```

### Operating characteristics

From `analysis/context_arms.py power` (simulation of criteria 1 to 3 and the held-out `REGRESSION` test, 4,000 draws;
criteria 4 and 5 are not simulated, so the `WIN` rates are upper bounds). Inputs: the within-city held-out seed sd
of October's trio and the bootstrap sd of a difference between two of its seeds (1,000 joint panorama resamples).
The pilot's five baseline seeds replace October's sds when they finish; the report recomputes the table then.

| type | seed sd | bootstrap sd of D | arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | effect at 80% power | recommended arm seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCurbRamp | 0.024 | 0.012 | 3 | 0.011 | 0.010 | 0.07 | 0.14 | 0.38 | +0.085 | 5 |
| NoCurbRamp | 0.024 | 0.012 | 5 | 0.012 | 0.016 | 0.10 | 0.22 | 0.58 | +0.065 |  |
| Obstacle | 0.009 | 0.010 | 3 | 0.005 | 0.005 | 0.16 | 0.41 | 0.87 | +0.050 | 5 |
| Obstacle | 0.009 | 0.010 | 5 | 0.009 | 0.009 | 0.26 | 0.59 | 0.97 | +0.040 |  |
| SurfaceProblem | 0.021 | 0.015 | 3 | 0.005 | 0.007 | 0.06 | 0.12 | 0.37 | +0.085 | 5 |
| SurfaceProblem | 0.021 | 0.015 | 5 | 0.010 | 0.013 | 0.09 | 0.19 | 0.55 | +0.070 |  |

Klone job 43403551 (2026-11-19), on the full shadow of the round-3 data: `analysis/context_arms.py power
--data-dir $WORK/data --variant baseline`, bootstrap pair `baseline-s43` minus `baseline-s44`, seed 0.

**What this means.** The rule is strict, and three seeds are not enough for the effects this experiment could
plausibly find. At three arm seeds a real +0.03 is detected 12% to 41% of the time; at five, 19% to 59%. The
smallest effect detected with 80% power is +0.065 to +0.085 for NoCurbRamp and SurfaceProblem, and +0.04 to +0.05 for
Obstacle. The false-`WIN` rate stays at or under about 1% in every row. Following the sign-off rule (five arm seeds
for any type whose power at +0.03 with three seeds is below 0.5), the **recommended default for stage A is five arm
seeds (`A5=1`) for all three target types**: 30 trainings instead of 18. Even then a `NO GAIN` rules out only
large gains, which is why every output words it as "no gain detected at this power". The rule itself is not
changed; whether to accept this power, or to relax a criterion before any arm is trained, is Jon's call.

### Deviations

None.

## Setup

| | |
|---|---|
| Data | the round-3 dataset on klone (`retrain-2026-10/<Type>/`), **read only**. Every job reads and writes a shadow of it on `/gscratch/scrubbed` instead (below) |
| Split | the round-3 `split.csv` (panorama-grouped, three deployments held out), byte-identical in the shadow; every run trains on its train rows and is scored on its val/test/test_heldout rows |
| Recipe | round 3 (above); `--batch-size` only moves the split between per-device batch and gradient accumulation, effective batch 32 either way |
| Seeds | `baseline-rerun` 42 to 46 (pilot); confirmatory arms 42, 43, 44 (stage A; 42 to 46 with `A5=1`); exploratory arms 42 |
| Evaluation | the baseline rerun: `evaluate.py --checkpoint <best> --tag <tag> --per-city`; an arm: `context_model.py predict`, then `evaluate.py --skip-inference --tag <tag> --per-city`; the decision by `analysis/context_arms.py report` |
| Missing side features | share of rows with no severity / no imagery date / no depth, from `preflight`: NoCurbRamp 5.5% / 0.0% / 0.0% (18,162 rows); Obstacle 6.2% / 0.0% / 0.0% (16,894); SurfaceProblem 5.9% / 0.0% / 0.0% (15,852) |
| Storage | the shadow and checkpoints on `/gscratch/scrubbed/$USER/context-arms-launch-2026-11/` (purged after 21 days unread); small durable artifacts (each run's records, metrics, predictions) on `/gscratch/makelab/$USER/context-arms-launch-2026-11-keep/` |

**The shadow data directory.** `evaluate.py`, `context_model.py predict` and the backfill write next to the split
they read. So that nothing is written into the round-3 directory, `analysis/context_arms.py shadow` builds a
directory on scratch whose `<Type>/split.csv`, `manifest.csv` and `generation_run.json` are byte-identical copies
(the split hash recorded by every run equals the round-3 baseline's), whose `correct/` and `incorrect/` are symlinks
to the round-3 crops, and which holds copies of October's outputs (`metrics.json`, `predictions.csv`, the
seed-43/44 files) for the drift row. `shadow_run.json` records the source path, the sha256 of every copied file, the
rawLabels snapshot the backfill must use and the git commit. `--only-staged STORE` keeps only the rows whose panorama
is in a staged store (the smoke shadow) and marks the shadow a subset, which the report never decides on.
`analysis/context_submit.sh` refuses a `DATA` without `shadow_run.json`.

**Each run's record.** Every training job copies `training_run.json`, `run_key.json`, the best checkpoint's
`trainer_state.json` and `config.json` and a `job.txt` (job id, node, GPU, start and end times) to
`<shadow>/<Type>/runs/<tag>/` before deleting the checkpoint (`KEEP_CKPT=0`, the default; `best` keeps the best one),
and copies the record, metrics and predictions to `$KEEP`. `check-run` verifies a finished run (record present,
seed, split hash, predictions covering exactly the shadow's val/test/test_heldout rows, one per `label_uid`), and
the report marks a decision `INCOMPLETE` when a record is missing or does not match. October's seed-42 model
predates run records; its `training_run.json` and `run_key.json`, and those of its seed-43/44 replicates, were pinned
on 2026-11-19 under `$KEEP/baseline-records/<Type>/` with a `SHA256SUMS` file (57 files, 1.7 MB).

## Launch checklist

On klone, in a clone of this branch on scratch (the jobs run the code of the clone they were submitted from; do not
pull into it while jobs are queued). Login-node steps import pandas only; never import torch or transformers on a
login node (see [`running-on-hyak.md`](running-on-hyak.md)).

```
CLONE=/gscratch/scrubbed/$USER/sva-context-arms-launch
WORK=/gscratch/scrubbed/$USER/context-arms-launch-2026-11
KEEP=/gscratch/makelab/$USER/context-arms-launch-2026-11-keep
SRC=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10
ENV=/gscratch/makelab/$USER/envs/sidewalk-validator-ai
cd $CLONE

# 1. shadows (seconds): the full one for the pilot and stage A, and the smoke subset
$ENV/bin/python analysis/context_arms.py shadow --src $SRC --dest $WORK/data --types NoCurbRamp,Obstacle,SurfaceProblem
$ENV/bin/python analysis/context_arms.py shadow --src $SRC --dest $WORK/smoke/data --types NoCurbRamp \
    --only-staged /gscratch/scrubbed/$USER/pano-stage/NoSidewalk

# 2. smoke (7 jobs, 1 epoch): every code path once; the launcher runs the preflight first
STAGE=smoke DATA=$WORK/smoke/data RUNS=$WORK/smoke/runs PANO_STORE=/gscratch/scrubbed/$USER/pano-stage/NoSidewalk \
    analysis/context_submit.sh NoCurbRamp
$ENV/bin/python analysis/context_arms.py check-run --data $WORK/smoke/data --label-type NoCurbRamp --seed-from-tag \
    --tags baseline-rerun,two_scale,view_heatmap,view_heatmap-r448,side_info,side_info_nohq

# 3. pilot (21 jobs, no panoramas needed)
STAGE=pilot DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem

# 4. stage A: first the restage (Jon's decision; below), then
STAGE=A DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP PANO_STORE=<staged store> DRY_RUN=1 \
    analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem      # read it, then again without DRY_RUN

# 5. the decision (no GPU; anywhere with the shadow)
$ENV/bin/python analysis/context_arms.py report --data-dir $WORK/data --label-types NoCurbRamp,Obstacle,SurfaceProblem \
    --out docs/context-arms-2026-11
```

Every invocation of the launcher appends its job ids and each job's full command to `context_jobs_<stage>.txt` in
the clone, refuses more than 40 jobs per invocation (`MAX_JOBS`) and more than 40 summed over every
`context_jobs_*.txt` (`SESSION_MAX_JOBS`; `ALLOW_OVER_SESSION_CAP=1` overrides deliberately), stops at the first
`sbatch` that fails without retrying, and runs `preflight` before any `sbatch`. Single jobs:

```
STEP=train LT=NoCurbRamp ARM=side_info_nohq SEED=43 TAG=side_info_nohq-s43 DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP \
    sbatch analysis/context_arms.slurm
STEP=backfill LT=NoCurbRamp DATA=$WORK/data PANO_STORE=<store> sbatch analysis/context_arms.slurm
```

`STEP=train` refuses an empty `TAG` and the tags `baseline` and `baseline-s<seed>` (October's files), and a run whose
`<shadow>/<Type>/runs/<tag>/.evaluated` exists exits at once ("already evaluated"), so a resubmitted job never
overwrites a finished one.

**The panorama restage (stage A).** The image arms' backfill re-renders each row's view from its panorama, and the
round-3 store on klone is gone. `preflight --backfill` writes the missing list as `<shadow>/<Type>/stage_files.txt`
(the format `analysis/ai_labels_stage.slurm` reads) with a size estimate:

| type | unique panoramas | at 11.4 MB each |
|---|---:|---:|
| NoCurbRamp | 13,666 | about 156 GB |
| Obstacle | 13,971 | about 159 GB |
| SurfaceProblem | 13,630 | about 155 GB |
| three target types, deduplicated | 39,240 | about 450 GB |

```
cat $WORK/data/{NoCurbRamp,Obstacle,SurfaceProblem}/stage_files.txt | sort -u > $WORK/stage/stage_files.txt
RUN=$WORK/stage DEST=/gscratch/scrubbed/$USER/pano-stage/context-arms sbatch analysis/ai_labels_stage.slurm
```

The stage job pulls sequential 1,000-file batches over one connection each and stops on the first connection error
without retrying (the makelab hosts ban a source that reconnects too fast); a resubmit resumes. The backfill then
runs per type inside stage A. Stage A should be launched within the scratch purge window of the shadow; if the shadow
has been purged, rebuilding it takes seconds, but the context images need the panoramas again.

## Cost

| stage | trainings | backfills | GPU-hours (estimate) |
|---|---:|---:|---:|
| smoke | 6 (1 epoch, 217 rows) | 1 | under 1 |
| pilot | 21 | 0 | about 21 x 2.5 = 50 |
| A | 21 (33 with `A5=1`) | 3 | about 70 (110), `two_scale` and `view_heatmap-r448` at about 2x and 3x the step time |
| B | 6 per guard type | 1 per guard type | about 20 |

Estimates from October's tight-crop runs (0:10 to 2:41 h per run on `ckpt-all`, NoCurbRamp 2:41) times the arm's
backbone passes and tokens. All on klone's free `ckpt-all` partition. Measured smoke throughput (1 epoch on 217
rows, so dominated by start-up; read it as a lower bound on the full-run cost per step):

| run (smoke, NoCurbRamp, 1 epoch, 135 train rows) | GPU | seconds per step (5 steps of 32) | train_runtime (s) | peak GPU memory |
|---|---|---:|---:|---:|
| `baseline-rerun` | A40 | 6.4 | 105 | 9.2 GB |
| `side_info` | A100 80 GB | 6.8 | 126 | 9.3 GB |
| `side_info_nohq` | L40 | 5.8 | 117 | 9.3 GB |
| `two_scale` | A40 | 5.6 | 113 | 13.2 GB |
| `view_heatmap` (224) | A40 | 5.0 | 103 | 9.1 GB |
| `view_heatmap-r448` | A100 80 GB | 10.0 | 173 | 20.2 GB |

Peak memory is Slurm's `gres/gpumem` (`sacct TRESUsageInMax`), at batch 8 per device. Five steps on a cold cache
are dominated by reading crops from GPFS (GPU utilisation 4% to 37%), so the step times do not show the arms'
compute ratio; the pilot's full runs measure the steady state of the tight-crop and side arms, and stage A's
`TIME` for `view_heatmap-r448` should be set from those. What the smoke does show: every arm fits at batch 8 on a
48 GB card with room to spare, `two_scale` included (the threat about running it at batch 4 does not arise).

**Backfill.** 217 rows in 268 s on an A40 (about 1.2 s per row, mostly reading the 11 MB panoramas from scratch);
at that rate a full type takes about 5 to 6 hours. `--verify 50`: re-cut tight crops against the stored round-3
WebPs, mean absolute difference 1.33 grey levels (worst crop 2.16; limit 12), max pixel difference 39. The smoke
read panoramas from a store staged separately from the round-3 one (for the NoSidewalk study), so a mismatch could
have come from source-image drift as well as from the renderer; none was seen.

## Results

Not yet run. The report writes, under [`context-arms-2026-11/`](context-arms-2026-11/):

- `decision.md` / `decision.json`: the verdict per type x confirmatory arm with every criterion and its value, the
  exploratory arms' intervals (never a verdict), the drift row, the operating characteristics recomputed from the
  baseline rerun's seeds, and the bootstrap's redraw and drop counts.
- `gate.md`: the report-only gate block per run.
- `slices_seeds.md`: type x severity and type x tag across seeds, with training rows, the thin flag and October's
  round-3 AUC beside.
- `arms.md` / `arms.json`, `seeds.md`, `slices.md`: per-run tables against October's seed-42 model (descriptive).
- `clipping.md`: the share of wide crops capped at the whole view, per type, and `wide_half_px` quantiles.
- `provenance.json`: sha256 of every file read, the arguments, the bootstrap seed, the git commit, the decision
  rule, the shares and shadow hashes, and each run's record status.

| type | arm | verdict | D within-city held-out [95% CI] |
|---|---|---|---:|
| NoCurbRamp | two_scale / view_heatmap | not yet run | not yet run |
| Obstacle | two_scale / view_heatmap | not yet run | not yet run |
| SurfaceProblem | two_scale / view_heatmap | not yet run | not yet run |
| target types | side_info / side_info_nohq (exploratory) | running (pilot) | running |

## What could ship

- **`two_scale`**: sidewalk-ai-api already renders the 2048 px view and computes the crop half-width; it would cut
  one more `centre_crop` and run the backbone twice, roughly doubling validator latency. Deployable with a small API
  change and a new model loader.
- **`view_heatmap`**: the API has the view and the depth; it would downsample the view and build the heatmap from
  the crop half-width. One backbone pass, but a 4-channel model is not loadable with
  `AutoModelForImageClassification`, so the API needs a loader for it.
- **`side_info`**: the `/process` request carries `label_type`, `panorama_id`, `x`, `y`, the camera pose fields,
  `coord_frame` and `city`. Depth and clipping the API computes itself, but severity, tags, the labeler's
  high-quality flag and the imagery capture date are not in the request today; SidewalkWebpage would have to send
  them. Least deployable; read its result as an upper bound on what side information adds, not as a candidate.

Any arm that wins needs its own serving-parity stage
([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5),
[`serving-parity-2026-10.md`](serving-parity-2026-10.md)) before a deploy. `publish_to_hf.py` copies checkpoint
files without loading them, so it does not refuse an arm checkpoint pointed at explicitly; a guard on
`training_run.json`'s `context_arm` (as it already refuses `fixed_crop_half_px`) belongs with any proposal to ship
one.

## Rerunning an arm on another backbone

The arms are written for DINOv2 and `context_model.from_base` refuses any other base checkpoint
(`check_backbone`: the base's `config.json` `model_type` must be `dinov2`), so an arm can never be silently built
on a backbone it was not written for. What a port to another backbone (for example the one chosen in
[#119](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/119)) has to supply:

- **Pooled feature.** The head reads `[CLS, mean(patch tokens)]`, so the port needs the number of prefix tokens
  (CLS plus any register tokens) to skip before the patch mean, or the backbone's own pooled output in their place.
- **The fourth channel** (`view_heatmap`): the first projection's weight (the patch embedding's convolution, or a
  ConvNet's stem) gets one extra input channel initialised to zero, and the step-0 identity test (the 4-channel model
  computes exactly the pretrained function before training) must pass on the new backbone.
- **The processor**: its mean, std and input size (the view arm bypasses the centre crop and resizes to `res`), and
  the patch size (448 = 2 x 224 is chosen to be a whole number of DINOv2 patches).
- **The baseline** for such a rerun is that backbone's own tight-crop run on the same recipe, never the DINOv2-L
  numbers here.

The backbone name recorded in each arm's `config.json` (`base_checkpoint`) is what a later reader checks.

## Threats to validity

- **Seed noise.** The baseline's within-city held-out AUC varies across seeds 42/43/44 with a sample sd of 0.024
  (NoCurbRamp), 0.009 (Obstacle) and 0.021 (SurfaceProblem)
  ([`confident-learning-2026-10/seeds.md`](confident-learning-2026-10/seeds.md)). The decision therefore compares
  seed means and puts the seed variance into the interval; a single-seed Δ of that size is not a finding.
- **Seed 42 is the low draw** of October's trio for NoCurbRamp and SurfaceProblem; a comparison against October's
  seed-42 model alone would flatter any arm. The decision uses the five-seed rerun.
- **The held-out split has been read** by earlier studies (round 3, confident learning, the deployment gate), so it
  is not a fresh test set; the pre-registration limits what is chosen after seeing it, it does not make it fresh.
- **Gold rows in train.** The round-3 split's train rows include 131 items decided in the masked re-label rounds, so
  these models cannot be scored on the Tier-A gold set.
- **The wide crop is often the whole view** at 4x (see the decisions above), so `two_scale` partly tests "tight
  crop plus whole view at 224", not a graded context ring.
- **Detail at 224.** The view arm sees the scene at 1/9 of the rendered resolution; a null result at 224 does not
  rule out the arm, which is what the 448 run is for.
- **Batch size.** Every arm ran at batch 8 per device in the smoke (`two_scale` peaked at 13.2 GB, `view_heatmap-r448`
  at 20.2 GB), so no arm needs a smaller per-device batch than the baseline.
- **Backfill fidelity.** The context images are re-rendered, not saved at generation time. `--verify` bounds the
  difference on the tight crop, and on the synthetic test store the hook and the backfill write byte-identical
  files; a GPU backfill of a numpy-rendered dataset (or the reverse) differs by rounding and prints a warning.
- **Side features are what the crowd saw**: severity, tags and the high-quality flag come from the labeler, so
  `side_info` can learn labeler habits that do not transfer to new deployments; the held-out deployments are the
  check.
- **Severity and tags are post-edit values.** rawLabels carries the label's current severity and tags, and
  validators edit them in the same screen where they cast the vote that sets the target: the 46 public labelEdits
  exports of 2026-11-05 hold 13,274 edits, almost all tag and severity edits
  ([`retype-2026-11.md`](retype-2026-11.md)). An edit can therefore carry the validator's judgment back into the
  features. The share of round-3 rows of the three target types with such an edit was not measured. Both side arms
  are upper bounds for this reason too.
- **`high_quality_user` is partly circular with the target.** The rawLabels flag is derived from the labeler's
  validation record, which includes the very validations that set this label's class, on the evaluation rows as
  much as on the train rows. Part of any `side_info` gain can therefore be the label's own votes leaking back in.
  The ablation `side_info_nohq` (the same arm with `high_quality_user` dropped; `--context-arm side_info_nohq`)
  runs beside it in the pilot, and both are reported.
