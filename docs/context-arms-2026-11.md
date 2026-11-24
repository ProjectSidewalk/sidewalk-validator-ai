# More context for the validator: two-scale crops, whole view with a point heatmap, side information (November 2026)

Issue [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14). Four ways (three arms and one ablation) to show the round-3
DINOv2-L validator more than its tight, depth-sized crop, each trained with the round-3 recipe on the round-3 split
and scored on the same rows as the round-3 model.

## Status

**Pilot and stage A done; the pre-registered decision was run once, on 2026-11-24 ([Results](#results)). No arm
wins on any target type: `two_scale` is `NO GAIN` on NoCurbRamp, Obstacle and SurfaceProblem, and `view_heatmap` is
`REGRESSION` on all three.** The panoramas were restaged on 2026-11-23 (39,231 from the lab store, 9 from an earlier
copy on klone; see [Provenance](#provenance)) and stage A ran on 2026-11-23/24 from commit aa8ef24.

**What the rule says to do next: stop.** Stage B runs only for an arm with a `WIN` in stage A, so it is not run: the
launcher's `WINNER` has no value to take, and no guard type (CurbRamp, Crosswalk) is trained. Under "What each outcome
triggers", [#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14) is to be closed as "no gain
detected under the round-3 recipe at the stated power", and the arms are not carried into round 4. The exploratory arms
(`view_heatmap-r448`, `side_info`, `side_info_nohq`) produce no verdict; under the rule, one that looks good earns a
new pre-registered confirmatory run, not a verdict, and whether to start one is Jon's call. Nothing has been launched.

| stage | what | jobs | state |
|---|---|---|---|
| smoke | NoCurbRamp on the 217 round-3 rows whose panorama is in the one surviving staged store; 1 epoch; every arm and the baseline rerun | 43403247 (backfill), 43403248 to 43403253 (six trainings), 43403800 (a resubmit of a finished run) | all completed, 2026-11-19 17:20 to 17:48 PST: `check-run` ok for all six tags, every checkpoint deleted after its record was copied, the backfill's `--verify 50` passed (below), the resubmit exited with "already evaluated" in 28 s. `report` on the smoke shadow was run before the three image-arm jobs had finished, so its image-arm rows read "not run"; rerun locally on all six runs it exits 0 with every row `INCOMPLETE` (subset shadow). The klone clone was fast-forwarded once while a smoke job ran (harmless here; the checklist says not to) |
| pilot | target types x (`baseline-rerun` seeds 42 to 46, `side_info` and `side_info_nohq` at seed 42); full round-3 recipe; **pilot code: 9dace0c** | 43404266, 43404267, 43404269 to 43404287 (21 jobs, ids and commands in `context_jobs_pilot.txt` in the scratch clone; copy it and `context_jobs_smoke.txt` to `$KEEP` when the pilot ends, scratch is purged) | all 21 completed (submitted 2026-11-19, first job started 17:49 PST per the job accounting; the last ended 2026-11-19 23:17 PST per the run records); scored by the one `report` run of 2026-11-24 together with stage A |
| A | confirmatory arms x target types x seeds 42 to 46 (`A5=1`; Jon's decision J5, 2026-11-20, before stage A was submitted; see [Deviations](#deviations)), `view_heatmap-r448` at seed 42, 3 backfills; **stage-A code: aa8ef24** | 43481236 to 43481271 (36 jobs: the 33 trainings' ids are in their records; the three ids no record names, 43481236, 43481248 and 43481260, are the backfills) | all 36 completed: launched 2026-11-23 with `A5=1` after the restage, first training started 15:10 PST, last ended 2026-11-24 04:46 PST (run records). Nine trainings were preempted and resumed from their last checkpoint, plus one in the pilot ([Provenance](#provenance)). `report` run once, 2026-11-24 |
| B | only for an arm with a `WIN` in stage A: that arm (the rule's confirmatory seeds, 42 to 46) and `baseline-rerun` (seeds 42 to 46) on CurbRamp and Crosswalk | 11 per guard type with a backfill | **not run: no `WIN` in stage A**, so the rule triggers no stage B |

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

## Pre-registered design and decision rule (2026-11-19; J5 decided 2026-11-20; the rest as proposed)

Written before any arm was trained on the round-3 data. `analysis/context_arms.py report` applies it mechanically
and writes `decision.md` / `decision.json`; nothing below is changed after a result is seen without an entry under
[Deviations](#deviations).

Edits to the proposal before any arm was trained (review fix round 1 and its follow-ups, 2026-11-19 PST; the
constants are unchanged): stage B trains the baseline at all five rule seeds; only the rule's seed lists are read; a
record must also match its tag's arm, view resolution, the recipe's epochs and the knobs the job file pins (best
metric, target, per-device batch); a verdict needs at least 200 kept bootstrap resamples.

One constant changed after the proposal, before stage A was submitted (no confirmatory arm had been trained beyond
the 1-epoch smoke), but after the pilot's trainings had finished: **five confirmatory seeds**, `seeds_confirmatory`
from `[42, 43, 44]` to `[42, 43, 44, 45, 46]`, Jon's decision J5 on 2026-11-20, following the power table below. The
timing is recorded under [Deviations](#deviations). The other constants are unchanged.

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
- Pilot (submitted 2026-11-19; trainings finished the same night; scored by the 2026-11-24 report): baseline reruns x
  5 seeds and the two side arms at seed 42, target types.
- Stage A (run 2026-11-23/24 after the panorama restage): confirmatory arms x target types x seeds 42 to 46 (30 trainings; Jon took
  the five-seed recommendation of the power table below, decision J5, 2026-11-20), `view_heatmap-r448` at seed 42
  (3 trainings), 3 backfills: 36 jobs, launched with `A5=1`.
- Stage B (only for an arm with a `WIN` in stage A): that arm at the rule's confirmatory seeds and `baseline-rerun`
  at seeds 42 to 46 (the rule's baseline seeds on every type; with three, a guard-type `REGRESSION` could never be
  read) x guard types, plus 2 backfills.
- Only the rule's seeds are read: `seeds_baseline` for the baseline, `seeds_confirmatory` for a confirmatory arm. A
  seed trained beyond those lists is listed in `decision.md` as "on disk, not used" and enters nothing, so a partly
  finished or extended stage cannot move a verdict. The launcher hands each stage preset's seed lists to `preflight`,
  which refuses a list the rule does not read (the rule reads five confirmatory seeds, so a stage-A or stage-B
  launch without `A5=1`, which trains three, is refused before any `sbatch`).
- No other run is added after looking at results without an entry under Deviations.

**Primary endpoint.** `D` = mean over the arm's seeds minus mean over the baseline's seeds of the **within-city AUC
on `test_heldout`**. Its 95% interval is `D ± t · se`, with
`se² = s_arm²/n_arm + s_base²/n_base + var_boot(D)`: the sample variance across seeds of each group, and the
variance of `D` over a joint bootstrap of panoramas within deployment applied to every run (the same resample for
every run; a deployment whose resample loses a class is redrawn up to 20 times, then the resample is dropped, and
the report counts both). `t` is the Welch t quantile for the two seed terms. Arm and baseline both have five seeds;
the Welch form is kept because the two groups' seed variances need not be equal.

**Verdict per type x confirmatory arm.** `WIN` when all of:
1. `D >= +0.02`;
2. the interval's lower bound is above 0;
3. every arm seed is above the baseline's seed mean;
4. the seed-mean difference is positive in at least 2 of the 3 held-out deployments (those meeting the minimum-n
   rule; if fewer than 3 qualify, in all that qualify);
5. no harm in distribution: the lower bound of the same interval for the within-city AUC on `test` is above
   **-0.02**.

`REGRESSION` when the upper bound of the interval for `D` is below 0, or the upper bound of the `test` interval is
below -0.01. `INCOMPLETE` when a required run, record or gate block is missing; when a run's record does not match
its tag (seed, split hash, context arm, view resolution, epochs other than the recipe's 25, or a best metric,
target or per-device batch other than the job file's `macro_f1`, hard and 8); when the joint
bootstrap kept fewer than 200 resamples on test or on held-out (without the bootstrap term the interval would be too
narrow; `report` refuses `--decision-boot` below 200); or when the data is a subset shadow.
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
the protection, and the simulated false-`WIN` rate is printed. Even with five arm seeds the power is low (table
below). Three held-out deployments, two private, 138 to 1,812 labels per type: the interval covers label sampling
and training seed, not "which deployments were held out". Five arm seeds narrow the interval, so the simulated null
rates are slightly higher than with three (table below: false `WIN` at 0 from 0.005 to 0.011 at three arm seeds to
0.009 to 0.012 at five; `REGRESSION` at 0 from 0.005 to 0.010 to 0.009 to 0.016). These are per-test rates,
and there are six uncorrected tests. The held-out split is not a fresh test set. The Tier-A gold set cannot score
these models.

Constants (`DECISION_RULE` in `analysis/context_arms.py` must equal this block; a test compares them):

```json
{"all_seeds_above_baseline_mean": true, "baseline_variant": "baseline-rerun", "gate": "report_only", "level": 0.95, "min_delta": 0.02, "min_positive_deployments": 2, "prior_rule": "majority2", "regression_test_upper": -0.01, "seeds_baseline": [42, 43, 44, 45, 46], "seeds_confirmatory": [42, 43, 44, 45, 46], "test_no_harm_lower": -0.02}
```

### Operating characteristics

From `analysis/context_arms.py power` (simulation of criteria 1 to 3 and the held-out `REGRESSION` test, 4,000 draws;
criteria 4 and 5 are not simulated, so the `WIN` rates are upper bounds). Inputs: the within-city held-out seed sd
of October's trio and the bootstrap sd of a difference between two of its seeds (1,000 joint panorama resamples).
The pilot's five baseline seeds replace October's sds in the table `report` prints; that table, from the 2026-11-24
run, is under [Results](#results).
The bootstrap sd of D comes from one seed pair at one bootstrap seed: at another bootstrap seed the review measured
0.0107 / 0.0137 against 0.0099 / 0.0151 (Obstacle / SurfaceProblem), which moves Obstacle's power at +0.03 with three
seeds from 0.41 to about 0.36. The conclusion below does not change; `power` now also prints the sd over every seed
pair.

| type | seed sd | bootstrap sd of D | arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | effect at 80% power | recommended arm seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCurbRamp | 0.024 | 0.012 | 3 | 0.011 | 0.010 | 0.07 | 0.14 | 0.38 | +0.085 | 5 |
| NoCurbRamp | 0.024 | 0.012 | 5 | 0.012 | 0.016 | 0.10 | 0.22 | 0.58 | +0.065 |  |
| Obstacle | 0.009 | 0.010 | 3 | 0.005 | 0.005 | 0.16 | 0.41 | 0.87 | +0.050 | 5 |
| Obstacle | 0.009 | 0.010 | 5 | 0.009 | 0.009 | 0.26 | 0.59 | 0.97 | +0.040 |  |
| SurfaceProblem | 0.021 | 0.015 | 3 | 0.005 | 0.007 | 0.06 | 0.12 | 0.37 | +0.085 | 5 |
| SurfaceProblem | 0.021 | 0.015 | 5 | 0.010 | 0.013 | 0.09 | 0.19 | 0.55 | +0.070 |  |

The rule reads 5 confirmatory arm seeds (`seeds_confirmatory`, J5), so the 5-arm-seed rows are the ones that apply;
the 3-arm-seed rows and the "recommended arm seeds" column are the comparison that J5 was taken from. `report` and
`power` print the same note under their tables.

Klone job 43403551 (2026-11-19), on the full shadow of the round-3 data: `analysis/context_arms.py power
--data-dir $WORK/data --variant baseline`, bootstrap pair `baseline-s43` minus `baseline-s44`, seed 0.

**What this means.** The rule is strict, and three seeds are not enough for the effects this experiment could
plausibly find. At three arm seeds a real +0.03 is detected 12% to 41% of the time; at five, 19% to 59%. The
smallest effect detected with 80% power is +0.065 to +0.085 for NoCurbRamp and SurfaceProblem, and +0.04 to +0.05 for
Obstacle. The false-`WIN` rate stays at or under about 1% in every row. Following the sign-off rule (five arm seeds
for any type whose power at +0.03 with three seeds is below 0.5), the **recommended default for stage A is five arm
seeds (`A5=1`) for all three target types**: 30 trainings instead of 18. Even then a `NO GAIN` rules out only
large gains, which is why every output words it as "no gain detected at this power". **Decided (J5, Jon,
2026-11-20, before stage A was submitted; timing under [Deviations](#deviations)): five arm seeds**, so
`seeds_confirmatory` is `[42, 43, 44, 45, 46]`; no other criterion was relaxed.

### Deviations

- **J5, five confirmatory seeds (2026-11-20): taken after the pilot's trainings had finished.** The five-seed
  recommendation (the sign-off rule: five arm seeds where power at +0.03 with three is below 0.5) was computed from
  October's baseline trio by the job `context-arms-launch-power-october`, which ended 2026-11-19 17:33 PST, before the
  pilot was submitted (its first job started 17:49 PST the same day). The pilot's 21 trainings all completed, the last
  by the morning of 2026-11-20. Jon confirmed five seeds (J5) on 2026-11-20, after that, so the pilot's
  `baseline-rerun` per-run outputs (each run's evaluate files) were on disk when J5 was confirmed. No pilot `report`
  or decision had been generated: the only `report` job run so far is the smoke's (2026-11-19 17:39 PST), and no
  decision file exists in the work or keep directories. Whether any individual pilot run's metrics were looked at before J5 was confirmed is not
  recorded. The change adds seeds and relaxes no criterion; stage A had not been submitted when it was made.
- **Nine panoramas came from an earlier copy, not from the lab store (2026-11-23, before stage A).** The restage
  pulled 39,231 of the 39,240 panoramas from the lab store. The other 9 are no longer on the lab store and were copied
  from an earlier copy on klone, the one round 3's own crops had been cut from: 4 for NoCurbRamp, 3 for Obstacle and
  2 for SurfaceProblem. They affect 9 rows (4 `train`, 5 `test_heldout`) out of 50,908. Nothing else about the data
  changed, and the restage finished before stage A was launched, so no stage-A result existed yet.

## Setup

| | |
|---|---|
| Data | the round-3 dataset on klone (`retrain-2026-10/<Type>/`), **read only**. Every job reads and writes a shadow of it on `/gscratch/scrubbed` instead (below) |
| Split | the round-3 `split.csv` (panorama-grouped, three deployments held out), byte-identical in the shadow; every run trains on its train rows and is scored on its val/test/test_heldout rows |
| Recipe | round 3 (above); `--batch-size` only moves the split between per-device batch and gradient accumulation, effective batch 32 either way |
| Seeds | `baseline-rerun` 42 to 46 (pilot); confirmatory arms 42 to 46 (stages A and B, `A5=1`; J5); exploratory arms 42 |
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
`trainer_state.json` and `config.json` and a `job.txt` (job id, node, GPU, start and end times, best checkpoint,
`KEEP_CKPT`) to
`<shadow>/<Type>/runs/<tag>/`, then writes the finished marker `.evaluated` there, before deleting the checkpoint
(`KEEP_CKPT=0`, the default; `best` keeps the best one), and copies the record, metrics and predictions to `$KEEP`.
`check-run` verifies a finished run (record present, seed, split hash, the context arm and view resolution the tag
stands for, `--epochs` when given, the best metric, target and per-device batch the job file pins, so a hand-submitted
`BATCH=4` run never counts as a seed, predictions covering exactly the shadow's val/test/test_heldout rows, one per
`label_uid`), and the report marks a decision `INCOMPLETE` when a record is missing or does not match (the same
checks, epochs fixed at 25). October's seed-42 model
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

# 4. stage A: first the restage (Jon's decision; below), then a dry run (never refused by the session cap)
STAGE=A A5=1 DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP PANO_STORE=<staged store> DRY_RUN=1 \
    analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem      # read it; then, once Jon has approved:
STAGE=A A5=1 DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP PANO_STORE=<staged store> ALLOW_OVER_SESSION_CAP=1 \
    analysis/context_submit.sh NoCurbRamp Obstacle SurfaceProblem      # the smoke and pilot logs hold 29 jobs

# 5. the decision (no GPU; anywhere with the shadow)
$ENV/bin/python analysis/context_arms.py report --data-dir $WORK/data --label-types NoCurbRamp,Obstacle,SurfaceProblem \
    --out docs/context-arms-2026-11
```

`A5=1` (five confirmatory seeds) is required for stages A and B: `seeds_confirmatory` is `[42, 43, 44, 45, 46]` in
`DECISION_RULE` and in the json block above (J5, 2026-11-20), and a launch without it trains three seeds, which the
preflight refuses before any `sbatch`.

Every invocation of the launcher appends its job ids and each job's full command to `context_jobs_<stage>.txt` in
the clone, refuses more than 40 jobs per invocation (`MAX_JOBS`) and, except under `DRY_RUN=1`, more than 40 summed
over every `context_jobs_*.txt` (`SESSION_MAX_JOBS`; `ALLOW_OVER_SESSION_CAP=1` overrides deliberately), refuses a
label type or tag listed twice and (where `squeue` exists) a type and tag already queued or running, stops at the
first `sbatch` that fails without retrying, and runs `preflight` before any `sbatch`. Single jobs:

```
STEP=train LT=NoCurbRamp ARM=side_info_nohq SEED=43 TAG=side_info_nohq-s43 DATA=$WORK/data RUNS=$WORK/runs KEEP=$KEEP \
    sbatch analysis/context_arms.slurm
STEP=backfill LT=NoCurbRamp DATA=$WORK/data PANO_STORE=<store> sbatch analysis/context_arms.slurm
```

`STEP=train` refuses an empty `TAG` and the tags `baseline` and `baseline-s<seed>` (October's files); a run whose
`<shadow>/<Type>/runs/<tag>/.evaluated` exists exits at once ("already evaluated", after redoing the `$KEEP` copy
and finishing any checkpoint delete a preemption cut short, with the `KEEP_CKPT` and best checkpoint that run recorded
in its `job.txt`), so a resubmitted job never overwrites a finished one. The job holds two exclusive locks for the
whole step: `<shadow>/<Type>/.train-<tag>.lock`, beside the outputs, and `$RUNS/<Type>/<tag>/.train.lock`, on the
checkpoints. A second job with the same type and tag (a stage resubmitted after an `sbatch` timeout, or the same run
under another `RUNS`) exits instead of writing the same predictions and record or training into the same directory.

**Time limit.** The jobs request 16 h, but the `ckpt-gpu` QOS caps them: `scontrol show job` reported
`TimeLimit=09:05:00` on every pilot job. The pilot's runs (about 4 h) fit. `view_heatmap-r448` at about 3x the step
time may not: a `TIMEOUT` is not requeued, so watch the first r448 run and resubmit the same command if it times out
(it resumes from its last checkpoint).

**The panorama restage (stage A).** The image arms' backfill re-renders each row's view from its panorama, and the
round-3 store on klone is gone. `analysis/context_arms.py stage-list` writes each type's missing list as
`<shadow>/<Type>/stage_files.txt` (the format `analysis/ai_labels_stage.slurm` reads) and one deduplicated list for
all types in `--out` (created), with a size estimate; a store that does not exist yet lacks everything, so it sizes
the restage before any launcher run. `preflight --backfill` writes the per-type lists too, marked store or not:

| type | unique panoramas | at 11.4 MB each |
|---|---:|---:|
| NoCurbRamp | 13,666 | about 156 GB |
| Obstacle | 13,971 | about 159 GB |
| SurfaceProblem | 13,630 | about 155 GB |
| three target types, deduplicated | 39,240 | about 450 GB |

```
$ENV/bin/python analysis/context_arms.py stage-list --data $WORK/data --types NoCurbRamp,Obstacle,SurfaceProblem \
    --pano-store /gscratch/scrubbed/$USER/pano-stage/context-arms --out $WORK/stage
RUN=$WORK/stage DEST=/gscratch/scrubbed/$USER/pano-stage/context-arms sbatch analysis/ai_labels_stage.slurm
```

`DEST` is a new store (or this one, to resume it), never an existing marked store: the stage job removes
`.pano-store` at its start and writes it back only after the pull, so a stopped pull would leave that store unmarked.

The stage job pulls sequential 1,000-file batches over one connection each and stops on the first connection error
without retrying (the makelab hosts ban a source that reconnects too fast); a resubmit resumes. The backfill then
runs per type inside stage A. Stage A should be launched within the scratch purge window of the shadow; if the shadow
has been purged, rebuilding it takes seconds, but the context images need the panoramas again.

## Cost

| stage | trainings | backfills | GPU-hours (estimate) |
|---|---:|---:|---:|
| smoke | 6 (1 epoch, 217 rows) | 1 | under 1 |
| pilot | 21 | 0 | about 21 x 2.5 = 50 |
| A | 33 | 3 | about 110, `two_scale` and `view_heatmap-r448` at about 2x and 3x the step time |
| B | 10 per guard type | 1 per guard type | about 33 |

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

The pre-registered `report` was run once, on 2026-11-24 (command and provenance below). It wrote, under
[`context-arms-2026-11/`](context-arms-2026-11/):

- [`decision.md`](context-arms-2026-11/decision.md) / `decision.json`: the verdict per type x confirmatory arm with
  every criterion and its value, the exploratory arms' intervals (never a verdict), the drift row, the operating
  characteristics recomputed from the baseline rerun's seeds, and the bootstrap's redraw and drop counts.
- [`gate.md`](context-arms-2026-11/gate.md): the report-only gate block per run.
- [`slices_seeds.md`](context-arms-2026-11/slices_seeds.md): type x severity and type x tag across seeds, on `test`
  and `test_heldout`, with training rows, the thin flag and October's round-3 AUC beside.
- [`arms.md`](context-arms-2026-11/arms.md) / `arms.json`, [`seeds.md`](context-arms-2026-11/seeds.md),
  [`slices.md`](context-arms-2026-11/slices.md): per-run tables against October's seed-42 model (descriptive;
  `arms.md` shows each arm's seed-42 run only).
- [`clipping.md`](context-arms-2026-11/clipping.md): the share of wide crops capped at the whole view, per type, and
  `wide_half_px` quantiles.
- `provenance.json`: sha256 of every file read, the arguments, the bootstrap seed, the git commit, the decision
  rule, the shares and shadow hashes, and each run's record status.

**The decision** (from `decision.md`; D = arm seed mean minus `baseline-rerun` seed mean of the within-city AUC;
95% interval D ± t·se with the seed and joint-bootstrap terms; the exploratory arms' single-seed intervals leave out
that arm's seed variance, so they are narrower than a confirmatory interval would be):

| type | arm | kind | seeds | verdict | D within-city held-out [95% CI] | D within-city test [95% CI] | arm seeds above baseline mean |
|---|---|---|---|---|---:|---:|---:|
| NoCurbRamp | `two_scale` | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | +0.013 [-0.008, +0.035] | +0.011 [-0.003, +0.025] | 5 of 5 |
| NoCurbRamp | `view_heatmap` | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.108 [-0.155, -0.061] | -0.047 [-0.073, -0.021] | 0 of 5 |
| NoCurbRamp | `view_heatmap-r448` | exploratory | 42 | none (exploratory) | -0.019 [-0.059, +0.021] | -0.017 [-0.042, +0.009] | 0 of 1 |
| NoCurbRamp | `side_info` | exploratory | 42 | none (exploratory) | -0.008 [-0.034, +0.018] | -0.001 [-0.018, +0.016] | 0 of 1 |
| NoCurbRamp | `side_info_nohq` | exploratory | 42 | none (exploratory) | -0.009 [-0.036, +0.017] | -0.001 [-0.022, +0.019] | 0 of 1 |
| Obstacle | `two_scale` | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | +0.005 [-0.018, +0.028] | +0.001 [-0.009, +0.012] | 4 of 5 |
| Obstacle | `view_heatmap` | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.055 [-0.100, -0.009] | -0.050 [-0.072, -0.027] | 0 of 5 |
| Obstacle | `view_heatmap-r448` | exploratory | 42 | none (exploratory) | -0.001 [-0.049, +0.047] | -0.004 [-0.025, +0.017] | 0 of 1 |
| Obstacle | `side_info` | exploratory | 42 | none (exploratory) | -0.005 [-0.032, +0.023] | -0.012 [-0.028, +0.004] | 0 of 1 |
| Obstacle | `side_info_nohq` | exploratory | 42 | none (exploratory) | +0.029 [-0.002, +0.060] | -0.013 [-0.031, +0.005] | 1 of 1 |
| SurfaceProblem | `two_scale` | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | -0.001 [-0.042, +0.039] | -0.001 [-0.016, +0.013] | 4 of 5 |
| SurfaceProblem | `view_heatmap` | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.075 [-0.122, -0.028] | -0.086 [-0.113, -0.058] | 0 of 5 |
| SurfaceProblem | `view_heatmap-r448` | exploratory | 42 | none (exploratory) | -0.017 [-0.078, +0.044] | -0.047 [-0.077, -0.018] | 0 of 1 |
| SurfaceProblem | `side_info` | exploratory | 42 | none (exploratory) | -0.025 [-0.067, +0.018] | -0.010 [-0.030, +0.010] | 0 of 1 |
| SurfaceProblem | `side_info_nohq` | exploratory | 42 | none (exploratory) | -0.005 [-0.041, +0.032] | -0.001 [-0.017, +0.015] | 0 of 1 |

Every confirmatory verdict rests on five arm seeds and five baseline seeds, every run's record matched its tag, every
run has a gate block, and the joint bootstrap kept 1,000 of 1,000 resamples on held-out and on test for every type
(none dropped; redraws for a resample that lost a class: held-out 13 for NoCurbRamp, all in private city A, and 0
for the other two types; test 46, 683 and 2,079). No seed on disk was left unused, and no row is `INCOMPLETE`.

**Drift row** (the rerun against October's round-3 trio, within-city AUC). No type is flagged: every held-out
difference is under two of October's seed sds.

| type | baseline-rerun within-city held-out, mean ± sd (5 seeds) | October round 3, mean ± sd (3 seeds) | rerun − October | test: rerun | test: October | drift flag |
|---|---:|---:|---:|---:|---:|---|
| NoCurbRamp | 0.894 ± 0.005 | 0.885 ± 0.024 | +0.009 | 0.909 ± 0.008 | 0.906 ± 0.003 | no |
| Obstacle | 0.778 ± 0.012 | 0.782 ± 0.009 | -0.004 | 0.867 ± 0.003 | 0.863 ± 0.007 | no |
| SurfaceProblem | 0.692 ± 0.014 | 0.706 ± 0.021 | -0.015 | 0.860 ± 0.006 | 0.863 ± 0.014 | no |

**Operating characteristics at the rule's five arm seeds**, recomputed by `report` from the rerun's five seeds
(criteria 1 to 3 simulated, so the `WIN` rates are upper bounds; the three-seed rows are in `decision.md`):

| type | seed sd (rerun) | bootstrap sd of D | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | effect at 80% power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCurbRamp | 0.005 | 0.011 | 0.007 | 0.008 | 0.26 | 0.60 | 0.97 | +0.040 |
| Obstacle | 0.012 | 0.010 | 0.010 | 0.012 | 0.22 | 0.50 | 0.92 | +0.045 |
| SurfaceProblem | 0.014 | 0.015 | 0.009 | 0.009 | 0.11 | 0.24 | 0.67 | +0.060 |

**Clipping** (`clipping.md`): the wide crop is the whole view for 46% of NoCurbRamp labels, 42% of Obstacle and 50%
of SurfaceProblem (clipped: 45%, 41%, 49%), under the "about 60%" that the design decisions named as the trigger for
a 2.5x second pass. So for about half of the labels `two_scale` saw the tight crop plus the whole view.

**Slices** (descriptive; no verdict reads them). The full type x severity and type x tag tables, on `test` and
`test_heldout`, with every arm, October's round-3 AUC and training counts, are in
[`slices_seeds.md`](context-arms-2026-11/slices_seeds.md). The held-out rows for the two confirmatory arms:

**NoCurbRamp, `test_heldout`** (seed-mean AUC difference against the baseline rerun's seed mean, k of 5 seeds above it)

| by | slice | n | incorrect | training rows | baseline rerun AUC | `two_scale` Δ (k/5) | `view_heatmap` Δ (k/5) |
|---|---|---:|---:|---:|---:|---:|---:|
| severity | 1 | 149 | 138 | 1,839 | 0.857 | +0.070 (5/5) | -0.011 (2/5) |
| severity | 2 | 158 | 109 | 3,595 | 0.878 | +0.004 (4/5) | -0.194 (0/5) |
| severity | 3 | 418 | 122 | 6,074 | 0.896 | +0.002 (3/5) | -0.101 (0/5) |
| tag | no alternate route | 348 | 78 | 4,670 | 0.894 | -0.004 (2/5) | -0.116 (0/5) |
| tag | alternate route present | 209 | 178 | 2,300 | 0.897 | +0.022 (5/5) | -0.109 (0/5) |
| tag | unclear if needed | 73 | 57 | 860 | 0.910 | +0.028 (5/5) | -0.137 (0/5) |
| tag | (none) | 125 | 79 | 4,631 | 0.903 | +0.005 (3/5) | -0.093 (0/5) |

**Obstacle, `test_heldout`** (seed-mean AUC difference against the baseline rerun's seed mean, k of 5 seeds above it)

| by | slice | n | incorrect | training rows | baseline rerun AUC | `two_scale` Δ (k/5) | `view_heatmap` Δ (k/5) |
|---|---|---:|---:|---:|---:|---:|---:|
| severity | 1 | 1,006 | 890 | 3,142 | 0.866 | +0.006 (3/5) | -0.046 (0/5) |
| severity | 2 | 335 | 158 | 2,844 | 0.850 | +0.007 (5/5) | -0.040 (0/5) |
| severity | 3 | 421 | 136 | 3,899 | 0.799 | -0.004 (2/5) | -0.038 (0/5) |
| severity | (none) | 50 | 41 | 669 | 0.813 | +0.011 (2/5) | -0.138 (0/5) |
| tag | pole | 439 | 355 | 2,087 | 0.932 | -0.003 (2/5) | -0.081 (0/5) |
| tag | trash/recycling can | 286 | 89 | 1,046 | 0.961 | -0.002 (2/5) | -0.049 (0/5) |
| tag | parked car | 164 | 98 | 1,293 | 0.779 | -0.025 (2/5) | -0.034 (0/5) |
| tag | vegetation | 156 | 87 | 1,136 | 0.857 | -0.017 (2/5) | -0.031 (1/5) |
| tag | sign | 139 | 124 | 486 | 0.791 | -0.014 (1/5) | -0.110 (0/5) |
| tag | outdoor dining area | 128 | 122 | 1 ⚠ thin | 0.631 | +0.150 (5/5) | +0.087 (5/5) |
| tag | narrow | 75 | 35 | 908 | 0.729 | +0.010 (3/5) | +0.045 (5/5) |
| tag | parked bike | 60 | 57 | 191 | 0.398 | +0.118 (5/5) | +0.095 (4/5) |
| tag | construction | 59 | 18 | 416 | 0.724 | +0.013 (4/5) | -0.167 (0/5) |
| tag | tree | 59 | 48 | 854 | 0.871 | +0.021 (4/5) | -0.038 (1/5) |
| tag | stairs | 47 | 21 | 35 ⚠ thin | 0.358 | -0.008 (4/5) | +0.049 (4/5) |
| tag | litter/garbage | 43 | 13 | 235 | 0.777 | +0.003 (2/5) | -0.018 (3/5) |
| tag | fire hydrant | 37 | 34 | 354 | 0.945 | -0.004 (3/5) | -0.073 (0/5) |
| tag | (none) | 154 | 116 | 1,933 | 0.814 | -0.013 (1/5) | -0.155 (0/5) |

**SurfaceProblem, `test_heldout`** (seed-mean AUC difference against the baseline rerun's seed mean, k of 5 seeds above it)

| by | slice | n | incorrect | training rows | baseline rerun AUC | `two_scale` Δ (k/5) | `view_heatmap` Δ (k/5) |
|---|---|---:|---:|---:|---:|---:|---:|
| severity | 1 | 574 | 337 | 5,050 | 0.703 | -0.004 (2/5) | -0.096 (0/5) |
| severity | 2 | 174 | 79 | 2,959 | 0.671 | +0.003 (3/5) | -0.119 (0/5) |
| severity | 3 | 74 | 43 | 1,833 | 0.814 | +0.034 (5/5) | -0.045 (1/5) |
| tag | brick/cobblestone | 288 | 191 | 337 | 0.590 | -0.020 (3/5) | -0.065 (0/5) |
| tag | narrow sidewalk | 148 | 120 | 1,160 | 0.662 | -0.003 (2/5) | -0.139 (0/5) |
| tag | cracks | 113 | 28 | 3,031 | 0.853 | +0.018 (4/5) | -0.117 (0/5) |
| tag | uneven/slanted | 94 | 38 | 1,574 | 0.663 | -0.014 (1/5) | -0.183 (0/5) |
| tag | sand/gravel | 80 | 39 | 284 | 0.562 | +0.041 (4/5) | +0.038 (5/5) |
| tag | utility panel | 68 | 32 | 904 | 0.594 | -0.035 (1/5) | -0.112 (0/5) |
| tag | bumpy | 63 | 13 | 1,982 | 0.751 | -0.058 (0/5) | -0.175 (0/5) |
| tag | grass | 57 | 17 | 2,636 | 0.814 | +0.014 (3/5) | -0.031 (0/5) |
| tag | height difference | 31 | 16 | 770 | 0.953 | +0.017 (5/5) | -0.146 (0/5) |

Not corrected for the number of slices; ⚠ thin = fewer than 100 training rows.

### Reading

This applies the rule as `decision.md` states it; nothing below changes a verdict.

**`two_scale`: `NO GAIN` on all three target types** ("no gain detected at this power"). D on the held-out
within-city AUC is +0.013 for NoCurbRamp, +0.005 for Obstacle and -0.001 for SurfaceProblem, all below the +0.02
the rule requires, and every interval includes 0. On NoCurbRamp criteria 3 to 5 pass (all five seeds above the
baseline mean, positive in 2 of 3 held-out deployments, no harm on test), but criteria 1 and 2 fail. What the
intervals rule out at 95%: a held-out gain larger than +0.035 (NoCurbRamp), +0.028 (Obstacle) or +0.039
(SurfaceProblem), and a held-out loss larger than -0.008, -0.018 or -0.042. On `test`, a loss larger than -0.003,
-0.009 or -0.016 is ruled out. Gains of the +0.02 to +0.03 size the rule was built to find are not ruled out: at five
seeds the simulated power at +0.03 is 0.60, 0.50 and 0.24.

**`view_heatmap` (224 px): `REGRESSION` on all three target types.** Both regression tests fire on every type: the
held-out interval lies wholly below 0 (D -0.108 [-0.155, -0.061] for NoCurbRamp, -0.055 [-0.100, -0.009] for
Obstacle, -0.075 [-0.122, -0.028] for SurfaceProblem), and the upper bound of the test interval is below -0.01
(-0.021, -0.027, -0.058). No seed of this arm is above the baseline's seed mean on any type. Under the rule a
`REGRESSION` on one type does not veto another, but here there is no `WIN` anywhere for it to veto.

**Exploratory arms (no verdict, one seed each).** `view_heatmap-r448`, `side_info` and `side_info_nohq` have their
intervals in the table above; the rule reads none of them. Their intervals omit the arm's seed variance (one seed),
so they are narrower than a five-seed interval would be. The largest held-out D among them is `side_info_nohq` on
Obstacle, +0.029 [-0.002, +0.060]; the smallest is `side_info` on SurfaceProblem, -0.025 [-0.067, +0.018]. On `test`,
`view_heatmap-r448` on SurfaceProblem has an interval wholly below -0.01 (-0.047 [-0.077, -0.018]); it is the only
exploratory row that would meet either regression test (held-out upper bound below 0, or test upper bound below
-0.01), and no exploratory interval lies wholly above 0. Under the rule an exploratory arm
that looks good earns a new pre-registered confirmatory run, not a verdict.

**What the rule says to do next.** There is no `WIN` on any type, so there is no stage-B `WINNER` and no guard type
to train. The rule's outcome for "No `WIN` anywhere" applies:
[#14](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/14) is to be closed as "no gain detected under
the round-3 recipe at the stated power", and the arms are not carried into round 4. The gate block (`gate.md`) is
report-only and no arm is proposed, so nobody needs to read it for a ship decision.

### Provenance

- **Code.** The pilot's 21 trainings ran from commit 9dace0c (2026-11-19), and stage A's 33 trainings and 3 backfills
  from aa8ef24 (2026-11-23), both recorded as not dirty in every run's `training_run.json`. Between the two commits,
  the training and inference code (`fine_tune_classifier.py`, `context_model.py`, `context_crops.py`, `evaluate.py`)
  is unchanged. Only `analysis/context_arms.py`, the job file's bookkeeping (locks, the "already evaluated" path,
  checkpoint pruning, the `$KEEP` copy), the launcher and tests changed. One visible consequence: the pilot's job file
  wrote the `.evaluated` marker beside the checkpoints, not in the run record, so the pilot's records have none. The
  report does not read the marker. The job file's `STEP=train` invocation is the same at both commits (checked with
  `git diff 9dace0c aa8ef24 -- analysis/context_arms.slurm`): the `fine_tune_classifier.py` command and its
  arguments, the defaults it takes (seed 42, 25 epochs, view resolution 224, batch 8), the `--context-arm` /
  `--context-view-res` arguments, the best-checkpoint lookup, the `evaluate.py` and `context_model.py predict`
  commands, and the backfill's `context_crops.py` call are byte-identical.
- **Run records.** Before the report, each of the 54 runs (15 `baseline-rerun`, 30 confirmatory, 3
  `view_heatmap-r448`, 6 side-arm) was checked with `check-run`'s checks (seed from the tag, split hash, arm and view
  resolution, 25 epochs, best metric, target and batch, predictions covering exactly the val, test and held-out rows),
  and every copy in `$KEEP` was compared with its copy in the shadow. All passed, and the four record files of every run were byte-identical in both places.
- **Preemptions.** Ten trainings were preempted on `ckpt-all` and resumed from their last checkpoint, as their records
  show (`resumed_from`, `slurm_restart_count`): in stage A, NoCurbRamp `two_scale` seeds 42 (1 restart) and 46 (2),
  NoCurbRamp `view_heatmap-r448` (3), Obstacle `two_scale` seeds 42 (2), 43 (1) and 44 (1), SurfaceProblem
  `two_scale` seeds 42 (1), 45 (1) and 46 (2); in the pilot, Obstacle `baseline-rerun` seed 42 (1). A resume
  continues the same recipe from the checkpoint's weights, optimizer and scheduler state, but it is not guaranteed to
  be bit-identical to an unbroken run, and the logged final train loss of a resumed run is not comparable with an
  unbroken one. All ten passed the record checks and count as
  ordinary seeds. The resumes are not spread evenly across arms. They hit 8 of the 15 `two_scale` runs, 1 of 15
  `baseline-rerun`, 0 of 15 `view_heatmap`, 1 of 3 `view_heatmap-r448` and 0 of 6 side-arm runs, so for `two_scale`
  resumption is partly confounded with the arm. As description only, not a finding: the resumed `two_scale` seeds
  are not systematically high or low against the unbroken ones (held-out within-city AUC, NoCurbRamp resumed 0.897
  and 0.911 against unbroken 0.900, 0.906 and 0.924; Obstacle resumed 0.794, 0.785 and 0.787 against 0.784 and
  0.768). One outlier is SurfaceProblem `two_scale` seed 42 (0.642, resumed once; the other four seeds are 0.699 to
  0.709). The rule reads all five seeds, and D without that seed would be about +0.011, still under +0.02, so this
  changes no verdict.
- **Panoramas.** 39,231 were restaged from the lab store on 2026-11-23, and 9 came from an earlier copy on klone, the
  one round 3's own crops had been cut from (4 / 3 / 2 for NoCurbRamp / Obstacle / SurfaceProblem; 9 rows: 4
  `train`, 5 `test_heldout`; see [Deviations](#deviations)). The stage-A backfills' `--verify` output was not copied
  off the cluster, so this document does not record it.
- **The report run.** `python analysis/context_arms.py report --data-dir <shadow> --label-types
  NoCurbRamp,Obstacle,SurfaceProblem --out docs/context-arms-2026-11`, run once on 2026-11-24 on a Mac
  (pandas 2.3.1, CPython 3.12) against a read-only copy of klone's shadow: per type `split.csv`, `manifest.csv`,
  every run's predictions, metrics and record, October's files, and the three backfills' `context/index.csv`. The
  images were not copied, and `report` reads none. The public-city list came from the live cities API, so the two
  private held-out deployments are lettered. The code was commit 8709891: master at b2a61d7 plus the `power_markdown` caption fix, which `report`
  does not call. A first attempt stopped before invoking `report`, because the context indexes had not been copied
  yet. It produced no output, and no number was seen before the one run. `provenance.json` holds the sha256 of every
  file read.
- **`provenance.json` regenerated, 2026-11-24.** The file first committed (from the run at 8709891) recorded local
  absolute paths in two of its arguments. `report` now writes its path arguments relative to the repository root,
  or by their last component when outside it (commit 1a74e80). The same command was re-run with that code, at commit
  aad8ad2 on a clean tree, into a scratch directory. All nine other outputs (`decision.md`, `decision.json`,
  `gate.md`, `slices_seeds.md`, `arms.md`, `arms.json`, `seeds.md`, `slices.md`, `clipping.md`) came out
  byte-identical to the committed files. Only `provenance.json` was replaced. It differs from the first only in
  `arguments.data_dir`, `arguments.shares`, `arguments.out` (which names that scratch directory, `regen-aad8ad2`)
  and `git_commit`; all 132 file hashes and 63 run entries are unchanged. So the committed outputs other than
  `provenance.json` come from the report code at 8709891, and `provenance.json` comes from aad8ad2.

## What could ship

Nothing, under the 2026-11-24 decision: no arm won on any type ([Results](#results)). These notes were written
before the result and are kept for any later confirmatory run.

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
