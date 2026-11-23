# Crop-constant sweep: 4000 / 6100 / 8000 (November 2026)

Issue: [#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21). Earlier steps:
[#51](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/51) (depth-sized against fixed-size crops) and
[`depth-ablation-2026-11.md`](depth-ablation-2026-11.md) (depth at the label; rules R0 to R3 in its section 7).
Code: [`analysis/crop_constant_sweep.py`](../analysis/crop_constant_sweep.py),
[`analysis/crop_constant_recrop.slurm`](../analysis/crop_constant_recrop.slurm),
[`analysis/crop_constant_submit.sh`](../analysis/crop_constant_submit.sh).

**Status: designed and pre-registered; nothing has been run.** The restage, the re-cut and the trainings wait for
Jon's go. The choices marked *provisional* in section 7 are implemented as recommended and can be changed with one
setting each before the run.

## 1. The question

Every crop is `int(C / depth_m)` px on each side of the label, with `C = 6100` inherited from the first version and
never tuned. The depth study found that, against a flat-ground reference, the served depth reads about 2.66× long, so
the crop is about 4.5 m wide at the label rather than the 11.9 m the formula suggests (depth doc, section 4). Whether
that size is the best one for validation was never measured. This sweep trains the validator on crops cut at
`C = 4000` (smaller crops), `6100` and `8000` (larger crops), on the same labels, and reads a pre-registered rule.

Two of the issue's boxes need no experiment: clipping at the view edge (38 of 64,465 round-3 crops at 6100, 34 at
4000, 54 at 8000; depth doc, section 5) and the publish-guard backstop (merged in
[#101](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/101); this PR adds the record check below).

## 2. Design

**Arms.** One re-cut pass renders each label's view once, exactly as round 3 did (`read_store_panorama` with its
8192 × 4096 resize, `crop_pose.pose_for_row(row, 0.95)` and `crop_pose.rig_y_norm`, the torch renderer), and cuts
three crops from that one view:

| arm | half-width | sized from |
|---|---|---|
| `c4000` | `int(4000 / depth_m)` | the **recorded** `depth_m`, through `depth_sized_crop(view, depth_m, constant=4000)` |
| `c6100` (baseline) | the **recorded** `crop_half_px` | through `centre_crop`; 17 round-3 rows differ from `int(6100 / depth_m)` by 1 px |
| `c8000` | `int(8000 / depth_m)` | the recorded `depth_m`, `constant=8000` |

No depth model runs and `depth_m` is never rewritten, so the arms differ in the constant only. Each arm directory
(`c4000/`, `c6100/`, `c8000/`) holds, per type, the crops, round 3's `split.csv` (same rows, same order, same
`split`; only `crop_half_px` and `clipped` replaced), a `manifest.csv` and a `generation_run.json` that copies round
3's record and adds `crop_depth_constant`, `derived_from` (the source split's sha256) and a `recrop` provenance
block. The deviation from the depth doc's section 9: no `sizing_depth_m` column, since it would equal `depth_m`.

**The baseline is regenerated.** c6100 is trained beside the arms on every seed, from the same re-cut, job file and
environment, so an arm is never compared with a run from another pipeline. Round 3's stored 6100 predictions are a
negative control that decides nothing (section 5).

**Training.** Round 3's recipe (hard target, 25 epochs) through `analysis/confident_learning.slurm`, unchanged:
`VARIANT=baseline DATA=$W/<arm> RUNS=$W/runs/<arm> SEED=<s> TAG=s<s> KEEP_CKPT=0`. One `RUNS` directory per arm, so
two arms at one seed never share a run directory. Checkpoints are deleted after evaluation; the GPU model of each
run is read from its log.

**Rows.** A row lost in the re-cut (missing panorama, a native size other than the recorded `pano_native_w/h`, a
Panoramax skip, an error) is removed from every arm.

**Seeds.** 42 to 46, each arm paired with c6100 at the same seed (*provisional*, decision 2; three seeds is the
alternative). The registered seed count `k` is fixed before the first training job is submitted, and `compare
--seeds` must name exactly those seeds.

**Types, staged.** Stage 1 is CurbRamp and SurfaceProblem. Stage 2 (NoCurbRamp, Obstacle, Crosswalk) runs only if
exactly one arm passes for both stage-1 types, and it trains that arm and c6100. Because the default changes only if
one arm passes for every swept type, a stage-1 failure settles "6100 stays", and stopping early cannot raise the
chance of a false change.

## 3. Gates before any training (stop on failure)

`gates` writes the arm directories only when every gate passes, so a failure leaves the trainings nothing to read
(and `--kill-on-invalid-dep` cancels them).

| gate | rule |
|---|---|
| G0 inputs | Round 3's `split.csv` and rawLabels pins match (section 9); `sidewalk-panorama-tools` equals round 3's commit; the versions of torch, torchvision, transformers, opencv and pillow are recorded. Checked by `preflight`, which also requires the store's marker and stops early if the panoramas missing from the store would already break G2. |
| G2 row loss | Per type, at most 0.5 % of rows lost, and at most 1 % of its `test` rows and of its `test_heldout` rows. A row never attempted fails the gate. |
| G3 re-cut parity | Per type, the c6100 re-cut against round 3's stored crop after the same WebP q95 encode and decode: per-row mean absolute difference at most 3.0 grey levels on at least 99.5 % of rows, and no row above 12. A missing stored crop or a different size counts as above 12. Reported per source store too. |

G3's basis: the segmentation-prior study measured 1.33 (max 2.47) on 800 rows; a 1 px shift gives 4.7 at the
median; the context-arms verify step fails above 12. G3 guards against a restage that delivers different files or a
drifted renderer; it does not decide which baseline is used. The bounds are exact fractions, so a share exactly on a
bound passes.

## 4. Decision rule (R3 as amended; fixed before any arm exists)

Written on 2026-11-23, before any arm was cut or trained. Known at the time: #51's result (+0.014 AUC in the training
deployments), sections 4 and 5 of the depth doc, the stored baselines' seed spread (SurfaceProblem test AUC 0.896 /
0.910 / 0.893), and the crop-size distribution at each constant. The amendments to R3 as registered on 2026-11-12
are logged, dated, in the depth doc's section 7.

- **Statistic**, per swept type and arm: the mean over the `k` seed pairs of the paired ΔAUC (arm minus c6100) on
  `test`.
- **Both intervals above 0** on `test`, at level `1 − 0.05 / m` with **m = 2 for every type**, whatever stage 2
  trains (97.5 %; lower bound at the 1.25th percentile):
  - the registered interval: 2,000 resamples of whole panoramas;
  - a hierarchical interval from the same replicates that also resamples the seed pairs with replacement.
  - In each replicate one panorama draw is shared by every drawn seed and by both models.
  - A lower bound of exactly 0 does not pass.
- **Seed guard on `test`**, by the registered seed count: with three seeds all three pairs ahead (Δ > 0); with five
  or more, at least `k − 1` of `k`. Both intervals already control the false-change rate, so the guard is not made
  stricter as seeds are added.
- **Held-out deployments:** the mean over seed pairs of the within-city ΔAUC on `test_heldout` is at least 0 (per
  deployment AUCs weighted by n_incorrect × n_correct, deployments with at least 10 rows and both classes).
- **Across types:** the default changes only if one arm passes for every swept type (a conjunction, so no further
  adjustment).
- **Both arms pass:** nothing changes, stage 2 is not triggered, and the result does not read as a constant effect.
  (R3 as registered would have taken the larger summed ΔAUC; this is stricter.)
- **A stage-1 pass with stage 2 not run** is a candidate only; the default does not change.
- **Unestimable is never a pass**: fewer runs than the registered seeds on either side (no reading on a subset), fewer
  than 10 `test` rows of a class, a run that records another arm's `crop_depth_constant`, or a family whose runs
  differ in `versions` or `git_commit`, record `git_dirty` other than false, or record `git_state_at_finish` (the
  checkout moved during a run). The cause is named.
- An arm whose two upper bounds are both below 0 is reported as **worse**; it does not "pass" by excluding zero.
- **R1 or R2 arms added later** re-read the whole family with the new m, and nothing changes until the family is
  complete.

The read-out also prints, per type and arm, the **MDD** (the Δ that both lower bounds would have needed to clear 0
at the same widths) and the few-seeds caveat: with three seeds, (1/3)³ = 3.7 % of hierarchical replicates draw one
given seed only (for example the worst), so the hierarchical level is nominal.

`rule_readout` implements this and is tested at every boundary named above (`tests/test_crop_constant_sweep.py`).

**Downstream if an arm passes** (out of scope here): retrain all five types and refit the calibration maps; change
`int(inv_depth * 6100)` in sidewalk-ai-api and its tests in the same release; change `SERVED_CROP_DEPTH_CONSTANT`
and the default here in one reviewed PR (with a CLI flag if one is wanted). It lands with a round.

## 5. Reported, not deciding

- Held-out deployments: pooled and within-city ΔAUC with both intervals, and leave-one-deployment-out (three held-out
  deployments). Non-public deployments are never named (`non-public deployment A`, `B`).
- Slices, per the standing rule: type × severity and type × tag, seed-mean Δ beside the baseline AUC, with training
  counts and the thin-slice flag (`train_n < 100`).
- Mechanism slices: the baseline `crop_half_px` quartile and the capture-year band.
- Recall at precision 0.92 per class, and the sensitivity curve (Δ at each constant).
- Negative control: stored round-3 6100 predictions against c6100 at seeds 42 to 44. The stored seed-42 run came
  from `fine_tune.slurm` (4 workers) and seeds 43 and 44 from `confident_learning.slurm` (7 workers), so a seed-42
  outlier there is not a pipeline effect.
- #51's fixed 238 px CurbRamp run as a context row (a different pipeline; context only).

## 6. Cost (klone `ckpt-all`, free)

| item | stage 1 (CurbRamp + SurfaceProblem) | stage 2 (other three types, only if triggered) |
|---|---|---|
| restage of panoramas | about 215 to 260 GB, about 22,500 panoramas | at most about 370 GB more (`stage-list --exclude-store` against the stage-1 store gives the real figure) |
| re-cut | 26,289 labels; 6 to 26 h of job time | 38,176 labels; 8 to 37 h |
| training, 3 seeds | 18 runs, about 32 GPU-h | 18 runs, about 32 GPU-h |
| training, 5 seeds | 30 runs, about 53 GPU-h | 30 runs, about 53 GPU-h |

- Training uses the 10-28 baseline runs' wall-clock: about 1.25 h for CurbRamp and 2.25 h for SurfaceProblem.
  Stage-2 times are scaled by train rows and unverified.
- The re-cut rate is 0.8 s per label (round 3's generation) to 3.5 s (the segmentation-prior job, decode-bound).
  `SMOKE=1` measures it first. The array is chunked so that chunk labels × 3.5 s × 1.5 fits inside `--time` (8 h by
  default: at most 5,485 labels a chunk), so a slow node does not time out in a requeue loop.
- Scrubbed storage: the restage plus about 6 GB of crops (78,867 files) for stage 1, and no kept checkpoints.
  Scrubbed purges after 21 days and was 98 % full on 11-03, so a real write test comes first (both job files do one)
  and training must follow the re-cut within the window.
- The stage job (`analysis/ai_labels_stage.slurm`, reused unchanged) opens one ssh connection per 1,000-file batch,
  sequentially (about 23 for stage 1, minutes apart), and stops without retrying on any rsync error other than 23
  (missing files). Its own default destination is the lab's durable store, which cannot take this volume, so the
  submit script always passes `DEST` and refuses one outside the working directory.
- Preconditions: klone can reach the panorama store (tested with one connection), and klone is free of the other
  runs now using it.

## 7. Decisions for Jon (recommendations are provisional and implemented as recommended)

1. **Scope.** Recommended: stage 1 only, stage 2 triggered by a single-arm pass on both types (`TYPES`, default
   `CurbRamp SurfaceProblem`). Alternatives: all five at once, or CurbRamp only (a candidate at most).
2. **Seeds: three or five.** Recommended: five (`SEEDS`, default `42,43,44,45,46`; about 53 GPU-h for stage 1
   against 32). On SurfaceProblem the seed noise in a Δ is about the size of the effect #51 found, so three seeds
   would probably miss a true +0.01. With five the guard is four of five.
3. **The restage.** A store of our own on scrubbed (`RESTAGE=1`, the default; about 215 to 260 GB for stage 1), or
   sharing the SurfaceProblem panoramas another study restages anyway (`RESTAGE=0 PANO_STORE=<that store>`), which
   leaves CurbRamp's 9,343 as the only extra. Recommended: share, if that store keeps the pano-store layout. Either
   way, reachability of the panorama store from klone is confirmed first.
4. **The go** for the klone run, once the PR is reviewed and klone is free.
5. **Ticking two boxes on the issue**: clipping and the publish-guard backstop. Recommended: tick both.

Made under the standing statistics delegation, listed for visibility: the three R3 amendments, the seed guard by seed
count, m = 2, the staging rule, sizing from the recorded depth, the G2 and G3 thresholds, no manifest column and no
CLI flag, deleting checkpoints after evaluation.

## 8. Results

*Not run yet. Filled by pasting from `docs/crop-constant-2026-11/tables.md` after `compare` and `report`.*

### 8.1 Gates (G0, G2, G3)
### 8.2 Rule read-out
### 8.3 Held-out deployments
### 8.4 Slices
### 8.5 Negative control

## 9. Reproduce

**Inputs** (`preflight` refuses any mismatch; the same pins as the depth doc's section 12):

| file | sha256 |
|---|---|
| `CurbRamp/split.csv` | `f2ec89f8032c9b6b7c1533bea7ac82d82b7c139847b14286dd53d932a2195610` |
| `SurfaceProblem/split.csv` | `c48cac6c889b1df2c125238401f8f886904ae2a28c8e43628f5e2e799abd8fdd` |
| `NoCurbRamp/split.csv` (stage 2) | `fbb1e53bd121bdd5eaca8b12d8fcf5a6ec3f6102920f1f68735eb78c9703ce33` |
| `Obstacle/split.csv` (stage 2) | `d41e865f45b5f0b91d3bc81c1a60dae5e3effea2d30dd92c7883207feb87e8aa` |
| `Crosswalk/split.csv` (stage 2) | `91001f64b778ec3d59b381d50911c6bd8ea35449052574d97eb4c9380d37f00f` |
| `rawlabels/*.csv` (`cd rawlabels && shasum -a 256 *.csv \| shasum -a 256`) | `42c0dcfab78943d3e329b21c3d86322a0894469d4fd11eb5ea2f576d5dd6999a` |
| `sidewalk-panorama-tools` | `7ce25ffbddae23d85f9976ec0a925707a0d64a89` (round 3's `package_source`) |

Every later record pins what it read: `stage_list.json` (the list's sha256 and the split pins), `preflight.json`
(pins, package source, versions, git commit), `gates.json` (G0 to G3, the sha256 of `recrop_parity.csv`), each arm's
`generation_run.json` (`derived_from`, `recrop`), each run's `training_run.json` (commit, versions,
`crop_depth_constant`), `compare.json` (the sha256 of every split, prediction and run record it read, and of the
stored control predictions) and `provenance.json` (`tables.md`, the figure, the public listing). Record those hashes
here after the run. `recrop_parity.csv` is per label and stays off the repo.

### klone recipe (Jon or the parent session; needs a live klone master and Jon's go)

Everything is written under `W=/gscratch/scrubbed/$USER/crop-constant-2026-11`; the recrop job refuses any other
working directory, and the submit script refuses a stage destination outside it. Round 3's data root and the conda
env are only read.

1. **Login node:** a real write test (`dd` 8 MiB into `$W`, then remove it) and `hyakstorage /gscratch/scrubbed`.
   Clone this branch into `$W/sidewalk-validator-ai` (no submodule is needed) and `mkdir -p $W/logs`. Confirm with
   one connection that klone can reach the panorama store.
2. **Stage** (decision 3; skip when sharing a store): from the clone,
   `python analysis/crop_constant_sweep.py stage-list --root $R3 --types CurbRamp,SurfaceProblem --out $W/stage_files.txt`,
   then `sbatch --chdir=$W --time=24:00:00 --export=ALL,RUN=$W,DEST=$W/store analysis/ai_labels_stage.slurm`. On a
   timeout or any rsync error other than 23, stop and do not resubmit.
3. **Smoke:** `cd $W && SMOKE=1 REPO=$W/sidewalk-validator-ai R3=$R3 PANO_STORE=$W/store sbatch --array=0
   $W/sidewalk-validator-ai/analysis/crop_constant_recrop.slurm`. It cuts 20 labels into `$W/smoke/` and prints the
   seconds per label; if that is above 3.5 s, raise `RECROP_TIME` before step 4.
4. **Submit** from the clone: `DRY_RUN=1 RESTAGE=0 R3=$R3 analysis/crop_constant_submit.sh` to read the commands,
   then the same without `DRY_RUN`. It chains preflight → recrop (array) → gates → the trainings
   (`RESTAGE=1` puts the stage job in front instead of step 2). Job ids go to `$W/crop_constant_jobs.txt`.
5. **Mac:** copy each arm's `split.csv` and `predictions_s*.csv`, `runs/*/*/*/training_run.json`, the logs,
   `preflight.json`, `gates.json` and `recrop_parity.csv` into `~/Data/sidewalk-validator-ai-crop-constant-2026-11/w`,
   then `compare --work <that copy> --seeds <the registered seeds> --control-root $R3 --logs-dir <logs>
   [--fixed238 <#51's CurbRamp predictions>] --out compare.json` and `report --compare compare.json --out-dir docs/crop-constant-2026-11`. Paste section 8 and pin
   the hashes above.
