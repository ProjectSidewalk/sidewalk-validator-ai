# Does a larger input size help DINOv2-L? (November 2026)

Every validator since 2025-08 trains and serves at 224 × 224 px, while `data_generation.py` stores each crop at a
640 px long side. The backbone study ([`backbone-2026-11.md`](backbone-2026-11.md)) and the context-arm study
([`context-arms-2026-11.md`](context-arms-2026-11.md)) both found no arm that beats the round-3 model, but both held
the input at 224, so neither says anything about resolution. The slices that stay weak are the fine-texture ones:
brick/cobblestone in SurfaceProblem (AUC about 0.63), missing tactile warning in CurbRamp, paint fading in Crosswalk,
stairs in Obstacle. This study retrains the round-3 model with the round-3 recipe at 448 px (518 px as a registered
later arm) against a 224 px rerun in the same environment, at three seeds, and decides with a rule written before
any run.

Related but a separate axis: [#21](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/21) (the crop
constant, [`crop-constant-2026-11.md`](crop-constant-2026-11.md)) changes **how much of the scene** a crop covers.
This study keeps every crop exactly as it is and changes only **how many pixels of it** the model sees.

## Status

**Pre-registered; no GPU result exists.** This document, the rule, the power table
([`resolution-2026-11/power.md`](resolution-2026-11/power.md)) and the laptop latency measurement
([`resolution-2026-11/latency_laptop_cpu.json`](resolution-2026-11/latency_laptop_cpu.json)) were committed before
any job was submitted. Nothing has run on klone. What exists:

- `fine_tune_classifier.py --input-res {224,448,518}` (default 224, byte-identical to today) and
  `--gradient-checkpointing` (a memory fallback, off by default).
- [`backbones.py`](../backbones.py): `input_sizes`, `tokens_at`, and an `input_res` argument to
  `preprocess_spec`, `preprocess_sha256`, `build_image_processor` and `backbone_settings`.
- `publish_to_hf.py`: `check_backbone` refuses a run whose `input_res` is not 224 (serving reads 224 today).
- [`analysis/resolution_study.py`](../analysis/resolution_study.py) (`power`, `bench`, `report`), the klone pair
  [`analysis/resolution_submit.sh`](../analysis/resolution_submit.sh) and
  [`analysis/resolution.slurm`](../analysis/resolution.slurm), and the pins in
  [`analysis/resolution_inputs.json`](../analysis/resolution_inputs.json).
- Tests: `tests/test_input_resolution.py`, `tests/test_resolution_study.py`, `tests/test_resolution_slurm.py`.

## Why now

1. **The crops hold more detail than the model sees.** The crop's native detail is `2 × crop_half_px` in the
   2048 px view, a median of about 470–510 px per type. At 224 px about 90% of crops are downsampled. Only 6–12%
   per type are under 256 px natively, so most of them have detail to give at 448.
2. **The weak slices are texture.** Brick/cobblestone, worn paint and tactile domes are spatial-frequency cues that
   a 2× downsample removes first.
3. **DINOv2 was pretrained for it.** `facebook/dinov2-large`'s position table is a 37 × 37 grid (518 px at patch
   14); 224 px interpolates it *down* to 16 × 16. At 518 the table is used as pretrained, and 448 needs a smaller
   interpolation (37 → 32) than 224 does.
4. **It is the cheapest unanswered axis.** It needs no new data, no new backbone and no new architecture, only a
   flag.

## What changes, and what is held fixed

The **only** difference between an arm and the baseline is the input size, applied identically everywhere a crop
becomes a tensor:

| path | 224 (today) | 448 | 518 |
|---|---|---|---|
| train augmentation | `RandomResizedCrop(224, scale 0.7–1.0)` + flip | `RandomResizedCrop(448, …)` + flip | `RandomResizedCrop(518, …)` + flip |
| val / test / held-out | shortest edge 256, centre-crop 224 | shortest edge 512, centre-crop 448 | shortest edge 592, centre-crop 518 |
| saved processor (evaluate.py, export_onnx.py, serving) | as above | as above | as above |
| tokens per image | 256 | 1,024 | 1,369 |
| position table | bicubic 37 → 16 | bicubic 37 → 32 | as pretrained |

- **The field of view is unchanged.** The resize scales with the crop (`res × 256 / 224`), so the centre crop
  keeps the same 87.5% of the stored crop at every size (`backbones.input_sizes`; tested).
- **Evaluation needs no new flag.** `evaluate.py` and `export_onnx.py` build their processor from the checkpoint's
  saved `preprocessor_config.json`, so the predict path that scores `test_heldout` uses the run's size by
  construction. `run_one` refuses to slim a run whose saved processor does not crop its size.
- **Position embeddings.** HF's `Dinov2Embeddings.interpolate_pos_encoding` runs on every forward pass: bicubic to
  `(H / 14, W / 14)`, the CLS position untouched, and the table returned unchanged when the grid already matches
  (518). `tests/test_input_resolution.py` checks this on a tiny DINOv2 config at 224, 448 and 518, including the
  gradient under checkpointing.
- **Held fixed (the round-3 recipe):** `facebook/dinov2-large` at its pinned revision, the round-3 split
  (`analysis/resolution_inputs.json` pins every `split.csv`), lr 1e-5, 25 epochs, per-device batch 8 × gradient
  accumulation 4 (effective batch 32, as round 3), hard targets, best checkpoint by val macro F1, the loss divided by
  the accumulation steps (`model_accepts_loss_kwargs = False`), fp32, and the backbone study's env (torch 2.7.1,
  transformers 4.56.2, pandas 2.3.1). The 224 arm is a **rerun in this env**, not round 3's own runs.
- **Refused combinations.** `--input-res` other than 224 needs `--output-dir` and is refused with `--backbone`,
  `--context-arm`, `--region-conditioning` or `--init-from`, so no run can mix two experiments. It enters the run
  key (`input_res`), so a resume across sizes is impossible. The default adds nothing to the key, so every existing
  run key and output is unchanged (tested).
- **Provenance.** `training_run.json` records `input_res`, `gradient_checkpointing`, `attn_implementation` and the
  `backbone` block's preprocessing and its hash at the run's size.

## Arms

| arm | tags | seeds | role |
|---|---|---|---|
| 224 px rerun | `res-r224`, `res-r224-s43`, `res-r224-s44` | 42, 43, 44 | baseline |
| 448 px | `res-r448`, `res-r448-s43`, `res-r448-s44` | 42, 43, 44 | the arm under test |
| 518 px | `res-r518`, … | 42, 43, 44 | **registered later arm**: submitted only on Jon's word (J3), judged by the same rule |

## Types

| type | role | train / val / test / held-out rows | why |
|---|---|---|---|
| Obstacle | target | 10,554 / 2,262 / 2,266 / 1,812 | stairs, small objects |
| SurfaceProblem | target | 10,500 / 2,250 / 2,255 / 847 | brick/cobblestone, cracks |
| NoCurbRamp | target | 12,199 / 2,607 / 2,613 / 743 | ramp edge versus curb at distance |
| CurbRamp | guard | 6,925 / 1,487 / 1,479 / 546 | the best model; must not get worse; holds missing tactile warning |
| Crosswalk | texture guard | 2,089 / 448 / 445 / 138 | in the study only for its paint-fading slice (J4) |

Each type has three held-out deployments. Private ones are lettered in every table (`private city A (US)`), never
named.

## Memory and time

**Memory.** At 224 px, DINOv2-L at batch 8 peaked at 8.0 GB in the backbone study. The context study's
`view_heatmap-r448` (a 224 pass plus a 448 pass, batch 8) peaked at 20.2 GB, so one 448 pass is about 17–18 GB and
518 about 21 GB (attention is SDPA). Both fit a 48 GB L40S (and the A40/L40 nodes of the constraint) at batch 8, so
the plan keeps batch 8 × accumulation 4 and **does not** use gradient checkpointing. The smoke job prints each size's
measured peak. If one does not fit, the fallbacks are, in order, `GC=1` (`--gradient-checkpointing`, same maths,
slower) and then `BATCH=4` (per-device 4 × accumulation 8, the same effective batch); both are recorded in
`training_run.json` and count as deviations. The report counts a run at another per-device batch only with
`--allow-batch`.

**Optimizer steps per run** (effective batch 32, 25 epochs): Obstacle about 8,250, NoCurbRamp 9,550,
SurfaceProblem 8,225, CurbRamp 5,425, Crosswalk 1,650; 33,100 per seed for all five types.

**Cost estimate** (from the backbone study's measured 1.21 s/step on CurbRamp and 1.62 s/step on SurfaceProblem
for `dinov2-large` at 224, mean about 1.47 s/step, loading and per-epoch evaluation included, on mixed
A40/L40/L40S). Per layer the work is about `12d² + 2Nd` MACs per token (d = 1024, N tokens), so a step's model work
grows 4.47× at 448 and 6.26× at 518. The lower bound below assumes half of today's step is fixed (data loading,
JPEG/WebP decode, evaluation) and does not grow. The upper bound scales the whole step.

| arm | runs | GPU-hours, lower | GPU-hours, upper | longest single run (NoCurbRamp) |
|---|---|---|---|---|
| 224 rerun | 15 | 41 | 41 | about 3.9 h |
| 448 | 15 | 111 | 182 | 10.6–17.4 h |
| 518 (later) | 15 | 147 | 254 | 14.1–24.4 h |
| smoke (1 job) + bench (1 job) | 2 | about 2 | about 2 | |

**Time limit.** The jobs request 16 h, but the `ckpt-gpu` QOS caps them at `TimeLimit=09:05:00` (context study,
"Time limit"), and a `TIMEOUT` is not requeued. At 448 the three large types run past it, so they need one or two
continuations: `resolution_submit.sh resume` resubmits every started row that has no done marker and is not queued or
running, and the run resumes from its last epoch checkpoint under the same run key. `cost.md` measures the step time
of each run's last attempt.

## Latency against the nightly budget

The nightly AI run scores up to 800 labels per city (`getLabelsToValidateWithAi`,
[`deploy-checklist-2026-10.md`](deploy-checklist-2026-10.md), F6). A larger input changes only the classifier's
share of a request; fetching, cropping and depth do not change.

Measured on the laptop CPU (Apple arm64, 4 threads, torch 2.7.1, fp32, random DINOv2-L weights, batch 1, median of 7;
[`resolution-2026-11/latency_laptop_cpu.json`](resolution-2026-11/latency_laptop_cpu.json)):

| size | tokens | preprocess ms | forward ms | total ms per crop | × 224 |
|---|---|---|---|---|---|
| 224 | 256 | 4.0 | 236.5 | 240.5 | 1.0 |
| 448 | 1,024 | 7.9 | 1,067.6 | 1,075.5 | 4.5 |
| 518 | 1,369 | 11.6 | 1,605.4 | 1,617.0 | 6.7 |

The API serves on makelab2's A40, so the number that matters is the A40's. `resolution_submit.sh bench` measures it
in one 1-hour job pinned to an A40 (`STEP=bench`, batch 1 and 16), and `report --bench` turns it into hours per
night at the cap across the deployments listed. **No latency threshold is pre-registered** (J5): the table is
descriptive, and adopting 448 is the rule's call on accuracy plus Jon's call on cost.

## The pre-registered decision rule

Computed by `analysis/resolution_study.py report` from the runs' records. Constants in `DECISION_RULE`, which a test
checks against this block:

```json
{
  "endpoint": "within-city AUC on test_heldout",
  "statistic": "seed mean of the arm minus seed mean of the in-env 224 rerun",
  "interval": "welch_t_seed_variance_plus_panorama_bootstrap",
  "level": 0.95,
  "min_delta": 0.015,
  "min_target_wins": 2,
  "target_types": [
    "Obstacle",
    "SurfaceProblem",
    "NoCurbRamp"
  ],
  "target_test_auc_floor": -0.005,
  "guard_type": "CurbRamp",
  "guard_floor": -0.005,
  "texture_slices": [
    [
      "SurfaceProblem",
      "tag",
      "brick/cobblestone"
    ],
    [
      "CurbRamp",
      "tag",
      "missing tactile warning"
    ],
    [
      "Crosswalk",
      "tag",
      "paint fading"
    ]
  ],
  "texture_splits": [
    "test",
    "test_heldout"
  ],
  "texture_fails_when": "interval_wholly_below_zero",
  "seeds": [
    42,
    43,
    44
  ],
  "baseline_res": 224,
  "arms": [
    448
  ],
  "later_arms": [
    518
  ],
  "prefer_larger_res_margin": 0.01,
  "within_min_n": 10,
  "slice_min_n": 30,
  "slice_min_class": 8
}
```

**The statistic.** For each run, the within-city AUC on `test_heldout` (round 3's definition: per-deployment AUC,
weighted by n_incorrect × n_correct, deployments with at least 10 labels and both classes). D = the arm's seed mean
minus the 224 rerun's seed mean. Its 95% interval is the context study's (`context_arms.seed_mean_delta`): Welch t
with the two seed variances plus the bootstrap variance of D, from one joint panorama bootstrap stratified by
deployment that resamples the same rows for every run (`context_arms.within_city_boot`, 1,000 resamples).

**A target type WINS** when all three hold:

1. D ≥ +0.015;
2. the interval's lower end > 0;
3. the seed-mean paired test AUC Δ (`backbone_comparison.paired`) ≥ −0.005 (no in-distribution loss bought with a
   held-out gain).

A target type is a **REGRESSION** when the interval's upper end < 0, and **NO GAIN** otherwise.

**The arm WINS** when all of these hold:

- at least 2 of the 3 target types win;
- **guard:** CurbRamp's D ≥ −0.005 **and** its seed-mean paired test AUC Δ ≥ −0.005 (point estimates: no
  regression beyond 0.005);
- **texture:** no texture slice *falls* on `test` or `test_heldout`. A slice falls when the same seed-mean interval,
  computed on the slice's pooled AUC, lies wholly below zero. A slice with fewer than 30 rows or fewer than 8 of
  either class on a split is *not evaluable* there, never a failure, and is listed as such (J6).

If the guard or a texture slice fails, or two target wins are out of reach, the arm reads **NO WIN** at once. While a
needed run is missing and a win is still possible, it reads **HELD**.

**On a WIN**, the proposal is to adopt the arm for every type, except a type whose interval lies wholly below zero or
whose test Δ is below −0.005 (that type keeps 224). **If 448 and 518 both win**, adopt 448 unless 518's summed D over
the target types exceeds 448's by at least 0.010 (518 costs about 1.5× 448 to serve). **No arm wins:** keep 224, and
the backbone and context nulls stand as resolution-independent at 224 only.

**Descriptive, never decisive:** pooled held-out AUC, per-seed paired deltas, per-deployment AUC, within-city AUC
on `test`, macro F1, recall at precision 0.92/0.95, the 0.92 gate under `majority2` (coverage and precision,
`prior_rules.gate_rows`), Crosswalk's type-level numbers, the stairs slice of Obstacle (35 train / 9 test / 47
held-out rows: too thin to judge), cost and latency.

### Operating characteristics (computed before any run)

From `resolution_study.py power` ([`resolution-2026-11/power.md`](resolution-2026-11/power.md)), with October's
round-3 within-city held-out seed sds and bootstrap sds (context study, klone job 43403551). Only criteria 1 and 2 are
simulated, so these are upper bounds:

| type | seeds per model | P(win) at true +0.000 | +0.015 | +0.030 | +0.050 | effect for 80% |
|---|---|---|---|---|---|---|
| Obstacle | 3 | 0.001 | 0.039 | 0.264 | 0.775 | +0.055 |
| SurfaceProblem | 3 | 0.002 | 0.017 | 0.066 | 0.231 | +0.095 |
| NoCurbRamp | 3 | 0.005 | 0.025 | 0.079 | 0.251 | +0.100 |
| **study (≥ 2 of 3)** | 3 | 0.000 | 0.002 | 0.041 | 0.341 | |
| Obstacle | 5 | 0.009 | 0.145 | 0.591 | 0.968 | +0.040 |
| SurfaceProblem | 5 | 0.010 | 0.059 | 0.195 | 0.555 | +0.070 |
| NoCurbRamp | 5 | 0.011 | 0.067 | 0.222 | 0.586 | +0.065 |
| **study (≥ 2 of 3)** | 5 | 0.000 | 0.021 | 0.238 | 0.800 | |

Read plainly: the false-win rate is near zero, but **at 3 seeds the study can detect only a large effect** (about
+0.05 within-city on two types at once gives a 1-in-3 chance). The seed spread of SurfaceProblem and NoCurbRamp
(0.021–0.024) is larger than the +0.015 bar itself. Five seeds raise the study's power at +0.05 to 0.80 and cost 10
more runs per arm (J1). The study as briefed registers 3 seeds; the table is here so that a null at 3 seeds is read
as "no large effect", not "no effect".

## Slices

`slices.md` reports **every type × severity and type × tag slice** for every arm against the 224 rerun, on `test`
and `test_heldout`, with each slice's training rows and the `thin` flag (< 100 training rows): the seed-mean AUC of
both models, D and the same interval, and how many arm seeds beat the baseline mean. Slices are not corrected for
multiplicity, and only the three texture slices enter the rule. Training rows for the texture slices (round-3 split):
brick/cobblestone 337 train / 77 test / 288 held-out (SurfaceProblem); missing tactile warning 1,482 / 303 / 141
(CurbRamp); paint fading 339 / 78 / 66 (Crosswalk).

## How to run

On the klone login node, from a checkout of this branch, after `analysis/backbone_env.sh` (the same overlay env as the
backbone study):

```bash
analysis/resolution_submit.sh view              # $OUT/data: the five types' pinned files, crops linked
analysis/resolution_submit.sh view smoke        # CurbRamp subsample for the smoke job
analysis/resolution_submit.sh smoke             # 1 job, 2 h: one-epoch runs at 224/448/518, guards, GPU tests,
                                                # bench, report; prints peak memory per size
analysis/resolution_submit.sh bench             # 1 job on an A40: latency per crop per size
MAX_PARALLEL=8 analysis/resolution_submit.sh main   # 30 elements: 5 types x {224, 448} x 3 seeds, seed 42 first
analysis/resolution_submit.sh resume            # after the 9:05 timeouts: continue every started, unfinished row
ALLOW_518=1 analysis/resolution_submit.sh r518  # only on Jon's word (J3); needs a cap raise
```

The launcher reuses the backbone study's safeguards: a real 8 MiB write test, free-space checks on scrubbed and
`/gscratch/makelab`, a ledger (`$OUT/resolution_jobs.txt`, copied to `$KEEP`) that counts elements against
`JOB_CAP` (proposed 56, J2), a done marker per run, squeue-based skipping of rows already queued or running, and no
`smoke-` tag outside the smoke job. `view` refuses any file whose hash differs from `resolution_inputs.json`. The
Obstacle, NoCurbRamp and Crosswalk pins beyond the backbone study's were hashed on the laptop copy of
`retrain-2026-10/`, whose split files and CurbRamp/SurfaceProblem files equal the klone pins. A mismatch on klone
stops `view` before any GPU job is spent.

Then on the laptop, after copying `$KEEP` back:

```bash
python analysis/resolution_study.py report --split-dir <retrain-2026-10 copy> --runs-dir <keep copy> \
    --public-cities <keep copy>/cities.json --bench <keep copy>/bench_nvidia_a40.json \
    --out docs/resolution-2026-11
```

It writes `decision.md`/`.json`, `runs.md` (per seed, per held-out deployment, uncounted runs with the reason),
`slices.md`, `cost.md`, `env.md` (the 224 rerun against round 3's own seed-42 predictions, not a gate), `gate.md`
and `provenance.json`. Provenance keys are relative (`split-dir/…`, `runs-dir/…`) with no absolute local paths. A run
is counted only if its records match its tag and the recipe (`consistency`): split hash, seed, `input_res` in the
run key, in the record and in the recorded preprocessing, `dinov2-large`, lr, epochs, best metric, target, per-device
batch, env versions, trained weights, and the loss divided by the steps. Predictions join the split one to one on
`label_uid` (`city:label_id`, `validate="one_to_one"`).

## What adopting 448 would need downstream

- **Serving.** Not checked here: whether sidewalk-ai-api builds its processor from the published
  `preprocessor_config.json` or hard-codes 224. Either way its train/serve parity test and its latency would have to
  be re-run at 448. `publish_to_hf.py` refuses a non-224 run until that is done (`check_backbone`).
- **Model cards and `explain.py`.** `hf_cards.py` writes the 224 preprocessing into every card, and `explain.py`
  assumes the 16 × 16 patch grid of a 224 view; both would need the size.
- **ONNX.** `export_onnx.py` takes its dummy input from the saved processor, so the graph is exported at the run's
  size; its parity check would have to be re-run.
- **Round 4** would train at 448 from the start; the context and backbone arms could then be revisited at 448.

## Threats to validity

1. **Native detail under 448.** About 36–46% of crops per type are under 448 px natively (Obstacle 0.455,
   NoCurbRamp 0.407, SurfaceProblem 0.361, CurbRamp 0.44, Crosswalk 0.441), so 448 upsamples them. 448 can only
   help a crop with detail above 224.
2. **Train/test resolution discrepancy.** `RandomResizedCrop` at scale 0.7–1.0 makes objects appear larger in
   training than at the centre crop of evaluation at any size (Touvron et al., FixRes). The discrepancy is the same
   at every size, so it does not favour an arm. A FixRes-style test-time size sweep is not run.
3. **Recipe fit.** lr 1e-5 and 25 epochs were set at 224. More tokens per image is not more images, so the
   recipe is held, but a larger input may want a different lr; this study does not tune it.
4. **Power** (above): 3 seeds detect only large effects.
5. **Reused held-out set.** The backbone, context, confident-learning and region-transfer studies read the same
   held-out deployments, so a winning arm's held-out number is optimistic. Round 4's headline should come from fresh
   data.
6. **City–class confound.** Pooled held-out AUC rewards between-deployment offsets; the endpoint is therefore
   within-city, and per-deployment Δ is reported.
7. **Multiplicity in slices.** Many slices, uncorrected; only the three texture slices enter the rule, and only as
   a harm screen.
8. **Environment.** The 224 arm is rerun in the study env, and `env.md` reports its drift from round 3 at seed 42.
   The torch-gated tests run in the smoke job and by hand, not in CI (no torch there).
9. **Preemption and time limit.** Resumes continue from epoch checkpoints under the same run key; `cost.md` reads
   each run's last attempt only, so a resumed run's step time is from a partial attempt.
10. **Storage and purge.** Weights live on scrubbed (21-day purge). `$KEEP` holds the small records needed to
    rebuild every table, but not to re-score a model.

## Hooks on shared files

This branch edits:

- `fine_tune_classifier.py`: `DEFAULT_INPUT_RES`/`INPUT_RES_CHOICES`, `resolution_settings`,
  `make_image_processor`, `train_crop_size`, `build_train_transform` (the old inline `Compose`, moved; tested equal),
  one spread in `run_key`, two argparse options and their refusals, the processor and transform lines in `main`, the
  `TrainingArguments` checkpointing spread, and additive `training_run.json` fields (`input_res`,
  `gradient_checkpointing`, `attn_implementation`; `backbone` carries the size).
- `backbones.py`: `input_sizes`, `tokens_at`, and an optional `input_res` on four functions. Every default is unchanged.
- `publish_to_hf.py`: one refusal in `check_backbone`; `tests/test_publish_to_hf.py`: one case.

## Decisions that are Jon's

The default this branch uses is listed with each; none blocks the smoke job.

- **J1. Seeds.** Default: 3 (as briefed), registered. Power at 3 seeds is low (above); 5 seeds would add 10 runs per
  arm (about +27 GPU-h at 224, +74–121 at 448). This must be decided before the main stage is read, not after.
- **J2. Job cap.** Default: 56 (smoke 1 + bench 1 + main 30 + up to 21 resumes + 3 spare). The 518 stage needs a
  raise.
- **J3. The 518 arm.** Default: not submitted (147–254 GPU-h). Submit with `ALLOW_518=1` only on Jon's word, ideally
  after reading 448.
- **J4. Crosswalk as a texture guard.** Default: included (6 runs, about 8–11 GPU-h) so that the paint-fading clause
  can be evaluated at all; its type-level numbers are descriptive. Alternative: drop it, and the paint-fading clause
  reads "not evaluable".
- **J5. Latency threshold.** Default: none pre-registered; the A40 table is descriptive.
- **J6. Reading "the texture slices do not fall".** Default: a harm screen, failing only when the seed-mean interval
  lies wholly below zero on test or held-out. A stricter reading (point estimate ≥ 0) would fail on noise in slices of
  66–303 rows.
- **J7. Memory fallbacks.** Default: batch 8 × 4, no checkpointing. If the smoke job shows a size does not fit, use
  `GC=1` first, then `BATCH=4`, recorded as a deviation.
