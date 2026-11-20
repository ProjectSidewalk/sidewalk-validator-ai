# Is DINOv2-L the right backbone? (November 2026)

Issue [#119](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/119). Every validator since 2025-08
fine-tunes `facebook/dinov2-large`, and the backbone has never been compared with anything. This study retrains the
round-3 validator on other backbones with the round-3 recipe, on the round-3 split, at three seeds each, against a
baseline retrained in the same environment, and decides with a rule written before any run.

## Status

**Pilot submitted (job `43406096`, 30 runs); no result read yet.** The GPU smoke test passed (job
`43404752`). What exists:

- [`backbones.py`](../backbones.py): the registry (pinned revisions), the fixed preprocessing, a DINOv3 classifier
  class, RampNet's ConvNeXt V2-B converted into the native `ConvNextV2ForImageClassification`, the weight prefetch.
- `fine_tune_classifier.py --backbone <key> [--learning-rate <lr>]` (requires `--output-dir`; refused together with
  `--context-arm`, `--region-conditioning` and `--init-from`). The default run, its run key and its outputs are
  unchanged.
- [`analysis/backbone_comparison.py`](../analysis/backbone_comparison.py) (`view`, `report`) and the klone chain
  `analysis/backbone_env.sh`, `backbone_prefetch.sh`, `backbone_submit.sh`, `backbone.slurm`.

The GPU smoke test and the pilot's job IDs are under [Runs](#runs). Every table under [Results](#results) reads "not
yet run" until `report` is run on the pilot's records.

## Why

DINOv2-L was the first backbone tried and the only one. Three things make a comparison worth its cost now: DINOv3
(2025) reports gains over DINOv2 at the same size on dense and fine-grained transfer; DINOv2 with registers
removes the high-norm artefact tokens that DINOv2-L shows on large uniform regions (sky, road), which are most of a
streetscape crop; and RampNet's ConvNeXt V2-B was trained on curb-ramp detection in street-level panoramas, the
closest pretraining to this task that exists. If none of them beats DINOv2-L by a margin worth a change, that is
worth knowing too: round 4 then keeps the current backbone with a recorded reason.

## Candidates

| key | source | revision | family | patch | prefix tokens | tokens / image | params (M) | input | gated | converted | eligible |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| `dinov2-large` | `facebook/dinov2-large` | `47b73eefe95e` | dinov2 | 14 | 1 | 256 | 304.4 | 256/224 | no | no | no |
| `dinov2-reg-large` | `facebook/dinov2-with-registers-large` | `e4c89a4e0558` | dinov2_with_registers | 14 | 5 | 256 | 304.4 | 256/224 | no | no | yes |
| `convnextv2-base` | `facebook/convnextv2-base-22k-384` | `53ec0af6e1bd` | convnextv2 | 32 (stride) | 0 | 49 | 88.7 | 256/224 | no | no | no |
| `rampnet-convnextv2-base` | `projectsidewalk/rampnet-model` | `606a11956743` | convnextv2 | 32 (stride) | 0 | 49 | 88.7 | 256/224 | no | yes | yes |
| `rampnet-crop-convnextv2-base` | `projectsidewalk/rampnet-crop-model` | `7aa79b8edb10` | convnextv2 | 32 (stride) | 0 | 49 | 88.7 | 256/224 | no | yes | no |
| `dinov3-vitl16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51` | dinov3_vit | 16 | 5 | 196 | 303.1 | 256/224 | yes | no | yes |
| `dinov3-vitl16-256` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51` | dinov3_vit | 16 | 5 | 256 | 303.1 | 292/256 | yes | no | no |
| `dinov3-vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bb` | dinov3_vit | 16 | 5 | 196 | 85.7 | 256/224 | yes | no | no |
| `dinov3-vits16` | `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c13799502` | dinov3_vit | 16 | 5 | 196 | 21.6 | 256/224 | yes | no | no |

(`python backbones.py list` prints this table; the full 40-character revisions are in `backbones.py`.)
Parameters are counted from the built classifier. The DINOv3 rows' weights are gated on HF; a token with access is
required to download them. None was on klone when the pilot was submitted, so **no DINOv3 arm is in the pilot**;
their code path was exercised at full ViT-L size with random weights in the smoke job (`SVA_BACKBONE_RANDOM_INIT=1`),
which a run's consistency check then refuses to count as a result. On 2026-11-19 the weights of `dinov3-vitl16`
(also used by `dinov3-vitl16-256`) and `dinov3-vitb16` were prefetched with a token into the HF cache, and the
`dinov3` stage (below) runs the 18 DINOv3 runs. Jobs read the cache offline, so they need no token. `rampnet-crop-convnextv2-base` is a registry row only (decision J4 below).

- **DINOv3** has no classification class in transformers 4.56, so `backbones.py` defines
  `SidewalkDinov3ForImageClassification` (`model_type` `sidewalk_dinov3_classifier`): `DINOv3ViTModel` under the
  attribute `dinov3_vit`, and a linear head on `[CLS, mean(patch tokens)]`, the pooling of
  `Dinov2ForImageClassification`; the four register tokens are left out of the mean. `backbones.register_auto()`
  makes `AutoModelForImageClassification` load it (`evaluate.py` and `export_onnx.py` call it).
- **RampNet** stores a timm ConvNeXt V2-B as `model.feature_extractor` (stem and four stages, 376 tensors) plus a
  4-tensor keypoint head. `backbones.convert_state_dict` maps the 376 one to one onto
  `ConvNextV2ForImageClassification`'s embeddings and encoder (GRN tensors reshaped from `(4C,)` to `(1, 1, 1, 4C)`),
  fails on any unmatched name or shape, and drops the head. The final layernorm comes from the `convnextv2-base`
  template. The converted directory also holds a two-class classifier (the template's), but it is a placeholder:
  `build_classifier` draws a new head for a converted row under the run's seed, as every other arm's head is drawn,
  so RampNet and its control differ only in stem and stage weights. (Until PR #121's fix round the saved head was
  loaded as is, so the pilot's RampNet runs share one head across seeds; see "Pilot deviations" below.) `convert-rampnet
  --verify-with-timm` checks the converted features against timm's on a seeded batch (max abs difference recorded
  in `conversion.json`).

## What is held fixed, and what is not

**Fixed:** the round-3 dataset and `split.csv`, the recipe (lr 1e-5, batch 8 × 4 accumulation, 25 epochs, warm-up
10%, best checkpoint by val macro-F1, hard targets, fp32), the evaluation rows, and the environment, which is the
same for every arm and the baseline.

- **Preprocessing.** Every arm gets the baseline's slow `BitImageProcessor`: resize the shortest edge to 256
  (bicubic), centre crop 224, ImageNet mean and std. DINOv3's own processor is fast-only and resizes straight to 224
  (a different field of view); serving parity rests on the slow PIL processor; and each checkpoint saves its
  processor, so evaluation, export and serving read the same one. `preprocess_sha256` in `training_run.json` records
  it. What still differs is the number of tokens per image (table above).
- **Loss scaling under gradient accumulation.** transformers 4.56 changed how the Trainer scales the loss for
  models whose `forward` takes `**kwargs` (it normalises by the items in the accumulated batch instead of dividing
  by the accumulation steps), and ConvNeXt V2's `forward` now does. `fine_tune_classifier.py` sets
  `model.accepts_loss_kwargs = False` on every model, the baseline included, which is what round 3 did under 4.55.0.
  A test checks that the first-step gradient norm at batch 2 × accumulation 4 equals that at batch 8 × accumulation
  1 for every family, and `training_run.json` records `model_accepts_loss_kwargs`.
- **DINOv3 positional augmentation.** DINOv3 jitters and rescales its RoPE patch coordinates in train mode. It is
  kept as released (library defaults; recorded under `pos_embed` in `training_run.json`); no other arm has a
  counterpart (decision J12). `backbones.build_classifier(..., pos_embed_aug="off")` turns it off.

**Varies:** the backbone, and in six explanatory rows the learning rate (3e-5).

## Setup

**Environment (J1).** An overlay venv on scrubbed over the round-3 conda env: the round-3 packages (Python 3.10,
torch 2.7.1, pandas 2.3.1) are imported from the base env, and `requirements-backbone.txt` adds transformers 4.56.2,
tokenizers 0.22 and timm 1.0.31, with every other package held at its installed version by a constraints file.
`analysis/backbone_env.sh` fails unless torch and pandas import from the base env and transformers from the venv.
`pip_freeze.txt` is in `$OUT` (its sha256 under [Runs](#runs)). Built 2026-11-19 on klone: torch 2.7.1+cu126 and
pandas 2.3.1 from the base env, transformers 4.56.2, tokenizers 0.22.2 and timm 1.0.31 from the venv
(huggingface-hub 0.34.4, safetensors 0.6.2, accelerate 1.10.0 unchanged). The baseline (`bb-dinov2-large`) is retrained in this
env, so every comparison is in-env; `env.md` compares it with round 3's own baseline (4.55.0) at each seed.

**Pinned inputs.** [`analysis/backbone_inputs.json`](../analysis/backbone_inputs.json) holds the sha256 of every
round-3 file the study reads: the five `split.csv` files, `deployment_priors.json`, the CurbRamp and SurfaceProblem
`manifest.csv` and `generation_run.json`, and round 3's seed 42/43/44 baseline metrics and predictions. `view`
refuses to build on a mismatch, every run's `run_key.json` must carry the pinned split hash, and `report` refuses a
`--split-dir` whose split differs.

**Storage.** klone paths (`$USER` = the account running the chain):

```
CLONE=/gscratch/scrubbed/$USER/sva-backbone-comparison          # this branch
OUT=/gscratch/scrubbed/$USER/backbone-comparison-2026-11         # views, runs, best weights, converted RampNet, logs
KEEP=/gscratch/makelab/$USER/backbone-comparison-2026-11-keep    # small run records only (under 1 GB)
R=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10  # round 3, read only
VENV=/gscratch/scrubbed/$USER/envs/sva-bb
HF_HOME=/gscratch/makelab/$USER/hf_cache                          # the four non-gated rows (3.2 GB, 1.9 GB of it new)
```

Nothing is written under `$R`. `view` builds `$OUT/data/<LT>/` with real copies of the small files (checked against
the pins after copying) and symlinks only for the crop directories `correct/` and `incorrect/`, so `evaluate.py`'s
tagged outputs land in the view. Scrubbed deletes files 21 days after their last write: the view, the venv and the
best weights (`$OUT/runs/<LT>/<TAG>/best/`, about 1.2 GB per ViT-L run) go then. The records in `$KEEP` are enough to
rebuild every table, but not to re-score or export a model. To rebuild what scrubbed loses: `analysis/backbone_env.sh`,
`analysis/backbone_submit.sh view`, and rerun the runs whose weights are needed.

## How to run

On klone, login node (git, small files, sbatch; the heavy work is in the jobs):

```bash
git clone --branch backbone-comparison https://github.com/ProjectSidewalk/sidewalk-validator-ai.git \
    /gscratch/scrubbed/$USER/sva-backbone-comparison
cd /gscratch/scrubbed/$USER/sva-backbone-comparison
analysis/backbone_env.sh                      # overlay venv; prints the three versions and where they import from
analysis/backbone_prefetch.sh                 # the four non-gated rows; with HF_TOKEN set, the DINOv3 rows too
#   with a token: HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
#                 GATED_KEYS="dinov3-vitl16,dinov3-vitl16-256,dinov3-vitb16" analysis/backbone_prefetch.sh
analysis/backbone_submit.sh view              # $OUT/data for CurbRamp and SurfaceProblem
analysis/backbone_submit.sh view smoke        # $OUT/smoke/data, seeded subsample (320/160/160/160, balanced)
analysis/backbone_submit.sh smoke             # one job, 2 h: convert RampNet, 5 one-epoch runs, guards, tests, report
MAX_PARALLEL=8 analysis/backbone_submit.sh pilot   # one array, 30 elements, at most 8 at a time
analysis/backbone_submit.sh dinov3            # 18 elements: 3 DINOv3 arms x 2 pilot types x 3 seeds
```

Every submit first writes 8 MiB to `$OUT` (a `touch` proves nothing on scrubbed), checks free space on scrubbed and
`/gscratch/makelab`, skips an arm the manifest lists as unavailable, refuses a `smoke-` tag outside the smoke job,
skips a row whose `$KEEP/<LT>/runs/<TAG>/done` exists, and refuses to go past 55 GPU jobs for the stream (array
elements counted, read from `$OUT/backbone_jobs.txt`; Jon's cap was 40 and he raised it to 55 on 2026-11-19 for the
DINOv3 runs). It submits with `EPOCHS=25` set explicitly and without the caller's conda state, and the job refuses
to train unless torch 2.7.1, transformers 4.56.2 and pandas 2.3.1 import. It appends the job id, element count,
stage, commit and the task file's sha256 to that ledger and copies the ledger, the task file,
`backbone_manifest.json`, `pip_freeze.txt`, `constraints.txt` and `cities.json` to `$KEEP`. `seeds`, `rest`, `resubmit` and `onnx` exist for later stages and are not part of the
pilot.

Each array element runs `run_one LT ARM SEED TAG LR` in `analysis/backbone.slurm` (`ckpt-all`, `--requeue`, any of
A40, L40, L40S, A100, H200): train (resumes after preemption), evaluate on the view (`--tag <TAG> --per-city`), copy
the small records to `$KEEP/<LT>/runs/<TAG>/`, copy the best checkpoint's weights, config, processor and trainer
state to `best/` and check them, and only then delete every `checkpoint-*`; a `done` marker in `$KEEP` makes a requeue
return at once. `gpu.txt` is appended on every attempt, so a requeue onto another GPU model is visible.

Tags: `bb-<key>` at seed 42, `bb-<key>-s<seed>`, `bb-<key>-lr3e5`; `smoke-<key>` and `smoke-rand-<key>` in the
smoke job. The in-env baseline is `bb-dinov2-large`.

Reading the results (laptop):

```bash
scp -r klone:/gscratch/makelab/$USER/backbone-comparison-2026-11-keep ~/Data/backbone-2026-11-keep
python analysis/backbone_comparison.py report --split-dir <a local copy of retrain-2026-10> \
    --runs-dir ~/Data/backbone-2026-11-keep --shares docs/deployment-prior-2026-11/shares.json \
    --label-types CurbRamp,SurfaceProblem --manifest <backbone_manifest.json> --pip-freeze <pip_freeze.txt> \
    --out docs/backbone-2026-11
```

`--split-dir` needs `<LT>/split.csv` and round 3's `predictions*.csv` (for `env.md` and the round-3 slice column);
the split hash is checked. `--public-cities` defaults to the live `/v3/api/cities` listing; private held-out
deployments are lettered and never named. Each run must pass a consistency check (split hash, and seed, backbone
and learning rate equal to its tag; real weights; loss divided by the accumulation steps) or it is listed as
INCOMPLETE and left out.

## The pre-registered decision rule

Written before any run (plan v2, signed off 2026-11-19); `analysis/backbone_comparison.py decide` applies it.

**Pilot:** CurbRamp and SurfaceProblem × seeds 42, 43, 44 × four arms: `dinov2-large` (in-env baseline),
`dinov2-reg-large`, `convnextv2-base` (control), `rampnet-convnextv2-base`. That is 24 runs. Every arm gets all
three seeds whatever its seed-42 number is: there is no screening stage, so no draw is used both to select an arm
and to judge it. Plus six learning-rate rows (below). The DINOv3 arms join with the same three seeds through the
`dinov3` stage (+18 runs for `dinov3-vitl16`, `-256` and `vitb16`), compared with the pilot's `dinov2-large` runs.

**Pilot deviations.** The pilot (job 43406096, commit b8944e8) ran before the fix round, so (a) its 8 RampNet rows
start every seed from one shared head (review B1; fixed in `build_classifier`, rerun is decision J16), and (b) a
preemption between deleting `checkpoint-*` and the `slimmed` marker would fail a finished element (review S6; fixed,
and a pilot element it hit is finished by `resubmit` from the fixed commit, which treats a complete `best/` with no
best checkpoint left as copied).

**Endpoints.** Primary: paired Δ held-out AUC against the same-seed in-env baseline on the same rows, read
together with the paired **within-city** Δ held-out AUC. Guards: Δ test AUC, Δ within-city test AUC, the gate
readout on `test`. Descriptive: each held-out deployment, Disagree at the gate, every type × severity and type ×
tag slice with its training count.

**Thresholds.** From the baseline's seed sd of held-out AUC (0.006 CurbRamp, 0.010 SurfaceProblem; n = 3 seeds
each, so the sds themselves are rough): the sd of a per-seed paired Δ is taken as √2 × that (0.0085 and 0.014), and
the bar for a three-seed mean is two standard errors, floored at 0.010: **+0.010 for CurbRamp, +0.016 for
SurfaceProblem**.

**Rule.**

- *Winner* — only arms in `ELIGIBLE` (`dinov2-reg-large`, `rampnet-convnextv2-base`, `dinov3-vitl16`). All of:
  1. on at least one pilot type, mean Δ held-out AUC over three seeds ≥ that type's threshold, with Δ > 0 in 3 of 3
     seeds, **and** mean within-city Δ held-out AUC > 0 on that type;
  2. on both pilot types, mean Δ held-out AUC ≥ −0.005, mean within-city Δ held-out AUC ≥ −0.005 and mean Δ test
     AUC ≥ −0.005;
  3. on both pilot types, under `majority2`, the Agree verdict's precision at the gate on `test` (mean over seeds)
     is ≥ 0.92 and not more than 0.01 below the in-env baseline's, with coverage not more than 2 points below the
     baseline's.
- *Between-deployment only*: condition 1's pooled bar is met but the within-city mean Δ is ≤ 0. Reported under
  that name; cannot win.
- *Other arms* (`convnextv2-base`, `dinov3-vitl16-256`, `dinov3-vitb16`) are explanatory. One is promoted to a
  winner only if it meets the rule with thresholds doubled (+0.020 / +0.032) and no eligible arm wins.
- *Non-inferior small* (any arm with under half the baseline's parameters): condition 2 and condition 3 hold over
  three seeds. Reported as a serving-cost option, not the round-4 recommendation unless Jon chooses it.
- *Several winners*: the larger sum of mean Δ held-out AUC over the two types; within 0.005, the arm that
  `AutoModelForImageClassification` loads with no custom code, then fewer parameters.
- *Confirmation* (later stage, Jon's call): the winner on the other three types at seed 42; the recommendation
  stands if no type has a Δ held-out interval entirely below zero and no type has a within-city Δ held-out interval
  entirely below zero.
- *No winner*: round 4 keeps `facebook/dinov2-large`; the doc records the null with its intervals.
- *RampNet reading*: its Δ against `convnextv2-base` as well as against the baseline. Its `test` numbers are marked
  "indirectly exposed" (threat 8); conclusions about it rest on held-out only.
- *Learning-rate rows* (explanatory, cannot win): `dinov2-large`, `convnextv2-base` and
  `rampnet-convnextv2-base` at lr 3e-5, seed 42, both pilot types (tags `…-lr3e5`), each compared with its own
  1e-5 seed-42 run and with the baseline at 3e-5. If an arm's 3e-5 run beats its 1e-5 run by at least the type's
  threshold on held-out AUC, the doc calls that arm's result "recipe-limited" and proposes a three-seed follow-up
  at 3e-5 for Jon.
- *Environment readout* (not a gate): `bb-dinov2-large` against round 3's `baseline` at the same seed; a mean
  difference beyond twice the seed sd on either type is called out at the top of the doc.
- *Multiplicity*: the family (eligible arms × two types, plus explanatory rows) is not corrected. `null.md` gives
  the expected number of null winners under this rule, and the doc states how many arms were compared.
- Slices never decide. A slice with ≥ `MIN_N` rows whose Δ interval is entirely below zero for a winner is listed
  under "where it got worse".

Computed before any run (`null_winner_rate`, 20,000 seeded draws, independent per-seed Δ with the sds above and the
gate condition assumed met): with three eligible arms, **0.021 expected null winners** (P(at least one) 0.021); at
the doubled thresholds, 0.00005. With DINOv3 not runnable, the pilot compares two eligible arms, so its null rate is
lower still.

Mechanics of the report (fixed with the rule): within-city AUC is the per-deployment AUC weighted by n_incorrect ×
n_correct over deployments with at least 10 labels of both classes (round 3's definition); its paired Δ has a 95%
interval from 1,000 panorama bootstrap draws resampled within each deployment (seed 0). Pooled Δ and slice Δ use the
existing panorama bootstrap (`confident_learning.overall_deltas`, `slices.paired_delta_table`). The gate readout is
`prior_rules.gate_rows` on each run's predictions under the `majority2` prior (decision) and `margin` (reference),
never in the same table; calibration is not part of the job.

## Runs

All from the clone at `b8944e84cc73` (branch `backbone-comparison`), in the env above (`pip_freeze.txt` sha256
`66aee6ea9199bde678e54d198dd000740d47d4fc0c401c5f4a5b9dea3c8337db`). Weights manifest
`$OUT/backbone_manifest.json` (sha256 `9574dd02196a…`): the four non-gated rows available, `dinov2-large`'s cached
`refs/main` equal to the pin; the DINOv3 rows skipped (no token). Storage read on 2026-11-19 before any job:
`/gscratch/makelab` 969 of 1,024 GB and 974,033 of 1,000,000 files; home 5 of 10 GB; an 8 MiB write to scrubbed
succeeded. The three rows not already cached added about 1.9 GB to the HF cache.

**GPU smoke test: passed on the first attempt.** Job `43404752` (`--time=02:00:00`, one job), A40 node, 41 min 56 s,
exit 0. Task file `$OUT/smoke/tasks_smoke.tsv` (sha256 `9d3fe6e62488…`); smoke subsample split sha256
`c5baa3ef3567…` (seed 0, 320/160/160/160 rows, balanced by class).

| path | outcome |
|---|---|
| RampNet conversion | 376 tensors mapped one to one, 4 head tensors dropped, 2 layernorm tensors from the template, new classifier; `verified_against_timm` true, max abs difference 3.8e-5 |
| `smoke-dinov2-large` | 1 epoch (10 steps), 480 predictions, all finite; `model_accepts_loss_kwargs` false; `resolved_revision` equal to the pin; `best/` holds the four files, no `checkpoint-*` left; `done` written; peak GPU memory 8.0 GB |
| `smoke-dinov2-reg-large` | same checks pass; 304.38M parameters; peak 8.1 GB |
| `smoke-convnextv2-base` | same checks pass; 87.69M parameters; peak 4.1 GB |
| `smoke-rampnet-convnextv2-base` | same checks pass on the converted weights; peak 4.1 GB |
| `smoke-rand-dinov3-vitl16` | random weights at full size: 303.13M parameters (registry 303.1M); RoPE augmentation as released (rescale 2.0); same checks pass; peak 7.3 GB |
| rerun of every line | each returned at the `done` guard |
| torch tests on the GPU node | `tests/test_backbones.py`, `tests/test_backbone_cli.py`: 40 passed, 3 skipped |
| `report --tag-prefix smoke-` | all 13 files written; the four real-weight runs complete; only `smoke-rand-dinov3-vitl16` INCOMPLETE (random init) |
| round-3 directory | nothing written under it since submission |

The smoke runs' wall time per step (28.8 s for DINOv2-L down to 2 s for RampNet's ConvNeXt) is dominated by model
loading and the first, cold read of the crops, so it does not restate the pilot's cost. The plan's estimate stands:
from round 3's timing (25 epochs on CurbRamp in 2.5 to 3 h on an L40S), about 3.5 to 4 h per ViT-L CurbRamp run and
5.5 to 6 h per SurfaceProblem run, less for ConvNeXt V2-B; roughly 140 GPU-hours on free `ckpt`. `cost.md` will
give the measured rate.

**Pilot: submitted 2026-11-19 (02:45 UTC on 11-20), not yet read.** Job `43406096`, one array `--array=1-30%8`,
job name `backbone-comparison-pilot`, `ckpt-all` with `--requeue`. Task file
`$OUT/tasks_pilot_20261120T024531Z.tsv` (sha256 `0d80a1fd70d5370b3117326196be67386c4ccd6ec8b887f8dfd478d4c36ec6e0`;
a copy is in `$KEEP`): the 24 main rows (seed 42 first, then 43, then 44) and the six 3e-5 rows. Command, from the
clone:

```bash
MAX_PARALLEL=8 analysis/backbone_submit.sh pilot    # sbatch analysis/backbone.slurm with STEP=train
```

GPU jobs used by this stream: 31 of 40 at the pilot (smoke 1, pilot 30); the cap is now 55, and the `dinov3` stage
takes 18 more (49 of 55), leaving 6 for resubmitting failed elements (`backbone_submit.sh resubmit`). Records land in `$KEEP/<LT>/runs/<TAG>/`; the ledger is `$KEEP/backbone_jobs.txt`.

## Results

| table | status |
|---|---|
| `arms.md` / `arms.json`: every run against the same-seed baseline, each held-out deployment | not yet run |
| `seeds.md`: mean ± sd over seeds | not yet run |
| `gate.md`: Agree and Disagree at the 0.92 gate, `majority2` and `margin` | not yet run |
| `slices.md`: type × severity and type × tag, with training counts and round 3's own AUC | not yet run |
| `cost.md`: parameters, tokens, model class, GPU, seconds per step, peak memory | not yet run |
| `env.md`: in-env baseline against round 3 | not yet run |
| `lr.md`: learning-rate rows | not yet run |
| `rampnet.md`: RampNet against its control, and the exposure table | not yet run |
| `decision.md` / `decision.json`: each arm's status and the recommendation | not yet run |
| `null.md`: expected null winners | written by `report` |

`report --out docs/backbone-2026-11` writes these files next to this page.

## What a different backbone would need downstream

Nothing below is built; it is what a winner would cost.

- sidewalk-ai-api loads `AutoModelForImageClassification` under transformers 5.20.2 with a PIL processor. A
  ConvNeXt V2 or DINOv2-with-registers checkpoint loads with no code change and carries the processor config
  already served.
- DINOv3 has no stock classification class in transformers 4.56 to 5.23, so serving would need the small wrapper
  class registered at start-up, or the ONNX export. 5.x introduces `DINOv3ViTEncoder`, so tensor names move
  between 4.56 and 5.x; the parity stage (`analysis/serving_parity.py`) would have to cover it.
- For the gated models: weights are gated on HF; a token with access is required. A fine-tuned checkpoint loads
  from its own files without hub access.
- `explain.py`, the model cards and `publish_to_hf.py` assume DINOv2 and would need the new patch grid and class
  names before a publish. Until then `explain.py` refuses a non-DINOv2 checkpoint and `publish_to_hf.py` refuses a
  run whose `training_run.json` names another backbone or learning rate, or whose best checkpoint is not `dinov2`.
- ONNX: the tiny-model tests export ConvNeXt V2 and the DINOv3 wrapper at opset 17 with agreement under 1e-4.
  `Dinov2WithRegisters` does not export at opset 17 (its position-embedding interpolation uses antialiased bicubic
  upsampling, which the exporter lacks); a winner from that family would need a newer opset or a fixed-size
  interpolation before an ONNX path.

## Threats to validity

1. **Recipe fairness.** lr 1e-5 was chosen for DINOv2-L; ConvNeXt may want more. The learning-rate rows bound this
   for one value only.
2. **Loss scaling.** All arms use "divide by the accumulation steps"; under transformers 4.56.2 the stock DINOv2
   classes would otherwise normalise by items in the accumulated batch. The two agree except on a final ragged
   batch; `env.md` reports the baseline's drift from round 3.
3. **DINOv3 positional augmentation** in train mode is kept as released and has no counterpart in the other arms.
4. **Tokens and field of view** differ by family at 224 input; ConvNeXt V2-B was pretrained at 384.
5. **City–class confound.** Pooled held-out AUC rewards between-deployment offsets; the rule therefore requires a
   within-city gain, and per-deployment Δ is reported.
6. **Seed noise and multiplicity.** Thresholds come from sds estimated on three seeds; the family is uncorrected;
   `null.md` quantifies the null winner rate.
7. **Reused held-out set.** Context arms, confident learning, region transfer and this study all read the same
   three held-out deployments, so the chosen arm's held-out number is optimistic. Round 4's headline should come
   from fresh data.
8. **RampNet exposure.** Stage two never trained on Project Sidewalk labels, but its keypoints came from a crop
   model that trained on CurbRamp labels of 12 deployments, all now in round 3's train and test splits
   ([`analysis/backbone_rampnet_exposure.json`](../analysis/backbone_rampnet_exposure.json)). Its `test` numbers are
   indirectly exposed; none of the held-out deployments is among the 12.
9. **RampNet conversion.** The bijection, the shape check and the timm parity check make a wrong map fail loudly.
10. **Environment.** The study env is the round-3 env plus transformers 4.56.2, tokenizers and timm; the baseline is
    retrained inside it.
11. **Storage and purge.** Weights and views live on scrubbed and are deleted 21 days after their last write; the
    small records in `$KEEP` are enough to rebuild every table but not to re-score or export a model.
12. **ONNX export of DINOv3** (RoPE from the input shape) is untested beyond the tiny-model test.
13. **Hot shared file.** `fine_tune_classifier.py` is also edited by the context-arms and round-4 streams (below).

## Hooks on shared files

So the other streams can sequence their merges. This branch edits:

- `fine_tune_classifier.py`: the argparse block (`--backbone`, `--backbone-dir`, `--learning-rate` and their
  checks), `run_key` (one spread of `backbone_run_settings`), the processor line
  (`backbones.build_image_processor`), the `build_model` tail, one `model.accepts_loss_kwargs = False` line after
  the model is built (applies to every stream's runs; a no-op under transformers 4.55.0), `learning_rate=` with
  `DEFAULT_LEARNING_RATE`, and additive `training_run.json` fields (`backbone`, `learning_rate`,
  `model_accepts_loss_kwargs`, `peak_gpu_memory_bytes`).
- `evaluate.py` and `export_onnx.py`: one `backbones.register_auto()` call each (two in `export_onnx.py`).
- `explain.py`: the `require_dinov2` guard.
- `publish_to_hf.py`: `check_backbone`.
- `tests/test_fine_tune_retype.py` (the recipe-constant assertion) and `tests/test_fine_tune_resume.py` (run-key
  tests for the new options).

The context stream's one-word addition of `side_info_nohq` to the `--context-arm` choices touches the same argparse
hunk and should merge first; this branch then rebases.

## Decisions that are Jon's

Listed with the default this branch uses; none blocks the pilot. Numbered as in the PR.

- **J1. Environment.** Default: the overlay venv above, baseline retrained inside it. Alternative: bump the one env.
- **J2. HF access for DINOv3.** Done 2026-11-19: Jon prefetched the weights with his token and raised the cap to 55;
  the `dinov3` stage was submitted the same night.
- **J3. Preprocessing.** Default: the baseline's 256/224 slow processor for every arm.
- **J4. RampNet weights.** Default: stage two only. The crop-model row trained directly on labels of deployments in
  round 3's `test` split, so only its held-out numbers would be readable.
- **J5. Smaller arm.** Default: `dinov3-vitb16` when DINOv3 becomes runnable.
- **J6. Learning-rate rows.** Default: the six 3e-5 rows in the pilot, explanatory only.
- **J8. Thresholds.** Default: the rule above.
- **J9. Confirmation seeds.** Default: seed 42 on the other three types for a winner; replicates only where an
  interval straddles zero.
- **J10. One backbone for all five types?** Default: yes.
- **J11. Merge order** across the three streams (above).
- **J12. DINOv3 positional augmentation.** Default: as released. Alternative: off.
- **J13. More families.** Not built. SigLIP 2 needs its own normalisation and mean pooling, so it breaks the
  fixed-preprocessing design and would be a second PR.
- **J16. The pilot's RampNet rows (shared head).** Default: rerun the 8 rows after the pilot finishes, from the
  fixed commit, under the same tags (the report parses only the planned tags). First move all three places a run
  lives aside, or `run_one` finds `evaluated`/`slimmed` and copies the old outputs back: `$RUNS/<LT>/<TAG>/`,
  `$KEEP/<LT>/runs/<TAG>/`, and the tagged files `$DATA/<LT>/{metrics,predictions,per_city,slices,report}_<TAG>.*`.
  Then `resubmit` a task file of those 8 rows. That needs about 8 more GPU jobs than the 55 cap leaves. Alternative: keep them and report the deviation beside every RampNet number.
