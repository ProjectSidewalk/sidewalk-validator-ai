# VLM scoreboard: klone run log (issue #63)

Every Slurm job behind the transformers legs and the production rows, written by the runner as the jobs ran (all on
2026-10-26, account `makelab`, partition `ckpt-all`, `--requeue`). Wall times are Slurm `Elapsed`. The per-crop
outputs carry label ids of private deployments, so they live outside the repo
(`/gscratch/scrubbed/jfroehli/vlm-scoreboard-2026-10/` on klone, `~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10/`
on the Mac); only this log, the code and the generated scoreboard are committed. No label ids or deployment names
appear here.

Launcher: `analysis/vlm_score.slurm` (`vlm_baseline.py score-torch` / `timing-torch` / `baseline --split-root`).
Held-out set: round-3 `test` + `test_heldout`, 13,144 crops. Prompt variant `definition` throughout.

## Environment (reproducibility notes)

- **Env `vlm-scoreboard`** on `/gscratch/scrubbed/jfroehli/envs/`: system Python 3.12.15, torch 2.14.1+cu126,
  transformers 5.20.2, accelerate 1.15.0, built with uv as `rampnet-eval` was. Molmo2-8B and Phi-4-reasoning-vision
  run in the existing **`molmo2`** env (transformers 4.57.1; Phi's Hub code fails to import under transformers 5);
  scikit-learn and matplotlib were added to it so `vlm_baseline.py` imports. Production rows: the unchanged
  validator-ai env (transformers 4.55.0).
- **torch 2.14.1+cu126 `prod` is broken on klone's driver 580.178.04.** `Tensor.prod` on any CUDA tensor raised
  "CUDA driver error: invalid argument" (probe job 42380553, A40: int64, int32 and float32 `prod` all fail; `sum`
  and `cumprod` work). Qwen3-VL, Qwen3.8 and Molmo2 take `prod` of their image grids on the GPU, so they died at
  random. `vlm_baseline.patch_cuda_prod` probes once per process and, if `prod` fails, routes CUDA `prod` through
  the last element of `cumprod` (the same product); every meta file records `cuda_prod_patched` (true on every leg).
- **Triton needs `Python.h`**, which the system python3.12 lacks; Molmo2's rotary embedding compiles a Triton helper
  at first use. The launcher puts the headers of a uv-managed CPython 3.12 on `CPATH` and keeps the Triton and
  inductor caches on scrubbed (home is over quota).
- **Weights** were pre-downloaded on the login node (`HF_HUB_DISABLE_XET=1`, only
  `*.safetensors|json|py|txt|model|tiktoken|jinja`); jobs run with `HF_HUB_OFFLINE=1`. klone's shared IP hit the
  Hub's anonymous API limit (HTTP 429) at first; the downloader retries every 150 s, one process per repo
  (about 600 GB in about 1.5 h).
- **Thinking off.** `enable_thinking=False` for Qwen3.8, Gemma 4, MiniCPM-V 4.7 and Nemotron 3.5 (their templates
  have the switch; Qwen3.8/MiniCPM render an empty `<think></think>`, Gemma 4 an empty thought channel). Qwen3-VL
  Instruct and Molmo2 have no switch. Muse Glimmer has none either (the model picks a channel after
  `<|start|>assistant`): the prompt ends with the final-answer channel ` to=user<|message|>` and
  `reasoning_strength="low"`. Phi-4-reasoning-vision is prefixed with `<nothink>`, the mode its system prompt
  defines, and takes `<image>` placeholders in plain-text messages (its processor has no chat template).
- **slurmctld outage**, about 06:43–07:33 PDT: `sbatch` timed out but six of the timed-out submissions were in fact
  queued (42379345, -353, -363, -377, -434, -445); they were cancelled before running.

## Batch size and the batch check

Each type's first batch is scored again row by row at batch 1 and the differences are logged
(`batch_check` in the meta file; `scores/batch_invariance.csv` compares the 50 batch-1 timing crops with the full run).
**In bf16 the scores are not batch-invariant, and no padding is involved:** on 16 crops of identical length
(493 tokens) Qwen3-VL-8B differed between batch 16 and batch 1 by max 0.054 / median 0.0027 in `p_incorrect`
(max 0.25 in log-odds) in bf16, and by max 2e-5 in fp32 (probe 42384547); Gemma 4 12B, 8 crops: bf16 max log-odds
0.26, fp32 6e-4 (probe 42385398). So the brief's 1e-3 target is not reachable in bf16 at any batch size; the check
is kept as a measurement, and the run aborts only above 0.25 in p (0.5 for Gemma 4 31B), which would mean a padding
bug. Early runs used a 0.02 threshold and a one-row check; the aborts below are those, none fired at 0.25.

| leg | batch | GPU (full run) | batch check (max abs dp over the first batch, per type) | aborts |
|---|---|---|---|---|
| `qwen3vl-8b-bf16` | 16 | 1×A40 | CurbRamp 2e-4, NoCurbRamp <1e-4, Obstacle <1e-4, SurfaceProblem <1e-4 (one-row check, saturated rows); Crosswalk 0.047 | one, at the then 0.02 threshold (job 42381720, Crosswalk); resumed in 42384745 |
| `gemma4-12b-bf16` | 16 | 1×A40 | 0.011 max (one-row check) | none |
| `molmo2-8b` | 16 | 1×A40 | 0.0095 max (one-row check) | none |
| `qwen3vl-32b-bf16` | 16 | 2×L40S | 0.028–0.090 max, 0.002–0.009 median (16 rows per type) | one, at the then 0.02 threshold (job 42383962, CurbRamp, 0.028) |
| `gemma4-31b` | 16 | 2×A40 | 0.007–0.103 max, ≤0.005 median | none (threshold 0.5) |
| `muse-glimmer-30b` | 16 | 2×A40 | 0.010–0.024 max, 0.002–0.007 median | none |
| `minicpm-v4.7-35b-a3b` | 1 | 1×H200 | n/a (batch 1) | a batch-16 run (42380601) aborted on NoCurbRamp at 0.062 under the 0.02 threshold; rerun at batch 1 before the bf16 finding, kept |
| `qwen3.8-27b` | 1 | 2×A40 | n/a (batch 1) | a batch-16 run (42380949) aborted on NoCurbRamp at 0.030 under the 0.02 threshold; rerun at batch 1, kept |
| `phi4-rv-15b` | 1 | 1×A40 | n/a (batch 1; its smoke showed 0.011 at batch 8) | none |
| `gemma4-31b-fewshot8` | 4 | 2×L40S / 2×A40 (one job per type) | ≤1e-16 on the first batches (about 3.1k-token prompts) | none |
| `nemotron3.5-super-vl-120b` | 4 (planned) | 2×H200 | not run yet | see the job table |
| `production` | 1 (unbatched) | 1×A40 | n/a | none |

The batch-1 timing crops against the full runs (`scores/batch_invariance.csv`, 50 crops per leg, same GPU type):
median |dp| 0.0024 (Molmo2) to 0.012 (Gemma 4 12B), max 0.019 to 0.22 (Gemma 4 31B); legs run at batch 1
reproduce their batch-1 timing scores exactly (0.0), so batch-1 runs are deterministic on one GPU type.

## Jobs

| job | name | leg / purpose | GPU | wall | outcome |
|---|---|---|---|---|---|
| 42379345, -353, -363, -377, -434, -445 | vlm-smoke | duplicates queued during the slurmctld outage | — | — | cancelled before start |
| 42379515 | smk-qvl8 | smoke qwen3vl-8b-bf16 | A40 | 3:57 | failed: CUDA `prod` (driver bug) |
| 42379517 | smk-molmo | smoke molmo2-8b | A40 | 3:57 | failed: Triton missing `Python.h` |
| 42379519 | smk-qvl32 | smoke qwen3vl-32b-bf16 | H200 | 6:00 | failed: CUDA `prod` |
| 42379892 | prod-dino | production, all 13,144 crops | A40 | 14:06 | done |
| 42379897 | smk-q38 | smoke qwen3.8-27b | H200 | 4:18 | done |
| 42379898 | smk-mcpm | smoke minicpm-v4.7-35b-a3b | H200 | 4:15 | done |
| 42379932, 42379933 | smk-qvl8, smk-molmo | smokes again | A40 | 2:49 | failed: CUDA `prod` |
| 42380553 | wrap | probe: `prod`/`sum`/`cumprod` per dtype | A40 | 0:37 | found the `prod` failure |
| 42380569 | q38 | qwen3.8-27b full, batch 16 | H200 | 3:15 | failed: CUDA `prod` |
| 42380601 | mcpm | minicpm full, batch 16 | H200 | 8:18 | CurbRamp done, then the batch check aborted (0.062 > 0.02); superseded |
| 42380801 | smk-qvl8 | smoke, with the `prod` patch | A40 | 3:14 | done |
| 42380811 | smk-molmo | smoke | A40 | 3:14 | done |
| 42380812 | smk-qvl32 | smoke | H200 | 3:37 | done |
| 42380858 | smk-phi | smoke phi4-rv-15b | A40 | 3:13 | failed: Hub code finds its vision weights only in a local path |
| 42380949 | q38 | qwen3.8-27b full, batch 16 | H200 | 7:35 | CurbRamp done, then the batch check aborted (0.030 > 0.02); superseded |
| 42381651 | smk-g12 | smoke gemma4-12b-bf16 | L40S | 4:01 | done |
| 42381701 | mcpm | **minicpm-v4.7-35b-a3b full, batch 1** | H200 | 1:12:18 | done, 0 preemptions |
| 42381720 | qvl8 | qwen3vl-8b-bf16 full, batch 16 | A40 | 37:26 | four types done, Crosswalk's check aborted (0.047 > 0.02) |
| 42381723 | molmo | **molmo2-8b full, batch 16** | A40 | 1:09:31 | done, 0 preemptions |
| 42381764 | smk-phi | smoke phi4-rv-15b | L40S | 3:27 | done |
| 42382139 | g12 | **gemma4-12b-bf16 full, batch 16** | A40 | 41:02 | done, 0 preemptions |
| 42382172 | phi | **phi4-rv-15b full, batch 1** | A40 | 3:14:14 | done, 0 preemptions |
| 42382174, -175, -177, -178 | t-* | batch-1 timing (before `p` was saved) | A40/L40S | ~3:00 each | superseded by the reruns below |
| 42382210 | t-minicpm | batch-1 timing | H200 | 6:36 | done |
| 42382969, -971, -972, -973 | t-* | batch-1 timing qwen3vl-8b, gemma4-12b, molmo2, phi4 | L40S, L40S, L40S, A40 | 3:09–3:34 | done |
| 42383925, -926, -927 | t-* | batch-1 timing qwen3vl-8b, gemma4-12b, molmo2 on the full runs' GPU | A40 | 1:17–3:39 | done |
| 42383962 | qvl32 | qwen3vl-32b full, batch 16 | 2×L40S | 3:48 | batch check aborted on its first batch (0.028 > 0.02) |
| 42383963 | smk-g31 | smoke gemma4-31b | 2×L40S | 5:12 | done |
| 42383964 | smk-muse | smoke muse-glimmer-30b | 2×A40 | 4:30 | done |
| 42383965 | q38 | **qwen3.8-27b full, batch 1** | 2×A40 | 2:08:46 | done, 0 preemptions |
| 42384547 | wrap | probe: bf16 vs fp32 batch invariance, Qwen3-VL-8B | A40 | 1:49 | bf16 numerics, not padding |
| 42384745 | qvl8 | **qwen3vl-8b-bf16 resumed (Crosswalk)** | A40 | 2:32 | done |
| 42384776 | qvl32 | **qwen3vl-32b-bf16 full, batch 16** | 2×L40S | 1:03:08 | done, 0 preemptions |
| 42385398 | wrap | probe: bf16 vs fp32, Gemma 4 12B | 2×L40S | 4:01 | bf16 numerics |
| 42385925 | g31 | **gemma4-31b full, batch 16** | 2×A40 | 1:39:09 | done, 0 preemptions |
| 42385926 | muse | **muse-glimmer-30b full, batch 16** | 2×A40 | 2:17:14 | done, 0 preemptions |
| 42389414, -418, -426 | t-* | batch-1 timing gemma4-31b, muse, qwen3.8 | 2×A40 | 3:10–5:19 | done |
| 42389429 | t-qvl32 | batch-1 timing qwen3vl-32b | 2×L40S | 3:44 | done |
| 42395024 | g31-fs8 | 8-shot, all types, 2×A40 | 2×A40 | 9:42 | cancelled at 64 rows (4.2 s/crop; split per type instead) |
| 42396176, 42396190 | g31-fs8-h/-a | 8-shot split by GPU type | — | — | cancelled before or just after start |
| 42396336, -337, -339, -340, -341 | fs8-* | **gemma4-31b-fewshot8**, one job per type, batch 4 | 2×L40S (Obstacle, SurfaceProblem, NoCurbRamp), 2×A40 (CurbRamp, Crosswalk) | Crosswalk 46:35; others running | 2.5 s/crop on 2×L40S, 4.2 s/crop on 2×A40 |
| 42383195 | smk-nemo | smoke nemotron3.5-super-vl-120b, batch 4 | 2×H200 | preempted once while loading (10:15), done on the second start | loads in 25 min (`attn=default`: sdpa refused), 0.108 s/crop, 248 GB peak |
| 42399869 | nemo | **nemotron3.5-super-vl-120b full, batch 4** | 2×H200 | queued | |
