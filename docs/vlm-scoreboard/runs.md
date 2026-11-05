# VLM scoreboard: klone run log (issue #63)

Every Slurm job behind the transformers, vLLM and production legs, written by the runner as the jobs ran (2026-10-26/27,
account `makelab`, partition `ckpt-all`, `--requeue`). Wall times are Slurm `Elapsed`. The per-crop
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
| `nemotron3.5-super-vl-120b` | 4 | 2×H200 | 0.007–0.111 max, 0.003–0.038 median (4 rows per type: the first batch only) | none |
| `production` | 1 (unbatched) | 1×A40 | n/a | none |

The batch-1 timing crops against the full runs (`scores/batch_invariance.csv`, 50 crops per leg, same GPU type):
median |dp| 0.0001 (Gemma 4 31B) to 0.012 (Gemma 4 12B), max 0.019 (Molmo2) to 0.22 (Gemma 4 31B); legs run at batch 1
reproduce their batch-1 timing scores exactly (0.0), so batch-1 runs are deterministic on one GPU type.

MiniCPM-V 4.7's batch-1 timing run (job 42382210, 0.964 s/crop on g3130) is 3.7× slower than its full run at the
same batch size on g3127 (0.264 s/crop), on the same crops and software; the cause is unknown (probably a shared
node), and the full-run row is the one to quote (the board picks it: `run_speed` prefers a full run on a tie).

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
| 42396336 | fs8-Obst | **gemma4-31b-fewshot8** Obstacle, batch 4 | 2×L40S | 3:00:40 | done, 0 preemptions |
| 42396337 | fs8-Surf | **gemma4-31b-fewshot8** SurfaceProblem | 2×L40S | 2:24:20 | done, 0 preemptions |
| 42396339 | fs8-NoCu | **gemma4-31b-fewshot8** NoCurbRamp | 2×L40S | 2:31:48 | done, 0 preemptions |
| 42396340 | fs8-Curb | **gemma4-31b-fewshot8** CurbRamp (64 rows came from 42395024) | 2×A40, then 2×L40S | 1:42 + 0:37 | done; preempted once at 13:23 and resumed (Slurm restart count 3) |
| 42396341 | fs8-Cros | **gemma4-31b-fewshot8** Crosswalk | 2×A40 | 46:35 | done, 0 preemptions |
| 42383195 | smk-nemo | smoke nemotron3.5-super-vl-120b, batch 4 | 2×H200 | 35:21 (second start) | preempted once while loading, then done: loads in 25 min (`attn=default`; sdpa refused), 0.108 s/crop, 248 GB peak |
| 42399869 | nemo | nemotron3.5-super-vl-120b full, batch 4 | 2×H200 | — | preempted while loading the weights; cancelled and resubmitted with parallel loading |
| 42401600 | nemo | **nemotron3.5-super-vl-120b full**, batch 4, `HF_ENABLE_PARALLEL_LOADING` | 2×H200 | 31:23 (third start) | preempted while loading at 14:15, requeued, ran 17:25–17:56 on g3127 and scored 1,728 CurbRamp rows, then **FAILED with `Disk quota exceeded`** writing the meta file on `/gscratch/scrubbed` (the shared fileset was at 99%; writes succeeded again at 18:37, so the quota hit was transient) |
| 42413443 | nemo | **nemotron3.5-super-vl-120b full**, resubmitted 18:40 after the quota failure; resumes from the 1,728 rows already in CurbRamp.csv | 2×H200 | 20:53 | ran 18:47–19:08 on g3128, loaded in 19 min, scored 64 more rows at 0.203 s/crop, then **FAILED again with `Disk quota exceeded`** on the CSV append |
| 42454116 | nemo | **nemotron3.5-super-vl-120b full**, `OUT=/gscratch/makelab/jfroehli/vlm-out` (the 1,792 scored rows copied there); submitted 2026-10-27 05:48 | 2×H200 | 41:16 | **done**, 0 preemptions: ran 06:55–07:36 on g3132 right after GLM released the node, loaded in 19 min, scored the remaining 11,352 crops at 0.105 s/crop (peak 248 GB). Five types from four job starts; `n_crops` 13,136 in timing.csv is the last partial batch excluded |

## vLLM

GLM-5.3-Flash (and a Qwen3-VL-8B smoke through the same path), served by vLLM in Apptainer and scored from the first
generated token's top-20 log-probs: `analysis/vlm_vllm.py` (client) and `analysis/vlm_vllm.slurm` (launcher). Outputs
on klone under `/gscratch/scrubbed/jfroehli/vlm-scoreboard-2026-10/` (`scores/glm5.3-flash/` for the full run,
`smoke-vllm/` for smokes), staged on the Mac under `~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10/scores/`.

### Setup (2026-10-26)

- Image: `vllm/vllm-openai:v0.32.0-x86_64` (vLLM 0.32.0, CUDA 13.0, FlashInfer 0.7.0.post1), pulled to
  `/gscratch/scrubbed/jfroehli/containers/` in a CPU job (`apptainer pull` on a login node risks the reaper). The
  recipe (https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) asks for vLLM ≥ 0.29.0 and FlashInfer ≥ 0.6.18. The
  `-cu129` variant of the same release was tried first and dies at import: its torchcodec links `libnvrtc.so.13`.
  klone's H200 driver is 580.178, which runs CUDA 13.0.
- Weights: `zai-org/GLM-5.3-Flash` revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, 306 GiB FP8, 69 files, downloaded
  on the login node in 2 h 10 min (`HF_HUB_DISABLE_XET=1`, safetensors/json/py/txt/jinja only).
- Serve flags: TP 4, `--max-model-len 8192`, `--limit-mm-per-prompt {"image":1,"video":0}`, `--max-logprobs 20`, BF16 KV
  cache (the recipe: Hopper cannot use FP8 KV for this model), no MTP speculative decoding (one token is generated).
- Thinking: GLM-5.3-Flash's template always opens `<think>` and has no off switch (the recipe: "Thinking is always
  on"). The client sends an assistant turn `<think></think>` with `continue_final_message=true`, which the template
  renders as an empty, closed think block, plus `reasoning_effort=low`. The rendered prompt tail from vLLM's
  `/tokenize` is recorded in each meta file.
- Client env: `/gscratch/scrubbed/jfroehli/envs/vllm-client` (Python 3.12, numpy, pandas, scikit-learn, matplotlib,
  pillow). The client runs on the host and posts each crop to the server as a base64 data URL.
- Three klone traps, each fixed in the launcher: (1) klone exports `http_proxy`, and its `no_proxy` omits `127.0.0.1`,
  so the `/health` probe went to the proxy and a healthy server timed out (job 42382723). (2) Apptainer's `--env`
  splits on commas, so a 4-GPU `CUDA_VISIBLE_DEVICES=0,1,2,3` broke the 4×H200 launch (job 42390223). (3) The first
  successful smoke used the transformers leg's `scores/` directory and model name. It appended nothing to that leg's
  CSV, but it merged its keys into that leg's `qwen3vl-8b-bf16/CurbRamp.meta.json`. That file was restored by hand
  (the added keys were removed; `gpu`/`batch_size` were reset to the CSV's `NVIDIA A40`/16). The client now refuses to
  write into a CSV or meta file it did not create.

### Smoke agreement: vLLM vs transformers, Qwen3-VL-8B-Instruct bf16, 40 CurbRamp crops

The same 40 crops (`--limit 40`, first 20 per class by `label_uid`) scored by both paths. The transformers side is the
runner's full CurbRamp run, on an A40 at batch 16; the vLLM side ran on 1×H200 at concurrency 8.

| measure | value |
|---|---|
| mean / median / max \|Δ p_incorrect\| | 0.009 / 0.0007 / 0.060 |
| crops with \|Δ\| > 0.02 | 8 of 40 |
| mean / max \|Δ log-odds\| | 0.17 / 0.50 |
| same verdict at 0.5 | 39 of 40 |
| same top token, same prompt token count | 40 of 40, 40 of 40 |
| top-20 mass outside (vLLM) | mean 1.1e-7, max 1.1e-6; no missing-side floors |

The prompt and image token counts match exactly, so preprocessing agrees. The residual is bf16 kernel noise:
FlashAttention 3 in vLLM against sdpa in transformers. In probability it is largest near 0.5. The mean is inside the
~0.02 expectation; the tail is not.

### Jobs

Wait = Slurm `Start − Submit`. All on `ckpt-all`, account `makelab`.

| job | what | GPU | wait | wall | preemptions | outcome |
|---|---|---|---|---|---|---|
| 42378643 | `apptainer pull` v0.32.0-cu129 | CPU, 16 cores | 0 min | 1:09:48 | 0 | image built; unusable (libnvrtc.so.13) |
| 42379510 | smoke Qwen3-VL-8B, cu129 image | 1×H200 | 7 min | 0:01:11 | 0 | failed: torchcodec import |
| 42380061 | `apptainer pull` v0.32.0 (CUDA 13) | CPU, 16 cores | 3 min | 0:44:22 | 0 | image built |
| 42382723 | smoke Qwen3-VL-8B | 1×H200 | 37 min | 1:00:26 | 0 | timeout: health probe behind proxy |
| 42390219 | smoke Qwen3-VL-8B | 1×H200 | 7 min | 0:03:39 | 0 | collided with the transformers leg's dir (above) |
| 42390223 | smoke GLM-5.3-Flash, 40/type | 4×H200 | 9 min | 0:00:26 | 0 | failed: `--env` comma split |
| 42391263 | smoke Qwen3-VL-8B, 40 CurbRamp | 1×H200 | 10 min | 0:07:21 | 0 | OK; server healthy in 180 s; 0.075 s/crop at concurrency 8 |
| 42391323 | smoke GLM-5.3-Flash, 40/type, `OUT=smoke-vllm`; submitted 2026-10-26 10:22 | 4×H200 | 8 h 34 min (to the first start) | — | 1 | started 18:56 on g3127 and was **preempted at 18:58** while loading the tokenizer; requeued; cancelled 2026-10-27 06:05 once the full run 42413444 was running (the full run's meta answers the same prefill question) |
| 42413444 | **full GLM-5.3-Flash**, launcher defaults (timing at concurrency 1, then all 13,144 crops); submitted 2026-10-26 18:40 | 4×H200 | about 11 h (started 2026-10-27 05:57 on g3132) | 0:55:13 | 0 | **done** (05:57:38–06:52:51): 0.256 s/crop at concurrency 1, 0.031 s/crop at concurrency 32. Weights loaded in 30 min, first-use kernel JIT (TileLang, FlashInfer) another 17 min, healthy after 2,823 s, then all 13,144 crops in 7 min. Every rendered prompt ends `<\|assistant\|><think></think>`; no yes/no side fell outside the top-20 log-probs (`n_answer_floor` empty); `top20_mass_outside` ≤ 5.2e-4; the meta carries `answer_floor_rule`, so the corrected client ran. Wrote to scrubbed without hitting the quota |
| 42454117 | **full GLM-5.3-Flash**, same defaults, `OUT=/gscratch/makelab/jfroehli/vlm-out/scores-vllm`; submitted 2026-10-27 05:48 | 4×H200 | — | — | — | hedge against the scrubbed quota; never started: cancelled while pending at 06:45, once 42413444 was writing rows cleanly |

## Notes

8-shot arm: the leg with the best mean per-type AUC among the finished zero-shot legs when it was launched was
`gemma4-31b`. Examples: 4 correct + 4 incorrect `train` crops per type, seed 20261021, interleaved correct/incorrect,
the same eight for every query of a type (the chosen `label_uid`s are in each type's meta file, outside the repo).
No query shared a panorama with its type's examples, so no example was swapped (`fewshot_swaps` = 0 for every run).
Prompts are about 3,100 tokens (nine images), so 2.5 s/crop on 2×L40S and 4.2 s/crop on 2×A40 at batch 4.

**`/gscratch/scrubbed` quota (2026-10-26/27).** Both Nemotron attempts on 26 Oct died with `OSError: [Errno 122] Disk quota exceeded` while appending to their CSV on scrubbed, 20–30 min into the job, after writes had succeeded for several chunks. `hyakstorage /gscratch/scrubbed` shows the shared fileset at 98% of its 153,600 GB quota (151,155 GB used), so writes fail whenever other users push it over the line and succeed again when they free space. Reads (the HF weight cache) are unaffected. Per-crop outputs for the remaining legs therefore go to `/gscratch/makelab/jfroehli/vlm-out` (`OUT=`), which has its own quota, and are staged to the Mac from there.

Nemotron: three preemptions in total (smoke once, full run twice), each during the 25-minute weight load on a
ckpt H200 pair. It was left queued on ckpt rather than moved to the lab's `gpu-l40s` allocation, which has no H200
and would need a whole 8×L40S node to hold the 249 GB of weights.

## Where it stands (2026-10-27 07:50)

- Every leg is in: 14 klone legs on all 13,144 crops (GLM-5.3-Flash 0.754, Nemotron 3.5 Super VL 0.738). Both staged
  to the Mac, `timing.csv` regenerated with `timing-summary`.

At the previous check (2026-10-26 15:05) the vLLM client, launcher, tests and container path were done and the
Qwen3-VL-8B smoke through vLLM agreed with transformers; the GLM-5.3-Flash smoke was unread and the full GLM run
waited on a reopened SSH master (Duo). Both were settled on 27 Oct by jobs 42391323 and 42413444 above.
