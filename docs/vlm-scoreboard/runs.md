# VLM scoreboard: klone run log (issue #63)

Every Slurm job behind the transformers legs and the production rows, kept by the runner as the jobs ran. Job ids
are klone's; wall times are Slurm `Elapsed`. Outputs carry private-city label ids, so they live outside the repo
(`/gscratch/scrubbed/jfroehli/vlm-scoreboard-2026-10/` on klone, `~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10/`
on the Mac); only this log, the code and the generated scoreboard are committed.

Launcher: `analysis/vlm_score.slurm` (`vlm_baseline.py score-torch` / `timing-torch` / `baseline --split-root`).
Held-out set: round-3 `test` + `test_heldout`, 13,144 crops. Prompt variant `definition` unless noted.

## Setup (2026-10-26)

- Env `vlm-scoreboard` on scrubbed: Python 3.12, torch 2.14.1+cu126, transformers 5.20.2, accelerate 1.15.0
  (built with uv like `rampnet-eval`). Molmo2 runs in the existing `molmo2` env (transformers 4.57.1); scikit-learn
  and matplotlib were added to it so `vlm_baseline.py` imports.
- Weights pre-downloaded on the login node with `HF_HUB_DISABLE_XET=1`, `*.safetensors|json|py|txt|model|tiktoken|jinja`
  only. klone's shared public IP hit the Hub's anonymous API rate limit (HTTP 429) on the first attempt; the
  downloader retries every 150 s, one process per repo.
- Thinking: `enable_thinking=False` for Qwen3.8, Gemma 4, MiniCPM-V 4.7 and Nemotron 3.5 (their templates have the
  switch); Qwen3-VL Instruct and Molmo2 have none. Muse Glimmer has no switch (the model picks a channel after
  `<|start|>assistant`), so the prompt is prefixed with the final-answer channel ` to=user<|message|>` and
  `reasoning_strength="low"`; Phi-4-reasoning-vision is prefixed with `<nothink>`, the mode its system prompt defines.

## Jobs

| job | leg | mode | types | partition | GPU | batch | wall | preemptions | outcome |
|---|---|---|---|---|---|---|---|---|---|

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
