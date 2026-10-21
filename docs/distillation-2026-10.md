# Cheaper validators: ONNX, int8, and the existing tiny models (October 2026)

Part of [#20](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/20). Each production validator is a
304M-parameter DINOv2 ViT-L (1.22 GB of fp32 weights). This note measures what can be had **without training**:
exporting the five production checkpoints to ONNX, quantizing them to int8, and scoring the five existing
`projectsidewalk/quantized_*_dinov2_tiny_onnx` models on the same held-out crops. It also records what is known about
where those tiny models came from. The ViT-S fine-tune and distillation the issue asks for need a GPU and a change to
`fine_tune_classifier.py`; they are written up as a recipe at the end, not run.

All counts below are over the **2,877 unique test crops from public deployments** (3,041 crop rows: CurbRamp reuses
the 164 Crosswalk crops), each scored **one crop per call**, as the API does.

**Short answer.**

- **Dynamic int8 (`quantize_dynamic`) is not safe to serve.** It cuts the file to 310 MB but 228 of the 3,041 crop
  rows (7.5%) vote differently from PyTorch, and the vote count falls from 73 to 35 for Obstacle and from 237 to 129
  for SurfaceProblem. Per-channel scales do not fix it (Obstacle: 36 changes, 47 votes). Its scores also depend on
  which crops share a batch (below), so it must be scored one crop at a time.
- **Weight-only int8 (`MatMulNBits`, fp32 activations) keeps nearly every vote.** 321 MB; 10 of 3,041 rows (0.3%)
  vote differently, all Agree verdicts sitting on a map step, most of them moved by under 1e-6 in confidence, and
  accuracy and AUC are unchanged to the third decimal. On this CPU it is slower than PyTorch fp32 and than ONNX fp32,
  so it buys file size (download, memory, browser), not speed.
- **ONNX fp32** reproduces PyTorch to 5e-5 in probability and changes no vote, but onnxruntime is slower than
  PyTorch on this Apple CPU: 1.1–2.1× at batch 1 and about 3× per crop at batch 16.
- **The existing tiny models are ViT-S** (22M parameters, 25 MB, about 10× faster than ViT-L int8, and 9–21× faster
  than PyTorch fp32, on this CPU). On these crops their accuracy and AUC are within a few points of ViT-L, above it
  on three of four types, but their verdicts differ from production's on 6–31% of crops, their training data is
  undocumented and very probably overlaps these test labels, and their provenance is not good enough to ship. They
  are evidence that a ViT-S is worth training properly, not a drop-in replacement.

## How the numbers were made

All numbers are from `export_onnx.py` (this PR), run 2026-10-20 on an Apple M1 Max (8 performance + 2 efficiency
cores, 32 GB), macOS 15.5, Python 3.12.14, torch 2.14.1, transformers 4.55.0 (requirements.txt's pin), onnx 1.23.2,
onnxruntime 1.31.0, onnx-ir 1.0.0. In the commands, `T` is the lower-case type (`crosswalk`, `curbramp`, `obstacle`,
`surfaceproblem`, `nocurbramp`), `Type` its CamelCase name, `REV` the production revision (table below), `TINY` the
tiny repo's type (`crosswalk`, `curb_ramp`, `obstacle`, `surface_problem`, `no_curb_ramp`), and `cities.json` the
`/v3/api/cities` response saved with the 2026-10-17 production pull (it is not committed; any current copy of the
listing works).

```
python export_onnx.py export --model projectsidewalk/sidewalk-validator-ai-$T --revision $REV --out models/$T
python export_onnx.py quantize models/$T/model.onnx                        # dynamic -> models/$T/model_int8.onnx
python export_onnx.py quantize models/$T/model.onnx --method weight-only   # -> models/$T/model_int8_wo.onnx
python export_onnx.py quantize models/obstacle/model.onnx --per-channel --out models/obstacle/model_int8_pc.onnx
hf download projectsidewalk/quantized_${TINY}_dinov2_tiny_onnx --local-dir tiny/$TINY
hf download projectsidewalk/sidewalk-validator-ai-dataset-$T --repo-type dataset --include "test/*" --local-dir data/$T
python export_onnx.py bench --label-type $Type --crops data/$T --model projectsidewalk/sidewalk-validator-ai-$T \
    --revision $REV --onnx onnx-fp32=models/$T/model.onnx --onnx onnx-int8=models/$T/model_int8.onnx \
    --onnx onnx-int8-weightonly=models/$T/model_int8_wo.onnx \
    --onnx tiny-int8-existing=tiny/$TINY/onnx/model_quantized.onnx \
    --accuracy-map ../sidewalk-ai-api/accuracy_mappings/$T.json --cities-json cities.json \
    --mps --threads 4 --out bench/
    # CurbRamp adds --no-accuracy; Obstacle adds --onnx onnx-int8-perchannel=models/obstacle/model_int8_pc.onnx
python export_onnx.py table bench/bench_{Crosswalk,CurbRamp,Obstacle,SurfaceProblem,NoCurbRamp}.csv --columns \
    label_type,variant,n_crops,size_mb,params_m,accuracy,macro_f1,auc,r_at_p95_correct,r_at_p95_incorrect,vs_ref_max_abs_dprob,vs_ref_verdict_flips,vs_ref_votes,vs_ref_vote_changes,b16_vs_b1_max_abs_dprob
python export_onnx.py table bench/bench_{Crosswalk,CurbRamp,Obstacle,SurfaceProblem,NoCurbRamp}.csv --columns \
    label_type,variant,lat_b1_ms,ms_per_crop_b16,load_avg_1m
```

The five `bench_<Type>.csv` files behind both tables are in [`distillation-2026-10/`](distillation-2026-10) (paths
made relative), and both tables regenerate exactly from them with the two `table` commands. The per-crop scores stay
outside the repo.

| type | model revision (sidewalk-ai-api `model_map.py`) | public test crops (correct / incorrect) | dataset revision |
|---|---|---|---|
| Crosswalk | `b7936e5bd17a4b96db301f954e5d3a5fe9eaf361` | 164 (69 / 95) | `1be513a` |
| CurbRamp | `969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a` | 164, the Crosswalk crops (see below) | `d396ee4` |
| Obstacle | `8f0fe8fc48748fac35098e74c1848d015126deef` | 647 (401 / 246) | `9fdc3c3` |
| SurfaceProblem | `0e1439daa5667c8a81f63e75f835bd51221df4c2` | 784 (536 / 248) | `23f32c0` |
| NoCurbRamp | `9f3f3bb2f40864a468fb65e43644ee5ba54e3e19` | 1,282 (762 / 520) | `e737c59` |

- **Crops.** The `test/` split of each public dataset repo. Hugging Face rate-limited this IP during the run, so the
  crops were taken from each dataset's auto-converted parquet (`/api/datasets/<repo>/parquet/default/test/*.parquet`),
  one request per type, instead of one request per file. The extracted file names match each repo's file list, and
  a separate review checked every image byte-for-byte against an LFS-verified download. `hf download ... --include
  "test/*"` gives the same directory.
- **Public deployments only.** The published test splits also include crops from deployments that are private today
  (530 of the 3,407 unique test crops). `bench --cities-json` drops them before any number is computed: a crop's city
  (the file-name prefix, either the schema's city or the deployment's subdomain) must match a public deployment in
  `/v3/api/cities`, with `calibration.per_city`'s matching rule and `data_generation.discover_cities`' subdomains.
  Only the dropped count is recorded (`n_crops_dropped_not_public`).
- **These test splits are the old file-level splits** (issue [#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)):
  crops of the same panorama can sit in train and test, so the absolute accuracy below flatters every model. It is
  fine for comparing variants of the same model on the same images, which is what this note does.
- **CurbRamp has no test set of its own.** Its published dataset is a byte-for-byte copy of the Crosswalk one
  (issue [#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7); the two test parquets have the same
  sha256 here). CurbRamp rows therefore report size, latency and parity on the 164 public Crosswalk crops, and no
  accuracy.
- **Preprocessing.** Every variant gets the slow PIL `BitImageProcessor` (`use_fast=False`, as `evaluate.py` and
  sidewalk-ai-api's `backend="pil"` do), from the variant's own `preprocessor_config.json`. The production and tiny
  configs are identical (resize shortest edge 256, bicubic, centre crop 224, ImageNet mean/std).
- **Scored one crop per call** (`--score-batch 1`, the default), as the API does. This matters for every graph with
  run-time activation quantization (`DynamicQuantizeLinear`): dynamic int8 and the tiny repos compute one activation
  scale per tensor over the whole batch, so a crop's output depends on its batch neighbours. `bench` always scores
  those at batch 1 and re-scores them at batch 16 to measure the dependence: **the same crop's probability moves by
  up to 0.34 (Crosswalk), 0.33 (CurbRamp), 0.82 (Obstacle), 0.67 (SurfaceProblem) and 0.76 (NoCurbRamp) for dynamic
  int8, and 0.18–0.43 for the tiny models** (`b16_vs_b1_max_abs_dprob`). The first version of this note scored at
  batch 16 and overstated the int8 damage (for example Obstacle votes 74 → 19 instead of 73 → 35).
- **Accuracy columns** are `evaluate.slice_metrics` on the test crops: accuracy, macro-F1, AUC of `p_incorrect`,
  and recall at 95% precision (`evaluate.recall_at_precision`) for each class.
- **Parity** is against PyTorch fp32 on the CPU, crop by crop: the largest |Δ softmax|, verdict flips, and
  **vote changes**. Softmax is computed in float32, as the API's `F.softmax` on float32 logits; the maps step every
  1e-6 near 1.0, about float32's rounding there. A vote is what the API would record: verdict, its confidence looked
  up in sidewalk-ai-api's live `accuracy_mappings/<type>.json` (main at `b96be89`, 2026-10-17, the Phase 0 maps)
  exactly as the API's loop does, and a vote only if the looked-up value is ≥ 0.92. A vote change is a crop that
  votes under one model and not the other, or votes for different verdicts. `vs_ref_votes` is how many of the crops
  vote under that variant.
- **Latency** is the median of 10 timed calls after 3 warm-up calls, on already-preprocessed tensors, 4 intra-op
  threads (torch `set_num_threads(4)`, onnxruntime `intra_op_num_threads=4`), onnxruntime's CPU provider with all
  graph optimizations. `lat_b1_ms` is one crop per call (the API's pattern), `ms_per_crop_b16` a 16-crop call divided
  by 16. No other benchmark ran during these measurements; the 1-minute load average recorded per row (2.6–8.5)
  includes this bench's own 4 threads, Spotlight indexing, and an occasional test run from another session. MPS runs
  on the M1's GPU and is shown only for scale; the API serves on an NVIDIA GPU, which was not measured.

## Results

| label_type | variant | n_crops | size_mb | params_m | accuracy | macro_f1 | auc | r_at_p95_correct | r_at_p95_incorrect | vs_ref_max_abs_dprob | vs_ref_verdict_flips | vs_ref_votes | vs_ref_vote_changes | b16_vs_b1_max_abs_dprob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Crosswalk | torch-fp32-cpu | 164 | 1218 | 304.4 | 0.9756 | 0.9749 | 0.9811 | 0.9565 | 0.9895 | 0 | 0 | 63 | 0 | — |
| Crosswalk | torch-fp32-mps | 164 | 1218 | 304.4 | 0.9756 | 0.9749 | 0.9811 | 0.9565 | 0.9895 | 7.808e-06 | 0 | 63 | 0 | — |
| Crosswalk | onnx-fp32 | 164 | 1218 | 304.4 | 0.9756 | 0.9749 | 0.9811 | 0.9565 | 0.9895 | 6.288e-06 | 0 | 63 | 0 | — |
| Crosswalk | onnx-int8 | 164 | 310.5 | 304.4 | 0.9634 | 0.9621 | 0.9762 | 0.9565 | 0.9895 | 0.3828 | 2 | 62 | 1 | 0.3381 |
| Crosswalk | onnx-int8-weightonly | 164 | 321.4 | 306.7 | 0.9756 | 0.9749 | 0.9809 | 0.9565 | 0.9895 | 0.01672 | 0 | 63 | 0 | — |
| Crosswalk | tiny-int8-existing | 164 | 24.63 | 22.14 | 0.939 | 0.9377 | 0.9779 | 0.913 | 0.9474 | 0.9893 | 10 | 66 | 9 | 0.1792 |
| CurbRamp | torch-fp32-cpu | 164 | 1218 | 304.4 | — | — | — | — | — | 0 | 0 | 41 | 0 | — |
| CurbRamp | torch-fp32-mps | 164 | 1218 | 304.4 | — | — | — | — | — | 7.987e-06 | 0 | 41 | 0 | — |
| CurbRamp | onnx-fp32 | 164 | 1218 | 304.4 | — | — | — | — | — | 2.414e-05 | 0 | 41 | 0 | — |
| CurbRamp | onnx-int8 | 164 | 310.5 | 304.4 | — | — | — | — | — | 0.7256 | 10 | 33 | 10 | 0.3252 |
| CurbRamp | onnx-int8-weightonly | 164 | 321.4 | 306.7 | — | — | — | — | — | 0.02304 | 0 | 41 | 0 | — |
| CurbRamp | tiny-int8-existing | 164 | 24.63 | 22.14 | — | — | — | — | — | 0.9977 | 50 | 58 | 37 | 0.2911 |
| Obstacle | torch-fp32-cpu | 647 | 1218 | 304.4 | 0.7759 | 0.7511 | 0.8264 | 0.1247 | 0.1301 | 0 | 0 | 73 | 0 | — |
| Obstacle | torch-fp32-mps | 647 | 1218 | 304.4 | 0.7759 | 0.7511 | 0.8264 | 0.1247 | 0.1301 | 2.581e-05 | 0 | 73 | 0 | — |
| Obstacle | onnx-fp32 | 647 | 1218 | 304.4 | 0.7759 | 0.7511 | 0.8264 | 0.1247 | 0.1301 | 5.156e-05 | 0 | 73 | 0 | — |
| Obstacle | onnx-int8 | 647 | 310.5 | 304.4 | 0.7604 | 0.7251 | 0.8249 | 0.2145 | 0.1585 | 0.9485 | 38 | 35 | 44 | 0.8227 |
| Obstacle | onnx-int8-perchannel | 647 | 311.6 | 304.8 | 0.7635 | 0.7313 | 0.8277 | 0.1621 | 0.1301 | 0.9495 | 36 | 47 | 36 | 0.8295 |
| Obstacle | onnx-int8-weightonly | 647 | 321.4 | 306.7 | 0.7759 | 0.7511 | 0.8263 | 0.1247 | 0.1301 | 0.05086 | 0 | 74 | 1 | — |
| Obstacle | tiny-int8-existing | 647 | 24.63 | 22.14 | 0.7944 | 0.7732 | 0.8642 | 0.4115 | 0.04472 | 0.9982 | 146 | 2 | 71 | 0.3862 |
| SurfaceProblem | torch-fp32-cpu | 784 | 1218 | 304.4 | 0.8112 | 0.7688 | 0.8673 | 0.5187 | 0.1774 | 0 | 0 | 237 | 0 | — |
| SurfaceProblem | torch-fp32-mps | 784 | 1218 | 304.4 | 0.8112 | 0.7688 | 0.8673 | 0.5187 | 0.1774 | 2.217e-05 | 0 | 237 | 0 | — |
| SurfaceProblem | onnx-fp32 | 784 | 1218 | 304.4 | 0.8112 | 0.7688 | 0.8673 | 0.5187 | 0.1774 | 2.098e-05 | 0 | 237 | 0 | — |
| SurfaceProblem | onnx-int8 | 784 | 310.5 | 304.4 | 0.824 | 0.7898 | 0.861 | 0.3974 | 0.2218 | 0.9121 | 40 | 129 | 110 | 0.6741 |
| SurfaceProblem | onnx-int8-weightonly | 784 | 321.4 | 306.7 | 0.8112 | 0.7681 | 0.8677 | 0.5224 | 0.1815 | 0.05802 | 2 | 237 | 6 | — |
| SurfaceProblem | tiny-int8-existing | 784 | 24.63 | 22.14 | 0.8214 | 0.7888 | 0.8806 | 0.5578 | 0.09677 | 0.9998 | 162 | 27 | 218 | 0.4337 |
| NoCurbRamp | torch-fp32-cpu | 1282 | 1218 | 304.4 | 0.876 | 0.8699 | 0.9447 | 0.8071 | 0.4923 | 0 | 0 | 437 | 0 | — |
| NoCurbRamp | torch-fp32-mps | 1282 | 1218 | 304.4 | 0.876 | 0.8699 | 0.9447 | 0.8071 | 0.4923 | 1.138e-05 | 0 | 437 | 0 | — |
| NoCurbRamp | onnx-fp32 | 1282 | 1218 | 304.4 | 0.876 | 0.8699 | 0.9447 | 0.8071 | 0.4923 | 9.745e-06 | 0 | 437 | 0 | — |
| NoCurbRamp | onnx-int8 | 1282 | 310.5 | 304.4 | 0.8775 | 0.8718 | 0.9437 | 0.7927 | 0.575 | 0.6949 | 40 | 392 | 63 | 0.7555 |
| NoCurbRamp | onnx-int8-weightonly | 1282 | 321.4 | 306.7 | 0.876 | 0.87 | 0.9446 | 0.8071 | 0.4904 | 0.02593 | 2 | 434 | 3 | — |
| NoCurbRamp | tiny-int8-existing | 1282 | 24.63 | 22.14 | 0.8846 | 0.88 | 0.9423 | 0.7992 | 0.4481 | 0.999 | 173 | 483 | 220 | 0.4116 |

Columns: `size_mb` is the weight file (`model.safetensors` for PyTorch, the `.onnx` file otherwise). `params_m` for
PyTorch is `sum(p.numel())`; for ONNX it is the initializer elements (the export folds the position embeddings to
224 × 224 and drops the unused mask token, about 1M fewer; weight-only adds its block scales). `r_at_p95_*` is the
recall at precision ≥ 0.95 for that class. The `vs_ref_*` columns compare with `torch-fp32-cpu`.
`b16_vs_b1_max_abs_dprob` is blank for graphs without run-time activation quantization.
**The `vs_ref_votes` and `vs_ref_vote_changes` of the tiny rows use the ViT-L validators' maps**, which were fitted
to ViT-L confidences; a tiny model in production would need its own map, so read those two columns as "how
differently it behaves", not as its vote rate. Verdict flips do not depend on any map.

**Latency, 4 CPU threads unless MPS:**

| label_type | variant | lat_b1_ms | ms_per_crop_b16 | load_avg_1m |
|---|---|---|---|---|
| Crosswalk | torch-fp32-cpu | 279.4 | 154.4 | 2.84 |
| Crosswalk | torch-fp32-mps | 56.83 | 32.72 | 3.5 |
| Crosswalk | onnx-fp32 | 523 | 475.9 | 2.81 |
| Crosswalk | onnx-int8 | 206 | 170.5 | 4.75 |
| Crosswalk | onnx-int8-weightonly | 670.4 | 500.7 | 4.96 |
| Crosswalk | tiny-int8-existing | 21 | 17.01 | 6.13 |
| CurbRamp | torch-fp32-cpu | 281.7 | 154.9 | 6.55 |
| CurbRamp | torch-fp32-mps | 56.86 | 34.89 | 6.43 |
| CurbRamp | onnx-fp32 | 517.7 | 479 | 5.58 |
| CurbRamp | onnx-int8 | 201.6 | 167.1 | 7.42 |
| CurbRamp | onnx-int8-weightonly | 666.9 | 500.8 | 7.33 |
| CurbRamp | tiny-int8-existing | 22.33 | 16.97 | 7.89 |
| Obstacle | torch-fp32-cpu | 295.2 | 157.8 | 6.69 |
| Obstacle | torch-fp32-mps | 56.82 | 32.34 | 4.3 |
| Obstacle | onnx-fp32 | 514.3 | 485.6 | 2.61 |
| Obstacle | onnx-int8 | 204.3 | 166.1 | 7.25 |
| Obstacle | onnx-int8-perchannel | 201.1 | 167.1 | 7.08 |
| Obstacle | onnx-int8-weightonly | 673.7 | 491.5 | 7.47 |
| Obstacle | tiny-int8-existing | 20.32 | 16.52 | 5.19 |
| SurfaceProblem | torch-fp32-cpu | 336.6 | 157.9 | 4.18 |
| SurfaceProblem | torch-fp32-mps | 56.61 | 35.79 | 5.39 |
| SurfaceProblem | onnx-fp32 | 516.1 | 474.1 | 3.7 |
| SurfaceProblem | onnx-int8 | 198.7 | 165.6 | 7.82 |
| SurfaceProblem | onnx-int8-weightonly | 657.8 | 495.4 | 8.45 |
| SurfaceProblem | tiny-int8-existing | 20.79 | 16.82 | 6.97 |
| NoCurbRamp | torch-fp32-cpu | 298.5 | 181.5 | 5.38 |
| NoCurbRamp | torch-fp32-mps | 56.87 | 36.47 | 7.34 |
| NoCurbRamp | onnx-fp32 | 514 | 472.1 | 7.01 |
| NoCurbRamp | onnx-int8 | 200.1 | 165.5 | 7.25 |
| NoCurbRamp | onnx-int8-weightonly | 659.4 | 496.6 | 7.8 |
| NoCurbRamp | tiny-int8-existing | 20.73 | 16.77 | 7.93 |

**Run-to-run spread.** `onnxruntime` is stable; PyTorch's batch-1 time on this CPU is not. Three interleaved rounds
of the Obstacle variants on its first 16 crops (`bench_torch` / `bench_onnx`, same settings;
[`distillation-2026-10/latency_check_obstacle.csv`](distillation-2026-10/latency_check_obstacle.csv)), min–max in ms:

| variant | b1 | b16, per crop |
|---|---|---|
| torch-fp32-cpu | 249–457 | 155–190 |
| onnx-fp32 | 514–518 | 484–494 |
| onnx-int8 | 205–212 | 172–175 |
| onnx-int8-weightonly | 679–693 | 511–513 |
| tiny-int8-existing | 21.5–21.8 | 17.4–17.7 |

So only ratios that hold across PyTorch's whole batch-1 range are stated below.

### What the results say

- **ONNX export is lossless.** fp32 ONNX matches PyTorch to at most 5.2e-5 in probability on every type, with no
  verdict flip and no vote change. MPS matches to 2.6e-5. Any of the three can serve today's models.
- **Dynamic int8 is the wrong quantization for these models.** Scored one crop at a time, it moves probabilities by
  up to 0.95 (`vs_ref_max_abs_dprob`). Accuracy moves −1.6 to +1.3 points (Crosswalk −1.2, Obstacle −1.6,
  SurfaceProblem +1.3, NoCurbRamp +0.2), but what matters for serving is the votes: 228 of 3,041 rows (7.5%) change,
  and the vote count falls 73 → 35 (Obstacle), 237 → 129 (SurfaceProblem), 437 → 392 (NoCurbRamp), 41 → 33
  (CurbRamp) and 63 → 62 (Crosswalk). Per-channel scales do not help much (Obstacle 36 changes, 47 votes). That its
  outputs move by up to 0.82 when only the batch changes shows how sensitive these models are to activation scales.
  Not tried: a map refitted to the int8 model's own confidences, static quantization with calibration data, or
  keeping some layers in fp32.
- **Weight-only int8 is the safe size reduction.** 1218 → 321 MB per model (about 0.9 GB saved per model),
  probabilities within 0.06, 10 vote changes in 3,041 rows (Obstacle 1, SurfaceProblem 6, NoCurbRamp 3). All are
  Agree verdicts whose served value sits at a map step. Most moved by less than 1e-6 in confidence (7 of the 11 that
  a float64 recomputation of the same scores flags; float32 flags 10), the size of float32 rounding near 1.0, so
  they say as much about the 1e-6 map grid as about quantization. Its cost is
  speed: `MatMulNBits` dequantizes the weights in every call, so on this CPU it is slower than PyTorch fp32 at both
  batch sizes and than ONNX fp32 (b1 658–693 vs 514–523 ms; b16 491–513 vs 472–494 ms per crop, within about 5%).
  The API runs PyTorch on a GPU, so this matters for CPU or browser use, or for an ONNX server.
- **Dynamic int8 is the only faster ViT-L, and only at batch 1.** At b1 it takes 199–212 ms against PyTorch's
  249–457 ms; at b16 it is no faster (165–175 vs 154–190 ms per crop). It costs the votes above.
- **Real CPU speed comes only from a smaller model.** The tiny ViT-S models take 20–22 ms per crop at b1 and 17 ms at
  b16, from a 25 MB file: about 10× faster than ViT-L dynamic int8 (199–212 ms; int8 against int8), about 9–11×
  faster than PyTorch fp32 at b16, and 11–21× at b1 across PyTorch's b1 spread. MPS (57 ms at b1) shows what even a
  laptop GPU does for ViT-L.
- **The tiny models look competitive, which is the reason to train a ViT-S properly.** On these crops, against
  ViT-L: accuracy Crosswalk −3.7, Obstacle +1.9, SurfaceProblem +1.0, NoCurbRamp +0.9 points; AUC −0.3, +3.8, +1.3,
  −0.2. But their verdicts differ from production's on 6–31% of crops (Crosswalk 10 of 164, CurbRamp 50 of 164,
  Obstacle 146 of 647, SurfaceProblem 162 of 784, NoCurbRamp 173 of 1,282), so they are a different classifier, not
  a compressed copy; and their training data very probably contains these crops' labels (next section), which would
  inflate exactly these numbers. Recall at 95% precision is mixed (Obstacle correct 0.41 vs 0.12, incorrect 0.04 vs
  0.13), which is what a different operating point looks like on small test sets.
- **No number here says how a small model does on the 0.92 gate under its own map**, which is what the switch
  decision needs. That requires the ViT-S fine-tune on the panorama-level split and a map fitted to it (follow-up).

## Where the `quantized_*_dinov2_tiny_onnx` repos came from

**Documented:** nothing. None of the five repos has a model card, and none of this repo's history, sidewalk-ai-api's,
SidewalkWebpage's, sidewalk-tagger-ai's, Planning's or sidewalk-auto-labeler's mentions them (`git log --all -S`,
`git grep`, and `gh search code`/`gh search issues` across the ProjectSidewalk org, 2026-10-20). The only mention
found anywhere is a third-party EDA note that lists them and says it did not download them. SidewalkWebpage's
`.onnx` history is unrelated: the LabelAId prediction-model MLPs (about 4–5 KB each), removed in February 2025.

**Read from the repos (Hugging Face API and the files, 2026-10-20):**

| fact | value |
|---|---|
| uploader | Hugging Face account `johnomeara`, the same account that uploaded the January 2025 and August 2025 ViT-L validators and the August 2025 datasets |
| dates | SurfaceProblem: 8 uploads 2025-04-17 23:55 to 2025-04-18 02:19 UTC; the other four one upload each, 07:16–07:18 UTC on 2025-04-18 |
| files | `config.json`, `preprocessor_config.json`, `ort_config.json`, `onnx/model_quantized.onnx` (24,625,231 bytes each, different sha256 per type); no fp32 weights, no training state, no card |
| architecture | `Dinov2ForImageClassification`, hidden 384, 12 layers, 6 heads, patch 14: **DINOv2 ViT-S/14** (`facebook/dinov2-small`), not a "tiny" variant; 22.1M parameters in the graph |
| labels | `id2label = {0: correct, 1: incorrect}`, the same as the ViT-L validators |
| preprocessing | identical to the ViT-L validators' `preprocessor_config.json` |
| exported with | transformers 4.48.3 (`config.json`); ONNX opset 14, IR 7, dynamic batch and spatial axes |
| quantized with | Hugging Face Optimum's `ORTQuantizer` (the tool that writes `ort_config.json`): dynamic, `QOperator`, `IntegerOps`, per-channel, `reduce_range`, uint8 weights and activations; graph producer `onnx.quantize 0.1.0` |
| configs | the five `config.json` files differ only in `_name_or_path` (`<type>_dinov2_tiny_onnx`) |

**Inferred, with the evidence:**

- **Each starts from `facebook/dinov2-small`, and every fp32 backbone tensor was updated, per type.** The strong
  evidence is the cross-type comparison: of the 148 fp32 tensors left unquantized in each graph (biases, layer norms,
  layer scales, embeddings), none is identical across the five types, in any of the 12 layers; the only identical
  tensors are quantization zero points. A linear probe would have left them identical. Their distance from
  `facebook/dinov2-small`'s is small (median relative L2 0.02–0.04% per type), about the size an fp16 round trip of
  dinov2-small itself would cause, so it confirms the starting point but not how far training moved them. The
  int8 weight matrices were not compared. The classifier is `768 → 2` (CLS token plus mean patch token),
  with biases of about ±1e-5. All of this is consistent with `fine_tune_classifier.py` run with
  `CHECKPOINT = "facebook/dinov2-small"`, though nothing confirms that script was used.
- **Trained before 2025-04-18**, between the January 2025 ViT-L upload (revisions `3e7119d`/`92ddf35`, same
  account) and the August 2025 production release, so on data of that period, not on the published August 2025
  datasets. Which labels, which split and which hyperparameters is **unknown**: no split, manifest or
  trainer state was uploaded.
- **Probably meant for in-browser or CPU validation** (int8 ONNX in the layout transformers.js reads,
  `onnx/model_quantized.onnx`). Nothing records that intent; it is an inference from the layout.

**Unknown:** the training data and split, whether any served or reported model ever used them, and whether an fp32
ViT-S checkpoint exists anywhere. The person who uploaded them would know; asking is cheaper than reverse
engineering further.

**What that means for the numbers above.** The tiny models were trained a few months before the August 2025
datasets were cut, from the same Project Sidewalk label pool, with an unknown split. So many of these test crops'
labels (or other labels on the same panoramas) were likely in their training data. Their accuracy here is therefore
not comparable with the ViT-L validators', which at least never trained on these test files. Their verdict flips
(how often they disagree with production) are still meaningful.

## Follow-up: ViT-S fine-tune and distillation on klone (not run)

What the issue's first experiment needs, and why it is not in this PR: it is GPU training, and the training path
(`fine_tune_classifier.py`, `fine_tune.slurm`) is frozen while the retrain runs.

1. **Proposed change to `fine_tune_classifier.py`** (for the retrain owner; not made here):
   - `--base-model` (default `facebook/dinov2-large`) replacing the `CHECKPOINT` constant, recorded in
     `training_run.json` and in `run_key` so a resumed run cannot silently switch backbones.
   - `--teacher PATH` and `--distill-alpha A` (default 0, i.e. off) and `--distill-temperature T` (default 2): a
     `Trainer` subclass whose `compute_loss` is `(1-A)·CE(y, s) + A·T²·KL(softmax(t/T) ‖ softmax(s/T))`, with the
     teacher's logits computed on the same augmented batch under `torch.no_grad()` (teacher in eval mode, fp16).
     Precomputing teacher logits on the un-augmented crops would be cheaper but would not match the student's
     augmented view.
   - Everything else unchanged (lr 1e-5 may be low for ViT-S; try 5e-5 as a second arm).
2. **Runs per label type**, on the new panorama-level `split.csv` the retrain produces (not the old file-level split
   used above): ViT-S CE only; ViT-S distilled from the retrained ViT-L (A = 0.5, T = 2). Same seed, same epochs.
3. **Slurm shape:** `fine_tune.slurm`'s, one GPU (`gpu-l40s` or `ckpt`), but ViT-S trains in roughly a fifth of
   ViT-L's time (an estimate from the parameter count, not measured), and distillation adds one teacher forward per
   batch (about ViT-L inference cost). All five types fit
   one job.
4. **Evaluate** with `evaluate.py --checkpoint <run> --tag vits` (and `--tag vits-distill`) on `test` and
   `test_heldout`, fit maps with `calibration/calibrate.py`, then `export_onnx.py export`, `quantize --method
   weight-only` and `quantize` (dynamic), and `export_onnx.py bench --cities-json ...` (batch 1) against the
   retrained ViT-L, with the new maps.
   Report AUC, recall at the gate, the share of votes kept, and the vote changes, per type.
5. **Decision rule** (the issue's): switch a type only if the small model is within about 1 point of ViT-L on AUC and
   on precision at the 0.92 gate, with its own fitted map.

## Draft issue for sidewalk-ai-api (not opened; for Jon to decide)

**Not justified yet.** The issue's bar is a small model within about 1 point of ViT-L, with ids to switch to. The
only small models that exist are the April 2025 tiny repos, whose training data is unknown and whose edge on these
crops may well be leakage; and dynamic int8 of ViT-L changes 7.5% of votes even scored one crop at a time.
Weight-only int8 keeps the votes, but the API runs PyTorch: it would help only if the server ran ONNX, where it
saves file size and could save memory (the ≥ 9–10 GB of VRAM issue #20 starts from) if the runtime keeps the weights
in int8, as onnxruntime's CPU kernel does; its GPU support was not checked. That is a serving-stack change worth its
own measurement, not a model switch. So no sidewalk-ai-api issue is proposed now.

Once the follow-up's ViT-S passes step 5, the issue would read:

> **Switch `<type>` validator to ViT-S (`projectsidewalk/sidewalk-validator-ai-<type>-vits@<sha>`)**
>
> Measured in sidewalk-validator-ai `docs/distillation-2026-10.md` (follow-up section) on the panorama-level test
> split and held-out cities: AUC `<x>` vs ViT-L `<y>`, precision at the 0.92 gate `<p>` vs `<q>` under each model's
> own map, `<n>`% of ViT-L's votes kept. 22M parameters vs 304M; CPU latency `<a>` vs `<b>` ms.
> Change: `model_map.py` `validator_models.<type>` (`model_id`, `revision`, `training_date`) and
> `accuracy_mappings/<type>.json` (the ViT-S map from `calibration/calibrate.py`). Roll out one type at a time,
> recording `validator_revision` per assessment
> ([SidewalkWebpage#5794](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5794)) so the switch can be
> evaluated against production outcomes.
