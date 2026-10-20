# Cheaper validators: ONNX, int8, and the existing tiny models (October 2026)

Part of [#20](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/20). Each production validator is a
304M-parameter DINOv2 ViT-L (1.22 GB of fp32 weights). This note measures what can be had **without training**:
exporting the five production checkpoints to ONNX, quantizing them to int8, and scoring the five existing
`projectsidewalk/quantized_*_dinov2_tiny_onnx` models on the same held-out crops. It also records what is known about
where those tiny models came from. The ViT-S fine-tune and distillation the issue asks for need a GPU and a change to
`fine_tune_classifier.py`; they are written up as a recipe at the end, not run.

**Short answer.**

- **Dynamic int8 (`quantize_dynamic`) is not safe to serve.** It cuts the file to 310 MB but changes served votes:
  359 of the 3,647 test crops (9.8%) vote differently from PyTorch, and the vote count falls most for Obstacle
  (74 → 19) and SurfaceProblem (253 → 88). Per-channel scales do not fix it (Obstacle 48 changes).
- **Weight-only int8 (`MatMulNBits`, fp32 activations) keeps nearly every vote.** 321 MB; 12 of 3,647 crops
  (0.3%) vote differently, all Agree verdicts sitting on a map step (11 of them moved by less than 5e-5 in
  confidence), and accuracy and AUC are unchanged to the third decimal. On this CPU it is
  slower than fp32 PyTorch, so it buys size (download, memory, browser), not speed.
- **ONNX fp32** reproduces PyTorch to about 1e-5 in probability and changes no vote, but onnxruntime's CPU kernels are
  about 1.5–2× slower than PyTorch's on this Apple CPU.
- **The existing tiny models are ViT-S (22M parameters, 25 MB, 14–20× faster than ViT-L on CPU) and score
  about as well as ViT-L, often better, on accuracy and AUC on these crops**, but their verdicts differ from
  production's on 5–21% of crops, their training data is undocumented and very probably overlaps these test crops,
  and their provenance is not good enough to ship.
  They are evidence that a ViT-S is worth training properly, not a drop-in replacement.

## How the numbers were made

All numbers are from `export_onnx.py` (this PR), run 2026-10-20 on an Apple M1 Max (8 performance + 2 efficiency
cores, 32 GB), macOS 15.5, Python 3.12.14, torch 2.14.1, transformers 4.55.0 (requirements.txt's pin), onnx 1.23.2,
onnxruntime 1.31.0, onnx-ir 1.0.0.

```
# per type T in crosswalk curbramp obstacle surfaceproblem nocurbramp, REV = the production revision (table below)
python export_onnx.py export --model projectsidewalk/sidewalk-validator-ai-$T --revision $REV --out models/$T
python export_onnx.py quantize models/$T/model.onnx                        # dynamic -> model_int8.onnx
python export_onnx.py quantize models/$T/model.onnx --method weight-only   # -> model_int8_wo.onnx
hf download projectsidewalk/quantized_<tiny name>_dinov2_tiny_onnx --local-dir tiny/<tiny name>
hf download projectsidewalk/sidewalk-validator-ai-dataset-$T --repo-type dataset --include "test/*" --local-dir data/$T
python export_onnx.py bench --label-type <Type> --crops data/$T --model projectsidewalk/sidewalk-validator-ai-$T \
    --revision $REV --onnx onnx-fp32=models/$T/model.onnx --onnx onnx-int8=models/$T/model_int8.onnx \
    --onnx onnx-int8-weightonly=models/$T/model_int8_wo.onnx \
    --onnx tiny-int8-existing=tiny/<tiny name>/onnx/model_quantized.onnx \
    --accuracy-map ../sidewalk-ai-api/accuracy_mappings/$T.json --mps --threads 4 --out bench/
    # CurbRamp adds --no-accuracy; Obstacle also ran --onnx onnx-int8-perchannel=... (quantize --per-channel)
python export_onnx.py table bench/bench_*.csv --columns <the columns shown>
```

The five `bench_<Type>.csv` files behind every table are in [`distillation-2026-10/`](distillation-2026-10)
(paths made relative); the per-crop scores stay outside the repo.

| type | model revision (sidewalk-ai-api `model_map.py`) | test crops (correct / incorrect) | dataset revision |
|---|---|---|---|
| Crosswalk | `b7936e5bd17a4b96db301f954e5d3a5fe9eaf361` | 240 (129 / 111) | `1be513a` |
| CurbRamp | `969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a` | 240, the Crosswalk crops (see below) | `d396ee4` |
| Obstacle | `8f0fe8fc48748fac35098e74c1848d015126deef` | 769 (443 / 326) | `9fdc3c3` |
| SurfaceProblem | `0e1439daa5667c8a81f63e75f835bd51221df4c2` | 1,024 (620 / 404) | `23f32c0` |
| NoCurbRamp | `9f3f3bb2f40864a468fb65e43644ee5ba54e3e19` | 1,374 (797 / 577) | `e737c59` |

- **Crops.** The `test/` split of each public dataset repo. Hugging Face rate-limited this IP during the run, so the
  crops were taken from each dataset's auto-converted parquet (`/api/datasets/<repo>/parquet/default/test/*.parquet`),
  one request per type, instead of one request per file. The extracted file names were checked against the repo's
  file list (all five identical), and the 496 Obstacle files that the per-file download had already fetched were
  byte-identical to the parquet's. `hf download ... --include "test/*"` gives the same directory.
- **These test splits are the old file-level splits** (issue [#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3)):
  crops of the same panorama can sit in train and test, so the absolute accuracy below flatters every model. It is
  fine for comparing variants of the same model on the same images, which is what this note does.
- **CurbRamp has no test set of its own.** Its published dataset is a byte-for-byte copy of the Crosswalk one
  (issue [#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7); the two test parquets have the same
  sha256 here). CurbRamp rows therefore report size, latency and parity on the 240 Crosswalk crops, and no accuracy.
- **Preprocessing.** Every variant gets the slow PIL `BitImageProcessor` (`use_fast=False`, as `evaluate.py` and
  sidewalk-ai-api's `backend="pil"` do), from the variant's own `preprocessor_config.json`. The production and tiny
  configs are identical (resize shortest edge 256, bicubic, centre crop 224, ImageNet mean/std).
- **Accuracy columns** are `evaluate.slice_metrics` on the test crops: accuracy, macro-F1, AUC of `p_incorrect`,
  and recall at 95% precision (`evaluate.recall_at_precision`) for each class.
- **Parity** is against PyTorch fp32 on the CPU, crop by crop: the largest |Δ softmax|, verdict flips, and
  **vote changes**. A vote is what the API would record: verdict, its confidence looked up in sidewalk-ai-api's live
  `accuracy_mappings/<type>.json` (main at `b96be89`, 2026-10-17, the Phase 0 maps) exactly as the API's loop does,
  and a vote only if the looked-up value is ≥ 0.92. A vote change is a crop that votes under one model and not the
  other, or votes for different verdicts. "votes" is how many of the crops vote under that variant.
- **Latency** is the median of 10 timed calls after 3 warm-up calls, on already-preprocessed tensors, 4 intra-op
  threads (torch `set_num_threads(4)`, onnxruntime `intra_op_num_threads=4`), onnxruntime's CPU provider with all
  graph optimizations. `b1` is one crop per call (the API's pattern), `b16/crop` is a 16-crop call divided by 16. Two
  other jobs shared the machine (1-minute load average 3–10, recorded per row in the CSVs), so treat differences under
  ~20% as noise. MPS runs on the M1's GPU and is shown only for scale; the API serves on an NVIDIA GPU, which was not
  measured.

## Results

| type | variant | MB on disk | params (M) | accuracy | macro-F1 | AUC | R@P95 correct | R@P95 incorrect | max \|Δp\| vs torch | verdict flips | votes | vote changes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Crosswalk | torch-fp32-cpu | 1218 | 304.4 | 0.9208 | 0.9195 | 0.9503 | 0.6047 | 0.8559 | 0 | 0 | 136 | 0 |
| Crosswalk | torch-fp32-mps | 1218 | 304.4 | 0.9208 | 0.9195 | 0.9503 | 0.6047 | 0.8559 | 1.195e-05 | 0 | 136 | 0 |
| Crosswalk | onnx-fp32 | 1218 | 304.4 | 0.9208 | 0.9195 | 0.9503 | 0.6047 | 0.8559 | 9.321e-06 | 0 | 136 | 0 |
| Crosswalk | onnx-int8 | 310.5 | 304.4 | 0.9167 | 0.9157 | 0.9513 | 0.6357 | 0.8649 | 0.71 | 5 | 130 | 6 |
| Crosswalk | onnx-int8-weightonly | 321.4 | 306.7 | 0.9208 | 0.9195 | 0.95 | 0.6124 | 0.8559 | 0.01672 | 0 | 136 | 0 |
| Crosswalk | tiny-int8-existing | 24.63 | 22.14 | 0.9083 | 0.9067 | 0.9652 | 0.8062 | 0.8378 | 0.9898 | 13 | 133 | 19 |
| CurbRamp | torch-fp32-cpu | 1218 | 304.4 | — | — | — | — | — | 0 | 0 | 63 | 0 |
| CurbRamp | torch-fp32-mps | 1218 | 304.4 | — | — | — | — | — | 8.744e-06 | 0 | 63 | 0 |
| CurbRamp | onnx-fp32 | 1218 | 304.4 | — | — | — | — | — | 3.191e-05 | 0 | 63 | 0 |
| CurbRamp | onnx-int8 | 310.5 | 304.4 | — | — | — | — | — | 0.8942 | 19 | 49 | 18 |
| CurbRamp | onnx-int8-weightonly | 321.4 | 306.7 | — | — | — | — | — | 0.02304 | 0 | 64 | 1 |
| CurbRamp | tiny-int8-existing | 24.63 | 22.14 | — | — | — | — | — | 0.996 | 86 | 106 | 71 |
| Obstacle | torch-fp32-cpu | 1218 | 304.4 | 0.7776 | 0.7683 | 0.8396 | 0.09707 | 0.3067 | 0 | 0 | 74 | 0 |
| Obstacle | torch-fp32-mps | 1218 | 304.4 | 0.7776 | 0.7683 | 0.8396 | 0.09707 | 0.3067 | 1.259e-05 | 0 | 74 | 0 |
| Obstacle | onnx-fp32 | 1218 | 304.4 | 0.7776 | 0.7683 | 0.8396 | 0.09707 | 0.3067 | 3.827e-05 | 0 | 74 | 0 |
| Obstacle | onnx-int8 | 310.5 | 304.4 | 0.7607 | 0.7404 | 0.8318 | 0.14 | 0.2975 | 0.8897 | 79 | 19 | 57 |
| Obstacle | onnx-int8-perchannel | 311.6 | 304.8 | 0.7646 | 0.7453 | 0.8301 | 0.1354 | 0.2699 | 0.893 | 74 | 28 | 48 |
| Obstacle | tiny-int8-existing | 24.63 | 22.14 | 0.814 | 0.8054 | 0.8845 | 0.535 | 0.07975 | 0.998 | 164 | 2 | 72 |
| Obstacle | onnx-int8-weightonly | 321.4 | 306.7 | 0.7776 | 0.7683 | 0.8396 | 0.09707 | 0.3221 | 0.05086 | 0 | 75 | 1 |
| SurfaceProblem | torch-fp32-cpu | 1218 | 304.4 | 0.8154 | 0.803 | 0.8905 | 0.5032 | 0.349 | 0 | 0 | 253 | 0 |
| SurfaceProblem | torch-fp32-mps | 1218 | 304.4 | 0.8154 | 0.803 | 0.8905 | 0.5032 | 0.349 | 2.315e-05 | 0 | 253 | 0 |
| SurfaceProblem | onnx-fp32 | 1218 | 304.4 | 0.8154 | 0.803 | 0.8905 | 0.5032 | 0.349 | 2.191e-05 | 0 | 253 | 0 |
| SurfaceProblem | onnx-int8 | 310.5 | 304.4 | 0.8232 | 0.8134 | 0.881 | 0.4726 | 0.198 | 0.9907 | 66 | 88 | 165 |
| SurfaceProblem | onnx-int8-weightonly | 321.4 | 306.7 | 0.8154 | 0.8028 | 0.8907 | 0.5016 | 0.3762 | 0.05802 | 2 | 252 | 7 |
| SurfaceProblem | tiny-int8-existing | 24.63 | 22.14 | 0.8281 | 0.8176 | 0.9032 | 0.571 | 0.3342 | 0.9998 | 209 | 27 | 236 |
| NoCurbRamp | torch-fp32-cpu | 1218 | 304.4 | 0.8777 | 0.8733 | 0.9449 | 0.7641 | 0.5823 | 0 | 0 | 448 | 0 |
| NoCurbRamp | torch-fp32-mps | 1218 | 304.4 | 0.8777 | 0.8733 | 0.9449 | 0.7641 | 0.5823 | 1.23e-05 | 0 | 448 | 0 |
| NoCurbRamp | onnx-fp32 | 1218 | 304.4 | 0.8777 | 0.8733 | 0.9449 | 0.7641 | 0.5823 | 1.299e-05 | 0 | 448 | 0 |
| NoCurbRamp | onnx-int8 | 310.5 | 304.4 | 0.8617 | 0.8572 | 0.941 | 0.7829 | 0.4939 | 0.8102 | 72 | 341 | 113 |
| NoCurbRamp | onnx-int8-weightonly | 321.4 | 306.7 | 0.8777 | 0.8733 | 0.9448 | 0.7629 | 0.5823 | 0.02593 | 2 | 445 | 3 |
| NoCurbRamp | tiny-int8-existing | 24.63 | 22.14 | 0.885 | 0.8816 | 0.9444 | 0.7704 | 0.5442 | 0.999 | 178 | 493 | 227 |

Columns: MB is the weight file (`model.safetensors` for PyTorch, the `.onnx` file otherwise). Params for PyTorch is
`sum(p.numel())`; for ONNX it is the initializer elements (the export folds the position embeddings to 224 × 224
and drops the unused mask token, about 1M fewer; weight-only adds its block scales). The Obstacle per-channel row is
an extra `quantize --per-channel` run; the Obstacle weight-only row came from a second `bench` run (its PyTorch
reference reproduced the first run's probabilities exactly) and was appended to the same CSV.
**The `votes` and `vote changes` of the tiny rows use the ViT-L validators' maps**, which were fitted to ViT-L
confidences; a tiny model in production would need its own map, so read those two columns as "how differently it
behaves", not as its vote rate. Verdict flips do not depend on any map.

**Latency (ms), 4 CPU threads unless MPS:**

| type | variant | b1 (ms) | b16, per crop (ms) | 1-min load avg |
|---|---|---|---|---|
| Crosswalk | torch-fp32-cpu | 344.3 | 181.6 | 8.19 |
| Crosswalk | torch-fp32-mps | 58.01 | 37.54 | 8.88 |
| Crosswalk | onnx-fp32 | 520.4 | 488.8 | 8.3 |
| Crosswalk | onnx-int8 | 206.4 | 171.4 | 8.08 |
| Crosswalk | onnx-int8-weightonly | 654.7 | 538.6 | 8.42 |
| Crosswalk | tiny-int8-existing | 24.31 | 20.01 | 7.63 |
| CurbRamp | torch-fp32-cpu | 294.1 | 169.2 | 6.92 |
| CurbRamp | torch-fp32-mps | 56.89 | 36.52 | 5.71 |
| CurbRamp | onnx-fp32 | 522.5 | 485.3 | 5.04 |
| CurbRamp | onnx-int8 | 206.6 | 167.4 | 7 |
| CurbRamp | onnx-int8-weightonly | 711.1 | 508.9 | 5.78 |
| CurbRamp | tiny-int8-existing | 21.21 | 17.2 | 6.19 |
| Obstacle | torch-fp32-cpu | 322.5 | 205.1 | 3.13 |
| Obstacle | torch-fp32-mps | 60.63 | 46.94 | 6.71 |
| Obstacle | onnx-fp32 | 625.5 | 566.6 | 6.83 |
| Obstacle | onnx-int8 | 255.1 | 203.3 | 6.75 |
| Obstacle | onnx-int8-perchannel | 240.5 | 174.5 | 10.06 |
| Obstacle | tiny-int8-existing | 22.37 | 17.84 | 7.15 |
| Obstacle | onnx-int8-weightonly | 657.6 | 495.1 | 5.94 |
| SurfaceProblem | torch-fp32-cpu | 416.7 | 176.4 | 3.88 |
| SurfaceProblem | torch-fp32-mps | 59.52 | 42.7 | 5.61 |
| SurfaceProblem | onnx-fp32 | 681.1 | 598.1 | 5.01 |
| SurfaceProblem | onnx-int8 | 212.9 | 226.6 | 13.9 |
| SurfaceProblem | onnx-int8-weightonly | 707.7 | 515.6 | 10.34 |
| SurfaceProblem | tiny-int8-existing | 21.11 | 17.12 | 10.04 |
| NoCurbRamp | torch-fp32-cpu | 350.8 | 193 | 9.16 |
| NoCurbRamp | torch-fp32-mps | 59.29 | 41.93 | 7.3 |
| NoCurbRamp | onnx-fp32 | 516.1 | 496.8 | 6.28 |
| NoCurbRamp | onnx-int8 | 199.5 | 166.5 | 7.43 |
| NoCurbRamp | onnx-int8-weightonly | 675.8 | 498.6 | 8.26 |
| NoCurbRamp | tiny-int8-existing | 21.02 | 16.75 | 8.41 |

Run-to-run spread: the Obstacle PyTorch CPU reference, run twice an hour apart, measured 322 and 273 ms at b1 and
205 and 168 ms per crop at b16, so differences under about 20% between rows are noise here.

Obstacle was also quantized with per-channel weight scales (`quantize --per-channel`), to check that the dynamic-int8
damage is not just per-tensor scales: 48 vote changes and 28 votes, against per-tensor's 57 and 19 (rows above), so not.

### What the results say

- **ONNX export is lossless.** fp32 ONNX matches PyTorch to at most 3.8e-5 in probability on every type, with no
  verdict flip and no vote change. MPS matches to 2.3e-5. Any of the three can serve today's models.
- **Dynamic int8 is the wrong quantization for these models.** It quantizes every activation to uint8 per call,
  and the rounding moves probabilities by up to 0.99 (max |Δp|), consistent with the large activation outliers ViTs
  are known for (not measured here). Accuracy drops 0.4–1.7 points on Crosswalk, Obstacle and NoCurbRamp and rises
  0.8 on SurfaceProblem, but what matters for serving is the votes: 9.8% of crops change, and the vote count falls
  from 74 to 19 (Obstacle) and from 253 to 88 (SurfaceProblem). Not tried: a map refitted to the int8 model's own
  confidences, static quantization with calibration data, or keeping some layers in fp32.
- **Weight-only int8 is the safe size reduction.** 1218 → 321 MB per model (about 0.9 GB saved per model),
  probabilities within 0.06, 12 vote changes in 3,647 crops, all Agree verdicts whose served value sits at a map
  step. Its cost is speed: `MatMulNBits` dequantizes the weights in every call, so on this CPU it is the slowest
  variant (b1 654–711 ms). The API runs PyTorch on a GPU, so this matters only for CPU or browser use.
- **Speed on the CPU comes only from a smaller model.** The fastest ViT-L variant (dynamic int8, 200–255 ms per
  crop at b1) is 1.3–2.0× faster than PyTorch fp32 (294–417 ms) and costs the votes above. The tiny ViT-S models
  take 21–24 ms per crop at b1, 14–20× faster than PyTorch ViT-L on the same 4 threads, from a 25 MB file. MPS
  (57–61 ms) shows what even a laptop GPU does for ViT-L.
- **The tiny models look competitive, which is the reason to train a ViT-S properly.** On these crops they are close to
  or better than ViT-L on accuracy (Crosswalk −1.3, Obstacle +3.6, SurfaceProblem +1.3, NoCurbRamp +0.7 points) and AUC
  (+1.5, +4.5, +1.3, −0.05). But they disagree with the production verdict on 5–21% of crops (Crosswalk 13 of 240,
  Obstacle 164 of 769, SurfaceProblem 209 of 1,024, NoCurbRamp 178 of 1,374), so they are a different classifier,
  not a compressed copy; and their training data very probably contains these crops' labels (next section), which
  would inflate exactly these numbers. Recall at 95% precision is mixed (Obstacle correct 0.54 vs 0.10, Obstacle
  incorrect 0.08 vs 0.31), which is what a different operating point looks like on small test sets.
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

- **Each is `facebook/dinov2-small` fully fine-tuned on that label type.** The 148 fp32 tensors left unquantized
  in each graph (biases, layer norms, layer scales, embeddings) are all within 0.02–0.04% (median relative L2
  distance, per type) of `facebook/dinov2-small`'s, and none is identical to it. Across the five types, every fp32
  tensor differs in all 12 layers; the only identical tensors are quantization zero points. A linear probe would
  have left the backbone identical across types. The classifier is `768 → 2` (CLS token plus mean patch token),
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
labels (or other labels on the same panoramas) were likely in their training data. Their accuracy here is therefore not comparable with
the ViT-L validators', which at least never trained on these test files. Their parity numbers (how often they vote
differently from production) are still meaningful.

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
   weight-only` and `quantize` (dynamic), and `export_onnx.py bench` against the retrained ViT-L, with the new maps.
   Report AUC, recall at the gate, the share of votes kept, and the vote changes, per type.
5. **Decision rule** (the issue's): switch a type only if the small model is within about 1 point of ViT-L on AUC and
   on precision at the 0.92 gate, with its own fitted map.

## Draft issue for sidewalk-ai-api (not opened; for Jon to decide)

**Not justified yet.** The issue's bar is a small model within about 1 point of ViT-L, with ids to switch to. The
only small models that exist are the April 2025 tiny repos, whose training data is unknown and whose apparent edge
on these crops is very likely leakage; and dynamic int8 of ViT-L changes about 10% of votes. Weight-only int8 is safe
but saves only memory, which the API (GPU, PyTorch) does not need. So no sidewalk-ai-api issue is proposed now.

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
