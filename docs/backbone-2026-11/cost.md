# Cost

Parameters counted from the built model; seconds per optimizer step are the last attempt's wall time (loading and per-epoch evaluation included) over the steps it ran, mean over the timed runs. A run requeued after training finished but before it was evaluated reruns as a no-op that trains no steps: it has no timing (counted in "timed"), and its peak GPU memory is that no-op's, so the peak column can understate such a run. GPU models from each run's `gpu.txt` (appended on every attempt). ONNX rows join when that later stage has run.

| arm | type | params (M) | tokens / image | model class | AutoModelForImageClassification, no custom code | GPUs | s / step | timed | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|
| `convnextv2-base` | CurbRamp | 87.7 | 49 | ConvNextV2ForImageClassification | yes | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 0.86 | 4 of 4 | 4.1 |
| `convnextv2-base` | SurfaceProblem | 87.7 | 49 | ConvNextV2ForImageClassification | yes | NVIDIA A100 80GB PCIe, 81920 MiB; NVIDIA A40, 46068 MiB; NVIDIA H200, 143771 MiB; NVIDIA L40, 46068 MiB | 1.03 | 4 of 4 | 4.1 |
| `dinov2-large` | CurbRamp | 304.4 | 256 | Dinov2ForImageClassification | yes | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.21 | 4 of 4 | 8.0 |
| `dinov2-large` | SurfaceProblem | 304.4 | 256 | Dinov2ForImageClassification | yes | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.62 | 4 of 4 | 8.0 |
| `dinov2-reg-large` | CurbRamp | 304.4 | 256 | Dinov2WithRegistersForImageClassification | yes | NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.12 | 3 of 3 | 8.1 |
| `dinov2-reg-large` | SurfaceProblem | 304.4 | 256 | Dinov2WithRegistersForImageClassification | yes | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.48 | 3 of 3 | 8.1 |
| `dinov3-vitb16` | CurbRamp | 85.7 | 196 | SidewalkDinov3ForImageClassification | no | NVIDIA A40, 46068 MiB; NVIDIA L40S, 46068 MiB | 0.86 | 3 of 3 | 2.4 |
| `dinov3-vitb16` | SurfaceProblem | 85.7 | 196 | SidewalkDinov3ForImageClassification | no | NVIDIA A100 80GB PCIe, 81920 MiB; NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB | 0.89 | 3 of 3 | 2.4 |
| `dinov3-vitl16` | CurbRamp | 303.1 | 196 | SidewalkDinov3ForImageClassification | no | NVIDIA A40, 46068 MiB | 1.42 | 3 of 3 | 7.3 |
| `dinov3-vitl16` | SurfaceProblem | 303.1 | 196 | SidewalkDinov3ForImageClassification | no | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.33 | 3 of 3 | 7.3 |
| `dinov3-vitl16-256` | CurbRamp | 303.1 | 256 | SidewalkDinov3ForImageClassification | no | NVIDIA A40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.51 | 3 of 3 | 8.1 |
| `dinov3-vitl16-256` | SurfaceProblem | 303.1 | 256 | SidewalkDinov3ForImageClassification | no | NVIDIA A40, 46068 MiB; NVIDIA L40, 46068 MiB; NVIDIA L40S, 46068 MiB | 1.39 | 3 of 3 | 8.1 |
| `rampnet-convnextv2-base` | CurbRamp | 87.7 | 49 | ConvNextV2ForImageClassification | yes | NVIDIA A40, 46068 MiB; NVIDIA H200, 143771 MiB; NVIDIA L40S, 46068 MiB | 0.49 | 4 of 4 | 4.1 |
| `rampnet-convnextv2-base` | SurfaceProblem | 87.7 | 49 | ConvNextV2ForImageClassification | yes | NVIDIA A100 80GB PCIe, 81920 MiB; NVIDIA A40, 46068 MiB; NVIDIA L40S, 46068 MiB | 0.53 | 4 of 4 | 4.1 |
