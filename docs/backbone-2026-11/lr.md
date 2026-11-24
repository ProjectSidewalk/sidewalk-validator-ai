# Learning rate (explanatory; cannot win)

Each 3e-5 run against the same arm at 1e-5 (seed 42) and against the baseline at 3e-5. An arm whose 3e-5 run beats its 1e-5 run by at least the type's threshold (CurbRamp +0.010, SurfaceProblem +0.016) is called recipe-limited. That bar was set for a three-seed mean; here it is applied to one paired run, whose seed-to-seed sd alone is about CurbRamp 0.009, SurfaceProblem 0.014, so "recipe-limited" flags a three-seed follow-up, not a result.

| type | arm | seed | lr | held-out AUC | Δ held-out vs own 1e-5 | Δ held-out vs baseline at the same lr | reading |
|---|---|---|---|---|---|---|---|
| CurbRamp | `dinov2-large` | 42 | 3e-05 | 0.894 | -0.024 [-0.042, -0.006] |  |  |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | 0.884 | +0.009 [-0.009, +0.025] | -0.010 [-0.032, +0.010] |  |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | 0.871 | +0.003 [-0.016, +0.024] | -0.023 [-0.044, -0.000] |  |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | 0.658 | -0.070 [-0.098, -0.041] |  |  |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | 0.670 | +0.035 [+0.011, +0.060] | +0.013 [-0.023, +0.046] | recipe-limited |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | 0.656 | -0.001 [-0.027, +0.028] | -0.002 [-0.034, +0.030] |  |
