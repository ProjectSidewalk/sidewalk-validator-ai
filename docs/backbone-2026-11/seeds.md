# Seeds: mean ± sample sd over seeds (lr 1e-5)

| arm | type | seeds | held-out AUC | test AUC | mean Δ held-out | seeds with Δ > 0 | mean Δ within-city held-out | mean Δ test |
|---|---|---|---|---|---|---|---|---|
| `convnextv2-base` | CurbRamp | 3 | 0.884 ± 0.008 | 0.939 | -0.028 | 0 of 3 | -0.018 | -0.026 |
| `convnextv2-base` | SurfaceProblem | 3 | 0.670 ± 0.031 | 0.861 | -0.036 | 1 of 3 | -0.027 | -0.035 |
| `dinov2-large` | CurbRamp | 3 | 0.912 ± 0.004 | 0.965 |  |  |  |  |
| `dinov2-large` | SurfaceProblem | 3 | 0.706 ± 0.020 | 0.895 |  |  |  |  |
| `dinov2-reg-large` | CurbRamp | 3 | 0.903 ± 0.013 | 0.961 | -0.009 | 0 of 3 | -0.006 | -0.004 |
| `dinov2-reg-large` | SurfaceProblem | 3 | 0.724 ± 0.013 | 0.898 | +0.018 | 2 of 3 | -0.014 | +0.002 |
| `dinov3-vitb16` | CurbRamp | 3 | 0.898 ± 0.007 | 0.949 | -0.014 | 0 of 3 | +0.006 | -0.016 |
| `dinov3-vitb16` | SurfaceProblem | 3 | 0.705 ± 0.006 | 0.893 | -0.000 | 2 of 3 | -0.002 | -0.003 |
| `dinov3-vitl16` | CurbRamp | 3 | 0.899 ± 0.009 | 0.960 | -0.014 | 0 of 3 | -0.001 | -0.006 |
| `dinov3-vitl16` | SurfaceProblem | 3 | 0.677 ± 0.005 | 0.905 | -0.028 | 0 of 3 | -0.026 | +0.010 |
| `dinov3-vitl16-256` | CurbRamp | 3 | 0.902 ± 0.011 | 0.958 | -0.010 | 1 of 3 | +0.001 | -0.007 |
| `dinov3-vitl16-256` | SurfaceProblem | 3 | 0.681 ± 0.006 | 0.902 | -0.024 | 0 of 3 | -0.042 | +0.006 |
| `rampnet-convnextv2-base` | CurbRamp | 3 | 0.883 ± 0.015 | 0.948 | -0.029 | 0 of 3 | -0.036 | -0.017 |
| `rampnet-convnextv2-base` | SurfaceProblem | 3 | 0.676 ± 0.018 | 0.845 | -0.030 | 1 of 3 | -0.045 | -0.051 |
