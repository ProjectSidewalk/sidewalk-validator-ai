# Environment: in-env baseline (transformers 4.56.2) against round 3 (4.55.0)

Same recipe, split and seed; a mean difference beyond twice the round-3 seed sd (CurbRamp 0.006, SurfaceProblem 0.01) is called out at the top of docs/backbone-2026-11.md.

| type | seed | round-3 test AUC | in-env test AUC | Δ test | round-3 held-out AUC | in-env held-out AUC | Δ held-out |
|---|---|---|---|---|---|---|---|
| CurbRamp | 42 | 0.968 | 0.967 | -0.001 [-0.005, +0.003] | 0.907 | 0.917 | +0.010 [-0.002, +0.024] |
| CurbRamp | 43 | 0.966 | 0.963 | -0.002 [-0.007, +0.002] | 0.914 | 0.909 | -0.006 [-0.020, +0.009] |
| CurbRamp | 44 | 0.963 | 0.965 | +0.003 [-0.002, +0.007] | 0.902 | 0.910 | +0.008 [-0.004, +0.021] |
| SurfaceProblem | 42 | 0.896 | 0.894 | -0.002 [-0.009, +0.006] | 0.710 | 0.728 | +0.017 [-0.003, +0.036] |
| SurfaceProblem | 43 | 0.910 | 0.899 | -0.011 [-0.020, -0.001] | 0.720 | 0.702 | -0.019 [-0.041, +0.004] |
| SurfaceProblem | 44 | 0.893 | 0.894 | +0.001 [-0.007, +0.009] | 0.730 | 0.688 | -0.042 [-0.061, -0.024] |

