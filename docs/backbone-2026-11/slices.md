# Slices: type × severity and type × tag against the in-env baseline

Same rows for both models; Δ with a 95% panorama-bootstrap interval; blank under 30 labels; ⚠ thin = fewer than 100 training rows; round-3 AUC is round 3's own seed-42 baseline on the slice. Intervals are not corrected for the number of slices.

## CurbRamp / `convnextv2-base` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.946 | -0.023 [-0.037, -0.011] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.877 | -0.071 [-0.117, -0.031] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.854 | -0.042 [-0.114, +0.019] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.936 | -0.048 [-0.099, -0.006] |

## CurbRamp / `convnextv2-base` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.945 | -0.026 [-0.042, -0.013] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.870 | -0.070 [-0.104, -0.037] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.964 | -0.022 [-0.051, -0.003] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.886 | -0.045 [-0.116, +0.012] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.879 | -0.057 [-0.118, -0.007] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.890 | -0.005 [-0.063, +0.047] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.908 | -0.061 [-0.130, -0.008] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.871 | -0.077 [-0.235, +0.025] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `convnextv2-base` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.872 | -0.042 [-0.069, -0.017] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.874 | -0.024 [-0.105, +0.059] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `convnextv2-base` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.900 | -0.036 [-0.066, -0.009] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.787 | -0.063 [-0.130, +0.001] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.901 | -0.060 [-0.123, -0.016] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## CurbRamp / `dinov2-reg-large` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.962 | -0.008 [-0.016, -0.000] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.922 | -0.027 [-0.057, -0.000] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.944 | +0.048 [-0.015, +0.116] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.976 | -0.008 [-0.031, +0.012] |

## CurbRamp / `dinov2-reg-large` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.962 | -0.009 [-0.019, -0.000] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.927 | -0.012 [-0.033, +0.006] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.981 | -0.006 [-0.019, +0.003] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.927 | -0.004 [-0.047, +0.038] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.906 | -0.030 [-0.079, +0.012] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.921 | +0.026 [-0.034, +0.090] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.948 | -0.020 [-0.068, +0.016] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.973 | +0.024 [-0.010, +0.083] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `dinov2-reg-large` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.923 | +0.009 [-0.008, +0.027] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.877 | -0.021 [-0.072, +0.024] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `dinov2-reg-large` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.942 | +0.006 [-0.009, +0.021] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.821 | -0.030 [-0.094, +0.025] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.939 | -0.022 [-0.059, +0.000] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## CurbRamp / `dinov3-vitb16` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.952 | -0.017 [-0.028, -0.008] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.911 | -0.038 [-0.068, -0.011] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.901 | +0.005 [-0.031, +0.043] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.974 | -0.010 [-0.028, +0.002] |

## CurbRamp / `dinov3-vitb16` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.960 | -0.012 [-0.023, -0.002] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.899 | -0.041 [-0.069, -0.014] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.956 | -0.031 [-0.061, -0.008] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.925 | -0.005 [-0.041, +0.031] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.913 | -0.023 [-0.067, +0.014] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.869 | -0.026 [-0.075, +0.021] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.939 | -0.029 [-0.085, +0.013] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.916 | -0.033 [-0.103, +0.029] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `dinov3-vitb16` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.918 | +0.004 [-0.015, +0.022] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.821 | -0.077 [-0.138, -0.025] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `dinov3-vitb16` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.933 | -0.003 [-0.022, +0.016] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.781 | -0.070 [-0.138, -0.009] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.915 | -0.045 [-0.130, +0.006] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## CurbRamp / `dinov3-vitl16` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.958 | -0.011 [-0.019, -0.004] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.896 | -0.053 [-0.084, -0.025] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.945 | +0.049 [-0.002, +0.105] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.984 | -0.000 [-0.012, +0.010] |

## CurbRamp / `dinov3-vitl16` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.963 | -0.008 [-0.017, +0.001] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.917 | -0.022 [-0.044, -0.003] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.974 | -0.013 [-0.031, -0.001] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.908 | -0.023 [-0.062, +0.011] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.874 | -0.062 [-0.112, -0.018] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.924 | +0.029 [-0.032, +0.091] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.931 | -0.038 [-0.086, -0.003] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.859 | -0.089 [-0.230, +0.000] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `dinov3-vitl16` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.907 | -0.008 [-0.027, +0.013] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.837 | -0.061 [-0.138, +0.007] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `dinov3-vitl16` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.919 | -0.017 [-0.042, +0.006] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.788 | -0.062 [-0.131, +0.003] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.914 | -0.046 [-0.095, -0.012] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## CurbRamp / `dinov3-vitl16-256` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.959 | -0.010 [-0.018, -0.003] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.909 | -0.039 [-0.076, -0.012] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.908 | +0.012 [-0.048, +0.079] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.982 | -0.002 [-0.026, +0.020] |

## CurbRamp / `dinov3-vitl16-256` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.962 | -0.009 [-0.018, -0.001] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.917 | -0.022 [-0.042, -0.003] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.973 | -0.014 [-0.029, -0.003] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.897 | -0.034 [-0.083, +0.010] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.887 | -0.049 [-0.096, -0.010] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.897 | +0.002 [-0.063, +0.065] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.925 | -0.043 [-0.089, -0.009] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.875 | -0.073 [-0.197, +0.015] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `dinov3-vitl16-256` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.909 | -0.005 [-0.025, +0.014] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.868 | -0.030 [-0.082, +0.021] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `dinov3-vitl16-256` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.916 | -0.019 [-0.042, +0.002] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.814 | -0.037 [-0.110, +0.039] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.942 | -0.019 [-0.078, +0.045] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## CurbRamp / `rampnet-convnextv2-base` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1080 | 421 | 5,083 | 0.969 | 0.969 | 0.954 | -0.016 [-0.027, -0.006] |
| 2 | 196 | 122 | 837 | 0.950 | 0.949 | 0.901 | -0.048 [-0.088, -0.014] |
| 3 | 102 | 74 | 551 | 0.906 | 0.896 | 0.874 | -0.022 [-0.098, +0.052] |
| (none) | 101 | 50 | 454 | 0.982 | 0.984 | 0.987 | +0.004 [-0.031, +0.039] |

## CurbRamp / `rampnet-convnextv2-base` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 862 | 341 | 3,933 | 0.974 | 0.972 | 0.962 | -0.010 [-0.020, +0.001] |
| missing tactile warning | 303 | 147 | 1,482 | 0.935 | 0.939 | 0.885 | -0.054 [-0.089, -0.024] |
| points into traffic | 156 | 64 | 770 | 0.988 | 0.987 | 0.981 | -0.006 [-0.021, +0.005] |
| not level with street | 123 | 93 | 513 | 0.933 | 0.931 | 0.893 | -0.038 [-0.092, +0.006] |
| narrow | 95 | 57 | 451 | 0.925 | 0.936 | 0.886 | -0.051 [-0.112, +0.006] |
| surface problem | 79 | 41 | 351 | 0.909 | 0.895 | 0.899 | +0.004 [-0.076, +0.076] |
| not enough landing space | 69 | 32 | 348 | 0.952 | 0.968 | 0.873 | -0.095 [-0.183, -0.024] |
| steep | 50 | 41 | 226 | 0.957 | 0.949 | 0.935 | -0.014 [-0.093, +0.065] |
| debris / pooled water | 9 | 0 | 74 ⚠ thin |  |  |  |  |
| parallel lines | 7 | 0 | 14 ⚠ thin |  |  |  |  |
| not aligned with crosswalk | 3 | 1 | 35 ⚠ thin |  |  |  |  |

## CurbRamp / `rampnet-convnextv2-base` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 444 | 171 | 5,083 | 0.913 | 0.914 | 0.871 | -0.044 [-0.072, -0.017] |
| 2 | 55 | 27 | 837 | 0.873 | 0.898 | 0.868 | -0.030 [-0.111, +0.042] |
| 3 | 23 | 10 | 551 |  |  |  |  |
| (none) | 24 | 20 | 454 |  |  |  |  |

## CurbRamp / `rampnet-convnextv2-base` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| (none) | 329 | 150 | 3,933 | 0.925 | 0.935 | 0.896 | -0.039 [-0.069, -0.012] |
| missing tactile warning | 141 | 45 | 1,482 | 0.858 | 0.851 | 0.747 | -0.103 [-0.172, -0.034] |
| points into traffic | 73 | 22 | 770 | 0.937 | 0.961 | 0.892 | -0.069 [-0.168, -0.011] |
| not level with street | 28 | 14 | 513 |  |  |  |  |
| not enough landing space | 22 | 0 | 348 |  |  |  |  |
| surface problem | 20 | 9 | 351 |  |  |  |  |
| narrow | 18 | 6 | 451 |  |  |  |  |
| debris / pooled water | 8 | 1 | 74 ⚠ thin |  |  |  |  |
| steep | 5 | 2 | 226 |  |  |  |  |

## SurfaceProblem / `convnextv2-base` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.867 | -0.039 [-0.057, -0.022] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.847 | -0.024 [-0.051, +0.004] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.865 | -0.036 [-0.074, -0.003] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.881 | -0.004 [-0.070, +0.067] |

## SurfaceProblem / `convnextv2-base` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.819 | -0.072 [-0.103, -0.040] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.841 | -0.044 [-0.077, -0.011] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.861 | -0.040 [-0.073, -0.006] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.873 | +0.002 [-0.032, +0.035] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.799 | -0.058 [-0.113, -0.007] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.789 | -0.051 [-0.111, +0.004] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.870 | -0.083 [-0.154, -0.026] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.897 | -0.037 [-0.079, +0.006] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.869 | -0.000 [-0.097, +0.102] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.907 | +0.018 [-0.049, +0.083] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.852 | -0.050 [-0.144, +0.031] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `convnextv2-base` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.630 | -0.104 [-0.140, -0.066] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.586 | -0.067 [-0.140, +0.013] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.719 | -0.095 [-0.203, +0.007] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `convnextv2-base` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.572 | -0.056 [-0.130, +0.018] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.571 | -0.099 [-0.210, +0.018] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.837 | -0.029 [-0.113, +0.052] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.496 | -0.171 [-0.263, -0.084] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.520 | -0.073 [-0.200, +0.050] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.556 | -0.042 [-0.171, +0.095] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.697 | -0.099 [-0.250, +0.026] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.813 | +0.017 [-0.141, +0.204] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.846 | -0.121 [-0.275, -0.004] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov2-reg-large` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.904 | -0.001 [-0.014, +0.013] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.882 | +0.011 [-0.009, +0.031] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.912 | +0.011 [-0.012, +0.032] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.905 | +0.021 [-0.028, +0.070] |

## SurfaceProblem / `dinov2-reg-large` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.902 | +0.011 [-0.013, +0.033] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.885 | +0.001 [-0.022, +0.025] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.905 | +0.004 [-0.019, +0.025] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.871 | +0.000 [-0.026, +0.026] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.867 | +0.009 [-0.023, +0.042] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.840 | +0.000 [-0.040, +0.035] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.931 | -0.022 [-0.085, +0.019] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.930 | -0.004 [-0.036, +0.023] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.911 | +0.041 [-0.036, +0.124] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.870 | -0.019 [-0.087, +0.043] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.917 | +0.014 [-0.039, +0.068] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov2-reg-large` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.693 | -0.042 [-0.077, -0.007] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.734 | +0.081 [+0.024, +0.139] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.860 | +0.046 [-0.026, +0.134] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `dinov2-reg-large` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.570 | -0.057 [-0.124, +0.009] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.744 | +0.074 [-0.010, +0.172] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.861 | -0.005 [-0.089, +0.068] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.674 | +0.008 [-0.085, +0.090] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.650 | +0.058 [-0.039, +0.157] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.516 | -0.082 [-0.167, -0.008] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.742 | -0.055 [-0.169, +0.054] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.882 | +0.086 [-0.015, +0.217] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.942 | -0.025 [-0.105, +0.046] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitb16` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.900 | -0.006 [-0.021, +0.010] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.885 | +0.014 [-0.008, +0.035] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.905 | +0.004 [-0.019, +0.029] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.903 | +0.018 [-0.031, +0.072] |

## SurfaceProblem / `dinov3-vitb16` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.886 | -0.004 [-0.029, +0.021] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.892 | +0.008 [-0.018, +0.033] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.910 | +0.008 [-0.013, +0.031] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.905 | +0.034 [-0.002, +0.069] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.847 | -0.011 [-0.046, +0.025] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.842 | +0.002 [-0.047, +0.046] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.939 | -0.014 [-0.054, +0.021] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.903 | -0.031 [-0.067, +0.005] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.891 | +0.021 [-0.047, +0.100] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.870 | -0.019 [-0.099, +0.046] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.873 | -0.030 [-0.095, +0.032] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitb16` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.695 | -0.039 [-0.074, -0.005] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.674 | +0.021 [-0.045, +0.081] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.749 | -0.065 [-0.136, -0.005] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `dinov3-vitb16` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.605 | -0.023 [-0.088, +0.046] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.649 | -0.022 [-0.105, +0.060] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.816 | -0.050 [-0.126, +0.019] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.599 | -0.068 [-0.141, +0.005] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.545 | -0.048 [-0.157, +0.051] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.582 | -0.016 [-0.116, +0.098] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.669 | -0.127 [-0.275, -0.015] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.857 | +0.061 [-0.028, +0.192] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.921 | -0.046 [-0.146, +0.017] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.913 | +0.007 [-0.006, +0.021] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.891 | +0.020 [+0.003, +0.038] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.919 | +0.019 [-0.006, +0.044] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.899 | +0.014 [-0.040, +0.074] |

## SurfaceProblem / `dinov3-vitl16` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.908 | +0.018 [-0.006, +0.041] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.902 | +0.018 [-0.006, +0.041] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.907 | +0.005 [-0.019, +0.029] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.902 | +0.031 [+0.000, +0.061] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.869 | +0.011 [-0.025, +0.047] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.868 | +0.028 [-0.012, +0.064] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.930 | -0.022 [-0.077, +0.019] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.939 | +0.005 [-0.016, +0.026] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.914 | +0.044 [-0.016, +0.119] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.839 | -0.049 [-0.130, +0.031] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.898 | -0.005 [-0.069, +0.055] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.678 | -0.057 [-0.093, -0.021] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.595 | -0.058 [-0.118, +0.001] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.818 | +0.004 [-0.082, +0.086] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.532 | -0.096 [-0.166, -0.026] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.635 | -0.036 [-0.144, +0.080] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.845 | -0.021 [-0.081, +0.039] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.540 | -0.127 [-0.202, -0.053] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.509 | -0.084 [-0.204, +0.035] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.581 | -0.017 [-0.109, +0.083] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.680 | -0.116 [-0.221, -0.011] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.865 | +0.068 [-0.038, +0.194] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.917 | -0.050 [-0.141, +0.004] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16-256` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.906 | +0.000 [-0.016, +0.016] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.874 | +0.003 [-0.018, +0.025] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.893 | -0.008 [-0.031, +0.015] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.887 | +0.003 [-0.047, +0.056] |

## SurfaceProblem / `dinov3-vitl16-256` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.892 | +0.002 [-0.022, +0.027] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.896 | +0.012 [-0.013, +0.034] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.883 | -0.018 [-0.043, +0.006] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.879 | +0.008 [-0.037, +0.050] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.866 | +0.008 [-0.025, +0.043] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.855 | +0.015 [-0.027, +0.052] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.908 | -0.045 [-0.105, +0.003] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.935 | +0.000 [-0.024, +0.026] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.844 | -0.026 [-0.103, +0.058] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.853 | -0.036 [-0.108, +0.027] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.847 | -0.056 [-0.135, +0.009] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16-256` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.687 | -0.048 [-0.085, -0.012] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.636 | -0.017 [-0.067, +0.032] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.777 | -0.038 [-0.118, +0.038] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `dinov3-vitl16-256` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.481 | -0.146 [-0.223, -0.072] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.679 | +0.009 [-0.091, +0.120] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.784 | -0.082 [-0.160, -0.006] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.585 | -0.082 [-0.162, -0.009] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.535 | -0.058 [-0.165, +0.050] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.552 | -0.046 [-0.159, +0.060] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.665 | -0.132 [-0.252, -0.021] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.919 | +0.123 [+0.024, +0.257] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.929 | -0.038 [-0.115, +0.009] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |

## SurfaceProblem / `rampnet-convnextv2-base` seed 42 / test / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 1107 | 517 | 5,050 | 0.902 | 0.905 | 0.813 | -0.092 [-0.117, -0.069] |
| 2 | 647 | 263 | 2,959 | 0.884 | 0.871 | 0.824 | -0.048 [-0.082, -0.015] |
| 3 | 378 | 145 | 1,833 | 0.899 | 0.901 | 0.813 | -0.088 [-0.135, -0.043] |
| (none) | 123 | 74 | 658 | 0.868 | 0.885 | 0.852 | -0.033 [-0.108, +0.044] |

## SurfaceProblem / `rampnet-convnextv2-base` seed 42 / test / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| cracks | 662 | 161 | 3,031 | 0.889 | 0.890 | 0.802 | -0.088 [-0.134, -0.046] |
| grass | 528 | 153 | 2,636 | 0.881 | 0.884 | 0.776 | -0.108 [-0.150, -0.062] |
| bumpy | 437 | 174 | 1,982 | 0.906 | 0.901 | 0.832 | -0.069 [-0.104, -0.034] |
| (none) | 379 | 263 | 1,490 | 0.897 | 0.871 | 0.844 | -0.027 [-0.070, +0.013] |
| uneven/slanted | 328 | 97 | 1,574 | 0.859 | 0.858 | 0.735 | -0.122 [-0.188, -0.064] |
| narrow sidewalk | 217 | 100 | 1,160 | 0.843 | 0.840 | 0.794 | -0.046 [-0.104, +0.007] |
| height difference | 163 | 32 | 770 | 0.954 | 0.953 | 0.772 | -0.181 [-0.276, -0.079] |
| utility panel | 162 | 98 | 904 | 0.915 | 0.934 | 0.842 | -0.093 [-0.149, -0.037] |
| very broken | 113 | 30 | 625 | 0.886 | 0.870 | 0.740 | -0.130 [-0.289, +0.009] |
| brick/cobblestone | 77 | 51 | 337 | 0.884 | 0.889 | 0.847 | -0.042 [-0.125, +0.033] |
| sand/gravel | 65 | 29 | 284 | 0.891 | 0.903 | 0.791 | -0.112 [-0.209, -0.019] |
| construction | 19 | 5 | 44 ⚠ thin |  |  |  |  |
| debris | 14 | 6 | 66 ⚠ thin |  |  |  |  |
| rail/tram track | 5 | 3 | 25 ⚠ thin |  |  |  |  |

## SurfaceProblem / `rampnet-convnextv2-base` seed 42 / test_heldout / by severity

| severity | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 337 | 5,050 | 0.701 | 0.735 | 0.637 | -0.098 [-0.140, -0.055] |
| 2 | 174 | 79 | 2,959 | 0.672 | 0.653 | 0.644 | -0.010 [-0.077, +0.065] |
| 3 | 74 | 43 | 1,833 | 0.823 | 0.815 | 0.791 | -0.024 [-0.108, +0.057] |
| (none) | 25 | 17 | 658 |  |  |  |  |

## SurfaceProblem / `rampnet-convnextv2-base` seed 42 / test_heldout / by tag

| tag | n | incorrect | training rows | round-3 AUC | baseline AUC | arm AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| brick/cobblestone | 288 | 191 | 337 | 0.540 | 0.628 | 0.466 | -0.162 [-0.244, -0.083] |
| narrow sidewalk | 148 | 120 | 1,160 | 0.696 | 0.670 | 0.774 | +0.103 [+0.000, +0.215] |
| cracks | 113 | 28 | 3,031 | 0.901 | 0.866 | 0.795 | -0.070 [-0.185, +0.029] |
| uneven/slanted | 94 | 38 | 1,574 | 0.704 | 0.667 | 0.590 | -0.077 [-0.161, +0.006] |
| sand/gravel | 80 | 39 | 284 | 0.615 | 0.593 | 0.540 | -0.053 [-0.163, +0.062] |
| utility panel | 68 | 32 | 904 | 0.458 | 0.598 | 0.435 | -0.163 [-0.317, -0.012] |
| bumpy | 63 | 13 | 1,982 | 0.726 | 0.796 | 0.711 | -0.085 [-0.245, +0.069] |
| grass | 57 | 17 | 2,636 | 0.803 | 0.796 | 0.854 | +0.058 [-0.069, +0.220] |
| height difference | 31 | 16 | 770 | 0.929 | 0.967 | 0.992 | +0.025 [-0.019, +0.100] |
| debris | 29 | 5 | 66 ⚠ thin |  |  |  |  |
| very broken | 21 | 8 | 625 |  |  |  |  |
| (none) | 20 | 12 | 1,490 |  |  |  |  |
| rail/tram track | 20 | 9 | 25 ⚠ thin |  |  |  |  |
| construction | 6 | 5 | 44 ⚠ thin |  |  |  |  |
