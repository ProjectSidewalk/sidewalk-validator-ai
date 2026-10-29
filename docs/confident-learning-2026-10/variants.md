### CurbRamp

| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | incorrect R@P0.92 held-out | correct R@P0.92 test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6925 | 0 | 0 | 0.968 |  | 0.973 (19) | 0.907 |  | 0.838 (3) | 0.091 (23) | 0.900 | 0.000 | 0.926 |
| symmetric | 5225 | 1700 | 0 | 0.959 | -0.009 [-0.015, -0.002] | 0.964 (19) | 0.889 | -0.018 [-0.038, +0.000] | 0.814 (3) | 0.000 (23) | 0.849 | 0.522 | 0.879 |
| unanimous3 | 4747 | 2178 | 0 | 0.962 | -0.005 [-0.012, +0.001] | 0.967 (19) | 0.900 | -0.007 [-0.027, +0.011] | 0.832 (3) | 0.000 (23) | 0.850 | 0.610 | 0.909 |
| cl-drop | 6380 | 545 | 0 | 0.962 | -0.006 [-0.012, -0.000] | 0.962 (19) | 0.920 | +0.012 [-0.002, +0.028] | 0.876 (3) | 0.000 (23) | 0.852 | 0.671 | 0.913 |
| cl-flip | 6925 | 0 | 545 | 0.958 | -0.009 [-0.015, -0.003] | 0.962 (19) | 0.916 | +0.009 [-0.005, +0.026] | 0.874 (3) | 0.000 (23) | 0.850 | 0.618 | 0.895 |

Held-out deployments, AUC per model (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 284 | 168 | 0.821 | 0.792 | 0.813 | 0.871 | 0.877 |
| private city B (US) | 202 | 30 | 0.881 | 0.876 | 0.882 | 0.878 | 0.846 |
| tucson-az | 60 | 30 | 0.958 | 0.943 | 0.968 | 0.960 | 0.954 |

Expert-rated train rows (aggregates; the rated file stays private):

| variant | rated train rows | target = expert | rows touched | of which expert overturns | rows untouched | of which expert overturns |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | 3 | 2 | 2 | 1 | 1 | 0 |
| unanimous3 | 3 | 2 | 2 | 1 | 1 | 0 |
| cl-drop | 3 | 2 | 0 | 0 | 3 | 1 |
| cl-flip | 3 | 2 | 0 | 0 | 3 | 1 |

### Crosswalk

| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | incorrect R@P0.92 held-out | correct R@P0.92 test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2089 | 0 | 0 | 0.980 |  | 0.983 (8) | 0.941 |  | 0.946 (3) | 0.750 (5) | 0.944 | 0.890 | 0.992 |
| symmetric | 1765 | 324 | 0 | 0.956 | -0.024 [-0.048, -0.006] | 0.973 (8) | 0.964 | +0.023 [+0.001, +0.050] | 0.959 (3) | 1.000 (5) | 0.939 | 0.945 | 0.996 |
| unanimous3 | 1479 | 610 | 0 | 0.963 | -0.017 [-0.046, +0.006] | 0.963 (8) | 0.948 | +0.007 [-0.022, +0.037] | 0.950 (3) | 0.500 (5) | 0.924 | 0.877 | 0.984 |
| cl-drop | 2008 | 81 | 0 | 0.972 | -0.008 [-0.027, +0.005] | 0.970 (8) | 0.973 | +0.032 [+0.004, +0.067] | 0.980 (3) | 0.750 (5) | 0.939 | 0.945 | 0.988 |
| cl-flip | 2089 | 0 | 81 | 0.977 | -0.003 [-0.009, +0.002] | 0.976 (8) | 0.972 | +0.031 [+0.003, +0.069] | 0.974 (3) | 1.000 (5) | 0.934 | 0.904 | 1.000 |

Held-out deployments, AUC per model (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 45 | 16 | 0.888 | 0.871 | 0.851 | 0.957 | 0.940 |
| private city B (US) | 77 | 47 | 0.972 | 0.990 | 0.984 | 0.991 | 0.984 |
| tucson-az | 16 | 10 | 0.767 | 0.917 | 0.917 | 0.908 | 1.000 |

Expert-rated train rows (aggregates; the rated file stays private):

| variant | rated train rows | target = expert | rows touched | of which expert overturns | rows untouched | of which expert overturns |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | 5 | 3 | 3 | 1 | 2 | 1 |
| unanimous3 | 5 | 3 | 3 | 1 | 2 | 1 |
| cl-drop | 5 | 3 | 0 | 0 | 5 | 2 |
| cl-flip | 5 | 3 | 0 | 0 | 5 | 2 |

### NoCurbRamp

| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | incorrect R@P0.92 held-out | correct R@P0.92 test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 12199 | 0 | 0 | 0.934 |  | 0.903 (18) | 0.881 |  | 0.861 (3) | 0.644 (90) | 0.656 | 0.562 | 0.836 |
| symmetric | 10236 | 1963 | 0 | 0.926 | -0.008 [-0.016, -0.000] | 0.866 (18) | 0.887 | +0.005 [-0.016, +0.028] | 0.871 (3) | 0.598 (90) | 0.721 | 0.578 | 0.816 |
| unanimous3 | 6891 | 5308 | 0 | 0.925 | -0.009 [-0.016, -0.002] | 0.878 (18) | 0.884 | +0.003 [-0.021, +0.027] | 0.880 (3) | 0.542 (90) | 0.601 | 0.359 | 0.808 |
| cl-drop | 10909 | 1290 | 0 | 0.928 | -0.005 [-0.012, +0.002] | 0.908 (18) | 0.923 | +0.041 [+0.016, +0.063] | 0.904 (3) | 0.584 (90) | 0.601 | 0.792 | 0.861 |
| cl-flip | 12199 | 0 | 1290 | 0.930 | -0.003 [-0.011, +0.004] | 0.909 (18) | 0.908 | +0.027 [+0.007, +0.049] | 0.896 (3) | 0.639 (90) | 0.556 | 0.487 | 0.834 |

Held-out deployments, AUC per model (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 125 | 121 | 0.800 | 0.769 | 0.800 | 0.866 | 0.741 |
| private city B (US) | 572 | 234 | 0.860 | 0.872 | 0.881 | 0.905 | 0.898 |
| tucson-az | 46 | 29 | 0.955 | 0.826 | 0.795 | 0.851 | 0.850 |

Expert-rated train rows (aggregates; the rated file stays private):

| variant | rated train rows | target = expert | rows touched | of which expert overturns | rows untouched | of which expert overturns |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | 13 | 11 | 3 | 0 | 10 | 2 |
| unanimous3 | 13 | 11 | 10 | 2 | 3 | 0 |
| cl-drop | 13 | 11 | 2 | 0 | 11 | 2 |
| cl-flip | 13 | 11 | 2 | 0 | 11 | 2 |

### Obstacle

| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | incorrect R@P0.92 held-out | correct R@P0.92 test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10554 | 0 | 0 | 0.883 |  | 0.870 (18) | 0.868 |  | 0.788 (3) | 0.556 (11) | 0.000 | 0.529 | 0.605 |
| symmetric | 7907 | 2647 | 0 | 0.869 | -0.014 [-0.024, -0.004] | 0.848 (18) | 0.844 | -0.024 [-0.039, -0.009] | 0.761 (3) | 0.694 (11) | 0.287 | 0.434 | 0.490 |
| unanimous3 | 6279 | 4275 | 0 | 0.856 | -0.027 [-0.039, -0.016] | 0.827 (18) | 0.860 | -0.008 [-0.023, +0.008] | 0.805 (3) | 0.889 (11) | 0.197 | 0.442 | 0.437 |
| cl-drop | 8608 | 1946 | 0 | 0.891 | +0.009 [-0.001, +0.019] | 0.871 (18) | 0.890 | +0.022 [+0.009, +0.036] | 0.816 (3) | 0.833 (11) | 0.000 | 0.704 | 0.000 |
| cl-flip | 10554 | 0 | 1946 | 0.878 | -0.005 [-0.016, +0.006] | 0.855 (18) | 0.877 | +0.009 [-0.006, +0.024] | 0.806 (3) | 0.667 (11) | 0.000 | 0.682 | 0.000 |

Held-out deployments, AUC per model (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 993 | 910 | 0.666 | 0.646 | 0.723 | 0.699 | 0.707 |
| private city B (US) | 628 | 196 | 0.896 | 0.865 | 0.882 | 0.921 | 0.898 |
| tucson-az | 191 | 119 | 0.792 | 0.746 | 0.769 | 0.809 | 0.782 |

Expert-rated train rows (aggregates; the rated file stays private):

| variant | rated train rows | target = expert | rows touched | of which expert overturns | rows untouched | of which expert overturns |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | 6 | 6 | 5 | 0 | 1 | 0 |
| unanimous3 | 6 | 6 | 5 | 0 | 1 | 0 |
| cl-drop | 6 | 6 | 0 | 0 | 6 | 0 |
| cl-flip | 6 | 6 | 0 | 0 | 6 | 0 |

### SurfaceProblem

| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | incorrect R@P0.92 held-out | correct R@P0.92 test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10500 | 0 | 0 | 0.896 |  | 0.858 (21) | 0.710 |  | 0.682 (3) | 0.323 (40) | 0.402 | 0.000 | 0.697 |
| symmetric | 7419 | 3081 | 0 | 0.863 | -0.032 [-0.045, -0.020] | 0.812 (21) | 0.624 | -0.087 [-0.115, -0.059] | 0.628 (3) | 0.357 (40) | 0.302 | 0.000 | 0.173 |
| unanimous3 | 6311 | 4189 | 0 | 0.866 | -0.030 [-0.043, -0.017] | 0.823 (21) | 0.655 | -0.055 [-0.085, -0.027] | 0.621 (3) | 0.364 (40) | 0.414 | 0.015 | 0.462 |
| cl-drop | 8998 | 1502 | 0 | 0.901 | +0.005 [-0.003, +0.015] | 0.866 (21) | 0.686 | -0.024 [-0.047, +0.002] | 0.658 (3) | 0.348 (40) | 0.000 | 0.000 | 0.714 |
| cl-flip | 10500 | 0 | 1502 | 0.897 | +0.001 [-0.008, +0.011] | 0.862 (21) | 0.726 | +0.016 [-0.009, +0.040] | 0.706 (3) | 0.476 (40) | 0.000 | 0.000 | 0.631 |

Held-out deployments, AUC per model (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | symmetric | unanimous3 | cl-drop | cl-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 356 | 214 | 0.612 | 0.596 | 0.538 | 0.600 | 0.653 |
| private city B (US) | 262 | 77 | 0.798 | 0.735 | 0.794 | 0.790 | 0.831 |
| tucson-az | 229 | 185 | 0.743 | 0.559 | 0.630 | 0.644 | 0.683 |

Expert-rated train rows (aggregates; the rated file stays private):

| variant | rated train rows | target = expert | rows touched | of which expert overturns | rows untouched | of which expert overturns |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | 10 | 6 | 7 | 2 | 3 | 2 |
| unanimous3 | 10 | 6 | 8 | 3 | 2 | 1 |
| cl-drop | 10 | 6 | 0 | 0 | 10 | 4 |
| cl-flip | 10 | 6 | 0 | 0 | 10 | 4 |
