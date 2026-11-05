# Seed replicates: the training seed varied, everything else fixed

Every run of a variant that was repeated with another training seed (the Slurm script's `TAG=<variant>-s<seed>`), the chain's own run counted as seed 42. Each replicate is compared with the baseline trained with the *same* seed, so Δ holds the seed fixed on both sides; its interval resamples whole panoramas of the evaluation split (`analysis/slices.py`), the same resample for both models. The across-seed rows give the spread of the seed itself (sample sd over the seeds) next to the spread of the difference.

### CurbRamp

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.968 |  | 0.973 (19) | 0.907 |  | 0.838 (3) | 0.000 |
| baseline | 43 | 0.966 |  | 0.969 (19) | 0.914 |  | 0.857 (3) | 0.614 |
| baseline | 44 | 0.963 |  | 0.966 (19) | 0.902 |  | 0.828 (3) | 0.000 |
| cl-drop | 42 | 0.962 | -0.006 [-0.012, -0.000] | 0.962 (19) | 0.920 | +0.012 [-0.002, +0.028] | 0.876 (3) | 0.671 |
| cl-drop | 43 | 0.961 | -0.005 [-0.011, +0.001] | 0.967 (19) | 0.926 | +0.011 [-0.002, +0.026] | 0.884 (3) | 0.697 |
| cl-drop | 44 | 0.963 | -0.000 [-0.006, +0.006] | 0.964 (19) | 0.907 | +0.005 [-0.009, +0.019] | 0.847 (3) | 0.526 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.965 ± 0.002 | — | 0.908 ± 0.006 | — | 0.841 ± 0.015 | 0.205 ± 0.355 |
| cl-drop | 42, 43, 44 | 0.962 ± 0.001 | -0.004 [-0.006, -0.000], 0/3 > 0 | 0.917 ± 0.010 | +0.009 [+0.005, +0.012], 3/3 > 0 | 0.869 ± 0.019 | 0.632 ± 0.092 |

### Crosswalk

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.980 |  | 0.983 (8) | 0.941 |  | 0.946 (3) | 0.890 |
| baseline | 43 | 0.984 |  | 0.988 (8) | 0.941 |  | 0.957 (3) | 0.904 |
| baseline | 44 | 0.974 |  | 0.976 (8) | 0.966 |  | 0.973 (3) | 0.904 |
| cl-drop | 42 | 0.972 | -0.008 [-0.027, +0.005] | 0.970 (8) | 0.973 | +0.032 [+0.004, +0.067] | 0.980 (3) | 0.945 |
| cl-drop | 43 | 0.975 | -0.009 [-0.022, +0.000] | 0.975 (8) | 0.968 | +0.027 [-0.001, +0.059] | 0.975 (3) | 0.904 |
| cl-drop | 44 | 0.979 | +0.005 [-0.003, +0.014] | 0.985 (8) | 0.968 | +0.002 [-0.014, +0.015] | 0.976 (3) | 0.918 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.979 ± 0.005 | — | 0.949 ± 0.014 | — | 0.959 ± 0.014 | 0.900 ± 0.008 |
| cl-drop | 42, 43, 44 | 0.975 ± 0.004 | -0.004 [-0.009, +0.005], 1/3 > 0 | 0.970 ± 0.003 | +0.020 [+0.002, +0.032], 3/3 > 0 | 0.977 ± 0.003 | 0.922 ± 0.021 |

### NoCurbRamp

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.934 |  | 0.903 (18) | 0.881 |  | 0.861 (3) | 0.562 |
| baseline | 43 | 0.936 |  | 0.909 (18) | 0.912 |  | 0.886 (3) | 0.747 |
| baseline | 44 | 0.934 |  | 0.907 (18) | 0.921 |  | 0.909 (3) | 0.719 |
| cl-drop | 42 | 0.928 | -0.005 [-0.012, +0.002] | 0.908 (18) | 0.923 | +0.041 [+0.016, +0.063] | 0.904 (3) | 0.792 |
| cl-drop | 43 | 0.930 | -0.006 [-0.012, -0.000] | 0.903 (18) | 0.915 | +0.002 [-0.013, +0.019] | 0.889 (3) | 0.763 |
| cl-drop | 44 | 0.930 | -0.004 [-0.011, +0.002] | 0.905 (18) | 0.927 | +0.006 [-0.009, +0.023] | 0.914 (3) | 0.758 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.934 ± 0.001 | — | 0.905 ± 0.021 | — | 0.885 ± 0.024 | 0.676 ± 0.100 |
| cl-drop | 42, 43, 44 | 0.929 ± 0.001 | -0.005 [-0.006, -0.004], 0/3 > 0 | 0.921 ± 0.006 | +0.017 [+0.002, +0.041], 3/3 > 0 | 0.902 ± 0.013 | 0.771 ± 0.018 |

### Obstacle

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.883 |  | 0.870 (18) | 0.868 |  | 0.788 (3) | 0.529 |
| baseline | 43 | 0.879 |  | 0.856 (18) | 0.862 |  | 0.772 (3) | 0.458 |
| baseline | 44 | 0.884 |  | 0.864 (18) | 0.878 |  | 0.787 (3) | 0.543 |
| cl-drop | 42 | 0.891 | +0.009 [-0.001, +0.019] | 0.871 (18) | 0.890 | +0.022 [+0.009, +0.036] | 0.816 (3) | 0.704 |
| cl-drop | 43 | 0.885 | +0.006 [-0.003, +0.016] | 0.862 (18) | 0.880 | +0.018 [+0.005, +0.032] | 0.777 (3) | 0.622 |
| cl-drop | 44 | 0.886 | +0.002 [-0.008, +0.013] | 0.866 (18) | 0.864 | -0.014 [-0.028, -0.001] | 0.784 (3) | 0.492 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.882 ± 0.003 | — | 0.869 ± 0.008 | — | 0.782 ± 0.009 | 0.510 ± 0.046 |
| cl-drop | 42, 43, 44 | 0.887 ± 0.004 | +0.006 [+0.002, +0.009], 3/3 > 0 | 0.878 ± 0.013 | +0.009 [-0.014, +0.022], 2/3 > 0 | 0.792 ± 0.021 | 0.606 ± 0.107 |

### SurfaceProblem

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.896 |  | 0.858 (21) | 0.710 |  | 0.682 (3) | 0.000 |
| baseline | 43 | 0.910 |  | 0.878 (21) | 0.720 |  | 0.716 (3) | 0.103 |
| baseline | 44 | 0.893 |  | 0.851 (21) | 0.730 |  | 0.721 (3) | 0.000 |
| cl-drop | 42 | 0.901 | +0.005 [-0.003, +0.015] | 0.866 (21) | 0.686 | -0.024 [-0.047, +0.002] | 0.658 (3) | 0.000 |
| cl-drop | 43 | 0.903 | -0.007 [-0.015, +0.001] | 0.866 (21) | 0.698 | -0.022 [-0.044, +0.001] | 0.668 (3) | 0.000 |
| cl-drop | 44 | 0.898 | +0.005 [-0.004, +0.015] | 0.861 (21) | 0.719 | -0.011 [-0.034, +0.013] | 0.683 (3) | 0.000 |
| cl-flip | 42 | 0.897 | +0.001 [-0.008, +0.011] | 0.862 (21) | 0.726 | +0.016 [-0.009, +0.040] | 0.706 (3) | 0.000 |
| cl-flip | 43 | 0.893 | -0.016 [-0.025, -0.007] | 0.856 (21) | 0.725 | +0.005 [-0.018, +0.026] | 0.707 (3) | 0.000 |
| cl-flip | 44 | 0.889 | -0.004 [-0.015, +0.006] | 0.851 (21) | 0.710 | -0.020 [-0.044, +0.006] | 0.677 (3) | 0.000 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.899 ± 0.009 | — | 0.720 ± 0.010 | — | 0.706 ± 0.021 | 0.034 ± 0.059 |
| cl-drop | 42, 43, 44 | 0.900 ± 0.002 | +0.001 [-0.007, +0.005], 2/3 > 0 | 0.701 ± 0.016 | -0.019 [-0.024, -0.011], 0/3 > 0 | 0.670 ± 0.012 | 0.000 ± 0.000 |
| cl-flip | 42, 43, 44 | 0.893 ± 0.004 | -0.006 [-0.016, +0.001], 1/3 > 0 | 0.721 ± 0.009 | +0.000 [-0.020, +0.016], 2/3 > 0 | 0.696 ± 0.017 | 0.000 ± 0.000 |
