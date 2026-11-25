# Seed replicates

Every run repeated with another training seed (`TAG=<run>-s<seed>`), the chain's own run counted as seed 42; each compared with the baseline of the same seed.

### NoCurbRamp

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.934 |  | 0.903 (18) | 0.881 |  | 0.861 (3) | 0.562 |
| baseline | 43 | 0.936 |  | 0.909 (18) | 0.912 |  | 0.886 (3) | 0.747 |
| baseline | 44 | 0.934 |  | 0.907 (18) | 0.921 |  | 0.909 (3) | 0.719 |
| baseline-rerun | 42 | 0.931 | -0.002 [-0.009, +0.004] | 0.905 (18) | 0.904 | +0.022 [+0.005, +0.041] | 0.894 (3) | 0.599 |
| baseline-rerun | 43 | 0.934 | -0.002 [-0.007, +0.003] | 0.907 (18) | 0.914 | +0.002 [-0.013, +0.017] | 0.890 (3) | 0.768 |
| baseline-rerun | 44 | 0.937 | +0.003 [-0.004, +0.010] | 0.909 (18) | 0.922 | +0.001 [-0.013, +0.015] | 0.903 (3) | 0.648 |
| baseline-rerun | 45 | 0.937 | no same-seed baseline | 0.923 (18) | 0.913 | no same-seed baseline | 0.892 (3) | 0.719 |
| baseline-rerun | 46 | 0.934 | no same-seed baseline | 0.903 (18) | 0.917 | no same-seed baseline | 0.892 (3) | 0.758 |
| two_scale | 42 | 0.936 | +0.002 [-0.005, +0.009] | 0.917 (18) | 0.917 | +0.036 [+0.015, +0.056] | 0.897 (3) | 0.721 |
| two_scale | 43 | 0.941 | +0.005 [-0.001, +0.011] | 0.922 (18) | 0.919 | +0.007 [-0.009, +0.023] | 0.900 (3) | 0.758 |
| two_scale | 44 | 0.941 | +0.007 [+0.001, +0.014] | 0.922 (18) | 0.924 | +0.003 [-0.012, +0.018] | 0.906 (3) | 0.724 |
| two_scale | 45 | 0.943 | no same-seed baseline | 0.922 (18) | 0.938 | no same-seed baseline | 0.924 (3) | 0.818 |
| two_scale | 46 | 0.942 | no same-seed baseline | 0.918 (18) | 0.933 | no same-seed baseline | 0.911 (3) | 0.826 |
| view_heatmap | 42 | 0.898 | -0.035 [-0.047, -0.025] | 0.851 (18) | 0.822 | -0.060 [-0.091, -0.029] | 0.779 (3) | 0.276 |
| view_heatmap | 43 | 0.901 | -0.035 [-0.046, -0.024] | 0.853 (18) | 0.814 | -0.099 [-0.134, -0.068] | 0.767 (3) | 0.188 |
| view_heatmap | 44 | 0.903 | -0.031 [-0.042, -0.020] | 0.866 (18) | 0.837 | -0.084 [-0.116, -0.058] | 0.805 (3) | 0.164 |
| view_heatmap | 45 | 0.905 | no same-seed baseline | 0.878 (18) | 0.837 | no same-seed baseline | 0.796 (3) | 0.424 |
| view_heatmap | 46 | 0.899 | no same-seed baseline | 0.862 (18) | 0.817 | no same-seed baseline | 0.784 (3) | 0.164 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.934 ± 0.001 | — | 0.905 ± 0.021 | — | 0.885 ± 0.024 | 0.676 ± 0.100 |
| baseline-rerun | 42, 43, 44, 45, 46 | 0.935 ± 0.002 | -0.000 [-0.002, +0.003], 1/3 > 0 | 0.914 ± 0.007 | +0.008 [+0.001, +0.022], 3/3 > 0 | 0.894 ± 0.005 | 0.698 ± 0.073 |
| two_scale | 42, 43, 44, 45, 46 | 0.940 ± 0.003 | +0.005 [+0.002, +0.007], 3/3 > 0 | 0.926 ± 0.009 | +0.015 [+0.003, +0.036], 3/3 > 0 | 0.907 ± 0.011 | 0.769 ± 0.050 |
| view_heatmap | 42, 43, 44, 45, 46 | 0.901 ± 0.003 | -0.034 [-0.035, -0.031], 0/3 > 0 | 0.825 ± 0.011 | -0.081 [-0.099, -0.060], 0/3 > 0 | 0.786 ± 0.015 | 0.243 ± 0.111 |

### Obstacle

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.883 |  | 0.870 (18) | 0.868 |  | 0.788 (3) | 0.529 |
| baseline | 43 | 0.879 |  | 0.856 (18) | 0.862 |  | 0.772 (3) | 0.458 |
| baseline | 44 | 0.884 |  | 0.864 (18) | 0.878 |  | 0.787 (3) | 0.543 |
| baseline-rerun | 42 | 0.887 | +0.004 [-0.003, +0.011] | 0.870 (18) | 0.858 | -0.009 [-0.021, +0.001] | 0.793 (3) | 0.426 |
| baseline-rerun | 43 | 0.885 | +0.006 [-0.002, +0.015] | 0.868 (18) | 0.865 | +0.003 [-0.010, +0.016] | 0.778 (3) | 0.371 |
| baseline-rerun | 44 | 0.884 | +0.001 [-0.007, +0.009] | 0.869 (18) | 0.876 | -0.002 [-0.013, +0.007] | 0.785 (3) | 0.500 |
| baseline-rerun | 45 | 0.884 | no same-seed baseline | 0.863 (18) | 0.859 | no same-seed baseline | 0.760 (3) | 0.380 |
| baseline-rerun | 46 | 0.881 | no same-seed baseline | 0.866 (18) | 0.865 | no same-seed baseline | 0.776 (3) | 0.442 |
| two_scale | 42 | 0.880 | -0.003 [-0.014, +0.007] | 0.863 (18) | 0.874 | +0.007 [-0.008, +0.021] | 0.794 (3) | 0.509 |
| two_scale | 43 | 0.892 | +0.013 [+0.003, +0.023] | 0.873 (18) | 0.877 | +0.014 [+0.001, +0.028] | 0.785 (3) | 0.519 |
| two_scale | 44 | 0.885 | +0.001 [-0.009, +0.012] | 0.869 (18) | 0.852 | -0.026 [-0.042, -0.012] | 0.787 (3) | 0.462 |
| two_scale | 45 | 0.887 | no same-seed baseline | 0.869 (18) | 0.857 | no same-seed baseline | 0.784 (3) | 0.527 |
| two_scale | 46 | 0.889 | no same-seed baseline | 0.868 (18) | 0.848 | no same-seed baseline | 0.768 (3) | 0.348 |
| view_heatmap | 42 | 0.845 | -0.037 [-0.054, -0.020] | 0.818 (18) | 0.805 | -0.063 [-0.086, -0.040] | 0.713 (3) | 0.000 |
| view_heatmap | 43 | 0.850 | -0.029 [-0.046, -0.012] | 0.823 (18) | 0.817 | -0.045 [-0.069, -0.021] | 0.718 (3) | 0.041 |
| view_heatmap | 44 | 0.848 | -0.036 [-0.053, -0.020] | 0.820 (18) | 0.792 | -0.086 [-0.112, -0.061] | 0.718 (3) | 0.000 |
| view_heatmap | 45 | 0.846 | no same-seed baseline | 0.818 (18) | 0.827 | no same-seed baseline | 0.728 (3) | 0.000 |
| view_heatmap | 46 | 0.838 | no same-seed baseline | 0.809 (18) | 0.817 | no same-seed baseline | 0.743 (3) | 0.000 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.882 ± 0.003 | — | 0.869 ± 0.008 | — | 0.782 ± 0.009 | 0.510 ± 0.046 |
| baseline-rerun | 42, 43, 44, 45, 46 | 0.884 ± 0.002 | +0.004 [+0.001, +0.006], 3/3 > 0 | 0.865 ± 0.007 | -0.003 [-0.009, +0.003], 1/3 > 0 | 0.778 ± 0.012 | 0.424 ± 0.052 |
| two_scale | 42, 43, 44, 45, 46 | 0.887 ± 0.004 | +0.004 [-0.003, +0.013], 2/3 > 0 | 0.862 ± 0.013 | -0.002 [-0.026, +0.014], 2/3 > 0 | 0.784 ± 0.010 | 0.473 ± 0.074 |
| view_heatmap | 42, 43, 44, 45, 46 | 0.845 ± 0.004 | -0.034 [-0.037, -0.029], 0/3 > 0 | 0.812 ± 0.014 | -0.065 [-0.086, -0.045], 0/3 > 0 | 0.724 ± 0.012 | 0.008 ± 0.018 |

### SurfaceProblem

| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.896 |  | 0.858 (21) | 0.710 |  | 0.682 (3) | 0.000 |
| baseline | 43 | 0.910 |  | 0.878 (21) | 0.720 |  | 0.716 (3) | 0.103 |
| baseline | 44 | 0.893 |  | 0.851 (21) | 0.730 |  | 0.721 (3) | 0.000 |
| baseline-rerun | 42 | 0.898 | +0.002 [-0.006, +0.010] | 0.863 (21) | 0.716 | +0.005 [-0.016, +0.027] | 0.677 (3) | 0.000 |
| baseline-rerun | 43 | 0.902 | -0.008 [-0.017, +0.001] | 0.864 (21) | 0.715 | -0.006 [-0.026, +0.017] | 0.703 (3) | 0.000 |
| baseline-rerun | 44 | 0.893 | +0.001 [-0.007, +0.008] | 0.850 (21) | 0.721 | -0.009 [-0.027, +0.009] | 0.701 (3) | 0.000 |
| baseline-rerun | 45 | 0.895 | no same-seed baseline | 0.860 (21) | 0.695 | no same-seed baseline | 0.676 (3) | 0.000 |
| baseline-rerun | 46 | 0.902 | no same-seed baseline | 0.862 (21) | 0.700 | no same-seed baseline | 0.702 (3) | 0.000 |
| two_scale | 42 | 0.891 | -0.004 [-0.014, +0.006] | 0.854 (21) | 0.693 | -0.017 [-0.042, +0.008] | 0.642 (3) | 0.000 |
| two_scale | 43 | 0.894 | -0.016 [-0.025, -0.006] | 0.862 (21) | 0.721 | +0.000 [-0.022, +0.023] | 0.699 (3) | 0.000 |
| two_scale | 44 | 0.898 | +0.005 [-0.005, +0.016] | 0.867 (21) | 0.725 | -0.005 [-0.029, +0.020] | 0.702 (3) | 0.097 |
| two_scale | 45 | 0.898 | no same-seed baseline | 0.861 (21) | 0.703 | no same-seed baseline | 0.709 (3) | 0.000 |
| two_scale | 46 | 0.893 | no same-seed baseline | 0.848 (21) | 0.710 | no same-seed baseline | 0.700 (3) | 0.000 |
| view_heatmap | 42 | 0.846 | -0.049 [-0.064, -0.036] | 0.769 (21) | 0.610 | -0.100 [-0.137, -0.066] | 0.614 (3) | 0.000 |
| view_heatmap | 43 | 0.846 | -0.064 [-0.079, -0.050] | 0.774 (21) | 0.638 | -0.083 [-0.115, -0.049] | 0.641 (3) | 0.000 |
| view_heatmap | 44 | 0.851 | -0.042 [-0.055, -0.028] | 0.785 (21) | 0.611 | -0.119 [-0.154, -0.079] | 0.605 (3) | 0.000 |
| view_heatmap | 45 | 0.850 | no same-seed baseline | 0.777 (21) | 0.601 | no same-seed baseline | 0.628 (3) | 0.013 |
| view_heatmap | 46 | 0.841 | no same-seed baseline | 0.768 (21) | 0.604 | no same-seed baseline | 0.597 (3) | 0.000 |

Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the number of seeds with Δ > 0):

| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | incorrect R@P0.92 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42, 43, 44 | 0.899 ± 0.009 | — | 0.720 ± 0.010 | — | 0.706 ± 0.021 | 0.034 ± 0.059 |
| baseline-rerun | 42, 43, 44, 45, 46 | 0.898 ± 0.004 | -0.002 [-0.008, +0.002], 2/3 > 0 | 0.709 ± 0.011 | -0.003 [-0.009, +0.005], 1/3 > 0 | 0.692 ± 0.014 | 0.000 ± 0.000 |
| two_scale | 42, 43, 44, 45, 46 | 0.895 ± 0.003 | -0.005 [-0.016, +0.005], 1/3 > 0 | 0.711 ± 0.013 | -0.007 [-0.017, +0.000], 1/3 > 0 | 0.691 ± 0.027 | 0.019 ± 0.043 |
| view_heatmap | 42, 43, 44, 45, 46 | 0.847 ± 0.004 | -0.052 [-0.064, -0.042], 0/3 > 0 | 0.613 ± 0.015 | -0.100 [-0.119, -0.083], 0/3 > 0 | 0.617 ± 0.018 | 0.003 ± 0.006 |
