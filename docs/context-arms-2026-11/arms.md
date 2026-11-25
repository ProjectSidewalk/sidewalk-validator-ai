# Context arms against the round-3 model

Each arm is trained with the round-3 recipe on the round-3 split and scored on the same rows; Δ = run − October's seed-42 round-3 model (`baseline`), descriptive only (the decision is in `decision.md`, against the baseline rerun's seed mean), with a 95% interval from resampling whole panoramas of the evaluation split, the same resample for both models. Within-city AUC: per-deployment AUCs weighted by n_incorrect x n_correct (deployments with at least 10 labels and both classes; count in brackets). Private deployments are lettered.

### NoCurbRamp

| run | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | macro F1 test / held-out | incorrect R@P0.92 / 0.95 held-out | correct R@P0.92 / 0.95 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.934 |  | 0.903 (18) | 0.881 |  | 0.861 (3) | 0.877 / 0.771 | 0.562 / 0.458 | 0.103 / 0.000 |
| baseline-rerun | 0.931 | -0.002 [-0.009, +0.004] | 0.905 (18) | 0.904 | +0.022 [+0.005, +0.041] | 0.894 (3) | 0.881 / 0.825 | 0.599 / 0.359 | 0.490 / 0.351 |
| two_scale | 0.936 | +0.002 [-0.005, +0.009] | 0.917 (18) | 0.917 | +0.036 [+0.015, +0.056] | 0.897 (3) | 0.867 / 0.833 | 0.721 / 0.404 | 0.593 / 0.000 |
| view_heatmap | 0.898 | -0.035 [-0.047, -0.025] | 0.851 (18) | 0.822 | -0.060 [-0.091, -0.029] | 0.779 (3) | 0.827 / 0.742 | 0.276 / 0.065 | 0.000 / 0.000 |
| view_heatmap-r448 | 0.917 | -0.017 [-0.026, -0.007] | 0.893 (18) | 0.899 | +0.018 [-0.007, +0.045] | 0.875 (3) | 0.838 / 0.784 | 0.544 / 0.471 | 0.373 / 0.000 |
| side_info | 0.931 | -0.002 [-0.009, +0.005] | 0.908 (18) | 0.905 | +0.023 [+0.002, +0.044] | 0.886 (3) | 0.879 / 0.820 | 0.617 / 0.471 | 0.000 / 0.000 |
| side_info_nohq | 0.933 | -0.001 [-0.008, +0.006] | 0.908 (18) | 0.905 | +0.023 [+0.006, +0.041] | 0.885 (3) | 0.878 / 0.821 | 0.721 / 0.448 | 0.409 / 0.000 |

Held-out deployments, AUC per run (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | baseline-rerun | two_scale | view_heatmap | view_heatmap-r448 | side_info | side_info_nohq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 125 | 121 | 0.800 | 0.855 | 0.876 | 0.977 | 0.924 | 0.866 | 0.913 |
| private city B (US) | 572 | 234 | 0.860 | 0.894 | 0.897 | 0.779 | 0.875 | 0.887 | 0.885 |
| tucson-az | 46 | 29 | 0.955 | 0.796 | 0.872 | 0.623 | 0.759 | 0.801 | 0.830 |

### Obstacle

| run | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | macro F1 test / held-out | incorrect R@P0.92 / 0.95 held-out | correct R@P0.92 / 0.95 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.883 |  | 0.870 (18) | 0.868 |  | 0.788 (3) | 0.808 / 0.716 | 0.529 / 0.265 | 0.000 / 0.000 |
| baseline-rerun | 0.887 | +0.004 [-0.003, +0.011] | 0.870 (18) | 0.858 | -0.009 [-0.021, +0.001] | 0.793 (3) | 0.812 / 0.730 | 0.426 / 0.125 | 0.000 / 0.000 |
| two_scale | 0.880 | -0.003 [-0.014, +0.007] | 0.863 (18) | 0.874 | +0.007 [-0.008, +0.021] | 0.794 (3) | 0.797 / 0.739 | 0.509 / 0.165 | 0.540 / 0.000 |
| view_heatmap | 0.845 | -0.037 [-0.054, -0.020] | 0.818 (18) | 0.805 | -0.063 [-0.086, -0.040] | 0.713 (3) | 0.765 / 0.668 | 0.000 / 0.000 | 0.000 / 0.000 |
| view_heatmap-r448 | 0.882 | -0.000 [-0.012, +0.013] | 0.863 (18) | 0.864 | -0.003 [-0.022, +0.016] | 0.778 (3) | 0.799 / 0.736 | 0.395 / 0.000 | 0.516 / 0.000 |
| side_info | 0.878 | -0.005 [-0.014, +0.004] | 0.855 (18) | 0.855 | -0.013 [-0.026, +0.000] | 0.774 (3) | 0.795 / 0.688 | 0.455 / 0.162 | 0.480 / 0.000 |
| side_info_nohq | 0.880 | -0.003 [-0.013, +0.007] | 0.855 (18) | 0.873 | +0.006 [-0.010, +0.020] | 0.808 (3) | 0.802 / 0.760 | 0.643 / 0.238 | 0.458 / 0.385 |

Held-out deployments, AUC per run (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | baseline-rerun | two_scale | view_heatmap | view_heatmap-r448 | side_info | side_info_nohq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 993 | 910 | 0.666 | 0.671 | 0.670 | 0.601 | 0.639 | 0.650 | 0.685 |
| private city B (US) | 628 | 196 | 0.896 | 0.903 | 0.908 | 0.827 | 0.909 | 0.883 | 0.912 |
| tucson-az | 191 | 119 | 0.792 | 0.775 | 0.764 | 0.579 | 0.700 | 0.786 | 0.850 |

### SurfaceProblem

| run | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | within-city held-out | macro F1 test / held-out | incorrect R@P0.92 / 0.95 held-out | correct R@P0.92 / 0.95 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.896 |  | 0.858 (21) | 0.710 |  | 0.682 (3) | 0.830 / 0.653 | 0.000 / 0.000 | 0.000 / 0.000 |
| baseline-rerun | 0.898 | +0.002 [-0.006, +0.010] | 0.863 (21) | 0.716 | +0.005 [-0.016, +0.027] | 0.677 (3) | 0.820 / 0.648 | 0.000 / 0.000 | 0.000 / 0.000 |
| two_scale | 0.891 | -0.004 [-0.014, +0.006] | 0.854 (21) | 0.693 | -0.017 [-0.042, +0.008] | 0.642 (3) | 0.816 / 0.649 | 0.000 / 0.000 | 0.000 / 0.000 |
| view_heatmap | 0.846 | -0.049 [-0.064, -0.036] | 0.769 (21) | 0.610 | -0.100 [-0.137, -0.066] | 0.614 (3) | 0.769 / 0.571 | 0.000 / 0.000 | 0.000 / 0.000 |
| view_heatmap-r448 | 0.869 | -0.027 [-0.041, -0.014] | 0.812 (21) | 0.655 | -0.056 [-0.091, -0.024] | 0.675 (3) | 0.796 / 0.611 | 0.000 / 0.000 | 0.000 / 0.000 |
| side_info | 0.894 | -0.002 [-0.011, +0.008] | 0.850 (21) | 0.694 | -0.017 [-0.042, +0.006] | 0.667 (3) | 0.826 / 0.650 | 0.000 / 0.000 | 0.000 / 0.000 |
| side_info_nohq | 0.898 | +0.002 [-0.006, +0.011] | 0.858 (21) | 0.714 | +0.003 [-0.020, +0.026] | 0.687 (3) | 0.831 / 0.653 | 0.000 / 0.000 | 0.000 / 0.000 |

Held-out deployments, AUC per run (blank under 10 labels or with one class):

| deployment | n | incorrect | baseline | baseline-rerun | two_scale | view_heatmap | view_heatmap-r448 | side_info | side_info_nohq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| private city A (outside the US) | 356 | 214 | 0.612 | 0.601 | 0.548 | 0.571 | 0.642 | 0.629 | 0.633 |
| private city B (US) | 262 | 77 | 0.798 | 0.836 | 0.810 | 0.688 | 0.744 | 0.806 | 0.813 |
| tucson-az | 229 | 185 | 0.743 | 0.682 | 0.699 | 0.644 | 0.676 | 0.564 | 0.670 |
