# VLM baseline report (generated 2026-10-21)

best VLM by mean per-type AUC: gemma4-12b

headline set: 500 crops (balanced_subset per_type=100); models on it: gemma4-12b, production, qwen3vl-2b, qwen3vl-8b

## headline

| model | label_type | n | auc | balanced_accuracy | r_incorrect_p95 | r_incorrect_p92 | r_correct_p95 | r_correct_p92 |
|---|---|---|---|---|---|---|---|---|
| gemma4-12b | Crosswalk | 100 | 0.939 | 0.850 | 0.580 | 0.840 | 0.500 | 0.600 |
| gemma4-12b | CurbRamp | 100 | 0.826 | 0.610 | 0.220 | 0.280 | 0.420 | 0.420 |
| gemma4-12b | NoCurbRamp | 100 | 0.620 | 0.510 | 0.060 | 0.060 | 0.020 | 0.020 |
| gemma4-12b | Obstacle | 100 | 0.500 | 0.480 | 0.040 | 0.040 | 0.020 | 0.020 |
| gemma4-12b | SurfaceProblem | 100 | 0.700 | 0.650 | 0.180 | 0.240 | 0.040 | 0.040 |
| production | Crosswalk | 100 | 0.964 | 0.870 | 0.800 | 0.820 | 0.820 | 0.880 |
| production | CurbRamp | 100 | 0.929 | 0.900 | 0.660 | 0.760 | 0.000 | 0.540 |
| production | NoCurbRamp | 100 | 0.959 | 0.890 | 0.840 | 0.860 | 0.680 | 0.760 |
| production | Obstacle | 100 | 0.869 | 0.720 | 0.280 | 0.320 | 0.640 | 0.640 |
| production | SurfaceProblem | 100 | 0.721 | 0.640 | 0.200 | 0.240 | 0.120 | 0.120 |
| qwen3vl-2b | Crosswalk | 100 | 0.924 | 0.850 | 0.680 | 0.780 | 0.640 | 0.740 |
| qwen3vl-2b | CurbRamp | 100 | 0.570 | 0.500 | 0.020 | 0.020 | 0.060 | 0.060 |
| qwen3vl-2b | NoCurbRamp | 100 | 0.546 | 0.500 | 0.000 | 0.000 | 0.020 | 0.020 |
| qwen3vl-2b | Obstacle | 100 | 0.486 | 0.490 | 0.060 | 0.060 | 0.000 | 0.000 |
| qwen3vl-2b | SurfaceProblem | 100 | 0.618 | 0.510 | 0.020 | 0.020 | 0.040 | 0.040 |
| qwen3vl-8b | Crosswalk | 100 | 0.895 | 0.830 | 0.180 | 0.320 | 0.180 | 0.720 |
| qwen3vl-8b | CurbRamp | 100 | 0.761 | 0.660 | 0.080 | 0.080 | 0.400 | 0.400 |
| qwen3vl-8b | NoCurbRamp | 100 | 0.457 | 0.460 | 0.020 | 0.020 | 0.000 | 0.000 |
| qwen3vl-8b | Obstacle | 100 | 0.515 | 0.520 | 0.040 | 0.040 | 0.060 | 0.060 |
| qwen3vl-8b | SurfaceProblem | 100 | 0.656 | 0.600 | 0.180 | 0.180 | 0.000 | 0.000 |
| gemma4-12b | all (pooled) | 500 | 0.723 | 0.620 | 0.168 | 0.196 | 0.040 | 0.060 |
| production | all (pooled) | 500 | 0.885 | 0.804 | 0.464 | 0.552 | 0.028 | 0.448 |
| qwen3vl-2b | all (pooled) | 500 | 0.608 | 0.570 | 0.008 | 0.008 | 0.132 | 0.144 |
| qwen3vl-8b | all (pooled) | 500 | 0.675 | 0.614 | 0.036 | 0.064 | 0.016 | 0.016 |

## by split

| model | label_type | split | n | auc | balanced_accuracy | r_incorrect_p95 | r_incorrect_p92 | r_correct_p95 | r_correct_p92 |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | Crosswalk | test | 60 | 0.946 | 0.883 | 0.667 | 0.900 | 0.700 | 0.700 |
| gemma4-12b | Crosswalk | test_heldout | 40 | 0.930 | 0.800 | 0.500 | 0.800 | 0.450 | 0.650 |
| gemma4-12b | CurbRamp | test | 60 | 0.772 | 0.633 | 0.167 | 0.167 | 0.300 | 0.400 |
| gemma4-12b | CurbRamp | test_heldout | 40 | 0.912 | 0.575 | 0.600 | 0.600 | 0.600 | 0.600 |
| gemma4-12b | NoCurbRamp | test | 60 | 0.591 | 0.500 | 0.100 | 0.100 | 0.000 | 0.000 |
| gemma4-12b | NoCurbRamp | test_heldout | 40 | 0.672 | 0.525 | 0.050 | 0.050 | 0.050 | 0.050 |
| gemma4-12b | Obstacle | test | 60 | 0.499 | 0.517 | 0.033 | 0.033 | 0.033 | 0.033 |
| gemma4-12b | Obstacle | test_heldout | 40 | 0.522 | 0.425 | 0.050 | 0.050 | 0.150 | 0.150 |
| gemma4-12b | SurfaceProblem | test | 60 | 0.754 | 0.700 | 0.233 | 0.233 | 0.067 | 0.067 |
| gemma4-12b | SurfaceProblem | test_heldout | 40 | 0.627 | 0.575 | 0.150 | 0.150 | 0.150 | 0.150 |
| production | Crosswalk | test | 60 | 0.980 | 0.883 | 0.800 | 0.900 | 0.900 | 0.900 |
| production | Crosswalk | test_heldout | 40 | 0.930 | 0.850 | 0.800 | 0.800 | 0.250 | 0.600 |
| production | CurbRamp | test | 60 | 0.924 | 0.917 | 0.767 | 0.867 | 0.000 | 0.900 |
| production | CurbRamp | test_heldout | 40 | 0.930 | 0.875 | 0.700 | 0.750 | 0.450 | 0.450 |
| production | NoCurbRamp | test | 60 | 0.964 | 0.883 | 0.900 | 0.900 | 0.733 | 0.867 |
| production | NoCurbRamp | test_heldout | 40 | 0.955 | 0.900 | 0.800 | 0.850 | 0.600 | 0.700 |
| production | Obstacle | test | 60 | 0.853 | 0.733 | 0.400 | 0.433 | 0.500 | 0.567 |
| production | Obstacle | test_heldout | 40 | 0.890 | 0.700 | 0.200 | 0.200 | 0.750 | 0.850 |
| production | SurfaceProblem | test | 60 | 0.839 | 0.733 | 0.300 | 0.300 | 0.133 | 0.133 |
| production | SurfaceProblem | test_heldout | 40 | 0.525 | 0.500 | 0.150 | 0.150 | 0.200 | 0.200 |
| qwen3vl-2b | Crosswalk | test | 60 | 0.938 | 0.850 | 0.733 | 0.767 | 0.700 | 0.833 |
| qwen3vl-2b | Crosswalk | test_heldout | 40 | 0.910 | 0.850 | 0.100 | 0.750 | 0.550 | 0.750 |
| qwen3vl-2b | CurbRamp | test | 60 | 0.482 | 0.500 | 0.067 | 0.067 | 0.033 | 0.033 |
| qwen3vl-2b | CurbRamp | test_heldout | 40 | 0.682 | 0.500 | 0.050 | 0.050 | 0.100 | 0.100 |
| qwen3vl-2b | NoCurbRamp | test | 60 | 0.504 | 0.500 | 0.000 | 0.000 | 0.033 | 0.033 |
| qwen3vl-2b | NoCurbRamp | test_heldout | 40 | 0.605 | 0.500 | 0.400 | 0.400 | 0.000 | 0.000 |
| qwen3vl-2b | Obstacle | test | 60 | 0.491 | 0.500 | 0.000 | 0.000 | 0.033 | 0.033 |
| qwen3vl-2b | Obstacle | test_heldout | 40 | 0.460 | 0.475 | 0.150 | 0.150 | 0.000 | 0.000 |
| qwen3vl-2b | SurfaceProblem | test | 60 | 0.630 | 0.533 | 0.033 | 0.033 | 0.067 | 0.067 |
| qwen3vl-2b | SurfaceProblem | test_heldout | 40 | 0.585 | 0.475 | 0.000 | 0.000 | 0.050 | 0.050 |
| qwen3vl-8b | Crosswalk | test | 60 | 0.909 | 0.883 | 0.133 | 0.133 | 0.867 | 0.867 |
| qwen3vl-8b | Crosswalk | test_heldout | 40 | 0.867 | 0.750 | 0.350 | 0.350 | 0.300 | 0.300 |
| qwen3vl-8b | CurbRamp | test | 60 | 0.698 | 0.650 | 0.100 | 0.100 | 0.367 | 0.367 |
| qwen3vl-8b | CurbRamp | test_heldout | 40 | 0.853 | 0.675 | 0.450 | 0.450 | 0.400 | 0.400 |
| qwen3vl-8b | NoCurbRamp | test | 60 | 0.423 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3vl-8b | NoCurbRamp | test_heldout | 40 | 0.525 | 0.400 | 0.100 | 0.100 | 0.000 | 0.000 |
| qwen3vl-8b | Obstacle | test | 60 | 0.497 | 0.517 | 0.033 | 0.033 | 0.033 | 0.033 |
| qwen3vl-8b | Obstacle | test_heldout | 40 | 0.557 | 0.525 | 0.050 | 0.050 | 0.200 | 0.200 |
| qwen3vl-8b | SurfaceProblem | test | 60 | 0.750 | 0.650 | 0.267 | 0.267 | 0.000 | 0.000 |
| qwen3vl-8b | SurfaceProblem | test_heldout | 40 | 0.487 | 0.525 | 0.100 | 0.100 | 0.050 | 0.050 |
| gemma4-12b | all (pooled) | test | 300 | 0.730 | 0.647 | 0.167 | 0.187 | 0.053 | 0.053 |
| gemma4-12b | all (pooled) | test_heldout | 200 | 0.714 | 0.580 | 0.240 | 0.240 | 0.070 | 0.160 |
| production | all (pooled) | test | 300 | 0.911 | 0.830 | 0.573 | 0.627 | 0.027 | 0.627 |
| production | all (pooled) | test_heldout | 200 | 0.838 | 0.765 | 0.270 | 0.370 | 0.190 | 0.380 |
| qwen3vl-2b | all (pooled) | test | 300 | 0.605 | 0.577 | 0.020 | 0.020 | 0.147 | 0.180 |
| qwen3vl-2b | all (pooled) | test_heldout | 200 | 0.611 | 0.560 | 0.010 | 0.010 | 0.110 | 0.120 |
| qwen3vl-8b | all (pooled) | test | 300 | 0.691 | 0.640 | 0.027 | 0.160 | 0.007 | 0.007 |
| qwen3vl-8b | all (pooled) | test_heldout | 200 | 0.652 | 0.575 | 0.070 | 0.240 | 0.060 | 0.060 |

## hard

| slice | model | n | n_incorrect | auc | balanced_accuracy |
|---|---|---|---|---|---|
| hard == 1, all types, subset | gemma4-12b | 6 | 6 |  | 0.667 |
| hard == 1, all types, subset | production | 6 | 6 |  | 0.333 |
| hard == 1, all types, subset | qwen3vl-2b | 6 | 6 |  | 0.500 |
| hard == 1, all types, subset | qwen3vl-8b | 6 | 6 |  | 0.667 |
| hard == 1, all types, full sample | production | 20 | 10 | 0.220 | 0.200 |

## full sample (models that scored all 1988 crops)

| model | label_type | n | auc | balanced_accuracy | r_incorrect_p95 | r_incorrect_p92 | r_correct_p95 | r_correct_p92 |
|---|---|---|---|---|---|---|---|---|
| production | Crosswalk | 388 | 0.965 | 0.905 | 0.848 | 0.894 | 0.784 | 0.895 |
| production | CurbRamp | 400 | 0.930 | 0.873 | 0.715 | 0.805 | 0.195 | 0.750 |
| production | NoCurbRamp | 400 | 0.908 | 0.833 | 0.490 | 0.710 | 0.485 | 0.590 |
| production | Obstacle | 400 | 0.828 | 0.732 | 0.115 | 0.140 | 0.000 | 0.230 |
| production | SurfaceProblem | 400 | 0.822 | 0.725 | 0.155 | 0.195 | 0.255 | 0.375 |
| production | all (pooled) | 1988 | 0.891 | 0.813 | 0.351 | 0.582 | 0.118 | 0.441 |

## per city

| model | city | n | auc | balanced_accuracy |
|---|---|---|---|---|
| gemma4-12b | seattle-wa | 71 | 0.733 | 0.621 |
| gemma4-12b | chicago-il | 52 | 0.656 | 0.657 |
| gemma4-12b | other (8 private deployments) | 259 | 0.736 | 0.626 |
| production | seattle-wa | 71 | 0.942 | 0.856 |
| production | chicago-il | 52 | 0.911 | 0.822 |
| production | other (8 private deployments) | 259 | 0.891 | 0.817 |
| qwen3vl-2b | seattle-wa | 71 | 0.660 | 0.589 |
| qwen3vl-2b | chicago-il | 52 | 0.463 | 0.480 |
| qwen3vl-2b | other (8 private deployments) | 259 | 0.655 | 0.607 |
| qwen3vl-8b | seattle-wa | 71 | 0.650 | 0.598 |
| qwen3vl-8b | chicago-il | 52 | 0.710 | 0.713 |
| qwen3vl-8b | other (8 private deployments) | 259 | 0.679 | 0.594 |

## bootstrap

| label_type | model_a | model_b | n | auc_a | auc_b | diff | a_lo | a_hi | b_lo | b_hi | diff_lo | diff_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CurbRamp | gemma4-12b | production | 100 | 0.826 | 0.929 | -0.103 | 0.739 | 0.900 | 0.870 | 0.977 | -0.191 | -0.024 |
| NoCurbRamp | gemma4-12b | production | 100 | 0.620 | 0.959 | -0.339 | 0.510 | 0.728 | 0.922 | 0.988 | -0.449 | -0.232 |
| Obstacle | gemma4-12b | production | 100 | 0.500 | 0.869 | -0.369 | 0.379 | 0.618 | 0.791 | 0.933 | -0.482 | -0.261 |
| SurfaceProblem | gemma4-12b | production | 100 | 0.700 | 0.721 | -0.022 | 0.586 | 0.793 | 0.619 | 0.824 | -0.152 | 0.091 |
| Crosswalk | gemma4-12b | production | 100 | 0.939 | 0.964 | -0.026 | 0.892 | 0.980 | 0.929 | 0.990 | -0.066 | 0.013 |

## latency

| model | variant | n | latency_median_s | latency_p90_s | answer_mass_median | size_gb |
|---|---|---|---|---|---|---|
| gemma4-12b | definition | 500 | 3.945 | 5.618 | 0.984 | 11.020 |
| production | definition | 500 | 0.156 | 0.227 |  | 1.218 |
| qwen3vl-2b | definition | 500 | 1.355 | 1.487 | 0.979 | 1.798 |
| qwen3vl-8b | definition | 500 | 4.751 | 5.404 | 0.986 | 5.777 |
| qwen3vl-8b | no-definition | 100 | 2.446 | 2.586 | 0.986 | 5.777 |

## ablation

| model | label_type | variant | n | n_incorrect | auc | balanced_accuracy | accuracy | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3vl-8b | CurbRamp | definition | 100 | 50 | 0.761 | 0.660 | 0.660 | 0.400 | 0.400 | 0.080 | 0.080 |
| qwen3vl-8b | CurbRamp | no-definition | 100 | 50 | 0.556 | 0.550 | 0.550 | 0.000 | 0.000 | 0.020 | 0.020 |

## greedy check

| file | n | agree | unparsable |
|---|---|---|---|
| gemma4-12b | 10 | 10 | 0 |
| qwen3vl-2b | 10 | 10 | 0 |
| qwen3vl-8b | 10 | 9 | 0 |
