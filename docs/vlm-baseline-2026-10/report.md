# VLM baseline report (generated 2026-10-21)

best VLM by mean per-type AUC on the subset: gemma4-12b

headline set: 500 crops (balanced_subset per_type=100); models on it: gemma4-12b, production, qwen3vl-2b, qwen3vl-8b; on the full sample: gemma4-12b, production

## headline (subset)

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

## full sample

| model | label_type | n | auc | balanced_accuracy | r_incorrect_p95 | r_incorrect_p92 | r_correct_p95 | r_correct_p92 |
|---|---|---|---|---|---|---|---|---|
| gemma4-12b | Crosswalk | 388 | 0.944 | 0.854 | 0.707 | 0.828 | 0.511 | 0.826 |
| gemma4-12b | CurbRamp | 400 | 0.790 | 0.595 | 0.195 | 0.200 | 0.335 | 0.360 |
| gemma4-12b | NoCurbRamp | 400 | 0.634 | 0.507 | 0.010 | 0.010 | 0.010 | 0.010 |
| gemma4-12b | Obstacle | 400 | 0.508 | 0.505 | 0.015 | 0.015 | 0.015 | 0.015 |
| gemma4-12b | SurfaceProblem | 400 | 0.735 | 0.665 | 0.000 | 0.115 | 0.030 | 0.030 |
| production | Crosswalk | 388 | 0.965 | 0.905 | 0.848 | 0.894 | 0.784 | 0.895 |
| production | CurbRamp | 400 | 0.930 | 0.873 | 0.715 | 0.805 | 0.195 | 0.750 |
| production | NoCurbRamp | 400 | 0.908 | 0.833 | 0.490 | 0.710 | 0.485 | 0.590 |
| production | Obstacle | 400 | 0.828 | 0.732 | 0.115 | 0.140 | 0.000 | 0.230 |
| production | SurfaceProblem | 400 | 0.822 | 0.725 | 0.155 | 0.195 | 0.255 | 0.375 |
| gemma4-12b | all (pooled) | 1988 | 0.724 | 0.623 | 0.163 | 0.191 | 0.033 | 0.033 |
| production | all (pooled) | 1988 | 0.891 | 0.813 | 0.351 | 0.582 | 0.118 | 0.441 |

## by split

| model | label_type | split | n | n_incorrect | auc | balanced_accuracy | accuracy | recall_correct | recall_incorrect | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 | set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | Crosswalk | test | 60 | 30 | 0.946 | 0.883 | 0.883 | 0.867 | 0.900 | 0.700 | 0.700 | 0.900 | 0.667 | subset |
| gemma4-12b | Crosswalk | test_heldout | 40 | 20 | 0.930 | 0.800 | 0.800 | 0.700 | 0.900 | 0.650 | 0.450 | 0.800 | 0.500 | subset |
| gemma4-12b | CurbRamp | test | 60 | 30 | 0.772 | 0.633 | 0.633 | 0.267 | 1.000 | 0.400 | 0.300 | 0.167 | 0.167 | subset |
| gemma4-12b | CurbRamp | test_heldout | 40 | 20 | 0.912 | 0.575 | 0.575 | 0.150 | 1.000 | 0.600 | 0.600 | 0.600 | 0.600 | subset |
| gemma4-12b | NoCurbRamp | test | 60 | 30 | 0.591 | 0.500 | 0.500 | 0.067 | 0.933 | 0.000 | 0.000 | 0.100 | 0.100 | subset |
| gemma4-12b | NoCurbRamp | test_heldout | 40 | 20 | 0.672 | 0.525 | 0.525 | 0.100 | 0.950 | 0.050 | 0.050 | 0.050 | 0.050 | subset |
| gemma4-12b | Obstacle | test | 60 | 30 | 0.499 | 0.517 | 0.517 | 0.367 | 0.667 | 0.033 | 0.033 | 0.033 | 0.033 | subset |
| gemma4-12b | Obstacle | test_heldout | 40 | 20 | 0.522 | 0.425 | 0.425 | 0.450 | 0.400 | 0.150 | 0.150 | 0.050 | 0.050 | subset |
| gemma4-12b | SurfaceProblem | test | 60 | 30 | 0.754 | 0.700 | 0.700 | 0.567 | 0.833 | 0.067 | 0.067 | 0.233 | 0.233 | subset |
| gemma4-12b | SurfaceProblem | test_heldout | 40 | 20 | 0.627 | 0.575 | 0.575 | 0.400 | 0.750 | 0.150 | 0.150 | 0.150 | 0.150 | subset |
| production | Crosswalk | test | 60 | 30 | 0.980 | 0.883 | 0.883 | 0.967 | 0.800 | 0.900 | 0.900 | 0.900 | 0.800 | subset |
| production | Crosswalk | test_heldout | 40 | 20 | 0.930 | 0.850 | 0.850 | 0.850 | 0.850 | 0.600 | 0.250 | 0.800 | 0.800 | subset |
| production | CurbRamp | test | 60 | 30 | 0.924 | 0.917 | 0.917 | 0.900 | 0.933 | 0.900 | 0.000 | 0.867 | 0.767 | subset |
| production | CurbRamp | test_heldout | 40 | 20 | 0.930 | 0.875 | 0.875 | 0.850 | 0.900 | 0.450 | 0.450 | 0.750 | 0.700 | subset |
| production | NoCurbRamp | test | 60 | 30 | 0.964 | 0.883 | 0.883 | 0.867 | 0.900 | 0.867 | 0.733 | 0.900 | 0.900 | subset |
| production | NoCurbRamp | test_heldout | 40 | 20 | 0.955 | 0.900 | 0.900 | 1.000 | 0.800 | 0.700 | 0.600 | 0.850 | 0.800 | subset |
| production | Obstacle | test | 60 | 30 | 0.853 | 0.733 | 0.733 | 0.833 | 0.633 | 0.567 | 0.500 | 0.433 | 0.400 | subset |
| production | Obstacle | test_heldout | 40 | 20 | 0.890 | 0.700 | 0.700 | 0.850 | 0.550 | 0.850 | 0.750 | 0.200 | 0.200 | subset |
| production | SurfaceProblem | test | 60 | 30 | 0.839 | 0.733 | 0.733 | 0.833 | 0.633 | 0.133 | 0.133 | 0.300 | 0.300 | subset |
| production | SurfaceProblem | test_heldout | 40 | 20 | 0.525 | 0.500 | 0.500 | 0.600 | 0.400 | 0.200 | 0.200 | 0.150 | 0.150 | subset |
| qwen3vl-2b | Crosswalk | test | 60 | 30 | 0.938 | 0.850 | 0.850 | 0.900 | 0.800 | 0.833 | 0.700 | 0.767 | 0.733 | subset |
| qwen3vl-2b | Crosswalk | test_heldout | 40 | 20 | 0.910 | 0.850 | 0.850 | 0.750 | 0.950 | 0.750 | 0.550 | 0.750 | 0.100 | subset |
| qwen3vl-2b | CurbRamp | test | 60 | 30 | 0.482 | 0.500 | 0.500 | 0.000 | 1.000 | 0.033 | 0.033 | 0.067 | 0.067 | subset |
| qwen3vl-2b | CurbRamp | test_heldout | 40 | 20 | 0.682 | 0.500 | 0.500 | 0.000 | 1.000 | 0.100 | 0.100 | 0.050 | 0.050 | subset |
| qwen3vl-2b | NoCurbRamp | test | 60 | 30 | 0.504 | 0.500 | 0.500 | 0.367 | 0.633 | 0.033 | 0.033 | 0.000 | 0.000 | subset |
| qwen3vl-2b | NoCurbRamp | test_heldout | 40 | 20 | 0.605 | 0.500 | 0.500 | 0.300 | 0.700 | 0.000 | 0.000 | 0.400 | 0.400 | subset |
| qwen3vl-2b | Obstacle | test | 60 | 30 | 0.491 | 0.500 | 0.500 | 0.000 | 1.000 | 0.033 | 0.033 | 0.000 | 0.000 | subset |
| qwen3vl-2b | Obstacle | test_heldout | 40 | 20 | 0.460 | 0.475 | 0.475 | 0.000 | 0.950 | 0.000 | 0.000 | 0.150 | 0.150 | subset |
| qwen3vl-2b | SurfaceProblem | test | 60 | 30 | 0.630 | 0.533 | 0.533 | 0.333 | 0.733 | 0.067 | 0.067 | 0.033 | 0.033 | subset |
| qwen3vl-2b | SurfaceProblem | test_heldout | 40 | 20 | 0.585 | 0.475 | 0.475 | 0.300 | 0.650 | 0.050 | 0.050 | 0.000 | 0.000 | subset |
| qwen3vl-8b | Crosswalk | test | 60 | 30 | 0.909 | 0.883 | 0.883 | 0.867 | 0.900 | 0.867 | 0.867 | 0.133 | 0.133 | subset |
| qwen3vl-8b | Crosswalk | test_heldout | 40 | 20 | 0.867 | 0.750 | 0.750 | 0.750 | 0.750 | 0.300 | 0.300 | 0.350 | 0.350 | subset |
| qwen3vl-8b | CurbRamp | test | 60 | 30 | 0.698 | 0.650 | 0.650 | 0.300 | 1.000 | 0.367 | 0.367 | 0.100 | 0.100 | subset |
| qwen3vl-8b | CurbRamp | test_heldout | 40 | 20 | 0.853 | 0.675 | 0.675 | 0.400 | 0.950 | 0.400 | 0.400 | 0.450 | 0.450 | subset |
| qwen3vl-8b | NoCurbRamp | test | 60 | 30 | 0.423 | 0.500 | 0.500 | 0.033 | 0.967 | 0.000 | 0.000 | 0.000 | 0.000 | subset |
| qwen3vl-8b | NoCurbRamp | test_heldout | 40 | 20 | 0.525 | 0.400 | 0.400 | 0.000 | 0.800 | 0.000 | 0.000 | 0.100 | 0.100 | subset |
| qwen3vl-8b | Obstacle | test | 60 | 30 | 0.497 | 0.517 | 0.517 | 0.967 | 0.067 | 0.033 | 0.033 | 0.033 | 0.033 | subset |
| qwen3vl-8b | Obstacle | test_heldout | 40 | 20 | 0.557 | 0.525 | 0.525 | 1.000 | 0.050 | 0.200 | 0.200 | 0.050 | 0.050 | subset |
| qwen3vl-8b | SurfaceProblem | test | 60 | 30 | 0.750 | 0.650 | 0.650 | 0.733 | 0.567 | 0.000 | 0.000 | 0.267 | 0.267 | subset |
| qwen3vl-8b | SurfaceProblem | test_heldout | 40 | 20 | 0.487 | 0.525 | 0.525 | 0.450 | 0.600 | 0.050 | 0.050 | 0.100 | 0.100 | subset |
| gemma4-12b | all (pooled) | test | 300 | 150 | 0.730 | 0.647 | 0.647 | 0.427 | 0.867 | 0.053 | 0.053 | 0.187 | 0.167 | subset |
| gemma4-12b | all (pooled) | test_heldout | 200 | 100 | 0.714 | 0.580 | 0.580 | 0.360 | 0.800 | 0.160 | 0.070 | 0.240 | 0.240 | subset |
| production | all (pooled) | test | 300 | 150 | 0.911 | 0.830 | 0.830 | 0.880 | 0.780 | 0.627 | 0.027 | 0.627 | 0.573 | subset |
| production | all (pooled) | test_heldout | 200 | 100 | 0.838 | 0.765 | 0.765 | 0.830 | 0.700 | 0.380 | 0.190 | 0.370 | 0.270 | subset |
| qwen3vl-2b | all (pooled) | test | 300 | 150 | 0.605 | 0.577 | 0.577 | 0.320 | 0.833 | 0.180 | 0.147 | 0.020 | 0.020 | subset |
| qwen3vl-2b | all (pooled) | test_heldout | 200 | 100 | 0.611 | 0.560 | 0.560 | 0.270 | 0.850 | 0.120 | 0.110 | 0.010 | 0.010 | subset |
| qwen3vl-8b | all (pooled) | test | 300 | 150 | 0.691 | 0.640 | 0.640 | 0.580 | 0.700 | 0.007 | 0.007 | 0.160 | 0.027 | subset |
| qwen3vl-8b | all (pooled) | test_heldout | 200 | 100 | 0.652 | 0.575 | 0.575 | 0.520 | 0.630 | 0.060 | 0.060 | 0.240 | 0.070 | subset |
| gemma4-12b | Crosswalk | test | 250 | 125 | 0.967 | 0.896 | 0.896 | 0.848 | 0.944 | 0.904 | 0.816 | 0.888 | 0.856 | full |
| gemma4-12b | Crosswalk | test_heldout | 138 | 73 | 0.893 | 0.774 | 0.783 | 0.631 | 0.918 | 0.354 | 0.292 | 0.630 | 0.575 | full |
| gemma4-12b | CurbRamp | test | 250 | 125 | 0.763 | 0.604 | 0.604 | 0.208 | 1.000 | 0.352 | 0.336 | 0.184 | 0.176 | full |
| gemma4-12b | CurbRamp | test_heldout | 150 | 75 | 0.838 | 0.580 | 0.580 | 0.160 | 1.000 | 0.547 | 0.360 | 0.373 | 0.373 | full |
| gemma4-12b | NoCurbRamp | test | 250 | 125 | 0.616 | 0.500 | 0.500 | 0.040 | 0.960 | 0.008 | 0.008 | 0.016 | 0.016 | full |
| gemma4-12b | NoCurbRamp | test_heldout | 150 | 75 | 0.673 | 0.520 | 0.520 | 0.067 | 0.973 | 0.027 | 0.027 | 0.080 | 0.080 | full |
| gemma4-12b | Obstacle | test | 250 | 125 | 0.500 | 0.508 | 0.508 | 0.440 | 0.576 | 0.024 | 0.024 | 0.024 | 0.024 | full |
| gemma4-12b | Obstacle | test_heldout | 150 | 75 | 0.518 | 0.500 | 0.500 | 0.507 | 0.493 | 0.000 | 0.000 | 0.000 | 0.000 | full |
| gemma4-12b | SurfaceProblem | test | 250 | 125 | 0.777 | 0.688 | 0.688 | 0.560 | 0.816 | 0.032 | 0.032 | 0.240 | 0.000 | full |
| gemma4-12b | SurfaceProblem | test_heldout | 150 | 75 | 0.668 | 0.627 | 0.627 | 0.413 | 0.840 | 0.160 | 0.107 | 0.080 | 0.080 | full |
| production | Crosswalk | test | 250 | 125 | 0.979 | 0.940 | 0.940 | 0.992 | 0.888 | 0.976 | 0.960 | 0.960 | 0.952 | full |
| production | Crosswalk | test_heldout | 138 | 73 | 0.919 | 0.841 | 0.841 | 0.846 | 0.836 | 0.569 | 0.338 | 0.822 | 0.822 | full |
| production | CurbRamp | test | 250 | 125 | 0.945 | 0.912 | 0.912 | 0.920 | 0.904 | 0.872 | 0.232 | 0.880 | 0.824 | full |
| production | CurbRamp | test_heldout | 150 | 75 | 0.890 | 0.807 | 0.807 | 0.787 | 0.827 | 0.387 | 0.373 | 0.680 | 0.640 | full |
| production | NoCurbRamp | test | 250 | 125 | 0.899 | 0.816 | 0.816 | 0.800 | 0.832 | 0.552 | 0.456 | 0.552 | 0.216 | full |
| production | NoCurbRamp | test_heldout | 150 | 75 | 0.936 | 0.860 | 0.860 | 0.973 | 0.747 | 0.653 | 0.453 | 0.800 | 0.747 | full |
| production | Obstacle | test | 250 | 125 | 0.821 | 0.740 | 0.740 | 0.824 | 0.656 | 0.000 | 0.000 | 0.192 | 0.160 | full |
| production | Obstacle | test_heldout | 150 | 75 | 0.838 | 0.720 | 0.720 | 0.813 | 0.627 | 0.693 | 0.653 | 0.053 | 0.053 | full |
| production | SurfaceProblem | test | 250 | 125 | 0.899 | 0.796 | 0.796 | 0.904 | 0.688 | 0.640 | 0.336 | 0.504 | 0.344 | full |
| production | SurfaceProblem | test_heldout | 150 | 75 | 0.659 | 0.607 | 0.607 | 0.667 | 0.547 | 0.187 | 0.187 | 0.040 | 0.040 | full |
| gemma4-12b | all (pooled) | test | 1250 | 625 | 0.740 | 0.639 | 0.639 | 0.419 | 0.859 | 0.040 | 0.038 | 0.190 | 0.174 | full |
| gemma4-12b | all (pooled) | test_heldout | 738 | 373 | 0.698 | 0.596 | 0.599 | 0.348 | 0.845 | 0.019 | 0.019 | 0.204 | 0.180 | full |
| production | all (pooled) | test | 1250 | 625 | 0.912 | 0.841 | 0.841 | 0.888 | 0.794 | 0.486 | 0.000 | 0.707 | 0.608 | full |
| production | all (pooled) | test_heldout | 738 | 373 | 0.844 | 0.766 | 0.766 | 0.816 | 0.716 | 0.362 | 0.181 | 0.126 | 0.115 | full |

## split difference (test - heldout)

| model | label_type | auc_test | auc_heldout | diff | diff_lo | diff_hi | n_test | n_heldout | set |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | CurbRamp | 0.772 | 0.912 | -0.140 | -0.286 | 0.007 | 60 | 40 | subset |
| gemma4-12b | NoCurbRamp | 0.591 | 0.672 | -0.081 | -0.313 | 0.145 | 60 | 40 | subset |
| gemma4-12b | Obstacle | 0.499 | 0.522 | -0.024 | -0.252 | 0.227 | 60 | 40 | subset |
| gemma4-12b | SurfaceProblem | 0.754 | 0.627 | 0.127 | -0.072 | 0.351 | 60 | 40 | subset |
| gemma4-12b | Crosswalk | 0.946 | 0.930 | 0.016 | -0.069 | 0.117 | 60 | 40 | subset |
| gemma4-12b | all (pooled) | 0.730 | 0.714 | 0.016 | -0.076 | 0.108 | 300 | 200 | subset |
| production | CurbRamp | 0.924 | 0.930 | -0.006 | -0.124 | 0.103 | 60 | 40 | subset |
| production | NoCurbRamp | 0.964 | 0.955 | 0.009 | -0.057 | 0.085 | 60 | 40 | subset |
| production | Obstacle | 0.853 | 0.890 | -0.037 | -0.183 | 0.119 | 60 | 40 | subset |
| production | SurfaceProblem | 0.839 | 0.525 | 0.314 | 0.103 | 0.532 | 60 | 40 | subset |
| production | Crosswalk | 0.980 | 0.930 | 0.050 | -0.026 | 0.158 | 60 | 40 | subset |
| production | all (pooled) | 0.911 | 0.838 | 0.074 | 0.008 | 0.141 | 300 | 200 | subset |
| qwen3vl-2b | CurbRamp | 0.482 | 0.682 | -0.200 | -0.417 | 0.025 | 60 | 40 | subset |
| qwen3vl-2b | NoCurbRamp | 0.504 | 0.605 | -0.101 | -0.313 | 0.149 | 60 | 40 | subset |
| qwen3vl-2b | Obstacle | 0.491 | 0.460 | 0.031 | -0.203 | 0.275 | 60 | 40 | subset |
| qwen3vl-2b | SurfaceProblem | 0.630 | 0.585 | 0.045 | -0.170 | 0.263 | 60 | 40 | subset |
| qwen3vl-2b | Crosswalk | 0.938 | 0.910 | 0.028 | -0.069 | 0.148 | 60 | 40 | subset |
| qwen3vl-2b | all (pooled) | 0.605 | 0.611 | -0.005 | -0.096 | 0.095 | 300 | 200 | subset |
| qwen3vl-8b | CurbRamp | 0.698 | 0.853 | -0.155 | -0.323 | 0.025 | 60 | 40 | subset |
| qwen3vl-8b | NoCurbRamp | 0.423 | 0.525 | -0.102 | -0.337 | 0.130 | 60 | 40 | subset |
| qwen3vl-8b | Obstacle | 0.497 | 0.557 | -0.061 | -0.299 | 0.169 | 60 | 40 | subset |
| qwen3vl-8b | SurfaceProblem | 0.750 | 0.487 | 0.263 | 0.056 | 0.472 | 60 | 40 | subset |
| qwen3vl-8b | Crosswalk | 0.909 | 0.868 | 0.041 | -0.088 | 0.192 | 60 | 40 | subset |
| qwen3vl-8b | all (pooled) | 0.691 | 0.652 | 0.038 | -0.060 | 0.138 | 300 | 200 | subset |
| gemma4-12b | CurbRamp | 0.763 | 0.838 | -0.074 | -0.160 | 0.009 | 250 | 150 | full |
| gemma4-12b | NoCurbRamp | 0.616 | 0.673 | -0.057 | -0.164 | 0.048 | 250 | 150 | full |
| gemma4-12b | Obstacle | 0.500 | 0.518 | -0.018 | -0.129 | 0.095 | 250 | 150 | full |
| gemma4-12b | SurfaceProblem | 0.777 | 0.668 | 0.108 | 0.009 | 0.207 | 250 | 150 | full |
| gemma4-12b | Crosswalk | 0.967 | 0.892 | 0.074 | 0.021 | 0.137 | 250 | 138 | full |
| gemma4-12b | all (pooled) | 0.740 | 0.698 | 0.042 | -0.004 | 0.088 | 1250 | 738 | full |
| production | CurbRamp | 0.945 | 0.890 | 0.055 | -0.000 | 0.115 | 250 | 150 | full |
| production | NoCurbRamp | 0.899 | 0.936 | -0.038 | -0.088 | 0.013 | 250 | 150 | full |
| production | Obstacle | 0.821 | 0.838 | -0.017 | -0.096 | 0.075 | 250 | 150 | full |
| production | SurfaceProblem | 0.899 | 0.659 | 0.240 | 0.143 | 0.337 | 250 | 150 | full |
| production | Crosswalk | 0.980 | 0.919 | 0.060 | 0.011 | 0.116 | 250 | 138 | full |
| production | all (pooled) | 0.912 | 0.844 | 0.068 | 0.036 | 0.099 | 1250 | 738 | full |

## within-city AUC

| model | split | label_type | auc_within_city | n_cities | n | set |
|---|---|---|---|---|---|---|
| gemma4-12b | test | all (pooled) | 0.725 | 7 | 300 | subset |
| gemma4-12b | test | CurbRamp | 0.746 | 1 | 60 | subset |
| gemma4-12b | test | NoCurbRamp | 0.500 | 1 | 60 | subset |
| gemma4-12b | test | Obstacle | 0.561 | 3 | 60 | subset |
| gemma4-12b | test | SurfaceProblem | 0.757 | 2 | 60 | subset |
| gemma4-12b | test | Crosswalk | 1.000 | 2 | 60 | subset |
| gemma4-12b | test_heldout | all (pooled) | 0.702 | 3 | 200 | subset |
| gemma4-12b | test_heldout | CurbRamp | 0.875 | 2 | 40 | subset |
| gemma4-12b | test_heldout | NoCurbRamp | 0.595 | 1 | 40 | subset |
| gemma4-12b | test_heldout | Obstacle | 0.333 | 2 | 40 | subset |
| gemma4-12b | test_heldout | SurfaceProblem | 0.641 | 2 | 40 | subset |
| gemma4-12b | test_heldout | Crosswalk | 0.924 | 2 | 40 | subset |
| production | test | all (pooled) | 0.922 | 7 | 300 | subset |
| production | test | CurbRamp | 0.936 | 1 | 60 | subset |
| production | test | NoCurbRamp | 1.000 | 1 | 60 | subset |
| production | test | Obstacle | 0.872 | 3 | 60 | subset |
| production | test | SurfaceProblem | 0.757 | 2 | 60 | subset |
| production | test | Crosswalk | 0.963 | 2 | 60 | subset |
| production | test_heldout | all (pooled) | 0.863 | 3 | 200 | subset |
| production | test_heldout | CurbRamp | 0.903 | 2 | 40 | subset |
| production | test_heldout | NoCurbRamp | 0.968 | 1 | 40 | subset |
| production | test_heldout | Obstacle | 0.889 | 2 | 40 | subset |
| production | test_heldout | SurfaceProblem | 0.658 | 2 | 40 | subset |
| production | test_heldout | Crosswalk | 0.924 | 2 | 40 | subset |
| qwen3vl-2b | test | all (pooled) | 0.616 | 7 | 300 | subset |
| qwen3vl-2b | test | CurbRamp | 0.587 | 1 | 60 | subset |
| qwen3vl-2b | test | NoCurbRamp | 0.686 | 1 | 60 | subset |
| qwen3vl-2b | test | Obstacle | 0.506 | 3 | 60 | subset |
| qwen3vl-2b | test | SurfaceProblem | 0.600 | 2 | 60 | subset |
| qwen3vl-2b | test | Crosswalk | 0.982 | 2 | 60 | subset |
| qwen3vl-2b | test_heldout | all (pooled) | 0.599 | 3 | 200 | subset |
| qwen3vl-2b | test_heldout | CurbRamp | 0.667 | 2 | 40 | subset |
| qwen3vl-2b | test_heldout | NoCurbRamp | 0.429 | 1 | 40 | subset |
| qwen3vl-2b | test_heldout | Obstacle | 0.531 | 2 | 40 | subset |
| qwen3vl-2b | test_heldout | SurfaceProblem | 0.632 | 2 | 40 | subset |
| qwen3vl-2b | test_heldout | Crosswalk | 0.930 | 2 | 40 | subset |
| qwen3vl-8b | test | all (pooled) | 0.697 | 7 | 300 | subset |
| qwen3vl-8b | test | CurbRamp | 0.667 | 1 | 60 | subset |
| qwen3vl-8b | test | NoCurbRamp | 0.157 | 1 | 60 | subset |
| qwen3vl-8b | test | Obstacle | 0.604 | 3 | 60 | subset |
| qwen3vl-8b | test | SurfaceProblem | 0.771 | 2 | 60 | subset |
| qwen3vl-8b | test | Crosswalk | 1.000 | 2 | 60 | subset |
| qwen3vl-8b | test_heldout | all (pooled) | 0.664 | 3 | 200 | subset |
| qwen3vl-8b | test_heldout | CurbRamp | 0.826 | 2 | 40 | subset |
| qwen3vl-8b | test_heldout | NoCurbRamp | 0.360 | 1 | 40 | subset |
| qwen3vl-8b | test_heldout | Obstacle | 0.580 | 2 | 40 | subset |
| qwen3vl-8b | test_heldout | SurfaceProblem | 0.496 | 2 | 40 | subset |
| qwen3vl-8b | test_heldout | Crosswalk | 0.854 | 2 | 40 | subset |
| gemma4-12b | test | all (pooled) | 0.733 | 17 | 1250 | full |
| gemma4-12b | test | CurbRamp | 0.810 | 8 | 250 | full |
| gemma4-12b | test | NoCurbRamp | 0.588 | 6 | 250 | full |
| gemma4-12b | test | Obstacle | 0.553 | 4 | 250 | full |
| gemma4-12b | test | SurfaceProblem | 0.706 | 6 | 250 | full |
| gemma4-12b | test | Crosswalk | 0.981 | 7 | 250 | full |
| gemma4-12b | test_heldout | all (pooled) | 0.697 | 3 | 738 | full |
| gemma4-12b | test_heldout | CurbRamp | 0.792 | 3 | 150 | full |
| gemma4-12b | test_heldout | NoCurbRamp | 0.619 | 2 | 150 | full |
| gemma4-12b | test_heldout | Obstacle | 0.314 | 3 | 150 | full |
| gemma4-12b | test_heldout | SurfaceProblem | 0.656 | 3 | 150 | full |
| gemma4-12b | test_heldout | Crosswalk | 0.890 | 3 | 138 | full |
| production | test | all (pooled) | 0.917 | 17 | 1250 | full |
| production | test | CurbRamp | 0.950 | 8 | 250 | full |
| production | test | NoCurbRamp | 0.919 | 6 | 250 | full |
| production | test | Obstacle | 0.820 | 4 | 250 | full |
| production | test | SurfaceProblem | 0.845 | 6 | 250 | full |
| production | test | Crosswalk | 0.985 | 7 | 250 | full |
| production | test_heldout | all (pooled) | 0.844 | 3 | 738 | full |
| production | test_heldout | CurbRamp | 0.834 | 3 | 150 | full |
| production | test_heldout | NoCurbRamp | 0.933 | 2 | 150 | full |
| production | test_heldout | Obstacle | 0.718 | 3 | 150 | full |
| production | test_heldout | SurfaceProblem | 0.675 | 3 | 150 | full |
| production | test_heldout | Crosswalk | 0.942 | 3 | 138 | full |

## hard (every subset row is incorrect: balanced accuracy there is recall of incorrect)

| model | n | n_incorrect | auc | balanced_accuracy | accuracy | recall_correct | recall_incorrect | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 | set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | 6 | 6 |  | 0.667 | 0.667 |  | 0.667 |  |  | 1.000 | 1.000 | subset |
| production | 6 | 6 |  | 0.333 | 0.333 |  | 0.333 |  |  | 1.000 | 1.000 | subset |
| qwen3vl-2b | 6 | 6 |  | 0.500 | 0.500 |  | 0.500 |  |  | 1.000 | 1.000 | subset |
| qwen3vl-8b | 6 | 6 |  | 0.667 | 0.667 |  | 0.667 |  |  | 1.000 | 1.000 | subset |
| gemma4-12b | 20 | 10 | 0.460 | 0.400 | 0.400 | 0.000 | 0.800 | 0.000 | 0.000 | 0.000 | 0.000 | full |
| production | 20 | 10 | 0.220 | 0.200 | 0.200 | 0.100 | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | full |

## per city

| model | city | public | n | n_incorrect | auc | balanced_accuracy | accuracy | recall_correct | recall_incorrect | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 | mean_within_type_auc | n_types | set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | seattle-wa | True | 71 | 30 | 0.733 | 0.621 | 0.578 | 0.342 | 0.900 | 0.073 | 0.073 | 0.167 | 0.167 | 0.664 | 4 | subset |
| gemma4-12b | chicago-il | True | 52 | 33 | 0.655 | 0.657 | 0.692 | 0.526 | 0.788 | 0.053 | 0.053 | 0.212 | 0.212 | 0.694 | 2 | subset |
| gemma4-12b | other (8 private deployments) | False | 259 | 120 | 0.736 | 0.626 | 0.610 | 0.403 | 0.850 | 0.201 | 0.065 | 0.092 | 0.092 | 0.745 | 5 | subset |
| production | seattle-wa | True | 71 | 30 | 0.942 | 0.856 | 0.859 | 0.878 | 0.833 | 0.805 | 0.805 | 0.767 | 0.467 | 0.936 | 4 | subset |
| production | chicago-il | True | 52 | 33 | 0.911 | 0.822 | 0.788 | 0.947 | 0.697 | 0.053 | 0.053 | 0.818 | 0.727 | 0.907 | 2 | subset |
| production | other (8 private deployments) | False | 259 | 120 | 0.891 | 0.817 | 0.819 | 0.842 | 0.792 | 0.547 | 0.424 | 0.517 | 0.358 | 0.890 | 5 | subset |
| qwen3vl-2b | seattle-wa | True | 71 | 30 | 0.660 | 0.589 | 0.535 | 0.244 | 0.933 | 0.098 | 0.098 | 0.133 | 0.133 | 0.621 | 4 | subset |
| qwen3vl-2b | chicago-il | True | 52 | 33 | 0.463 | 0.480 | 0.538 | 0.263 | 0.697 | 0.158 | 0.158 | 0.000 | 0.000 | 0.725 | 2 | subset |
| qwen3vl-2b | other (8 private deployments) | False | 259 | 120 | 0.655 | 0.607 | 0.587 | 0.338 | 0.875 | 0.209 | 0.209 | 0.008 | 0.008 | 0.668 | 5 | subset |
| qwen3vl-8b | seattle-wa | True | 71 | 30 | 0.650 | 0.598 | 0.578 | 0.463 | 0.733 | 0.122 | 0.122 | 0.200 | 0.200 | 0.614 | 4 | subset |
| qwen3vl-8b | chicago-il | True | 52 | 33 | 0.710 | 0.713 | 0.692 | 0.789 | 0.636 | 0.053 | 0.053 | 0.212 | 0.212 | 0.694 | 2 | subset |
| qwen3vl-8b | other (8 private deployments) | False | 259 | 120 | 0.679 | 0.594 | 0.591 | 0.554 | 0.633 | 0.058 | 0.058 | 0.017 | 0.017 | 0.670 | 5 | subset |
| gemma4-12b | seattle-wa | True | 328 | 156 | 0.727 | 0.582 | 0.570 | 0.337 | 0.827 | 0.017 | 0.017 | 0.346 | 0.269 | 0.748 | 5 | full |
| gemma4-12b | chicago-il | True | 206 | 116 | 0.706 | 0.653 | 0.675 | 0.478 | 0.828 | 0.056 | 0.056 | 0.198 | 0.198 | 0.686 | 5 | full |
| gemma4-12b | tucson-az | True | 85 | 53 | 0.628 | 0.540 | 0.588 | 0.344 | 0.736 | 0.000 | 0.000 | 0.283 | 0.075 | 0.683 | 4 | full |
| gemma4-12b | mendota-il | True | 53 | 43 | 0.823 | 0.638 | 0.849 | 0.300 | 0.977 | 0.300 | 0.300 | 0.674 | 0.674 | 0.909 | 2 | full |
| gemma4-12b | amsterdam | True | 49 | 39 | 0.661 | 0.549 | 0.755 | 0.200 | 0.897 | 0.100 | 0.100 | 0.051 | 0.051 | 0.704 | 2 | full |
| gemma4-12b | spgg | True | 43 | 19 | 0.842 | 0.614 | 0.581 | 0.333 | 0.895 | 0.167 | 0.167 | 0.263 | 0.263 | 0.604 | 3 | full |
| gemma4-12b | pittsburgh-pa | True | 40 | 17 | 0.714 | 0.673 | 0.650 | 0.522 | 0.824 | 0.217 | 0.217 | 0.235 | 0.235 | 0.729 | 1 | full |
| gemma4-12b | teaneck-nj | True | 34 | 23 | 0.787 | 0.662 | 0.735 | 0.455 | 0.870 | 0.000 | 0.000 | 0.478 | 0.478 | 0.923 | 1 | full |
| gemma4-12b | other (10 private deployments) | False | 971 | 455 | 0.729 | 0.642 | 0.627 | 0.405 | 0.879 | 0.099 | 0.074 | 0.132 | 0.075 | 0.730 | 5 | full |
| production | seattle-wa | True | 328 | 156 | 0.920 | 0.851 | 0.854 | 0.907 | 0.795 | 0.762 | 0.000 | 0.744 | 0.635 | 0.927 | 5 | full |
| production | chicago-il | True | 206 | 116 | 0.943 | 0.869 | 0.859 | 0.944 | 0.793 | 0.767 | 0.011 | 0.871 | 0.733 | 0.900 | 5 | full |
| production | tucson-az | True | 85 | 53 | 0.739 | 0.664 | 0.635 | 0.781 | 0.547 | 0.156 | 0.156 | 0.226 | 0.132 | 0.799 | 4 | full |
| production | mendota-il | True | 53 | 43 | 0.998 | 0.950 | 0.981 | 0.900 | 1.000 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 2 | full |
| production | amsterdam | True | 49 | 39 | 0.939 | 0.899 | 0.898 | 0.900 | 0.897 | 0.500 | 0.500 | 0.923 | 0.923 | 1.000 | 2 | full |
| production | spgg | True | 43 | 19 | 0.833 | 0.764 | 0.767 | 0.792 | 0.737 | 0.500 | 0.000 | 0.158 | 0.158 | 0.396 | 3 | full |
| production | pittsburgh-pa | True | 40 | 17 | 0.747 | 0.700 | 0.725 | 0.870 | 0.529 | 0.217 | 0.217 | 0.471 | 0.471 | 0.562 | 1 | full |
| production | teaneck-nj | True | 34 | 23 | 0.961 | 0.913 | 0.882 | 1.000 | 0.826 | 0.091 | 0.091 | 0.957 | 0.957 | 1.000 | 1 | full |
| production | other (10 private deployments) | False | 971 | 455 | 0.877 | 0.795 | 0.797 | 0.824 | 0.767 | 0.562 | 0.444 | 0.283 | 0.180 | 0.869 | 5 | full |

## per city x type

| model | label_type | city | public | n | n_incorrect | auc | balanced_accuracy | accuracy | recall_correct | recall_incorrect | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 | set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | Crosswalk | other (8 private deployments) | False | 57 | 18 | 0.936 | 0.855 | 0.842 | 0.821 | 0.889 | 0.923 | 0.795 | 0.444 | 0.444 | subset |
| production | Crosswalk | other (8 private deployments) | False | 57 | 18 | 0.932 | 0.838 | 0.860 | 0.897 | 0.778 | 0.897 | 0.718 | 0.722 | 0.556 | subset |
| qwen3vl-2b | Crosswalk | other (8 private deployments) | False | 57 | 18 | 0.929 | 0.855 | 0.842 | 0.821 | 0.889 | 0.821 | 0.744 | 0.722 | 0.056 | subset |
| qwen3vl-8b | Crosswalk | other (8 private deployments) | False | 57 | 18 | 0.877 | 0.784 | 0.807 | 0.846 | 0.722 | 0.769 | 0.692 | 0.111 | 0.111 | subset |
| gemma4-12b | CurbRamp | other (8 private deployments) | False | 50 | 25 | 0.858 | 0.540 | 0.540 | 0.080 | 1.000 | 0.480 | 0.360 | 0.480 | 0.440 | subset |
| production | CurbRamp | other (8 private deployments) | False | 50 | 25 | 0.904 | 0.880 | 0.880 | 0.880 | 0.880 | 0.040 | 0.040 | 0.680 | 0.640 | subset |
| qwen3vl-2b | CurbRamp | other (8 private deployments) | False | 50 | 25 | 0.613 | 0.500 | 0.500 | 0.000 | 1.000 | 0.080 | 0.080 | 0.040 | 0.040 | subset |
| qwen3vl-8b | CurbRamp | other (8 private deployments) | False | 50 | 25 | 0.749 | 0.640 | 0.640 | 0.320 | 0.960 | 0.320 | 0.320 | 0.040 | 0.040 | subset |
| gemma4-12b | NoCurbRamp | seattle-wa | True | 23 | 6 | 0.500 | 0.529 | 0.304 | 0.059 | 1.000 | 0.235 | 0.235 | 0.000 | 0.000 | subset |
| gemma4-12b | NoCurbRamp | other (8 private deployments) | False | 46 | 22 | 0.689 | 0.519 | 0.500 | 0.083 | 0.955 | 0.042 | 0.042 | 0.091 | 0.091 | subset |
| production | NoCurbRamp | seattle-wa | True | 23 | 6 | 1.000 | 0.971 | 0.957 | 0.941 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | subset |
| production | NoCurbRamp | other (8 private deployments) | False | 46 | 22 | 0.951 | 0.911 | 0.913 | 0.958 | 0.864 | 0.750 | 0.625 | 0.864 | 0.864 | subset |
| qwen3vl-2b | NoCurbRamp | seattle-wa | True | 23 | 6 | 0.686 | 0.647 | 0.478 | 0.294 | 1.000 | 0.294 | 0.294 | 0.167 | 0.167 | subset |
| qwen3vl-2b | NoCurbRamp | other (8 private deployments) | False | 46 | 22 | 0.585 | 0.528 | 0.522 | 0.375 | 0.682 | 0.042 | 0.042 | 0.273 | 0.273 | subset |
| qwen3vl-8b | NoCurbRamp | seattle-wa | True | 23 | 6 | 0.157 | 0.500 | 0.261 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | subset |
| qwen3vl-8b | NoCurbRamp | other (8 private deployments) | False | 46 | 22 | 0.494 | 0.409 | 0.391 | 0.000 | 0.818 | 0.000 | 0.000 | 0.091 | 0.091 | subset |
| gemma4-12b | Obstacle | other (8 private deployments) | False | 52 | 26 | 0.553 | 0.500 | 0.500 | 0.385 | 0.615 | 0.077 | 0.077 | 0.038 | 0.038 | subset |
| production | Obstacle | other (8 private deployments) | False | 52 | 26 | 0.890 | 0.769 | 0.769 | 0.846 | 0.692 | 0.538 | 0.500 | 0.577 | 0.462 | subset |
| qwen3vl-2b | Obstacle | other (8 private deployments) | False | 52 | 26 | 0.573 | 0.481 | 0.481 | 0.000 | 0.962 | 0.000 | 0.000 | 0.269 | 0.269 | subset |
| qwen3vl-8b | Obstacle | other (8 private deployments) | False | 52 | 26 | 0.617 | 0.500 | 0.500 | 0.962 | 0.038 | 0.115 | 0.115 | 0.038 | 0.038 | subset |
| gemma4-12b | SurfaceProblem | other (8 private deployments) | False | 54 | 29 | 0.691 | 0.614 | 0.630 | 0.400 | 0.828 | 0.000 | 0.000 | 0.276 | 0.276 | subset |
| production | SurfaceProblem | other (8 private deployments) | False | 54 | 29 | 0.772 | 0.679 | 0.685 | 0.600 | 0.759 | 0.240 | 0.240 | 0.310 | 0.310 | subset |
| qwen3vl-2b | SurfaceProblem | other (8 private deployments) | False | 54 | 29 | 0.641 | 0.534 | 0.556 | 0.240 | 0.828 | 0.120 | 0.120 | 0.035 | 0.035 | subset |
| qwen3vl-8b | SurfaceProblem | other (8 private deployments) | False | 54 | 29 | 0.615 | 0.565 | 0.574 | 0.440 | 0.690 | 0.000 | 0.000 | 0.241 | 0.241 | subset |
| gemma4-12b | Crosswalk | chicago-il | True | 38 | 25 | 0.972 | 0.883 | 0.895 | 0.846 | 0.920 | 0.692 | 0.692 | 0.920 | 0.920 | full |
| gemma4-12b | Crosswalk | seattle-wa | True | 35 | 24 | 0.992 | 0.864 | 0.914 | 0.727 | 1.000 | 0.909 | 0.909 | 1.000 | 1.000 | full |
| gemma4-12b | Crosswalk | mendota-il | True | 31 | 29 | 1.000 | 0.750 | 0.968 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | full |
| gemma4-12b | Crosswalk | other (10 private deployments) | False | 209 | 73 | 0.929 | 0.861 | 0.852 | 0.831 | 0.890 | 0.890 | 0.699 | 0.630 | 0.356 | full |
| production | Crosswalk | chicago-il | True | 38 | 25 | 0.954 | 0.920 | 0.895 | 1.000 | 0.840 | 0.308 | 0.308 | 0.920 | 0.920 | full |
| production | Crosswalk | seattle-wa | True | 35 | 24 | 1.000 | 0.979 | 0.971 | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 | full |
| production | Crosswalk | mendota-il | True | 31 | 29 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | full |
| production | Crosswalk | other (10 private deployments) | False | 209 | 73 | 0.937 | 0.861 | 0.885 | 0.941 | 0.781 | 0.897 | 0.750 | 0.753 | 0.740 | full |
| gemma4-12b | CurbRamp | seattle-wa | True | 63 | 40 | 0.881 | 0.696 | 0.778 | 0.391 | 1.000 | 0.565 | 0.565 | 0.425 | 0.275 | full |
| gemma4-12b | CurbRamp | chicago-il | True | 41 | 13 | 0.832 | 0.643 | 0.512 | 0.286 | 1.000 | 0.607 | 0.286 | 0.385 | 0.385 | full |
| gemma4-12b | CurbRamp | other (10 private deployments) | False | 194 | 94 | 0.771 | 0.555 | 0.541 | 0.110 | 1.000 | 0.270 | 0.190 | 0.245 | 0.223 | full |
| production | CurbRamp | seattle-wa | True | 63 | 40 | 0.980 | 0.891 | 0.921 | 0.783 | 1.000 | 0.870 | 0.870 | 0.975 | 0.950 | full |
| production | CurbRamp | chicago-il | True | 41 | 13 | 0.920 | 0.885 | 0.927 | 1.000 | 0.769 | 1.000 | 0.964 | 0.923 | 0.846 | full |
| production | CurbRamp | other (10 private deployments) | False | 194 | 94 | 0.904 | 0.825 | 0.825 | 0.820 | 0.830 | 0.480 | 0.400 | 0.702 | 0.670 | full |
| gemma4-12b | NoCurbRamp | seattle-wa | True | 100 | 28 | 0.618 | 0.460 | 0.270 | 0.028 | 0.893 | 0.000 | 0.000 | 0.071 | 0.071 | full |
| gemma4-12b | NoCurbRamp | chicago-il | True | 52 | 45 | 0.378 | 0.489 | 0.846 | 0.000 | 0.978 | 0.000 | 0.000 | 0.133 | 0.133 | full |
| gemma4-12b | NoCurbRamp | other (10 private deployments) | False | 176 | 87 | 0.668 | 0.517 | 0.511 | 0.056 | 0.977 | 0.022 | 0.022 | 0.058 | 0.058 | full |
| production | NoCurbRamp | seattle-wa | True | 100 | 28 | 0.940 | 0.858 | 0.890 | 0.931 | 0.786 | 0.958 | 0.764 | 0.643 | 0.179 | full |
| production | NoCurbRamp | chicago-il | True | 52 | 45 | 0.813 | 0.670 | 0.846 | 0.429 | 0.911 | 0.429 | 0.429 | 0.867 | 0.867 | full |
| production | NoCurbRamp | other (10 private deployments) | False | 176 | 87 | 0.914 | 0.840 | 0.841 | 0.899 | 0.782 | 0.607 | 0.382 | 0.736 | 0.667 | full |
| gemma4-12b | Obstacle | seattle-wa | True | 76 | 36 | 0.574 | 0.590 | 0.592 | 0.625 | 0.556 | 0.050 | 0.050 | 0.139 | 0.139 | full |
| gemma4-12b | Obstacle | chicago-il | True | 42 | 23 | 0.430 | 0.406 | 0.405 | 0.421 | 0.391 | 0.053 | 0.053 | 0.000 | 0.000 | full |
| gemma4-12b | Obstacle | other (10 private deployments) | False | 196 | 91 | 0.554 | 0.542 | 0.536 | 0.448 | 0.637 | 0.000 | 0.000 | 0.000 | 0.000 | full |
| production | Obstacle | seattle-wa | True | 76 | 36 | 0.832 | 0.754 | 0.763 | 0.925 | 0.583 | 0.000 | 0.000 | 0.500 | 0.389 | full |
| production | Obstacle | chicago-il | True | 42 | 23 | 0.915 | 0.800 | 0.786 | 0.947 | 0.652 | 0.789 | 0.263 | 0.696 | 0.478 | full |
| production | Obstacle | other (10 private deployments) | False | 196 | 91 | 0.800 | 0.722 | 0.725 | 0.752 | 0.692 | 0.514 | 0.486 | 0.055 | 0.055 | full |
| gemma4-12b | SurfaceProblem | seattle-wa | True | 54 | 28 | 0.674 | 0.626 | 0.630 | 0.538 | 0.714 | 0.000 | 0.000 | 0.071 | 0.071 | full |
| gemma4-12b | SurfaceProblem | chicago-il | True | 33 | 10 | 0.817 | 0.698 | 0.697 | 0.696 | 0.700 | 0.609 | 0.130 | 0.400 | 0.400 | full |
| gemma4-12b | SurfaceProblem | tucson-az | True | 33 | 23 | 0.674 | 0.670 | 0.697 | 0.600 | 0.739 | 0.300 | 0.300 | 0.130 | 0.130 | full |
| gemma4-12b | SurfaceProblem | other (10 private deployments) | False | 196 | 110 | 0.727 | 0.637 | 0.668 | 0.384 | 0.891 | 0.023 | 0.023 | 0.273 | 0.255 | full |
| production | SurfaceProblem | seattle-wa | True | 54 | 28 | 0.883 | 0.764 | 0.759 | 0.885 | 0.643 | 0.654 | 0.462 | 0.179 | 0.179 | full |
| production | SurfaceProblem | chicago-il | True | 33 | 10 | 0.896 | 0.750 | 0.849 | 1.000 | 0.500 | 0.739 | 0.304 | 0.600 | 0.600 | full |
| production | SurfaceProblem | tucson-az | True | 33 | 23 | 0.535 | 0.487 | 0.364 | 0.800 | 0.174 | 0.100 | 0.100 | 0.087 | 0.087 | full |
| production | SurfaceProblem | other (10 private deployments) | False | 196 | 110 | 0.788 | 0.703 | 0.709 | 0.651 | 0.754 | 0.360 | 0.337 | 0.236 | 0.218 | full |

## bootstrap

| set | label_type | model_a | model_b | n | auc_a | auc_b | diff | a_lo | a_hi | b_lo | b_hi | diff_lo | diff_hi | city_a_lo | city_a_hi | city_b_lo | city_b_hi | city_diff_lo | city_diff_hi | city_n_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subset | CurbRamp | gemma4-12b | production | 100 | 0.826 | 0.929 | -0.103 | 0.739 | 0.900 | 0.870 | 0.977 | -0.191 | -0.024 | 0.751 | 0.891 | 0.872 | 0.969 | -0.170 | -0.041 | 22 |
| subset | NoCurbRamp | gemma4-12b | production | 100 | 0.620 | 0.959 | -0.339 | 0.510 | 0.728 | 0.922 | 0.988 | -0.449 | -0.232 | 0.526 | 0.756 | 0.893 | 0.987 | -0.444 | -0.173 | 18 |
| subset | Obstacle | gemma4-12b | production | 100 | 0.500 | 0.869 | -0.369 | 0.379 | 0.618 | 0.791 | 0.933 | -0.482 | -0.261 | 0.374 | 0.598 | 0.724 | 0.963 | -0.475 | -0.259 | 15 |
| subset | SurfaceProblem | gemma4-12b | production | 100 | 0.700 | 0.721 | -0.022 | 0.586 | 0.793 | 0.619 | 0.824 | -0.152 | 0.091 | 0.619 | 0.822 | 0.609 | 0.856 | -0.142 | 0.092 | 21 |
| subset | Crosswalk | gemma4-12b | production | 100 | 0.939 | 0.964 | -0.026 | 0.892 | 0.980 | 0.929 | 0.990 | -0.066 | 0.013 | 0.875 | 0.976 | 0.931 | 0.990 | -0.078 | 0.007 | 20 |
| subset | all (pooled) | gemma4-12b | production | 500 | 0.723 | 0.885 | -0.162 | 0.678 | 0.768 | 0.853 | 0.912 | -0.210 | -0.117 | 0.683 | 0.762 | 0.833 | 0.923 | -0.209 | -0.108 | 31 |
| full | CurbRamp | gemma4-12b | production | 400 | 0.790 | 0.930 | -0.140 | 0.747 | 0.833 | 0.903 | 0.955 | -0.183 | -0.098 | 0.714 | 0.832 | 0.885 | 0.964 | -0.229 | -0.091 | 29 |
| full | NoCurbRamp | gemma4-12b | production | 400 | 0.634 | 0.908 | -0.274 | 0.579 | 0.687 | 0.878 | 0.935 | -0.331 | -0.216 | 0.527 | 0.721 | 0.798 | 0.933 | -0.341 | -0.114 | 24 |
| full | Obstacle | gemma4-12b | production | 400 | 0.508 | 0.828 | -0.319 | 0.454 | 0.564 | 0.785 | 0.868 | -0.378 | -0.256 | 0.426 | 0.559 | 0.735 | 0.901 | -0.431 | -0.250 | 28 |
| full | SurfaceProblem | gemma4-12b | production | 400 | 0.735 | 0.822 | -0.087 | 0.689 | 0.784 | 0.779 | 0.863 | -0.137 | -0.037 | 0.658 | 0.812 | 0.736 | 0.901 | -0.164 | -0.019 | 28 |
| full | Crosswalk | gemma4-12b | production | 388 | 0.944 | 0.965 | -0.020 | 0.921 | 0.964 | 0.947 | 0.979 | -0.043 | -0.001 | 0.893 | 0.974 | 0.933 | 0.986 | -0.064 | 0.001 | 26 |
| full | all (pooled) | gemma4-12b | production | 1988 | 0.724 | 0.891 | -0.166 | 0.702 | 0.747 | 0.876 | 0.905 | -0.189 | -0.145 | 0.684 | 0.755 | 0.840 | 0.923 | -0.199 | -0.125 | 41 |

## latency (timing runs)

| model | n | p10_s | median_s | p90_s | load_1min_before | load_1min_after | date | reason_p10_s | reason_median_s | reason_p90_s | prompt_tokens_median | size_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b | 50 | 2.149 | 2.328 | 2.422 | 2.630 | 7.740 | 2026-10-21T14:12:27 | 3.379 | 3.716 | 3.953 | 353.000 | 11.020 |
| production | 50 | 0.068 | 0.071 | 0.074 | 2.760 | 3.830 | 2026-10-21T14:08:17 |  |  |  |  | 1.218 |
| qwen3vl-2b | 50 | 0.685 | 0.717 | 0.727 | 2.870 | 3.120 | 2026-10-21T14:09:33 |  |  |  | 493.000 | 1.798 |
| qwen3vl-8b | 50 | 2.065 | 2.193 | 2.297 | 2.950 | 2.630 | 2026-10-21T14:10:26 |  |  |  | 493.000 | 5.777 |

## ablation

| model | label_type | variant | n | n_incorrect | auc | balanced_accuracy | accuracy | recall_correct | recall_incorrect | r_correct_p92 | r_correct_p95 | r_incorrect_p92 | r_incorrect_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3vl-8b | CurbRamp | definition | 100 | 50 | 0.761 | 0.660 | 0.660 | 0.340 | 0.980 | 0.400 | 0.400 | 0.080 | 0.080 |
| qwen3vl-8b | CurbRamp | no-definition | 100 | 50 | 0.556 | 0.550 | 0.550 | 0.820 | 0.280 | 0.000 | 0.000 | 0.020 | 0.020 |

## greedy check

| model | label_type | n | agree | unparsable |
|---|---|---|---|---|
| gemma4-12b | all | 10 | 8 | 0 |
| gemma4-12b | Crosswalk | 2 | 0 | 0 |
| gemma4-12b | CurbRamp | 2 | 2 | 0 |
| gemma4-12b | NoCurbRamp | 2 | 2 | 0 |
| gemma4-12b | Obstacle | 2 | 2 | 0 |
| gemma4-12b | SurfaceProblem | 2 | 2 | 0 |
| qwen3vl-2b | all | 10 | 10 | 0 |
| qwen3vl-2b | Crosswalk | 2 | 2 | 0 |
| qwen3vl-2b | CurbRamp | 2 | 2 | 0 |
| qwen3vl-2b | NoCurbRamp | 2 | 2 | 0 |
| qwen3vl-2b | Obstacle | 2 | 2 | 0 |
| qwen3vl-2b | SurfaceProblem | 2 | 2 | 0 |
| qwen3vl-8b | all | 10 | 10 | 0 |
| qwen3vl-8b | Crosswalk | 2 | 2 | 0 |
| qwen3vl-8b | CurbRamp | 2 | 2 | 0 |
| qwen3vl-8b | NoCurbRamp | 2 | 2 | 0 |
| qwen3vl-8b | Obstacle | 2 | 2 | 0 |
| qwen3vl-8b | SurfaceProblem | 2 | 2 | 0 |

