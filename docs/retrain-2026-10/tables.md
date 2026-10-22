## headline

| label_type | split | n | n_correct | n_incorrect | n_panos | n_cities | AUC [95% CI] | within_city_auc | within_city_n_cities | accuracy | balanced_accuracy | test − held-out [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| CurbRamp | test | 1479 | 812 | 667 | 1319 | 36 | 0.968 [0.958, 0.976] | 0.973 | 19 | 0.915 | 0.914 | +0.060 [0.034, 0.091] |
| CurbRamp | test_heldout | 546 | 318 | 228 | 469 | 3 | 0.907 [0.877, 0.933] | 0.838 | 3 | 0.826 | 0.834 | +0.060 [0.034, 0.091] |
| NoCurbRamp | test | 2613 | 1443 | 1170 | 1974 | 34 | 0.934 [0.924, 0.943] | 0.903 | 18 | 0.879 | 0.877 | +0.052 [0.023, 0.082] |
| NoCurbRamp | test_heldout | 743 | 359 | 384 | 542 | 3 | 0.881 [0.854, 0.908] | 0.861 | 3 | 0.774 | 0.778 | +0.052 [0.023, 0.082] |
| Obstacle | test | 2266 | 1305 | 961 | 1963 | 34 | 0.883 [0.869, 0.896] | 0.870 | 18 | 0.814 | 0.805 | +0.015 [-0.010, 0.040] |
| Obstacle | test_heldout | 1812 | 587 | 1225 | 1261 | 3 | 0.868 [0.847, 0.888] | 0.788 | 3 | 0.724 | 0.759 | +0.015 [-0.010, 0.040] |
| SurfaceProblem | test | 2255 | 1256 | 999 | 1937 | 37 | 0.896 [0.881, 0.910] | 0.858 | 21 | 0.832 | 0.830 | +0.185 [0.147, 0.225] |
| SurfaceProblem | test_heldout | 847 | 371 | 476 | 757 | 3 | 0.710 [0.675, 0.745] | 0.682 | 3 | 0.658 | 0.653 | +0.185 [0.147, 0.225] |
| Crosswalk | test | 445 | 247 | 198 | 374 | 27 | 0.980 [0.961, 0.993] | 0.983 | 8 | 0.957 | 0.954 | +0.039 [-0.004, 0.085] |
| Crosswalk | test_heldout | 138 | 65 | 73 | 130 | 3 | 0.941 [0.897, 0.980] | 0.946 | 3 | 0.906 | 0.907 | +0.039 [-0.004, 0.085] |

## per city

### CurbRamp / test

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| seattle-wa | 367 | 141 | 226 | 346 | 0.991 [0.980, 0.998] | 0.962 |
| other (private deployments) | 340 | 201 | 139 | 271 | 0.966 [0.947, 0.982] | 0.900 |
| chicago-il | 220 | 131 | 89 | 207 | 0.960 [0.934, 0.982] | 0.905 |
| spgg | 105 | 22 | 83 | 100 | 0.906 [0.812, 0.979] | 0.886 |
| amsterdam | 68 | 35 | 33 | 48 | 0.943 [0.887, 0.986] | 0.897 |

### CurbRamp / test_heldout

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| private city A (outside the US) | 284 | 116 | 168 | 225 | 0.821 [0.761, 0.872] | 0.729 |
| private city B (US) | 202 | 172 | 30 | 190 | 0.881 [0.791, 0.951] | 0.921 |
| tucson-az | 60 | 30 | 30 | 54 | 0.958 [0.888, 1.000] | 0.967 |

### NoCurbRamp / test

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| seattle-wa | 964 | 740 | 224 | 734 | 0.914 [0.888, 0.939] | 0.897 |
| chicago-il | 501 | 75 | 426 | 381 | 0.859 [0.797, 0.911] | 0.898 |
| other (private deployments) | 302 | 177 | 125 | 250 | 0.942 [0.914, 0.967] | 0.874 |
| mendota-il | 173 | 17 | 156 | 82 | 0.880 [0.783, 0.960] | 0.913 |
| spgg | 111 | 104 | 7 | 96 | 0.499 [0.157, 0.819] | 0.865 |
| amsterdam | 74 | 11 | 63 | 57 | 0.791 [0.661, 0.908] | 0.824 |
| columbus-oh | 69 | 44 | 25 | 53 | 0.845 [0.730, 0.934] | 0.739 |
| teaneck-nj | 68 | 38 | 30 | 44 | 0.968 [0.909, 1.000] | 0.941 |
| cdmx | 62 | 48 | 14 | 49 | 0.749 [0.551, 0.923] | 0.839 |
| newberg-or | 60 | 35 | 25 | 43 | 0.968 [0.919, 0.997] | 0.900 |
| pittsburgh-pa | 53 | 36 | 17 | 48 | 0.719 [0.562, 0.853] | 0.698 |

### NoCurbRamp / test_heldout

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| private city B (US) | 572 | 338 | 234 | 423 | 0.860 [0.824, 0.893] | 0.781 |
| private city A (outside the US) | 125 | 4 | 121 | 84 | 0.800 [0.681, 0.903] | 0.704 |
| tucson-az | 46 | 17 | 29 | 35 | 0.955 — | 0.870 |

### Obstacle / test

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| seattle-wa | 682 | 368 | 314 | 613 | 0.855 [0.825, 0.883] | 0.793 |
| other (private deployments) | 631 | 429 | 202 | 517 | 0.895 [0.867, 0.920] | 0.835 |
| chicago-il | 437 | 240 | 197 | 377 | 0.905 [0.874, 0.933] | 0.815 |
| amsterdam | 68 | 21 | 47 | 52 | 0.862 [0.741, 0.953] | 0.779 |
| st-louis-mo | 53 | 41 | 12 | 51 | 0.926 [0.826, 0.993] | 0.925 |
| spgg | 50 | 32 | 18 | 48 | 0.832 [0.694, 0.939] | 0.760 |

### Obstacle / test_heldout

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| private city A (outside the US) | 993 | 83 | 910 | 539 | 0.666 [0.596, 0.727] | 0.655 |
| private city B (US) | 628 | 432 | 196 | 550 | 0.896 [0.866, 0.924] | 0.844 |
| tucson-az | 191 | 72 | 119 | 172 | 0.792 [0.724, 0.853] | 0.691 |

### SurfaceProblem / test

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| other (private deployments) | 584 | 220 | 364 | 406 | 0.926 [0.898, 0.949] | 0.872 |
| seattle-wa | 558 | 247 | 311 | 520 | 0.830 [0.794, 0.865] | 0.771 |
| chicago-il | 314 | 190 | 124 | 291 | 0.882 [0.840, 0.922] | 0.831 |
| amsterdam | 91 | 12 | 79 | 78 | 0.854 [0.730, 0.942] | 0.901 |
| oradell-nj | 82 | 79 | 3 | 59 | 0.869 [0.688, 1.000] | 0.915 |
| columbus-oh | 79 | 64 | 15 | 75 | 0.727 [0.556, 0.865] | 0.747 |
| newberg-or | 71 | 50 | 21 | 64 | 0.855 [0.728, 0.956] | 0.803 |
| pittsburgh-pa | 70 | 61 | 9 | 66 | 0.863 [0.735, 0.964] | 0.886 |
| spgg | 67 | 53 | 14 | 65 | 0.814 [0.669, 0.955] | 0.821 |
| st-louis-mo | 65 | 64 | 1 | 63 | 0.859 [0.769, 0.937] | 0.938 |
| cdmx | 51 | 50 | 1 | 44 | 0.360 [0.239, 0.481] | 0.922 |

### SurfaceProblem / test_heldout

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| private city A (outside the US) | 356 | 142 | 214 | 295 | 0.612 [0.552, 0.669] | 0.618 |
| private city B (US) | 262 | 185 | 77 | 257 | 0.798 [0.736, 0.859] | 0.767 |
| tucson-az | 229 | 44 | 185 | 205 | 0.743 [0.668, 0.811] | 0.594 |

### Crosswalk / test

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| other (private deployments) | 171 | 153 | 18 | 150 | 0.882 [0.766, 0.980] | 0.947 |
| seattle-wa | 59 | 25 | 34 | 52 | 1.000 [1.000, 1.000] | 0.983 |
| chicago-il | 57 | 20 | 37 | 47 | 0.946 [0.881, 0.991] | 0.860 |

### Crosswalk / test_heldout

| city | n | n_correct | n_incorrect | n_panos | AUC [95% CI] | accuracy |
|---|---:|---:|---:|---:|---|---:|
| private city B (US) | 77 | 30 | 47 | 75 | 0.972 [0.929, 1.000] | 0.935 |
| private city A (outside the US) | 45 | 29 | 16 | 42 | 0.888 — | 0.889 |
| tucson-az | 16 | 6 | 10 | 13 | 0.767 — | 0.812 |

## class share

### CurbRamp

| split | city | n | share_of_split | incorrect_share |
|---|---|---:|---:|---:|
| test_heldout | private city A (outside the US) | 284 | 0.520 | 0.592 |
| test_heldout | private city B (US) | 202 | 0.370 | 0.149 |
| test_heldout | tucson-az | 60 | 0.110 | 0.500 |
| train | seattle-wa | 1722 | 0.249 | 0.632 |
| train | other (private deployments) | 1559 | 0.225 | 0.412 |
| train | chicago-il | 1095 | 0.158 | 0.356 |
| train | spgg | 545 | 0.079 | 0.817 |
| train | teaneck-nj | 258 | 0.037 | 0.120 |
| train | amsterdam | 238 | 0.034 | 0.618 |
| train | columbus-oh | 181 | 0.026 | 0.221 |
| train | newberg-or | 167 | 0.024 | 0.383 |
| train | vancouver-wa | 150 | 0.022 | 0.140 |
| train | pittsburgh-pa | 148 | 0.021 | 0.311 |
| train | mendota-il | 137 | 0.020 | 0.409 |
| train | oradell-nj | 133 | 0.019 | 0.263 |
| train | cdmx | 128 | 0.018 | 0.430 |
| train | burnaby | 103 | 0.015 | 0.097 |
| train | st-louis-mo | 80 | 0.012 | 0.013 |
| train | walla-walla-wa | 67 | 0.010 | 0.090 |
| train | knox-oh | 53 | 0.008 | 0.377 |

### NoCurbRamp

| split | city | n | share_of_split | incorrect_share |
|---|---|---:|---:|---:|
| test_heldout | private city B (US) | 572 | 0.770 | 0.409 |
| test_heldout | private city A (outside the US) | 125 | 0.168 | 0.968 |
| test_heldout | tucson-az | 46 | 0.062 | 0.630 |
| train | seattle-wa | 4228 | 0.347 | 0.241 |
| train | chicago-il | 2248 | 0.184 | 0.868 |
| train | other (private deployments) | 1764 | 0.145 | 0.502 |
| train | mendota-il | 500 | 0.041 | 0.810 |
| train | newberg-or | 464 | 0.038 | 0.239 |
| train | spgg | 420 | 0.034 | 0.100 |
| train | columbus-oh | 344 | 0.028 | 0.413 |
| train | teaneck-nj | 338 | 0.028 | 0.325 |
| train | amsterdam | 302 | 0.025 | 0.798 |
| train | pittsburgh-pa | 296 | 0.024 | 0.402 |
| train | cdmx | 286 | 0.023 | 0.210 |
| train | st-louis-mo | 200 | 0.016 | 0.200 |
| train | burnaby | 192 | 0.016 | 0.271 |
| train | walla-walla-wa | 159 | 0.013 | 0.264 |
| train | oradell-nj | 146 | 0.012 | 0.473 |
| train | paterson-nj | 55 | 0.005 | 0.891 |

### Obstacle

| split | city | n | share_of_split | incorrect_share |
|---|---|---:|---:|---:|
| test_heldout | private city A (outside the US) | 993 | 0.548 | 0.916 |
| test_heldout | private city B (US) | 628 | 0.347 | 0.312 |
| test_heldout | tucson-az | 191 | 0.105 | 0.623 |
| train | other (private deployments) | 3398 | 0.322 | 0.375 |
| train | seattle-wa | 2990 | 0.283 | 0.441 |
| train | chicago-il | 1839 | 0.174 | 0.433 |
| train | amsterdam | 319 | 0.030 | 0.571 |
| train | st-louis-mo | 252 | 0.024 | 0.183 |
| train | columbus-oh | 201 | 0.019 | 0.592 |
| train | walla-walla-wa | 199 | 0.019 | 0.648 |
| train | newberg-or | 188 | 0.018 | 0.399 |
| train | spgg | 188 | 0.018 | 0.340 |
| train | pittsburgh-pa | 172 | 0.016 | 0.628 |
| train | cdmx | 110 | 0.010 | 0.345 |
| train | mendota-il | 99 | 0.009 | 0.899 |
| train | oradell-nj | 89 | 0.008 | 0.382 |
| train | waltham-ma | 69 | 0.007 | 0.043 |
| train | teaneck-nj | 64 | 0.006 | 0.375 |
| train | paterson-nj | 61 | 0.006 | 0.705 |
| train | gainesville-fl | 55 | 0.005 | 0.018 |

### SurfaceProblem

| split | city | n | share_of_split | incorrect_share |
|---|---|---:|---:|---:|
| test_heldout | private city A (outside the US) | 356 | 0.420 | 0.601 |
| test_heldout | private city B (US) | 262 | 0.309 | 0.294 |
| test_heldout | tucson-az | 229 | 0.270 | 0.808 |
| train | seattle-wa | 2691 | 0.256 | 0.583 |
| train | other (private deployments) | 2542 | 0.242 | 0.609 |
| train | chicago-il | 1361 | 0.130 | 0.451 |
| train | oradell-nj | 543 | 0.052 | 0.072 |
| train | amsterdam | 438 | 0.042 | 0.849 |
| train | pittsburgh-pa | 397 | 0.038 | 0.171 |
| train | columbus-oh | 364 | 0.035 | 0.176 |
| train | spgg | 361 | 0.034 | 0.150 |
| train | newberg-or | 286 | 0.027 | 0.290 |
| train | st-louis-mo | 257 | 0.024 | 0.035 |
| train | cdmx | 216 | 0.021 | 0.125 |
| train | teaneck-nj | 215 | 0.020 | 0.181 |
| train | mendota-il | 152 | 0.014 | 0.125 |
| train | walla-walla-wa | 134 | 0.013 | 0.149 |
| train | burnaby | 124 | 0.012 | 0.266 |
| train | knox-oh | 66 | 0.006 | 0.152 |
| train | maywood-nj | 58 | 0.006 | 0.121 |

### Crosswalk

| split | city | n | share_of_split | incorrect_share |
|---|---|---:|---:|---:|
| test_heldout | private city B (US) | 77 | 0.558 | 0.610 |
| test_heldout | private city A (outside the US) | 45 | 0.326 | 0.356 |
| test_heldout | tucson-az | 16 | 0.116 | 0.625 |
| train | other (private deployments) | 852 | 0.408 | 0.100 |
| train | seattle-wa | 274 | 0.131 | 0.690 |
| train | chicago-il | 222 | 0.106 | 0.599 |
| train | mendota-il | 171 | 0.082 | 0.901 |
| train | teaneck-nj | 142 | 0.068 | 0.613 |
| train | amsterdam | 114 | 0.055 | 0.737 |

## recall at precision

| label_type | split | class | precision | recall | threshold |
|---|---|---|---:|---:|---:|
| CurbRamp | test | correct | 0.920 | 0.926 | 0.457 |
| CurbRamp | test | correct | 0.950 | 0.876 | 0.998 |
| CurbRamp | test | incorrect | 0.920 | 0.900 | 0.816 |
| CurbRamp | test | incorrect | 0.950 | 0.811 | 1.000 |
| CurbRamp | test_heldout | correct | 0.920 | 0.726 | 0.968 |
| CurbRamp | test_heldout | correct | 0.950 | 0.689 | 0.997 |
| CurbRamp | test_heldout | incorrect | 0.920 | 0.000 | — |
| CurbRamp | test_heldout | incorrect | 0.950 | 0.000 | — |
| NoCurbRamp | test | correct | 0.920 | 0.836 | 0.958 |
| NoCurbRamp | test | correct | 0.950 | 0.713 | 0.998 |
| NoCurbRamp | test | incorrect | 0.920 | 0.656 | 0.996 |
| NoCurbRamp | test | incorrect | 0.950 | 0.426 | 1.000 |
| NoCurbRamp | test_heldout | correct | 0.920 | 0.103 | 1.000 |
| NoCurbRamp | test_heldout | correct | 0.950 | 0.000 | — |
| NoCurbRamp | test_heldout | incorrect | 0.920 | 0.562 | 0.909 |
| NoCurbRamp | test_heldout | incorrect | 0.950 | 0.458 | 0.983 |
| Obstacle | test | correct | 0.920 | 0.605 | 1.000 |
| Obstacle | test | correct | 0.950 | 0.000 | — |
| Obstacle | test | incorrect | 0.920 | 0.000 | — |
| Obstacle | test | incorrect | 0.950 | 0.000 | — |
| Obstacle | test_heldout | correct | 0.920 | 0.000 | — |
| Obstacle | test_heldout | correct | 0.950 | 0.000 | — |
| Obstacle | test_heldout | incorrect | 0.920 | 0.529 | 0.977 |
| Obstacle | test_heldout | incorrect | 0.950 | 0.265 | 1.000 |
| SurfaceProblem | test | correct | 0.920 | 0.697 | 1.000 |
| SurfaceProblem | test | correct | 0.950 | 0.000 | — |
| SurfaceProblem | test | incorrect | 0.920 | 0.402 | 1.000 |
| SurfaceProblem | test | incorrect | 0.950 | 0.000 | — |
| SurfaceProblem | test_heldout | correct | 0.920 | 0.000 | — |
| SurfaceProblem | test_heldout | correct | 0.950 | 0.000 | — |
| SurfaceProblem | test_heldout | incorrect | 0.920 | 0.000 | — |
| SurfaceProblem | test_heldout | incorrect | 0.950 | 0.000 | — |
| Crosswalk | test | correct | 0.920 | 0.992 | 0.125 |
| Crosswalk | test | correct | 0.950 | 0.955 | 0.999 |
| Crosswalk | test | incorrect | 0.920 | 0.944 | 0.001 |
| Crosswalk | test | incorrect | 0.950 | 0.934 | 0.007 |
| Crosswalk | test_heldout | correct | 0.920 | 0.862 | 0.977 |
| Crosswalk | test_heldout | correct | 0.950 | 0.338 | 1.000 |
| Crosswalk | test_heldout | incorrect | 0.920 | 0.890 | 0.976 |
| Crosswalk | test_heldout | incorrect | 0.950 | 0.863 | 0.995 |

## hard

| label_type | n | n_correct | n_incorrect | auc | accuracy | balanced_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CurbRamp | 23 | 22 | 1 | 0.091 | 0.478 | 0.250 |
| NoCurbRamp | 90 | 48 | 42 | 0.644 | 0.633 | 0.644 |
| Obstacle | 11 | 9 | 2 | 0.556 | 0.455 | 0.667 |
| SurfaceProblem | 40 | 11 | 29 | 0.323 | 0.475 | 0.384 |
| Crosswalk | 5 | 4 | 1 | 0.750 | 0.600 | 0.750 |

## production comparison

| label_type | split | n | n_panos | n_deployments | auc_round3 | auc_production | Δ [label CI] | Δ [pano CI] | within_city_round3 | within_city_production | within_city_n_cities |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| Crosswalk | test | 250 | 221 | 23 | 0.980 | 0.979 | +0.000 [-0.013, 0.014] | +0.000 [-0.013, 0.015] | 0.985 | 0.985 | 7 |
| Crosswalk | test_heldout | 138 | 130 | 3 | 0.941 | 0.919 | +0.021 [-0.024, 0.067] | +0.021 [-0.020, 0.065] | 0.946 | 0.942 | 3 |
| CurbRamp | test | 250 | 247 | 26 | 0.948 | 0.945 | +0.003 [-0.025, 0.031] | +0.003 [-0.023, 0.030] | 0.968 | 0.950 | 8 |
| CurbRamp | test_heldout | 150 | 142 | 3 | 0.919 | 0.890 | +0.029 [-0.003, 0.064] | +0.029 [-0.003, 0.062] | 0.857 | 0.834 | 3 |
| NoCurbRamp | test | 250 | 245 | 21 | 0.903 | 0.899 | +0.005 [-0.027, 0.037] | +0.005 [-0.028, 0.039] | 0.897 | 0.919 | 6 |
| NoCurbRamp | test_heldout | 150 | 142 | 3 | 0.936 | 0.936 | -0.000 [-0.039, 0.037] | -0.000 [-0.040, 0.037] | 0.917 | 0.933 | 2 |
| Obstacle | test | 250 | 245 | 25 | 0.895 | 0.821 | +0.074 [0.024, 0.124] | +0.074 [0.028, 0.126] | 0.883 | 0.820 | 4 |
| Obstacle | test_heldout | 150 | 139 | 3 | 0.856 | 0.838 | +0.018 [-0.038, 0.077] | +0.018 [-0.039, 0.073] | 0.705 | 0.718 | 3 |
| SurfaceProblem | test | 250 | 242 | 25 | 0.913 | 0.899 | +0.014 [-0.028, 0.057] | +0.014 [-0.027, 0.055] | 0.889 | 0.845 | 6 |
| SurfaceProblem | test_heldout | 150 | 146 | 3 | 0.639 | 0.659 | -0.020 [-0.081, 0.049] | -0.020 [-0.083, 0.044] | 0.601 | 0.675 | 3 |
| all (pooled) | test | 1250 | 1195 | 38 | 0.926 | 0.912 | +0.014 [-0.002, 0.029] | +0.014 [-0.002, 0.030] | 0.921 | 0.917 | 17 |
| all (pooled) | test_heldout | 738 | 668 | 3 | 0.847 | 0.844 | +0.003 [-0.019, 0.025] | +0.003 [-0.020, 0.027] | 0.844 | 0.844 | 3 |
