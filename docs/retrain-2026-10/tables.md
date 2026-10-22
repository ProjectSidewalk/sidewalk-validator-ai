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
| spgg | 111 | 104 | 7 | 96 | 0.499 — | 0.865 |
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
| private city A (outside the US) | 125 | 4 | 121 | 84 | 0.800 — | 0.704 |
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
| oradell-nj | 82 | 79 | 3 | 59 | 0.869 — | 0.915 |
| columbus-oh | 79 | 64 | 15 | 75 | 0.727 [0.556, 0.865] | 0.747 |
| newberg-or | 71 | 50 | 21 | 64 | 0.855 [0.728, 0.956] | 0.803 |
| pittsburgh-pa | 70 | 61 | 9 | 66 | 0.863 [0.735, 0.964] | 0.886 |
| spgg | 67 | 53 | 14 | 65 | 0.814 [0.669, 0.955] | 0.821 |
| st-louis-mo | 65 | 64 | 1 | 63 | 0.859 — | 0.938 |
| cdmx | 51 | 50 | 1 | 44 | 0.360 — | 0.922 |

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

| label_type | n | n_correct | n_incorrect | AUC [95% CI] | accuracy | balanced_accuracy |
|---|---:|---:|---:|---|---:|---:|
| CurbRamp | 23 | 22 | 1 | 0.091 — | 0.478 | 0.250 |
| NoCurbRamp | 90 | 48 | 42 | 0.644 [0.528, 0.749] | 0.633 | 0.644 |
| Obstacle | 11 | 9 | 2 | 0.556 — | 0.455 | 0.667 |
| SurfaceProblem | 40 | 11 | 29 | 0.323 [0.163, 0.503] | 0.475 | 0.384 |
| Crosswalk | 5 | 4 | 1 | 0.750 — | 0.600 | 0.750 |

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

## calibration (file prior)

| label_type | prior_incorrect_share | agree_clears_at | disagree_map_max_shipped | disagree_map_max_uncapped | disagree_clears_at_uncapped | agree_test_coverage | agree_test_precision | agree_test_n | agree_test_ece | agree_test_precision_city_own | agree_test_coverage_city_own | agree_heldout_coverage | agree_heldout_precision | agree_heldout_n | agree_heldout_ece | agree_heldout_precision_city_own | agree_heldout_coverage_city_own | disagree_uncapped_test_coverage | disagree_uncapped_test_precision | disagree_uncapped_test_n | disagree_uncapped_heldout_coverage | disagree_uncapped_heldout_precision | disagree_uncapped_heldout_n | agree_fit_test_heldout_coverage | agree_fit_test_heldout_precision | agree_fit_test_heldout_n | disagree_fit_test_heldout_coverage | disagree_fit_test_heldout_precision | disagree_fit_test_heldout_n |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | 0.1177 | 0.99992 | 0.8023 | 0.8023 | never | 0.8825 | 0.994 | 692 | 0.03448 | 0.9939 | 0.8825 | 0.7512 | 0.9912 | 198 | 0.07171 | 0.9923 | 0.7521 | 0 | — | 0 | 0 | — | 0 | 0.6874 | 0.9915 | 181 | 0 | — | 0 |
| NoCurbRamp | 0.3737 | 0.9999 | 0.9199 | 0.9403 | 0.999994 | 0.4402 | 0.9727 | 627 | 0.05532 | 0.973 | 0.4404 | 0.3724 | 0.9331 | 158 | 0.0607 | 0.9144 | 0.3594 | 0.2063 | 0.9783 | 249 | 0.05567 | 1 | 17 | 0.5242 | 0.9234 | 224 | 0.01965 | 1 | 6 |
| Obstacle | 0.3578 | 1.0 | 0.8923 | 0.8923 | never | 0.4751 | 0.9587 | 631 | 0.04667 | 0.9627 | 0.4798 | 0.4887 | 0.9698 | 325 | 0.03648 | 0.9263 | 0.3995 | 0 | — | 0 | 0 | — | 0 | 0.4887 | 0.9698 | 325 | 0 | — | 0 |
| SurfaceProblem | 0.3198 | 0.999997 | 0.903 | 0.903 | never | 0.6756 | 0.9569 | 819 | 0.04402 | 0.9581 | 0.6766 | 0.5278 | 0.8897 | 178 | 0.07786 | 0.8373 | 0.5059 | 0 | — | 0 | 0 | — | 0 | 0.4719 | 0.8903 | 159 | 0 | — | 0 |
| Crosswalk | 0.06114 | 0.9925 | 0.9199 | 0.9768 | 0.999989 | 0.9915 | 0.9953 | 255 | 0.03887 | 0.9954 | 0.9915 | 0.8639 | 0.9956 | 56 | 0.09318 | 0.9939 | 0.8628 | 0.422 | 1 | 98 | 0.319 | 0.6425 | 32 | 0.7967 | 0.9964 | 51 | 0.7455 | 0.5411 | 64 |

## alternative prior

| label_type | eval_split | prior_incorrect_share | agree_coverage | agree_precision | agree_n | agree_clears_at | disagree_coverage | disagree_precision | disagree_n | disagree_clears_at |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| CurbRamp | test | 0.06 | 0.886 | 0.9971 | 692 | 0.99992 | 0 | — | 0 | never |
| CurbRamp | test_heldout | 0.06 | 0.7552 | 0.9958 | 198 | 0.99992 | 0 | — | 0 | never |
| NoCurbRamp | test | 0.2 | 0.7108 | 0.9854 | 984 | 0.9993 | 0 | — | 0 | never |
| NoCurbRamp | test_heldout | 0.2 | 0.635 | 0.9569 | 260 | 0.9993 | 0 | — | 0 | never |
| Obstacle | test | 0.22 | 0.5 | 0.9787 | 631 | 1.0 | 0 | — | 0 | never |
| Obstacle | test_heldout | 0.22 | 0.5286 | 0.9845 | 325 | 1.0 | 0 | — | 0 | never |
| SurfaceProblem | test | 0.15 | 0.6986 | 0.9834 | 819 | 0.999997 | 0 | — | 0 | never |
| SurfaceProblem | test_heldout | 0.15 | 0.5582 | 0.9555 | 178 | 0.999997 | 0 | — | 0 | never |
| Crosswalk | test | 0.04 | 0.9916 | 0.997 | 255 | 0.9925 | 0.377 | 1 | 98 | 0.999989 |
| Crosswalk | test_heldout | 0.04 | 0.8649 | 0.9972 | 56 | 0.9925 | 0.2901 | 0.5349 | 32 | 0.999989 |

## gate per deployment

| label_type | prior | split | city | city_incorrect_share | n_agree | gated | n_wrong | coverage_pooled | coverage_city_own | precision_pooled | precision_city_own | precision_raw |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | file | test | chicago-il | 0.405 | 136 | 120 | 7 | 0.912 | 0.912 | 0.990 | 0.990 | 0.942 |
| CurbRamp | file | test | other (private deployments) | 0.409 | 195 | 168 | 4 | 0.898 | 0.898 | 0.996 | 0.996 | 0.976 |
| CurbRamp | file | test | seattle-wa | 0.616 | 149 | 135 | 4 | 0.942 | 0.942 | 0.995 | 0.995 | 0.970 |
| CurbRamp | file | test_heldout | private city A (outside the US) | 0.592 | 71 | 17 | 2 | 0.265 | 0.266 | 0.976 | 0.979 | 0.882 |
| CurbRamp | file | test_heldout | private city B (US) | 0.149 | 176 | 157 | 7 | 0.901 | 0.902 | 0.991 | 0.992 | 0.955 |
| CurbRamp | file | test_heldout | tucson-az | 0.500 | 28 | 24 | 0 | 0.857 | 0.857 | 1.000 | 1.000 | 1.000 |
| CurbRamp | alt | test | chicago-il | 0.405 | 136 | 120 | 7 | 0.916 | 0.916 | 0.995 | 0.995 | 0.942 |
| CurbRamp | alt | test | other (private deployments) | 0.409 | 195 | 168 | 4 | 0.902 | 0.902 | 0.998 | 0.998 | 0.976 |
| CurbRamp | alt | test | seattle-wa | 0.616 | 149 | 135 | 4 | 0.946 | 0.946 | 0.998 | 0.998 | 0.970 |
| CurbRamp | alt | test_heldout | private city A (outside the US) | 0.592 | 71 | 17 | 2 | 0.269 | 0.269 | 0.988 | 0.990 | 0.882 |
| CurbRamp | alt | test_heldout | private city B (US) | 0.149 | 176 | 157 | 7 | 0.903 | 0.903 | 0.996 | 0.996 | 0.955 |
| CurbRamp | alt | test_heldout | tucson-az | 0.500 | 28 | 24 | 0 | 0.857 | 0.857 | 1.000 | 1.000 | 1.000 |
| NoCurbRamp | file | test | cdmx | 0.226 | 54 | 32 | 3 | 0.601 | 0.602 | 0.929 | 0.930 | 0.906 |
| NoCurbRamp | file | test | chicago-il | 0.850 | 66 | 14 | 1 | 0.227 | 0.228 | 0.946 | 0.947 | 0.929 |
| NoCurbRamp | file | test | other (private deployments) | 0.414 | 193 | 123 | 5 | 0.655 | 0.655 | 0.970 | 0.970 | 0.959 |
| NoCurbRamp | file | test | seattle-wa | 0.232 | 761 | 294 | 8 | 0.392 | 0.392 | 0.980 | 0.980 | 0.973 |
| NoCurbRamp | file | test | spgg | 0.063 | 99 | 50 | 4 | 0.501 | 0.501 | 0.940 | 0.940 | 0.920 |
| NoCurbRamp | file | test_heldout | private city A (outside the US) | 0.968 | 39 | 3 | 3 | 0.073 | 0.075 | 0.000 | 0.000 | 0.000 |
| NoCurbRamp | file | test_heldout | private city B (US) | 0.409 | 407 | 150 | 15 | 0.394 | 0.383 | 0.942 | 0.925 | 0.900 |
| NoCurbRamp | file | test_heldout | tucson-az | 0.630 | 17 | 5 | 0 | 0.319 | 0.309 | 1.000 | 1.000 | 1.000 |
| NoCurbRamp | alt | test | cdmx | 0.226 | 54 | 36 | 4 | 0.686 | 0.686 | 0.963 | 0.963 | 0.889 |
| NoCurbRamp | alt | test | chicago-il | 0.850 | 66 | 23 | 3 | 0.407 | 0.407 | 0.956 | 0.956 | 0.870 |
| NoCurbRamp | alt | test | other (private deployments) | 0.414 | 193 | 143 | 9 | 0.785 | 0.785 | 0.980 | 0.980 | 0.937 |
| NoCurbRamp | alt | test | seattle-wa | 0.232 | 761 | 556 | 20 | 0.754 | 0.754 | 0.989 | 0.989 | 0.964 |
| NoCurbRamp | alt | test | spgg | 0.063 | 99 | 69 | 4 | 0.693 | 0.693 | 0.981 | 0.982 | 0.942 |
| NoCurbRamp | alt | test_heldout | private city A (outside the US) | 0.968 | 39 | 7 | 7 | 0.143 | 0.153 | 0.000 | 0.000 | 0.000 |
| NoCurbRamp | alt | test_heldout | private city B (US) | 0.409 | 407 | 247 | 35 | 0.662 | 0.656 | 0.963 | 0.952 | 0.858 |
| NoCurbRamp | alt | test_heldout | tucson-az | 0.630 | 17 | 6 | 0 | 0.408 | 0.402 | 1.000 | 1.000 | 1.000 |
| Obstacle | file | test | chicago-il | 0.451 | 259 | 125 | 3 | 0.503 | 0.510 | 0.982 | 0.984 | 0.976 |
| Obstacle | file | test | other (private deployments) | 0.320 | 437 | 204 | 6 | 0.478 | 0.482 | 0.978 | 0.980 | 0.971 |
| Obstacle | file | test | seattle-wa | 0.460 | 387 | 189 | 16 | 0.504 | 0.509 | 0.935 | 0.941 | 0.915 |
| Obstacle | file | test_heldout | private city A (outside the US) | 0.916 | 350 | 21 | 12 | 0.097 | 0.068 | 0.738 | 0.524 | 0.429 |
| Obstacle | file | test_heldout | private city B (US) | 0.312 | 460 | 269 | 11 | 0.631 | 0.604 | 0.989 | 0.972 | 0.959 |
| Obstacle | file | test_heldout | tucson-az | 0.623 | 109 | 35 | 11 | 0.365 | 0.336 | 0.891 | 0.762 | 0.686 |
| Obstacle | alt | test | chicago-il | 0.451 | 259 | 125 | 3 | 0.540 | 0.544 | 0.991 | 0.992 | 0.976 |
| Obstacle | alt | test | other (private deployments) | 0.320 | 437 | 204 | 6 | 0.498 | 0.500 | 0.989 | 0.990 | 0.971 |
| Obstacle | alt | test | seattle-wa | 0.460 | 387 | 189 | 16 | 0.531 | 0.534 | 0.966 | 0.969 | 0.915 |
| Obstacle | alt | test_heldout | private city A (outside the US) | 0.916 | 350 | 21 | 12 | 0.123 | 0.087 | 0.847 | 0.685 | 0.429 |
| Obstacle | alt | test_heldout | private city B (US) | 0.312 | 460 | 269 | 11 | 0.640 | 0.625 | 0.994 | 0.986 | 0.959 |
| Obstacle | alt | test_heldout | tucson-az | 0.623 | 109 | 35 | 11 | 0.378 | 0.358 | 0.942 | 0.864 | 0.686 |
| SurfaceProblem | file | test | chicago-il | 0.395 | 187 | 133 | 10 | 0.729 | 0.730 | 0.954 | 0.955 | 0.925 |
| SurfaceProblem | file | test | columbus-oh | 0.190 | 60 | 32 | 2 | 0.550 | 0.550 | 0.962 | 0.963 | 0.938 |
| SurfaceProblem | file | test | newberg-or | 0.296 | 52 | 36 | 2 | 0.722 | 0.723 | 0.966 | 0.967 | 0.944 |
| SurfaceProblem | file | test | oradell-nj | 0.037 | 76 | 62 | 1 | 0.819 | 0.819 | 0.990 | 0.991 | 0.984 |
| SurfaceProblem | file | test | other (private deployments) | 0.623 | 235 | 137 | 13 | 0.608 | 0.609 | 0.942 | 0.943 | 0.905 |
| SurfaceProblem | file | test | pittsburgh-pa | 0.129 | 57 | 45 | 2 | 0.786 | 0.786 | 0.973 | 0.974 | 0.956 |
| SurfaceProblem | file | test | seattle-wa | 0.557 | 247 | 128 | 19 | 0.544 | 0.546 | 0.907 | 0.909 | 0.852 |
| SurfaceProblem | file | test | spgg | 0.209 | 59 | 42 | 3 | 0.737 | 0.738 | 0.957 | 0.958 | 0.929 |
| SurfaceProblem | file | test | st-louis-mo | 0.015 | 62 | 58 | 1 | 0.935 | 0.935 | 0.990 | 0.990 | 0.983 |
| SurfaceProblem | file | test_heldout | private city A (outside the US) | 0.601 | 64 | 5 | 2 | 0.082 | 0.080 | 0.804 | 0.723 | 0.600 |
| SurfaceProblem | file | test_heldout | private city B (US) | 0.294 | 190 | 119 | 9 | 0.670 | 0.655 | 0.971 | 0.955 | 0.924 |
| SurfaceProblem | file | test_heldout | tucson-az | 0.808 | 123 | 54 | 34 | 0.474 | 0.458 | 0.616 | 0.506 | 0.370 |
| SurfaceProblem | alt | test | chicago-il | 0.395 | 187 | 133 | 10 | 0.747 | 0.748 | 0.982 | 0.983 | 0.925 |
| SurfaceProblem | alt | test | columbus-oh | 0.190 | 60 | 32 | 2 | 0.566 | 0.566 | 0.985 | 0.986 | 0.938 |
| SurfaceProblem | alt | test | newberg-or | 0.296 | 52 | 36 | 2 | 0.752 | 0.753 | 0.987 | 0.987 | 0.944 |
| SurfaceProblem | alt | test | oradell-nj | 0.037 | 76 | 62 | 1 | 0.822 | 0.822 | 0.996 | 0.996 | 0.984 |
| SurfaceProblem | alt | test | other (private deployments) | 0.623 | 235 | 137 | 13 | 0.634 | 0.635 | 0.977 | 0.978 | 0.905 |
| SurfaceProblem | alt | test | pittsburgh-pa | 0.129 | 57 | 45 | 2 | 0.784 | 0.784 | 0.990 | 0.990 | 0.956 |
| SurfaceProblem | alt | test | seattle-wa | 0.557 | 247 | 128 | 19 | 0.574 | 0.575 | 0.963 | 0.964 | 0.852 |
| SurfaceProblem | alt | test | spgg | 0.209 | 59 | 42 | 3 | 0.763 | 0.763 | 0.983 | 0.984 | 0.929 |
| SurfaceProblem | alt | test | st-louis-mo | 0.015 | 62 | 58 | 1 | 0.935 | 0.935 | 0.996 | 0.996 | 0.983 |
| SurfaceProblem | alt | test_heldout | private city A (outside the US) | 0.601 | 64 | 5 | 2 | 0.084 | 0.083 | 0.916 | 0.874 | 0.600 |
| SurfaceProblem | alt | test_heldout | private city B (US) | 0.294 | 190 | 119 | 9 | 0.689 | 0.682 | 0.989 | 0.983 | 0.924 |
| SurfaceProblem | alt | test_heldout | tucson-az | 0.808 | 123 | 54 | 34 | 0.505 | 0.492 | 0.810 | 0.732 | 0.370 |
| Crosswalk | file | test | other (private deployments) | 0.105 | 160 | 160 | 8 | 1.000 | 1.000 | 0.996 | 0.996 | 0.950 |
| Crosswalk | file | test_heldout | private city A (outside the US) | 0.356 | 32 | 26 | 3 | 0.821 | 0.821 | 0.992 | 0.990 | 0.885 |
| Crosswalk | file | test_heldout | private city B (US) | 0.610 | 33 | 27 | 1 | 0.891 | 0.890 | 0.998 | 0.997 | 0.963 |
| Crosswalk | file | test_heldout | tucson-az | 0.625 | 3 | 3 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Crosswalk | alt | test | other (private deployments) | 0.105 | 160 | 160 | 8 | 1.000 | 1.000 | 0.997 | 0.997 | 0.950 |
| Crosswalk | alt | test_heldout | private city A (outside the US) | 0.356 | 32 | 26 | 3 | 0.821 | 0.821 | 0.995 | 0.993 | 0.885 |
| Crosswalk | alt | test_heldout | private city B (US) | 0.610 | 33 | 27 | 1 | 0.893 | 0.892 | 0.999 | 0.998 | 0.963 |
| Crosswalk | alt | test_heldout | tucson-az | 0.625 | 3 | 3 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## saturation

| label_type | agree_clears_at | map_value_at_0.999999 | n_test_agree | share_agree_at_1 | share_agree_ge_0.999999 | share_disagree_at_1 |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | 0.99992 | 0.943 | 814 | 0.506 | 0.692 | 0.525 |
| NoCurbRamp | 0.9999 | 0.961 | 1454 | 0.006 | 0.048 | 0.011 |
| Obstacle | 1.0 | 0.788 | 1370 | 0.461 | 0.601 | 0.344 |
| SurfaceProblem | 0.999997 | 0.923 | 1254 | 0.293 | 0.585 | 0.359 |
| Crosswalk | 0.9925 | 0.966 | 258 | 0.000 | 0.132 | 0.021 |
