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

| label_type | prior_incorrect_share | own_mix_correct_weight | agree_clears_at | disagree_map_max_shipped | disagree_map_max_uncapped | disagree_clears_at_uncapped | agree_test_coverage | agree_test_precision | agree_test_n | agree_test_ece | agree_test_precision_city_own | agree_test_coverage_city_own | agree_heldout_coverage | agree_heldout_precision | agree_heldout_n | agree_heldout_ece | agree_heldout_precision_city_own | agree_heldout_coverage_city_own | disagree_uncapped_test_coverage | disagree_uncapped_test_precision | disagree_uncapped_test_n | disagree_uncapped_heldout_coverage | disagree_uncapped_heldout_precision | disagree_uncapped_heldout_n | agree_fit_test_heldout_coverage | agree_fit_test_heldout_precision | agree_fit_test_heldout_n | disagree_fit_test_heldout_coverage | disagree_fit_test_heldout_precision | disagree_fit_test_heldout_n |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | 0.1177 | 7.789 | 0.99992 | 0.8023 | 0.8023 | never | 0.8825 | 0.994 | 692 | 0.03448 | 0.9952 | 0.8839 | 0.7512 | 0.9912 | 198 | 0.07171 | 0.9939 | 0.7536 | 0 | — | 0 | 0 | — | 0 | 0.6874 | 0.9915 | 181 | 0 | — | 0 |
| NoCurbRamp | 0.3737 | 1.852 | 0.9999 | 0.9199 | 0.9403 | 0.999994 | 0.4402 | 0.9727 | 627 | 0.05532 | 0.9799 | 0.4472 | 0.3724 | 0.9331 | 158 | 0.0607 | 0.9351 | 0.3739 | 0.2063 | 0.9783 | 249 | 0.05567 | 1 | 17 | 0.5242 | 0.9234 | 224 | 0.01965 | 1 | 6 |
| Obstacle | 0.3578 | 1.573 | 1.0 | 0.8923 | 0.8923 | never | 0.4751 | 0.9587 | 631 | 0.04667 | 0.9651 | 0.4827 | 0.4887 | 0.9698 | 325 | 0.03648 | 0.9309 | 0.4073 | 0 | — | 0 | 0 | — | 0 | 0.4887 | 0.9698 | 325 | 0 | — | 0 |
| SurfaceProblem | 0.3198 | 2.559 | 0.999997 | 0.903 | 0.903 | never | 0.6756 | 0.9569 | 819 | 0.04402 | 0.9711 | 0.6878 | 0.5278 | 0.8897 | 178 | 0.07786 | 0.8832 | 0.525 | 0 | — | 0 | 0 | — | 0 | 0.4719 | 0.8903 | 159 | 0 | — | 0 |
| Crosswalk | 0.06114 | 12.7 | 0.9925 | 0.9199 | 0.9768 | 0.999989 | 0.9915 | 0.9953 | 255 | 0.03887 | 0.9954 | 0.9915 | 0.8639 | 0.9956 | 56 | 0.09318 | 0.994 | 0.8629 | 0.422 | 1 | 98 | 0.319 | 0.6425 | 32 | 0.7967 | 0.9964 | 51 | 0.7455 | 0.5411 | 64 |

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

| label_type | prior | split | city | city_incorrect_share | n_agree | gated | n_wrong | coverage_pooled | coverage_city_own | precision_pooled | precision_city_own | precision_raw | wilson_lower_own |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CurbRamp | file | test | chicago-il | 0.405 | 136 | 120 | 7 | 0.912 | 0.914 | 0.990 | 0.992 | 0.942 | 0.955 |
| CurbRamp | file | test | other (private deployments) | 0.409 | 195 | 168 | 4 | 0.898 | 0.900 | 0.996 | 0.997 | 0.976 | 0.972 |
| CurbRamp | file | test | seattle-wa | 0.616 | 149 | 135 | 4 | 0.942 | 0.943 | 0.995 | 0.996 | 0.970 | 0.965 |
| CurbRamp | file | test_heldout | private city A (outside the US) | 0.592 | 71 | 17 | 2 | 0.265 | 0.267 | 0.976 | 0.983 | 0.882 | 0.789 |
| CurbRamp | file | test_heldout | private city B (US) | 0.149 | 176 | 157 | 7 | 0.901 | 0.902 | 0.991 | 0.994 | 0.955 | 0.965 |
| CurbRamp | file | test_heldout | tucson-az | 0.500 | 28 | 24 | 0 | 0.857 | 0.857 | 1.000 | 1.000 | 1.000 | 0.862 |
| CurbRamp | alt | test | chicago-il | 0.405 | 136 | 120 | 7 | 0.916 | 0.914 | 0.995 | 0.992 | 0.942 | 0.955 |
| CurbRamp | alt | test | other (private deployments) | 0.409 | 195 | 168 | 4 | 0.902 | 0.900 | 0.998 | 0.997 | 0.976 | 0.972 |
| CurbRamp | alt | test | seattle-wa | 0.616 | 149 | 135 | 4 | 0.946 | 0.943 | 0.998 | 0.996 | 0.970 | 0.965 |
| CurbRamp | alt | test_heldout | private city A (outside the US) | 0.592 | 71 | 17 | 2 | 0.269 | 0.267 | 0.988 | 0.983 | 0.882 | 0.789 |
| CurbRamp | alt | test_heldout | private city B (US) | 0.149 | 176 | 157 | 7 | 0.903 | 0.902 | 0.996 | 0.994 | 0.955 | 0.965 |
| CurbRamp | alt | test_heldout | tucson-az | 0.500 | 28 | 24 | 0 | 0.857 | 0.857 | 1.000 | 1.000 | 1.000 | 0.862 |
| NoCurbRamp | file | test | cdmx | 0.226 | 54 | 32 | 3 | 0.601 | 0.609 | 0.929 | 0.947 | 0.906 | 0.812 |
| NoCurbRamp | file | test | chicago-il | 0.850 | 66 | 14 | 1 | 0.227 | 0.240 | 0.946 | 0.960 | 0.929 | 0.727 |
| NoCurbRamp | file | test | other (private deployments) | 0.414 | 193 | 123 | 5 | 0.655 | 0.668 | 0.970 | 0.978 | 0.959 | 0.934 |
| NoCurbRamp | file | test | seattle-wa | 0.232 | 761 | 294 | 8 | 0.392 | 0.396 | 0.980 | 0.985 | 0.973 | 0.964 |
| NoCurbRamp | file | test | spgg | 0.063 | 99 | 50 | 4 | 0.501 | 0.498 | 0.940 | 0.955 | 0.920 | 0.859 |
| NoCurbRamp | file | test_heldout | private city A (outside the US) | 0.968 | 39 | 3 | 3 | 0.073 | 0.072 | 0.000 | 0.000 | 0.000 | 0.000 |
| NoCurbRamp | file | test_heldout | private city B (US) | 0.409 | 407 | 150 | 15 | 0.394 | 0.395 | 0.942 | 0.943 | 0.900 | 0.894 |
| NoCurbRamp | file | test_heldout | tucson-az | 0.630 | 17 | 5 | 0 | 0.319 | 0.320 | 1.000 | 1.000 | 1.000 | 0.566 |
| NoCurbRamp | alt | test | cdmx | 0.226 | 54 | 36 | 4 | 0.686 | 0.679 | 0.963 | 0.937 | 0.889 | 0.808 |
| NoCurbRamp | alt | test | chicago-il | 0.850 | 66 | 23 | 3 | 0.407 | 0.384 | 0.956 | 0.925 | 0.870 | 0.748 |
| NoCurbRamp | alt | test | other (private deployments) | 0.414 | 193 | 143 | 9 | 0.785 | 0.769 | 0.980 | 0.965 | 0.937 | 0.921 |
| NoCurbRamp | alt | test | seattle-wa | 0.232 | 761 | 556 | 20 | 0.754 | 0.746 | 0.989 | 0.980 | 0.964 | 0.965 |
| NoCurbRamp | alt | test | spgg | 0.063 | 99 | 69 | 4 | 0.693 | 0.695 | 0.981 | 0.968 | 0.942 | 0.896 |
| NoCurbRamp | alt | test_heldout | private city A (outside the US) | 0.968 | 39 | 7 | 7 | 0.143 | 0.168 | 0.000 | 0.000 | 0.000 | 0.000 |
| NoCurbRamp | alt | test_heldout | private city B (US) | 0.409 | 407 | 247 | 35 | 0.662 | 0.637 | 0.963 | 0.918 | 0.858 | 0.877 |
| NoCurbRamp | alt | test_heldout | tucson-az | 0.630 | 17 | 6 | 0 | 0.408 | 0.384 | 1.000 | 1.000 | 1.000 | 0.610 |
| Obstacle | file | test | chicago-il | 0.451 | 259 | 125 | 3 | 0.503 | 0.515 | 0.982 | 0.985 | 0.976 | 0.944 |
| Obstacle | file | test | other (private deployments) | 0.320 | 437 | 204 | 6 | 0.478 | 0.484 | 0.978 | 0.981 | 0.971 | 0.952 |
| Obstacle | file | test | seattle-wa | 0.460 | 387 | 189 | 16 | 0.504 | 0.512 | 0.935 | 0.944 | 0.915 | 0.902 |
| Obstacle | file | test_heldout | private city A (outside the US) | 0.916 | 350 | 21 | 12 | 0.097 | 0.070 | 0.738 | 0.541 | 0.429 | 0.339 |
| Obstacle | file | test_heldout | private city B (US) | 0.312 | 460 | 269 | 11 | 0.631 | 0.606 | 0.989 | 0.974 | 0.959 | 0.947 |
| Obstacle | file | test_heldout | tucson-az | 0.623 | 109 | 35 | 11 | 0.365 | 0.339 | 0.891 | 0.774 | 0.686 | 0.613 |
| Obstacle | alt | test | chicago-il | 0.451 | 259 | 125 | 3 | 0.540 | 0.515 | 0.991 | 0.985 | 0.976 | 0.944 |
| Obstacle | alt | test | other (private deployments) | 0.320 | 437 | 204 | 6 | 0.498 | 0.484 | 0.989 | 0.981 | 0.971 | 0.952 |
| Obstacle | alt | test | seattle-wa | 0.460 | 387 | 189 | 16 | 0.531 | 0.512 | 0.966 | 0.944 | 0.915 | 0.902 |
| Obstacle | alt | test_heldout | private city A (outside the US) | 0.916 | 350 | 21 | 12 | 0.123 | 0.070 | 0.847 | 0.541 | 0.429 | 0.339 |
| Obstacle | alt | test_heldout | private city B (US) | 0.312 | 460 | 269 | 11 | 0.640 | 0.606 | 0.994 | 0.974 | 0.959 | 0.947 |
| Obstacle | alt | test_heldout | tucson-az | 0.623 | 109 | 35 | 11 | 0.378 | 0.339 | 0.942 | 0.774 | 0.686 | 0.613 |
| SurfaceProblem | file | test | chicago-il | 0.395 | 187 | 133 | 10 | 0.729 | 0.739 | 0.954 | 0.969 | 0.925 | 0.924 |
| SurfaceProblem | file | test | columbus-oh | 0.190 | 60 | 32 | 2 | 0.550 | 0.558 | 0.962 | 0.975 | 0.938 | 0.851 |
| SurfaceProblem | file | test | newberg-or | 0.296 | 52 | 36 | 2 | 0.722 | 0.738 | 0.966 | 0.978 | 0.944 | 0.866 |
| SurfaceProblem | file | test | oradell-nj | 0.037 | 76 | 62 | 1 | 0.819 | 0.821 | 0.990 | 0.994 | 0.984 | 0.930 |
| SurfaceProblem | file | test | other (private deployments) | 0.623 | 235 | 137 | 13 | 0.608 | 0.622 | 0.942 | 0.961 | 0.905 | 0.914 |
| SurfaceProblem | file | test | pittsburgh-pa | 0.129 | 57 | 45 | 2 | 0.786 | 0.785 | 0.973 | 0.982 | 0.956 | 0.891 |
| SurfaceProblem | file | test | seattle-wa | 0.557 | 247 | 128 | 19 | 0.544 | 0.560 | 0.907 | 0.936 | 0.852 | 0.880 |
| SurfaceProblem | file | test | spgg | 0.209 | 59 | 42 | 3 | 0.737 | 0.751 | 0.957 | 0.971 | 0.929 | 0.869 |
| SurfaceProblem | file | test | st-louis-mo | 0.015 | 62 | 58 | 1 | 0.935 | 0.935 | 0.990 | 0.993 | 0.983 | 0.926 |
| SurfaceProblem | file | test_heldout | private city A (outside the US) | 0.601 | 64 | 5 | 2 | 0.082 | 0.082 | 0.804 | 0.793 | 0.600 | 0.370 |
| SurfaceProblem | file | test_heldout | private city B (US) | 0.294 | 190 | 119 | 9 | 0.670 | 0.668 | 0.971 | 0.969 | 0.924 | 0.920 |
| SurfaceProblem | file | test_heldout | tucson-az | 0.808 | 123 | 54 | 34 | 0.474 | 0.471 | 0.616 | 0.601 | 0.370 | 0.468 |
| SurfaceProblem | alt | test | chicago-il | 0.395 | 187 | 133 | 10 | 0.747 | 0.739 | 0.982 | 0.969 | 0.925 | 0.924 |
| SurfaceProblem | alt | test | columbus-oh | 0.190 | 60 | 32 | 2 | 0.566 | 0.558 | 0.985 | 0.975 | 0.938 | 0.851 |
| SurfaceProblem | alt | test | newberg-or | 0.296 | 52 | 36 | 2 | 0.752 | 0.738 | 0.987 | 0.978 | 0.944 | 0.866 |
| SurfaceProblem | alt | test | oradell-nj | 0.037 | 76 | 62 | 1 | 0.822 | 0.821 | 0.996 | 0.994 | 0.984 | 0.930 |
| SurfaceProblem | alt | test | other (private deployments) | 0.623 | 235 | 137 | 13 | 0.634 | 0.622 | 0.977 | 0.961 | 0.905 | 0.914 |
| SurfaceProblem | alt | test | pittsburgh-pa | 0.129 | 57 | 45 | 2 | 0.784 | 0.785 | 0.990 | 0.982 | 0.956 | 0.891 |
| SurfaceProblem | alt | test | seattle-wa | 0.557 | 247 | 128 | 19 | 0.574 | 0.560 | 0.963 | 0.936 | 0.852 | 0.880 |
| SurfaceProblem | alt | test | spgg | 0.209 | 59 | 42 | 3 | 0.763 | 0.751 | 0.983 | 0.971 | 0.929 | 0.869 |
| SurfaceProblem | alt | test | st-louis-mo | 0.015 | 62 | 58 | 1 | 0.935 | 0.935 | 0.996 | 0.993 | 0.983 | 0.926 |
| SurfaceProblem | alt | test_heldout | private city A (outside the US) | 0.601 | 64 | 5 | 2 | 0.084 | 0.082 | 0.916 | 0.793 | 0.600 | 0.370 |
| SurfaceProblem | alt | test_heldout | private city B (US) | 0.294 | 190 | 119 | 9 | 0.689 | 0.668 | 0.989 | 0.969 | 0.924 | 0.920 |
| SurfaceProblem | alt | test_heldout | tucson-az | 0.808 | 123 | 54 | 34 | 0.505 | 0.471 | 0.810 | 0.601 | 0.370 | 0.468 |
| Crosswalk | file | test | other (private deployments) | 0.105 | 160 | 160 | 8 | 1.000 | 1.000 | 0.996 | 0.996 | 0.950 | 0.969 |
| Crosswalk | file | test_heldout | private city A (outside the US) | 0.356 | 32 | 26 | 3 | 0.821 | 0.821 | 0.992 | 0.990 | 0.885 | 0.854 |
| Crosswalk | file | test_heldout | private city B (US) | 0.610 | 33 | 27 | 1 | 0.891 | 0.890 | 0.998 | 0.997 | 0.963 | 0.870 |
| Crosswalk | file | test_heldout | tucson-az | 0.625 | 3 | 3 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.439 |
| Crosswalk | alt | test | other (private deployments) | 0.105 | 160 | 160 | 8 | 1.000 | 1.000 | 0.997 | 0.996 | 0.950 | 0.969 |
| Crosswalk | alt | test_heldout | private city A (outside the US) | 0.356 | 32 | 26 | 3 | 0.821 | 0.821 | 0.995 | 0.990 | 0.885 | 0.854 |
| Crosswalk | alt | test_heldout | private city B (US) | 0.610 | 33 | 27 | 1 | 0.893 | 0.890 | 0.999 | 0.997 | 0.963 | 0.870 |
| Crosswalk | alt | test_heldout | tucson-az | 0.625 | 3 | 3 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.439 |

## saturation

| label_type | agree_clears_at | map_value_at_0.999999 | n_test_agree | share_agree_at_1 | share_agree_ge_0.999999 | share_disagree_at_1 |
|---|---|---:|---:|---:|---:|---:|
| CurbRamp | 0.99992 | 0.943 | 814 | 0.506 | 0.692 | 0.525 |
| NoCurbRamp | 0.9999 | 0.961 | 1454 | 0.006 | 0.048 | 0.011 |
| Obstacle | 1.0 | 0.788 | 1370 | 0.461 | 0.601 | 0.344 |
| SurfaceProblem | 0.999997 | 0.923 | 1254 | 0.293 | 0.585 | 0.359 |
| Crosswalk | 0.9925 | 0.966 | 258 | 0.000 | 0.132 | 0.021 |

## by severity and tag

### CurbRamp / test / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,080 | 659 / 421 | 0.969 [0.960, 0.979] | 0.921 | 27% | 5,083 |
| 2 | 196 | 74 / 122 | 0.950 [0.914, 0.978] | 0.893 | 20% | 837 |
| 3 | 102 | 28 / 74 | 0.906 [0.813, 0.974] | 0.863 | 29% | 551 |
| (none) | 101 | 51 / 50 | 0.982 [0.958, 0.998] | 0.941 | 36% | 454 |

### CurbRamp / test / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,080 | 421 | 0.969 | 0.957 | +0.013 [+0.003, +0.023] | 27% |
| 2 | 196 | 122 | 0.950 | 0.936 | +0.014 [-0.023, +0.053] | 20% |
| 3 | 102 | 74 | 0.906 | 0.946 | -0.040 [-0.117, +0.023] | 29% |
| (none) | 101 | 50 | 0.982 | 0.984 | -0.001 [-0.020, +0.016] | 36% |

### CurbRamp / test / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 862 | 521 / 341 | 0.974 [0.963, 0.983] | 0.923 | 28% | 3,933 |
| missing tactile warning | 303 | 156 / 147 | 0.935 [0.904, 0.963] | 0.871 | 21% | 1,482 |
| points into traffic | 156 | 92 / 64 | 0.988 [0.970, 1.000] | 0.968 | 28% | 770 |
| not level with street | 123 | 30 / 93 | 0.933 [0.862, 0.988] | 0.902 | 21% | 513 |
| narrow | 95 | 38 / 57 | 0.925 [0.861, 0.979] | 0.853 | 23% | 451 |
| surface problem | 79 | 38 / 41 | 0.909 [0.833, 0.971] | 0.835 | 28% | 351 |
| not enough landing space | 69 | 37 / 32 | 0.952 [0.891, 0.995] | 0.899 | 19% | 348 |
| steep | 50 | 9 / 41 | 0.957 [0.886, 1.000] | 0.880 | 28% | 226 |
| debris / pooled water | 9 | 9 / 0 |  |  | 67% | 74 ⚠ thin |
| parallel lines | 7 | 7 / 0 |  |  | 100% | 14 ⚠ thin |
| not aligned with crosswalk | 3 | 2 / 1 |  |  | 100% | 35 ⚠ thin |

### CurbRamp / test / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 862 | 341 | 0.974 | 0.958 | +0.016 [+0.006, +0.027] | 28% |
| missing tactile warning | 303 | 147 | 0.935 | 0.931 | +0.004 [-0.028, +0.036] | 21% |
| points into traffic | 156 | 64 | 0.988 | 0.969 | +0.019 [-0.004, +0.046] | 28% |
| not level with street | 123 | 93 | 0.933 | 0.930 | +0.003 [-0.065, +0.068] | 21% |
| narrow | 95 | 57 | 0.925 | 0.910 | +0.014 [-0.047, +0.084] | 23% |
| surface problem | 79 | 41 | 0.909 | 0.922 | -0.012 [-0.097, +0.065] | 28% |
| not enough landing space | 69 | 32 | 0.952 | 0.959 | -0.007 [-0.059, +0.048] | 19% |
| steep | 50 | 41 | 0.957 | 0.916 | +0.041 [-0.018, +0.114] | 28% |
| debris / pooled water | 9 | 0 |  |  |  | 67% |
| parallel lines | 7 | 0 |  |  |  | 100% |
| not aligned with crosswalk | 3 | 1 |  |  |  | 100% |

### CurbRamp / test_heldout / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 444 | 273 / 171 | 0.913 [0.883, 0.940] | 0.813 | 57% | 5,083 |
| 2 | 55 | 28 / 27 | 0.873 [0.756, 0.966] | 0.873 | 47% | 837 |
| 3 | 23 | 13 / 10 |  |  | 61% | 551 |
| (none) | 24 | 4 / 20 |  |  | 83% | 454 |

### CurbRamp / test_heldout / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 444 | 171 | 0.913 | 0.884 | +0.029 [+0.010, +0.051] | 57% |
| 2 | 55 | 27 | 0.873 | 0.852 | +0.021 [-0.064, +0.105] | 47% |
| 3 | 23 | 10 |  |  |  | 61% |
| (none) | 24 | 20 |  |  |  | 83% |

### CurbRamp / test_heldout / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 329 | 179 / 150 | 0.925 [0.892, 0.955] | 0.839 | 60% | 3,933 |
| missing tactile warning | 141 | 96 / 45 | 0.858 [0.795, 0.920] | 0.759 | 50% | 1,482 |
| points into traffic | 73 | 51 / 22 | 0.937 [0.842, 1.000] | 0.945 | 51% | 770 |
| not level with street | 28 | 14 / 14 |  |  | 68% | 513 |
| not enough landing space | 22 | 22 / 0 |  |  | 77% | 348 |
| surface problem | 20 | 11 / 9 |  |  | 55% | 351 |
| narrow | 18 | 12 / 6 |  |  | 56% | 451 |
| debris / pooled water | 8 | 7 / 1 |  |  | 88% | 74 ⚠ thin |
| steep | 5 | 3 / 2 |  |  | 60% | 226 |

### CurbRamp / test_heldout / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 329 | 150 | 0.925 | 0.899 | +0.026 [+0.001, +0.052] | 60% |
| missing tactile warning | 141 | 45 | 0.858 | 0.791 | +0.066 [+0.001, +0.131] | 50% |
| points into traffic | 73 | 22 | 0.937 | 0.898 | +0.038 [-0.025, +0.126] | 51% |
| not level with street | 28 | 14 |  |  |  | 68% |
| not enough landing space | 22 | 0 |  |  |  | 77% |
| surface problem | 20 | 9 |  |  |  | 55% |
| narrow | 18 | 6 |  |  |  | 56% |
| debris / pooled water | 8 | 1 |  |  |  | 88% |
| steep | 5 | 2 |  |  |  | 60% |

### NoCurbRamp / test / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 340 | 110 / 230 | 0.894 [0.855, 0.930] | 0.868 | 34% | 1,839 |
| 2 | 743 | 429 / 314 | 0.915 [0.894, 0.934] | 0.844 | 48% | 3,595 |
| 3 | 1,391 | 859 / 532 | 0.946 [0.931, 0.960] | 0.902 | 35% | 6,074 |
| (none) | 139 | 45 / 94 | 0.918 [0.868, 0.963] | 0.856 | 42% | 691 |

### NoCurbRamp / test / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 340 | 230 | 0.894 | 0.900 | -0.007 [-0.041, +0.027] | 34% |
| 2 | 743 | 314 | 0.915 | 0.925 | -0.010 [-0.029, +0.008] | 48% |
| 3 | 1,391 | 532 | 0.946 | 0.937 | +0.010 [-0.005, +0.024] | 35% |
| (none) | 139 | 94 | 0.918 | 0.922 | -0.004 [-0.060, +0.053] | 42% |

### NoCurbRamp / test / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 1,032 | 718 / 314 | 0.944 [0.925, 0.960] | 0.902 | 41% | 4,670 |
| (none) | 979 | 451 / 528 | 0.945 [0.930, 0.960] | 0.891 | 35% | 4,631 |
| alternate route present | 478 | 231 / 247 | 0.879 [0.845, 0.911] | 0.816 | 37% | 2,300 |
| unclear if needed | 181 | 61 / 120 | 0.845 [0.767, 0.914] | 0.807 | 30% | 860 |

### NoCurbRamp / test / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 1,032 | 314 | 0.944 | 0.941 | +0.003 [-0.015, +0.019] | 41% |
| (none) | 979 | 528 | 0.945 | 0.944 | +0.001 [-0.012, +0.014] | 35% |
| alternate route present | 478 | 247 | 0.879 | 0.883 | -0.004 [-0.032, +0.024] | 37% |
| unclear if needed | 181 | 120 | 0.845 | 0.848 | -0.004 [-0.076, +0.065] | 30% |

### NoCurbRamp / test_heldout / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 149 | 11 / 138 | 0.781 [0.574, 0.935] | 0.691 | 58% | 1,839 |
| 2 | 158 | 49 / 109 | 0.843 [0.769, 0.909] | 0.703 | 82% | 3,595 |
| 3 | 418 | 296 / 122 | 0.873 [0.831, 0.911] | 0.823 | 91% | 6,074 |
| (none) | 18 | 3 / 15 |  |  | 50% | 691 |

### NoCurbRamp / test_heldout / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 149 | 138 | 0.781 | 0.882 | -0.101 [-0.243, +0.014] | 58% |
| 2 | 158 | 109 | 0.843 | 0.854 | -0.011 [-0.064, +0.046] | 82% |
| 3 | 418 | 122 | 0.873 | 0.881 | -0.008 [-0.047, +0.032] | 91% |
| (none) | 18 | 15 |  |  |  | 50% |

### NoCurbRamp / test_heldout / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 348 | 270 / 78 | 0.867 [0.817, 0.911] | 0.836 | 92% | 4,670 |
| alternate route present | 209 | 31 / 178 | 0.856 [0.775, 0.925] | 0.727 | 59% | 2,300 |
| (none) | 125 | 46 / 79 | 0.858 [0.782, 0.925] | 0.720 | 65% | 4,631 |
| unclear if needed | 73 | 16 / 57 | 0.865 [0.746, 0.967] | 0.685 | 78% | 860 |

### NoCurbRamp / test_heldout / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 348 | 78 | 0.867 | 0.881 | -0.015 [-0.060, +0.030] | 92% |
| alternate route present | 209 | 178 | 0.856 | 0.880 | -0.024 [-0.093, +0.042] | 59% |
| (none) | 125 | 79 | 0.858 | 0.888 | -0.030 [-0.093, +0.034] | 65% |
| unclear if needed | 73 | 57 | 0.865 | 0.873 | -0.008 [-0.094, +0.078] | 78% |

### Obstacle / test / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 679 | 275 / 404 | 0.862 [0.833, 0.889] | 0.775 | 27% | 3,142 |
| 2 | 641 | 397 / 244 | 0.853 [0.821, 0.883] | 0.796 | 33% | 2,844 |
| 3 | 803 | 581 / 222 | 0.903 [0.877, 0.927] | 0.866 | 29% | 3,899 |
| (none) | 143 | 52 / 91 | 0.877 [0.809, 0.936] | 0.797 | 36% | 669 |

### Obstacle / test / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 679 | 404 | 0.862 | 0.837 | +0.025 [-0.008, +0.058] | 27% |
| 2 | 641 | 244 | 0.853 | 0.812 | +0.041 [+0.005, +0.077] | 33% |
| 3 | 803 | 222 | 0.903 | 0.845 | +0.058 [+0.029, +0.089] | 29% |
| (none) | 143 | 91 | 0.877 | 0.811 | +0.066 [-0.022, +0.167] | 36% |

### Obstacle / test / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| pole | 468 | 249 / 219 | 0.873 [0.842, 0.903] | 0.795 | 29% | 2,087 |
| (none) | 358 | 177 / 181 | 0.891 [0.857, 0.923] | 0.810 | 34% | 1,933 |
| parked car | 286 | 166 / 120 | 0.862 [0.817, 0.905] | 0.797 | 42% | 1,293 |
| trash/recycling can | 236 | 170 / 66 | 0.934 [0.899, 0.962] | 0.890 | 53% | 1,046 |
| vegetation | 230 | 152 / 78 | 0.831 [0.770, 0.885] | 0.787 | 37% | 1,136 |
| narrow | 204 | 156 / 48 | 0.881 [0.823, 0.925] | 0.824 | 29% | 908 |
| tree | 204 | 84 / 120 | 0.893 [0.847, 0.933] | 0.814 | 22% | 854 |
| parked scooter/motorcycle | 105 | 80 / 25 | 0.896 [0.808, 0.965] | 0.857 | 76% | 527 |
| construction | 99 | 76 / 23 | 0.816 [0.719, 0.907] | 0.808 | 30% | 416 |
| sign | 99 | 56 / 43 | 0.914 [0.846, 0.968] | 0.848 | 41% | 486 |
| fire hydrant | 82 | 37 / 45 | 0.865 [0.775, 0.940] | 0.817 | 41% | 354 |
| utility cabinet | 45 | 42 / 3 |  | 0.889 | 100% | 223 |
| parked bike | 41 | 25 / 16 | 0.960 [0.894, 1.000] | 0.854 | 49% | 191 |
| litter/garbage | 39 | 23 / 16 | 0.889 [0.753, 1.000] | 0.872 | 31% | 235 |
| garage entrance | 29 | 13 / 16 |  |  | 48% | 95 ⚠ thin |
| street vendor | 24 | 22 / 2 |  |  | 88% | 114 |
| mailbox | 12 | 6 / 6 |  |  | 67% | 36 ⚠ thin |
| height difference | 10 | 7 / 3 |  |  | 40% | 74 ⚠ thin |
| stairs | 10 | 4 / 6 |  |  | 30% | 35 ⚠ thin |

### Obstacle / test / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| pole | 468 | 219 | 0.873 | 0.825 | +0.047 [+0.011, +0.085] | 29% |
| (none) | 358 | 181 | 0.891 | 0.788 | +0.103 [+0.060, +0.153] | 34% |
| parked car | 286 | 120 | 0.862 | 0.822 | +0.040 [-0.004, +0.084] | 42% |
| trash/recycling can | 236 | 66 | 0.934 | 0.942 | -0.009 [-0.041, +0.026] | 53% |
| vegetation | 230 | 78 | 0.831 | 0.836 | -0.005 [-0.071, +0.058] | 37% |
| narrow | 204 | 48 | 0.881 | 0.924 | -0.044 [-0.104, +0.008] | 29% |
| tree | 204 | 120 | 0.893 | 0.837 | +0.056 [-0.008, +0.119] | 22% |
| parked scooter/motorcycle | 105 | 25 | 0.896 | 0.667 | +0.229 [+0.111, +0.357] | 76% |
| construction | 99 | 23 | 0.816 | 0.951 | -0.135 [-0.241, -0.033] | 30% |
| sign | 99 | 43 | 0.914 | 0.945 | -0.031 [-0.095, +0.025] | 41% |
| fire hydrant | 82 | 45 | 0.865 | 0.909 | -0.043 [-0.144, +0.052] | 41% |
| utility cabinet | 45 | 3 |  |  |  | 100% |
| parked bike | 41 | 16 | 0.960 | 0.955 | +0.005 [-0.063, +0.094] | 49% |
| litter/garbage | 39 | 16 | 0.889 | 0.921 | -0.033 [-0.154, +0.071] | 31% |
| garage entrance | 29 | 16 |  |  |  | 48% |
| street vendor | 24 | 2 |  |  |  | 88% |
| mailbox | 12 | 6 |  |  |  | 67% |
| height difference | 10 | 3 |  |  |  | 40% |
| stairs | 10 | 6 |  |  |  | 30% |

### Obstacle / test_heldout / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,006 | 116 / 890 | 0.868 [0.825, 0.909] | 0.703 | 70% | 3,142 |
| 2 | 335 | 177 / 158 | 0.870 [0.831, 0.907] | 0.764 | 52% | 2,844 |
| 3 | 421 | 285 / 136 | 0.811 [0.765, 0.852] | 0.746 | 54% | 3,899 |
| (none) | 50 | 9 / 41 | 0.829 [0.650, 0.986] | 0.700 | 78% | 669 |

### Obstacle / test_heldout / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,006 | 890 | 0.868 | 0.833 | +0.035 [-0.003, +0.076] | 70% |
| 2 | 335 | 158 | 0.870 | 0.805 | +0.065 [+0.022, +0.110] | 52% |
| 3 | 421 | 136 | 0.811 | 0.777 | +0.034 [-0.001, +0.068] | 54% |
| (none) | 50 | 41 | 0.829 | 0.837 | -0.008 [-0.182, +0.143] | 78% |

### Obstacle / test_heldout / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| pole | 439 | 84 / 355 | 0.927 [0.890, 0.956] | 0.790 | 49% | 2,087 |
| trash/recycling can | 286 | 197 / 89 | 0.939 [0.904, 0.968] | 0.871 | 71% | 1,046 |
| parked car | 164 | 66 / 98 | 0.767 [0.686, 0.843] | 0.622 | 63% | 1,293 |
| vegetation | 156 | 69 / 87 | 0.833 [0.763, 0.896] | 0.750 | 57% | 1,136 |
| (none) | 154 | 38 / 116 | 0.856 [0.781, 0.922] | 0.760 | 66% | 1,933 |
| sign | 139 | 15 / 124 | 0.822 [0.676, 0.942] | 0.662 | 75% | 486 |
| outdoor dining area | 128 | 6 / 122 |  | 0.633 | 100% | 1 ⚠ thin |
| narrow | 75 | 40 / 35 | 0.763 [0.646, 0.862] | 0.667 | 48% | 908 |
| parked bike | 60 | 3 / 57 |  | 0.467 | 100% | 191 |
| construction | 59 | 41 / 18 | 0.730 [0.585, 0.851] | 0.678 | 51% | 416 |
| tree | 59 | 11 / 48 | 0.874 [0.683, 0.987] | 0.831 | 66% | 854 |
| stairs | 47 | 26 / 21 | 0.397 [0.221, 0.564] | 0.447 | 83% | 35 ⚠ thin |
| litter/garbage | 43 | 30 / 13 | 0.810 [0.651, 0.939] | 0.791 | 65% | 235 |
| fire hydrant | 37 | 3 / 34 |  | 0.838 | 46% | 354 |
| parked scooter/motorcycle | 37 | 0 / 37 |  | 0.378 | 100% | 527 |
| mailbox | 24 | 11 / 13 |  |  | 62% | 36 ⚠ thin |
| height difference | 13 | 1 / 12 |  |  | 92% | 74 ⚠ thin |
| garage entrance | 1 | 0 / 1 |  |  | 100% | 95 ⚠ thin |

### Obstacle / test_heldout / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| pole | 439 | 355 | 0.927 | 0.891 | +0.036 [-0.009, +0.080] | 49% |
| trash/recycling can | 286 | 89 | 0.939 | 0.934 | +0.005 [-0.028, +0.038] | 71% |
| parked car | 164 | 98 | 0.767 | 0.748 | +0.020 [-0.045, +0.089] | 63% |
| vegetation | 156 | 87 | 0.833 | 0.763 | +0.070 [-0.001, +0.152] | 57% |
| (none) | 154 | 116 | 0.856 | 0.791 | +0.065 [-0.007, +0.138] | 66% |
| sign | 139 | 124 | 0.822 | 0.848 | -0.027 [-0.164, +0.091] | 75% |
| outdoor dining area | 128 | 122 |  |  |  | 100% |
| narrow | 75 | 35 | 0.763 | 0.567 | +0.196 [+0.074, +0.320] | 48% |
| parked bike | 60 | 57 |  |  |  | 100% |
| construction | 59 | 18 | 0.730 | 0.650 | +0.080 [-0.048, +0.214] | 51% |
| tree | 59 | 48 | 0.874 | 0.824 | +0.050 [-0.082, +0.255] | 66% |
| stairs | 47 | 21 | 0.397 | 0.273 | +0.125 [-0.088, +0.315] | 83% |
| litter/garbage | 43 | 13 | 0.810 | 0.710 | +0.100 [-0.067, +0.274] | 65% |
| fire hydrant | 37 | 34 |  |  |  | 46% |
| parked scooter/motorcycle | 37 | 37 |  |  |  | 100% |
| mailbox | 24 | 13 |  |  |  | 62% |
| height difference | 13 | 12 |  |  |  | 92% |
| garage entrance | 1 | 1 |  |  |  | 100% |

### SurfaceProblem / test / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,107 | 590 / 517 | 0.902 [0.882, 0.920] | 0.833 | 25% | 5,050 |
| 2 | 647 | 384 / 263 | 0.884 [0.853, 0.911] | 0.832 | 25% | 2,959 |
| 3 | 378 | 233 / 145 | 0.899 [0.863, 0.930] | 0.847 | 25% | 1,833 |
| (none) | 123 | 49 / 74 | 0.868 [0.797, 0.925] | 0.789 | 27% | 658 |

### SurfaceProblem / test / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,107 | 517 | 0.902 | 0.893 | +0.009 [-0.010, +0.029] | 25% |
| 2 | 647 | 263 | 0.884 | 0.889 | -0.006 [-0.034, +0.020] | 25% |
| 3 | 378 | 145 | 0.899 | 0.870 | +0.029 [-0.006, +0.062] | 25% |
| (none) | 123 | 74 | 0.868 | 0.825 | +0.043 [-0.028, +0.114] | 27% |

### SurfaceProblem / test / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| cracks | 662 | 501 / 161 | 0.889 [0.854, 0.920] | 0.852 | 27% | 3,031 |
| grass | 528 | 375 / 153 | 0.881 [0.842, 0.916] | 0.833 | 31% | 2,636 |
| bumpy | 437 | 263 / 174 | 0.906 [0.875, 0.933] | 0.831 | 30% | 1,982 |
| (none) | 379 | 116 / 263 | 0.897 [0.858, 0.933] | 0.863 | 30% | 1,490 |
| uneven/slanted | 328 | 231 / 97 | 0.859 [0.811, 0.903] | 0.829 | 28% | 1,574 |
| narrow sidewalk | 217 | 117 / 100 | 0.843 [0.784, 0.892] | 0.765 | 22% | 1,160 |
| height difference | 163 | 131 / 32 | 0.954 [0.916, 0.982] | 0.914 | 21% | 770 |
| utility panel | 162 | 64 / 98 | 0.915 [0.861, 0.959] | 0.877 | 62% | 904 |
| very broken | 113 | 83 / 30 | 0.886 [0.809, 0.953] | 0.850 | 25% | 625 |
| brick/cobblestone | 77 | 26 / 51 | 0.884 [0.798, 0.952] | 0.779 | 53% | 337 |
| sand/gravel | 65 | 36 / 29 | 0.891 [0.806, 0.958] | 0.800 | 22% | 284 |
| construction | 19 | 14 / 5 |  |  | 37% | 44 ⚠ thin |
| debris | 14 | 8 / 6 |  |  | 36% | 66 ⚠ thin |
| rail/tram track | 5 | 2 / 3 |  |  | 40% | 25 ⚠ thin |

### SurfaceProblem / test / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| cracks | 662 | 161 | 0.889 | 0.883 | +0.006 [-0.026, +0.036] | 27% |
| grass | 528 | 153 | 0.881 | 0.863 | +0.018 [-0.018, +0.054] | 31% |
| bumpy | 437 | 174 | 0.906 | 0.884 | +0.023 [-0.007, +0.051] | 30% |
| (none) | 379 | 263 | 0.897 | 0.886 | +0.011 [-0.034, +0.058] | 30% |
| uneven/slanted | 328 | 97 | 0.859 | 0.865 | -0.006 [-0.051, +0.043] | 28% |
| narrow sidewalk | 217 | 100 | 0.843 | 0.848 | -0.005 [-0.054, +0.046] | 22% |
| height difference | 163 | 32 | 0.954 | 0.928 | +0.026 [-0.029, +0.091] | 21% |
| utility panel | 162 | 98 | 0.915 | 0.907 | +0.008 [-0.043, +0.055] | 62% |
| very broken | 113 | 30 | 0.886 | 0.914 | -0.028 [-0.108, +0.048] | 25% |
| brick/cobblestone | 77 | 51 | 0.884 | 0.922 | -0.038 [-0.114, +0.022] | 53% |
| sand/gravel | 65 | 29 | 0.891 | 0.898 | -0.007 [-0.101, +0.083] | 22% |
| construction | 19 | 5 |  |  |  | 37% |
| debris | 14 | 6 |  |  |  | 36% |
| rail/tram track | 5 | 3 |  |  |  | 40% |

### SurfaceProblem / test_heldout / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 574 | 237 / 337 | 0.701 [0.655, 0.744] | 0.666 | 49% | 5,050 |
| 2 | 174 | 95 / 79 | 0.672 [0.588, 0.750] | 0.598 | 40% | 2,959 |
| 3 | 74 | 31 / 43 | 0.823 [0.718, 0.913] | 0.716 | 45% | 1,833 |
| (none) | 25 | 8 / 17 |  |  | 36% | 658 |

### SurfaceProblem / test_heldout / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 574 | 337 | 0.701 | 0.684 | +0.017 [-0.023, +0.053] | 49% |
| 2 | 174 | 79 | 0.672 | 0.634 | +0.037 [-0.019, +0.101] | 40% |
| 3 | 74 | 43 | 0.823 | 0.870 | -0.047 [-0.138, +0.040] | 45% |
| (none) | 25 | 17 |  |  |  | 36% |

### SurfaceProblem / test_heldout / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| brick/cobblestone | 288 | 97 / 191 | 0.540 [0.477, 0.609] | 0.608 | 93% | 337 |
| narrow sidewalk | 148 | 28 / 120 | 0.696 [0.585, 0.800] | 0.642 | 59% | 1,160 |
| cracks | 113 | 85 / 28 | 0.901 [0.821, 0.966] | 0.867 | 85% | 3,031 |
| uneven/slanted | 94 | 56 / 38 | 0.704 [0.602, 0.804] | 0.617 | 36% | 1,574 |
| sand/gravel | 80 | 41 / 39 | 0.615 [0.485, 0.751] | 0.625 | 78% | 284 |
| utility panel | 68 | 36 / 32 | 0.458 [0.314, 0.602] | 0.485 | 56% | 904 |
| bumpy | 63 | 50 / 13 | 0.726 [0.551, 0.898] | 0.651 | 51% | 1,982 |
| grass | 57 | 40 / 17 | 0.803 [0.628, 0.949] | 0.860 | 70% | 2,636 |
| height difference | 31 | 15 / 16 | 0.929 [0.804, 1.000] | 0.903 | 65% | 770 |
| debris | 29 | 24 / 5 |  |  | 93% | 66 ⚠ thin |
| very broken | 21 | 13 / 8 |  |  | 62% | 625 |
| (none) | 20 | 8 / 12 |  |  | 65% | 1,490 |
| rail/tram track | 20 | 11 / 9 |  |  | 65% | 25 ⚠ thin |
| construction | 6 | 1 / 5 |  |  | 100% | 44 ⚠ thin |

### SurfaceProblem / test_heldout / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| brick/cobblestone | 288 | 191 | 0.540 | 0.593 | -0.053 [-0.129, +0.027] | 93% |
| narrow sidewalk | 148 | 120 | 0.696 | 0.669 | +0.026 [-0.055, +0.108] | 59% |
| cracks | 113 | 28 | 0.901 | 0.874 | +0.027 [-0.039, +0.101] | 85% |
| uneven/slanted | 94 | 38 | 0.704 | 0.595 | +0.109 [+0.025, +0.198] | 36% |
| sand/gravel | 80 | 39 | 0.615 | 0.542 | +0.073 [-0.042, +0.182] | 78% |
| utility panel | 68 | 32 | 0.458 | 0.510 | -0.052 [-0.167, +0.072] | 56% |
| bumpy | 63 | 13 | 0.726 | 0.757 | -0.031 [-0.198, +0.153] | 51% |
| grass | 57 | 17 | 0.803 | 0.866 | -0.063 [-0.212, +0.054] | 70% |
| height difference | 31 | 16 | 0.929 | 0.963 | -0.033 [-0.142, +0.051] | 65% |
| debris | 29 | 5 |  |  |  | 93% |
| very broken | 21 | 8 |  |  |  | 62% |
| (none) | 20 | 12 |  |  |  | 65% |
| rail/tram track | 20 | 9 |  |  |  | 65% |
| construction | 6 | 5 |  |  |  | 100% |

### Crosswalk / test / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 287 | 228 / 59 | 0.969 [0.936, 0.991] | 0.951 | 43% | 1,394 |
| 2 | 49 | 10 / 39 | 0.941 [0.856, 1.000] | 0.939 | 24% | 187 |
| 3 | 79 | 1 / 78 |  | 0.975 | 23% | 369 |
| (none) | 30 | 8 / 22 | 1.000 [1.000, 1.000] | 1.000 | 23% | 139 |

### Crosswalk / test / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 287 | 59 | 0.969 | 0.962 | +0.007 [-0.019, +0.035] | 43% |
| 2 | 49 | 39 | 0.941 | 0.967 | -0.026 [-0.085, +0.015] | 24% |
| 3 | 79 | 78 |  |  |  | 23% |
| (none) | 30 | 22 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 23% |

### Crosswalk / test / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 324 | 199 / 125 | 0.988 [0.974, 0.998] | 0.966 | 37% | 1,557 |
| paint fading | 78 | 34 / 44 | 0.965 [0.909, 1.000] | 0.949 | 27% | 339 |
| uneven surface | 17 | 4 / 13 |  |  | 29% | 64 ⚠ thin |
| bumpy | 15 | 7 / 8 |  |  | 47% | 58 ⚠ thin |
| broken surface | 14 | 10 / 4 |  |  | 43% | 58 ⚠ thin |
| very long crossing | 12 | 4 / 8 |  |  | 33% | 68 ⚠ thin |
| no pedestrian priority | 10 | 2 / 8 |  |  | 40% | 33 ⚠ thin |
| brick/cobblestone | 5 | 1 / 4 |  |  | 40% | 48 ⚠ thin |
| too close to traffic | 4 | 4 / 0 |  |  | 100% | 20 ⚠ thin |
| rail/tram track | 3 | 1 / 2 |  |  | 33% | 10 ⚠ thin |

### Crosswalk / test / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 324 | 125 | 0.988 | 0.977 | +0.011 [-0.003, +0.030] | 37% |
| paint fading | 78 | 44 | 0.965 | 0.967 | -0.002 [-0.031, +0.026] | 27% |
| uneven surface | 17 | 13 |  |  |  | 29% |
| bumpy | 15 | 8 |  |  |  | 47% |
| broken surface | 14 | 4 |  |  |  | 43% |
| very long crossing | 12 | 8 |  |  |  | 33% |
| no pedestrian priority | 10 | 8 |  |  |  | 40% |
| brick/cobblestone | 5 | 4 |  |  |  | 40% |
| too close to traffic | 4 | 0 |  |  |  | 100% |
| rail/tram track | 3 | 2 |  |  |  | 33% |

### Crosswalk / test_heldout / by severity

| Severity | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 69 | 56 / 13 | 0.782 [0.632, 0.939] | 0.870 | 52% | 1,394 |
| 2 | 47 | 5 / 42 |  | 0.979 | 91% | 187 |
| 3 | 17 | 4 / 13 |  |  | 65% | 369 |
| (none) | 5 | 0 / 5 |  |  | 100% | 139 |

### Crosswalk / test_heldout / by severity: round 3 vs production

| Severity | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 69 | 13 | 0.782 | 0.780 | +0.001 [-0.165, +0.208] | 52% |
| 2 | 47 | 42 |  |  |  | 91% |
| 3 | 17 | 13 |  |  |  | 65% |
| (none) | 5 | 5 |  |  |  | 100% |

### Crosswalk / test_heldout / by tag

| Tag | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 | largest one-deployment share | training rows |
|---|---:|---:|---:|---:|---:|---:|
| paint fading | 66 | 16 / 50 | 0.944 [0.863, 1.000] | 0.939 | 77% | 339 |
| (none) | 55 | 40 / 15 | 0.817 [0.670, 0.952] | 0.836 | 58% | 1,557 |
| broken surface | 12 | 6 / 6 |  |  | 58% | 58 ⚠ thin |
| rail/tram track | 6 | 5 / 1 |  |  | 100% | 10 ⚠ thin |
| brick/cobblestone | 5 | 3 / 2 |  |  | 80% | 48 ⚠ thin |
| no pedestrian priority | 5 | 0 / 5 |  |  | 100% | 33 ⚠ thin |
| uneven surface | 4 | 2 / 2 |  |  | 50% | 64 ⚠ thin |
| bumpy | 2 | 0 / 2 |  |  | 50% | 58 ⚠ thin |
| very long crossing | 1 | 1 / 0 |  |  | 100% | 68 ⚠ thin |

### Crosswalk / test_heldout / by tag: round 3 vs production

| Tag | n | incorrect | AUC, round 3 | AUC, production | Δ [95% CI] | largest one-deployment share |
|---|---:|---:|---:|---:|---:|---:|
| paint fading | 66 | 50 | 0.944 | 0.917 | +0.026 [-0.004, +0.071] | 77% |
| (none) | 55 | 15 | 0.817 | 0.782 | +0.035 [-0.147, +0.239] | 58% |
| broken surface | 12 | 6 |  |  |  | 58% |
| rail/tram track | 6 | 1 |  |  |  | 100% |
| brick/cobblestone | 5 | 2 |  |  |  | 80% |
| no pedestrian priority | 5 | 5 |  |  |  | 100% |
| uneven surface | 4 | 2 |  |  |  | 50% |
| bumpy | 2 | 2 |  |  |  | 50% |
| very long crossing | 1 | 0 |  |  |  | 100% |
