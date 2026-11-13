# Agree at the 0.92 gate by severity and tag: margin prior (baseline) vs majority2 prior

Uncapped maps fit on val; cells are coverage / precision, reweighted to the rule's prior, blank under 30 Agree verdicts in the slice. n = Agree verdicts in the slice; train = training rows of the slice (correct / incorrect). A label with several tags counts in each.

## Crosswalk, test, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 234 | 0.991 / 0.997 | 0.991 / 0.998 | +0.000 | +0.001 | 1,073 / 321 |
| 2 | 13 |  /  |  /  |  |  | 44 / 143 |
| 3 | 3 |  /  |  /  |  |  | 12 / 357 |
| (none) | 8 |  /  |  /  |  |  | 27 / 112 |

## Crosswalk, test, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 202 | 1.000 / 0.998 | 1.000 / 0.998 | +0.000 | +0.001 | 922 / 635 |
| paint fading | 38 | 0.942 / 0.990 | 0.942 / 0.993 | -0.000 | +0.003 | 158 / 181 |
| broken surface | 10 |  /  |  /  |  |  | 32 / 26 |
| very long crossing | 8 |  /  |  /  |  |  | 36 / 32 |
| bumpy | 7 |  /  |  /  |  |  | 21 / 37 |
| too close to traffic | 4 |  /  |  /  |  |  | 17 / 3 |
| uneven surface | 4 |  /  |  /  |  |  | 15 / 49 |
| no pedestrian priority | 3 |  /  |  /  |  |  | 11 / 22 |
| brick/cobblestone | 1 |  /  |  /  |  |  | 5 / 43 |
| rail/tram track | 1 |  /  |  /  |  |  | 4 / 6 |

## Crosswalk, test_heldout, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 0.867 / 0.995 | 0.867 / 0.996 | +0.000 | +0.001 | 1,073 / 321 |
| 2 | 6 |  /  |  /  |  |  | 44 / 143 |
| 3 | 2 |  /  |  /  |  |  | 12 / 357 |
| (none) | 1 |  /  |  /  |  |  | 27 / 112 |

## Crosswalk, test_heldout, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 45 | 0.891 / 0.993 | 0.892 / 0.995 | +0.001 | +0.002 | 922 / 635 |
| paint fading | 14 |  /  |  /  |  |  | 158 / 181 |
| broken surface | 6 |  /  |  /  |  |  | 32 / 26 |
| rail/tram track | 5 |  /  |  /  |  |  | 4 / 6 |
| brick/cobblestone | 3 |  /  |  /  |  |  | 5 / 43 |
| uneven surface | 2 |  /  |  /  |  |  | 15 / 49 |
| very long crossing | 1 |  /  |  /  |  |  | 36 / 32 |

## CurbRamp, test, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 658 | 0.893 / 0.994 | 0.895 / 0.997 | +0.002 | +0.002 | 3,062 / 2,021 |
| 2 | 75 | 0.833 / 0.985 | 0.838 / 0.992 | +0.005 | +0.007 | 377 / 460 |
| 3 | 28 |  /  |  /  |  |  | 144 / 407 |
| (none) | 53 | 0.910 / 0.996 | 0.914 / 0.998 | +0.004 | +0.002 | 212 / 242 |

## CurbRamp, test, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 533 | 0.921 / 0.994 | 0.924 / 0.997 | +0.003 | +0.003 | 2,381 / 1,552 |
| missing tactile warning | 149 | 0.783 / 0.989 | 0.786 / 0.994 | +0.003 | +0.005 | 770 / 712 |
| points into traffic | 93 | 0.862 / 1.000 | 0.864 / 1.000 | +0.002 | +0.000 | 382 / 388 |
| surface problem | 33 | 0.787 / 0.986 | 0.790 / 0.992 | +0.003 | +0.006 | 176 / 175 |
| narrow | 32 | 0.808 / 0.993 | 0.814 / 0.996 | +0.006 | +0.003 | 234 / 217 |
| not enough landing space | 32 | 0.834 / 1.000 | 0.836 / 1.000 | +0.002 | +0.000 | 164 / 184 |
| not level with street | 28 |  /  |  /  |  |  | 108 / 405 |
| steep | 11 |  /  |  /  |  |  | 49 / 177 |
| debris / pooled water | 9 |  /  |  /  |  |  | 56 / 18 |
| parallel lines | 7 |  /  |  /  |  |  | 12 / 2 |
| not aligned with crosswalk | 2 |  /  |  /  |  |  | 31 / 4 |

## CurbRamp, test_heldout, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 230 | 0.754 / 0.993 | 0.757 / 0.996 | +0.004 | +0.003 | 3,062 / 2,021 |
| 2 | 29 |  /  |  /  |  |  | 377 / 460 |
| 3 | 13 |  /  |  /  |  |  | 144 / 407 |
| (none) | 3 |  /  |  /  |  |  | 212 / 242 |

## CurbRamp, test_heldout, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 148 | 0.752 / 0.995 | 0.755 / 0.997 | +0.003 | +0.002 | 2,381 / 1,552 |
| missing tactile warning | 86 | 0.676 / 0.989 | 0.682 / 0.994 | +0.006 | +0.005 | 770 / 712 |
| points into traffic | 53 | 0.878 / 0.992 | 0.879 / 0.995 | +0.001 | +0.004 | 382 / 388 |
| not enough landing space | 21 |  /  |  /  |  |  | 164 / 184 |
| narrow | 14 |  /  |  /  |  |  | 234 / 217 |
| not level with street | 11 |  /  |  /  |  |  | 108 / 405 |
| surface problem | 8 |  /  |  /  |  |  | 176 / 175 |
| debris / pooled water | 7 |  /  |  /  |  |  | 56 / 18 |
| steep | 2 |  /  |  /  |  |  | 49 / 177 |

## NoCurbRamp, test, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 115 | 0.341 / 0.920 | 0.581 / 0.956 | +0.240 | +0.036 | 614 / 1,225 |
| 2 | 445 | 0.415 / 0.975 | 0.692 / 0.977 | +0.277 | +0.002 | 2,044 / 1,551 |
| 3 | 845 | 0.469 / 0.979 | 0.734 / 0.990 | +0.265 | +0.011 | 3,863 / 2,211 |
| (none) | 49 | 0.381 / 0.916 | 0.664 / 0.948 | +0.283 | +0.032 | 217 / 474 |

## NoCurbRamp, test, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 707 | 0.466 / 0.984 | 0.736 / 0.991 | +0.270 | +0.007 | 3,339 / 1,331 |
| (none) | 446 | 0.434 / 0.980 | 0.716 / 0.990 | +0.282 | +0.010 | 2,117 / 2,514 |
| alternate route present | 257 | 0.398 / 0.924 | 0.641 / 0.942 | +0.243 | +0.018 | 1,129 / 1,171 |
| unclear if needed | 60 | 0.320 / 0.959 | 0.518 / 0.944 | +0.199 | -0.015 | 246 / 614 |

## NoCurbRamp, test_heldout, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 53 | 0.169 / 0.705 | 0.347 / 0.551 | +0.177 | -0.154 | 614 / 1,225 |
| 2 | 82 | 0.349 / 0.801 | 0.540 / 0.876 | +0.190 | +0.075 | 2,044 / 1,551 |
| 3 | 324 | 0.401 / 0.968 | 0.667 / 0.977 | +0.266 | +0.010 | 3,863 / 2,211 |
| (none) | 4 |  /  |  /  |  |  | 217 / 474 |

## NoCurbRamp, test_heldout, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 283 | 0.425 / 0.975 | 0.666 / 0.986 | +0.240 | +0.010 | 3,339 / 1,331 |
| alternate route present | 80 | 0.236 / 0.749 | 0.413 / 0.762 | +0.177 | +0.012 | 1,129 / 1,171 |
| (none) | 73 | 0.309 / 0.818 | 0.665 / 0.895 | +0.356 | +0.077 | 2,117 / 2,514 |
| unclear if needed | 33 | 0.313 / 0.926 | 0.562 / 0.869 | +0.249 | -0.057 | 246 / 614 |

## Obstacle, test, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 310 | 0.419 / 0.956 | 0.456 / 0.973 | +0.037 | +0.017 | 1,382 / 1,760 |
| 2 | 418 | 0.511 / 0.941 | 0.529 / 0.963 | +0.018 | +0.022 | 1,818 / 1,026 |
| 3 | 579 | 0.496 / 0.976 | 0.505 / 0.985 | +0.009 | +0.009 | 2,627 / 1,272 |
| (none) | 63 | 0.301 / 0.914 | 0.324 / 0.945 | +0.023 | +0.032 | 270 / 399 |

## Obstacle, test, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| pole | 275 | 0.444 / 0.961 | 0.470 / 0.976 | +0.026 | +0.015 | 1,160 / 927 |
| (none) | 199 | 0.448 / 0.937 | 0.470 / 0.961 | +0.022 | +0.023 | 782 / 1,151 |
| trash/recycling can | 172 | 0.709 / 0.987 | 0.723 / 0.992 | +0.014 | +0.005 | 759 / 287 |
| parked car | 162 | 0.335 / 0.942 | 0.345 / 0.964 | +0.010 | +0.022 | 775 / 518 |
| vegetation | 153 | 0.540 / 0.933 | 0.553 / 0.958 | +0.014 | +0.025 | 743 / 393 |
| narrow | 146 | 0.557 / 0.981 | 0.568 / 0.988 | +0.011 | +0.007 | 669 / 239 |
| tree | 90 | 0.396 / 0.955 | 0.421 / 0.972 | +0.025 | +0.017 | 421 / 433 |
| construction | 87 | 0.510 / 0.964 | 0.531 / 0.978 | +0.021 | +0.014 | 342 / 74 |
| parked scooter/motorcycle | 83 | 0.492 / 0.981 | 0.505 / 0.988 | +0.013 | +0.007 | 411 / 116 |
| sign | 61 | 0.504 / 0.949 | 0.520 / 0.968 | +0.016 | +0.019 | 277 / 209 |
| utility cabinet | 41 | 0.636 / 0.971 | 0.638 / 0.982 | +0.002 | +0.011 | 206 / 17 |
| fire hydrant | 34 | 0.454 / 0.949 | 0.470 / 0.968 | +0.016 | +0.019 | 190 / 164 |
| parked bike | 27 |  /  |  /  |  |  | 112 / 79 |
| litter/garbage | 24 |  /  |  /  |  |  | 153 / 82 |
| street vendor | 24 |  /  |  /  |  |  | 97 / 17 |
| garage entrance | 12 |  /  |  /  |  |  | 31 / 64 |
| mailbox | 7 |  /  |  /  |  |  | 17 / 19 |
| height difference | 5 |  /  |  /  |  |  | 36 / 38 |
| stairs | 4 |  /  |  /  |  |  | 16 / 19 |

## Obstacle, test_heldout, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 377 | 0.382 / 0.915 | 0.445 / 0.946 | +0.063 | +0.031 | 1,382 / 1,760 |
| 2 | 220 | 0.530 / 0.980 | 0.545 / 0.988 | +0.016 | +0.008 | 1,818 / 1,026 |
| 3 | 304 | 0.534 / 0.988 | 0.545 / 0.993 | +0.011 | +0.004 | 2,627 / 1,272 |
| (none) | 18 |  /  |  /  |  |  | 270 / 399 |

## Obstacle, test_heldout, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| trash/recycling can | 220 | 0.774 / 0.991 | 0.784 / 0.995 | +0.010 | +0.003 | 759 / 287 |
| pole | 160 | 0.477 / 0.937 | 0.511 / 0.961 | +0.034 | +0.023 | 1,160 / 927 |
| parked car | 110 | 0.250 / 0.955 | 0.266 / 0.972 | +0.016 | +0.017 | 775 / 518 |
| vegetation | 86 | 0.409 / 0.970 | 0.423 / 0.982 | +0.014 | +0.011 | 743 / 393 |
| (none) | 63 | 0.343 / 0.942 | 0.364 / 0.964 | +0.021 | +0.022 | 782 / 1,151 |
| sign | 54 | 0.347 / 0.897 | 0.416 / 0.935 | +0.069 | +0.037 | 277 / 209 |
| outdoor dining area | 49 | 0.062 / 1.000 | 0.088 / 1.000 | +0.026 | +0.000 | 0 / 1 |
| construction | 46 | 0.471 / 0.970 | 0.482 / 0.981 | +0.011 | +0.012 | 342 / 74 |
| narrow | 43 | 0.466 / 0.983 | 0.485 / 0.989 | +0.018 | +0.007 | 669 / 239 |
| litter/garbage | 31 | 0.256 / 1.000 | 0.261 / 1.000 | +0.005 | +0.000 | 153 / 82 |
| parked bike | 31 | 0.111 / 1.000 | 0.170 / 1.000 | +0.059 | +0.000 | 112 / 79 |
| parked scooter/motorcycle | 23 |  /  |  /  |  |  | 411 / 116 |
| tree | 19 |  /  |  /  |  |  | 421 / 433 |
| mailbox | 18 |  /  |  /  |  |  | 17 / 19 |
| stairs | 16 |  /  |  /  |  |  | 16 / 19 |
| fire hydrant | 9 |  /  |  /  |  |  | 190 / 164 |
| height difference | 5 |  /  |  /  |  |  | 36 / 38 |

## SurfaceProblem, test, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 587 | 0.647 / 0.958 | 0.668 / 0.981 | +0.021 | +0.022 | 2,641 / 2,409 |
| 2 | 377 | 0.713 / 0.953 | 0.728 / 0.978 | +0.016 | +0.025 | 1,768 / 1,191 |
| 3 | 241 | 0.700 / 0.967 | 0.721 / 0.985 | +0.021 | +0.018 | 1,191 / 642 |
| (none) | 49 | 0.604 / 0.910 | 0.635 / 0.957 | +0.032 | +0.047 | 239 / 419 |

## SurfaceProblem, test, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| cracks | 495 | 0.765 / 0.974 | 0.779 / 0.988 | +0.013 | +0.014 | 2,245 / 786 |
| grass | 373 | 0.757 / 0.963 | 0.772 / 0.983 | +0.015 | +0.020 | 1,898 / 738 |
| bumpy | 261 | 0.680 / 0.979 | 0.706 / 0.990 | +0.025 | +0.011 | 1,178 / 804 |
| uneven/slanted | 237 | 0.698 / 0.958 | 0.714 / 0.981 | +0.016 | +0.022 | 1,148 / 426 |
| height difference | 129 | 0.755 / 0.994 | 0.765 / 0.997 | +0.009 | +0.003 | 643 / 127 |
| (none) | 122 | 0.606 / 0.929 | 0.637 / 0.967 | +0.031 | +0.037 | 419 / 1,071 |
| narrow sidewalk | 110 | 0.550 / 0.936 | 0.571 / 0.970 | +0.021 | +0.034 | 605 / 555 |
| very broken | 86 | 0.798 / 0.964 | 0.814 / 0.983 | +0.016 | +0.019 | 503 / 122 |
| utility panel | 62 | 0.552 / 0.963 | 0.569 / 0.983 | +0.017 | +0.020 | 346 / 558 |
| sand/gravel | 35 | 0.743 / 0.951 | 0.769 / 0.977 | +0.026 | +0.026 | 169 / 115 |
| brick/cobblestone | 17 |  /  |  /  |  |  | 95 / 242 |
| construction | 12 |  /  |  /  |  |  | 24 / 20 |
| debris | 9 |  /  |  /  |  |  | 45 / 21 |
| rail/tram track | 4 |  /  |  /  |  |  | 21 / 4 |

## SurfaceProblem, test_heldout, by severity

| severity | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 225 | 0.472 / 0.871 | 0.491 / 0.937 | +0.019 | +0.066 | 2,641 / 2,409 |
| 2 | 105 | 0.612 / 0.902 | 0.644 / 0.953 | +0.032 | +0.051 | 1,768 / 1,191 |
| 3 | 38 | 0.656 / 0.942 | 0.703 / 0.973 | +0.047 | +0.031 | 1,191 / 642 |
| (none) | 9 |  /  |  /  |  |  | 239 / 419 |

## SurfaceProblem, test_heldout, by tag

| tag | n | margin cov / prec | majority2 cov / prec | Δ coverage | Δ precision | train (c / i) |
|---|---:|---:|---:|---:|---:|---:|
| cracks | 88 | 0.702 / 0.987 | 0.712 / 0.994 | +0.011 | +0.007 | 2,245 / 786 |
| brick/cobblestone | 60 | 0.127 / 0.438 | 0.112 / 0.633 | -0.015 | +0.195 | 95 / 242 |
| narrow sidewalk | 59 | 0.433 / 0.713 | 0.493 / 0.846 | +0.060 | +0.133 | 605 / 555 |
| sand/gravel | 51 | 0.487 / 0.804 | 0.485 / 0.900 | -0.002 | +0.097 | 169 / 115 |
| uneven/slanted | 48 | 0.555 / 0.966 | 0.587 / 0.984 | +0.032 | +0.018 | 1,148 / 426 |
| utility panel | 45 | 0.560 / 0.747 | 0.563 / 0.867 | +0.002 | +0.120 | 346 / 558 |
| grass | 42 | 0.759 / 0.950 | 0.758 / 0.977 | -0.001 | +0.027 | 1,898 / 738 |
| bumpy | 36 | 0.329 / 1.000 | 0.337 / 1.000 | +0.008 | +0.000 | 1,178 / 804 |
| debris | 22 |  /  |  /  |  |  | 45 / 21 |
| height difference | 16 |  /  |  /  |  |  | 643 / 127 |
| very broken | 15 |  /  |  /  |  |  | 503 / 122 |
| rail/tram track | 11 |  /  |  /  |  |  | 21 / 4 |
| (none) | 3 |  /  |  /  |  |  | 419 / 1,071 |
| construction | 1 |  /  |  /  |  |  | 24 / 20 |

