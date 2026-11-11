# The 0.92 gate under each prior rule (uncapped maps, fit on val)

Cell: coverage / precision (gated n), reweighted to the rule's prior.

## Agree, test

| Type | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|
| Crosswalk | 0.991 / 0.995 (255) | 0.992 / 0.997 (255) | 0.992 / 0.997 (255) | 0.992 / 0.997 (255) |
| CurbRamp | 0.883 / 0.994 (692) | 0.887 / 0.998 (692) | 0.886 / 0.997 (692) | 0.886 / 0.997 (692) |
| NoCurbRamp | 0.440 / 0.973 (627) | 0.598 / 0.981 (834) | 0.710 / 0.985 (984) | 0.708 / 0.983 (984) |
| Obstacle | 0.475 / 0.959 (631) | 0.509 / 0.986 (631) | 0.500 / 0.979 (631) | 0.494 / 0.974 (631) |
| SurfaceProblem | 0.676 / 0.957 (819) | 0.703 / 0.988 (819) | 0.699 / 0.984 (819) | 0.696 / 0.980 (819) |

## Disagree, test

| Type | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|
| Crosswalk | 0.422 / 1.000 (98) | 0.370 / 1.000 (98) | 0.370 / 1.000 (98) | 0.388 / 1.000 (98) |
| CurbRamp | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| NoCurbRamp | 0.206 / 0.978 (249) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| Obstacle | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| SurfaceProblem | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |

## Agree, test_heldout

| Type | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|
| Crosswalk | 0.864 / 0.996 (56) | 0.865 / 0.997 (56) | 0.865 / 0.997 (56) | 0.865 / 0.997 (56) |
| CurbRamp | 0.751 / 0.991 (198) | 0.756 / 0.997 (198) | 0.756 / 0.996 (198) | 0.755 / 0.995 (198) |
| NoCurbRamp | 0.372 / 0.933 (158) | 0.548 / 0.951 (224) | 0.634 / 0.955 (260) | 0.630 / 0.950 (260) |
| Obstacle | 0.489 / 0.970 (325) | 0.544 / 0.990 (325) | 0.528 / 0.984 (325) | 0.520 / 0.981 (325) |
| SurfaceProblem | 0.528 / 0.890 (178) | 0.564 / 0.967 (178) | 0.559 / 0.957 (178) | 0.554 / 0.947 (178) |

## Disagree, test_heldout

| Type | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|
| Crosswalk | 0.319 / 0.643 (32) | 0.286 / 0.520 (32) | 0.287 / 0.520 (32) | 0.296 / 0.559 (32) |
| CurbRamp | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| NoCurbRamp | 0.056 / 1.000 (17) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| Obstacle | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |
| SurfaceProblem | 0 (never clears) | 0 (never clears) | 0 (never clears) | 0 (never clears) |

