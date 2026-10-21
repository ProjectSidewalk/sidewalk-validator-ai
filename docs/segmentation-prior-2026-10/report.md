## Segmenter sanity

| label_type | n_views | any_sidewalk | sidewalk_frac_view_median | views_sidewalk_above_horizon_gt_5pct | sidewalk_px_above_horizon_share | point_on_sidewalk | on_road | point_on_ground_object | near_sidewalk_1m | depth_over_ground_range_median | depth_over_ground_range_q25 | depth_over_ground_range_q75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 800 | 0.981 | 0.039 | 0.005 | 0.001 | 0.365 | 0.113 | 0.025 | 0.801 | 2.592 | 2.246 | 2.925 |
| Obstacle | 400 | 0.975 | 0.038 | 0.003 | 0.000 | 0.172 | 0.048 | 0.005 | 0.782 | 2.497 | 2.012 | 2.811 |
| SurfaceProblem | 400 | 0.988 | 0.042 | 0.007 | 0.003 | 0.557 | 0.177 | 0.045 | 0.820 | 2.703 | 2.423 | 3.002 |

## Class at the label point (top 8 per type)

| label_type | class | share |
|---|---|---|
| Obstacle | Sidewalk | 0.158 |
| Obstacle | Vegetation | 0.133 |
| Obstacle | Trash Can | 0.117 |
| Obstacle | Pole | 0.102 |
| Obstacle | Car | 0.100 |
| Obstacle | Terrain | 0.052 |
| Obstacle | Building | 0.043 |
| Obstacle | Road | 0.037 |
| SurfaceProblem | Sidewalk | 0.515 |
| SurfaceProblem | Road | 0.158 |
| SurfaceProblem | Terrain | 0.080 |
| SurfaceProblem | Building | 0.052 |
| SurfaceProblem | Pedestrian Area | 0.035 |
| SurfaceProblem | Vegetation | 0.028 |
| SurfaceProblem | Catch Basin | 0.022 |
| SurfaceProblem | Curb | 0.018 |

## Each feature alone (AUC for incorrect; dir '-' = higher value means more likely correct; auc_oriented = max(auc, 1 - auc); auc_within_city = per-city AUCs weighted by n_pos * n_neg (stratified Mann-Whitney), _rows = weighted by rows; cities with both classes and >= 10 rows)

| label_type | split | feature | n | auc | dir | auc_oriented | auc_lo | auc_hi | auc_within_city | auc_within_city_rows | n_cities_within | mean_correct | mean_incorrect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Obstacle | both | point_on_sidewalk | 400 | 0.527 | + | 0.527 | 0.490 | 0.564 | 0.529 | 0.542 | 7 | 0.145 | 0.200 |
| Obstacle | both | point_on_sidewalk_disc | 400 | 0.535 | + | 0.535 | 0.496 | 0.573 | 0.550 | 0.576 | 7 | 0.155 | 0.225 |
| Obstacle | both | sidewalk_frac_disc | 400 | 0.549 | + | 0.549 | 0.500 | 0.599 | 0.542 | 0.589 | 7 | 0.163 | 0.234 |
| Obstacle | both | dist_px_to_sidewalk | 400 | 0.503 | + | 0.503 | 0.445 | 0.558 | 0.529 | 0.465 | 7 | 75.019 | 193.388 |
| Obstacle | both | dist_m_to_sidewalk | 400 | 0.515 | + | 0.515 | 0.458 | 0.571 | 0.529 | 0.479 | 7 | 2.744 | 9.312 |
| Obstacle | both | dist_m_depth_to_sidewalk | 400 | 0.507 | + | 0.507 | 0.448 | 0.562 | 0.522 | 0.471 | 7 | 3.377 | 10.833 |
| Obstacle | both | near_sidewalk_1m | 400 | 0.432 | - | 0.568 | 0.392 | 0.471 | 0.423 | 0.444 | 7 | 0.850 | 0.715 |
| Obstacle | both | sidewalk_frac_crop | 400 | 0.541 | + | 0.541 | 0.484 | 0.600 | 0.472 | 0.518 | 7 | 0.172 | 0.215 |
| Obstacle | both | sidewalk_frac_view | 400 | 0.532 | + | 0.532 | 0.478 | 0.589 | 0.465 | 0.491 | 7 | 0.057 | 0.092 |
| Obstacle | both | road_frac_crop | 400 | 0.487 | - | 0.513 | 0.430 | 0.545 | 0.598 | 0.546 | 7 | 0.251 | 0.246 |
| Obstacle | both | curb_frac_crop | 400 | 0.414 | - | 0.586 | 0.360 | 0.470 | 0.413 | 0.435 | 7 | 0.033 | 0.025 |
| Obstacle | both | on_road | 400 | 0.507 | + | 0.507 | 0.486 | 0.529 | 0.494 | 0.494 | 7 | 0.040 | 0.055 |
| Obstacle | both | connected_to_bottom | 400 | 0.535 | + | 0.535 | 0.505 | 0.566 | 0.513 | 0.513 | 7 | 0.075 | 0.145 |
| SurfaceProblem | both | point_on_sidewalk | 400 | 0.408 | - | 0.593 | 0.362 | 0.457 | 0.433 | 0.410 | 9 | 0.650 | 0.465 |
| SurfaceProblem | both | point_on_sidewalk_disc | 400 | 0.407 | - | 0.593 | 0.360 | 0.456 | 0.443 | 0.418 | 9 | 0.660 | 0.475 |
| SurfaceProblem | both | sidewalk_frac_disc | 400 | 0.408 | - | 0.592 | 0.358 | 0.463 | 0.439 | 0.413 | 9 | 0.640 | 0.473 |
| SurfaceProblem | both | dist_px_to_sidewalk | 400 | 0.613 | + | 0.613 | 0.561 | 0.661 | 0.588 | 0.610 | 9 | 57.393 | 153.433 |
| SurfaceProblem | both | dist_m_to_sidewalk | 400 | 0.617 | + | 0.617 | 0.567 | 0.666 | 0.592 | 0.615 | 9 | 1.467 | 5.672 |
| SurfaceProblem | both | dist_m_depth_to_sidewalk | 400 | 0.615 | + | 0.615 | 0.565 | 0.664 | 0.591 | 0.614 | 9 | 2.132 | 6.632 |
| SurfaceProblem | both | near_sidewalk_1m | 400 | 0.425 | - | 0.575 | 0.388 | 0.461 | 0.439 | 0.431 | 9 | 0.895 | 0.745 |
| SurfaceProblem | both | sidewalk_frac_crop | 400 | 0.458 | - | 0.542 | 0.400 | 0.516 | 0.422 | 0.410 | 9 | 0.229 | 0.225 |
| SurfaceProblem | both | sidewalk_frac_view | 400 | 0.443 | - | 0.557 | 0.387 | 0.503 | 0.412 | 0.408 | 9 | 0.078 | 0.090 |
| SurfaceProblem | both | road_frac_crop | 400 | 0.541 | + | 0.541 | 0.482 | 0.595 | 0.520 | 0.512 | 9 | 0.273 | 0.321 |
| SurfaceProblem | both | curb_frac_crop | 400 | 0.452 | - | 0.548 | 0.395 | 0.507 | 0.480 | 0.462 | 9 | 0.030 | 0.026 |
| SurfaceProblem | both | on_road | 400 | 0.568 | + | 0.568 | 0.530 | 0.603 | 0.535 | 0.552 | 9 | 0.110 | 0.245 |
| SurfaceProblem | both | connected_to_bottom | 400 | 0.495 | - | 0.505 | 0.463 | 0.527 | 0.460 | 0.467 | 9 | 0.130 | 0.120 |
| Obstacle | test | point_on_sidewalk | 250 | 0.520 | + | 0.520 | 0.474 | 0.566 | 0.533 | 0.544 | 4 | 0.144 | 0.184 |
| Obstacle | test | point_on_sidewalk_disc | 250 | 0.524 | + | 0.524 | 0.474 | 0.572 | 0.543 | 0.558 | 4 | 0.160 | 0.208 |
| Obstacle | test | sidewalk_frac_disc | 250 | 0.517 | + | 0.517 | 0.456 | 0.580 | 0.500 | 0.538 | 4 | 0.178 | 0.213 |
| Obstacle | test | dist_px_to_sidewalk | 250 | 0.537 | + | 0.537 | 0.465 | 0.610 | 0.596 | 0.534 | 4 | 43.335 | 246.132 |
| Obstacle | test | dist_m_to_sidewalk | 250 | 0.538 | + | 0.538 | 0.466 | 0.610 | 0.596 | 0.549 | 4 | 0.704 | 12.375 |
| Obstacle | test | dist_m_depth_to_sidewalk | 250 | 0.536 | + | 0.536 | 0.462 | 0.609 | 0.592 | 0.542 | 4 | 1.052 | 14.583 |
| Obstacle | test | near_sidewalk_1m | 250 | 0.420 | - | 0.580 | 0.370 | 0.471 | 0.378 | 0.399 | 4 | 0.856 | 0.696 |
| Obstacle | test | sidewalk_frac_crop | 250 | 0.444 | - | 0.556 | 0.376 | 0.518 | 0.378 | 0.412 | 4 | 0.182 | 0.163 |
| Obstacle | test | sidewalk_frac_view | 250 | 0.453 | - | 0.547 | 0.383 | 0.526 | 0.382 | 0.412 | 4 | 0.056 | 0.051 |
| Obstacle | test | road_frac_crop | 250 | 0.595 | + | 0.595 | 0.524 | 0.666 | 0.679 | 0.622 | 4 | 0.224 | 0.277 |
| Obstacle | test | curb_frac_crop | 250 | 0.439 | - | 0.561 | 0.369 | 0.508 | 0.390 | 0.407 | 4 | 0.033 | 0.027 |
| Obstacle | test | on_road | 250 | 0.516 | + | 0.516 | 0.489 | 0.543 | 0.512 | 0.516 | 4 | 0.040 | 0.072 |
| Obstacle | test | connected_to_bottom | 250 | 0.488 | - | 0.512 | 0.460 | 0.516 | 0.486 | 0.487 | 4 | 0.064 | 0.040 |
| Obstacle | test_heldout | point_on_sidewalk | 150 | 0.540 | + | 0.540 | 0.479 | 0.600 | 0.522 | 0.539 | 3 | 0.147 | 0.227 |
| Obstacle | test_heldout | point_on_sidewalk_disc | 150 | 0.553 | + | 0.553 | 0.490 | 0.618 | 0.563 | 0.598 | 3 | 0.147 | 0.253 |
| Obstacle | test_heldout | sidewalk_frac_disc | 150 | 0.603 | + | 0.603 | 0.523 | 0.685 | 0.622 | 0.651 | 3 | 0.136 | 0.268 |
| Obstacle | test_heldout | dist_px_to_sidewalk | 150 | 0.444 | - | 0.556 | 0.348 | 0.538 | 0.400 | 0.382 | 3 | 127.826 | 105.481 |
| Obstacle | test_heldout | dist_m_to_sidewalk | 150 | 0.474 | - | 0.526 | 0.379 | 0.568 | 0.399 | 0.394 | 3 | 6.144 | 4.206 |
| Obstacle | test_heldout | dist_m_depth_to_sidewalk | 150 | 0.457 | - | 0.543 | 0.360 | 0.551 | 0.389 | 0.386 | 3 | 7.251 | 4.583 |
| Obstacle | test_heldout | near_sidewalk_1m | 150 | 0.453 | - | 0.547 | 0.388 | 0.519 | 0.511 | 0.498 | 3 | 0.840 | 0.747 |
| Obstacle | test_heldout | sidewalk_frac_crop | 150 | 0.695 | + | 0.695 | 0.603 | 0.780 | 0.651 | 0.645 | 3 | 0.155 | 0.302 |
| Obstacle | test_heldout | sidewalk_frac_view | 150 | 0.660 | + | 0.660 | 0.573 | 0.749 | 0.624 | 0.587 | 3 | 0.058 | 0.160 |
| Obstacle | test_heldout | road_frac_crop | 150 | 0.340 | - | 0.660 | 0.250 | 0.426 | 0.444 | 0.455 | 3 | 0.295 | 0.195 |
| Obstacle | test_heldout | curb_frac_crop | 150 | 0.376 | - | 0.624 | 0.282 | 0.467 | 0.458 | 0.469 | 3 | 0.031 | 0.021 |
| Obstacle | test_heldout | on_road | 150 | 0.493 | - | 0.507 | 0.465 | 0.523 | 0.459 | 0.467 | 3 | 0.040 | 0.027 |
| Obstacle | test_heldout | connected_to_bottom | 150 | 0.613 | + | 0.613 | 0.552 | 0.677 | 0.563 | 0.545 | 3 | 0.093 | 0.320 |
| SurfaceProblem | test | point_on_sidewalk | 250 | 0.372 | - | 0.628 | 0.312 | 0.432 | 0.397 | 0.368 | 6 | 0.640 | 0.384 |
| SurfaceProblem | test | point_on_sidewalk_disc | 250 | 0.364 | - | 0.636 | 0.305 | 0.423 | 0.397 | 0.368 | 6 | 0.656 | 0.384 |
| SurfaceProblem | test | sidewalk_frac_disc | 250 | 0.347 | - | 0.653 | 0.282 | 0.414 | 0.379 | 0.353 | 6 | 0.646 | 0.390 |
| SurfaceProblem | test | dist_px_to_sidewalk | 250 | 0.670 | + | 0.670 | 0.608 | 0.731 | 0.646 | 0.672 | 6 | 30.776 | 157.895 |
| SurfaceProblem | test | dist_m_to_sidewalk | 250 | 0.675 | + | 0.675 | 0.613 | 0.735 | 0.653 | 0.680 | 6 | 0.277 | 5.486 |
| SurfaceProblem | test | dist_m_depth_to_sidewalk | 250 | 0.672 | + | 0.672 | 0.610 | 0.733 | 0.648 | 0.675 | 6 | 0.864 | 6.180 |
| SurfaceProblem | test | near_sidewalk_1m | 250 | 0.388 | - | 0.612 | 0.341 | 0.436 | 0.383 | 0.375 | 6 | 0.928 | 0.704 |
| SurfaceProblem | test | sidewalk_frac_crop | 250 | 0.399 | - | 0.601 | 0.329 | 0.471 | 0.377 | 0.365 | 6 | 0.192 | 0.174 |
| SurfaceProblem | test | sidewalk_frac_view | 250 | 0.418 | - | 0.582 | 0.345 | 0.488 | 0.404 | 0.414 | 6 | 0.054 | 0.065 |
| SurfaceProblem | test | road_frac_crop | 250 | 0.589 | + | 0.589 | 0.521 | 0.656 | 0.526 | 0.537 | 6 | 0.244 | 0.337 |
| SurfaceProblem | test | curb_frac_crop | 250 | 0.424 | - | 0.576 | 0.352 | 0.495 | 0.431 | 0.415 | 6 | 0.032 | 0.024 |
| SurfaceProblem | test | on_road | 250 | 0.596 | + | 0.596 | 0.553 | 0.642 | 0.559 | 0.584 | 6 | 0.072 | 0.264 |
| SurfaceProblem | test | connected_to_bottom | 250 | 0.524 | + | 0.524 | 0.495 | 0.555 | 0.510 | 0.508 | 6 | 0.040 | 0.088 |
| SurfaceProblem | test_heldout | point_on_sidewalk | 150 | 0.467 | - | 0.533 | 0.386 | 0.544 | 0.462 | 0.458 | 3 | 0.667 | 0.600 |
| SurfaceProblem | test_heldout | point_on_sidewalk_disc | 150 | 0.480 | - | 0.520 | 0.403 | 0.556 | 0.480 | 0.476 | 3 | 0.667 | 0.627 |
| SurfaceProblem | test_heldout | sidewalk_frac_disc | 150 | 0.501 | + | 0.501 | 0.414 | 0.591 | 0.487 | 0.482 | 3 | 0.630 | 0.610 |
| SurfaceProblem | test_heldout | dist_px_to_sidewalk | 150 | 0.529 | + | 0.529 | 0.447 | 0.610 | 0.541 | 0.538 | 3 | 101.754 | 145.996 |
| SurfaceProblem | test_heldout | dist_m_to_sidewalk | 150 | 0.530 | + | 0.530 | 0.449 | 0.610 | 0.542 | 0.541 | 3 | 3.449 | 5.981 |
| SurfaceProblem | test_heldout | dist_m_depth_to_sidewalk | 150 | 0.530 | + | 0.530 | 0.449 | 0.612 | 0.544 | 0.543 | 3 | 4.246 | 7.387 |
| SurfaceProblem | test_heldout | near_sidewalk_1m | 150 | 0.487 | - | 0.513 | 0.426 | 0.549 | 0.485 | 0.496 | 3 | 0.840 | 0.813 |
| SurfaceProblem | test_heldout | sidewalk_frac_crop | 150 | 0.523 | + | 0.523 | 0.428 | 0.613 | 0.460 | 0.463 | 3 | 0.292 | 0.311 |
| SurfaceProblem | test_heldout | sidewalk_frac_view | 150 | 0.470 | - | 0.530 | 0.378 | 0.561 | 0.419 | 0.402 | 3 | 0.119 | 0.130 |
| SurfaceProblem | test_heldout | road_frac_crop | 150 | 0.468 | - | 0.532 | 0.372 | 0.560 | 0.514 | 0.483 | 3 | 0.322 | 0.296 |
| SurfaceProblem | test_heldout | curb_frac_crop | 150 | 0.488 | - | 0.512 | 0.397 | 0.577 | 0.520 | 0.516 | 3 | 0.027 | 0.028 |
| SurfaceProblem | test_heldout | on_road | 150 | 0.520 | + | 0.520 | 0.457 | 0.583 | 0.515 | 0.515 | 3 | 0.173 | 0.213 |
| SurfaceProblem | test_heldout | connected_to_bottom | 150 | 0.447 | - | 0.553 | 0.380 | 0.514 | 0.420 | 0.419 | 3 | 0.280 | 0.173 |

## Lift (out-of-fold AUC; `_lo`/`_hi` resample rows, `_city_lo`/`_city_hi` resample whole cities; deltas are paired; city CIs blank below 8 cities; within-city = per-city AUCs weighted by n_pos * n_neg, _rows by rows; cities with both classes and >= 10 rows)

| label_type | slice | model | n | n_incorrect | n_cities | auc | auc_lo | auc_hi | auc_city_lo | auc_city_hi | auc_within_city | auc_within_city_rows | n_cities_within | delta_vs | delta_auc | delta_lo | delta_hi | delta_city_lo | delta_city_hi | delta_within_city | delta_within_city_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Obstacle | all (5-fold CV) | prod | 400 | 200 | 28.000 | 0.828 | 0.787 | 0.867 | 0.740 | 0.897 | 0.785 | 0.793 | 7.000 |  |  |  |  |  |  |  |  |
| Obstacle | all (5-fold CV) | cls | 400 | 200 | 28.000 | 0.845 | 0.804 | 0.882 | 0.795 | 0.873 | 0.799 | 0.814 | 7.000 |  |  |  |  |  |  |  |  |
| Obstacle | all (5-fold CV) | cls_feat | 400 | 200 | 28.000 | 0.847 | 0.807 | 0.887 | 0.788 | 0.872 | 0.810 | 0.814 | 7.000 | cls | 0.003 | -0.017 | 0.022 | -0.021 | 0.022 | 0.011 | -0.001 |
| Obstacle | all (5-fold CV) | feat | 400 | 200 | 28.000 | 0.634 | 0.582 | 0.687 | 0.555 | 0.676 | 0.644 | 0.660 | 7.000 |  |  |  |  |  |  |  |  |
| Obstacle | all (5-fold CV) | prod_feat | 400 | 200 | 28.000 | 0.834 | 0.793 | 0.873 | 0.760 | 0.879 | 0.806 | 0.814 | 7.000 | prod | 0.006 | -0.016 | 0.026 | -0.034 | 0.039 | 0.021 | 0.021 |
| Obstacle | test (5-fold CV) | prod | 250 | 125 | 25.000 | 0.821 | 0.765 | 0.869 | 0.732 | 0.918 | 0.820 | 0.823 | 4.000 |  |  |  |  |  |  |  |  |
| Obstacle | test (5-fold CV) | cls | 250 | 125 | 25.000 | 0.813 | 0.756 | 0.863 | 0.760 | 0.850 | 0.803 | 0.809 | 4.000 |  |  |  |  |  |  |  |  |
| Obstacle | test (5-fold CV) | cls_feat | 250 | 125 | 25.000 | 0.823 | 0.768 | 0.872 | 0.738 | 0.865 | 0.835 | 0.829 | 4.000 | cls | 0.010 | -0.016 | 0.039 | -0.039 | 0.032 | 0.032 | 0.020 |
| Obstacle | test (5-fold CV) | feat | 250 | 125 | 25.000 | 0.580 | 0.509 | 0.649 | 0.483 | 0.630 | 0.645 | 0.644 | 4.000 |  |  |  |  |  |  |  |  |
| Obstacle | test (5-fold CV) | prod_feat | 250 | 125 | 25.000 | 0.810 | 0.751 | 0.860 | 0.718 | 0.891 | 0.828 | 0.829 | 4.000 | prod | -0.011 | -0.035 | 0.014 | -0.065 | 0.014 | 0.008 | 0.006 |
| Obstacle | test_heldout (5-fold CV) | prod | 150 | 75 | 3.000 | 0.838 | 0.769 | 0.900 |  |  | 0.718 | 0.757 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | cls | 150 | 75 | 3.000 | 0.890 | 0.834 | 0.942 |  |  | 0.789 | 0.821 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | cls_feat | 150 | 75 | 3.000 | 0.876 | 0.812 | 0.935 |  |  | 0.762 | 0.796 | 3.000 | cls | -0.014 | -0.052 | 0.018 |  |  | -0.027 | -0.026 |
| Obstacle | test_heldout (5-fold CV) | feat | 150 | 75 | 3.000 | 0.723 | 0.641 | 0.803 |  |  | 0.642 | 0.679 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | prod_feat | 150 | 75 | 3.000 | 0.864 | 0.803 | 0.923 |  |  | 0.765 | 0.796 | 3.000 | prod | 0.026 | -0.009 | 0.066 |  |  | 0.047 | 0.039 |
| Obstacle | test_heldout (fit on test) | prod | 150 | 75 | 3.000 | 0.838 | 0.768 | 0.901 |  |  | 0.718 | 0.757 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (fit on test) | cls | 150 | 75 | 3.000 | 0.846 | 0.774 | 0.909 |  |  | 0.771 | 0.799 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (fit on test) | cls_feat | 150 | 75 | 3.000 | 0.800 | 0.720 | 0.874 |  |  | 0.729 | 0.771 | 3.000 | cls | -0.045 | -0.081 | -0.010 |  |  | -0.042 | -0.028 |
| Obstacle | test_heldout (fit on test) | feat | 150 | 75 | 3.000 | 0.410 | 0.320 | 0.507 |  |  | 0.446 | 0.499 | 3.000 |  |  |  |  |  |  |  |  |
| Obstacle | test_heldout (fit on test) | prod_feat | 150 | 75 | 3.000 | 0.793 | 0.715 | 0.867 |  |  | 0.667 | 0.719 | 3.000 | prod | -0.044 | -0.081 | -0.007 |  |  | -0.051 | -0.038 |
| Obstacle | hard | - | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | prod | 400 | 200 | 28.000 | 0.822 | 0.780 | 0.860 | 0.741 | 0.898 | 0.751 | 0.744 | 9.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | cls | 400 | 200 | 28.000 | 0.810 | 0.766 | 0.852 | 0.742 | 0.875 | 0.743 | 0.740 | 9.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | cls_feat | 400 | 200 | 28.000 | 0.813 | 0.771 | 0.853 | 0.733 | 0.890 | 0.747 | 0.749 | 9.000 | cls | 0.003 | -0.017 | 0.024 | -0.017 | 0.027 | 0.004 | 0.009 |
| SurfaceProblem | all (5-fold CV) | feat | 400 | 200 | 28.000 | 0.656 | 0.601 | 0.708 | 0.599 | 0.726 | 0.594 | 0.628 | 9.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | prod_feat | 400 | 200 | 28.000 | 0.825 | 0.784 | 0.864 | 0.754 | 0.890 | 0.778 | 0.775 | 9.000 | prod | 0.002 | -0.017 | 0.022 | -0.015 | 0.018 | 0.027 | 0.030 |
| SurfaceProblem | test (5-fold CV) | prod | 250 | 125 | 25.000 | 0.899 | 0.860 | 0.936 | 0.839 | 0.945 | 0.845 | 0.808 | 6.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | cls | 250 | 125 | 25.000 | 0.872 | 0.824 | 0.916 | 0.795 | 0.925 | 0.848 | 0.818 | 6.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | cls_feat | 250 | 125 | 25.000 | 0.884 | 0.839 | 0.926 | 0.796 | 0.936 | 0.874 | 0.842 | 6.000 | cls | 0.012 | -0.009 | 0.032 | -0.021 | 0.039 | 0.026 | 0.024 |
| SurfaceProblem | test (5-fold CV) | feat | 250 | 125 | 25.000 | 0.698 | 0.633 | 0.767 | 0.615 | 0.783 | 0.631 | 0.683 | 6.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | prod_feat | 250 | 125 | 25.000 | 0.892 | 0.853 | 0.931 | 0.834 | 0.932 | 0.863 | 0.832 | 6.000 | prod | -0.007 | -0.026 | 0.010 | -0.031 | 0.008 | 0.018 | 0.024 |
| SurfaceProblem | test_heldout (5-fold CV) | prod | 150 | 75 | 3.000 | 0.659 | 0.571 | 0.742 |  |  | 0.675 | 0.670 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | cls | 150 | 75 | 3.000 | 0.681 | 0.592 | 0.763 |  |  | 0.658 | 0.650 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | cls_feat | 150 | 75 | 3.000 | 0.669 | 0.581 | 0.751 |  |  | 0.644 | 0.641 | 3.000 | cls | -0.012 | -0.061 | 0.035 |  |  | -0.014 | -0.009 |
| SurfaceProblem | test_heldout (5-fold CV) | feat | 150 | 75 | 3.000 | 0.586 | 0.491 | 0.674 |  |  | 0.563 | 0.564 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | prod_feat | 150 | 75 | 3.000 | 0.692 | 0.601 | 0.774 |  |  | 0.710 | 0.708 | 3.000 | prod | 0.033 | -0.008 | 0.076 |  |  | 0.034 | 0.038 |
| SurfaceProblem | test_heldout (fit on test) | prod | 150 | 75 | 3.000 | 0.659 | 0.567 | 0.744 |  |  | 0.675 | 0.670 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | cls | 150 | 75 | 3.000 | 0.560 | 0.468 | 0.648 |  |  | 0.526 | 0.527 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | cls_feat | 150 | 75 | 3.000 | 0.554 | 0.465 | 0.642 |  |  | 0.528 | 0.529 | 3.000 | cls | -0.006 | -0.055 | 0.042 |  |  | 0.002 | 0.002 |
| SurfaceProblem | test_heldout (fit on test) | feat | 150 | 75 | 3.000 | 0.493 | 0.401 | 0.582 |  |  | 0.493 | 0.479 | 3.000 |  |  |  |  |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | prod_feat | 150 | 75 | 3.000 | 0.641 | 0.550 | 0.726 |  |  | 0.643 | 0.643 | 3.000 | prod | -0.018 | -0.050 | 0.015 |  |  | -0.032 | -0.027 |
| SurfaceProblem | hard | - | 3 | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Per city (n >= 30; private pooled)

| label_type | city | n | n_incorrect | cls | cls_feat | feat | prod | prod_feat |
|---|---|---|---|---|---|---|---|---|
| Obstacle | chicago-il | 42 | 23 | 0.819 | 0.842 | 0.613 | 0.915 | 0.883 |
| Obstacle | other (6 private deployments) | 196 | 91 | 0.879 | 0.869 | 0.666 | 0.800 | 0.837 |
| Obstacle | seattle-wa | 76 | 36 | 0.819 | 0.872 | 0.666 | 0.832 | 0.846 |
| SurfaceProblem | chicago-il | 33 | 10 | 0.813 | 0.848 | 0.696 | 0.896 | 0.896 |
| SurfaceProblem | other (6 private deployments) | 196 | 110 | 0.785 | 0.792 | 0.664 | 0.788 | 0.798 |
| SurfaceProblem | seattle-wa | 54 | 28 | 0.883 | 0.878 | 0.544 | 0.883 | 0.880 |
| SurfaceProblem | tucson-az | 33 | 23 | 0.526 | 0.526 | 0.522 | 0.535 | 0.617 |

## Production checkpoints re-scored on the round-3 crops

| label_type | split | n | auc | balanced_accuracy | recall_incorrect_at_p90 | recall_correct_at_p95 |
|---|---|---|---|---|---|---|
| Obstacle | test | 250 | 0.821 | 0.740 | 0.384 | 0.000 |
| Obstacle | test_heldout | 150 | 0.838 | 0.720 | 0.053 | 0.653 |
| SurfaceProblem | test | 250 | 0.899 | 0.796 | 0.592 | 0.336 |
| SurfaceProblem | test_heldout | 150 | 0.659 | 0.607 | 0.040 | 0.187 |

## Production-verdict strata (783 of 800 sample labels have an assessment)

| label_type | pool | human_class | n | not_on_sidewalk | not_on_lo | not_on_hi | not_near_1m | not_near_lo | not_near_hi |
|---|---|---|---|---|---|---|---|---|---|
| Obstacle | ai_disagree | all | 159 | 0.830 | 0.764 | 0.881 | 0.296 | 0.230 | 0.371 |
| Obstacle | ai_disagree | correct | 40 | 0.925 | 0.801 | 0.974 | 0.350 | 0.221 | 0.505 |
| Obstacle | ai_disagree | incorrect | 119 | 0.798 | 0.717 | 0.861 | 0.277 | 0.205 | 0.364 |
| Obstacle | contested_agree | all | 103 | 0.816 | 0.730 | 0.879 | 0.233 | 0.162 | 0.323 |
| Obstacle | contested_agree | correct | 29 | 0.862 | 0.694 | 0.945 | 0.172 | 0.076 | 0.345 |
| Obstacle | contested_agree | incorrect | 74 | 0.797 | 0.692 | 0.873 | 0.257 | 0.171 | 0.367 |
| Obstacle | other | all | 14 | 0.929 | 0.685 | 0.987 | 0.357 | 0.163 | 0.612 |
| Obstacle | other | correct | 7 | 1.000 | 0.646 | 1.000 | 0.000 | 0.000 | 0.354 |
| Obstacle | other | incorrect | 7 | 0.857 | 0.487 | 0.974 | 0.714 | 0.359 | 0.918 |
| Obstacle | uncontested_agree | all | 124 | 0.823 | 0.746 | 0.880 | 0.089 | 0.050 | 0.152 |
| Obstacle | uncontested_agree | correct | 124 | 0.823 | 0.746 | 0.880 | 0.089 | 0.050 | 0.152 |
| SurfaceProblem | ai_disagree | all | 171 | 0.544 | 0.469 | 0.617 | 0.263 | 0.203 | 0.334 |
| SurfaceProblem | ai_disagree | correct | 41 | 0.415 | 0.278 | 0.566 | 0.195 | 0.102 | 0.340 |
| SurfaceProblem | ai_disagree | incorrect | 130 | 0.585 | 0.499 | 0.666 | 0.285 | 0.214 | 0.367 |
| SurfaceProblem | contested_agree | all | 80 | 0.450 | 0.346 | 0.559 | 0.188 | 0.117 | 0.287 |
| SurfaceProblem | contested_agree | correct | 12 | 0.417 | 0.193 | 0.680 | 0.083 | 0.015 | 0.354 |
| SurfaceProblem | contested_agree | incorrect | 68 | 0.456 | 0.343 | 0.573 | 0.206 | 0.127 | 0.316 |
| SurfaceProblem | other | all | 3 | 0.333 | 0.061 | 0.792 | 0.333 | 0.061 | 0.792 |
| SurfaceProblem | other | correct | 1 | 1.000 | 0.207 | 1.000 | 1.000 | 0.207 | 1.000 |
| SurfaceProblem | other | incorrect | 2 | 0.000 | 0.000 | 0.658 | 0.000 | 0.000 | 0.658 |
| SurfaceProblem | uncontested_agree | all | 146 | 0.322 | 0.252 | 0.401 | 0.075 | 0.043 | 0.130 |
| SurfaceProblem | uncontested_agree | correct | 146 | 0.322 | 0.252 | 0.401 | 0.075 | 0.043 | 0.130 |

## Served verdict vs the re-scored checkpoint (p_incorrect > 0.5 means Disagree)

| label_type | n | same_verdict |
|---|---|---|
| all | 783 | 0.945 |
| Obstacle | 386 | 0.940 |
| SurfaceProblem | 397 | 0.950 |
