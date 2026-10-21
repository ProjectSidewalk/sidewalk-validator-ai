## Segmenter sanity

| label_type | n_views | any_sidewalk | sidewalk_frac_view_median | views_sidewalk_above_horizon_gt_5pct | sidewalk_px_above_horizon_share | point_on_sidewalk | on_road | near_sidewalk_1m |
|---|---|---|---|---|---|---|---|---|
| all | 800 | 0.981 | 0.039 | 0.005 | 0.001 | 0.365 | 0.113 | 0.704 |
| Obstacle | 400 | 0.975 | 0.038 | 0.003 | 0.000 | 0.172 | 0.048 | 0.657 |
| SurfaceProblem | 400 | 0.988 | 0.042 | 0.007 | 0.003 | 0.557 | 0.177 | 0.750 |

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

## Each feature alone (AUC for incorrect; dir '-' = higher value means more likely correct)

| label_type | split | feature | n | auc | dir | auc_lo | auc_hi | mean_correct | mean_incorrect |
|---|---|---|---|---|---|---|---|---|---|
| Obstacle | both | point_on_sidewalk | 400 | 0.527 | + | 0.490 | 0.564 | 0.145 | 0.200 |
| Obstacle | both | point_on_sidewalk_disc | 400 | 0.535 | + | 0.496 | 0.573 | 0.155 | 0.225 |
| Obstacle | both | sidewalk_frac_disc | 400 | 0.549 | + | 0.500 | 0.599 | 0.163 | 0.234 |
| Obstacle | both | dist_px_to_sidewalk | 400 | 0.503 | + | 0.445 | 0.558 | 75.019 | 193.388 |
| Obstacle | both | dist_m_to_sidewalk | 400 | 0.507 | + | 0.448 | 0.562 | 3.377 | 10.833 |
| Obstacle | both | near_sidewalk_1m | 400 | 0.468 | - | 0.422 | 0.514 | 0.690 | 0.625 |
| Obstacle | both | sidewalk_frac_crop | 400 | 0.541 | + | 0.484 | 0.600 | 0.172 | 0.215 |
| Obstacle | both | sidewalk_frac_view | 400 | 0.532 | + | 0.478 | 0.589 | 0.057 | 0.092 |
| Obstacle | both | road_frac_crop | 400 | 0.487 | - | 0.430 | 0.545 | 0.251 | 0.246 |
| Obstacle | both | curb_frac_crop | 400 | 0.414 | - | 0.360 | 0.470 | 0.033 | 0.025 |
| Obstacle | both | on_road | 400 | 0.507 | + | 0.486 | 0.529 | 0.040 | 0.055 |
| Obstacle | both | connected_to_bottom | 400 | 0.535 | + | 0.505 | 0.566 | 0.075 | 0.145 |
| SurfaceProblem | both | point_on_sidewalk | 400 | 0.408 | - | 0.362 | 0.457 | 0.650 | 0.465 |
| SurfaceProblem | both | point_on_sidewalk_disc | 400 | 0.407 | - | 0.360 | 0.456 | 0.660 | 0.475 |
| SurfaceProblem | both | sidewalk_frac_disc | 400 | 0.408 | - | 0.358 | 0.463 | 0.640 | 0.473 |
| SurfaceProblem | both | dist_px_to_sidewalk | 400 | 0.613 | + | 0.561 | 0.661 | 57.393 | 153.433 |
| SurfaceProblem | both | dist_m_to_sidewalk | 400 | 0.615 | + | 0.565 | 0.664 | 2.132 | 6.632 |
| SurfaceProblem | both | near_sidewalk_1m | 400 | 0.395 | - | 0.352 | 0.437 | 0.855 | 0.645 |
| SurfaceProblem | both | sidewalk_frac_crop | 400 | 0.458 | - | 0.400 | 0.516 | 0.229 | 0.225 |
| SurfaceProblem | both | sidewalk_frac_view | 400 | 0.443 | - | 0.387 | 0.503 | 0.078 | 0.090 |
| SurfaceProblem | both | road_frac_crop | 400 | 0.541 | + | 0.482 | 0.595 | 0.273 | 0.321 |
| SurfaceProblem | both | curb_frac_crop | 400 | 0.452 | - | 0.395 | 0.507 | 0.030 | 0.026 |
| SurfaceProblem | both | on_road | 400 | 0.568 | + | 0.530 | 0.603 | 0.110 | 0.245 |
| SurfaceProblem | both | connected_to_bottom | 400 | 0.495 | - | 0.463 | 0.527 | 0.130 | 0.120 |
| Obstacle | test | point_on_sidewalk | 250 | 0.520 | + | 0.474 | 0.566 | 0.144 | 0.184 |
| Obstacle | test | point_on_sidewalk_disc | 250 | 0.524 | + | 0.474 | 0.572 | 0.160 | 0.208 |
| Obstacle | test | sidewalk_frac_disc | 250 | 0.517 | + | 0.456 | 0.580 | 0.178 | 0.213 |
| Obstacle | test | dist_px_to_sidewalk | 250 | 0.537 | + | 0.465 | 0.610 | 43.335 | 246.132 |
| Obstacle | test | dist_m_to_sidewalk | 250 | 0.536 | + | 0.462 | 0.609 | 1.052 | 14.583 |
| Obstacle | test | near_sidewalk_1m | 250 | 0.456 | - | 0.397 | 0.518 | 0.688 | 0.600 |
| Obstacle | test | sidewalk_frac_crop | 250 | 0.444 | - | 0.376 | 0.518 | 0.182 | 0.163 |
| Obstacle | test | sidewalk_frac_view | 250 | 0.453 | - | 0.383 | 0.526 | 0.056 | 0.051 |
| Obstacle | test | road_frac_crop | 250 | 0.595 | + | 0.524 | 0.666 | 0.224 | 0.277 |
| Obstacle | test | curb_frac_crop | 250 | 0.439 | - | 0.369 | 0.508 | 0.033 | 0.027 |
| Obstacle | test | on_road | 250 | 0.516 | + | 0.489 | 0.543 | 0.040 | 0.072 |
| Obstacle | test | connected_to_bottom | 250 | 0.488 | - | 0.460 | 0.516 | 0.064 | 0.040 |
| Obstacle | test_heldout | point_on_sidewalk | 150 | 0.540 | + | 0.479 | 0.600 | 0.147 | 0.227 |
| Obstacle | test_heldout | point_on_sidewalk_disc | 150 | 0.553 | + | 0.490 | 0.618 | 0.147 | 0.253 |
| Obstacle | test_heldout | sidewalk_frac_disc | 150 | 0.603 | + | 0.523 | 0.685 | 0.136 | 0.268 |
| Obstacle | test_heldout | dist_px_to_sidewalk | 150 | 0.444 | - | 0.348 | 0.538 | 127.826 | 105.481 |
| Obstacle | test_heldout | dist_m_to_sidewalk | 150 | 0.457 | - | 0.360 | 0.551 | 7.251 | 4.583 |
| Obstacle | test_heldout | near_sidewalk_1m | 150 | 0.487 | - | 0.410 | 0.560 | 0.693 | 0.667 |
| Obstacle | test_heldout | sidewalk_frac_crop | 150 | 0.695 | + | 0.603 | 0.780 | 0.155 | 0.302 |
| Obstacle | test_heldout | sidewalk_frac_view | 150 | 0.660 | + | 0.573 | 0.749 | 0.058 | 0.160 |
| Obstacle | test_heldout | road_frac_crop | 150 | 0.340 | - | 0.250 | 0.426 | 0.295 | 0.195 |
| Obstacle | test_heldout | curb_frac_crop | 150 | 0.376 | - | 0.282 | 0.467 | 0.031 | 0.021 |
| Obstacle | test_heldout | on_road | 150 | 0.493 | - | 0.465 | 0.523 | 0.040 | 0.027 |
| Obstacle | test_heldout | connected_to_bottom | 150 | 0.613 | + | 0.552 | 0.677 | 0.093 | 0.320 |
| SurfaceProblem | test | point_on_sidewalk | 250 | 0.372 | - | 0.312 | 0.432 | 0.640 | 0.384 |
| SurfaceProblem | test | point_on_sidewalk_disc | 250 | 0.364 | - | 0.305 | 0.423 | 0.656 | 0.384 |
| SurfaceProblem | test | sidewalk_frac_disc | 250 | 0.347 | - | 0.282 | 0.414 | 0.646 | 0.390 |
| SurfaceProblem | test | dist_px_to_sidewalk | 250 | 0.670 | + | 0.608 | 0.731 | 30.776 | 157.895 |
| SurfaceProblem | test | dist_m_to_sidewalk | 250 | 0.672 | + | 0.610 | 0.733 | 0.864 | 6.180 |
| SurfaceProblem | test | near_sidewalk_1m | 250 | 0.348 | - | 0.297 | 0.401 | 0.872 | 0.568 |
| SurfaceProblem | test | sidewalk_frac_crop | 250 | 0.399 | - | 0.329 | 0.471 | 0.192 | 0.174 |
| SurfaceProblem | test | sidewalk_frac_view | 250 | 0.418 | - | 0.345 | 0.488 | 0.054 | 0.065 |
| SurfaceProblem | test | road_frac_crop | 250 | 0.589 | + | 0.521 | 0.656 | 0.244 | 0.337 |
| SurfaceProblem | test | curb_frac_crop | 250 | 0.424 | - | 0.352 | 0.495 | 0.032 | 0.024 |
| SurfaceProblem | test | on_road | 250 | 0.596 | + | 0.553 | 0.642 | 0.072 | 0.264 |
| SurfaceProblem | test | connected_to_bottom | 250 | 0.524 | + | 0.495 | 0.555 | 0.040 | 0.088 |
| SurfaceProblem | test_heldout | point_on_sidewalk | 150 | 0.467 | - | 0.386 | 0.544 | 0.667 | 0.600 |
| SurfaceProblem | test_heldout | point_on_sidewalk_disc | 150 | 0.480 | - | 0.403 | 0.556 | 0.667 | 0.627 |
| SurfaceProblem | test_heldout | sidewalk_frac_disc | 150 | 0.501 | + | 0.414 | 0.591 | 0.630 | 0.610 |
| SurfaceProblem | test_heldout | dist_px_to_sidewalk | 150 | 0.529 | + | 0.447 | 0.610 | 101.754 | 145.996 |
| SurfaceProblem | test_heldout | dist_m_to_sidewalk | 150 | 0.530 | + | 0.449 | 0.612 | 4.246 | 7.387 |
| SurfaceProblem | test_heldout | near_sidewalk_1m | 150 | 0.473 | - | 0.407 | 0.536 | 0.827 | 0.773 |
| SurfaceProblem | test_heldout | sidewalk_frac_crop | 150 | 0.523 | + | 0.428 | 0.613 | 0.292 | 0.311 |
| SurfaceProblem | test_heldout | sidewalk_frac_view | 150 | 0.470 | - | 0.378 | 0.561 | 0.119 | 0.130 |
| SurfaceProblem | test_heldout | road_frac_crop | 150 | 0.468 | - | 0.372 | 0.560 | 0.322 | 0.296 |
| SurfaceProblem | test_heldout | curb_frac_crop | 150 | 0.488 | - | 0.397 | 0.577 | 0.027 | 0.028 |
| SurfaceProblem | test_heldout | on_road | 150 | 0.520 | + | 0.457 | 0.583 | 0.173 | 0.213 |
| SurfaceProblem | test_heldout | connected_to_bottom | 150 | 0.447 | - | 0.380 | 0.514 | 0.280 | 0.173 |

## Lift (out-of-fold AUC, 95% bootstrap CI; delta is paired)

| label_type | slice | model | n | n_incorrect | auc | auc_lo | auc_hi | delta_vs | delta_auc | delta_lo | delta_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Obstacle | all (5-fold CV) | prod | 400 | 200 | 0.828 | 0.787 | 0.867 |  |  |  |  |
| Obstacle | all (5-fold CV) | cls | 400 | 200 | 0.845 | 0.804 | 0.882 |  |  |  |  |
| Obstacle | all (5-fold CV) | cls_feat | 400 | 200 | 0.842 | 0.800 | 0.882 | cls | -0.002 | -0.019 | 0.013 |
| Obstacle | all (5-fold CV) | feat | 400 | 200 | 0.608 | 0.554 | 0.662 |  |  |  |  |
| Obstacle | all (5-fold CV) | prod_feat | 400 | 200 | 0.831 | 0.791 | 0.868 | prod | 0.003 | -0.017 | 0.025 |
| Obstacle | test (5-fold CV) | prod | 250 | 125 | 0.821 | 0.765 | 0.869 |  |  |  |  |
| Obstacle | test (5-fold CV) | cls | 250 | 125 | 0.813 | 0.756 | 0.863 |  |  |  |  |
| Obstacle | test (5-fold CV) | cls_feat | 250 | 125 | 0.815 | 0.758 | 0.868 | cls | 0.002 | -0.021 | 0.025 |
| Obstacle | test (5-fold CV) | feat | 250 | 125 | 0.551 | 0.480 | 0.620 |  |  |  |  |
| Obstacle | test (5-fold CV) | prod_feat | 250 | 125 | 0.813 | 0.757 | 0.862 | prod | -0.008 | -0.034 | 0.018 |
| Obstacle | test_heldout (5-fold CV) | prod | 150 | 75 | 0.838 | 0.769 | 0.900 |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | cls | 150 | 75 | 0.890 | 0.834 | 0.942 |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | cls_feat | 150 | 75 | 0.875 | 0.812 | 0.934 | cls | -0.015 | -0.049 | 0.014 |
| Obstacle | test_heldout (5-fold CV) | feat | 150 | 75 | 0.707 | 0.619 | 0.791 |  |  |  |  |
| Obstacle | test_heldout (5-fold CV) | prod_feat | 150 | 75 | 0.855 | 0.792 | 0.912 | prod | 0.017 | -0.017 | 0.054 |
| Obstacle | test_heldout (fit on test) | prod | 150 | 75 | 0.838 | 0.768 | 0.901 |  |  |  |  |
| Obstacle | test_heldout (fit on test) | cls | 150 | 75 | 0.846 | 0.774 | 0.909 |  |  |  |  |
| Obstacle | test_heldout (fit on test) | cls_feat | 150 | 75 | 0.797 | 0.714 | 0.870 | cls | -0.049 | -0.083 | -0.016 |
| Obstacle | test_heldout (fit on test) | feat | 150 | 75 | 0.367 | 0.277 | 0.466 |  |  |  |  |
| Obstacle | test_heldout (fit on test) | prod_feat | 150 | 75 | 0.765 | 0.685 | 0.840 | prod | -0.073 | -0.113 | -0.032 |
| Obstacle | hard | - | 1 | 1 |  |  |  |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | prod | 400 | 200 | 0.822 | 0.780 | 0.860 |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | cls | 400 | 200 | 0.810 | 0.766 | 0.852 |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | cls_feat | 400 | 200 | 0.818 | 0.776 | 0.857 | cls | 0.008 | -0.009 | 0.025 |
| SurfaceProblem | all (5-fold CV) | feat | 400 | 200 | 0.656 | 0.601 | 0.709 |  |  |  |  |
| SurfaceProblem | all (5-fold CV) | prod_feat | 400 | 200 | 0.823 | 0.782 | 0.862 | prod | 0.000 | -0.018 | 0.018 |
| SurfaceProblem | test (5-fold CV) | prod | 250 | 125 | 0.899 | 0.860 | 0.936 |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | cls | 250 | 125 | 0.872 | 0.824 | 0.916 |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | cls_feat | 250 | 125 | 0.884 | 0.841 | 0.926 | cls | 0.013 | -0.005 | 0.031 |
| SurfaceProblem | test (5-fold CV) | feat | 250 | 125 | 0.697 | 0.627 | 0.765 |  |  |  |  |
| SurfaceProblem | test (5-fold CV) | prod_feat | 250 | 125 | 0.888 | 0.848 | 0.928 | prod | -0.011 | -0.030 | 0.006 |
| SurfaceProblem | test_heldout (5-fold CV) | prod | 150 | 75 | 0.659 | 0.571 | 0.742 |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | cls | 150 | 75 | 0.681 | 0.592 | 0.763 |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | cls_feat | 150 | 75 | 0.686 | 0.601 | 0.767 | cls | 0.004 | -0.034 | 0.041 |
| SurfaceProblem | test_heldout (5-fold CV) | feat | 150 | 75 | 0.580 | 0.486 | 0.668 |  |  |  |  |
| SurfaceProblem | test_heldout (5-fold CV) | prod_feat | 150 | 75 | 0.687 | 0.597 | 0.767 | prod | 0.028 | -0.008 | 0.067 |
| SurfaceProblem | test_heldout (fit on test) | prod | 150 | 75 | 0.659 | 0.567 | 0.744 |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | cls | 150 | 75 | 0.560 | 0.468 | 0.648 |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | cls_feat | 150 | 75 | 0.554 | 0.467 | 0.642 | cls | -0.006 | -0.054 | 0.043 |
| SurfaceProblem | test_heldout (fit on test) | feat | 150 | 75 | 0.495 | 0.403 | 0.584 |  |  |  |  |
| SurfaceProblem | test_heldout (fit on test) | prod_feat | 150 | 75 | 0.644 | 0.554 | 0.730 | prod | -0.015 | -0.046 | 0.018 |
| SurfaceProblem | hard | - | 3 | 3 |  |  |  |  |  |  |  |

## Per city (n >= 30; private pooled)

| label_type | city | n | n_incorrect | cls | cls_feat | feat | prod | prod_feat |
|---|---|---|---|---|---|---|---|---|
| Obstacle | chicago-il | 42 | 23 | 0.819 | 0.828 | 0.499 | 0.915 | 0.863 |
| Obstacle | other (6 private deployments) | 196 | 91 | 0.879 | 0.869 | 0.646 | 0.800 | 0.830 |
| Obstacle | seattle-wa | 76 | 36 | 0.819 | 0.850 | 0.677 | 0.832 | 0.875 |
| SurfaceProblem | chicago-il | 33 | 10 | 0.813 | 0.848 | 0.704 | 0.896 | 0.909 |
| SurfaceProblem | other (6 private deployments) | 196 | 110 | 0.785 | 0.797 | 0.665 | 0.788 | 0.798 |
| SurfaceProblem | seattle-wa | 54 | 28 | 0.883 | 0.874 | 0.577 | 0.883 | 0.871 |
| SurfaceProblem | tucson-az | 33 | 23 | 0.526 | 0.543 | 0.504 | 0.535 | 0.552 |

## Production model on this sample

| label_type | split | n | auc | balanced_accuracy | recall_incorrect_at_p90 | recall_correct_at_p95 |
|---|---|---|---|---|---|---|
| Obstacle | test | 250 | 0.821 | 0.740 | 0.384 | 0.000 |
| Obstacle | test_heldout | 150 | 0.838 | 0.720 | 0.053 | 0.653 |
| SurfaceProblem | test | 250 | 0.899 | 0.796 | 0.592 | 0.336 |
| SurfaceProblem | test_heldout | 150 | 0.659 | 0.607 | 0.040 | 0.187 |

## Production-verdict strata (783 of 800 sample labels have an assessment)

| label_type | pool | human_class | n | not_on_sidewalk | not_on_lo | not_on_hi | not_near_1m | not_near_lo | not_near_hi |
|---|---|---|---|---|---|---|---|---|---|
| Obstacle | ai_disagree | all | 159 | 0.830 | 0.764 | 0.881 | 0.409 | 0.335 | 0.486 |
| Obstacle | ai_disagree | correct | 40 | 0.925 | 0.801 | 0.974 | 0.475 | 0.329 | 0.625 |
| Obstacle | ai_disagree | incorrect | 119 | 0.798 | 0.717 | 0.861 | 0.387 | 0.304 | 0.476 |
| Obstacle | contested_agree | all | 103 | 0.816 | 0.730 | 0.879 | 0.330 | 0.247 | 0.426 |
| Obstacle | contested_agree | correct | 29 | 0.862 | 0.694 | 0.945 | 0.345 | 0.199 | 0.527 |
| Obstacle | contested_agree | incorrect | 74 | 0.797 | 0.692 | 0.873 | 0.324 | 0.229 | 0.437 |
| Obstacle | other | all | 14 | 0.929 | 0.685 | 0.987 | 0.429 | 0.214 | 0.674 |
| Obstacle | other | correct | 7 | 1.000 | 0.646 | 1.000 | 0.143 | 0.026 | 0.513 |
| Obstacle | other | incorrect | 7 | 0.857 | 0.487 | 0.974 | 0.714 | 0.359 | 0.918 |
| Obstacle | uncontested_agree | all | 124 | 0.823 | 0.746 | 0.880 | 0.258 | 0.189 | 0.342 |
| Obstacle | uncontested_agree | correct | 124 | 0.823 | 0.746 | 0.880 | 0.258 | 0.189 | 0.342 |
| SurfaceProblem | ai_disagree | all | 171 | 0.544 | 0.469 | 0.617 | 0.368 | 0.300 | 0.443 |
| SurfaceProblem | ai_disagree | correct | 41 | 0.415 | 0.278 | 0.566 | 0.244 | 0.138 | 0.393 |
| SurfaceProblem | ai_disagree | incorrect | 130 | 0.585 | 0.499 | 0.666 | 0.408 | 0.327 | 0.494 |
| SurfaceProblem | contested_agree | all | 80 | 0.450 | 0.346 | 0.559 | 0.275 | 0.189 | 0.381 |
| SurfaceProblem | contested_agree | correct | 12 | 0.417 | 0.193 | 0.680 | 0.333 | 0.138 | 0.609 |
| SurfaceProblem | contested_agree | incorrect | 68 | 0.456 | 0.343 | 0.573 | 0.265 | 0.174 | 0.380 |
| SurfaceProblem | other | all | 3 | 0.333 | 0.061 | 0.792 | 0.333 | 0.061 | 0.792 |
| SurfaceProblem | other | correct | 1 | 1.000 | 0.207 | 1.000 | 1.000 | 0.207 | 1.000 |
| SurfaceProblem | other | incorrect | 2 | 0.000 | 0.000 | 0.658 | 0.000 | 0.000 | 0.658 |
| SurfaceProblem | uncontested_agree | all | 146 | 0.322 | 0.252 | 0.401 | 0.096 | 0.058 | 0.155 |
| SurfaceProblem | uncontested_agree | correct | 146 | 0.322 | 0.252 | 0.401 | 0.096 | 0.058 | 0.155 |
