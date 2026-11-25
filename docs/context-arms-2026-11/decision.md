# Context arms: pre-registered decision

Applied mechanically from `DECISION_RULE` (below and in `decision.json`; docs/context-arms-2026-11.md has the rule in words). D = seed-mean minus the `baseline-rerun` seed-mean of the within-city AUC; interval D ± t·se with se² = s_arm²/n_arm + s_base²/n_base + var_boot (joint panorama bootstrap within deployment) and the Welch t at 95%. `NO GAIN` means no gain detected at this power; read it next to the power table. Exploratory arms never produce a `WIN`; a single seed's interval omits that arm's seed variance. Only the rule's seeds are read (`seeds_baseline`, `seeds_confirmatory`); any other seed on disk is listed as not used and enters nothing.

```json
{"all_seeds_above_baseline_mean": true, "baseline_variant": "baseline-rerun", "gate": "report_only", "level": 0.95, "min_delta": 0.02, "min_positive_deployments": 2, "prior_rule": "majority2", "regression_test_upper": -0.01, "seeds_baseline": [42, 43, 44, 45, 46], "seeds_confirmatory": [42, 43, 44, 45, 46], "test_no_harm_lower": -0.02}
```

## NoCurbRamp

| arm | kind | seeds | verdict | D held-out within-city [CI] | D test within-city [CI] | arm seeds above baseline mean | held-out deployments (D) | reason |
|---|---|---|---|---:|---:|---:|---|---|
| two_scale | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | +0.013 [-0.008, +0.035] | +0.011 [-0.003, +0.025] | 5 of 5 | private city A (outside the US) -0.001, private city B (US) +0.013, tucson-az +0.023 | no gain detected at this power |
| view_heatmap | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.108 [-0.155, -0.061] | -0.047 [-0.073, -0.021] | 0 of 5 | private city A (outside the US) +0.053, private city B (US) -0.108, tucson-az -0.215 |  |
| view_heatmap-r448 | exploratory | 42 | exploratory | -0.019 [-0.059, +0.021] | -0.017 [-0.042, +0.009] | 0 of 1 | private city A (outside the US) +0.060, private city B (US) -0.019, tucson-az -0.087 | exploratory arms never produce a WIN |
| side_info | exploratory | 42 | exploratory | -0.008 [-0.034, +0.018] | -0.001 [-0.018, +0.016] | 0 of 1 | private city A (outside the US) +0.002, private city B (US) -0.008, tucson-az -0.044 | exploratory arms never produce a WIN |
| side_info_nohq | exploratory | 42 | exploratory | -0.009 [-0.036, +0.017] | -0.001 [-0.022, +0.019] | 0 of 1 | private city A (outside the US) +0.050, private city B (US) -0.010, tucson-az -0.016 | exploratory arms never produce a WIN |

NoCurbRamp / two_scale: 1. D >= min_delta: 0.0133 (fail); 2. interval lower bound > 0: -0.0081 (fail); 3. every arm seed above the baseline seed mean: 5 of 5 (pass); 4. positive in >= 2 of 3 held-out deployments: 2 (pass); 5. test lower bound > -0.02: -0.0026 (pass); REGRESSION: held-out upper bound < 0: 0.0348 (fail); REGRESSION: test upper bound < -0.01: 0.0248 (fail)

NoCurbRamp / view_heatmap: 1. D >= min_delta: -0.108 (fail); 2. interval lower bound > 0: -0.1547 (fail); 3. every arm seed above the baseline seed mean: 0 of 5 (fail); 4. positive in >= 2 of 3 held-out deployments: 1 (fail); 5. test lower bound > -0.02: -0.0732 (fail); REGRESSION: held-out upper bound < 0: -0.0612 (pass); REGRESSION: test upper bound < -0.01: -0.0211 (pass)

Drift row (`baseline-rerun` seeds [42, 43, 44, 45, 46] vs October's round-3 seeds [42, 43, 44]):

| endpoint | rerun mean ± sd | October mean ± sd | rerun − October |
|---|---:|---:|---:|
| within-city held-out | 0.894 ± 0.005 | 0.885 ± 0.024 | +0.009 |
| within-city test | 0.909 ± 0.008 | 0.906 ± 0.003 | +0.003 |

Operating characteristics (seed sd from baseline-rerun, 5 seeds, 0.005; bootstrap sd of D from paired bootstrap of baseline-rerun - baseline-rerun-s43, 0.011; criteria 1-3 simulated, so WIN rates are upper bounds):

| arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | 80% power at |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.004 | 0.004 | 0.16 | 0.44 | 0.90 | +0.045 |
| 5 | 0.007 | 0.008 | 0.26 | 0.60 | 0.97 | +0.040 |

The rule reads 5 confirmatory arm seeds (`seeds_confirmatory`), so the 5-arm-seed row is the one that applies; the other row is for comparison.

## Obstacle

| arm | kind | seeds | verdict | D held-out within-city [CI] | D test within-city [CI] | arm seeds above baseline mean | held-out deployments (D) | reason |
|---|---|---|---|---:|---:|---:|---|---|
| two_scale | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | +0.005 [-0.018, +0.028] | +0.001 [-0.009, +0.012] | 4 of 5 | private city A (outside the US) +0.009, private city B (US) +0.002, tucson-az +0.009 | no gain detected at this power |
| view_heatmap | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.055 [-0.100, -0.009] | -0.050 [-0.072, -0.027] | 0 of 5 | private city A (outside the US) -0.026, private city B (US) -0.073, tucson-az -0.126 |  |
| view_heatmap-r448 | exploratory | 42 | exploratory | -0.001 [-0.049, +0.047] | -0.004 [-0.025, +0.017] | 0 of 1 | private city A (outside the US) +0.001, private city B (US) +0.002, tucson-az -0.050 | exploratory arms never produce a WIN |
| side_info | exploratory | 42 | exploratory | -0.005 [-0.032, +0.023] | -0.012 [-0.028, +0.004] | 0 of 1 | private city A (outside the US) +0.012, private city B (US) -0.023, tucson-az +0.036 | exploratory arms never produce a WIN |
| side_info_nohq | exploratory | 42 | exploratory | +0.029 [-0.002, +0.060] | -0.013 [-0.031, +0.005] | 1 of 1 | private city A (outside the US) +0.047, private city B (US) +0.006, tucson-az +0.100 | exploratory arms never produce a WIN |

Obstacle / two_scale: 1. D >= min_delta: 0.0052 (fail); 2. interval lower bound > 0: -0.018 (fail); 3. every arm seed above the baseline seed mean: 4 of 5 (fail); 4. positive in >= 2 of 3 held-out deployments: 3 (pass); 5. test lower bound > -0.02: -0.0092 (pass); REGRESSION: held-out upper bound < 0: 0.0283 (fail); REGRESSION: test upper bound < -0.01: 0.0118 (fail)

Obstacle / view_heatmap: 1. D >= min_delta: -0.0546 (fail); 2. interval lower bound > 0: -0.0999 (fail); 3. every arm seed above the baseline seed mean: 0 of 5 (fail); 4. positive in >= 2 of 3 held-out deployments: 0 (fail); 5. test lower bound > -0.02: -0.072 (fail); REGRESSION: held-out upper bound < 0: -0.0093 (pass); REGRESSION: test upper bound < -0.01: -0.0271 (pass)

Drift row (`baseline-rerun` seeds [42, 43, 44, 45, 46] vs October's round-3 seeds [42, 43, 44]):

| endpoint | rerun mean ± sd | October mean ± sd | rerun − October |
|---|---:|---:|---:|
| within-city held-out | 0.778 ± 0.012 | 0.782 ± 0.009 | -0.004 |
| within-city test | 0.867 ± 0.003 | 0.863 ± 0.007 | +0.004 |

Operating characteristics (seed sd from baseline-rerun, 5 seeds, 0.012; bootstrap sd of D from paired bootstrap of baseline-rerun - baseline-rerun-s43, 0.010; criteria 1-3 simulated, so WIN rates are upper bounds):

| arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | 80% power at |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.005 | 0.005 | 0.14 | 0.34 | 0.78 | +0.055 |
| 5 | 0.010 | 0.012 | 0.22 | 0.50 | 0.92 | +0.045 |

The rule reads 5 confirmatory arm seeds (`seeds_confirmatory`), so the 5-arm-seed row is the one that applies; the other row is for comparison.

## SurfaceProblem

| arm | kind | seeds | verdict | D held-out within-city [CI] | D test within-city [CI] | arm seeds above baseline mean | held-out deployments (D) | reason |
|---|---|---|---|---:|---:|---:|---|---|
| two_scale | confirmatory | 42, 43, 44, 45, 46 | NO GAIN | -0.001 [-0.042, +0.039] | -0.001 [-0.016, +0.013] | 4 of 5 | private city A (outside the US) -0.009, private city B (US) -0.001, tucson-az +0.029 | no gain detected at this power |
| view_heatmap | confirmatory | 42, 43, 44, 45, 46 | REGRESSION | -0.075 [-0.122, -0.028] | -0.086 [-0.113, -0.058] | 0 of 5 | private city A (outside the US) -0.063, private city B (US) -0.123, tucson-az -0.035 |  |
| view_heatmap-r448 | exploratory | 42 | exploratory | -0.017 [-0.078, +0.044] | -0.047 [-0.077, -0.018] | 0 of 1 | private city A (outside the US) -0.004, private city B (US) -0.064, tucson-az +0.019 | exploratory arms never produce a WIN |
| side_info | exploratory | 42 | exploratory | -0.025 [-0.067, +0.018] | -0.010 [-0.030, +0.010] | 0 of 1 | private city A (outside the US) -0.017, private city B (US) -0.002, tucson-az -0.093 | exploratory arms never produce a WIN |
| side_info_nohq | exploratory | 42 | exploratory | -0.005 [-0.041, +0.032] | -0.001 [-0.017, +0.015] | 0 of 1 | private city A (outside the US) -0.014, private city B (US) +0.005, tucson-az +0.013 | exploratory arms never produce a WIN |

SurfaceProblem / two_scale: 1. D >= min_delta: -0.0012 (fail); 2. interval lower bound > 0: -0.0417 (fail); 3. every arm seed above the baseline seed mean: 4 of 5 (fail); 4. positive in >= 2 of 3 held-out deployments: 1 (fail); 5. test lower bound > -0.02: -0.0159 (pass); REGRESSION: held-out upper bound < 0: 0.0393 (fail); REGRESSION: test upper bound < -0.01: 0.013 (fail)

SurfaceProblem / view_heatmap: 1. D >= min_delta: -0.0748 (fail); 2. interval lower bound > 0: -0.1218 (fail); 3. every arm seed above the baseline seed mean: 0 of 5 (fail); 4. positive in >= 2 of 3 held-out deployments: 0 (fail); 5. test lower bound > -0.02: -0.1132 (fail); REGRESSION: held-out upper bound < 0: -0.0278 (pass); REGRESSION: test upper bound < -0.01: -0.0578 (pass)

Drift row (`baseline-rerun` seeds [42, 43, 44, 45, 46] vs October's round-3 seeds [42, 43, 44]):

| endpoint | rerun mean ± sd | October mean ± sd | rerun − October |
|---|---:|---:|---:|
| within-city held-out | 0.692 ± 0.014 | 0.706 ± 0.021 | -0.015 |
| within-city test | 0.860 ± 0.006 | 0.863 ± 0.014 | -0.003 |

Operating characteristics (seed sd from baseline-rerun, 5 seeds, 0.014; bootstrap sd of D from paired bootstrap of baseline-rerun - baseline-rerun-s43, 0.015; criteria 1-3 simulated, so WIN rates are upper bounds):

| arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | 80% power at |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.005 | 0.005 | 0.06 | 0.15 | 0.48 | +0.070 |
| 5 | 0.009 | 0.009 | 0.11 | 0.24 | 0.67 | +0.060 |

The rule reads 5 confirmatory arm seeds (`seeds_confirmatory`), so the 5-arm-seed row is the one that applies; the other row is for comparison.

