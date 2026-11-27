# Power of the pre-registered rule (computed before any run; amended 2026-11-27)

From `analysis/resolution_study.py power` (`simulate_full_rule`, 20,000 simulated studies per row, seed 0). Every check of the rule is simulated with the interval the report uses (Satterthwaite over the two seed variances and the bootstrap term): each target's criteria 1-3, the guard's two non-inferiority checks, and the six texture checks at 99%. The guard, the test floors and the texture slices are simulated at a true change of 0, so the guard and texture columns are the chance that a clause vetoes a real win by chance. Checks are independent draws. Inputs (`POWER_*` in the module): the targets' within-city held-out seed and bootstrap sds from October (context study, klone job 43403551); pooled test AUC from the context study's five-seed rerun; the CurbRamp guard from the backbone study's in-env baseline; texture slices scaled from one run's slice bootstrap. `study: criteria 1-2 only` is the quantity the first table reported.

| seeds per model | true effect on every target | Obstacle win | SurfaceProblem win | NoCurbRamp win | guard passes | texture screen passes | study: criteria 1-2 only | **study: full rule** |
|---|---|---|---|---|---|---|---|---|
| 3 | +0.000 | 0.025 | 0.024 | 0.025 | 0.989 | 0.968 | 0.002 | **0.002** |
| 3 | +0.015 | 0.210 | 0.085 | 0.084 | 0.989 | 0.968 | 0.039 | **0.038** |
| 3 | +0.030 | 0.632 | 0.218 | 0.197 | 0.989 | 0.968 | 0.250 | **0.239** |
| 3 | +0.050 | 0.966 | 0.499 | 0.446 | 0.989 | 0.968 | 0.707 | **0.676** |
| 5 | +0.000 | 0.024 | 0.024 | 0.024 | 0.992 | 0.968 | 0.001 | **0.001** |
| 5 | +0.015 | 0.253 | 0.106 | 0.107 | 0.992 | 0.968 | 0.059 | **0.056** |
| 5 | +0.030 | 0.730 | 0.304 | 0.311 | 0.992 | 0.968 | 0.408 | **0.390** |
| 5 | +0.050 | 0.990 | 0.681 | 0.681 | 0.992 | 0.968 | 0.894 | **0.858** |

Inputs, (seed sd, bootstrap sd of a paired difference):

- targets_within_city_heldout: {"NoCurbRamp": [0.024, 0.012], "Obstacle": [0.009, 0.01], "SurfaceProblem": [0.021, 0.015]}
- test_auc: {"NoCurbRamp": [0.0023, 0.0032], "Obstacle": [0.0021, 0.0038], "SurfaceProblem": [0.004, 0.004], "CurbRamp": [0.0019, 0.0039]}
- guard_within_city_heldout: [0.011, 0.01]
- texture: {"SurfaceProblem:test": 0.028, "SurfaceProblem:test_heldout": 0.024, "CurbRamp:test": 0.01, "CurbRamp:test_heldout": 0.024, "Crosswalk:test": 0.018, "Crosswalk:test_heldout": 0.025}
