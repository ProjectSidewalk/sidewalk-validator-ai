# RampNet's backbone

The converted RampNet ConvNeXt V2-B against the `convnextv2-base` control (the same architecture and final layernorm, ImageNet stem and stages) and against the in-env baseline. Its `test` numbers are indirectly exposed: RampNet's crop model trained on CurbRamp labels of deployments now in round 3's `test` split. Conclusions about it rest on held-out only.

| type | seed | Δ held-out vs control | Δ within-city held-out vs control | Δ test vs control | Δ held-out vs baseline | Δ within-city held-out vs baseline |
|---|---|---|---|---|---|---|
| CurbRamp | 42 | -0.007 [-0.036, +0.022] | -0.035 [-0.080, +0.013] | +0.012 [+0.001, +0.022] (indirectly exposed) | -0.050 [-0.075, -0.023] | -0.076 [-0.121, -0.029] |
| CurbRamp | 43 | +0.009 [-0.012, +0.030] | +0.000 [-0.034, +0.034] | +0.013 [+0.004, +0.021] (indirectly exposed) | -0.011 [-0.031, +0.009] | -0.004 [-0.043, +0.035] |
| CurbRamp | 44 | -0.005 [-0.030, +0.018] | -0.020 [-0.056, +0.018] | +0.004 [-0.007, +0.015] (indirectly exposed) | -0.027 [-0.052, -0.004] | -0.027 [-0.071, +0.018] |
| SurfaceProblem | 42 | +0.022 [-0.012, +0.051] | -0.032 [-0.079, +0.012] | -0.042 [-0.058, -0.027] (indirectly exposed) | -0.070 [-0.106, -0.035] | -0.117 [-0.165, -0.071] |
| SurfaceProblem | 43 | -0.003 [-0.031, +0.025] | -0.006 [-0.044, +0.030] | +0.006 [-0.004, +0.017] (indirectly exposed) | -0.027 [-0.059, +0.009] | -0.033 [-0.080, +0.015] |
| SurfaceProblem | 44 | -0.001 [-0.033, +0.029] | -0.015 [-0.057, +0.026] | -0.012 [-0.025, +0.001] (indirectly exposed) | +0.007 [-0.024, +0.038] | +0.016 [-0.031, +0.060] |

## Exposure (`analysis/backbone_rampnet_exposure.json`)

RampNet stage two trained on open-government curb-ramp locations in three cities that are not Project Sidewalk deployments, with keypoints placed by a crop model. That crop model trained on Project Sidewalk CurbRamp labels (Agree - Disagree >= 2) from the 12 deployments below, every one of them in round 3's train, val and test splits (one of them in val for CurbRamp only); none of round 3's three held-out deployments is among them.

| deployment | RampNet stage | round-3 splits |
|---|---|---|
| blackhawk-hills-il | crop model (CurbRamp labels) | train, test (val: CurbRamp only) |
| chicago-il | crop model (CurbRamp labels) | train, val, test |
| cliffside-park-nj | crop model (CurbRamp labels) | train, val, test |
| columbus-oh | crop model (CurbRamp labels) | train, val, test |
| knox-oh | crop model (CurbRamp labels) | train, val, test |
| mendota-il | crop model (CurbRamp labels) | train, val, test |
| newberg-or | crop model (CurbRamp labels) | train, val, test |
| oradell-nj | crop model (CurbRamp labels) | train, val, test |
| pittsburgh-pa | crop model (CurbRamp labels) | train, val, test |
| seattle-wa | crop model (CurbRamp labels) | train, val, test |
| st-louis-mo | crop model (CurbRamp labels) | train, val, test |
| teaneck-nj | crop model (CurbRamp labels) | train, val, test |
