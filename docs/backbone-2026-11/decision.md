# Decision

Recommendation: **keep dinov2-large among the arms run (no arm met the rule)**

Arms compared: 6; the family is not corrected for multiplicity (null.md gives the expected number of null winners under this rule).

| arm | status | condition 1 met on | condition 2 | condition 3 | eligible |
|---|---|---|---|---|---|
| `convnextv2-base` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem fail (coverage), 3 shared seeds | no |
| `dinov2-reg-large` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem pass, 3 shared seeds | yes |
| `dinov3-vitb16` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem pass, 3 shared seeds | no |
| `dinov3-vitl16` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem pass, 3 shared seeds | yes |
| `dinov3-vitl16-256` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem pass, 3 shared seeds | no |
| `rampnet-convnextv2-base` | worse |  | False | CurbRamp pass, 3 shared seeds; SurfaceProblem pass, 3 shared seeds | yes |

## Condition 3: causes and flags (descriptive; they never decide)

Causes of unestimable clauses, and of a relative-precision clause read as pass because the baseline gates no Agree on every shared seed; flags for fewer than three shared seeds, a seed where one model gates no Agree and the other some, a seed whose Δ coverage is below -0.06, an arm precision mean below the baseline's by more than the margin over the seeds where both are defined, and, for a winner, a point estimate beyond a margin whose interval spans it (gate.md).

- `convnextv2-base` CurbRamp, flag: seed 43: Δ coverage -0.110, below -0.06
- `convnextv2-base` SurfaceProblem, flag: seed 42: Δ coverage -0.177, below -0.06
- `convnextv2-base` SurfaceProblem, flag: seed 44: Δ coverage -0.354, below -0.06
- `dinov2-reg-large` SurfaceProblem, flag: seed 42: Δ coverage -0.127, below -0.06
- `dinov3-vitl16` SurfaceProblem, flag: seed 42: Δ coverage -0.203, below -0.06
- `rampnet-convnextv2-base` SurfaceProblem, flag: seed 43: Δ coverage -0.164, below -0.06
- `rampnet-convnextv2-base` SurfaceProblem, flag: seed 44: Δ coverage -0.092, below -0.06
