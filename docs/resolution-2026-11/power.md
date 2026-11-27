# Power of the pre-registered rule (computed before any run)

From `analysis/resolution_study.py power` (`context_arms.simulate_rule`, 4,000 draws, seed 0). Inputs: October's within-city held-out seed sd and the bootstrap sd of a two-seed difference per type (docs/context-arms-2026-11.md, klone job 43403551). Only criteria 1 and 2 are simulated (the test floor, guard and texture clauses can only remove wins), so every rate is an upper bound. The study row assumes the three types independent.

| type | seeds per model | seed sd | bootstrap sd | P(win) at +0.000 | P(win) at +0.015 | P(win) at +0.030 | P(win) at +0.050 | effect at 80% |
|---|---|---|---|---|---|---|---|---|
| Obstacle | 3 | 0.009 | 0.010 | 0.001 | 0.039 | 0.264 | 0.775 | +0.055 |
| SurfaceProblem | 3 | 0.021 | 0.015 | 0.002 | 0.017 | 0.066 | 0.231 | +0.095 |
| NoCurbRamp | 3 | 0.024 | 0.012 | 0.005 | 0.025 | 0.079 | 0.251 | +0.100 |
| study (>= 2 of 3) | 3 |  |  | 0.000 | 0.002 | 0.041 | 0.341 |  |
| Obstacle | 5 | 0.009 | 0.010 | 0.009 | 0.145 | 0.591 | 0.968 | +0.040 |
| SurfaceProblem | 5 | 0.021 | 0.015 | 0.010 | 0.059 | 0.195 | 0.555 | +0.070 |
| NoCurbRamp | 5 | 0.024 | 0.012 | 0.011 | 0.067 | 0.222 | 0.586 | +0.065 |
| study (>= 2 of 3) | 5 |  |  | 0.000 | 0.021 | 0.238 | 0.800 |  |
