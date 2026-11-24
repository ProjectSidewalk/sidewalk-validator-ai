# Expected winners under no true difference

Seeded Monte Carlo (20,000 draws, seed 0): each eligible arm's per-seed paired Δ held-out, Δ within-city held-out and Δ test are N(0, sd) with sd CurbRamp 0.0085, SurfaceProblem 0.0141; condition 3 is treated as passing, and since it can only remove winners the rates are upper bounds over it. Δ test is independent. Δ within-city held-out is read from the same held-out rows as Δ held-out, so the two are positively correlated; ρ is that per-seed correlation, from 0 (the pre-registered, independent calculation) to 1 (the bound). Their true correlation is not known, so the null winner rate is the range 0.021 to 0.056 at the pre-registered thresholds.

| thresholds | ρ | arms | expected winners | P(at least one) |
|---|---|---|---|---|
| CurbRamp +0.010, SurfaceProblem +0.016 | 0 | 3 | 0.0213 | 0.0212 |
| CurbRamp +0.010, SurfaceProblem +0.016 | 0.5 | 3 | 0.0433 | 0.0428 |
| CurbRamp +0.010, SurfaceProblem +0.016 | 1 | 3 | 0.0559 | 0.0551 |
| CurbRamp +0.020, SurfaceProblem +0.032 (doubled) | 0 | 3 | 0.0001 | 0.0001 |
| CurbRamp +0.020, SurfaceProblem +0.032 (doubled) | 0.5 | 3 | 0.0001 | 0.0001 |
| CurbRamp +0.020, SurfaceProblem +0.032 (doubled) | 1 | 3 | 0.0001 | 0.0001 |
