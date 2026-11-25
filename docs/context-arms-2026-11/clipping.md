# How much context the wide crop adds

The wide crop asks for 4x the depth-sized half-width, capped at the whole 2048 px view (half-width 1024). `clipped` = the requested context did not fit; `whole view` = the wide crop is the view itself. Quantiles of the wide half-width in view px.

| type | labels | clipped | whole view | q10 | q25 | q50 | q75 | q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCurbRamp | 18,162 | 45% | 46% | 552 | 748 | 980 | 1024 | 1024 |
| Obstacle | 16,894 | 41% | 42% | 492 | 688 | 940 | 1024 | 1024 |
| SurfaceProblem | 15,852 | 49% | 50% | 596 | 788 | 1020 | 1024 | 1024 |
