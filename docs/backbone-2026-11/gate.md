# The 0.92 gate

Coverage and precision of each verdict at the 0.92 gate, uncapped map fitted on val under each rule's deployment prior (`analysis/prior_rules.py gate_rows`). One table per rule; `majority2` is the rule's input, `margin` a reference.

## Condition 3: Agree at the gate on `test` under `majority2`

The arm's Agree precision, mean over every paired seed, must be ≥ 0.92 (unestimable when a seed's run gates no Agree). Δ = seed mean of (arm − in-env baseline, same seed); the interval is a nominal 95% percentile interval from a paired hierarchical bootstrap (1000 replicates): the shared seeds resampled with replacement, then whole panoramas with one resample shared by every drawn seed and both models, each run on its own val-fitted map. The coverage clause fails only when the interval lies wholly below −0.02, the precision clause only when it lies wholly below −0.01. With three seeds the interval's upper end sits near the best seed's difference, so a clause fails roughly when every shared seed is past the margin: about a 12.5% one-sided false-fail level at a true drop exactly at the margin, not 2.5% (docs/backbone-2026-11.md, condition 3). Dropped = replicates left out because a drawn seed's run gated no Agree (or the resample had one class); fewer than half kept leaves the interval unestimable, except that a baseline that gates no Agree on every shared seed (the arm on none) reads the precision clause as pass; a baseline undefined on some seeds only stays unestimable (causes listed below). Shared seeds = the seeds paired; under 3 is marked. Per-seed values are descriptive.

| arm | type | seeds | precision by seed | baseline precision by seed | mean precision | coverage by seed | baseline coverage by seed | Δ coverage [nominal 95%] | dropped | Δ precision [nominal 95%] | dropped | shared seeds | condition 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `convnextv2-base` | CurbRamp | 42, 43, 44 | 0.994, 0.995, 0.996 | 0.997, 0.996, 0.997 | 0.995 | 0.881, 0.739, 0.777 | 0.876, 0.850, 0.821 | -0.050 [-0.106, +0.007] | 0 | -0.001 [-0.003, +0.000] | 0 | 3 | pass |
| `convnextv2-base` | SurfaceProblem | 42, 43, 44 | 0.976, 0.973, 0.980 | 0.979, 0.985, 0.985 | 0.977 | 0.498, 0.506, 0.174 | 0.675, 0.529, 0.528 | -0.185 [-0.348, -0.036] | 0 | -0.006 [-0.014, +0.002] | 0 | 3 | fail (coverage) |
| `dinov2-reg-large` | CurbRamp | 42, 43, 44 | 0.996, 0.995, 0.996 | 0.997, 0.996, 0.997 | 0.996 | 0.901, 0.906, 0.856 | 0.876, 0.850, 0.821 | +0.039 [+0.014, +0.068] | 0 | -0.001 [-0.002, +0.000] | 0 | 3 | pass |
| `dinov2-reg-large` | SurfaceProblem | 42, 43, 44 | 0.985, 0.985, 0.985 | 0.979, 0.985, 0.985 | 0.985 | 0.548, 0.517, 0.627 | 0.675, 0.529, 0.528 | -0.013 [-0.121, +0.086] | 0 | +0.002 [-0.003, +0.007] | 0 | 3 | pass |
| `dinov3-vitb16` | CurbRamp | 42, 43, 44 | 0.995, 0.995, 0.995 | 0.997, 0.996, 0.997 | 0.995 | 0.885, 0.836, 0.863 | 0.876, 0.850, 0.821 | +0.012 [-0.022, +0.044] | 0 | -0.002 [-0.003, -0.001] | 0 | 3 | pass |
| `dinov3-vitb16` | SurfaceProblem | 42, 43, 44 | 0.983, 0.980, 0.986 | 0.979, 0.985, 0.985 | 0.983 | 0.647, 0.632, 0.489 | 0.675, 0.529, 0.528 | +0.012 [-0.055, +0.101] | 0 | +0.000 [-0.006, +0.006] | 0 | 3 | pass |
| `dinov3-vitl16` | CurbRamp | 42, 43, 44 | 0.995, 0.997, 0.996 | 0.997, 0.996, 0.997 | 0.996 | 0.899, 0.852, 0.857 | 0.876, 0.850, 0.821 | +0.020 [-0.008, +0.049] | 0 | -0.001 [-0.003, +0.000] | 0 | 3 | pass |
| `dinov3-vitl16` | SurfaceProblem | 42, 43, 44 | 0.987, 0.988, 0.988 | 0.979, 0.985, 0.985 | 0.988 | 0.473, 0.601, 0.608 | 0.675, 0.529, 0.528 | -0.017 [-0.184, +0.094] | 0 | +0.005 [+0.000, +0.010] | 0 | 3 | pass |
| `dinov3-vitl16-256` | CurbRamp | 42, 43, 44 | 0.995, 0.996, 0.996 | 0.997, 0.996, 0.997 | 0.996 | 0.900, 0.829, 0.850 | 0.876, 0.850, 0.821 | +0.011 [-0.023, +0.042] | 0 | -0.001 [-0.002, +0.000] | 0 | 3 | pass |
| `dinov3-vitl16-256` | SurfaceProblem | 42, 43, 44 | 0.984, 0.989, 0.990 | 0.979, 0.985, 0.985 | 0.987 | 0.618, 0.541, 0.556 | 0.675, 0.529, 0.528 | -0.006 [-0.056, +0.038] | 0 | +0.004 [+0.001, +0.009] | 0 | 3 | pass |
| `rampnet-convnextv2-base` | CurbRamp | 42, 43, 44 | 0.996, 0.994, 0.994 | 0.997, 0.996, 0.997 | 0.994 | 0.877, 0.932, 0.942 | 0.876, 0.850, 0.821 | +0.068 [+0.004, +0.127] | 0 | -0.002 [-0.004, -0.001] | 0 | 3 | pass |
| `rampnet-convnextv2-base` | SurfaceProblem | 42, 43, 44 | 0.953, 0.986, 0.977 | 0.979, 0.985, 0.985 | 0.972 | 0.804, 0.365, 0.437 | 0.675, 0.529, 0.528 | -0.042 [-0.166, +0.112] | 0 | -0.011 [-0.025, +0.003] | 0 | 3 | pass |

### Causes and flags (descriptive; they never decide)

- `convnextv2-base` CurbRamp, flag: seed 43: Δ coverage -0.110, below -0.06
- `convnextv2-base` SurfaceProblem, flag: seed 42: Δ coverage -0.177, below -0.06
- `convnextv2-base` SurfaceProblem, flag: seed 44: Δ coverage -0.354, below -0.06
- `dinov2-reg-large` SurfaceProblem, flag: seed 42: Δ coverage -0.127, below -0.06
- `dinov3-vitl16` SurfaceProblem, flag: seed 42: Δ coverage -0.203, below -0.06
- `rampnet-convnextv2-base` SurfaceProblem, flag: seed 43: Δ coverage -0.164, below -0.06
- `rampnet-convnextv2-base` SurfaceProblem, flag: seed 44: Δ coverage -0.092, below -0.06

## prior rule `majority2`

| type | arm | seed | lr | split | verdict | n | coverage | precision | gated n |
|---|---|---|---|---|---|---|---|---|---|
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test | Agree | 799 | 0.876 | 0.997 | 677 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test | Disagree | 680 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test_heldout | Agree | 276 | 0.696 | 0.994 | 185 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test_heldout | Disagree | 270 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test | Agree | 786 | 0.902 | 0.996 | 688 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test | Disagree | 693 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test_heldout | Agree | 274 | 0.790 | 0.994 | 208 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test_heldout | Disagree | 272 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test | Agree | 806 | 0.850 | 0.996 | 663 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test | Disagree | 673 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test_heldout | Agree | 262 | 0.727 | 0.996 | 179 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test_heldout | Disagree | 284 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test | Agree | 817 | 0.821 | 0.997 | 644 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test | Disagree | 662 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test_heldout | Agree | 278 | 0.676 | 0.996 | 175 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test_heldout | Disagree | 268 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test | Agree | 825 | 0.881 | 0.994 | 693 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test | Disagree | 654 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 293 | 0.764 | 0.990 | 217 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 253 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test | Agree | 828 | 0.822 | 0.995 | 642 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test | Disagree | 651 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 310 | 0.748 | 0.990 | 226 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 236 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test | Agree | 832 | 0.739 | 0.995 | 572 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test | Disagree | 647 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 302 | 0.580 | 0.991 | 170 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 244 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test | Agree | 770 | 0.777 | 0.996 | 562 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test | Disagree | 709 | 0.040 | 1.000 | 79 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 275 | 0.586 | 0.991 | 156 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 271 | 0.014 | 1.000 | 13 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test | Agree | 812 | 0.901 | 0.996 | 700 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test | Disagree | 667 | 0.011 | 1.000 | 15 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Agree | 302 | 0.641 | 0.994 | 183 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Disagree | 244 | 0.000 |  | 0 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test | Agree | 792 | 0.906 | 0.995 | 702 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test | Disagree | 687 | 0.056 | 0.867 | 74 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Agree | 261 | 0.763 | 0.993 | 192 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Disagree | 285 | 0.003 | 1.000 | 3 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test | Agree | 797 | 0.856 | 0.996 | 658 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test | Disagree | 682 | 0.032 | 1.000 | 46 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Agree | 265 | 0.709 | 0.995 | 178 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Disagree | 281 | 0.001 | 1.000 | 1 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test | Agree | 779 | 0.885 | 0.995 | 667 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test | Disagree | 700 | 0.037 | 1.000 | 65 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Agree | 270 | 0.724 | 0.993 | 189 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Disagree | 276 | 0.003 | 1.000 | 3 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test | Agree | 806 | 0.836 | 0.995 | 642 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test | Disagree | 673 | 0.064 | 1.000 | 100 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Agree | 244 | 0.781 | 0.994 | 182 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Disagree | 302 | 0.041 | 0.597 | 31 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test | Agree | 809 | 0.863 | 0.995 | 665 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test | Disagree | 670 | 0.062 | 1.000 | 98 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Agree | 275 | 0.654 | 0.994 | 171 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Disagree | 271 | 0.058 | 0.439 | 26 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test | Agree | 795 | 0.899 | 0.995 | 695 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test | Disagree | 684 | 0.014 | 1.000 | 21 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Agree | 236 | 0.787 | 0.995 | 181 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Disagree | 310 | 0.003 | 1.000 | 3 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test | Agree | 817 | 0.852 | 0.997 | 662 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test | Disagree | 662 | 0.142 | 0.942 | 183 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Agree | 263 | 0.690 | 0.995 | 176 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Disagree | 283 | 0.028 | 0.635 | 18 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test | Agree | 824 | 0.857 | 0.996 | 679 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test | Disagree | 655 | 0.058 | 1.000 | 73 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Agree | 245 | 0.759 | 0.995 | 181 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Disagree | 301 | 0.017 | 1.000 | 19 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test | Agree | 815 | 0.900 | 0.995 | 706 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test | Disagree | 664 | 0.000 |  | 0 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Agree | 257 | 0.793 | 0.995 | 195 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Disagree | 289 | 0.000 |  | 0 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test | Agree | 804 | 0.829 | 0.996 | 636 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test | Disagree | 675 | 0.156 | 1.000 | 233 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Agree | 245 | 0.738 | 0.995 | 174 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Disagree | 301 | 0.043 | 0.799 | 40 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test | Agree | 796 | 0.850 | 0.996 | 652 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test | Disagree | 683 | 0.020 | 1.000 | 31 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Agree | 245 | 0.742 | 0.997 | 171 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Disagree | 301 | 0.003 | 1.000 | 4 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Agree | 791 | 0.877 | 0.996 | 670 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Disagree | 688 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 265 | 0.823 | 0.991 | 212 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 281 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Agree | 836 | 0.871 | 0.995 | 697 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Disagree | 643 | 0.066 | 1.000 | 83 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 284 | 0.751 | 0.990 | 208 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 262 | 0.053 | 0.353 | 19 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Agree | 800 | 0.932 | 0.994 | 724 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Disagree | 679 | 0.073 | 0.904 | 107 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 292 | 0.876 | 0.990 | 252 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 254 | 0.144 | 0.369 | 47 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Agree | 806 | 0.942 | 0.994 | 732 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Disagree | 673 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 264 | 0.873 | 0.991 | 224 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 282 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test | Agree | 1249 | 0.675 | 0.979 | 796 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test | Disagree | 1006 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test_heldout | Agree | 376 | 0.545 | 0.952 | 172 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test_heldout | Disagree | 471 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test | Agree | 1351 | 0.387 | 0.981 | 471 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test | Disagree | 904 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test_heldout | Agree | 414 | 0.259 | 0.926 | 92 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test_heldout | Disagree | 433 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test | Agree | 1257 | 0.529 | 0.985 | 614 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test | Disagree | 998 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test_heldout | Agree | 391 | 0.376 | 0.944 | 122 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test_heldout | Disagree | 456 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test | Agree | 1263 | 0.528 | 0.985 | 613 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test | Disagree | 992 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test_heldout | Agree | 365 | 0.382 | 0.961 | 111 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test_heldout | Disagree | 482 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test | Agree | 1237 | 0.498 | 0.976 | 568 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test | Disagree | 1018 | 0.011 | 1.000 | 18 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 354 | 0.435 | 0.941 | 124 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 493 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test | Agree | 1335 | 0.736 | 0.965 | 914 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test | Disagree | 920 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 461 | 0.570 | 0.924 | 226 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 386 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test | Agree | 1300 | 0.506 | 0.973 | 604 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test | Disagree | 955 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 447 | 0.417 | 0.949 | 144 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 400 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test | Agree | 1285 | 0.174 | 0.980 | 201 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test | Disagree | 970 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 426 | 0.109 | 0.978 | 33 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 421 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test | Agree | 1216 | 0.548 | 0.985 | 622 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test | Disagree | 1039 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Agree | 364 | 0.451 | 0.957 | 136 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Disagree | 483 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test | Agree | 1244 | 0.517 | 0.985 | 593 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test | Disagree | 1011 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Agree | 365 | 0.397 | 0.976 | 108 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Disagree | 482 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test | Agree | 1244 | 0.627 | 0.985 | 727 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test | Disagree | 1011 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Agree | 317 | 0.500 | 0.968 | 131 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Disagree | 530 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test | Agree | 1199 | 0.647 | 0.983 | 725 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test | Disagree | 1056 | 0.004 | 1.000 | 6 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Agree | 339 | 0.511 | 0.954 | 141 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Disagree | 508 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test | Agree | 1232 | 0.632 | 0.980 | 722 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test | Disagree | 1023 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Agree | 343 | 0.496 | 0.963 | 138 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Disagree | 504 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test | Agree | 1304 | 0.489 | 0.986 | 573 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test | Disagree | 951 | 0.040 | 0.936 | 56 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Agree | 416 | 0.309 | 0.972 | 95 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Disagree | 431 | 0.003 | 1.000 | 3 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test | Agree | 1255 | 0.473 | 0.987 | 546 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test | Disagree | 1000 | 0.007 | 1.000 | 11 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Agree | 292 | 0.383 | 0.974 | 86 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Disagree | 555 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test | Agree | 1259 | 0.601 | 0.988 | 691 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test | Disagree | 996 | 0.007 | 1.000 | 10 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Agree | 315 | 0.467 | 0.963 | 121 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Disagree | 532 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test | Agree | 1270 | 0.608 | 0.988 | 702 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test | Disagree | 985 | 0.002 | 1.000 | 3 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Agree | 312 | 0.538 | 0.972 | 129 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Disagree | 535 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test | Agree | 1217 | 0.618 | 0.984 | 702 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test | Disagree | 1038 | 0.012 | 1.000 | 19 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Agree | 288 | 0.608 | 0.957 | 149 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Disagree | 559 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test | Agree | 1193 | 0.541 | 0.989 | 602 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test | Disagree | 1062 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Agree | 289 | 0.471 | 0.974 | 109 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Disagree | 558 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test | Agree | 1239 | 0.556 | 0.990 | 630 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test | Disagree | 1016 | 0.012 | 1.000 | 19 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Agree | 301 | 0.525 | 0.979 | 122 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Disagree | 546 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Agree | 1262 | 0.804 | 0.953 | 975 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Disagree | 993 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 345 | 0.710 | 0.929 | 221 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 502 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Agree | 1265 | 0.599 | 0.968 | 706 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Disagree | 990 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 316 | 0.516 | 0.944 | 144 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 531 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Agree | 1311 | 0.365 | 0.986 | 424 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Disagree | 944 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 331 | 0.249 | 0.962 | 68 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 516 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Agree | 1195 | 0.437 | 0.977 | 479 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Disagree | 1060 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 296 | 0.313 | 0.977 | 73 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 551 | 0.000 |  | 0 |

## prior rule `margin`

| type | arm | seed | lr | split | verdict | n | coverage | precision | gated n |
|---|---|---|---|---|---|---|---|---|---|
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test | Agree | 799 | 0.822 | 0.996 | 632 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test | Disagree | 680 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test_heldout | Agree | 276 | 0.648 | 0.991 | 171 |
| CurbRamp | `dinov2-large` | 42 | 1e-05 | test_heldout | Disagree | 270 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test | Agree | 786 | 0.900 | 0.993 | 688 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test | Disagree | 693 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test_heldout | Agree | 274 | 0.787 | 0.990 | 208 |
| CurbRamp | `dinov2-large` | 42 | 3e-05 | test_heldout | Disagree | 272 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test | Agree | 806 | 0.847 | 0.993 | 663 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test | Disagree | 673 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test_heldout | Agree | 262 | 0.723 | 0.994 | 179 |
| CurbRamp | `dinov2-large` | 43 | 1e-05 | test_heldout | Disagree | 284 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test | Agree | 817 | 0.818 | 0.994 | 644 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test | Disagree | 662 | 0.000 |  | 0 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test_heldout | Agree | 278 | 0.671 | 0.993 | 175 |
| CurbRamp | `dinov2-large` | 44 | 1e-05 | test_heldout | Disagree | 268 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test | Agree | 825 | 0.803 | 0.990 | 630 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test | Disagree | 654 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 293 | 0.635 | 0.979 | 183 |
| CurbRamp | `convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 253 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test | Agree | 828 | 0.818 | 0.990 | 642 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test | Disagree | 651 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 310 | 0.746 | 0.981 | 226 |
| CurbRamp | `convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 236 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test | Agree | 832 | 0.695 | 0.992 | 539 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test | Disagree | 647 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 302 | 0.522 | 0.984 | 153 |
| CurbRamp | `convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 244 | 0.000 |  | 0 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test | Agree | 770 | 0.773 | 0.993 | 562 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test | Disagree | 709 | 0.059 | 1.000 | 79 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 275 | 0.584 | 0.983 | 156 |
| CurbRamp | `convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 271 | 0.022 | 1.000 | 13 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test | Agree | 812 | 0.817 | 0.994 | 630 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test | Disagree | 667 | 0.014 | 1.000 | 15 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Agree | 302 | 0.533 | 0.992 | 151 |
| CurbRamp | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Disagree | 244 | 0.000 |  | 0 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test | Agree | 792 | 0.789 | 0.993 | 605 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test | Disagree | 687 | 0.072 | 0.922 | 74 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Agree | 261 | 0.644 | 0.990 | 161 |
| CurbRamp | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Disagree | 285 | 0.005 | 1.000 | 3 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test | Agree | 797 | 0.853 | 0.993 | 658 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test | Disagree | 682 | 0.043 | 1.000 | 46 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Agree | 265 | 0.706 | 0.991 | 178 |
| CurbRamp | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Disagree | 281 | 0.002 | 1.000 | 1 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test | Agree | 779 | 0.883 | 0.991 | 667 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test | Disagree | 700 | 0.053 | 1.000 | 65 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Agree | 270 | 0.722 | 0.986 | 189 |
| CurbRamp | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Disagree | 276 | 0.005 | 1.000 | 3 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test | Agree | 806 | 0.777 | 0.993 | 594 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test | Disagree | 673 | 0.090 | 1.000 | 100 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Agree | 244 | 0.691 | 0.990 | 161 |
| CurbRamp | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Disagree | 302 | 0.054 | 0.730 | 31 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test | Agree | 809 | 0.775 | 0.993 | 593 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test | Disagree | 670 | 0.087 | 1.000 | 98 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Agree | 275 | 0.563 | 0.989 | 148 |
| CurbRamp | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Disagree | 271 | 0.067 | 0.588 | 26 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test | Agree | 795 | 0.804 | 0.993 | 615 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test | Disagree | 684 | 0.019 | 1.000 | 21 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Agree | 236 | 0.680 | 0.993 | 156 |
| CurbRamp | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Disagree | 310 | 0.004 | 1.000 | 3 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test | Agree | 817 | 0.848 | 0.994 | 662 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test | Disagree | 662 | 0.186 | 0.967 | 183 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Agree | 263 | 0.688 | 0.990 | 176 |
| CurbRamp | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Disagree | 283 | 0.036 | 0.760 | 18 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test | Agree | 824 | 0.854 | 0.992 | 679 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test | Disagree | 655 | 0.076 | 1.000 | 73 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Agree | 245 | 0.757 | 0.990 | 181 |
| CurbRamp | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Disagree | 301 | 0.027 | 1.000 | 19 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test | Agree | 815 | 0.771 | 0.994 | 598 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test | Disagree | 664 | 0.000 |  | 0 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Agree | 257 | 0.655 | 0.993 | 160 |
| CurbRamp | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Disagree | 289 | 0.000 |  | 0 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test | Agree | 804 | 0.826 | 0.993 | 636 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test | Disagree | 675 | 0.214 | 1.000 | 233 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Agree | 245 | 0.735 | 0.991 | 174 |
| CurbRamp | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Disagree | 301 | 0.062 | 0.879 | 40 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test | Agree | 796 | 0.847 | 0.992 | 652 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test | Disagree | 683 | 0.028 | 1.000 | 31 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Agree | 245 | 0.738 | 0.994 | 171 |
| CurbRamp | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Disagree | 301 | 0.006 | 1.000 | 4 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Agree | 791 | 0.874 | 0.992 | 670 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Disagree | 688 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 265 | 0.821 | 0.984 | 212 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 281 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Agree | 836 | 0.868 | 0.990 | 697 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Disagree | 643 | 0.087 | 1.000 | 83 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 284 | 0.749 | 0.982 | 208 |
| CurbRamp | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 262 | 0.058 | 0.498 | 19 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Agree | 800 | 0.759 | 0.994 | 576 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Disagree | 679 | 0.098 | 0.945 | 107 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 292 | 0.607 | 0.989 | 170 |
| CurbRamp | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 254 | 0.154 | 0.515 | 47 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Agree | 806 | 0.869 | 0.991 | 671 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Disagree | 673 | 0.000 |  | 0 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 264 | 0.764 | 0.986 | 195 |
| CurbRamp | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 282 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test | Agree | 1249 | 0.278 | 0.957 | 336 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test | Disagree | 1006 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test_heldout | Agree | 376 | 0.195 | 0.921 | 63 |
| SurfaceProblem | `dinov2-large` | 42 | 1e-05 | test_heldout | Disagree | 471 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test | Agree | 1351 | 0.369 | 0.959 | 471 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test | Disagree | 904 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test_heldout | Agree | 414 | 0.246 | 0.849 | 92 |
| SurfaceProblem | `dinov2-large` | 42 | 3e-05 | test_heldout | Disagree | 433 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test | Agree | 1257 | 0.510 | 0.967 | 614 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test | Disagree | 998 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test_heldout | Agree | 391 | 0.355 | 0.885 | 122 |
| SurfaceProblem | `dinov2-large` | 43 | 1e-05 | test_heldout | Disagree | 456 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test | Agree | 1263 | 0.508 | 0.967 | 613 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test | Disagree | 992 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test_heldout | Agree | 365 | 0.357 | 0.917 | 111 |
| SurfaceProblem | `dinov2-large` | 44 | 1e-05 | test_heldout | Disagree | 482 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test | Agree | 1237 | 0.480 | 0.948 | 568 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test | Disagree | 1018 | 0.101 | 0.972 | 119 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 354 | 0.407 | 0.878 | 124 |
| SurfaceProblem | `convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 493 | 0.014 | 0.524 | 8 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test | Agree | 1335 | 0.171 | 0.941 | 218 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test | Disagree | 920 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 461 | 0.077 | 0.870 | 31 |
| SurfaceProblem | `convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 386 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test | Agree | 1300 | 0.155 | 0.971 | 189 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test | Disagree | 955 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 447 | 0.115 | 0.960 | 39 |
| SurfaceProblem | `convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 400 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test | Agree | 1285 | 0.166 | 0.958 | 201 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test | Disagree | 970 | 0.000 |  | 0 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 426 | 0.098 | 0.952 | 33 |
| SurfaceProblem | `convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 421 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test | Agree | 1216 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test | Disagree | 1039 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Agree | 364 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 42 | 1e-05 | test_heldout | Disagree | 483 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test | Agree | 1244 | 0.425 | 0.978 | 502 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test | Disagree | 1011 | 0.038 | 1.000 | 44 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Agree | 365 | 0.284 | 0.957 | 83 |
| SurfaceProblem | `dinov2-reg-large` | 43 | 1e-05 | test_heldout | Disagree | 482 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test | Agree | 1244 | 0.413 | 0.977 | 491 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test | Disagree | 1011 | 0.000 |  | 0 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Agree | 317 | 0.294 | 0.935 | 81 |
| SurfaceProblem | `dinov2-reg-large` | 44 | 1e-05 | test_heldout | Disagree | 530 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test | Agree | 1199 | 0.338 | 0.984 | 385 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test | Disagree | 1056 | 0.070 | 0.960 | 84 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Agree | 339 | 0.241 | 0.958 | 65 |
| SurfaceProblem | `dinov3-vitb16` | 42 | 1e-05 | test_heldout | Disagree | 508 | 0.001 | 1.000 | 1 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test | Agree | 1232 | 0.179 | 0.991 | 207 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test | Disagree | 1023 | 0.046 | 0.969 | 54 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Agree | 343 | 0.092 | 1.000 | 24 |
| SurfaceProblem | `dinov3-vitb16` | 43 | 1e-05 | test_heldout | Disagree | 504 | 0.006 | 0.423 | 3 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test | Agree | 1304 | 0.156 | 0.994 | 189 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test | Disagree | 951 | 0.052 | 0.970 | 56 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Agree | 416 | 0.096 | 0.988 | 30 |
| SurfaceProblem | `dinov3-vitb16` | 44 | 1e-05 | test_heldout | Disagree | 431 | 0.005 | 1.000 | 3 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test | Agree | 1255 | 0.370 | 0.978 | 442 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test | Disagree | 1000 | 0.126 | 0.953 | 140 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Agree | 292 | 0.275 | 0.970 | 64 |
| SurfaceProblem | `dinov3-vitl16` | 42 | 1e-05 | test_heldout | Disagree | 555 | 0.010 | 0.687 | 7 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test | Agree | 1259 | 0.495 | 0.977 | 592 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test | Disagree | 996 | 0.135 | 0.978 | 151 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Agree | 315 | 0.369 | 0.942 | 98 |
| SurfaceProblem | `dinov3-vitl16` | 43 | 1e-05 | test_heldout | Disagree | 532 | 0.020 | 0.668 | 13 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test | Agree | 1270 | 0.376 | 0.987 | 450 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test | Disagree | 985 | 0.044 | 0.966 | 49 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Agree | 312 | 0.332 | 0.976 | 81 |
| SurfaceProblem | `dinov3-vitl16` | 44 | 1e-05 | test_heldout | Disagree | 535 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test | Agree | 1217 | 0.246 | 0.969 | 288 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test | Disagree | 1038 | 0.016 | 1.000 | 19 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Agree | 288 | 0.294 | 0.966 | 69 |
| SurfaceProblem | `dinov3-vitl16-256` | 42 | 1e-05 | test_heldout | Disagree | 559 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test | Agree | 1193 | 0.524 | 0.976 | 602 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test | Disagree | 1062 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Agree | 289 | 0.442 | 0.945 | 109 |
| SurfaceProblem | `dinov3-vitl16-256` | 43 | 1e-05 | test_heldout | Disagree | 558 | 0.000 |  | 0 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test | Agree | 1239 | 0.422 | 0.981 | 497 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test | Disagree | 1016 | 0.107 | 0.959 | 122 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Agree | 301 | 0.353 | 0.964 | 87 |
| SurfaceProblem | `dinov3-vitl16-256` | 44 | 1e-05 | test_heldout | Disagree | 546 | 0.016 | 0.801 | 12 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Agree | 1262 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test | Disagree | 993 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Agree | 345 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 1e-05 | test_heldout | Disagree | 502 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Agree | 1265 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test | Disagree | 990 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Agree | 316 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 42 | 3e-05 | test_heldout | Disagree | 531 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Agree | 1311 | 0.223 | 0.987 | 271 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test | Disagree | 944 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Agree | 331 | 0.101 | 0.972 | 27 |
| SurfaceProblem | `rampnet-convnextv2-base` | 43 | 1e-05 | test_heldout | Disagree | 516 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Agree | 1195 | 0.240 | 0.969 | 271 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test | Disagree | 1060 | 0.000 |  | 0 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Agree | 296 | 0.166 | 0.971 | 40 |
| SurfaceProblem | `rampnet-convnextv2-base` | 44 | 1e-05 | test_heldout | Disagree | 551 | 0.000 |  | 0 |
