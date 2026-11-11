# Observed: AI Agree votes at served accuracy >= 0.92, confirmed by the human majority

Production pull `ai_assessments_2026-11-11.csv` (sha256 `6891a1c150685975`), 53 schemas, AI dates 2025-08-13 .. 2026-11-11. Round 3 = assessments dated 2026-10-28 or later; older = before 2026-10-27; 2026-10-27 itself (the changeover day) is in neither. Cell: share confirmed [95% Wilson interval] (decided n); share blank under 30 decided.

## round3

| Type | gated Agree | with a human vote | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|---:|---:|
| Crosswalk | 1,911 | 1,147 | 0.998 [0.991, 1.000] (630) | 1.000 [0.994, 1.000] (629) | 0.999 [0.993, 1.000] (864) | 0.998 [0.992, 0.999] (885) |
| CurbRamp | 10,980 | 3,629 | 0.994 [0.964, 0.999] (154) | 1.000 [0.976, 1.000] (153) | 0.999 [0.993, 1.000] (821) | 0.998 [0.991, 0.999] (834) |
| NoCurbRamp | 511 | 411 | (21) | (21) | 1.000 [0.969, 1.000] (122) | 1.000 [0.971, 1.000] (129) |
| Obstacle | 799 | 400 | 0.974 [0.911, 0.993] (78) | 1.000 [0.952, 1.000] (76) | 0.989 [0.960, 0.997] (179) | 0.945 [0.904, 0.969] (199) |
| SurfaceProblem | 1,883 | 736 | 0.962 [0.872, 0.990] (53) | 1.000 [0.930, 1.000] (51) | 0.991 [0.966, 0.997] (212) | 0.987 [0.962, 0.995] (226) |

## older

| Type | gated Agree | with a human vote | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|---:|---:|
| Crosswalk | 80,739 | 43,360 | 0.997 [0.996, 0.997] (14,059) | 0.999 [0.998, 0.999] (14,025) | 0.998 [0.997, 0.998] (22,799) | 0.996 [0.995, 0.997] (23,095) |
| CurbRamp | 278,781 | 161,365 | 0.990 [0.988, 0.991] (24,375) | 0.998 [0.998, 0.999] (24,166) | 0.996 [0.995, 0.996] (58,616) | 0.992 [0.991, 0.992] (60,944) |
| NoCurbRamp | 74,014 | 55,538 | 0.938 [0.933, 0.942] (12,534) | 0.976 [0.973, 0.979] (12,039) | 0.969 [0.967, 0.971] (25,273) | 0.954 [0.951, 0.956] (28,323) |
| Obstacle | 7 | 5 | (3) | (3) | (4) | (4) |
| SurfaceProblem | 95,982 | 57,765 | 0.951 [0.947, 0.956] (8,899) | 0.994 [0.992, 0.995] (8,522) | 0.979 [0.977, 0.981] (20,877) | 0.965 [0.962, 0.967] (22,970) |

## Round 3 by severity (majority2)

| Type | severity | gated Agree | confirmed (decided) |
|---|---|---:|---:|
| Crosswalk | (none) | 34 |  (20) |
| Crosswalk | 1 | 1,664 | 0.999 (812) |
| Crosswalk | 2 | 180 | 0.976 (42) |
| Crosswalk | 3 | 33 |  (11) |
| CurbRamp | (none) | 854 |  (28) |
| CurbRamp | 1 | 8,998 | 0.999 (747) |
| CurbRamp | 2 | 956 | 1.000 (50) |
| CurbRamp | 3 | 172 |  (9) |
| NoCurbRamp | (none) | 26 |  (4) |
| NoCurbRamp | 1 | 71 |  (19) |
| NoCurbRamp | 2 | 139 | 1.000 (39) |
| NoCurbRamp | 3 | 275 | 1.000 (67) |
| Obstacle | (none) | 60 |  (12) |
| Obstacle | 1 | 190 | 0.946 (56) |
| Obstacle | 2 | 286 | 0.942 (52) |
| Obstacle | 3 | 263 | 0.937 (79) |
| SurfaceProblem | (none) | 93 |  (11) |
| SurfaceProblem | 1 | 914 | 0.971 (105) |
| SurfaceProblem | 2 | 622 | 1.000 (70) |
| SurfaceProblem | 3 | 254 | 1.000 (40) |

## Round 3 replayed through each prior's map (Agree verdicts)

The round-3 production Agree confidences served through the val-fitted, uncapped Agree map of each prior: how many would clear 0.92, and the share of those the human majority2 rule confirms. served = share of rows where the margin map's lookup equals the served accuracy (checks the replay and the round-3 date split).

| Type | Agree rows | served | margin: gated, confirmed | majority2: gated, confirmed |
|---|---:|---:|---:|---:|
| Crosswalk | 1,957 | 1.000 | 1,911, 0.998 [0.992, 0.999] (885) | 1,911, 0.998 [0.992, 0.999] (885) |
| CurbRamp | 11,632 | 1.000 | 10,980, 0.998 [0.991, 0.999] (834) | 10,980, 0.998 [0.991, 0.999] (834) |
| NoCurbRamp | 1,587 | 1.000 | 511, 1.000 [0.971, 1.000] (129) | 987, 0.996 [0.979, 0.999] (265) |
| Obstacle | 1,800 | 1.000 | 799, 0.945 [0.904, 0.969] (199) | 799, 0.945 [0.904, 0.969] (199) |
| SurfaceProblem | 2,589 | 1.000 | 1,883, 0.987 [0.962, 0.995] (226) | 1,883, 0.987 [0.962, 0.995] (226) |

## Every AI-assessed label of the pull: incorrect share under each rule

| Type | margin | symmetric3 | symmetric2 | majority2 |
|---|---:|---:|---:|---:|
| Crosswalk | 6.1% (21,048) | 3.8% (20,535) | 3.8% (34,158) | 4.4% (34,923) |
| CurbRamp | 12.0% (33,698) | 4.4% (31,004) | 5.3% (76,291) | 6.9% (81,575) |
| NoCurbRamp | 35.8% (22,147) | 25.7% (19,120) | 20.8% (38,103) | 22.6% (43,999) |
| Obstacle | 37.2% (19,561) | 17.3% (14,860) | 22.7% (32,087) | 25.8% (39,105) |
| SurfaceProblem | 32.2% (21,263) | 12.2% (16,419) | 15.0% (45,756) | 18.2% (54,202) |
