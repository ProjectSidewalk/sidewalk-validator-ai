# Deployment prior under four vote rules (issue #56)

Human-authored, geometry-valid labels of the cached rawLabels (54 cities). Cell: incorrect share (correct / incorrect counts).

| Rule | Crosswalk | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem |
|---|---:|---:|---:|---:|---:|
| margin | 6.1% (19,671 / 1,281) | 11.8% (28,470 / 3,797) | 37.4% (13,030 / 7,774) | 35.8% (12,247 / 6,822) | 32.0% (13,790 / 6,484) |
| symmetric3 | 3.8% (19,671 / 771) | 4.2% (28,470 / 1,255) | 26.9% (13,030 / 4,790) | 15.8% (12,247 / 2,304) | 11.5% (13,790 / 1,792) |
| symmetric2 | 3.8% (32,628 / 1,281) | 5.2% (68,886 / 3,797) | 20.7% (29,864 / 7,774) | 22.1% (24,068 / 6,822) | 14.5% (38,080 / 6,484) |
| majority2 | 4.4% (33,136 / 1,524) | 6.8% (72,228 / 5,291) | 22.5% (32,677 / 9,479) | 25.4% (28,084 / 9,555) | 17.5% (42,761 / 9,101) |

Hard check against the deployment_priors.json margin counts: Crosswalk ok, CurbRamp ok, NoCurbRamp ok, Obstacle ok, SurfaceProblem ok

## Crosswalk, incorrect share by severity

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| severity | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| 1 | 97,845 | 0.023 (18,849) | 0.012 (18,643) | 0.014 (30,565) | 0.018 (31,103) |
| 2 | 10,785 | 0.251 (880) | 0.170 (794) | 0.144 (1,538) | 0.158 (1,602) |
| 3 | 4,777 | 0.757 (618) | 0.679 (468) | 0.567 (826) | 0.577 (925) |
| (none) | 7,559 | 0.260 (605) | 0.166 (537) | 0.160 (980) | 0.170 (1,030) |

## Crosswalk, incorrect share by tag

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| tag | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| (none) | 87,991 | 0.050 (16,558) | 0.033 (16,263) | 0.031 (26,580) | 0.036 (27,082) |
| paint fading | 24,007 | 0.101 (2,780) | 0.052 (2,637) | 0.058 (4,825) | 0.067 (4,988) |
| very long crossing | 5,161 | 0.063 (663) | 0.039 (646) | 0.038 (1,109) | 0.045 (1,135) |
| broken surface | 3,203 | 0.056 (680) | 0.039 (668) | 0.037 (1,026) | 0.041 (1,053) |
| bumpy | 2,022 | 0.117 (420) | 0.077 (402) | 0.085 (578) | 0.095 (599) |
| uneven surface | 1,887 | 0.179 (341) | 0.103 (312) | 0.107 (569) | 0.119 (599) |
| brick/cobblestone | 1,795 | 0.383 (141) | 0.163 (104) | 0.171 (316) | 0.183 (333) |
| too close to traffic | 1,078 | 0.013 (239) | 0.000 (236) | 0.009 (320) | 0.018 (325) |
| no pedestrian priority | 846 | 0.149 (241) | 0.101 (228) | 0.115 (313) | 0.131 (327) |
| rail/tram track | 559 | 0.065 (155) | 0.046 (152) | 0.040 (247) | 0.039 (257) |
| level with sidewalk | 55 |  (0) |  (0) |  (3) |  (3) |

## CurbRamp, incorrect share by severity

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| severity | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| 1 | 410,819 | 0.091 (26,987) | 0.031 (25,337) | 0.040 (61,395) | 0.054 (64,917) |
| 2 | 43,104 | 0.199 (3,002) | 0.070 (2,588) | 0.088 (6,777) | 0.110 (7,508) |
| 3 | 13,826 | 0.350 (1,414) | 0.161 (1,095) | 0.186 (2,666) | 0.216 (3,099) |
| (none) | 38,024 | 0.303 (864) | 0.146 (705) | 0.142 (1,845) | 0.162 (1,995) |

## CurbRamp, incorrect share by tag

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| tag | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| (none) | 387,217 | 0.093 (20,009) | 0.031 (18,735) | 0.040 (46,485) | 0.052 (48,903) |
| missing tactile warning | 61,715 | 0.134 (6,617) | 0.044 (5,995) | 0.061 (14,494) | 0.082 (15,883) |
| points into traffic | 33,567 | 0.142 (2,919) | 0.064 (2,675) | 0.064 (6,436) | 0.082 (6,873) |
| narrow | 19,182 | 0.155 (1,850) | 0.055 (1,655) | 0.072 (3,988) | 0.098 (4,433) |
| surface problem | 14,983 | 0.124 (1,884) | 0.053 (1,742) | 0.063 (3,743) | 0.083 (4,112) |
| not level with street | 14,361 | 0.318 (1,558) | 0.127 (1,218) | 0.186 (2,662) | 0.223 (3,073) |
| not enough landing space | 13,050 | 0.157 (1,380) | 0.059 (1,237) | 0.068 (3,159) | 0.095 (3,487) |
| debris / pooled water | 4,855 | 0.049 (390) | 0.013 (376) | 0.019 (1,011) | 0.027 (1,076) |
| steep | 4,673 | 0.367 (668) | 0.193 (524) | 0.232 (1,058) | 0.259 (1,243) |
| not aligned with crosswalk | 1,297 | 0.020 (306) | 0.007 (302) | 0.015 (389) | 0.027 (401) |
| tactile warning | 1,144 |  (2) |  (2) | 0.000 (34) | 0.000 (34) |
| parallel lines | 792 |  (17) |  (16) | 0.007 (149) | 0.007 (149) |
| not visible | 5 |  (0) |  (0) |  (0) |  (0) |

## NoCurbRamp, incorrect share by severity

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| severity | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| 1 | 13,835 | 0.599 (2,946) | 0.497 (2,349) | 0.414 (4,256) | 0.420 (4,968) |
| 2 | 39,998 | 0.366 (5,883) | 0.246 (4,946) | 0.185 (11,665) | 0.204 (13,089) |
| 3 | 64,860 | 0.294 (10,955) | 0.210 (9,793) | 0.160 (20,182) | 0.177 (22,287) |
| (none) | 11,057 | 0.624 (1,020) | 0.475 (732) | 0.414 (1,535) | 0.422 (1,812) |

## NoCurbRamp, incorrect share by tag

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| tag | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| (none) | 54,570 | 0.442 (7,744) | 0.338 (6,529) | 0.262 (13,072) | 0.279 (14,641) |
| no alternate route | 53,450 | 0.237 (8,131) | 0.153 (7,329) | 0.116 (16,604) | 0.134 (18,195) |
| alternate route present | 18,182 | 0.449 (3,932) | 0.318 (3,178) | 0.267 (6,609) | 0.280 (7,691) |
| unclear if needed | 5,549 | 0.643 (1,405) | 0.532 (1,071) | 0.458 (1,973) | 0.462 (2,374) |

## Obstacle, incorrect share by severity

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| severity | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| 1 | 51,580 | 0.533 (5,763) | 0.283 (3,756) | 0.363 (8,468) | 0.376 (10,655) |
| 2 | 58,170 | 0.289 (5,149) | 0.117 (4,145) | 0.172 (8,658) | 0.207 (10,494) |
| 3 | 81,800 | 0.234 (7,010) | 0.086 (5,877) | 0.137 (11,977) | 0.178 (14,201) |
| (none) | 11,812 | 0.541 (1,147) | 0.318 (773) | 0.347 (1,787) | 0.368 (2,289) |

## Obstacle, incorrect share by tag

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| tag | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| pole | 49,256 | 0.394 (3,713) | 0.178 (2,738) | 0.229 (6,401) | 0.261 (7,838) |
| (none) | 46,022 | 0.503 (3,067) | 0.257 (2,052) | 0.333 (4,628) | 0.358 (5,881) |
| tree | 26,730 | 0.436 (1,405) | 0.207 (999) | 0.250 (2,453) | 0.279 (2,898) |
| vegetation | 17,800 | 0.298 (1,970) | 0.125 (1,581) | 0.179 (3,275) | 0.218 (4,060) |
| parked car | 15,107 | 0.332 (2,256) | 0.148 (1,769) | 0.211 (3,538) | 0.260 (4,457) |
| narrow | 13,194 | 0.206 (1,712) | 0.087 (1,489) | 0.121 (2,922) | 0.148 (3,409) |
| trash/recycling can | 12,173 | 0.219 (2,088) | 0.103 (1,818) | 0.143 (3,205) | 0.175 (3,802) |
| sign | 9,779 | 0.396 (956) | 0.153 (681) | 0.243 (1,562) | 0.267 (1,893) |
| garage entrance | 8,969 | 0.573 (117) | 0.194 (62) | 0.344 (195) | 0.364 (236) |
| parked scooter/motorcycle | 7,456 | 0.200 (1,066) | 0.057 (905) | 0.118 (1,809) | 0.144 (2,117) |
| construction | 4,696 | 0.151 (879) | 0.045 (781) | 0.091 (1,458) | 0.117 (1,654) |
| litter/garbage | 3,596 | 0.256 (437) | 0.107 (364) | 0.149 (752) | 0.176 (879) |
| stairs | 3,565 | 0.454 (119) | 0.177 (79) | 0.219 (247) | 0.228 (272) |
| height difference | 3,363 | 0.500 (100) | 0.138 (58) | 0.294 (170) | 0.314 (210) |
| street vendor | 3,277 | 0.096 (218) | 0.015 (200) | 0.047 (451) | 0.083 (508) |
| fire hydrant | 3,070 | 0.394 (592) | 0.206 (452) | 0.262 (890) | 0.296 (1,118) |
| parked bike | 2,722 | 0.403 (419) | 0.206 (315) | 0.278 (608) | 0.305 (729) |
| utility cabinet | 2,111 | 0.042 (474) | 0.009 (458) | 0.026 (774) | 0.041 (823) |
| outdoor dining area | 331 | 0.895 (133) | 0.725 (51) | 0.763 (156) | 0.764 (182) |
| mailbox | 237 | 0.429 (91) | 0.288 (73) | 0.336 (116) | 0.336 (128) |
| drainage | 171 |  (4) |  (0) |  (14) |  (15) |
| cart | 14 |  (0) |  (0) |  (1) |  (1) |
| electrical box | 8 |  (0) |  (0) |  (1) |  (1) |

## SurfaceProblem, incorrect share by severity

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| severity | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| 1 | 101,458 | 0.351 (10,139) | 0.124 (7,513) | 0.163 (21,847) | 0.189 (25,590) |
| 2 | 64,453 | 0.279 (5,804) | 0.106 (4,682) | 0.122 (13,217) | 0.160 (15,456) |
| 3 | 43,388 | 0.243 (3,215) | 0.068 (2,613) | 0.103 (7,545) | 0.131 (8,476) |
| (none) | 9,562 | 0.475 (1,116) | 0.243 (774) | 0.271 (1,955) | 0.297 (2,340) |

## SurfaceProblem, incorrect share by tag

Blank under 30 decided labels. n = labels in the slice; n_rule = decided under the rule.

| tag | n | margin (n) | symmetric3 (n) | symmetric2 (n) | majority2 (n) |
|---|---:|---:|---:|---:|---:|
| cracks | 75,469 | 0.164 (6,053) | 0.040 (5,274) | 0.066 (14,968) | 0.087 (16,620) |
| grass | 59,612 | 0.179 (5,258) | 0.033 (4,461) | 0.076 (12,390) | 0.095 (13,831) |
| bumpy | 32,652 | 0.263 (3,964) | 0.089 (3,208) | 0.130 (8,039) | 0.160 (9,336) |
| uneven/slanted | 31,182 | 0.200 (2,887) | 0.052 (2,437) | 0.077 (7,465) | 0.104 (8,370) |
| narrow sidewalk | 26,526 | 0.379 (2,163) | 0.128 (1,541) | 0.157 (5,202) | 0.210 (6,641) |
| (none) | 24,782 | 0.590 (2,371) | 0.342 (1,476) | 0.333 (4,208) | 0.354 (4,868) |
| height difference | 21,035 | 0.129 (1,560) | 0.041 (1,417) | 0.059 (3,423) | 0.083 (3,805) |
| very broken | 15,865 | 0.112 (1,326) | 0.048 (1,237) | 0.056 (2,650) | 0.083 (2,919) |
| utility panel | 15,705 | 0.434 (1,855) | 0.195 (1,304) | 0.245 (3,285) | 0.269 (4,183) |
| sand/gravel | 6,273 | 0.226 (619) | 0.101 (533) | 0.107 (1,313) | 0.129 (1,491) |
| brick/cobblestone | 5,722 | 0.547 (982) | 0.331 (665) | 0.360 (1,492) | 0.377 (1,965) |
| debris | 2,923 | 0.136 (221) | 0.035 (198) | 0.062 (481) | 0.093 (527) |
| construction | 1,182 | 0.246 (122) | 0.107 (103) | 0.090 (335) | 0.132 (394) |
| rail/tram track | 637 | 0.138 (116) | 0.038 (104) | 0.082 (196) | 0.103 (213) |
| uncovered manhole | 122 |  (0) |  (0) |  (1) |  (1) |

