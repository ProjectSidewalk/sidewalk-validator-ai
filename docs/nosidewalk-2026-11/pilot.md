| arm | split | runs | rows (incorrect) | AUC | within-city AUC | Agree gate coverage | Agree gate precision |
|---|---|---|---|---|---|---|---|
| depth | test | 3 | 110 (46) | 0.878 (0.873–0.886) | 0.905 (0.901–0.912) | 0.000 | n/a (map) |
| depth | test_heldout | 3 | 127 (80) | 0.778 (0.758–0.803) | 0.779 (0.757–0.803) | 0.000 | n/a (map) |
| whole | test | 3 | 110 (46) | 0.916 (0.899–0.932) | 0.931 (0.925–0.939) | 0.000 | n/a (map) |
| whole | test_heldout | 3 | 127 (80) | 0.802 (0.760–0.840) | 0.786 (0.741–0.823) | 0.000 | n/a (map) |

Per run, AUC with a 95 % interval from resampling block faces; Agree gate precision with its Wilson 95 % lower bound:

| arm | seed | split | AUC [95 % CI] | gate n | gate precision (lower) |
|---|---|---|---|---|---|
| depth | 42 | test | 0.886 [0.713, 0.993] | 0 | n/a |
| depth | 42 | test_heldout | 0.758 [0.647, 0.862] | 0 | n/a |
| depth | 43 | test | 0.873 [0.684, 0.982] | 0 | n/a |
| depth | 43 | test_heldout | 0.803 [0.692, 0.910] | 0 | n/a |
| depth | 44 | test | 0.876 [0.694, 0.993] | 0 | n/a |
| depth | 44 | test_heldout | 0.772 [0.668, 0.871] | 0 | n/a |
| whole | 42 | test | 0.899 [0.732, 0.998] | 0 | n/a |
| whole | 42 | test_heldout | 0.805 [0.701, 0.895] | 0 | n/a |
| whole | 43 | test | 0.932 [0.791, 1.000] | 0 | n/a |
| whole | 43 | test_heldout | 0.760 [0.647, 0.862] | 0 | n/a |
| whole | 44 | test | 0.915 [0.769, 1.000] | 0 | n/a |
| whole | 44 | test_heldout | 0.840 [0.759, 0.921] | 0 | n/a |
