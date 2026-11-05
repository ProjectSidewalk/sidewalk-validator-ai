<!-- analysis/ai_labels_relabel.py score, 2026-10-28 -->

| cell | n | voted | Unsure | rater sides with validator | rater sides with original human vote |
|---|---:|---:|---:|---|---|
| contested | 31 | 31 | 1 | 4/30 = 13.3% [5%, 30%] | 26/30 = 86.7% [70%, 95%] |
| confirmed | 30 | 30 | 1 | 27/29 = 93.1% [78%, 98%] | 27/29 = 93.1% [78%, 98%] |
| missed | 22 | 22 | 1 | 15/21 = 71.4% [50%, 86%] | 6/21 = 28.6% [14%, 50%] |
| decoy | 50 | 49 | 0 | 48/49 = 98.0% [89%, 100%] | 48/49 = 98.0% [89%, 100%] |

Overall (Unsure excluded): rater sides with the original vote 107/129 = 82.9% [76%, 88%], with the validator 94/129 = 72.9% [65%, 80%].
Validator Disagree precision against the rater, packet only (contested + confirmed): 31/59 = 52.5% [40%, 65%]; by p_incorrect bin: < 0.99: 2/14; 0.99–0.999: 6/13; > 0.999: 23/32.
Including the 28 validator Disagrees the packet left out because the rater had already voted on them ({'Agree': 25, 'Disagree': 2, 'Unsure': 1}): Disagree precision against the rater over all 89 Disagrees (86 decided) 33/86 = 38.4% [29%, 49%]; by bin: < 0.99: 3/25; 0.99–0.999: 6/18; > 0.999: 24/43.
Read-back check: 132 labels gained exactly one vote by the rater, 1 unchanged, nothing else changed.
