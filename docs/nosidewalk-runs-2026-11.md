# NoSidewalk runs: validate the run, not the label? (November 2026)

**Status: exploratory analysis for
[#102](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/102); nothing here changes serving.** The
step (c) decision of 2026-11-14 on [`nosidewalk-2026-11.md`](nosidewalk-2026-11.md) (serve the five fold models as a
per-label ensemble, one masked round per deployment) stands whatever this page finds.

NoSidewalk labels are usually placed in sequence: one labeler walks a stretch without a sidewalk and drops a label
every panorama or so. A **run** here is one labeler's NoSidewalk labels on one block face, `(city, street_edge_id,
street_side, user_id)`, contiguous or not. The validator scores each label alone. #102 asks four things:

1. **Do the labels of a run stand or fall together?** Measured on existing expert votes, descriptive only.
2. **Does pooling the image score over the run or the face beat the single-label score?**
3. **Do neighbour features that need no image** (run length, labelers on the face, distance to the nearest
   NoSidewalk label, CurbRamp / Obstacle / SurfaceProblem labels on the same face) **add to the image score?**
4. **Should serving vote per run?** A recommendation under a rule fixed before the pooled results were computed.

Code: [`analysis/nosidewalk_runs.py`](../analysis/nosidewalk_runs.py) (`step-a-neighbours`, `agreement`,
`pooling`, `features`, `report`) and [`analysis/pull_expert_vote_times.sh`](../analysis/pull_expert_vote_times.sh).

## Units and rules

- Labels are keyed `city:label_id` (label ids are per city), faces `city:street_edge_id:side`. Faces and runs of a
  label come only from the pull that defined them: between the 2026-10-18 and 2026-11-05 pulls `street_edge_id`
  changed for about a tenth of NoSidewalk labels, so step (a) uses the 2026-11-05 pull (it reproduces every split
  row's face), step (b) the 2026-11-11 pulls, and the type comparison of item 1 the 2026-10-18 all-type pull.
- **A label's verdict in item 1 is its own expert vote**: every expert Agree / Disagree on that label points the same
  way; crowd votes are ignored (the stricter `data_generation.expert_verdict`, where one contradicting crowd vote
  drops the label, is a sensitivity). Never `label_class`: the face rule copies one verdict across a face, which
  would make agreement 1 by construction. Labels placed by an expert or by SidewalkAI are left out and counted.
- Expert and labeler identities stay in memory. Outputs carry a per-city `labeler_idx` and the experts' positions
  (1, 2). Label-level files stay outside the repo, in `~/Data/sidewalk-validator-ai-nosidewalk-2026-11/runs/`.
- Deployments that are not public in the `/v3/api/cities` snapshot of the run day are dropped before any statistic.
- **Every pooled score has its own map.** A pooled score is the mean logit of the per-label `p_correct` over the
  group, and averaging compresses it toward the group mean, so a served accuracy read off the single-label map no
  longer means what it says. Each group definition gets a map fitted on step (a)'s pooled out-of-fold scores of the
  same definition (leave-one-fold-out within step (a); all folds for step (b)); `serve_pooled` refuses any other map.

**What was seen before the rule of item 4 was written.** The planner and the plan's reviewer had computed item 1's
per-city agreements and references and the expert-vote timing in Seattle, Chicago and Oradell. Nobody had computed
any pooled AUC or pooled gate on step (a) or step (b) when the rule was written. Since then, before the step (a)
scores existed, the implementer's smoke run (placeholder step (a) scores) and the code review both produced the
step (b) **vote-pooled** rows, which do not depend on step (a) scores, so the implementer and the reviewer have seen
those rows. They are not the primary gate (score pooling over the run), and the rule was fixed before them. The
smoke run's score-pooled rows went through maps fitted on placeholder scores, so they are not the numbers below.

## Results

<!-- nosidewalk-runs-results:start -->
**Not yet computed: item 2 (pooled scores), item 4 (the recommendation); item 3 (neighbour features).** They need the step (a) labels re-scored with the kept gate2 weights on klone; until those scores are copied back and pinned, this page reports item 1 and the run census only, and no statement below is about pooling or neighbour features.

#### Item 1: do the labels of a run get the same expert verdict?

Descriptive only. NoSidewalk on the 2026-11-05 pull; a label's verdict is its **own** expert votes (every expert vote on it the same way, crowd ignored): 2,400 labels (19,835 placed by an expert and 0 by SidewalkAI left out, 1 where the experts split, 119 without a face). Pair agreement is the share of within-run label pairs with the same verdict, with a 95 % CI from resampling runs within city. Null 1 permutes verdicts within the city, null 2 within the labeler (across their faces); p is one-sided over 1,000 permutations.

| city | labels | runs of 2+ | pairs | pair agreement [95 % CI] | all-same runs | ICC | null 1: city (p) | null 2: labeler (p) | same labeler, other face | same face, other labeler | any two in the city |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seattle-wa | 700 | 77 | 368 | 0.984 [0.955, 1.000] | 0.961 | 0.955 | 0.636 (p 0.001) | 0.944 (p 0.002) | 0.941 | 0.960 | 0.635 |
| chicago-il | 215 | 22 | 300 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.532 (p 0.001) | 0.804 (p 0.001) | 0.726 | 1.000 | 0.531 |
| oradell-nj | 1,076 | 142 | 6,570 | 0.989 [0.969, 0.999] | 0.965 | 0.641 | 0.959 (p 0.001) | 0.970 (p 0.001) | 0.926 | 0.942 | 0.958 |
| other (24 cities, pooled) | 290 | 40 | 84 | 0.952 [0.868, 1.000] | 0.975 | n/a | 0.915 (p 0.157) | 0.934 (p 0.299) | 0.925 | 0.889 | 0.928 |
| **macro mean**, 3 cities with >= 10 runs of 2+ (chicago-il, oradell-nj, seattle-wa) | 1,991 | | 7,238 | 0.991 [0.979, 0.998] | 0.975 | 0.865 | 0.709 (p 0.001) | 0.906 (p 0.001) | 0.864 | 0.967 | 0.708 |

Within-run pairs by the time between the expert's two votes, and by the page they were cast on (`pull_expert_vote_times.sh`, each expert's latest vote before the pull; cells under 30 pairs blank). Of the pull's expert votes on these labels 2,289 match a timed vote, 0 were changed around the pull and 2 have no timed vote; a pair with either is "time unknown".

| scope | different expert | same expert, < 1 h | same expert, 1 h - 1 day | same expert, >= 1 day | same expert, time unknown |
|---|---|---|---|---|---|
| all cities (pair-weighted pool, not a macro mean; 90% of them oradell-nj) | 7: n/a | 2,188: 0.984 | 176: 1.000 | 4,950: 0.991 | 1: n/a |
| seattle-wa | 0 | 360: 0.983 | 0 | 8: n/a | 0 |
| chicago-il | 0 | 299: 1.000 | 0 | 0 | 1: n/a |
| oradell-nj | 1: n/a | 1,458: 0.983 | 176: 1.000 | 4,935: 0.990 | 0 |

| scope (same-expert pairs) | AdminUserDashboard | Gallery | LabelMap | Old data | Validate | mixed | other | unknown |
|---|---|---|---|---|---|---|---|---|
| all cities (pair-weighted pool, not a macro mean; 90% of them oradell-nj) | 77: 0.922 | 11: n/a | 275: 1.000 | 364: 0.989 | 6,554: 0.989 | 32: 1.000 | 1: n/a | 1: n/a |
| seattle-wa | 19: n/a | 0 | 0 | 340: 0.988 | 4: n/a | 5: n/a | 0 | 0 |
| chicago-il | 19: n/a | 1: n/a | 275: 1.000 | 3: n/a | 0 | 0 | 1: n/a | 1: n/a |
| oradell-nj | 0 | 0 | 0 | 12: n/a | 6,530: 0.989 | 27: n/a | 0 | 0 |

By tag (pairs where both labels carry it, pair-weighted over all cities): street has no sidewalks 1,915 pairs, 0.966; street has a sidewalk 200 pairs, 1.000; gravel/dirt road 67 pairs, 1.000. NoSidewalk has no severity.

Sensitivities of the macro mean: `data_generation.expert_verdict` (a contradicting crowd vote drops the label): 0.992 over 3 cities (null 1 0.776 (p 0.001), null 2 0.925 (p 0.001)); without Oradell: 0.992 over 2 cities (null 1 0.584 (p 0.001), null 2 0.874 (p 0.001)); the 2026-10-18 pull: 0.992 over 3 cities (null 1 0.699 (p 0.001), null 2 0.908 (p 0.001)).

All seven types on the 2026-10-18 pull (macro mean over cities with >= 10 runs of 2+; same rules):

| type | labels with own verdict | pairs | cities | pair agreement [95 % CI] | null 1: city (p) | null 2: labeler (p) | same labeler, other face | same face, other labeler | share of same-expert pairs voted < 1 h apart |
|---|---|---|---|---|---|---|---|---|---|
| NoSidewalk | 2,325 | 6,917 | 3 | 0.992 [0.981, 0.999] | 0.699 (p 0.001) | 0.908 (p 0.001) | 0.860 | 0.981 | 29% |
| CurbRamp | 22,398 | 3,937 | 16 | 0.949 [0.938, 0.961] | 0.901 (p 0.001) | 0.928 (p 0.001) | 0.947 | 0.947 | 43% |
| NoCurbRamp | 12,960 | 2,343 | 13 | 0.915 [0.891, 0.935] | 0.695 (p 0.001) | 0.803 (p 0.001) | 0.770 | 0.926 | 28% |
| Obstacle | 8,291 | 4,450 | 14 | 0.895 [0.858, 0.917] | 0.623 (p 0.001) | 0.798 (p 0.001) | 0.757 | 0.890 | 51% |
| SurfaceProblem | 14,899 | 14,838 | 18 | 0.943 [0.926, 0.955] | 0.850 (p 0.001) | 0.901 (p 0.001) | 0.911 | 0.924 | 24% |
| Crosswalk | 4,659 | 261 | 6 | 0.982 [0.960, 0.998] | 0.818 (p 0.001) | 0.919 (p 0.001) | 0.942 | 0.967 | 43% |
| Signal | 1,321 | 110 | 3 | 0.986 [0.958, 1.000] | 0.926 (p 0.037) | 0.987 (p 0.728) | 0.977 | 0.958 | 60% |

Type x severity and type x tag (within-run pairs where both labels carry the same severity or the tag, a pair-weighted pool over cities, not a macro mean, so the cities with the most pairs set it; blank under 30 pairs; top three tags by pairs):

| type | severity 1 | severity 2 | severity 3 | tags |
|---|---|---|---|---|
| NoSidewalk | no severity | |  | street has no sidewalks 1,656: 0.963; street has a sidewalk 199: 1.000; gravel/dirt road 67: 1.000 |
| CurbRamp | 2,722: 0.928 | 206: 0.913 | 93: 0.935 | missing tactile warning 689: 0.940; points into traffic 295: 0.912; narrow 82: 0.915 |
| NoCurbRamp | 158: 0.905 | 320: 0.869 | 1,219: 0.931 | no alternate route 825: 0.908; unclear if needed 202: 0.975; alternate route present 138: 0.877 |
| Obstacle | 1,335: 0.928 | 553: 0.837 | 1,106: 0.910 | pole 764: 0.846; trash/recycling can 521: 0.985; parked car 452: 0.927 |
| SurfaceProblem | 6,675: 0.905 | 1,552: 0.907 | 1,178: 0.914 | cracks 3,586: 0.895; grass 3,079: 0.976; uneven/slanted 1,391: 0.940 |
| Crosswalk | 176: 0.994 | 7: n/a | 24: n/a | paint fading 17: n/a; broken surface 13: n/a; rail/tram track 11: n/a |
| Signal | no severity | |  | one button 50: 1.000; button waist height 29: n/a; two buttons 9: n/a |

#### Runs of every type (the issue's table)

A run is one labeler's labels of one type on one block face; a label without a face is a run of one. Public cities only.

| type (pull) | cities | labels | in a run of 2+ | of 3+ | run length of the median label | multi-label faces with one labeler | seattle-wa: labels, in 2+, median run |
|---|---|---|---|---|---|---|---|
| NoSidewalk 2026-11-05 | 45 | 237,421 | 86% | 78% | 7 | 58% | 51,646, 82%, 6 |
| NoSidewalk 2026-10-18 | 45 | 232,258 | 85% | 76% | 7 | 59% | 51,642, 81%, 6 |
| CurbRamp 2026-10-18 | 46 | 541,936 | 66% | 41% | 2 | 35% | 100,933, 54%, 2 |
| NoCurbRamp 2026-10-18 | 46 | 122,675 | 43% | 19% | 1 | 27% | 44,663, 44%, 1 |
| Obstacle 2026-10-18 | 46 | 158,013 | 72% | 56% | 3 | 52% | 17,982, 61%, 2 |
| SurfaceProblem 2026-10-18 | 46 | 192,017 | 76% | 61% | 3 | 68% | 37,545, 71%, 3 |
| Crosswalk 2026-10-18 | 45 | 67,325 | 15% | 4% | 1 | 33% | 6,280, 14%, 1 |
| Signal 2026-10-18 | 39 | 15,168 | 31% | 8% | 1 | 42% | 1,671, 25%, 1 |

NoSidewalk on the 2026-11-05 pull, cities with >= 1,000 labels:

| city | labels | in a run of 2+ | of 3+ | run length of the median label | multi-label faces with one labeler |
|---|---|---|---|---|---|
| seattle-wa | 51,646 | 82% | 74% | 6 | 50% |
| washington-dc | 45,213 | 88% | 82% | 10 | 39% |
| chicago-il | 14,733 | 82% | 72% | 6 | 84% |
| st-louis-mo | 14,535 | 95% | 90% | 9 | 95% |
| knox-oh | 12,617 | 94% | 90% | 48 | 92% |
| mendota-il | 12,412 | 91% | 85% | 11 | 13% |
| columbus-oh | 9,371 | 88% | 79% | 6 | 43% |
| gainesville-fl | 7,962 | 94% | 81% | 5 | 99% |
| spgg | 7,470 | 65% | 50% | 3 | 33% |
| teaneck-nj | 7,226 | 85% | 75% | 6 | 31% |
| newberg-or | 7,037 | 92% | 84% | 9 | 29% |
| burnaby | 6,916 | 93% | 89% | 9 | 39% |
| walla-walla-wa | 6,420 | 91% | 81% | 6 | 67% |
| danville-il | 5,890 | 94% | 90% | 8 | 86% |
| cdmx | 5,744 | 58% | 43% | 2 | 40% |
| oradell-nj | 5,679 | 91% | 84% | 11 | 14% |
| amsterdam | 4,002 | 78% | 71% | 7 | 74% |
| pittsburgh-pa | 3,509 | 76% | 63% | 4 | 78% |
| maywood-nj | 1,524 | 90% | 81% | 5 | 47% |
| paterson-nj | 1,268 | 75% | 59% | 4 | 72% |
| laurens-ia | 1,250 | 89% | 85% | 11 | 91% |
| tucson-az | 1,215 | 88% | 80% | 8 | 43% |
<!-- nosidewalk-runs-results:end -->

### Reading of item 1 (items 2 to 4 follow when the step (a) scores are in)

- **Within a run, the experts' own verdicts almost always agree** (macro mean 0.991 over Seattle, Chicago and
  Oradell), above both nulls in every named city (p 0.001-0.002). The labeler null is the one that matters: a run
  is one labeler, and a labeler's labels on *other* faces already agree 0.73-0.94 of the time.
- **In Seattle and Chicago this cannot be told apart from the sitting.** 360 of Seattle's 368 pairs and 299 of
  Chicago's 300 were judged by one expert less than an hour apart (Limits). The numbers are what a run that stands
  or falls together would give, and also what one expert judging a run as a block would give.
- **Oradell is the only city with pairs judged days apart** (4,935 pairs, 0.990 against 0.983 for pairs voted under
  an hour apart). That is above its labeler null (0.970), but Oradell's labels are about 98 % correct, so every
  number there sits near 1 and the margin is small.
- **NoSidewalk agrees within a run more than the other types do** on the same rules (0.992 against 0.90-0.95 for
  CurbRamp, NoCurbRamp, Obstacle and SurfaceProblem on the 2026-10-18 pull), with the same caveat for each type that
  its pairs are mostly judged together.
- Item 1 alone does not support validating the run instead of the label: it is consistent with that, but the
  clean test is the masked round under Follow-ups, and items 2 and 3 are the next evidence.

## Limits

- **Item 1 cannot separate the run from the sitting where it matters.** The votes are unmasked. In Seattle and
  Chicago almost every within-run pair was judged by one expert within an hour, often on a page that shows one
  labeler's labels together (Seattle's are mostly imported "Old data" votes with real timestamps, Chicago's mostly
  LabelMap). There, "the labels of a run stand or fall together" and "the expert judged the run as a block" predict
  the same numbers. Oradell has pairs judged days apart, but 98 % of its labels are correct, so its agreement sits
  close to chance by arithmetic.
- **Which labels got an expert vote is not random**: experts looked at faces and labelers they had reason to look
  at. The nulls compare against the same labels, not against the population.
- **Step (a) truth is entangled with item 1.** 546 of the 885 split rows carry a verdict copied from their face, and
  the label-level expert verdicts on one face come from the same sittings. Step (a) is therefore secondary; the
  step (b) ratings (masked, one label per face) are the primary test of pooling.
- **Step (b) is small.** 235 decided ratings, 31 incorrect, 2 of them in the pass stratum. Weighted rates are set by
  agree-below labels whose inclusion weights reach about 365.
- **Selection through the rated label.** Step (b)'s faces were drawn through one rated label each, 60 / 20 / 20 by
  stratum, and the labels beside it follow that label's stratum.
- **Crop paths.** Step (a)'s pilot crops came from `data_generation.py`, step (b)'s and this page's re-scored ones
  from `ai_labels_eval.py crops`; both arms of item 2 use the re-scored values (crop parity in item 2's step (a)
  table, once it is computed). The 13 split rows without a face sit on no step (a) face, so they are neither
  re-scored nor pooled.
- **Distance is a straight line** between label points, a proxy for distance along the street.
- Shared panoramas: a panorama can carry labels of two faces, so face-grouped resampling is not panorama-grouped.
- **Step (b) and step (a) held-out faces.** 7 of the 250 rated step (b) labels sit on step (a) held-out faces (3 of
  them were held-out split rows). Nothing is fitted on held-out rows, so this is not a leak. `pilot_heldout` in
  `face_scores.csv` marks only the held-out split rows, so the step (b) sensitivity "without neighbours on trained
  panoramas or on step (a) held-out faces" drops every label on those faces, taken from the step (a) table; a rated
  label itself always counts in its own pool.

## Follow-ups (one issue, filed by Jon or at his word)

- **A masked round that rates 2-3 labels per run**, in separate sittings. It is the only clean test of item 1 and
  the pre-registered test of a per-run gate. It needs a rater.
- Scoring labels on neighbouring faces within a distance along the street, and true along-street distance from the
  street geometry.

## How to rerun

```
D=~/Data/sidewalk-validator-ai-nosidewalk-2026-11
# experts in the environment only; EXPERT1_ID, EXPERT2_ID are derived from EXPERTS so their order cannot drift
export EXPERTS="<name>=<uuid>,<name>=<uuid>"
export EXPERT1_ID=$(echo "$EXPERTS" | cut -d, -f1 | cut -d= -f2) EXPERT2_ID=$(echo "$EXPERTS" | cut -d, -f2 | cut -d= -f2)
python analysis/nosidewalk_runs.py step-a-neighbours --pull $D/data/rawlabels-ns-2026-11-05 --gate $D/gate2 \
    --cities-json $D/runs/cities_2026-11-17.json --out $D/runs/step-a
# klone (docs/running-on-hyak.md): stage + crop + score with the kept gate2 weights, labels split in two by city
RUN=$R DEST=$STORE sbatch analysis/ai_labels_stage.slurm
RUN=$R/a RUNS=<gate2 runs> STORE=$STORE sbatch --dependency=afterok:<stage> nosidewalk_round.slurm   # and $R/b
ssh makelab1 "SCHEMAS='<public schemas>' EXPERT1_ID=$EXPERT1_ID EXPERT2_ID=$EXPERT2_ID bash -s" \
    < analysis/pull_expert_vote_times.sh > $D/runs/expert_vote_times.csv \
    || { echo "vote-time pull failed: the CSV lacks a schema" >&2; mv $D/runs/expert_vote_times.csv{,.failed}; }
python analysis/fetch_all.py $D/runs/rawlabels-contra-2026-11-17 --label-types CurbRamp,Obstacle,SurfaceProblem,NoSidewalk
python analysis/nosidewalk_runs.py agreement --ns-pull $D/data/rawlabels-ns-2026-11-05 \
    --all-pull ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels --vote-times $D/runs/expert_vote_times.csv \
    --cityparams $D/round-b/gallery_jon/cityparams_snapshot.conf --cities-json $D/runs/cities_2026-11-17.json \
    --out-json $D/runs/agreement.json
K=$D/round-b/private-key-do-not-share
python analysis/nosidewalk_runs.py pooling --step-a $D/runs/step-a --scores $D/runs/step-a-scores/a $D/runs/step-a-scores/b \
    --gate $D/gate2 --gate1 $D/gate --priors $D/data/whole/deployment_priors.json --key $K/key.csv \
    --sheet $K/votes_jon.csv --face-scores $D/round-b/neighbours/face_scores.csv --out-json $D/runs/pooling.json \
    --out-dir $D/runs
python analysis/nosidewalk_runs.py features --step-a $D/runs/step-a --scores $D/runs/step-a-scores/a \
    $D/runs/step-a-scores/b --gate $D/gate2 --key $K/key.csv --sheet $K/votes_jon.csv \
    --face-scores $D/round-b/neighbours/face_scores.csv --ns-pull $D/data/rawlabels-ns-2026-11-05 \
    --round-b-pull $D/round-b/rawlabels --contra-pull $D/runs/rawlabels-contra-2026-11-17 \
    --cities-json $D/runs/cities_2026-11-17.json --out-json $D/runs/features.json
python analysis/nosidewalk_runs.py report --agreement $D/runs/agreement.json --pooling $D/runs/pooling.json \
    --features $D/runs/features.json --cities-json $D/runs/cities_2026-11-17.json \
    --out-json docs/nosidewalk-runs-2026-11/runs.json --out-md docs/nosidewalk-runs-2026-11/runs.md \
    --doc docs/nosidewalk-runs-2026-11.md
```

`step-a-neighbours`, `agreement`, `pooling`, `features` and `report` stop on an input whose sha256 differs from
`PINS` in the module (`--allow-unpinned` is for tests only). Pinned: the 2026-11-05 NoSidewalk pull, the
2026-11-11 round (b) pull, the 2026-11-17 contradicting-label pull, the 2026-10-18 all-type pull (in the form
`deployment_gate.py` records it; it equals `docs/deployment-gate-2026-11/provenance.json`
`slices.rawlabels_files_sha256`), the cities snapshot, the cityparams snapshot, the expert vote times, gate2's
`folds.csv`, `folds.json`, five `split.csv` and five out-of-fold prediction files (as in
`docs/nosidewalk-2026-11/neighbours.json`), `step_a_faces.csv`, and round (b)'s key, sheet, `face_scores.csv` and
deployment priors. Pins are keyed by input, not by file name. The R3 step (a) predictions are matched by content:
`pooling` and `features` stop unless the files they read are exactly the `r3_*` files in `PINS`, whatever the score
directories are called, and they stop while none is pinned. They also stop when R3 scored under the plan's 90 % of
the step (a) labels (the plan's fallback, subset pooling over the split rows, is a different run that this code does
not report). `report` checks the cities snapshot and records the sha256 of the three JSONs it reads, which
are this page's own outputs. `agreement`, `pooling` and `features` refuse to replace an output whose bytes would
change unless given `--force`. Seeds: `--seed 0`, 1,000 permutations and 1,000 bootstrap resamples; 50 m for group
D; vote-gap strata at 1 h and 1 day.
