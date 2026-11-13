# NoSidewalk in auto-validation: census, face rule and pilot (November 2026)

**Status: phase A (pipeline and census) in PR [#91](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/91), not merged; pilot run 2026-11-05 on klone; not served.**
Issue [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42). This page reports how much NoSidewalk
training data the public rawLabels hold, why the label-level rule finds almost none, the block-face rule that replaces
it for this type, the deployment prior, and the pilot that tests whether a NoSidewalk validator is worth serving.
Signal (also in #42) is counted here and follows later with the label-level rule.

**Pilot in short:** the whole-view crop beats the depth-sized crop on test in all three seeds (AUC 0.916 vs 0.878,
within-city 0.931 vs 0.905, so both clear the 0.85 bar), and both fall to about 0.78 within-city on the held-out
cities. The gate half of the serving bar cannot be judged at this scale: no Agree verdict reaches 0.92 in any run,
because the calibration fit has only about 70 Agree labels. The 5-fold out-of-fold run of 2026-11-11 is the bar's
own test: within-city AUC 0.924, and 240 labels clear the gate at a prior-weighted precision of 0.979 (Wilson 95 %
lower bound 0.966), so the bar is met on that basis; unweighted the bound is 0.890, and the held-out cities stay
under it. The whole-view crop is the NoSidewalk crop from here on; nothing is served yet. [Results](#results).

## The census (public rawLabels, 2026-11-05)

A NoSidewalk- and Signal-only pull of the 45 public deployments (`analysis/fetch_all.py --label-types`), counted by
`analysis/type_readiness.py` with the two expert accounts in `EXPERTS`. Every number below is in
[`nosidewalk-2026-11/readiness.json`](nosidewalk-2026-11/readiness.json); the full tables (per city, tags, the
vote-pair histogram, Signal) are in [`nosidewalk-2026-11/readiness.md`](nosidewalk-2026-11/readiness.md).

NoSidewalk: **237,291 labels**, all human-authored, 99.5 % GSV (Mapillary 1,257, Panoramax 3). Only 11,111 carry any
human vote, 2,097 two or more, 327 three or more.

| rule | correct | incorrect | cities with >= 20 incorrect |
|---|---|---|---|
| label margin only (net >= 3 agree / net >= 2 disagree) | 79 | 80 | 1 |
| label margin + expert ([#38](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/38)'s rule) | 2,263 | 280 | 3 |
| face rule, census count (votes summed, every label on a decided face) | 15,575 on 1,576 faces | 590 on 167 faces | n/a |
| **training rule** (label rule, then the face rule below on the undecided labels) | 12,137 on 1,219 faces | 530 on 162 faces | 4 |
| training rule after the cap of 5 labels per face and class | 4,909 on 1,219 faces | 398 on 162 faces | 3 |

The training rule by the rule that decided each label (before the cap):

| class | margin | expert | face_margin | face_expert |
|---|---|---|---|---|
| correct | 79 | 2,184 | 4,369 | 5,505 |
| incorrect | 80 | 200 | 76 | 174 |

Incorrect labels / faces by city under the training rule after the cap: seattle-wa 234 / 84, chicago-il 61 / 32,
amsterdam 41 / 21, tucson-az 17 / 5, oradell-nj 8 / 5, then 7 or fewer labels each. Under the census count the same column read seattle-wa 301,
chicago-il 98, tucson-az 73, amsterdam 51.

**The census count and the training rule differ, on purpose.** The census summed votes over a face, so one validator
who disagreed with three labels of a face counted as three; the training rule counts distinct people (below). That
costs most in Tucson: 73 incorrect labels by the census count, 23 under the training rule, 17 (on 5 faces) after the
cap. Tucson alone would be a meaningless held-out set, so the pilot holds out `tucson-az` and `chicago-il` together
(below and "Open questions").

Signal: 15,455 labels in 39 cities; label margin 1,577 correct / 186 incorrect, with experts 2,245 / 202. It has
enough label-level verdicts to follow the five shipped types without a face rule.

## Why the label rule starves: the website settles faces at two votes

The vote pairs (human agree, human disagree) of voted NoSidewalk labels: (1, 0) 8,010; **(2, 0) 1,579**; (0, 1) 1,004;
(1, 1) 148; (1, 2) 86; (3, 0) 67; (2, 1) 59; (0, 2) 43. The training rule needs net three Agree votes; 1,579 labels
stop at exactly two. SidewalkWebpage serves NoSidewalk per block face
([SidewalkWebpage#5285](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5285),
`app/models/validation/ValidationQueuePolicy.scala`) and treats a face as settled at two agreeing human votes
(`SettledMargin = 2`, `FaceSettledSupport = 2`), so the third vote almost never comes. The website's votes are also
spread over the face's labels: a face with ten labels and two Agree votes on two different labels is settled for the
website while each of its labels, alone, is unusable.

Pooling by face does not mix classes. Among the 2,426 label-rule labels (with experts) that have a face, on 1,131
faces, only **11 faces carry both classes**. (The census script counted 1,200 faces because it also grouped labels
with a street edge but no `street_side`; those have no face here.) Incorrect faces are 88 % single-labeler (correct
faces 40 %); labels per incorrect face: median 2, p90 8, max 30. 10,085 labels (4 %) have no `street_side` and so no
face.

## The face rule

`nosidewalk_faces.py`, behind `data_generation.py --verdict-unit face`. A face is `(city, street_edge_id,
street_side)`, the unit SidewalkWebpage validates by; `face_id` = `<city>:<street_edge_id>:<side>`.

1. The label rule runs first, exactly as for every type (margin, then expert), and is never overridden.
2. For each label it leaves undecided, the face's pooled human votes decide. `support` = distinct human validators who
   voted Agree on any label of the face, `opposition` = distinct human validators who voted Disagree on any label of
   it.
   - `correct` if support >= 2 and opposition = 0; `incorrect` if opposition >= 2 and support = 0
     (`class_source = face_margin`);
   - otherwise, with `EXPERTS`: an expert voted on the face, the experts all one way, and no human voted the other
     way anywhere on the face: the expert's verdict (`class_source = face_expert`);
   - a contradicted face (support > 0 and opposition > 0) decides nothing beyond the label rule.
3. Distinct validators, not votes: one person agreeing with three labels of a face is one piece of evidence. AI votes
   never count. Labels without a face keep the label rule only.
4. At most 5 labels per face and class (`--max-labels-per-face`, the 5 with the lowest `label_rank`, so the choice is
   stable as the pool changes), before the 55:45 downsample; without it one 30-label face would be 5 % of the
   negatives.
5. The split groups by face (`split_data.py --group-col face_id`), joined with the panorama groups because one
   panorama can see both sides of a street; the 3 m merge still applies on top.

The label and face rules cannot disagree: a label-level `incorrect` needs a Disagree vote, which makes the face's
opposition positive, so the face cannot be `correct` (and the other way round). The code checks it anyway.

## The deployment prior

The margin-rule count (79 / 80) is not a sample of production: the validation queue serves lone-labeler faces first,
which are where false NoSidewalk calls concentrate, so the labels that collect enough votes are the doubtful ones.
`calibration/nosidewalk_prior.py` uses instead the
[SidewalkWebpage#5222](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5222) measurement
(`SidewalkPresenceBasis.scala`): against Seattle's sidewalk inventory, a face with one NoSidewalk label is right 69 %
of the time, two labels 80 %, three or more 84 %. Weighted by the share of production labels on faces of each size
(2026-11-05: 4.9 % of labels on 1-label faces, 5.2 % on 2-label faces, 89.9 % on 3+), **P(correct) = 0.831**. It is
written into `deployment_priors.json` as pseudo-counts over the 227,035 labels with a face, with a `source` naming the
method and the Seattle measurement. `data_generation.py` still writes its margin-only entry first (computed before the face rule and the cap);
`nosidewalk_prior.py` replaces it, and the pilot runs it after every `data_generation.py`. The caveats (one city, a
per-face measurement used per label, the inventory's own errors) are in [`decisions.md`](decisions.md).

## Pilot design (phase B, klone; run 2026-11-05)

`nosidewalk_submit.sh` submits it in one command (see [`running-on-hyak.md`](running-on-hyak.md)):

- **Data.** `EXPERTS` set; every deployment the cache holds (a NoSidewalk-only pull of the public ones by default; point
  `CACHE` at a full cache to include private ones, as round 3 did); `--verdict-unit face --max-labels-per-face 5`.
- **Split.** Face-grouped, `tucson-az` (round 3's public held-out city) and `chicago-il` held out together
  (`HOLDOUT=tucson-az,chicago-il`, the launcher's default). The run's selection (cap, then the 55:45 downsample)
  was 400 incorrect and 488 correct labels; `test_heldout` got 127 of them (80 incorrect), training 537 (227).
- **Two crop arms x 3 seeds (42, 43, 44).** Depth-sized (the default) and whole-view (`--fixed-crop-half-px 1024`):
  a NoSidewalk label marks an absence along a stretch, and #42 asked whether the crop size matters.
- **Evaluate and calibrate.** `evaluate.py --tag s<seed> --per-city` (with the severity and tag slices, on by default),
  then `calibration/calibrate.py` under the face-size prior, Agree only: the Disagree verdict is held back as for the
  five shipped types ([#57](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/57)).
- **Report.** `analysis/nosidewalk_pilot.py`: per arm and seed, AUC with a 95 % interval from resampling block faces,
  within-city AUC (the incorrect share differs by city), and the Agree verdict's coverage and precision at the 0.92
  gate on test and on `test_heldout` (whose two cities also give a within-city AUC), spliced into the results section below.

**Go / no-go bar for serving**, judged after phase B: within-city AUC >= 0.85 on test **and** the Agree verdict's 5-fold
out-of-fold gate precision with a Wilson 95 % lower bound >= 0.92 on >= 100 test labels. Below that the pilot still
answers #42's crop question, and the lever is more negatives: a masked expert round on NoSidewalk faces outside
Seattle, like [#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78), 300-500 faces.

## Cross-repository work (listed, not done here)

- **sidewalk-ai-api**: `nosidewalk` in `VALIDATOR_LABEL_TYPES` and `MODEL_MAP`, and an `accuracy_mappings` entry.
- **SidewalkWebpage**: NoSidewalk in `LabelType.aiLabelTypes`; and a policy decision for that repository: whether the
  AI's Agree vote counts toward `FaceSettledSupport` and toward the sidewalk-presence basis.
- **Signal** follows the same path with label-level verdicts; `analysis/type_readiness.py` already counts it.

## Open questions for review

- **The held-out cities.** Under the training rule after the cap, incorrect labels / faces: seattle-wa 234 / 84,
  chicago-il 61 / 32, amsterdam 41 / 21, tucson-az 17 / 5. Tucson alone (the plan's choice, the census count suggested
  73) is five incorrect faces, so the pilot held out `tucson-az,chicago-il`; in the run that was 80 incorrect labels
  (Chicago 63, Tucson 17) against 47 correct (Tucson only 2), so the held-out numbers are in effect Chicago's.
- **The test split is in effect Seattle.** 44 of test's 46 incorrect labels are Seattle's, and Seattle is the only
  test city with both classes, so test's within-city AUC is Seattle's AUC. The other 12 test cities contribute 42
  correct labels and 2 incorrect ones (Amsterdam). More negatives outside Seattle (next steps, b) fixes both this and
  the gate's sample size.
- **Is the gate's sample size the only obstacle?** With the thin-block guard off, the Agree verdict would pass the
  gate on most test labels at about 0.95–0.97 precision, but that number comes from isotonic blocks of 7 to 48
  validation labels. The 5-fold out-of-fold run (next steps, a) is the test the bar asks for.
- The cap of 5 shrinks the correct class most (12,137 to 4,909) but the 55:45 downsample would cut it to about 486
  anyway; it matters for the incorrect class (530 to 398 in the census, 400 in the run's one-day-newer cache).

## Results

<!-- pilot-results:start -->
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
<!-- pilot-results:end -->

The table above is `analysis/nosidewalk_pilot.py`'s output, also in
[`nosidewalk-2026-11/pilot.md`](nosidewalk-2026-11/pilot.md) and [`pilot.json`](nosidewalk-2026-11/pilot.json).
AUC is of the `incorrect` score; within-city AUC is the per-city AUC averaged with weights n_incorrect × n_correct
over cities with both classes and at least 10 labels, as in [`retrain-2026-10.md`](retrain-2026-10.md). On test that
is Seattle alone; on `test_heldout` it is Chicago and Tucson, with Tucson's weight 34 against Chicago's 2,835.

**The run.** 888 labels selected (400 incorrect, 488 correct) on 832 panoramas in 26 cities; 885 crops per arm (3
Seattle panoramas were not in the store: `no_pano`). Both arms use the same split, label for label: train 537
(310 correct / 227 incorrect), val 111 (65 / 46), test 110 (64 / 46), `test_heldout` 127 (47 / 80: Chicago 108,
Tucson 19). No val, test or held-out label lies within 10 m of a training label. Selected labels by the rule that decided them:
expert 260, face_expert 326, face_margin 223, margin 79. Depth-sized crops have a median half-width of 247 px (p10 123, p90 395)
on the native panorama; the whole-view crop is 1024 px for every label.

### Whole-view against depth-sized crops, paired by seed

Same split, same seed, only the crop differs:

| split | seed | depth AUC | whole AUC | whole − depth | depth within-city | whole within-city | whole − depth |
|---|---|---:|---:|---:|---:|---:|---:|
| test | 42 | 0.886 | 0.899 | +0.013 | 0.912 | 0.925 | +0.012 |
| test | 43 | 0.873 | 0.932 | +0.059 | 0.901 | 0.939 | +0.038 |
| test | 44 | 0.876 | 0.915 | +0.039 | 0.901 | 0.930 | +0.029 |
| test_heldout | 42 | 0.758 | 0.805 | +0.047 | 0.757 | 0.794 | +0.037 |
| test_heldout | 43 | 0.803 | 0.760 | −0.043 | 0.803 | 0.741 | −0.062 |
| test_heldout | 44 | 0.772 | 0.840 | +0.068 | 0.778 | 0.823 | +0.045 |

On test the whole view wins in every seed (mean +0.037 AUC, +0.026 within-city); on the held-out cities it wins in two
of three (mean +0.024, within-city +0.007). Each run's face-bootstrap interval is about ±0.15 wide on 64 test faces,
so no single difference is significant; the consistent sign on test is the evidence, and it is modest.

### The serving bar

- **Within-city AUC ≥ 0.85 on test: both arms pass** (depth 0.905, whole 0.931; every seed above 0.90). On the held-out
  cities both arms drop to about 0.78 within-city (depth 0.779, whole 0.786), a drop of 0.13–0.15. That is the same
  cross-city loss the five shipped types show in round 3 ([`retrain-2026-10.md`](retrain-2026-10.md): CurbRamp 0.973
  to 0.838 within-city, Obstacle 0.870 to 0.788, SurfaceProblem 0.858 to 0.682), and it rests on one held-out city.
- **Agree gate: not evaluable at this scale.** The Agree verdict's coverage at 0.92 is 0.000 in all six runs, and this
  is structural, not a property of the model. Calibration fits isotonic curves on the 111 validation labels (65 to 73
  of them Agree verdicts), and `calibration/_isotonic_map.py` serves any isotonic block with fewer than
  `THIN_BLOCK_N` = 300 labels at its 90 % Wilson lower bound rather than its mean. No Agree block can reach 300 labels
  (the top block holds 7 to 48), and a block whose labels are all correct clears 0.92 only from about 32 labels
  (the guard's 90 % bound, 1 / (1 + z² / n); a block at 0.98 needs about 56, at 0.97 about 80), so at this size every
  Agree block is served below 0.92: the highest served Agree value per run
  is depth 0.882 / 0.899 / 0.889 and whole 0.868 / 0.917 / 0.896 (seeds 42 / 43 / 44). `calibration_report.json`
  records this as `uncapped_reaches_gate.correct = false` in every run.

What the Agree verdict does on test, reweighted to the deployment prior (P(correct) = 0.831), per seed 42 / 43 / 44:

| arm | Agree verdicts on test | Agree precision (all verdicts) | guard off: gate coverage | guard off: gate precision | guard off, `test_heldout`: gate precision |
|---|---|---|---|---|---|
| depth | 66 / 69 / 66 | 0.957 / 0.959 / 0.957 | 1.00 / 0.95 / 0.73 | 0.957 / 0.957 / 0.948 | 0.888 / 0.913 / 0.922 |
| whole | 63 / 67 / 63 | 0.971 / 0.973 / 0.971 | 0.97 / 1.00 / 0.97 | 0.970 / 0.973 / 0.970 | 0.900 / 0.882 / 0.918 |

"Guard off" serves each block's mean instead of its lower bound (`build_api_map(curves, conservative=False)` on the
same fits; a diagnostic, never a map to serve). It says the Agree verdict is about 0.96–0.97 precise on test and about
0.88–0.92 on the held-out cities, the latter below the gate, but on blocks this thin it is an optimistic estimate.
The bar's own test, the 5-fold out-of-fold gate precision on at least 100 labels, needs the next step (a).

The Disagree verdict is held back as for the five shipped types. Its uncapped gate numbers on test (from each run's
`calibration_report.json`, `hold_back.uncapped_eval`), coverage / precision: depth 0.70 / 0.84, 0.93 / 0.78,
0.67 / 0.83; whole 0.93 / 0.65, 1.00 / 0.79, and 0.00 (no verdict reached the gate) for seed 44. The held-back
Disagree verdict would be 0.65–0.84 precise at the gate, so holding it back is right.

### Per city

Only cities with 50 or more labels in a split get a row (`evaluate.py --per-city`, `per_city_s<seed>.md`). Test's
other cities: oradell-nj 15, newberg-or 9, burnaby 4, walla-walla-wa 4, cdmx 3, amsterdam 2, columbus-oh 2, and five
cities with 1 label each; all their labels are correct except Amsterdam's 2. AUC and accuracy at 0.5, seeds 42 / 43 / 44:

| split | city | n (correct / incorrect) | depth AUC | whole AUC | depth accuracy | whole accuracy |
|---|---|---|---|---|---|---|
| test | seattle-wa | 66 (22 / 44) | 0.912 / 0.901 / 0.901 | 0.925 / 0.939 / 0.930 | 0.864 / 0.864 / 0.864 | 0.909 / 0.909 / 0.894 |
| test_heldout | chicago-il | 108 (45 / 63) | 0.754 / 0.801 / 0.776 | 0.792 / 0.738 / 0.820 | 0.667 / 0.676 / 0.694 | 0.648 / 0.602 / 0.676 |
| test_heldout | tucson-az | 19 (2 / 17) | 1.000 / 1.000 / 0.941 | 1.000 / 1.000 / 1.000 | 0.368 / 0.526 / 0.421 | 0.474 / 0.421 / 0.684 |

In Chicago the model ranks reasonably but at 0.5 calls too many labels correct (depth s42: recall of the incorrect
class 0.49, against 0.80 in Seattle). Tucson's AUC rests on 2 correct labels and means nothing; its accuracy says
most of its 17 incorrect labels score below 0.5.

### Severity and tags

NoSidewalk has no severity: every label's is empty, so the severity slice is one row equal to the whole split. Tag
slices from `slices_s<seed>.md` (AUC only where the smaller class has 8 or more labels), with the training rows per tag
(a label counts once per tag; train has 537 rows):

| tag | train (correct / incorrect) | test (c / i) | `test_heldout` (c / i) | depth AUC, seeds 42 / 43 / 44 | whole AUC, seeds 42 / 43 / 44 |
|---|---|---|---|---|---|
| (none) | 217 (104 / 113) | 61 (24 / 37) | 18 (4 / 14) | test 0.916 / 0.916 / 0.912 | test 0.909 / 0.945 / 0.936 |
| street has no sidewalks | 210 (154 / 56) | 34 (30 / 4) | 36 (23 / 13) | held-out 0.629 / 0.722 / 0.732 | held-out 0.769 / 0.903 / 0.866 |
| street has a sidewalk | 70 (36 / 34) | 9 (6 / 3) | 39 (17 / 22) | held-out 0.917 / 0.920 / 0.866 | held-out 0.880 / 0.858 / 0.890 |
| ends abruptly | 32 (15 / 17) | 6 (4 / 2) | 16 (6 / 10) | too few | too few |
| gravel/dirt road | 29 (23 / 6) | 4 (3 / 1) | 14 (2 / 12) | too few | too few |
| shared pedestrian/car space | 23 (9 / 14) | 6 (4 / 2) | 23 (0 / 23) | too few | too few |
| pedestrian lane marking | 5 (0 / 5) | 0 | 0 | — | — |

The largest arm difference anywhere is on held-out labels tagged "street has no sidewalks" (whole view +0.15 on
average over seeds, 36 labels, 13 incorrect): the tag that asserts the absence along the whole stretch is the one
the wider view helps most. It is a small slice; read it as consistent with the crop recommendation, not as proof.

### The 5-fold out-of-fold gate (step (a), run 2026-11-11)

The bar's own test. `analysis/nosidewalk_gate.py` pools the whole-view crops of train, val and test (758 rows in
450 face groups: 440 faces, plus the 12 face-less labels in 10 panorama groups) into five face-grouped folds;
fold k's model trains on three folds, selects its checkpoint on one and scores the fifth, so every pooled row is
scored once by a model that never saw its face. The map that serves a fold is fitted on the other four folds'
out-of-fold predictions (leave-one-fold-out calibration under the face-size prior, Disagree held back, thin-block
guard on), so each row's served accuracy is out of sample for the model and for the map. The gate precision is read
**reweighted to the deployment prior**, the basis the map is fitted under and the one `calibration_report.json`
reports, with its Wilson 95 % lower bound taken on the raw passing count and mapped through the two-class weight
ratio (the weights are fixed by the class, so the weighted precision is a monotone function of the raw one and the
bound keeps its coverage; an effective-sample-size bound would be far too conservative here, since the weight is
set by the outcome); the pooled rows' own 58:42 mix gives an unweighted
precision that is reported beside it and is not the bar's number. The count of >= 100 passing labels is the raw
count. Job files `nosidewalk_gate.slurm` / `nosidewalk_gate_submit.sh`; the two held-out cities are scored by every
fold model under a map fitted on all pooled rows.

What this estimates, and what it does not: the map serving fold k is fitted on four *other* fold models'
predictions, so the number is for the recipe "a model of this kind plus a pooled out-of-fold map", not for one
deployed model with its own map; a final model trained on every row is sharper than the 60 % fold models; the pooled
AUC mixes five score scales; folds are grouped by face, so a panorama carrying two faces can straddle folds (21
panoramas, 48 rows; the pilot split allows the same); and the mean-of-models row is descriptive only.

<!-- gate-results:start -->
Pooled out-of-fold rows: 758 (319 incorrect) in 450 face groups, 5 folds; the map serving each fold is fitted on the other folds' out-of-fold predictions under the face-size prior (P(correct) = 0.831), Disagree held back, thin-block guard on. Precision is reported both reweighted to that prior (the basis the map is fitted under, comparable with `calibration_report.json`) and unweighted on the pooled rows' own class mix.

| rows | AUC [95 % CI, face resampling] | within-city AUC (cities) | Agree gate n (coverage) | precision, prior-weighted (Wilson 95 % lower) | precision, unweighted (Wilson 95 % lower) |
|---|---|---|---|---|---|
| pooled | 0.928 [0.894, 0.956] | 0.924 (7) | 240 (0.317) | 0.979 (0.966) | 0.929 (0.890) |

**Bar:** within-city AUC >= 0.85 is met; the Agree gate's prior-weighted Wilson 95 % lower bound >= 0.92 on >= 100 passing labels is met (240 labels pass of 447 Agree verdicts; weighted lower bound 0.966, unweighted 0.890). **PASSES.**

Per fold (each fold's map fitted on the other folds):

| fold | rows | Agree gate n (coverage) | precision, prior-weighted (Wilson 95 % lower) | precision, unweighted (Wilson 95 % lower) |
|---|---|---|---|---|
| 0 | 156 | 78 (0.500) | 0.966 (0.935) | 0.885 (0.795) |
| 1 | 152 | 2 (0.013) | 1.000 (0.656) | 1.000 (0.342) |
| 2 | 151 | 55 (0.364) | 0.978 (0.944) | 0.927 (0.827) |
| 3 | 149 | 50 (0.336) | 0.982 (0.948) | 0.940 (0.838) |
| 4 | 150 | 55 (0.367) | 0.995 (0.970) | 0.982 (0.904) |

Per city (>= 10 pooled rows):

| city | rows | Agree gate n (coverage) | precision, prior-weighted (Wilson 95 % lower) | precision, unweighted (Wilson 95 % lower) |
|---|---|---|---|---|
| seattle-wa | 406 | 91 (0.224) | 0.978 (0.960) | 0.868 (0.784) |
| oradell-nj | 92 | 57 (0.620) | 0.963 (0.820) | 0.982 (0.907) |
| cdmx | 45 | 11 (0.244) | 1.000 (0.721) | 1.000 (0.741) |
| amsterdam | 41 | 0 | n/a | n/a |
| newberg-or | 38 | 25 (0.658) | n/a (one class) | 1.000 (0.867) |
| walla-walla-wa | 38 | 17 (0.447) | 1.000 (0.830) | 1.000 (0.816) |
| columbus-oh | 18 | 3 (0.167) | 1.000 (0.596) | 1.000 (0.438) |
| houston-tx | 11 | 1 (0.091) | 0.000 (0.000) | 0.000 (0.000) |
| paterson-nj | 10 | 3 (0.300) | 0.936 (0.658) | 0.667 (0.208) |

Held-out cities (`test_heldout`), each fold model under a map fitted on all 758 pooled rows; the mean-of-models row is descriptive only (averaged probabilities are not what the map was fitted on):

| model | rows (incorrect) | AUC | within-city AUC | Agree gate n (coverage) | precision, prior-weighted (Wilson 95 % lower) | precision, unweighted (Wilson 95 % lower) |
|---|---|---|---|---|---|---|
| fold 0 | 127 (80) | 0.814 | 0.802 | 34 (0.268) | 0.964 (0.926) | 0.765 (0.600) |
| fold 1 | 127 (80) | 0.770 | 0.768 | 6 (0.047) | 0.977 (0.866) | 0.833 (0.436) |
| fold 2 | 127 (80) | 0.791 | 0.768 | 24 (0.189) | 0.969 (0.925) | 0.792 (0.595) |
| fold 3 | 127 (80) | 0.819 | 0.805 | 22 (0.173) | 0.966 (0.916) | 0.773 (0.566) |
| fold 4 | 127 (80) | 0.782 | 0.749 | 28 (0.220) | 0.962 (0.916) | 0.750 (0.566) |
| mean of 5 models | 127 (80) | 0.810 | 0.795 | 8 (0.063) | 0.983 (0.904) | 0.875 (0.529) |
<!-- gate-results:end -->

**Reading.** On the pooled out-of-fold rows the bar is met on the prior-weighted basis: within-city AUC 0.924 over
seven cities, and 240 of the 447 Agree verdicts (31.7 % of all rows) clear 0.92 at a weighted precision of 0.979 with
a Wilson 95 % lower bound of 0.966. Four things qualify that:

- **The verdict depends on the basis.** Unweighted, on the pooled rows' own 58:42 mix, the same 240 labels are 0.929
  precise with a lower bound of 0.890, under the gate. The weighted number is the one the map is fitted for and the
  one `calibration_report.json` reports for the five shipped types, and it inherits the face-size prior's caveats
  (one Seattle inventory measurement, per face, applied per label; [decisions.md](decisions.md)). Which basis the
  bar means is a decision for the issue, not for this page; both numbers are reported above.
- **Fold 1 is an outlier.** Its model (best checkpoint at step 105, val macro-F1 0.862) passes only 2 of its 152
  rows through the gate; the other four folds pass 50–78 each at weighted precision 0.966–0.995. A pooled coverage
  of 31.7 % therefore understates what four of five models do (33–50 %) and the gate precision is driven by those
  four. Seed variation of this size was already visible in the pilot's three seeds.
- **Seattle and Oradell carry the gate.** Of the 240 passing labels, 91 are Seattle's (weighted precision 0.978,
  lower bound 0.960) and 57 Oradell's (0.963, 0.820); no other city has more than 25, and Amsterdam's 41 rows pass
  none. Seattle is the only city whose own lower bound clears 0.92, and no city reaches the 100-label count, so the
  bar is met only in aggregate.
- **The held-out cities do not clear it.** Each fold model reaches within-city AUC 0.75–0.81 on Chicago and Tucson
  (the pilot's 0.79), and its Agree verdicts at the gate are 0.96–0.98 precise weighted on 6–34 labels (lower
  bounds 0.87–0.93, far short of the 100-label count), 0.75–0.83 unweighted. The class mix there is 37 % correct, so the weighting matters
  even more; and these are the sharper-than-fold-model caveats in reverse: a model that has never seen a city is
  not what the pooled estimate describes. Step (b), more negatives outside Seattle, is still the lever for this.

**Decision (2026-11-11, [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42)).** The bar is
read on the **prior-weighted** basis, the one the maps are fitted under; the unweighted number stays in every report
as the pessimistic view. On that basis the pooled gate passes. Serving still waits: step (b) below comes first, because
the pass is an aggregate one and the held-out cities do not clear it.

What the pooled number is: the map serving fold k was fitted on four *other* fold models' out-of-fold predictions,
so this is the precision of the recipe (a whole-view NoSidewalk model of this size plus a pooled out-of-fold map),
not of one deployed model with its own map. A deployed model would be trained on all 758 rows (sharper than the
fold models' 450) with a map fitted on the pooled out-of-fold predictions, which is the configuration step (c)
would ship; its gate precision is expected at or above this estimate on the training cities and is unmeasured on
new ones.

### The masked expert round outside Seattle (step (b), design fixed 2026-11-11)

Step (a) scored labels that already had a settled verdict, and most of its evidence is Seattle's. Step (b) samples
the **deployment population** of four other cities instead: every human-placed NoSidewalk label with a block face,
whatever its votes. One expert rates 250 of them in production Gallery with the votes masked
(`analysis/nosidewalk_round.py`, design `analysis/gt_rounds/nosidewalk_b.json`; the Gallery lists, the vote pull and
the pins are `analysis/gt_rounds.py`'s). Within a city the rated pass-stratum labels are a random sample of the
labels the gate would pass, so that city's Agree precision is read directly on its natural class mix, with a plain
Wilson bound and no prior. Across cities the rated set follows the design's quotas, not the population (see "What is
read" below). The round also measures the share of correct labels that the face-size prior assumes (0.831).

**Two phases, so the model chooses what is rated without biasing the estimate.**

1. *Eligible labels* (rawLabels pull of 2026-11-11): NoSidewalk, human-placed, with a block face and a usable pano
   point, author not excluded (Gallery hides those), never placed or voted on by the rater, and on no face or
   panorama among the 758 rows the fold models trained on. Labels with crowd votes stay in. Chicago and Tucson were
   never trained on; Oradell and CDMX were.

   | city | NoSidewalk labels | no face | placed by the rater | voted on by the rater | trained face or panorama | eligible | phase 1 |
   |---|---|---|---|---|---|---|---|
   | chicago-il | 14,741 | 416 | 195 | 59 | 0 | 14,071 | 400 |
   | tucson-az | 1,215 | 11 | 335 | 9 | 0 | 860 | 300 |
   | oradell-nj | 5,679 | 188 | 2,063 | 1,145 | 852 | 1,431 | 250 |
   | cdmx | 5,744 | 827 | 0 | 5 | 336 | 4,576 | 250 |

   No label was AI-placed, had an unusable pano point or had an excluded author. In Oradell and Tucson the rater
   placed a large share of the labels (36 % and 28 %), so the eligible population there is the *other* labelers'
   labels; the round says nothing about the rater's own.
2. *Phase 1*: a seeded simple random sample of each city's eligible labels (the counts above, 1,200 in all) is
   cropped whole-view and scored by each of the five fold models of a second gate run (below), served through the
   map fitted on all 758 of that run's out-of-fold predictions, as the held-out table above is.
3. *Strata*: **pass** when at least 3 of the 5 models give Agree at served accuracy >= 0.92; **agree-below** when
   not, with a median `p_correct` >= 0.5; **model-incorrect** otherwise.
4. *Phase 2*, the 250 rated items: Chicago 80, Tucson 60, Oradell 55, CDMX 55, each split 60 / 20 / 20 over the
   strata (150 / 50 / 50 in all), at most one label per face, in a seeded shuffled order. A short cell is topped up
   from the same stratum of another city. Each item carries its inclusion weight (eligible / scored in its city,
   times scored / picked in its cell), so every estimate can be reweighted to the population.

**The rating rule.** Agree: no sidewalk on the labeled side of the street at that spot. Disagree: a sidewalk is
there, or the label marks something else. Unsure: only when the imagery cannot show it. Gallery's vote counts,
comments and AI icons are hidden by the masking CSS; the rater cannot tell an item's stratum.

**What is read, fixed before any rating.**

- **The bar:** in the pass stratum as sampled, correct / decided with a Wilson 95 % lower bound >= 0.92 on >= 100
  decided labels. Unsure votes leave the denominator and are counted; the same rate with every Unsure counted as
  incorrect is printed beside it as a sensitivity check. With 150 items the bound clears 0.92 from 145
  correct (0.967) up, so a true precision near 0.96 would be a coin flip; that is the price of 250 ratings.
- **What that rate is.** It is the precision under the design's city mix, which spreads the pass items almost
  evenly (Chicago 35 %, Tucson 21 %, Oradell 22 %, CDMX 22 % of them), where the four cities' pass population is
  about 75 % Chicago, 3 % Tucson, 14 % Oradell and 8 % CDMX. The bar therefore asks that the gate hold across
  cities, not that it hold in a Chicago-dominated average. The population-weighted rate is reported beside it with
  a bootstrap interval; its effective sample size is about 88, under the bar's 100, so it is not the bar. Where the
  two differ, the per-city rows say why.
- Beside those: per city, and per fold model on its own gate verdicts (weighted precision, coverage, AUC,
  within-city AUC).
- The population share of correct labels against the prior's 0.831 (this population leaves out the rater's own
  labels and trained faces, which the prior does not); severity and tag slices with training counts; the crowd
  majority against the expert.
- At the vote pull, the timestamp of each vote is checked against the pool pull of 2026-11-11: a vote on a key label
  cast before the lists existed would not be a masked rating.

The same ratings can later score the final all-rows model of step (c) by reweighting.

**The second gate run (`gate2`).** Step (a) deleted its fold checkpoints, so the five fold models were retrained on
the same fold files with the same seed (`KEEP_BEST=1 ./nosidewalk_gate_submit.sh`, which keeps each fold's best
checkpoint). Retrained weights are not bit-identical, so the run has its own out-of-fold predictions, map and
report; it is also a replication of step (a).

It agrees with step (a) and is a little stronger (run 2026-11-11, jobs 43156012–43156017, one A40 or L40 per
fold, 11–15 minutes each):

| run | pooled AUC [95 % CI] | within-city AUC | Agree gate n (coverage) | precision, prior-weighted (lower) | precision, unweighted (lower) |
|---|---|---|---|---|---|
| step (a) | 0.928 [0.894, 0.956] | 0.924 | 240 (0.317) | 0.979 (0.966) | 0.929 (0.890) |
| `gate2` | 0.936 [0.907, 0.961] | 0.939 | 265 (0.350) | 0.985 (0.974) | 0.947 (0.913) |

Fold 1 is again the weakest model (31 passing rows of 152; 2 in step (a)); the others pass 53–71. On the 127
held-out rows the five models reach within-city AUC 0.78–0.82 and pass 18–45 labels at a weighted precision of
0.94–0.97 (lower bounds 0.89–0.93), the same picture as step (a). Best checkpoints: fold 0 step 150, 1 step 75,
2 step 270, 3 step 120, 4 step 165.

**Phase 1 as scored** (2026-11-11, jobs 43156311 staging, 43156312 crops and scores, 32 minutes): 1,197 of the
1,200 labels (3 panoramas are missing from the store). Labels per stratum, and the items picked for rating:

| city | scored | pass | agree-below | model-incorrect | rated: pass / agree-below / model-incorrect |
|---|---|---|---|---|---|
| chicago-il | 400 | 179 (45 %) | 166 | 55 | 53 / 16 / 21 |
| tucson-az | 299 | 88 (29 %) | 155 | 56 | 31 / 12 / 12 |
| oradell-nj | 249 | 204 (82 %) | 36 | 9 | 33 / 11 / 6 |
| cdmx | 249 | 38 (15 %) | 175 | 36 | 33 / 11 / 11 |

Two cells fell short under the one-label-per-face rule and were topped up from Chicago: Tucson's pass cell (31 of
36; its 88 passing labels sit on few faces) and Oradell's model-incorrect cell (6 of 11). The rated set is 250
items on 250 faces, 150 / 50 / 50 over the strata; each city's weights sum to its eligible count. A face is used
once, in the stratum of its first-ranked label, so the cell weights (scored / picked) are approximate where
faces repeat, which is mainly Tucson.

**Pins** (per-label files stay outside the repo, under `~/Data/sidewalk-validator-ai-nosidewalk-2026-11/`):
`round-b/pool/pool.json` `f973f826fa20fb61d2bc339153596d4070a11049ccd023df3ac4ca7e6fa78807` (it pins the rawLabels
pulls, the excluded-author pull, the pilot split and the seed's hash), `pool/labels.csv`
`e41ea4501ded5bbc3fc06ffaf20d857c6b88a9ffd3413836bcd4f0a7fa95ac2f`, `scored/manifest.csv`
`88430ecb40691b6833f6b4768625800b2491b6891ec2faace583f18539c7f2d1`, `gate2/report/gate.json`
`cbb7ea06aacc6a3ed68fb7611123186c4ab16103320581ee56b001aee840077f`, the key
`d2d30d05912e1a12cd75f56822a32413cb250dd048a5c17ff5bc13ae949172a8`, the Gallery lists
`0fd1e0554e262760e200d64bbb7e5ddba5f24556b97281338afa6d0f6f3d7c6f`; the key's `manifest.json` pins the ten
prediction files. `gate2` folds: `folds.csv` sha256 unchanged from step (a) (`895d00b4…`). Code: branch
`nosidewalk-round-b` at 7649325 for the fold models, crops and scores. Job logs in `klone-logs/round-b/`.

**Results.**

<!-- round-b-results:start -->
250 items, 248 rated (13 Unsure, 2 unanswered). Precision cells are correct / decided = rate (Wilson 95 % lower bound); Unsure votes are outside the denominator.

**Bar:** in the pass stratum, 142 / 144 = 0.986 (0.951); a lower bound >= 0.92 on >= 100 decided labels is met. **PASSES.** Reweighted to the cities' population: 0.998 (Wilson 95 % lower bound 0.953 at the weights' effective sample size of 84; no bootstrap interval is given, because resampling cannot vary a cell with no misses). With every Unsure in the pass stratum counted as incorrect: 142 / 149 = 0.953 (0.906).

Share of correct NoSidewalk labels in these cities' eligible population: 0.878 [0.819, 0.932] (the deployment prior assumes 0.831 over every label with a face; this population leaves out the rater's own labels and trained faces. The interval resamples within (city, stratum) cells with the weights fixed, so it is too narrow where a cell has no misses, and Unsure and unanswered items drop out with their weight).

| stratum | items | Unsure | correct / decided (Wilson 95 % lower) |
|---|---|---|---|
| pass | 150 | 5 | 142 / 144 = 0.986 (0.951) |
| agree_below | 50 | 2 | 42 / 48 = 0.875 (0.753) |
| model_incorrect | 50 | 6 | 20 / 43 = 0.465 (0.325) |

| city | decided (incorrect) | pass stratum (Wilson 95 % lower) | population share correct |
|---|---|---|---|
| chicago-il | 84 (8) | 51 / 51 = 1.000 (0.930) | 0.925 |
| tucson-az | 54 (12) | 29 / 31 = 0.935 (0.793) | 0.788 |
| oradell-nj | 49 (2) | 33 / 33 = 1.000 (0.896) | 0.981 |
| cdmx | 48 (9) | 29 / 29 = 1.000 (0.883) | 0.713 |

Per fold model, on its own gate verdicts (weights are the items' inverse inclusion probabilities):

| model | gate passes, rated items unweighted (Wilson 95 % lower) | precision, weighted | coverage, weighted | AUC, weighted | within-city AUC, weighted (cities) |
|---|---|---|---|---|---|
| fold 0 | 115 / 117 = 0.983 (0.940) | 0.997 | 0.318 | 0.883 | 0.926 (4) |
| fold 1 | 83 / 84 = 0.988 (0.936) | 0.998 | 0.264 | 0.910 | 0.943 (4) |
| fold 2 | 155 / 158 = 0.981 (0.946) | 0.975 | 0.622 | 0.871 | 0.941 (4) |
| fold 3 | 121 / 121 = 1.000 (0.969) | 1.000 | 0.372 | 0.923 | 0.942 (4) |
| fold 4 | 153 / 157 = 0.975 (0.936) | 0.963 | 0.536 | 0.819 | 0.753 (4) |

| severity | decided | pass stratum (Wilson 95 % lower) | population share correct | training rows |
|---|---|---|---|---|
| (none) | 235 | 142 / 144 = 0.986 (0.951) | 0.878 | 758 |

| tag | decided | pass stratum (Wilson 95 % lower) | population share correct | training rows |
|---|---|---|---|---|
| (none) | 48 | 23 / 23 = 1.000 (0.857) | 0.892 | 325 |
| ends abruptly | 22 | 4 / 5 = 0.800 (0.376) | 0.812 | 47 |
| gravel/dirt road | 20 | 14 / 15 = 0.933 (0.702) | 0.857 | 38 |
| shared pedestrian/car space | 20 | 5 / 6 = 0.833 (0.436) | 0.328 | 40 |
| street has a sidewalk | 29 | 14 / 14 = 1.000 (0.785) | 0.849 | 93 |
| street has no sidewalks | 127 | 95 / 95 = 1.000 (0.961) | 0.949 | 278 |

Slice population shares rest on few items with unequal weights and carry no interval; read them as direction only.

Crowd majority (human votes at the pool pull) against the expert, decided items:

| crowd | expert correct | expert incorrect |
|---|---|---|
| correct | 34 | 2 |
| incorrect | 2 | 12 |
| none or tie | 168 | 17 |
<!-- round-b-results:end -->

**Reading (rated 2026-11-12 and 2026-11-13).** The bar is met: of the 144 decided labels the gate would pass, the
expert called 142 correct (0.986, Wilson 95 % lower bound 0.951), against a bar of 0.92 on at least 100. Both
misses are in Tucson (29 of 31), one tagged "ends abruptly" and "gravel/dirt road", one "shared pedestrian/car
space"; Chicago (51 of 51), Oradell (33 of 33) and CDMX (29 of 29) have none. Chicago and Tucson were never
trained on. The bar is met pooled over the four cities, not in each: Chicago alone clears 0.92 (lower bound
0.930), Oradell and CDMX are too small to clear it alone, and Tucson, with both misses (29 / 31 = 0.935, lower
bound 0.793), gives no evidence that the gate holds there. What qualifies the result:

- **Unsure votes.** Five pass-stratum labels were rated Unsure (three in CDMX, two in Chicago). The bar leaves
  them out, as fixed in advance. Counting all five as incorrect gives 142 / 149 = 0.953 with a lower bound of
  0.906, under 0.92. The verdict survives up to three of the five being wrong (lower bound 0.923 at three,
  0.914 at four). Unsure is rarer in the pass stratum (5 of 150) than among the labels the model calls incorrect
  (6 of 50).
- **The population-weighted rate is 0.998, with a lower bound of 0.953** (Wilson at the weights' effective
  sample size of about 84). It is not the bar; it is high because Chicago holds three quarters of the passing
  population and has no misses.
- **Two labels could not be rated.** Gallery reported two CDMX labels as unavailable (one pass, one
  model-incorrect), so 248 of 250 have a vote. They are outside every denominator.
- **This is a 3-of-5 vote of the fold models, not the model that would ship.** Per model, the gate passes 84 to
  158 of the rated items at 0.975 to 1.000 unweighted; fold 4 is the weakest ranker (weighted within-city AUC
  0.753 against 0.926 to 0.943 for the others; low in three of the four cities and driven by a few heavily
  weighted below-gate items) and has the lowest weighted precision on its own passes (0.963; 0.975 unweighted).
  No AUC here carries an interval.
- **The order of the strata is as designed.** Agree below the gate is 42 / 48 = 0.875 and the labels the model
  calls incorrect are 20 / 43 = 0.465, so the model's Disagree is right about half the time here. Disagree stays
  held back.
- **The prior.** The share of correct labels in these four cities' eligible population is 0.878 [0.819, 0.932],
  against the 0.831 the deployment prior assumes; the interval contains it, though it is a different population
  from the prior's (the rater's own labels and trained faces are left out) and the interval is too narrow.
  Unsure and unanswered items drop out with their weight; spreading that weight over the decided items of the
  same cell gives 0.869. It differs by city: Oradell 0.981, Chicago 0.925, Tucson 0.788, CDMX 0.713. CDMX drew
  the highest Unsure share (5 of 53 answered; Chicago 6 of 90), and the rater remarked that it was far harder to
  judge than the other three.
- **Tags.** The tag table's population shares are unstable: "shared pedestrian/car space" (0.328) rests on 20
  decided items with an effective sample size of about 6.
- **One rater.** There is no second expert and no measure of rater agreement in this round. Every vote was cast
  after the label pull of 2026-11-11, so no earlier vote of the rater's is counted: a read-only query of
  `label_validation.end_timestamp` for the rater's votes on the key's labels, run on 2026-11-13 before the
  Oradell top-up, found 243 votes on 243 labels, the earliest at 2026-11-12 05:06 PST and none before the pull.
  The five votes added afterwards did not exist at that query, so they too are later than the pull. The query is not part of the repo's code.

Pins: `votes_jon.csv` `09a2624f93ff5ca203dfd8f0a4038129252d12334d863046986931e99ce957c9` (pulled 2026-11-13, 248
of 250 voted; beside the key, outside the repo), the aggregates
[`nosidewalk-2026-11/round_b.json`](nosidewalk-2026-11/round_b.json). Regenerate with `analysis/nosidewalk_round.py
report` on the pinned key, sheet and `scores.csv`.

### Cost and jobs

klone, `gpu-l40s` (one L40S) for data and training, `ckpt-all` CPU for staging and the report. Staging 829
panoramas (8.9 GB) took 2 minutes; crops 11 min 43 s (depth, including depth estimation) and 17 min 44 s (whole);
training 25 epochs on 537 crops took 380–398 s per run (`train_runtime`). Job ids: stage 42916414; data depth
42917458, whole 42917573; train depth s42 / s43 / s44 42918974 / 42919041 / 42919049, whole 42919040 / 42919046 /
42919051; report 42919052. A first submission chain (42917199, a duplicate left by an `sbatch` socket timeout,
42917208 and its dependents) was cancelled and its outputs wiped before the clean run, and the first training
attempt (42917537 and its siblings) failed because compute nodes cannot reach Hugging Face's Xet server; the job
file now sets `HF_HOME` and `HF_HUB_OFFLINE=1` ([`running-on-hyak.md`](running-on-hyak.md#nosidewalk-pilot)).

Gate run (2026-11-11): `ckpt-all`, one A40 per fold (nodes g3054, g3064, g3072, g3074), 25 epochs on about 450 rows
in 617–637 s of `train_runtime` per fold, 13 minutes wall-clock per job including scoring; the five fold jobs ran in
parallel (43134271–43134275) and the CPU report job (43134276, n3266) took 52 s. Fold checkpoints were deleted after
scoring (3.8 MB left per fold: `training_run.json`, `run_key.json`, `trainer_state.best.json`, the split and prior).
Free partition, no preemption.

### Recommendation and next steps

**The whole-view crop (`--fixed-crop-half-px 1024`) is the NoSidewalk crop from here on.** It answers #42's crop
question: better on test in every seed, no worse on the held-out cities on average, and best on the tag that marks an
absence along the whole street. **NoSidewalk is not served yet**: the test half of the bar passes, and the gate half
passes on the prior-weighted basis in the 5-fold run of 2026-11-11 (weighted lower bound 0.966 on 240 labels) but not
unweighted (0.890) and not on the held-out cities; see "The 5-fold out-of-fold gate" above. The masked expert round of step (b)
meets its bar pooled over four cities outside Seattle (0.986, lower bound 0.951, on 144 labels; only Chicago
clears it alone).

- (a) **5-fold out-of-fold training over the 758 pooled train / val / test crops** (whole view; the 127 held-out
   rows stay held out), calibrating on the pooled out-of-fold predictions, so each fold's Agree map is fitted on
   350–365 Agree verdicts instead of about 70 and the bar's Wilson lower bound on ≥ 100 labels can be computed.
   Five training runs of this size are well under an hour of GPU time. **Done 2026-11-11**: see "The 5-fold
   out-of-fold gate" above. The basis is decided: prior-weighted (2026-11-11).
- (b) **A masked expert round outside Seattle**, before any serving work (decided 2026-11-11): one rater, 250
   labels chosen score-first from four cities' deployment population, so the gate is read where step (a) could
   not reach. Design and results in "The masked expert round outside Seattle" above. **Done 2026-11-13: the bar
   is met pooled over the four cities** (142 / 144 = 0.986, lower bound 0.951; only Chicago clears it alone), with the caveats in that section's reading; whether that
   unlocks step (c) is the maintainer's decision.
- (c) **Cross-repository work** (sidewalk-ai-api, SidewalkWebpage; listed above) after step (b), if its bar is
   met; a final model on all 758 rows with a map fitted on
   the pooled out-of-fold predictions is the configuration to ship, gated per deployment as
   [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52) asks.

## Provenance

Pilot run (phase B, 2026-11-05):

- rawLabels: NoSidewalk-only public pull of 2026-11-05 (46 deployments, every one public in the live `/v3/api/cities`
  that day; `generation_run.json` `city_source`); `EXPERTS` with 2 accounts (values never written down).
- Code: branch `nosidewalk` at 8eafcbb (the PR head when the jobs were submitted; the logs do not record the commit),
  sidewalk-panorama-tools 7ce25ffbddae23d85f9976ec0a925707a0d64a89, depth encoder ViT-B (vkitti), tilt beta 0.95,
  `--verdict-unit face --max-labels-per-face 5`, selection seed 42, training seeds 42, 43, 44, 25 epochs. The job file
  then gained the `HF_HOME` / `HF_HUB_OFFLINE` lines (the commit that adds this section); nothing else changed.
- Panoramas: a staged copy at `/gscratch/scrubbed/jfroehli/pano-stage/NoSidewalk`, with the store marker line
  "Project Sidewalk pano store; see docs/ops.md#the-store-marker" (`generation_run.json` `pano_store_marker`).
- Selection: `selection_sha256` `f6a46bde78dfa8e65d43d4859046b7ec497e83afd7c60e9dcbd6d155aba97411`, the same in both
  arms. Per-label files stay outside the repo (`~/Data/sidewalk-validator-ai-nosidewalk-2026-11/`): `split.csv`
  sha256 `690bebc0214684901caf9e0c7a87491caed2b5c385c563a662de59cf5858a6fb` (depth) and
  `94d6ead5d0d0f979aee8e334376a139f16e969af44046124e317b857e00f58e8` (whole; the files differ only in crop columns,
  the uids and assignments are identical).
- Prior: `calibration/nosidewalk_prior.py`, P(correct) = 0.8306 (188,680 / 38,485 pseudo-counts over 227,165
  labels with a face).
- `pilot.json` sha256 `a340ef99fe01e778ab73eb39798fff36b9cd9ee7750bb0e2e8fee9208f4e1704`; regenerated on a laptop from
  the run's predictions with `python analysis/nosidewalk_pilot.py --data <copy> --out-json ... --out-md ... --doc
  docs/nosidewalk-2026-11.md`, byte-identical to klone's report job (42919052).
- Job ids: see "Cost and jobs" above.

Gate run (step (a), 2026-11-11):

- Inputs: the pilot's whole-arm `split.csv` (sha256 `94d6ead5d0d0f979aee8e334376a139f16e969af44046124e317b857e00f58e8`,
  as above) and `whole/deployment_priors.json` (the face-size prior, P(correct) = 0.8306); no new crops or pulls.
- Code: branch `nosidewalk-gate`, training and scoring at 0a58709 (the five fold jobs), the report at 246d622 (the
  clone was pulled between the two; `fine_tune_classifier.py`, `evaluate.py` and `nosidewalk_gate.py predict` did not
  change between them). `fine_tune_classifier.py` seed 42, 25 epochs, `--num-workers 7`, checkpoint by val macro-F1;
  the fold models' `training_run.json` files predate the provenance fields of
  [#94](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/94).
- Folds: `analysis/nosidewalk_gate.py folds --folds 5 --seed 42`; `folds.csv` sha256
  `895d00b4f5fb5567cf5b2b0253db5dba986ac42085c4827fe6123e3101df25ac` (758 rows, 450 face groups; per fold 149–156
  test rows, 149–156 val, 450–459 train; `test_heldout` 127 in every fold file). Best checkpoints (val macro-F1):
  fold 0 step 360 (0.891), 1 step 105 (0.862), 2 step 345 (0.890), 3 step 135 (0.959), 4 step 195 (0.856).
- Predictions (`fold<k>/predictions.csv`, sha256): fold 0 `1170b8f2e572cbe8da2b822a9e4756dd5b3f726f0c9b0a986b95d9c152a1e234`,
  1 `4a18a0d774441330d1f01d166a46bf2b0c9a01b41d3193485a05e5cda36fd7b4`, 2 `dd424332c2f2b61162dfcfb8ff9a28fb3809a4b9fcca59958d6408d2b4e150db`,
  3 `3c8bc0cfa63b0b7537380896812f7f946b0bfab354df5defec64f65cc3d25876`, 4 `c1527363c6de7357d82b0b0a789b8237802475ce139317896a45cb775ff9b1de`.
- Report: klone's report job ran at 246d622, whose weighted Wilson bound used an effective-sample-size (Kish)
  correction (`gate.md` sha256 `f1cc33846011bb77d76671580a34f86694ee276b959636597df7661f4fb638c6`, kept as `gate.local.md` in the results copy; pooled weighted
  lower bound 0.951); review round 2 replaced that bound with the exact one (the raw Wilson bound mapped through the
  two-class weight ratio; the commit that adds this paragraph), and the report was regenerated on a laptop from the
  copied predictions and split files (`--n-boot 1000 --seed 0`, the defaults): every number other than the weighted
  lower bounds is identical to klone's, and the results block above is spliced from that run (`gate.json` sha256
  `8800e364d3ea916093d3a2310173e8aa6865666c7078b90f537c26c352e097a0`, `gate.md`
  `ece584d2d610f893e59035fd471ddd95d66c841906301da011640093d5d16b08`).
- Per-label files stay outside the repo: `~/Data/sidewalk-validator-ai-nosidewalk-2026-11/gate/` (fold split files,
  predictions, report) and `klone-logs/gate/` (the six job logs).
- Job ids: see "Cost and jobs" above.

Census (phase A):


- rawLabels: NoSidewalk- and Signal-only pull of the 45 public deployments listed by `/v3/api/cities`, 2026-11-05
  (`labelType` query parameter); public/private resolved from the live listing when the census ran.
- Census: `analysis/type_readiness.py --pattern '{city}__{label_type}.csv' --label-type NoSidewalk --label-type
  Signal` (that pull's file layout; `fetch_all.py --label-types` writes `<city>.csv`) with 2 expert accounts in
  `EXPERTS`, at commit b932358 (the code commit of this change).
- `readiness.json` sha256 `a4ce2b2ba24fd1f22a131ed32703f8f8ce509c367abd5529f729399a9bcd7823`.
- Seeds: selection and cap 42; pilot training 42, 43, 44.
