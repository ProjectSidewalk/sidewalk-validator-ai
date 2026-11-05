# NoSidewalk in auto-validation: census, face rule and pilot design (November 2026)

**Status: phase A: pipeline and census; pilot not yet run.**
Issue [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42). This page reports how much NoSidewalk
training data the public rawLabels hold, why the label-level rule finds almost none, the block-face rule that replaces
it for this type, the deployment prior, and the pilot that will say whether a NoSidewalk validator is worth serving.
Signal (also in #42) is counted here and follows later with the label-level rule.

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

## Pilot design (phase B, klone; not run)

`nosidewalk_submit.sh` submits it in one command (see [`running-on-hyak.md`](running-on-hyak.md)):

- **Data.** `EXPERTS` set; every deployment the cache holds (a NoSidewalk-only pull of the public ones by default; point
  `CACHE` at a full cache to include private ones, as round 3 did); `--verdict-unit face --max-labels-per-face 5`.
- **Split.** Face-grouped, `tucson-az` (round 3's public held-out city) and `chicago-il` held out together
  (`HOLDOUT=tucson-az,chicago-il`, the launcher's default): after the cap, 78 incorrect labels on 37 faces and 475
  correct labels in `test_heldout`; training keeps 320 of the 398 negatives.
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
  73) is five incorrect faces, so the default is `tucson-az,chicago-il` (78 / 37 held out, 15 % of the negatives taken
  from training). Jon can override with `HOLDOUT=` on the launcher, e.g. `HOLDOUT=chicago-il` alone; amsterdam (41
  incorrect, 6 correct) is no better a choice.
- The cap of 5 shrinks the correct class most (12,137 to 4,909) but the 55:45 downsample would cut it to about 486
  anyway; it matters for the incorrect class (530 to 398).

## Results

<!-- pilot-results:start -->
_Not yet run. `analysis/nosidewalk_pilot.py --doc docs/nosidewalk-2026-11.md` fills this section: per arm and split,
AUC, within-city AUC and the Agree gate's coverage and precision (mean and range over seeds), and per run the
face-bootstrap AUC interval and the gate precision's Wilson lower bound._
<!-- pilot-results:end -->

Placeholders to fill with the run: the per-city tables and the type x severity and type x tag slices
(`<arm>/NoSidewalk/per_city.md` and `metrics.json` from `evaluate.py`, with training counts), the job ids, and the
crop counts per arm.

## Provenance

- rawLabels: NoSidewalk- and Signal-only pull of the 45 public deployments listed by `/v3/api/cities`, 2026-11-05
  (`labelType` query parameter); public/private resolved from the live listing when the census ran.
- Census: `analysis/type_readiness.py --pattern '{city}__{label_type}.csv' --label-type NoSidewalk --label-type
  Signal` (that pull's file layout; `fetch_all.py --label-types` writes `<city>.csv`) with 2 expert accounts in
  `EXPERTS`, at commit b932358 (the code commit of this change).
- `readiness.json` sha256 `a4ce2b2ba24fd1f22a131ed32703f8f8ce509c367abd5529f729399a9bcd7823`.
- Seeds: selection and cap 42; pilot training 42, 43, 44.
