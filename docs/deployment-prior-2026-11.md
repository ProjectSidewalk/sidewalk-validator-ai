# Deployment prior: every vote rule recorded, the ≥2-vote majority by default (November 2026)

Issue [#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56). Report only: this measures what
the choice of prior does and makes `majority2` the default for the NEXT `data_generation.py` run. No served map is
refitted or published here, and sidewalk-ai-api is untouched; the refit is a separate PR once these numbers are read.

**Short version.** `deployment_priors.json` was counted under the training margin rule, which is asymmetric
(`incorrect` at net human Disagree > 1, `correct` only at net Agree > 2), so it put the incorrect share 10–15 points
too high for NoCurbRamp, Obstacle and SurfaceProblem. On one population (the round-3 rawLabels), the plain
≥ 2-vote human majority gives 4.4 / 6.8 / 22.5 / 25.4 / 17.5% incorrect against the file's 6.1 / 11.8 / 37.4 / 35.8 /
32.0%. Refitting the round-3 maps under the majority prior changes one operating point a lot (NoCurbRamp Agree
coverage 0.44 → 0.71 on the test split) and raises the Agree precision the maps claim for every type, by 0.002–0.023 on the test split
(0.001–0.057 on the unseen deployments). The round-3 Agree votes at the gate are confirmed by the human majority
99.8% (Crosswalk), 99.8% (CurbRamp), 100% (NoCurbRamp), 98.7% (SurfaceProblem) and 94.5% (Obstacle, 95% interval
0.904–0.969) of the time. For Obstacle both maps sit at or above the top of that interval (A claims 0.959 / 0.970,
D 0.974 / 0.981 on test / unseen deployments), and the A−D gap (about 0.015) is smaller than the interval's width
(about 0.065), so the production data do not separate the two priors there; they say Obstacle's maps, under either
prior, may be optimistic. NoCurbRamp Disagree, which cleared the gate
only under the margin prior, never clears under any of the other three; the Disagree hold-back stays (decision D3).

## The problem

`calibration/calibrate.py` reweights the val split from the training mix (55:45) to `deployment_priors.json` before
it fits the confidence → precision maps, and the maps decide which verdicts clear the 0.92 gate. That prior was
written by `data_generation.deployment_prior_entry` from the labels the TRAINING rule selects, and the training rule
is asymmetric. A label with two Disagree votes and no Agree is `incorrect`; one with two Agree votes and no Disagree is
undecided. The rule was built to select confident training targets, not to estimate how often a label is wrong, and
it counts the thin Disagree margins that it refuses on the Agree side. "Symmetric" has also meant two different rules
in this repo, so the four rules are named apart here.

## The four rules, on one population

| Rule | Name in code (`data_generation.PRIOR_RULES`) | correct | incorrect |
|---|---|---|---|
| A | `margin` (the training rule, unchanged) | net Agree > 2 | net Disagree > 1 |
| B | `symmetric3` (what `analysis/confident_learning.py` calls `symmetric`) | net Agree > 2 | net Disagree > 2 |
| C | `symmetric2` | net Agree ≥ 2 | net Disagree ≥ 2 |
| D | `majority2` (**the new default prior**) | ≥ 2 Agree/Disagree votes, more Agree | ≥ 2 Agree/Disagree votes, more Disagree; ties out |

Votes are human votes only (`validator_type == "Human"`), Unsure is not counted, and only human-authored labels with a
usable panorama point count (`data_generation.voted_labels`), as for the old prior. The expert rule (one expert
vote deciding a label) never enters any of them; an expert's vote is tallied as one human vote like any other.

Incorrect share (correct / incorrect labels), round-3 rawLabels of the 54 deployments in the round-3 prior entry
([`deployment-prior-2026-11/shares.md`](deployment-prior-2026-11/shares.md)):

| Rule | Crosswalk | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem |
|---|---:|---:|---:|---:|---:|
| A margin | 6.1% (19,671 / 1,281) | 11.8% (28,470 / 3,797) | 37.4% (13,030 / 7,774) | 35.8% (12,247 / 6,822) | 32.0% (13,790 / 6,484) |
| B symmetric3 | 3.8% (19,671 / 771) | 4.2% (28,470 / 1,255) | 26.9% (13,030 / 4,790) | 15.8% (12,247 / 2,304) | 11.5% (13,790 / 1,792) |
| C symmetric2 | 3.8% (32,628 / 1,281) | 5.2% (68,886 / 3,797) | 20.7% (29,864 / 7,774) | 22.1% (24,068 / 6,822) | 14.5% (38,080 / 6,484) |
| D majority2 | 4.4% (33,136 / 1,524) | 6.8% (72,228 / 5,291) | 22.5% (32,677 / 9,479) | 25.4% (28,084 / 9,555) | 17.5% (42,761 / 9,101) |

**Hard check: rule A reproduces the round-3 `deployment_priors.json` exactly** (all ten counts and the five `n_rows`),
so the four rows differ only in how they read the votes. The same four rules on every label the production AI has
assessed (the 2026-11-11 pull, below) give almost the same shares (A 6.1 / 12.0 / 35.8 / 37.2 / 32.2%, D 4.4 / 6.9 /
22.6 / 25.8 / 18.2%; [`observed.md`](deployment-prior-2026-11/observed.md)), which repeats the finding of the
2026-10-17 pull in [`deploy-checklist-2026-10.md`](deploy-checklist-2026-10.md): the gap is the rule, not the
population. The plan's first draft of this table mixed those sources (B from the confident-learning run, C and D from
the 2026-10-17 pull); the table above puts all four on one population; B is unchanged and C and D move by at most 0.8 points.

Why D and not B or C (decision D1). D is what the website treats as a settled label, so it is the prior of the label
mix the AI is asked about. B drops thin Disagree margins but keeps the thin-Agree exclusion, so it inverts the old bias
on Obstacle and SurfaceProblem (15.8% and 11.5%). C and D are close; D decides 2–22% more labels (the 2–1 and 3–2 splits; most for Obstacle and
SurfaceProblem), which is why its incorrect shares sit 0.6–3.3 points above C's.

## The gate, before and after

Each cell: Agree coverage / precision (gated n) at 0.92, of an uncapped map fitted on `val` and reweighted to the
rule's prior (`calibrate.calibrate(pred, pi, "val", split)`), as in round 3. **The margin rows reproduce every
round-3 `calibration_report.json` exactly** (test and `eval-heldout`, both verdicts, coverage, precision and gated
n; `gate --check-reports`). "Observed" is from production, next section. All four rules in
[`gate.md`](deployment-prior-2026-11/gate.md).

| Type | Agree test, A | Agree test, D | Agree unseen deployments, A | Agree unseen, D | Observed, round 3 (served map = A), confirmed by majority2 |
|---|---:|---:|---:|---:|---:|
| Crosswalk | 0.991 / 0.995 (255) | 0.992 / 0.997 (255) | 0.864 / 0.996 (56) | 0.865 / 0.997 (56) | 0.998 [0.992, 0.999] (885) |
| CurbRamp | 0.883 / 0.994 (692) | 0.886 / 0.997 (692) | 0.751 / 0.991 (198) | 0.755 / 0.995 (198) | 0.998 [0.991, 0.999] (834) |
| NoCurbRamp | 0.440 / 0.973 (627) | **0.708** / 0.983 (984) | 0.372 / 0.933 (158) | **0.630** / 0.950 (260) | 1.000 [0.971, 1.000] (129) |
| Obstacle | 0.475 / 0.959 (631) | 0.494 / 0.974 (631) | 0.489 / 0.970 (325) | 0.520 / 0.981 (325) | **0.945 [0.904, 0.969]** (199) |
| SurfaceProblem | 0.676 / 0.957 (819) | 0.696 / 0.980 (819) | 0.528 / 0.890 (178) | 0.554 / 0.947 (178) | 0.987 [0.962, 0.995] (226) |

Disagree (uncapped, for the record; it stays held back):

| Type | test, A | test, B / C / D | unseen, A | unseen, D |
|---|---:|---:|---:|---:|
| Crosswalk | 0.422 / 1.000 (98) | 0.370 / 0.370 / 0.388, all at 1.000 (98) | 0.319 / 0.643 (32) | 0.296 / 0.559 (32) |
| NoCurbRamp | 0.206 / 0.978 (249) | never clears | 0.056 / 1.000 (17) | never clears |
| CurbRamp, Obstacle, SurfaceProblem | never clears | never clears | never clears | never clears |

Reading:

- **Only NoCurbRamp's Agree gate moves.** For the other four types the D map gates exactly the same verdicts as the
  A map (same gated n on both splits); only the precision it claims for them rises. For NoCurbRamp the gate lets
  through 57% more Agree verdicts on test (627 → 984) and 65% more on the unseen deployments.
- **NoCurbRamp Disagree exists only under rule A.** Its 21% coverage came from the inflated incorrect share. Crosswalk
  Disagree clears under every prior and is 0.52–0.64 precise on the unseen deployments (one false Disagree in 32,
  reweighted to a 4–6% incorrect share), so the hold-back is needed whichever prior is used (D3).
- **The caveat.** `calibrate.py` reweights val classes as the MARGIN rule defines them, and val has no thin-margin
  labels at all (the margin rule left them out of the dataset). Those are the hardest labels, and in production the
  validator sees them, so whichever prior is chosen the map's precision is an upper bound on what production sees.
  The sound map is one fitted on the revision's own production outcomes (`calibrate_from_production.py`).

## Observed: what production says (decision D4)

From the fresh production pull `ai_assessments_2026-11-11.csv` (`analysis/pull_ai_assessments.sh`; 1,045,714 rows
in 53 schemas, AI assessment dates 2025-08-13 .. 2026-11-11). It holds private-city rows and stays outside the repo;
only the aggregates in [`observed.md`](deployment-prior-2026-11/observed.md) / `observed.json` are committed. Backup
schemas (`_bak_` in the name; none had assessments), and 29,554 deleted labels were dropped; labels are keyed by
`(schema, label_id)`, unique in the pull.

**Separating round-3 votes.** The pull's `model_id` names the type's model repo only, with no revision, so the
revision is told apart by date: the round-3 revisions went live on 2026-10-27 at about 12:36 PDT (sidewalk-ai-api
1.2.0), and the pull stores the assessment's date only. Round 3 = assessments dated 2026-10-28 or later (23,932 labels);
older models = before 2026-10-27 (990,298, after dropping one row whose label type differs from the model's); the changeover day itself (1,929) is in neither group. The split is
confirmed by replaying the round-3 confidences through the round-3 Agree maps: for every one of the 19,565 round-3
Agree rows the map's lookup equals the served accuracy exactly (`served_check` in `observed.json`). The pull keeps each
label's LATEST assessment, so a label the old model assessed and round 3 re-assessed counts as round 3.

**Observed column.** For AI Agree votes at served accuracy ≥ 0.92, the share the human majority confirms, with its
95% Wilson interval and the number of labels the rule decides. Round 3 under every rule:

| Type | gated Agree | with a decisive human vote | A margin | B symmetric3 | C symmetric2 | D majority2 |
|---|---:|---:|---:|---:|---:|---:|
| Crosswalk | 1,911 | 1,147 | 0.998 (630) | 1.000 (629) | 0.999 (864) | 0.998 [0.992, 0.999] (885) |
| CurbRamp | 10,980 | 3,629 | 0.994 (154) | 1.000 (153) | 0.999 (821) | 0.998 [0.991, 0.999] (834) |
| NoCurbRamp | 511 | 411 | n < 30 (21) | n < 30 (21) | 1.000 (122) | 1.000 [0.971, 1.000] (129) |
| Obstacle | 799 | 400 | 0.974 (78) | 1.000 (76) | 0.989 (179) | 0.945 [0.904, 0.969] (199) |
| SurfaceProblem | 1,883 | 736 | 0.962 (53) | 1.000 (51) | 0.991 (212) | 0.987 [0.962, 0.995] (226) |

The older models since August 2025, for comparison (D, majority2): Crosswalk 0.996 (23,095 decided), CurbRamp 0.992
(60,944), NoCurbRamp 0.954 (28,323), SurfaceProblem 0.965 (22,970); the old Obstacle map gated only 7 Agree votes.

**Replay under the D map.** The same round-3 Agree confidences served through the D-prior map: for four types it
gates exactly the verdicts the served map gated. For NoCurbRamp it would gate 987 instead of 511, and the majority
confirms 0.996 [0.979, 0.999] of those 265 the rule decides, so the extra coverage held up on the outcomes there are.

What the observed column can and cannot say:

- It is measured on the labels with at least two decisive human votes, 8–46% of the gated votes depending on the
  type, two weeks in; which labels get votes is not random. The votes may also predate the AI's (the pull does not
  time them), and a voter may see the AI's Agree. These limits make it an encouraging sign, not a precision estimate.
- Each rule reads the same votes differently, so the columns are not independent estimates. B confirms 1.000
  everywhere because it calls a label incorrect only at three net Disagree votes, which no gated Agree label has yet;
  D decides the most labels, 2–1 splits included, so it is the noisiest reading and the one the prior now uses.
- **Obstacle is the exception, but it does not pick between A and D.** Under D, 0.945 [0.904, 0.969] (199 decided):
  the A map's claims (0.959 test, 0.970 unseen deployments) and the D map's (0.974, 0.981) for the same verdicts all
  sit at or above the top of the interval, and the A−D gap (about 0.015) is smaller than the interval's width (about
  0.065). Read under rule A instead, the observed rate is 0.974 (78 decided), equal to D's test claim. The shortfall is
  thin-margin: 9 of the 11 unconfirmed labels are 1 Agree / 2 Disagree splits (the other 2 are 0 / 2), and the
  unanimous decided labels are confirmed 166 of 168. It is even across severities (0.946 / 0.942 / 0.937 at 1 / 2 /
  3). Jon's decision (2026-11-12): Obstacle takes the `majority2` prior like the other four types; it is the first
  type to refit on production outcomes (`calibrate_from_production.py`) once enough human votes have accumulated.
- **Untested caveat: queue settling.** If SidewalkWebpage counts the AI's vote toward settling a label, then a label
  with an AI Agree and one human Disagree stays in the validation queue and draws more votes, while one with an AI
  Agree and a human Agree settles and leaves it. Contested labels would then be over-represented among the decided
  ones, which pulls every observed share down, the most where the AI and the crowd disagree most (Obstacle). This
  pull cannot test it (the votes are not timed and the queue rule was not checked).

## Slices

Agree at the gate per type × severity and type × tag, A map (baseline) against D map, with the training rows of each
slice (correct / incorrect): [`slices.md`](deployment-prior-2026-11/slices.md) (test and unseen deployments, every
tag). Rates are blank under 30 Agree verdicts in the slice. Test split, by severity:

| Type | severity | Agree verdicts | A: coverage / precision | D: coverage / precision | train (c / i) |
|---|---|---:|---:|---:|---:|
| Crosswalk | 1 | 234 | 0.991 / 0.997 | 0.991 / 0.998 | 1,073 / 321 |
| Crosswalk | 2, 3, none | 13, 3, 8 | | | 44 / 143, 12 / 357, 27 / 112 |
| CurbRamp | 1 | 658 | 0.893 / 0.994 | 0.895 / 0.997 | 3,062 / 2,021 |
| CurbRamp | 2 | 75 | 0.833 / 0.985 | 0.838 / 0.992 | 377 / 460 |
| CurbRamp | 3 | 28 | | | 144 / 407 |
| CurbRamp | none | 53 | 0.910 / 0.996 | 0.914 / 0.998 | 212 / 242 |
| NoCurbRamp | 1 | 115 | 0.341 / 0.920 | 0.581 / 0.956 | 614 / 1,225 |
| NoCurbRamp | 2 | 445 | 0.415 / 0.975 | 0.692 / 0.977 | 2,044 / 1,551 |
| NoCurbRamp | 3 | 845 | 0.469 / 0.979 | 0.734 / 0.990 | 3,863 / 2,211 |
| NoCurbRamp | none | 49 | 0.381 / 0.916 | 0.664 / 0.948 | 217 / 474 |
| Obstacle | 1 | 310 | 0.419 / 0.956 | 0.456 / 0.973 | 1,382 / 1,760 |
| Obstacle | 2 | 418 | 0.511 / 0.941 | 0.529 / 0.963 | 1,818 / 1,026 |
| Obstacle | 3 | 579 | 0.496 / 0.976 | 0.505 / 0.985 | 2,627 / 1,272 |
| Obstacle | none | 63 | 0.301 / 0.914 | 0.324 / 0.945 | 270 / 399 |
| SurfaceProblem | 1 | 587 | 0.647 / 0.958 | 0.668 / 0.981 | 2,641 / 2,409 |
| SurfaceProblem | 2 | 377 | 0.713 / 0.953 | 0.728 / 0.978 | 1,768 / 1,191 |
| SurfaceProblem | 3 | 241 | 0.700 / 0.967 | 0.721 / 0.985 | 1,191 / 642 |
| SurfaceProblem | none | 49 | 0.604 / 0.910 | 0.635 / 0.957 | 239 / 419 |

The NoCurbRamp coverage gain is spread evenly over severities (+0.24 to +0.28) and tags (+0.20 to +0.28), so the D map
does not open the gate for one kind of label only. Severity-1 NoCurbRamp and the unrated labels are where the A map
was weakest (0.920, 0.916); about two thirds of their training rows are incorrect.

How wrong a type's labels are depends strongly on severity under every rule
([`shares.md`](deployment-prior-2026-11/shares.md) has the tags too). Under D: Crosswalk 1.8 / 15.8 / 57.7% incorrect
at severity 1 / 2 / 3, NoCurbRamp 42.0 / 20.4 / 17.7%, Obstacle 37.6 / 20.7 / 17.8%, SurfaceProblem 18.9 / 16.0 /
13.1%, CurbRamp 5.4 / 11.0 / 21.6%. Rule A overstates the incorrect share at every severity (e.g. Obstacle 53.3 / 28.9 / 23.4% under A
against 37.6 / 20.7 / 17.8% under D). A per-severity prior is a possible
refinement for the refit PR; it is not done here.

## What does NOT change

- **The training set.** `eligible_labels` still selects under the margin rule (A), byte for byte as before (a test
  checks the refactor against the old code). The selection hash of a re-run is unchanged; `--prior-rule` is not a
  resume setting.
- **The served maps.** Nothing is refitted, published or pinned. The round-3 maps (A prior, Disagree held back) stay
  live; sidewalk-ai-api is untouched.
- **The Disagree hold-back** stays (D3): Crosswalk Disagree is not trustworthy on unseen deployments under any prior.
- **NoSidewalk** keeps its face-size prior (`calibration/nosidewalk_prior.py`), which overwrites its entry.
- **Old prior files** keep working: `calibrate.load_prior` reads a file without `rules` as the margin prior, and
  `retrain_report.correct_weight` (which divides by margin-rule selections) reads `rules.margin` when present.

## Code

- `data_generation.py`: `voted_labels` (geometry, author, human tallies) is split out of `eligible_labels`;
  `PRIOR_RULES`, `prior_class(agree, disagree, rule)` and `prior_rule_counts`; `deployment_prior_entry(...,
  voted=, rule=)` writes `rule`, `rules: {rule: {correct, incorrect}}` and the chosen rule's counts at the top level
  (without `voted` its output is unchanged); `--prior-rule` (default `majority2`).
- `calibration/calibrate.py`: `--prior-rule` / `load_prior(rule=)` read `rules[rule]`; `calibration_report.json`
  records `prior_rule`.
- `analysis/retrain_report.py`: `correct_weight` uses the margin counts, and the gate tables reweight to the prior
  the map was fitted under (`prior_rule` in its `calibration_report.json`, `margin` when absent), not to the file's top
  level.
- `hf_cards.py`: the dataset and model cards name the rule the prior was counted under (or NoSidewalk's face-size
  method).
- `analysis/prior_rules.py`: this report (`shares`, `gate`, `slices`, `observed`).

## Replicating

No random draws: the counts are deterministic and the isotonic fits have no seed. The round-3 predictions come from
the split with seed 42 ([`retrain-2026-10.md`](retrain-2026-10.md)).

| Input | Path (Jon's Mac) | sha256 |
|---|---|---|
| rawLabels, 54 deployments | `~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels/` | `966e7391c77a6007…` (over the sorted per-file sha256 lines; `shares.json`) |
| round-3 prior | `…/round3/deployment_priors.json` | `80d4e685bbce5006…` |
| predictions, split | `…/round3/<Type>/predictions.csv`, `split.csv` | per type in `gate.json` and `slices.json` |
| production pull, 2026-11-11 (PRIVATE) | `~/Data/sidewalk-validator-ai-prior-rule-2026-11/ai_assessments_2026-11-11.csv` | `6891a1c150685975ca4b45136e2eaefd9c3401d20923ce6f6eb7dd53ae71db64` |

```bash
R=~/Data/sidewalk-validator-ai-retrain-2026-10/round3
O=docs/deployment-prior-2026-11
A=~/Data/sidewalk-validator-ai-prior-rule-2026-11/ai_assessments_2026-11-11.csv
python analysis/prior_rules.py shares --rawlabels-cache $R/rawlabels --priors $R/deployment_priors.json --out $O
python analysis/prior_rules.py gate --data-dir $R --shares $O/shares.json --check-reports --out $O
python analysis/prior_rules.py slices --data-dir $R --shares $O/shares.json --out $O
python analysis/prior_rules.py observed --assessments $A --data-dir $R --shares $O/shares.json --out $O
```

`shares` exits non-zero if rule A does not reproduce the prior file, and `gate --check-reports` if the A rows do not
reproduce the round-3 calibration reports. Run on a CPU (pandas 2.3.3, scikit-learn 1.9.1), about 40 s in all.

## Departures from the plan

The plan (by Fable 5.2) was followed with these changes, each for the reason given:

- The incorrect share per rule by severity and tag is computed by `shares`, not `slices`, so the rawLabels are read
  once; `slices` covers the gate.
- `observed` also replays the round-3 confidences through each prior's map. That makes the D map's NoCurbRamp gate
  measurable on outcomes, and the exact match with the served accuracy confirms the date split. Wilson intervals were
  added because the decided counts are small.
- `load_prior(rule=)` raises `KeyError` for a rule the entry does not record, instead of falling back to the top level
  under a different rule, which would silently mislabel the report. It names NoSidewalk's `face_size` prior as such.
- `eligible_labels` takes `voted=` so `main` parses the votes once.
- The model card names the rule as well as the dataset card does.
- The plan's starting table mixed populations; C and D move by up to 0.8 points on the one population used here.

## Follow-ups

- **The refit PR.** Regenerate `deployment_priors.json` with `data_generation.py --select-only` (it now writes every
  rule, `majority2` on top), refit all five maps under `majority2` with the Disagree hold-back (Jon's decision, 2026-11-12: every type,
  Obstacle included). NoCurbRamp gains the most and its outcomes support it.
- **Obstacle on production outcomes.** Obstacle is the first type to refit with `calibrate_from_production.py` once
  round 3 has enough decided human votes; that map then supersedes its `majority2` one.
- **sidewalk-ai-api re-pin**, after the refit publishes new map revisions.
- The second box of #56 (should the TRAINING rule be symmetric?) is a retrain question and stays open; the
  confident-learning run ([`confident-learning-2026-10.md`](confident-learning-2026-10.md)) found that rule B alone,
  without rebalancing, costs AUC.
