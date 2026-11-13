# Per-deployment status of the AI's votes: when has the AI been checked in a deployment? (November 2026)

Part of [#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52) (option 1, shadow mode until checked),
for [SidewalkWebpage#5832](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5832) (per city, type and verdict
mode `off` / `shadow` / `vote`). Related: [#15](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/15)
(per-city calibration), [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42) (NoSidewalk).
Tool: [`analysis/deployment_gate.py`](../analysis/deployment_gate.py). Generated public files:
[`deployment-gate-2026-11/`](deployment-gate-2026-11/).

## Short version

- **This is a status report, not a switch.** For every deployment, label type and verdict it says how the AI's gated
  assessments fared against the human reference under the models serving now. It emits no serving mode for any
  deployment: whether existing deployments keep voting while they build evidence is an open decision (below). A
  private file, kept outside the repository, records the mode each stratum would get under each candidate policy, for
  comparison; nothing applies it, and a person would apply any change to the website's city settings.
- **Under the round-3 models, per-deployment evidence exists in one deployment only.** Two weeks after the round-3
  revisions went live, almost all decided gated rows come from one public deployment (headline below). Four of its
  five Agree strata pass; Obstacle is undecided. Every other deployment has at most a handful of decided rows. The
  reason is structural: a label is assessed once (the website skips assessed labels), so current-era evidence comes
  only from labels assessed since the switch, and humans have voted on few of them yet.
- **Disagree is `no_gated` everywhere**: the round-3 Disagree maps are capped below 0.92, so no Disagree verdict is a
  vote today. In the replay of the previous models (below), no Disagree stratum in any deployment would ever have
  passed: pooled per type, their gated Disagree votes were right between 51% and 71% of the time.
- **Every extra look raises the chance of a lucky pass, so the rule looks only at fixed checkpoints.** At the default bar (n >= 50, Wilson
  95% lower bound >= 0.92, checkpoints 50, 100, 200, ...), a deployment whose true precision is exactly 0.92 passes at
  some checkpoint by n = 2,000 with probability 0.075, not 0.025; re-running the check at every new vote raises that
  to 0.18. A z of 2.09 instead of 1.96 holds the checkpoint schedule to 5% (power table below).
- **Which reference is used matters.** In vote mode, the AI's Agree plus one human Agree settles a label, so labels
  confirmed by a single human never reach two votes and the default `majority2` reference never counts them, while
  contested labels are sent for more votes. The `any_vote` sensitivity (at least one human vote) is reported beside
  every stratum, and shadow-mode and vote-mode evidence are kept apart once the pull records which is which.
- **A `pass` means "its Agree votes are rarely wrong there", not "the model transfers there".** A deployment where
  most labels are correct can pass with a model that agrees with everything; each stratum reports the deployment's
  own base rate beside its precision.

<!-- BEGIN GENERATED: headline (analysis/deployment_gate.py) -->

- Pull `ai_assessments_2026-11-11.csv` (sha256 `6891a1c150685975`), AI dates 2025-08-13 .. 2026-11-11; era key: date (no validator_revision in the pull): current >= 2026-10-28, previous < 2026-10-27.
- Current-era strata (deployment x type x verdict): 135; by status: pass 4, undecided 1, insufficient 60, no_gated 70.
- 99.5% of the current era's decided gated rows come from one deployment (seattle-wa); 5 deployments have any.
- Pooled current-era decided gated Agree (majority2) after the re-label exclusions, share confirmed in brackets: Crosswalk 885 (0.998), CurbRamp 834 (0.998), NoCurbRamp 129 (1.000), Obstacle 186 (0.946), SurfaceProblem 226 (0.987).
- Acceptance check (the same counts before exclusions, against `docs/deployment-prior-2026-11/observed.md`): Crosswalk 885, CurbRamp 834, NoCurbRamp 129, Obstacle 199, SurfaceProblem 226 -- matches.

<!-- END GENERATED: headline -->

## The rule

A **stratum** is one deployment x label type x verdict (Agree, Disagree) x evidence regime (`vote`, `shadow`, or
`unrecorded` when the pull does not say), in one model era.

- **Gated assessment**: the AI's latest assessment of a label, verdict Agree or Disagree, served accuracy (the recorded
  `ai_accuracy`) at least 0.92 (`calibration._isotonic_map.GATE`, the website's `ai-validation-min-accuracy`).
- **Decided** (`majority2`, the default reference, the rule adopted for the deployment prior in
  [#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)): at least two human Agree or Disagree votes
  on the label, not tied. The AI's own vote is excluded in the pull's SQL and Unsure votes are not counted.
  `data_generation.prior_class` gives the class of the **label** (`correct` = the humans agree with the label). An AI
  **Agree** is right when the label is `correct`; an AI **Disagree** is right when the label is `incorrect`.
- **Checkpoints**: a stratum is evaluated on its first C decided rows, in assessment order (`ai_date`, then
  `label_id`, within one schema, so the per-city `label_id` is never compared across deployments), where C is the
  largest of 50, 100, 200, 400, 800, 1600, 3200, ... not above its decided count. Rows past C wait for the next
  checkpoint.
- **Status** at checkpoint C with k right, Wilson 95% interval:

| status | condition |
|---|---|
| `pass` | C >= 50 and the lower bound >= 0.92 |
| `fail` | C >= 50 and the upper bound < 0.92 |
| `undecided` | C >= 50 and the interval contains 0.92 |
| `insufficient` | gated assessments exist, but fewer than 50 are decided |
| `no_gated` | assessments of this verdict exist, none at or above 0.92 (every held-back Disagree) |
| `never_votes` | the type's policy says this verdict never votes (NoSidewalk Disagree) |
| `not_served` | production evidence for a type the API does not serve |

- **Era**: the 2026-11-11 pull has no model revision, so the era is a date split: current = assessed on 2026-10-28
  or later (the first full day of the round-3 revisions), previous = before 2026-10-27, the changeover day in
  neither. The assessment table does record the revision (`validator_revision`, SidewalkWebpage evolution 420); this
  PR changes `analysis/pull_ai_assessments.sh` to select it, and with it the tool keys the era on each model's newest
  revision instead of a date. Status uses the current era only; the previous era's status is reported as context.
- **Every value is a flag** (`--min-n`, `--gate`, `--z`, `--schedule`, `--reference`, `--since`) and is recorded in
  `deployment_status.json`.

Per stratum the report also carries: `n_assessed`, `n_gated`, `n` (decided), `k`, the precision and interval at the
checkpoint, `precision_all` (all decided rows), `more_needed_if_perfect` (the fewest further decided rows, all right,
that would pass at a checkpoint), `base_rate` (the share of all decided AI-assessed labels of that deployment and type
whose human class matches the verdict) and `lift`, the status under the `tie_half` and `any_vote` references, the
previous era's status, whether the deployment has previous-era assessments of the type (`existing`), and `evidence`,
`evidence_ref`, `basis`, `min_n` and `model_revision`.

### The output contract

`deployment-gate-2026-11/deployment_status.json` (public) holds:

- `types`: every type with `served` and, per verdict, `eligible` or `never`. An unserved type is `off` on the website,
  not `shadow`: shadow needs the API to score the type.
- `deployments[city_id][type][verdict][regime]`: the stratum, public deployments only.
- `private`: pooled counts for private deployments per type and verdict; a cell is shown only when at least three
  private deployments contribute, otherwise it reads "fewer than 3".
- `summary`: strata by status, how many change status under each sensitivity reference, the largest one-deployment
  share of the decided rows, and the acceptance check.

It contains no `mode`. The private `deployment_status_all.json` (under `~/Data`, never committed) lists every
deployment by schema, with the mode each stratum would get under two policies: `strict` (vote only on `pass`) and
`grandfather` (an existing deployment keeps voting unless it `fail`s; a new one as `strict`).

## What the website needs (SidewalkWebpage#5832)

- A per city, type and verdict mode `off` / `shadow` / `vote`, with `shadow` the default for a new deployment.
- Shadow mode must score and store assessments even when tag suggestions are off, or there is nothing to measure.
- Human validators must still see the deployment's labels: the evidence is human votes on assessed labels.
- The pull must say whether a vote was cast for each assessment (`ai_vote_cast`, added here): the decided population
  differs between the modes (above), so the two are separate strata and are never pooled.
- Switching a stratum to `vote` is a person's decision, made from this report and applied to the city settings. When
  the switch happens, the stored assessments can back-fill votes (#5832's gap 4).
- Nothing changes for Disagree while its maps are held back below 0.92.

## Results today (2026-11-11 pull)

### Strata by status, current era

<!-- BEGIN GENERATED: status_counts (analysis/deployment_gate.py) -->

| Type | Verdict | pass | undecided | fail | insufficient | no_gated |
|---|---|---:|---:|---:|---:|---:|
| Crosswalk | Agree | 1 | 0 | 0 | 14 | 0 |
| Crosswalk | Disagree | 0 | 0 | 0 | 0 | 8 |
| CurbRamp | Agree | 1 | 0 | 0 | 15 | 0 |
| CurbRamp | Disagree | 0 | 0 | 0 | 0 | 16 |
| NoCurbRamp | Agree | 1 | 0 | 0 | 6 | 4 |
| NoCurbRamp | Disagree | 0 | 0 | 0 | 0 | 13 |
| Obstacle | Agree | 0 | 1 | 0 | 10 | 3 |
| Obstacle | Disagree | 0 | 0 | 0 | 0 | 12 |
| SurfaceProblem | Agree | 1 | 0 | 0 | 15 | 0 |
| SurfaceProblem | Disagree | 0 | 0 | 0 | 0 | 14 |

<!-- END GENERATED: status_counts -->

### Public deployments

<!-- BEGIN GENERATED: public_table (analysis/deployment_gate.py) -->

Public deployments' Agree strata with at least one decided row. Disagree is `no_gated` everywhere (the round-3 Disagree maps are capped below 0.92), so it is not shown. n = decided rows (checkpoint the status is evaluated at); precision [Wilson 95%] at the checkpoint; base rate = share of the deployment's decided AI-assessed labels of the type that humans call correct; more needed = further decided rows, all right, that would pass.

| Deployment | Type | Status | n (checkpoint) | precision [95%] | base rate | more needed | previous era |
|---|---|---|---:|---:|---:|---:|---|
| houston-tx | CurbRamp | insufficient | 1 (-) |  | 1.000 | 49 | insufficient |
| houston-tx | Obstacle | insufficient | 1 (-) |  | 1.000 | 49 | no_gated |
| houston-tx | SurfaceProblem | insufficient | 1 (-) |  | 0.444 | 99 | insufficient |
| la-ca | CurbRamp | insufficient | 1 (-) |  | 1.000 | 49 | insufficient |
| seattle-wa | Crosswalk | pass | 883 (800) | 0.998 [0.991, 0.999] | 0.908 | 0 | pass |
| seattle-wa | CurbRamp | pass | 831 (800) | 0.998 [0.991, 0.999] | 0.952 | 0 | pass |
| seattle-wa | NoCurbRamp | pass | 129 (100) | 1.000 [0.963, 1.000] | 0.853 | 0 | pass |
| seattle-wa | Obstacle | undecided | 184 (100) | 0.970 [0.915, 0.990] | 0.698 | 216 | insufficient |
| seattle-wa | SurfaceProblem | pass | 222 (200) | 0.995 [0.972, 0.999] | 0.860 | 0 | undecided |
| st-louis-mo | SurfaceProblem | insufficient | 1 (-) |  | 0.000 | 99 | pass |
| walla-walla-wa | Crosswalk | insufficient | 2 (-) |  | 1.000 | 48 | pass |
| walla-walla-wa | CurbRamp | insufficient | 1 (-) |  | 1.000 | 49 | pass |
| walla-walla-wa | Obstacle | insufficient | 1 (-) |  | 0.000 | 99 | no_gated |
| walla-walla-wa | SurfaceProblem | insufficient | 2 (-) |  | 1.000 | 48 | pass |

Another 54 public Agree strata in 18 deployments have current-era assessments but no decided row (insufficient 47, no_gated 7); each needs at least 50 decided rows. Their previous-era status is in `deployment_status.json`.

<!-- END GENERATED: public_table -->

### Private deployments (pooled)

<!-- BEGIN GENERATED: private_counts (analysis/deployment_gate.py) -->

Private deployments with a current-era assessment: fewer than 3. Per type and verdict, a cell is shown only when at least 3 private deployments contribute.

| Type | Verdict | deployments | strata by status | decided n |
|---|---|---|---|---:|
| CurbRamp | Agree | fewer than 3 | suppressed | suppressed |
| CurbRamp | Disagree | fewer than 3 | suppressed | suppressed |
| NoCurbRamp | Agree | fewer than 3 | suppressed | suppressed |
| NoCurbRamp | Disagree | fewer than 3 | suppressed | suppressed |
| Obstacle | Agree | fewer than 3 | suppressed | suppressed |
| Obstacle | Disagree | fewer than 3 | suppressed | suppressed |
| SurfaceProblem | Agree | fewer than 3 | suppressed | suppressed |
| SurfaceProblem | Disagree | fewer than 3 | suppressed | suppressed |

<!-- END GENERATED: private_counts -->

### The reference: `majority2` against `tie_half` and `any_vote`

<!-- BEGIN GENERATED: sensitivity (analysis/deployment_gate.py) -->

Strata whose status differs from the binding `majority2` reference: `tie_half` 0, `any_vote` 2. Public Agree strata where any reference has at least one decided row; cell: status (decided n), and for `any_vote` the precision at the checkpoint.

| Deployment | Type | majority2 | tie_half | any_vote |
|---|---|---|---|---|
| chicago-il | Crosswalk | insufficient (0) | insufficient (0) | insufficient (1) |
| chicago-il | CurbRamp | insufficient (0) | insufficient (0) | insufficient (9) |
| chicago-il | SurfaceProblem | insufficient (0) | insufficient (0) | insufficient (1) |
| houston-tx | CurbRamp | insufficient (1) | insufficient (1) | insufficient (26) |
| houston-tx | Obstacle | insufficient (1) | insufficient (1) | insufficient (7) |
| houston-tx | SurfaceProblem | insufficient (1) | insufficient (1) | insufficient (16) |
| la-ca | Crosswalk | insufficient (0) | insufficient (0) | insufficient (14) |
| la-ca | CurbRamp | insufficient (1) | insufficient (1) | insufficient (26) |
| paterson-nj | CurbRamp | insufficient (0) | insufficient (0) | insufficient (2) |
| seattle-wa | Crosswalk | pass (883) | pass (885) | pass (1,100, 0.994) |
| seattle-wa | CurbRamp | pass (831) | pass (833) | pass (3,444, 0.974) |
| seattle-wa | NoCurbRamp | pass (129) | pass (134) | pass (381, 0.980) |
| seattle-wa | Obstacle | undecided (184) | undecided (195) | undecided (330, 0.910) |
| seattle-wa | SurfaceProblem | pass (222) | pass (227) | pass (649, 0.948) |
| st-louis-mo | Crosswalk | insufficient (0) | insufficient (0) | insufficient (17) |
| st-louis-mo | CurbRamp | insufficient (0) | insufficient (1) | insufficient (26) |
| st-louis-mo | NoCurbRamp | insufficient (0) | insufficient (0) | insufficient (21) |
| st-louis-mo | Obstacle | insufficient (0) | insufficient (0) | insufficient (23) |
| st-louis-mo | SurfaceProblem | insufficient (1) | insufficient (1) | undecided (62, 0.980) |
| walla-walla-wa | Crosswalk | insufficient (2) | insufficient (2) | insufficient (13) |
| walla-walla-wa | CurbRamp | insufficient (1) | insufficient (1) | pass (91, 1.000) |
| walla-walla-wa | NoCurbRamp | insufficient (0) | insufficient (0) | insufficient (4) |
| walla-walla-wa | Obstacle | insufficient (1) | insufficient (1) | insufficient (13) |
| walla-walla-wa | SurfaceProblem | insufficient (2) | insufficient (2) | insufficient (3) |

<!-- END GENERATED: sensitivity -->

## Replay on the previous models (vote-mode evidence, end-of-pull votes)

The rule replayed on the previous era, stratum by stratum: at each checkpoint, the status of the decided rows so far;
a stratum **graduates** at its first `pass`. Rows up to and including that checkpoint are what shadow mode would have
withheld; rows after it are what it would have let through. Counts only: the pull has no vote timestamps, so the
replay makes no claim about dates or waiting times. Items of the masked re-label rounds (their votes were cast in
production on items sampled by AI verdict) are excluded.

Limits: whether a row is decided, and its majority, uses votes as of the end of the pull, so an early checkpoint sees
votes that did not exist yet; contested labels are decided later, so the replay understates early false
graduation. Every previous-era vote was collected in vote mode, so this says nothing about how evidence accrues in
shadow mode. The previous era is not one model: API releases before 1.0.2 loaded the hub's main branch.

<!-- BEGIN GENERATED: backtest (analysis/deployment_gate.py) -->

Previous era, vote-mode evidence, votes as of the end of the pull; count-based replay. Checkpoints 50, 100, 200, 400, 800, 1600 then doubling. Cells: decided rows, precision [Wilson 95%].

| Type | Verdict | strata (n >= min) | final pass / undecided / fail | graduated | n at graduation (median, range) | withheld in graduated | kept after graduation | never graduated | strata under min |
|---|---|---:|---|---:|---|---|---|---|---|
| Crosswalk | Agree | 46 (24) | 23 / 1 / 0 | 23 | 50 (50-100) | 1,350, 0.996 [0.991, 0.998] | 21,333, 0.997 [0.997, 0.998] | 79, 0.975 [0.912, 0.993] | 301, 0.927 [0.892, 0.951] |
| Crosswalk | Disagree | 44 (13) | 0 / 3 / 10 | 0 |  | 0 | 0 | 1,956, 0.525 [0.502, 0.547] | 436, 0.383 [0.339, 0.429] |
| CurbRamp | Agree | 49 (31) | 29 / 1 / 1 | 29 | 50 (50-400) | 3,100, 0.980 [0.974, 0.984] | 57,260, 0.993 [0.992, 0.994] | 209, 0.871 [0.819, 0.910] | 313, 0.990 [0.972, 0.997] |
| CurbRamp | Disagree | 43 (18) | 0 / 0 / 18 | 0 |  | 0 | 0 | 7,817, 0.509 [0.498, 0.520] | 257, 0.370 [0.313, 0.430] |
| NoCurbRamp | Agree | 45 (21) | 7 / 12 / 2 | 8 | 200 (50-3200) | 4,350, 0.946 [0.938, 0.952] | 18,213, 0.972 [0.969, 0.974] | 5,314, 0.911 [0.903, 0.918] | 398, 0.789 [0.746, 0.826] |
| NoCurbRamp | Disagree | 46 (19) | 0 / 3 / 16 | 0 |  | 0 | 0 | 9,037, 0.713 [0.704, 0.722] | 306, 0.523 [0.467, 0.578] |
| Obstacle | Agree | 2 (0) | 0 / 0 / 0 | 0 |  | 0 | 0 | 0 | 4, 1.000 [0.510, 1.000] |
| Obstacle | Disagree | 35 (10) | 0 / 3 / 7 | 0 |  | 0 | 0 | 2,925, 0.687 [0.670, 0.703] | 335, 0.660 [0.607, 0.708] |
| SurfaceProblem | Agree | 46 (31) | 23 / 7 / 1 | 23 | 100 (50-1600) | 3,850, 0.963 [0.956, 0.968] | 11,916, 0.981 [0.979, 0.984] | 6,964, 0.937 [0.931, 0.942] | 183, 0.962 [0.923, 0.981] |
| SurfaceProblem | Disagree | 34 (10) | 0 / 0 / 10 | 0 |  | 0 | 0 | 5,385, 0.663 [0.650, 0.675] | 328, 0.543 [0.489, 0.596] |

<!-- END GENERATED: backtest -->

Reading. Under the previous models, the rule would have let Agree votes through in most deployments that had enough
human votes: Crosswalk and CurbRamp strata graduated at the first checkpoint (median n 50), SurfaceProblem at a median
of 100 and NoCurbRamp at 200. The votes let through after graduation were more precise than those withheld before it
(for example NoCurbRamp 0.972 kept against 0.946 withheld), and the strata that never graduated, or never reached 50
decided rows, held the weakest Agree votes (NoCurbRamp 0.911 and 0.789). Every Disagree stratum stayed in shadow:
none passed at any checkpoint. These are end-of-pull votes collected in vote mode; read them as what the rule
separates, not as how fast a new deployment would graduate.

## How many decided labels a deployment needs, and how often a lucky pass happens

Exact binomial arithmetic, no simulation. At n = 50 only a perfect 50 of 50 passes; with crowd noise that is
uncommon even for a good model (the single-look row at 0.98 and 0.99), so the realistic graduation n is in the
hundreds. At the pooled current-era precisions (0.945 for Obstacle up to 0.998 for Crosswalk) the table shows how
often a deployment would pass at each n.

<!-- BEGIN GENERATED: power (analysis/deployment_gate.py) -->

Rule: gate 0.92, z 1.96, min n 50, checkpoints 50, 100, 200, 400, 800, 1600 then doubling. Exact binomial, no simulation.

Errors allowed at n: 50: 0, 67: 1, 87: 2, 100: 2, 200: 8, 400: 21, 800: 48, 1600: 106.

Single look: the largest P(pass | true precision = gate) over n = 50..2000 is 0.0261.

| true precision | n=50 | n=67 | n=87 | n=100 | n=200 | n=400 | n=800 | n=1600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.88 | 0.002 | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.9 | 0.005 | 0.007 | 0.006 | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 |
| 0.92 | 0.015 | 0.026 | 0.026 | 0.011 | 0.018 | 0.022 | 0.019 | 0.022 |
| 0.94 | 0.045 | 0.084 | 0.100 | 0.057 | 0.147 | 0.307 | 0.538 | 0.865 |
| 0.95 | 0.077 | 0.146 | 0.184 | 0.118 | 0.327 | 0.646 | 0.913 | 0.998 |
| 0.96 | 0.130 | 0.246 | 0.319 | 0.232 | 0.593 | 0.915 | 0.997 | 1.000 |
| 0.97 | 0.218 | 0.399 | 0.514 | 0.420 | 0.850 | 0.995 | 1.000 | 1.000 |
| 0.98 | 0.364 | 0.612 | 0.747 | 0.677 | 0.980 | 1.000 | 1.000 | 1.000 |
| 0.99 | 0.605 | 0.855 | 0.943 | 0.921 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

P(EVER pass by n = 2,000) under each look schedule:

| looks | 0.88 | 0.9 | 0.92 | 0.94 | 0.95 | 0.96 | z holding p = gate to 5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| checkpoints | 0.002 | 0.008 | 0.075 | 0.889 | 0.999 | 1.000 | 2.09 |
| every 50 | 0.002 | 0.010 | 0.130 | 0.967 | 1.000 | 1.000 | 2.35 |
| every 10 | 0.003 | 0.017 | 0.165 | 0.972 | 1.000 | 1.000 | 2.44 |
| every row | 0.004 | 0.020 | 0.181 | 0.974 | 1.000 | 1.000 | 2.49 |

<!-- END GENERATED: power -->

## Severity and tag slices

The standing rule (every performance study reports type x severity and type x tag against a baseline, with training
counts). Pooled over deployments, so in the current era these are in effect one deployment's numbers (the `largest`
column). The baseline is the previous era. The slice that bears on the status is the last table: the severity mix
inside each stratum that passes or is undecided, so a stratum that passes overall but is weak at high severity is
visible.

<!-- BEGIN GENERATED: slices (analysis/deployment_gate.py) -->

Gated Agree rows decided under `majority2`, pooled over deployments; confirmed share blank under 30 decided; train = round-3 `split.csv` training rows (correct / incorrect); largest = the largest one-deployment share of the slice.
Tags joined from cached public rawLabels: 794,881 of 808,584 public-deployment rows have an export (private deployments are never joined; rows without one are `(not exported)`).

**Crosswalk by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 812 | 0.999 | 21,574 | 0.997 | 100% | 1073 / 321 |
| 2 | 42 | 0.976 | 746 | 0.987 | 100% | 44 / 143 |
| 3 | 11 |  | 163 | 0.963 | 100% | 12 / 357 |
| (none) | 20 |  | 580 | 0.986 | 100% | 27 / 112 |

**Crosswalk by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 567 | 0.998 | 5,193 | 0.998 | 100% | 922 / 635 |
| paint fading | 259 | 0.996 | 1,508 | 0.996 | 100% | 158 / 181 |
| broken surface | 64 | 1.000 | 408 | 0.998 | 100% | 32 / 26 |
| bumpy | 42 | 1.000 | 238 | 1.000 | 100% | 21 / 37 |
| very long crossing | 16 |  | 204 | 1.000 | 100% | 36 / 32 |
| uneven surface | 13 |  | 143 | 0.993 | 100% | 15 / 49 |
| brick/cobblestone | 12 |  | 40 | 1.000 | 100% | 5 / 43 |
| rail/tram track | 10 |  | 70 | 0.986 | 100% | 4 / 6 |
| (not exported) | 2 |  | 15,839 | 0.996 | 100% | 0 / 0 |

**CurbRamp by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 747 | 0.999 | 53,152 | 0.993 | 100% | 3062 / 2021 |
| 2 | 50 | 1.000 | 4,848 | 0.986 | 98% | 377 / 460 |
| 3 | 9 |  | 1,510 | 0.973 | 100% | 144 / 407 |
| (none) | 28 |  | 1,372 | 0.983 | 100% | 212 / 242 |

**CurbRamp by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| (none) | 654 | 0.997 | 32,806 | 0.995 | 100% | 2381 / 1552 |
| missing tactile warning | 74 | 1.000 | 7,213 | 0.986 | 100% | 770 / 712 |
| points into traffic | 69 | 1.000 | 4,282 | 0.993 | 100% | 382 / 388 |
| narrow | 35 | 1.000 | 2,104 | 0.987 | 100% | 234 / 217 |
| not enough landing space | 17 |  | 1,900 | 0.986 | 100% | 164 / 184 |
| surface problem | 17 |  | 1,683 | 0.988 | 100% | 176 / 175 |
| debris / pooled water | 16 |  | 581 | 0.993 | 100% | 56 / 18 |
| not level with street | 7 |  | 961 | 0.961 | 100% | 108 / 405 |
| (not exported) | 3 |  | 14,150 | 0.989 | 33% | 0 / 0 |
| steep | 0 |  | 403 | 0.973 |  | 49 / 177 |
| parallel lines | 0 |  | 121 | 1.000 |  | 12 / 2 |

**NoCurbRamp by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 19 |  | 2,226 | 0.911 | 100% | 614 / 1225 |
| 2 | 39 | 1.000 | 9,165 | 0.950 | 100% | 2044 / 1551 |
| 3 | 67 | 1.000 | 16,051 | 0.963 | 100% | 3863 / 2211 |
| (none) | 4 |  | 833 | 0.934 | 100% | 217 / 474 |

**NoCurbRamp by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 68 | 1.000 | 12,754 | 0.970 | 100% | 3339 / 1331 |
| (none) | 44 | 1.000 | 8,448 | 0.955 | 100% | 2117 / 2514 |
| alternate route present | 17 |  | 3,835 | 0.923 | 100% | 1129 / 1171 |
| (not exported) | 0 |  | 2,777 | 0.926 |  | 0 / 0 |
| unclear if needed | 0 |  | 785 | 0.892 |  | 246 / 614 |

**Obstacle by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 0.939 | 0 |  | 100% | 1382 / 1760 |
| 2 | 51 | 0.941 | 3 |  | 98% | 1818 / 1026 |
| 3 | 74 | 0.946 | 1 |  | 99% | 2627 / 1272 |
| (none) | 12 |  | 0 |  | 100% | 270 / 399 |

**Obstacle by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| narrow | 51 | 1.000 | 0 |  | 100% | 669 / 239 |
| trash/recycling can | 50 | 0.980 | 0 |  | 100% | 759 / 287 |

**SurfaceProblem by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 105 | 0.971 | 11,006 | 0.961 | 96% | 2641 / 2409 |
| 2 | 70 | 1.000 | 7,151 | 0.964 | 100% | 1768 / 1191 |
| 3 | 40 | 1.000 | 3,966 | 0.980 | 100% | 1191 / 642 |
| (none) | 11 |  | 790 | 0.943 | 100% | 239 / 419 |

**SurfaceProblem by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| cracks | 120 | 0.992 | 9,151 | 0.973 | 100% | 2245 / 786 |
| bumpy | 89 | 1.000 | 3,206 | 0.965 | 100% | 1178 / 804 |
| grass | 86 | 0.988 | 7,645 | 0.973 | 100% | 1898 / 738 |
| uneven/slanted | 46 | 1.000 | 3,867 | 0.972 | 100% | 1148 / 426 |
| height difference | 30 | 1.000 | 2,284 | 0.985 | 100% | 643 / 127 |
| narrow sidewalk | 23 |  | 1,567 | 0.939 | 100% | 605 / 555 |
| sand/gravel | 21 |  | 451 | 0.971 | 100% | 169 / 115 |
| very broken | 15 |  | 1,304 | 0.985 | 100% | 503 / 122 |
| (none) | 7 |  | 1,457 | 0.932 | 100% | 419 / 1071 |
| utility panel | 5 |  | 92 | 0.891 | 100% | 346 / 558 |
| (not exported) | 1 |  | 2,745 | 0.969 | 100% | 0 / 0 |
| construction | 1 |  | 48 | 1.000 | 100% | 24 / 20 |
| brick/cobblestone | 0 |  | 196 | 0.923 |  | 95 / 242 |
| debris | 0 |  | 113 | 0.956 |  | 45 / 21 |

**Severity within each current-era pass or undecided stratum** (public deployments)

| Deployment | Type | status | severity: n (confirmed) |
|---|---|---|---|
| seattle-wa | Crosswalk | pass | 1: 810 (0.999); 2: 42 (0.976); 3: 11 (); (none): 20 () |
| seattle-wa | CurbRamp | pass | 1: 745 (0.999); 2: 49 (1.000); 3: 9 (); (none): 28 () |
| seattle-wa | NoCurbRamp | pass | 1: 19 (); 2: 39 (1.000); 3: 67 (1.000); (none): 4 () |
| seattle-wa | Obstacle | undecided | 1: 49 (0.939); 2: 50 (0.960); 3: 73 (0.945); (none): 12 () |
| seattle-wa | SurfaceProblem | pass | 1: 101 (0.990); 2: 70 (1.000); 3: 40 (1.000); (none): 11 () |

<!-- END GENERATED: slices -->

## Evidence beyond production votes (NoSidewalk step (c))

NoSidewalk is not served, so it has no production evidence. Its step (c) evidence is a masked expert round
(`docs/nosidewalk-2026-11/round_b.*`, [#42](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42)): one
expert's verdict per label, about a fold ensemble rather than the single model that would be served. The schema holds
it without ingesting it here:

- `expert_round_stratum(k, n, rule, evidence_ref, model_revision, "NoSidewalk", "Agree", basis=..., min_n=...)` builds
  a stratum with `evidence: masked_expert_round`, evaluated once on all rows (a round is one look), with its own
  `min_n` (step (c) may want 100) and `basis` (`raw` or `prior_weighted`, the ruling on #42). `add_stratum(report,
  city_id, "NoSidewalk", "Agree", "expert_round", rec)` puts it into a report.
- NoSidewalk Disagree is `never_votes`; NoSidewalk is `served: false` until the API serves it, which reads `off`.
- `model_revision` names the model the evidence is about, so evidence about the fold ensemble is visibly not about the
  served model.
- Whether a pooled rule is admissible for NoSidewalk (four cities of about 30 to 50 pass-stratum ratings each are
  `insufficient` one by one) is a decision, not something this tool settles. NoSidewalk's crowd reference is per
  block face, so `majority2` per label is not its production reference either.

## What this does not establish

- **Selection.** Humans do not vote on a random sample of assessed labels: the queue favours unsettled labels, the
  Triage queue routes contested ones to experts, and Gallery can filter by AI verdict. Shadow mode changes that
  selection again. The status measures precision on the labels humans chose to validate.
- **The reference is noisy.** Expert and crowd agree 86 to 91% of the time; a type such as Obstacle may never clear a
  lower bound of 0.92 in a small deployment however good the model is.
- **Repeated looks.** The 2.5% one-sided error of a single look does not hold for a check run again and again; the
  power table gives the rate for each look schedule.
- **Era by date.** Until a pull with `validator_revision`, a model swap inside an era would mix revisions.
- **Transfer.** A pass bounds how often the AI's votes are wrong there, not whether the model learned anything about
  the deployment (base rate and lift).
- **Votes of excluded validators** are in today's human counts; the updated pull reports them separately
  (`h_agree_excl`, `h_dis_excl`) and the tool then adds a `majority2_no_excluded` sensitivity. The updated pull also
  counts only votes on the label's current type; today's counts include votes cast on a label's earlier type.
- **Per-deployment slices** are too thin to report except inside the strata that pass or are undecided.

A faster source of per-deployment evidence for existing deployments than waiting for new labels: score the round-3
models offline on each deployment's already human-decided labels outside the training split (the round-3 `test` and
`test_heldout` rows per city). That is also the only kind of evidence NoSidewalk can have before it is served.

## Replicating

Inputs (private, under `~/Data`, never committed): the pull
`~/Data/sidewalk-validator-ai-prior-rule-2026-11/ai_assessments_2026-11-11.csv` (sha256 in the provenance block), the
masked re-label keys (four files, listed by basename and sha256 in the provenance block), the cached public rawLabels
and the round-3 `split.csv` files under `~/Data/sidewalk-validator-ai-retrain-2026-10/round3`. The listing is fetched
live; its sha256 and public and private counts are recorded. Nothing is random, so there is no seed. The private
outputs and a copy of the listing are in `~/Data/sidewalk-validator-ai-deployment-gate-2026-11/run-2026-11-13/`.

```bash
D=~/Data/sidewalk-validator-ai-deployment-gate-2026-11
A=~/Data/sidewalk-validator-ai-prior-rule-2026-11/ai_assessments_2026-11-11.csv
R=~/Data/sidewalk-validator-ai-retrain-2026-10/round3
X="$HOME/Data/sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv
   $HOME/Data/sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv
   $HOME/Data/sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv
   $HOME/Data/sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv"
O=docs/deployment-gate-2026-11
python analysis/deployment_gate.py gate --assessments $A --exclude-items $X --extra-terms $D/extra_private_terms.txt \
    --private-out $D/run-2026-11-13 --public-out $O --expect 885,834,129,199,226
python analysis/deployment_gate.py backtest --assessments $A --exclude-items $X --extra-terms $D/extra_private_terms.txt \
    --private-out $D/run-2026-11-13 --public-out $O
python analysis/deployment_gate.py power --public-out $O
python analysis/deployment_gate.py slices --assessments $A --exclude-items $X --extra-terms $D/extra_private_terms.txt \
    --rawlabels-cache $R/rawlabels --train-root $R --public-out $O
python analysis/deployment_gate.py doc --public-out $O --doc docs/deployment-gate-2026-11.md
python analysis/deployment_gate.py doc --public-out $O --doc docs/deployment-gate-2026-11.md --check   # no data needed
```

`--expect` is the acceptance check: the pooled current-era decided gated Agree counts per type, before the re-label
exclusions, must equal those of [`deployment-prior-2026-11/observed.md`](deployment-prior-2026-11/observed.md) (same
pull, same rule), or the command exits 1. `tests/test_deployment_gate.py` asserts the committed result.
`--extra-terms` is a private file of further non-public terms (for example a deployment being onboarded that the
listing does not show yet) that every public file is checked against.

With a fresh pull (the changed `pull_ai_assessments.sh`, run by someone holding a database-host session), the same
`gate` line keys the era on the model revision and splits shadow from vote evidence; drop `--expect`, which belongs
to the 2026-11-11 pull.

<!-- BEGIN GENERATED: provenance (analysis/deployment_gate.py) -->

Versions: python 3.12.14, pandas 2.3.3, numpy 2.5.4.
Pull `ai_assessments_2026-11-11.csv`, sha256 `6891a1c150685975ca4b45136e2eaefd9c3401d20923ce6f6eb7dd53ae71db64`, 1,045,714 rows, 1,007,374 kept (backup_schema_rows 0, internal_schema_rows 9,002, deleted_rows 29,337, model_type_mismatch 1); eras previous 981,513, current 23,932, changeover 1,929.
Listing: live `https://sidewalk-teaneck.cs.washington.edu/v3/api/cities`, sha256 `4932727583564bf9`, 60 deployments, 46 public; 73 non-public terms checked.
Excluded re-label keys (rows in the pull): `sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv` sha256 `ae3468a6409150c6` (300 items, 296 in the pull); `sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv` sha256 `57116bdfa129b66f` (500 items, 500 in the pull); `sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv` sha256 `54fa16fc1dd5a35c` (133 items, 0 in the pull); `sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv` sha256 `d2d30d05912e1a12` (250 items, 0 in the pull); 796 rows removed.
rawLabels cache: 54 files, sha256 of the sorted file hashes `99e286e784c01f5a`; split.csv sha256: Crosswalk `91001f64b778`, CurbRamp `f2ec89f8032c`, NoCurbRamp `fbb1e53bd121`, Obstacle `d41e865f45b5`, SurfaceProblem `c48cac6c889b`.

| command | run on | code (commit; dirty = uncommitted changes) | flags |
|---|---|---|---|
| gate | 2026-11-13 | `0dec41fcd99e` | `--assessments ai_assessments_2026-11-11.csv --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --expect 885,834,129,199,226 --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --private-out run-2026-11-13 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --z 1.96` |
| backtest | 2026-11-13 | `0dec41fcd99e` | `--assessments ai_assessments_2026-11-11.csv --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --private-out run-2026-11-13 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --z 1.96` |
| power | 2026-11-13 | `0dec41fcd99e` | `--gate 0.92 --min-n 50 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --up-to 2000 --z 1.96` |
| slices | 2026-11-13 | `0dec41fcd99e` | `--assessments ai_assessments_2026-11-11.csv --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --public-out deployment-gate-2026-11 --rawlabels-cache rawlabels --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --train-root round3 --z 1.96` |

<!-- END GENERATED: provenance -->

## Follow-ups

- A fresh pull with the revision, vote-cast, excluded-validator and type-matched columns.
- A pull with the time of each label's second human vote, for a real (time-ordered) replay.
- The offline per-deployment evidence above, for existing deployments and for NoSidewalk.
- Re-run the report after the deployment-prior map refit ([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)),
  reading the served accuracy from the new maps.
- A deployment being onboarded in a new country
  ([SidewalkWebpage#5960](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5960)) appears here once it is live,
  scoring in shadow mode, with human votes. Until its deployment is public it is only part of the pooled counts.
