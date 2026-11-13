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
- **Every figure here covers public deployments only.** Non-public deployments are left out of every public number,
  pooled ones included, so no public figure minus the listed public rows says anything about them; their figures are
  only in a private file outside the repository. Two exceptions, both open (see "Pooled counts that include
  non-public deployments" below): the acceptance check's expected counts, which this doc and the provenance repeat,
  and the pooled counts of the same pull that `deployment-prior-2026-11/observed.md` on master already publishes.
- **Under the round-3 models, per-deployment evidence exists in one deployment only.** Two weeks after the round-3
  revisions went live, almost all decided gated rows of the public deployments come from one deployment (headline
  below). Four of its five Agree strata pass; Obstacle is undecided: 0.970 [0.915, 0.990] on the first 100 decided
  rows, which set the status, but 0.951 [0.910, 0.974] over all 184, so the rows past the checkpoint are weaker. It
  passed at the first checkpoint (50 of 50) and lost the pass at 100: under a sticky rule it would read `pass`,
  which is one of the decisions below.
  Every other public deployment has at most a handful of decided rows. The reason is structural: a label is assessed
  once (the website skips assessed labels), so current-era evidence comes only from labels assessed since the switch,
  and humans have voted on few of them yet.
- **Disagree is `no_gated` in every public deployment**: the round-3 Disagree maps are capped below 0.92, so no
  Disagree verdict is a vote today. In the replay of the previous models (below), no Disagree stratum of a public
  deployment would ever have passed: over all their previous-era decided gated Disagree rows (the replay's `all
  decided` column), the votes were right between 53% (CurbRamp) and 79% (Obstacle) of the time per type.
- **Every extra look raises the chance of a lucky pass, so the rule looks only at fixed checkpoints.** At the default
  bar (n >= 50, Wilson 95% lower bound >= 0.92, checkpoints 50, 100, 200, ...), a deployment whose true precision is
  exactly 0.92 passes at some checkpoint by n = 2,000 with probability at least 0.075, not 0.025 (0.10 by n = 6,400);
  re-running the check at every new vote raises that to at least 0.18. "At least": the exact figures assume each
  checkpoint judges the first rows of the next, and real re-runs are not nested (power section). A z of 2.09 instead
  of 1.96 holds the nested checkpoint schedule to 5%.
- **Which reference is used matters.** In vote mode, the AI's Agree plus one human Agree settles a label, so labels
  confirmed by a single human never reach two votes and the default `majority2` reference never counts them, while
  contested labels are sent for more votes. The `any_vote` sensitivity (at least one human vote) is reported beside
  every stratum, and shadow-mode and vote-mode evidence are kept apart once the pull records which is which.
- **A `pass` means "its Agree votes are rarely wrong there", not "the model transfers there".** A deployment where
  most labels are correct can pass with a model that agrees with everything; each stratum reports the deployment's
  own base rate beside its precision.

<!-- BEGIN GENERATED: headline (analysis/deployment_gate.py) -->

- Pull `ai_assessments_2026-11-11.csv` (sha256 `6891a1c150685975`), AI dates of the public deployments' rows 2025-08-13 .. 2026-11-11; era key: date (no validator_revision in the pull): current >= 2026-10-28, previous < 2026-10-27.
- Every figure below and in the generated files covers **public deployments only**, pooled figures included; non-public deployments are left out (their figures are only in the private file).
- Current-era strata (public deployment x type x verdict x regime): 127; by status: pass 4, undecided 1, insufficient 56, no_gated 66.
- 99.5% of the public deployments' current-era decided gated rows come from one deployment (seattle-wa); 5 public deployments have any.
- Pooled current-era decided gated Agree (majority2) of the public deployments after the re-label exclusions, share confirmed in brackets: Crosswalk 885 (0.998), CurbRamp 834 (0.998), NoCurbRamp 129 (1.000), Obstacle 186 (0.946), SurfaceProblem 226 (0.987).
- Acceptance check (every deployment's counts before the exclusions, computed privately, against `docs/deployment-prior-2026-11/observed.md`): passes.

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
  `label_id`, within one schema, so the per-city `label_id` is never compared across deployments; a stable sort, so
  the order is deterministic for one pull), where C is the largest of 50, 100, 200, 400, 800, 1600, 3200, ... not
  above its decided count. Rows past C wait for the next checkpoint. The first checkpoint may not be below the minimum
  n (the tool refuses such a schedule). Across pulls the prefix is not fixed: a label assessed early but decided late
  (contested labels get extra votes) enters the first C rows in a later pull and pushes another row out.
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

- **A pass can be lost.** `status` is re-judged at every run at the stratum's current checkpoint, so a stratum that
  passes at 50 and not at 100 reads `undecided` at 100. Each stratum also reports `first_pass_checkpoint`, the first
  checkpoint that passed. A sticky rule (a pass is kept unless the stratum later fails) would act on that field; the
  power figures and the replay describe the sticky rule (the chance of passing at ANY checkpoint). Which rule the
  website follows is a decision for the owner (below).

- **Era**: the 2026-11-11 pull has no model revision, so the era is a date split: current = assessed on `--since`
  (2026-10-28, the first full day of the round-3 revisions) or later, previous = before `--before` (2026-10-27), the
  days between in neither. The assessment table does record the revision (`validator_revision`, SidewalkWebpage
  evolution 420); this PR changes `analysis/pull_ai_assessments.sh` to select it, and with it the tool keys the era on
  each model's current revision instead of a date. The current revision is the one on the newest day of the PUBLIC
  deployments' rows, so a non-public deployment scored later, or by another revision on the same day, moves no public
  row between eras. If two revisions of one model appear on that day the run stops and asks for
  `--current-revision MODEL=REVISION`; no tie is broken on `label_id`, which is numbered per deployment. The
  provenance records each model's current revision and whether it came from the public rows or the flag. Whether the
  revision and vote-cast columns are used at all is also decided on the public rows: a column that is present but
  empty on every public row is treated as absent, and the output says "present but empty". An EMPTY revision is never read as the previous era: it is also what
  a schema missing the column looks like, so such rows go to an `unknown` era, excluded from both and counted (and
  the tool warns). `--unrevisioned-before DAY` is the explicit exception: empty-revision rows assessed before that day
  (when revisions began to be recorded) are previous; the stderr warning counts the public deployments' rows only. Days
  are given as YYYY-MM-DD and refused otherwise. The pull script now exits 1 when any schema's table or column
  check or query fails, so a partial pull is not mistaken for a complete one. Status uses the current era only; the
  previous era's status is reported as context.
- **Regime**: `vote` when the assessment is linked to a vote (`ai_vote_cast`), `shadow` when it was scored without
  one. The link means AI validation was ON for the deployment: in vote mode the website casts a vote for every
  assessment, the verdict at or above the gate and an **Unsure** below it. So on a gated row (the only rows a status
  uses) `vote` also means the verdict counted; on an ungated row it means an Unsure was cast. Known limit:
  SidewalkWebpage evolution 355 removed the vote link of AI votes it voided (the off-target-marker repair), so those
  vote-mode rows read `shadow`; they predate the current era. The pull carries no field that recovers them.
- **Every value is a flag** (`--min-n`, `--gate`, `--z`, `--schedule`, `--reference`, `--since`, `--before`,
  `--unrevisioned-before`, `--current-revision`) and is recorded in `deployment_status.json` or the provenance.

Per stratum the report also carries: `n_assessed`, `n_gated`, `n` (decided), `k`, the precision and interval at the
checkpoint, `precision_all` and its interval (all decided rows), `first_pass_checkpoint`, `more_needed_if_perfect`
(the fewest further decided rows, all right, that would pass at a checkpoint; null for `no_gated`, `never_votes` and
`not_served`, where no number of rows helps), `base_rate` (the share of all decided AI-assessed labels of that deployment and type
whose human class matches the verdict) and `lift`, the status under the `tie_half` and `any_vote` references, the
previous era's status, whether the deployment has previous-era assessments of the type (`existing`), and `evidence`,
`evidence_ref`, `basis`, `min_n` and `model_revision`.

### The output contract

`deployment-gate-2026-11/deployment_status.json` (public) holds:

- `types`: every type with `served` and, per verdict, `eligible` or `never`. An unserved type is `off` on the website,
  not `shadow`: shadow needs the API to score the type.
- `deployments[city_id][type][verdict][regime]`: the stratum, public deployments only.
- `private`: a fixed note (one key, `note`, the same text whatever the non-public deployments are): non-public
  deployments are left out of every figure in the file.
- `summary`: over the public strata only: strata by status, how many change status under each sensitivity reference,
  the largest one-deployment share of the decided rows, the acceptance counts, and the outcome of the acceptance
  check (`{"ok": true, "against": ...}`). The counts the check computes include non-public deployments and are
  written only to the private file, but the counts it compares them with are the `--expect` flag, which the
  provenance records, and since the check passes the two are equal (see "Pooled counts that include non-public
  deployments" below).
- `pull`, `exclusions`: the pull's file name and sha256, and its counts over public deployments only.

It contains no `mode`. The private `deployment_status_all.json` (under `~/Data`, never committed) lists every
deployment by schema, with the mode each stratum would get under two policies: `strict` (vote only on `pass`) and
`grandfather` (an existing deployment keeps voting unless it `fail`s; a new one as `strict`), plus the pooled figures
that include non-public deployments (`summary_all`, `private_cells`) and the acceptance counts.

`schema_version` (1) promises the key names and meanings below; a change to either bumps it. One stratum
(seattle-wa Obstacle Agree from the committed file, floats rounded to three places, `sensitivity` left out; it holds
the same fields per other reference):

```json
{"evidence": "production_votes", "evidence_ref": "ai_assessments_2026-11-11.csv@6891a1c150685975",
 "basis": "raw", "reference": "majority2", "min_n": 50,
 "model_revision": "unrecorded; current era by date (ai_date >= 2026-10-28)",
 "n_assessed": 795, "n_gated": 345, "n": 184, "k": 175.0,
 "precision_all": 0.951, "ci_all_lo": 0.910, "ci_all_hi": 0.974,
 "checkpoint": 100, "k_checkpoint": 97.0, "precision": 0.970, "ci_lo": 0.915, "ci_hi": 0.990,
 "status": "undecided", "first_pass_checkpoint": 50, "more_needed_if_perfect": 216,
 "base_rate": 0.698, "base_n": 682, "lift": 0.253, "existing": true,
 "previous_era": {"n": 0, "checkpoint": null, "precision": null, "ci_lo": null, "ci_hi": null,
                  "status": "insufficient"}}
```

| field | type | meaning |
|---|---|---|
| `evidence` | `production_votes` / `masked_expert_round` | where the outcomes come from |
| `evidence_ref` | string | the input it was computed from (pull file and sha256 prefix, or a round's file and commit) |
| `basis` | `raw` / `prior_weighted` | whether `k` is a raw count or prior-weighted |
| `reference` | `majority2` / `tie_half` / `any_vote` / `expert` | what decides a row |
| `min_n` | int | the fewest decided rows a status other than `insufficient` needs |
| `model_revision` | string | the model the evidence is about |
| `n_assessed`, `n_gated` | int or null | assessments of this verdict, and those at or above the gate (null for an expert round) |
| `n` | int | decided gated rows |
| `k` | float | rows the reference says the AI got right; fractional under `tie_half` (a tie counts 0.5) and `prior_weighted` |
| `precision_all`, `ci_all_lo`, `ci_all_hi` | float or null | k / n over every decided row, Wilson 95% |
| `checkpoint` | int or null | the checkpoint the status is evaluated at; null below the first one and for an expert round (one look, all rows) |
| `k_checkpoint`, `precision`, `ci_lo`, `ci_hi` | float or null | the same on the first `checkpoint` rows; these set the status |
| `status` | see the status table | re-judged at the current checkpoint |
| `first_pass_checkpoint` | int or null | the first checkpoint that passed |
| `more_needed_if_perfect` | int or null | further decided rows, all right, that would pass; 0 = passing; null where no number helps or past 25,600 |
| `base_rate`, `base_n`, `lift` | float, int, float | share of the deployment's decided AI-assessed labels of the type whose class matches the verdict, its n, and `precision_all - base_rate` |
| `existing` | bool or null | the deployment has previous-era assessments of the type |
| `previous_era` | object or null | the previous era's n, checkpoint, precision, interval and status, as context |
| `sensitivity` | object | per other reference: n, k, checkpoint, precision, interval, status |

The regime key under `deployments[city][type][verdict]` is `vote`, `shadow`, `unrecorded` (the pull does not say),
or `expert_round` (a masked expert round added with `add_stratum`).

## What the website needs (SidewalkWebpage#5832)

- A per city, type and verdict mode `off` / `shadow` / `vote`, with `shadow` the default for a new deployment.
- Shadow mode must score and store assessments even when tag suggestions are off, or there is nothing to measure.
- Human validators must still see the deployment's labels: the evidence is human votes on assessed labels.
- The pull must say whether a vote was cast for each assessment (`ai_vote_cast`, added here): the decided population
  differs between the modes (above), so the two are separate strata and are never pooled. Today the link only says
  that AI validation was on (an Unsure is cast below the gate); for gated rows that is the same as "the verdict
  counted".
- Switching a stratum to `vote` is a person's decision, made from this report and applied to the city settings. When
  the switch happens, the stored assessments can back-fill votes (#5832's gap 4).
- Nothing changes for Disagree while its maps are held back below 0.92.

## Results today (2026-11-11 pull)

### Strata by status, current era

<!-- BEGIN GENERATED: status_counts (analysis/deployment_gate.py) -->

Public deployments only.

| Type | Verdict | pass | undecided | fail | insufficient | no_gated |
|---|---|---:|---:|---:|---:|---:|
| Crosswalk | Agree | 1 | 0 | 0 | 14 | 0 |
| Crosswalk | Disagree | 0 | 0 | 0 | 0 | 8 |
| CurbRamp | Agree | 1 | 0 | 0 | 14 | 0 |
| CurbRamp | Disagree | 0 | 0 | 0 | 0 | 15 |
| NoCurbRamp | Agree | 1 | 0 | 0 | 5 | 4 |
| NoCurbRamp | Disagree | 0 | 0 | 0 | 0 | 12 |
| Obstacle | Agree | 0 | 1 | 0 | 9 | 3 |
| Obstacle | Disagree | 0 | 0 | 0 | 0 | 11 |
| SurfaceProblem | Agree | 1 | 0 | 0 | 14 | 0 |
| SurfaceProblem | Disagree | 0 | 0 | 0 | 0 | 13 |

<!-- END GENERATED: status_counts -->

### Public deployments

<!-- BEGIN GENERATED: public_table (analysis/deployment_gate.py) -->

Public deployments' Agree strata with at least one decided row. Disagree is `no_gated` in every public deployment (the round-3 Disagree maps are capped below 0.92), so it is not shown. n = decided rows (checkpoint the status is evaluated at); at checkpoint = precision [Wilson 95%] of the first `checkpoint` decided rows, which sets the status; all decided = precision [Wilson 95%] of every decided row, including those past the checkpoint; first pass = the first checkpoint that passed (a sticky rule would keep it; the status is re-judged at the current checkpoint); base rate = share of the deployment's decided AI-assessed labels of the type that humans call correct; more needed = further decided rows, all right, that would pass.

| Deployment | Type | Status | first pass | n (checkpoint) | at checkpoint [95%] | all decided [95%] | base rate | more needed | previous era |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| houston-tx | CurbRamp | insufficient |  | 1 (-) |  | 1.000 [0.207, 1.000] | 1.000 | 49 | insufficient |
| houston-tx | Obstacle | insufficient |  | 1 (-) |  | 1.000 [0.207, 1.000] | 1.000 | 49 | no_gated |
| houston-tx | SurfaceProblem | insufficient |  | 1 (-) |  | 0.000 [0.000, 0.793] | 0.444 | 99 | insufficient |
| la-ca | CurbRamp | insufficient |  | 1 (-) |  | 1.000 [0.207, 1.000] | 1.000 | 49 | insufficient |
| seattle-wa | Crosswalk | pass | 50 | 883 (800) | 0.998 [0.991, 0.999] | 0.998 [0.992, 0.999] | 0.908 | 0 | pass |
| seattle-wa | CurbRamp | pass | 100 | 831 (800) | 0.998 [0.991, 0.999] | 0.998 [0.991, 0.999] | 0.952 | 0 | pass |
| seattle-wa | NoCurbRamp | pass | 50 | 129 (100) | 1.000 [0.963, 1.000] | 1.000 [0.971, 1.000] | 0.853 | 0 | pass |
| seattle-wa | Obstacle | undecided | 50 | 184 (100) | 0.970 [0.915, 0.990] | 0.951 [0.910, 0.974] | 0.698 | 216 | insufficient |
| seattle-wa | SurfaceProblem | pass | 50 | 222 (200) | 0.995 [0.972, 0.999] | 0.995 [0.975, 0.999] | 0.860 | 0 | undecided |
| st-louis-mo | SurfaceProblem | insufficient |  | 1 (-) |  | 0.000 [0.000, 0.793] | 0.000 | 99 | pass |
| walla-walla-wa | Crosswalk | insufficient |  | 2 (-) |  | 1.000 [0.342, 1.000] | 1.000 | 48 | pass |
| walla-walla-wa | CurbRamp | insufficient |  | 1 (-) |  | 1.000 [0.207, 1.000] | 1.000 | 49 | pass |
| walla-walla-wa | Obstacle | insufficient |  | 1 (-) |  | 0.000 [0.000, 0.793] | 0.000 | 99 | no_gated |
| walla-walla-wa | SurfaceProblem | insufficient |  | 2 (-) |  | 1.000 [0.342, 1.000] | 1.000 | 48 | pass |

Another 54 public Agree strata in 18 deployments have current-era assessments but no decided row (insufficient 47, no_gated 7); each needs at least 50 decided rows. Their previous-era status is in `deployment_status.json`.

<!-- END GENERATED: public_table -->

### Non-public deployments

<!-- BEGIN GENERATED: private_counts (analysis/deployment_gate.py) -->

Non-public deployments are left out of every figure in this file, pooled figures included. Their strata, and every figure that includes them, are only in the private file kept outside the repository.

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

## Replay on the previous models (public deployments, end-of-pull votes)

The rule replayed on the previous era of the public deployments, stratum by stratum (deployment x type x verdict x
evidence regime; vote and shadow evidence are never pooled): at each checkpoint, the status of the decided rows so
far; a stratum **graduates** at its first `pass` and stays graduated (the sticky rule). Rows up to and including that checkpoint are what shadow mode would have
withheld; rows after it are what it would have let through. Counts only: the pull has no vote timestamps, so the
replay makes no claim about dates or waiting times. Items of the masked re-label rounds (their votes were cast in
production on items sampled by AI verdict) are excluded.

Limits: whether a row is decided, and its majority, uses votes as of the end of the pull, so an early checkpoint sees
votes that did not exist yet; contested labels are decided later, so the replay understates early false
graduation. The 2026-11-11 pull does not record the regime, so every row is `unrecorded`; its previous era was
collected in vote mode, so this says nothing about how evidence accrues in shadow mode. The previous era is not one model: API releases before 1.0.2 loaded the hub's main branch.

<!-- BEGIN GENERATED: backtest (analysis/deployment_gate.py) -->

Previous era, votes as of the end of the pull, public deployments only; count-based replay. Checkpoints 50, 100, 200, 400, 800, 1600 then doubling; a stratum graduates at its first pass and stays graduated. Strata are split by evidence regime and never pooled across it. Cells: decided rows, precision [Wilson 95%].

| Type | Verdict | Regime | strata (n >= min) | final pass / undecided / fail | graduated | n at graduation (median, range) | withheld in graduated | kept after graduation | never graduated | strata under min | all decided |
|---|---|---|---:|---|---:|---|---|---|---|---|---|
| Crosswalk | Agree | unrecorded | 37 (18) | 17 / 1 / 0 | 17 | 50 (50-100) | 1,050, 0.995 [0.989, 0.998] | 5,857, 0.998 [0.997, 0.999] | 79, 0.975 [0.912, 0.993] | 238, 0.992 [0.970, 0.998] | 7,224, 0.997 [0.996, 0.998] |
| Crosswalk | Disagree | unrecorded | 36 (9) | 0 / 3 / 6 | 0 |  | 0 | 0 | 1,568, 0.588 [0.563, 0.612] | 355, 0.423 [0.372, 0.474] | 1,923, 0.557 [0.535, 0.580] |
| CurbRamp | Agree | unrecorded | 40 (25) | 24 / 1 / 0 | 24 | 50 (50-400) | 2,350, 0.980 [0.973, 0.985] | 44,030, 0.993 [0.993, 0.994] | 52, 0.962 [0.870, 0.989] | 303, 0.993 [0.976, 0.998] | 46,735, 0.993 [0.992, 0.993] |
| CurbRamp | Disagree | unrecorded | 35 (12) | 0 / 0 / 12 | 0 |  | 0 | 0 | 5,328, 0.541 [0.527, 0.554] | 240, 0.346 [0.289, 0.408] | 5,568, 0.532 [0.519, 0.545] |
| NoCurbRamp | Agree | unrecorded | 37 (16) | 6 / 8 / 2 | 7 | 200 (100-3200) | 4,300, 0.945 [0.938, 0.951] | 18,186, 0.972 [0.970, 0.974] | 2,676, 0.885 [0.873, 0.897] | 342, 0.854 [0.812, 0.887] | 25,504, 0.957 [0.954, 0.959] |
| NoCurbRamp | Disagree | unrecorded | 38 (14) | 0 / 1 / 13 | 0 |  | 0 | 0 | 7,376, 0.715 [0.704, 0.725] | 260, 0.469 [0.409, 0.530] | 7,636, 0.706 [0.696, 0.716] |
| Obstacle | Agree | unrecorded | 1 (0) | 0 / 0 / 0 | 0 |  | 0 | 0 | 0 | 2, 1.000 [0.342, 1.000] | 2, 1.000 [0.342, 1.000] |
| Obstacle | Disagree | unrecorded | 28 (6) | 0 / 2 / 4 | 0 |  | 0 | 0 | 1,529, 0.808 [0.788, 0.827] | 281, 0.680 [0.623, 0.732] | 1,810, 0.788 [0.769, 0.807] |
| SurfaceProblem | Agree | unrecorded | 38 (25) | 18 / 6 / 1 | 18 | 50 (50-1600) | 3,000, 0.960 [0.953, 0.967] | 10,645, 0.981 [0.978, 0.984] | 6,347, 0.937 [0.931, 0.943] | 177, 0.960 [0.921, 0.981] | 20,169, 0.964 [0.961, 0.966] |
| SurfaceProblem | Disagree | unrecorded | 28 (4) | 0 / 0 / 4 | 0 |  | 0 | 0 | 2,600, 0.654 [0.636, 0.672] | 328, 0.543 [0.489, 0.596] | 2,928, 0.642 [0.624, 0.659] |

<!-- END GENERATED: backtest -->

Reading. Under the previous models, the rule would have let Agree votes through in most public deployments that had
enough human votes: Crosswalk, CurbRamp and SurfaceProblem strata graduated at a median of 50 decided rows, and
NoCurbRamp at 200. The votes let through after graduation were more precise than those withheld before it (for
example NoCurbRamp 0.972 kept against 0.945 withheld), and the strata that never graduated, or never reached 50
decided rows, held the weakest Agree votes (NoCurbRamp 0.885 and 0.854). Every Disagree stratum stayed in shadow:
none passed at any checkpoint, and over all decided rows the gated Disagree votes were right 0.532 (CurbRamp) to
0.788 (Obstacle) of the time per type. These are end-of-pull votes collected in vote mode; read them as what the rule
separates, not as how fast a new deployment would graduate.

## How many decided labels a deployment needs, and how often a lucky pass happens

Exact binomial arithmetic, no simulation. At n = 50 only a perfect 50 of 50 passes; with crowd noise that is
uncommon even for a good model (the single-look row at 0.98 and 0.99), so the realistic graduation n is in the
hundreds. At the public deployments' pooled current-era precisions (0.946 for Obstacle up to 1.000 for NoCurbRamp,
headline) the table shows how often a deployment would pass at each n.

Two qualifications, also in `power.json`. First, `P(EVER pass)` is the chance of passing at ANY look: it describes
a sticky rule. Second, the exact figures assume nested looks (the rows judged at one checkpoint are the first rows of
those judged at the next). The rows are ordered by (`ai_date`, `label_id`) within a deployment, deterministic for
one pull, but a label assessed early and decided late enters that prefix in a later pull and pushes another row out,
so real re-runs are not nested and every figure below is a lower bound on the real rate.

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

Past n = 2,000 the checkpoints keep doubling and the rate keeps growing: at true precision = gate, P(ever pass) under the checkpoints is 0.102 by n = 6,400, 0.128 by n = 25,600.

- ever_pass is the chance of passing at ANY look, i.e. of a sticky rule (a pass is kept). A rule that re-judges each stratum at its current checkpoint can lose a pass later; ever_pass bounds how often it passes at some point, not how often it is passing at the end.
- The rates assume NESTED looks: the rows judged at one checkpoint are the first rows of those judged at the next. Real re-runs are not nested. The rule takes the first C decided rows ordered by (ai_date, label_id) within a deployment, which is deterministic for one pull, but a label assessed early and decided late (contested labels get extra votes) enters that prefix in a later pull and pushes another row out. Over real re-runs the false-pass rate is therefore at least the stated rate, not exactly it: each figure here is a lower bound.
- ever_pass is by n = 2,000; the checkpoints keep doubling past it and the rate keeps growing (ever_pass_checkpoints_at_gate_longer).

<!-- END GENERATED: power -->

## Severity and tag slices

The standing rule (every performance study reports type x severity and type x tag against a baseline, with training
counts). Pooled over public deployments, so in the current era these are in effect one deployment's numbers (the `largest`
column). The baseline is the previous era. The slice that bears on the status is the last table: the severity mix
inside each stratum that passes or is undecided, so a stratum that passes overall but is weak at high severity is
visible.

<!-- BEGIN GENERATED: slices (analysis/deployment_gate.py) -->

Gated Agree rows decided under `majority2`, pooled over public deployments only (non-public deployments are left out of every table); confirmed share blank under 30 decided; train = round-3 `split.csv` training rows (correct / incorrect); largest = the largest one-deployment share of the slice.
Tags joined from cached public rawLabels: 794,881 of 808,584 public-deployment rows have an export (private deployments are never joined; rows without one are `(not exported)`).

**Crosswalk by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 812 | 0.999 | 6,504 | 0.998 | 100% | 1073 / 321 |
| 2 | 42 | 0.976 | 366 | 0.995 | 100% | 44 / 143 |
| 3 | 11 |  | 115 | 0.991 | 100% | 12 / 357 |
| (none) | 20 |  | 239 | 0.992 | 100% | 27 / 112 |

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

**CurbRamp by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 747 | 0.999 | 40,504 | 0.994 | 100% | 3062 / 2021 |
| 2 | 50 | 1.000 | 3,862 | 0.986 | 98% | 377 / 460 |
| 3 | 9 |  | 1,252 | 0.971 | 100% | 144 / 407 |
| (none) | 28 |  | 1,117 | 0.986 | 100% | 212 / 242 |

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
| steep | 0 |  | 403 | 0.973 |  | 49 / 177 |
| parallel lines | 0 |  | 121 | 1.000 |  | 12 / 2 |

**NoCurbRamp by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 19 |  | 1,892 | 0.924 | 100% | 614 / 1225 |
| 2 | 39 | 1.000 | 8,480 | 0.952 | 100% | 2044 / 1551 |
| 3 | 67 | 1.000 | 14,364 | 0.965 | 100% | 3863 / 2211 |
| (none) | 4 |  | 768 | 0.935 | 100% | 217 / 474 |

**NoCurbRamp by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| no alternate route | 68 | 1.000 | 12,754 | 0.970 | 100% | 3339 / 1331 |
| (none) | 44 | 1.000 | 8,448 | 0.955 | 100% | 2117 / 2514 |
| alternate route present | 17 |  | 3,835 | 0.923 | 100% | 1129 / 1171 |
| unclear if needed | 0 |  | 785 | 0.892 |  | 246 / 614 |

**Obstacle by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 0.939 | 0 |  | 100% | 1382 / 1760 |
| 2 | 51 | 0.941 | 2 |  | 98% | 1818 / 1026 |
| 3 | 74 | 0.946 | 0 |  | 99% | 2627 / 1272 |
| (none) | 12 |  | 0 |  | 100% | 270 / 399 |

**Obstacle by tag** (slices with at least min n in either era)

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| narrow | 51 | 1.000 | 0 |  | 100% | 669 / 239 |
| trash/recycling can | 50 | 0.980 | 0 |  | 100% | 759 / 287 |

**SurfaceProblem by severity**

| slice | current n | current confirmed | previous n | previous confirmed (baseline) | largest (current) | train |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 105 | 0.971 | 9,691 | 0.960 | 96% | 2641 / 2409 |
| 2 | 70 | 1.000 | 6,143 | 0.963 | 100% | 1768 / 1191 |
| 3 | 40 | 1.000 | 3,610 | 0.980 | 100% | 1191 / 642 |
| (none) | 11 |  | 725 | 0.946 | 100% | 239 / 419 |

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
  power table gives the rate for each look schedule, as a lower bound (re-runs are not nested).
- **Era by date.** Until a pull with `validator_revision`, a model swap inside an era would mix revisions.
- **Vote links voided by evolution 355** read as shadow (above); they are in the previous era only.
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
live; its sha256 and the numbers of listed and of public deployments are recorded. Nothing is random, so there is no
seed. The private outputs (`deployment_status_all.{json,csv}`, `backtest_all.json`, `slices_all.json`,
`provenance_private.json`, with every figure that includes non-public deployments) and a copy of the listing are in
`~/Data/sidewalk-validator-ai-deployment-gate-2026-11/run-2026-11-13-r1/` (the run before the review round-1 fixes
is in `run-2026-11-13/`).

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
    --private-out $D/run-2026-11-13-r1 --public-out $O --expect 885,834,129,199,226
python analysis/deployment_gate.py backtest --assessments $A --exclude-items $X --extra-terms $D/extra_private_terms.txt \
    --private-out $D/run-2026-11-13-r1 --public-out $O
python analysis/deployment_gate.py power --public-out $O
python analysis/deployment_gate.py slices --assessments $A --exclude-items $X --extra-terms $D/extra_private_terms.txt \
    --rawlabels-cache $R/rawlabels --train-root $R --private-out $D/run-2026-11-13-r1 --public-out $O
python analysis/deployment_gate.py doc --public-out $O --doc docs/deployment-gate-2026-11.md
python analysis/deployment_gate.py doc --public-out $O --doc docs/deployment-gate-2026-11.md --check   # no data needed
```

`--expect` is the acceptance check: the pooled current-era decided gated Agree counts per type, before the re-label
exclusions and over EVERY deployment, must equal those of
[`deployment-prior-2026-11/observed.md`](deployment-prior-2026-11/observed.md) (same pull, same rule), or the command
exits 1. Those counts include non-public deployments. `deployment_status.json` records only the outcome and the
counts the check computes go to the private file, but the expected counts themselves are public: they are the
`--expect` value in the command above, in `deployment-gate-2026-11/provenance.json` and so in the flags table below,
and in the docstrings of `analysis/deployment_gate.py`, and they are the counts `observed.md` already publishes. See
"Pooled counts that include non-public deployments" under the open decisions. `tests/test_deployment_gate.py`
asserts the committed outcome.
`--extra-terms` is a private file of further non-public terms (for example a deployment being onboarded that the
listing does not show yet) that every public file is checked against.

With a fresh pull (the changed `pull_ai_assessments.sh`, run by someone holding a database-host session), the same
`gate` line keys the era on the model revision and splits shadow from vote evidence; drop `--expect`, which belongs
to the 2026-11-11 pull, and consider `--unrevisioned-before` (above). The script exits 1 if any schema failed; do not
use such a pull.

<!-- BEGIN GENERATED: provenance (analysis/deployment_gate.py) -->

Versions: python 3.12.14, pandas 2.3.3, numpy 2.5.4.
Pull `ai_assessments_2026-11-11.csv`, sha256 `6891a1c150685975ca4b45136e2eaefd9c3401d20923ce6f6eb7dd53ae71db64`. Public deployments only: 829,678 rows, 808,584 kept after the drops and the exclusions (deleted_rows 20,393, model_type_mismatch 1); eras changeover 1,121, current 15,666, previous 791,797.
Listing: live `https://sidewalk-teaneck.cs.washington.edu/v3/api/cities`, sha256 `4932727583564bf9`, 60 deployments, 46 public. Every public file is checked against the non-public terms before it is written.
Excluded re-label keys (rows of public deployments in the pull): `sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv` sha256 `ae3468a6409150c6` (200); `sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv` sha256 `57116bdfa129b66f` (500); `sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv` sha256 `54fa16fc1dd5a35c` (0); `sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv` sha256 `d2d30d05912e1a12` (0); 700 rows removed.
rawLabels cache: 54 files, sha256 of the sorted file hashes `99e286e784c01f5a`; split.csv sha256: Crosswalk `91001f64b778`, CurbRamp `f2ec89f8032c`, NoCurbRamp `fbb1e53bd121`, Obstacle `d41e865f45b5`, SurfaceProblem `c48cac6c889b`.

| command | run on | code (commit; dirty = uncommitted changes) | flags |
|---|---|---|---|
| gate | 2026-11-13 | `659531bee686` | `--assessments ai_assessments_2026-11-11.csv --before 2026-10-27 --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --expect 885,834,129,199,226 --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --private-out run-2026-11-13-r1 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --z 1.96` |
| backtest | 2026-11-13 | `659531bee686` | `--assessments ai_assessments_2026-11-11.csv --before 2026-10-27 --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --private-out run-2026-11-13-r1 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --z 1.96` |
| power | 2026-11-13 | `659531bee686` | `--before 2026-10-27 --gate 0.92 --min-n 50 --public-out deployment-gate-2026-11 --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --up-to 2000 --z 1.96` |
| slices | 2026-11-13 | `659531bee686` | `--assessments ai_assessments_2026-11-11.csv --before 2026-10-27 --exclude-items sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv sidewalk-validator-ai-relabel-2026-10/round2/private-key-do-not-share/key.csv sidewalk-validator-ai-ai-labels-2026-10/relabel/private-key-do-not-share/key.csv sidewalk-validator-ai-nosidewalk-2026-11/round-b/private-key-do-not-share/key.csv --extra-terms extra_private_terms.txt --gate 0.92 --min-n 50 --private-out run-2026-11-13-r1 --public-out deployment-gate-2026-11 --rawlabels-cache rawlabels --reference majority2 --schedule 50,100,200,400,800,1600 --since 2026-10-28 --train-root round3 --z 1.96` |

<!-- END GENERATED: provenance -->

## Open decisions (for the owner)

- **Sticky pass, or re-judged at each checkpoint.** Sticky: a stratum that passes at a checkpoint keeps voting unless
  it later reaches `fail`; its false-pass rate is the `P(EVER pass)` row (at least 0.075 by n = 2,000 at z 1.96), and
  a real decline shows only once the upper bound falls below 0.92, which takes many rows. Today seattle-wa Obstacle
  Agree would vote. Re-judged: the status is that of the current checkpoint, so a single checkpoint's error rate
  applies (at most 0.026 at the gate) but a stratum can switch between voting and not voting from one run to the
  next; seattle-wa Obstacle Agree stays in shadow. Either way the report carries both `status` and
  `first_pass_checkpoint`.
- **Existing deployments**: vote only on `pass` (strict), or keep voting unless `fail` (grandfather). The private
  file shows both.
- **The bar**: n >= 50 and Wilson 95% >= 0.92 at checkpoints 50, 100, 200, ...; keep z 1.96 or raise it (2.09 holds
  the nested checkpoint rate to 5%).
- **The reference**: `majority2` binds, with `tie_half` and `any_vote` as sensitivities.
- **Previous-era evidence**: context only today.
- **NoSidewalk**: per-deployment masked rounds, or a pooled rule.
- **Pooled counts that include non-public deployments.** `docs/deployment-prior-2026-11/observed.md` and
  `observed.json` on master pool every deployment of the same pull. Set against this PR's public files they give, per
  label type, the non-public deployments' current-era gated Agree counts and their current-era decided counts, and
  their previous-era decided and confirmed counts, so their pooled previous-era precision; they also give a
  deleted-label total and the number of schemas in the pull. The current-era part comes from very few deployments.
  This PR also repeats the private-inclusive acceptance counts itself: the `--expect` value in
  `deployment-gate-2026-11/provenance.json` and the generated flags table, the Replicating command, and the module
  and `acceptance_counts` docstrings in `analysis/deployment_gate.py`. Nothing here changes `observed.*`. The option,
  if the owner decides to close it: regenerate `observed.*` over public deployments only, read the expected counts
  from a private file under `~/Data`, and keep them out of the provenance flags.

## Follow-ups

- A fresh pull with the revision, vote-cast, excluded-validator and type-matched columns.
- A pull with the time of each label's second human vote, for a real (time-ordered) replay, and to order the rows
  of a stratum by when they were decided, which makes successive runs nested.
- The offline per-deployment evidence above, for existing deployments and for NoSidewalk.
- Re-run the report after the deployment-prior map refit ([#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)),
  reading the served accuracy from the new maps.
- A deployment being onboarded in a new country
  ([SidewalkWebpage#5960](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5960)) appears here once it is live,
  scoring in shadow mode, with human votes. Until its deployment is public it appears in no public figure: every
  public figure, pooled ones included, covers public deployments only.
