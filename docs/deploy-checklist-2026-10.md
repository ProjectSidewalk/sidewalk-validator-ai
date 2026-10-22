# Pre-deploy red team and checklist: shipping the round-3 validators (October 2026)

The round-3 models (five label types, tilt-corrected depth-sized crops, panorama-grouped split, held-out cities;
`docs/calibration-2026-10.md` and the retrain report) are about to replace the 2025-08-16 production models that
[sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) serves. This document asks one question of the
whole chain, adversarially: **what breaks when they go live?** It reads this repo at `c68459a`, sidewalk-panorama-tools
`v1.0.0` (`7ce25ff`), sidewalk-ai-api `tilt-fields` at `174a169`
([#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17)) and the serving PR stacked on it,
[sidewalk-ai-api#19](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/19) (`round3-serving`, `c6bf89d`, which
closes [sidewalk-ai-api#18](https://github.com/ProjectSidewalk/sidewalk-ai-api/issues/18)), and SidewalkWebpage
`develop` after [#5795](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/5795) (camera pose and `coord_frame`
in the `/process` request). Every claim below was checked against code or reproduced on the round-3 predictions; each
finding says what evidence would prove it wrong, so the next person can retire it rather than argue with it.

Three parts: ranked findings (most severe first), the ordered checklist across the four repositories, and rollback.
Private deployments are written as "private city A (outside the US)" and "private city B (US)", as in the retrain
report.

## 1. Findings, ranked

| # | Severity | Finding | One-line fix |
|---|---|---|---|
| F1 | **High** | The Disagree maps are fitted under a class prior that the training rule itself distorts, and the one type where Disagree clears the gate under every prior (Crosswalk) fails on the held-out cities. sidewalk-ai-api#19 ships these maps **uncapped**, so merging it as it stands turns Crosswalk and NoCurbRamp Disagree votes on. A lone AI Disagree sets `label.correct = FALSE` and hides the label. | Fit with `calibrate.py --hold-back` (the flag the calibration work is adding) so every `incorrect` map is capped at `HOLD_BACK_CAP` (0.9199) and `publish_to_hf.py` stages the capped map and report; swap #19's maps for those; release Disagree per type only from production outcomes on the new revision. |
| F2 | **High** | The Agree maps move a lot with the same prior choice (NoCurbRamp coverage 44% under the file prior, 71% under the production estimate, 86% at 90:10), so the prior decides whether a third of NoCurbRamp labels get an AI vote. | Decide and document the prior; evaluate on `test_heldout` as well as `test`; gate per deployment where the model has not been measured ([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)). |
| F3 | **High** (silent outage) | The website parses the whole `/process` response inside one `try`; any field it cannot parse (`validator_training_date` or `tagger_training_date` not `MM-dd-yyyy`, an `image_source` outside the `AiImageSource` enum) drops the response for every label everywhere, with nothing but a warning per label. #19's entries parse today; nothing stops the next edit from breaking them. | Add a format test for both dates and the enum in sidewalk-ai-api; have the smoke test parse the whole response the way `AiService` does. |
| F4 | Medium | Nothing binds `accuracy_mappings/<type>.json` to the pinned model revision; #19's `regression/parity_expected.csv` has no accuracy column, so a stale map passes the smoke test. | Add `validation_estimated_accuracy` to the expected file; record the map's sha256 (or load the map from the HF snapshot at the pinned revision) in the API. |
| F5 | Medium | Training only ever saw panoramas from the lab's store; `/process` falls back to stitching Google tiles (zoom 4, or zoom 3 upsampled) on a cache miss. That path's pixels were never compared with training's, and the share of production assessments that take it is unmeasured here. | Measure the `ai_image_source` share; run `serving_parity.py` on one panorama through both paths before the deploy. |
| F6 | Medium | The new model never re-assesses a label the old one assessed (`aiAssessmentIsCurrent` is a label-type match only). Old Disagree-hidden labels stay hidden ([#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)), and the first days of new-revision data come from the nightly actor's backlog order (up to 800 labels per city per day), not a random sample. | Measure by revision and by city; do not read the first 48 h as a platform-wide precision. |
| F7 | Medium | Shadow mode exists today without code: `ai-validation-enabled = false` keeps calling the API (tag suggestions are on in every one of the 57 cities where the validator runs) and writes `label_ai_assessment` with no vote. But a shadow-period assessment then suppresses the vote: `getLabelsToValidateWithAi` skips any label already assessed as its current type, so those labels never get a vote when validation is switched back on unless the shadow assessments are deleted first. | Shadow with `ai-validation-enabled = false`; plan the deletion (or a re-assess path) before switching on; both belong in #52. |
| F8 | Low–Medium | Train/serve parity was measured on arm64/mps only; GPU depth and GPU classifier noise are unmeasured, so a correct deploy can miss the smoke test's 0.01 tolerance (sidewalk-ai-api#18 says so). | Run the smoke test on a `dev` container on makelab2 built from the PR branch, before merging; measure before loosening. |
| F9 | Low | `coord_frame` is hard-coded `"level"`. Training rendered AI-authored labels *without* tilt (all 220 AI-authored CurbRamp rows have `tilt_applied = 0`), so when [#5790](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5790) starts sending AI labels they must go as `"rig"`. | Precondition for #5790, not for this deploy. |

Verified non-issues (checked, no action): the website drops poses beyond ±30° and the API rejects them, and no
round-3 training row has one (0 of 64,465 ok rows; `camera_pitch` is null in 0 rows and roll is null everywhere); the
depth checkpoint is the same file (`depth_anything_v2_metric_vkitti_vitb.pth`, pinned to HF commit `e96fd0c8` in the
API) and the same encoder (`vitb`) on both sides; `generation_run.json` records the package at `7ce25ff`, which is the
`v1.0.0` tag the API pins; transformers 4.55 (`use_fast=False`) and 5.20 (`backend="pil"`) produce identical tensors
(parity doc); every configured city's `ai-validation-min-accuracy` is `0.92` (60 explicit entries, `staging` inherits
one of them); the map lookup walks the dict in insertion order and the maps are written ascending with values rounded
down; the on-submit path filters `panoSource == Gsv` in `ExploreController` before calling the AI, so Mapillary labels
do not reach it; Obstacle Agree, which the Phase-0 map (sidewalk-ai-api#16) clears only at confidence 0.999996 (map
max 0.947), clears on the new map at confidence 1.0.

### F1. Disagree: the prior is distorted by the training rule, and the gate-clearing type fails where it matters

**Scenario.** sidewalk-ai-api#19 merges with its maps as they are (byte-identical to `calibrate()` under the file
prior: Crosswalk `incorrect` reaches 0.977, NoCurbRamp `incorrect` 0.940). Crosswalk Disagree clears the gate for 42%
of its verdicts and NoCurbRamp Disagree for 21%. In a deployment outside the training countries a share of those
Crosswalk Disagrees are wrong by the human majority; each one hides a correct label from clustering, the API and the
Gallery (`LabelTable.majorityCorrect`: a lone AI Disagree sets `correct = FALSE`) and counts against the labeler
([SidewalkWebpage#4402](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4402)). That is the 2025 failure
again: the served Disagree accuracy was 0.93 and the observed one 0.46 (`docs/calibration-2026-10.md`).

**Evidence.** `calibration/calibrate.py` on the round-3 `predictions.csv` (fit on `val`, the default) under three
priors: the `deployment_priors.json` the generation run wrote (measured 2026-10-20), a production estimate (the
≥2-vote majority share with ties counted as correct, 0.041/0.065/0.204/0.224/0.152; the ties-out variant in the
attribution table below moves the Agree numbers by at most 0.01 and the held-out Crosswalk Disagree to 0.293 at
0.548), and the 90:10 of [#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4).
Coverage and precision are at the 0.92 gate, reweighted to the prior:

| Type | Prior (incorrect share) | Eval split | Agree cov / prec | Disagree cov / prec (gated n) |
|---|---|---|---|---|
| CurbRamp | file 0.118 | test | 0.883 / 0.994 | 0 / – |
| CurbRamp | file 0.118 | test_heldout | 0.751 / 0.991 | 0 / – |
| CurbRamp | production (ties→correct) 0.065 | test · test_heldout | 0.886 / 0.997 · 0.755 / 0.995 | 0 / – |
| NoCurbRamp | file 0.374 | test | 0.440 / 0.973 | **0.206 / 0.978** |
| NoCurbRamp | file 0.374 | test_heldout | 0.372 / 0.933 | 0.056 / 1.000 |
| NoCurbRamp | production (ties→correct) 0.204 | test · test_heldout | 0.710 / 0.985 · 0.634 / 0.956 | 0 / – |
| NoCurbRamp | 90:10 | test | 0.857 / 0.991 | 0 / – |
| Obstacle | file 0.358 | test · test_heldout | 0.475 / 0.959 · 0.489 / 0.970 | 0 / – |
| Obstacle | production (ties→correct) 0.224 | test · test_heldout | 0.499 / 0.978 · 0.528 / 0.984 | 0 / – |
| SurfaceProblem | file 0.320 | test | 0.676 / 0.957 | 0 / – |
| SurfaceProblem | file 0.320 | test_heldout | 0.528 / **0.890** | 0 / – |
| SurfaceProblem | production (ties→correct) 0.152 | test · test_heldout | 0.698 / 0.983 · 0.558 / 0.955 | 0 / – |
| Crosswalk | file 0.061 | test | 0.991 / 0.995 | **0.422 / 1.000** |
| Crosswalk | file 0.061 | test_heldout (n=138) | 0.864 / 0.996 | **0.319 / 0.643** (32 gated, **1 wrong**) |
| Crosswalk | production (ties→correct) 0.041 | test | 0.992 / 0.997 | **0.380 / 1.000** (98) |
| Crosswalk | production (ties→correct) 0.041 | test_heldout | 0.865 / 0.997 | **0.292 / 0.541** (32, 1 wrong) |
| Crosswalk | 90:10 | test_heldout | 0.862 / 0.992 | 0.356 / 0.754 |

Three things in that table. Disagree never clears the gate for CurbRamp, Obstacle or SurfaceProblem under any prior,
so for those three the cap costs nothing. NoCurbRamp Disagree's 21% coverage exists only under the file prior and is 0
under the production estimate. Crosswalk Disagree clears under every prior, and on the held-out cities its precision
is 0.64 (file prior) or 0.54 (production prior), which is **one** false Disagree among 32 gated verdicts (unweighted
0.969) reweighted to a 4–6% incorrect share: the held-out set cannot say whether the real rate is 0.54 or 0.97, only
that the test-split 1.000 is not to be trusted. The cap is needed under every prior for that reason alone.

**Where the prior gap comes from.** It is not a training-versus-production population difference. `deployment_priors.json`
is written by `data_generation.py` under the asymmetric margin rule that defines the training classes: `correct` needs
net human Agree > 2, `incorrect` needs net human Disagree > 1 (`eligible_labels`, "unchanged since the first
models"). Applying exactly that rule to the production ≥2-vote rows of the 2026-10-17 pull reproduces the file prior;
a symmetric rule or the plain ≥2-vote majority does not:

| Type | File prior | Asymmetric rule on production rows (n) | Symmetric \|net\| ≥ 2 (n) | ≥2-vote majority, ties out (n) | ≥1-vote majority (n) |
|---|---|---|---|---|---|
| Crosswalk | 0.061 | 0.059 (19,751) | 0.036 (32,404) | 0.042 (33,124) | 0.051 (62,406) |
| CurbRamp | 0.118 | 0.119 (33,173) | 0.053 (73,968) | 0.070 (79,147) | 0.085 (198,203) |
| NoCurbRamp | 0.374 | 0.359 (21,696) | 0.212 (36,815) | 0.230 (42,467) | 0.201 (76,025) |
| Obstacle | 0.358 | 0.365 (18,569) | 0.229 (29,672) | 0.262 (36,194) | 0.275 (75,396) |
| SurfaceProblem | 0.320 | 0.318 (20,903) | 0.149 (44,609) | 0.182 (52,878) | 0.189 (114,733) |

The rule admits an `incorrect` label on two net Disagree votes but a `correct` one only on three net Agree votes, so it
keeps a larger share of the incorrect labels than of the correct ones and the "prior" it writes is the class mix of
that filtered set, not of the labels the website sends. The website's own notion of the truth (`majorityCorrect`) is
the ≥2-vote majority, whose incorrect share is roughly half the file's for the three hard types. Whether the Triage
queue makes the ≥2-vote sample more or less contested than the traffic is **unverified**, and the evidence points both
ways: ≥1-vote majorities have a *higher* incorrect share than ≥2-vote ones for four types and a lower one for
NoCurbRamp. Neither prior is fully right for `calibrate.py`, because it reweights classes that the asymmetric rule
defined; the only map that is sound by construction is one fitted on production outcomes for the new revision
(`calibrate_from_production.py`). **Follow-up issue** (to be filed by the coordinator): make the margin rule symmetric
or derive the prior from the ≥2-vote majority, and say which in the calibration report.

**What would prove this wrong.** Gated Disagree verdicts from the new revision whose ≥2-human outcome, in the first
few hundred per type, has a 90% Wilson lower bound at or above 0.92 (`calibrate_from_production.py` restricted to the
new revision). Then release that type's Disagree map and this finding retires for it.

### F2. Agree: coverage swings with the prior

**Scenario.** NoCurbRamp Agree covers 44% of verdicts under the file prior, 71% under the production estimate and 86%
at 90:10, so a choice no measurement pins decides whether a third of NoCurbRamp labels get an AI vote. SurfaceProblem
Agree on the held-out cities is 0.890 under the file prior (below the gate's 0.92) but 0.955 under the production
estimate, so the same choice decides whether SurfaceProblem votes in a new country are trustworthy on paper.

**Evidence.** The F1 table; SurfaceProblem's held-out AUC of 0.71 (0.90 on the training cities) in the retrain
results; private city A is the weakest city for every type but Crosswalk.

**What would prove this wrong.** Per-city Agree precision for the new revision at or above 0.92 in the deployments that
were held out, after enough ≥2-human outcomes (a few hundred per type) to say so with a Wilson bound.

### F3. One unparseable field is a silent, total outage

**Scenario.** A later edit writes a `MODEL_MAP` date as `"2026-10-21"`, or the API adds an `image_source` value the
website's `AiImageSource` enum does not have. `AiService.callAiApi` parses `validator_training_date` and
`tagger_training_date` with `DateTimeFormatter.ofPattern("MM-dd-yyyy")` and the source with `AiImageSource.withName`,
all inside one `try`; the exception is caught, logged as "Failed to parse AI API response for label N", and the result
is `None`. No vote, no assessment row, no failure row, on every label in every city, until someone reads the logs. The
nightly sweep retries the same labels forever.

**Evidence.** `app/service/AiService.scala` (`develop`), the parse and the `catch`; #19's entries (`10-21-2026`) parse,
and its tests check the revision format (`pinned_revision()`) but not the dates or the enum; the smoke test
(`regression/parity_smoke.py`) checks `validation_result` and `p(incorrect)` only.

**What would prove this wrong.** A test in sidewalk-ai-api that fails on a non-`MM-dd-yyyy` date in any entry and on
an `image_source` outside `{cache, download}`, plus the smoke test parsing the whole response with the website's rules.
Either makes the mistake impossible rather than unlikely.

### F4. The accuracy map and the model revision are not bound to each other

**Scenario.** The model map is bumped to the new HF revisions, the five `accuracy_mappings/*.json` are forgotten (or
the reverse). The old map, fitted to the old model's confidences, gates the new model. The deploy smoke test checks
`validation_result` and `p(incorrect)` only, so it passes.

**Evidence.** `main.py` loads `accuracy_mappings/{label_type}.json` from the checkout and the model by
`pinned_revision()` from HF, with no cross-check; #19's `regression/parity_expected.csv` has no accuracy column;
`publish_to_hf.py` stages the same map as `confidence_precision_map.json` in the model repo, but
`download_hf_models.py` does not fetch it. #19's placeholder guard stops an *unpinned* model from loading, which is a
different hazard.

**What would prove this wrong.** Nothing; it is a process gap. Close it in #19 or right after: fetch
`confidence_precision_map.json` at the pinned revision and refuse to start if it differs from the checked-in map (or
drop the checked-in copy), and add `validation_estimated_accuracy` to the expected file when it is regenerated.

### F5. The Google-download path was never compared with training

**Scenario.** A label's panorama is not in the store (`SIDEWALK_SCRAPES_DIR`), so `Panorama._fetch_panorama` stitches
tiles from Google at zoom 4 (8192 px; or zoom 3, 4096 px, when zoom 4 has no tile), crops black borders, and resizes
to 8192×4096. Training read every panorama from the store (native 16384×8192 for 87–92% of rows in every type,
13312×6656 for most of the rest, and 0.1–0.5% at 3328×1664, resized once with `cv2.INTER_LINEAR`). Different
downsampling, different JPEG generations, and for the zoom-3 fallback a 2× upsample: a different input distribution
for every cache miss, on top of the two skews #19 fixes.

**Evidence.** `sidewalk_ai_api/panorama.py`; the round-3 manifests' `pano_native_w/h`; the parity doc tests the cache
path only. `label_ai_assessment.ai_image_source` records `cache` or `download` per assessment, so the share is one
query away; the read-only production query was not run for this document (the session's permission classifier
declined it), so the share is unknown here.

**What would prove this wrong.** A small `download` share in production, or a parity run of one panorama through both
paths showing |Δp| at the level of the WebP skew or less.

### F6. New-revision votes cover only not-yet-assessed labels

**Scenario.** The first week's data for the new revision comes from `getLabelsToValidateWithAi`, which skips any label
with an assessment of its current type and orders by unvalidated-first, oldest imagery first, up to 800 labels per
city per nightly run. The numbers are per city backlog, not a platform sample, and labels the 2025 model hid stay
hidden until #5793 is decided.

**Evidence.** `LabelTable.getLabelsToValidateWithAi` and `aiAssessmentIsCurrent` (`develop`).

**What would prove this wrong.** Nothing; it is by design. The consequence is only in how the first 48 h are read.

### F7. Shadow mode exists today, and it has a trap

`AiService` calls the API whenever `ai-validation-enabled` **or** `ai-tag-suggestions-enabled` is on for the city and
always writes `label_ai_assessment`; it submits a vote only when `ai-validation-enabled` is on. On `develop` the two
sets are identical (57 cities each), so `ai-validation-enabled = false` is a true shadow mode in every city where the
validator runs: the new revision's verdict, accuracy and confidence are recorded for every label and no vote is cast.
(`ai-validation-min-accuracy.<city>` above 1.0 would also keep the assessments while casting Unsure votes; it is not
needed anywhere today.)

The trap: `getLabelsToValidateWithAi` skips any label that already has an assessment of its current type
(`aiAssessmentIsCurrent`, F6), and `AiService` writes that assessment during the shadow. So every label assessed while
a city is shadowed is permanently skipped once validation is switched back on, and never gets the vote the shadow was
meant to earn it, unless the shadow-period rows (by `validator_training_date` or `validator_revision`, and
`label_validation_id IS NULL`) are deleted first, or a re-assess path is added. Either belongs in #52's design.

## 2. Checklist, in order

Preconditions, all verifiable before anything is built:

- [ ] **Hard blocker:** SidewalkWebpage #5795 is on `master` and deployed. Today it is not: `origin/master` is the
      2026-10-19 release (`4e177f4`), which carries neither #5795 nor the `validator_revision` evolution. The new
      models are trained on tilt-corrected crops, so serving them before the website sends `camera_pitch` is a skew on
      nearly every crop (training applied tilt to 99.9% of rows). Check a response's `tilt_applied` for a request from
      production, or the API log for `camera_pitch` in the form.
- [ ] Useful, not blocking: `label_ai_assessment.validator_revision`
      ([#5794](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5794)) on production. Until it is,
      `validator_training_date` (`10-21-2026` vs `08-16-2025`) separates the two models. `analysis/pull_ai_assessments.sh`
      pulls neither column yet; add both.
- [ ] Decide the deployment prior per type and write the decision into the calibration report (F1, F2). Whatever is
      chosen, evaluate on `test_heldout` too and keep both reports.
- [ ] Record the `ai_image_source` share per city for the last month (F5) and run `analysis/serving_parity.py` on one
      panorama through the cache and download paths.

**sidewalk-validator-ai**

1. [ ] `python -m calibration.calibrate --hold-back incorrect` (the flag the calibration work is adding; until it
       lands, do not post-process the maps by hand, because `publish_to_hf.py` would then stage a
       `calibration_report.json` for the uncapped map) for the five types, fit `val`, eval `test` and `test_heldout`.
       Every `incorrect` map capped at `HOLD_BACK_CAP` (F1).
2. [ ] `publish_to_hf.py --label_type <T> --out <root>` for each type; read the generated cards; fill
       `card_overrides.json` (no `TODO:` left). `--push` for the dry run, then `--push --yes` (Jon). Record the five
       model-repo commit shas. The old revisions remain addressable; nothing is rewritten.
3. [ ] README "Published artifacts" table and the retrain report point at the new revisions.

**sidewalk-ai-api** (#19 on top of #17; #19 already carries the opencv pin, the WebP round trip, the float64 crop
half-size, `API_VERSION` 1.2.0, the smoke test and the placeholder guard)

4. [ ] Replace #19's five uncapped `accuracy_mappings/*.json` with the capped maps from step 1 (the one open item in
       #19). Its monotonicity and grid tests still apply.
5. [ ] Replace every `PLACEHOLDER_REVISION` with the step-2 shas; confirm `training_date` (`10-21-2026`) is the date
       the cards say. Add the format test for both dates and the `image_source` enum (F3), and the map-to-revision
       check (F4).
6. [ ] Regenerate `regression/parity_expected.csv` for the new revisions. The real workflow, in this repo:
       `analysis/serving_parity.py train-worker --models` (training venv) and `api-worker --models` (API venv) against
       the #19 checkout with the step-5 revisions pinned, then `expected --train <train out> --api <api out> --out
       <csv>`, and copy the result over. `expected` writes `expected_p_incorrect` and `expected_validation_result` only;
       the `validation_estimated_accuracy` column (F4) is new code in `expected_values` and in `parity_smoke.py`, not
       a flag. The file in #19 holds the 2025-08 values and every row fails until then. **Generate it on x86, on the
       server's own CPU or on klone, never on a Mac:** opencv 4.12's `INTER_CUBIC` crop resize differs by CPU type
       (arm64 NEON vs x86 differs by 1 level on 2.5–4.2% of values; through the validators that is max |Δp| 0.050 and
       12 of 105 smoke rows over the 0.01 tolerance in an SSE-only emulation; the size on AVX-512 hardware is
       unmeasured). JPEG decode and `INTER_LINEAR` are byte-identical across the two. Found in the round-1 review of #19.
7. [ ] On makelab2, from the PR branch, **before merging**: `./docker_build.sh dev && ./docker_start.sh dev`; confirm
       VRAM fits beside prod (`nvidia-smi`; the dev container needs ~9–10 GB); `python3 regression/parity_smoke.py`
       against `127.0.0.1:60655` (F8): all `in_smoke_set` rows within 0.01, accuracy column equal, and the whole
       response parses the way the website parses it (F3). If it fails by a little, measure GPU depth and classifier
       deltas before touching the tolerance. `python3 regression/compare.py` prod vs dev: verdicts will differ (new
       models); the request/response shape, status codes and `Server-Timing` must not, within reason.
8. [ ] Review gate per the repo rule. After #17 merges, retarget #19 to `main` before #17's branch is deleted (the
       repo does not retarget stacked PRs), then merge.

**makelab2** (the API host)

9. [ ] Record what production runs now, before touching it: `docker ps` (container, image id),
       `docker inspect sidewalk-ai-api --format '{{.Image}} {{index .Config.Labels "org.opencontainers.image.revision"}}'`,
       `docker images sidewalk-ai-api` (version tags exist only for builds since 2026-10-14; an older image has only
       `latest`), and from inside the container the maps and revisions it serves:
       `docker exec sidewalk-ai-api sh -c 'cat API_VERSION; sha256sum accuracy_mappings/*.json; grep revision sidewalk_ai_api/model_map.py'`.
       Keep that output with the deploy notes; it is the rollback target and the answer to F4's "which map was live".
10. [ ] `git pull` the merged `main`, `./docker_build.sh && ./docker_start.sh`. `docker_build.sh` prints no tag line;
        read the new `<version>-<date>-<sha>` tag from `docker images sidewalk-ai-api` and write it down. One to two
        minutes of downtime; the nightly sweep picks up anything that failed meanwhile.
11. [ ] First response from production: `validator_revision` is a step-2 sha, `tilt_applied` is `true` for a request
        with `camera_pitch`, `api_version` is `1.2.0`.

**SidewalkWebpage**

12. [ ] No gate to flip: every configured city is at `0.92`. For deployments to shadow (the held-out-like ones,
        SurfaceProblem's weak countries; #52): `ai-validation-enabled = false` records without voting in every city
        where the validator runs today. Before switching a shadowed city back on, delete its shadow-period
        assessments (F7) or the labels assessed meanwhile never get a vote. Write both into #52.
13. [ ] Watch the website logs for `Failed to parse AI API response` and `returned error status` in the first nightly
        run after the deploy (F3, F8).

**First 48 hours** (all by `validator_training_date = 2026-10-21`, or `validator_revision` once it exists; read-only
on makelab1, port 5434):

- [ ] Volume per city and type, and the `ai_image_source` split (F5, F6).
- [ ] Gate pass rate per type and verdict: `validation_accuracy >= 0.92` grouped by `validation_result`. Disagree
      must be 0 while capped; Agree should land near the calibration report's coverage for the prior chosen, and a
      large gap is the prior being wrong (F2).
- [ ] Error rows: `label_ai_failure` growth, 502s, 400s (`InvalidField`), and worker restarts in `docker logs`.
- [ ] Latency: `Server-Timing` totals vs before (the render is unchanged; the WebP round trip adds a few ms).

**After one to two weeks**

- [ ] `analysis/pull_ai_assessments.sh` (with the two new columns) and `calibration/calibrate_from_production.py
      --since <deploy date>` restricted to the new revision: Agree precision per type and per city against ≥2-human
      majorities; Disagree precision of the *recorded* verdicts (they are stored even while capped).
- [ ] Release a type's Disagree map only where the Wilson lower bound clears 0.92 in production (F1). Release
      SurfaceProblem Agree in shadowed deployments on the same rule (F2, #52).
- [ ] Decide #5793 (what to do with the 2025 model's Disagree votes) with the new numbers in hand.

## 3. Rollback

- **API**: `docker images sidewalk-ai-api`, `docker tag sidewalk-ai-api:<previous-tag> sidewalk-ai-api`,
  `./docker_start.sh`. If production predates version tagging (builds before 2026-10-14 carry only `latest`), the
  rebuild moves `latest` and the old image is left untagged, so there is no `<previous-tag>`: use the image ID
  recorded in step 9, `docker tag <image id> sidewalk-ai-api`, then `./docker_start.sh`. Each image carries its own models and maps, so this restores the 2025 models and whatever maps
  that image was built with. Those are the Phase-0 maps (every Disagree held back) **only if production was built
  from sidewalk-ai-api#16 or later**; an older image serves the original 2025 maps, whose Obstacle `incorrect` map
  reaches 1.0. Step 9's record says which: an `API_VERSION` of 1.0.5 or later, or a `sha256sum` of
  `accuracy_mappings/obstacle.json` matching #16's, means Phase 0. Website: nothing to do. The previous image predates
  #17, so it does not know the pose fields; it ignores unknown form fields, so the website's requests keep working,
  uncorrected as before.
- **Hugging Face**: nothing to undo; the previous revisions are still pinned by the previous image and still exist in
  the repos.
- **Data**: assessments and votes cast by the new revision stay. Because Disagree was capped, the only votes to
  consider retracting are Agree votes, which for the 2025 models were right 94–99% of the time at the gate against
  ≥2-human majorities (CurbRamp 96.6%, NoCurbRamp 94.8%, Obstacle 94.1%, SurfaceProblem 94.1%, Crosswalk 98.6%;
  `docs/calibration-2026-10.md`, shipped-map gate precision) and are not a hiding risk. If a retraction is ever needed,
  `validator_training_date` (or `validator_revision`) plus `label_validation_id` on `label_ai_assessment` identify
  exactly the rows, and the Disagree→Unsure conversion of #5793 is the precedent for how.
- **Calibration**: if the first-48-h gate pass rates are far from the report (F2), the fastest safe move is to cap the
  affected Agree map too and rebuild, not to reinstate the old model.

## Appendix: how the numbers were made

- Calibration table: `calibration.calibrate.calibrate(df, prior, "val", eval_split)` on each type's
  `retrain-2026-10/<T>/predictions.csv` (val/test/test_heldout rows, e.g. CurbRamp 1,487/1,479/546), priors from
  `retrain-2026-10/deployment_priors.json` (`measured_on` 2026-10-20), the ≥2-vote majority shares of the production
  pull with ties counted as correct (`h_dis > h_agree`), and `{correct: 0.9, incorrect: 0.1}`. Coverage and precision are the report's `gate_coverage` /
  `gate_precision` per verdict; the gated n and the wrong count come from `verdicts()` and `api_lookup_many` on the
  same rows.
- Prior attribution: `ai_assessments_rounded4.csv` from the 2026-10-17 pull (`analysis/pull_ai_assessments.sh`,
  52 schemas), `net = h_agree - h_dis`; asymmetric rule `net > 2` / `-net > 1` as in `eligible_labels`; symmetric
  `|net| >= 2`; majorities with ties excluded.
- Training-side facts (`tilt_applied`, pose ranges, native sizes, `include_ai_labels`, package pin) from the round-3
  `manifest.csv` and `generation_run.json` of each type.
- Serving-side facts from sidewalk-ai-api `174a169` and #19 at `c6bf89d` (`main.py`,
  `sidewalk_ai_api/{panorama,validator,model_map}.py`, `Dockerfile`, `docker_*.sh`, `download_hf_models.py`,
  `regression/parity_smoke.py`, `tests/test_app.py`) and SidewalkWebpage `develop` and `master`
  (`AiService.scala`, `LabelTable.scala`, `LabelAiAssessmentTable.scala`, `cityparams.conf`).
