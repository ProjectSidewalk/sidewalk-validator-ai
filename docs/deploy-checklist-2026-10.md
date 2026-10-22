# Pre-deploy red team and checklist: shipping the round-3 validators (October 2026)

The round-3 models (five label types, tilt-corrected depth-sized crops, panorama-grouped split, held-out cities;
`docs/calibration-2026-10.md` and the retrain report) are about to replace the 2025-08-16 production models that
[sidewalk-ai-api](https://github.com/ProjectSidewalk/sidewalk-ai-api) serves. This document asks one question of the
whole chain, adversarially: **what breaks when they go live?** It reads this repo at `c68459a`, sidewalk-panorama-tools
`v1.0.0` (`7ce25ff`), sidewalk-ai-api `tilt-fields` at `174a169`
([#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17)) with
[sidewalk-ai-api#18](https://github.com/ProjectSidewalk/sidewalk-ai-api/issues/18), and SidewalkWebpage `develop` after
[#5795](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/5795) (camera pose and `coord_frame` in the `/process`
request). Every claim below was checked against code or reproduced on the round-3 predictions; each finding says what
evidence would prove it wrong, so the next person can retire it rather than argue with it.

Three parts: ranked findings (most severe first), the ordered checklist across the four repositories, and rollback.
Private deployments are written as "private city A (outside the US)" and "private city B (US)", as in the retrain
report.

## 1. Findings, ranked

| # | Severity | Finding | One-line fix |
|---|---|---|---|
| F1 | **High** | The Disagree maps are fitted under a class prior that production contradicts, and the two types where Disagree would clear the 0.92 gate (Crosswalk, NoCurbRamp) are the two where the evidence says it should not. A lone AI Disagree sets `label.correct = FALSE` and hides the label. | Ship every `incorrect` map capped at `HOLD_BACK_CAP` (0.9199); release Disagree per type only from production outcomes on the new revision. |
| F2 | **High** | The Agree maps move a lot with the same prior choice (NoCurbRamp coverage 44% vs 86%), and SurfaceProblem Agree on the held-out cities is below the gate's precision (0.890) under the file prior. | Decide and document the prior; evaluate on `test_heldout` as well as `test`; gate SurfaceProblem per deployment ([#52](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/52)). |
| F3 | **High** (silent outage) | `MODEL_MAP.training_date` must be `MM-dd-yyyy`; the website parses it inside a `try` and drops the whole response on failure. An ISO date in the new entries stops every AI vote everywhere with nothing but a warning per label. | Add a format test in sidewalk-ai-api; have the smoke test parse the response the way `AiService` does. |
| F4 | Medium | Nothing binds `accuracy_mappings/<type>.json` to the pinned model revision; the deploy smoke test has no accuracy column, so a stale map passes it. | Add `validation_estimated_accuracy` to `expected_process.csv`; record the map's sha256 (or load the map from the HF snapshot) in the API. |
| F5 | Medium | Training only ever saw panoramas from the lab's store; `/process` falls back to stitching Google tiles (zoom 4, or zoom 3 upsampled) on a cache miss. That path's pixels were never compared with training's, and the share of production assessments that take it is unmeasured here. | Measure the `ai_image_source` share; run `serving_parity.py` on one panorama through both paths before the deploy. |
| F6 | Medium | The new model never re-assesses a label the old one assessed (`aiAssessmentIsCurrent` is a label-type match only). Old Disagree-hidden labels stay hidden ([#5793](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5793)), and the first days of new-revision data come from the nightly queue's backlog order, not a random sample. | Measure by `validator_revision` and by city; do not read the first 48 h as a platform-wide precision. |
| F7 | Medium | The only no-code "shadow mode" is `ai-validation-min-accuracy.<city>` above 1.0, which still casts an AI *Unsure* vote per label (counted in `unsure_count` and the queue order); `ai-validation-enabled = false` records nothing at all. | Use the min-accuracy lever knowingly for a city-level shadow; a true assess-without-voting mode is the SidewalkWebpage side of #52. |
| F8 | Low–Medium | Train/serve parity was measured on arm64/mps only; GPU depth and GPU classifier noise are unmeasured, so a correct deploy can miss the smoke test's 0.01 tolerance (sidewalk-ai-api#18 says so). | Run the smoke test on the `dev` container on makelab2 before switching prod; measure before loosening. |
| F9 | Low | The on-submit path (`validateLabelsWithAi` → `getLabelDataForAi`) has neither the GSV filter nor the already-assessed guard of the nightly path. | Pre-existing; note for the website; harmless unless a Mapillary city enables AI validation. |
| F10 | Low | `coord_frame` is hard-coded `"level"`. Training rendered AI-authored labels *without* tilt (220 CurbRamp rows), so when [#5790](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5790) starts sending AI labels they must go as `"rig"`. | Precondition for #5790, not for this deploy. |

Verified non-issues (checked, no action): the website drops poses beyond ±30° and the API rejects them, and no
round-3 training row has one (0 of 64,465 ok rows; `camera_pitch` is null in 0 rows and roll is null everywhere); the
depth checkpoint is the same file (`depth_anything_v2_metric_vkitti_vitb.pth`, pinned to HF commit `e96fd0c8` in the
API) and the same encoder (`vitb`) on both sides; `generation_run.json` records the package at `7ce25ff`, which is the
`v1.0.0` tag the API pins; transformers 4.55 (`use_fast=False`) and 5.20 (`backend="pil"`) produce identical tensors
(parity doc); every city's `ai-validation-min-accuracy` is `0.92` (48 explicit entries, `staging` inherits one of them); the
map lookup walks the dict in insertion order and the maps are written ascending with values rounded down; Obstacle
Agree, which could never clear the gate on the 2025 map (max 0.915), now does at confidence 1.0.

### F1. Disagree: the prior is unverified and the gate-clearing cases fail where it matters

**Scenario.** The new `incorrect` maps go live uncapped. Crosswalk Disagree clears the gate for 42% of its verdicts
and NoCurbRamp Disagree for 21%. In a deployment outside the training countries, roughly a third of those Crosswalk
Disagrees are wrong by the human majority, each one hides a correct label from clustering, the API and the Gallery
(`LabelTable.majorityCorrect`, a lone AI Disagree sets `correct = FALSE`) and counts against the labeler
([SidewalkWebpage#4402](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4402)). That is the 2025 failure
again: the served Disagree accuracy was 0.93 and the observed one 0.46 (`docs/calibration-2026-10.md`).

**Evidence.** `calibration/calibrate.py` on the round-3 `predictions.csv` (fit on `val`, the default), under the
`deployment_priors.json` the generation run wrote (measured 2026-10-20) and, for sensitivity, the 90:10 prior of
[#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4). Coverage and precision are at the 0.92 gate,
reweighted to the prior:

| Type | Prior (incorrect share) | Eval split | Agree cov / prec | Disagree cov / prec |
|---|---|---|---|---|
| CurbRamp | file 0.118 | test | 0.883 / 0.994 | 0 / – |
| CurbRamp | file 0.118 | test_heldout | 0.751 / 0.991 | 0 / – |
| NoCurbRamp | file 0.374 | test | 0.440 / 0.973 | **0.206 / 0.978** |
| NoCurbRamp | file 0.374 | test_heldout | 0.372 / 0.933 | 0.056 / 1.000 |
| NoCurbRamp | 90:10 | test | 0.857 / 0.991 | 0 / – |
| Obstacle | file 0.358 | test | 0.475 / 0.959 | 0 / – |
| Obstacle | file 0.358 | test_heldout | 0.489 / 0.970 | 0 / – |
| SurfaceProblem | file 0.320 | test | 0.676 / 0.957 | 0 / – |
| SurfaceProblem | file 0.320 | test_heldout | 0.528 / **0.890** | 0 / – |
| SurfaceProblem | 90:10 | test_heldout | 0.566 / 0.972 | 0 / – |
| Crosswalk | file 0.061 | test | 0.991 / 0.995 | **0.422 / 1.000** |
| Crosswalk | file 0.061 | test_heldout (n=138) | 0.864 / 0.996 | **0.319 / 0.643** |
| Crosswalk | 90:10 | test_heldout | 0.862 / 0.992 | 0.356 / 0.754 |

Two things in that table. First, Disagree never clears the gate for CurbRamp, Obstacle or SurfaceProblem under any
prior, so for those three the cap costs nothing. Second, where it does clear, it fails the next check: Crosswalk
Disagree is perfect on the test split and 0.64 on the held-out cities; NoCurbRamp Disagree's 21% coverage exists only
under the file prior and vanishes at 90:10.

The file prior is the class mix among labels that met the training vote thresholds. A production estimate from the
2026-10-17 pull (`analysis/pull_ai_assessments.sh`; AI-assessed labels with at least two human Agree/Disagree votes,
52 schemas) gives a much smaller incorrect share, and that sample is itself enriched for contested labels by the
Triage queue, so the true traffic share is smaller still:

| Type | File prior | Production, ≥2 human votes (n) | Production, ties excluded |
|---|---|---|---|
| Crosswalk | 0.061 | 0.041 (34,037) | 0.042 |
| CurbRamp | 0.118 | 0.065 (85,563) | 0.070 |
| NoCurbRamp | 0.374 | 0.204 (47,761) | 0.230 |
| Obstacle | 0.358 | 0.224 (42,413) | 0.262 |
| SurfaceProblem | 0.320 | 0.152 (63,557) | 0.182 |

Overstating the incorrect share makes every Disagree map optimistic (and every Agree map pessimistic), which is the
direction the 2025 maps failed in. Importance weighting also assumes `p(image | class)` is the same in training and
production, and it is not: training labels are decisively voted, production sends everything, including the labels
humans would mark Unsure.

**What would prove this wrong.** Gated Disagree verdicts from the new revision whose ≥2-human outcome, in the first
few hundred per type, has a 90% Wilson lower bound at or above 0.92 (`calibrate_from_production.py` with the new
revision). Then release that type's Disagree map and this finding retires for it.

### F2. Agree: prior-sensitive coverage, and SurfaceProblem below the gate in new countries

**Scenario.** NoCurbRamp Agree covers 44% of verdicts under the file prior and 86% under 90:10, so the choice of
prior, which no measurement pins, decides whether half of NoCurbRamp labels get an AI vote. SurfaceProblem Agree votes
in a deployment like the held-out cities run at 0.89 precision while the website trusts them at ≥0.92.

**Evidence.** The table above; SurfaceProblem's held-out AUC of 0.71 (0.90 on the training cities) in the retrain
results; private city A is the weakest city for every type but Crosswalk.

**What would prove this wrong.** Per-city Agree precision for the new revision at or above 0.92 in the deployments that
were held out, after enough ≥2-human outcomes (a few hundred per type) to say so with a Wilson bound.

### F3. A wrong `training_date` string is a silent, total outage

**Scenario.** The new `MODEL_MAP` entries are written as `"2026-10-21"`. `AiService.callAiApi` parses
`validator_training_date` with `DateTimeFormatter.ofPattern("MM-dd-yyyy")` inside a `try`; a
`DateTimeParseException` is caught, logged as "Failed to parse AI API response for label N", and the result is
`None`. No vote, no assessment row, no failure row, on every label in every city, until someone reads the logs. The
nightly sweep retries the same labels forever.

**Evidence.** `app/service/AiService.scala` (`develop`), the parse and the `catch`; `sidewalk-ai-api/tests/test_app.py`
has no test on the format; the entries are hand-edited (`publish_to_hf.py` produces no `MODEL_MAP` line).

**What would prove this wrong.** A test in sidewalk-ai-api that fails on a non-`MM-dd-yyyy` string, or the smoke test
parsing the response with the website's rules. Either makes the mistake impossible rather than unlikely.

### F4. The accuracy map and the model revision are not bound to each other

**Scenario.** The model map is bumped to the new HF revisions, the five `accuracy_mappings/*.json` are forgotten (or
the reverse). The old map, fitted to the old model's confidences, gates the new model. The deploy smoke test
(`docs/serving-parity-2026-10/expected_process.csv`) checks `validation_result` and `p(incorrect)` only, so it passes.

**Evidence.** `main.py` loads `accuracy_mappings/{label_type}.json` from the checkout and the model by
`MODEL_MAP[...]["revision"]` from HF, with no cross-check; `expected_process.csv` has no accuracy column;
`publish_to_hf.py` stages the same map as `confidence_precision_map.json` in the model repo, but
`download_hf_models.py` does not fetch it.

**What would prove this wrong.** Nothing; it is a process gap. Close it in the API PR: fetch
`confidence_precision_map.json` at the pinned revision and refuse to start if it differs from the checked-in map (or
drop the checked-in copy), and add the accuracy column when `serving_parity.py expected` is regenerated.

### F5. The Google-download path was never compared with training

**Scenario.** A label's panorama is not in the store (`SIDEWALK_SCRAPES_DIR`), so `Panorama._fetch_panorama` stitches
tiles from Google at zoom 4 (8192 px; or zoom 3, 4096 px, when zoom 4 has no tile), crops black borders, and resizes
to 8192×4096. Training read every panorama from the store (native 16384×8192 for 87–92% of rows in the two types checked and
13312×6656 for most of the rest, resized once with `cv2.INTER_LINEAR`). Different downsampling, different JPEG generations, and for the
zoom-3 fallback a 2× upsample: a different input distribution for every cache miss, on top of the two skews #18 fixes.

**Evidence.** `sidewalk_ai_api/panorama.py`; the round-3 manifests' `pano_native_w/h`; the parity doc tests the cache
path only. `label_ai_assessment.ai_image_source` records `cache` or `download` per assessment, so the share is one
query away; the read-only production query was not run for this document (the session's permission classifier
declined it), so the share is unknown here.

**What would prove this wrong.** A small `download` share in production, or a parity run of one panorama through both
paths showing |Δp| at the level of the WebP skew or less.

### F6. New-revision votes cover only not-yet-assessed labels

**Scenario.** The first week's data for the new revision comes from `getLabelsToValidateWithAi`, which skips any label
with an assessment of its current type and orders by unvalidated-first, oldest imagery first. The numbers are per
city backlog, not a platform sample, and labels the 2025 model hid stay hidden until #5793 is decided.

**Evidence.** `LabelTable.getLabelsToValidateWithAi` and `aiAssessmentIsCurrent` (`develop`).

**What would prove this wrong.** Nothing; it is by design. The consequence is only in how the first 48 h are read.

### F7. What shadow mode exists today

Setting `ai-validation-min-accuracy.<city>` above 1.0 keeps assessments flowing (`label_ai_assessment` is written
either way) and turns every AI verdict into an Unsure vote. That is a usable per-deployment shadow, with a side
effect: the Unsure vote is a real `label_validation` row by the AI user, counted in `unsure_count` and in
`ValidationQueuePolicy.totalVotes`. `ai-validation-enabled = false` is not a shadow mode, it records nothing.
An assess-without-voting mode is the SidewalkWebpage half of #52.

## 2. Checklist, in order

Preconditions, all verifiable before anything is built:

- [ ] SidewalkWebpage #5795 is on **production**, not only `develop` (the API ignores unknown fields, so the order is
      safe either way, but tilt correction needs the fields to arrive). Check a recent request in the API logs for
      `camera_pitch`, or the `tilt_applied` field in a response.
- [ ] `label_ai_assessment.validator_revision` is populated on production for recent rows
      ([#5794](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5794)); it is the only way to tell the two
      models apart afterwards. `analysis/pull_ai_assessments.sh` does not pull the column yet; add it.
- [ ] Decide the deployment prior per type and write the decision into the calibration report (F1, F2). Whatever is
      chosen, evaluate on `test_heldout` too and keep both reports.
- [ ] Record the `ai_image_source` share per city for the last month (F5) and run `analysis/serving_parity.py` on one
      panorama through the cache and download paths.

**sidewalk-validator-ai**

1. [ ] `python -m calibration.calibrate` for the five types (fit `val`; eval `test` and `test_heldout`). Cap every
       `incorrect` map at `HOLD_BACK_CAP` (F1). `calibrate.py` has no hold-back flag today; `calibrate_from_production.py`
       does. Add one (or post-process) so the capped maps are produced by code, not by hand.
2. [ ] `publish_to_hf.py --label_type <T> --out <root>` for each type; read the generated cards; fill
       `card_overrides.json` (no `TODO:` left). `--push` for the dry run, then `--push --yes` (Jon). Record the five
       model-repo commit shas. The old revisions remain addressable; nothing is rewritten.
3. [ ] README "Published artifacts" table and the retrain report point at the new revisions.

**sidewalk-ai-api** (one PR on top of `tilt-fields`, which merges with it)

4. [ ] `MODEL_MAP.validator_models`: new `revision` per type, `training_date` in **`MM-dd-yyyy`** (F3). Add a test that
       asserts the format for every entry.
5. [ ] `accuracy_mappings/*.json` replaced by the staged `confidence_precision_map.json` files, and a startup check or
       test that they match the pinned revision's copy (F4).
6. [ ] #18: `opencv-python==4.12.0.88`, the in-memory q95 WebP round trip on a `validator_image` (tagger keeps the raw
       crop), the `int(6100 / float(depth))` crop rule.
7. [ ] Regenerate `expected_process.csv` with the new models (`serving_parity.py expected`), add
       `validation_estimated_accuracy` to it, and ship the smoke test as a script in the API repo.
8. [ ] Bump `API_VERSION`. Review gate per the repo rule, then merge.

**makelab2** (the API host)

9. [ ] `./docker_build.sh dev && ./docker_start.sh dev`; confirm VRAM fits beside prod (`nvidia-smi`; the dev container
       needs ~9–10 GB).
10. [ ] Smoke test against `127.0.0.1:60655` (F8): all `in_smoke_set` rows within 0.01, accuracy column equal. If it
        fails by a little, measure GPU depth and classifier deltas before touching the tolerance.
11. [ ] `python3 regression/compare.py` prod vs dev. Verdicts will differ (new models); what must not differ is the
        request/response shape, the status codes, and `Server-Timing` within reason.
12. [ ] `./docker_build.sh && ./docker_start.sh`. Write down the version tag it prints
        (`sidewalk-ai-api:<version>-<date>-<sha>`); that is the rollback target. One to two minutes of downtime; the
        nightly sweep picks up anything that failed meanwhile.
13. [ ] First response from production: `validator_revision` is the new sha, `tilt_applied` is `true` for a request
        with `camera_pitch`, `api_version` is the new one.

**SidewalkWebpage**

14. [ ] No flag to flip: every configured city is at `0.92`. For deployments to shadow (the held-out-like ones, SurfaceProblem's
        weak countries; #52), either set `ai-validation-min-accuracy.<city>` above 1.0 with F7's side effect in mind,
        or wait for #52's mechanism. Write the choice into #52.
15. [ ] Watch the website logs for `Failed to parse AI API response` and `returned error status` in the first nightly
        run after the deploy (F3, F8).

**First 48 hours** (all by `validator_revision = <new sha>`, read-only on makelab1, port 5434):

- [ ] Volume per city and type, and the `ai_image_source` split (F5, F6).
- [ ] Gate pass rate per type and verdict: `validation_accuracy >= 0.92` grouped by `validation_result`. Disagree
      must be 0 while capped; Agree should land near the calibration report's coverage for the prior chosen, and a
      large gap is the prior being wrong (F2).
- [ ] Error rows: `label_ai_failure` growth, 502s, 400s (`InvalidField`), and worker restarts in `docker logs`.
- [ ] Latency: `Server-Timing` totals vs before (the render is unchanged; the WebP round trip adds a few ms).

**After one to two weeks**

- [ ] `analysis/pull_ai_assessments.sh` (with `validator_revision`) and `calibration/calibrate_from_production.py
      --since <deploy date>` restricted to the new revision: Agree precision per type and per city against ≥2-human
      majorities; Disagree precision of the *recorded* verdicts (they are stored even while capped).
- [ ] Release a type's Disagree map only where the Wilson lower bound clears 0.92 in production (F1). Release
      SurfaceProblem Agree in shadowed deployments on the same rule (F2, #52).
- [ ] Decide #5793 (what to do with the 2025 model's Disagree votes) with the new numbers in hand.

## 3. Rollback

- **API**: `docker images sidewalk-ai-api`, `docker tag sidewalk-ai-api:<previous-tag> sidewalk-ai-api`,
  `./docker_start.sh`. Each image carries its own models and maps, so this restores the 2025 models *and* the maps
  that were serving before (the Phase-0 maps, which hold back every Disagree). Website: nothing to do; it keeps
  sending the pose fields, which the previous image already accepts (`tilt-fields` is in both) or ignores.
- **Hugging Face**: nothing to undo; the previous revisions are still pinned by the previous image and still exist in
  the repos.
- **Data**: assessments and votes cast by the new revision stay. Because Disagree was capped, the only votes to
  consider retracting are Agree votes, which were right 96–99% of the time for the old model and are not a hiding
  risk. If a retraction is ever needed, `validator_revision` plus `label_validation_id` on `label_ai_assessment`
  identify exactly the rows, and the Disagree→Unsure conversion of #5793 is the precedent for how.
- **Calibration**: if the first-48-h gate pass rates are far from the report (F2), the fastest safe move is to cap the
  affected Agree map too and rebuild, not to reinstate the old model.

## Appendix: how the numbers were made

- Calibration table: `calibration.calibrate.calibrate(df, prior, "val", eval_split)` on each type's
  `retrain-2026-10/<T>/predictions.csv` (val/test/test_heldout rows, e.g. CurbRamp 1,487/1,479/546), priors from
  `retrain-2026-10/deployment_priors.json` (`measured_on` 2026-10-20) and `{correct: 0.9, incorrect: 0.1}`. Coverage
  and precision are the report's `gate_coverage` / `gate_precision` per verdict; "gate at" is the first map key at or
  above 0.92.
- Production prior: `ai_assessments_rounded4.csv` from the 2026-10-17 pull (`analysis/pull_ai_assessments.sh`), rows
  with `h_agree + h_dis >= 2`, incorrect = `h_dis > h_agree`; ties reported separately.
- Training-side facts (`tilt_applied`, pose ranges, native sizes, `include_ai_labels`, package pin) from the round-3
  `manifest.csv` and `generation_run.json` of each type.
- Serving-side facts from sidewalk-ai-api `174a169` (`main.py`, `sidewalk_ai_api/{panorama,validator,model_map}.py`,
  `Dockerfile`, `docker_*.sh`, `download_hf_models.py`, `tests/test_app.py`) and SidewalkWebpage `develop`
  (`AiService.scala`, `LabelTable.scala`, `LabelAiAssessmentTable.scala`, `cityparams.conf`).
