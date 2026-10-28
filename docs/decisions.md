# Decisions

Choices about the training data that are deliberate, not oversights, with the evidence behind each one and
what would change it. One section per decision, newest first.

## Labels on Mapillary and Panoramax imagery are cropped with their provider's pose (October 2026)

Issue [#72](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/72), step 1. **Supersedes** the next
section ("Labels on non-GSV imagery stay out of training").

**Decision.** `data_generation.py` no longer drops labels whose `pano_source` is not `gsv` (`valid_geometry` has no
`non_gsv` check). Each human label's crop is corrected with a pose read the way its provider stores it
(`crop_pose.pose_for_row`), and a label with no usable pose is rendered uncorrected and counted, not dropped. The
manifest gains `pose_source` (where the pose came from) and `pose_pitch`/`pose_roll` (the pose handed to
`label_view`, in the geometry package's sign); `camera_pitch`/`camera_roll` stay the database columns as stored, and
`tilt_beta` is now the beta of the row's provider. `generation_run.json` records `source_beta`, `mapillary_api` and
per-provider `pose_counts`; the first two are resume settings, because both change non-GSV crops.

**Why the original reason no longer holds.** The pano store has held Mapillary and Panoramax panoramas since September
2026 (ProjectSidewalk/SidewalkWebpage#4865), sidewalk-ai-api reads that store before Google, and the round-3 CurbRamp
validator, never trained on Mapillary, scored AI-placed Mapillary labels at AUC 0.98 (Richmond) and 0.93 (Laurens) on
rig-frame crops (#69). What was missing was only the per-source pose, below.

The geometry package's sign is pitch > 0 = nose down, roll > 0 = left side up, with the click off the rig pixel by
beta · T(b), T(b) = pitch cos b + roll sin b.

| `pano_source` | `pose_source` | Pose handed to `label_view` | beta | Evidence |
|---|---|---|---|---|
| gsv | `gsv_db` | DB `camera_pitch`/`camera_roll` unchanged (as before) | 0.95 (`--tilt-beta`) | measured, sidewalk-panorama-tools#54 |
| mapillary | `mapillary_db` | DB (−`camera_pitch`, +`camera_roll`) | 1.0 | sign: pixel test below; beta unmeasured |
| mapillary | `mapillary_api` | Graph API `computed_rotation` → `mapillary_pitch_roll` → (−pitch, +roll), only with `--mapillary-token-env` | 1.0 | same |
| panoramax | `panoramax_db` | DB (−`camera_pitch`, +`camera_roll`), only when both are non-zero | 1.0 | sign: pixel test below; beta unmeasured |
| any | `image_estimate` | *reserved* for the image-based tilt estimator (ProjectSidewalk/RampNet PR 273), the fallback once it lands | — | not implemented |
| any | `none` | no correction; the reason is counted in the run summary | — | |
| (AI label) | `rig_frame` | no correction: placed on the rig image | — | as before |

**The signs were measured on pixels** (`tests/test_pose_levelling.py`, committed with its fixtures). Each panorama is
resampled into the gravity frame through the package's own `tilt_matrix` at the stored pose under each of the four
sign readings, and the lean of its near-vertical edges is measured (the vertical-edge estimator of
sidewalk-panorama-tools `reports/scripts/tilt_frame.py`). The right reading flattens the lean profile; the wrong ones
add tilt. Median lean amplitude, degrees:

| Set | n | before | (+p, +r) | (+p, −r) | **(−p, +r)** | (−p, −r) |
|---|---|---|---|---|---|---|
| Panoramax fixtures (Bayonne FR, Etalab 2.0) | 8 | 4.98 | 6.43 | 7.22 | **0.42** (best on 8/8) | 4.15 |
| Mapillary fixtures (Richmond VA, Laurens IA, CC BY-SA 4.0) | 6 | 3.59 | 4.46 | 5.72 | **0.31** (best on 6/6) | 4.85 |
| All 80 Mapillary panoramas fetched (40 per city, 20 most tilted + 20 at random, DB pose present) | 80 | 1.82 | 1.98 | 2.73 | **0.95** (best on 52) | 2.32 |
| the same, tilt ≥ 5° | 46 | 2.45 | 2.34 | 3.55 | **0.88** (best on 36) | 3.41 |

So **both website viewers store pitch positive nose UP and roll in the package's sign**: the opposite pitch sign from
GSV's DB column. For Panoramax this agrees with the desk check (sidewalk-panorama-tools
`reports/2026-10-06-non-gsv-levelling.md`: streetlevel pitch = −`pers:pitch`, roll = +`pers:roll`). For Mapillary,
`MapillaryViewer.extractPitchRoll` computes pitch as `asin(forward_up)`, the elevation of the forward axis.

**Mapillary DB pose = the parse of `computed_rotation`.** On the 80 panoramas above, the DB `camera_pitch` and
`camera_roll` equal `crop_pose.mapillary_pitch_roll(computed_rotation)` to within 3e-14 degrees, roll with the same
sign, so `mapillary_db` and `mapillary_api` take the same conversion. (The port was also checked against the
auto-labeler's own derived poses on 4,999 panoramas of another Mapillary deployment: within 1.3e-14 degrees, and the
same 22 over-tilt refusals.) Since SidewalkWebpage#5725 the server fills an empty Mapillary pose from the same vector,
so a DB pose no longer means a person viewed the panorama.

**Panoramax 0/0 and lone angles.** `PanoramaxViewer` stores `pers:pitch || 0` and `pers:roll ?? undefined`, and
levels the sphere only when both angles are non-zero (mirroring Panoramax's own viewer). A label on a panorama with a
zero or missing angle was therefore clicked on the un-levelled image and needs no correction; a 0/0 pose is also not
a measurement (Bayonne: pose absent 42%, real 37%, exactly 0/0 21%). These rows get `pose_source = none`, counted as
`panoramax_zero_pose`, `panoramax_no_pose` or `panoramax_single_angle_unlevelled`.

**Beta.** 1.0 for Mapillary and Panoramax (`crop_pose.SOURCE_BETA`), on the assumption that both viewers level the
whole sphere before the click. It is unmeasured; issue #72 step 2 measures it on Mapillary with sidewalk-auto-labeler's
`label_frame_beta.py`, and the table is one constant to change.

**Caveats.**

- One of the 80 Mapillary panoramas (Richmond, 11000 px wide, stored tilt 17.6°) reads as already level (lean
  amplitude 0.85° before, about 7° after levelling under every reading). Its pixels and its `computed_rotation`
  disagree, so its labels would be mis-corrected by about the full tilt. How common this is per contributor/camera is
  not measured; the image-based estimator would catch it.
- `mapillary_api` needs a token (`--mapillary-token-env MAPILLARY_ACCESS_TOKEN`); answers are cached in
  `<out>/<label_type>/mapillary_rotations.json`, and a lookup without an answer stops the run before any crop is cut.
  Without the flag, Mapillary labels with no DB pose are rendered uncorrected (`none/mapillary_no_db_pose_api_off`).
- Serving: sidewalk-ai-api must apply the same per-source pose before a model trained on these crops is used on
  Mapillary or Panoramax traffic (issue #72 step 3), and SidewalkWebpage still sends GSV labels only (step 4).
- Anything that re-renders crops from manifest columns (`analysis/sidewalk_seg.py`, `analysis/serving_parity.py`)
  reads `camera_pitch`/`camera_roll`; for non-GSV rows it must use `pose_pitch`/`pose_roll` instead.

## Every performance study is sliced by severity and tag (October 2026)

The seven label types are coarse. The severity a labeler picks and the tags they add carry far more of what is in
the crop: a CurbRamp tagged `missing tactile warning` or `surface problem`, a Crosswalk tagged `paint fading` or
`brick/cobblestone`. A model can do well on a type overall and badly on one of these. So every performance study
reports label type × severity and label type × tag next to its comparative baseline:

- **Model evaluations.** `evaluate.py` writes the slices by default (`metrics.json` `slices`, `slices.md`).
- **Retrain reports.** `analysis/retrain_report.py` gives them for the new model and, from `predictions_prod.csv`
  (`evaluate.py --checkpoint <served model> --tag prod`), round 3 against production on the same rows, with a
  panorama-resampled interval on the difference.
- **Production measurements.** They report the same slices against human votes.

**How the slices are counted.**

- A label with several tags counts once per tag.
- Missing severity and untagged labels are their own `(none)` slice, never dropped.
- Rates on fewer than 30 labels are left blank; the count is still shown. An AUC also needs at least 8 labels of
  each class.
- Each slice reports its largest one-deployment share, so a slice that is mostly one city's labels is visible.
- Intervals are not corrected for the number of slices: read a table of them as a screen.

**Where we need more data.** Each slice carries its training-row count, and fewer than 100 is flagged `⚠ thin`.

The implementation is [`analysis/slices.py`](../analysis/slices.py). New experiments should carry `severity` and
`tags` through to their outputs so the slices can be computed. `split.csv` has them since round 3.

## Labels on non-GSV imagery stay out of training (October 2026) — superseded

**Superseded by the section above ([#72](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/72)).** Kept
as the record of why the drop existed.

Issue [#9](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/9), item 4.

**Decision.** `data_generation.py` keeps dropping labels whose `pano_source` is not `gsv`, before selection.
They are counted as `non_gsv` in the "dropped for unusable pano geometry" line, which is printed after loading
and again in the end-of-run summary. In the 2026-10-17 rawLabels pull that is every label in three of the 41
public deployments: Richmond VA (13,348, Mapillary), Laurens IA (3,458, Mapillary) and Bayonne FR (75,
Panoramax). No other city has any.

**Why.** Reading the panoramas is not the problem. The pano store holds Mapillary and Panoramax panoramas
under the same `<city_id>/<pano_id[:2]>/<pano_id>.jpg` layout, and `read_store_panorama` would read them.
The problem is putting a crop on the right pixel. The crop is cut at the stored `pano_x`/`pano_y` after a
rig-tilt correction, and the size of that correction (`--tilt-beta`) was measured for GSV only
([sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54)). The
pano-store desk check for the other two sources
([sidewalk-panorama-tools#190](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/190), report
`reports/2026-10-06-non-gsv-levelling.md` from
[sidewalk-panorama-tools#210](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/pull/210)) found:

- **Panoramax** panoramas are not gravity-levelled when they carry a non-zero pose. Their pose is the STAC
  `pers:pitch`/`pers:roll`, with a measured sign convention different from GSV's: streetlevel pitch = −`pers:pitch`,
  roll = +`pers:roll`. The Project Sidewalk viewer levels these panoramas before the click, so the stored `pano_y`
  is expected to be off by the full tilt (β = 1), not GSV's ~0.95. The median tilt amplitude on Bayonne's labelled
  posed panoramas is about 5°, and four of 16 are at 10° or more. Panoramas with exactly one non-zero angle take a
  different viewer code path, which would need its own handling.
- **Mapillary** is unmeasured: the check had no API token, so it saw no pixels. The pose would come from the
  Graph API `computed_rotation`. Project Sidewalk's own copy (`camera_pitch`/`camera_roll`) is null for most
  panoramas: 313 of 9,282 in Richmond and 863 of 4,579 in Laurens have one.

Applying the GSV correction to these labels would put many crops off the feature, by a few degrees on Panoramax
and by an unknown amount on Mapillary. Serving does not need them either: SidewalkWebpage sends the validator GSV
labels only, on both of its paths, and sidewalk-ai-api's download fallback fetches GSV panoramas only. So there is
no production traffic from these sources to match.

**What would unblock it.**

1. A per-source pose reader in `sidewalk_panorama_tools.geometry`, used by `label_view`: Panoramax from
   `pers:pitch`/`pers:roll` with β = 1 and the single-angle case, and Mapillary from `computed_rotation`.
2. A crop-level check like #54's, in each source, showing that labels land on the corrected pixel.
3. Serving those sources in sidewalk-ai-api with the same correction, so training and serving cut the same crop.

Once all three land, remove the `non_gsv` check from `valid_geometry` and record `pano_source` (already a manifest
column) so the new rows can be ablated.
