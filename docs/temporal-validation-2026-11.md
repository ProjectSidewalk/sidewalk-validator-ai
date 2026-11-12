# Temporal validation: was the label wrong, or did the world change? (issue #17)

> **Status, 2026-11-12: design pre-registered; Stage 0 preview only; no gate has been run.** Every threshold below is
> **proposed, to be confirmed by Jon before any data is read for the gate**. The thresholds live in one file,
> [`temporal-validation-2026-11/gates.json`](temporal-validation-2026-11/gates.json) (dated, `confirmed_on: null`),
> so a later change shows in its diff. The Stage 0 numbers in this page come from camera positions *inverted* from the
> public rawLabels, not read from the database, and are marked **Preview (not the gate)**.

Issue: [#17](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/17). Code:
[`analysis/temporal_pairs.py`](../analysis/temporal_pairs.py) (Stage 0),
[`analysis/pull_pano_cameras.sh`](../analysis/pull_pano_cameras.sh) (the read-only database pull for the gate run),
[`tests/test_temporal_pairs.py`](../tests/test_temporal_pairs.py). Generated preview tables:
[`temporal-validation-2026-11/preview/`](temporal-validation-2026-11/preview/).

## 1. The premise, corrected

The issue says that a human validator who sees newer imagery will disagree with a label that was right when it was
placed, so some Disagree votes mean "the world changed", not "the label was wrong". The code says otherwise. Every
non-admin view shows the label's **own** panorama:

- **Validate** loads the label's `pano_id`, or the self-hosted backup of that same panorama when it has expired
  (SidewalkWebpage `428341df9`, 2026-11-11: `frontend/js/validate/label/LabelContainer.js:418-421` calls
  `setPanorama(panoId, backupImage, ...)`; `frontend/js/validate/panorama/PanoManager.js:523` is `setPanorama`, and
  `:175` sets `defaultNavigation: false` with the comment "Nothing in Validate follows a pano's links"). Mobile
  Validate shares these modules. The 2023 Validate also set the panorama by id
  (`public/javascripts/SVValidate/src/panorama/Panorama.js:275-292` at `f4b2dbc36`, 2023-05-31).
- **Gallery, LabelMap and the label popup** load `meta.pano_id`
  (`frontend/js/common/label-detail/LabelDetail.js:937` at `428341df9`).
- **One exception:** the admin popup enables click-to-go navigation
  (`frontend/js/common/label-detail/PopupPanoManager.js:253` at `b19a00fa2`: `defaultNavigation: !!this.#admin`), so
  an admin can walk to another panorama before voting.

So human Disagree votes judge a label against the image it was placed on; "the world changed" cannot explain them,
and this study does not try to re-read votes.

**What the experiment is for instead: currency.** A label that was right on its capture date may no longer be true
(a ramp built, an obstacle removed). The consumer is **re-audit priority** among streets that have newer imagery, and
a staleness note on exports. A model flag is worth building only if it beats two free baselines: (i) **imagery age
alone**, and (ii) the human signal of a **contradicting label on the newer panorama** (CurbRamp on A and NoCurbRamp at
the same point on B, or the reverse). Set (ii) is also a pool of likely real changes for estimating the model's
sensitivity. A `changed` value can never come from a single-image API request (section 10).

## 2. Data and scope

- **Deployments:** public ones only, checked against the live `/v3/api/cities` listing at run time (the module
  refuses to write tables naming a deployment that is not public in the listing it is given).
- **Imagery:** GSV only. The world bearing of a raster column is documented and replayed for GSV only (`ps_math`); the
  stored click is rig-frame on Mapillary and conditionally levelled on Panoramax (`crop_pose.py`), and nobody has
  measured their heading convention.
- **Pairs:** both panoramas already in the label store (a label on each), so no new imagery is needed. `pano_history`
  and historical-link captures are out of the pilot.
- **Camera positions.** rawLabels carries no camera position. Two sources:
  - **Inverted (preview).** Since SidewalkWebpage evolution 352 (`conf/evolutions/default/352.sql`, 2026-08-11,
    `3b59c9a56`) every stored `approximation2` position was recomputed as a spherical destination (R = 6,371 km) from
    the panorama's position along `camera_heading + lon` at the fitted range of the click's depression, and live
    submissions use the same estimator (`frontend/js/explore/label/Label.js` `toLatLng`,
    `app/service/PanoDataService.scala` `toLatLng`). Inverting it gives the camera exactly. Labels left at `depth`
    (pre-2021 positions from GSV depth data, which 352 did not touch) invert to the wrong place; rawLabels does not
    expose `computation_method`, so a panorama is kept only if all its labels invert to within 5 cm of their median
    (`inversion_agreement`), and a single-label panorama is kept unchecked. This check is bookkeeping, not a test of the
    heading: positions and clicks share the heading, so a wrong heading inverts consistently.
  - **Database (the gate).** `analysis/pull_pano_cameras.sh` reads `pano_data` (`lat`, `lng`, `camera_heading`,
    `camera_pitch`, `capture_date`, `width`, ...) and, with `MODE=methods`, `label_point.computation_method`, from each
    public schema on the database host, read-only. Only these positions can give a Gate 0 verdict; `positions
    --db-panos` also reports how far the inverted positions are from them per city (centimetres expected; if not, this
    page's preview was wrong and will say so).

## 3. Pairing, arms and leakage

**Anchor:** a label of one of the five served types (CurbRamp, NoCurbRamp, Obstacle, SurfaceProblem, Crosswalk) on
panorama A, keyed `label_uid = "<city>:<label_id>"` (asserted unique; every merge passes `validate=`); GSV; usable
pano point (`data_generation.valid_geometry`); **vote-decided** by `data_generation.eligible_labels` (margin rule; the
expert rule only when `EXPERTS` is set, which the run records as a boolean); human-authored; a parseable capture
month; A has a camera position; not a user photosphere.

**Partner B:** another GSV panorama in the same city with a camera position and a capture month, within 5 m of A.
The primary baseline cap (3 m, or 5 m if the 3-5 m bin passes too) is Gate 0's call. Arms by `dt = month_B -
month_A`:

| arm | rule | used for |
|---|---|---|
| `long_newer` | `dt >= +24` months | the primary arm (the only long arm in the pilot) |
| `control` | `1 <= |dt| <= 3` months | the near-in-time floor C2 (Stage 2) and the Stage 0 residual |
| `same_month` | `dt = 0` | the Stage 0 residual only (era comparison); **never a floor** |

One partner per (anchor, arm): the smallest baseline among QC-passing candidates, ties broken by the larger `|dt|`
and then `pano_id`; `pair_uid = "<label_uid>|<arm>"`. The negative control (section 7) is a flag on `long_newer`
pairs, not an arm.

**Leakage.** A pair is `clean` when neither A's nor B's `pano_id` appears in a `train` or `val` row of that type's
round-3 `split.csv`, matched on **`pano_id` alone across all cities** (it is the provider's global id; neighbouring
deployments such as the New Jersey cluster can hold one panorama in two schemas). The census reports the clean share
per type. Asserting that the local `split.csv` files are the ones behind the pinned Hugging Face revisions is a Stage 2
precondition; where that hash is recorded is unverified.

## 4. Re-projection

Positions are east/north metres about each pair's own mid-latitude on the R = 6,371 km sphere (the site's), so the
inversion stays exact and no city-wide cosine can scale a baseline (a test checks this at 70 degrees north). With
`h = 2.3412 m` (the site's fitted camera height):

1. A's click in the level frame: `lon_A = (pano_x / pano_width - 0.5) * 360`, `lat_A = (0.5 - pano_y / pano_height) *
   180`; bearing `theta_A = camera_heading_A + lon_A`.
2. Range `r_A = blend_range(-lat_A)`: the site's estimator, the flat-ground cotangent `h / tan(dep)` for depressions
   of 11.25 degrees or more and, closer to the horizon, a straight line with the cotangent's value and slope at 11.25
   degrees, clamped to [0, 50] m (copied from `ps_math.predict_blend_distance` with provenance, since the repo pins
   panorama-tools v1.0.0 and `ps_math` arrived in v1.2.0; identical to it to machine precision, checked against v1.2.0
   when written, and a parity test runs whenever `ps_math` imports). Its 0.41 m median error was measured on what it was
   fitted to: ground-contact clicks against provider depth.
3. Ground point `P = r_A (sin theta_A, cos theta_A)` east/north of A.
4. From B: `v = P - (B - A)`, `d_B = |v|`, `theta_B = atan2(v_E, v_N)`, `lon_B = wrap180(theta_B -
   camera_heading_B)`, and **`lat_B = -blend_depression(d_B)`**, the depression at which the same estimator gives
   `d_B`. (The plan wrote `-atan(h / d_B)`; that breaks the identity below 11.25 degrees, where the fitted tail is not
   the cotangent. With the inverse of the same estimator, a partner at A's own position returns A's click exactly. In
   the preview the change moves no median by more than 3 px; section 6.3.)
5. (Stage 2) The B crop is `data_generation.create_perspective_crop` at that level-frame point with B's
   `camera_pitch`, `camera_roll=None` when blank (as the A crop does), `apply_tilt=True`, beta 0.95, and
   `fixed_crop_half_px = round(crop_half_px_A * r_A / d_B)` (the same physical extent). Beta 0.95 was measured on human
   clicks; a computed point is exactly level-frame, a difference of 5% of a 1-3 degree tilt, under 3 px.

A synthetic two-camera test paints a disc on flat ground into two equirectangular panoramas from two positions and
headings (and, in two variants, a 3 degree rig pitch on B), re-projects A's click and renders B's view with the
training renderer: the disc lands within 3 px of the view centre (6 px with beta 0.95 against a fully applied pitch).

**Geometry QC, in this order, each rejection counted:** `no_camera_position`, `photosphere`, `near_horizon` (A's
depression below 5 degrees, where the tail is a bounded guess), `baseline` (b > 5 m), `b_range` (`d_B` outside
3-25 m), `range_ratio` (`d_B / r_A` outside 0.6-1.6), `b_near_horizon`, `parallax` (angle A-P-B above 25 degrees).
Stage 2 adds `no_pano`/`bad_pano`, `clipped` and `off_ground` (rescaled Depth Anything depth against `r_A` beyond a
factor 1.5, reported per type). Two checks of the first plan are gone because they are circular: the heading
self-check and the "website position" range check both re-derive what the label's stored lat/lng was computed from.
Depth is the only independent range check, and it needs a crop.

**Error budget (a prior; the Stage 0 measurement supersedes it).** A Monte Carlo of the exact 3D geometry at r = 8 m,
b = 3 m, heading sigma 2.2 degrees gives a median / p90 miss at B's view centre of 55 / 127 px at 0.5 m relative
position error per axis, 93 / 213 px at 1 m and 176 / 407 px at 2 m. Further terms: B's camera 0.3 m higher at
r = 5 m adds about 50 px vertically; an 8% grade along the ray at r = 15 m moves the ground point about 5 m; a click
1 m up a pole at 8 m is about 150 px off. The median round-3 crop half-width is about 238 px.

## 5. The stages and their gates

Four stages, cheapest first. A failed gate ends the study with a documented negative result: the tables of the stage
that failed are published as the result.

| stage | what | cost | gate |
|---|---|---|---|
| 0 | census of pairs; label-to-label re-projection residual | CPU, Mac | Gate 0: residual, era, yield |
| 1 | kill stage: same-panorama jitter flip rate (C1) | about 2,000 crops on klone | Gate 1: noise flip rate per type |
| 2 | partner crops and scoring; floors | klone GPU | Gate 2: flips available, net rate above the floor |
| 3 | masked hand-check, two raters | about 2 h + 30 min | primary precision, Wilson lower bound |

**Gate 0** (after Stage 0 on **database** positions; the preview does not count). All of, thresholds proposed:

| rule | proposed threshold |
|---|---|
| G0a residual | on unique matches of vote-decided anchors, 30 degree window, baseline <= 3 m: `long_newer` median <= **100 px** and >= **80%** within **240 px**, and the same for `control`; types are judged separately when they have n >= **50**, and a type that fails is dropped |
| G0b era | `long_newer` median residual <= **1.5x** the `same_month` median |
| G0c yield (public deployments) | >= **800** clean vote-correct `long_newer` pairs; >= **2** types with >= **150**; a negative-control pool >= **200** over CurbRamp, NoCurbRamp and Crosswalk |
| cap | 5 m instead of 3 m if the 3-5 m baseline bin of both arms also passes G0a |

**Gate 1** (Stage 1). 400 vote-correct, A0-verdict-correct, test-split anchors per surviving type, one jittered crop
each. A type passes if the Wilson upper 95% bound of its flip rate is <= **0.10** (about 7.0% observed at n = 400).
Continue if >= **2** types pass, with those types only. The stage publishes the implied floor `f` and the minimum true
change rate for a flag precision of 0.6 at sensitivity 0.7 (`f = 0.05` needs 9.7%, `f = 0.08` needs 14.6%; precision
is `pi s / (pi s + (1 - pi) f)`).

**Gate 2** (Stage 2). >= **60** `changed_c2i` flips among clean vote-correct `long_newer` pairs in the surviving types,
**and** the pooled model-detected change rate minus the floor has a lower 95% bound **above 0**.

**Stage 3 outcome (primary).** Among those flips, the share a masked rater marks "changed" on B ("not the same spot",
"occluded" and "can't tell" count as not changed). **Go rule: Wilson lower 95% bound >= 0.5** (0.60 observed at
n = 100). The primary is reported as unreliable if Cohen's kappa on the **50**-item overlap is below **0.6**.

## 6. Stage 0: the label-to-label residual

For every QC-passing (anchor, partner) whose B carries a human label of the same type, A's label is re-projected
into B and compared with B's own clicks, which are independent human measurements of where that feature is. B labels
the crowd voted incorrect (net Disagree > 1) and AI-placed labels are not measurements and are left out. Recorded per
candidate: the angle to the nearest same-type B click as view pixels (`1024 tan(angle)`, the 2048 px, 90 degree
training view), signed azimuth and elevation components (a heading or position error is mostly azimuthal; a camera
height, grade or range error is in elevation), the number of same-type B clicks within 30 degrees, `matched` (nearest
within 30 degrees) and `unique` (exactly one within 30 degrees). The gate statistic uses unique matches of
vote-decided anchors. Beside it, the **click floor**: the same statistic for two different users' same-type clicks on
one panorama within 15 degrees, with no geometry in between.

Two biases point in opposite directions and both are reported: two people click different parts of one feature,
which inflates the residual (the click floor shows by how much); and the uniqueness filter keeps isolated features,
which are easier, which deflates it (the `matched_any` table shows the residual without it).

### 6.1 Preview (not the gate)

**What this is and what it is not.** Camera positions are inverted from the public rawLabels (section 2), not read
from the database, so the module prints the Gate 0 values but no verdict. Limits:

- positions are **inverted, not read**: exact where `inversion_agreement` holds, unchecked on single-label panoramas
  (about a fifth of the gate-population pairs have at least one such panorama; `by_arm_positions_checked`), and
  panoramas whose labels disagree by more than 5 cm are excluded: 43,404 of 422,896 labelled GSV panoramas over the
  41 deployments, of them **26,178 in seattle-wa** (the pre-2021 `depth` rows evolution 352 left untouched), 7,027 in
  spgg, 4,203 in columbus-oh, and **4,183 in newberg-or, where 95% of multi-label panoramas disagree (median 1.7 m)**,
  for a reason not checked (its labels may not have been recomputed);
- residuals are **truncated** at the 30 degree window (591 px); the planner's preview truncated at 15 degrees;
- the gate statistic is on **unique-candidate matches** only;
- 41 deployments: every public deployment with a CSV in the 2026-10-17 cache. Five deployments public today are not in
  that cache (auckland, cuenca, la-ca, washington-dc, west-chester-pa), and the cache is not today's data.

Inputs: rawLabels cache `~/Data/sidewalk-validator-ai-prod-2026-10-17/rawlabels_cache/` (41 files; per-file sha256 in
[`preview/provenance.json`](temporal-validation-2026-11/preview/provenance.json)); the live `/v3/api/cities` listing
fetched 2026-11-12 (sha256 `a4a2fab7379cada5d583f1bffa329a5b1f065741af3f51123b0a15403ceb5aca`; 46 public deployments,
all 41 named here among them); round-3 `split.csv` files from `~/Data/sidewalk-validator-ai-retrain-2026-10/round3/`
(sha256s in the same file). Full tables: [`preview/residual.md`](temporal-validation-2026-11/preview/residual.md),
[`preview/census.md`](temporal-validation-2026-11/preview/census.md),
[`preview/gate0.json`](temporal-validation-2026-11/preview/gate0.json), CSVs beside them.

**Residual by arm** (unique matches, vote-decided anchors, baseline <= 3 m):

| arm | n | median_px | p75_px | p90_px | within_100 | within_240 | abs_az_median_px | abs_el_median_px | el_median_px |
|---|---|---|---|---|---|---|---|---|---|
| same_month | 306 | 74.5 | 154.6 | 260.0 | 0.647 | 0.873 | 59.1 | 26.9 | 1.1 |
| control | 104 | 88.6 | 160.8 | 345.6 | 0.558 | 0.837 | 76.4 | 20.2 | -1.9 |
| long_newer | 463 | 76.5 | 146.3 | 322.1 | 0.607 | 0.86 | 57.9 | 26.9 | 4.4 |

**By type** (exploratory; Obstacle and SurfaceProblem are thin):

| arm | label_type | n | median_px | p75_px | p90_px | within_100 | within_240 |
|---|---|---|---|---|---|---|---|
| control | Crosswalk | 23 | 122.9 | 156.7 | 207.2 | 0.348 | 0.913 |
| control | CurbRamp | 40 | 57.0 | 139.8 | 350.1 | 0.7 | 0.825 |
| control | NoCurbRamp | 29 | 90.6 | 146.7 | 309.6 | 0.586 | 0.828 |
| control | Obstacle | 4 | 52.9 | 91.1 | 97.7 | 0.75 | 1.0 |
| control | SurfaceProblem | 8 | 225.1 | 461.5 | 556.6 | 0.25 | 0.625 |
| long_newer | Crosswalk | 97 | 79.9 | 147.5 | 342.3 | 0.598 | 0.835 |
| long_newer | CurbRamp | 235 | 68.2 | 122.6 | 263.9 | 0.672 | 0.894 |
| long_newer | NoCurbRamp | 77 | 91.1 | 147.7 | 240.6 | 0.532 | 0.896 |
| long_newer | Obstacle | 17 | 106.9 | 263.7 | 346.5 | 0.471 | 0.706 |
| long_newer | SurfaceProblem | 37 | 123.0 | 284.4 | 452.3 | 0.432 | 0.703 |

**Era and camera generation, `long_newer`** (exploratory): B's click sits about 40 px *above* the prediction when A
is a 13,312 px panorama (the older camera) and B a 16,384 px one, against about 7 px below for 16,384 to 16,384. A
camera-height difference between generations would do this; the gate run's elevation split will show whether it
holds on database positions.

| width_pair | n | median_px | p75_px | p90_px | within_100 | within_240 | abs_az_median_px | abs_el_median_px | el_median_px |
|---|---|---|---|---|---|---|---|---|---|
| 13312 -> 16384 | 125 | 98.4 | 159.5 | 236.5 | 0.512 | 0.904 | 66.4 | 43.6 | 42.1 |
| 16384 -> 16384 | 332 | 67.5 | 138.0 | 344.0 | 0.639 | 0.84 | 55.1 | 20.5 | -6.6 |

**Click floor** (two users, same panorama, same type, within 15 degrees; no uniqueness filter): median 49 px, p75
113, p90 197 (n = 76,865 pairs); by type, CurbRamp 36, Crosswalk 66, SurfaceProblem 66, Obstacle 71, NoCurbRamp 73.

**Census** (vote-correct anchors with a partner; `le3m` = baseline <= 3 m; `clean` = neither panorama is a round-3
train/val panorama of the type):

| label_type | decided_anchors | vote_correct_anchors | long_newer_le3m | long_newer_le5m | long_newer_le3m_clean | clean_share | negative_control_pool | contradicting_on_b | control_le3m | control_le5m | unchecked_inversion_share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CurbRamp | 16489 | 14361 | 527 | 916 | 392 | 0.744 | 209 | 5 | 114 | 236 | 0.237 |
| NoCurbRamp | 11998 | 6644 | 229 | 362 | 82 | 0.358 | 30 | 50 | 71 | 114 | 0.288 |
| Obstacle | 9230 | 5893 | 68 | 130 | 26 | 0.382 | 3 | 0 | 22 | 39 | 0.441 |
| SurfaceProblem | 11457 | 8195 | 110 | 189 | 59 | 0.536 | 20 | 0 | 27 | 42 | 0.536 |
| Crosswalk | 6580 | 5499 | 213 | 312 | 185 | 0.869 | 66 | 0 | 84 | 133 | 0.221 |
| all | 55754 | 40592 | 1147 | 1909 | 744 | 0.649 | 328 | 55 | 318 | 564 | 0.285 |

Contradicting labels on the newer panorama (`long_newer`, 3 m, vote-correct anchors): 50 NoCurbRamp anchors have a
CurbRamp click near the re-projected point on B, against 5 CurbRamp anchors with a NoCurbRamp click. Ramps being
built is the direction one would expect; nobody has looked at these pairs yet (they are stratum S4 of Stage 3).

**Gate 0 values on the preview (no verdict).**

| rule | preview value | proposed threshold | pass |
|---|---|---|---|
| G0a residual long_newer pooled | n 463, median 76.5 px, 86.0% within 240 px | median <= 100 px and >= 80% within 240 px | no verdict (preview) |
| G0a residual control pooled | n 104, median 88.6 px, 83.7% within 240 px | median <= 100 px and >= 80% within 240 px | no verdict (preview) |
| G0a residual long_newer CurbRamp | n 235, median 68.2 px, 89.4% within 240 px | as pooled; a failing type is dropped | no verdict (preview) |
| G0a residual long_newer NoCurbRamp | n 77, median 91.1 px, 89.6% within 240 px | as pooled; a failing type is dropped | no verdict (preview) |
| G0a residual long_newer Crosswalk | n 97, median 79.9 px, 83.5% within 240 px | as pooled; a failing type is dropped | no verdict (preview) |
| G0b era | 1.027 | long_newer / same_month median <= 1.5 | no verdict (preview) |
| G0c yield: clean vote-correct long_newer pairs | 744 | >= 800 | no verdict (preview) |
| G0c yield: types with enough pairs | CurbRamp, Crosswalk | >= 2 types with >= 150 | no verdict (preview) |
| G0c yield: negative-control pool (presence types) | 305 | >= 200 | no verdict (preview) |

Reading, as a preview only: the median residual is about 1.5x the click floor and well under the median crop
half-width, and it does not grow from same-month to long-gap pairs, so cross-year position error does not look
like the binding problem (median 76.5 px for `long_newer`, 74.5 px for `same_month`, ratio 1.03). Yield does: at
3 m, clean vote-correct long-gap pairs over all 41 deployments come to 744, under the proposed 800; only CurbRamp
(392) and Crosswalk (185) reach 150; Obstacle and SurfaceProblem have tens; and the round-3 leakage filter removes
most NoCurbRamp and Obstacle pairs (clean share 0.36 and 0.38). Widening to 5 m raises the uncleaned long-gap count
from 1,147 to 1,909, but the 3-5 m bin of the `control` arm has only 71% within 240 px (n = 72). These numbers were seen before the thresholds were confirmed; section 11 says what
that implies.

### 6.2 Cross-check against the planner's preview script

The planner's unreviewed script (stdlib, three cities) reported a long-gap median of 56-65 px and p90 about 170 px.
Run on the same three cities with its settings, this module reproduces it to the pixel (seattle-wa `long_newer`:
n 1,730, median 65, p75 110, p90 169 in both; chicago-il n 844 against 843, median 56, p90 175 in both; the few-pair
differences come from this module's per-pair longitude scaling at the 3 m boundary and its exclusion of AI-placed
anchors). Stepping from its settings to this module's, on the same three cities:

| step | seattle-wa `long_newer` n / median / p90 | chicago-il | oradell-nj |
|---|---|---|---|
| 1 planner's settings (all labels as anchors, any B label, `-atan(h/d)` on B, no parallax QC, 15 degree truncation) | 1,730 / 65 / 169 | 844 / 56 / 175 | 67 / 48 / 181 |
| 2 + the estimator's own inverse on B (section 4) | 1,729 / 65 / 169 | 844 / 56 / 174 | 67 / 51 / 181 |
| 3 + the module's QC (parallax, `b_near_horizon`) | 1,666 / 64 / 168 | 832 / 56 / 173 | 65 / 49 / 168 |
| 4 + B clicks human and not voted incorrect | 1,667 / 64 / 168 | 827 / 56 / 173 | 65 / 49 / 168 |
| 5 + the 30 degree window (no 15 degree truncation) | 1,859 / 72 / 278 | 948 / 66 / 339 | 77 / 55 / 346 |
| 6 + vote-decided anchors only (the gate population) | 142 / 101 / 367 | 107 / 66 / 336 | 36 / 54 / 323 |

So the two implementations agree, the 15 degree truncation flattered the tail (p90 about 170 px becomes about 280-340
px), and restricting anchors to vote-decided labels, as the gate requires, cuts n by an order of magnitude and, in
Seattle, raises the median to about 100 px. Why Seattle's vote-decided anchors re-project worse is not known; they
are older on average, and Seattle is where the inversion is weakest.

## 7. Stages 1-3 (specified, not implemented)

**Stage 1, the kill stage (C1 jitter).** For test-split anchors, re-cut A with its centre displaced by a vector drawn
(seeded per `label_uid`, seed 20261112) from the Stage 0 residual distribution of that type's `long_newer` unique
pairs, measured on database positions. The flip rate A0 -> J is centring error plus model instability: a **lower
bound** on the real floor (same image, same era). It needs only A panoramas.

**Stage 2, partner crops and scoring.** B is cut with the fixed-extent rule (section 4); the depth-sized B crop is a
sensitivity arm. For packet-eligible pairs, 1024 px copies of the A and B views and a full-resolution 1024 px window
of B around the predicted point are saved and copied back the same day. Per crop, `verdict, confidence =
predict_verdicts(...)`; `served` through the type's **uncapped** round-3 map (the served maps hold Disagree under the
gate, so no gated flip could occur with them); `gated = served >= 0.92`. Pair outcomes from A0 vs B:
`consistent_correct`, `consistent_incorrect`, `changed_c2i`, `changed_i2c`, raw and gated. The number reported net of
the floor is the **model-detected change rate**, not a change rate.

Floors and controls:

- **C2 near-in-time:** `control` pairs (1-3 months), same pipeline. They share viewpoint change and occlusion with the
  long arm, not era, camera generation or season.
- **Negative control:** `long_newer` pairs whose B carries a human label of the same type, by another user, within 30
  degrees of the re-projected point: the feature demonstrably persisted, with the long arm's era and season
  confounds. B is cut at the **re-projected** point, never at B's click, and the flip rate is reported by residual bin
  so that conditioning on a near match is visible. Usable for CurbRamp, NoCurbRamp and Crosswalk, where it is **the
  floor**; C2 is reported beside it. Obstacle and SurfaceProblem have only C2 and probably too little of it; they are
  reported apart, not pooled in.
- Flip rates by (capture-year bin of A, of B) and by (`pano_width` of A, of B): round 3 trained mostly on 16,384 px
  panoramas, so an older, smaller panorama can move a verdict without any change in the world.

Statistics: cluster bootstrap (1,000 resamples, seed recorded) over connected components of the A-B panorama graph,
built once over all arms; Wilson 95% intervals for hand-check proportions; a secondary population rate over
vote-correct anchors the model calls correct on A (strata `changed_c2i` and `consistent_correct`, Jeffreys draws per
stratum); slices by type x severity and type x tag (`MIN_N = 30`) with round-3 training counts and slice AUCs as the
baseline. Per-type, per-city and per-slice tables are exploratory: no per-cell significance claims.

**Stage 3, the masked hand-check.** 200 items, one seeded shuffle:

| stratum | n | purpose |
|---|---:|---|
| S1 `changed_c2i` flips, clean, vote-correct, `long_newer` | 100 | the primary: about +-9 points pooled; per type descriptive |
| S2 `consistent_correct`, same population | 50 | miss rate (0 of 50 bounds it at 7.1%) |
| S3 decoys: 15 `control` flips + 15 negative-control pairs | 30 | rater false-positive rate on pairs that look like long-gap pairs |
| S4 pairs with a contradicting label on B | 20 | likely real changes: the sensitivity pool and the human baseline |

Each item shows A's crop and B's crop side by side, under them the 1024 px copies of both views with the crop square
drawn, and the full-resolution 1024 px window of B, so "not the same spot" can be told from "changed". Hidden:
verdicts, probabilities, votes, stratum, arm, dates, city, label id. Q1 "is this a correct <type> label on A?"; Q2 "at
the same place on B: same / changed / not the same spot / occluded / can't tell". A **second rater** rates a seeded
50-item overlap (25 from S1); kappa is on Q2 collapsed to changed vs not. Raters are named in public text only as "a
project lead". The packet is an offline `rating_page` packet; `gt_rounds.py` is not used. Views, crops and the key live
under `~/Data/sidewalk-validator-ai-temporal-2026-11/`, sha256 pinned here.

## 8. Results

| stage | result |
|---|---|
| 0 (gate run, database positions) | pending: needs the database pull (section 12) |
| 1 | pending Gate 0 |
| 2 | pending Gate 1 |
| 3 | pending Gate 2 |

## 9. The gate run (after merge, when makelab1 is cleared; Jon or the parent session)

```bash
# SCHEMAS = the public deployments' schemas: today's /v3/api/cities listing (visibility "public") mapped through
# SidewalkWebpage conf/cityparams.conf. One connection: reuse a live control master; on one timeout, stop.
D=~/Data/sidewalk-validator-ai-temporal-2026-11
ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago ...' bash -s" < analysis/pull_pano_cameras.sh > $D/db/panos.csv
ssh makelab1 "MODE=methods SCHEMAS='sidewalk_seattle sidewalk_chicago ...' bash -s" \
    < analysis/pull_pano_cameras.sh > $D/db/methods.csv
python analysis/fetch_all.py ...      # fresh public rawLabels, the same day, into $D/rawlabels
curl -s https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > $D/listing/cities-$(date +%F).json
python analysis/temporal_pairs.py positions --rawlabels $D/rawlabels --listing $D/listing/cities-<date>.json \
    --db-panos $D/db/panos.csv --cityparams ~/Git/SidewalkWebpage/conf/cityparams.conf --out $D/gate/panos.csv
python analysis/temporal_pairs.py census --rawlabels $D/rawlabels --listing $D/listing/cities-<date>.json \
    --panos $D/gate/panos.csv --round3 ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 --out $D/gate
python analysis/temporal_pairs.py residual --run $D/gate --listing $D/listing/cities-<date>.json \
    --out docs/temporal-validation-2026-11/gate
```

Expected: `gate/gate0.json` with every rule's value and pass/fail and a verdict, and in `$D/gate/panos.json` a
`compare_positions` table showing the inverted positions within centimetres of the database's on `approximation3`
panoramas. The thresholds must be confirmed (`confirmed_on` set in `gates.json`) before this runs.

## 10. If it works: a `changed` signal

A separate `temporal_status` field, never a vote: the compared panorama's id and capture date and the model-detected
change, computable only where a second capture is on hand (the service never fetches newer imagery). Conditional on
Stage 3 passing and on the flag beating both baselines of section 1.

## 11. Limitations

- **Selection.** Only places audited twice, on two captures, by people who labelled the same feature; and only
  vote-decided anchors. Neither is a random street.
- **Flat ground, ground-contact clicks.** Grades and off-ground clicks (Obstacle) are unaddressed in Stage 0; the
  Stage 2 depth check is the only independent test.
- **Season and era.** The 1-3 month control shares neither with the long arm; only the negative control does.
- **The negative control conditions on a match** and exists for three types only.
- **Two raters**, one of them a project lead.
- **Yield per type** may end the study at Gate 0 (section 6.1).
- **The preview was read before the thresholds were confirmed.** Thresholds chosen after seeing it could be fitted to
  it; the gate run uses database positions and fresh rawLabels, which limits but does not remove that risk.
- **Unverified:** that `pano_data.lat/lng` is the provider-reported camera position unchanged since the label was
  placed; that every schema is at the evolutions the pull script needs (298, 306, 319, 352); why newberg-or's labels do
  not invert; whether the 2026-10-17 cache reflects evolution 352 in every deployment.

## 12. Reproduce the preview

```bash
C=~/Data/sidewalk-validator-ai-prod-2026-10-17/rawlabels_cache
L=~/Data/sidewalk-validator-ai-temporal-2026-11/listing/cities-2026-11-12.json   # sha256 a4a2fab7...ceb5aca
R=~/Data/sidewalk-validator-ai-temporal-2026-11/preview-run
python analysis/temporal_pairs.py positions --rawlabels $C --listing $L --out $R/panos.csv
python analysis/temporal_pairs.py census --rawlabels $C --listing $L --panos $R/panos.csv \
    --round3 ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 --out $R
python analysis/temporal_pairs.py residual --run $R --listing $L --out docs/temporal-validation-2026-11/preview
```

No sampling in Stage 0 (the seed, 20261112, is recorded for the later stages). The per-label run files (`$R`) hold
label ids and stay outside the repo; only aggregates are committed. Every input's sha256, the settings and the git
commit are in [`preview/provenance.json`](temporal-validation-2026-11/preview/provenance.json).
