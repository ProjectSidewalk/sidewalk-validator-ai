# Decisions

Choices about the training data that are deliberate, not oversights, with the evidence behind each one and
what would change it. One section per decision, newest first.

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

## Labels on non-GSV imagery stay out of training (October 2026)

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
