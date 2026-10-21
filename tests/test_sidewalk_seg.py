"""CPU tests for the pure parts of analysis/sidewalk_seg.py (issue #44): mask features on synthetic masks, the
pixel-to-metre conversion, the horizon, the production-verdict strata and the public-city filter."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis import sidewalk_seg as ss

SW, ROAD, CURB, OTHER = 1, 2, 3, 0


def feats(mask, crop_half_px=256, depth_m=10.0, pitch_deg=-20.0, **kw):
    return ss.mask_features(mask, [SW], [ROAD], [CURB], crop_half_px, depth_m, pitch_deg, **kw)


def blank(m=128):
    return np.full((m, m), OTHER, np.uint8)


def test_no_sidewalk():
    f = feats(blank())
    assert not f["sidewalk_any"] and not f["point_on_sidewalk"] and not f["point_on_sidewalk_disc"]
    assert math.isnan(f["dist_px_to_sidewalk"]) and math.isnan(f["dist_m_to_sidewalk"])
    assert f["near_sidewalk_1m"] is False and f["connected_to_bottom"] is False
    assert f["sidewalk_frac_view"] == 0 and f["sidewalk_frac_crop"] == 0
    assert math.isnan(f["sidewalk_above_horizon_frac"])


def test_point_on_sidewalk_band_reaching_bottom():
    m = blank()
    m[60:, :] = SW  # covers the centre (64) and the bottom rows
    f = feats(m)
    assert f["point_on_sidewalk"] and f["point_on_sidewalk_disc"]
    assert f["dist_px_to_sidewalk"] == 0 and f["dist_m_to_sidewalk"] == 0 and f["near_sidewalk_1m"]
    assert f["connected_to_bottom"]
    assert f["centre_class"] == SW
    assert f["sidewalk_above_horizon_frac"] == 0  # horizon at row 651/2048 -> 40.7 of 128, band starts at 60


def test_distance_in_view_pixels_and_metres():
    m = blank(256)
    m[:, 128 + 10:] = SW  # 10 mask px right of the centre column = 80 view px at 256/2048
    # Ground range 12.8 m: 2.5 / sin(-pitch) = 12.8 -> pitch = -asin(2.5 / 12.8)
    pitch = -math.degrees(math.asin(2.5 / 12.8))
    f = feats(m, depth_m=30.0, pitch_deg=pitch)
    assert f["dist_px_to_sidewalk"] == pytest.approx(80.0)
    assert f["ground_range_m"] == pytest.approx(12.8)
    assert f["dist_m_to_sidewalk"] == pytest.approx(80.0 * 12.8 / 1024)  # 1.0 m, from the ground range
    assert f["dist_m_depth_to_sidewalk"] == pytest.approx(80.0 * 30.0 / 1024)  # the depth-based column
    assert f["near_sidewalk_1m"]  # thresholded on the ground range, not on depth_m
    f = feats(m, depth_m=1.0, pitch_deg=pitch + 0.5)  # farther ground point
    assert not f["near_sidewalk_1m"]


def test_ground_range():
    assert ss.ground_range_m(-90.0) == pytest.approx(2.5)
    assert ss.ground_range_m(-30.0) == pytest.approx(5.0)
    assert ss.ground_range_m(-1.0) == ss.ground_range_m(10.0) == pytest.approx(2.5 / math.sin(math.radians(2)))


def test_connectedness_uses_component_of_nearest_pixel():
    m = blank()
    m[50:56, 60:70] = SW     # small patch just above the centre, not reaching the bottom
    m[120:, :20] = SW        # a separate patch at the bottom-left, farther away
    f = feats(m)
    assert not f["point_on_sidewalk"]
    assert not f["connected_to_bottom"]  # the nearest component is the floating patch
    m[56:, 60:62] = SW       # now join the patch to the bottom with a thin strip
    assert feats(m)["connected_to_bottom"]


def test_disc_majority_differs_from_centre_pixel():
    m = blank(2048)
    m[1024, 1024] = SW       # a single sidewalk pixel exactly at the centre
    f = feats(m)
    assert f["point_on_sidewalk"] and not f["point_on_sidewalk_disc"]
    assert 0 < f["sidewalk_frac_disc"] < 0.01


def test_crop_and_view_fractions_and_road():
    m = blank(256)
    m[:, :128] = ROAD
    m[:, 128:] = SW
    m[:, 127] = CURB
    f = feats(m, crop_half_px=512)  # crop square = 64 mask px half-width
    assert f["sidewalk_frac_view"] == pytest.approx(0.5)
    assert f["sidewalk_frac_crop"] == pytest.approx(0.5)
    assert f["curb_frac_crop"] == pytest.approx(1 / 128)
    assert f["road_frac_crop"] == pytest.approx(63 / 128)
    assert f["point_on_sidewalk"] and not f["on_road"]


def test_crop_square_clipped_to_view():
    m = blank(64)
    m[:, :] = SW
    f = feats(m, crop_half_px=5000)
    assert f["sidewalk_frac_crop"] == 1.0


def test_sidewalk_above_horizon():
    m = blank(2048)
    m[:200, :] = SW          # sidewalk only near the top: all of it above a horizon at row ~651
    m[1500:, :] = SW
    f = feats(m, pitch_deg=-20.0)
    assert f["sidewalk_above_horizon_frac"] == pytest.approx(200 / (200 + 548))


def test_mask_must_be_square():
    with pytest.raises(ValueError):
        feats(np.zeros((10, 20), np.uint8))


def test_geometry_helpers():
    assert ss.px_to_m(102.4, 10.0) == pytest.approx(1.0)
    assert ss.horizon_row(0.0) == 1024
    assert ss.horizon_row(-45.0) == pytest.approx(0.0)
    assert ss.view_pitch_deg(0.5) == 0
    assert ss.view_pitch_deg(1.0) == -90
    assert ss.store_tail("chicago-il", "CAoSFENJ") == "chicago-il/CA/CAoSFENJ.jpg"


def test_class_ids():
    id2 = {"13": "Road", "15": "Sidewalk", "9": "Curb Cut", "11": "Pedestrian Area", "2": "Curb"}
    assert ss.class_ids(id2, ss.SIDEWALK_CLASSES) == [9, 11, 15]
    with pytest.raises(ValueError):
        ss.class_ids(id2, ("Pothole",))


def test_production_pool():
    assert ss.production_pool("Disagree", 3) == "ai_disagree"
    assert ss.production_pool("Agree", 1) == "contested_agree"
    assert ss.production_pool("Agree", 0) == "uncontested_agree"
    assert ss.production_pool("Agree", float("nan")) == "other"
    assert ss.production_pool(float("nan"), 0) == "other"
    assert ss.production_pool("Unsure", 0) == "other"


def test_schema_label_uid_and_public_filter():
    cities = {"seattle-wa": {"visibility": "public", "country_id": "usa"},
              "hidden-tw": {"visibility": "private", "country_id": "taiwan"},
              "quiet-sc": {"visibility": "private", "country_id": "usa"}}
    assert ss.schema_label_uid("sidewalk_seattle", 7, cities) == "seattle-wa:7"
    assert ss.schema_label_uid("sidewalk_hidden", 7, cities) == "hidden-tw:7"
    assert ss.schema_label_uid("sidewalk_unknown", 7, cities) is None
    shown = ss.display_cities(pd.Series(["seattle-wa", "hidden-tw", "quiet-sc", "hidden-tw"]), cities)
    assert shown.tolist()[0] == "seattle-wa"
    assert set(shown.tolist()[1:]) == {"other (2 private deployments)"}
    assert not any("hidden" in s or "quiet" in s for s in shown)


def test_feature_matrix_imputes_missing_distance_as_far():
    df = pd.DataFrame({c: [np.nan, 0.0] for c, _ in ss.FEATURES})
    df["point_on_sidewalk"] = [False, True]
    X = ss.feature_matrix(df)
    j = [c for c, _ in ss.FEATURES].index("dist_px_to_sidewalk")
    assert X[0, j] == ss.NO_SIDEWALK_DIST["dist_px_to_sidewalk"] and X[1, j] == 0
    assert X[:, 0].tolist() == [0.0, 1.0]


def test_select_labels_filters_types_and_checks_uid():
    s = pd.DataFrame({c: [0, 0, 0] for c in ss.LABEL_COLUMNS})
    s["label_uid"] = ["a-wa:1", "a-wa:2", "a-wa:3"]
    s["city"] = "a-wa"
    s["label_id"] = [1, 2, 3]
    s["pano_id"] = ["xyz1", "xyz2", "xyz3"]
    s["label_type"] = ["Obstacle", "CurbRamp", "SurfaceProblem"]
    out = ss.select_labels(s)
    assert out.label_uid.tolist() == ["a-wa:1", "a-wa:3"]
    assert out.store_tail.tolist() == ["a-wa/xy/xyz1.jpg", "a-wa/xy/xyz3.jpg"]
    s.loc[0, "label_uid"] = "b-wa:1"
    with pytest.raises(ValueError):
        ss.select_labels(s)


def test_auc_helpers():
    assert ss.oriented_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == (0.0, "-")
    auc, d = ss.oriented_auc([0, 0, 0], [1, 2, 3])
    assert math.isnan(auc) and d == ""
    lo, hi = ss.bootstrap_auc([0, 1] * 20, [0.2, 0.8] * 20, n_boot=50, other=[0.5, 0.5] * 20)
    assert lo == hi == pytest.approx(0.5)
    lo, hi = ss.wilson(0, 10)
    assert lo == pytest.approx(0, abs=1e-12) and 0.2 < hi < 0.35


def test_within_city_auc_ignores_between_city_differences():
    # Within each city the score is useless (constant), but city b has more incorrect labels and higher scores.
    y = [0, 1] * 10 + [0, 1, 1, 1, 1] * 4
    city = ["a"] * 20 + ["b"] * 20
    score = [0.1] * 20 + [0.9] * 20
    pooled, _ = ss.oriented_auc(y, score)
    within, n = ss.within_city_auc(y, score, city)
    assert pooled > 0.6 and within == pytest.approx(0.5) and n == 2
    assert ss.within_city_auc([0, 1], [0.1, 0.9], ["a", "a"])[1] == 0  # below min_n


def test_city_bootstrap_auc_is_wider_than_row_bootstrap_with_few_cities():
    rng = np.random.default_rng(0)
    city = np.repeat(list("abc"), 60)
    y = rng.integers(0, 2, 180)
    # A score that helps a lot in city a, a little in b, not at all in c.
    score = y * np.repeat([2.0, 0.5, 0.0], 60) + rng.normal(size=180)
    rlo, rhi = ss.bootstrap_auc(y, score, n_boot=300)
    clo, chi = ss.city_bootstrap_auc(y, score, city, n_boot=300)
    assert chi - clo > rhi - rlo


def test_grouped_folds_keep_panoramas_disjoint():
    y = np.tile([0, 1], 50)
    groups = np.repeat(np.arange(25), 4)
    folds = ss.grouped_folds(y, groups, n_splits=5)
    assert len(folds) == 5
    seen = np.zeros(len(y), int)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_lift_and_transfer_predictions_smoke():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({c: rng.random(n) for c, _ in ss.FEATURES})
    df["label_class"] = np.where(np.arange(n) % 2, "incorrect", "correct")
    df["pano_id"] = [f"p{i // 2}" for i in range(n)]
    df["p_incorrect"] = np.clip(0.5 + 0.3 * (np.arange(n) % 2) - 0.15 + rng.normal(0, 0.1, n), 0.01, 0.99)
    df["split"] = np.where(np.arange(n) < 80, "test", "test_heldout")
    cls = rng.normal(size=(n, 16))
    oof = ss.lift_predictions(df, cls, pca=4)
    assert list(oof.columns) == list(ss.MODEL_NAMES) and not oof.isna().any().any()
    assert ((oof >= 0) & (oof <= 1)).all().all()
    tp = ss.transfer_predictions(df, cls, pca=4)
    assert len(tp) == 40 and set(tp.index) == set(df.index[df.split == "test_heldout"])
