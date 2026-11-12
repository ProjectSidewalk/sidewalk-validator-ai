"""Tests for analysis/temporal_pairs.py (issue #17, Stage 0). CPU only, synthetic data."""
import doctest
import json
import math
import os
import random
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import temporal_pairs as tp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = tp.CAMERA_HEIGHT_M


def test_doctests():
    assert doctest.testmod(tp).failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# 1. The range estimator
# ---------------------------------------------------------------------------------------------------------------------

def test_blend_range_is_the_cotangent_above_the_blend_angle():
    dep = np.array([11.25, 15.0, 30.0, 60.0])
    assert np.allclose(tp.blend_range(dep), H / np.tan(np.radians(dep)))


def test_blend_range_is_continuous_in_value_and_slope_and_bounded():
    e = 1e-6
    lo, hi = tp.blend_range(tp.BLEND_DEG - e), tp.blend_range(tp.BLEND_DEG + e)
    assert abs(lo - hi) < 1e-4
    slope_lo = (tp.blend_range(tp.BLEND_DEG - e) - tp.blend_range(tp.BLEND_DEG - 2 * e)) / e
    slope_hi = (tp.blend_range(tp.BLEND_DEG + 2 * e) - tp.blend_range(tp.BLEND_DEG + e)) / e
    assert abs(slope_lo - slope_hi) < 1e-3
    horizon = float(tp.blend_range(0.0))
    assert 23.8 < horizon < 23.9  # the structural maximum, never the 50 m cap
    assert float(tp.blend_range(-10.0)) == horizon


def test_blend_depression_inverts_blend_range_everywhere_below_the_horizon():
    dep = np.linspace(0.5, 80, 400)
    assert np.allclose(tp.blend_depression(tp.blend_range(dep)), dep, atol=1e-9)
    assert float(tp.blend_depression(40.0)) == 0.0  # beyond the horizon value: at the horizon


def test_blend_range_matches_ps_math_when_available():
    ps_math = pytest.importorskip("sidewalk_panorama_tools.ps_math")  # panorama-tools >= v1.2.0; the repo pins v1.0.0
    dep = np.linspace(-5, 85, 181)
    assert np.allclose(tp.blend_range(dep), ps_math.predict_blend_distance(dep))
    assert (tp.CAMERA_HEIGHT_M, tp.BLEND_DEG, tp.DIST_CAP_M) == (
        ps_math.BLEND_CAMERA_HEIGHT_M, ps_math.BLEND_DEG, ps_math.DIST_CAP_M)


# ---------------------------------------------------------------------------------------------------------------------
# 2. destination / invert_camera / longitude scaling
# ---------------------------------------------------------------------------------------------------------------------

def _site_label_position(cam_lat, cam_lng, pano_x, pano_y, width, height, camera_heading):
    """Evolution 352's SQL, step by step, written independently of the module (352.sql: angles, distances,
    new_latitudes, new_positions)."""
    depression = 180.0 * pano_y / height - 90.0
    bearing = math.fmod(camera_heading - 180.0 + (pano_x / width) * 360.0, 360.0)
    a = math.radians(11.25)
    if depression >= 11.25:
        dist = H / math.tan(math.radians(depression))
    else:
        dist = min(H / math.tan(a) + H * (math.pi / 180.0) / math.sin(a) ** 2 * (11.25 - max(depression, 0.0)), 50.0)
    ad = dist / 1000.0 / 6371.0
    la = math.radians(cam_lat)
    new_lat = math.asin(math.sin(la) * math.cos(ad) + math.cos(la) * math.sin(ad) * math.cos(math.radians(bearing)))
    new_lng = math.degrees(math.radians(cam_lng) + math.atan2(
        math.sin(math.radians(bearing)) * math.sin(ad) * math.cos(la), math.cos(ad) - math.sin(la) * math.sin(new_lat)))
    return math.degrees(new_lat), new_lng


@pytest.mark.parametrize("lat", [47.6, 70.0, -33.4])
@pytest.mark.parametrize("pano_y", [4096 + 600, 4096 + 2000, 4096 + 100])  # tail, cotangent, near the horizon
def test_invert_camera_recovers_the_camera_from_the_sites_formula(lat, pano_y):
    cam = (lat, -122.3)
    lab = _site_label_position(*cam, 12000.0, pano_y, 16384, 8192, 217.3)
    got = tp.invert_camera(lab[0], lab[1], 12000.0, pano_y, 16384, 8192, 217.3)
    e, n = tp.enu_delta(cam[0], cam[1], got[0], got[1])
    assert math.hypot(e, n) < 1e-3


def test_destination_matches_the_sites_formula():
    lab = _site_label_position(47.6, -122.3, 3000.0, 5000.0, 16384, 8192, 10.0)
    lon, lat = tp.label_lonlat(3000.0, 5000.0, 16384, 8192)
    got = tp.destination(47.6, -122.3, float(tp.blend_range(-lat)), 10.0 + lon)
    assert abs(got[0] - lab[0]) < 1e-11 and abs(got[1] - lab[1]) < 1e-11


@pytest.mark.parametrize("lat", [0.0, 47.6, 60.0, 70.0])
@pytest.mark.parametrize("bearing", [0.0, 45.0, 90.0, 200.0])
def test_enu_delta_scales_longitude_by_the_pair_latitude(lat, bearing):
    """The first preview scaled longitude wrongly; at 70 N a degree of longitude is about a third of a degree of
    latitude, so a wrong cosine misplaces a 5 m baseline by metres."""
    lat_b, lng_b = tp.destination(lat, 15.0, 5.0, bearing)
    e, n = tp.enu_delta(lat, 15.0, lat_b, lng_b)
    assert abs(e - 5.0 * math.sin(math.radians(bearing))) < 1e-4
    assert abs(n - 5.0 * math.cos(math.radians(bearing))) < 1e-4


def test_enu_delta_wrong_latitude_would_be_metres_off_at_high_latitude():
    lat_b, lng_b = tp.destination(70.0, 15.0, 5.0, 90.0)
    wrong_east = (lng_b - 15.0) * tp.M_PER_DEG * math.cos(math.radians(47.6))  # a Seattle cosine at 70 N
    assert abs(wrong_east - 5.0) > 4.0
    assert abs(float(tp.enu_delta(70.0, 15.0, lat_b, lng_b)[0]) - 5.0) < 1e-4


# ---------------------------------------------------------------------------------------------------------------------
# 3. Camera positions from labels
# ---------------------------------------------------------------------------------------------------------------------

def _labels_on_pano(city, pano_id, cam, heading, clicks, month="2020-05", width=16384, first_id=1, user="u"):
    rows = []
    for i, (x, y) in enumerate(clicks):
        lat, lng = _site_label_position(cam[0], cam[1], x, y, width, width // 2, heading)
        rows.append({"label_id": first_id + i, "user_id": f"{user}{i}", "pano_id": pano_id, "pano_source": "gsv",
                     "label_type": "CurbRamp", "severity": 1, "tags": "[]", "validations": "[]",
                     "image_capture_date": month, "pano_x": x, "pano_y": y, "pano_width": width,
                     "pano_height": width // 2, "camera_heading": heading, "camera_pitch": 0.5, "camera_roll": None,
                     "latitude": lat, "longitude": lng, "city": city})
    return rows


def test_panos_from_rawlabels_agreeing_outlier_and_single():
    rows = (_labels_on_pano("c", "A" * 22, (47.6, -122.3), 30.0, [(1000, 5000), (9000, 4600), (15000, 6000)])
            + _labels_on_pano("c", "B" * 22, (47.6001, -122.3), 80.0, [(1000, 5000), (9000, 4600)], first_id=10)
            + _labels_on_pano("c", "C" * 22, (47.6002, -122.3), 80.0, [(2000, 5000)], first_id=20))
    rows[3]["latitude"] += 1e-5  # a depth-era-style label on pano B, about 1.1 m off
    labels, _ = tp.prepare_labels(pd.DataFrame(rows))
    panos, stats = tp.panos_from_rawlabels(labels)
    got = panos.set_index("pano_id")
    assert list(got.index) == ["A" * 22, "C" * 22]
    e, n = tp.enu_delta(47.6, -122.3, got.loc["A" * 22, "lat"], got.loc["A" * 22, "lng"])
    assert math.hypot(e, n) < 1e-3 and got.loc["A" * 22, "agreement_m"] < 1e-3
    assert np.isnan(got.loc["C" * 22, "agreement_m"])  # one label: unchecked, kept
    assert stats["c"]["dropped_inversion_disagrees"] == 1
    assert set(panos["position_source"]) == {"inverted"}


def test_compare_positions_finds_a_planted_offset():
    inv = pd.DataFrame({"city": ["c", "c"], "pano_id": ["p1", "p2"], "lat": [47.6, 47.7], "lng": [-122.3, -122.3],
                        "camera_heading": [10.0, 20.0]})
    lat2, lng2 = tp.destination(47.7, -122.3, 2.0, 90.0)
    db = inv.copy()
    db.loc[1, ["lat", "lng"]] = [lat2, lng2]
    out = tp.compare_positions(inv, db).iloc[0]
    assert out["n"] == 2 and out["median_m"] == pytest.approx(1.0, abs=1e-3) and out["share_over_tol"] == 0.5


def test_prepare_labels_keys_on_city_and_label_id():
    rows = (_labels_on_pano("seattle-wa", "A" * 22, (47.6, -122.3), 30.0, [(1000, 5000)], first_id=7)
            + _labels_on_pano("chicago-il", "B" * 22, (41.9, -87.6), 30.0, [(1000, 5000)], first_id=7))
    labels, _ = tp.prepare_labels(pd.DataFrame(rows))
    assert sorted(labels["label_uid"]) == ["chicago-il:7", "seattle-wa:7"]
    assert "user_id" not in labels.columns
    with pytest.raises(AssertionError):
        tp.prepare_labels(pd.DataFrame(rows[:1] * 2))


# ---------------------------------------------------------------------------------------------------------------------
# 4. Re-projection
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lat_a", [-7.0, -11.25, -20.0, -45.0])
def test_reproject_identity(lat_a):
    r = tp.reproject(33.0, lat_a, 120.0, 120.0, 0.0, 0.0)
    assert r["lon_b"] == pytest.approx(33.0, abs=1e-9) and r["lat_b"] == pytest.approx(lat_a, abs=1e-9)


def test_reproject_pure_heading_change():
    r = tp.reproject(10.0, -20.0, 100.0, 130.0, 0.0, 0.0)
    assert r["lon_b"] == pytest.approx(-20.0) and r["lat_b"] == pytest.approx(-20.0)


def test_reproject_known_triangle():
    # A looks north (heading 0, lon 0) at a point 5 m away; B stands 5 m east of A, heading north.
    dep = math.degrees(math.atan(H / 5.0))
    r = tp.reproject(0.0, -dep, 0.0, 0.0, 5.0, 0.0)
    assert r["r_a"] == pytest.approx(5.0)
    assert r["d_b"] == pytest.approx(math.hypot(5, 5))
    assert r["lon_b"] == pytest.approx(-45.0)  # north-west of B
    assert r["parallax_deg"] == pytest.approx(45.0)
    assert r["lat_b"] == pytest.approx(-math.degrees(math.atan(H / math.hypot(5, 5))))  # cotangent region


def test_reproject_round_trip_a_b_a():
    r = tp.reproject(-70.0, -18.0, 15.0, 250.0, 1.2, -2.1)
    back = tp.reproject(r["lon_b"], r["lat_b"], 250.0, 15.0, -1.2, 2.1)
    assert back["lon_b"] == pytest.approx(-70.0, abs=1e-9) and back["lat_b"] == pytest.approx(-18.0, abs=1e-9)


@pytest.mark.parametrize("pano_x", [0.0, 16384.0])
def test_reproject_at_the_seam(pano_x):
    lon, lat = tp.label_lonlat(pano_x, 6000, 16384, 8192)
    r = tp.reproject(lon, lat, 90.0, 90.0, 0.0, 0.0)
    assert abs(float(tp.wrap180(r["lon_b"] - 180.0))) < 1e-9  # both ends of the row look straight back
    assert r["x_norm_b"] % 1.0 == pytest.approx(0.0, abs=1e-12) or r["x_norm_b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------------------------------
# 5. Synthetic two-camera scene, rendered with the training crop's own renderer
# ---------------------------------------------------------------------------------------------------------------------

def _paint_ground_disc(cam_e, cam_n, heading, centre, radius, width, camera_pitch=None):
    """An equirect panorama (rig frame) of a flat grey world with a red disc on the ground, camera at height H."""
    from sidewalk_panorama_tools.geometry import tilt_matrix
    from sidewalk_panorama_tools.pose import direction_rfu
    height = width // 2
    lon = ((np.arange(width) + 0.5) / width - 0.5) * 360.0
    lat = (0.5 - (np.arange(height) + 0.5) / height) * 180.0
    d = direction_rfu(lon[None, :], lat[:, None])  # rig-frame rays, (H, W, 3)
    tilt = tilt_matrix(camera_pitch, None, beta=1.0) if camera_pitch else None
    if tilt is not None:
        d = d @ tilt  # row-vector form of tilt.T @ v: rig -> level
    lon_l = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
    down = d[..., 2] < -1e-6
    rng = np.where(down, H * np.hypot(d[..., 0], d[..., 1]) / np.where(down, -d[..., 2], 1.0), np.inf)
    brg = np.radians(heading + lon_l)
    e, n = cam_e + rng * np.sin(brg), cam_n + rng * np.cos(brg)
    hit = down & (np.hypot(e - centre[0], n - centre[1]) <= radius)
    img = np.full((height, width, 3), 128, np.uint8)
    img[hit] = (255, 0, 0)
    return img


def _click_on(cam, heading, point):
    """Level-frame (lon, lat) of a ground point from a camera."""
    ve, vn = point[0] - cam[0], point[1] - cam[1]
    lon = float(tp.wrap180(math.degrees(math.atan2(ve, vn)) - heading))
    return lon, -math.degrees(math.atan(H / math.hypot(ve, vn)))


@pytest.mark.parametrize("camera_pitch,beta,tol", [(None, 0.95, 3.0), (3.0, 1.0, 3.0), (3.0, 0.95, 6.0)])
def test_two_camera_scene_lands_the_feature_at_the_view_centre(camera_pitch, beta, tol):
    import data_generation as dg
    cam_a, heading_a = (0.0, 0.0), 30.0
    cam_b, heading_b = (2.0, 1.5), 200.0
    feature = (5.5, 4.0)  # 6.8 m from A, 4.3 m from B: both in the cotangent region, where flat ground is exact
    lon_a, lat_a = _click_on(cam_a, heading_a, feature)
    r = tp.reproject(lon_a, lat_a, heading_a, heading_b, cam_b[0] - cam_a[0], cam_b[1] - cam_a[1])
    pano_b = _paint_ground_disc(*cam_b, heading_b, feature, 0.12, 4096, camera_pitch=camera_pitch)
    view, _ = dg.render_label_view(pano_b, float(r["x_norm_b"]), float(r["y_norm_b"]), camera_pitch=camera_pitch,
                                   apply_tilt=True, beta=beta)
    ys, xs = np.nonzero((view[..., 0] > 200) & (view[..., 1] < 60))
    assert len(xs) > 50
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
    assert math.hypot(cx - dg.VIEW_SIZE / 2, cy - dg.VIEW_SIZE / 2) < tol, (cx, cy)


# ---------------------------------------------------------------------------------------------------------------------
# 6-9. QC, neighbours, partners, arms, leakage
# ---------------------------------------------------------------------------------------------------------------------

def test_geometry_qc_one_row_per_reason_in_order():
    good = {"has_camera": True, "photosphere": False, "dep_a": 20.0, "baseline_m": 2.0, "d_b": 6.0, "r_a": 6.4,
            "dep_b": 21.0, "parallax_deg": 10.0}
    assert tp.geometry_qc(good) == "ok"
    bad = {"no_camera_position": {"has_camera": False}, "photosphere": {"photosphere": True},
           "near_horizon": {"dep_a": 4.0}, "baseline": {"baseline_m": 5.5}, "b_range": {"d_b": 2.0},
           "range_ratio": {"d_b": 12.0}, "b_near_horizon": {"dep_b": 4.0}, "parallax": {"parallax_deg": 30.0}}
    assert tuple(bad) == tp.QC_ORDER
    for reason, change in bad.items():
        assert tp.geometry_qc({**good, **change}) == reason
    # first failing rule wins
    assert tp.geometry_qc({**good, "dep_a": 4.0, "parallax_deg": 40.0}) == "near_horizon"


def _random_panos(n, lat0, seed, spread_m=60.0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        lat, lng = tp.destination(lat0, 10.0, rnd.uniform(0, spread_m), rnd.uniform(0, 360))
        rows.append({"city": "c", "pano_id": f"p{i:03d}", "lat": lat, "lng": lng})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("lat0", [47.6, 70.0])
def test_neighbours_equals_brute_force(lat0):
    panos = _random_panos(150, lat0, seed=3)
    got = tp.neighbours(panos, 5.0)
    want = set()
    for a in panos.itertuples():
        for b in panos.itertuples():
            if a.pano_id != b.pano_id:
                e, n = tp.enu_delta(a.lat, a.lng, b.lat, b.lng)
                if math.hypot(e, n) <= 5.0:
                    want.add((a.pano_id, b.pano_id))
    assert set(zip(got["pano_a"], got["pano_b"])) == want and len(want) > 20


def test_neighbours_keeps_cities_apart():
    p = _random_panos(20, 47.6, seed=1, spread_m=2)  # every pair within 4 m
    both = pd.concat([p, p.assign(city="d")], ignore_index=True)
    got = tp.neighbours(both, 5.0)
    assert len(got) == 2 * 20 * 19


def test_arms():
    assert list(tp.arm_of([0, 1, 3, -1, -3, 4, 23, 24, 60, -24])) == [
        "same_month", "control", "control", "control", "control", None, None, "long_newer", "long_newer", None]


def _cands(rows):
    base = {"qc": "ok", "dt_months": 30.0, "arm": "long_newer", "baseline_m": 2.0, "label_type": "CurbRamp",
            "city": "c", "pano_id": "A" * 22}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["cand_uid"] = df["label_uid"] + "|" + df["pano_b"]
    return df


def test_choose_partners_one_per_anchor_arm_deterministic_and_per_city():
    rows = [
        {"label_uid": "seattle-wa:7", "city": "seattle-wa", "pano_b": "B1", "baseline_m": 2.0},
        {"label_uid": "seattle-wa:7", "city": "seattle-wa", "pano_b": "B2", "baseline_m": 1.0},
        {"label_uid": "seattle-wa:7", "city": "seattle-wa", "pano_b": "B3", "baseline_m": 1.0, "dt_months": 40.0},
        {"label_uid": "seattle-wa:7", "city": "seattle-wa", "pano_b": "B4", "baseline_m": 0.5, "qc": "parallax"},
        {"label_uid": "seattle-wa:7", "city": "seattle-wa", "pano_b": "B5", "baseline_m": 0.7, "arm": "control",
         "dt_months": 2.0},
        {"label_uid": "chicago-il:7", "city": "chicago-il", "pano_b": "B9", "baseline_m": 4.0},
    ]
    c = _cands(rows)
    first = tp.choose_partners(c)
    for seed in range(5):
        again = tp.choose_partners(c.sample(frac=1, random_state=seed))
        pd.testing.assert_frame_equal(first.sort_values("pair_uid").reset_index(drop=True),
                                      again.sort_values("pair_uid").reset_index(drop=True))
    got = dict(zip(first["pair_uid"], first["pano_b"]))
    assert got == {"seattle-wa:7|long_newer": "B3", "seattle-wa:7|control": "B5", "chicago-il:7|long_newer": "B9"}


def test_leakage_is_keyed_on_pano_id_across_cities(tmp_path):
    for t in tp.TYPES:
        os.makedirs(tmp_path / t)
        pd.DataFrame({"label_uid": ["hackensack-nj:1", "x:2"], "pano_id": ["SHARED" + "x" * 16, "TEST" + "y" * 18],
                      "split": ["train", "test"]}).to_csv(tmp_path / t / "split.csv", index=False)
    leak, shas = tp.leakage_panos(str(tmp_path))
    pairs = pd.DataFrame({"label_type": ["CurbRamp", "CurbRamp", "CurbRamp"],
                          "pano_id": ["SHARED" + "x" * 16, "OTHER" + "z" * 17, "OTHER" + "z" * 17],
                          "pano_b": ["Q" * 22, "SHARED" + "x" * 16, "TEST" + "y" * 18],
                          "city": ["teaneck-nj"] * 3})  # a different deployment from the training row
    assert tp.mark_clean(pairs, leak).tolist() == [False, False, True]
    assert set(shas) == set(tp.TYPES)


# ---------------------------------------------------------------------------------------------------------------------
# 10-11. Residual, click floor, contradicting labels
# ---------------------------------------------------------------------------------------------------------------------

def _blabel(uid, pano, lon, lat, user="u1", t="CurbRamp", vote="", author="human", city="c"):
    return {"label_uid": uid, "city": city, "pano_id": pano, "label_type": t, "lon": lon, "lat": lat,
            "user_key": user, "vote_class": vote, "author": author}


def test_residual_planted_offset_and_flags():
    c = _cands([{"label_uid": "c:1", "pano_b": "B1", "lon_b": 10.0, "lat_b": -20.0, "user_key": "ua"},
                {"label_uid": "c:2", "pano_b": "B2", "lon_b": 10.0, "lat_b": -20.0, "user_key": "ua"},
                {"label_uid": "c:3", "pano_b": "B3", "lon_b": 10.0, "lat_b": -20.0, "user_key": "ua"}])
    labels = pd.DataFrame([
        _blabel("c:11", "B1", 10.0, -18.0, user="ub"),                    # 2 deg above: unique
        _blabel("c:12", "B2", 13.0, -20.0, user="ua"), _blabel("c:13", "B2", 30.0, -20.0, user="ub"),  # two within 30
        _blabel("c:14", "B3", 50.0, -20.0),                               # none within 30: unmatched
        _blabel("c:15", "B1", 11.0, -20.0, vote="incorrect"),            # voted incorrect: not a measurement
        _blabel("c:16", "B1", 10.5, -20.0, author="ai"),                 # AI-placed: not a measurement
    ])
    res = tp.residuals(c, labels).set_index("cand_uid")
    r1 = res.loc["c:1|B1"]
    assert r1["unique"] and r1["matched"] and r1["n_b_labels"] == 1
    assert r1["res_px"] == pytest.approx(1024 * math.tan(math.radians(2.0)), rel=1e-6)
    assert r1["el_px"] == pytest.approx(1024 * math.tan(math.radians(2.0)), rel=1e-6) and abs(r1["az_px"]) < 1e-6
    assert not r1["same_user"] and r1["n_within_other_user"] == 1
    r2 = res.loc["c:2|B2"]
    assert r2["matched"] and not r2["unique"] and r2["n_within"] == 2 and r2["same_user"]
    expected_az = 1024 * math.tan(math.radians(3.0 * math.cos(math.radians(20.0))))
    assert r2["az_px"] == pytest.approx(expected_az, rel=1e-6)
    assert not res.loc["c:3|B3", "matched"]


def test_click_floor_ignores_same_user_pairs():
    labels = pd.DataFrame([_blabel("c:1", "P", 0.0, -20.0, user="a"), _blabel("c:2", "P", 1.0, -20.0, user="a"),
                           _blabel("c:3", "P", 0.0, -21.0, user="b"), _blabel("c:4", "P", 40.0, -20.0, user="c")])
    fl = tp.click_floor(labels)
    # a-b pairs: (c:1, c:3) 1 deg, (c:2, c:3) ~1.4 deg; c:4 is beyond 15 deg of everything; a-a is ignored
    assert len(fl) == 2 and fl["res_px"].min() == pytest.approx(1024 * math.tan(math.radians(1.0)), rel=1e-6)


def test_contradicting_on_b_only_curb_ramp_vs_no_curb_ramp():
    c = _cands([{"label_uid": "c:1", "pano_b": "B1", "lon_b": 0.0, "lat_b": -20.0, "user_key": "u"},
                {"label_uid": "c:2", "pano_b": "B2", "lon_b": 0.0, "lat_b": -20.0, "user_key": "u",
                 "label_type": "NoCurbRamp"},
                {"label_uid": "c:3", "pano_b": "B3", "lon_b": 0.0, "lat_b": -20.0, "user_key": "u"}])
    labels = pd.DataFrame([_blabel("c:9", "B1", 2.0, -20.0, t="NoCurbRamp"),
                           _blabel("c:8", "B2", 2.0, -20.0, t="CurbRamp"),
                           _blabel("c:7", "B3", 2.0, -20.0, t="Obstacle")])
    got = tp.contradicting_on_b(c, labels)
    assert got.to_dict() == {"c:1|B1": True, "c:2|B2": True, "c:3|B3": False}


# ---------------------------------------------------------------------------------------------------------------------
# 12. Gate 0
# ---------------------------------------------------------------------------------------------------------------------

def _gate_inputs(ln_median=60.0, ctl_median=70.0, sm_median=55.0, within=0.9, clean=300, neg=100):
    row = lambda arm, m, n=500: {"arm": arm, "n": n, "median_px": m, "within_240": within}  # noqa: E731
    tables = {"by_arm": pd.DataFrame([row("same_month", sm_median), row("control", ctl_median),
                                      row("long_newer", ln_median)]),
              "by_arm_type": pd.DataFrame([{**row("long_newer", ln_median, 200), "label_type": t} for t in tp.TYPES]),
              "by_arm_baseline": pd.DataFrame([{**row(a, 60.0), "baseline_bin": "3-5"} for a in
                                               ("long_newer", "control")])}
    census = pd.DataFrame([{"label_type": t, "long_newer_le3m_clean": clean, "negative_control_pool": neg}
                           for t in tp.TYPES])
    return tables, census


def test_gate0_passes_and_each_rule_fails():
    gates = tp.load_gates()
    tables, census = _gate_inputs()
    ok = tp.gate0(tables, census, gates, {"db"})
    assert ok["verdict"] == "pass" and ok["baseline_cap_m"] == 5.0 and ok["types_dropped"] == []
    for kw, rule in [({"ln_median": 120.0}, "G0a residual long_newer pooled"),
                     ({"ctl_median": 120.0}, "G0a residual control pooled"),
                     ({"within": 0.7}, "G0a residual long_newer pooled"),
                     ({"ln_median": 90.0, "sm_median": 50.0}, "G0b era"),
                     ({"clean": 100}, "G0c yield: clean vote-correct long_newer pairs"),
                     ({"neg": 10}, "G0c yield: negative-control pool (presence types)")]:
        out = tp.gate0(*_gate_inputs(**kw), gates, {"db"})
        assert out["verdict"] == "fail", kw
        assert [r for r in out["rules"] if r["rule"] == rule][0]["pass"] is False, kw


def test_gate0_drops_a_failing_type():
    gates = tp.load_gates()
    tables, census = _gate_inputs()
    tables["by_arm_type"].loc[tables["by_arm_type"]["label_type"] == "Obstacle", "median_px"] = 150.0
    out = tp.gate0(tables, census, gates, {"db"})
    assert out["types_dropped"] == ["Obstacle"] and "Obstacle" not in out["types_kept"]


@pytest.mark.parametrize("sources", [{"inverted"}, {"db", "inverted"}])
def test_gate0_refuses_a_verdict_on_inverted_positions(sources):
    tables, census = _gate_inputs()
    out = tp.gate0(tables, census, tp.load_gates(), sources)
    assert out["verdict"] is None and "refused" in out and out["baseline_cap_m"] is None
    assert all(r["pass"] is None for r in out["rules"])


def test_gates_file_is_dated_and_marked_proposed():
    g = tp.load_gates()
    assert g["proposed_on"] == "2026-11-12" and g["status"].startswith("proposed")
    assert {"gate0", "gate1", "gate2", "stage3"} <= set(g)


# ---------------------------------------------------------------------------------------------------------------------
# End to end on a tiny synthetic city (positions -> census -> residual), then the shell script
# ---------------------------------------------------------------------------------------------------------------------

def test_end_to_end_preview_refuses_a_verdict(tmp_path):
    cache, run, out, r3 = tmp_path / "cache", tmp_path / "run", tmp_path / "out", tmp_path / "r3"
    os.makedirs(cache)
    votes = json.dumps([{"user_id": f"v{i}", "validation": "Agree", "validator_type": "Human"} for i in range(4)])
    rows = []
    cam_a, cam_b = (47.6, -122.3), tp.destination(47.6, -122.3, 2.0, 90.0)
    rows += _labels_on_pano("tiny-city", "A" * 22, cam_a, 0.0, [(8192, 5500), (2000, 5000)], month="2015-06")
    rows += _labels_on_pano("tiny-city", "B" * 22, cam_b, 0.0, [(7500, 5600), (3000, 5000)], month="2019-06",
                            first_id=10, user="w")
    for r in rows:
        r["validations"] = votes
    pd.DataFrame(rows).to_csv(cache / "tiny-city.csv", index=False)
    listing = tmp_path / "cities.json"
    listing.write_text(json.dumps({"cities": [{"city_id": "tiny-city", "visibility": "public",
                                               "url": "https://sidewalk-tiny.example.org"}]}))
    for t in tp.TYPES:
        os.makedirs(r3 / t)
        pd.DataFrame({"label_uid": ["z:1"], "pano_id": ["Z" * 22], "split": ["train"]}).to_csv(
            r3 / t / "split.csv", index=False)
    tp.main(["positions", "--rawlabels", str(cache), "--listing", str(listing), "--out", str(run / "panos.csv")])
    tp.main(["census", "--rawlabels", str(cache), "--listing", str(listing), "--panos", str(run / "panos.csv"),
             "--round3", str(r3), "--out", str(run)])
    tp.main(["residual", "--run", str(run), "--listing", str(listing), "--out", str(out)])
    g0 = json.loads((out / "gate0.json").read_text())
    assert g0["verdict"] is None and g0["banner"].startswith("PREVIEW")
    assert (out / "residual.md").read_text().startswith("**PREVIEW")
    by_arm = pd.read_csv(out / "by_arm.csv", comment="#")
    assert by_arm.loc[0, "arm"] == "long_newer" and by_arm.loc[0, "n"] >= 1
    committed = "".join((out / f).read_text() for f in os.listdir(out))
    assert "tiny-city:" not in committed  # no label keys in the aggregates


def test_end_to_end_refuses_a_private_deployment(tmp_path):
    run = tmp_path / "run"
    os.makedirs(run)
    (run / "selection.json").write_text("{}")
    pd.DataFrame({"city": ["secret-city"], "pano_id": ["A" * 22], "pano_b": ["B" * 22]}).to_csv(
        run / "candidates.csv.gz", index=False)
    pd.DataFrame({"pano_id": ["A" * 22], "vote_class": [""]}).to_csv(run / "labels.csv.gz", index=False)
    listing = tmp_path / "cities.json"
    listing.write_text(json.dumps({"cities": [{"city_id": "secret-city", "visibility": "private"}]}))
    with pytest.raises(SystemExit, match="not public"):
        tp.main(["residual", "--run", str(run), "--listing", str(listing), "--out", str(tmp_path / "out")])


SCRIPT = os.path.join(REPO, "analysis", "pull_pano_cameras.sh")


def test_pull_script_parses():
    subprocess.run(["bash", "-n", SCRIPT], check=True)


@pytest.mark.parametrize("env,code", [({"SCHEMAS": "sidewalk_seattle; DROP"}, 2), ({"SCHEMAS": "public"}, 2),
                                      ({"SCHEMAS": ""}, 2), ({"SCHEMAS": "sidewalk_x", "MODE": "users"}, 2)])
def test_pull_script_refuses_before_any_query(tmp_path, env, code):
    marker = tmp_path / "psql_called"
    stub = tmp_path / "psql"
    stub.write_text(f"#!/bin/sh\ntouch {marker}\n")
    stub.chmod(0o755)
    full = {"PATH": f"{tmp_path}:{os.environ['PATH']}", **env}
    p = subprocess.run(["bash", SCRIPT], env=full, capture_output=True, text=True)
    assert p.returncode == code and not marker.exists()


def test_pull_script_queries_each_schema_read_only(tmp_path):
    log, sql = tmp_path / "calls", tmp_path / "sql"
    stub = tmp_path / "psql"
    stub.write_text(f'#!/bin/sh\nprintf "%s|%s\\n" "$PGOPTIONS" "$1 $2 $3 $4 $5 $6 $7" >> {log}\n'
                    f'printf "%s\\n" "$*" >> {sql}\n')
    stub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "SCHEMAS": "sidewalk_a sidewalk_b"}
    p = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("schema,pano_id,lat,lng,camera_heading")
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 2 and all("default_transaction_read_only=on" in c and "-p 5434" in c for c in calls)
    assert "sidewalk_a.pano_data" in sql.read_text() and "user_id" not in sql.read_text()
    p = subprocess.run(["bash", SCRIPT], env={**env, "MODE": "methods"}, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("schema,label_id,computation_method")
