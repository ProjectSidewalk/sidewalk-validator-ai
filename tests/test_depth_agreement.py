"""Tests for analysis/depth_agreement.py (issue #21). CPU only, synthetic fixtures, no network.

The geometry tests need ``sidewalk_panorama_tools`` (installed in CI at the pinned v1.0.0) and skip without it.
"""
import doctest
import json
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import depth_agreement as da

LABEL_TYPES = da.LABEL_TYPES
W, H = 16384, 8192


def test_doctests():
    pytest.importorskip("sidewalk_panorama_tools")
    res = doctest.testmod(da)
    assert res.failed == 0 and res.attempted > 0


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic round 3
# ---------------------------------------------------------------------------------------------------------------------

def y_for_depression(dep):
    return (90.0 + dep) / 180.0 * H


def make_round3(root, cities=("pub-a", "pub-b", "secret-town"), n_per=30, seed=0, ratio=2.5, h=2.5,
                extra_rows=None):
    """``<root>/<Type>/split.csv`` and ``<root>/rawlabels/<city>.csv``: n_per labels per (type, city), every label
    ViT-B = ``ratio`` x the flat-ground distance at camera height ``h`` (times a little noise), two labels per
    panorama, label_id starting at 1 in every city (so ids collide across cities, as in production)."""
    rng = np.random.default_rng(seed)
    raw = {c: [] for c in cities}
    for t in LABEL_TYPES:
        rows = []
        for city in cities:
            for i in range(n_per):
                lid = str(LABEL_TYPES.index(t) * 1000 + i + 1)
                pano = f"P{city[:3]}{t[:3]}{i // 2:04d}x"
                dep = float(rng.uniform(5, 40))
                trig = h / math.sin(math.radians(dep))
                depth = round(ratio * trig * float(np.exp(rng.normal(0, 0.05))), 4)
                rows.append({
                    "label_uid": f"{city}:{lid}", "city": city, "label_id": lid, "pano_id": pano, "label_type": t,
                    "label_class": "correct" if i % 2 else "incorrect", "author": "human",
                    "severity": [1.0, 2.0, 3.0, None][i % 4], "tags": ['["narrow"]', "[]", '["a","b"]'][i % 3],
                    "image_capture_date": ["2012-05", "2019-07", "2023-01"][i % 3], "pano_source": "gsv",
                    "pano_path": f"/gscratch/x/pano-stage/{t}/{city}/{pano[:2]}/{pano}.jpg",
                    "camera_pitch": 1.0, "camera_roll": None, "tilt_applied": 1, "tilt_beta": 0.95,
                    "depth_m": depth, "crop_half_px": int(6100 / depth), "clipped": int(int(6100 / depth) > 1024),
                    "status": "ok", "split": ["train", "train", "val", "test"][i % 4]})
                raw[city].append({"label_id": lid, "pano_id": pano, "pano_x": float(rng.uniform(0, W)),
                                  "pano_y": y_for_depression(dep), "pano_width": W, "pano_height": H,
                                  "user_id": "not-read"})
        df = pd.DataFrame(rows)
        if extra_rows is not None:
            df = extra_rows(t, df)
        os.makedirs(os.path.join(root, t), exist_ok=True)
        df.to_csv(os.path.join(root, t, "split.csv"), index=False)
    os.makedirs(os.path.join(root, "rawlabels"), exist_ok=True)
    for c, r in raw.items():
        pd.DataFrame(r).to_csv(os.path.join(root, "rawlabels", f"{c}.csv"), index=False)
    return root


@pytest.fixture
def r3(tmp_path):
    return make_round3(str(tmp_path / "round3"))


PUBLIC = frozenset({"pub-a", "pub-b"})


# ---------------------------------------------------------------------------------------------------------------------
# T1-T6: trig
# ---------------------------------------------------------------------------------------------------------------------

def test_t1_depression_and_trig():
    assert da.depression_deg(np.array([H / 2, H]), H).tolist() == [0.0, 90.0]
    assert da.trig_m(np.array([30.0]), 2.5)[0] == pytest.approx(5.0)
    assert np.isnan(da.trig_m(np.array([2.0, 1.0, -5.0]), 2.5)).all()
    assert np.isfinite(da.trig_m(np.array([2.01]), 2.5)).all()


def test_t2_ratio_scales_with_height_and_slopes_do_not(r3):
    f = da.load_frame(r3, os.path.join(r3, "rawlabels"))
    a, b = da.trig_frame(f, 2.5), da.trig_frame(f, 2.9)
    np.testing.assert_allclose(b.ratio, a.ratio * 2.5 / 2.9)
    sa = da.slope_bracket(np.log(a.depth_m), np.log(a.trig_m))
    sb = da.slope_bracket(np.log(b.depth_m), np.log(b.trig_m))
    for k in ("beta_yx", "beta_xy", "rma", "r"):
        assert sa[k] == pytest.approx(sb[k], abs=1e-12)


def test_t3_ai_and_near_horizon_excluded_and_counted(tmp_path):
    def extra(t, df):
        if t == "CurbRamp":
            df.loc[0, "author"] = "ai"
        return df
    root = make_round3(str(tmp_path / "r"), extra_rows=extra)
    raw = pd.read_csv(os.path.join(root, "rawlabels", "pub-b.csv"))
    raw.loc[raw.label_id == 1001, "pano_y"] = y_for_depression(1.5)       # NoCurbRamp, pub-b, near the horizon
    raw.to_csv(os.path.join(root, "rawlabels", "pub-b.csv"), index=False)
    tf = da.trig_frame(da.load_frame(root, os.path.join(root, "rawlabels")))
    assert tf.set_index("label_uid").loc["pub-a:1", "why_excluded"] == "ai_authored"
    assert tf.set_index("label_uid").loc["pub-b:1001", "why_excluded"] == "near_horizon"
    assert tf.usable.sum() == len(tf) - 2
    out = da.trig_tables(tf, PUBLIC, n_boot=0)
    rows = {(r["label_type"], r["why_excluded"]): r["n"] for r in _t(out, "rows")["rows"]}
    assert rows[("CurbRamp", "ai_authored")] == 1 and rows[("NoCurbRamp", "near_horizon")] == 1
    assert _t(out, "ai_rows")["rows"][0]["n"] == 1
    head = {r["label_type"]: r["n"] for r in _t(out, "by_type")["rows"]}
    assert head["CurbRamp"] == 89 and head["all types"] == len(tf) - 2


def _t(out, name):
    return next(t for t in out["tables"] if t["name"] == name)


def test_t4_private_city_never_named(r3):
    tf = da.trig_frame(da.load_frame(r3, os.path.join(r3, "rawlabels")))
    out = da.trig_tables(tf, PUBLIC, n_boot=0)
    text = json.dumps(out)
    assert "secret-town" not in text and "secret" not in text
    cities = [r["city"] for r in _t(out, "city")["rows"]]
    assert cities == ["pub-a", "pub-b", "non-public deployments (1)"]
    pooled = _t(out, "city")["rows"][-1]
    assert pooled["n"] == 150 and pooled["train_n"] == sum(1 for i in range(30) if i % 4 < 2) * 5
    assert "secret" not in da.report_markdown(out)


def test_t5_small_cells_marked_and_train_n_everywhere(tmp_path):
    r3 = make_round3(str(tmp_path / "small"), n_per=10)       # 30 per type: every severity cell is under 20
    tf = da.trig_frame(da.load_frame(r3, os.path.join(r3, "rawlabels")))
    out = da.trig_tables(tf, PUBLIC, n_boot=0)
    sev = _t(out, "type_severity")
    assert all("train_n" in r for r in sev["rows"])
    small = [r for r in sev["rows"] if r["small"]]
    assert small and all(r["ratio_med"] is None and r["n"] < da.MIN_CELL_N for r in small)
    big = [r for r in _t(out, "by_type")["rows"] if not r["small"]]
    assert big and all(r["ratio_med"] is not None for r in big)
    tag = _t(out, "type_tag")
    assert {r["tag"] for r in tag["rows"]} == {"narrow", "a", "b", da.NONE}
    md = da.table_markdown(sev)
    assert "n too small" in md
    for name in ("type_severity", "type_tag", "type_class", "type_year", "type_split", "city", "by_type"):
        assert all(isinstance(r["train_n"], int) for r in _t(out, name)["rows"])


def test_t6_trig_json_records_input_hashes(r3, tmp_path):
    pub = tmp_path / "public.json"
    pub.write_text(json.dumps(sorted(PUBLIC)))
    out = tmp_path / "trig"
    argv = ["trig", "--root", r3, "--rawlabels-dir", os.path.join(r3, "rawlabels"), "--out-dir", str(out),
            "--public-cities", str(pub), "--n-boot", "0"]
    da.main(argv)
    rec = json.loads((out / "trig.json").read_text())
    assert set(rec["inputs"]) == set(da.ROUND3_PINS)
    assert rec["inputs"]["CurbRamp/split.csv"] == da.sha256_file(os.path.join(r3, "CurbRamp", "split.csv"))
    assert rec["inputs_match_round3_pins"] is False
    assert rec["rows_in"] == 450 and rec["rows_used"] == 450
    first = rec["inputs"]["rawlabels/*.csv"]
    with open(os.path.join(r3, "rawlabels", "pub-a.csv"), "a") as f:
        f.write("\n")
    da.main(argv)
    assert json.loads((out / "trig.json").read_text())["inputs"]["rawlabels/*.csv"] != first
    with pytest.raises(SystemExit, match="differ from ROUND3_PINS"):
        da.main(argv + ["--check-pins"])
    tables = (out / "trig_tables.json").read_text()
    assert "secret-town" not in tables and "pub-a:" not in tables   # no per-label row in the aggregates


def test_combined_sha256_matches_shasum_pipeline(tmp_path):
    if shutil.which("shasum") is None:
        pytest.skip("no shasum")
    for name, text in (("b-x.csv", "1\n"), ("a.csv", "2\n"), ("bx.csv", "3\n")):
        (tmp_path / name).write_text(text)
    want = subprocess.run("shasum -a 256 *.csv | shasum -a 256", shell=True, cwd=tmp_path, capture_output=True,
                          text=True, env={**os.environ, "LC_ALL": "C"}).stdout.split()[0]
    assert da.combined_sha256(str(tmp_path))[0] == want


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic depth artifacts
# ---------------------------------------------------------------------------------------------------------------------

def make_artifact(planes, h=64, w=128, pitch=0.0, roll=0.0):
    """A DepthGrid from planes ``[(n, d), ...]`` ({p: p . n = d}, artifact frame: x left, y back, z down): each
    cell takes the nearest plane its ray meets in front of the camera; none: depth -1, index 0. Plane i of the
    list is index i + 1 (index 0 is the no-plane sentinel)."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    v = da.cell_ray(rr, cc, h, w)
    best = np.full((h, w), np.inf)
    idx = np.zeros((h, w), int)
    for i, (n, d) in enumerate(planes, start=1):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = d / (v @ np.asarray(n, float))
        hit = np.isfinite(t) & (t > 0) & (t < best)
        best[hit], idx[hit] = t[hit], i
    depth = np.where(idx > 0, best, -1.0)
    pn = np.vstack([[0, 0, 0]] + [np.asarray(n, float) for n, _ in planes])
    pd_ = np.array([0.0] + [float(d) for _, d in planes])
    return da.DepthGrid(depth, idx, pn, pd_, pitch, roll)


GROUND = ((0.0, 0.0, 1.0), 2.5)             # level ground 2.5 m below the camera (z down)
FACADE_8M = ((0.0, 1.0, 0.0), -8.0)         # a wall 8 m ahead (forward is -y)


def at(depression, bearing=0.0):
    """Pano pixel (x, y) of a gravity direction."""
    return (bearing + 180.0) / 360.0 * W, y_for_depression(depression)


def test_g1_reconstruction_identity_and_label_ray_at_cell_centres():
    g = make_artifact([GROUND, FACADE_8M])
    h, w = g.depth.shape
    for r, c in [(40, 64), (50, 10), (33, 127), (60, 90), (20, 64)]:
        i = g.plane_indices[r, c]
        if i:
            assert g.depth[r, c] == pytest.approx(abs(g.planes_d[i] / (da.cell_ray(r, c, h, w) @ g.planes_n[i])))
        x, y = (c + 0.5) / w * W, (r + 0.5) / h * H
        np.testing.assert_allclose(da.label_ray(x, y, W, H), da.cell_ray(r, c, h, w), atol=1e-12)


def test_g2_axes():
    np.testing.assert_allclose(da.label_ray(W / 2, H / 2, W, H), [0, -1, 0], atol=1e-12)   # forward is -y
    np.testing.assert_allclose(da.label_ray(W / 4, H / 2, W, H), [1, 0, 0], atol=1e-12)    # 90 deg left is +x
    np.testing.assert_allclose(da.label_ray(W / 2, H, W, H), [0, 0, 1], atol=1e-12)        # down is +z


@pytest.mark.parametrize("dep", [10.0, 20.0, 45.0])
def test_g3_level_ground_is_flat_ground_trig(dep):
    g = make_artifact([GROUND])
    out = da.google_depth(g, *at(dep, 30.0), W, H, dep)
    assert out["g_status"] == "ground"
    assert out["g_ray_m"] == pytest.approx(2.5 / math.sin(math.radians(dep)), rel=1e-12)
    assert out["trig_cam_m"] == pytest.approx(out["g_ray_m"], rel=1e-12)
    assert out["cam_height_m"] == pytest.approx(2.5)


def test_g4_cell_value_differs_from_exact_intersection_near_horizon():
    g = make_artifact([GROUND], h=256, w=512)
    out = da.google_depth(g, *at(4.0), W, H, 4.0)
    exact = 2.5 / math.sin(math.radians(4.0))
    assert out["g_ray_m"] == pytest.approx(exact, rel=1e-12)
    assert abs(out["g_cell_m"] / exact - 1) > 0.01


def test_g5_facade():
    g = make_artifact([GROUND, FACADE_8M])
    el = 5.0
    out = da.google_depth(g, *at(-el), W, H, -el)
    assert out["g_status"] == "facade"
    assert out["g_ray_m"] == pytest.approx(8.0 / math.cos(math.radians(el)), rel=1e-12)
    out0 = da.google_depth(g, *at(-0.01), W, H)
    assert out0["g_ray_m"] == pytest.approx(8.0, rel=1e-4)


def _pixel_of_cell(r, c, h=64, w=128):
    return (c + 0.5) / w * W, (r + 0.5) / h * H


def test_g6_sky_no_plane_and_the_five_cell_minimum():
    g = make_artifact([GROUND])
    assert da.google_depth(g, *at(-20.0), W, H)["g_status"] == "sky"
    idx = g.plane_indices.copy()
    depth = g.depth.copy()
    r, c = 50, 60
    idx[r - 1:r + 2, c - 1:c + 2] = 0
    depth[r - 1:r + 2, c - 1:c + 2] = -1
    for k, (rr, cc) in enumerate([(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1), (r, c + 1)]):
        if k < 4:
            idx[rr, cc], depth[rr, cc] = 1, 5.0
    four = g._replace(plane_indices=idx.copy(), depth=depth.copy())
    assert da.google_depth(four, *_pixel_of_cell(r, c), W, H)["g_status"] == "no_plane"
    idx[r, c + 1], depth[r, c + 1] = 1, 5.0
    five = g._replace(plane_indices=idx, depth=depth)
    out = da.google_depth(five, *_pixel_of_cell(r, c), W, H)
    assert out["g_status"] == "ground" and out["n_valid"] == 5


def test_g7_columns_wrap_rows_do_not():
    h, w = 64, 128
    base = make_artifact([GROUND])
    r = 50
    idx = np.zeros((h, w), int)
    idx[r - 1:r + 2, w - 1] = 1          # three cells across the seam
    idx[r - 1, 0] = idx[r + 1, 0] = 1    # two at column 0, centre empty
    depth = np.where(idx > 0, 5.0, -1.0)
    g = base._replace(plane_indices=idx, depth=depth)
    assert da.google_depth(g, *_pixel_of_cell(r, 0), W, H)["n_valid"] == 5      # wrapped: 3 + 2
    idx = np.zeros((h, w), int)
    idx[h - 1, :] = 1                    # the bottom row: adjacent to row 0 only if rows wrapped
    idx[0, 0:2] = idx[1, 0:2] = 1        # four cells near the top
    depth = np.where(idx > 0, 5.0, -1.0)
    out = da.google_depth(base._replace(plane_indices=idx, depth=depth), *_pixel_of_cell(0, 1), W, H)
    assert out["n_valid"] == 4 and out["g_status"] == "sky"


def test_g8_centre_without_plane_takes_most_common_lowest_on_tie():
    g = make_artifact([GROUND, ((0.0, 0.0, 1.0), 3.0)])
    r, c = 50, 60
    idx = g.plane_indices.copy()
    block = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])
    idx[r - 1:r + 2, c - 1:c + 2] = block
    depth = np.where(idx > 0, 5.0, -1.0)
    out = da.google_depth(g._replace(plane_indices=idx, depth=depth), *_pixel_of_cell(r, c), W, H)
    assert out["plane"] == 1
    idx[r - 1, c - 1] = 2
    out = da.google_depth(g._replace(plane_indices=idx, depth=np.where(idx > 0, 5.0, -1.0)),
                          *_pixel_of_cell(r, c), W, H)
    assert out["plane"] == 2


def test_g9_bad_artifacts_never_raise(tmp_path):
    g = make_artifact([GROUND])
    assert da.load_depth_grid(str(tmp_path / "none.depth.npz")) == (None, "no_artifact")
    full = dict(depth=g.depth.astype(np.float32), plane_indices=g.plane_indices.astype(np.uint8),
                planes_n=g.planes_n.astype(np.float32), planes_d=g.planes_d.astype(np.float32),
                pitch=np.float64(math.radians(1.5)), roll=np.float64(-0.01))
    np.savez(tmp_path / "ok.npz", format_version=3, **full)
    grid, status = da.load_depth_grid(str(tmp_path / "ok.npz"))
    assert status is None and grid.pitch_deg == pytest.approx(1.5) and grid.depth.shape == (64, 128)
    np.savez(tmp_path / "v2.npz", format_version=2, depth=full["depth"])
    assert da.load_depth_grid(str(tmp_path / "v2.npz")) == (None, "old_format")
    np.savez(tmp_path / "v0.npz", depth=full["depth"])
    assert da.load_depth_grid(str(tmp_path / "v0.npz")) == (None, "old_format")
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    assert da.load_depth_grid(str(tmp_path / "junk.npz")) == (None, "unreadable")
    np.savez(tmp_path / "nofield.npz", format_version=3, depth=full["depth"])
    assert da.load_depth_grid(str(tmp_path / "nofield.npz")) == (None, "unreadable")
    bad = dict(full, plane_indices=np.full((64, 128), 9, np.uint8))
    np.savez(tmp_path / "badidx.npz", format_version=3, **bad)
    assert da.load_depth_grid(str(tmp_path / "badidx.npz")) == (None, "unreadable")


def test_g10_range_sign_and_negation():
    h, w = 64, 128
    g = make_artifact([GROUND])
    # A ray exactly on the horizon is parallel to level ground.
    idx = g.plane_indices.copy()
    idx[h // 2 - 2:h // 2 + 1, 10:13] = 1
    gp = g._replace(plane_indices=idx, depth=np.where(idx > 0, 5.0, -1.0))
    assert da.google_depth(gp, (11.5) / w * W, H / 2, W, H)["g_status"] == "out_of_range"
    # A hit 286 m away.
    g_fine = make_artifact([GROUND], h=1024, w=2048)
    assert da.google_depth(g_fine, *at(0.5), W, H)["g_status"] == "out_of_range"
    # The neighbour-plane fallback above the horizon: ground painted onto cells just above it.
    r = h // 2 - 1
    idx = g.plane_indices.copy()
    idx[r - 1:r + 1, 20:23] = 1
    idx[r, 21] = 0
    gb = g._replace(plane_indices=idx, depth=np.where(idx > 0, 5.0, -1.0))
    out = da.google_depth(gb, *_pixel_of_cell(r, 21), W, H)
    assert out["g_status"] == "out_of_range" and out["plane"] == 1
    assert np.isfinite(out["g_ray_abs_m"]) and out["g_ray_abs_m"] > 0
    # Negating every (n, d) changes nothing.
    neg = g._replace(planes_n=-g.planes_n, planes_d=-g.planes_d)
    for dep in (8.0, 25.0):
        assert da.google_depth(neg, *at(dep), W, H)["g_ray_m"] == pytest.approx(
            da.google_depth(g, *at(dep), W, H)["g_ray_m"], rel=1e-12)
    gneg = gb._replace(planes_n=-gb.planes_n, planes_d=-gb.planes_d)
    assert da.google_depth(gneg, *_pixel_of_cell(r, 21), W, H)["g_status"] == "out_of_range"


def _row(x, y, pitch=None, roll=None, author="human"):
    return {"pano_x": x, "pano_y": y, "pano_width": W, "pano_height": H, "pano_source": "gsv", "pano_id": "p",
            "camera_pitch": pitch, "camera_roll": roll, "author": author}


def _centre_of_view(x, y, pitch, roll=None, beta=0.95):
    from sidewalk_panorama_tools.geometry import label_view, sampling_grid
    view, tilt = label_view(x / W, y / H, fov_h_deg=90.0, size=2048, camera_pitch=pitch, camera_roll=roll,
                            beta=beta)
    sx, sy = sampling_grid(view, tilt, W, H)
    xs, ys = sx[1023:1025, 1023:1025].ravel(), sy[1023:1025, 1023:1025].ravel()
    ang = xs / W * 2 * math.pi                     # seam-aware mean of x
    mx = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * math.pi) * W) % W
    return mx, float(ys.mean())


@pytest.mark.parametrize("x,y,pitch,roll", [(12000.0, 5200.0, 3.0, None), (20.0, 6000.0, -2.0, None),
                                            (8192.0, y_for_depression(23.0), 1.3, None),
                                            (4000.0, 4700.0, 2.0, 1.5)])
def test_g11_lookup_pixel_is_the_view_centre(x, y, pitch, roll):
    pytest.importorskip("sidewalk_panorama_tools")
    lx, ly = da.crop_ray_pixel(_row(x, y, pitch, roll))
    cx, cy = _centre_of_view(x, y, pitch, roll)
    dx = (lx - cx + W / 2) % W - W / 2
    assert abs(dx) < 0.01 and abs(ly - cy) < 0.01
    assert da.crop_tilt_applied(_row(x, y, pitch, roll))


def test_g11_ai_and_no_pose_rows_use_the_stored_pixel():
    pytest.importorskip("sidewalk_panorama_tools")
    assert da.crop_ray_pixel(_row(12000.0, 5200.0, 3.0, author="ai")) == (12000.0, 5200.0)
    assert da.crop_ray_pixel(_row(12000.0, 5200.0, None, None)) == (12000.0, 5200.0)
    assert da.crop_ray_pixel(_row(12000.0, 5200.0, 0.0, None)) == (12000.0, 5200.0)
    assert not da.crop_tilt_applied(_row(12000.0, 5200.0, 3.0, author="ai"))


def test_g11_tilted_rig_over_level_ground():
    pytest.importorskip("sidewalk_panorama_tools")
    from sidewalk_panorama_tools import pose as pano_pose
    pitch, beta = 3.0, 0.95
    R = pano_pose.rig_from_gravity(beta * pitch, 0.0)      # v_rig = R v_gravity; the same map on -RFU vectors
    n_rig = R @ np.array(GROUND[0])
    g = make_artifact([(tuple(n_rig), GROUND[1])], h=256, w=512)
    for dep, bearing in ((15.0, 0.0), (25.0, 140.0)):
        x, y = at(dep, bearing)
        row = _row(x, y, pitch)
        out = da.google_depth(g, *da.crop_ray_pixel(row), W, H, dep)
        assert out["g_ray_m"] == pytest.approx(2.5 / math.sin(math.radians(dep)), rel=1e-9)
        stored = da.google_depth(g, x, y, W, H, dep)
        assert abs(stored["g_ray_m"] / out["g_ray_m"] - 1) > 0.05


@pytest.mark.skipif(not os.environ.get("PANORAMA_TOOLS_CHECKOUT"), reason="PANORAMA_TOOLS_CHECKOUT not set")
def test_g12_cross_check_with_croprunner(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(os.environ["PANORAMA_TOOLS_CHECKOUT"])
    monkeypatch.chdir(os.environ["PANORAMA_TOOLS_CHECKOUT"])
    CropRunner = pytest.importorskip("CropRunner")
    g = make_artifact([GROUND, FACADE_8M], h=256, w=512)
    np.savez(tmp_path / "a.depth.npz", format_version=3, depth=g.depth.astype(np.float32),
             plane_indices=g.plane_indices.astype(np.uint8), planes_n=g.planes_n.astype(np.float32),
             planes_d=g.planes_d.astype(np.float32), pitch=np.float64(0.0), roll=np.float64(0.0),
             heading=np.float64(0.0))
    grid, reason = CropRunner.load_depth_grid(str(tmp_path / "a.depth.npz"))
    mine, _ = da.load_depth_grid(str(tmp_path / "a.depth.npz"))
    checked = 0
    for dep, bearing in ((8.0, 10.0), (15.0, -60.0), (30.0, 170.0)):
        x, y = at(dep, bearing)
        est = CropRunner.label_distance(x, y, W, H, grid, reason)
        out = da.google_depth(mine, x, y, W, H, dep)
        if out["g_status"] == "ground" and est.source == CropRunner.DISTANCE_SOURCE_DEPTH:
            assert out["g_ray_abs_m"] * math.cos(math.radians(dep)) == pytest.approx(est.distance_m, rel=1e-5)
            checked += 1
    assert checked >= 2
    del sys.modules["CropRunner"]


def test_g13_ground_plane_ignores_a_ceiling():
    h = 64
    ground_tilted = ((0.0, 0.1, 0.995), 2.5)                # verticality 0.995
    ceiling = ((0.0, 0.0, 1.0), -3.0)                       # perfectly flat, 3 m above
    g = make_artifact([ground_tilted, ceiling])
    idx = g.plane_indices.copy()
    idx[h // 2:h // 2 + 2, :20] = 2                         # the ceiling also covers a few below-horizon cells
    g = g._replace(plane_indices=idx)
    assert (g.plane_indices == 2).sum() > (g.plane_indices == 1).sum()
    assert da.ground_plane(g) == 1


# ---------------------------------------------------------------------------------------------------------------------
# 14-17b: sampling
# ---------------------------------------------------------------------------------------------------------------------

def big_frame(tmp_path, n_per=60):
    root = make_round3(str(tmp_path / "big"), cities=("pub-a", "pub-b", "pub-c", "secret-town"), n_per=n_per)
    return root, da.load_frame(root, os.path.join(root, "rawlabels"))


def test_14_deterministic_and_rank_independent(tmp_path):
    _, f = big_frame(tmp_path)
    pub = {"pub-a", "pub-b", "pub-c"}
    a = da.draw_sample(f, pub, n_per_type=10, oversample=3)
    b = da.draw_sample(f, pub, n_per_type=10, oversample=3)
    pd.testing.assert_frame_equal(a, b)
    assert da.rank(1, "x:1") != da.rank(2, "x:1")
    dropped = a.label_uid.iloc[5]
    c = da.draw_sample(f[f.label_uid != dropped], pub, n_per_type=10, oversample=3)
    common = a[a.label_uid != dropped].label_uid
    ranks_a = a.set_index("label_uid")["rank"]
    ranks_c = c.set_index("label_uid")["rank"]
    shared = common[common.isin(ranks_c.index)]
    assert (ranks_a[shared] == ranks_c[shared]).all() and len(shared) >= len(common) - 1


def test_15_sample_rules(tmp_path):
    _, f = big_frame(tmp_path)
    pub = {"pub-a", "pub-b", "pub-c"}
    s = da.draw_sample(f, pub, n_per_type=10, oversample=3)
    assert not s.city.eq("secret-town").any()
    assert s.label_uid.is_unique
    assert not s.duplicated(["label_type", "city", "pano_id"]).any()
    for t, g in s.groupby("label_type"):
        g = g.sort_values("order")
        assert len(g) == 30 and g.set_a.sum() == 10
        assert g.set_a.tolist() == [True] * 10 + [False] * 20       # Set A is the prefix
        first9 = g.head(9).city.value_counts()
        assert first9.max() - first9.min() == 0                     # round-robin over the three cities
        assert np.diff(g["order"]).min() == 1
        a = g[g.set_a]
        rows = f[(f.label_type == t) & f.city.isin(set(a.city))]
        assert a.inclusion_weight.sum() == pytest.approx(len(rows))
        assert g.inclusion_weight_cand.sum() == pytest.approx(len(f[(f.label_type == t) & f.city.isin(pub)]))
        assert g[~g.set_a].inclusion_weight.isna().all()


def test_16_shared_label_ids_and_loud_joins(tmp_path):
    root, f = big_frame(tmp_path, n_per=4)
    assert f.label_uid.is_unique
    assert set(f[f.label_id == "1"].label_uid) == {"pub-a:1", "pub-b:1", "pub-c:1", "secret-town:1"}
    raw = pd.read_csv(os.path.join(root, "rawlabels", "pub-a.csv"))
    pd.concat([raw, raw.head(1)]).to_csv(os.path.join(root, "rawlabels", "pub-a.csv"), index=False)
    with pytest.raises(ValueError, match="twice"):
        da.load_frame(root, os.path.join(root, "rawlabels"))
    raw.iloc[1:].to_csv(os.path.join(root, "rawlabels", "pub-a.csv"), index=False)
    with pytest.raises(ValueError, match="no rawLabels row"):
        da.load_frame(root, os.path.join(root, "rawlabels"))


def test_17_sample_outputs_and_store_root(tmp_path):
    root, _ = big_frame(tmp_path, n_per=12)
    pub = tmp_path / "pub.json"
    pub.write_text(json.dumps(["pub-a", "pub-b", "pub-c"]))
    out, public_out = tmp_path / "data", tmp_path / "public"
    da.main(["sample", "--root", root, "--rawlabels-dir", os.path.join(root, "rawlabels"), "--out-dir", str(out),
             "--public-out", str(public_out), "--public-cities", str(pub), "--n-per-type", "4", "--oversample", "2"])
    full = pd.read_csv(out / "sample.csv")
    public = pd.read_csv(public_out / "sample_public.csv")
    assert len(full) == len(public) == 5 * 8
    assert not any(c in public for c in ("pano_path", "depth_m", "crop_half_px", "store_root"))
    assert not public.city.eq("secret-town").any()
    lines = (out / "stage_files.txt").read_text().split()
    assert all(re.fullmatch(r"pub-[abc]/[A-Za-z0-9]{2}/[A-Za-z0-9]+\.depth\.npz", ln) for ln in lines)
    assert len(lines) == len(full[["city", "pano_id"]].drop_duplicates())
    rec = json.loads((public_out / "sample.json").read_text())
    assert rec["seed"] == da.SEED and "secret" not in json.dumps(rec)
    p = "/s/pano-stage/CurbRamp/pub-a/Pp/Ppub0001x.jpg"
    assert da.store_root(p, "pub-a", "Ppub0001x") == "/s/pano-stage/CurbRamp"
    with pytest.raises(ValueError):
        da.store_root(p, "pub-b", "Ppub0001x")
    with pytest.raises(ValueError):
        da.store_root("/s/pub-a/Pp/Ppub0001x.png", "pub-a", "Ppub0001x")


def test_17b_set_g_depends_on_file_level_status_only():
    sample = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(6)], "label_type": "CurbRamp",
                           "order": range(6)})
    google = pd.DataFrame({"label_uid": sample.label_uid,
                           "g_status": ["ground", "no_artifact", "facade", "ground", "old_format", "ground"]})
    base = da.set_g(sample, google, n_per_type=3)
    assert base.tolist() == [True, False, True, True, False, False]
    google.loc[0, "g_status"] = "sky"
    assert da.set_g(sample, google, n_per_type=3).tolist() == base.tolist()
    google.loc[0, "g_status"] = "unreadable"
    assert da.set_g(sample, google, n_per_type=3).tolist() == [False, False, True, True, False, True]
    assert da.set_g(sample, google.iloc[1:], n_per_type=3)[0] == False  # noqa: E712  (absent = no_artifact)


# ---------------------------------------------------------------------------------------------------------------------
# 18-20: depth pass and job files
# ---------------------------------------------------------------------------------------------------------------------

def test_18_centre_depths():
    d = np.full((2048, 2048), 30.0)
    d[1024, 1024] = 10.0
    d[1008:1041, 1008:1020] = 50.0
    c, b = da.centre_depths(d)
    assert c == 10.0 and b == 30.0


class FakeModel:
    def __init__(self, value):
        self.value = value

    def infer_image(self, bgr):
        assert bgr.shape == (8, 8, 3) and bgr.flags["C_CONTIGUOUS"]
        return np.full((64, 64), self.value)


def _sample_rows():
    rows = []
    for i, (city, pano) in enumerate([("pub-a", "AAx"), ("pub-a", "AAx"), ("pub-b", "BBy"), ("pub-b", "missing")]):
        rows.append({"label_uid": f"{city}:{i}", "city": city, "pano_id": pano, "pano_x": 100.0 * i,
                     "pano_y": 5000.0, "pano_width": W, "pano_height": H, "pano_source": "gsv",
                     "camera_pitch": 2.0, "camera_roll": np.nan, "author": "human",
                     "pano_path": f"/store/root/{city}/{pano[:2]}/{pano}.jpg"})
    return pd.DataFrame(rows)


def test_19_depth_pass_reader_device_resume_and_statuses(tmp_path):
    reads, renders = [], []

    def reader(store, city, pano_id):
        reads.append((store, city, pano_id))
        if pano_id == "missing":
            return "no_pano", {}
        return np.zeros((4, 8, 3), np.uint8), {}

    def render(pano_rgb, x_norm, y_norm, camera_pitch=None, camera_roll=None, apply_tilt=True, beta=0.95,
               device=None):
        renders.append(dict(device=device, apply_tilt=apply_tilt, beta=beta, pitch=camera_pitch))
        if x_norm > 150 / W and x_norm < 250 / W:
            raise RuntimeError("boom")
        return np.zeros((8, 8, 3), np.uint8), True

    out = str(tmp_path / "depths.csv")
    loader = lambda enc, device: FakeModel({"vitb": 20.0, "vitl": 30.0}[enc])  # noqa: E731
    da.depth_pass(_sample_rows(), out, ("vitb", "vitl"), "cuda", reader, loader, render)
    assert reads == [("/store/root", "pub-a", "AAx"), ("/store/root", "pub-b", "BBy"),
                     ("/store/root", "pub-b", "missing")]
    assert all(r["device"] == "cuda" and r["apply_tilt"] and r["beta"] == 0.95 and r["pitch"] == 2.0
               for r in renders)
    d = pd.read_csv(out).set_index("label_uid")
    assert d.loc["pub-a:0", "status"] == "ok" and d.loc["pub-a:0", "d_vitl_m"] == 30.0
    assert d.loc["pub-a:0", "renderer"] == "torch" and d.loc["pub-a:0", "d_vitb_box_m"] == 20.0
    assert d.loc["pub-b:2", "status"] == "error:RuntimeError"
    assert d.loc["pub-b:3", "status"] == "no_pano"
    reads.clear()
    da.depth_pass(_sample_rows(), out, ("vitb", "vitl"), "cuda", reader, loader, render)
    assert reads == [("/store/root", "pub-b", "BBy")]            # only the errored label is retried
    assert len(da.load_depths(out)) == 4
    da.depth_pass(_sample_rows(), str(tmp_path / "cpu.csv"), ("vitb",), None, reader, loader, render,
                  pano_store="/fresh")
    assert pd.read_csv(tmp_path / "cpu.csv").renderer.eq("numpy").all() and reads[-1][0] == "/fresh"


SLURM = ["depth_agreement.slurm", "depth_agreement_stage.slurm"]
ANALYSIS = os.path.join(os.path.dirname(__file__), "..", "analysis")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", SLURM)
def test_20_slurm_parses_and_guards_cwd(name):
    path = os.path.join(ANALYSIS, name)
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = open(path).read()
    assert "W=$(pwd -P)" in text                                  # the physical path, symlinks resolved
    guard = next(ln for ln in text.splitlines() if ln.startswith('[[ "$W" =~'))
    for w, ok in (("/gscratch/scrubbed/jfroehli/depth-ablation-2026-11", True),
                  ("/mmfs1/gscratch/scrubbed/jfroehli/depth-ablation-2026-11", True),
                  ("/gscratch/scrubbed/jfroehli/depth-ablation-2026-11/sub", True),
                  ("/gscratch/makelab/jfroehli/sidewalk-validator-ai", False),
                  ("/gscratch/scrubbed/jfroehli/nosidewalk-gate", False),
                  # a same-named directory inside another study's directory, or a look-alike name, is refused
                  ("/gscratch/scrubbed/jfroehli/nosidewalk-gate/depth-ablation-2026-11", False),
                  ("/gscratch/scrubbed/jfroehli/depth-ablation-2026-11-old", False),
                  ("/gscratch/makelab/x/gscratch/scrubbed/u/depth-ablation-2026-11", False), ("/tmp", False)):
        r = subprocess.run(["bash", "-c", f'W="{w}"; {guard}; echo passed'], capture_output=True, text=True)
        assert ("passed" in r.stdout) == ok, (w, r.stdout)
    assert "#SBATCH --account=makelab" in text and "#SBATCH --partition=ckpt-all" in text
    assert "/gscratch/makelab/jfroehli/sidewalk-validator-ai" not in text.replace("never", "")


def test_20_gpu_job_is_offline_and_pins_vitb():
    text = open(os.path.join(ANALYSIS, "depth_agreement.slurm")).read()
    assert "HF_HUB_OFFLINE=1" in text and "depth-pass" in text and "google-depth" in text
    assert "e96fd0c8cedf1825a55b9c61309f754982948df0" in text
    assert "#SBATCH --no-requeue" in open(os.path.join(ANALYSIS, "depth_agreement_stage.slurm")).read()


# ---------------------------------------------------------------------------------------------------------------------
# 21-30: compare and report
# ---------------------------------------------------------------------------------------------------------------------

def _pairs_frame(n=400, cities=8, seed=0):
    rng = np.random.default_rng(seed)
    city = np.array([f"c{i % cities}" for i in range(n)])
    pano = np.array([f"p{i // 2}" for i in range(n)])
    return rng, city, pano


def test_21_exact_double():
    rng, city, pano = _pairs_frame()
    b = np.exp(rng.normal(3, 0.4, len(city)))
    ps = da.pair_stats(2 * b, b, pano, city, n_boot=50)
    assert ps["s"] == pytest.approx(2.0) and ps["matched_constant"] == pytest.approx(12200)
    assert ps["bracket"]["beta_yx"] == pytest.approx(1.0) and ps["bracket"]["beta_xy"] == pytest.approx(1.0)
    assert ps["bracket"]["verdict"] == "inside"
    assert all(ps["moved_raw"][f"t={t:g}"]["share"] == 1.0 for t in da.CROP_DIFF_THRESHOLDS)
    assert all(ps["moved_matched"][f"t={t:g}"]["share"] == 0.0 for t in da.CROP_DIFF_THRESHOLDS)


def test_22_errors_in_variables_bracket():
    rng = np.random.default_rng(1)
    x = rng.normal(3, 0.4, 4000)
    a, b = x, x + rng.normal(0, 0.25, len(x))
    s = da.slope_bracket(a, b)
    assert s["beta_yx"] < 0.9 < 1 < s["beta_xy"]                 # a one-direction slope alone reads "compressed"
    assert da.bracket_verdict(s["beta_yx"], s["beta_xy"]) == "straddles"
    a2 = 0.7 * x + rng.normal(0, 0.01, len(x))
    s2 = da.slope_bracket(a2, x)
    assert da.bracket_verdict(s2["beta_yx"], s2["beta_xy"]) == "outside"


def test_23_crop_difference_is_symmetric():
    s = da.crop_moved_shares(np.array([math.log(1.11), -math.log(1.11), math.log(1.09), -math.log(1.09)]))
    assert s["t=0.1"]["count"] == 2 and s["t=0.1"]["share"] == 0.5


def test_24_weights_and_kish_n():
    L = np.log(np.array([1.0, 1.3, 1.0, 2.0, 0.5]))
    plain = da.crop_moved_shares(L)
    eq = da.crop_moved_shares(L, weights=np.full(5, 3.0))
    assert eq["t=0.1"]["share"] == plain["t=0.1"]["share"] and eq["n_eff"] == pytest.approx(5)
    uneq = da.crop_moved_shares(L, weights=np.array([10, 1, 1, 1, 1.0]))
    assert uneq["n_eff"] < 5 and uneq["t=0.1"]["share"] == pytest.approx(3 / 14)


def test_25_bootstraps():
    rng, city, pano = _pairs_frame(n=300, cities=6)
    v = rng.normal(0, 1, len(city))
    a = da.median_with_ci(v, pd.Series(city) + ":" + pano, city, n_boot=200, seed=3)
    b = da.median_with_ci(v, pd.Series(city) + ":" + pano, city, n_boot=200, seed=3)
    assert a == b
    dup = da.median_with_ci(np.repeat(v, 2), np.repeat(pd.Series(city) + ":" + pano, 2), np.repeat(city, 2),
                            n_boot=200, seed=3)
    assert (dup[2] - dup[1]) >= 0.9 * (a[2] - a[1])              # duplicating labels does not narrow it
    effect = v + np.array([int(c[1:]) for c in city]) * 0.5     # a city effect
    within = da.median_with_ci(effect, pd.Series(city) + ":" + pano, city, n_boot=300, seed=4)
    whole = da.median_with_ci(effect, city, None, n_boot=300, seed=4)
    assert (whole[2] - whole[1]) > (within[2] - within[1])


def test_26_physical_width():
    assert da.physical_width_m(7.0, 7.0) == pytest.approx(2 * 6100 / 1024)
    assert da.physical_width_m(20.0, 10.0) == pytest.approx(6100 / 1024)


def test_27_merge_sources():
    sample = pd.DataFrame({"label_uid": ["a:1", "a:2"], "set_a": [True, True]})
    depths = pd.DataFrame({"label_uid": ["a:1", "a:2"], "status": "ok", "d_vitb_m": [1.0, 2.0]})
    google = pd.DataFrame({"label_uid": ["a:1"], "g_status": ["ground"], "g_ray_m": [1.0]})
    trig = pd.DataFrame({"label_uid": ["a:1", "a:2"], "trig_m": [1.0, 2.0], "ratio": [1.0, 1.0],
                         "why_excluded": ["", ""]})
    m = da.merge_sources(sample, depths, google, trig)
    assert m.set_index("label_uid").g_status.to_dict() == {"a:1": "ground", "a:2": "no_artifact"}
    with pytest.raises(ValueError, match="duplicated"):
        da.merge_sources(sample, pd.concat([depths, depths.head(1)]), google, trig)
    assert len(da.load_depths_frame(pd.concat([depths, depths.head(1)]))) == 2


def test_28_clipping_table(tmp_path):
    root = str(tmp_path / "r")
    for t in da.LABEL_TYPES:
        os.makedirs(os.path.join(root, t))
        d = np.array([3.0, 5.0, 5.96, 5.95, 6.0, 30.0, 2.0])
        pd.DataFrame({"depth_m": d, "clipped": (np.floor(6100 / d) > 1024).astype(int),
                      "status": ["ok"] * 6 + ["no_pano"]}).to_csv(os.path.join(root, t, "split.csv"), index=False)
    tb = da.clipping_table(root).set_index("label_type")
    assert tb.loc["CurbRamp", "n"] == 6
    assert tb.loc["CurbRamp", "c6100_clipped"] == tb.loc["CurbRamp", "recorded_clipped"] == 3
    ok = np.array([3.0, 5.0, 5.96, 5.95, 6.0, 30.0])
    for c in (4000, 6100, 8000):
        assert tb.loc["CurbRamp", f"c{c}_share"] == pytest.approx(np.mean(np.floor(c / ok) > 1024))
    assert tb.loc["all types", "n"] == 30


def test_29_rule_readout():
    def r1(overall, types, weighted=None, jitter=0.01, t25=None):
        cell = {"overall": overall, "weighted": weighted if weighted is not None else overall, "by_type": types}
        return {"jitter_share_10": jitter, "t=0.1": cell, "t=0.25": t25 or cell}
    base = {"CurbRamp": 0.90, "Obstacle": 0.97}
    assert da.rule_readout({"r1": r1(0.95, base)})["R1"]["met"]
    out = da.rule_readout({"r1": r1(0.96, {**base, "Obstacle": 0.895})})["R1"]
    assert not out["met"] and out["types_below_90"] == ["Obstacle"] and "Obstacle" in out["marginal_types"]
    out = da.rule_readout({"r1": r1(0.90, base, jitter=0.06, t25=r1(0.99, {"CurbRamp": 0.99})["t=0.1"])})["R1"]
    assert out["tolerance"] == 0.25 and out["jitter_fallback"] and out["met"]
    out = da.rule_readout({"r1": r1(0.96, base, weighted=0.94)})["R1"]
    assert not out["met"] and out["weighted_disagrees"]
    assert da.rule_readout({"r0": {"n": 1000, "within": 990}})["R0"]["met"]
    assert not da.rule_readout({"r0": {"n": 1000, "within": 989}})["R0"]["met"]
    for (lo, hi), v in (((0.92, 1.05), "inside"), ((0.6, 0.85), "outside"), ((0.85, 0.97), "straddles")):
        assert da.rule_readout({"r2": {"beta_yx": lo, "beta_xy": hi}})["R2"]["verdict"] == v


def _synthetic_merged(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in da.LABEL_TYPES:
        for i in range(60):
            dep = rng.uniform(6, 35)
            trig = 2.5 / math.sin(math.radians(dep))
            d = 2.7 * trig * math.exp(rng.normal(0, 0.1))
            rows.append({"label_uid": f"c{i % 5}:{t}{i}", "city": f"c{i % 5}", "pano_id": f"p{t}{i // 2}",
                         "label_type": t, "severity": 1.0, "tags": "[]", "image_capture_date": "2020-01",
                         "set_a": i < 40, "set_g": i % 3 != 0, "order": i, "inclusion_weight": 1.0 + i % 2,
                         "status": "ok", "renderer": "torch", "depth_m": round(d, 4), "d_vitb_m": d,
                         "d_vitb_box_m": d * math.exp(rng.normal(0, 0.02)), "d_vitl_m": 1.2 * d,
                         "d_vitl_box_m": 1.2 * d, "g_status": "ground" if i % 4 else "facade",
                         "g_ray_m": trig * math.exp(rng.normal(0, 0.05)), "trig_m": trig, "vitb_trig_ratio": d / trig,
                         "why_excluded": "", "cam_height_m": 2.5, "camera_pitch": 1.0, "art_pitch_deg": 1.1,
                         "art_roll_deg": 0.2})
    return pd.DataFrame(rows)


def test_compare_end_to_end_on_synthetic_sources():
    tables, readout, stats = da.compare(_synthetic_merged(), n_boot=50)
    assert readout["R0"]["met"] and readout["R0"]["n"] == 200
    assert readout["R1"]["met"] and readout["R1"]["tolerance"] == 0.1
    assert stats["vitl_vitb"]["s"] == pytest.approx(1.2)
    assert readout["R2"]["verdict"] in ("inside", "straddles", "outside")
    names = {t["name"] for t in tables}
    assert {"pairs", "jitter", "vitl_type_severity", "vitl_type_tag", "coverage_selection", "artifact_pose"} <= names


def test_30_report_is_deterministic_and_aggregate_only(r3, tmp_path):
    pub = tmp_path / "public.json"
    pub.write_text(json.dumps(sorted(PUBLIC)))
    da.main(["trig", "--root", r3, "--rawlabels-dir", os.path.join(r3, "rawlabels"), "--out-dir",
             str(tmp_path / "trig"), "--public-cities", str(pub), "--n-boot", "20"])
    da.main(["clipping", "--root", r3, "--out", str(tmp_path / "clipping.csv")])
    outs = []
    for k in (1, 2):
        assets = tmp_path / f"assets{k}"
        da.main(["report", "--trig-dir", str(tmp_path / "trig"), "--clipping", str(tmp_path / "clipping.csv"),
                 "--assets", str(assets), "--no-figures"])
        outs.append((assets / "tables.md").read_text())
    assert outs[0] == outs[1]
    md = outs[0]
    assert "secret" not in md and not re.search(r"pub-[ab]:\d", md)   # no private city, no label uid
    assert "non-public deployments (1)" in md and "Clipping at the view edge" in md


def test_report_figure_is_written(r3, tmp_path):
    pytest.importorskip("matplotlib")
    pub = tmp_path / "public.json"
    pub.write_text(json.dumps(sorted(PUBLIC)))
    da.main(["trig", "--root", r3, "--rawlabels-dir", os.path.join(r3, "rawlabels"), "--out-dir",
             str(tmp_path / "trig"), "--public-cities", str(pub), "--n-boot", "0"])
    da.main(["report", "--trig-dir", str(tmp_path / "trig"), "--assets", str(tmp_path / "a")])
    assert (tmp_path / "a" / "trig_hist.png").stat().st_size > 1000
    prov = json.loads((tmp_path / "a" / "provenance.json").read_text())
    assert set(prov["outputs"]) == {"tables.md", "trig_hist.png"}


# ---------------------------------------------------------------------------------------------------------------------
# Review round (PR #104): tests for mutations the suite above did not catch, and for the review's fixes
# ---------------------------------------------------------------------------------------------------------------------

VITL_GATE = r'^VITL_(REV|SHA256)=|^\[\[ "\$VITL_'


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("env,passes", [
    ({}, False), ({"VITL_REV": "main"}, False), ({"VITL_REV": "TODO-pin-revision"}, False),
    ({"VITL_REV": "0123456789abcdef0123456789abcdef01234567"}, False),                       # no checkpoint sha256
    ({"VITL_REV": "0123456789ABCDEF0123456789abcdef01234567", "VITL_SHA256": "a" * 64}, False),
    ({"VITL_REV": "0123456789abcdef0123456789abcdef01234567", "VITL_SHA256": "a" * 63}, False),
    ({"VITL_REV": "0123456789abcdef0123456789abcdef01234567", "VITL_SHA256": "a" * 64}, True)])
def test_r_gpu_job_refuses_to_run_until_vitl_is_pinned(env, passes):
    """The ViT-L checkpoint is not pinned in the PR. The job must stop, not run on whatever file is there."""
    text = open(os.path.join(ANALYSIS, "depth_agreement.slurm")).read()
    gate = [ln for ln in text.splitlines() if re.search(VITL_GATE, ln)]
    assert len(gate) == 4
    r = subprocess.run(["bash", "-c", "\n".join(gate) + "\necho gate-passed"], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **env})
    assert ("gate-passed" in r.stdout) == passes, r.stdout
    assert passes or ("refusing to run" in r.stdout and r.returncode == 2)
    lines = text.splitlines()
    first_python = next(k for k, ln in enumerate(lines) if ln.startswith("python "))
    assert max(lines.index(g) for g in gate) < first_python            # the gate comes before any work
    assert any("sha256sum --check" in ln and "VITL_SHA256" in ln for ln in lines[:first_python])
    assert 'VITL_REV=${VITL_REV:-}' in text and "TRANSFORMERS_OFFLINE=1" in text


@pytest.mark.parametrize("name", SLURM)
def test_r_jobs_test_a_real_write_before_any_work(name):
    lines = open(os.path.join(ANALYSIS, name)).read().splitlines()
    dd = next(k for k, ln in enumerate(lines) if "dd if=/dev/zero" in ln and 'of="$W/.write_test' in ln)
    work = next(k for k, ln in enumerate(lines) if ln.startswith(("python ", "timeout ")))
    guard = next(k for k, ln in enumerate(lines) if ln.startswith('[[ "$W" =~'))
    assert guard < dd < work


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_r_stage_list_must_be_store_relative(tmp_path):
    text = open(os.path.join(ANALYSIS, "depth_agreement_stage.slurm")).read()
    check = next(ln for ln in text.splitlines() if ln.startswith("if grep -qE"))
    for body, ok in (("a/bc/bcd.depth.npz\n", True), ("/etc/passwd\n", False), ("a/../../x.jpg\n", False),
                     ("../x\n", False), ("a/b..c/x..y.jpg\n", True)):
        lst = tmp_path / "list.txt"
        lst.write_text(body)
        r = subprocess.run(["bash", "-c", f'LIST="{lst}"; {check}; echo passed'], capture_output=True, text=True)
        assert ("passed" in r.stdout) == ok, (body, r.stdout)
