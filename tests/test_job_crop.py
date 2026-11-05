"""job_crop.py: the port of SidewalkWebpage's crop-job geometry (CropSizingRule v2, CropGeometry, CropService).

``tests/fixtures/job_crop/`` is a verbatim copy of SidewalkWebpage ``test/resources/crops/`` at 5d89dc179
(``mechanics.json``, ``sizing-v2.json``, ``synthetic-pano.png``, ``expected/*.png``). Those files were written by
panorama-tools' ``CropRunner.py`` itself (commit c1d0751, see that directory's README) for the Scala suite
``test/service/CropGeometrySpec.scala``, so passing them here pins this port to the same reference the backend is
pinned to. The cases below that are not fixture-driven are the Scala spec's own, ported verbatim, plus hand-computed
ones.
"""
import doctest
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import job_crop as jc

FIX = Path(__file__).parent / "fixtures" / "job_crop"


def _box(d):
    return jc.CropBox(d["left"], d["top"], d["width"], d["height"], d["shifted"])


def _pano_rgb():
    """The synthetic pano as RGB (the reference wrote PNGs with PIL, so RGB order is what its pixels mean)."""
    return cv2.imread(str(FIX / "synthetic-pano.png"))[..., ::-1]


def test_doctests():
    result = doctest.testmod(jc)
    assert result.attempted > 0 and result.failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# CropSizingRule
# ---------------------------------------------------------------------------------------------------------------------

def test_reference_crop_size_at_the_horizon_and_both_clamps():
    d0 = jc.V1_DIST_INTERCEPT
    assert jc.reference_crop_size(0.0) == pytest.approx(jc.V1_SIZE_COEF * d0 ** jc.V1_SIZE_EXP)
    zero_at = -jc.V1_DIST_INTERCEPT / jc.V1_DIST_SLOPE  # about -1299.6: distance reaches 0 here
    assert zero_at == pytest.approx(-1299.6, abs=0.1)
    assert jc.reference_crop_size(zero_at - 1) == jc.V1_SIZE_MAX
    assert jc.reference_crop_size(-1290.0) == jc.V1_SIZE_MAX  # tiny distance, huge size: clamped to 1500
    assert jc.reference_crop_size(100000.0) == jc.V1_SIZE_MIN  # far field


def test_predict_crop_size_is_the_identity_at_the_reference_height_and_scales_with_it():
    y = 3000.0
    ref = jc.predict_crop_size(y, 6656)
    assert ref == pytest.approx(jc.reference_crop_size(6656 / 2 - y))
    k = 8192 / 6656
    assert jc.predict_crop_size(y * k, 8192) == pytest.approx(ref * k)  # same angle, more pixels


def test_window_width_clamps_as_an_angle():
    assert jc.window_width(0, 2048, 1024) == pytest.approx(8 / 360 * 2048)  # 45.5 px at the floor
    assert jc.window_width(0, 2048, 1024) == pytest.approx(45.511, abs=1e-3)
    assert jc.window_width(8191, 16384, 8192) == pytest.approx(4096.0)  # 90 degrees at the ceiling
    y, w, h = 4096.0, 16384, 8192  # unclamped, by hand
    deg = jc.predict_crop_size(y, h) * 2.5 / h * 180
    assert 8 < deg < 90
    assert jc.window_width(y, w, h) == pytest.approx(deg / 360 * w)


def test_window_width_is_an_azimuthal_span():
    # CropGeometrySpec: on a square pano the floor is a fraction of the WIDTH; doubling the width doubles it.
    w = h = 4096
    assert jc.window_width(0, w, h) == pytest.approx(8 / 360 * w)
    assert jc.window_width(0, 2 * h, h) == pytest.approx(2 * jc.window_width(0, w, h))


def test_sizing_fixture_constants_match():
    s = json.loads((FIX / "sizing-v2.json").read_text())
    assert s["crop_rule_version"] == jc.RULE_VERSION
    assert s["crop_size_scale"] == jc.SCALE
    assert (s["crop_min_fov_deg"], s["crop_max_fov_deg"]) == (jc.MIN_FOV_DEG, jc.MAX_FOV_DEG)
    assert s["crop_aspect_w_over_h"] == jc.ASPECT
    assert s["crop_max_stored_width"] == jc.MAX_STORED_WIDTH


def test_every_sizing_fixture_row_the_reference_wrote():
    rows = json.loads((FIX / "sizing-v2.json").read_text())["rows"]
    assert len(rows) > 50
    for r in rows:
        w, h, x, y = r["pano_width"], r["pano_height"], r["pano_x"], r["pano_y"]
        win = jc.window_width(y, w, h)
        assert jc.predict_crop_size(y, h) == pytest.approx(r["predict_crop_size"], abs=1e-6), r
        assert win == pytest.approx(r["window_width"], abs=1e-6), r
        assert jc.compute_crop_box(x, y, win, w, h) == _box(r["box"]), r


# ---------------------------------------------------------------------------------------------------------------------
# CropGeometry
# ---------------------------------------------------------------------------------------------------------------------

def test_compute_crop_box_interior_seam_and_poles():
    # The cases panorama-tools pins for its own compute_crop_box (CropGeometrySpec, ported verbatim).
    assert jc.compute_crop_box(300, 128, 248.33, 512, 256) == (176, 46, 248, 165, False)
    assert jc.compute_crop_box(0, 128, 248.33, 512, 256) == (-124 % 512, 46, 248, 165, False)
    assert jc.compute_crop_box(300, 8, 248.33, 512, 256) == (176, 0, 248, 165, True)
    bottom = jc.compute_crop_box(300, 250, 200.0, 512, 256)
    assert bottom.top == 256 - 133 and bottom.shifted


def test_compute_crop_box_hand_computed_on_a_full_size_pano():
    # (a) centre of a 16384x8192 pano with a 1000-px request: width 1000, height round(666.67) = 667.
    assert jc.compute_crop_box(8192, 4096, 1000.0, 16384, 8192) == (7692, 4096 - 334, 1000, 667, False)
    # (b) x = 20: left = round(20 - 500) = -480, wrapped.
    assert jc.compute_crop_box(20, 4096, 1000.0, 16384, 8192).left == 16384 - 480
    # (c) near the top and the bottom: shifted onto the edge.
    top = jc.compute_crop_box(8192, 100, 1000.0, 16384, 8192)
    assert (top.top, top.shifted) == (0, True)
    bot = jc.compute_crop_box(8192, 8150, 1000.0, 16384, 8192)
    assert (bot.top, bot.shifted) == (8192 - 667, True)


def test_compute_crop_box_caps_on_the_binding_axis():
    landscape = jc.compute_crop_box(300, 250, 1500, 512, 256)
    assert (landscape.width, landscape.height, landscape.top) == (384, 256, 0)  # height * 1.5 binds on 2:1
    portrait = jc.compute_crop_box(100, 300, 400, 200, 600)  # a tall test pano: the width term binds
    assert portrait.width == 200 and 0 <= portrait.left < 200 and 0 <= portrait.top <= 600 - portrait.height


def test_compute_crop_box_rounds_half_to_even():
    # width = round(503.21) = 503, so left = round(x - 251.5); x = 100 gives -151.5 -> -152 (even), not -151.
    for x, raw_left in ((100.0, -152), (100.5, -151), (101.0, -150), (250.25, -1)):
        b = jc.compute_crop_box(x, 512, 503.21, 2048, 1024)
        assert (b.width, b.height) == (503, 335)
        assert b.left == raw_left % 2048, x
    # a width that is itself a .5 tie: round(250.5) = 250, height round(166.67) = 167
    assert jc.compute_crop_box(1000, 500, 250.5, 2048, 1024).width == 250


def test_label_position_across_the_seam_and_after_a_shift():
    box = jc.compute_crop_box(20, 4096, 1000.0, 16384, 8192)  # left = 15904
    x, y = jc.label_position_in_crop(20, 4096, box, 16384)
    assert x == 20 - box.left + 16384 and y == 4096 - box.top
    top = jc.compute_crop_box(700, 20, 300, 1024, 512)
    assert jc.label_position_in_crop(700, 20, top, 1024) == (150.0, 20.0)
    fx, fy = jc.label_fraction_in_crop(700, 20, top, 1024)
    assert fx == pytest.approx(0.5) and fy == pytest.approx(20 / top.height) and 0 <= fy <= 1


def test_label_position_matches_every_mechanics_fixture():
    for c in json.loads((FIX / "mechanics.json").read_text()):
        got = jc.label_position_in_crop(c["pano_x"], c["pano_y"], _box(c["box"]), c["pano_width"])
        assert got == pytest.approx(tuple(c["label_in_crop"]), abs=1e-9), c["case"]


def test_segments_one_run_or_two_summing_to_the_width():
    assert jc.segments(jc.CropBox(100, 0, 50, 33, False), 1024) == [(100, 50, 0)]
    runs = jc.segments(jc.CropBox(1000, 0, 50, 33, False), 1024)
    assert runs == [(1000, 24, 0), (0, 26, 24)]
    assert sum(w for _, w, _ in runs) == 50
    assert jc.segments(jc.CropBox(974, 0, 50, 33, False), 1024) == [(974, 50, 0)]  # ends exactly at the edge


# ---------------------------------------------------------------------------------------------------------------------
# CropService
# ---------------------------------------------------------------------------------------------------------------------

def test_stored_size_caps_never_upscales_and_rounds_half_up():
    assert jc.stored_size(jc.CropBox(0, 0, 4096, 2731, False)) == (1440, 960)
    assert jc.stored_size(jc.CropBox(0, 0, 1200, 800, False)) == (1200, 800)
    assert jc.stored_size(jc.CropBox(0, 0, 3000, 2000, False)) == (1440, 960)
    # 2880 x 1925 at scale 0.5: height 962.5, which Java's Math.round takes up to 963 and round() to even, 962.
    assert jc.stored_size(jc.CropBox(0, 0, 2880, 1925, False)) == (1440, 963)
    assert round(1925 * 0.5) == 962  # the rint the backend does NOT use here


def test_cut_window_reproduces_every_mechanics_fixture_pixel_for_pixel():
    pano = _pano_rgb()
    assert pano.shape == (512, 1024, 3)
    cases = json.loads((FIX / "mechanics.json").read_text())
    assert {c["case"] for c in cases} >= {"interior", "seam_left", "seam_right", "pole_top", "pole_bottom"}
    for c in cases:
        box = jc.compute_crop_box(c["pano_x"], c["pano_y"], c["requested_width"], 1024, 512)
        assert box == _box(c["box"]), c["case"]
        expected = cv2.imread(str(FIX / "expected" / f"{c['case']}.png"))[..., ::-1]
        np.testing.assert_array_equal(jc.cut_window(pano, box), expected, err_msg=c["case"])


def test_sizing_end_to_end_fixture():
    e2e = json.loads((FIX / "sizing-v2.json").read_text())["e2e"]
    pano = _pano_rgb()
    win = jc.window_width(e2e["pano_y"], 1024, 512)
    assert win == pytest.approx(e2e["window_width"], abs=1e-6)
    box = jc.compute_crop_box(e2e["pano_x"], e2e["pano_y"], win, 1024, 512)
    assert box == _box(e2e["box"])
    expected = cv2.imread(str(FIX / "expected" / "sizing_e2e.png"))[..., ::-1]
    np.testing.assert_array_equal(jc.stored_crop(jc.cut_window(pano, box)), expected)


def test_cut_window_stitches_the_seam_left_to_right():
    pano = np.tile(np.arange(100, dtype=np.uint8)[None, :, None], (10, 1, 3))  # column index as the pixel value
    box = jc.CropBox(90, 2, 20, 5, False)
    cut = jc.cut_window(pano, box)
    assert cut.shape == (5, 20, 3)
    assert cut[0, :, 0].tolist() == list(range(90, 100)) + list(range(0, 10))


def test_stored_crop_downscales_to_the_cap_and_never_upscales():
    wide = np.zeros((2000, 3000, 3), np.uint8)
    assert jc.stored_crop(wide).shape == (960, 1440, 3)
    small = np.zeros((40, 60, 3), np.uint8)
    assert jc.stored_crop(small) is small


def test_window_for_puts_an_unshifted_label_at_the_centre():
    box, (fx, fy) = jc.window_for(512, 256, 1024, 512)
    assert not box.shifted and fx == pytest.approx(0.5, abs=0.01) and fy == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------------------------------------------------
# cut_job_crop and the data_generation hook helpers
# ---------------------------------------------------------------------------------------------------------------------

def test_cut_job_crop_info_and_skips():
    pano = _pano_rgb()
    crop, info = jc.cut_job_crop(pano, 512, 300, 1024, 512)
    box, (fx, fy) = jc.window_for(512, 300, 1024, 512)
    assert crop.shape == (box.height, box.width, 3)
    assert info["crop_half_px"] == box.width // 2 and info["clipped"] == int(box.shifted) == 0
    assert info["depth_m"] is None and info["tilt_applied"] == 0
    assert (info["left"], info["top"], info["width"], info["height"]) == tuple(box[:4])
    assert (info["fx"], info["fy"]) == (round(fx, 6), round(fy, 6))
    assert (info["stored_w"], info["stored_h"], info["pano_native_w"], info["pano_native_h"]) == (
        box.width, box.height, 1024, 512)
    assert set(info) >= set(jc.SIDECAR_COLUMNS) - {"label_uid"}

    _, top = jc.cut_job_crop(pano, 700, 2, 1024, 512)
    assert top["clipped"] == top["shifted"] == 1

    for y in (-1, 512, 600):
        with pytest.raises(jc.JobCropSkipped) as e:
            jc.cut_job_crop(pano, 10, y, 1024, 512)
        assert e.value.reason == "out_of_frame"
    with pytest.raises(jc.JobCropSkipped) as e:
        jc.cut_job_crop(pano, 10, 600, 2048, 1024)  # the dims check comes first, as in the backend
    assert e.value.reason == "dims_mismatch"
    with pytest.raises(jc.JobCropSkipped, match="dims_mismatch"):
        jc.cut_job_crop(pano, 10, 10, "1024", 500.0)
    # no recorded dims: cut anyway, against the stored file
    for missing in (None, float("nan"), ""):
        assert jc.cut_job_crop(pano, 10, 10, missing, missing)[1]["pano_native_w"] == 1024
    assert jc.cut_job_crop(pano, 10, 10, 1024.0, "512")[1]["pano_native_h"] == 512  # floats and strings compare


def test_read_native_pano_caches_one_entry(tmp_path, monkeypatch):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(p1), np.zeros((4, 8, 3), np.uint8))
    cv2.imwrite(str(p2), np.ones((4, 8, 3), np.uint8))
    reads = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda p: reads.append(p) or real(p))
    monkeypatch.setattr(jc, "_NATIVE_CACHE", {"path": None, "img": None})
    jc.read_native_pano(str(p1))
    jc.read_native_pano(str(p1))
    jc.read_native_pano(str(p2))
    assert reads == [str(p1), str(p2)]
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot decode"):
        jc.read_native_pano(str(tmp_path / "bad.jpg"))


def test_crop_for_row_returns_rgb_manifest_fields_and_appends_the_sidecar(tmp_path, monkeypatch):
    path = tmp_path / "pano.png"
    bgr = np.zeros((512, 1024, 3), np.uint8)
    bgr[..., 0] = 200  # blue in OpenCV order
    cv2.imwrite(str(path), bgr)
    monkeypatch.setattr(jc, "_NATIVE_CACHE", {"path": None, "img": None})
    side = tmp_path / jc.SIDECAR_FILE
    row = {"pano_x": 512, "pano_y": 300, "pano_width": 1024, "pano_height": 512}
    crop, fields = jc.crop_for_row(str(path), row, "c:1", str(side))
    assert tuple(crop[0, 0]) == (0, 0, 200)  # RGB
    assert set(fields) == set(jc.MANIFEST_KEYS)
    jc.crop_for_row(str(path), {**row, "pano_x": 10}, "c:2", str(side))
    jc.crop_for_row(str(path), {**row, "pano_x": 20}, "c:1", str(side))  # redone: appended again
    lines = side.read_text().splitlines()
    assert lines[0] == ",".join(jc.SIDECAR_COLUMNS) and len(lines) == 4


def test_compact_sidecar_keeps_the_last_row_of_ok_labels_only(tmp_path):
    side = tmp_path / jc.SIDECAR_FILE
    assert jc.compact_sidecar(str(side), ["x"]).empty  # no file yet
    info = {"left": 1, "top": 2, "width": 3, "height": 2, "shifted": 0, "fx": .5, "fy": .5, "stored_w": 3,
            "stored_h": 2, "pano_native_w": 10, "pano_native_h": 5, "rule_version": "v2"}
    jc.append_sidecar(str(side), "c:1", info)
    jc.append_sidecar(str(side), "c:2", info)
    jc.append_sidecar(str(side), "c:1", {**info, "left": 9})
    df = jc.compact_sidecar(str(side), ["c:1"])
    assert df.label_uid.tolist() == ["c:1"] and df.left.tolist() == [9]
    import pandas as pd
    assert pd.read_csv(side).to_dict("records") == df.to_dict("records")


def test_finish_run_counts(tmp_path, capsys):
    import pandas as pd
    info = {"left": 1, "top": 2, "width": 3, "height": 2, "fx": .5, "fy": .5, "stored_w": 3,
            "stored_h": 2, "pano_native_w": 10, "pano_native_h": 5, "rule_version": "v2"}
    side = str(tmp_path / jc.SIDECAR_FILE)
    jc.append_sidecar(side, "c:1", {**info, "shifted": 1})
    jc.append_sidecar(side, "c:2", {**info, "shifted": 0})
    manifest = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3", "c:4", "c:5"],
                             "status": ["ok", "ok", "out_of_frame", "dims_mismatch", "ok"]})
    s = jc.finish_run(str(tmp_path), manifest)
    assert s == {"job_crops": 2, "shifted": 1, "out_of_frame": 1, "dims_mismatch": 1, "missing_from_sidecar": 1}
    assert "job-window crops" in capsys.readouterr().out


def test_skipped_is_not_a_fatal_runtime_error():
    import data_generation as dg
    assert not dg.is_fatal(jc.JobCropSkipped("out_of_frame", "pano_y 9 outside [0, 8)"))
    assert math.isfinite(jc.window_width(0, 1024, 512))
