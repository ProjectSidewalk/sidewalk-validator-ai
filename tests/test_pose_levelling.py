"""The sign of each provider's pose, measured on pixels (issue #72).

``crop_pose.pose_for_row`` turns a label's stored ``camera_pitch``/``camera_roll`` into the pose the geometry package's
``label_view`` corrects with. Whether that conversion is right is a property of the pixels, not of any provider's
documentation, so this test asks the panoramas themselves:

    resample the stored (rig-frame) panorama into the gravity frame with the package's own ``tilt_matrix`` at the
    candidate pose, and measure how much its near-vertical edges lean.

In a gravity-levelled panorama world-vertical edges are vertical columns, so the lean profile over bearing is flat. In
a rig-frame one they lean sinusoidally with the tilt. Levelling with the RIGHT pose flattens the profile; levelling
with a wrong sign adds tilt instead. All four sign readings of the stored (pitch, roll) are tried per provider; the
reading ``pose_for_row`` uses must be the one that flattens, and the other three must not.

The instrument is a port of the minimum of sidewalk-panorama-tools ``reports/scripts/tilt_frame.py`` (``lean_profile``,
the #54 F2 vertical-edge estimator) and ``reports/scripts/non_gsv_levelling.py`` (``fit_sinusoid``,
``level_with_pose``), except that the levelling goes through ``sidewalk_panorama_tools.geometry`` (``tilt_matrix``,
``rays_to_lonlat``), the exact transform the crop path applies, rather than the study's own ``pano_pose`` copy.

Fixtures (tests/fixtures/pose_levelling/<source>/, licences in each ATTRIBUTION.txt): eight posed Bayonne FR Panoramax
pictures (Etalab 2.0) and six Richmond VA / Laurens IA Mapillary thumbnails (CC BY-SA 4.0), 2048 x 1024 each, with
the poses the Project Sidewalk database stores for them and their sha256s. The Mapillary six were chosen among 80
fetched panoramas with a DB pose as clear, strongly tilted cases from four contributors; on all 80 the same reading
was best on 52, and the median lean amplitude went 1.82 -> 0.95 degrees under it vs 1.98 / 2.32 / 2.73 under the
other three (docs/decisions.md). The test skips a source whose fixture directory is absent.
"""
import hashlib
import json
import math
import os

import numpy as np
import pytest
from PIL import Image
from sidewalk_panorama_tools.geometry import (lonlat_to_pano_px, lonlat_to_ray, pano_px_to_lonlat, rays_to_lonlat,
                                              tilt_matrix)

import crop_pose

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pose_levelling")

# The instrument's parameters (tilt_frame.py's, with non_gsv_levelling.py's wider lean window: these tilts reach 20
# degrees, which a 12-degree window would truncate).
N_BINS = 12
EL_LO, EL_HI = -25.0, 35.0
MAG_PERCENTILE = 95.0
MAX_ABS_LEAN_DEG = 30.0
MIN_PIXELS = 50
MIN_DEFINED_BINS = 6
WORK_WIDTH = 2048

#: (pitch sign, roll sign) applied to the STORED columns before they reach tilt_matrix.
READINGS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def load_gray(path, width=WORK_WIDTH):
    """Grayscale float32 at the working width (2:1)."""
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (width, width // 2):
            im = im.resize((width, width // 2), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def _band_rows(h, el_lo, el_hi):
    top = int(np.floor((0.5 - el_hi / 180.0) * h))
    bottom = int(np.ceil((0.5 - el_lo / 180.0) * h))
    return max(1, top), min(h - 1, bottom)


def lean_profile(gray):
    """[(bin centre bearing, magnitude-weighted mean lean of strong near-vertical edges or None)] over 12 bins
    (tilt_frame.lean_profile: Sobel over the -25..+35 degree band, top 5% gradient magnitude)."""
    h, w = gray.shape
    top, bottom = _band_rows(h, EL_LO, EL_HI)
    rows = gray[top - 1:bottom + 1]
    dx = np.roll(rows, -1, axis=1) - np.roll(rows, 1, axis=1)
    gx = dx[:-2] + 2 * dx[1:-1] + dx[2:]
    dy = rows[2:] - rows[:-2]
    gy = np.roll(dy, 1, axis=1) + 2 * dy + np.roll(dy, -1, axis=1)
    mag = np.hypot(gx, gy)
    lean = -((np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0 - 90.0)
    keep = (mag > np.percentile(mag, MAG_PERCENTILE)) & (np.abs(lean) < MAX_ABS_LEAN_DEG) & (mag > 0)
    bins = (np.broadcast_to(np.arange(w), keep.shape) * N_BINS) // w
    out = []
    for b in range(N_BINS):
        sel = keep & (bins == b)
        centre = (b + 0.5) / N_BINS * 360.0 - 180.0
        out.append((centre, float(np.sum(mag[sel] * lean[sel]) / np.sum(mag[sel])) if sel.sum() >= MIN_PIXELS
                    else None))
    return out


def lean_amplitude(profile):
    """hypot(A, B) of the least-squares fit lean(b) = c0 + A cos b + B sin b (non_gsv_levelling.fit_sinusoid); None
    with fewer than 6 defined bins."""
    pts = [(c, v) for c, v in profile if v is not None]
    if len(pts) < MIN_DEFINED_BINS:
        return None
    b = np.radians([c for c, _ in pts])
    X = np.column_stack([np.ones_like(b), np.cos(b), np.sin(b)])
    (_, a, bb), *_ = np.linalg.lstsq(X, np.array([v for _, v in pts]), rcond=None)
    return float(math.hypot(a, bb))


def level(gray, pitch, roll):
    """The rig-frame panorama resampled into the gravity frame at the package-sign pose (pitch, roll), over the rows
    the lean band reads: each output pixel's level-frame ray goes through ``tilt_matrix(pitch, roll, beta=1)`` into
    the rig frame, exactly as ``label_view`` + ``render_view`` map a crop pixel, and is sampled bilinearly."""
    h, w = gray.shape
    top, bottom = _band_rows(h, EL_LO - 10, EL_HI + 10)
    ys, xs = np.mgrid[top:bottom, 0:w]
    lon, lat = pano_px_to_lonlat(xs + 0.5, ys + 0.5, w, h)
    rlon, rlat = rays_to_lonlat(lonlat_to_ray(lon, lat), tilt_matrix(pitch, roll, beta=1.0))
    sx, sy = lonlat_to_pano_px(rlon, rlat, w, h)
    sx = sx - 0.5
    sy = np.clip(sy - 0.5, 0, h - 1.000001)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = (sx - x0).astype(np.float32), (sy - y0).astype(np.float32)
    x0 %= w
    x1, y1 = (x0 + 1) % w, np.minimum(y0 + 1, h - 1)
    v = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x1] * fx * (1 - fy)
         + gray[y1, x0] * (1 - fx) * fy + gray[y1, x1] * fx * fy)
    out = gray.copy()
    out[top:bottom] = v
    return out


def load_fixtures(source):
    """(poses.json entries, directory); skips the test when the source's fixtures are absent."""
    d = os.path.join(FIXTURES, source)
    if not os.path.exists(os.path.join(d, "poses.json")):
        pytest.skip(f"no {source} levelling fixtures in {d}")
    with open(os.path.join(d, "poses.json")) as f:
        return json.load(f)["panoramas"], d


def image_path(entry, d):
    """The fixture file for ``entry``, after checking its bytes are the ones poses.json records."""
    name = entry["pano_id"] + (".sd.jpg" if os.path.exists(os.path.join(d, entry["pano_id"] + ".sd.jpg")) else ".jpg")
    path = os.path.join(d, name)
    with open(path, "rb") as f:
        body = f.read()
    assert (hashlib.sha256(body).hexdigest(), len(body)) == (entry["sha256"], entry["bytes"]), \
        f"{path} is not the file poses.json records"
    return path


@pytest.fixture(scope="module")
def measured():
    """{source: [(entry, amp_before, {reading: amp_after}, production_pose)]}, measured once per module."""
    cache = {}

    def get(source):
        if source not in cache:
            entries, d = load_fixtures(source)
            rows = []
            for e in entries:
                gray = load_gray(image_path(e, d))
                before = lean_amplitude(lean_profile(gray))
                after = {(sp, sr): lean_amplitude(lean_profile(level(gray, sp * e["camera_pitch"],
                                                                     sr * e["camera_roll"])))
                         for sp, sr in READINGS}
                pose = crop_pose.pose_for_row({"pano_source": source, "pano_id": e["pano_id"], "author": "human",
                                               "camera_pitch": e["camera_pitch"],
                                               "camera_roll": e["camera_roll"]}, gsv_beta=0.95)
                rows.append((e, before, after, pose))
            cache[source] = rows
        return cache[source]
    return get


@pytest.mark.parametrize("source", ["mapillary", "panoramax"])
def test_exactly_one_sign_reading_levels_the_pixels_and_it_is_the_one_pose_for_row_uses(measured, source):
    rows = measured(source)
    assert all(b is not None and all(v is not None for v in a.values()) for _, b, a, _ in rows)
    # The reading pose_for_row applies, read off the production code rather than restated here.
    used = set()
    for e, _, _, pose in rows:
        assert pose.pose_source == f"{source}_db" and pose.beta == crop_pose.SOURCE_BETA[source]
        used.add((round(pose.camera_pitch / e["camera_pitch"]), round(pose.camera_roll / e["camera_roll"])))
    assert len(used) == 1
    chosen = used.pop()
    before = np.median([b for _, b, _, _ in rows])
    after = {r: float(np.median([a[r] for _, _, a, _ in rows])) for r in READINGS}
    best_counts = {r: sum(min(a, key=a.get) == r for _, _, a, _ in rows) for r in READINGS}
    # The chosen reading flattens: the median lean amplitude at least halves, and it is the best reading on every
    # picture but at most one.
    assert after[chosen] < 0.5 * before, (source, before, after)
    assert best_counts[chosen] >= len(rows) - 1, (source, best_counts)
    # No other reading comes close: each leaves at least twice the chosen reading's median, and none flattens.
    for r in READINGS:
        if r != chosen:
            assert after[r] > 2 * after[chosen] and after[r] > 0.5 * before, (source, r, before, after)
    # Both website viewers store pitch positive nose UP and roll in the package's sign (crop_pose docstring).
    assert chosen == (-1, 1)


def test_mapillary_db_pose_is_the_parse_of_computed_rotation():
    """The website's DB columns for a Mapillary pano are crop_pose.mapillary_pitch_roll of its Graph API rotation (the
    same numbers, same signs), so ``mapillary_db`` and ``mapillary_api`` rows get the same conversion."""
    entries, _ = load_fixtures("mapillary")
    for e in entries:
        p, r = crop_pose.mapillary_pitch_roll(e["computed_rotation"])
        assert p == pytest.approx(e["camera_pitch"], abs=1e-9) and r == pytest.approx(e["camera_roll"], abs=1e-9)
