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

The instrument lives in ``analysis/pose_levelling.py`` (a port of sidewalk-panorama-tools' vertical-edge lean
estimator, levelling through the package's own ``tilt_matrix``, the exact transform the crop path applies).

Fixtures (tests/fixtures/pose_levelling/<source>/, licences in each ATTRIBUTION.txt): eight posed Bayonne FR Panoramax
pictures (Etalab 2.0) and six Richmond VA / Laurens IA Mapillary thumbnails (CC BY-SA 4.0), 2048 x 1024 each, with
the poses the Project Sidewalk database stores for them and their sha256s. The Mapillary six were chosen among 80
fetched panoramas with a DB pose as clear, strongly tilted cases from four contributors; on all 80 the same reading
was best on 52, and the median lean amplitude went 1.82 -> 0.95 degrees under it vs 1.98 / 2.32 / 2.73 under the
other three (docs/decisions.md; ``python analysis/pose_levelling.py`` reproduces it from
docs/pose-levelling-2026-10/mapillary-80.json). The test skips a source whose fixture directory is absent.
"""
import hashlib
import json
import os

import numpy as np
import pytest

import crop_pose
from analysis import pose_levelling as pl

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pose_levelling")
READINGS = pl.READINGS


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
                before, after = pl.measure(pl.load_gray(image_path(e, d)), e["camera_pitch"], e["camera_roll"])
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


def test_the_80_panorama_mapillary_record_gives_the_decisions_table():
    """docs/decisions.md's 80-panorama row, re-derived from the committed record (no imagery needed)."""
    with open(os.path.join(os.path.dirname(FIXTURES), "..", "..", "docs", "pose-levelling-2026-10",
                           "mapillary-80.json")) as f:
        rec = json.load(f)["panoramas"]
    assert len(rec) == 80 and {p["city"] for p in rec} == {"richmond-va", "laurens-ia"}
    med = {k: float(np.median([p["amp_after"][k] for p in rec])) for k in rec[0]["amp_after"]}
    best = {k: sum(min(p["amp_after"], key=p["amp_after"].get) == k for p in rec) for k in med}
    assert round(float(np.median([p["amp_before"] for p in rec])), 2) == 1.82
    assert {k: round(v, 2) for k, v in med.items()} == {"+1,+1": 1.98, "+1,-1": 2.73, "-1,+1": 0.95, "-1,-1": 2.32}
    assert best == {"+1,+1": 12, "+1,-1": 6, "-1,+1": 52, "-1,-1": 10}
    for p in rec:  # the DB pose is the parse of the rotation, same signs
        pitch, roll = crop_pose.mapillary_pitch_roll(p["computed_rotation"])
        assert abs(pitch - p["camera_pitch"]) < 1e-9 and abs(roll - p["camera_roll"]) < 1e-9
