"""Temporal validation, Stage 0 (issue #17): pair labelled panoramas across capture dates, re-project each label into
the partner panorama, and measure how far that lands from the partner's own human clicks.

The design and the gates are in ``docs/temporal-validation-2026-11.md``; every threshold is in
``docs/temporal-validation-2026-11/gates.json`` (proposed until Jon confirms them). This module is CPU only and
reads no imagery: it needs rawLabels CSVs, camera positions, the round-3 ``split.csv`` files and the public city
listing.

Pipeline (three subcommands, each writing into a run directory outside the repo, except the last one's aggregates)::

    # 1. One camera position per panorama: inverted from the public rawLabels (PREVIEW), or from the DB pull
    #    (analysis/pull_pano_cameras.sh; the only source a gate verdict accepts)
    python analysis/temporal_pairs.py positions --rawlabels CACHE --listing cities.json --out RUN/panos.csv
    python analysis/temporal_pairs.py positions --rawlabels CACHE --listing cities.json \\
        --db-panos DB/panos.csv --cityparams SidewalkWebpage/conf/cityparams.conf --out RUN/panos.csv

    # 2. Anchors, partner panoramas, re-projection and geometry QC (per-label files; they stay in RUN)
    python analysis/temporal_pairs.py census --rawlabels CACHE --listing cities.json --panos RUN/panos.csv \\
        --round3 ROUND3 --out RUN

    # 3. The label-to-label residual, the click floor, the census tables and the Gate 0 check (aggregates only)
    python analysis/temporal_pairs.py residual --run RUN --listing cities.json \\
        --out docs/temporal-validation-2026-11/preview

Every output of ``residual`` says PREVIEW or GATE in its first line, from the camera positions it was given, and
:func:`gate0` refuses a verdict unless every position came from the database.

Frames and units (the same as ``sidewalk_panorama_tools.geometry`` and the site):

* A stored click ``(pano_x, pano_y)`` on a ``pano_width x pano_height`` panorama is, in the level frame,
  ``lon = (pano_x / pano_width - 0.5) * 360`` degrees clockwise from the camera heading and
  ``lat = (0.5 - pano_y / pano_height) * 180`` degrees above the horizon. Its world bearing is
  ``camera_heading + lon``; its depression is ``-lat``.
* Positions are on the sphere the site uses, R = 6,371 km (``CommonUtils.calculateDestination``,
  SidewalkWebpage evolution 352). Offsets between two panoramas are east/north metres about the pair's mid-latitude
  (:func:`enu_delta`), never about a city-wide reference, so a city's extent cannot scale them.
* Ranges on the ground come from the site's own fitted estimator (:func:`blend_range`); view pixels are those of
  the training crop's 2048 px, 90 degree view (focal length 1,024 px).
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generation as dg  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Constants. Geometry constants carry their provenance; gate thresholds are NOT here (gates.json).
# ---------------------------------------------------------------------------------------------------------------------

TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
PRESENCE_TYPES = ("CurbRamp", "NoCurbRamp", "Crosswalk")  # the types with a usable negative-control pool

# The site's range estimator (approximation3): SidewalkWebpage PanoDataService.LatLngEstimation, evolution 352's
# constants CTE, and sidewalk_panorama_tools.ps_math (BLEND_CAMERA_HEIGHT_M, BLEND_DEG, DIST_CAP_M; ps_math exists
# from panorama-tools v1.2.0, this repo pins v1.0.0, so they are copied; a test checks parity when ps_math imports).
CAMERA_HEIGHT_M = 2.341219672825709
BLEND_DEG = 11.25
DIST_CAP_M = 50.0
EARTH_R_M = 6371000.0  # CommonUtils.EARTH_RADIUS_KM * 1000; 352.sql
M_PER_DEG = EARTH_R_M * math.pi / 180.0  # 111,194.93 m per degree of latitude on that sphere

# The training view (data_generation.py): 2048 px over 90 degrees, so a focal length of 1,024 px.
FOCAL_PX = (dg.VIEW_SIZE / 2) / math.tan(math.radians(dg.VIEW_FOV_DEG / 2))
assert abs(FOCAL_PX - 1024.0) < 1e-9, FOCAL_PX

# Pairing.
B_SEARCH_M = 5.0          # partners searched within this baseline; the primary cap (3 or 5 m) is Gate 0's call
LONG_MONTHS = 24          # long_newer: B captured at least this many months after A
CONTROL_MONTHS = (1, 3)   # control: 1 <= |dt| <= 3 months; dt = 0 is same_month, used only in the residual table
AGREE_TOL_M = 0.05        # inversion_agreement: every label on a panorama must invert to within this of the median

# Geometry QC (in this order; each rejection is counted under its first failing rule).
MIN_DEPRESSION_DEG = 5.0  # near_horizon / b_near_horizon: the estimator's linear tail is a bounded guess above this
B_RANGE_M = (3.0, 25.0)   # b_range: the re-projected point's range from B
RANGE_RATIO = (0.6, 1.6)  # range_ratio: d_B / r_A
MAX_PARALLAX_DEG = 25.0   # parallax: the angle A-P-B at the re-projected ground point
QC_ORDER = ("no_camera_position", "photosphere", "near_horizon", "baseline", "b_range", "range_ratio",
            "b_near_horizon", "parallax")

# Residual matching.
RESIDUAL_WINDOW_DEG = 30.0  # B's same-type clicks within this of the re-projected point are candidates
FLOOR_WINDOW_DEG = 15.0     # click floor: two users' same-type clicks on one panorama within this
CONTRA_WINDOW_DEG = 15.0    # a contradicting label on B (CurbRamp vs NoCurbRamp) within this
SEED = 20261112             # no sampling in Stage 0; recorded for the later stages' jitter draws

GATES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "docs", "temporal-validation-2026-11", "gates.json")
PREVIEW_BANNER = ("PREVIEW (not the gate): camera positions inverted from public rawLabels, not read from the "
                  "database; no gate verdict is given on these numbers.")
GATE_BANNER = "GATE: camera positions read from the database (pano_data)."

LABEL_COLUMNS = ["label_id", "user_id", "pano_id", "pano_source", "label_type", "severity", "tags", "validations",
                 "image_capture_date", "pano_x", "pano_y", "pano_width", "pano_height", "camera_heading",
                 "camera_pitch", "camera_roll", "latitude", "longitude"]


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------------

def capture_month(s):
    """A capture date (``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``) as a month index ``year * 12 + month - 1``; a bare
    year has no month and gives None, as does anything unparseable.

    >>> capture_month("2019-07"), capture_month("2019-07-23"), capture_month("2019"), capture_month(float("nan"))
    (24234, 24234, None, None)
    """
    if not isinstance(s, str):
        return None
    parts = s.strip().split("-")
    if len(parts) < 2:
        return None
    try:
        y, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (1 <= m <= 12 and 1900 <= y <= 2100):
        return None
    return y * 12 + m - 1


def is_photosphere(pano_id):
    """True for a user photosphere or anything that is not a 22-character GSV panorama id.

    >>> is_photosphere("8AsY4nlOJQAo8P5YYaEjzA"), is_photosphere("CAoSLEFGMVFpcE1"), is_photosphere(None)
    (False, True, True)
    """
    return not isinstance(pano_id, str) or pano_id.startswith("CAoS") or len(pano_id) != 22


def wrap180(deg):
    """Angles wrapped into (-180, 180].

    >>> wrap180(np.array([180.0, -180.0, 190.0, 360.0])).tolist()
    [180.0, 180.0, -170.0, 0.0]
    """
    return 180.0 - np.mod(180.0 - np.asarray(deg, float), 360.0)


def label_lonlat(pano_x, pano_y, pano_width, pano_height):
    """A stored click as level-frame ``(lon, lat)`` degrees: lon clockwise from the camera heading, lat above the
    horizon (the inverse of the site's ``calculatePanoXYFromPov``; ``geometry.pano_px_to_lonlat`` without the wrap).

    >>> [round(v, 9) for v in label_lonlat(8192, 4096 + 4096 * 20 / 90, 16384, 8192)]
    [0.0, -20.0]
    """
    lon = (np.asarray(pano_x, float) / np.asarray(pano_width, float) - 0.5) * 360.0
    lat = (0.5 - np.asarray(pano_y, float) / np.asarray(pano_height, float)) * 180.0
    return (float(lon), float(lat)) if np.ndim(lon) == 0 else (lon, lat)


def blend_range(dep_deg, h=CAMERA_HEIGHT_M, blend_deg=BLEND_DEG, cap_m=DIST_CAP_M):
    """The site's ground range (m) for a click ``dep_deg`` below the horizon (approximation3).

    Flat-ground cotangent ``h / tan(dep)`` at or steeper than ``blend_deg``; below it a straight line with the
    cotangent's value and slope at ``blend_deg`` (slope ``-h (pi/180) / sin^2(blend)`` metres per degree), held at
    its horizon value above the horizon, clamped to ``[0, cap_m]``. The same function as
    ``ps_math.predict_blend_distance`` and SidewalkWebpage ``PanoDataService.estimateDistanceFromPanoM``.
    Its cited 0.41 m median error is for ground-contact clicks against provider depth, what it was fitted on.

    >>> round(float(blend_range(20.0)), 3), round(float(blend_range(11.25)), 3), round(float(blend_range(0.0)), 3)
    (6.432, 11.77, 23.848)
    """
    dep = np.asarray(dep_deg, float)
    a = math.radians(blend_deg)
    value_at_blend = h / math.tan(a)
    slope = -h * (math.pi / 180.0) / math.sin(a) ** 2
    cot = h / np.tan(np.radians(np.maximum(dep, 1e-9)))
    tail = value_at_blend + slope * (np.maximum(dep, 0.0) - blend_deg)
    return np.clip(np.where(dep >= blend_deg, cot, tail), 0.0, cap_m)


def blend_depression(range_m, h=CAMERA_HEIGHT_M, blend_deg=BLEND_DEG):
    """The inverse of :func:`blend_range`: the depression (degrees, positive down) at which the site's estimator
    gives ``range_m``; 0 (the horizon) for ranges at or beyond the tail's horizon value (about 23.85 m).

    Re-projecting into B with the *same* estimator A's range came from keeps the identity exact: a partner at A's
    own position and heading gets A's click back, at any depression (the flat-ground ``atan(h / d)`` would not,
    below 11.25 degrees, where the fitted tail and the cotangent part ways).

    >>> round(float(blend_depression(blend_range(20.0))), 9), round(float(blend_depression(blend_range(7.0))), 9)
    (20.0, 7.0)
    """
    d = np.asarray(range_m, float)
    a = math.radians(blend_deg)
    value_at_blend = h / math.tan(a)
    slope = -h * (math.pi / 180.0) / math.sin(a) ** 2
    with np.errstate(divide="ignore"):
        cot_inv = np.degrees(np.arctan(h / np.maximum(d, 1e-12)))
    tail_inv = blend_deg + (d - value_at_blend) / slope
    return np.maximum(np.where(d <= value_at_blend, cot_inv, tail_inv), 0.0)


def destination(lat, lng, dist_m, bearing_deg, radius_m=EARTH_R_M):
    """The point ``dist_m`` from ``(lat, lng)`` along ``bearing_deg`` on a sphere: the site's
    ``CommonUtils.calculateDestination`` (evolution 352's SQL port of it), vectorised.

    >>> [round(v, 9) for v in destination(47.6, -122.3, 0.0, 90.0)]
    [47.6, -122.3]
    """
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lng, float))
    ad = np.asarray(dist_m, float) / radius_m
    b = np.radians(np.asarray(bearing_deg, float))
    la2 = np.arcsin(np.clip(np.sin(la) * np.cos(ad) + np.cos(la) * np.sin(ad) * np.cos(b), -1.0, 1.0))
    lo2 = lo + np.arctan2(np.sin(b) * np.sin(ad) * np.cos(la), np.cos(ad) - np.sin(la) * np.sin(la2))
    out = np.degrees(la2), np.degrees(lo2)
    return (float(out[0]), float(out[1])) if np.ndim(out[0]) == 0 else out


def invert_camera(label_lat, label_lng, pano_x, pano_y, pano_width, pano_height, camera_heading, iterations=3):
    """The camera position the site must have used to place a label at ``(label_lat, label_lng)``.

    Since evolution 352 (2026-08-11) every stored ``approximation2`` position was recomputed as
    ``destination(camera, blend_range(depression), camera_heading + lon)`` from the stored click, and live
    submissions use the same estimator, so the camera is the point from which that destination lands on the label.
    The first guess walks back along the reverse bearing; each iteration then moves the guess by the miss of the
    forward destination. Great-circle bearings change by a few nano-degrees over 50 m, so one iteration already
    lands within a micrometre; three are kept as margin. Depth-era labels (``computation_method = 'depth'``) were
    not recomputed and invert to the wrong place; :func:`panos_from_rawlabels` catches them on panoramas with two
    or more labels.

    >>> lat, lng = destination(47.6, -122.3, float(blend_range(15.0)), 30.0 + 10.0)
    >>> [round(v, 9) for v in invert_camera(lat, lng, (10 / 360 + 0.5) * 16384, 4096 + 4096 * 15 / 90,
    ...                                     16384, 8192, 30.0)]
    [47.6, -122.3]
    """
    lon, lat = label_lonlat(pano_x, pano_y, pano_width, pano_height)
    d = blend_range(-np.asarray(lat, float))
    brg = np.asarray(camera_heading, float) + np.asarray(lon, float)
    llat, llng = np.asarray(label_lat, float), np.asarray(label_lng, float)
    clat, clng = destination(llat, llng, d, brg + 180.0)
    for _ in range(iterations):
        flat, flng = destination(clat, clng, d, brg)
        clat, clng = clat - (flat - llat), clng - (flng - llng)
    return (float(clat), float(clng)) if np.ndim(clat) == 0 else (clat, clng)


def enu_delta(lat_a, lng_a, lat_b, lng_b):
    """B's position relative to A as ``(east_m, north_m)`` on the R = 6,371 km sphere.

    Longitude is scaled by the cosine of the pair's own mid-latitude, never by a city-wide or per-label one: at
    60 degrees north a degree of longitude is half a degree of latitude, so the wrong cosine misplaces a 5 m
    baseline by metres. Over a few metres the error of this local projection is of order (d / R)^2, nothing.

    >>> lat_b, lng_b = destination(65.0, 10.0, 5.0, 90.0)
    >>> [round(float(v), 4) + 0.0 for v in enu_delta(65.0, 10.0, lat_b, lng_b)]
    [5.0, 0.0]
    """
    lat_a, lng_a, lat_b, lng_b = (np.asarray(v, float) for v in (lat_a, lng_a, lat_b, lng_b))
    mid = np.radians((lat_a + lat_b) / 2.0)
    east = wrap180(lng_b - lng_a) * M_PER_DEG * np.cos(mid)
    north = (lat_b - lat_a) * M_PER_DEG
    return east, north


def angular_sep_deg(lon1, lat1, lon2, lat2):
    """Great-circle angle (degrees) between two view directions, haversine form (stable for small angles).

    >>> round(float(angular_sep_deg(0.0, -20.0, 0.0, -25.0)), 9), round(float(angular_sep_deg(170.0, 0, -170.0, 0)), 9)
    (5.0, 20.0)
    """
    p1, p2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dl = np.radians(wrap180(np.asarray(lon2, float) - np.asarray(lon1, float)))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def view_px(angle_deg):
    """An angle off the view centre as pixels of the 2048 px, 90 degree training view: ``1024 tan(angle)``.

    >>> round(float(view_px(45.0)), 6)
    1024.0
    """
    return FOCAL_PX * np.tan(np.radians(np.asarray(angle_deg, float)))


def reproject(lon_a, lat_a, heading_a, heading_b, d_east, d_north, h=CAMERA_HEIGHT_M):
    """Re-project a click on A into partner B. All arguments broadcast; returns a dict of arrays.

    ``lon_a, lat_a`` are the click's level-frame angles on A (:func:`label_lonlat`), ``heading_a/heading_b`` the
    two camera headings, ``(d_east, d_north)`` B's position relative to A in metres (:func:`enu_delta`).

    Derivation (flat ground, both cameras at the fitted height ``h``):

    1. A's range ``r_A = blend_range(-lat_a)`` along bearing ``theta_A = heading_a + lon_a`` puts the ground point
       at ``P = r_A (sin theta_A, cos theta_A)`` east/north of A.
    2. From B, ``v = P - (d_east, d_north)``; its length is B's range ``d_B`` and its bearing
       ``theta_B = atan2(v_E, v_N)``, so the point sits ``lon_B = wrap180(theta_B - heading_b)`` from B's centre
       column.
    3. B's elevation is the depression at which the same estimator gives ``d_B``:
       ``lat_B = -blend_depression(d_B)``. (The plan's ``-atan(h / d_B)`` breaks the identity below 11.25 degrees,
       where the fitted tail is not the cotangent; see :func:`blend_depression`.)
    4. The level-frame normalised point on B is ``x = lon_B / 360 + 0.5``, ``y = 0.5 - lat_B / 180``; with B's
       ``camera_pitch`` it is what ``data_generation.render_label_view`` takes.

    Also returned: ``parallax_deg``, the angle A-P-B (how differently the two cameras see the feature).

    >>> r = reproject(0.0, -20.0, 0.0, 0.0, 0.0, 0.0)          # B = A: the click comes back
    >>> round(float(r["lon_b"]), 9), round(float(r["lat_b"]), 9), round(float(r["parallax_deg"]), 6)
    (0.0, -20.0, 0.0)
    >>> r = reproject(0.0, -20.0, 0.0, 90.0, 0.0, 0.0)         # B turned 90 degrees right: the point is 90 to its left
    >>> round(float(r["lon_b"]), 9)
    -90.0
    """
    lon_a, lat_a = np.asarray(lon_a, float), np.asarray(lat_a, float)
    r_a = blend_range(-lat_a, h=h)
    theta_a = np.radians(np.asarray(heading_a, float) + lon_a)
    pe, pn = r_a * np.sin(theta_a), r_a * np.cos(theta_a)
    ve, vn = pe - np.asarray(d_east, float), pn - np.asarray(d_north, float)
    d_b = np.hypot(ve, vn)
    theta_b = np.degrees(np.arctan2(ve, vn))
    lon_b = wrap180(theta_b - np.asarray(heading_b, float))
    lat_b = -blend_depression(d_b, h=h)
    # Parallax at P: the angle between P->A = -P and P->B = -v.
    dot = pe * ve + pn * vn
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_par = np.clip(dot / (r_a * d_b), -1.0, 1.0)
    parallax = np.where((r_a > 0) & (d_b > 0), np.degrees(np.arccos(cos_par)), np.nan)
    return {"r_a": r_a, "theta_a": np.degrees(theta_a), "p_east": pe, "p_north": pn, "d_b": d_b,
            "theta_b": theta_b, "lon_b": lon_b, "lat_b": lat_b, "x_norm_b": lon_b / 360.0 + 0.5,
            "y_norm_b": 0.5 - lat_b / 180.0, "parallax_deg": parallax}


def qc_reasons(df):
    """The first failing geometry rule per row (``"ok"`` when none), in :data:`QC_ORDER`.

    Needs columns ``has_camera``, ``photosphere`` (either panorama), ``dep_a``, ``baseline_m``, ``d_b``, ``r_a``,
    ``dep_b`` and ``parallax_deg``. Two checks of the first plan are deliberately absent: the heading self-check
    and the website-position range check are circular (the label's lat/lng was computed from the same click and
    heading), so they cannot catch an error; the independent range check needs a depth map and comes in Stage 2.
    """
    rules = [
        ("no_camera_position", ~df["has_camera"].astype(bool)),
        ("photosphere", df["photosphere"].astype(bool)),
        ("near_horizon", ~(df["dep_a"] >= MIN_DEPRESSION_DEG)),
        ("baseline", ~(df["baseline_m"] <= B_SEARCH_M)),
        ("b_range", ~((df["d_b"] >= B_RANGE_M[0]) & (df["d_b"] <= B_RANGE_M[1]))),
        ("range_ratio", ~((df["d_b"] / df["r_a"] >= RANGE_RATIO[0]) & (df["d_b"] / df["r_a"] <= RANGE_RATIO[1]))),
        ("b_near_horizon", ~(df["dep_b"] >= MIN_DEPRESSION_DEG)),
        ("parallax", ~(df["parallax_deg"] <= MAX_PARALLAX_DEG)),
    ]
    assert tuple(r for r, _ in rules) == QC_ORDER
    out = pd.Series("ok", index=df.index, dtype=object)
    undecided = pd.Series(True, index=df.index)
    for reason, bad in rules:
        hit = undecided & bad.fillna(True)
        out[hit] = reason
        undecided &= ~hit
    return out


def geometry_qc(row):
    """:func:`qc_reasons` for one row (a dict or Series): ``"ok"`` or the first failing rule's name."""
    return qc_reasons(pd.DataFrame([dict(row)])).iloc[0]


def arm_of(dt_months):
    """The pairing arm of a capture gap ``dt = month_B - month_A``: ``long_newer`` (dt >= 24), ``control``
    (1 <= |dt| <= 3), ``same_month`` (dt = 0), else None (not used in the pilot).

    >>> arm_of(np.array([0, 1, -3, 4, 24, -30])).tolist()
    ['same_month', 'control', 'control', None, 'long_newer', None]
    """
    dt = np.asarray(dt_months, float)
    lo, hi = CONTROL_MONTHS
    out = np.select([dt >= LONG_MONTHS, (np.abs(dt) >= lo) & (np.abs(dt) <= hi), dt == 0],
                    ["long_newer", "control", "same_month"], default="")
    return np.where(out == "", None, out).astype(object)


def user_key(user_id):
    """A short one-way key for a user id, so the run files can tell two users apart without holding their ids."""
    return hashlib.sha256(f"temporal-pairs:{user_id}".encode()).hexdigest()[:12]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Labels and camera positions
# ---------------------------------------------------------------------------------------------------------------------

def prepare_labels(raw):
    """rawLabels rows (one or more cities, with a ``city`` column) -> one row per usable GSV label, and drop counts.

    Keeps every label type (more labels per panorama make the inversion check stronger). Adds ``label_uid``
    (``city:label_id``, asserted unique), ``lon``/``lat`` (level frame), ``capture_month``, ``author``,
    ``vote_class`` (the margin rule of ``data_generation.eligible_labels``: ``correct`` / ``incorrect`` / ``""``)
    and ``user_key``; the raw ``user_id`` is dropped.
    """
    drops = Counter()
    df = raw.copy()
    df["label_uid"] = df["city"].astype(str) + ":" + df["label_id"].astype(str)
    assert df["label_uid"].is_unique, "label_uid (city:label_id) is not unique"
    gsv = df["pano_source"].astype(str).str.strip().str.lower() == "gsv"
    drops["non_gsv"] = int((~gsv).sum())
    df = df[gsv]
    keep, why = dg.valid_geometry(df)
    drops.update(why)
    df = df[keep].copy()
    for c in ("pano_x", "pano_y", "pano_width", "pano_height", "camera_heading", "camera_pitch", "latitude",
              "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lon"], df["lat"] = label_lonlat(df["pano_x"].to_numpy(), df["pano_y"].to_numpy(),
                                        df["pano_width"].to_numpy(), df["pano_height"].to_numpy())
    df["capture_month"] = df["image_capture_date"].map(capture_month).astype("Int64")
    df["author"] = df["user_id"].map(dg.label_author)
    df["user_key"] = df["user_id"].map(user_key)
    counts = df["validations"].apply(dg.human_vote_counts)
    net = np.array([a - d for a, d in counts]) if len(df) else np.zeros(0)
    df["vote_class"] = np.where(net > 2, "correct", np.where(-net > 1, "incorrect", ""))
    df["photosphere"] = df["pano_id"].map(is_photosphere)
    return df.drop(columns=["user_id"]), drops


def panos_from_rawlabels(labels, tol_m=AGREE_TOL_M):
    """One camera position per ``(city, pano_id)`` by inverting the site's estimator on each label
    (:func:`invert_camera`): ``(panos, stats)``.

    ``panos`` columns: ``city, pano_id, lat, lng, camera_heading, camera_pitch, capture_month, pano_width, n_labels,
    agreement_m, position_source`` (``"inverted"``). ``agreement_m`` is the largest distance of any label's inverted
    camera from the panorama's median position (NaN with one label: unchecked). Dropped and counted per city:
    panoramas whose labels disagree by more than ``tol_m`` (depth-era positions, or a camera that moved), and
    panoramas whose rows disagree on heading, capture month or width. ``stats`` is ``{city: {...}}``.

    The ``inversion_agreement`` check is bookkeeping, not a test of the heading: label positions and clicks share
    the heading, so a wrong heading inverts consistently.
    """
    df = labels[np.isfinite(labels["latitude"]) & np.isfinite(labels["longitude"])].copy()
    clat, clng = invert_camera(df["latitude"].to_numpy(), df["longitude"].to_numpy(), df["pano_x"].to_numpy(),
                               df["pano_y"].to_numpy(), df["pano_width"].to_numpy(), df["pano_height"].to_numpy(),
                               df["camera_heading"].to_numpy())
    df["clat"], df["clng"] = clat, clng
    df = df[np.isfinite(df["clat"]) & np.isfinite(df["clng"])]  # e.g. a blank camera_heading
    g = df.groupby(["city", "pano_id"], sort=True)
    panos = g.agg(lat=("clat", "median"), lng=("clng", "median"), camera_heading=("camera_heading", "first"),
                  camera_pitch=("camera_pitch", "first"), capture_month=("capture_month", "first"),
                  pano_width=("pano_width", "first"), n_labels=("label_uid", "size"),
                  n_heading=("camera_heading", "nunique"), n_month=("capture_month", "nunique"),
                  n_width=("pano_width", "nunique")).reset_index()
    df = df.merge(panos[["city", "pano_id", "lat", "lng"]].rename(columns={"lat": "mlat", "lng": "mlng"}),
                  on=["city", "pano_id"], validate="many_to_one")
    e, n = enu_delta(df["mlat"].to_numpy(), df["mlng"].to_numpy(), df["clat"].to_numpy(), df["clng"].to_numpy())
    df["off_m"] = np.hypot(e, n)
    agree = df.groupby(["city", "pano_id"])["off_m"].max().rename("agreement_m").reset_index()
    panos = panos.merge(agree, on=["city", "pano_id"], validate="one_to_one")
    panos.loc[panos["n_labels"] < 2, "agreement_m"] = np.nan
    panos["drop"] = np.select(
        [panos["agreement_m"] > tol_m, panos["n_heading"] > 1, panos["n_month"] > 1, panos["n_width"] > 1],
        ["inversion_disagrees", "two_headings", "two_months", "two_widths"], default="")
    stats = {}
    for city, p in panos.groupby("city"):
        multi = p["agreement_m"].dropna()
        stats[city] = {
            "panos": int(len(p)), "multi_label_panos": int(len(multi)),
            "agreement_m_p50": _q(multi, 0.5), "agreement_m_p99": _q(multi, 0.99),
            **{f"dropped_{k}": int(v) for k, v in p.loc[p["drop"] != "", "drop"].value_counts().items()},
            "share_multi_label_dropped": (round(float((multi > tol_m).mean()), 4) if len(multi) else None),
        }
    panos = panos[panos["drop"] == ""].drop(columns=["drop", "n_heading", "n_month", "n_width"])
    panos["position_source"] = "inverted"
    return panos.reset_index(drop=True), stats


def panos_from_db(csv_path, schema_city):
    """Camera positions from ``analysis/pull_pano_cameras.sh`` (MODE=panos), ``position_source = "db"``.

    ``schema_city`` maps schema -> city_id (``analysis/gt_rounds.city_ids``). GSV rows with a position, heading
    and a parseable capture month are kept.
    """
    db = pd.read_csv(csv_path, dtype={"pano_id": str, "schema": str, "capture_date": str})
    unknown = sorted(set(db["schema"]) - set(schema_city))
    if unknown:
        raise ValueError(f"schemas without a city_id: {unknown}")
    db["city"] = db["schema"].map(schema_city)
    db = db[db["source"].astype(str).str.lower() == "gsv"]
    out = pd.DataFrame({
        "city": db["city"], "pano_id": db["pano_id"], "lat": pd.to_numeric(db["lat"], errors="coerce"),
        "lng": pd.to_numeric(db["lng"], errors="coerce"),
        "camera_heading": pd.to_numeric(db["camera_heading"], errors="coerce"),
        "camera_pitch": pd.to_numeric(db["camera_pitch"], errors="coerce"),
        "capture_month": db["capture_date"].map(capture_month).astype("Int64"),
        "pano_width": pd.to_numeric(db["width"], errors="coerce"), "n_labels": db["n_labels"],
        "agreement_m": np.nan, "position_source": "db"})
    ok = np.isfinite(out["lat"]) & np.isfinite(out["lng"]) & np.isfinite(out["camera_heading"])
    out = out[ok & out["capture_month"].notna()]
    assert not out.duplicated(["city", "pano_id"]).any(), "pano_id twice in one schema"
    return out.reset_index(drop=True)


def compare_positions(inverted, db):
    """Per city: how far the inverted camera position is from the database's, on panoramas in both.

    Columns ``city, n, median_m, p99_m, share_over_tol, heading_diff_p99_deg``. On ``approximation3`` panoramas the
    two should agree within centimetres; if they do not, the preview was wrong and the doc must say so.
    """
    m = inverted.merge(db, on=["city", "pano_id"], suffixes=("_inv", "_db"), validate="one_to_one")
    e, n = enu_delta(m["lat_db"].to_numpy(), m["lng_db"].to_numpy(), m["lat_inv"].to_numpy(), m["lng_inv"].to_numpy())
    m["dist_m"] = np.hypot(e, n)
    m["heading_diff"] = np.abs(wrap180(m["camera_heading_inv"] - m["camera_heading_db"]))
    rows = []
    for city, g in m.groupby("city"):
        rows.append({"city": city, "n": len(g), "median_m": _q(g["dist_m"], 0.5), "p99_m": _q(g["dist_m"], 0.99),
                     "share_over_tol": round(float((g["dist_m"] > AGREE_TOL_M).mean()), 4),
                     "heading_diff_p99_deg": _q(g["heading_diff"], 0.99)})
    return pd.DataFrame(rows, columns=["city", "n", "median_m", "p99_m", "share_over_tol", "heading_diff_p99_deg"])


def neighbours(panos, radius_m):
    """Every ordered pair of distinct panoramas in one city within ``radius_m``:
    ``city, pano_a, pano_b, d_east, d_north, baseline_m`` (B relative to A, :func:`enu_delta`).

    A grid hash in city-local metres finds candidates (cells of ``1.1 * radius_m``; the city-wide cosine distorts
    them by well under 10% inside a city, so the 3 x 3 neighbourhood still covers the radius) and the exact pair
    offset decides.
    """
    out = []
    cell = 1.1 * radius_m
    for city, p in panos.groupby("city", sort=True):
        p = p[["pano_id", "lat", "lng"]].reset_index(drop=True)
        lat0, lng0 = float(p["lat"].median()), float(p["lng"].median())
        x = (p["lng"] - lng0) * M_PER_DEG * math.cos(math.radians(lat0))
        y = (p["lat"] - lat0) * M_PER_DEG
        p["cx"], p["cy"] = np.floor(x / cell).astype(np.int64), np.floor(y / cell).astype(np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = p.rename(columns={"pano_id": "pano_b", "lat": "lat_b", "lng": "lng_b"})
                q = q.assign(cx=q["cx"] - dx, cy=q["cy"] - dy)
                m = p.rename(columns={"pano_id": "pano_a", "lat": "lat_a", "lng": "lng_a"}).merge(
                    q, on=["cx", "cy"], validate="many_to_many")
                m = m[m["pano_a"] != m["pano_b"]]
                if not len(m):
                    continue
                e, n = enu_delta(m["lat_a"].to_numpy(), m["lng_a"].to_numpy(), m["lat_b"].to_numpy(),
                                 m["lng_b"].to_numpy())
                m = m.assign(d_east=e, d_north=n, baseline_m=np.hypot(e, n))
                m = m[m["baseline_m"] <= radius_m]
                out.append(m.assign(city=city)[["city", "pano_a", "pano_b", "d_east", "d_north", "baseline_m"]])
    cols = ["city", "pano_a", "pano_b", "d_east", "d_north", "baseline_m"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def anchors(labels, panos, experts=frozenset(), pool="decided"):
    """Anchor labels: ``(anchors, drops)``.

    One of the five served types, human-authored, a parseable capture month, not a photosphere, on a panorama
    with a camera position, and (``pool="decided"``, the default and the gate's population) vote-decided by
    ``data_generation.eligible_labels``: the margin rule, plus the expert rule when ``experts`` is non-empty.
    ``pool="all"`` keeps undecided labels too (``label_class = ""``), for exploratory residual tables only.
    """
    drops = Counter()
    df = labels[labels["label_type"].isin(TYPES)].copy()
    if pool == "decided":
        # eligible_labels reads user_id only to tag AI-authored rows; the prepared frame holds no user ids, so a
        # stand-in carries the same author, and its expert rule reads the voters' ids from `validations` as usual.
        stand_in = df.assign(user_id=np.where(df["author"] == "ai", dg.AI_USER_ID, "human"))
        parts = []
        for t in TYPES:
            elig, _ = dg.eligible_labels(stand_in, t, experts=experts)
            parts.append(elig[["label_uid", "label_class", "class_source"]])
        dec = pd.concat(parts, ignore_index=True)
        drops["not_vote_decided"] = int(len(df) - len(dec))
        df = df.merge(dec, on="label_uid", validate="one_to_one")
    else:
        df["label_class"], df["class_source"] = df["vote_class"], np.where(df["vote_class"] != "", "margin", "")
    rules = (("ai_authored", lambda d: d["author"] != "human"),
             ("no_capture_month", lambda d: d["capture_month"].isna()),
             ("photosphere", lambda d: d["photosphere"].astype(bool)))
    for reason, rule in rules:  # each label counted once, under its first failing rule
        bad = rule(df).to_numpy(bool)
        drops[reason] += int(bad.sum())
        df = df[~bad]
    has = df[["city", "pano_id"]].merge(panos[["city", "pano_id"]].assign(_has=True), on=["city", "pano_id"],
                                        how="left", validate="many_to_one")["_has"].notna().to_numpy()
    drops["no_camera_position"] += int((~has).sum())
    df = df[has]
    cols = ["label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "class_source", "severity",
            "tags", "user_key", "lon", "lat", "capture_month", "camera_heading", "pano_width", "photosphere"]
    return df[cols].reset_index(drop=True), drops


def candidate_pairs(anc, panos, neigh):
    """Every (anchor, partner panorama) within the search radius in one of the three arms, re-projected, with its
    QC verdict (``qc``). Columns from A carry no suffix; B's carry ``_b``.
    """
    a = anc.merge(neigh, left_on=["city", "pano_id"], right_on=["city", "pano_a"], validate="many_to_many")
    pa = panos[["city", "pano_id", "camera_heading", "pano_width", "n_labels", "agreement_m", "position_source"]]
    a = a.drop(columns=["camera_heading", "pano_width"]).merge(
        pa, on=["city", "pano_id"], validate="many_to_one")
    pb = panos[["city", "pano_id", "camera_heading", "camera_pitch", "capture_month", "pano_width", "n_labels",
                "agreement_m", "position_source"]].rename(columns=lambda c: c if c == "city" else f"{c}_b")
    a = a.merge(pb.rename(columns={"pano_id_b": "pano_b"}), on=["city", "pano_b"], validate="many_to_one")
    a["dt_months"] = (a["capture_month_b"].astype("float") - a["capture_month"].astype("float"))
    a["arm"] = arm_of(a["dt_months"].to_numpy())
    a = a[a["arm"].notna()].copy()
    r = reproject(a["lon"].to_numpy(), a["lat"].to_numpy(), a["camera_heading"].to_numpy(),
                  a["camera_heading_b"].to_numpy(), a["d_east"].to_numpy(), a["d_north"].to_numpy())
    for k, v in r.items():
        a[k] = v
    a["dep_a"], a["dep_b"] = -a["lat"], -a["lat_b"]
    a["has_camera"] = True
    a["photosphere"] = a["pano_b"].map(is_photosphere) | a["photosphere"]
    a["qc"] = qc_reasons(a)
    a["cand_uid"] = a["label_uid"] + "|" + a["pano_b"]
    assert a["cand_uid"].is_unique
    return a.reset_index(drop=True)


def choose_partners(cands):
    """One partner per (anchor, arm) among the QC-passing candidates: the smallest baseline, ties broken by the
    larger ``|dt|`` and then ``pano_b``, so the choice does not depend on row order. Adds ``pair_uid =
    "<label_uid>|<arm>"``.
    """
    ok = cands[cands["qc"] == "ok"].assign(_absdt=lambda d: -d["dt_months"].abs())
    ok = ok.sort_values(["label_uid", "arm", "baseline_m", "_absdt", "pano_b"], kind="mergesort")
    out = ok.drop_duplicates(["label_uid", "arm"]).drop(columns="_absdt").copy()
    out["pair_uid"] = out["label_uid"] + "|" + out["arm"]
    assert out["pair_uid"].is_unique
    return out.reset_index(drop=True)


def leakage_panos(round3_dir):
    """``({type: {pano_id}}, {type: sha256})``: the panoramas of each type's round-3 ``train`` and ``val`` rows.

    Keyed on ``pano_id`` alone, across all cities: a GSV pano_id is the provider's global id, and neighbouring
    deployments (the New Jersey cluster) can hold the same panorama in two schemas.
    """
    out, shas = {}, {}
    for t in TYPES:
        p = os.path.join(round3_dir, t, "split.csv")
        s = pd.read_csv(p, usecols=["pano_id", "split"], dtype={"pano_id": str})
        out[t] = set(s.loc[s["split"].isin(["train", "val"]), "pano_id"])
        shas[t] = sha256_file(p)
    return out, shas


def mark_clean(pairs, leak):
    """``clean`` = neither A's nor B's pano_id is a round-3 train/val panorama of the anchor's type."""
    used = pd.Series([pa in leak.get(t, ()) or pb in leak.get(t, ())
                      for t, pa, pb in zip(pairs["label_type"], pairs["pano_id"], pairs["pano_b"])],
                     index=pairs.index, dtype=bool)
    return ~used


# ---------------------------------------------------------------------------------------------------------------------
# Residual, click floor, contradicting labels
# ---------------------------------------------------------------------------------------------------------------------

def b_labels(labels):
    """B-side measurements: human-authored labels of the five types that the crowd did not vote incorrect."""
    keep = labels["label_type"].isin(TYPES) & (labels["author"] == "human") & (labels["vote_class"] != "incorrect")
    return labels.loc[keep, ["city", "pano_id", "label_type", "lon", "lat", "user_key", "label_uid"]]


def residuals(cands, labels, window_deg=RESIDUAL_WINDOW_DEG):
    """For every QC-passing candidate whose B carries a same-type human label: A's label re-projected into B against
    B's own clicks.

    One row per candidate (``cand_uid``) with ``n_b_labels`` (same type on B), ``n_within`` (within
    ``window_deg``), ``n_within_other_user``, ``matched`` (nearest within the window), ``unique``
    (``n_within == 1``), and for the nearest click ``sep_deg``, ``res_px`` (:func:`view_px`), ``az_px`` and
    ``el_px`` (signed; ``el_px > 0``: B's click is above the prediction) and ``same_user``. A heading or position
    error shows mostly in azimuth, a camera-height, grade or range error in elevation.
    """
    ok = cands[cands["qc"] == "ok"]
    bl = b_labels(labels).rename(columns={"pano_id": "pano_b", "lon": "lon_l", "lat": "lat_l",
                                          "user_key": "user_key_l", "label_uid": "label_uid_l"})
    m = ok[["cand_uid", "city", "pano_b", "label_type", "lon_b", "lat_b", "user_key"]].merge(
        bl, on=["city", "pano_b", "label_type"], validate="many_to_many")
    if not len(m):
        return pd.DataFrame(columns=["cand_uid", "n_b_labels", "n_within", "n_within_other_user", "matched",
                                     "unique", "sep_deg", "res_px", "az_px", "el_px", "same_user"])
    m["sep_deg"] = angular_sep_deg(m["lon_b"], m["lat_b"], m["lon_l"], m["lat_l"])
    m["within"] = m["sep_deg"] <= window_deg
    m["same_user"] = m["user_key"] == m["user_key_l"]
    g = m.groupby("cand_uid", sort=False)
    agg = pd.DataFrame({"n_b_labels": g.size(), "n_within": g["within"].sum(),
                        "n_within_other_user": (m["within"] & ~m["same_user"]).groupby(m["cand_uid"]).sum()})
    near = m.loc[g["sep_deg"].idxmin()].set_index("cand_uid")
    daz = wrap180(near["lon_l"] - near["lon_b"]) * np.cos(np.radians(near["lat_b"]))
    # Every assignment aligns on cand_uid (view_px returns bare arrays, so they are re-indexed first).
    agg["sep_deg"] = near["sep_deg"]
    agg["res_px"] = pd.Series(view_px(near["sep_deg"].to_numpy()), index=near.index)
    agg["az_px"] = pd.Series(view_px(daz.to_numpy()), index=near.index)
    agg["el_px"] = pd.Series(view_px((near["lat_l"] - near["lat_b"]).to_numpy()), index=near.index)
    agg["same_user"] = near["same_user"]
    agg["matched"] = agg["n_within"] >= 1
    agg["unique"] = agg["n_within"] == 1
    return agg.reset_index()


def click_floor(labels, window_deg=FLOOR_WINDOW_DEG):
    """The same statistic for two different users' same-type clicks on one panorama, within ``window_deg``: how
    far apart two people click on (presumably) one feature, with no geometry in between. One row per label pair.
    """
    bl = b_labels(labels)
    m = bl.merge(bl, on=["city", "pano_id", "label_type"], suffixes=("_1", "_2"), validate="many_to_many")
    m = m[(m["label_uid_1"] < m["label_uid_2"]) & (m["user_key_1"] != m["user_key_2"])]
    m = m.assign(sep_deg=angular_sep_deg(m["lon_1"], m["lat_1"], m["lon_2"], m["lat_2"]))
    m = m[m["sep_deg"] <= window_deg]
    return pd.DataFrame({"city": m["city"].to_numpy(), "label_type": m["label_type"].to_numpy(),
                         "res_px": view_px(m["sep_deg"].to_numpy())})


def contradicting_on_b(cands, labels, window_deg=CONTRA_WINDOW_DEG):
    """Per QC-passing CurbRamp or NoCurbRamp candidate: does B carry a human label of the *other* of the two types
    within ``window_deg`` of the re-projected point? (A ramp on A and "no ramp" at the same spot on B, or the
    reverse: the human signal of a change the model's flag must beat.) Returns a boolean Series by ``cand_uid``.
    """
    ok = cands[(cands["qc"] == "ok") & cands["label_type"].isin(["CurbRamp", "NoCurbRamp"])]
    other = ok["label_type"].map({"CurbRamp": "NoCurbRamp", "NoCurbRamp": "CurbRamp"})
    bl = b_labels(labels).rename(columns={"pano_id": "pano_b", "label_type": "other_type", "lon": "lon_l",
                                          "lat": "lat_l"})
    m = ok[["cand_uid", "city", "pano_b", "lon_b", "lat_b"]].assign(other_type=other.to_numpy()).merge(
        bl[["city", "pano_b", "other_type", "lon_l", "lat_l"]], on=["city", "pano_b", "other_type"],
        validate="many_to_many")
    hit = angular_sep_deg(m["lon_b"], m["lat_b"], m["lon_l"], m["lat_l"]) <= window_deg
    found = set(m.loc[hit, "cand_uid"])
    return pd.Series([c in found for c in ok["cand_uid"]], index=ok["cand_uid"].to_numpy(), dtype=bool)


# ---------------------------------------------------------------------------------------------------------------------
# Tables and the gate
# ---------------------------------------------------------------------------------------------------------------------

def _q(s, p):
    s = pd.Series(s, dtype=float).dropna()
    return round(float(s.quantile(p)), 4) if len(s) else None


def summarize(res_px, az_px=None, el_px=None):
    """n, median, p75, p90 of residual pixels, shares within 100 and 240 px, and the azimuth/elevation split."""
    r = pd.Series(res_px, dtype=float).dropna()
    out = {"n": int(len(r)), "median_px": _r(r.median()), "p75_px": _r(r.quantile(0.75)),
           "p90_px": _r(r.quantile(0.9)), "within_100": _r(np.mean(r <= 100), 3),
           "within_240": _r(np.mean(r <= 240), 3)}
    if az_px is not None:
        out["abs_az_median_px"] = _r(pd.Series(az_px, dtype=float).abs().median())
        out["abs_el_median_px"] = _r(pd.Series(el_px, dtype=float).abs().median())
        out["el_median_px"] = _r(pd.Series(el_px, dtype=float).median())
    return out


def _r(v, nd=1):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


BASELINE_BINS = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0)]
RANGE_BINS = [(0.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 50.0)]


def _bin_label(v, bins):
    for lo, hi in bins:
        if (lo == bins[0][0] and lo <= v <= hi) or lo < v <= hi:
            return f"{lo:g}-{hi:g}"
    return None


def era(year):
    """Capture-year bin: ``-2011``, ``2012-15``, ``2016-19``, ``2020-``.

    >>> era(2010), era(2014), era(2019), era(2024)
    ('-2011', '2012-15', '2016-19', '2020-')
    """
    return "-2011" if year <= 2011 else "2012-15" if year <= 2015 else "2016-19" if year <= 2019 else "2020-"


def residual_frame(cands, res):
    """QC-passing candidates with a same-type B label, joined to their residual row, with the bins the tables use."""
    ok = cands[cands["qc"] == "ok"]
    m = ok.merge(res, on="cand_uid", validate="one_to_one")
    m["baseline_bin"] = [_bin_label(v, BASELINE_BINS) for v in m["baseline_m"]]
    m["range_bin"] = [_bin_label(v, RANGE_BINS) for v in m["r_a"]]
    ya, yb = m["capture_month"].astype(float) // 12, m["capture_month_b"].astype(float) // 12
    m["era_pair"] = [f"{era(a)} -> {era(b)}" for a, b in zip(ya, yb)]
    m["width_pair"] = [f"{int(a)} -> {int(b)}" for a, b in zip(m["pano_width"], m["pano_width_b"])]
    # Was each camera position checked? A database position is; an inverted one only when its panorama holds two or
    # more labels that agree (one label: a depth-era position would invert to the wrong place unnoticed).
    checked = lambda src, n: (src == "db") | (n >= 2)  # noqa: E731
    m["positions_checked"] = np.where(checked(m["position_source"], m["n_labels"])
                                      & checked(m["position_source_b"], m["n_labels_b"]), "both", "not both")
    return m


ARMS = ("same_month", "control", "long_newer")


def residual_tables(rf, floor, primary_m=3.0, min_n=30):
    """The Stage 0 tables, as ``{name: DataFrame}``. Gate population unless the name says otherwise: unique matches
    (``n_within == 1``) of vote-decided anchors at baselines up to ``primary_m``. Cells with ``n < min_n`` are left
    out of the per-city, era and width tables (and pooled into one ``other`` row per arm in the per-city one).
    Per-type, per-city and per-bin tables are exploratory: no per-cell significance claims.
    """
    t = {}
    gate = rf[rf["unique"] & (rf["baseline_m"] <= primary_m)]

    def rows(df, keys):
        out = []
        for k, g in df.groupby(keys, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            out.append({**dict(zip(keys, k)), **summarize(g["res_px"], g["az_px"], g["el_px"])})
        return pd.DataFrame(out)

    def arm_sorted(df):
        """Arms in ARMS order, then the second key; bins by their numeric lower edge, not as text."""
        if "arm" in df:
            second = [c for c in df.columns if c in ("label_type", "baseline_bin", "range_bin", "city",
                                                       "positions_checked")]
            df = df.assign(_o=df["arm"].map({a: i for i, a in enumerate(ARMS)}))
            if second and second[0].endswith("_bin"):
                df = df.assign(_k=df[second[0]].map(lambda b: float(str(b).split("-")[0])))
                second = ["_k"]
            df = df.sort_values(["_o"] + second, kind="mergesort").drop(columns=["_o", "_k"], errors="ignore")
        return df.reset_index(drop=True)

    t["by_arm"] = arm_sorted(rows(gate, ["arm"]))
    t["by_arm_type"] = arm_sorted(rows(gate, ["arm", "label_type"]))
    wide = rf[rf["unique"]]
    t["by_arm_baseline"] = arm_sorted(rows(wide, ["arm", "baseline_bin"]))
    t["by_arm_range"] = arm_sorted(rows(gate, ["arm", "range_bin"]))
    t["by_arm_positions_checked"] = arm_sorted(rows(gate, ["arm", "positions_checked"]))
    byc = rows(gate, ["arm", "city"])
    if len(byc):
        small = byc[byc["n"] < min_n]
        other = []
        for arm in small["arm"].unique():
            g = gate[(gate["arm"] == arm) & gate["city"].isin(small.loc[small["arm"] == arm, "city"])]
            other.append({"arm": arm, "city": f"other ({small['arm'].eq(arm).sum()} deployments, n<{min_n} each)",
                          **summarize(g["res_px"], g["az_px"], g["el_px"])})
        byc = pd.concat([byc[byc["n"] >= min_n], pd.DataFrame(other)], ignore_index=True)
    t["by_arm_city"] = arm_sorted(byc)
    ln = gate[gate["arm"] == "long_newer"]
    e = rows(ln, ["era_pair"])
    t["long_newer_by_era"] = e[e["n"] >= min_n].reset_index(drop=True) if len(e) else e
    w = rows(ln, ["width_pair"])
    t["long_newer_by_width"] = w[w["n"] >= min_n].reset_index(drop=True) if len(w) else w
    anym = rf[rf["matched"] & (rf["baseline_m"] <= primary_m)]
    t["matched_any_by_arm"] = arm_sorted(rows(anym, ["arm"]))
    cov = []
    for arm, g in rf[rf["baseline_m"] <= primary_m].groupby("arm"):
        cov.append({"arm": arm, "with_same_type_b_label": len(g), "matched_30deg": int(g["matched"].sum()),
                    "unique": int(g["unique"].sum()), "unmatched_share": _r(1 - g["matched"].mean(), 3),
                    "same_user_share_of_unique": _r(g.loc[g["unique"], "same_user"].mean(), 3)})
    t["coverage"] = arm_sorted(pd.DataFrame(cov))
    fl = []
    if len(floor):
        fl.append({"city": "all", "label_type": "all", **summarize(floor["res_px"])})
        for k, g in floor.groupby("label_type"):
            fl.append({"city": "all", "label_type": k, **summarize(g["res_px"])})
        for k, g in floor.groupby("city"):
            if len(g) >= min_n:
                fl.append({"city": k, "label_type": "all", **summarize(g["res_px"])})
    t["click_floor"] = pd.DataFrame(fl)
    return t


def census_tables(anc, cands, pairs, primary_m=3.0, wide_m=B_SEARCH_M):
    """Yield, per city and type and pooled per type: decided and vote-correct anchors; vote-correct anchors with a
    ``long_newer`` partner at ``primary_m`` and ``wide_m``; how many of those are clean of round-3 train/val
    panoramas; the negative-control pool (clean, vote-correct ``long_newer`` pairs whose B carries a same-type label
    by another user within the residual window); pairs with a contradicting label on B; ``control`` partners; and
    QC rejections over all candidates.
    """
    def counts(c, p):
        vc = p["label_class"] == "correct"
        ln = p[(p["arm"] == "long_newer") & vc]
        ln3 = ln[ln["baseline_m"] <= primary_m]
        ctl = p[(p["arm"] == "control") & vc]
        return {
            "decided_anchors": int(c["label_uid"].nunique()) if len(c) else 0,
            "vote_correct_anchors": int(c.loc[c["label_class"] == "correct", "label_uid"].nunique()) if len(c) else 0,
            f"long_newer_le{primary_m:g}m": int(len(ln3)), f"long_newer_le{wide_m:g}m": int(len(ln)),
            f"long_newer_le{primary_m:g}m_clean": int(ln3["clean"].sum()),
            "clean_share": _r(ln3["clean"].mean(), 3) if len(ln3) else None,
            "negative_control_pool": int((ln3["clean"] & ln3["negative_control"]).sum()),
            "contradicting_on_b": int(ln3["contradicting"].sum()),
            f"control_le{primary_m:g}m": int((ctl["baseline_m"] <= primary_m).sum()),
            f"control_le{wide_m:g}m": int(len(ctl)),
            "unchecked_inversion_share": _r((ln3["n_labels"].lt(2) | ln3["n_labels_b"].lt(2)).mean(), 3)
            if len(ln3) else None,
        }

    by_type, by_city_type = [], []
    for t in TYPES:
        by_type.append({"label_type": t, **counts(anc[anc["label_type"] == t], pairs[pairs["label_type"] == t])})
    by_type.append({"label_type": "all", **counts(anc, pairs)})
    for (city, t), _ in anc.groupby(["city", "label_type"]):
        by_city_type.append({"city": city, "label_type": t,
                             **counts(anc[(anc["city"] == city) & (anc["label_type"] == t)],
                                      pairs[(pairs["city"] == city) & (pairs["label_type"] == t)])})
    qc = cands.groupby(["arm", "qc"]).size().unstack(fill_value=0).reset_index()
    return {"census_by_type": pd.DataFrame(by_type), "census_by_city_type": pd.DataFrame(by_city_type),
            "qc_rejections": qc}


def load_gates(path=GATES_PATH):
    with open(path) as f:
        return json.load(f)


def gate0(tables, census, gates, position_sources):
    """Gate 0 from the Stage 0 tables: every rule's value and threshold, and pass/fail only on database positions.

    ``tables`` are :func:`residual_tables` at the primary baseline (plus ``by_arm_baseline`` for the cap call),
    ``census`` is :func:`census_tables`' ``census_by_type``, ``gates`` the parsed ``gates.json``,
    ``position_sources`` the set of ``position_source`` values behind the pairs. With anything but ``{"db"}`` the
    result has ``verdict = None`` and every rule's ``pass = None``: a preview cannot pass or fail the gate.

    Rules (thresholds from ``gates["gate0"]``): G0a residual on unique matches, for ``long_newer`` and ``control``
    pooled and per type with ``n >= type_min_n`` (a failing type is dropped, listed in ``types_dropped``); G0b
    ``long_newer`` median over ``same_month`` median; G0c yield (clean vote-correct ``long_newer`` pairs, types
    with enough of them, the negative-control pool over the presence types). ``baseline_cap_m`` is the wide cap
    when the 3-5 m bin of both arms passes G0a too, else the primary one.
    """
    g = gates["gate0"]
    preview = set(position_sources) != {"db"}

    def res_ok(row):
        return (row["median_px"] is not None and row["median_px"] <= g["residual_median_px_max"]
                and row["within_240"] >= g["residual_within_share_min"])

    rules, types_kept, types_dropped = [], set(TYPES), []
    by_arm = tables["by_arm"].set_index("arm") if len(tables["by_arm"]) else pd.DataFrame()
    for arm in ("long_newer", "control"):
        row = by_arm.loc[arm].to_dict() if arm in by_arm.index else {"n": 0, "median_px": None, "within_240": None}
        rules.append({"rule": f"G0a residual {arm} pooled", "n": int(row["n"]), "median_px": row["median_px"],
                      "within_240": row["within_240"],
                      "threshold": f"median <= {g['residual_median_px_max']} px and >= "
                                   f"{g['residual_within_share_min']:.0%} within {g['residual_within_px']} px",
                      "pass": bool(row["n"] and res_ok(row))})
    bt = tables["by_arm_type"]
    for t in TYPES:
        for arm in ("long_newer", "control"):
            sel = bt[(bt["arm"] == arm) & (bt["label_type"] == t)] if len(bt) else bt
            if not len(sel) or sel.iloc[0]["n"] < g["type_min_n"]:
                continue
            row = sel.iloc[0].to_dict()
            ok = res_ok(row)
            rules.append({"rule": f"G0a residual {arm} {t}", "n": int(row["n"]), "median_px": row["median_px"],
                          "within_240": row["within_240"], "threshold": "as pooled; a failing type is dropped",
                          "pass": bool(ok)})
            if not ok and t in types_kept:
                types_kept.discard(t)
                types_dropped.append(t)
    sm = by_arm.loc["same_month", "median_px"] if "same_month" in by_arm.index else None
    ln = by_arm.loc["long_newer", "median_px"] if "long_newer" in by_arm.index else None
    ratio = (ln / sm) if (sm and ln is not None) else None
    rules.append({"rule": "G0b era", "value": _r(ratio, 3), "threshold": f"long_newer / same_month median <= "
                  f"{g['era_ratio_max']}", "pass": bool(ratio is not None and ratio <= g["era_ratio_max"])})
    c = census.set_index("label_type")
    key = f"long_newer_le{g['primary_baseline_m']:g}m_clean"
    kept = [t for t in TYPES if t in types_kept]
    total = int(sum(c.loc[t, key] for t in kept))
    enough = [t for t in kept if c.loc[t, key] >= g["yield_per_type_min"]]
    neg = int(sum(c.loc[t, "negative_control_pool"] for t in PRESENCE_TYPES if t in kept))
    rules.append({"rule": "G0c yield: clean vote-correct long_newer pairs", "value": total,
                  "threshold": f">= {g['yield_long_newer_min']}", "pass": bool(total >= g["yield_long_newer_min"])})
    rules.append({"rule": "G0c yield: types with enough pairs", "value": enough,
                  "threshold": f">= {g['yield_types_min']} types with >= {g['yield_per_type_min']}",
                  "pass": len(enough) >= g["yield_types_min"]})
    rules.append({"rule": "G0c yield: negative-control pool (presence types)", "value": neg,
                  "threshold": f">= {g['negative_control_min']}", "pass": bool(neg >= g["negative_control_min"])})
    bb = tables["by_arm_baseline"]
    wide_bin = f"{BASELINE_BINS[-1][0]:g}-{BASELINE_BINS[-1][1]:g}"
    wide_ok = all(len(bb[(bb["arm"] == a) & (bb["baseline_bin"] == wide_bin)])
                  and res_ok(bb[(bb["arm"] == a) & (bb["baseline_bin"] == wide_bin)].iloc[0].to_dict())
                  for a in ("long_newer", "control")) if len(bb) else False
    out = {"gate": "0", "thresholds": g, "thresholds_status": gates.get("status"),
           "thresholds_confirmed_on": gates.get("confirmed_on"), "position_sources": sorted(position_sources),
           "types_kept": kept, "types_dropped": types_dropped,
           "baseline_cap_m": g["wide_baseline_m"] if wide_ok else g["primary_baseline_m"], "rules": rules}
    if preview:
        for r in rules:
            r["pass"] = None
        out.update(verdict=None, baseline_cap_m=None,
                   refused="camera positions are not all from the database (pano_data): PREVIEW, not the gate")
    else:
        # Per-type residual rules only drop types; the pooled residual, era and yield rules decide.
        deciding = [r for r in rules if r["rule"].endswith("pooled") or r["rule"].startswith(("G0b", "G0c"))]
        out["verdict"] = "pass" if all(r["pass"] for r in deciding) else "fail"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# I/O and subcommands
# ---------------------------------------------------------------------------------------------------------------------

def public_cities(listing_path):
    """``(public city_ids, listing sha256, fetched_on)`` from a saved ``/v3/api/cities`` response."""
    with open(listing_path) as f:
        listing = json.load(f)
    entries = dg.listing_entries(listing)
    pub = sorted(c["city_id"] for c in entries if c.get("visibility") == "public")
    return pub, sha256_file(listing_path), (listing.get("fetched_on") if isinstance(listing, dict) else None)


def load_labels(cache, cities):
    """rawLabels for ``cities`` from the cache (``data_generation.load_city``), prepared; and per-file sha256s."""
    frames, shas, drops = [], {}, Counter()
    for city in cities:
        path = os.path.join(cache, f"{city}.csv")
        raw = dg.load_city(city, cache_dir=cache)
        cols = [c for c in LABEL_COLUMNS if c in raw.columns]
        lab, d = prepare_labels(raw[cols + ["city"]])
        frames.append(lab)
        drops.update(d)
        shas[f"{city}.csv"] = sha256_file(path)
    return pd.concat(frames, ignore_index=True), shas, drops


def cached_public(cache, listing):
    pub, sha, fetched = public_cities(listing)
    have = sorted(f[:-4] for f in os.listdir(cache) if f.endswith(".csv"))
    return [c for c in have if c in pub], sha, fetched


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def md_table(df):
    """A DataFrame as a GitHub markdown table (no extra dependency)."""
    if df is None or not len(df):
        return "_(no rows)_\n"
    cols = list(df.columns)
    fmt = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def cmd_positions(a):
    cities, listing_sha, fetched = cached_public(a.rawlabels, a.listing)
    labels, shas, drops = load_labels(a.rawlabels, cities)
    if a.db_methods:
        meth = pd.read_csv(a.db_methods, dtype={"schema": str})
        sc = _schema_city(meth["schema"].unique(), a.cityparams)
        meth["label_uid"] = meth["schema"].map(sc) + ":" + meth["label_id"].astype(str)
        keep = set(meth.loc[meth["computation_method"] == "approximation3", "label_uid"])
        drops["not_approximation3"] = int((~labels["label_uid"].isin(keep)).sum())
        labels = labels[labels["label_uid"].isin(keep)]
    inv, stats = panos_from_rawlabels(labels)
    info = {"made_on": date.today().isoformat(), "git_head": git_head(), "cities": cities,
            "listing": {"path": os.path.basename(a.listing), "sha256": listing_sha, "fetched_on": fetched},
            "rawlabels_sha256": shas, "label_drops": dict(drops), "inversion": stats, "agree_tol_m": AGREE_TOL_M}
    panos = inv
    if a.db_panos:
        db = pd.read_csv(a.db_panos, usecols=["schema"], dtype=str)
        sc = _schema_city(db["schema"].unique(), a.cityparams)
        dbp = panos_from_db(a.db_panos, sc)
        dbp = dbp[dbp["city"].isin(cities)]
        info["db_panos_sha256"] = sha256_file(a.db_panos)
        info["compare_positions"] = compare_positions(inv, dbp).to_dict("records")
        panos = dbp
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    panos.to_csv(a.out, index=False)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1, default=str)
    print(f"{len(panos)} panoramas ({panos['position_source'].iloc[0] if len(panos) else '-'}) -> {a.out}")


def _schema_city(schemas, cityparams):
    if not cityparams:
        raise SystemExit("--cityparams is required with database inputs (schema -> city_id)")
    from analysis.gt_rounds import city_ids
    return city_ids(list(schemas), cityparams)


def cmd_census(a):
    cities, listing_sha, fetched = cached_public(a.rawlabels, a.listing)
    labels, shas, drops = load_labels(a.rawlabels, cities)
    panos = pd.read_csv(a.panos, dtype={"pano_id": str})
    panos["capture_month"] = panos["capture_month"].astype("Int64")
    panos = panos[panos["city"].isin(cities)]
    experts = dg.parse_experts(os.environ.get(dg.EXPERTS_ENV))
    anc, adrops = anchors(labels, panos, experts=experts, pool=a.anchor_pool)
    neigh = neighbours(panos, B_SEARCH_M)
    cands = candidate_pairs(anc, panos, neigh)
    leak, split_shas = leakage_panos(a.round3)
    cands["clean"] = mark_clean(cands, leak)
    os.makedirs(a.out, exist_ok=True)
    keep = ["cand_uid", "label_uid", "city", "label_id", "pano_id", "pano_b", "label_type", "label_class",
            "class_source", "severity", "tags", "user_key", "lon", "lat", "capture_month", "capture_month_b",
            "dt_months", "arm", "camera_heading", "camera_heading_b", "camera_pitch_b", "pano_width", "pano_width_b",
            "n_labels", "n_labels_b", "position_source", "position_source_b", "d_east", "d_north", "baseline_m",
            "r_a", "d_b", "lon_b", "lat_b", "x_norm_b", "y_norm_b", "parallax_deg", "qc", "clean"]
    anc.drop(columns=["tags"]).to_csv(os.path.join(a.out, "anchors.csv.gz"), index=False)
    cands[keep].to_csv(os.path.join(a.out, "candidates.csv.gz"), index=False)
    choose_partners(cands)[keep + ["pair_uid"]].to_csv(os.path.join(a.out, "pairs.csv"), index=False)
    labels[["label_uid", "city", "pano_id", "label_type", "lon", "lat", "author", "vote_class", "user_key"]].to_csv(
        os.path.join(a.out, "labels.csv.gz"), index=False)
    sel = {"made_on": date.today().isoformat(), "git_head": git_head(), "cities": cities,
           "listing": {"path": os.path.basename(a.listing), "sha256": listing_sha, "fetched_on": fetched},
           "rawlabels_sha256": shas, "panos_sha256": sha256_file(a.panos), "round3_split_sha256": split_shas,
           "positions_info": _read_json(os.path.splitext(a.panos)[0] + ".json"),
           "anchor_pool": a.anchor_pool, "experts_set": bool(experts), "label_drops": dict(drops),
           "anchor_drops": dict(adrops), "position_sources": sorted(panos["position_source"].unique().tolist()),
           "settings": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple, str))
                        and k not in ("PREVIEW_BANNER", "GATE_BANNER", "GATES_PATH")}}
    with open(os.path.join(a.out, "selection.json"), "w") as f:
        json.dump(sel, f, indent=1, default=str)
    print(f"{len(anc)} anchors, {len(cands)} candidates ({int((cands['qc'] == 'ok').sum())} pass QC) -> {a.out}")


def cmd_residual(a):
    with open(os.path.join(a.run, "selection.json")) as f:
        sel = json.load(f)
    pub, listing_sha, fetched = public_cities(a.listing)
    cands = pd.read_csv(os.path.join(a.run, "candidates.csv.gz"), dtype={"pano_id": str, "pano_b": str})
    labels = pd.read_csv(os.path.join(a.run, "labels.csv.gz"), dtype={"pano_id": str})
    labels["vote_class"] = labels["vote_class"].fillna("")
    not_public = sorted(set(cands["city"]) - set(pub))
    if not_public:  # the committed tables name deployments: re-check against today's listing
        raise SystemExit(f"deployments not public in {a.listing}: {not_public}")
    gates = load_gates(a.gates)
    primary = gates["gate0"]["primary_baseline_m"]
    res = residuals(cands, labels)
    floor = click_floor(labels)
    contra = contradicting_on_b(cands, labels)
    cands = cands.merge(res[["cand_uid", "n_within_other_user"]], on="cand_uid", how="left", validate="one_to_one")
    cands["negative_control"] = cands["n_within_other_user"].fillna(0) > 0
    cands["contradicting"] = cands["cand_uid"].isin(contra.index[contra.to_numpy()])
    pairs = choose_partners(cands)
    rf = residual_frame(cands, res)
    tables = residual_tables(rf, floor, primary_m=primary)
    anc = pd.read_csv(os.path.join(a.run, "anchors.csv.gz"), usecols=["label_uid", "city", "label_type",
                                                                        "label_class"])
    census = census_tables(anc, cands, pairs, primary_m=primary, wide_m=gates["gate0"]["wide_baseline_m"])
    sources = set(cands["position_source"]) | set(cands["position_source_b"])
    g0 = gate0(tables, census["census_by_type"], gates, sources)
    banner = GATE_BANNER if sources == {"db"} else PREVIEW_BANNER
    os.makedirs(a.out, exist_ok=True)
    for name, df in {**tables, **census}.items():
        with open(os.path.join(a.out, f"{name}.csv"), "w") as f:
            f.write(f"# {banner}\n")
            df.to_csv(f, index=False)
    with open(os.path.join(a.out, "residual.md"), "w") as f:
        f.write(f"**{banner}**\n\nGenerated by `analysis/temporal_pairs.py residual` on {date.today().isoformat()}. "
                f"Gate population: unique matches (one same-type B click within {RESIDUAL_WINDOW_DEG:g} deg) of "
                f"vote-decided anchors, baseline <= {primary:g} m, unless a table says otherwise. Exploratory per "
                f"cell; residuals truncated at {RESIDUAL_WINDOW_DEG:g} deg ({view_px(RESIDUAL_WINDOW_DEG):.0f} px).\n")
        for name, df in tables.items():
            f.write(f"\n### {name}\n\n{md_table(df)}")
    with open(os.path.join(a.out, "census.md"), "w") as f:
        f.write(f"**{banner}**\n\nGenerated by `analysis/temporal_pairs.py residual` on {date.today().isoformat()}.\n")
        for name, df in census.items():
            f.write(f"\n### {name}\n\n{md_table(df)}")
    with open(os.path.join(a.out, "gate0.json"), "w") as f:
        json.dump({"banner": banner, **g0}, f, indent=1, default=_json_default)
    prov = {"banner": banner, "made_on": date.today().isoformat(), "git_head": git_head(),
            "run_selection": {k: sel.get(k) for k in ("made_on", "git_head", "cities", "listing", "rawlabels_sha256",
                                                      "panos_sha256", "positions_info", "round3_split_sha256",
                                                      "anchor_pool", "experts_set", "label_drops", "anchor_drops",
                                                      "position_sources", "settings")},
            "public_check": {"listing": os.path.basename(a.listing), "sha256": listing_sha, "fetched_on": fetched,
                             "all_named_deployments_public": True},
            "gates_sha256": sha256_file(a.gates)}
    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1, default=str)
    print(banner)
    print(md_table(tables["by_arm"]))
    print(json.dumps({k: g0[k] for k in ("verdict", "types_kept", "baseline_cap_m")}, default=_json_default))


def _read_json(path):
    """A JSON file's content, or None when it is absent (positions run elsewhere)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("positions", help="one camera position per panorama (inverted, or from the DB pull)")
    s.add_argument("--rawlabels", required=True, help="rawLabels cache directory (<city_id>.csv)")
    s.add_argument("--listing", required=True, help="a saved /v3/api/cities response (public deployments only)")
    s.add_argument("--db-panos", help="analysis/pull_pano_cameras.sh MODE=panos output; makes this a GATE run")
    s.add_argument("--db-methods", help="MODE=methods output: invert only approximation3 labels")
    s.add_argument("--cityparams", help="SidewalkWebpage conf/cityparams.conf (schema -> city_id)")
    s.add_argument("--out", required=True, help="RUN/panos.csv (outside the repo)")
    s = sub.add_parser("census", help="anchors, partners, re-projection and QC (per-label files in RUN)")
    s.add_argument("--rawlabels", required=True)
    s.add_argument("--listing", required=True)
    s.add_argument("--panos", required=True)
    s.add_argument("--round3", required=True, help="round-3 run root with <Type>/split.csv")
    s.add_argument("--anchor-pool", choices=("decided", "all"), default="decided",
                   help="decided (the gate's population) or all labels (exploratory)")
    s.add_argument("--out", required=True, help="RUN directory (outside the repo)")
    s = sub.add_parser("residual", help="residual, click floor, census tables and Gate 0 (aggregates only)")
    s.add_argument("--run", required=True)
    s.add_argument("--listing", required=True, help="today's /v3/api/cities response: every named deployment "
                                                    "must be public in it")
    s.add_argument("--gates", default=GATES_PATH)
    s.add_argument("--out", required=True)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {"positions": cmd_positions, "census": cmd_census, "residual": cmd_residual}[a.cmd](a)


if __name__ == "__main__":
    main()
