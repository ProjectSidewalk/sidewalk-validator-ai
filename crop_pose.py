"""The rig pose a label's crop is corrected with, per imagery provider (issue #72).

A human label is clicked in a gravity-levelled viewer, while the stored equirectangular panorama is in the camera
rig's frame, so ``data_generation.py`` renders the crop with ``sidewalk_panorama_tools.geometry.label_view``'s tilt
correction. That package takes the pose in ONE sign convention (``pano_pose``'s measured one): **pitch > 0 = nose
down, roll > 0 = left side up**, so a level ray at bearing b sits at rig elevation ``lat + beta * T(b)`` with
``T(b) = pitch cos b + roll sin b``. Each provider stores its pose differently, and :func:`pose_for_row` converts it:

=============  ====================  ===========================================  ======  ==================
pano_source    pose_source           package (pitch, roll) from                   beta    evidence
=============  ====================  ===========================================  ======  ==================
gsv            ``gsv_db``            DB (camera_pitch, camera_roll) unchanged     0.95    panorama-tools #54
mapillary      ``mapillary_db``      DB (-camera_pitch, +camera_roll)             1.0     pixel test (#72)
mapillary      ``mapillary_api``     Graph API computed_rotation, then as above  1.0     pixel test (#72)
panoramax      ``panoramax_db``      DB (-camera_pitch, +camera_roll), both != 0  1.0     pixel test (#72)
(any)          ``image_estimate``    RESERVED: RampNet PR 273's estimator         --      not implemented
(any)          ``none``              no usable pose: no correction                --
(AI label)     ``rig_frame``         placed on the rig image: no correction       --
=============  ====================  ===========================================  ======  ==================

**Signs were measured on pixels, not taken from documentation** (tests/test_pose_levelling.py): resampling a
panorama into the gravity frame with the package's own ``tilt_matrix`` straightens its near-vertical edges under
exactly one of the four sign readings. Both website viewers store the pitch as the elevation of the camera's forward
axis (positive = nose UP; ``MapillaryViewer.extractPitchRoll`` computes ``asin(forward_up)``, and Panoramax's
``pers:pitch`` is GPano's "degrees above the horizon"), the opposite of GSV's DB column, and both store the roll in
the package's sign.

**Mapillary.** The DB columns are the website's ``extractPitchRoll`` of the Graph API ``computed_rotation`` (filled
server-side from the same vector since SidewalkWebpage#5725), so they equal :func:`mapillary_pitch_roll` of that
vector exactly (checked on 80 Richmond VA / Laurens IA panoramas: |difference| < 1e-13 degrees, roll the same sign).
A label with no DB pose takes the API vector when ``data_generation.py`` was given a token
(``--mapillary-token-env``); its responses are cached on disk, so a re-run makes no calls.

**Panoramax.** ``PanoramaxViewer`` stores ``pers:pitch || 0`` and ``pers:roll ?? undefined`` unchanged, and levels
the sphere ONLY when both angles are non-zero (it mirrors Panoramax's own viewer, which distrusts a lone angle). A
label on a panorama with a zero or missing angle was therefore clicked on the un-levelled (rig-frame) image and
needs no correction; an exact 0/0 is also no measurement (Bayonne: absent 42%, 0/0 21% of panoramas,
panorama-tools ``reports/2026-10-06-non-gsv-levelling.md``). Those rows get ``pose_source = none`` with their own
reason in the counts.

**Beta** for Mapillary and Panoramax is 1.0 because both viewers level the whole sphere before the click. It is
UNMEASURED (GSV's 0.95 was measured on 3,518 label pairs); issue #72 step 2 measures it on Mapillary.
"""
import json
import math
import os
import time
from collections import namedtuple

#: Beta (the fraction of the rig tilt in the stored click) per pano_source. GSV's comes from ``--tilt-beta``
#: (default 0.95, measured); the others are the unmeasured "the viewer levels fully" assumption (module docstring).
SOURCE_BETA = {"mapillary": 1.0, "panoramax": 1.0}

#: Every ``pose_source`` value. ``image_estimate`` is reserved for the image-based tilt estimator (RampNet PR 273),
#: the fallback for panoramas with no usable metadata; nothing produces it yet.
POSE_SOURCES = ("gsv_db", "mapillary_db", "mapillary_api", "panoramax_db", "image_estimate", "none", "rig_frame")

#: ``(camera_pitch, camera_roll)`` in the geometry package's sign (pitch > 0 nose down, roll > 0 left side up), or
#: None each for no correction; ``pose_source`` (:data:`POSE_SOURCES`); ``beta`` for ``label_view`` (None when no
#: correction applies); ``reason`` says why (counted in the run summary, not written to the manifest).
Pose = namedtuple("Pose", ["camera_pitch", "camera_roll", "pose_source", "beta", "reason"])


def _finite(v):
    """``float(v)`` if it is a finite number, else None (None, NaN, blanks and strings are "no value")."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --- Mapillary: Graph API computed_rotation -> (pitch, roll) ---------------------------------------------------------
# Ported from sidewalk-auto-labeler geo.py (rotation_matrix, opensfm_pose, mapillary_pose_status,
# mapillary_pitch_roll; ProjectSidewalk/sidewalk-auto-labeler#42, whose study locked the convention four ways).
# Output is in the WEBSITE's sign (the DB columns' sign: pitch positive nose UP), not the geometry package's;
# :func:`pose_for_row` converts both the DB and the API value in one place.

#: A pose past this tilt (camera-up vs world-up) is a failed reconstruction, not a rig (30 of 191,286 panoramas in
#: the auto-labeler's five Mapillary runs, up to 170 degrees); it is treated as no pose.
MAX_POSE_TILT_DEG = 45.0


def rotation_matrix(rvec):
    """Rodrigues: axis-angle vector -> 3x3 rotation matrix (nested lists)."""
    rx, ry, rz = (float(v) for v in rvec)
    th = math.sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / th, ry / th, rz / th
    c, s = math.cos(th), math.sin(th)
    v = 1.0 - c
    return [[c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]]


def opensfm_pose(rvec):
    """Heading, pitch, roll and tilt (degrees) of OpenSfM's world->camera axis-angle rotation (world = ENU, camera =
    OpenCV x right, y down, z forward). Pitch is the forward axis's elevation (positive up), roll is in the website's
    sign, tilt the angle between camera-up and world-up.

    >>> p = opensfm_pose([math.pi / 2, 0.0, 0.0])    # a level camera facing north
    >>> [round(p[k], 9) + 0.0 for k in ('heading_deg', 'pitch_deg', 'roll_deg', 'tilt_deg')]
    [0.0, 0.0, 0.0, 0.0]
    """
    R = rotation_matrix(rvec)
    fwd_e, fwd_n, fwd_u = R[2]
    right_e, right_n, right_u = R[0]
    up_u = -R[1][2]
    heading = math.atan2(fwd_e, fwd_n)
    pitch = math.asin(max(-1.0, min(1.0, fwd_u)))
    cp, sp = math.cos(heading), math.sin(heading)
    ca, sa = math.cos(pitch), math.sin(pitch)
    level_right = (cp, -sp, 0.0)
    pitched_up = (-sa * sp, -sa * cp, ca)
    roll = math.atan2(right_e * pitched_up[0] + right_n * pitched_up[1] + right_u * pitched_up[2],
                      right_e * level_right[0] + right_n * level_right[1] + right_u * level_right[2])
    tilt = math.acos(max(-1.0, min(1.0, up_u)))
    return {'heading_deg': math.degrees(heading) % 360.0, 'pitch_deg': math.degrees(pitch),
            'roll_deg': -math.degrees(roll), 'tilt_deg': math.degrees(tilt)}


def mapillary_pose_status(rvec):
    """``ok``, ``missing`` (no vector), ``malformed`` (not three finite numbers) or ``over_tilt``."""
    if rvec is None:
        return "missing"
    if not isinstance(rvec, (list, tuple)) or len(rvec) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in rvec):
        return "malformed"
    if opensfm_pose(rvec)['tilt_deg'] > MAX_POSE_TILT_DEG:
        return "over_tilt"
    return "ok"


def mapillary_pitch_roll(rvec):
    """(pitch, roll) in degrees, in the website's DB sign, from ``computed_rotation``; (None, None) unless
    :func:`mapillary_pose_status` is ``ok``.

    Richmond VA pano 2163793620710887 (the website's viewer stored 2.2318, -6.3550):

    >>> pitch, roll = mapillary_pitch_roll([1.3857263583832, 0.71804330335161, -0.58250746512038])
    >>> round(pitch, 4), round(roll, 4)
    (2.2318, -6.355)
    >>> mapillary_pitch_roll([math.pi, 0.0, 0.0])      # upside down: not a pose
    (None, None)
    """
    if mapillary_pose_status(rvec) != "ok":
        return None, None
    pose = opensfm_pose(rvec)
    return pose['pitch_deg'], pose['roll_deg']


def website_to_package(pitch, roll):
    """The website's Mapillary/Panoramax (pitch positive nose up, roll) -> the package's (pitch positive nose down,
    roll): ``(-pitch, +roll)``, the reading tests/test_pose_levelling.py measured on pixels for both sources.

    >>> website_to_package(2.0, -3.0)
    (-2.0, -3.0)
    """
    return -pitch + 0.0, roll + 0.0


# --- the per-row pose -----------------------------------------------------------------------------------------------

def pose_for_row(row, gsv_beta, mapillary_rotations=None):
    """The pose a label's crop is corrected with (module docstring's table), as a :data:`Pose`.

    ``row`` is a mapping with ``pano_source``, ``pano_id``, ``camera_pitch``, ``camera_roll`` and optionally
    ``author`` (``"ai"`` labels were placed on the rig image and get ``rig_frame``, no correction). ``gsv_beta`` is
    ``--tilt-beta``. ``mapillary_rotations`` maps a Mapillary pano_id to its Graph API ``computed_rotation`` (None
    for a pano the API has no rotation for); pass None when no lookup was made.

    >>> pose_for_row({"pano_source": "gsv", "pano_id": "a", "camera_pitch": 1.5, "camera_roll": None}, 0.95)
    Pose(camera_pitch=1.5, camera_roll=None, pose_source='gsv_db', beta=0.95, reason='db')
    >>> pose_for_row({"pano_source": "mapillary", "pano_id": "m", "camera_pitch": 2.0, "camera_roll": -6.0}, 0.95)
    Pose(camera_pitch=-2.0, camera_roll=-6.0, pose_source='mapillary_db', beta=1.0, reason='db')
    >>> pose_for_row({"pano_source": "panoramax", "pano_id": "x", "camera_pitch": 0, "camera_roll": 0}, 0.95)
    Pose(camera_pitch=None, camera_roll=None, pose_source='none', beta=None, reason='panoramax_zero_pose')
    """
    src = str(row.get("pano_source") or "").strip().lower()
    if str(row.get("author") or "human") == "ai":
        return Pose(None, None, "rig_frame", None, "ai_label")
    pitch, roll = _finite(row.get("camera_pitch")), _finite(row.get("camera_roll"))
    if src == "gsv":
        # Exactly the behaviour from before #72: the DB columns as they are (None/NaN is 0 to label_view, and an
        # all-zero pose gives no tilt matrix, so tilt_applied says whether anything was corrected).
        if pitch is None and roll is None:
            return Pose(None, None, "none", None, "gsv_no_db_pose")
        return Pose(pitch, roll, "gsv_db", gsv_beta, "db")
    if src == "mapillary":
        beta = SOURCE_BETA["mapillary"]
        if pitch is not None and roll is not None:
            return Pose(*website_to_package(pitch, roll), "mapillary_db", beta, "db")
        if mapillary_rotations is None:
            return Pose(None, None, "none", None, "mapillary_no_db_pose_api_off")
        pano_id = str(row.get("pano_id"))
        if pano_id not in mapillary_rotations:
            return Pose(None, None, "none", None, "mapillary_api_not_fetched")
        rvec = mapillary_rotations[pano_id]
        status = mapillary_pose_status(rvec)
        if status != "ok":
            return Pose(None, None, "none", None, f"mapillary_api_{status}")
        return Pose(*website_to_package(*mapillary_pitch_roll(rvec)), "mapillary_api", beta, "api")
    if src == "panoramax":
        if pitch is None and roll is None:
            return Pose(None, None, "none", None, "panoramax_no_pose")
        if not pitch and not roll:
            return Pose(None, None, "none", None, "panoramax_zero_pose")
        if not pitch or not roll:  # the viewer did not level it: the click is already in the rig frame
            return Pose(None, None, "none", None, "panoramax_single_angle_unlevelled")
        return Pose(*website_to_package(pitch, roll), "panoramax_db", SOURCE_BETA["panoramax"], "db")
    return Pose(None, None, "none", None, f"unknown_source:{src or 'blank'}")


# --- Mapillary Graph API lookup, cached ----------------------------------------------------------------------------

GRAPH_API = "https://graph.mapillary.com"
MAPILLARY_CACHE_FILE = "mapillary_rotations.json"
API_PAUSE_S = 0.1


def needs_mapillary_rotation(row):
    """True for a human Mapillary label with no complete DB pose (the rows the API lookup is for)."""
    return (str(row.get("pano_source") or "").strip().lower() == "mapillary"
            and str(row.get("author") or "human") != "ai"
            and (_finite(row.get("camera_pitch")) is None or _finite(row.get("camera_roll")) is None))


def load_rotation_cache(path):
    """``{pano_id: computed_rotation or None}`` from the cache file, or ``{}``."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v.get("computed_rotation") for k, v in json.load(f).items()}


def fetch_mapillary_rotations(pano_ids, cache_path, token, session=None, pause_s=API_PAUSE_S):
    """``{pano_id: computed_rotation or None}`` for ``pano_ids``, read from ``cache_path`` and the Graph API for the
    rest; returns ``(rotations, n_called, failed)``.

    Only answers are cached: a 200 (its ``computed_rotation``, None when it has none) or a 404 (None). Anything else
    (another status, a network error) is listed in ``failed`` and not cached, so the caller can stop before rendering
    and a re-run retries exactly those. The token is sent in a header and never written or printed; the cache holds
    only the rotation, the HTTP status and the fetch date.
    """
    import requests  # only a run with a token needs it
    raw = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
    session = session or requests.Session()
    failed, n_called = [], 0
    todo = [p for p in dict.fromkeys(str(p) for p in pano_ids) if p not in raw]
    try:
        for pano_id in todo:
            n_called += 1
            try:
                r = session.get(f"{GRAPH_API}/{pano_id}", params={"fields": "id,computed_rotation"},
                                headers={"Authorization": f"OAuth {token}"}, timeout=(10, 60))
            except Exception as e:  # noqa: BLE001 - counted, retried on the next run
                failed.append((pano_id, type(e).__name__))
                continue
            if r.status_code == 200:
                body = r.json()
                if not isinstance(body, dict) or "error" in body:
                    failed.append((pano_id, "error_body"))
                    continue
                raw[pano_id] = {"computed_rotation": body.get("computed_rotation"), "status": 200,
                                "fetched": time.strftime("%Y-%m-%d")}
            elif r.status_code == 404 and "OAuthException" not in r.text:
                raw[pano_id] = {"computed_rotation": None, "status": 404, "fetched": time.strftime("%Y-%m-%d")}
            else:
                failed.append((pano_id, f"http_{r.status_code}"))
            if pause_s:
                time.sleep(pause_s)
    finally:
        if todo:
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(raw, f, indent=1, sort_keys=True)
                f.write("\n")
            os.replace(tmp, cache_path)
    wanted = set(str(p) for p in pano_ids)
    return {k: v.get("computed_rotation") for k, v in raw.items() if k in wanted}, n_called, failed
