"""Which sign reading of a provider's stored pose levels its panoramas? The pixel instrument behind issue #72.

``crop_pose.pose_for_row`` turns a label's stored ``camera_pitch``/``camera_roll`` into the geometry package's sign.
Whether that conversion is right is a property of the pixels: resample the stored (rig-frame) panorama into the
gravity frame with the package's own ``tilt_matrix`` at a candidate pose and measure how much its near-vertical edges
lean. Levelling with the right pose flattens the lean profile; a wrong sign adds tilt instead. All four sign readings
of the stored (pitch, roll) are tried.

The instrument is a port of the minimum of sidewalk-panorama-tools ``reports/scripts/tilt_frame.py`` (``lean_profile``,
the #54 F2 vertical-edge estimator) and ``reports/scripts/non_gsv_levelling.py`` (``fit_sinusoid``,
``level_with_pose``), except that the levelling goes through ``sidewalk_panorama_tools.geometry`` (``tilt_matrix``,
``rays_to_lonlat``), the exact transform the crop path applies, rather than the study's own ``pano_pose`` copy.

``tests/test_pose_levelling.py`` runs it on the committed fixtures. This script reproduces the wider Mapillary
measurement in docs/decisions.md (80 Richmond VA / Laurens IA panoramas with a stored pose) from
``docs/pose-levelling-2026-10/mapillary-80.json``, which records each panorama's id, stored pose, Graph API rotation,
the thumbnail's sha256 and the amplitudes measured on 2026-10-28. The imagery is not committed (Mapillary, CC BY-SA
4.0); ``fetch`` downloads the 2048-wide thumbnails with a token::

    MAPILLARY_ACCESS_TOKEN=... python analysis/pose_levelling.py fetch --images ~/Data/mly-80
    python analysis/pose_levelling.py analyze --images ~/Data/mly-80     # prints the table, checks the record

A re-served thumbnail can differ in bytes from the one measured; ``analyze`` reports how many hashes differ.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(REPO, "docs", "pose-levelling-2026-10", "mapillary-80.json")

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
    from sidewalk_panorama_tools.geometry import (lonlat_to_pano_px, lonlat_to_ray, pano_px_to_lonlat,
                                                  rays_to_lonlat, tilt_matrix)
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


def measure(gray, camera_pitch, camera_roll):
    """``(amplitude before, {reading: amplitude after levelling with (sp * camera_pitch, sr * camera_roll)})``."""
    before = lean_amplitude(lean_profile(gray))
    after = {(sp, sr): lean_amplitude(lean_profile(level(gray, sp * camera_pitch, sr * camera_roll)))
             for sp, sr in READINGS}
    return before, after


def summarise(rows):
    """``rows`` of (before, {reading: after}) -> {n, before_median, after_median {reading}, best {reading: n}}."""
    return {"n": len(rows), "before_median": float(np.median([b for b, _ in rows])),
            "after_median": {f"{sp:+d},{sr:+d}": float(np.median([a[(sp, sr)] for _, a in rows]))
                             for sp, sr in READINGS},
            "best": {f"{sp:+d},{sr:+d}": sum(min(a, key=a.get) == (sp, sr) for _, a in rows) for sp, sr in READINGS}}


def _fetch(args):
    import requests
    token = (os.environ.get("MAPILLARY_ACCESS_TOKEN") or "").strip()
    if not token:
        sys.exit("MAPILLARY_ACCESS_TOKEN is unset")
    os.makedirs(args.images, exist_ok=True)
    s = requests.Session()
    for p in json.load(open(args.record))["panoramas"]:
        dest = os.path.join(args.images, f"{p['pano_id']}.jpg")
        if os.path.exists(dest):
            continue
        r = s.get(f"https://graph.mapillary.com/{p['pano_id']}", params={"fields": "thumb_2048_url"},
                  headers={"Authorization": f"OAuth {token}"}, timeout=60)
        r.raise_for_status()
        img = s.get(r.json()["thumb_2048_url"], timeout=60)
        img.raise_for_status()
        with open(dest, "wb") as f:
            f.write(img.content)
        time.sleep(0.3)


def _analyze(args):
    rec = json.load(open(args.record))
    rows, rows5, changed, drift = [], [], 0, 0.0
    for p in rec["panoramas"]:
        path = os.path.join(args.images, f"{p['pano_id']}.jpg")
        with open(path, "rb") as f:
            changed += hashlib.sha256(f.read()).hexdigest() != p["sha256"]
        before, after = measure(load_gray(path), p["camera_pitch"], p["camera_roll"])
        rows.append((before, after))
        if math.hypot(p["camera_pitch"], p["camera_roll"]) >= 5:
            rows5.append((before, after))
        drift = max(drift, abs(before - p["amp_before"]))
    print(json.dumps({"all": summarise(rows), "tilt>=5": summarise(rows5)}, indent=1))
    print(f"{changed} of {len(rows)} thumbnails differ in bytes from the record; max |amp_before - recorded| "
          f"{drift:.3f} deg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--images", required=True, help="directory of <pano_id>.jpg thumbnails")
        sp.add_argument("--record", default=RECORD)
    args = ap.parse_args(argv)
    (_fetch if args.cmd == "fetch" else _analyze)(args)


if __name__ == "__main__":
    main()
