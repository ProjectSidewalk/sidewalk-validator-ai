"""The crop SidewalkWebpage's nightly crop job cuts, as a training-crop geometry (issue #53).

The website's crop job (``CropService.cutCrops``) cuts every label's crop from the self-hosted panorama store as a 3:2
window around the label's native pixel: no perspective projection, no tilt correction, no depth. This module ports
that geometry one to one so ``data_generation.py --job-window-crops`` can build a training set of exactly those
windows, and a model trained on it can be compared with the depth-sized one (``analysis/job_crop_compare.py``).

Ported from ProjectSidewalk/SidewalkWebpage at 5d89dc179:

* ``app/service/CropGeometry.scala`` ``CropSizingRule`` (lines 16-101): :func:`reference_crop_size`,
  :func:`predict_crop_size`, :func:`window_width` and the constants below (rule version ``v2``).
* ``app/service/CropGeometry.scala`` ``CropGeometry`` (111-226): :func:`compute_crop_box`,
  :func:`label_position_in_crop`, :func:`label_fraction_in_crop`, :func:`segments`.
* ``app/service/CropService.scala``: ``windowFor`` (177-181) is :func:`window_for`, ``storedSize`` (171-174) is
  :func:`stored_size`, ``storedCrop`` (164) is :func:`stored_crop`, and the skip rules of ``cutCrops``/``cutCrop``
  (493-560) are :func:`cut_job_crop`'s :class:`JobCropSkipped` reasons.

**Faithful on geometry, approximate on resampling.** Every box, segment and stored size is the backend's, integer for
integer: the Scala uses ``Math.rint`` (round half to even, which is Python's ``round``) and ``Math.floorMod`` (Python's
``%``) because it was itself ported from panorama-tools' ``CropRunner.py``, and ``tests/test_job_crop.py`` checks the
port against the golden fixtures that ``CropRunner.py`` wrote for the Scala suite (copied into
``tests/fixtures/job_crop/``), pixel for pixel on the cut. The one rounding that differs is ``storedSize``, which uses
Java's ``Math.round`` (half UP), reproduced as ``floor(x + 0.5)``. The downscale to at most 1440 px wide is not
byte-identical: the backend's ``ImageUtils.scaleToMaxEdge`` halves repeatedly and finishes with a bilinear draw, and
this uses one ``cv2.resize(..., INTER_AREA)``. Both average the same source pixels into the same output size; the
pixels differ in the low bits. A window at most 1440 px wide is not resampled at all and is exact; on a 16384-px pano
that is a label above about y = 4750, 650 px below the horizon (the window is 764 px wide at the horizon and reaches
the 90-degree ceiling, 4096 px, about 1200 px below it).

**What the backend skips, this skips** (with the same reason as a manifest ``status``):

* ``dims_mismatch``: the label's recorded ``pano_width``/``pano_height`` differ from the stored file's size. The label
  was placed on a frame of another size (the old scraper upscaled small panos), so the backend skips the pano's
  labels "rather than mis-centring" (CropService.scala 505-511). Missing recorded dims are cut anyway, as there.
* ``out_of_frame``: ``pano_y < 0`` or ``pano_y >= height`` (CropService.scala 538); x wraps, so it is never out.

Example::

    >>> box, (fx, fy) = window_for(8192, 4096, 16384, 8192)  # a label on the horizon of a 16384-px pano
    >>> box
    CropBox(left=7810, top=3842, width=764, height=509, shifted=False)
    >>> stored_size(box), (round(fx, 3), round(fy, 3))
    ((764, 509), (0.5, 0.499))
    >>> stored_size(window_for(8192, 6000, 16384, 8192)[0])  # near field: the 90-degree ceiling, downscaled
    (1440, 960)
"""
import csv
import math
import os
from collections import namedtuple

import numpy as np

# CropSizingRule (CropGeometry.scala 16-101). The 2013 pano-y -> distance -> size regression, fit on 6656-px-high
# panos, so its constants only mean anything in that reference space.
V1_REF_HEIGHT = 6656.0
V1_DIST_INTERCEPT = 19.80546390
V1_DIST_SLOPE = 0.01523952
V1_SIZE_COEF = 8725.6
V1_SIZE_EXP = -1.192
V1_SIZE_MIN = 50.0
V1_SIZE_MAX = 1500.0
SCALE = 2.5          # the window ships at 2.5x the regression's size
MIN_FOV_DEG = 8.0    # the scaled window is clamped as an angle, so it means the same on any pano resolution
MAX_FOV_DEG = 90.0
RULE_VERSION = "v2"  # CropSizingRule.Version; the golden fixtures are versioned with it

# CropGeometry (CropGeometry.scala 111-226) and CropService.storedCrop.
ASPECT = 1.5             # windows are 3:2, width over height
MAX_STORED_WIDTH = 1440  # a ceiling, never a target: narrower windows are stored at their own size

#: Where data_generation.py writes the per-label window facts the manifest has no column for (``<out>/<LT>/``).
SIDECAR_FILE = "job_crops.csv"
SIDECAR_COLUMNS = ["label_uid", "left", "top", "width", "height", "shifted", "fx", "fy", "stored_w", "stored_h",
                   "pano_native_w", "pano_native_h", "rule_version"]
#: The keys of :func:`crop_for_row`'s info that are manifest columns; everything else goes to the sidecar.
MANIFEST_KEYS = ("tilt_applied", "depth_m", "crop_half_px", "clipped")


class CropBox(namedtuple("CropBox", "left top width height shifted")):
    """A crop window on a panorama (``CropGeometry.CropBox``).

    ``left`` is in ``[0, pano_width)`` and the window may run past the right edge (see :func:`segments`); ``top`` is
    in ``[0, pano_height - height]``; ``shifted`` says the window moved vertically to stay inside the pano, so the
    label is off-centre in it."""
    __slots__ = ()


class JobCropSkipped(RuntimeError):
    """The backend would not cut this label's crop; ``reason`` (``dims_mismatch`` or ``out_of_frame``) is the
    manifest status."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# ---------------------------------------------------------------------------------------------------------------------
# CropSizingRule
# ---------------------------------------------------------------------------------------------------------------------

def reference_crop_size(ref_y_offset):
    """The 2013 regression in its own 6656-px-high space: crop size in reference pixels, clamped to [50, 1500].

    ``ref_y_offset`` is the label's height above the horizon in reference pixels. Far below the horizon the predicted
    distance reaches 0 and the size is the 1500 ceiling.

    >>> reference_crop_size(-2000.0)
    1500.0
    >>> round(reference_crop_size(0.0), 2)
    248.33
    """
    distance = max(0.0, V1_DIST_INTERCEPT + V1_DIST_SLOPE * ref_y_offset)
    size = V1_SIZE_COEF * distance ** V1_SIZE_EXP if distance > 0 else 0.0
    if size > V1_SIZE_MAX or distance == 0:
        return V1_SIZE_MAX
    if size < V1_SIZE_MIN:
        return V1_SIZE_MIN
    return size


def predict_crop_size(pano_y, pano_height):
    """The size the regression asks for, in this pano's native pixels (the identity conversion at height 6656)."""
    ref_offset = (pano_height / 2.0 - pano_y) * (V1_REF_HEIGHT / pano_height)
    return reference_crop_size(ref_offset) * (pano_height / V1_REF_HEIGHT)


def window_width(pano_y, pano_width, pano_height):
    """The window width to cut, in native pixels (float): :func:`predict_crop_size` x :data:`SCALE`, as an angle
    through the height, clamped to [8, 90] degrees, back to pixels through the width. Not capped to the pano:
    :func:`compute_crop_box` owns that.

    >>> window_width(2000, 16384, 8192)  # far above the horizon the 8-degree floor binds: 8 / 360 * 16384
    364.0888888888889
    >>> window_width(7000, 16384, 8192)  # near field: the 90-degree ceiling
    4096.0
    """
    deg = predict_crop_size(pano_y, pano_height) * SCALE / pano_height * 180.0
    clamped = min(max(deg, MIN_FOV_DEG), MAX_FOV_DEG)
    return clamped / 360.0 * pano_width


# ---------------------------------------------------------------------------------------------------------------------
# CropGeometry
# ---------------------------------------------------------------------------------------------------------------------

def _rint(x):
    """Scala's ``Math.rint(x).toInt``: round half to even, which is Python's ``round``."""
    return int(round(x))


def compute_crop_box(pano_x, pano_y, crop_width, pano_width, pano_height):
    """The integer 3:2 window centred on a label: x wraps at the seam, y clamps by shifting (never pads).

    The width is capped at ``pano_width`` and at ``pano_height * 1.5``. ``pano_y`` is not validated (the caller
    rejects an out-of-frame y first, :func:`cut_job_crop`).

    >>> compute_crop_box(300, 128, 248.33, 512, 256)
    CropBox(left=176, top=46, width=248, height=165, shifted=False)
    >>> compute_crop_box(0, 128, 248.33, 512, 256).left  # wraps: -124 mod 512
    388
    >>> compute_crop_box(300, 8, 248.33, 512, 256)       # shifted off the top pole
    CropBox(left=176, top=0, width=248, height=165, shifted=True)
    """
    width = _rint(min(crop_width, float(pano_width), pano_height * ASPECT))
    height = _rint(width / ASPECT)
    left = _rint(pano_x - width / 2.0) % pano_width
    ideal_top = _rint(pano_y - height / 2.0)
    top = max(0, min(ideal_top, pano_height - height))
    return CropBox(left, top, width, height, top != ideal_top)


def label_position_in_crop(pano_x, pano_y, box, pano_width):
    """Where the labelled pixel lands in the cut window, ``(x, y)`` in window pixels (floats, unchecked).

    Handles the seam (``(pano_x - left) % pano_width`` is never negative) and the vertical shift.

    >>> label_position_in_crop(20, 100, CropBox(13000, 50, 400, 267, False), 16384)
    (3404.0, 50.0)
    """
    dx = (pano_x - box.left) % pano_width
    return float(dx), float(pano_y - box.top)


def label_fraction_in_crop(pano_x, pano_y, box, pano_width):
    """:func:`label_position_in_crop` as fractions of the window, which hold for the stored file at any scale."""
    x, y = label_position_in_crop(pano_x, pano_y, box, pano_width)
    return x / box.width, y / box.height


def segments(box, pano_width):
    """The source runs that fill a window, in crop order: ``[(src_x, width, dst_x), ...]``, one run or two across the
    seam.

    >>> segments(CropBox(100, 0, 50, 33, False), 1024)
    [(100, 50, 0)]
    >>> segments(CropBox(1000, 0, 50, 33, False), 1024)
    [(1000, 24, 0), (0, 26, 24)]
    """
    if box.left + box.width <= pano_width:
        return [(box.left, box.width, 0)]
    first = pano_width - box.left
    return [(box.left, first, 0), (0, box.width - first, first)]


# ---------------------------------------------------------------------------------------------------------------------
# CropService
# ---------------------------------------------------------------------------------------------------------------------

def _java_round(x):
    """Java's ``Math.round(double)``: half UP (``floor(x + 0.5)``), unlike :func:`_rint`."""
    return int(math.floor(x + 0.5))


def stored_size(box):
    """``(w, h)`` the backend stores a window of ``box``'s size at: at most :data:`MAX_STORED_WIDTH` on the long
    edge, never upscaled, rounded half up as ``ImageUtils.scaleToMaxEdge`` rounds.

    >>> stored_size(CropBox(0, 0, 4096, 2731, False))
    (1440, 960)
    >>> stored_size(CropBox(0, 0, 1200, 800, False))
    (1200, 800)
    """
    w, h = box.width, box.height
    scale = min(1.0, MAX_STORED_WIDTH / max(w, h))
    return max(1, _java_round(w * scale)), max(1, _java_round(h * scale))


def window_for(pano_x, pano_y, pano_width, pano_height):
    """``CropService.windowFor``: the window the job cuts for a label, and the label's ``(fx, fy)`` in it."""
    width = window_width(float(pano_y), pano_width, pano_height)
    box = compute_crop_box(float(pano_x), float(pano_y), width, pano_width, pano_height)
    return box, label_fraction_in_crop(float(pano_x), float(pano_y), box, pano_width)


def cut_window(pano, box):
    """The window's pixels from a full pano array (any channel order), seam runs stitched left to right:
    ``(box.height, box.width, C)``."""
    pano_width = pano.shape[1]
    rows = pano[box.top:box.top + box.height]
    return np.concatenate([rows[:, src:src + w] for src, w, _ in segments(box, pano_width)], axis=1)


def stored_crop(window):
    """The stored form of a cut window: downscaled with ``cv2.INTER_AREA`` to :func:`stored_size`, never upscaled
    (a window already small enough is returned as is). See the module docstring on resampling parity."""
    h, w = window.shape[:2]
    size = stored_size(CropBox(0, 0, w, h, False))
    if size == (w, h):
        return window
    import cv2
    return cv2.resize(window, size, interpolation=cv2.INTER_AREA)


def _given(v):
    """A recorded dimension that is present (not None, not NaN, not empty)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def cut_job_crop(pano_bgr, pano_x, pano_y, recorded_w=None, recorded_h=None):
    """One label's job crop from a NATIVE-resolution pano array: ``(crop, info)``, the crop in ``pano_bgr``'s channel
    order and at its stored size.

    ``info``: the manifest fields ``tilt_applied`` 0 (no correction on this path), ``depth_m`` None, ``crop_half_px``
    (``box.width // 2``: the native half-width of the 3:2 window, not a view half-width) and ``clipped``
    (``int(box.shifted)``), plus the sidecar fields of :data:`SIDECAR_COLUMNS` (box, label fraction, stored size,
    native pano size, rule version).

    Raises :class:`JobCropSkipped` ``dims_mismatch`` when recorded dims are given and differ from the array's, and
    ``out_of_frame`` when ``pano_y`` is outside ``[0, height)``; the backend checks the dims first.
    """
    h, w = pano_bgr.shape[:2]
    if (_given(recorded_w) and int(float(recorded_w)) != w) or (_given(recorded_h) and int(float(recorded_h)) != h):
        raise JobCropSkipped("dims_mismatch", f"recorded {recorded_w}x{recorded_h}, stored file {w}x{h}")
    if not 0 <= float(pano_y) < h:
        raise JobCropSkipped("out_of_frame", f"pano_y {pano_y} outside [0, {h})")
    box, (fx, fy) = window_for(pano_x, pano_y, w, h)
    crop = stored_crop(cut_window(pano_bgr, box))
    info = {"tilt_applied": 0, "depth_m": None, "crop_half_px": box.width // 2, "clipped": int(box.shifted),
            "left": box.left, "top": box.top, "width": box.width, "height": box.height, "shifted": int(box.shifted),
            "fx": round(fx, 6), "fy": round(fy, 6), "stored_w": crop.shape[1], "stored_h": crop.shape[0],
            "pano_native_w": w, "pano_native_h": h, "rule_version": RULE_VERSION}
    return crop, info


_NATIVE_CACHE = {"path": None, "img": None}


def read_native_pano(path):
    """The stored pano at its native size, BGR (``cv2.imread``), with a one-entry cache keyed on ``path``:
    data_generation.py walks the selection pano by pano, so consecutive labels on one pano read it once.

    Raises ``ValueError`` if the file cannot be decoded (the store reader decoded it a moment earlier, so this is a
    real error row, not a missing pano)."""
    if _NATIVE_CACHE["path"] != path:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cv2.imread cannot decode {path}")
        _NATIVE_CACHE.update(path=path, img=img)
    return _NATIVE_CACHE["img"]


# ---------------------------------------------------------------------------------------------------------------------
# data_generation.py hook
# ---------------------------------------------------------------------------------------------------------------------

def crop_for_row(pano_path, row, label_uid, sidecar_path):
    """``data_generation.py --job-window-crops``: one selected label's job crop, ``(crop_rgb, manifest_fields)``.

    Reads the native pano at ``pano_path`` (the file the store reader just read; :func:`read_native_pano` caches it),
    cuts with the label's ``pano_x``/``pano_y`` in the recorded ``pano_width``x``pano_height`` frame (never rescaled:
    a frame of another size is ``dims_mismatch``, as in the backend), appends the window facts to the sidecar at
    ``sidecar_path`` (:func:`append_sidecar`), and returns the crop as RGB with only the :data:`MANIFEST_KEYS`.
    Raises :class:`JobCropSkipped` for a label the backend would not cut."""
    crop_bgr, info = cut_job_crop(read_native_pano(pano_path), row["pano_x"], row["pano_y"],
                                  row.get("pano_width"), row.get("pano_height"))
    append_sidecar(sidecar_path, label_uid, info)
    return np.ascontiguousarray(crop_bgr[..., ::-1]), {k: info[k] for k in MANIFEST_KEYS}


def append_sidecar(path, label_uid, info):
    """Append one label's row to the sidecar CSV (header written on first use), flushed at once so a run that dies
    keeps what it cut. A label redone later appends again; :func:`compact_sidecar` keeps the last row."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIDECAR_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"label_uid": label_uid, **info})


def compact_sidecar(path, ok_uids):
    """Rewrite the sidecar atomically with one row per label (the last one written) for exactly the labels in
    ``ok_uids`` (the manifest's ok rows after the run), in that order of first appearance. Returns the kept rows as a
    DataFrame (empty when there is no sidecar).

    Run after every generation run, so a resumed or partly redone run leaves a sidecar that matches its manifest:
    rows of labels that were redone, failed, or left the selection are dropped."""
    import pandas as pd
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=SIDECAR_COLUMNS)
    df = pd.read_csv(path, dtype={"label_uid": str})
    df = df.drop_duplicates("label_uid", keep="last")
    df = df[df["label_uid"].isin(set(ok_uids))].reset_index(drop=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, columns=SIDECAR_COLUMNS)
    os.replace(tmp, path)
    return df


def finish_run(base_dir, manifest):
    """Post-loop step of ``--job-window-crops``: compact ``<base_dir>/job_crops.csv`` to the ok rows of ``manifest``
    (the final manifest DataFrame) and print one summary line. Returns the summary dict."""
    ok_uids = manifest.loc[manifest["status"] == "ok", "label_uid"].tolist()
    side = compact_sidecar(os.path.join(base_dir, SIDECAR_FILE), ok_uids)
    status = manifest["status"].fillna("")
    summary = {"job_crops": len(side), "shifted": int(side["shifted"].astype(int).sum()) if len(side) else 0,
               "out_of_frame": int((status == "out_of_frame").sum()),
               "dims_mismatch": int((status == "dims_mismatch").sum()),
               "missing_from_sidecar": len(set(ok_uids) - set(side["label_uid"]))}
    print(f"  job-window crops (whole manifest, {os.path.join(base_dir, SIDECAR_FILE)}): {summary}")
    return summary
