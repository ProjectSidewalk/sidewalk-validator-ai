"""The "more context" validator arms of issue #14: three ways to show the DINOv2 classifier more than the tight crop.

=================  ==========================================================================  =====================
arm                input                                                                       head input
=================  ==========================================================================  =====================
``two_scale``      the tight crop and the wide crop (``context_crops.py``, 4x the half-width),  ``4 x hidden``
                   one shared backbone, ``[CLS, mean(patches)]`` of each, concatenated
``view_heatmap``   the whole label-centred view (448 px stored, ``--context-view-res`` 224 by   ``2 x hidden``
                   default) plus a 4th channel: a Gaussian at the label point whose width is
                   the tight crop's extent
``side_info``      the tight crop plus a normalised vector of non-image features from          ``2 x hidden + n``
                   split.csv (severity, tag count, imagery age, log depth, high-quality user,
                   clipped, and missing-value flags)
=================  ==========================================================================  =====================

Every arm pools like ``Dinov2ForImageClassification`` (``[CLS, mean of the patch tokens]``) and trains with the same
recipe as round 3 through ``fine_tune_classifier.py --context-arm <arm>``.

**Why the heatmap encodes scale.** At evaluation the label is always at the view centre, so a fixed-width Gaussian
would be the same image for every label: a constant channel the model can only ignore. Its width is the tight crop's
half-width mapped into the input (``heatmap_sigma``: half of the crop's half-width, at least 4 px), so the channel
says both *where* the label is and *how big* the depth-sized crop around it is. The issue's wording was "a Gaussian
at the label position"; the plain point is ``heatmap_sigma(..., floor=s)`` with ``crop_half_px=0``.

**Why the view skips the centre crop.** The checkpoint's processor resizes to 256 and centre-crops 224, which drops the
outer 12.5% of each side: for a whole-view input that is exactly the context the arm is about. The view arm resizes to
``res`` x ``res`` instead (train: ``RandomResizedCrop`` with the heatmap mapped through the same crop and flip).

**The 4-channel patch embedding** copies the pretrained RGB filters into channels 0-2 and zero-initialises channel 3,
so the model computes exactly the pretrained function at step 0 and learns to use the heatmap.

torch, torchvision and transformers are imported inside functions and the classes are built by ``build_classes()``, so
this module imports on the CPU-only CI. The model is never registered with ``AutoModel`` (``model_type``
``context_dinov2``): ``evaluate.py``, ``publish_to_hf.py`` and sidewalk-ai-api cannot load it by accident. Score a
checkpoint with this module, then let ``evaluate.py`` compute the metrics::

    python context_model.py predict --label_type NoCurbRamp --data-dir D --checkpoint RUN/checkpoint-N --tag view_heatmap
    python evaluate.py --label_type NoCurbRamp --data-dir D --skip-inference --tag view_heatmap --per-city

Pattern: ``region_model.py`` (issue #33, PR #88).
"""
import numpy as np
import pandas as pd

ARMS = ("two_scale", "view_heatmap", "side_info")
MODEL_TYPE = "context_dinov2"
SIDE_FEATURES = ["severity", "n_tags", "age_years", "log_depth_m", "high_quality_user", "clipped"]
SIDE_MISSING = ["severity_missing", "age_missing", "depth_missing"]
SIDE_COLUMNS = SIDE_FEATURES + SIDE_MISSING
#: split.csv columns each arm reads (besides image and label), for ``fine_tune_classifier.extra_frame_columns``.
FRAME_COLUMNS = {
    "two_scale": ("label_uid", "split", "crop_half_px"),
    "view_heatmap": ("label_uid", "split", "crop_half_px"),
    "side_info": ("label_uid", "split", "severity", "tags", "image_capture_date", "depth_m", "high_quality_user",
                  "clipped"),
}
VIEW_SIZE = 2048            # data_generation.VIEW_SIZE (not imported: that module pulls in the generation stack)
HEATMAP_SIGMA_FLOOR = 4.0   # px at the model input
TRAIN_SCALE = (0.7, 1.0)    # fine_tune_classifier.py's RandomResizedCrop scale
TRAIN_RATIO = (3 / 4, 4 / 3)


# ---------------------------------------------------------------------------------------------------------------------
# Side information (pure pandas)
# ---------------------------------------------------------------------------------------------------------------------

def _capture_month(values):
    """``image_capture_date`` (``YYYY-MM`` or a longer ISO date) as a Timestamp of the month; NaT when unparseable."""
    return pd.to_datetime(pd.Series(values).astype(str).str.slice(0, 7), format="%Y-%m", errors="coerce")


def age_reference(df):
    """The reference date for ``age_years``: the latest ``image_capture_date`` in ``df`` (``YYYY-MM``), or None.

    >>> age_reference(pd.DataFrame({"image_capture_date": ["2019-04", None, "2024-09"]}))
    '2024-09'
    """
    d = _capture_month(df["image_capture_date"]).max()
    return None if pd.isna(d) else d.strftime("%Y-%m")


def _flag(v):
    return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0


def side_feature_frame(df, reference_date=None):
    """The raw side features of each row of ``df`` (split.csv columns), as ``SIDE_COLUMNS``.

    ``severity`` as a float; ``n_tags`` the number of tags (``analysis.slices.parse_tags``: JSON list or ``;``-joined);
    ``age_years`` = (``reference_date`` - ``image_capture_date``) in years (``reference_date`` defaults to
    ``age_reference(df)``); ``log_depth_m`` = log of a positive ``depth_m``; ``high_quality_user`` and ``clipped`` as
    0/1 (missing -> 0); and a 0/1 ``*_missing`` flag for severity, age and depth, whose own values stay NaN here
    (``normalise`` fills them with the training median).

    >>> f = side_feature_frame(pd.DataFrame({"severity": [3, None], "tags": ['["a","b"]', None],
    ...     "image_capture_date": ["2020-09", "2024-09"], "depth_m": [10.0, 0], "high_quality_user": [True, None],
    ...     "clipped": [0, 1]}))
    >>> f[["severity", "n_tags", "age_years", "high_quality_user", "clipped"]].values.tolist()
    [[3.0, 2.0, 4.0, 1.0, 0.0], [nan, 0.0, 0.0, 0.0, 1.0]]
    >>> f[SIDE_MISSING].values.tolist()
    [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    """
    from analysis.slices import parse_tags
    ref = pd.Timestamp(reference_date or age_reference(df) or "2000-01")
    out = pd.DataFrame(index=df.index)
    out["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype(float)
    out["n_tags"] = [float(len(parse_tags(t))) for t in df["tags"]]
    cap = _capture_month(df["image_capture_date"]).set_axis(df.index)
    out["age_years"] = ((ref - cap).dt.days / 365.25).round(6)
    depth = pd.to_numeric(df["depth_m"], errors="coerce").astype(float)
    out["log_depth_m"] = np.log(depth.where(depth > 0))
    out["high_quality_user"] = [_flag(v) for v in df["high_quality_user"]]
    out["clipped"] = [_flag(v) for v in df["clipped"]]
    out["severity_missing"] = out["severity"].isna().astype(float)
    out["age_missing"] = out["age_years"].isna().astype(float)
    out["depth_missing"] = out["log_depth_m"].isna().astype(float)
    return out[SIDE_COLUMNS]


def fit_normaliser(feats):
    """Per-column ``median`` (NaN fill), ``mean`` and ``std`` (floored at 1e-6) of the training rows' side features,
    as JSON-ready lists in ``SIDE_COLUMNS`` order. A column with no value at all gets median 0.

    >>> s = fit_normaliser(pd.DataFrame({c: [1.0, 3.0] for c in SIDE_COLUMNS}))
    >>> s["median"][0], s["mean"][0], s["std"][0]
    (2.0, 2.0, 1.0)
    """
    x = feats[SIDE_COLUMNS].to_numpy(dtype=float)
    med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isnan(x), med, x)
    return {"columns": list(SIDE_COLUMNS), "median": [float(v) for v in med],
            "mean": [float(v) for v in filled.mean(axis=0)],
            "std": [float(max(v, 1e-6)) for v in filled.std(axis=0)]}


def normalise(feats, stats):
    """``float32`` array: NaN -> the training median, then ``(x - mean) / std``."""
    if list(stats["columns"]) != list(SIDE_COLUMNS):
        raise ValueError(f"side-feature columns differ: {stats['columns']} vs {SIDE_COLUMNS}")
    x = feats[SIDE_COLUMNS].to_numpy(dtype=float)
    x = np.where(np.isnan(x), np.asarray(stats["median"]), x)
    return ((x - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# The point heatmap (numpy)
# ---------------------------------------------------------------------------------------------------------------------

def heatmap(h, w, cx, cy, sigma):
    """An ``h`` x ``w`` float32 Gaussian with peak 1.0 at pixel index ``(cx, cy)``; ``sigma`` in px, a number or
    ``(sigma_x, sigma_y)`` (a non-square training crop stretches it).

    >>> m = heatmap(9, 9, 4, 4, 2.0)
    >>> float(m[4, 4]), round(float(m[4, 6]), 4), bool(m[4, 2] == m[4, 6] == m[2, 4])
    (1.0, 0.6065, True)
    """
    sx, sy = (sigma, sigma) if np.isscalar(sigma) else sigma
    x = (np.arange(w, dtype=np.float32) - cx) / sx
    y = (np.arange(h, dtype=np.float32) - cy) / sy
    return np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2)).astype(np.float32)


def heatmap_sigma(crop_half_px, res, view_size=VIEW_SIZE, floor=HEATMAP_SIGMA_FLOOR):
    """The heatmap's sigma in model-input px for a view resized to ``res``: half of the tight crop's half-width
    (``crop_half_px`` view px), at least ``floor``.

    >>> heatmap_sigma(238, 224), heatmap_sigma(20, 224), heatmap_sigma(238, 448)
    (13.015625, 4.0, 26.03125)
    """
    return max(floor, float(crop_half_px) * res / view_size / 2)


def map_point(cx, cy, crop, res, flip=False):
    """Where pixel index ``(cx, cy)`` of an image lands after ``torchvision``'s ``resized_crop(i, j, h, w) -> res``
    and an optional horizontal flip, in pixel index coordinates (pixel centres are mapped, hence the half pixels).

    >>> map_point(10, 10, (0, 0, 20, 20), 20), map_point(10, 10, (0, 0, 20, 20), 20, flip=True)
    ((10.0, 10.0), (9.0, 10.0))
    >>> map_point(13.5, 9.5, (4, 8, 8, 8), 16)
    (11.5, 11.5)
    """
    i, j, h, w = crop
    x = (cx + 0.5 - j) * res / w - 0.5
    y = (cy + 0.5 - i) * res / h - 0.5
    return (res - 1 - x if flip else x), y
