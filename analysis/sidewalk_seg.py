"""Sidewalk segmentation as a localisation prior for Obstacle / SurfaceProblem: first experiment of issue #44.

The crop classifier sees a few metres of pixels around the label point and nothing that says whether the point is on
the pedestrian path. This script runs an off-the-shelf street-scene segmenter on the 2048x2048 label-centred
perspective views the training crops are cut from, derives point-vs-sidewalk features from the mask, and measures
what they say about the human verdict, alone and on top of the production model's frozen features. It is a feature
or a routing signal, never a gate. Offline, existing crops, no retraining.

Stages (each one writes to disk and can be re-run on its own):

``filelist`` (Mac)
    From the held-out ``sample.csv``: the Obstacle / SurfaceProblem rows as ``labels.csv`` (the manifest fields the
    render needs) and ``stage_files.txt``, the unique store-relative pano paths ``<city>/<pano_id[:2]>/<pano_id>.jpg``
    that ``analysis/seg_prior_stage.slurm`` pulls from the pano store in one rsync connection.
``render-segment`` (klone GPU job body, ``analysis/seg_prior.slurm``)
    Per label, exactly as ``data_generation.py`` rendered it: ``read_store_panorama`` (8192x4096 BGR), the view from
    ``render_label_view`` with the label's ``camera_pitch``/``camera_roll``, ``tilt_beta`` and ``apply_tilt =
    (author == "human")``, on the GPU torch path. The crop is re-cut from that view at the manifest's ``depth_m``
    with ``depth_sized_crop`` and compared with the training crop (mean absolute difference, so WebP noise shows as
    about 1 grey level and a wrong view as tens). The view, downscaled to ``SEG_SIZE`` (1024), goes through
    the segmenter; the per-pixel class ids are saved as 8-bit PNGs at 1024 (features) and 512 (sheets), with a
    512 JPEG of the view. Pano coordinates (``pano_x``... ``pano_height``) come from the retrain's rawLabels
    export, joined on ``label_uid``. Writes ``views.csv`` (one row per label: status, crop check, timings) and
    ``seg_run.json`` (model, revision, class list, versions, throughput).
``features`` (Mac)
    Per label, from the 1024 mask, the label point (the view centre) and the manifest's ``depth_m``:
    :func:`mask_features`. Writes one PRIVATE per-label CSV; keep it under ``~/Data``.
``cls`` (Mac, MPS)
    The production validators' frozen pooled vector (DINOv2 ``[CLS ; mean(patch tokens)]``, the 2048-d input of
    the classifier head) and their ``p_incorrect`` for every sample crop, preprocessed as ``evaluate.py`` does.
``report`` (Mac)
    Segmenter sanity, each feature's AUC alone (incorrect = positive, bootstrap CI), the lift of the features over
    the frozen production features under 5-fold CV grouped by panorama, per type / split / public city, and the
    production-verdict strata. Aggregates only; public cities named, private ones pooled.
``sheets`` (Mac)
    Contact sheets for the segmentation hand-check (public deployments only), written outside the repo, plus one
    small sample sheet for the doc.

**Mask geometry.** The label point is the view centre. ``f = 1024`` px (90 degree horizontal FOV over 2048 px),
so a lateral offset of ``d`` view pixels at range ``R`` is about ``d * R / f`` metres (small angle, and the nearest
sidewalk pixel is taken to be at the label's range, which overstates the distance for sidewalk nearer the camera and
understates it for sidewalk beyond the label). ``R`` is the **ground range** implied by the view pitch for a camera
``CAMERA_HEIGHT_M`` (2.5 m) above flat ground, ``R = h / sin(-pitch)`` with the pitch clamped at ``MIN_GROUND_PITCH_DEG``
(-2 degrees, so ``R <= 72`` m): :func:`ground_range_m`. It assumes the label point is on the ground, which holds for
surface problems and for the base of an obstacle, not for a point clicked on an obstacle's body: a point above the ground is seen at a shallower
depression angle, so ``R`` is then too LONG (camera 2.5 m up, obstacle 10 m away, click 1 m up: true range ~10.1 m,
formula 16.9 m), the metres are too large and ``near_sidewalk_1m`` is biased toward False for Obstacle. The manifest's Depth Anything (vkitti) ``depth_m`` gives a second, depth-based distance column; on these
views it is about 2.5-2.7 times the ground range, so the depth-based metres are that much larger. The view is
gravity-levelled and roll-free (``sidewalk_panorama_tools.geometry.label_view``), so the horizon is the row
``1024 + f * tan(pitch)``, where ``pitch = (0.5 - pano_y / pano_height) * 180`` degrees.

**Classes** (Mapillary Vistas v1.2, 65 classes, as the segmenter's ``id2label`` names them): ``SIDEWALK_CLASSES``,
``ROAD_CLASSES`` and ``CURB_CLASSES`` below.

**Privacy.** ``labels.csv``, ``views.csv``, the masks, the features CSV and the sheets hold label-level rows from
private deployments: they stay under ``~/Data`` (Mac) or the scrubbed scratch dir (klone). ``report`` names a city
only through ``calibration.per_city.public_name``; ``sheets`` draws from public cities only.

**Dependencies.** numpy, pandas, scikit-learn, opencv and pillow (all in CI) for the pure parts; ``render-segment``
and ``cls`` also need torch and transformers (``requirements.txt``) and ``sidewalk-panorama-tools``.

Usage (``$R`` = ``~/Data/sidewalk-validator-ai-retrain-2026-10/round3``, ``$O`` = ``~/Data/sidewalk-validator-ai-seg-2026-10``,
``$P`` = ``~/Data/sidewalk-validator-ai-prod-2026-10-17``)::

    python analysis/sidewalk_seg.py filelist --sample $R/sample.csv --out-dir $O
    # klone: see analysis/seg_prior_stage.slurm and analysis/seg_prior.slurm, then pull views.csv, seg_run.json,
    # masks_1024/, masks_512/, views_512/ into $O/klone. The 2026-10-21 run's views.csv predates the pitch column, so
    # the coordinates were also extracted there (a CPU sbatch job; rawLabels holds the private deployments too):
    #   python analysis/sidewalk_seg.py coords --labels $W/labels.csv --rawlabels-dir $RAWLABELS --out $W/coords.csv
    python analysis/sidewalk_seg.py features --labels $O/labels.csv --run-dir $O/klone --coords $O/coords.csv \\
        --out $O/features.csv
    python analysis/sidewalk_seg.py cls --labels $O/labels.csv --crops-dir $R/crops --out $O/cls.npz
    python analysis/sidewalk_seg.py report --features $O/features.csv --cls $O/cls.npz \\
        --assessments $P/ai_assessments_rounded4.csv --cities-json $P/cities.json --run-dir $O/klone \\
        --out-dir docs/segmentation-prior-2026-10
    python analysis/sidewalk_seg.py sheets --features $O/features.csv --run-dir $O/klone \\
        --cities-json $P/cities.json --out-dir $O/sheets --doc-fig docs/segmentation-prior-2026-10/sample_views.png
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

LABEL_TYPES = ("Obstacle", "SurfaceProblem")
VIEW_SIZE = 2048          # data_generation.VIEW_SIZE
F_PX = 1024.0             # focal length in view pixels: 90 degree FOV over 2048 px
SEG_SIZE = 1024           # the view is downscaled to this before segmentation; features use the mask at this size
SHEET_SIZE = 512          # small mask + view JPEG for the contact sheets
DISC_RADIUS_PX = 15       # view pixels, for the disc-majority version of point_on_sidewalk
BOTTOM_FRAC = 0.10        # connected_to_bottom: the sidewalk component reaches the bottom 10% of the view
NEAR_M = 1.0              # near_sidewalk_1m
SEG_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

CAMERA_HEIGHT_M = 2.5     # GSV camera height above the ground, for the pitch-implied ground range
MIN_GROUND_PITCH_DEG = -2.0  # pitch clamp for the ground range (labels at or above the horizon get 72 m)

#: Sidewalk = the walkable-surface classes. Ground objects (Manhole, Catch Basin, Pothole) are NOT counted as
#: sidewalk even when they sit in one, so a label on a manhole cover in the sidewalk reads as "not on sidewalk";
#: :func:`mask_features` returns the centre class so those can be counted.
SIDEWALK_CLASSES = ("Sidewalk", "Curb Cut", "Pedestrian Area")
GROUND_OBJECT_CLASSES = ("Manhole", "Catch Basin", "Pothole")
ROAD_CLASSES = ("Road", "Service Lane", "Bike Lane", "Parking", "Crosswalk - Plain", "Lane Marking - Crosswalk",
                "Lane Marking - General")
CURB_CLASSES = ("Curb",)

#: Features measured in report: (column, kind). Booleans are scored as 0/1.
FEATURES = (
    ("point_on_sidewalk", "bool"),
    ("point_on_sidewalk_disc", "bool"),
    ("sidewalk_frac_disc", "frac"),
    ("dist_px_to_sidewalk", "dist"),
    ("dist_m_to_sidewalk", "dist"),
    ("dist_m_depth_to_sidewalk", "dist"),
    ("near_sidewalk_1m", "bool"),
    ("sidewalk_frac_crop", "frac"),
    ("sidewalk_frac_view", "frac"),
    ("road_frac_crop", "frac"),
    ("curb_frac_crop", "frac"),
    ("on_road", "bool"),
    ("connected_to_bottom", "bool"),
)
#: A missing distance (no sidewalk in the view) is scored as this far: the view diagonal, and its metres at the
#: largest range each column can take (72 m ground range; 80 m, Depth Anything's max_depth).
NO_SIDEWALK_DIST = {"dist_px_to_sidewalk": VIEW_SIZE * math.sqrt(2),
                    "dist_m_to_sidewalk": VIEW_SIZE * math.sqrt(2) * 72.0 / F_PX,
                    "dist_m_depth_to_sidewalk": VIEW_SIZE * math.sqrt(2) * 80.0 / F_PX}


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def store_tail(city, pano_id):
    """The pano store's relative path for a panorama, as ``data_generation.pano_store_path`` lays it out.

    >>> store_tail("seattle-wa", "abcDEF")
    'seattle-wa/ab/abcDEF.jpg'
    """
    return f"{city}/{pano_id[:2]}/{pano_id}.jpg"


def view_pitch_deg(y_norm):
    """Pitch of the label-centred view in degrees (negative = below the horizon), as ``label_view`` aims it.

    >>> round(view_pitch_deg(0.5 + 20 / 180), 6)
    -20.0
    """
    return (0.5 - float(y_norm)) * 180.0


def horizon_row(pitch_deg, size=VIEW_SIZE, f_px=F_PX):
    """Row of the horizon in a level, roll-free pinhole view of ``size`` px aimed at ``pitch_deg``.

    A ray at elevation 0 projects to ``y = size / 2 + f * tan(pitch)`` in every column, so looking 20 degrees down
    puts the horizon 373 px above the centre:

    >>> round(horizon_row(-20.0))
    651
    """
    return size / 2 + f_px * math.tan(math.radians(pitch_deg))


def ground_range_m(pitch_deg, camera_h_m=CAMERA_HEIGHT_M, min_pitch_deg=MIN_GROUND_PITCH_DEG):
    """Slant range in metres to a ground point seen ``pitch_deg`` below the horizon from ``camera_h_m`` above flat
    ground: ``h / sin(-pitch)``, with the pitch clamped at ``min_pitch_deg`` (points at or above the horizon cannot be
    on the ground; they get the clamp's ~72 m).

    >>> round(ground_range_m(-30.0), 3), round(ground_range_m(5.0), 1)
    (5.0, 71.6)
    """
    p = min(float(pitch_deg), min_pitch_deg)
    return camera_h_m / math.sin(math.radians(-p))


def px_to_m(dist_px, depth_m, f_px=F_PX):
    """Lateral metres for ``dist_px`` view pixels at ``depth_m`` (small angle, same depth): ``dist_px * depth / f``.

    >>> px_to_m(102.4, 10.0)
    1.0
    """
    return float(dist_px) * float(depth_m) / f_px


def class_ids(id2label, names):
    """The integer ids of ``names`` in a segmenter's ``id2label`` (keys may be strings); every name must exist.

    >>> class_ids({"0": "Road", "1": "Sidewalk"}, ("Sidewalk",))
    [1]
    """
    lookup = {v: int(k) for k, v in id2label.items()}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"segmenter has no class {missing}")
    return sorted(lookup[n] for n in names)


def mask_features(mask, sidewalk_ids, road_ids, curb_ids, crop_half_px, depth_m, pitch_deg, view_size=VIEW_SIZE,
                  f_px=F_PX, disc_radius_px=DISC_RADIUS_PX, bottom_frac=BOTTOM_FRAC, near_m=NEAR_M,
                  camera_h_m=CAMERA_HEIGHT_M):
    """Point-vs-sidewalk features of one label from a square class-id ``mask`` of its label-centred view.

    The mask may be any size ``m``; distances and the crop square are given in view pixels (``view_size``) and
    scaled by ``m / view_size``. The label point is the mask's centre pixel ``(m // 2, m // 2)``.

    Returns a dict:

    * ``sidewalk_any``: any sidewalk pixel in the view.
    * ``point_on_sidewalk``: the centre pixel is sidewalk. ``sidewalk_frac_disc`` is the sidewalk share of the
      ``disc_radius_px`` disc around it and ``point_on_sidewalk_disc`` is that share > 0.5.
    * ``dist_px_to_sidewalk``: view pixels from the centre to the nearest sidewalk pixel (0 if on it, NaN if there is
      none); ``dist_m_to_sidewalk`` the same in metres at the pitch-implied ground range ``ground_range_m``
      (:func:`ground_range_m`, :func:`px_to_m`); ``dist_m_depth_to_sidewalk`` the same at the manifest's ``depth_m``
      (Depth Anything vkitti, same-depth approximation); ``near_sidewalk_1m`` is ``dist_m_to_sidewalk <= near_m``
      (False without sidewalk).
    * ``sidewalk_frac_crop``, ``road_frac_crop``, ``curb_frac_crop``: shares inside the crop square (half-width
      ``crop_half_px`` view px, clipped to the view); ``sidewalk_frac_view`` over the whole view.
    * ``on_road``: the centre pixel is a road-class pixel.
    * ``connected_to_bottom``: the 8-connected sidewalk component that contains the centre, or else holds the
      nearest sidewalk pixel, reaches the bottom ``bottom_frac`` of the view (a "the path continues to the camera"
      proxy). False without sidewalk.
    * ``sidewalk_above_horizon_frac``: share of the sidewalk pixels above :func:`horizon_row` (NaN without sidewalk);
      a sanity check, since sidewalk can only be below the horizon.
    * ``centre_class``: the class id at the centre.

    Example: a sidewalk band across the bottom quarter of a 64 px mask, label in the middle::

        >>> m = np.zeros((64, 64), np.uint8); m[48:, :] = 1
        >>> f = mask_features(m, [1], [2], [3], crop_half_px=256, depth_m=10.0, pitch_deg=-30.0)
        >>> f["point_on_sidewalk"], f["dist_px_to_sidewalk"], round(f["dist_m_to_sidewalk"], 3), f["dist_m_depth_to_sidewalk"]
        (False, 512.0, 2.5, 5.0)
    """
    import cv2
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square 2-D, got {mask.shape}")
    m = mask.shape[0]
    s = m / view_size  # mask px per view px
    c = m // 2
    sw = np.isin(mask, sidewalk_ids)
    road = np.isin(mask, road_ids)
    curb = np.isin(mask, curb_ids)
    rng_m = ground_range_m(pitch_deg, camera_h_m)
    out = {"centre_class": int(mask[c, c]), "sidewalk_any": bool(sw.any()), "ground_range_m": rng_m,
           "point_on_sidewalk": bool(sw[c, c]), "on_road": bool(road[c, c])}

    yy, xx = np.ogrid[:m, :m]
    r = disc_radius_px * s
    disc = (yy - c) ** 2 + (xx - c) ** 2 <= max(r, 0.5) ** 2
    out["sidewalk_frac_disc"] = float(sw[disc].mean())
    out["point_on_sidewalk_disc"] = out["sidewalk_frac_disc"] > 0.5

    nearest = None
    if sw.any():
        pts = np.argwhere(sw)
        d2 = ((pts - c) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        nearest = tuple(pts[i])
        dist_px = math.sqrt(float(d2[i])) / s
        out["dist_px_to_sidewalk"] = dist_px
        out["dist_m_to_sidewalk"] = px_to_m(dist_px, rng_m, f_px)
        out["dist_m_depth_to_sidewalk"] = px_to_m(dist_px, depth_m, f_px)
        out["near_sidewalk_1m"] = out["dist_m_to_sidewalk"] <= near_m
    else:
        out["dist_px_to_sidewalk"] = float("nan")
        out["dist_m_to_sidewalk"] = float("nan")
        out["dist_m_depth_to_sidewalk"] = float("nan")
        out["near_sidewalk_1m"] = False

    half = crop_half_px * s
    y0, y1 = max(0, int(round(c - half))), min(m, int(round(c + half)))
    box = (slice(y0, max(y1, y0 + 1)), slice(y0, max(y1, y0 + 1)))  # the view is square, so x bounds = y bounds
    out["sidewalk_frac_crop"] = float(sw[box].mean())
    out["road_frac_crop"] = float(road[box].mean())
    out["curb_frac_crop"] = float(curb[box].mean())
    out["sidewalk_frac_view"] = float(sw.mean())

    if nearest is None:
        out["connected_to_bottom"] = False
        out["sidewalk_above_horizon_frac"] = float("nan")
    else:
        _, lab = cv2.connectedComponents(sw.astype(np.uint8), connectivity=8)
        comp = lab[nearest]
        bottom = m - max(1, int(math.ceil(bottom_frac * m)))
        out["connected_to_bottom"] = bool((lab[bottom:] == comp).any())
        h = horizon_row(pitch_deg, view_size, f_px) * s
        above = int(min(max(math.floor(h), 0), m))
        out["sidewalk_above_horizon_frac"] = float(sw[:above].sum() / sw.sum())
    return out


def production_pool(ai_result, h_dis):
    """Production-verdict stratum of one assessed label: ``ai_disagree``, ``contested_agree`` (AI Agree with at least
    one human Disagree), ``uncontested_agree``, or ``other`` (anything else, e.g. no assessment).

    >>> [production_pool(a, d) for a, d in [("Disagree", 0), ("Agree", 2), ("Agree", 0), (None, 0)]]
    ['ai_disagree', 'contested_agree', 'uncontested_agree', 'other']
    """
    if ai_result == "Disagree":
        return "ai_disagree"
    if ai_result == "Agree":
        try:
            n = float(h_dis)
        except (TypeError, ValueError):
            n = float("nan")
        if not math.isfinite(n):
            return "other"
        return "contested_agree" if n >= 1 else "uncontested_agree"
    return "other"


def schema_label_uid(schema, label_id, cities):
    """``<city_id>:<label_id>`` for a production row keyed ``schema`` = ``sidewalk_<city>``, or None if the schema's
    city part matches no listed city (:func:`calibration.per_city.match_city_id`).

    >>> schema_label_uid("sidewalk_seattle", 12, {"seattle-wa": "public"})
    'seattle-wa:12'
    >>> schema_label_uid("sidewalk_nowhere", 12, {"seattle-wa": "public"}) is None
    True
    """
    from calibration.per_city import match_city_id
    city = match_city_id(str(schema).removeprefix("sidewalk_"), cities)
    return None if city is None else f"{city}:{int(label_id)}"


def display_cities(cities_series, cities):
    """Map city_ids to what a table may show: the public city_id, or the pooled private label (with its count of
    distinct private deployments). Uses ``calibration.per_city.public_name``; nothing else decides.

    >>> display_cities(pd.Series(["a-wa", "b-wa", "c-wa"]), {"a-wa": "public", "b-wa": "private", "c-wa": "private"}).tolist()
    ['a-wa', 'other (2 private deployments)', 'other (2 private deployments)']
    """
    from calibration.per_city import private_label, public_name
    pub = cities_series.map(lambda c: public_name(c, cities))
    n_private = cities_series[pub.isna()].nunique()
    return pub.fillna(private_label(n_private))


def oriented_auc(y, score):
    """``(auc, direction)``: AUC of ``score`` for ``y == 1`` and ``"+"`` when higher means more likely incorrect,
    ``"-"`` when lower does (AUC < 0.5). NaN when ``y`` has one class or ``score`` is constant.

    >>> oriented_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    (1.0, '+')
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    if len(np.unique(y)) < 2 or np.nanstd(score) == 0:
        return float("nan"), ""
    auc = float(roc_auc_score(y, score))
    return auc, "+" if auc >= 0.5 else "-"


def bootstrap_auc(y, score, n_boot=2000, seed=0, other=None):
    """Percentile 95% CI of the AUC by resampling rows; with ``other`` (a second score on the same rows), the CI of
    ``AUC(score) - AUC(other)`` (paired). Resamples with one class are skipped. Returns ``(lo, hi)``.

    >>> bootstrap_auc([0, 1] * 50, [0.2, 0.8] * 50, n_boot=200)   # perfectly separated
    (1.0, 1.0)
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    a = np.asarray(score, float)
    b = None if other is None else np.asarray(other, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        v = roc_auc_score(y[i], a[i])
        if b is not None:
            v -= roc_auc_score(y[i], b[i])
        stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def within_city_auc(y, score, city, min_n=10, weight="pairs"):
    """City-stratified AUC: the per-city AUCs of ``score`` for ``y == 1``, averaged over cities with both classes and
    at least ``min_n`` rows. Returns ``(auc, n_cities)``; NaN if no city qualifies. Unlike the pooled AUC it cannot be
    raised by a score that only tells cities with many incorrect labels from cities with few.

    ``weight="pairs"`` (the reported number) weights each city by its ``n_pos * n_neg`` comparable pairs: the
    stratified Mann-Whitney estimator, i.e. the share of within-city (incorrect, correct) pairs ranked correctly. A
    lopsided city (7 incorrect of 58) then counts for its few pairs, not its many rows. ``weight="rows"`` weights by
    city row count (kept as a second column).

    >>> y = [0, 1] * 10 + [1] * 10; s = list(range(20)) + [100] * 10; c = ["a"] * 20 + ["b"] * 10
    >>> within_city_auc(y, s, c)   # city b has one class, so only city a counts
    (0.55, 1)
    """
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(score, float), "c": np.asarray(city)})
    aucs, ws = [], []
    for _, g in d.groupby("c"):
        if len(g) < min_n or g.y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(g.y, g.s))
        n_pos = int(g.y.sum())
        ws.append(len(g) if weight == "rows" else n_pos * (len(g) - n_pos))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=ws)), len(aucs)


def city_bootstrap_auc(y, score, city, n_boot=2000, seed=0, other=None):
    """Percentile 95% CI of the pooled AUC (or, with ``other``, of ``AUC(score) - AUC(other)``) resampling whole
    cities with replacement, as PR #46's ``road_prior.city_delta_ci`` does: the interval for a claim about unseen
    cities, where labels within a city are not independent. With few cities it is wide and coarse. ``(lo, hi)``.

    >>> y = np.array([0, 1] * 20); c = np.repeat(list("abcd"), 10)
    >>> lo, hi = city_bootstrap_auc(y, np.where(y == 1, 0.6, 0.4), c, n_boot=50, other=np.full(40, 0.5))
    >>> bool(lo > 0)
    True
    """
    from sklearn.metrics import roc_auc_score
    y, city = np.asarray(y).astype(int), np.asarray(city)
    a = np.asarray(score, float)
    b = None if other is None else np.asarray(other, float)
    members = [np.flatnonzero(city == g) for g in np.unique(city)]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if y[i].min() == y[i].max():
            continue
        v = roc_auc_score(y[i], a[i])
        if b is not None:
            v -= roc_auc_score(y[i], b[i])
        stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def wilson(k, n, z=1.96):
    """Wilson 95% interval for ``k`` of ``n``: ``(lo, hi)``; NaNs for ``n == 0``.

    >>> [round(v, 3) for v in wilson(5, 10)]
    [0.237, 0.763]
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def feature_matrix(df):
    """The numeric feature matrix used by the lift models: every :data:`FEATURES` column, booleans as 0/1 and missing
    distances (no sidewalk) as :data:`NO_SIDEWALK_DIST`.

    >>> feature_matrix(pd.DataFrame({c: [np.nan] for c, _ in FEATURES})).shape
    (1, 13)
    """
    cols = []
    for c, kind in FEATURES:
        v = pd.to_numeric(df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(df[c])
                          if kind == "bool" else df[c], errors="coerce").astype(float)
        if kind == "dist":
            v = v.fillna(NO_SIDEWALK_DIST[c])
        cols.append(v.fillna(0.0).to_numpy())
    return np.column_stack(cols)


def sha256_file(path):
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cities(path):
    """``{city_id: entry}`` from a ``/v3/api/cities`` listing (``{"status", "cities": [...]}``)."""
    with open(path) as f:
        listing = json.load(f)
    return {c["city_id"]: c for c in listing["cities"]}


# ---------------------------------------------------------------------------------------------------------------------
# filelist
# ---------------------------------------------------------------------------------------------------------------------

LABEL_COLUMNS = ["label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "author", "human_agree",
                 "human_disagree", "camera_pitch", "camera_roll", "tilt_applied", "tilt_beta", "depth_m",
                 "crop_half_px", "clipped", "hard", "crop_path", "split", "pano_native_w", "pano_native_h"]


def select_labels(sample, label_types=LABEL_TYPES):
    """The rows of ``sample.csv`` this experiment uses, with ``label_uid`` checked unique and ``store_tail`` added."""
    df = sample[sample.label_type.isin(label_types)][LABEL_COLUMNS].copy()
    if not df.label_uid.is_unique:
        raise ValueError("label_uid is not unique in the sample")
    expected = df.city + ":" + df.label_id.astype(str)
    if not (expected == df.label_uid).all():
        raise ValueError("label_uid != city:label_id")
    df["store_tail"] = [store_tail(c, p) for c, p in zip(df.city, df.pano_id)]
    return df.reset_index(drop=True)


def cmd_filelist(args):
    labels = select_labels(pd.read_csv(args.sample))
    os.makedirs(args.out_dir, exist_ok=True)
    labels.to_csv(os.path.join(args.out_dir, "labels.csv"), index=False)
    tails = sorted(set(labels.store_tail))
    with open(os.path.join(args.out_dir, "stage_files.txt"), "w") as f:
        f.write("\n".join(tails) + "\n")
    print(f"{len(labels)} labels, {len(tails)} panoramas -> {args.out_dir}/labels.csv, stage_files.txt")


# ---------------------------------------------------------------------------------------------------------------------
# render-segment (GPU job body)
# ---------------------------------------------------------------------------------------------------------------------

def join_pano_coords(labels, rawlabels_dir):
    """Add ``pano_x, pano_y, pano_width, pano_height`` from ``<rawlabels_dir>/<city>.csv``, joined one-to-one on
    ``label_uid``. The rows must all match (the sample was drawn from these exports)."""
    parts = []
    for city in sorted(labels.city.unique()):
        raw = pd.read_csv(os.path.join(rawlabels_dir, f"{city}.csv"),
                          usecols=["label_id", "pano_id", "pano_x", "pano_y", "pano_width", "pano_height"])
        raw = raw.rename(columns={"pano_id": "raw_pano_id"})
        raw["label_uid"] = city + ":" + raw.label_id.astype(str)
        parts.append(raw.drop(columns="label_id"))
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw.label_uid.isin(labels.label_uid)]
    out = labels.merge(raw, on="label_uid", how="left", validate="one_to_one")
    if out.pano_x.isna().any():
        raise ValueError(f"{int(out.pano_x.isna().sum())} labels have no rawLabels row")
    if (out.raw_pano_id != out.pano_id).any():
        raise ValueError("rawLabels pano_id differs from the manifest's for some labels")
    return out.drop(columns="raw_pano_id")


def cmd_coords(args):
    out = join_pano_coords(pd.read_csv(args.labels), args.rawlabels_dir)
    out[["label_uid", "pano_x", "pano_y", "pano_width", "pano_height"]].to_csv(args.out, index=False)
    print(f"{len(out)} rows -> {args.out}")


def _nan_to_none(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def cmd_render_segment(args):
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    import data_generation as dg

    labels = join_pano_coords(pd.read_csv(args.labels), args.rawlabels_dir)
    if args.limit:
        labels = labels.head(args.limit)
    device = "cuda" if torch.cuda.is_available() else None
    if device is None and not args.allow_cpu:
        sys.exit("no CUDA device (pass --allow-cpu for a smoke test)")
    seg_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.seg_model, revision=args.seg_revision)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.seg_model, revision=args.seg_revision)
    model = model.to(seg_device).eval()
    id2label = {str(k): v for k, v in model.config.id2label.items()}

    out = args.out_dir
    for sub in ("masks_1024", "masks_512", "views_512"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows, last = [], (None, None, None)
    t_render = t_seg = t_read = 0.0
    labels = labels.sort_values(["city", "pano_id", "label_uid"], kind="mergesort")
    for k, r in enumerate(labels.itertuples(index=False), start=1):
        rec = {"label_uid": r.label_uid, "status": "", "pano_store": "", "half_check": None, "crop_mad": None,
               "crop_p99": None, "pano_y_norm": r.pano_y / r.pano_height,
               "view_pitch_deg": view_pitch_deg(r.pano_y / r.pano_height)}
        key = (r.city, r.pano_id)
        try:
            if last[0] != key:
                t0 = time.time()
                read, info = dg.read_store_panorama(args.store, r.city, r.pano_id)
                where = "stage"
                if isinstance(read, str) and args.extra_store and os.path.exists(
                        dg.pano_store_path(args.extra_store, r.city, r.pano_id)):
                    read, info = dg.read_store_panorama(args.extra_store, r.city, r.pano_id)
                    where = "extra"
                last = (key, read, {**info, "where": where})
                t_read += time.time() - t0
            pano_bgr, info = last[1], last[2]
            rec["pano_store"] = info["where"]
            if isinstance(pano_bgr, str):
                rec["status"] = pano_bgr
                rows.append(rec)
                continue
            rec["native_match"] = (int(info["pano_native_w"]) == int(r.pano_native_w)
                                   and int(info["pano_native_h"]) == int(r.pano_native_h))
            t0 = time.time()
            pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])
            view, tilt_applied = dg.render_label_view(
                pano_rgb, r.pano_x / r.pano_width, r.pano_y / r.pano_height,
                camera_pitch=_nan_to_none(r.camera_pitch), camera_roll=_nan_to_none(r.camera_roll),
                apply_tilt=r.author == "human", beta=float(r.tilt_beta), device=device)
            rec["tilt_applied"] = int(tilt_applied)
            crop, half, _ = dg.depth_sized_crop(view, float(r.depth_m))
            rec["half_check"] = int(half) == int(r.crop_half_px)
            if args.crops_dir:
                ref = np.asarray(Image.open(os.path.join(args.crops_dir, r.label_type, r.crop_path)).convert("RGB"))
                if ref.shape == crop.shape:
                    diff = np.abs(ref.astype(np.int16) - crop.astype(np.int16))
                    rec["crop_mad"] = float(diff.mean())
                    rec["crop_p99"] = float(np.percentile(diff, 99))
                else:
                    rec["crop_mad"] = float("inf")
            if device == "cuda":
                torch.cuda.synchronize()
            t_render += time.time() - t0

            t0 = time.time()
            small = cv2.resize(view, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
            inputs = processor(images=Image.fromarray(small), size={"height": SEG_SIZE, "width": SEG_SIZE},
                               return_tensors="pt").to(seg_device)
            with torch.no_grad():
                outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(SEG_SIZE, SEG_SIZE)])[0]
            seg = seg.to("cpu").numpy().astype(np.uint8)
            t_seg += time.time() - t0
            stem = r.label_uid.replace(":", "_")
            Image.fromarray(seg).save(os.path.join(out, "masks_1024", f"{stem}.png"), optimize=True)
            Image.fromarray(cv2.resize(seg, (SHEET_SIZE, SHEET_SIZE), interpolation=cv2.INTER_NEAREST)).save(
                os.path.join(out, "masks_512", f"{stem}.png"), optimize=True)
            Image.fromarray(cv2.resize(view, (SHEET_SIZE, SHEET_SIZE), interpolation=cv2.INTER_AREA)).save(
                os.path.join(out, "views_512", f"{stem}.jpg"), quality=88)
            rec["status"] = "ok"
        except Exception as e:  # one bad label must not end the job; it is counted in views.csv
            if isinstance(e, dg.PanoStoreUnavailable):
                raise
            import traceback
            traceback.print_exc()
            rec["status"] = f"error:{type(e).__name__}"
        rows.append(rec)
        if k % 25 == 0 or k == len(labels):
            print(f"{k}/{len(labels)} {r.label_uid} {rec['status']} mad={rec.get('crop_mad')} "
                  f"read {t_read:.0f}s render {t_render:.0f}s seg {t_seg:.0f}s", flush=True)

    views = pd.DataFrame(rows)
    views.to_csv(os.path.join(out, "views.csv"), index=False)
    n_ok = int((views.status == "ok").sum())
    import transformers
    import sidewalk_panorama_tools
    run = {
        "date": datetime.date.today().isoformat(), "seg_model": args.seg_model, "seg_revision": args.seg_revision,
        "seg_input_size": SEG_SIZE, "mask_sizes": [SEG_SIZE, SHEET_SIZE], "id2label": id2label,
        "sidewalk_classes": list(SIDEWALK_CLASSES), "road_classes": list(ROAD_CLASSES),
        "curb_classes": list(CURB_CLASSES), "n_labels": len(views), "n_ok": n_ok,
        "status_counts": views.status.value_counts().to_dict(),
        "pano_store_counts": views.pano_store.value_counts().to_dict(),
        "seconds": {"read": round(t_read, 1), "render_and_crop_check": round(t_render, 1),
                    "segment": round(t_seg, 1)},
        "seg_views_per_s": round(n_ok / t_seg, 3) if t_seg else None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else seg_device,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__, "opencv": cv2.__version__,
                     "sidewalk_panorama_tools": getattr(sidewalk_panorama_tools, "__version__", "?"),
                     "panorama_tools_source": dg.package_source()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(out, "seg_run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(json.dumps({k: run[k] for k in ("n_labels", "n_ok", "status_counts", "seconds", "seg_views_per_s")}))
    ok = views[views.status == "ok"]
    if len(ok) and "crop_mad" in ok and ok.crop_mad.notna().any():
        print("crop check: mean|diff| median %.2f, max %.2f; half_check all %s" % (
            ok.crop_mad.median(), ok.crop_mad.max(), bool(ok.half_check.all())))


# ---------------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------------

def cmd_features(args):
    from PIL import Image
    labels = pd.read_csv(args.labels)
    with open(os.path.join(args.run_dir, "seg_run.json")) as f:
        run = json.load(f)
    views = pd.read_csv(os.path.join(args.run_dir, "views.csv"))
    ids = {g: class_ids(run["id2label"], names) for g, names in
           (("sidewalk", SIDEWALK_CLASSES), ("road", ROAD_CLASSES), ("curb", CURB_CLASSES))}
    if "view_pitch_deg" not in views:
        # views.csv from the first klone run (job 42113381) predates the pitch column: take pano_y from a coords CSV
        # (label_uid, pano_y, pano_height; see the `coords` stage) instead.
        if not args.coords:
            raise SystemExit("views.csv has no view_pitch_deg: pass --coords")
        coords = pd.read_csv(args.coords)
        coords["view_pitch_deg"] = [view_pitch_deg(y / h) for y, h in zip(coords.pano_y, coords.pano_height)]
        views = views.merge(coords[["label_uid", "view_pitch_deg"]], on="label_uid", how="left",
                            validate="one_to_one")
    df = labels.merge(views, on="label_uid", how="left", validate="one_to_one")
    rows = []
    for r in df.itertuples(index=False):
        rec = {"label_uid": r.label_uid}
        if r.status == "ok":
            mask = np.asarray(Image.open(os.path.join(args.run_dir, "masks_1024", r.label_uid.replace(":", "_") + ".png")))
            rec.update(mask_features(mask, ids["sidewalk"], ids["road"], ids["curb"], r.crop_half_px, r.depth_m,
                                     r.view_pitch_deg))
        rows.append(rec)
    feats = pd.DataFrame(rows)
    out = df.merge(feats, on="label_uid", how="left", validate="one_to_one")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    inputs = {"labels": args.labels, "views": os.path.join(args.run_dir, "views.csv"),
              "seg_run": os.path.join(args.run_dir, "seg_run.json")}
    if args.coords:
        inputs["coords"] = args.coords
    write_sidecar(args.out, "features", inputs)
    print(f"{len(out)} rows ({int((out.status == 'ok').sum())} ok) -> {args.out}; sha256 {sha256_file(args.out)}")


# ---------------------------------------------------------------------------------------------------------------------
# cls (production frozen features)
# ---------------------------------------------------------------------------------------------------------------------

#: Production validator revisions, as pinned in sidewalk-ai-api's model_map.py (read 2026-10-21).
PROD_MODELS = {
    "Obstacle": ("projectsidewalk/sidewalk-validator-ai-obstacle", "8f0fe8fc48748fac35098e74c1848d015126deef"),
    "SurfaceProblem": ("projectsidewalk/sidewalk-validator-ai-surfaceproblem",
                       "0e1439daa5667c8a81f63e75f835bd51221df4c2"),
}


def cmd_cls(args):
    """``Dinov2ForImageClassification.forward`` feeds its head ``cat([cls_token, patch_tokens.mean(1)])`` of the last
    hidden state after the final layernorm; that 2048-d vector is what is saved. Checked per batch: the head applied
    to it reproduces the model's logits."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    labels = pd.read_csv(args.labels)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    uids, feats, p_inc = [], [], []
    for lt, (repo, rev) in PROD_MODELS.items():
        processor = AutoImageProcessor.from_pretrained(repo, revision=rev, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(repo, revision=rev).to(device).eval()
        inc = {v: int(k) for k, v in model.config.id2label.items()}["incorrect"]
        sub = labels[labels.label_type == lt]
        for start in range(0, len(sub), args.batch):
            chunk = sub.iloc[start:start + args.batch]
            imgs = [Image.open(os.path.join(args.crops_dir, lt, p)).convert("RGB") for p in chunk.crop_path]
            inputs = processor(imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                seq = model.dinov2(**inputs)[0]
                pooled = torch.cat([seq[:, 0], seq[:, 1:].mean(dim=1)], dim=1)
                logits = model.classifier(pooled)
                ref = model(**inputs).logits
            assert torch.allclose(logits, ref, atol=1e-4), "pooled vector does not reproduce the logits"
            uids += chunk.label_uid.tolist()
            feats.append(pooled.cpu().numpy().astype(np.float32))
            p_inc.append(ref.softmax(-1)[:, inc].cpu().numpy())
        print(f"{lt}: {len(sub)} crops")
    np.savez_compressed(args.out, label_uid=np.array(uids), cls=np.concatenate(feats),
                        p_incorrect=np.concatenate(p_inc),
                        models=np.array([f"{lt}={repo}@{rev}" for lt, (repo, rev) in PROD_MODELS.items()]))
    write_sidecar(args.out, "cls", {"labels": args.labels})
    print(f"-> {args.out}; sha256 {sha256_file(args.out)}")


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def grouped_folds(y, groups, n_splits=5, seed=20261021):
    """Stratified folds that keep each group (panorama) in one fold: a list of ``(train_idx, test_idx)``."""
    from sklearn.model_selection import StratifiedGroupKFold
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(y)), y, groups))


def _fit_predict(Xtr, ytr, gtr, Xte, pca=None, seed=0):
    """Standardise (+ optional PCA to ``pca`` components, fitted on the training fold), then an L2 logistic regression
    whose C is picked by inner 3-fold CV grouped by panorama (AUC). Returns P(incorrect) for ``Xte``."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    inner = grouped_folds(ytr, gtr, n_splits=3, seed=seed)
    steps = [StandardScaler()]
    if pca:
        steps += [PCA(n_components=min(pca, len(ytr) - 1), random_state=seed), StandardScaler()]
    steps.append(LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=inner, scoring="roc_auc", max_iter=5000))
    model = make_pipeline(*steps)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def lift_predictions(df, cls, pca=64, seed=20261021):
    """Out-of-fold P(incorrect) for one label type's rows under 5-fold CV grouped by panorama, for the models

    ``prod`` (the production score itself, no fitting), ``cls`` (frozen production features), ``cls_feat``
    (+ segmentation features), ``feat`` (segmentation features only) and ``prod_feat`` (logit of the production
    score + segmentation features). The CLS block is reduced to ``pca`` components inside each fold so the dozen
    segmentation features are not drowned by 2048 coefficients under one shared penalty. ``df`` rows align with
    ``cls`` rows."""
    y = (df.label_class == "incorrect").astype(int).to_numpy()
    g = df.pano_id.to_numpy()
    F = feature_matrix(df)
    prod = _logit(df.p_incorrect.to_numpy())[:, None]
    out = {"prod": df.p_incorrect.to_numpy(float)}
    for name in ("cls", "cls_feat", "feat", "prod_feat"):
        out[name] = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(grouped_folds(y, g, 5, seed)):
        out["cls"][te] = _fit_predict(cls[tr], y[tr], g[tr], cls[te], pca=pca, seed=seed + k)
        out["feat"][te] = _fit_predict(F[tr], y[tr], g[tr], F[te], seed=seed + k)
        out["prod_feat"][te] = _fit_predict(np.hstack([prod, F])[tr], y[tr], g[tr], np.hstack([prod, F])[te],
                                            seed=seed + k)
        # CLS + features: PCA on the CLS block only, then the features appended (both standardised).
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(cls[tr])
        p = PCA(n_components=min(pca, len(tr) - 1), random_state=seed).fit(sc.transform(cls[tr]))
        Ztr = np.hstack([p.transform(sc.transform(cls[tr])), F[tr]])
        Zte = np.hstack([p.transform(sc.transform(cls[te])), F[te]])
        out["cls_feat"][te] = _fit_predict(Ztr, y[tr], g[tr], Zte, seed=seed + k)
    return pd.DataFrame(out, index=df.index)


def transfer_predictions(df, cls, pca=64, seed=20261021):
    """Fit on the ``test`` rows, predict the ``test_heldout`` rows (three cities never in train): the same five
    models as :func:`lift_predictions`. Returns predictions for the held-out rows."""
    tr = (df.split == "test").to_numpy()
    te = (df.split == "test_heldout").to_numpy()
    y = (df.label_class == "incorrect").astype(int).to_numpy()
    g = df.pano_id.to_numpy()
    F = feature_matrix(df)
    prod = _logit(df.p_incorrect.to_numpy())[:, None]
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(cls[tr])
    p = PCA(n_components=pca, random_state=seed).fit(sc.transform(cls[tr]))
    Z = np.hstack([p.transform(sc.transform(cls)), F])
    out = {"prod": df.p_incorrect.to_numpy(float)[te],
           "cls": _fit_predict(cls[tr], y[tr], g[tr], cls[te], pca=pca, seed=seed),
           "cls_feat": _fit_predict(Z[tr], y[tr], g[tr], Z[te], seed=seed),
           "feat": _fit_predict(F[tr], y[tr], g[tr], F[te], seed=seed),
           "prod_feat": _fit_predict(np.hstack([prod, F])[tr], y[tr], g[tr], np.hstack([prod, F])[te], seed=seed)}
    return pd.DataFrame(out, index=df.index[te])


MODEL_NAMES = {"prod": "production score (no fit)", "cls": "CLS head", "cls_feat": "CLS + seg features",
               "feat": "seg features only", "prod_feat": "production score + seg features"}


#: Below this many cities a city-resampled CI is not estimable (3 cities give only 10 distinct resamples): blank.
MIN_CITIES_FOR_CITY_CI = 8


def _lift_rows(y, preds, city, slice_name, n_boot, seed):
    """One row per model: pooled AUC with row- and city-resampled CIs, the within-city AUC, and for the two
    "+ seg" models the paired delta against their base in all three forms."""
    rows = []
    n_cities = len(np.unique(city))
    city_ci = n_cities >= MIN_CITIES_FOR_CITY_CI
    nan2 = (float("nan"), float("nan"))
    for m in MODEL_NAMES:
        auc, _ = oriented_auc(y, preds[m])
        lo, hi = bootstrap_auc(y, preds[m], n_boot, seed)
        clo, chi = city_bootstrap_auc(y, preds[m], city, n_boot, seed) if city_ci else nan2
        wc, nwc = within_city_auc(y, preds[m], city)
        wcr, _ = within_city_auc(y, preds[m], city, weight="rows")
        row = {"slice": slice_name, "model": m, "n": len(y), "n_incorrect": int(np.sum(y)), "n_cities": n_cities,
               "auc": auc, "auc_lo": lo, "auc_hi": hi, "auc_city_lo": clo, "auc_city_hi": chi,
               "auc_within_city": wc, "auc_within_city_rows": wcr, "n_cities_within": nwc}
        base = {"cls_feat": "cls", "prod_feat": "prod"}.get(m)
        if base:
            dlo, dhi = bootstrap_auc(y, preds[m], n_boot, seed, other=preds[base])
            dclo, dchi = city_bootstrap_auc(y, preds[m], city, n_boot, seed, other=preds[base]) if city_ci else nan2
            row.update(delta_vs=base, delta_auc=auc - oriented_auc(y, preds[base])[0], delta_lo=dlo, delta_hi=dhi,
                       delta_city_lo=dclo, delta_city_hi=dchi,
                       delta_within_city=wc - within_city_auc(y, preds[base], city)[0],
                       delta_within_city_rows=wcr - within_city_auc(y, preds[base], city, weight="rows")[0])
        rows.append(row)
    return rows


def _fmt(v, nd=3):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def cmd_report(args):
    from evaluate import slice_metrics
    cities = load_cities(args.cities_json)
    feats = pd.read_csv(args.features)
    z = np.load(args.cls, allow_pickle=False)
    cls_df = pd.DataFrame({"label_uid": z["label_uid"], "p_incorrect": z["p_incorrect"]})
    cls_df["cls_row"] = np.arange(len(cls_df))
    df = feats.merge(cls_df, on="label_uid", how="left", validate="one_to_one")
    if df.p_incorrect.isna().any():
        raise SystemExit("some labels have no CLS features")
    df["display_city"] = display_cities(df.city, cities)
    df["y"] = (df.label_class == "incorrect").astype(int)
    os.makedirs(args.out_dir, exist_ok=True)
    md = []
    ok = df[df.status == "ok"].copy()

    # 1. Segmenter sanity.
    with open(os.path.join(args.run_dir, "seg_run.json")) as f:
        run = json.load(f)
    ground_ids = class_ids(run["id2label"], GROUND_OBJECT_CLASSES)
    sanity = []
    for lt, g in [("all", ok)] + list(ok.groupby("label_type")):
        sanity.append({"label_type": lt, "n_views": len(g), "any_sidewalk": g.sidewalk_any.mean(),
                       "sidewalk_frac_view_median": g.sidewalk_frac_view.median(),
                       "views_sidewalk_above_horizon_gt_5pct": (g.sidewalk_above_horizon_frac > 0.05).mean(),
                       "sidewalk_px_above_horizon_share": float(
                           (g.sidewalk_above_horizon_frac * g.sidewalk_frac_view).sum() / g.sidewalk_frac_view.sum()),
                       "point_on_sidewalk": g.point_on_sidewalk.mean(), "on_road": g.on_road.mean(),
                       "point_on_ground_object": g.centre_class.isin(ground_ids).mean(),
                       "near_sidewalk_1m": g.near_sidewalk_1m.mean(),
                       "depth_over_ground_range_median": float((g.depth_m / g.ground_range_m).median()),
                       "depth_over_ground_range_q25": float((g.depth_m / g.ground_range_m).quantile(0.25)),
                       "depth_over_ground_range_q75": float((g.depth_m / g.ground_range_m).quantile(0.75))})
    sanity = pd.DataFrame(sanity)
    sanity.to_csv(os.path.join(args.out_dir, "sanity.csv"), index=False)
    md += ["## Segmenter sanity", "", _md_table(sanity), ""]
    id2label = run["id2label"]
    cc = (ok.groupby("label_type").centre_class.value_counts(normalize=True).rename("share").reset_index())
    cc["class"] = cc.centre_class.astype(int).astype(str).map(id2label)
    cc = cc.groupby("label_type").head(8)
    cc.to_csv(os.path.join(args.out_dir, "centre_class.csv"), index=False)
    md += ["## Class at the label point (top 8 per type)", "", _md_table(cc[["label_type", "class", "share"]]), ""]

    # 2. Each feature alone, per type x split, with incorrect as the positive class.
    rows = []
    slices = [((lt, "both"), g) for lt, g in ok.groupby("label_type")] + list(ok.groupby(["label_type", "split"]))
    for (lt, split), s in slices:
        F = feature_matrix(s)
        for j, (c, _) in enumerate(FEATURES):
            auc, sign = oriented_auc(s.y, F[:, j])
            lo, hi = bootstrap_auc(s.y, F[:, j], args.n_boot, 1) if math.isfinite(auc) else (np.nan, np.nan)
            wc, nwc = within_city_auc(s.y, F[:, j], s.city)
            wcr, _ = within_city_auc(s.y, F[:, j], s.city, weight="rows")
            rows.append({"label_type": lt, "split": split, "feature": c, "n": len(s), "auc": auc, "dir": sign,
                         "auc_oriented": max(auc, 1 - auc) if math.isfinite(auc) else np.nan,
                         "auc_lo": lo, "auc_hi": hi, "auc_within_city": wc, "auc_within_city_rows": wcr,
                         "n_cities_within": nwc,
                         "mean_correct": F[s.y.to_numpy() == 0, j].mean(),
                         "mean_incorrect": F[s.y.to_numpy() == 1, j].mean()})
    alone = pd.DataFrame(rows)
    alone.to_csv(os.path.join(args.out_dir, "feature_auc.csv"), index=False)
    md += ["## Each feature alone (AUC for incorrect; dir '-' = higher value means more likely correct; "
           "auc_oriented = max(auc, 1 - auc); auc_within_city = per-city AUCs weighted by n_pos * n_neg (stratified Mann-Whitney), "
           "_rows = weighted by rows; cities with both classes and >= 10 rows)", "", _md_table(alone), ""]

    # 3. Lift over the frozen production features.
    lift_rows, city_rows = [], []
    oof_all = []
    for lt, g in df.groupby("label_type"):
        g = g[g.status == "ok"]
        cls = z["cls"][g.cls_row.to_numpy()]
        preds = lift_predictions(g, cls, pca=args.pca)
        oof_all.append(pd.concat([g[["label_uid", "label_type", "split", "city", "display_city", "y", "hard"]],
                                  preds], axis=1))
        lift_rows += [{"label_type": lt, **r} for r in _lift_rows(g.y.to_numpy(), preds, g.city.to_numpy(),
                                                                  "all (5-fold CV)", args.n_boot, 2)]
        for split, s in g.groupby("split"):
            lift_rows += [{"label_type": lt, **r} for r in _lift_rows(s.y.to_numpy(), preds.loc[s.index],
                                                                      s.city.to_numpy(), f"{split} (5-fold CV)",
                                                                      args.n_boot, 3)]
        tp = transfer_predictions(g, cls, pca=args.pca)
        s = g.loc[tp.index]
        lift_rows += [{"label_type": lt, **r} for r in _lift_rows(s.y.to_numpy(), tp, s.city.to_numpy(),
                                                                  "test_heldout (fit on test)", args.n_boot, 4)]
        for city, s in g.groupby("display_city"):
            if len(s) < args.min_city_n or s.y.nunique() < 2:
                continue
            for m in MODEL_NAMES:
                auc, _ = oriented_auc(s.y, preds.loc[s.index, m])
                city_rows.append({"label_type": lt, "city": city, "n": len(s), "n_incorrect": int(s.y.sum()),
                                  "model": m, "auc": auc})
        hard = g[g.hard == 1]
        lift_rows.append({"label_type": lt, "slice": "hard", "model": "-", "n": len(hard),
                          "n_incorrect": int(hard.y.sum()), "auc": float("nan")})
    lift = pd.DataFrame(lift_rows)
    lift.to_csv(os.path.join(args.out_dir, "lift.csv"), index=False)
    by_city = pd.DataFrame(city_rows)
    if len(by_city):
        by_city = by_city.pivot_table(index=["label_type", "city", "n", "n_incorrect"], columns="model",
                                      values="auc").reset_index()
    by_city.to_csv(os.path.join(args.out_dir, "lift_by_city.csv"), index=False)
    md += ["## Lift (out-of-fold AUC; `_lo`/`_hi` resample rows, `_city_lo`/`_city_hi` resample whole cities; deltas "
           "are paired; city CIs blank below 8 cities; within-city = per-city AUCs weighted by n_pos * n_neg, _rows "
           "by rows; cities with both classes and >= 10 rows)",
           "", _md_table(lift), "",
           f"## Per city (n >= {args.min_city_n}; private pooled)", "", _md_table(by_city), ""]

    # Production model's own slice metrics on this sample (evaluate.slice_metrics; plain accuracy is NOT a
    # production number on a 50/50 sample).
    pm = []
    for (lt, split), g in df.groupby(["label_type", "split"]):
        sm = slice_metrics(pd.DataFrame({"true": g.label_class, "p_incorrect": g.p_incorrect,
                                         "p_correct": 1 - g.p_incorrect}))
        pm.append({"label_type": lt, "split": split, "n": sm["n"], "auc": sm["auc"],
                   "balanced_accuracy": sm["balanced_accuracy"],
                   "recall_incorrect_at_p90": sm["recall_at_precision"]["incorrect"]["0.90"]["recall"],
                   "recall_correct_at_p95": sm["recall_at_precision"]["correct"]["0.95"]["recall"]})
    pm = pd.DataFrame(pm)
    pm.to_csv(os.path.join(args.out_dir, "production_on_sample.csv"), index=False)
    md += ["## Production checkpoints re-scored on the round-3 crops", "", _md_table(pm), ""]

    # 4. Production-verdict strata.
    strata, n_join, agree = production_strata(ok, args.assessments, cities)
    strata.to_csv(os.path.join(args.out_dir, "strata.csv"), index=False)
    agree.to_csv(os.path.join(args.out_dir, "served_vs_rescored.csv"), index=False)
    md += [f"## Production-verdict strata ({n_join} of {len(ok)} sample labels have an assessment)", "",
           _md_table(strata), "",
           "## Served verdict vs the re-scored checkpoint (p_incorrect > 0.5 means Disagree)", "",
           _md_table(agree), ""]

    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    oof = pd.concat(oof_all)
    inputs = {"features_csv": args.features, "cls_npz": args.cls, "assessments": args.assessments,
              "cities_json": args.cities_json, "views_csv": os.path.join(args.run_dir, "views.csv")}
    if args.private_dir:
        os.makedirs(args.private_dir, exist_ok=True)
        inputs["oof_predictions_csv"] = os.path.join(args.private_dir, "oof_predictions.csv")
        oof.to_csv(inputs["oof_predictions_csv"], index=False)
    write_provenance(args, run, ok, inputs)
    print("\n".join(md))


def git_state(exclude=None):
    """``(commit, dirty)`` of this checkout; dirty means a TRACKED file differs from HEAD (untracked files ignored),
    outside ``exclude`` (the report's own output directory, which the run itself rewrites)."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    spec = ["--", "."] + ([f":!{os.path.relpath(os.path.abspath(exclude), REPO)}"] if exclude else [])
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no", *spec))


def safetensors_param_count(path):
    """Number of parameters in a ``.safetensors`` file, read from its JSON header (no torch needed)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def segmenter_params_m(run):
    """Segmenter size in millions of parameters from the local HF cache (None if the weights are not cached)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(run["seg_model"], "model.safetensors", revision=run["seg_revision"])
    except Exception:
        return None
    return round(safetensors_param_count(p) / 1e6, 1) if isinstance(p, str) else None


def write_provenance(args, run, ok, inputs):
    """``provenance.json`` in the output dir: input sha256s and row counts, output sha256s, the segmenter run record
    (model, revision, job, device, throughput, versions), the CLS models, the commits the ``features`` and ``cls``
    stages ran at (from their sidecar JSONs), packages, commit and hardware. ``--extra-provenance`` merges a
    hand-entered record (klone job ids and timings, copied from ``sacct``) under ``hand_entered``."""
    import platform
    import sklearn

    outs = {f: sha256_file(os.path.join(args.out_dir, f)) for f in sorted(os.listdir(args.out_dir))
            if f.endswith((".csv", ".md")) and f != "provenance.json"}
    z = np.load(args.cls, allow_pickle=False)
    rec = {
        "date": datetime.date.today().isoformat(),
        "git_commit": git_state()[0], "git_dirty": git_state(args.out_dir)[1],
        "command": "python analysis/sidewalk_seg.py " + " ".join(sys.argv[1:]).replace(os.path.expanduser("~"), "~"),
        "inputs": {k: {"sha256": sha256_file(p), "rows": (int(sum(1 for _ in open(p)) - 1) if p.endswith(".csv")
                                                         else None)} for k, p in inputs.items()},
        "n_labels_ok": int(len(ok)),
        "outputs": outs,
        "segmenter_run": {k: run.get(k) for k in ("date", "seg_model", "seg_revision", "seg_input_size",
                                                  "mask_sizes", "sidewalk_classes", "road_classes", "curb_classes",
                                                  "n_labels", "n_ok", "status_counts", "pano_store_counts",
                                                  "seconds", "seg_views_per_s", "device", "versions",
                                                  "slurm_job_id")},
        "cls_models": [str(m) for m in z["models"]],
        "stage_records": {k: _sidecar(p) for k, p in (("features", args.features), ("cls", args.cls))},
        "segmenter_params_m": segmenter_params_m(run),
        "settings": {"n_boot": args.n_boot, "pca": args.pca, "min_city_n": args.min_city_n, "folds": 5,
                     "fold_seed": 20261021, "f_px": F_PX, "disc_radius_px": DISC_RADIUS_PX, "near_m": NEAR_M,
                     "bottom_frac": BOTTOM_FRAC},
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
        "hardware": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0] or platform.release()}",
    }
    for item in getattr(args, "hash_extra", None) or []:
        name, path = item.split("=", 1)
        rec["inputs"][name] = {"sha256": sha256_file(os.path.expanduser(path)), "rows": None}
    if getattr(args, "extra_provenance", None):
        with open(args.extra_provenance) as f:
            rec["hand_entered"] = json.load(f)
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")


def _sidecar(path):
    """The ``<path>.json`` stage record written next to a stage output, or None (older outputs have none)."""
    try:
        with open(path + ".json") as f:
            return json.load(f)
    except OSError:
        return None


def write_sidecar(path, stage, inputs):
    """Write ``<path>.json``: stage, date, commit, dirty flag, input and output sha256 (the stage record)."""
    commit, dirty = git_state()
    rec = {"stage": stage, "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
           "inputs": {k: sha256_file(p) for k, p in inputs.items()}, "output_sha256": sha256_file(path)}
    with open(path + ".json", "w") as f:
        json.dump(rec, f, indent=1)


def production_strata(ok, assessments_path, cities):
    """Join the sample to the production assessments (one row per ``label_uid``, ``validate="one_to_one"``) and
    tabulate, per type and pool, the share of labels not on a sidewalk / not within 1 m of one, overall and by the
    sample's human class; and how often the re-scored checkpoint's verdict (``p_incorrect > 0.5``) matches the served
    one. Returns ``(table, n_joined, agreement)``."""
    a = pd.read_csv(assessments_path, usecols=["schema", "label_id", "label_type", "ai_result", "ai_accuracy",
                                               "h_agree", "h_dis"])
    a = a[a.label_type.isin(LABEL_TYPES)]
    a["label_uid"] = [schema_label_uid(s, i, cities) for s, i in zip(a.schema, a.label_id)]
    a = a[a.label_uid.isin(set(ok.label_uid))]
    if a.label_uid.duplicated().any():
        raise ValueError("production assessments are not one row per label_uid")
    j = ok.merge(a[["label_uid", "ai_result", "ai_accuracy", "h_agree", "h_dis"]], on="label_uid", how="left",
                 validate="one_to_one")
    j["pool"] = [production_pool(r, d) for r, d in zip(j.ai_result, j.h_dis)]
    served = j[j.ai_result.isin(["Agree", "Disagree"])]
    agree = pd.DataFrame([{"label_type": lt, "n": len(g),
                           "same_verdict": float(((g.p_incorrect > 0.5) == (g.ai_result == "Disagree")).mean())}
                          for lt, g in [("all", served)] + list(served.groupby("label_type"))])
    rows = []
    for (lt, pool), g in j.groupby(["label_type", "pool"]):
        for cls_name, s in [("all", g)] + list(g.groupby("label_class")):
            n = len(s)
            k1 = int((~s.point_on_sidewalk.astype(bool)).sum())
            k2 = int((~s.near_sidewalk_1m.astype(bool)).sum())
            lo1, hi1 = wilson(k1, n)
            lo2, hi2 = wilson(k2, n)
            rows.append({"label_type": lt, "pool": pool, "human_class": cls_name, "n": n,
                         "not_on_sidewalk": k1 / n if n else np.nan, "not_on_lo": lo1, "not_on_hi": hi1,
                         "not_near_1m": k2 / n if n else np.nan, "not_near_lo": lo2, "not_near_hi": hi2})
    return pd.DataFrame(rows), int(j.ai_result.notna().sum()), agree


# ---------------------------------------------------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------------------------------------------------

#: Overlay colours (RGB) for the class groups in the contact sheets.
OVERLAY = {"sidewalk": (0, 200, 80), "road": (170, 60, 220), "curb": (255, 200, 0)}


def overlay_tile(view, mask, ids, crop_half_px, alpha=0.45):
    """The 512 view with the sidewalk / road / curb mask blended in, the crop square and a centre mark."""
    import cv2
    m = mask.shape[0]
    tile = view.astype(np.float32).copy()
    for g, col in OVERLAY.items():
        sel = np.isin(mask, ids[g])
        tile[sel] = (1 - alpha) * tile[sel] + alpha * np.array(col, np.float32)
    tile = tile.clip(0, 255).astype(np.uint8)
    c = m // 2
    h = int(round(crop_half_px * m / VIEW_SIZE))
    cv2.rectangle(tile, (max(0, c - h), max(0, c - h)), (min(m - 1, c + h), min(m - 1, c + h)), (255, 255, 255), 2)
    cv2.drawMarker(tile, (c, c), (255, 0, 0), cv2.MARKER_CROSS, 22, 3)
    return tile


def cmd_sheets(args):
    import cv2
    from PIL import Image, ImageDraw
    cities = load_cities(args.cities_json)
    with open(os.path.join(args.run_dir, "seg_run.json")) as f:
        run = json.load(f)
    ids = {g: class_ids(run["id2label"], names) for g, names in
           (("sidewalk", SIDEWALK_CLASSES), ("road", ROAD_CLASSES), ("curb", CURB_CLASSES))}
    from calibration.per_city import public_name
    df = pd.read_csv(args.features)
    df = df[(df.status == "ok") & df.city.map(lambda c: public_name(c, cities) is not None)]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def tile_for(r, size):
        stem = r.label_uid.replace(":", "_")
        view = np.asarray(Image.open(os.path.join(args.run_dir, "views_512", stem + ".jpg")).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(args.run_dir, "masks_512", stem + ".png")))
        t = overlay_tile(view, mask, ids, r.crop_half_px)
        return cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) if size != t.shape[0] else t

    made = []
    for lt in LABEL_TYPES:
        pool = df[df.label_type == lt]
        order = rng.permutation(len(pool))
        for k in range(2):
            pick = pool.iloc[order[k * 40:(k + 1) * 40]]
            if pick.empty:
                continue
            size, cols = 256, 8
            rows_n = int(math.ceil(len(pick) / cols))
            sheet = Image.new("RGB", (cols * size, rows_n * (size + 16)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(pick.itertuples(index=False)):
                x, y = (i % cols) * size, (i // cols) * (size + 16)
                sheet.paste(Image.fromarray(tile_for(r, size)), (x, y + 16))
                flag = "on" if r.point_on_sidewalk else f"{r.dist_m_to_sidewalk:.1f}m" if r.sidewalk_any else "none"
                draw.text((x + 3, y + 2), f"{r.label_uid} {r.label_class[:3]} sw:{flag}", fill="black")
            path = os.path.join(args.out_dir, f"sheet_{lt.lower()}_{k + 1}.jpg")
            sheet.save(path, quality=80, optimize=True)
            made.append((path, os.path.getsize(path)))
    if args.doc_fig:
        # Six views: per type, one correct on sidewalk, one incorrect off it, one more at random.
        picks = []
        for lt in LABEL_TYPES:
            pool = df[df.label_type == lt]
            for q in (pool[(pool.label_class == "correct") & pool.point_on_sidewalk.astype(bool)],
                      pool[(pool.label_class == "incorrect") & ~pool.point_on_sidewalk.astype(bool)], pool):
                if len(q):
                    picks.append(q.iloc[int(rng.integers(len(q)))])
        size = 200
        sheet = Image.new("RGB", (3 * size, 2 * (size + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(picks[:6]):
            x, y = (i % 3) * size, (i // 3) * (size + 14)
            sheet.paste(Image.fromarray(tile_for(r, size)), (x, y + 14))
            draw.text((x + 3, y + 1), f"{r.label_type} {r.label_class}", fill="black")
        sheet.quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(args.doc_fig, optimize=True)  # <= 120 KB
        made.append((args.doc_fig, os.path.getsize(args.doc_fig)))
    for p, n in made:
        print(f"{p}: {n / 1024:.0f} KB")


# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("filelist", help="labels.csv and stage_files.txt from the held-out sample")
    s.add_argument("--sample", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_filelist)

    s = sub.add_parser("render-segment", help="render each label's view, check the crop, segment (GPU job body)")
    s.add_argument("--labels", required=True)
    s.add_argument("--store", required=True, help="staged pano store root (<city>/<pp>/<pano_id>.jpg + marker)")
    s.add_argument("--extra-store", default=None, help="read-only fallback store for panos the main one lacks")
    s.add_argument("--rawlabels-dir", required=True, help="<city>.csv rawLabels exports (pano_x, pano_y, ...)")
    s.add_argument("--crops-dir", default=None, help="<LabelType>/<crop_path> training crops to check against")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--seg-model", default=SEG_MODEL)
    s.add_argument("--seg-revision", required=True, help="full commit sha (jobs run with HF_HUB_OFFLINE=1)")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--allow-cpu", action="store_true", help="smoke test: numpy render, segment on mps/cpu")
    s.set_defaults(func=cmd_render_segment)

    s = sub.add_parser("features", help="per-label mask features (private CSV)")
    s.add_argument("--labels", required=True)
    s.add_argument("--run-dir", required=True, help="the pulled render-segment output directory")
    s.add_argument("--coords", default=None, help="label_uid,pano_y,pano_height CSV, if views.csv lacks the pitch")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("coords", help="label_uid + pano coordinates from rawLabels exports (klone; private CSV)")
    s.add_argument("--labels", required=True)
    s.add_argument("--rawlabels-dir", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_coords)

    s = sub.add_parser("cls", help="frozen production pooled features + p_incorrect for every sample crop")
    s.add_argument("--labels", required=True)
    s.add_argument("--crops-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=16)
    s.set_defaults(func=cmd_cls)

    s = sub.add_parser("report", help="aggregate tables (public cities named, private pooled)")
    s.add_argument("--features", required=True)
    s.add_argument("--cls", required=True)
    s.add_argument("--assessments", required=True)
    s.add_argument("--cities-json", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--private-dir", default=None, help="where label-level out-of-fold predictions go (not the repo)")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--pca", type=int, default=64)
    s.add_argument("--min-city-n", type=int, default=30)
    s.add_argument("--extra-provenance", default=None,
                   help="hand-entered JSON stored under provenance.json's hand_entered (klone job ids, timings)")
    s.add_argument("--hash-extra", action="append", default=[], metavar="NAME=PATH",
                   help="another private file whose sha256 provenance.json should pin (e.g. the sample figure)")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("sheets", help="contact sheets for the hand-check (public cities only)")
    s.add_argument("--features", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--cities-json", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--doc-fig", default=None)
    s.add_argument("--seed", type=int, default=20261021)
    s.set_defaults(func=cmd_sheets)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
