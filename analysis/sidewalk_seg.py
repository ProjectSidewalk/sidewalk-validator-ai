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
so a lateral offset of ``d`` view pixels at the label's depth is about ``d * depth_m / f`` metres. That is a
small-angle, same-depth approximation: it treats the nearest sidewalk pixel as being at the label's depth, which
overstates the distance for sidewalk nearer the camera and understates it for sidewalk beyond the label. The
view is gravity-levelled and roll-free (``sidewalk_panorama_tools.geometry.label_view``), so the horizon is the
row ``1024 + f * tan(pitch)``, where ``pitch = (0.5 - pano_y / pano_height) * 180`` degrees.

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
    # masks_1024/, masks_512/, views_512/ into $O/klone
    python analysis/sidewalk_seg.py features --labels $O/labels.csv --run-dir $O/klone --out $O/features.csv
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

SIDEWALK_CLASSES = ("Sidewalk", "Curb Cut", "Pedestrian Area")
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
    ("near_sidewalk_1m", "bool"),
    ("sidewalk_frac_crop", "frac"),
    ("sidewalk_frac_view", "frac"),
    ("road_frac_crop", "frac"),
    ("curb_frac_crop", "frac"),
    ("on_road", "bool"),
    ("connected_to_bottom", "bool"),
)
#: A missing distance (no sidewalk in the view) is scored as this far: the view diagonal, and its metres at 80 m depth.
NO_SIDEWALK_DIST = {"dist_px_to_sidewalk": VIEW_SIZE * math.sqrt(2),
                    "dist_m_to_sidewalk": VIEW_SIZE * math.sqrt(2) * 80.0 / F_PX}


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
                  f_px=F_PX, disc_radius_px=DISC_RADIUS_PX, bottom_frac=BOTTOM_FRAC, near_m=NEAR_M):
    """Point-vs-sidewalk features of one label from a square class-id ``mask`` of its label-centred view.

    The mask may be any size ``m``; distances and the crop square are given in view pixels (``view_size``) and
    scaled by ``m / view_size``. The label point is the mask's centre pixel ``(m // 2, m // 2)``.

    Returns a dict:

    * ``sidewalk_any``: any sidewalk pixel in the view.
    * ``point_on_sidewalk``: the centre pixel is sidewalk. ``sidewalk_frac_disc`` is the sidewalk share of the
      ``disc_radius_px`` disc around it and ``point_on_sidewalk_disc`` is that share > 0.5.
    * ``dist_px_to_sidewalk``: view pixels from the centre to the nearest sidewalk pixel (0 if on it, NaN if there is
      none); ``dist_m_to_sidewalk`` the same in metres via :func:`px_to_m`; ``near_sidewalk_1m`` is ``dist_m <= near_m``
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
        >>> f = mask_features(m, [1], [2], [3], crop_half_px=256, depth_m=10.0, pitch_deg=-20.0)
        >>> f["point_on_sidewalk"], f["dist_px_to_sidewalk"], round(f["dist_m_to_sidewalk"], 3), f["connected_to_bottom"]
        (False, 512.0, 5.0, True)
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
    out = {"centre_class": int(mask[c, c]), "sidewalk_any": bool(sw.any()),
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
        out["dist_m_to_sidewalk"] = px_to_m(dist_px, depth_m, f_px)
        out["near_sidewalk_1m"] = out["dist_m_to_sidewalk"] <= near_m
    else:
        out["dist_px_to_sidewalk"] = float("nan")
        out["dist_m_to_sidewalk"] = float("nan")
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
    (1, 12)
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
               "crop_p99": None}
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
    coords = None
    if args.rawlabels_dir:
        coords = join_pano_coords(labels, args.rawlabels_dir)[["label_uid", "pano_y", "pano_height"]]
    df = labels.merge(views, on="label_uid", how="left", validate="one_to_one")
    if coords is not None:
        df = df.merge(coords, on="label_uid", how="left", validate="one_to_one")
    elif "view_pitch_deg" not in df:
        raise SystemExit("need --rawlabels-dir (pano_y) for the horizon check")
    rows = []
    for r in df.itertuples(index=False):
        rec = {"label_uid": r.label_uid}
        if r.status == "ok":
            mask = np.asarray(Image.open(os.path.join(args.run_dir, "masks_1024", r.label_uid.replace(":", "_") + ".png")))
            pitch = view_pitch_deg(r.pano_y / r.pano_height)
            rec.update(mask_features(mask, ids["sidewalk"], ids["road"], ids["curb"], r.crop_half_px, r.depth_m, pitch))
            rec["view_pitch_deg"] = pitch
        rows.append(rec)
    feats = pd.DataFrame(rows)
    out = df.drop(columns=[c for c in ("pano_y", "pano_height") if c in df]).merge(
        feats, on="label_uid", how="left", validate="one_to_one")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
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


def _lift_rows(y, preds, slice_name, n_boot, seed):
    rows = []
    for m in MODEL_NAMES:
        auc, _ = oriented_auc(y, preds[m])
        lo, hi = bootstrap_auc(y, preds[m], n_boot, seed)
        row = {"slice": slice_name, "model": m, "n": len(y), "n_incorrect": int(np.sum(y)), "auc": auc,
               "auc_lo": lo, "auc_hi": hi}
        base = {"cls_feat": "cls", "prod_feat": "prod"}.get(m)
        if base:
            dlo, dhi = bootstrap_auc(y, preds[m], n_boot, seed, other=preds[base])
            row.update(delta_vs=base, delta_auc=auc - oriented_auc(y, preds[base])[0], delta_lo=dlo, delta_hi=dhi)
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
    sanity = []
    for lt, g in [("all", ok)] + list(ok.groupby("label_type")):
        sanity.append({"label_type": lt, "n_views": len(g), "any_sidewalk": g.sidewalk_any.mean(),
                       "sidewalk_frac_view_median": g.sidewalk_frac_view.median(),
                       "views_sidewalk_above_horizon_gt_5pct": (g.sidewalk_above_horizon_frac > 0.05).mean(),
                       "sidewalk_px_above_horizon_share": float(
                           (g.sidewalk_above_horizon_frac * g.sidewalk_frac_view).sum() / g.sidewalk_frac_view.sum()),
                       "point_on_sidewalk": g.point_on_sidewalk.mean(), "on_road": g.on_road.mean(),
                       "near_sidewalk_1m": g.near_sidewalk_1m.mean()})
    sanity = pd.DataFrame(sanity)
    sanity.to_csv(os.path.join(args.out_dir, "sanity.csv"), index=False)
    md += ["## Segmenter sanity", "", _md_table(sanity), ""]
    with open(os.path.join(args.run_dir, "seg_run.json")) as f:
        run = json.load(f)
    id2label = run["id2label"]
    cc = (ok.groupby("label_type").centre_class.value_counts(normalize=True).rename("share").reset_index())
    cc["class"] = cc.centre_class.astype(int).astype(str).map(id2label)
    cc = cc.groupby("label_type").head(8)
    cc.to_csv(os.path.join(args.out_dir, "centre_class.csv"), index=False)
    md += ["## Class at the label point (top 8 per type)", "", _md_table(cc[["label_type", "class", "share"]]), ""]

    # 2. Each feature alone, per type x split, with incorrect as the positive class.
    rows = []
    for (lt, split), g in ok.groupby(["label_type", "split"]):
        for slice_name, s in (("all", g), ("correct-class mean", None)):
            if s is None:
                continue
            F = feature_matrix(s)
            for j, (c, _) in enumerate(FEATURES):
                auc, sign = oriented_auc(s.y, F[:, j])
                lo, hi = bootstrap_auc(s.y, F[:, j], args.n_boot, 1) if math.isfinite(auc) else (np.nan, np.nan)
                rows.append({"label_type": lt, "split": split, "feature": c, "n": len(s), "auc": auc, "dir": sign,
                             "auc_lo": lo, "auc_hi": hi,
                             "mean_correct": F[s.y.to_numpy() == 0, j].mean(),
                             "mean_incorrect": F[s.y.to_numpy() == 1, j].mean()})
    alone = pd.DataFrame(rows)
    alone.to_csv(os.path.join(args.out_dir, "feature_auc.csv"), index=False)
    md += ["## Each feature alone (AUC for incorrect; dir '-' = higher value means more likely correct)", "",
           _md_table(alone), ""]

    # 3. Lift over the frozen production features.
    lift_rows, city_rows = [], []
    oof_all = []
    for lt, g in df.groupby("label_type"):
        g = g[g.status == "ok"]
        cls = z["cls"][g.cls_row.to_numpy()]
        preds = lift_predictions(g, cls, pca=args.pca)
        oof_all.append(pd.concat([g[["label_uid", "label_type", "split", "display_city", "y", "hard"]], preds], axis=1))
        lift_rows += [{"label_type": lt, **r} for r in _lift_rows(g.y.to_numpy(), preds, "all (5-fold CV)",
                                                                  args.n_boot, 2)]
        for split, s in g.groupby("split"):
            lift_rows += [{"label_type": lt, **r} for r in _lift_rows(s.y.to_numpy(), preds.loc[s.index],
                                                                      f"{split} (5-fold CV)", args.n_boot, 3)]
        tp = transfer_predictions(g, cls, pca=args.pca)
        s = g.loc[tp.index]
        lift_rows += [{"label_type": lt, **r} for r in _lift_rows(s.y.to_numpy(), tp, "test_heldout (fit on test)",
                                                                  args.n_boot, 4)]
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
    md += ["## Lift (out-of-fold AUC, 95% bootstrap CI; delta is paired)", "", _md_table(lift), "",
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
    md += ["## Production model on this sample", "", _md_table(pm), ""]

    # 4. Production-verdict strata.
    strata, n_join = production_strata(ok, args.assessments, cities)
    strata.to_csv(os.path.join(args.out_dir, "strata.csv"), index=False)
    md += [f"## Production-verdict strata ({n_join} of {len(ok)} sample labels have an assessment)", "",
           _md_table(strata), ""]

    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    oof = pd.concat(oof_all)
    if args.private_dir:
        os.makedirs(args.private_dir, exist_ok=True)
        oof.to_csv(os.path.join(args.private_dir, "oof_predictions.csv"), index=False)
    print("\n".join(md))


def production_strata(ok, assessments_path, cities):
    """Join the sample to the production assessments (one row per ``label_uid``, ``validate="one_to_one"``) and
    tabulate, per type and pool, the share of labels not on a sidewalk / not within 1 m of one, overall and by the
    sample's human class. Returns ``(table, n_joined)``."""
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
    return pd.DataFrame(rows), int(j.ai_result.notna().sum())


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
        sheet.save(args.doc_fig, optimize=True)
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
    s.add_argument("--rawlabels-dir", required=True, help="for pano_y (the horizon check)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_features)

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
