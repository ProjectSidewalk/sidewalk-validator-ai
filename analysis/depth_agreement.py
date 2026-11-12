"""Depth at the label: how the served depth model compares with a flat-ground reference, a larger encoder and
Google's depth map (issue #21). Write-up: ``docs/depth-ablation-2026-11.md``.

Every crop is ``int(6100 / depth_m)`` px wide on each side, with ``depth_m`` read at the centre of a 2048 px view
by Depth Anything V2 metric (vKITTI, ViT-B), the model sidewalk-ai-api serves. Nobody had checked that depth
against anything. This module provides the references and the comparisons:

==================  ======  ==================================================================================
subcommand          where   what
==================  ======  ==================================================================================
``trig``            Mac     ViT-B against the flat-ground distance ``h / sin(depression)`` for every round-3
                            row, at three camera heights; aggregates only leave ``~/Data``
``clipping``        Mac     how many crops touch the view edge, at 4000 / 6100 / 8000
``sample``          Mac     the seeded, city-balanced sample of public-city labels for the klone arms
``depth-pass``      klone   ViT-B (re-run) and ViT-L depth at each sampled label, rendered as round 3 did
``google-depth``    either  Google's plane-model depth along the crop's own centre ray (``.depth.npz``)
``compare``         Mac     the agreement statistics and the read-out of rules R0 to R2
``report``          Mac     ``tables.md`` and binned figures for the doc (pasted once; the doc is not rewritten)
==================  ======  ==================================================================================

The flat-ground reference is an assumption (a nominal camera height, level ground, a label on the ground), not a
measurement, so ``depth_m / trig_m`` is ViT-B *relative to that reference*, never "ViT-B's error".

Nothing here edits a hot file: ``data_generation`` and ``crop_pose`` are imported, and the geometry comes from
the pinned ``sidewalk_panorama_tools`` (imported lazily, so ``trig``, ``clipping`` and ``sample`` run without it).

Example (the commands of the doc's "Reproduce" section)::

    R3=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; D=~/Data/sidewalk-validator-ai-depth-2026-11
    python analysis/depth_agreement.py trig --root $R3 --rawlabels-dir $R3/rawlabels \\
        --cities-json $D/cities.json --check-pins --out-dir $D/trig
    python analysis/depth_agreement.py clipping --root $R3 --out $D/clipping.csv
    python analysis/depth_agreement.py sample --root $R3 --rawlabels-dir $R3/rawlabels \\
        --cities-json $D/cities.json --out-dir $D --public-out data/depth-agreement-2026-11
    python analysis/depth_agreement.py report --trig-dir $D/trig --clipping $D/clipping.csv \\
        --assets docs/depth-ablation-2026-11
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import crop_pose  # noqa: E402  (imported, never edited: a hot file)
import data_generation as dg  # noqa: E402  (imported, never edited: a hot file)

SEED = 20261112
N_PER_TYPE = 200
OVERSAMPLE = 3
LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
GROUND_TYPES = ("CurbRamp", "NoCurbRamp", "SurfaceProblem", "Crosswalk")
CAMERA_HEIGHTS_M = (2.5, 2.3, 2.9)          # nominal first; 2.3 and 2.9 are the sensitivity
MIN_DEPRESSION_DEG = 2.0
GSV_BETA = 0.95
MIN_VALID_CELLS = 5
MIN_GROUND_VERTICALITY = 0.7
MIN_FORMAT_VERSION = 3
RANGE_M = (0.5, 80.0)
BOX_HALF_PX = 16                            # the 33 x 33 px box of the jitter measure
CROP_DIFF_THRESHOLDS = (0.10, 0.25, 0.50)
SLOPE_BAND = (0.9, 1.1)
MIN_CELL_N = 20
N_BOOT = 2000
FILE_LEVEL_UNUSABLE = ("no_artifact", "old_format", "unreadable")
FIXED_HALF_PX_51 = 238                      # #51's fixed-size control (CurbRamp's median depth-sized half-width)
F_PX = dg.VIEW_SIZE / 2                     # focal length in view px: a 90 degree view 2048 px wide
YEAR_BANDS = ((2007, 2012), (2013, 2017), (2018, 2019), (2020, 2022), (2023, 2026))
HIST_MIN_COUNT = 10                         # 2-D histogram bins with fewer labels are blanked in the figure data
POOLED_LABEL = "non-public deployments ({k})"
NONE = "(none)"

#: sha256 of round 3's inputs (docs/depth-ablation-2026-11.md, "Reproduce"). ``rawlabels/*.csv`` is the combined
#: hash ``cd rawlabels && shasum -a 256 *.csv | shasum -a 256``; :func:`combined_sha256` reproduces it.
ROUND3_PINS = {
    "CurbRamp/split.csv": "f2ec89f8032c9b6b7c1533bea7ac82d82b7c139847b14286dd53d932a2195610",
    "NoCurbRamp/split.csv": "fbb1e53bd121bdd5eaca8b12d8fcf5a6ec3f6102920f1f68735eb78c9703ce33",
    "Obstacle/split.csv": "d41e865f45b5f0b91d3bc81c1a60dae5e3effea2d30dd92c7883207feb87e8aa",
    "SurfaceProblem/split.csv": "c48cac6c889b1df2c125238401f8f886904ae2a28c8e43628f5e2e799abd8fdd",
    "Crosswalk/split.csv": "91001f64b778ec3d59b381d50911c6bd8ea35449052574d97eb4c9380d37f00f",
    "rawlabels/*.csv": "42c0dcfab78943d3e329b21c3d86322a0894469d4fd11eb5ea2f576d5dd6999a",
}


# ---------------------------------------------------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path):
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def combined_sha256(directory, pattern="*.csv"):
    """The hash of ``cd <directory> && shasum -a 256 <pattern> | shasum -a 256``: one line per file, sorted by name.

    Returns ``(combined, {file_name: sha256})``.
    """
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(directory, pattern)))
    per = {name: sha256_file(os.path.join(directory, name)) for name in files}
    text = "".join(f"{per[name]}  {name}\n" for name in files)
    return hashlib.sha256(text.encode()).hexdigest(), per


def input_pins(root, rawlabels_dir, label_types=LABEL_TYPES):
    """``(pins, per_rawlabels_file)``: the split files' hashes keyed as :data:`ROUND3_PINS`, plus the combined
    rawLabels hash. Per-file rawLabels hashes are keyed by file name, which names deployments: keep that record
    local (``trig.json``), never committed."""
    pins = {f"{t}/split.csv": sha256_file(os.path.join(root, t, "split.csv")) for t in label_types}
    combined, per = combined_sha256(rawlabels_dir)
    pins["rawlabels/*.csv"] = combined
    return pins, per


def check_pins(pins, expected=ROUND3_PINS):
    """The keys of ``pins`` whose hash differs from ``expected`` (or is absent from it); empty when all match.

    >>> check_pins({"a": "1", "b": "2"}, {"a": "1", "b": "3"})
    ['b']
    """
    return sorted(k for k, v in pins.items() if expected.get(k) != v)


def git_commit():
    """``(commit, dirty)`` of the repo this file is in; ``(None, None)`` when git cannot say."""
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def write_json(path, obj):
    """Write ``obj`` as indented, key-sorted JSON (deterministic bytes for the same content)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")


# ---------------------------------------------------------------------------------------------------------------------
# Public deployments
# ---------------------------------------------------------------------------------------------------------------------

def public_from_listing(listing):
    """The public city_ids of a ``/v3/api/cities`` listing (``data_generation.discover_cities``, public only)."""
    return frozenset(dg.discover_cities(listing, public_only=True))


def resolve_public(public_cities=None, cities_json=None):
    """``(public_city_ids, record)`` from, in order: a JSON list of city ids (``--public-cities``, tests and
    re-runs), a saved listing (``--cities-json``, the day's snapshot), or the live API (one request).

    ``record`` says which, with the sha256 of the file or of the canonical JSON of the live response.
    """
    if public_cities:
        with open(public_cities) as f:
            ids = json.load(f)
        return frozenset(ids), {"from": "public_cities", "path": os.path.abspath(public_cities),
                                "sha256": sha256_file(public_cities), "n_public": len(ids)}
    if cities_json:
        with open(cities_json) as f:
            listing = json.load(f)
        pub = public_from_listing(listing)
        return pub, {"from": "cities_json", "path": os.path.abspath(cities_json), "sha256": sha256_file(cities_json),
                     "fetched_on": listing.get("fetched_on") if isinstance(listing, dict) else None,
                     "n_public": len(pub)}
    listing, source = dg.load_city_listing(refresh=True)
    canon = json.dumps(listing, sort_keys=True).encode()
    pub = public_from_listing(listing)
    return pub, {"from": "api", "url": source["url"], "fetched_on": source["fetched_on"],
                 "sha256": hashlib.sha256(canon).hexdigest(), "n_public": len(pub)}


# ---------------------------------------------------------------------------------------------------------------------
# Frame: round-3 split rows with their stored pano coordinates
# ---------------------------------------------------------------------------------------------------------------------

SPLIT_DTYPES = {"label_id": str, "pano_id": str, "city": str, "label_uid": str}


def load_frame(root, rawlabels_dir, label_types=LABEL_TYPES):
    """Round-3 ``split.csv`` rows of ``label_types`` joined one-to-one to their rawLabels ``pano_x``, ``pano_y``,
    ``pano_width``, ``pano_height`` on ``label_uid`` (``city:label_id``; ``label_id`` alone is per-city).

    Raises ValueError on a duplicated ``label_uid`` (in the splits or in an export), a label with no export row,
    or an export whose ``pano_id`` differs from the split's.
    """
    parts = [pd.read_csv(os.path.join(root, t, "split.csv"), dtype=SPLIT_DTYPES, low_memory=False)
             for t in label_types]
    frame = pd.concat(parts, ignore_index=True)
    if not frame.label_uid.is_unique:
        raise ValueError(f"{int(frame.label_uid.duplicated().sum())} duplicated label_uid in the split files")
    if not (frame.label_uid == frame.city + ":" + frame.label_id).all():
        raise ValueError("label_uid is not city:label_id on some rows")
    raw = []
    for city in sorted(frame.city.unique()):
        r = pd.read_csv(os.path.join(rawlabels_dir, f"{city}.csv"), dtype={"label_id": str, "pano_id": str},
                        usecols=["label_id", "pano_id", "pano_x", "pano_y", "pano_width", "pano_height"])
        r["label_uid"] = city + ":" + r.label_id
        raw.append(r.drop(columns="label_id").rename(columns={"pano_id": "raw_pano_id"}))
    raw = pd.concat(raw, ignore_index=True)
    raw = raw[raw.label_uid.isin(frame.label_uid)]
    if not raw.label_uid.is_unique:
        raise ValueError(f"{int(raw.label_uid.duplicated().sum())} labels appear twice in their rawLabels export")
    out = frame.merge(raw, on="label_uid", how="left", validate="one_to_one")
    if out.pano_y.isna().any():
        raise ValueError(f"{int(out.pano_y.isna().sum())} labels have no rawLabels row")
    if (out.raw_pano_id != out.pano_id).any():
        raise ValueError(f"{int((out.raw_pano_id != out.pano_id).sum())} labels' rawLabels pano_id differs")
    return out.drop(columns="raw_pano_id")


def year_band(image_capture_date):
    """The capture-year band of a ``YYYY-MM`` date (:data:`YEAR_BANDS`), ``(none)`` when unknown.

    >>> [year_band(v) for v in ("2019-05", "2008-01", None, "2030-01")]
    ['2018-2019', '2007-2012', '(none)', '(none)']
    """
    try:
        year = int(str(image_capture_date)[:4])
    except ValueError:
        return NONE
    for lo, hi in YEAR_BANDS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return NONE


# ---------------------------------------------------------------------------------------------------------------------
# trig: the flat-ground reference (CPU, every row)
# ---------------------------------------------------------------------------------------------------------------------

def depression_deg(pano_y, pano_height):
    """Gravity depression of a stored (level-frame) pixel row, in degrees below the horizon.

    >>> depression_deg(np.array([4096.0, 8192.0, 0.0]), 8192).tolist()
    [0.0, 90.0, -90.0]
    """
    return np.asarray(pano_y, dtype=float) / np.asarray(pano_height, dtype=float) * 180.0 - 90.0


def trig_m(depression, camera_height_m, min_depression_deg=MIN_DEPRESSION_DEG):
    """Ray length to level ground ``camera_height_m`` below the camera: ``h / sin(depression)``; NaN at or above
    ``min_depression_deg`` from the horizon, where ``1 / sin`` explodes.

    >>> trig_m(np.array([30.0, 2.0, 1.0]), 2.5).round(6).tolist()
    [5.0, nan, nan]
    """
    d = np.asarray(depression, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = camera_height_m / np.sin(np.radians(d))
    return np.where(d > min_depression_deg, out, np.nan)


def physical_width_m(d_src, d_ref, constant=dg.CROP_DEPTH_CONSTANT, f_px=F_PX):
    """Width in metres, at the reference distance ``d_ref``, of a crop sized ``int(constant / d_src)`` px per side.

    >>> float(physical_width_m(10.0, 10.0))  # 2 * 6100 / 1024
    11.9140625
    """
    return 2.0 * constant / f_px * np.asarray(d_ref, dtype=float) / np.asarray(d_src, dtype=float)


def trig_frame(frame, camera_height_m=CAMERA_HEIGHTS_M[0], heights=CAMERA_HEIGHTS_M):
    """``frame`` plus ``depression_deg``, ``trig_m`` (at ``camera_height_m``), ``trig_m_<h>`` for every height,
    ``ratio`` (``depth_m / trig_m``), ``ln_ratio``, ``width_m``, ``why_excluded`` and ``usable``.

    Excluded from the headline (``usable`` False), with the reason recorded: AI-authored rows (``ai_authored``;
    their stored pixel is in the rig frame, so the depression is off by the tilt) and rows within
    ``MIN_DEPRESSION_DEG`` of the horizon or above it (``near_horizon``). AI rows still get their numbers, for a
    row of their own.
    """
    tf = frame.copy()
    tf["depth_m"] = pd.to_numeric(tf.depth_m, errors="coerce")
    tf["depression_deg"] = depression_deg(tf.pano_y, tf.pano_height)
    for h in heights:
        tf[f"trig_m_{h}"] = trig_m(tf.depression_deg, h)
    tf["trig_m"] = trig_m(tf.depression_deg, camera_height_m)
    tf["ratio"] = tf.depth_m / tf.trig_m
    tf["ln_ratio"] = np.log(tf.ratio)
    tf["width_m"] = physical_width_m(tf.depth_m, tf.trig_m)
    ai = tf.get("author", pd.Series("human", index=tf.index)).astype(str).eq("ai")
    near = ~(tf.depression_deg > MIN_DEPRESSION_DEG)
    bad_depth = ~(np.isfinite(tf.depth_m) & (tf.depth_m > 0))
    tf["why_excluded"] = np.select([bad_depth, ai, near], ["no_depth", "ai_authored", "near_horizon"], "")
    tf["usable"] = tf.why_excluded.eq("")
    return tf


def _slice_columns(df):
    """Add the slice keys used by every table: ``severity_s``, ``tags_list``, ``year_band``."""
    from analysis import slices
    out = df.copy()
    sev = pd.to_numeric(out.get("severity", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["severity_s"] = [NONE if not np.isfinite(v) else str(int(v)) for v in sev]
    out["tags_list"] = [slices.parse_tags(v) or [NONE] for v in out.get("tags", pd.Series(None, index=out.index))]
    out["year_band"] = [year_band(v) for v in out.get("image_capture_date", pd.Series(None, index=out.index))]
    return out


def _quantiles(values, qs=(0.5, 0.05, 0.95)):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [np.nan] * len(qs)
    return [float(np.quantile(v, q)) for q in qs]


def summarize(df, keys, values, train=None, min_n=MIN_CELL_N, explode_tags=False):
    """One row per group of ``keys``: ``n``, ``train_n`` (rows of ``train`` in the same group), and for each column
    in ``values`` its median, 5th and 95th percentile; ``small`` is True and the statistics blank when n < min_n
    (the row stays, so a thin slice is visible, never dropped).

    >>> d = pd.DataFrame({"t": ["a"] * 25 + ["b"] * 3, "x": list(range(25)) + [1, 2, 3]})
    >>> s = summarize(d, ["t"], ["x"], train=d)
    >>> s[["t", "n", "train_n", "x_med", "small"]].values.tolist()
    [['a', 25, 25, 12.0, False], ['b', 3, 3, nan, True]]
    """
    def expand(d):
        return d.explode("tags_list").rename(columns={"tags_list": "tag"}) if explode_tags else d
    data = expand(df)
    tr = expand(train) if train is not None else None
    rows = []
    for key, g in data.groupby(list(keys), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        rec["n"] = int(len(g))
        if tr is not None:
            m = np.ones(len(tr), bool)
            for k, v in zip(keys, key):
                m &= (tr[k] == v).to_numpy()
            rec["train_n"] = int(m.sum())
        small = len(g) < min_n
        for col in values:
            med, p05, p95 = _quantiles(g[col]) if not small else (np.nan, np.nan, np.nan)
            rec.update({f"{col}_med": med, f"{col}_p05": p05, f"{col}_p95": p95})
        rec["small"] = small
        rows.append(rec)
    return pd.DataFrame(rows)


def pool_cities(df, public, city_col="city", k=None):
    """``df`` with every city not in ``public`` replaced by one pooled label naming only how many there are
    (``k``; default: how many non-public cities ``df`` has). Pass the same ``k`` to every frame that must share
    the label (the rows and their training counts).

    >>> pool_cities(pd.DataFrame({"city": ["a", "x", "y"]}), {"a"}).city.tolist()
    ['a', 'non-public deployments (2)', 'non-public deployments (2)']
    """
    out = df.copy()
    if k is None:
        k = len(set(out[city_col]) - set(public))
    out[city_col] = out[city_col].where(out[city_col].isin(public), POOLED_LABEL.format(k=k))
    return out


# --- slopes and bootstraps -------------------------------------------------------------------------------------------

def slope_bracket(ln_a, ln_b, w=None):
    """Errors-in-variables summary of ``ln a`` against ``ln b``: ``beta_yx`` (OLS of a on b), ``beta_xy`` (the
    inverse of OLS of b on a), the reduced-major-axis slope ``rma = sign(r) sd(a) / sd(b)``, and ``r``.

    Noise in b pulls ``beta_yx`` toward 0, noise in a pushes ``beta_xy`` away from it, so with independent errors
    the true slope lies between them. ``w`` are optional frequency weights (the bootstrap's).

    >>> b = np.log(np.arange(1.0, 50.0)); s = slope_bracket(np.log(2) + b, b)
    >>> [round(float(s[k]), 6) for k in ("beta_yx", "beta_xy", "rma", "r")]
    [1.0, 1.0, 1.0, 1.0]
    """
    a, b = np.asarray(ln_a, float), np.asarray(ln_b, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    sw = w.sum()
    ma, mb = (w * a).sum() / sw, (w * b).sum() / sw
    da, db = a - ma, b - mb
    vaa, vbb, vab = (w * da * da).sum() / sw, (w * db * db).sum() / sw, (w * da * db).sum() / sw
    r = vab / math.sqrt(vaa * vbb) if vaa > 0 and vbb > 0 else np.nan
    return {"beta_yx": vab / vbb if vbb > 0 else np.nan,
            "beta_xy": vaa / vab if vab != 0 else np.nan,
            "rma": (np.sign(r) * math.sqrt(vaa / vbb)) if vbb > 0 and np.isfinite(r) else np.nan,
            "r": r}


def bracket_verdict(beta_yx, beta_xy, band=SLOPE_BAND):
    """R2's three-valued reading of a slope bracket against ``band``.

    >>> bracket_verdict(0.95, 1.05), bracket_verdict(0.6, 0.8), bracket_verdict(0.747, 0.968)
    ('inside', 'outside', 'straddles')
    """
    lo, hi = sorted((beta_yx, beta_xy))
    if band[0] <= lo and hi <= band[1]:
        return "inside"
    if hi < band[0] or lo > band[1]:
        return "outside"
    return "straddles"


def _codes(values):
    return pd.factorize(pd.Series(values).astype(str), sort=True)[0]


def boot_weights(cluster, strata=None, n_boot=N_BOOT, seed=SEED):
    """Yield ``n_boot`` integer row-weight arrays from a cluster bootstrap: within each stratum, draw as many
    clusters as it has, with replacement; a row's weight is how often its cluster was drawn.

    Panoramas within city: ``cluster = city:pano_id``, ``strata = city`` (cities fixed). Whole cities (the
    "unseen deployment" reading): ``cluster = city``, ``strata = None``. Seeded, so the same call gives the same
    intervals.
    """
    cl = _codes(cluster)
    st = np.zeros(len(cl), int) if strata is None else _codes(strata)
    n_cl = cl.max() + 1 if len(cl) else 0
    cl_stratum = np.zeros(n_cl, int)
    cl_stratum[cl] = st
    order = np.argsort(cl_stratum, kind="stable")          # clusters grouped by stratum
    sizes = np.bincount(cl_stratum, minlength=st.max() + 1 if len(st) else 0)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    slot_start = np.repeat(starts, sizes)                  # for every draw slot: where its stratum's clusters begin
    slot_size = np.repeat(sizes, sizes)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = order[slot_start + (rng.random(n_cl) * slot_size).astype(int)]
        counts = np.bincount(pick, minlength=n_cl)
        yield counts[cl]


def weighted_quantile(values, weights, q):
    """The ``q`` quantile of ``values`` repeated ``weights`` times (lower value at a tie), without repeating.

    >>> float(weighted_quantile(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 5]), 0.5))
    3.0
    """
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values)[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    return v[min(np.searchsorted(cw, q * cw[-1], side="left"), len(v) - 1)]


def _ci(samples, level=0.95):
    s = np.asarray([x for x in samples if np.isfinite(x)])
    if not len(s):
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def bracket_with_ci(ln_a, ln_b, cluster, strata, n_boot=N_BOOT, seed=SEED):
    """:func:`slope_bracket` with 95 % cluster-bootstrap intervals on each end and on the RMA slope."""
    point = slope_bracket(ln_a, ln_b)
    boots = [slope_bracket(ln_a, ln_b, w) for w in boot_weights(cluster, strata, n_boot, seed)] if n_boot else []
    out = dict(point)
    for k in ("beta_yx", "beta_xy", "rma"):
        out[f"{k}_lo"], out[f"{k}_hi"] = _ci([b[k] for b in boots])
    out["verdict"] = bracket_verdict(point["beta_yx"], point["beta_xy"])
    out["n"] = int(len(np.asarray(ln_a)))
    return out


def median_with_ci(values, cluster, strata, n_boot=N_BOOT, seed=SEED):
    """``(median, lo, hi)``: the median with a 95 % cluster-bootstrap interval."""
    v = np.asarray(values, float)
    med = float(np.median(v)) if len(v) else np.nan
    if not n_boot or not len(v):
        return med, np.nan, np.nan
    order = np.argsort(v, kind="stable")      # sort once; each resample only re-weights
    vs = v[order]
    meds = []
    for w in boot_weights(cluster, strata, n_boot, seed):
        cw = np.cumsum(w[order])
        meds.append(vs[min(np.searchsorted(cw, 0.5 * cw[-1], side="left"), len(vs) - 1)])
    lo, hi = _ci(meds)
    return med, lo, hi


def crop_moved_shares(ln_ratio, thresholds=CROP_DIFF_THRESHOLDS, weights=None):
    """For each t: the share of labels whose crop differs by more than t, ``|ln ratio| > ln(1 + t)`` (symmetric: a
    crop 10 % larger and one 10 % smaller in log terms count alike), with a Wilson interval on the Kish effective
    n. ``{"n", "n_eff", "t=0.1": {"share", "count", "lo", "hi"}, ...}``.

    >>> s = crop_moved_shares(np.log([1.2, 1 / 1.2, 1.0, 1.0]))
    >>> s["t=0.1"]["share"], s["n_eff"]
    (0.5, 4.0)
    """
    x = np.abs(np.asarray(ln_ratio, float))
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(x) & np.isfinite(w)
    x, w = x[keep], w[keep]
    n_eff = float(w.sum() ** 2 / (w * w).sum()) if len(w) and (w * w).sum() > 0 else 0.0
    out = {"n": int(len(x)), "n_eff": n_eff}
    for t in thresholds:
        moved = x > math.log1p(t)
        share = float((w * moved).sum() / w.sum()) if len(w) else np.nan
        lo, hi = wilson(share, n_eff)
        out[f"t={t:g}"] = {"share": share, "count": int(moved.sum()), "lo": lo, "hi": hi}
    return out


def wilson(p, n, z=1.959963984540054):
    """Wilson score interval of a proportion ``p`` on (effective) sample size ``n``.

    >>> [round(x, 4) for x in wilson(0.95, 200)]
    [0.9104, 0.9726]
    """
    if not n or not np.isfinite(p):
        return np.nan, np.nan
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


# --- trig tables ------------------------------------------------------------------------------------------------------

def table(name, title, df, note=None, digits=3):
    """A table as stored in ``trig_tables.json`` / ``compare_tables.json`` and rendered by ``report``: rounded
    values, NaN as None."""
    rows = []
    for rec in df.to_dict("records"):
        out = {}
        for k, v in rec.items():
            if isinstance(v, (float, np.floating)):
                v = None if not np.isfinite(v) else round(float(v), digits)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            out[k] = v
        rows.append(out)
    return {"name": name, "title": title, "note": note, "columns": list(df.columns), "rows": rows}


def _type_order(df, col="label_type"):
    rank = {t: i for i, t in enumerate(LABEL_TYPES)}
    return df.sort_values(by=col, key=lambda s: s.map(lambda v: rank.get(v, len(rank))), kind="stable")


def hist2d_counts(x, y, bins=24, min_count=HIST_MIN_COUNT):
    """A binned 2-D histogram (edges and counts) with bins under ``min_count`` set to 0: aggregate figure data
    that does not place any single label."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    xe = np.quantile(x[keep], [0.005, 0.995])
    ye = np.quantile(y[keep], [0.005, 0.995])
    counts, xedges, yedges = np.histogram2d(x[keep], y[keep], bins=bins, range=[xe, ye])
    counts[counts < min_count] = 0
    return {"x_edges": [round(v, 4) for v in xedges], "y_edges": [round(v, 4) for v in yedges],
            "counts": counts.astype(int).tolist(), "min_count": min_count}


def trig_tables(tf, public, n_boot=N_BOOT, seed=SEED):
    """The aggregate tables of the trig arm from a :func:`trig_frame` (nominal height): a list of :func:`table`
    dicts plus figure data. Only cities in ``public`` are named; all others are pooled into one row."""
    sl = _slice_columns(tf)
    use = sl[sl.usable]
    train = sl[sl.split.eq("train")] if "split" in sl else None
    vals = ["ratio", "width_m", "trig_m", "depth_m", "depression_deg"]
    tables = []

    by_type = _type_order(summarize(use, ["label_type"], vals, train))
    allrow = summarize(use.assign(label_type="all types"), ["label_type"], vals, train.assign(label_type="all types"))
    ground = use[use.label_type.isin(GROUND_TYPES)]
    grow = summarize(ground.assign(label_type="ground types"), ["label_type"], vals,
                     train[train.label_type.isin(GROUND_TYPES)].assign(label_type="ground types"))
    head = pd.concat([by_type, grow, allrow], ignore_index=True)
    cis = []
    for _, r in head.iterrows():
        g = use if r.label_type == "all types" else ground if r.label_type == "ground types" else \
            use[use.label_type == r.label_type]
        _, lo, hi = median_with_ci(g.ratio.to_numpy(), g.city + ":" + g.pano_id, g.city, n_boot, seed)
        cis.append((lo, hi))
    head["ratio_med_lo"], head["ratio_med_hi"] = zip(*cis)
    tables.append(table("by_type", "ViT-B / flat-ground ratio, implied crop width and distances by label type",
                        head[["label_type", "n", "train_n", "ratio_med", "ratio_med_lo", "ratio_med_hi",
                              "ratio_p05", "ratio_p95", "width_m_med", "width_m_p05", "width_m_p95", "trig_m_med",
                              "depth_m_med", "depression_deg_med", "small"]],
                        note="Usable rows: human-authored, depression > 2 deg. Interval: 95 % bootstrap of the "
                             "median over panoramas within city. train_n, here and in every slice: rows of the "
                             "slice in the round-3 train split, used here or not (AI-authored and near-horizon "
                             "rows included), i.e. what the classifier trained on."))

    heights = []
    for h in CAMERA_HEIGHTS_M:
        rr = use.depth_m / use[f"trig_m_{h}"]
        for t, g in [("all types", rr), ("ground types", rr[use.label_type.isin(GROUND_TYPES)])] + \
                [(t, rr[use.label_type == t]) for t in LABEL_TYPES]:
            med, p05, p95 = _quantiles(g)
            heights.append({"camera_height_m": h, "label_type": t, "n": int(g.notna().sum()), "ratio_med": med,
                            "ratio_p05": p05, "ratio_p95": p95,
                            "width_m_med": float(np.median(physical_width_m(1.0, 1.0 / g.dropna())))})
    tables.append(table("heights", "Camera-height sensitivity: the ratio scales as 1 / h, nothing else moves",
                        pd.DataFrame(heights)))

    for name, title, keys, explode in (
            ("type_severity", "By label type x severity", ["label_type", "severity_s"], False),
            ("type_tag", "By label type x tag (a label with several tags counts in each)", ["label_type", "tag"],
             True),
            ("type_class", "By label type x label class", ["label_type", "label_class"], False),
            ("type_year", "By label type x capture-year band", ["label_type", "year_band"], False),
            ("type_split", "By label type x round-3 split", ["label_type", "split"], False)):
        s = _type_order(summarize(use, keys, ["ratio", "width_m"], train, explode_tags=explode))
        tables.append(table(name, title, s[keys + ["n", "train_n", "ratio_med", "ratio_p05", "ratio_p95",
                                                  "width_m_med", "small"]]))

    k_private = len(set(sl.city) - set(public))
    pooled = pool_cities(use, public, k=k_private)
    ptrain = pool_cities(train, public, k=k_private)
    s = summarize(pooled, ["city"], ["ratio", "width_m", "depression_deg"], ptrain)
    is_pool = s.city.str.startswith("non-public")
    s = pd.concat([s[~is_pool].sort_values("city"), s[is_pool]], ignore_index=True)
    tables.append(table("city", "By deployment (public deployments named; all others pooled)",
                        s[["city", "n", "train_n", "ratio_med", "ratio_p05", "ratio_p95", "width_m_med",
                           "depression_deg_med", "small"]],
                        note="Hilly deployments break the flat-ground assumption; a deployment far from the "
                             "pooled median is where to look first."))

    slopes = []
    for t, g in [("all types", use), ("ground types", ground)] + [(t, use[use.label_type == t]) for t in LABEL_TYPES]:
        b = bracket_with_ci(np.log(g.depth_m.to_numpy()), np.log(g.trig_m.to_numpy()), g.city + ":" + g.pano_id,
                            g.city, n_boot, seed)
        slopes.append({"label_type": t, **b})
    tables.append(table("slopes", "Proportionality in log space: ln ViT-B against ln flat-ground distance",
                        pd.DataFrame(slopes)[["label_type", "n", "beta_yx", "beta_yx_lo", "beta_yx_hi", "beta_xy",
                                              "beta_xy_lo", "beta_xy_hi", "rma", "rma_lo", "rma_hi", "r", "verdict"]],
                        note="beta_yx: OLS of ln ViT-B on ln trig; beta_xy: inverse of OLS of ln trig on ln ViT-B. "
                             "The true slope lies in the bracket when the errors are independent; the verdict is "
                             "against the [0.9, 1.1] band. Height-independent."))

    dec = use.assign(trig_decile=pd.qcut(use.trig_m, 10, labels=False) + 1)
    d = dec.groupby("trig_decile").agg(n=("ratio", "size"), trig_m_med=("trig_m", "median"),
                                       depth_m_med=("depth_m", "median"), ratio_med=("ratio", "median"),
                                       width_m_med=("width_m", "median")).reset_index()
    tables.append(table("trig_deciles", "Descriptive only: by decile of the flat-ground distance", d,
                        note="Binning on a noisy reference produces a falling ratio even under exact "
                             "proportionality; read the slope bracket, not this table, for proportionality."))

    s_med = float(np.median(use.ratio))
    c_match = dg.CROP_DEPTH_CONSTANT / s_med
    clip = []
    for src, dist, c in (("ViT-B (depth_m)", use.depth_m, dg.CROP_DEPTH_CONSTANT),
                         ("flat ground at the scale-matched constant", use.trig_m, c_match),
                         ("flat ground at 6100 (not scale-matched)", use.trig_m, dg.CROP_DEPTH_CONSTANT)):
        half = np.floor(c / dist)
        clip.append({"sizing_source": src, "constant": c, "n": int(len(dist)),
                     "clipped_share": float((half > dg.VIEW_SIZE // 2).mean())})
    tables.append(table("clip_by_source", "Clipped share when crops are sized by each source at its matched "
                        "constant", pd.DataFrame(clip), digits=5))

    cr = sl[sl.label_type.eq("CurbRamp")]
    ln = np.log(pd.to_numeric(cr.crop_half_px) / FIXED_HALF_PX_51)
    ys = crop_moved_shares(ln)
    tables.append(table("yardstick_51", "#51's contrast as a yardstick: CurbRamp crops moved by the fixed-238 "
                        "control", pd.DataFrame([{"t": k[2:], "share": v["share"], "count": v["count"],
                                                  "n": ys["n"]} for k, v in ys.items() if k.startswith("t=")]),
                        note="|ln(crop_half_px / 238)| > ln(1 + t), all round-3 CurbRamp rows."))

    excl = sl.groupby(["label_type", "why_excluded"]).size().rename("n").reset_index()
    excl["why_excluded"] = excl.why_excluded.replace("", "used")
    tables.append(table("rows", "Rows used and excluded, by reason", _type_order(excl)))
    ai = sl[sl.why_excluded.eq("ai_authored")]
    if len(ai):
        med, p05, p95 = _quantiles(ai.ratio)
        tables.append(table("ai_rows", "AI-authored rows on their own (rig-frame pixel: depression off by the tilt)",
                            pd.DataFrame([{"n": len(ai), "ratio_med": med, "ratio_p05": p05, "ratio_p95": p95}])))

    figures = {"hist_ln_trig_ln_vitb": hist2d_counts(np.log(use.trig_m), np.log(use.depth_m)),
               "deciles": d[["trig_m_med", "depth_m_med", "ratio_med"]].round(4).to_dict("list")}
    return {"tables": tables, "figures": figures,
            "headline": {"camera_height_m": CAMERA_HEIGHTS_M[0], "n": int(len(use)), "ratio_median": s_med,
                         "scale_matched_constant": c_match}}


def cmd_trig(args):
    pins, per_raw = input_pins(args.root, args.rawlabels_dir)
    bad = check_pins(pins)
    if args.check_pins and bad:
        sys.exit(f"input hashes differ from ROUND3_PINS for: {', '.join(bad)}; stopping.")
    public, listing_rec = resolve_public(args.public_cities, args.cities_json)
    frame = load_frame(args.root, args.rawlabels_dir)
    tf = trig_frame(frame, args.camera_height_m[0], tuple(args.camera_height_m))
    out = trig_tables(tf, public, args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    keep = ["label_uid", "city", "pano_id", "label_type", "author", "split", "depth_m", "pano_y", "pano_height",
            "depression_deg", "trig_m"] + [f"trig_m_{h}" for h in args.camera_height_m] + \
        ["ratio", "width_m", "why_excluded", "usable"]
    csv_path = os.path.join(args.out_dir, "trig.csv")
    tf[keep].to_csv(csv_path, index=False, float_format="%.6g")
    tables_path = os.path.join(args.out_dir, "trig_tables.json")
    write_json(tables_path, out)
    commit, dirty = git_commit()
    write_json(os.path.join(args.out_dir, "trig.json"), {
        "stage": "trig", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "inputs": pins, "inputs_match_round3_pins": not bad, "rawlabels_files": per_raw,
        "public_listing": listing_rec, "camera_heights_m": list(args.camera_height_m),
        "rows_in": int(len(tf)), "rows_used": int(tf.usable.sum()),
        "excluded": tf.why_excluded[~tf.usable].value_counts().to_dict(),
        "outputs": {"trig.csv": sha256_file(csv_path), "trig_tables.json": sha256_file(tables_path)}})
    h = out["headline"]
    print(f"{len(tf)} rows, {h['n']} used; median ViT-B / flat ground at {h['camera_height_m']} m = "
          f"{h['ratio_median']:.3f} -> {args.out_dir}")


# ---------------------------------------------------------------------------------------------------------------------
# clipping
# ---------------------------------------------------------------------------------------------------------------------

def clipping_table(root, constants=(4000, 6100, 8000), label_types=LABEL_TYPES, file="split.csv"):
    """Clipped crops by label type, pooled over deployments (none named): ``recorded`` is round 3's own ``clipped``
    column; ``c<C>`` the share a constant C would clip, ``floor(C / depth_m) > VIEW_SIZE / 2`` (what
    ``data_generation.centre_crop`` does). Rows whose ``status`` is not ``ok`` are ignored."""
    rows = []
    frames = []
    for t in label_types:
        df = pd.read_csv(os.path.join(root, t, file), low_memory=False)
        if "status" in df:
            df = df[df.status.astype(str).eq("ok")]
        df = df.assign(label_type=t)
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    for t, df in [(t, f) for t, f in zip(label_types, frames)] + [("all types", allf)]:
        d = pd.to_numeric(df.depth_m, errors="coerce")
        rec = {"label_type": t, "n": int(len(df)), "recorded_clipped": int(pd.to_numeric(df.clipped).sum())}
        for c in constants:
            clipped = np.floor(c / d) > dg.VIEW_SIZE // 2
            rec[f"c{c}_clipped"] = int(clipped.sum())
            rec[f"c{c}_share"] = float(clipped.mean()) if len(df) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def cmd_clipping(args):
    t = clipping_table(args.root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t.to_csv(args.out, index=False, float_format="%.6g")
    print(t.to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------------------------------------

def rank(seed, label_uid):
    """A label's seeded rank in [0, 1): the first 12 hex digits of ``sha256("depth-agreement:<seed>:<uid>")``.
    Independent of every other label, so adding or removing one never reorders the rest.

    >>> 0 <= rank(SEED, "seattle-wa:12345") < 1, rank(SEED, "a:1") == rank(SEED, "a:1")
    (True, True)
    """
    return int(hashlib.sha256(f"depth-agreement:{seed}:{label_uid}".encode()).hexdigest()[:12], 16) / 16 ** 12


def draw_sample(frame, public, seed=SEED, n_per_type=N_PER_TYPE, oversample=OVERSAMPLE):
    """The candidates: per label type, public cities only, one label per (city, panorama) (lowest rank), cities
    taken round-robin in rank order (every city's best label, then every city's second, ...), the first
    ``n_per_type * oversample`` kept. Adds ``rank``, ``stratum`` (the type), ``order`` (0-based within the type),
    ``set_a`` (the first ``n_per_type``), ``inclusion_weight`` (for Set A rows: the city's round-3 rows of that
    type over its Set A rows; NaN elsewhere) and ``inclusion_weight_cand`` (the same over its candidates).

    The estimand is city-balanced: each public city has about equal say; the weights give the row-mix-weighted
    sensitivity.
    """
    pub = frame[frame.city.isin(public)].copy()
    pub["rank"] = [rank(seed, u) for u in pub.label_uid]
    out = []
    for t in LABEL_TYPES:
        g = pub[pub.label_type == t]
        if not len(g):
            continue
        city_rows = g.groupby("city").size()
        one = g.sort_values("rank").drop_duplicates(["city", "pano_id"], keep="first")
        one = one.assign(pos=one.sort_values("rank").groupby("city").cumcount())
        one = one.sort_values(["pos", "rank"]).head(n_per_type * oversample).copy()
        one["stratum"] = t
        one["order"] = np.arange(len(one))
        one["set_a"] = one.order < n_per_type
        n_a = one[one.set_a].groupby("city").size()
        n_c = one.groupby("city").size()
        one["inclusion_weight"] = np.where(one.set_a, one.city.map(city_rows) / one.city.map(n_a), np.nan)
        one["inclusion_weight_cand"] = one.city.map(city_rows) / one.city.map(n_c)
        out.append(one.drop(columns="pos"))
    sample = pd.concat(out, ignore_index=True)
    assert sample.label_uid.is_unique
    return sample


def store_root(pano_path, city, pano_id):
    """The pano store root of a manifest ``pano_path``: the path with ``/<city>/<pano_id[:2]>/<pano_id>.jpg``
    stripped, checked to rebuild ``pano_path`` exactly; raises ValueError otherwise.

    >>> store_root("/gscratch/x/pano-stage/CurbRamp/seattle-wa/ab/abc.jpg", "seattle-wa", "abc")
    '/gscratch/x/pano-stage/CurbRamp'
    """
    tail = f"/{city}/{pano_id[:2]}/{pano_id}.jpg"
    if not str(pano_path).endswith(tail):
        raise ValueError(f"pano_path {pano_path!r} does not end in {tail!r}")
    root = str(pano_path)[:-len(tail)]
    if dg.pano_store_path(root, city, pano_id) != pano_path:
        raise ValueError(f"store root {root!r} does not rebuild {pano_path!r}")
    return root


def depth_artifact_relpath(city, pano_id):
    """Store-relative path of a panorama's Google depth artifact.

    >>> depth_artifact_relpath("seattle-wa", "abcDEF")
    'seattle-wa/ab/abcDEF.depth.npz'
    """
    return f"{city}/{pano_id[:2]}/{pano_id}.depth.npz"


PUBLIC_SAMPLE_COLUMNS = ["label_uid", "city", "label_type", "label_class", "rank", "order", "set_a"]


def cmd_sample(args):
    public, listing_rec = resolve_public(args.public_cities, args.cities_json)
    frame = load_frame(args.root, args.rawlabels_dir)
    sample = draw_sample(frame, public, args.seed, args.n_per_type, args.oversample)
    sample["store_root"] = [store_root(p, c, i) for p, c, i in zip(sample.pano_path, sample.city, sample.pano_id)]
    os.makedirs(args.out_dir, exist_ok=True)
    sample_path = os.path.join(args.out_dir, "sample.csv")
    sample.to_csv(sample_path, index=False)
    panos = sample[["city", "pano_id"]].drop_duplicates()
    stage = sorted(depth_artifact_relpath(c, p) for c, p in zip(panos.city, panos.pano_id))
    if args.stage_panos:
        stage += sorted(f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(panos.city, panos.pano_id))
    stage_path = os.path.join(args.out_dir, "stage_files.txt")
    with open(stage_path, "w") as f:
        f.write("\n".join(stage) + "\n")
    os.makedirs(args.public_out, exist_ok=True)
    pub_path = os.path.join(args.public_out, "sample_public.csv")
    sample[PUBLIC_SAMPLE_COLUMNS].to_csv(pub_path, index=False, float_format="%.12f")
    commit, dirty = git_commit()
    counts = sample.groupby("label_type").agg(candidates=("set_a", "size"), set_a=("set_a", "sum"),
                                              cities=("city", "nunique"))
    write_json(os.path.join(args.public_out, "sample.json"), {
        "stage": "sample", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
        "seed": args.seed, "n_per_type": args.n_per_type, "oversample": args.oversample,
        "rank": "int(sha256('depth-agreement:<seed>:<label_uid>')[:12], 16) / 16**12",
        "public_listing": {k: v for k, v in listing_rec.items() if k != "path"},
        "inputs": input_pins(args.root, args.rawlabels_dir)[0],
        "counts": {t: {k: int(v) for k, v in r.items()} for t, r in counts.iterrows()},
        "stage_files": len(stage), "panoramas": int(len(panos)),
        "outputs": {"sample.csv (local only)": sha256_file(sample_path), "stage_files.txt": sha256_file(stage_path),
                    "sample_public.csv": sha256_file(pub_path)}})
    print(f"{len(sample)} candidates ({int(sample.set_a.sum())} in Set A), {len(panos)} panoramas -> {sample_path}, "
          f"{stage_path}, {pub_path}")


# ---------------------------------------------------------------------------------------------------------------------
# The crop's centre ray, and Google's depth along it (CPU)
# ---------------------------------------------------------------------------------------------------------------------

def crop_ray_pixel(row, gsv_beta=GSV_BETA):
    """The rig pano pixel ``(x, y)`` the crop's view is centred on, i.e. where the label's ray meets the rig-frame
    rasters (the JPEG, and the depth artifact's planes).

    The pose and beta are ``crop_pose.pose_for_row(row, gsv_beta)``, exactly what ``data_generation`` rendered
    the crop with; the click latitude offset is added to y as its main loop does (``y_norm = pano_y / H +
    offset / 180``). When that pose applies a tilt (``geometry.tilt_matrix`` is not None), the pixel is
    ``pano_pose.corrected_pixel``; otherwise (AI-authored rows, no pose, beta 0, a zero pose) the stored pixel.

    >>> crop_ray_pixel({"pano_x": 100.0, "pano_y": 5000.0, "pano_width": 16384, "pano_height": 8192,
    ...                 "pano_source": "gsv", "pano_id": "p", "camera_pitch": None, "camera_roll": None})
    (100.0, 5000.0)
    """
    from sidewalk_panorama_tools import pose as pano_pose
    from sidewalk_panorama_tools.geometry import tilt_matrix
    pose = crop_pose.pose_for_row(row, gsv_beta)
    W, H = float(row["pano_width"]), float(row["pano_height"])
    x = float(row["pano_x"])
    y = float(row["pano_y"]) + (pose.click_lat_offset_deg or 0.0) / 180.0 * H
    apply = pose.beta is not None and tilt_matrix(pose.camera_pitch, pose.camera_roll, pose.beta or 0.0) is not None
    if not apply:
        return x % W, y
    p = 0.0 if pose.camera_pitch is None or not np.isfinite(pose.camera_pitch) else float(pose.camera_pitch)
    r = 0.0 if pose.camera_roll is None or not np.isfinite(pose.camera_roll) else float(pose.camera_roll)
    return pano_pose.corrected_pixel(x, y, W, H, p, r, pose.beta)


def crop_tilt_applied(row, gsv_beta=GSV_BETA):
    """Whether the crop's render applied a tilt (what the manifest's ``tilt_applied`` records)."""
    from sidewalk_panorama_tools.geometry import tilt_matrix
    pose = crop_pose.pose_for_row(row, gsv_beta)
    return pose.beta is not None and tilt_matrix(pose.camera_pitch, pose.camera_roll, pose.beta or 0.0) is not None


DepthGrid = namedtuple("DepthGrid", "depth plane_indices planes_n planes_d pitch_deg roll_deg")


def load_depth_grid(npz_path):
    """``(DepthGrid, None)`` from a ``.depth.npz``, or ``(None, status)``: ``no_artifact`` (no file),
    ``old_format`` (``format_version`` absent or below 3, so no plane fields), ``unreadable`` (anything else: a
    corrupt zip, missing or misshapen fields). Never raises. ``pitch_deg``/``roll_deg`` are NaN when absent."""
    if not os.path.exists(npz_path):
        return None, "no_artifact"
    try:
        with np.load(npz_path) as art:
            if "format_version" not in art.files or int(art["format_version"]) < MIN_FORMAT_VERSION:
                return None, "old_format"
            depth = np.asarray(art["depth"], dtype=np.float64)
            idx = np.asarray(art["plane_indices"]).astype(np.intp)
            pn = np.asarray(art["planes_n"], dtype=np.float64)
            pd_ = np.asarray(art["planes_d"], dtype=np.float64)
            pitch = float(art["pitch"]) if "pitch" in art.files else np.nan
            roll = float(art["roll"]) if "roll" in art.files else np.nan
    except Exception:  # a truncated member, a bad zip, a non-scalar pose: all "unreadable"
        return None, "unreadable"
    if (depth.ndim != 2 or idx.shape != depth.shape or pn.ndim != 2 or pn.shape[1] != 3 or pd_.shape != (len(pn),)
            or (idx.size and (idx.min() < 0 or idx.max() >= len(pn)))):
        return None, "unreadable"
    return DepthGrid(depth, idx, pn, pd_, math.degrees(pitch), math.degrees(roll)), None


def cell_ray(r, c, h, w):
    """The artifact's unit ray of cell (r, c): ``(sin t cos f, sin t sin f, cos t)``, ``t = (h - r - 0.5) / h pi``,
    ``f = (w - c - 0.5) / w 2 pi + pi / 2`` (x left, y back, z down: the artifact frame is -RFU).

    >>> cell_ray(255, 256, 256, 512).round(2).tolist()  # bottom row, centre column: straight down
    [-0.0, -0.01, 1.0]
    """
    t = (h - np.asarray(r, float) - 0.5) / h * math.pi
    f = (w - np.asarray(c, float) - 0.5) / w * 2 * math.pi + math.pi / 2
    return np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], axis=-1)


def label_ray(lookup_x, lookup_y, pano_width, pano_height):
    """The label's continuous ray in the artifact frame: ``-RFU(b, el)`` with ``b = x / W 360 - 180``,
    ``el = 90 - y / H 180``. At a cell centre it equals :func:`cell_ray`."""
    b = lookup_x / pano_width * 360.0 - 180.0
    el = 90.0 - lookup_y / pano_height * 180.0
    br, er = math.radians(b), math.radians(el)
    return -np.array([math.cos(er) * math.sin(br), math.cos(er) * math.cos(br), math.sin(er)])


def plane_sign(grid, plane):
    """sigma, fixed per plane so that the plane's own cells lie in front of the camera: the sign of the median of
    ``planes_d / (v_cell . planes_n)`` over every cell of the raster carrying that plane (+1 when it has none).
    The payload's own sign convention for ``d`` is not relied on, and a negated ``(n, d)`` gives the same
    intersection."""
    h, w = grid.depth.shape
    rr, cc = np.nonzero(grid.plane_indices == plane)
    if not len(rr):
        return 1.0
    v = cell_ray(rr, cc, h, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = grid.planes_d[plane] / (v @ grid.planes_n[plane])
    t = t[np.isfinite(t)]
    return 1.0 if not len(t) or np.median(t) >= 0 else -1.0


def ground_plane(grid, min_vertical=MIN_GROUND_VERTICALITY):
    """The ground plane index: the near-horizontal plane (``|n_z| / |n| >= min_vertical``) referenced by the most
    below-horizon cells (rows from ``(h + 1) // 2``), ties by verticality then lowest index; None when none
    qualifies. The rule of panorama-tools' ``downloaders.gsv.ground_plane_from_artifact``."""
    idx = grid.plane_indices
    support = np.bincount(idx[(idx.shape[0] + 1) // 2:].ravel(), minlength=len(grid.planes_n))
    best = None
    for i in range(1, len(grid.planes_n)):
        if support[i] == 0:
            continue
        length = float(np.linalg.norm(grid.planes_n[i]))
        if length == 0:
            continue
        vert = abs(float(grid.planes_n[i][2])) / length
        if vert < min_vertical:
            continue
        if best is None or (support[i], vert) > best[0]:
            best = ((int(support[i]), vert), i)
    return None if best is None else best[1]


def google_depth(grid, lookup_x, lookup_y, pano_width, pano_height, depression=np.nan):
    """Google's plane-model distance along the crop's centre ray: a dict with ``g_status`` (``ground``,
    ``facade``, ``sky``, ``no_plane``, ``out_of_range``), ``g_ray_m`` (signed exact intersection, the reported
    value), ``g_ray_abs_m`` (CropRunner's ``|d / (v . n)|``), ``g_cell_m`` (the raster cell), ``plane``,
    ``verticality``, ``cam_height_m`` (``|d| / |n|`` of :func:`ground_plane`), ``trig_cam_m`` (``cam_height_m /
    sin(depression)``, depression from the STORED y), ``row``, ``col``, ``n_valid``.

    Lookup: truncation to a cell, 3 x 3 neighbourhood with columns wrapped and rows not; a cell counts when
    ``depth > 0`` and its plane index > 0. Fewer than ``MIN_VALID_CELLS``: ``sky`` above the horizon row split,
    else ``no_plane``. Plane: the centre cell's if it counts, else the most common counted plane (lowest index on
    a tie). Verticality >= 0.7 is ground. A ray parallel to the plane, behind the camera, or outside RANGE_M is
    ``out_of_range``.

    Example: level ground 2.5 m below a level camera, a label 30 degrees down::

        >>> h, w = 64, 128
        >>> g = DepthGrid(np.ones((h, w)), np.ones((h, w), int), np.array([[0, 0, 0], [0, 0, 1.0]]),
        ...               np.array([0, 2.5]), 0.0, 0.0)
        >>> round(google_depth(g, 8192.0, 4096 + 30 / 180 * 8192, 16384, 8192, 30.0)["g_ray_m"], 6)
        5.0
    """
    h, w = grid.depth.shape
    out = {"g_status": None, "g_ray_m": np.nan, "g_ray_abs_m": np.nan, "g_cell_m": np.nan, "plane": None,
           "verticality": np.nan, "cam_height_m": np.nan, "trig_cam_m": np.nan, "row": None, "col": None,
           "n_valid": 0}
    gp = ground_plane(grid)
    if gp is not None:
        out["cam_height_m"] = abs(float(grid.planes_d[gp])) / float(np.linalg.norm(grid.planes_n[gp]))
        out["trig_cam_m"] = float(trig_m(np.array([depression]), out["cam_height_m"])[0])
    col = int(lookup_x / pano_width * w) % w
    row = min(max(int(lookup_y / pano_height * h), 0), h - 1)
    out.update(row=row, col=col, g_cell_m=float(grid.depth[row, col]))
    rows = [r for r in (row - 1, row, row + 1) if 0 <= r < h]
    cols = [(col + dc) % w for dc in (-1, 0, 1)]
    depth = grid.depth[np.ix_(rows, cols)]
    idx = grid.plane_indices[np.ix_(rows, cols)]
    with np.errstate(invalid="ignore"):
        counted = np.isfinite(depth) & (depth > 0) & (idx > 0)
    out["n_valid"] = int(counted.sum())
    if counted.sum() < MIN_VALID_CELLS:
        out["g_status"] = "sky" if row < (h + 1) // 2 else "no_plane"
        return out
    centre = (rows.index(row), 1)
    plane = int(idx[centre]) if counted[centre] else int(np.argmax(np.bincount(idx[counted])))
    n, d = grid.planes_n[plane], float(grid.planes_d[plane])
    length = float(np.linalg.norm(n))
    out["plane"] = plane
    out["verticality"] = abs(float(n[2])) / length if length else np.nan
    v = label_ray(lookup_x, lookup_y, pano_width, pano_height)
    dot = float(v @ n)
    if length == 0 or dot == 0:
        out["g_status"] = "out_of_range"
        return out
    t_signed = plane_sign(grid, plane) * d / dot
    out["g_ray_abs_m"] = abs(d / dot)
    if not (np.isfinite(t_signed) and t_signed > 0 and RANGE_M[0] <= t_signed <= RANGE_M[1]):
        out["g_status"] = "out_of_range"
        return out
    out["g_ray_m"] = t_signed
    out["g_status"] = "ground" if out["verticality"] >= MIN_GROUND_VERTICALITY else "facade"
    return out


def set_g(sample, google, n_per_type=N_PER_TYPE):
    """Set G membership (a boolean Series on ``sample``'s index): per type, the first ``n_per_type`` candidates in
    ``order`` whose artifact is usable at the FILE level (``g_status`` not in FILE_LEVEL_UNUSABLE). Membership
    never depends on a depth value or on whether the ray hit ground."""
    status = sample.label_uid.map(google.set_index("label_uid").g_status).fillna("no_artifact")
    usable = ~status.isin(FILE_LEVEL_UNUSABLE)
    member = pd.Series(False, index=sample.index)
    for _, g in sample[usable].sort_values("order").groupby("label_type"):
        member.loc[g.index[:n_per_type]] = True
    return member


def cmd_google_depth(args):
    sample = pd.read_csv(args.sample, dtype=SPLIT_DTYPES, low_memory=False)
    rows = []
    for r in sample.to_dict("records"):
        grid, status = load_depth_grid(os.path.join(args.store, depth_artifact_relpath(r["city"], r["pano_id"])))
        lx, ly = crop_ray_pixel(r)
        dep = float(depression_deg(r["pano_y"], r["pano_height"]))
        rec = {"label_uid": r["label_uid"], "lookup_x": lx, "lookup_y": ly, "depression_deg": dep,
               "art_pitch_deg": np.nan, "art_roll_deg": np.nan}
        if grid is None:
            rec["g_status"] = status
        else:
            rec.update(google_depth(grid, lx, ly, float(r["pano_width"]), float(r["pano_height"]), dep))
            rec.update(art_pitch_deg=grid.pitch_deg, art_roll_deg=grid.roll_deg)
        rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.6g")
    print(out.g_status.value_counts().to_string())


# ---------------------------------------------------------------------------------------------------------------------
# depth pass (GPU on klone)
# ---------------------------------------------------------------------------------------------------------------------

DEPTH_PASS_RETRY = ("error",)       # statuses re-tried on resume (prefix match); ok/no_pano/bad_pano are final


def centre_depths(depth, box_half=BOX_HALF_PX):
    """``(centre, box_median)``: the value at the view centre (what data_generation records) and the median of
    the ``2 box_half + 1`` square around it (the jitter measure).

    >>> d = np.arange(64 * 64, dtype=float).reshape(64, 64)
    >>> centre_depths(d, 1)
    (2080.0, 2080.0)
    """
    c = depth.shape[0] // 2
    box = depth[c - box_half:c + box_half + 1, c - box_half:c + box_half + 1]
    return float(depth[c, c]), float(np.median(box))


def _done_uids(out_csv):
    if not os.path.exists(out_csv):
        return set()
    prev = pd.read_csv(out_csv, dtype={"label_uid": str, "status": str})
    final = ~prev.status.astype(str).str.startswith(DEPTH_PASS_RETRY)
    return set(prev.label_uid[final])


def depth_pass(sample, out_csv, encoders=("vitb", "vitl"), device="cuda", pano_reader=dg.read_store_panorama,
               model_loader=dg.load_depth_model, render=dg.render_label_view, pano_store=None, gsv_beta=GSV_BETA):
    """Depth at every sampled label, for each encoder, from the view round 3 rendered. Append-only and resumable
    (labels already written with a final status are skipped; ``error:*`` rows are retried and the later row wins).

    Per panorama: ``pano_reader(store, city, pano_id)``, i.e. ``read_store_panorama`` and its 8192 x 4096 resize,
    with ``store`` the row's own store root (:func:`store_root`) unless ``pano_store`` overrides it (a fresh
    staging). Per label: the pose of ``crop_pose.pose_for_row`` passed to ``render`` exactly as data_generation's
    loop passes it, with ``device`` (``"cuda"``: the torch renderer of round 3; None: numpy, ``--allow-cpu`` smoke
    tests, marked ``renderer=numpy`` so ``compare`` keeps them out of R0). Each encoder's model (loaded once) runs
    on the BGR view; written: ``label_uid, status, renderer, tilt_applied, d_<enc>_m, d_<enc>_box_m``.
    """
    import csv
    done = _done_uids(out_csv)
    todo = sample[~sample.label_uid.isin(done)].sort_values(["city", "pano_id"], kind="stable")
    if not len(todo):
        print("depth-pass: nothing to do")
        return
    models = {enc: model_loader(enc, device) for enc in encoders}
    fields = ["label_uid", "status", "renderer", "tilt_applied"] + \
        [f"d_{e}_{k}" for e in encoders for k in ("m", "box_m")]
    new = not os.path.exists(out_csv)
    renderer = "torch" if device is not None else "numpy"
    last = (None, None)
    with open(out_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if new:
            wr.writeheader()
        for r in todo.to_dict("records"):
            rec = {"label_uid": r["label_uid"], "renderer": renderer}
            try:
                key = (r["city"], r["pano_id"])
                if last[0] != key:
                    store = pano_store or store_root(r["pano_path"], r["city"], r["pano_id"])
                    last = (key, pano_reader(store, r["city"], r["pano_id"])[0])
                pano = last[1]
                if isinstance(pano, str):
                    rec["status"] = pano
                else:
                    pose = crop_pose.pose_for_row(r, gsv_beta)
                    y_norm = r["pano_y"] / r["pano_height"] + (pose.click_lat_offset_deg or 0.0) / 180.0
                    view_rgb, tilt = render(np.ascontiguousarray(pano[..., ::-1]), r["pano_x"] / r["pano_width"],
                                            y_norm, pose.camera_pitch, pose.camera_roll,
                                            apply_tilt=pose.beta is not None, beta=pose.beta or 0.0, device=device)
                    view_bgr = np.ascontiguousarray(view_rgb[..., ::-1])
                    rec["tilt_applied"] = int(tilt)
                    for enc, model in models.items():
                        c, b = centre_depths(model.infer_image(view_bgr))
                        rec[f"d_{enc}_m"], rec[f"d_{enc}_box_m"] = round(c, 4), round(b, 4)
                    rec["status"] = "ok"
            except Exception as e:
                if dg.is_fatal(e):
                    raise
                rec["status"] = f"error:{type(e).__name__}"
            wr.writerow(rec)
            f.flush()


def cmd_depth_pass(args):
    sample = pd.read_csv(args.sample, dtype=SPLIT_DTYPES, low_memory=False)
    device = dg.resolve_device(args.allow_cpu)
    depth_pass(sample, args.out, tuple(args.encoders.split(",")), device, pano_store=args.pano_store)


# ---------------------------------------------------------------------------------------------------------------------
# compare (CPU, after the klone run)
# ---------------------------------------------------------------------------------------------------------------------

def load_depths(path):
    """``depths.csv`` with a retried label's later row kept (the pass is append-only)."""
    return load_depths_frame(pd.read_csv(path, dtype={"label_uid": str}))


def load_depths_frame(d):
    """One row per label from an append-only depth-pass table: the last row written wins."""
    return d.drop_duplicates("label_uid", keep="last")


def merge_sources(sample, depths, google, trig):
    """One row per sampled label: the sample, the encoder depths, Google's depth and the trig columns, joined
    one-to-one on ``label_uid``. Raises on a duplicated uid in any input. A label absent from ``google`` is kept
    with ``g_status = no_artifact``; one absent from ``depths`` has ``status`` NaN."""
    for name, df in (("sample", sample), ("depths", depths), ("google", google), ("trig", trig)):
        if not df.label_uid.is_unique:
            raise ValueError(f"{name} has {int(df.label_uid.duplicated().sum())} duplicated label_uid")
    tcols = ["label_uid", "trig_m", "ratio", "why_excluded"]
    out = sample.merge(depths, on="label_uid", how="left", validate="one_to_one")
    out = out.merge(google, on="label_uid", how="left", validate="one_to_one", suffixes=("", "_g"))
    out = out.merge(trig[tcols].rename(columns={"ratio": "vitb_trig_ratio"}), on="label_uid", how="left",
                    validate="one_to_one")
    out["g_status"] = out.g_status.fillna("no_artifact")
    return out


def pair_stats(a, b, pano, city, weights=None, n_boot=N_BOOT, seed=SEED):
    """Agreement of two depth sources at the same labels, ``L = ln(a / b)``: median ratio ``s`` with 95 %
    intervals from panoramas within city and from whole cities, spread (IQR, 5-95 %, 1.4826 MAD), the slope
    bracket, crop-moved shares raw and after the scale match (``L - ln s``), city-balanced and weighted, and
    Spearman's rho. Pairs where either value is not finite and positive are dropped (counted in ``n_dropped``)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    pano, city = np.asarray(pano)[keep], np.asarray(city)[keep]
    w = None if weights is None else np.asarray(weights, float)[keep]
    a, b = a[keep], b[keep]
    L = np.log(a / b)
    out = {"n": int(keep.sum()), "n_dropped": int((~keep).sum())}
    if not len(L):
        return out
    cl = pd.Series(city).astype(str) + ":" + pd.Series(pano).astype(str)
    med, lo, hi = median_with_ci(L, cl, city, n_boot, seed)
    _, clo, chi = median_with_ci(L, city, None, n_boot, seed + 1)
    s = math.exp(med)
    q = np.quantile(L, [0.05, 0.25, 0.75, 0.95])
    out.update({"s": s, "s_lo": math.exp(lo) if np.isfinite(lo) else np.nan,
                "s_hi": math.exp(hi) if np.isfinite(hi) else np.nan,
                "s_city_lo": math.exp(clo) if np.isfinite(clo) else np.nan,
                "s_city_hi": math.exp(chi) if np.isfinite(chi) else np.nan,
                "matched_constant": dg.CROP_DEPTH_CONSTANT * s,
                "L_iqr": float(q[2] - q[1]), "L_p05": float(q[0]), "L_p95": float(q[3]),
                "L_mad": float(1.4826 * np.median(np.abs(L - np.median(L)))),
                "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")) if len(a) > 2 else np.nan})
    out["bracket"] = bracket_with_ci(np.log(a), np.log(b), cl, city, n_boot, seed)
    out["moved_raw"] = crop_moved_shares(L)
    out["moved_matched"] = crop_moved_shares(L - med)
    if w is not None:
        out["moved_matched_weighted"] = crop_moved_shares(L - med, weights=w)
    return out


def coverage_selection(df):
    """Selection on Google coverage (Set A): ViT-B ``depth_m``, ``trig_m`` and their ratio (median and IQR) by type
    for labels whose artifact is usable at the file level against those whose is not, and for ground hits against
    every other status."""
    a = df[df.set_a].copy()
    a["covered"] = np.where(~a.g_status.isin(FILE_LEVEL_UNUSABLE), "usable artifact", "no usable artifact")
    a["hit"] = np.where(a.g_status.eq("ground"), "ground hit", "not a ground hit")
    rows = []
    for col in ("covered", "hit"):
        for (t, k), g in a.groupby(["label_type", col]):
            rec = {"label_type": t, "group": k, "n": len(g)}
            for v in ("depth_m", "trig_m", "vitb_trig_ratio"):
                x = pd.to_numeric(g[v], errors="coerce").dropna()
                rec[f"{v}_med"] = float(x.median()) if len(x) else np.nan
                rec[f"{v}_iqr"] = float(x.quantile(0.75) - x.quantile(0.25)) if len(x) else np.nan
            rows.append(rec)
    return _type_order(pd.DataFrame(rows))


def rule_readout(stats):
    """Read rules R0 to R2 (``docs/depth-ablation-2026-11.md``, "Decision rules") from the numbers ``compare``
    computes. ``stats``::

        {"r0": {"n": int, "within": int},                               # GPU-rendered Set A rows
         "r1": {"jitter_share_10": float,                               # ViT-B centre vs box, share moved > 10 %
                "t=0.1": {"overall": float, "weighted": float, "by_type": {type: float}},
                "t=0.25": {...}},                                       # scale-matched shares WITHIN the tolerance
         "r2": {"beta_yx": float, "beta_xy": float}}

    R1 at a tolerance: met when the overall share >= 0.95, every type >= 0.90 (point estimates), and the
    row-mix-weighted share is not on the other side of 0.95. Types in [0.86, 0.94) are listed as marginal. When
    the jitter share exceeds 5 %, R1 is read at 25 % instead of 10 %.

    >>> r = rule_readout({"r0": {"n": 100, "within": 99},
    ...     "r1": {"jitter_share_10": 0.01, "t=0.1": {"overall": 0.95, "weighted": 0.96,
    ...            "by_type": {"CurbRamp": 0.90}}}, "r2": {"beta_yx": 0.95, "beta_xy": 1.05}})
    >>> r["R0"]["met"], r["R1"]["met"], r["R1"]["tolerance"], r["R2"]["verdict"]
    (True, True, 0.1, 'inside')
    """
    out = {}
    r0 = stats.get("r0")
    if r0:
        share = r0["within"] / r0["n"] if r0["n"] else np.nan
        out["R0"] = {"share": share, "n": r0["n"], "met": bool(share >= 0.99)}
    r1 = stats.get("r1")
    if r1:
        tol = 0.25 if r1.get("jitter_share_10", 0) > 0.05 else 0.10
        cell = r1[f"t={tol:g}"]
        types = cell["by_type"]
        failing = sorted(t for t, v in types.items() if not v >= 0.90)
        marginal = sorted(t for t, v in types.items() if 0.86 <= v < 0.94)
        weighted_ok = cell.get("weighted") is None or (cell["weighted"] >= 0.95) == (cell["overall"] >= 0.95)
        met = bool(cell["overall"] >= 0.95 and not failing and weighted_ok)
        out["R1"] = {"tolerance": tol, "jitter_fallback": tol == 0.25, "overall": cell["overall"],
                     "weighted": cell.get("weighted"), "types_below_90": failing, "marginal_types": marginal,
                     "weighted_disagrees": not weighted_ok, "met": met,
                     "reading": "ViT-B stays; encoder question closed" if met else
                     "encoder becomes a sweep arm; default changes only under R3"}
    r2 = stats.get("r2")
    if r2:
        out["R2"] = {**r2, "verdict": bracket_verdict(r2["beta_yx"], r2["beta_xy"])}
    return out


def _within_share(L):
    x = np.asarray(L, float)
    x = x[np.isfinite(x)]
    return {t: float((np.abs(x) <= math.log1p(t)).mean()) if len(x) else np.nan for t in CROP_DIFF_THRESHOLDS}


def compare(merged, n_boot=N_BOOT, seed=SEED):
    """Every statistic of the klone arms from :func:`merge_sources`' output: ``(tables, readout, stats)``."""
    m = merged.copy()
    for c in ("depth_m", "d_vitb_m", "d_vitb_box_m", "d_vitl_m", "d_vitl_box_m", "g_ray_m", "trig_m",
              "trig_cam_m", "cam_height_m", "camera_pitch", "art_pitch_deg", "art_roll_deg"):
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    a = m[m.set_a & m.status.eq("ok")]
    gmask = m["set_g"] if "set_g" in m else pd.Series(False, index=m.index)
    g_ground = m[gmask & m.g_status.eq("ground") & m.label_type.isin(GROUND_TYPES)]
    tables, stats = [], {}

    gpu = a[a.renderer.eq("torch")]
    within = int((np.abs(gpu.d_vitb_m / gpu.depth_m - 1) < 0.001).sum())
    stats["r0"] = {"n": int(len(gpu)), "within": within}

    a_trig = a[a.why_excluded.eq("")]
    pairs = {"vitl_vitb": (a, "d_vitl_m", "d_vitb_m"),
             "vitb_google": (g_ground, "depth_m", "g_ray_m"),
             "vitl_google": (g_ground, "d_vitl_m", "g_ray_m"),
             "vitb_trig": (a_trig, "depth_m", "trig_m"),
             "google_trig": (g_ground, "g_ray_m", "trig_m")}
    rows = []
    for name, (df, ca, cb) in pairs.items():
        if not len(df) or ca not in df or cb not in df:
            continue
        ps = pair_stats(df[ca], df[cb], df.pano_id, df.city, df.get("inclusion_weight"), n_boot, seed)
        stats[name] = ps
        if "s" not in ps:
            continue
        br = ps["bracket"]
        rows.append({"pair": name, "n": ps["n"], "s": ps["s"], "s_lo": ps["s_lo"], "s_hi": ps["s_hi"],
                     "s_city_lo": ps["s_city_lo"], "s_city_hi": ps["s_city_hi"], "L_iqr": ps["L_iqr"],
                     "beta_yx": br["beta_yx"], "beta_xy": br["beta_xy"], "rma": br["rma"], "verdict": br["verdict"],
                     "moved10_raw": ps["moved_raw"]["t=0.1"]["share"],
                     "moved25_matched": ps["moved_matched"]["t=0.25"]["share"],
                     "moved25_matched_lo": ps["moved_matched"]["t=0.25"]["lo"],
                     "moved25_matched_hi": ps["moved_matched"]["t=0.25"]["hi"], "spearman": ps["spearman"]})
    tables.append(table("pairs", "Agreement of each pair (a / b)", pd.DataFrame(rows)))

    if "vitl_vitb" in stats and "s" in stats["vitl_vitb"]:
        s = stats["vitl_vitb"]["s"]
        L = np.log(a.d_vitb_m / (a.d_vitl_m / s))
        jit = np.log(a.d_vitb_m / a.d_vitb_box_m)
        r1 = {"jitter_share_10": float((np.abs(jit) > math.log1p(0.10)).mean())}
        wts = a.inclusion_weight.to_numpy(float)
        for t in (0.10, 0.25):
            ok = np.isfinite(L)
            within_t = np.abs(L) <= math.log1p(t)
            r1[f"t={t:g}"] = {"overall": float(within_t[ok].mean()),
                              "weighted": float((wts * within_t)[ok].sum() / wts[ok].sum()),
                              "by_type": {k: float(within_t[(a.label_type == k) & ok].mean())
                                          for k in LABEL_TYPES if ((a.label_type == k) & ok).any()}}
        stats["r1"] = r1
        jrows = [{"source": enc, **{f"moved_{k[2:]}": v["share"] for k, v in
                                   crop_moved_shares(np.log(a[f"d_{enc}_m"] / a[f"d_{enc}_box_m"])).items()
                                   if k.startswith("t=")}} for enc in ("vitb", "vitl")]
        tables.append(table("jitter", "Jitter floor: centre pixel against the 33 x 33 box median",
                            pd.DataFrame(jrows)))
        sl = _slice_columns(a.assign(L=L))
        for name, keys, ex in (("vitl_type", ["label_type"], False), ("vitl_type_severity", ["label_type",
                               "severity_s"], False), ("vitl_type_tag", ["label_type", "tag"], True),
                               ("vitl_city", ["city"], False), ("vitl_year", ["year_band"], False)):
            t = summarize(sl, keys, ["L"], explode_tags=ex)
            tables.append(table(name, f"ViT-L vs ViT-B (scale-matched ln ratio) by {', '.join(keys)}", t))
    if "bracket" in stats.get("vitb_google", {}):
        stats["r2"] = {k: stats["vitb_google"]["bracket"][k] for k in ("beta_yx", "beta_xy")}
    if len(g_ground):
        wrows = []
        for t, g in g_ground.groupby("label_type"):
            med, p05, p95 = _quantiles(physical_width_m(g.depth_m, g.g_ray_m))
            wrows.append({"label_type": t, "n": len(g), "width_m_med": med, "width_m_p05": p05, "width_m_p95": p95})
        tables.append(table("width_google", "Physical crop width at Google's ground distance", pd.DataFrame(wrows)))
        h = m.cam_height_m.dropna()
        tables.append(table("cam_height", "Per-panorama camera height from Google's ground plane",
                            pd.DataFrame([{"n": len(h), **dict(zip(("med", "p05", "p95"), _quantiles(h)))}])))
    tables.append(table("coverage", "Google artifact status by type (all candidates)",
                        m.groupby(["label_type", "g_status"]).size().rename("n").reset_index()))
    tables.append(table("coverage_selection", "Selection on coverage (Set A)", coverage_selection(m)))
    if "art_roll_deg" in m:
        pr = m.dropna(subset=["art_roll_deg"])
        tables.append(table("artifact_pose", "Artifact roll and artifact pitch minus DB camera_pitch (degrees)",
                            pd.DataFrame([{"quantity": "roll", "n": len(pr),
                                           **dict(zip(("med", "p05", "p95"), _quantiles(pr.art_roll_deg)))},
                                          {"quantity": "pitch - DB pitch", "n": len(pr),
                                           **dict(zip(("med", "p05", "p95"),
                                                      _quantiles(pr.art_pitch_deg - pr.camera_pitch)))}])))
    return tables, rule_readout(stats), stats


def cmd_compare(args):
    sample = pd.read_csv(args.sample, dtype=SPLIT_DTYPES, low_memory=False)
    google = pd.read_csv(args.google, dtype={"label_uid": str})
    sample["set_g"] = set_g(sample, google).to_numpy()
    merged = merge_sources(sample, load_depths(args.depths), google,
                           pd.read_csv(args.trig, dtype={"label_uid": str}))
    tables, readout, stats = compare(merged, args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    merged.to_csv(os.path.join(args.out_dir, "agreement.csv"), index=False)     # per label: stays in ~/Data
    write_json(os.path.join(args.out_dir, "compare_tables.json"), {"tables": tables, "readout": readout})
    write_json(os.path.join(args.out_dir, "compare_stats.json"), stats)
    commit, dirty = git_commit()
    write_json(os.path.join(args.out_dir, "compare.json"), {
        "stage": "compare", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
        "inputs": {p: sha256_file(p) for p in (args.sample, args.depths, args.google, args.trig)},
        "seed": args.seed, "n_boot": args.n_boot})
    print(json.dumps(readout, indent=1, default=_json_default))


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:
            return f"{v:.2g}"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def table_markdown(t, small_note="n too small"):
    """One stored table as Markdown; a row flagged ``small`` shows its counts and "n too small" for its numbers."""
    cols = [c for c in t["columns"] if c != "small"]
    count_cols = {"n", "train_n", "label_type", "city", "severity_s", "tag", "label_class", "year_band", "split",
                  "trig_decile"}
    lines = [f"**{t['title']}**", ""]
    if t.get("note"):
        lines += [t["note"], ""]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in t["rows"]:
        if r.get("small"):
            cells = [_fmt(r.get(c)) if c in count_cols else "" for c in cols]
            first_stat = next((i for i, c in enumerate(cols) if c not in count_cols), None)
            if first_stat is not None:
                cells[first_stat] = small_note
        else:
            cells = [_fmt(r.get(c)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report_markdown(trig_out=None, clipping=None, compare_out=None):
    """``tables.md``: every stored table, in order, as Markdown. Deterministic for the same inputs."""
    parts = ["# Depth agreement: tables (generated by `analysis/depth_agreement.py report`)", "",
             "Aggregates only. Every number is pasted into `docs/depth-ablation-2026-11.md` from here.", ""]
    if trig_out:
        h = trig_out["headline"]
        parts += ["## Flat-ground reference (trig)", "",
                  f"Camera height {h['camera_height_m']} m nominal; {h['n']} usable rows; median ViT-B / flat "
                  f"ground {h['ratio_median']:.3f}; scale-matched constant {h['scale_matched_constant']:.0f}.", ""]
        parts += [table_markdown(t) for t in trig_out["tables"]]
    if clipping is not None:
        parts += ["## Clipping at the view edge", "",
                  table_markdown(table("clipping", "Clipped crops by type (pooled over deployments)", clipping,
                                       digits=5))]
    if compare_out:
        parts += ["## Klone arms", "", "```json", json.dumps(compare_out["readout"], indent=1, sort_keys=True,
                                                              default=_json_default), "```", ""]
        parts += [table_markdown(t) for t in compare_out["tables"]]
    return "\n".join(parts)


def report_figures(trig_out, out_dir):
    """The binned figures: a 2-D histogram of ln ViT-B against ln flat-ground distance (bins under
    HIST_MIN_COUNT blanked) and the descriptive decile curve. Returns the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    hst = trig_out["figures"]["hist_ln_trig_ln_vitb"]
    c = np.array(hst["counts"], float).T
    c[c == 0] = np.nan
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
    mesh = ax.pcolormesh(np.exp(hst["x_edges"]), np.exp(hst["y_edges"]), c, cmap="viridis")
    lim = [math.exp(min(hst["x_edges"][0], hst["y_edges"][0])), math.exp(max(hst["x_edges"][-1],
                                                                              hst["y_edges"][-1]))]
    ax.plot(lim, lim, color="0.4", lw=1, ls="--", label="equal")
    ax.plot(lim, [v * trig_out["headline"]["ratio_median"] for v in lim], color="C3", lw=1,
            label=f"{trig_out['headline']['ratio_median']:.2f} x")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*np.exp([hst["x_edges"][0], hst["x_edges"][-1]]))
    ax.set_ylim(*np.exp([hst["y_edges"][0], hst["y_edges"][-1]]))
    ax.set_xlabel(f"flat-ground distance at {trig_out['headline']['camera_height_m']} m camera (m)")
    ax.set_ylabel("ViT-B depth_m (m)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(mesh, ax=ax, label=f"labels per bin (bins < {hst['min_count']} blank)")
    fig.tight_layout()
    p = os.path.join(out_dir, "trig_hist.png")
    fig.savefig(p)
    plt.close(fig)
    written.append(p)
    return written


def cmd_report(args):
    trig_out = clipping = compare_out = None
    inputs = {}
    if args.trig_dir:
        p = os.path.join(args.trig_dir, "trig_tables.json")
        with open(p) as f:
            trig_out = json.load(f)
        inputs["trig_tables.json"] = sha256_file(p)
        inputs["trig.json"] = sha256_file(os.path.join(args.trig_dir, "trig.json"))
    if args.clipping:
        clipping = pd.read_csv(args.clipping)
        inputs["clipping.csv"] = sha256_file(args.clipping)
    if args.compare_dir:
        p = os.path.join(args.compare_dir, "compare_tables.json")
        with open(p) as f:
            compare_out = json.load(f)
        inputs["compare_tables.json"] = sha256_file(p)
    os.makedirs(args.assets, exist_ok=True)
    md = report_markdown(trig_out, clipping, compare_out)
    with open(os.path.join(args.assets, "tables.md"), "w") as f:
        f.write(md)
    figs = report_figures(trig_out, args.assets) if trig_out and not args.no_figures else []
    commit, dirty = git_commit()
    write_json(os.path.join(args.assets, "provenance.json"), {
        "stage": "report", "git_commit": commit, "git_dirty": dirty, "inputs": inputs,
        "outputs": {os.path.basename(p): sha256_file(p) for p in [os.path.join(args.assets, "tables.md")] + figs}})
    print(f"-> {args.assets}/tables.md" + (f", {', '.join(os.path.basename(p) for p in figs)}" if figs else ""))


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def listing(sp):
        sp.add_argument("--public-cities", help="JSON list of public city_ids (tests, re-runs)")
        sp.add_argument("--cities-json", help="a saved /v3/api/cities listing (the day's snapshot); default: live")

    t = sub.add_parser("trig", help="ViT-B against the flat-ground reference, every round-3 row (CPU)")
    t.add_argument("--root", required=True, help="round-3 dataset root (<Type>/split.csv)")
    t.add_argument("--rawlabels-dir", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--camera-height-m", type=float, nargs="+", default=list(CAMERA_HEIGHTS_M),
                   help="first is the headline height (default: 2.5 2.3 2.9)")
    t.add_argument("--check-pins", action="store_true", help="stop unless the inputs match ROUND3_PINS")
    t.add_argument("--n-boot", type=int, default=N_BOOT)
    t.add_argument("--seed", type=int, default=SEED)
    listing(t)
    t.set_defaults(func=cmd_trig)

    c = sub.add_parser("clipping", help="clipped crops by type at 4000 / 6100 / 8000 (CPU)")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_clipping)

    s = sub.add_parser("sample", help="the seeded, city-balanced public-city sample (CPU)")
    s.add_argument("--root", required=True)
    s.add_argument("--rawlabels-dir", required=True)
    s.add_argument("--out-dir", required=True, help="sample.csv and stage_files.txt (keep in ~/Data)")
    s.add_argument("--public-out", required=True, help="sample_public.csv and sample.json (committable)")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--n-per-type", type=int, default=N_PER_TYPE)
    s.add_argument("--oversample", type=int, default=OVERSAMPLE)
    s.add_argument("--stage-panos", action="store_true", help="also list the panorama JPEGs for staging")
    listing(s)
    s.set_defaults(func=cmd_sample)

    d = sub.add_parser("depth-pass", help="ViT-B / ViT-L depth at every sampled label (GPU)")
    d.add_argument("--sample", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--encoders", default="vitb,vitl")
    d.add_argument("--pano-store", help="read every panorama from this store instead of each row's own")
    d.add_argument("--allow-cpu", action="store_true", help=dg.DEPTH_DEVICE_HELP)
    d.set_defaults(func=cmd_depth_pass)

    g = sub.add_parser("google-depth", help="Google's depth along each crop's centre ray (CPU)")
    g.add_argument("--sample", required=True)
    g.add_argument("--store", required=True, help="root holding <city>/<id[:2]>/<id>.depth.npz")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_google_depth)

    m = sub.add_parser("compare", help="agreement statistics and the R0-R2 read-out (CPU)")
    m.add_argument("--sample", required=True)
    m.add_argument("--depths", required=True)
    m.add_argument("--google", required=True)
    m.add_argument("--trig", required=True, help="trig.csv from the trig subcommand")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--n-boot", type=int, default=N_BOOT)
    m.add_argument("--seed", type=int, default=SEED)
    m.set_defaults(func=cmd_compare)

    r = sub.add_parser("report", help="tables.md and binned figures from the stored aggregates (CPU)")
    r.add_argument("--trig-dir")
    r.add_argument("--clipping")
    r.add_argument("--compare-dir")
    r.add_argument("--assets", required=True)
    r.add_argument("--no-figures", action="store_true")
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
