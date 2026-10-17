"""Build a correct/incorrect crop dataset for one Project Sidewalk label type.

For every label of ``--label_type`` across the cities listed in ``CITIES``, this script:

1. pulls the label and its validations from the public v3 API (``/v3/api/rawLabels``);
2. keeps human-authored labels the crowd clearly endorsed (net >= 3 human agree votes) or clearly rejected
   (net >= 2 human disagree votes), downsampling the ``correct`` class to ``CORRECT_PROPORTION``, and records the
   PRE-downsampling class counts in ``deployment_priors.json`` (the prior ``calibration/calibrate.py`` reweights to);
3. reassembles the Google Street View panorama from tiles, renders a 2048x2048, 90-degree perspective view
   CENTRED on the label with rig-tilt correction (``sidewalk_panorama_tools.geometry``), estimates metric depth at
   the view centre with Depth Anything V2, and cuts a crop whose size scales with 1/depth so near and far features
   fill the frame similarly;
4. writes ``<out>/<label_type>/<correct|incorrect>/<city>_<label_id>.webp`` and appends one row per label to
   ``<out>/<label_type>/manifest.csv`` (the input of ``split_data.py``, which also joins the ``hard`` flag).

The run is resumable: a label whose manifest row says ``ok``, whose crop exists, and which is still selected with
the same class is skipped on restart. ``generation_run.json`` records a hash of the selection, so resuming against a
changed selection (new votes, other cities) stops with counts unless ``--force-reselect`` is given.

The view is rendered on the GPU with the package's torch recipe (``sidewalk_panorama_tools.torch_geometry``) and depth
runs on the same device. CUDA is required unless ``--allow-cpu`` (smoke tests only), which renders with the numpy
``render_view`` instead. ``deployment_priors.json`` is written only by a full run (no ``--cities``, no ``--limit``);
other runs write ``deployment_priors.partial.json``.

**Geometry.** The view is the shared one from ``sidewalk-panorama-tools`` (``label_view`` + ``render_view``), the
implementation sidewalk-ai-api is meant to pin too, so training and serving cut the same crop
(https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5). The label sits at the view centre, so the old
out-of-frame failure mode (issue #9, item 1) cannot occur. **Frame rule:** human labels are clicked in the
gravity-levelled viewer frame while the stored equirect is in the rig frame, so they are rendered with the tilt
correction (the label's ``camera_pitch``/``camera_roll``, a fraction ``--tilt-beta`` applied;
https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54). Labels authored by the AI account (RampNet,
Vancouver WA / Richmond VA / Laurens IA) were placed on the rig-frame image directly, so they are rendered without
it. They are excluded from the training selection by default anyway: production never sends them to the validator
(issue #9, item 2).

The serving-side twin of ``create_perspective_crop`` lives in
https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/main.py ; it applies the same tilt correction with the same
package in ProjectSidewalk/sidewalk-ai-api#17. **Deployment order:** models retrained on these crops expect
tilt-corrected input, so that API change must be deployed before a retrained model is published.

Lineage and background:
  * Weld et al., "Deep Learning for Automatically Detecting Sidewalk Accessibility Problems Using
    Streetscape Imagery", ASSETS 2019, https://doi.org/10.1145/3308561.3353798 (first auto-validation
    of Project Sidewalk labels; depth-sized crops around the label point).
  * Duan et al., "Scaling Crowd+AI Sidewalk Accessibility Assessments", ASSETS 2022,
    https://doi.org/10.1145/3517428.3550381 (validated vs unvalidated training data; cross-city transfer).
  * Depth Anything V2 (Yang et al., NeurIPS 2024), https://github.com/DepthAnything/Depth-Anything-V2 .
  * Project Sidewalk platform: Saha et al., CHI 2019, https://doi.org/10.1145/3290605.3300292 .

Known limitations are tracked in the repo's issues; the audit that found them is
docs/audit-2026-10-14.md.

Usage::

    python data_generation.py --label_type CurbRamp                       # full run, writes ./CurbRamp/
    python data_generation.py --label_type CurbRamp --limit 20 --out /tmp/smoke --allow-cpu
    python data_generation.py --label_type CurbRamp --select-only --rawlabels-cache ~/rawlabels  # priors, no GPU
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:  # advisory lock for the shared deployment_priors.json (five label types run in parallel under SLURM)
    import fcntl
except ImportError:  # pragma: no cover - Windows; the SLURM jobs run on Linux
    fcntl = None

# One seed for every shuffle in this file, so a dataset can be regenerated exactly.
SEED = 42

# The SidewalkAI account (role "AI"). Its labels (RampNet-placed curb ramps) are in the rig frame, and production
# never sends them to the validator. Its votes carry validator_type "AI" and are excluded from every count.
AI_USER_ID = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"

# Google's anonymous tile endpoint returns 403 PERMISSION_DENIED without browser-like headers
# (measured 2026-10-14; the same headers are used by sidewalk-ai-api/sidewalk_ai_api/panorama.py).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# City deployments to pull labels from: city_id (as /v3/api/cities names it, and as it appears in label_uid and in
# crop filenames) -> the deployment's subdomain, sidewalk-<subdomain>.cs.washington.edu. Not the full list of 59
# deployments; discovering them from /v3/api/cities is https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8.
# keelung-tw, new-taipei-tw and taipei are reached by hostname because /v3/api/cities lists them without a ``url``.
CITIES = {
    "cdmx": "cdmx", "la-piedad": "la-piedad", "spgg": "spgg", "amsterdam": "amsterdam",
    "blackhawk-hills-il": "blackhawk-hills", "chicago-il": "chicago", "cliffside-park-nj": "cliffside-park",
    "columbus-oh": "columbus", "knox-oh": "knox", "mendota-il": "mendota", "newberg-or": "newberg",
    "oradell-nj": "oradell", "pittsburgh-pa": "pittsburgh", "seattle-wa": "sea", "st-louis-mo": "st-louis",
    "teaneck-nj": "teaneck", "keelung-tw": "keelung", "new-taipei-tw": "new-taipei", "taipei": "taipei",
}

CORRECT_PROPORTION = 0.55

# Crop geometry. The view matches sidewalk-ai-api for an 8192-wide pano (width // 4).
VIEW_FOV_DEG = 90.0
VIEW_SIZE = 2048
CROP_DEPTH_CONSTANT = 6100  # crop_half_px = int(6100 / depth_m); inherited from the first version, see issue #21
CROP_LONG_SIDE = 640
WEBP_QUALITY = 95
DEFAULT_TILT_BETA = 0.95  # fraction of the rig tilt in the stored click, for the DB camera_pitch pose record

# Depth Anything V2 metric checkpoints (the vendored repo's configs). The API serves vkitti / vitb.
DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
DEPTH_DATASET = 'vkitti'
DEPTH_MAX_M = 80

MANIFEST_COLUMNS = [
    "label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "author", "human_agree",
    "human_disagree", "severity", "tags", "high_quality_user", "image_capture_date", "pano_source", "pano_zoom", "camera_pitch",
    "camera_roll", "tilt_applied", "tilt_beta", "depth_m", "crop_half_px", "clipped", "crop_path", "status",
]
# (``hard`` is not a manifest column: split_data.py --hard-examples joins it, so a new list needs no regeneration.)

TILE_TIMEOUT = (5, 30)        # (connect, read) seconds per tile request
CSV_TIMEOUT = (10, 300)       # rawLabels exports run to ~175 MB
DEPTH_DEVICE_HELP = "CUDA is required; --allow-cpu renders with numpy and runs depth on the CPU (smoke tests only)"



# ---------------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------------

def load_city(city_id, subdomain=None, cache_dir=None):
    """rawLabels of one deployment as a DataFrame with a ``city`` column (the city_id).

    With ``cache_dir``, ``<cache_dir>/<city_id>.csv`` is read instead of the API when it exists (a saved
    ``/v3/api/rawLabels?filetype=csv`` export).
    """
    cached = os.path.join(cache_dir, f"{city_id}.csv") if cache_dir else None
    if cached and os.path.exists(cached):
        df = pd.read_csv(cached, low_memory=False)
    else:
        import requests
        subdomain = subdomain or CITIES.get(city_id, city_id)
        url = f'https://sidewalk-{subdomain}.cs.washington.edu/v3/api/rawLabels?filetype=csv'
        for attempt in (1, 2):  # one retry
            try:
                response = requests.get(url, timeout=CSV_TIMEOUT)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                print(f"rawLabels for {city_id} failed ({e}); retrying once")
                time.sleep(10)
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    df['city'] = city_id
    return df


def human_vote_counts(validations_json):
    """Count (agree, disagree) votes from human validators only.

    The API's ``agree_count``/``disagree_count`` columns include the SidewalkAI account's own votes,
    so using them would train the validator on its own output. The ``validations`` column carries a
    ``validator_type`` per vote ("Human" or "AI"), which is what we count here; anything not "Human" is ignored.

    >>> human_vote_counts('[{"user_id":"a","validation":"Agree","validator_type":"AI"},'
    ...                   ' {"user_id":"b","validation":"Agree","validator_type":"Human"},'
    ...                   ' {"user_id":"c","validation":"Disagree","validator_type":"Human"}]')
    (1, 1)
    >>> human_vote_counts(float('nan'))
    (0, 0)
    >>> human_vote_counts('[{"validation":"Agree"}, {"validation":"Agree","validator_type":"Bot"}]')
    (0, 0)
    """
    if not isinstance(validations_json, str) or not validations_json:
        return 0, 0
    votes = json.loads(validations_json)
    # Fail closed: only votes explicitly typed "Human" count, so a new or missing validator_type is excluded.
    agree = sum(1 for v in votes if v.get("validator_type") == "Human" and v.get("validation") == "Agree")
    disagree = sum(1 for v in votes if v.get("validator_type") == "Human" and v.get("validation") == "Disagree")
    return agree, disagree


def label_author(user_id):
    """``"ai"`` for a label placed by the SidewalkAI account, else ``"human"``.

    >>> label_author("51b0b927-3c8a-45b2-93de-bd878d1e5cf4"), label_author("someone-else")
    ('ai', 'human')
    """
    return "ai" if str(user_id) == AI_USER_ID else "human"


def valid_geometry(df):
    """Boolean mask of rows whose stored pano point is usable, and the reason counts for the rest.

    Drops (counted by reason): non-GSV providers (the tile fetcher cannot fetch them; issue #9), non-finite or
    non-positive ``pano_width``/``pano_height``, non-finite ``pano_x``, and ``pano_y`` outside ``[0, pano_height]``
    (this covers the legacy labels whose pano coordinates were never converted and are negative: Seattle 193,
    CDMX 14, Pittsburgh 8, SPGG 1).

    >>> m, why = valid_geometry(pd.DataFrame({"pano_source": ["gsv", "gsv", "mapillary", "gsv"],
    ...     "pano_x": [1, 1, 1, float("nan")], "pano_y": [5, -3, 5, 5], "pano_width": [10, 10, 10, 10],
    ...     "pano_height": [10, 10, 10, 10]}))
    >>> m.tolist(), dict(why)
    ([True, False, False, False], {'non_gsv': 1, 'pano_y_out_of_range': 1, 'pano_x_nonfinite': 1})
    """
    num = {c: pd.to_numeric(df[c], errors='coerce') for c in ('pano_x', 'pano_y', 'pano_width', 'pano_height')}
    checks = [
        ('non_gsv', df['pano_source'].astype(str).str.lower() == 'gsv'),
        ('bad_pano_size', np.isfinite(num['pano_width']) & np.isfinite(num['pano_height'])
         & (num['pano_width'] > 0) & (num['pano_height'] > 0)),
        ('pano_y_out_of_range', np.isfinite(num['pano_y']) & (num['pano_y'] >= 0) & (num['pano_y'] <= num['pano_height'])),
        ('pano_x_nonfinite', np.isfinite(num['pano_x'])),
    ]
    keep = pd.Series(True, index=df.index)
    why = Counter()
    for reason, ok in checks:  # each dropped row is counted once, under its first failing check
        bad = keep & ~ok
        if bad.any():
            why[reason] = int(bad.sum())
        keep &= ok
    return keep, why


def eligible_labels(all_labels, label_type):
    """Labels of one type with a usable pano point that meet the vote thresholds: ``(eligible_df, dropped)``.

    Adds ``human_agree``, ``human_disagree``, ``author`` (``human``/``ai``) and ``label_class``; no downsampling.
    AI-authored labels are KEPT here (tagged ``author = ai``): the prior counts only human-authored ones, and the
    selection drops AI ones unless asked. ``dropped`` counts rows removed by :func:`valid_geometry`.

    Rule (unchanged since the first models): ``incorrect`` if net human disagree votes > 1, ``correct`` if net
    human agree votes > 2; everything else (unvalidated, unsure, contested) is left out.
    """
    df = all_labels[all_labels['label_type'] == label_type]
    keep, dropped = valid_geometry(df)
    df = df[keep].copy()
    df['author'] = df['user_id'].map(label_author) if 'user_id' in df else 'human'
    counts = df['validations'].apply(human_vote_counts)
    df['human_agree'] = [a for a, _ in counts]
    df['human_disagree'] = [d for _, d in counts]
    net = df['human_agree'] - df['human_disagree']
    df['label_class'] = np.where(net > 2, 'correct', np.where(-net > 1, 'incorrect', ''))
    return df[df['label_class'] != ''].copy(), dropped


def select_training_labels(eligible, seed=SEED, correct_proportion=CORRECT_PROPORTION, include_ai_labels=False):
    """Balance the :func:`eligible_labels` frame to ``correct_proportion`` correct, dropping AI-authored labels unless
    ``include_ai_labels``.

    The ``correct`` class is downsampled so it makes up ``correct_proportion`` of the result. Note that this ratio is
    not the deployment prior (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4);
    :func:`deployment_prior_entry` records that.

    **Stable under a changing pool.** The kept ``correct`` labels are the ``amount_correct`` with the lowest
    ``sha256(f"{seed}:{label_uid}")``, not a shuffle of the whole pool. Each label's rank is its own, so when one label
    enters or leaves the pool (new votes) the selected ``correct`` set changes by at most one label in and one out,
    and a resumed run keeps nearly all of its crops. The final order is still a seeded shuffle.
    """
    df = eligible if include_ai_labels else eligible[eligible['author'] == 'human']
    # A canonical order first, so the seeded samples below do not depend on the API's row order.
    df = df.sort_values(['city', 'label_id'], kind='mergesort').reset_index(drop=True)
    filtered_disagree = df[df['label_class'] == 'incorrect']
    print(f"Incorrect Examples: {len(filtered_disagree)}")

    filtered_agree = df[df['label_class'] == 'correct'].copy()
    amount_correct = int((correct_proportion / (1 - correct_proportion)) * len(filtered_disagree))
    filtered_agree['_rank'] = [label_rank(seed, f"{c}:{i}")
                               for c, i in zip(filtered_agree['city'], filtered_agree['label_id'])]
    filtered_agree = (filtered_agree.sort_values('_rank', kind='mergesort').head(amount_correct)
                      .drop(columns='_rank').reset_index(drop=True))
    print(f"Correct Examples: {len(filtered_agree)}")

    combined = pd.concat([filtered_agree, filtered_disagree])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the combined dataframe


def label_rank(seed, label_uid):
    """A label's seeded, pool-independent rank for the downsample: ``sha256(f"{seed}:{label_uid}")`` as hex."""
    return hashlib.sha256(f"{seed}:{label_uid}".encode()).hexdigest()


def deployment_prior_entry(eligible, cities, n_rows, measured_on=None):
    """The pre-downsampling class counts of human-authored :func:`eligible_labels`: the class mix the validator meets
    in production (which never sends it AI-authored labels), for ``calibration/calibrate.py``. ``n_rows`` is the
    number of rawLabels rows of this label type the counts were drawn from."""
    counts = eligible.loc[eligible['author'] == 'human', 'label_class'].value_counts()
    return {"correct": int(counts.get('correct', 0)), "incorrect": int(counts.get('incorrect', 0)),
            "measured_on": measured_on or date.today().isoformat(),
            "source": f"rawLabels of {', '.join(cities)}, human votes only",
            "cities": list(cities), "n_rows": int(n_rows)}


def selection_hash(selection):
    """sha256 over the sorted ``label_uid,label_class`` lines of a selection: changes when any label enters, leaves,
    or changes class.

    >>> sel = pd.DataFrame({"label_uid": ["b:2", "a:1"], "label_class": ["correct", "incorrect"]})
    >>> selection_hash(sel) == selection_hash(sel.iloc[::-1])
    True
    """
    lines = sorted(f"{u},{c}" for u, c in zip(selection['label_uid'], selection['label_class']))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def update_deployment_priors(path, label_type, entry):
    """Set ``label_type``'s entry in the JSON at ``path``, keeping the other label types' entries.

    Concurrent writers (five label types under one SLURM job) are serialised with an advisory lock on
    ``<path>.lock``, and the file is replaced atomically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        priors = {}
        if os.path.exists(path):
            with open(path) as f:
                priors = json.load(f)
        priors[label_type] = entry
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(dict(sorted(priors.items())), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    return priors


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------

class ManifestWriter:
    """Append-only, flushed-per-row ``manifest.csv``, resumable.

    On open, an existing manifest is read and de-duplicated (last row per ``label_uid`` wins). A row is kept, and its
    label skipped (:meth:`is_done`), only if its status is ``ok``, its crop file exists, and its ``label_uid`` is in
    ``selection`` (``{label_uid: label_class}``, the CURRENT full selection) with the same class. Every other row is
    dropped: a label still selected is attempted again and gets exactly one fresh row, and a label no longer selected
    leaves the manifest (its crop file stays on disk, unused). ``self.stale`` counts the dropped ``ok`` rows. The kept
    rows are rewritten atomically before any append, so ``label_uid`` stays unique in the file at every moment.

    Example::

        with ManifestWriter("CurbRamp/manifest.csv", "CurbRamp", {"seattle-wa:1": "correct"}) as m:
            if not m.is_done("seattle-wa:1"):
                m.write({...})
    """

    def __init__(self, path, base_dir, selection, keep_ok=True):
        self.path = path
        self.base_dir = base_dir
        self.done = set()
        self.stale = Counter()
        kept = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            old = old.drop_duplicates('label_uid', keep='last')
            for row in old.to_dict('records'):
                if row.get('status') != 'ok' or not os.path.exists(os.path.join(base_dir, row.get('crop_path', ''))):
                    continue
                uid = row['label_uid']
                if not keep_ok:
                    self.stale['settings_changed'] += 1
                elif uid not in selection:
                    self.stale['no_longer_selected'] += 1
                elif selection[uid] != row.get('label_class'):
                    self.stale['class_changed'] += 1
                else:
                    kept.append(row)
                    self.done.add(uid)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction='ignore')
            w.writeheader()
            w.writerows(kept)
        os.replace(tmp, path)
        self._seen = set(self.done)
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=MANIFEST_COLUMNS, extrasaction='ignore')

    def is_done(self, label_uid):
        return label_uid in self.done

    def write(self, row):
        uid = row['label_uid']
        assert uid not in self._seen, f"duplicate label_uid {uid} in {self.path}"
        self._seen.add(uid)
        self._w.writerow({c: _csv_value(row.get(c)) for c in MANIFEST_COLUMNS})
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _csv_value(v):
    """Manifest cell: None/NaN as empty, everything else as-is (the csv module stringifies)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return v


def manifest_row(row, label_type, crop_path, tilt_beta):
    """The label-level manifest fields for one selected label (the crop fields are filled in after rendering)."""
    city = row['city']
    return {
        "label_uid": f"{city}:{row['label_id']}", "city": city, "label_id": row['label_id'],
        "pano_id": row['pano_id'], "label_type": label_type, "label_class": row['label_class'],
        "author": row['author'], "human_agree": row['human_agree'], "human_disagree": row['human_disagree'],
        "severity": row.get('severity'), "tags": row.get('tags'), "high_quality_user": row.get('high_quality_user'),
        "image_capture_date": row.get('image_capture_date'), "pano_source": row.get('pano_source'),
        "pano_zoom": None,
        "camera_pitch": row.get('camera_pitch'), "camera_roll": row.get('camera_roll'),
        "tilt_applied": 0, "tilt_beta": tilt_beta, "depth_m": None, "crop_half_px": None, "clipped": None,
        "crop_path": crop_path, "status": "",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Panorama
# ---------------------------------------------------------------------------------------------------------------------

class IncompletePanorama(Exception):
    """Some tiles of the panorama's grid were still missing after one retry (recorded as ``incomplete_pano``)."""


def _http_get(url):
    """One tile request; a seam for the tests, which replace it with a stub."""
    import requests
    return requests.get(url, headers=HEADERS, timeout=TILE_TIMEOUT)


def fetch_panorama(pano_id, info=None):
    """The panorama as an 8192x4096 BGR array, or None if it cannot be found (black or missing first tiles).

    Raises :class:`IncompletePanorama` if any tile of the discovered ``(max_x + 1) x (max_y + 1)`` grid is still
    missing after one retry (assembling it anyway would leave a black hole in the pano and in a crop), or if the
    assembled image is not 2:1 (``|w / h - 2| >= 0.01``), which is what a transient failure during the size probe
    produces: a too-narrow grid that would be stretched to 8192 wide. ``info``, if given, receives ``zoom``.
    """
    import cv2
    from PIL import Image

    chosen_zoom = None
    http_errors = Counter()  # non-200 status codes for this pano, reported once (tiles are fetched in parallel)
    errors_lock = threading.Lock()

    def _note(status):
        with errors_lock:
            http_errors[status] += 1

    def _fetch_tile(x, y, zoom=4):
        nonlocal chosen_zoom

        if chosen_zoom != None:
            zoom = chosen_zoom
        
        url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={pano_id}"
            f"&x={x}&y={y}&zoom={zoom}"
        )
        try:
            response = _http_get(url)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if chosen_zoom != None or not _is_black_tile(tile):
                    chosen_zoom = zoom
                    return x, y, tile
            else:
                _note(response.status_code)
        except Exception as e:
            _note(type(e).__name__)
            print(e)

        if chosen_zoom == None:
            # Try fallback with zoom=3
            fallback_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/tile"
                f"?cb_client=maps_sv.tactile&panoid={pano_id}"
                f"&x={x}&y={y}&zoom=3"
            )
            try:
                response = _http_get(fallback_url)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    chosen_zoom = 3
                    return x, y, tile
                _note(response.status_code)
            except Exception as e:
                _note(type(e).__name__)
                print(e)

        return x, y, None

    def _is_black_tile(tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _find_panorama_dimensions():
        tiles_cache = {}
        x, y = 5, 2

        is_first = True

        while True:
            tile_info = _fetch_tile(x, y)
            if tile_info is None:
                return None
            tile = tile_info[2]

            if tile is None:
                return None  # Invalid panorama

            if is_first:
                is_first = False
                if _is_black_tile(tile):
                    return None  # Invalid panorama

            tiles_cache[(x, y)] = tile

            if _is_black_tile(tile):
                y = y - 1

                while True:
                    tile_info = _fetch_tile(x, y)
                    if tile_info is None:
                        return None
                    tile = tile_info[2]
                    if tile is None:
                        # A failed request (429, timeout) is not the right edge: retry once before treating it so.
                        print(f"pano {pano_id}: probe tile ({x}, {y}) failed; retrying once")
                        tile = _fetch_tile(x, y)[2]
                    tiles_cache[(x, y)] = tile

                    if _is_black_tile(tile):
                        return x - 1, y, tiles_cache

                    x += 1

            x += 1
            y += 1

    def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()

        # 100 workers, as in sidewalk-ai-api. A zoom-4 pano is 128 tiles (a handful fetched by the probe above), so
        # fewer than 128 threads run whether the pool is 100 or the earlier 1,000: tidying, not a rate-limit fix.
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(_fetch_tile, x, y))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    x, y, tile = result
                    if tile is not None:
                        tiles_cache[(x, y)] = tile

        return tiles_cache

    def _assemble_panorama(tiles, max_x, max_y):
        if not tiles:
            return None
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))

        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))

        return panorama

    def _crop(image):
        img_array = np.array(image)
        y_nonzero, x_nonzero, _ = np.nonzero(img_array)
        if y_nonzero.size > 0 and x_nonzero.size > 0:
            return img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        return img_array # Return original if all black

    def _report():
        if http_errors:
            # e.g. 403 means the request headers stopped working (audit finding 2); 400 is common for retired panos
            print(f"pano {pano_id}: non-200 tile responses {dict(http_errors)}")

    dimension_result = _find_panorama_dimensions()
    if dimension_result is None:
        _report()
        return None

    max_x, max_y, initial_tiles = dimension_result
    full_tiles = {k: t for k, t in _fetch_remaining_tiles(max_x, max_y, initial_tiles).items() if t is not None}
    missing = [(x, y) for x in range(max_x + 1) for y in range(max_y + 1) if (x, y) not in full_tiles]
    for x, y in missing:  # one retry each
        tile = _fetch_tile(x, y)[2]
        if tile is not None:
            full_tiles[(x, y)] = tile
    missing = [xy for xy in missing if xy not in full_tiles]
    _report()
    if missing:
        raise IncompletePanorama(f"pano {pano_id}: {len(missing)} of {(max_x + 1) * (max_y + 1)} tiles missing "
                                 f"after a retry, e.g. {missing[:3]}")
    assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
    if assembled_panorama is None:
        return None
    cropped_panorama = _crop(assembled_panorama)
    h, w = cropped_panorama.shape[:2]
    if abs(w / h - 2) >= 0.01:
        raise IncompletePanorama(f"pano {pano_id}: assembled {w}x{h} is not 2:1 (the size probe likely missed tiles)")
    if info is not None:
        info['zoom'] = chosen_zoom
    return cv2.cvtColor(cv2.resize(cropped_panorama, (8192, 4096), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------------------------------------------------
# Crop geometry (CPU, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def render_label_view(pano_rgb, x_norm, y_norm, camera_pitch=None, camera_roll=None, apply_tilt=True,
                      beta=DEFAULT_TILT_BETA, size=VIEW_SIZE, fov_deg=VIEW_FOV_DEG, device=None, dtype=None):
    """Render the ``size`` x ``size`` view centred on a label: ``(view_rgb uint8, tilt_applied)``.

    ``apply_tilt`` is True for human-authored labels (level-frame click on a rig-frame pano) and False for
    AI-authored ones (already rig-frame). ``tilt_applied`` is False also when the pose is unknown or zero (the
    package returns no tilt matrix then). The label is at the view centre ``(size / 2, size / 2)``.

    ``device=None`` renders with the package's numpy ``render_view`` (the ``--allow-cpu`` path, and what the CPU
    tests pin); a torch device renders with :func:`render_view_torch`, the GPU path of a real run.
    """
    from sidewalk_panorama_tools.geometry import label_view, render_view
    if not apply_tilt:
        camera_pitch = camera_roll = None
    view, tilt = label_view(x_norm, y_norm, fov_h_deg=fov_deg, size=size, camera_pitch=camera_pitch,
                            camera_roll=camera_roll, beta=beta)
    if device is None:
        return render_view(pano_rgb, view, tilt), tilt is not None
    return render_view_torch(pano_rgb, view, tilt, device, dtype), tilt is not None


def render_view_torch(pano_rgb, view, tilt, device, dtype=None):
    """The package's torch recipe (``sidewalk_panorama_tools.torch_geometry``), rounded to uint8 like ``render_view``.

    ``sampling_grid_torch`` -> ``normalized_grid_torch`` -> ``wrap_pad_torch`` -> ``grid_sample(bilinear, border,
    align_corners=False)``. Anti-aliasing is not implemented here: the reduction factor must be 1, which holds for
    an 8192-wide pano and a 2048 px 90-degree view (asserted). ``pano_rgb`` is an (H, W, 3) uint8 array or an
    (H, W, 3) tensor already on ``device``.
    """
    import torch
    import torch.nn.functional as F
    from sidewalk_panorama_tools.torch_geometry import normalized_grid_torch, sampling_grid_torch, wrap_pad_torch
    dtype = dtype or torch.float32
    H, W = pano_rgb.shape[:2]
    k = view.reduction_factor(W)
    assert k == 1, f"reduction factor {k} for a {W}-wide pano; render_view_torch does not box-reduce"
    pano = torch.as_tensor(pano_rgb, device=device).permute(2, 0, 1)[None].to(dtype)
    sx, sy = sampling_grid_torch(view, tilt, W, H, device=device, dtype=dtype)
    grid = normalized_grid_torch(sx, sy, W, H)
    out = F.grid_sample(wrap_pad_torch(pano), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def depth_sized_crop(view_img, depth_m):
    """Cut the depth-sized crop around the view centre and resize its long side to ``CROP_LONG_SIDE``.

    Returns ``(crop, crop_half_px, clipped)``; ``clipped`` is True when the square touched a frame edge and was
    cut short (near features: ``crop_half_px > size / 2``). The resize rule is sidewalk-ai-api's.

    >>> crop, half, clipped = depth_sized_crop(np.zeros((2048, 2048, 3), np.uint8), 10.0)
    >>> crop.shape, half, clipped
    ((640, 640, 3), 610, False)
    """
    import cv2
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError(f"bad depth {depth_m}")
    img_h, img_w = view_img.shape[:2]
    center_x, center_y = img_w // 2, img_h // 2
    crop_half = int(CROP_DEPTH_CONSTANT / depth_m)
    start_x = max(0, center_x - crop_half)
    start_y = max(0, center_y - crop_half)
    end_x = min(img_w, center_x + crop_half)
    end_y = min(img_h, center_y + crop_half)
    clipped = (center_x - crop_half < 0 or center_y - crop_half < 0 or center_x + crop_half > img_w
               or center_y + crop_half > img_h)
    region = view_img[start_y:end_y, start_x:end_x]
    crop_height, crop_width = region.shape[:2]
    if crop_width == 0 or crop_height == 0:
        raise ValueError(f"empty crop (crop_half {crop_half})")
    if crop_width > crop_height:
        resize_width, resize_height = CROP_LONG_SIDE, int(CROP_LONG_SIDE * crop_height / crop_width)
    else:
        resize_width, resize_height = int(CROP_LONG_SIDE * crop_width / crop_height), CROP_LONG_SIDE
    crop = cv2.resize(region, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    return crop, crop_half, bool(clipped)


def save_webp(rgb, path):
    """Write an RGB uint8 array as WebP at ``WEBP_QUALITY``."""
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb).save(path, format="WEBP", quality=WEBP_QUALITY)


# ---------------------------------------------------------------------------------------------------------------------
# GPU path (not covered by the unit tests)
# ---------------------------------------------------------------------------------------------------------------------

def resolve_device(allow_cpu):
    """``"cuda"`` when available; otherwise None (numpy render, CPU depth) if ``allow_cpu``, else exit with a message.

    A silent CPU fallback would turn a days-long GPU job into a months-long one, so it is opt-in."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        if not allow_cpu:
            sys.exit("torch is not installed; " + DEPTH_DEVICE_HELP)
        return None
    if not allow_cpu:
        sys.exit("No CUDA device found. " + DEPTH_DEVICE_HELP)
    print("WARNING: --allow-cpu: rendering with numpy and running depth on the CPU (smoke tests only).")
    return None


def load_depth_model(encoder, device):
    """Depth Anything V2 metric (vkitti, max 80 m) with ``encoder``; the checkpoint must be in the working dir.

    The vendored ``infer_image`` moves its input to cuda, else mps, else cpu on its own, so with ``device`` None
    (``--allow-cpu``) the model goes where that rule sends the input (mps on a Mac) or the two would not match."""
    import torch
    sys.path.insert(0, 'Depth-Anything-V2/metric_depth')
    from depth_anything_v2.dpt import DepthAnythingV2
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = DepthAnythingV2(**{**DEPTH_MODEL_CONFIGS[encoder], 'max_depth': DEPTH_MAX_M})
    model.load_state_dict(torch.load(f'depth_anything_v2_metric_{DEPTH_DATASET}_{encoder}.pth', map_location=device))
    return model.to(device).eval()


def create_perspective_crop(depth_model, pano_rgb, label_x_norm, label_y_norm, camera_pitch=None, camera_roll=None,
                            apply_tilt=True, beta=DEFAULT_TILT_BETA, device=None):
    """Render the label-centred view and cut a depth-sized crop around the centre.

    Returns ``(crop_rgb, info)`` with ``info`` = ``tilt_applied``, ``depth_m``, ``crop_half_px``, ``clipped``.
    Depth is inferred on the BGR view (``infer_image`` takes an OpenCV BGR image, as sidewalk-ai-api passes it); the
    Depth Anything code picks its own device (CUDA when present).
    """
    view_rgb, tilt_applied = render_label_view(pano_rgb, label_x_norm, label_y_norm, camera_pitch, camera_roll,
                                               apply_tilt=apply_tilt, beta=beta, device=device)
    depth = depth_model.infer_image(np.ascontiguousarray(view_rgb[..., ::-1]))
    c = VIEW_SIZE // 2
    depth_m = float(depth[c, c])
    crop, crop_half, clipped = depth_sized_crop(view_rgb, depth_m)
    return crop, {"tilt_applied": int(tilt_applied), "depth_m": round(depth_m, 4), "crop_half_px": crop_half,
                  "clipped": int(clipped)}


#: Settings that change what a crop looks like; a resume with different values would mix two datasets.
RESUME_SETTINGS = ("tilt_beta", "depth_encoder", "depth_dataset", "renderer", "include_ai_labels", "package_source")


def package_source(dist="sidewalk-panorama-tools"):
    """The commit a git-installed distribution came from (PEP 610 ``direct_url.json``), else its version, else None.

    The package's version string stays "1.0.0" across commits, so the commit is what identifies the geometry.
    """
    from importlib import metadata
    try:
        d = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        return None
    raw = d.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return f"{info.get('url')}@{commit}"
    return d.version


def check_resume(run_path, new_hash, settings):
    """Compare a resumed run with the ``generation_run.json`` of the previous one.

    Returns ``(old_hash or None, changed_settings)``: the previous selection hash if it differs from ``new_hash``,
    and ``{name: (old, new)}`` for every :data:`RESUME_SETTINGS` entry that differs. The caller refuses to resume on
    either unless ``--force-reselect``.

    >>> check_resume("/nonexistent/generation_run.json", "abc", {"tilt_beta": 0.95})
    (None, {})
    """
    if not os.path.exists(run_path):
        return None, {}
    with open(run_path) as f:
        old = json.load(f)
    old_hash = old.get("selection_sha256")
    old_settings = old.get("settings", {})
    changed = {k: (old_settings.get(k), settings.get(k)) for k in RESUME_SETTINGS
               if k in old_settings and old_settings.get(k) != settings.get(k)}
    return (old_hash if old_hash and old_hash != new_hash else None), changed


def count_ok_rows(manifest_path):
    """Number of ``ok`` rows in a manifest (0 if there is none)."""
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return 0
    return int((pd.read_csv(manifest_path, dtype=str, usecols=["status"])["status"] == "ok").sum())


def is_fatal(exc):
    """Errors that must stop the run instead of becoming one ``error:*`` row: broken invariants and GPU failures."""
    if isinstance(exc, AssertionError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "CUDA" in str(exc)


MAX_CONSECUTIVE_ERRORS = 50
MAX_CONSECUTIVE_MISSING_PANOS = 50  # panoramas in a row that could not be fetched (no_pano / incomplete_pano)
MISSING_STATUSES = ("no_pano", "incomplete_pano")


def parse_cities(arg):
    """``--cities a,b`` -> ``{city_id: subdomain}``; entries may be ``city_id`` (looked up in ``CITIES``) or
    ``city_id=subdomain``.

    >>> parse_cities("seattle-wa,vancouver-wa=vancouver")
    {'seattle-wa': 'sea', 'vancouver-wa': 'vancouver'}
    """
    if not arg:
        return dict(CITIES)
    out = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        cid, _, sub = item.partition("=")
        out[cid] = sub or CITIES.get(cid, cid)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument('--label_type', required=True, help='Specify the label type')
    p.add_argument('--out', default='.', help='output root; crops and manifest go under <out>/<label_type>/')
    p.add_argument('--limit', type=int, default=None, help='process only the first N selected labels (smoke tests)')
    p.add_argument('--manifest', default=None, help='default <out>/<label_type>/manifest.csv')
    p.add_argument('--priors', default=None,
                   help='shared deployment_priors.json; default <out>/deployment_priors.json. Partial runs (--cities '
                        'or --limit) write deployment_priors.partial.json beside it instead')
    p.add_argument('--depth-encoder', choices=['vitb', 'vitl'], default='vitb', help='the API serves vitb')
    p.add_argument('--tilt-beta', type=float, default=DEFAULT_TILT_BETA)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--cities', default=None, help='comma list of city_id[=subdomain], overriding CITIES')
    p.add_argument('--include-ai-labels', action='store_true',
                   help='keep AI-authored labels in the selection (rendered without tilt correction)')
    p.add_argument('--rawlabels-cache', default=None,
                   help='directory of saved rawLabels exports <city_id>.csv, read instead of the API when present')
    p.add_argument('--select-only', action='store_true',
                   help='select labels and write the priors, then stop (no GPU, no panoramas)')
    p.add_argument('--allow-cpu', action='store_true', help=DEPTH_DEVICE_HELP)
    p.add_argument('--max-missing-frac', type=float, default=0.2,
                   help='exit non-zero if more than this fraction of attempted labels ended no_pano/incomplete_pano')
    p.add_argument('--force-reselect', action='store_true',
                   help='resume although the selection changed since the last run (stale manifest rows are dropped)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    label_type = args.label_type
    base_dir = os.path.join(args.out, label_type)
    manifest_path = args.manifest or os.path.join(base_dir, "manifest.csv")
    run_path = os.path.join(base_dir, "generation_run.json")
    priors_path = args.priors or os.path.join(args.out, "deployment_priors.json")
    partial = args.cities is not None or args.limit is not None
    if partial:
        priors_path = os.path.splitext(priors_path)[0] + ".partial.json"
    cities = parse_cities(args.cities)
    device = None if args.select_only else resolve_device(args.allow_cpu)  # fail fast, before the downloads

    all_cities = pd.concat([load_city(c, s, args.rawlabels_cache) for c, s in cities.items()], ignore_index=True)
    n_rows = int((all_cities['label_type'] == label_type).sum())

    eligible, dropped = eligible_labels(all_cities, label_type)  # once, for both the prior and the selection
    print(f"{n_rows} {label_type} rows; dropped for unusable pano geometry: {dict(dropped) or 'none'}")
    entry = deployment_prior_entry(eligible, list(cities), n_rows)

    def write_priors():
        update_deployment_priors(priors_path, label_type, entry)
        print(f"Deployment prior ({priors_path}{', PARTIAL run' if partial else ''}): "
              f"{entry['correct']} correct, {entry['incorrect']} incorrect")

    combined_filtered = select_training_labels(eligible, seed=args.seed, include_ai_labels=args.include_ai_labels)
    combined_filtered['label_uid'] = combined_filtered['city'] + ":" + combined_filtered['label_id'].astype(str)
    assert combined_filtered['label_uid'].is_unique, "label_uid (city:label_id) is not unique in the selection"
    selection = dict(zip(combined_filtered['label_uid'], combined_filtered['label_class']))  # before --limit
    sel_hash = selection_hash(combined_filtered)
    if args.limit is not None:
        combined_filtered = combined_filtered.head(args.limit)
    if args.select_only:
        write_priors()
        print(f"--select-only: {len(combined_filtered)} labels selected; stopping before rendering.")
        return

    settings = {"tilt_beta": args.tilt_beta, "depth_encoder": args.depth_encoder, "depth_dataset": DEPTH_DATASET,
                "renderer": "torch" if device else "numpy", "include_ai_labels": args.include_ai_labels,
                "package_source": package_source()}
    previous, changed = check_resume(run_path, sel_hash, settings)
    if (previous or changed) and not args.force_reselect:
        n_ok = count_ok_rows(manifest_path)
        why = []
        if previous:
            why.append(f"the selection changed (sha256 {previous[:12]} -> {sel_hash[:12]})")
        if changed:
            why.append("settings changed: " + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changed.items()))
        # generation_run.json is left untouched, so the record still describes the crops on disk.
        sys.exit(f"Refusing to resume in {base_dir}: {'; '.join(why)}. The manifest holds {n_ok} ok rows made "
                 f"under the previous run. Re-run with --force-reselect to accept: stale ok rows are dropped "
                 f"(all of them if a setting changed) and redone; old crop files stay on disk until overwritten.")
    write_priors()  # only once the resume check has passed
    os.makedirs(base_dir, exist_ok=True)
    manifest = ManifestWriter(manifest_path, base_dir, selection, keep_ok=not changed)
    if previous or changed:
        print(f"WARNING: --force-reselect: selection changed: {bool(previous)}, settings changed: {changed or 'none'}; "
              f"dropped ok rows {dict(manifest.stale) or 'none'}, kept {len(manifest.done)}.")

    with open(run_path, "w") as f:  # read by fine_tune_classifier.py
        json.dump({"label_type": label_type, "settings": settings, **settings, "seed": args.seed,
                   "cities": list(cities), "limit": args.limit, "selection_size": len(selection),
                   "selection_sha256": sel_hash, "started": date.today().isoformat()}, f, indent=2)
        f.write("\n")

    depth_model = load_depth_model(args.depth_encoder, device)
    for cls in ("correct", "incorrect"):
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    # Process panorama by panorama (stable, so the order within a pano is the selection's) and reuse the last pano.
    combined_filtered = combined_filtered.sort_values('pano_id', kind='mergesort')
    total_images = len(combined_filtered)
    start_time = time.time()
    status_counts = Counter()
    last_pano = (None, None, None)  # (pano_id, BGR array or the status string it failed with, tile zoom)
    consecutive_errors = consecutive_missing = 0
    zoom_counts = Counter()
    with manifest:
        for processed_count, (_, row) in enumerate(combined_filtered.iterrows(), start=1):
            crop_path = f"{row['label_class']}/{row['city']}_{row['label_id']}.webp"
            rec = manifest_row(row, label_type, crop_path, args.tilt_beta)
            if manifest.is_done(rec['label_uid']):
                status_counts['skipped (already ok)'] += 1
                continue
            try:
                if last_pano[0] != row['pano_id']:
                    info = {}
                    try:  # the column was renamed from gsv_panorama_id in the v3 API
                        fetched = fetch_panorama(row['pano_id'], info=info)
                        last_pano = (row['pano_id'], 'no_pano' if fetched is None else fetched, info.get('zoom'))
                    except IncompletePanorama as e:
                        print(e)
                        last_pano = (row['pano_id'], 'incomplete_pano', None)
                    if isinstance(last_pano[1], str):
                        consecutive_missing += 1
                        if consecutive_missing >= MAX_CONSECUTIVE_MISSING_PANOS:
                            sys.exit(f"Aborting: {consecutive_missing} panoramas in a row could not be fetched "
                                     f"(last: {last_pano[1]}). Check the tile endpoint; the run is resumable.")
                    else:
                        consecutive_missing = 0
                        zoom_counts[last_pano[2]] += 1
                pano_bgr = last_pano[1]
                rec['pano_zoom'] = last_pano[2]
                if isinstance(pano_bgr, str):
                    rec['status'] = pano_bgr
                else:
                    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])
                    crop, info = create_perspective_crop(
                        depth_model, pano_rgb, row['pano_x'] / row['pano_width'], row['pano_y'] / row['pano_height'],
                        camera_pitch=row.get('camera_pitch'), camera_roll=row.get('camera_roll'),
                        apply_tilt=row['author'] == 'human', beta=args.tilt_beta, device=device)
                    save_webp(crop, os.path.join(base_dir, crop_path))
                    rec.update(info)
                    rec['status'] = 'ok'
            except Exception as e:
                if is_fatal(e):
                    raise
                traceback.print_exc()
                rec['status'] = f"error:{type(e).__name__}"
            manifest.write(rec)
            status_counts[rec['status']] += 1
            consecutive_errors = consecutive_errors + 1 if rec['status'].startswith('error:') else 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                sys.exit(f"Aborting: {consecutive_errors} consecutive labels failed with error:* (last: "
                         f"{rec['status']}). The manifest is resumable once the cause is fixed.")
            elapsed = time.time() - start_time
            left = elapsed / processed_count * (total_images - processed_count)
            print(f"Processed {processed_count}/{total_images}: {rec['label_uid']} {rec['status']}. "
                  f"Elapsed {timedelta(seconds=int(elapsed))}, left ~{timedelta(seconds=int(left))}")

    final = pd.read_csv(manifest_path, dtype=str)
    assert final['label_uid'].is_unique, f"{manifest_path}: label_uid is not unique"
    n_ok = int((final['status'] == 'ok').sum())
    attempted = sum(n for k, n in status_counts.items() if not k.startswith('skipped'))
    missing = sum(status_counts[k] for k in MISSING_STATUSES)
    print("Processing complete.")
    print(f"  dropped for unusable pano geometry (before selection): {dict(dropped) or 'none'}")
    print(f"  status counts: {dict(status_counts)}; {n_ok} ok rows in {manifest_path}")
    print(f"  panoramas fetched by tile zoom: {dict(zoom_counts) or 'none'} "
          f"(zoom-3 fallbacks: {zoom_counts.get(3, 0)})")
    if attempted and missing / attempted > args.max_missing_frac:
        sys.exit(f"{missing} of {attempted} attempted labels ({missing / attempted:.0%}) ended no_pano/incomplete_pano, "
                 f"above --max-missing-frac {args.max_missing_frac}: the dataset would shrink silently.")
    if n_ok == 0:
        # e.g. the tile endpoint refusing every request: fail the SLURM job instead of an empty dataset with exit 0
        sys.exit(f"No crops in {manifest_path}: every label failed (no_pano or error; non-200 tile responses are reported per pano above).")


if __name__ == "__main__":
    main()
