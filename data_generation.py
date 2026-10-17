"""Build a correct/incorrect crop dataset for one Project Sidewalk label type.

For every label of ``--label_type`` across the cities listed in ``CITIES``, this script:

1. pulls the label and its validations from the public v3 API (``/v3/api/rawLabels``);
2. keeps human-authored labels the crowd clearly endorsed (net >= 3 human agree votes) or clearly rejected
   (net >= 2 human disagree votes), downsampling the majority class to ``CORRECT_PROPORTION``, and records the
   PRE-downsampling class counts in ``deployment_priors.json`` (the prior ``calibration/calibrate.py`` reweights to);
3. reassembles the Google Street View panorama from tiles, renders a 2048x2048, 90-degree perspective view
   CENTRED on the label with rig-tilt correction (``sidewalk_panorama_tools.geometry``), estimates metric depth at
   the view centre with Depth Anything V2, and cuts a crop whose size scales with 1/depth so near and far features
   fill the frame similarly;
4. writes ``<out>/<label_type>/<correct|incorrect>/<city>_<label_id>.webp`` and appends one row per label to
   ``<out>/<label_type>/manifest.csv`` (the input of ``split_data.py``).

The run is resumable: a label whose manifest row says ``ok`` and whose crop exists is skipped on restart.

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
https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/main.py ; it still renders without tilt correction and
at ``pano_width // 4`` pixels, so parity needs the companion change there (ProjectSidewalk/sidewalk-ai-api#8).

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
    python data_generation.py --label_type CurbRamp --limit 20 --out /tmp/smoke
    python data_generation.py --label_type CurbRamp --hard-examples analysis/hard_examples_2026-10-17.csv
"""
import argparse
import csv
import io
import json
import os
import sys
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
# crop filenames) -> the deployment's subdomain, sidewalk-<subdomain>.cs.washington.edu. Not the full list of ~52
# deployments; discovering them from /v3/api/cities is https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8.
# The three Taiwan deployments are private (url null in /v3/api/cities) but serve rawLabels.
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
    "human_disagree", "severity", "tags", "high_quality_user", "image_capture_date", "pano_source", "camera_pitch",
    "camera_roll", "tilt_applied", "tilt_beta", "depth_m", "crop_half_px", "clipped", "hard", "crop_path", "status",
]


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
        subdomain = subdomain or CITIES.get(city_id, city_id)
        df = pd.read_csv(f'https://sidewalk-{subdomain}.cs.washington.edu/v3/api/rawLabels?filetype=csv',
                         low_memory=False)
    df['city'] = city_id
    return df


def human_vote_counts(validations_json):
    """Count (agree, disagree) votes from human validators only.

    The API's ``agree_count``/``disagree_count`` columns include the SidewalkAI account's own votes,
    so using them would train the validator on its own output. The ``validations`` column carries a
    ``validator_type`` per vote ("Human" or "AI"), which is what we count here.

    >>> human_vote_counts('[{"user_id":"a","validation":"Agree","validator_type":"AI"},'
    ...                   ' {"user_id":"b","validation":"Agree","validator_type":"Human"},'
    ...                   ' {"user_id":"c","validation":"Disagree","validator_type":"Human"}]')
    (1, 1)
    >>> human_vote_counts(float('nan'))
    (0, 0)
    """
    if not isinstance(validations_json, str) or not validations_json:
        return 0, 0
    votes = json.loads(validations_json)
    agree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Agree")
    disagree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Disagree")
    return agree, disagree


def label_author(user_id):
    """``"ai"`` for a label placed by the SidewalkAI account, else ``"human"``.

    >>> label_author("51b0b927-3c8a-45b2-93de-bd878d1e5cf4"), label_author("someone-else")
    ('ai', 'human')
    """
    return "ai" if str(user_id) == AI_USER_ID else "human"


def eligible_labels(all_labels, label_type, include_ai_labels=False):
    """Labels of one type that meet the vote thresholds, with ``human_agree``, ``human_disagree``, ``author`` and
    ``label_class`` added; no downsampling.

    Rule (unchanged since the first models): ``incorrect`` if net human disagree votes > 1, ``correct`` if net
    human agree votes > 2; everything else (unvalidated, unsure, contested) is left out. GSV panoramas only (the tile
    fetcher below cannot fetch other providers), and legacy DC labels whose pano coordinates were never converted
    (negative pano_y) are dropped (issue #9). AI-authored labels are dropped unless ``include_ai_labels``.
    """
    df = all_labels[all_labels['label_type'] == label_type].copy()
    df = df[(df['pano_source'].astype(str).str.lower() == 'gsv') & (df['pano_y'] >= 0)]
    df['author'] = df['user_id'].map(label_author) if 'user_id' in df else 'human'
    if not include_ai_labels:
        df = df[df['author'] == 'human']
    counts = df['validations'].apply(human_vote_counts)
    df['human_agree'] = [a for a, _ in counts]
    df['human_disagree'] = [d for _, d in counts]
    net = df['human_agree'] - df['human_disagree']
    df['label_class'] = np.where(net > 2, 'correct', np.where(-net > 1, 'incorrect', ''))
    return df[df['label_class'] != ''].copy()


def select_training_labels(all_labels, label_type, seed=SEED, correct_proportion=CORRECT_PROPORTION,
                           include_ai_labels=False):
    """Pick the labels of one type that the crowd clearly endorsed or rejected, balanced to ``correct_proportion``.

    See :func:`eligible_labels` for the rule. The ``correct`` class is downsampled so it makes up
    ``correct_proportion`` of the result. Note that this ratio is not the deployment prior
    (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4); :func:`deployment_prior_entry` records that.
    """
    df = eligible_labels(all_labels, label_type, include_ai_labels)
    filtered_disagree = df[df['label_class'] == 'incorrect']
    print(f"Incorrect Examples: {len(filtered_disagree)}")

    filtered_agree = df[df['label_class'] == 'correct']
    filtered_agree = filtered_agree.sample(frac=1, random_state=seed).reset_index(drop=True)
    amount_correct = int((correct_proportion / (1 - correct_proportion)) * len(filtered_disagree))
    filtered_agree = filtered_agree[:amount_correct]
    print(f"Correct Examples: {len(filtered_agree)}")

    combined = pd.concat([filtered_agree, filtered_disagree])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the combined dataframe


def deployment_prior_entry(all_labels, label_type, cities, measured_on=None):
    """The pre-downsampling class counts of human-authored labels meeting the vote thresholds: the class mix the
    validator meets in production (which never sends it AI-authored labels), for ``calibration/calibrate.py``."""
    counts = eligible_labels(all_labels, label_type, include_ai_labels=False)['label_class'].value_counts()
    return {"correct": int(counts.get('correct', 0)), "incorrect": int(counts.get('incorrect', 0)),
            "measured_on": measured_on or date.today().isoformat(),
            "source": f"rawLabels of {', '.join(cities)}, human votes only"}


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


def load_hard_examples(path):
    """The set of ``label_uid``s in a hard-examples CSV (``label_uid,reason``), or an empty set for None."""
    if not path:
        return set()
    return set(pd.read_csv(path)['label_uid'].astype(str))


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------

class ManifestWriter:
    """Append-only, flushed-per-row ``manifest.csv``, resumable.

    On open, an existing manifest is read and de-duplicated (last row per ``label_uid`` wins). Rows whose status is
    ``ok`` and whose crop file exists are kept and their labels are skipped (:meth:`is_done`); every other row is
    dropped so the label is attempted again and gets exactly one fresh row. The kept rows are rewritten atomically
    before any append, so ``label_uid`` stays unique in the file at every moment.

    Example::

        with ManifestWriter("CurbRamp/manifest.csv", "CurbRamp") as m:
            if not m.is_done("seattle-wa:1"):
                m.write({...})
    """

    def __init__(self, path, base_dir):
        self.path = path
        self.base_dir = base_dir
        self.done = set()
        kept = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            old = old.drop_duplicates('label_uid', keep='last')
            for row in old.to_dict('records'):
                if row.get('status') == 'ok' and os.path.exists(os.path.join(base_dir, row.get('crop_path', ''))):
                    kept.append(row)
                    self.done.add(row['label_uid'])
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


def manifest_row(row, label_type, crop_path, hard, tilt_beta):
    """The label-level manifest fields for one selected label (the crop fields are filled in after rendering)."""
    city = row['city']
    return {
        "label_uid": f"{city}:{row['label_id']}", "city": city, "label_id": row['label_id'],
        "pano_id": row['pano_id'], "label_type": label_type, "label_class": row['label_class'],
        "author": row['author'], "human_agree": row['human_agree'], "human_disagree": row['human_disagree'],
        "severity": row.get('severity'), "tags": row.get('tags'), "high_quality_user": row.get('high_quality_user'),
        "image_capture_date": row.get('image_capture_date'), "pano_source": row.get('pano_source'),
        "camera_pitch": row.get('camera_pitch'), "camera_roll": row.get('camera_roll'),
        "tilt_applied": 0, "tilt_beta": tilt_beta, "depth_m": None, "crop_half_px": None, "clipped": None,
        "hard": int(hard), "crop_path": crop_path, "status": "",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Panorama
# ---------------------------------------------------------------------------------------------------------------------

def fetch_panorama(pano_id):
    import cv2
    import requests
    from PIL import Image

    chosen_zoom = None

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
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if chosen_zoom != None or not _is_black_tile(tile):
                    chosen_zoom = zoom
                    return x, y, tile
        except Exception as e:
            print(e)

        if chosen_zoom == None:
            # Try fallback with zoom=3
            fallback_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/tile"
                f"?cb_client=maps_sv.tactile&panoid={pano_id}"
                f"&x={x}&y={y}&zoom=3"
            )
            try:
                response = requests.get(fallback_url, headers=HEADERS)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    chosen_zoom = 3
                    return x, y, tile
            except Exception as e:
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
                    tiles_cache[(x, y)] = tile

                    if _is_black_tile(tile):
                        return x - 1, y, tiles_cache

                    x += 1

            x += 1
            y += 1

    def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()

        # 100 workers, as in sidewalk-ai-api. Five label-type processes run in parallel under SLURM, and the
        # earlier 1,000-thread pool per process is the kind of burst that gets a source IP rate-limited.
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

    dimension_result = _find_panorama_dimensions()
    if dimension_result is None:
        return None

    max_x, max_y, initial_tiles = dimension_result
    full_tiles = _fetch_remaining_tiles(max_x, max_y, initial_tiles)
    assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
    if assembled_panorama is None:
        return None
    cropped_panorama = _crop(assembled_panorama)
    return cv2.cvtColor(cv2.resize(cropped_panorama, (8192, 4096), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------------------------------------------------
# Crop geometry (CPU, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def render_label_view(pano_rgb, x_norm, y_norm, camera_pitch=None, camera_roll=None, apply_tilt=True,
                      beta=DEFAULT_TILT_BETA, size=VIEW_SIZE, fov_deg=VIEW_FOV_DEG):
    """Render the ``size`` x ``size`` view centred on a label: ``(view_rgb, tilt_applied)``.

    ``apply_tilt`` is True for human-authored labels (level-frame click on a rig-frame pano) and False for
    AI-authored ones (already rig-frame). ``tilt_applied`` is False also when the pose is unknown or zero (the
    package returns no tilt matrix then). The label is at the view centre ``(size / 2, size / 2)``.
    """
    from sidewalk_panorama_tools.geometry import label_view, render_view
    if not apply_tilt:
        camera_pitch = camera_roll = None
    view, tilt = label_view(x_norm, y_norm, fov_h_deg=fov_deg, size=size, camera_pitch=camera_pitch,
                            camera_roll=camera_roll, beta=beta)
    return render_view(pano_rgb, view, tilt), tilt is not None


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

def load_depth_model(encoder):
    """Depth Anything V2 metric (vkitti, max 80 m) with ``encoder``; the checkpoint must be in the working dir."""
    import torch
    sys.path.insert(0, 'Depth-Anything-V2/metric_depth')
    from depth_anything_v2.dpt import DepthAnythingV2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnythingV2(**{**DEPTH_MODEL_CONFIGS[encoder], 'max_depth': DEPTH_MAX_M})
    model.load_state_dict(torch.load(f'depth_anything_v2_metric_{DEPTH_DATASET}_{encoder}.pth', map_location=device))
    return model.to(device).eval()


def create_perspective_crop(depth_model, pano_rgb, label_x_norm, label_y_norm, camera_pitch=None, camera_roll=None,
                            apply_tilt=True, beta=DEFAULT_TILT_BETA):
    """Render the label-centred view and cut a depth-sized crop around the centre.

    Returns ``(crop_rgb, info)`` with ``info`` = ``tilt_applied``, ``depth_m``, ``crop_half_px``, ``clipped``.
    Depth is inferred on the BGR view (``infer_image`` takes an OpenCV BGR image, as sidewalk-ai-api passes it).
    """
    view_rgb, tilt_applied = render_label_view(pano_rgb, label_x_norm, label_y_norm, camera_pitch, camera_roll,
                                               apply_tilt=apply_tilt, beta=beta)
    depth = depth_model.infer_image(np.ascontiguousarray(view_rgb[..., ::-1]))
    c = VIEW_SIZE // 2
    depth_m = float(depth[c, c])
    crop, crop_half, clipped = depth_sized_crop(view_rgb, depth_m)
    return crop, {"tilt_applied": int(tilt_applied), "depth_m": round(depth_m, 4), "crop_half_px": crop_half,
                  "clipped": int(clipped)}


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
    p.add_argument('--priors', default=None, help='shared deployment_priors.json; default <out>/deployment_priors.json')
    p.add_argument('--depth-encoder', choices=['vitb', 'vitl'], default='vitb', help='the API serves vitb')
    p.add_argument('--tilt-beta', type=float, default=DEFAULT_TILT_BETA)
    p.add_argument('--hard-examples', default=None, help='CSV with a label_uid column; those rows get hard = 1')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--cities', default=None, help='comma list of city_id[=subdomain], overriding CITIES')
    p.add_argument('--include-ai-labels', action='store_true',
                   help='keep AI-authored labels in the selection (rendered without tilt correction)')
    p.add_argument('--rawlabels-cache', default=None,
                   help='directory of saved rawLabels exports <city_id>.csv, read instead of the API when present')
    p.add_argument('--select-only', action='store_true',
                   help='select labels and write deployment_priors.json, then stop (no GPU, no panoramas)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    label_type = args.label_type
    base_dir = os.path.join(args.out, label_type)
    manifest_path = args.manifest or os.path.join(base_dir, "manifest.csv")
    priors_path = args.priors or os.path.join(args.out, "deployment_priors.json")
    cities = parse_cities(args.cities)

    all_cities = pd.concat([load_city(c, s, args.rawlabels_cache) for c, s in cities.items()], ignore_index=True)

    entry = deployment_prior_entry(all_cities, label_type, list(cities))
    update_deployment_priors(priors_path, label_type, entry)
    print(f"Deployment prior ({priors_path}): {entry['correct']} correct, {entry['incorrect']} incorrect")

    combined_filtered = select_training_labels(all_cities, label_type, seed=args.seed,
                                               include_ai_labels=args.include_ai_labels)
    if args.limit is not None:
        combined_filtered = combined_filtered.head(args.limit)
    uids = combined_filtered['city'] + ":" + combined_filtered['label_id'].astype(str)
    assert uids.is_unique, "label_uid (city:label_id) is not unique in the selection"
    if args.select_only:
        print(f"--select-only: {len(combined_filtered)} labels selected; stopping before rendering.")
        return

    hard = load_hard_examples(args.hard_examples)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "generation_run.json"), "w") as f:  # read by fine_tune_classifier.py
        json.dump({"label_type": label_type, "depth_encoder": args.depth_encoder, "depth_dataset": DEPTH_DATASET,
                   "tilt_beta": args.tilt_beta, "seed": args.seed, "cities": list(cities),
                   "include_ai_labels": args.include_ai_labels, "hard_examples": args.hard_examples,
                   "limit": args.limit, "started": date.today().isoformat()}, f, indent=2)
        f.write("\n")
    depth_model = load_depth_model(args.depth_encoder)
    for cls in ("correct", "incorrect"):
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    total_images = len(combined_filtered)
    start_time = time.time()
    status_counts = Counter()
    with ManifestWriter(manifest_path, base_dir) as manifest:
        for processed_count, (_, row) in enumerate(combined_filtered.iterrows(), start=1):
            crop_path = f"{row['label_class']}/{row['city']}_{row['label_id']}.webp"
            rec = manifest_row(row, label_type, crop_path, f"{row['city']}:{row['label_id']}" in hard, args.tilt_beta)
            if manifest.is_done(rec['label_uid']):
                status_counts['skipped (already ok)'] += 1
                continue
            try:
                pano_bgr = fetch_panorama(row['pano_id'])  # column was renamed from gsv_panorama_id in the v3 API
                if pano_bgr is None:
                    rec['status'] = 'no_pano'
                else:
                    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])
                    crop, info = create_perspective_crop(
                        depth_model, pano_rgb, row['pano_x'] / row['pano_width'], row['pano_y'] / row['pano_height'],
                        camera_pitch=row.get('camera_pitch'), camera_roll=row.get('camera_roll'),
                        apply_tilt=row['author'] == 'human', beta=args.tilt_beta)
                    save_webp(crop, os.path.join(base_dir, crop_path))
                    rec.update(info)
                    rec['status'] = 'ok'
            except Exception as e:
                traceback.print_exc()
                rec['status'] = f"error:{type(e).__name__}"
            manifest.write(rec)
            status_counts[rec['status']] += 1
            elapsed = time.time() - start_time
            left = elapsed / processed_count * (total_images - processed_count)
            print(f"Processed {processed_count}/{total_images}: {rec['label_uid']} {rec['status']}. "
                  f"Elapsed {timedelta(seconds=int(elapsed))}, left ~{timedelta(seconds=int(left))}")

    final = pd.read_csv(manifest_path, dtype=str)
    assert final['label_uid'].is_unique, f"{manifest_path}: label_uid is not unique"
    print("Processing complete. Status counts:", dict(status_counts))


if __name__ == "__main__":
    main()
