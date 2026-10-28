"""Build a correct/incorrect crop dataset for one Project Sidewalk label type.

For every label of ``--label_type`` across the public deployments ``/v3/api/cities`` lists (or ``--cities``), this
script:

1. pulls the label and its validations from the public v3 API (``/v3/api/rawLabels``);
2. keeps human-authored labels the crowd clearly endorsed (net >= 3 human agree votes) or clearly rejected
   (net >= 2 human disagree votes), or that an expert validator (env ``EXPERTS``) voted on with no other human voting
   the other way (:func:`expert_verdict`; a different ``EXPERTS`` is a selection change), downsampling the ``correct`` class to ``CORRECT_PROPORTION``, and records the
   PRE-downsampling class counts in ``deployment_priors.json`` (the prior ``calibration/calibrate.py`` reweights to);
3. reads the panorama from the lab's pano store (``--pano-store``), renders a 2048x2048, 90-degree perspective view
   CENTRED on the label with rig-tilt correction (``sidewalk_panorama_tools.geometry``), estimates metric depth at
   the view centre with Depth Anything V2, and cuts a crop whose size scales with 1/depth so near and far features
   fill the frame similarly;
4. writes ``<out>/<label_type>/<correct|incorrect>/<city>_<label_id>.webp`` and appends one row per label to
   ``<out>/<label_type>/manifest.csv`` (the input of ``split_data.py``, which also joins the ``hard`` flag).

The run is resumable: a label whose manifest row says ``ok``, whose crop exists, and which is still selected with
the same class is skipped on restart. ``generation_run.json`` records a hash of the selection, so resuming against a
changed selection (new votes, other cities, toggling ``--include-ai-labels``) stops with counts unless ``--force-reselect`` is given.

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

**Panoramas** come only from the pano store, or a staged copy of it with the same layout:
``<pano-store>/<city>/<pano_id[:2]>/<pano_id>.jpg``, ``<city>`` being the city_id (the layout sidewalk-ai-api
reads through ``SIDEWALK_SCRAPES_DIR``), with a ``.pano-store`` marker file at the root. Each is resized to 8192x4096
with ``cv2.INTER_LINEAR``, the same method as the API's cache path (byte-identical pixels also need matching opencv
builds). Nothing is downloaded: a pano missing from the store is ``no_pano``, an unreadable or non-2:1 file is
``bad_pano``, and a store that stops answering (a dropped mount, a vanished marker) stops the run. Labels on non-GSV
imagery (Mapillary, Panoramax) are dropped before selection and counted as ``non_gsv`` in the summary: their pose
convention differs from GSV's and is not handled yet (docs/decisions.md).

**Cities** are the public deployments ``/v3/api/cities`` lists with a ``url``, sorted by city_id
(https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8). With ``--rawlabels-cache`` the listing is read
from ``<cache>/cities.json``, so a run against a complete cache makes no API call; every listed city must then have its
``<city_id>.csv`` there (no live download into a snapshot). A missing listing is fetched and saved, except into a cache
that already holds CSVs (a snapshot from before discovery), which needs ``--refresh-cities``. ``deployment_priors.json``
and ``generation_run.json`` record the list and where it came from; resuming with a different list stops like a changed
selection, and ``--select-only`` against such a run writes ``deployment_priors.partial.json``. The pano store needs a
directory only for cities with a selected label. The 19-city default from before discovery is reproduced with
``--cities`` and those 19 ids (docs/training-data.md, Cities).

Usage::

    python data_generation.py --label_type CurbRamp --pano-store /path/to/Panoramas   # full run, writes ./CurbRamp/
    python data_generation.py --label_type CurbRamp --pano-store /path/to/Panoramas --limit 20 --out /tmp/smoke \
        --allow-cpu
    python data_generation.py --label_type CurbRamp --select-only --rawlabels-cache ~/rawlabels  # priors, no GPU
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
import time
import traceback
from collections import Counter
from datetime import date, timedelta
from urllib.parse import urlparse

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
# Expert validators' user_ids, from the environment as EXPERTS="<any name>=<user_id>,..." (the form analysis/analyze.py
# reads). Kept out of the repo on purpose: named accounts do not belong in public code.
EXPERTS_ENV = "EXPERTS"

# The deployment directory: every deployment's city_id, visibility and url. The default city list is its public
# deployments with a url (the rule analysis/fetch_all.py uses); any deployment serves the same listing.
CITIES_API = "https://sidewalk-teaneck.cs.washington.edu/v3/api/cities"
CITIES_CACHE_FILE = "cities.json"  # the listing's file name inside --rawlabels-cache

LISTING_RETRY_S = 10  # pause before the one retry of the listing fetch

# city_id -> subdomain (sidewalk-<subdomain>.cs.washington.edu) for public deployments /v3/api/cities lists without a
# url; an entry here puts such a deployment in the default list. Empty: every public deployment has a url as of
# 2026-10-19. Any deployment the API lists without a url is reached with ``--cities <city_id>=<subdomain>``.
HOST_OVERRIDES = {}

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
    "label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "class_source", "author", "human_agree",
    "human_disagree", "severity", "tags", "high_quality_user", "image_capture_date", "pano_source", "pano_path",
    "pano_native_w", "pano_native_h", "camera_pitch",
    "camera_roll", "tilt_applied", "tilt_beta", "depth_m", "crop_half_px", "clipped", "crop_path", "status",
]
# ``class_source`` is the rule that set ``label_class``: ``margin`` (the crowd vote margin) or ``expert``
# (:func:`expert_verdict`; empty on rows from before it existed). ``pano_source`` is the imagery provider from rawLabels (e.g. ``gsv``); ``pano_path`` is the absolute pano-store file
# that was read (empty when it was missing, and on rows made by the removed tile fetcher, which a resume redoes), and
# ``pano_native_w``/``pano_native_h`` its size before the resize (a 2:1 file with black bands passes the 2:1 check, so
# the native size is what finds them later). (``hard`` is not a manifest column: split_data.py --hard-examples joins
# it, so a new list needs no regeneration.)

CSV_TIMEOUT = (10, 300)       # rawLabels exports run to ~175 MB
PANO_SIZE = (8192, 4096)      # (width, height) every store pano is resized to, as sidewalk-ai-api does
PANO_STORE_MARKER = ".pano-store"  # must exist at the store root (any contents; its first line is recorded)
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
        url = f'{deployment_base_url(subdomain or HOST_OVERRIDES.get(city_id, city_id))}/v3/api/rawLabels?filetype=csv'
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


def deployment_base_url(host):
    """A deployment's base URL from its subdomain, or ``host`` itself when it is already a URL.

    >>> deployment_base_url("sea"), deployment_base_url("https://example.org/")
    ('https://sidewalk-sea.cs.washington.edu', 'https://example.org')
    """
    return host.rstrip("/") if "://" in host else f"https://sidewalk-{host}.cs.washington.edu"


def subdomain_from_url(url):
    """The ``<subdomain>`` of a ``https://sidewalk-<subdomain>.cs.washington.edu`` url, else the url itself (which
    :func:`deployment_base_url` accepts as is).

    >>> subdomain_from_url("https://sidewalk-sea.cs.washington.edu/"), subdomain_from_url("https://example.org")
    ('sea', 'https://example.org')
    """
    host = urlparse(url).hostname or ""
    prefix, suffix = "sidewalk-", ".cs.washington.edu"
    if host.startswith(prefix) and host.endswith(suffix) and len(host) > len(prefix) + len(suffix):
        return host[len(prefix):-len(suffix)]
    return url.rstrip("/")


class ListingError(RuntimeError):
    """The ``/v3/api/cities`` listing could not be fetched, or is not a listing."""


def listing_entries(listing):
    """The entries of a ``/v3/api/cities`` listing, checked: raises :class:`ListingError` unless ``listing`` is a dict
    (or, for an old ``analysis/fetch_all.py`` cache, a bare list) whose ``cities`` is a non-empty list of dicts, each
    with a string ``city_id`` matching ``[a-z0-9-]+``, no id twice.

    >>> [c["city_id"] for c in listing_entries({"cities": [{"city_id": "a"}, {"city_id": "b"}]})]
    ['a', 'b']
    >>> listing_entries({"cities": [{"city_id": "a"}, {"city_id": "a"}]})
    Traceback (most recent call last):
    ...
    data_generation.ListingError: /v3/api/cities listing names city_id a twice
    """
    entries = listing if isinstance(listing, list) else listing.get("cities") if isinstance(listing, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ListingError(f"/v3/api/cities listing has no non-empty 'cities' list (got {type(listing).__name__}"
                           f"{' with keys ' + ', '.join(map(str, listing)) if isinstance(listing, dict) else ''})")
    seen = set()
    for c in entries:
        cid = c.get("city_id") if isinstance(c, dict) else None
        if not isinstance(cid, str) or not cid.strip():
            raise ListingError(f"/v3/api/cities listing has an entry without a string city_id: {c!r:.200}")
        if not re.fullmatch(r"[a-z0-9-]+", cid):  # ids become cache file names, store paths and shell arguments
            raise ListingError(f"/v3/api/cities listing has a city_id that is not [a-z0-9-]+: {cid!r:.100}")
        if cid in seen:
            raise ListingError(f"/v3/api/cities listing names city_id {cid} twice")
        seen.add(cid)
    return entries


def _listing_url(c):
    """An entry's ``url``, stripped, if it is an http(s) url with a host; None when absent; warns and returns None
    when it is something else."""
    url = c.get("url")
    if url is None or (isinstance(url, str) and not url.strip()):
        return None
    parsed = urlparse(url.strip()) if isinstance(url, str) else None
    if parsed is None or parsed.scheme not in ("http", "https") or not parsed.hostname:
        print(f"WARNING: /v3/api/cities gives {c['city_id']} the url {url!r}, not an http(s) url; skipped.")
        return None
    return url.strip()


def discover_cities(listing, public_only=True):
    """``{city_id: subdomain}`` from a ``/v3/api/cities`` listing (checked by :func:`listing_entries`), sorted by
    city_id.

    With ``public_only`` (the default list) only ``visibility == "public"`` entries count; a visibility other than
    ``public``/``private`` (or none) is excluded with a warning. Without it (host lookup for ``--cities``) every
    entry counts. An entry needs an http(s) ``url``, or a :data:`HOST_OVERRIDES` entry; a public deployment with
    neither is skipped with a warning.

    >>> discover_cities({"cities": [
    ...     {"city_id": "teaneck-nj", "visibility": "public", "url": "https://sidewalk-teaneck.cs.washington.edu"},
    ...     {"city_id": "seattle-wa", "visibility": "public", "url": " https://sidewalk-sea.cs.washington.edu/ "},
    ...     {"city_id": "example-private", "visibility": "private", "url": None}]})
    {'seattle-wa': 'sea', 'teaneck-nj': 'teaneck'}
    """
    out, skipped, odd = {}, [], []
    for c in listing_entries(listing):
        cid, vis = c["city_id"], c.get("visibility")
        if vis not in ("public", "private"):
            odd.append(f"{cid} ({vis!r})")
        if public_only and vis != "public":
            continue
        url = _listing_url(c)
        if url:
            out[cid] = subdomain_from_url(url)
        elif cid in HOST_OVERRIDES:
            out[cid] = HOST_OVERRIDES[cid]
        elif public_only:
            skipped.append(cid)
    if odd:
        print(f"WARNING: /v3/api/cities gives an unknown visibility to {', '.join(odd)}; "
              f"{'excluded from the default list' if public_only else 'not used for the default list'}.")
    if skipped:
        print(f"WARNING: /v3/api/cities lists public deployments without a url, skipped (add them to HOST_OVERRIDES "
              f"to include them): {', '.join(sorted(skipped))}")
    return dict(sorted(out.items()))


def _get_json(url):
    import requests
    response = requests.get(url, timeout=(10, 60))
    response.raise_for_status()
    return response.json()


def fetch_city_listing(url=CITIES_API, fetch=None):
    """The ``/v3/api/cities`` response, checked, with one retry; raises :class:`ListingError`."""
    for attempt in (1, 2):
        try:
            response = (fetch or _get_json)(url)
            break
        except Exception as e:  # requests errors, a non-JSON body, ...
            if attempt == 2:
                raise ListingError(f"could not fetch {url} ({type(e).__name__}: {e})") from e
            print(f"{url} failed ({e}); retrying once")
            time.sleep(LISTING_RETRY_S)
    if not isinstance(response, dict):
        raise ListingError(f"{url} did not return a JSON object (got {type(response).__name__})")
    listing_entries(response)
    return response


def cache_has_csvs(cache_dir):
    """True when ``cache_dir`` holds any ``*.csv`` (saved rawLabels exports)."""
    return bool(cache_dir) and os.path.isdir(cache_dir) and any(n.endswith(".csv") for n in os.listdir(cache_dir))


def load_city_listing(cache_dir=None, refresh=False, url=CITIES_API, fetch=None):
    """The ``/v3/api/cities`` listing and where it came from: ``(listing, source)``; raises :class:`ListingError`.

    With ``cache_dir``, ``<cache_dir>/cities.json`` is read when it exists (no network) unless ``refresh``. Otherwise
    the API is fetched (:func:`fetch_city_listing`) and, with ``cache_dir``, written there atomically as
    ``{"fetched_on", "url", "cities"}``, but only after it is checked, and never into a cache that already holds
    rawLabels CSVs without a listing (a snapshot from before city discovery) unless ``refresh``: that would pair the
    old snapshot with a new list. ``source`` is ``{"from": "cache" | "api", "url", "fetched_on"}``; ``url`` and
    ``fetched_on`` are None when a cache file does not record them (an old fetch_all.py list).
    """
    path = os.path.join(cache_dir, CITIES_CACHE_FILE) if cache_dir else None
    if path and os.path.exists(path) and not refresh:
        try:
            with open(path) as f:
                listing = json.load(f)
        except (OSError, ValueError) as e:
            raise ListingError(f"{path} is not readable JSON ({e}); --refresh-cities rewrites it") from e
        try:
            listing_entries(listing)
        except ListingError as e:
            raise ListingError(f"{path}: {e}; --refresh-cities rewrites it") from e
        meta = listing if isinstance(listing, dict) else {}
        return listing, {"from": "cache", "url": meta.get("url"), "fetched_on": meta.get("fetched_on")}
    if path and not refresh and cache_has_csvs(cache_dir):
        raise ListingError(
            f"--rawlabels-cache {cache_dir} holds rawLabels CSVs but no {CITIES_CACHE_FILE}: a snapshot from before "
            f"city discovery (issue #8). Writing today's listing into it would pair that snapshot with a different "
            f"city list. Use --cities to run on its cities, or --refresh-cities to write the current listing into it "
            f"(every listed city then needs its <city_id>.csv: re-pull with FRESH=1 sbatch data_generation.slurm or "
            f"analysis/fetch_all.py).")
    response = fetch_city_listing(url, fetch)
    listing = {"fetched_on": date.today().isoformat(), "url": url, "cities": response["cities"]}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(listing, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
    return listing, {"from": "api", "url": url, "fetched_on": listing["fetched_on"]}


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


def parse_experts(value):
    """Expert user_ids from an ``EXPERTS`` value, ``"<name>=<user_id>,..."``, as a frozenset.

    Parsed as ``analysis/analyze.py`` parses it, so one value means the same experts everywhere: an entry without
    ``=`` is an error here (analyze.py would silently skip it).

    >>> sorted(parse_experts("a=u1, b=u2 ,,"))
    ['u1', 'u2']
    >>> parse_experts(None)
    frozenset()
    >>> parse_experts("u1")
    Traceback (most recent call last):
    ...
    ValueError: EXPERTS entries must be <name>=<user_id>; got 1 without '='
    """
    entries = [kv.strip() for kv in (value or "").split(",") if kv.strip()]
    bare = [kv for kv in entries if "=" not in kv]
    if bare:  # the count only: the value is a user_id and does not belong in a log
        raise ValueError(f"EXPERTS entries must be <name>=<user_id>; got {len(bare)} without '='")
    ids = (kv.split("=", 1)[1].strip() for kv in entries)
    return frozenset(i for i in ids if i)


def expert_verdict(validations_json, experts):
    """``"correct"``/``"incorrect"`` if expert validators voted on this label, all the same way, and no other human
    voted the other way; else ``""``.

    The crowd margin rule (:func:`eligible_labels`) needs three net Agree votes, so it ignores a label one expert
    checked and nobody else did. Checked on labels the experts also voted on (2026-10-19, all cities), a single
    non-expert Disagree was right only 24-78% of the time depending on the type, so an expert vote is the stronger
    signal; a contradicting human vote still sends the label back to the margin rule. Only ``validator_type ==
    "Human"`` votes count, as in :func:`human_vote_counts`; Unsure and other values are ignored.

    >>> v = lambda *vs: json.dumps([{"user_id": u, "validation": x, "validator_type": "Human"} for u, x in vs])
    >>> expert_verdict(v(("e", "Disagree")), {"e"}), expert_verdict(v(("e", "Agree"), ("c", "Agree")), {"e"})
    ('incorrect', 'correct')
    >>> expert_verdict(v(("e", "Agree"), ("c", "Disagree")), {"e"}), expert_verdict(v(("c", "Agree")), {"e"})
    ('', '')
    >>> expert_verdict(v(("e", "Agree"), ("f", "Disagree")), {"e", "f"}), expert_verdict(v(("e", "Agree")), set())
    ('', '')
    """
    if not experts or not isinstance(validations_json, str) or not validations_json:
        return ""
    votes = [v for v in json.loads(validations_json)
             if v.get("validator_type") == "Human" and v.get("validation") in ("Agree", "Disagree")]
    expert_votes = {v["validation"] for v in votes if str(v.get("user_id")) in experts}
    if len(expert_votes) != 1:
        return ""
    vote = expert_votes.pop()
    if any(v["validation"] != vote for v in votes):
        return ""
    return "correct" if vote == "Agree" else "incorrect"


def label_author(user_id):
    """``"ai"`` for a label placed by the SidewalkAI account, else ``"human"``.

    >>> label_author("51b0b927-3c8a-45b2-93de-bd878d1e5cf4"), label_author("someone-else")
    ('ai', 'human')
    """
    return "ai" if str(user_id) == AI_USER_ID else "human"


def valid_geometry(df):
    """Boolean mask of rows whose stored pano point is usable, and the reason counts for the rest.

    Drops (counted by reason): non-GSV providers (``non_gsv``: the pano store holds Mapillary and Panoramax
    panoramas, but their pose convention differs from GSV's and no crop path handles it yet; docs/decisions.md,
    issue #9 item 4), non-finite or
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


def eligible_labels(all_labels, label_type, experts=frozenset()):
    """Labels of one type with a usable pano point that meet the vote thresholds: ``(eligible_df, dropped)``.

    Adds ``human_agree``, ``human_disagree``, ``author`` (``human``/``ai``), ``label_class`` and ``class_source``;
    no downsampling.
    AI-authored labels are KEPT here (tagged ``author = ai``): the prior counts only human-authored ones, and the
    selection drops AI ones unless asked. ``dropped`` counts rows removed by :func:`valid_geometry`.

    Rule: ``incorrect`` if net human disagree votes > 1, ``correct`` if net human agree votes > 2 (``class_source =
    margin``, unchanged since the first models); otherwise the :func:`expert_verdict` of ``experts``, if any
    (``class_source = expert``); everything else (unvalidated, unsure, contested) is left out. The two rules cannot
    disagree: an expert verdict requires that no human voted the other way, so the opposite margin is impossible.
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
    df['class_source'] = np.where(df['label_class'] != '', 'margin', '')
    if experts:
        verdict = df['validations'].apply(expert_verdict, experts=experts)
        by_expert = (df['label_class'] == '') & (verdict != '')
        df.loc[by_expert, 'label_class'] = verdict[by_expert]
        df.loc[by_expert, 'class_source'] = 'expert'
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
    and a resumed run keeps nearly all of its crops. A whole group entering at once (another city, or AI labels with
    ``include_ai_labels``) can displace as many kept labels as it brings ranks below the cutoff. The final order is
    still a seeded shuffle.
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


def deployment_prior_entry(eligible, cities, n_rows, measured_on=None, city_source=None):
    """The pre-downsampling class counts of human-authored :func:`eligible_labels`: the class mix the validator meets
    in production (which never sends it AI-authored labels), for ``calibration/calibrate.py``. ``n_rows`` is the
    number of rawLabels rows of this label type the counts were drawn from; ``city_source``, when given, says where
    the city list came from (the ``source`` of :func:`load_city_listing`, or ``{"from": "--cities"}``).

    Only ``class_source == "margin"`` labels count: the labels experts vote on are not a sample of production (they
    check what looks wrong), so adding them would move the prior with who happened to validate, and an ``EXPERTS``
    setting would change calibration as well as the training set."""
    human = eligible[(eligible['author'] == 'human') & (eligible.get('class_source', 'margin') == 'margin')]
    counts = human['label_class'].value_counts()
    entry = {"correct": int(counts.get('correct', 0)), "incorrect": int(counts.get('incorrect', 0)),
             "measured_on": measured_on or date.today().isoformat(),
             "source": f"rawLabels of {', '.join(cities)}, human votes only",
             "cities": list(cities), "n_rows": int(n_rows)}
    if city_source is not None:
        entry["city_source"] = dict(city_source)
    return entry


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
    ``selection`` (``{label_uid: label_class}``, the CURRENT full selection) with the same class, and it has a
    ``pano_path`` (an ``ok`` row without one was made by the removed tile fetcher, not from the store). Every other
    row is dropped: a label still selected is attempted again and gets exactly one fresh row, and a label no longer
    selected leaves the manifest (its crop file stays on disk, unused). ``self.stale`` counts the dropped ``ok`` rows
    by reason. The kept rows are rewritten atomically before any append, so ``label_uid`` stays unique in the file at
    every moment.

    Example::

        with ManifestWriter("CurbRamp/manifest.csv", "CurbRamp", {"seattle-wa:1": "correct"}) as m:
            if not m.is_done("seattle-wa:1"):
                m.write({...})
    """

    def __init__(self, path, base_dir, selection, keep_ok=True, sources=None):
        self.path = path
        self.base_dir = base_dir
        self.done = set()
        self.stale = Counter()
        self.previously_missing_panos = set()  # (city, pano_id) whose last row was a missing status
        kept = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            old = old.drop_duplicates('label_uid', keep='last')
            for row in old.to_dict('records'):
                if row.get('status') in MISSING_STATUSES + LEGACY_MISSING_STATUSES:
                    self.previously_missing_panos.add((row.get('city'), row.get('pano_id')))
                if row.get('status') != 'ok' or not os.path.exists(os.path.join(base_dir, row.get('crop_path', ''))):
                    continue
                uid = row['label_uid']
                if not keep_ok:
                    self.stale['settings_changed'] += 1
                elif uid not in selection:
                    self.stale['no_longer_selected'] += 1
                elif selection[uid] != row.get('label_class'):
                    self.stale['class_changed'] += 1
                elif not row.get('pano_path'):
                    self.stale['not_from_store'] += 1
                else:
                    if sources is not None:  # the rule that decides a label can change with new votes
                        row['class_source'] = sources.get(uid, '')
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
        "class_source": row.get('class_source'),
        "author": row['author'], "human_agree": row['human_agree'], "human_disagree": row['human_disagree'],
        "severity": row.get('severity'), "tags": row.get('tags'), "high_quality_user": row.get('high_quality_user'),
        "image_capture_date": row.get('image_capture_date'), "pano_source": row.get('pano_source'),
        "pano_path": None, "pano_native_w": None, "pano_native_h": None,
        "camera_pitch": row.get('camera_pitch'), "camera_roll": row.get('camera_roll'),
        "tilt_applied": 0, "tilt_beta": tilt_beta, "depth_m": None, "crop_half_px": None, "clipped": None,
        "crop_path": crop_path, "status": "",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Panorama
# ---------------------------------------------------------------------------------------------------------------------

def pano_store_path(store, city, pano_id):
    """Where the pano store keeps a panorama: ``<store>/<city>/<pano_id[:2]>/<pano_id>.jpg``.

    The layout sidewalk-ai-api reads (``SIDEWALK_SCRAPES_DIR``), keyed by the city_id, so one
    store serves both. Nothing in it is provider-specific.

    >>> pano_store_path("/store", "seattle-wa", "abcDEF")
    '/store/seattle-wa/ab/abcDEF.jpg'
    """
    return os.path.join(store, city, pano_id[:2], f"{pano_id}.jpg")


class PanoStoreUnavailable(RuntimeError):
    """The store itself stopped answering (an I/O error other than a missing file, or its marker or a city directory
    vanished mid-run): fatal, never a ``no_pano`` row, since a dropped mount would otherwise shrink the dataset
    silently under ``--max-missing-frac``."""


def _require_store(store, city):
    """Raise :class:`PanoStoreUnavailable` unless the marker (a file) and ``city``'s directory are still there."""
    for what, path, is_kind in (("marker", os.path.join(store, PANO_STORE_MARKER), stat.S_ISREG),
                                ("city directory", os.path.join(store, city), stat.S_ISDIR)):
        try:
            mode = os.stat(path).st_mode
        except OSError as e:
            raise PanoStoreUnavailable(f"pano store {what} {path} is gone ({e}); is the store still mounted?") from e
        if not is_kind(mode):
            raise PanoStoreUnavailable(f"pano store {what} {path} is the wrong kind of file; is the store intact?")


def read_store_panorama(store, city, pano_id):
    """The panorama from the store: ``(result, info)``.

    ``result`` is an 8192x4096 BGR array (``PANO_SIZE``), or the status string ``no_pano`` (no file) or ``bad_pano``
    (``cv2.imread`` cannot decode it, or it is not 2:1, ``|w / h - 2| >= 0.01``, checked before resizing so a truncated
    or cropped file is never stretched into a crop). ``info`` holds the manifest fields ``pano_path`` (None for
    ``no_pano``), ``pano_native_w`` and ``pano_native_h`` (None unless decoded).

    The read and the resize are the API's cache path (``cv2.imread`` + ``cv2.resize(..., INTER_LINEAR)``); identical
    bytes also need the same opencv build. Only ``FileNotFoundError`` is ``no_pano``, and only once the store's marker
    and city directory are confirmed present; any other ``OSError`` raises :class:`PanoStoreUnavailable`.
    """
    import cv2
    path = pano_store_path(store, city, pano_id)
    info = {"pano_path": path, "pano_native_w": None, "pano_native_h": None}
    try:
        os.stat(path)
    except FileNotFoundError:
        _require_store(store, city)
        return "no_pano", {**info, "pano_path": None}
    except OSError as e:
        raise PanoStoreUnavailable(f"pano store read failed for {path}: {e}") from e
    img = cv2.imread(path)
    if img is None:
        _require_store(store, city)  # a mount that dropped between the stat and the read is an outage, not a bad file
        print(f"pano {city}/{pano_id}: {path} is unreadable")
        return "bad_pano", info
    h, w = img.shape[:2]
    info.update(pano_native_w=w, pano_native_h=h)
    if abs(w / h - 2) >= 0.01:
        print(f"pano {city}/{pano_id}: {path} is {w}x{h}, not 2:1")
        return "bad_pano", info
    return cv2.resize(img, PANO_SIZE, interpolation=cv2.INTER_LINEAR), info


def check_pano_store_marker(store):
    """Exit unless ``store`` is a directory with a ``.pano-store`` marker file (any contents; the store or a verified
    staged copy writes it); return the marker's first line. Run before any download, so a mistyped path or an
    unmounted store fails in seconds."""
    if not os.path.isdir(store):
        sys.exit(f"--pano-store {store} is not a directory.")
    marker = os.path.join(store, PANO_STORE_MARKER)
    if not os.path.isfile(marker):
        sys.exit(f"--pano-store {store} has no {PANO_STORE_MARKER} marker file: it is not a pano store, or not a "
                 f"verified staged copy of one (or the mount is not up).")
    with open(marker, errors="replace") as f:
        return f.readline().strip()


def check_pano_store(store, cities, allow_missing_cities=False):
    """Check the store for the cities that have selected labels: ``(marker_first_line, absent_cities)``.

    Exits unless the marker is there (:func:`check_pano_store_marker`) and there is a directory for every one of
    ``cities``. With ``allow_missing_cities``, absent cities only warn (their labels are ``no_pano``), but at least
    one must be present. Run after selection and before the depth model loads, so a store laid out by other city
    names fails in seconds instead of after 50 ``no_pano`` rows, and so a city with no directory cannot hide its
    ``no_pano`` rows among other cities' successes. Listed cities with no selected label need no directory."""
    marker_line = check_pano_store_marker(store)
    present = [c for c in cities if os.path.isdir(os.path.join(store, c))]
    absent = [c for c in cities if c not in present]
    layout = "the store is laid out <store>/<city_id>/<pano_id[:2]>/<pano_id>.jpg"
    if not present:
        sys.exit(f"--pano-store {store} has no directory for any selected city ({', '.join(cities)}); {layout}.")
    if absent and not allow_missing_cities:
        sys.exit(f"--pano-store {store} has no directory for {', '.join(absent)}; {layout}. Pass "
                 f"--allow-missing-cities to run anyway (their labels will all be no_pano).")
    if absent:
        print(f"WARNING: --allow-missing-cities: no pano-store directory for {', '.join(absent)}; their labels will "
              f"all be no_pano.")
    return marker_line, absent


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
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError(f"bad depth {depth_m}")
    return centre_crop(view_img, int(CROP_DEPTH_CONSTANT / depth_m))


def centre_crop(view_img, crop_half):
    """Cut a square of half-width ``crop_half`` px around the view centre, clipped to the frame, and resize its long
    side to ``CROP_LONG_SIDE``: ``(crop, crop_half, clipped)``, as :func:`depth_sized_crop` returns.

    >>> crop, half, clipped = centre_crop(np.zeros((2048, 2048, 3), np.uint8), 300)
    >>> crop.shape, half, clipped
    ((640, 640, 3), 300, False)
    """
    import cv2
    img_h, img_w = view_img.shape[:2]
    center_x, center_y = img_w // 2, img_h // 2
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
                            apply_tilt=True, beta=DEFAULT_TILT_BETA, device=None, fixed_crop_half_px=None):
    """Render the label-centred view and cut a crop around the centre, sized by depth unless ``fixed_crop_half_px``.

    Returns ``(crop_rgb, info)`` with ``info`` = ``tilt_applied``, ``depth_m``, ``crop_half_px``, ``clipped``.
    Depth is inferred on the BGR view (``infer_image`` takes an OpenCV BGR image, as sidewalk-ai-api passes it); the
    Depth Anything code picks its own device (CUDA when present). With ``fixed_crop_half_px`` every crop has that
    half-width and no depth is inferred (``depth_m`` is None): the fixed-size control of issue #21, which serving
    does not implement.
    """
    view_rgb, tilt_applied = render_label_view(pano_rgb, label_x_norm, label_y_norm, camera_pitch, camera_roll,
                                               apply_tilt=apply_tilt, beta=beta, device=device)
    if fixed_crop_half_px is not None:
        crop, crop_half, clipped = centre_crop(view_rgb, fixed_crop_half_px)
        return crop, {"tilt_applied": int(tilt_applied), "depth_m": None, "crop_half_px": crop_half,
                      "clipped": int(clipped)}
    depth = depth_model.infer_image(np.ascontiguousarray(view_rgb[..., ::-1]))
    c = VIEW_SIZE // 2
    depth_m = float(depth[c, c])
    crop, crop_half, clipped = depth_sized_crop(view_rgb, depth_m)
    return crop, {"tilt_applied": int(tilt_applied), "depth_m": round(depth_m, 4), "crop_half_px": crop_half,
                  "clipped": int(clipped)}


#: Settings that change what a crop looks like; a resume with different values would mix two datasets.
#: ``include_ai_labels`` is not one: it changes which labels are selected, not how any crop is cut (a label's author
#: already decides its tilt), so flipping it is a selection change. That still needs ``--force-reselect``, and keeps
#: every human-authored crop that is still selected; AI correct labels compete in the correct-class downsample, so
#: some human correct labels near the cutoff can be swapped out. It stays in ``generation_run.json`` as provenance.
RESUME_SETTINGS = ("tilt_beta", "depth_encoder", "depth_dataset", "renderer", "package_source")
#: Settings recorded only when they are not the default, so earlier runs' records stay as they were. Unlike
#: :data:`RESUME_SETTINGS` they are compared when present on EITHER side, so switching to or from one is caught.
OPTIONAL_SETTINGS = ("fixed_crop_half_px",)
#: Settings that only matter when crops are sized by depth. A fixed-size run does not record them (no depth model is
#: loaded), and :func:`check_resume` does not compare them when both runs are fixed-size, so ``--depth-encoder``
#: cannot block a fixed-size resume (also one recorded before they were left out).
DEPTH_SETTINGS = ("depth_encoder", "depth_dataset")
#: The largest ``--fixed-crop-half-px``: a half-width of ``VIEW_SIZE // 2`` is exactly the whole view, unclipped, and
#: every larger value clips to that same crop.
MAX_FIXED_CROP_HALF_PX = VIEW_SIZE // 2


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


def check_resume(run_path, new_hash, settings, cities=None):
    """Compare a resumed run with the ``generation_run.json`` of the previous one.

    Returns ``(old_hash or None, changed_settings, city_change)``: the previous selection hash if it differs from
    ``new_hash``; ``{name: (old, new)}`` for every setting that differs; and, when ``cities`` is given and the
    recorded city list is a different set, ``{"added": [...], "removed": [...]}`` (else ``{}``). A
    :data:`RESUME_SETTINGS` entry is compared only if the old record has it (a record older than the setting resumes),
    except that :data:`DEPTH_SETTINGS` are skipped when both runs have ``fixed_crop_half_px``; an
    :data:`OPTIONAL_SETTINGS` entry is compared whenever either side has it. The caller refuses to resume on any of
    them unless ``--force-reselect``. The city list is compared on its own because the selection hash misses a city
    that contributes no selected label; like a selection change (and unlike a settings change) it keeps the crops of
    labels still selected.

    >>> check_resume("/nonexistent/generation_run.json", "abc", {"tilt_beta": 0.95})
    (None, {}, {})
    """
    if not os.path.exists(run_path):
        return None, {}, {}
    with open(run_path) as f:
        old = json.load(f)
    old_hash = old.get("selection_sha256")
    old_settings = old.get("settings", {})
    both_fixed = old_settings.get("fixed_crop_half_px") is not None and settings.get("fixed_crop_half_px") is not None
    compared = [k for k in RESUME_SETTINGS if not (both_fixed and k in DEPTH_SETTINGS)]
    changed = {k: (old_settings.get(k), settings.get(k)) for k in compared
               if k in old_settings and old_settings.get(k) != settings.get(k)}
    changed.update({k: (old_settings.get(k), settings.get(k)) for k in OPTIONAL_SETTINGS
                    if old_settings.get(k) != settings.get(k)})
    city_change = {}
    if cities is not None and "cities" in old and set(old["cities"]) != set(cities):
        city_change = {"added": sorted(set(cities) - set(old["cities"])),
                       "removed": sorted(set(old["cities"]) - set(cities))}
    return (old_hash if old_hash and old_hash != new_hash else None), changed, city_change


def recorded_cities(run_path):
    """The city list ``generation_run.json`` at ``run_path`` records, or None (no file, or no list)."""
    if not os.path.exists(run_path):
        return None
    with open(run_path) as f:
        return json.load(f).get("cities")


def count_ok_rows(manifest_path):
    """Number of ``ok`` rows in a manifest (0 if there is none)."""
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return 0
    return int((pd.read_csv(manifest_path, dtype=str, usecols=["status"])["status"] == "ok").sum())


def is_fatal(exc):
    """Errors that must stop the run instead of becoming one ``error:*`` row: broken invariants, GPU failures and a
    pano store that stopped answering."""
    if isinstance(exc, (AssertionError, PanoStoreUnavailable)) or type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "CUDA" in str(exc)


MAX_CONSECUTIVE_ERRORS = 50
MAX_CONSECUTIVE_MISSING_PANOS = 50  # panoramas in a row missing from the store or unusable (MISSING_STATUSES)
MISSING_STATUSES = ("no_pano", "bad_pano")
# Written by the removed tile fetcher. Such rows are never ok, so a resume drops and retries them, and they cannot reach
# the final count; they are only read so a pano that failed under the old fetcher is not counted as a new outage.
LEGACY_MISSING_STATUSES = ("incomplete_pano",)


def parse_cities(arg, known=None):
    """``--cities a,b`` -> ``{city_id: subdomain}``; entries may be ``city_id`` or ``city_id=subdomain``. A bare
    city_id's host is looked up in ``known`` (the deployments :func:`discover_cities` found), then in
    :data:`HOST_OVERRIDES`, else it is the city_id itself. Without ``arg``, ``known`` itself (the default list).

    >>> parse_cities("seattle-wa,vancouver-wa=vancouver,x", {"seattle-wa": "sea"})
    {'seattle-wa': 'sea', 'vancouver-wa': 'vancouver', 'x': 'x'}
    """
    known = known or {}
    if not arg:
        return dict(known)
    out = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        cid, _, sub = item.partition("=")
        out[cid] = sub or known.get(cid) or HOST_OVERRIDES.get(cid, cid)
    return out


def resolve_cities(arg, cache_dir=None, refresh=False, fetch=None):
    """The cities to pull and where the list came from: ``(cities, city_source)``.

    Without ``arg`` (no ``--cities``, or a blank one): the public deployments of :func:`load_city_listing`; a listing
    that cannot be had stops the run. With it: :func:`parse_cities`, ``city_source`` ``{"from": "--cities"}``. The
    listing is consulted only for a host, i.e. when some bare city_id has neither a saved ``<cache_dir>/<city_id>.csv``
    nor a :data:`HOST_OVERRIDES` entry (so an offline smoke test stays offline); it is read from the cache or
    fetched, never written, and if it cannot be had the city_id is used as the subdomain, with a warning.
    """
    if not arg or not any(i.strip() for i in arg.split(",")):
        try:
            listing, source = load_city_listing(cache_dir, refresh, fetch=fetch)
            cities = discover_cities(listing)
        except ListingError as e:
            sys.exit(f"City list: {e}")
        if not cities:
            sys.exit(f"/v3/api/cities ({source}) lists no public deployment with a url.")
        return cities, source
    bare = [i.strip() for i in arg.split(",") if i.strip() and "=" not in i]
    needs_host = [c for c in bare if c not in HOST_OVERRIDES
                  and not (cache_dir and os.path.exists(os.path.join(cache_dir, f"{c}.csv")))]
    known = {}
    if needs_host:
        path = os.path.join(cache_dir, CITIES_CACHE_FILE) if cache_dir else None
        try:
            if path and os.path.exists(path) and not refresh:
                listing = load_city_listing(cache_dir, fetch=fetch)[0]
            else:
                listing = fetch_city_listing(fetch=fetch)
            known = discover_cities(listing, public_only=False)
        except ListingError as e:
            print(f"WARNING: {e}; using the city_id as the subdomain for {', '.join(needs_host)}.")
    return parse_cities(arg, known), {"from": "--cities"}


def missing_cached_csvs(cities, cache_dir):
    """The cities with no ``<cache_dir>/<city_id>.csv``."""
    return [c for c in cities if not os.path.exists(os.path.join(cache_dir, f"{c}.csv"))]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument('--label_type', required=True, help='Specify the label type')
    p.add_argument('--out', default='.', help='output root; crops and manifest go under <out>/<label_type>/')
    p.add_argument('--pano-store', default=None,
                   help='root of the pano store or of a staged copy with the same layout, read as '
                        '<DIR>/<city_id>/<pano_id[:2]>/<pano_id>.jpg; must hold a .pano-store marker file; required '
                        'unless --select-only')
    p.add_argument('--allow-missing-cities', action='store_true',
                   help='run although some cities with selected labels have no pano-store directory (their labels '
                        'are no_pano)')
    p.add_argument('--limit', type=int, default=None, help='process only the first N selected labels (smoke tests)')
    p.add_argument('--manifest', default=None, help='default <out>/<label_type>/manifest.csv')
    p.add_argument('--priors', default=None,
                   help='shared deployment_priors.json; default <out>/deployment_priors.json. Partial runs (--cities '
                        'or --limit) write deployment_priors.partial.json beside it instead')
    p.add_argument('--depth-encoder', choices=['vitb', 'vitl'], default='vitb', help='the API serves vitb')
    p.add_argument('--tilt-beta', type=float, default=DEFAULT_TILT_BETA)
    p.add_argument('--fixed-crop-half-px', type=int, default=None,
                   help='ablation only (issue #21): give every crop this half-width in view pixels instead of sizing '
                        'it by depth, and skip depth inference (--depth-encoder is then ignored and not recorded). '
                        f'Must be 1 to {MAX_FIXED_CROP_HALF_PX}: the view is {VIEW_SIZE} px square, so a half-width of '
                        f'{MAX_FIXED_CROP_HALF_PX} is the whole view and larger values clip to it. Serving sizes by depth, '
                        'so publish_to_hf.py refuses models trained this way')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--cities', default=None,
                   help='comma list of city_id[=subdomain], overriding the public deployments /v3/api/cities lists')
    p.add_argument('--include-ai-labels', action='store_true',
                   help='keep AI-authored labels in the selection (rendered without tilt correction)')
    p.add_argument('--rawlabels-cache', default=None,
                   help='directory of saved rawLabels exports <city_id>.csv, read instead of the API when present; '
                        'also holds the /v3/api/cities listing as cities.json (read if present, else fetched and '
                        'written, unless the directory already holds CSVs); without --cities every listed city needs '
                        'its CSV here')
    p.add_argument('--refresh-cities', action='store_true',
                   help='fetch the /v3/api/cities listing again; with --rawlabels-cache and no --cities, rewrite '
                        '<cache>/cities.json (also into a cache made before city discovery)')
    p.add_argument('--select-only', action='store_true',
                   help='select labels and write the priors, then stop (no GPU, no panoramas)')
    p.add_argument('--allow-cpu', action='store_true', help=DEPTH_DEVICE_HELP)
    p.add_argument('--max-missing-frac', type=float, default=0.2,
                   help='exit non-zero if more than this fraction of attempted labels ended no_pano/bad_pano')
    p.add_argument('--force-reselect', action='store_true',
                   help='resume although the selection changed since the last run (stale manifest rows are dropped)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pano_store:  # once, so every pano_path in the manifest and the run record is absolute
        args.pano_store = os.path.abspath(args.pano_store)
    label_type = args.label_type
    base_dir = os.path.join(args.out, label_type)
    manifest_path = args.manifest or os.path.join(base_dir, "manifest.csv")
    run_path = os.path.join(base_dir, "generation_run.json")
    priors_path = args.priors or os.path.join(args.out, "deployment_priors.json")
    partial = (args.cities is not None and args.cities.strip(", ") != "") or args.limit is not None
    if partial:
        priors_path = os.path.splitext(priors_path)[0] + ".partial.json"
    if args.fixed_crop_half_px is not None and not 0 < args.fixed_crop_half_px <= MAX_FIXED_CROP_HALF_PX:
        sys.exit(f"--fixed-crop-half-px must be 1 to {MAX_FIXED_CROP_HALF_PX} (half the {VIEW_SIZE} px view; larger "
                 f"values clip to the whole view), got {args.fixed_crop_half_px}")
    experts = parse_experts(os.environ.get(EXPERTS_ENV))  # a malformed value fails before the downloads
    if not args.select_only and not args.pano_store:  # before the city listing, which may need the network
        build_parser().error("--pano-store is required unless --select-only")
    if not args.select_only:  # fail fast, before the downloads; the city directories are checked after selection
        check_pano_store_marker(args.pano_store)
    default_cities = not args.cities or not any(i.strip() for i in args.cities.split(","))
    cities, city_source = resolve_cities(args.cities, args.rawlabels_cache, args.refresh_cities)
    print(f"{len(cities)} cities (from {city_source['from']}"
          f"{', fetched ' + city_source['fetched_on'] if city_source.get('fetched_on') else ''}): "
          f"{', '.join(cities)}")
    if default_cities and args.rawlabels_cache:
        missing = missing_cached_csvs(cities, args.rawlabels_cache)
        if missing:  # never a live download into a snapshot: the five label types would see different votes
            sys.exit(f"--rawlabels-cache {args.rawlabels_cache} has no <city_id>.csv for {len(missing)} listed "
                     f"cities: {', '.join(missing)}. Pull them into the cache (FRESH=1 sbatch data_generation.slurm, "
                     f"or analysis/fetch_all.py), or pass --cities to run without them.")
    device = None if args.select_only else resolve_device(args.allow_cpu)

    all_cities = pd.concat([load_city(c, s, args.rawlabels_cache) for c, s in cities.items()], ignore_index=True)
    n_rows = int((all_cities['label_type'] == label_type).sum())

    eligible, dropped = eligible_labels(all_cities, label_type, experts)  # once, for both the prior and the selection
    print(f"{n_rows} {label_type} rows; dropped for unusable pano geometry: {dict(dropped) or 'none'}")
    if experts:
        by_expert = eligible.loc[eligible['class_source'] == 'expert']
        print(f"Expert rule ({len(experts)} accounts from ${EXPERTS_ENV}) added, before downsampling: "
              + ", ".join(f"{a} {c} {n}" for (a, c), n in by_expert.groupby(['author', 'label_class']).size().items())
              if len(by_expert) else f"Expert rule ({len(experts)} accounts) added no labels")
    else:
        print(f"${EXPERTS_ENV} is not set: crowd margin rule only (no expert votes)")
    entry = deployment_prior_entry(eligible, list(cities), n_rows, city_source=city_source)

    def write_priors():
        update_deployment_priors(priors_path, label_type, entry)
        print(f"Deployment prior ({priors_path}{', PARTIAL run' if partial else ''}): "
              f"{entry['correct']} correct, {entry['incorrect']} incorrect")

    combined_filtered = select_training_labels(eligible, seed=args.seed, include_ai_labels=args.include_ai_labels)
    combined_filtered['label_uid'] = combined_filtered['city'] + ":" + combined_filtered['label_id'].astype(str)
    assert combined_filtered['label_uid'].is_unique, "label_uid (city:label_id) is not unique in the selection"
    selection = dict(zip(combined_filtered['label_uid'], combined_filtered['label_class']))  # before --limit
    selection_cities = combined_filtered['city'].tolist()
    sel_hash = selection_hash(combined_filtered)
    source_counts = combined_filtered['class_source'].value_counts().to_dict()
    sources = dict(zip(combined_filtered['label_uid'], combined_filtered['class_source']))  # both before --limit
    if args.limit is not None:
        combined_filtered = combined_filtered.head(args.limit)
    if args.select_only:
        recorded = recorded_cities(run_path)
        if not partial and recorded is not None and set(recorded) != set(cities):
            # The real priors must describe the crops in this output dir: a prior over another city list next to
            # them would be read by fine_tune/evaluate/calibrate without a comparison.
            priors_path = os.path.splitext(priors_path)[0] + ".partial.json"
            print(f"--select-only: {run_path} records a different city list (added: "
                  f"{', '.join(sorted(set(cities) - set(recorded))) or 'none'}; removed: "
                  f"{', '.join(sorted(set(recorded) - set(cities))) or 'none'}), so the prior goes to "
                  f"{priors_path}, not the real deployment_priors.json.")
        update_deployment_priors(priors_path, label_type, entry)
        print(f"Deployment prior ({priors_path}{', PARTIAL run' if partial else ''}): "
              f"{entry['correct']} correct, {entry['incorrect']} incorrect")
        print(f"--select-only: {len(combined_filtered)} labels selected; stopping before rendering.")
        return

    # Directories only for cities with a selected label (before --limit, so a limit run checks the same cities).
    selected_cities = sorted(set(selection_cities))
    store_marker, absent_cities = check_pano_store(args.pano_store, selected_cities, args.allow_missing_cities)
    unselected_without_dir = [c for c in cities if c not in selected_cities
                              and not os.path.isdir(os.path.join(args.pano_store, c))]
    if unselected_without_dir:
        print(f"Note: no pano-store directory for {', '.join(unselected_without_dir)}; none of their labels is "
              f"selected, so none is needed.")

    settings = {"tilt_beta": args.tilt_beta, "depth_encoder": args.depth_encoder, "depth_dataset": DEPTH_DATASET,
                "renderer": "torch" if device else "numpy", "include_ai_labels": args.include_ai_labels,
                "package_source": package_source()}
    if args.fixed_crop_half_px is not None:  # no depth model is loaded: recording an encoder would misstate the crops
        settings = {k: v for k, v in settings.items() if k not in DEPTH_SETTINGS}
        settings["fixed_crop_half_px"] = args.fixed_crop_half_px
    previous, changed, city_change = check_resume(run_path, sel_hash, settings, list(cities))
    if (previous or changed or city_change) and not args.force_reselect:
        n_ok = count_ok_rows(manifest_path)
        why = []
        if previous:
            why.append(f"the selection changed (sha256 {previous[:12]} -> {sel_hash[:12]})")
        if city_change:
            why.append("the city list changed (added: " + (", ".join(city_change["added"]) or "none")
                       + "; removed: " + (", ".join(city_change["removed"]) or "none") + ")")
        if changed:
            why.append("settings changed: " + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changed.items()))
        # generation_run.json is left untouched, so the record still describes the crops on disk.
        sys.exit(f"Refusing to resume in {base_dir}: {'; '.join(why)}. The manifest holds {n_ok} ok rows made "
                 f"under the previous run. Re-run with --force-reselect to accept: stale ok rows are dropped "
                 f"(all of them if a setting changed) and redone; old crop files stay on disk until overwritten.")
    write_priors()  # only once the resume check has passed
    os.makedirs(base_dir, exist_ok=True)
    manifest = ManifestWriter(manifest_path, base_dir, selection, keep_ok=not changed, sources=sources)
    if previous or changed or city_change:
        print(f"WARNING: --force-reselect: selection changed: {bool(previous)}, city list changed: "
              f"{city_change or 'no'}, settings changed: {changed or 'none'}; "
              f"dropped ok rows {dict(manifest.stale) or 'none'}, kept {len(manifest.done)}.")
    not_from_store = manifest.stale.get('not_from_store', 0)
    if not_from_store:
        print(f"Redoing {not_from_store} ok rows made by the removed tile fetcher (no pano_path): their crops were "
              f"not cut from the pano store.")

    # pano_store is provenance, not a resume setting: it says where the pixels were read, not which labels or how
    # they were rendered, so moving or remounting the store must not invalidate every crop (see RESUME_SETTINGS).
    with open(run_path, "w") as f:  # read by fine_tune_classifier.py
        json.dump({"label_type": label_type, "settings": settings, **settings, "seed": args.seed,
                   "pano_store": args.pano_store, "pano_store_marker": store_marker,
                   "cities": list(cities), "city_source": city_source, "limit": args.limit, "selection_size": len(selection),
                   "n_experts": len(experts),
                   "class_source_counts": source_counts,
                   "selection_sha256": sel_hash, "started": date.today().isoformat()}, f, indent=2)
        f.write("\n")

    depth_model = load_depth_model(args.depth_encoder, device) if args.fixed_crop_half_px is None else None
    for cls in ("correct", "incorrect"):
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    # Process panorama by panorama (stable, so the order within a pano is the selection's) and reuse the last pano.
    combined_filtered = combined_filtered.sort_values('pano_id', kind='mergesort')
    total_images = len(combined_filtered)
    start_time = time.time()
    status_counts = Counter()
    last_pano = (None, None, None)  # ((city, pano_id), BGR array or the status string it failed with, info)
    consecutive_errors = consecutive_missing = 0
    with manifest:
        for processed_count, (_, row) in enumerate(combined_filtered.iterrows(), start=1):
            crop_path = f"{row['label_class']}/{row['city']}_{row['label_id']}.webp"
            rec = manifest_row(row, label_type, crop_path, args.tilt_beta)
            if manifest.is_done(rec['label_uid']):
                status_counts['skipped (already ok)'] += 1
                continue
            try:
                # Keyed on (city, pano_id): the store path includes the city, so the same pano_id in two cities is
                # two reads (the sort is by pano_id, so they can be adjacent).
                if last_pano[0] != (row['city'], row['pano_id']):
                    if row['city'] in absent_cities:  # --allow-missing-cities; not a store failure
                        read, pano_info = "no_pano", {"pano_path": None}
                    else:
                        read, pano_info = read_store_panorama(args.pano_store, row['city'], row['pano_id'])
                    last_pano = ((row['city'], row['pano_id']), read, pano_info)
                    if isinstance(last_pano[1], str):
                        # Only panoramas missing for the first time count: a resume re-fetches exactly the panos that
                        # failed before, back to back, and must not read that as an outage.
                        if (row['city'], row['pano_id']) not in manifest.previously_missing_panos:
                            consecutive_missing += 1
                        if consecutive_missing >= MAX_CONSECUTIVE_MISSING_PANOS:
                            sys.exit(f"Aborting: {consecutive_missing} panoramas in a row were missing or unusable "
                                     f"(last: {last_pano[1]}, {row['city']}/{row['pano_id']}). Check the --pano-store "
                                     f"path ({args.pano_store}) and that its city directory names match the city_ids "
                                     f"({pano_store_path(args.pano_store, row['city'], row['pano_id'])} was looked "
                                     f"for); the run is resumable.")
                    else:
                        consecutive_missing = 0
                pano_bgr = last_pano[1]
                rec.update(last_pano[2])
                if isinstance(pano_bgr, str):
                    rec['status'] = pano_bgr
                else:
                    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])
                    crop, info = create_perspective_crop(
                        depth_model, pano_rgb, row['pano_x'] / row['pano_width'], row['pano_y'] / row['pano_height'],
                        camera_pitch=row.get('camera_pitch'), camera_roll=row.get('camera_roll'),
                        apply_tilt=row['author'] == 'human', beta=args.tilt_beta, device=device,
                        fixed_crop_half_px=args.fixed_crop_half_px)
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
    # Cumulative over the whole manifest (this run and previous ones), so a resume that only retries the panos that
    # failed before is judged on the dataset, not on its own all-missing batch.
    attempted = len(final)
    missing = int(final['status'].isin(MISSING_STATUSES).sum())
    if not any(not k.startswith('skipped') for k in status_counts):
        print(f"Nothing to do: all {len(final)} attempted labels in {manifest_path} are already ok.")
    print("Processing complete.")
    print(f"  dropped for unusable pano geometry (before selection): {dict(dropped) or 'none'}")
    print(f"  status counts: {dict(status_counts)}; {n_ok} ok rows in {manifest_path}")
    if not_from_store:
        print(f"  redone because they predate the pano store (no pano_path): {not_from_store} ok rows")
    if attempted and missing / attempted > args.max_missing_frac:
        sys.exit(f"{missing} of {attempted} attempted labels in the manifest ({missing / attempted:.0%}) are "
                 f"no_pano/bad_pano, "
                 f"above --max-missing-frac {args.max_missing_frac}: the dataset would shrink silently.")
    if n_ok == 0:
        # e.g. a store without these cities' panos: fail the SLURM job instead of an empty dataset with exit 0
        sys.exit(f"No crops in {manifest_path}: every label failed (no_pano/bad_pano or error; check --pano-store "
                 f"{args.pano_store}).")


if __name__ == "__main__":
    main()
