"""Street-network / OSM prior for CurbRamp (driveway vs curb ramp), first measurement of issue #45.

A common *incorrect* CurbRamp label is a driveway apron: in a tight crop a driveway cut looks like a curb ramp. Real
curb ramps cluster at intersections and marked crossings; driveways sit mid-block. This script measures how much
that road-network context says about the human verdict, with no GPU and no retraining, before anyone touches the
serving path. It is a *signal*, never a gate: mid-block ramps exist.

Stages (each one writes to disk and can be re-run on its own):

``fetch-streets``
    ``GET <city url>/v3/api/streets?filetype=csv`` for every public city in ``cities.json`` into a dated cache
    directory, plus ``manifest.json`` (status, bytes, error text per city). Project Sidewalk splits street edges at
    intersections, so the edge endpoints shared by two or more edges are the intersection set, with no OSM pull.
``fetch-osm``
    ``highway=crossing`` nodes and ``highway=service`` ways (with their ``service`` tag) straight from the Overpass
    API, in 0.1-degree tiles that contain labels (padded ``--pad-m``), one request at a time, with exponential
    backoff and mirror fallback; each tile is cached so an interrupted pull resumes. A city whose pull fails
    is recorded as missing in ``manifest.json``; nothing downstream requires OSM.
``features``
    One row per CurbRamp / Crosswalk label with at least one human Agree/Disagree vote: straight-line distance to
    the nearest intersection node (``d_intersection_m``: degree >= 2; ``d_junction3_m``: degree >= 3), the label's
    own street ``way_type``, OSM distance to the nearest crossing, whether the label's own OSM way is a driveway,
    driveway ways within 20 / 50 m, the human vote counts and majority (AI excluded, ties dropped), the AI's cast
    vote, and the production assessment from ``labels_oof.csv`` (``ai_result``, ``oof_accuracy``). Writes one
    PRIVATE CSV (label-level rows); keep it under ``~/Data``.
``evaluate``
    Per feature alone: AUC for predicting a human-majority Disagree (bootstrap 95% CI), pooled and per public city.
    Lift over the production confidence: logistic model, baseline = logit of ``oof_accuracy`` turned into
    P(Agree), vs baseline + street features (+ OSM features on the OSM-covered cities), under leave-one-city-out
    CV. Disagree rate vs distance plots (Wilson CIs). Crosswalk the same way as a sanity check.
``stratify``
    Among CurbRamp labels with a production verdict, the share of AI Disagrees, contested Agrees (AI Agree, human
    majority Disagree) and expert-overturned Agrees that are far (> 15 / 25 / 40 m) from any intersection and
    > 25 m from any crossing, next to the uncontested pools; then, for AI Disagrees, how often the human majority
    confirms them by road context within ``oof_accuracy`` quartiles (the routing question).
``packet``
    A blinded hand-coding packet of contested CurbRamp labels (driveway / real ramp / other / can't tell), stratified
    by distance bucket and city, shuffled with a secret seed; the key goes to a separate private directory and only
    its sha256 is meant for the repo.

**Privacy.** Every input and the ``features`` CSV hold label-level rows from the private pull; keep them outside the
repo. ``evaluate`` and ``stratify`` write aggregates only and name a city only if ``cities.json`` lists it as
``visibility == "public"`` (the same filter as ``calibration/per_city.py``). Expert vote columns are role-keyed
(``expert1``, ``expert2``, ... in ``--experts`` order), as in ``hard_examples.py``; no name or user id is written.

**Dependencies**: numpy, pandas, scipy (via scikit-learn), scikit-learn, matplotlib and requests, all in
``requirements.txt``. ``osmnx`` was the first plan for ``fetch-osm`` and was dropped on 2026-10-20: it pins one
resolved IP per host, and one of overpass-api.de's two addresses refused connections that night, so every osmnx
request stalled while plain ``requests`` (which falls through to the next address) succeeded.

Usage (``$D`` is the pull directory, e.g. ``~/Data/sidewalk-validator-ai-prod-2026-10-17``)::

    python analysis/road_prior.py fetch-streets --cities-json $D/cities.json --out-dir $D/streets_cache/2026-10-20
    python analysis/road_prior.py fetch-osm --cities-json $D/cities.json --rawlabels $D/rawlabels_cache \\
        --out-dir $D/osm_cache/2026-10-20
    python analysis/road_prior.py features --cities-json $D/cities.json --rawlabels $D/rawlabels_cache \\
        --streets-dir $D/streets_cache/2026-10-20 --osm-dir $D/osm_cache/2026-10-20 \\
        --labels-oof $D/phase0/labels_oof.csv --ai-vs-humans $D/ai_vs_humans.csv --experts "$EXPERT_COLUMNS" \\
        --out $D/road_prior/features.csv
    python analysis/road_prior.py evaluate --features $D/road_prior/features.csv --cities-json $D/cities.json \\
        --out-dir $D/road_prior/evaluate --fig-dir docs/road-prior-2026-10
    python analysis/road_prior.py stratify --features $D/road_prior/features.csv --cities-json $D/cities.json \\
        --out-dir $D/road_prior/stratify
    python analysis/road_prior.py packet --features $D/road_prior/features.csv --cities-json $D/cities.json \\
        --out-dir $D/road-prior-packet --key-dir $D/road-prior-packet/private-key-do-not-share --seed $SECRET_SEED
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.per_city import load_cities, match_city_id, public_name, wilson_interval  # noqa: E402

LABEL_TYPES = ("CurbRamp", "Crosswalk")
EARTH_RADIUS_M = 6_371_008.8
KEY_DECIMALS = 7  # endpoint coordinates are rounded to 1e-7 deg (about 1 cm) before matching
STREET_FEATURES = ["d_intersection_m", "d_junction3_m", "own_way_is_service"]
# own_way_is_service is reported alone but kept out of the models: the streets API has almost no service edges
STREET_MODEL = ["d_intersection_m", "d_junction3_m"]
OSM_MODEL = ["d_crossing_m", "n_driveways_20m", "n_driveways_50m"]
OSM_FEATURES = ["d_crossing_m", "own_way_is_driveway", "n_driveways_20m", "n_driveways_50m"]
DISTANCE_BINS = [0, 5, 10, 15, 25, 40, 60, 100, np.inf]
FAR_INTERSECTION_M = (15, 25, 40)
FAR_CROSSING_M = 25
OVERPASS_URLS = ["https://overpass-api.de/api", "https://overpass.private.coffee/api", "https://overpass.kumi.systems/api"]


# ---------------------------------------------------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------------------------------------------------

def public_cities(cities_json):
    """``{city_id: base url}`` of the public deployments in a saved ``/v3/api/cities`` response.

    >>> import tempfile
    >>> p = os.path.join(tempfile.mkdtemp(), "c.json")
    >>> _ = open(p, "w").write(json.dumps({"cities": [
    ...     {"city_id": "a-wa", "url": "https://a/", "visibility": "public"},
    ...     {"city_id": "b-wa", "url": "https://b", "visibility": "private"}]}))
    >>> public_cities(p)
    {'a-wa': 'https://a'}
    """
    with open(cities_json) as f:
        cities = json.load(f)
    cities = cities.get("cities", cities) if isinstance(cities, dict) else cities
    return {c["city_id"]: c["url"].rstrip("/") for c in cities
            if c.get("visibility") == "public" and c.get("url")}


def display_city(city_id, cities):
    """``city_id`` if ``cities`` (from ``load_cities``) lists it as public, else ``"other (private)"``.

    >>> display_city("seattle-wa", {"seattle-wa": "public"})
    'seattle-wa'
    >>> display_city("x", {"x": "private"})
    'other (private)'
    """
    return city_id if public_name(city_id, cities) == city_id else "other (private)"


# ---------------------------------------------------------------------------------------------------------------------
# Geometry (local equirectangular metres; adequate at city scale for distances under a few km)
# ---------------------------------------------------------------------------------------------------------------------

def to_xy(lat, lng, lat0, lng0):
    """Local metres east (x) and north (y) of ``(lat0, lng0)``.

    >>> x, y = to_xy(np.array([0.0]), np.array([0.001]), 0.0, 0.0)
    >>> round(float(x[0]), 2), round(float(y[0]), 2)
    (111.2, 0.0)
    """
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    k = np.pi / 180 * EARTH_RADIUS_M
    return (lng - lng0) * k * np.cos(np.radians(lat0)), (lat - lat0) * k


def nearest_distance(points_xy, nodes_xy):
    """Distance (m) from each point to its nearest node; NaN everywhere if there are no nodes.

    >>> nearest_distance(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[3.0, 4.0]])).tolist()
    [5.0, 8.06225774829855]
    """
    from scipy.spatial import cKDTree
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    nodes_xy = np.asarray(nodes_xy, dtype=float).reshape(-1, 2)
    if len(nodes_xy) == 0:
        return np.full(len(points_xy), np.nan)
    d, _ = cKDTree(nodes_xy).query(points_xy)
    return d


def densify(coords_xy, step_m=5.0):
    """Points along a polyline no more than ``step_m`` apart (vertices included).

    >>> densify(np.array([[0.0, 0.0], [10.0, 0.0]]), 5.0).tolist()
    [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]
    """
    coords_xy = np.asarray(coords_xy, dtype=float).reshape(-1, 2)
    if len(coords_xy) < 2:
        return coords_xy
    out = [coords_xy[:1]]
    for a, b in zip(coords_xy[:-1], coords_xy[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step_m)))
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a + t * (b - a))
    return np.vstack(out)


def count_ways_within(points_xy, ways_xy, radius_m, step_m=5.0):
    """Number of distinct ways (polylines) passing within ``radius_m`` of each point.

    Each way is densified to points ``step_m`` apart, so a way counts if one of those points is within the radius
    (error at most ``step_m / 2``).

    >>> ways = [np.array([[0.0, 10.0], [100.0, 10.0]]), np.array([[0.0, 30.0], [100.0, 30.0]])]
    >>> count_ways_within(np.array([[50.0, 0.0], [50.0, 20.0]]), ways, 20.0).tolist()
    [1, 2]
    """
    from scipy.spatial import cKDTree
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if not ways_xy:
        return np.zeros(len(points_xy), dtype=int)
    pts, owner = [], []
    for i, w in enumerate(ways_xy):
        d = densify(w, step_m)
        pts.append(d)
        owner.append(np.full(len(d), i))
    pts, owner = np.vstack(pts), np.concatenate(owner)
    hits = cKDTree(pts).query_ball_point(points_xy, r=radius_m)
    return np.array([len(set(owner[h])) for h in hits], dtype=int)


# ---------------------------------------------------------------------------------------------------------------------
# Streets API: intersection nodes
# ---------------------------------------------------------------------------------------------------------------------

def parse_point(s):
    """``"lng,lat"`` -> ``(lat, lng)`` floats.

    >>> parse_point("-122.5,47.25")
    (47.25, -122.5)
    """
    lng, lat = (float(v) for v in str(s).split(","))
    return lat, lng


def street_nodes(streets):
    """Endpoint nodes of a streets table with their degree (number of edge endpoints at that point).

    Args:
        streets: DataFrame with ``start_point`` and ``end_point`` as ``"lng,lat"`` strings (the streets API CSV).

    Returns:
        DataFrame ``lat, lng, degree``, one row per distinct endpoint (coordinates rounded to ``KEY_DECIMALS``).
        Degree 1 is a dead end (or the edge of the audited area); >= 2 is where Project Sidewalk split a street, which
        it does at intersections; >= 3 is a junction of three or more edges.

    >>> s = pd.DataFrame({"start_point": ["0,0", "0,0", "0,0"], "end_point": ["0,0.001", "0.001,0", "0,-0.001"]})
    >>> street_nodes(s).sort_values(["lat", "lng"]).degree.tolist()
    [1, 3, 1, 1]
    """
    pts = pd.concat([streets["start_point"], streets["end_point"]], ignore_index=True).map(parse_point)
    df = pd.DataFrame(pts.tolist(), columns=["lat", "lng"]).round(KEY_DECIMALS)
    return df.groupby(["lat", "lng"]).size().rename("degree").reset_index()


def intersection_summary(nodes):
    """Counts and dead-end share of a ``street_nodes`` table.

    >>> intersection_summary(pd.DataFrame({"degree": [1, 1, 2, 3]}))
    {'nodes': 4, 'deg_ge2': 2, 'deg_ge3': 1, 'dead_end_share': 0.5}
    """
    n = len(nodes)
    return {"nodes": n, "deg_ge2": int((nodes.degree >= 2).sum()), "deg_ge3": int((nodes.degree >= 3).sum()),
            "dead_end_share": float((nodes.degree == 1).mean()) if n else float("nan")}


def fetch_streets(cities, out_dir, timeout=600, retries=3, session=None):
    """Download each city's streets CSV once into ``out_dir`` (skips files already there); returns the manifest."""
    import requests
    session = session or requests.Session()
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for cid, url in sorted(cities.items()):
        out = os.path.join(out_dir, f"{cid}.csv")
        if os.path.exists(out):
            manifest.setdefault(cid, {"status": "ok", "bytes": os.path.getsize(out)})
            continue
        err = None
        for attempt in range(retries):
            try:
                r = session.get(f"{url}/v3/api/streets?filetype=csv", timeout=timeout)
                if r.status_code == 200 and r.text.startswith("street_edge_id"):
                    with open(out + ".tmp", "w") as f:
                        f.write(r.text)
                    os.replace(out + ".tmp", out)
                    err = None
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]!r}"
            except Exception as e:  # network errors are findings, recorded per city
                err = f"{type(e).__name__}: {e}"
            time.sleep(5 * 2 ** attempt)
        manifest[cid] = ({"status": "ok", "bytes": os.path.getsize(out),
                          "fetched_on": datetime.date.today().isoformat()} if err is None
                         else {"status": "failed", "error": err})
        print(f"{cid}: {manifest[cid]}", flush=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# Votes
# ---------------------------------------------------------------------------------------------------------------------

def human_votes(validations):
    """``(h_agree, h_dis, h_unsure, h_maj, ai_vote)`` from a rawLabels ``validations`` JSON string.

    Human votes only (``validator_type == "Human"``); the AI's vote is returned separately and never counted.
    ``h_maj`` is ``"Agree"``/``"Disagree"`` for a strict majority of the Agree/Disagree votes, ``None`` for a tie
    or no votes. Never use ``agree_count``/``disagree_count``: they include the AI.

    >>> v = json.dumps([{"user_id": "a", "validation": "Agree", "validator_type": "AI"},
    ...                 {"user_id": "b", "validation": "Disagree", "validator_type": "Human"},
    ...                 {"user_id": "c", "validation": "Agree", "validator_type": "Human"}])
    >>> human_votes(v)
    (1, 1, 0, None, 'Agree')
    """
    if not isinstance(validations, str) or not validations:
        return 0, 0, 0, None, None
    a = d = u = 0
    ai = None
    for v in json.loads(validations):
        if v.get("validator_type") == "AI":
            ai = v.get("validation")
            continue
        if v.get("validator_type") != "Human":
            continue
        if v.get("validation") == "Agree":
            a += 1
        elif v.get("validation") == "Disagree":
            d += 1
        else:
            u += 1
    maj = "Agree" if a > d else "Disagree" if d > a else None
    return a, d, u, maj, ai


# ---------------------------------------------------------------------------------------------------------------------
# OSM (optional)
# ---------------------------------------------------------------------------------------------------------------------

USER_AGENT = "sidewalk-validator-ai/road_prior.py (https://github.com/ProjectSidewalk/sidewalk-validator-ai)"
TILE_DEG = 0.1  # Overpass query tile side; tiles without labels are skipped


def label_tiles(lat, lng, tile_deg=TILE_DEG):
    """Sorted ``(i, j)`` grid indices (``floor(lat / tile_deg)``, ``floor(lng / tile_deg)``) that contain a label.

    >>> label_tiles(np.array([0.05, 0.06, 0.25]), np.array([0.05, 0.05, -0.05]))
    [(0, 0), (2, -1)]
    """
    i = np.floor(np.asarray(lat) / tile_deg).astype(int)
    j = np.floor(np.asarray(lng) / tile_deg).astype(int)
    return sorted(set(zip(i.tolist(), j.tolist())))


def tile_bbox(i, j, tile_deg=TILE_DEG, pad_m=200.0):
    """``(south, west, north, east)`` of tile ``(i, j)``, padded by ``pad_m`` so labels near its edge see across."""
    s, w = i * tile_deg, j * tile_deg
    dlat = pad_m / (np.pi / 180 * EARTH_RADIUS_M)
    dlng = dlat / np.cos(np.radians(s + tile_deg / 2))
    return (round(s - dlat, 6), round(w - dlng, 6), round(s + tile_deg + dlat, 6), round(w + tile_deg + dlng, 6))


def overpass_query(bbox):
    """Overpass QL for ``highway=crossing`` nodes and ``highway=service`` ways (with geometry) in a
    ``(south, west, north, east)`` box.

    >>> overpass_query((1, 2, 3, 4)).count('(1,2,3,4)')
    2
    """
    b = ",".join(f"{v:g}" for v in bbox)
    return (f'[out:json][timeout:600];(node["highway"="crossing"]({b});way["highway"="service"]({b}););'
            "out tags geom;")


def parse_overpass(elements):
    """``(crossings, service)`` tables from Overpass JSON ``elements`` (``crossings``: ``osmid, lat, lng``; ``service``:
    ``osmid, service, coords`` with ``coords`` a JSON ``[[lng, lat], ...]``).

    >>> c, s = parse_overpass([
    ...     {"type": "node", "id": 1, "lat": 1.0, "lon": 2.0, "tags": {"highway": "crossing"}},
    ...     {"type": "way", "id": 5, "tags": {"highway": "service", "service": "driveway"},
    ...      "geometry": [{"lat": 1.0, "lon": 2.0}, {"lat": 1.001, "lon": 2.0}]},
    ...     {"type": "way", "id": 6, "tags": {"highway": "service"}, "geometry": [{"lat": 1.0, "lon": 2.0}]}])
    >>> c.osmid.tolist(), s.osmid.tolist(), s.service.fillna("-").tolist(), s.coords[0]
    ([1], [5, 6], ['driveway', '-'], '[[2.0, 1.0], [2.0, 1.001]]')
    """
    cr, sv = [], []
    for e in elements:
        tags = e.get("tags", {})
        if e.get("type") == "node" and tags.get("highway") == "crossing":
            cr.append({"osmid": e["id"], "lat": e["lat"], "lng": e["lon"]})
        elif e.get("type") == "way" and tags.get("highway") == "service" and e.get("geometry"):
            sv.append({"osmid": e["id"], "service": tags.get("service"),
                       "coords": json.dumps([[g["lon"], g["lat"]] for g in e["geometry"] if g])})
    crossings = pd.DataFrame(cr, columns=["osmid", "lat", "lng"]).drop_duplicates("osmid")
    service = pd.DataFrame(sv, columns=["osmid", "service", "coords"]).drop_duplicates("osmid")
    return crossings.reset_index(drop=True), service.reset_index(drop=True)


def overpass_post(query, urls, retries=3, session=None, base_sleep=30):
    """POST ``query`` to each URL in turn with exponential backoff; returns ``(elements, url)`` or raises with the
    last error text. Sequential by design: the public instances ban parallel clients."""
    import requests
    session = session or requests.Session()
    last = None
    for url in urls:
        url = url.rstrip("/")
        url = url if url.endswith("/interpreter") else url + "/interpreter"
        for attempt in range(retries):
            try:
                r = session.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=900)
                if r.status_code == 200:
                    j = r.json()
                    if "remark" in j and "error" in j["remark"].lower():
                        last = f"{url}: remark {j['remark'][:200]}"
                    else:
                        return j.get("elements", []), url
                else:
                    last = f"{url}: HTTP {r.status_code} {r.text[:200]!r}"
            except Exception as e:
                last = f"{url}: {type(e).__name__}: {str(e)[:200]}"
            print(f"    attempt {attempt + 1} failed: {last}", flush=True)
            time.sleep(base_sleep * 2 ** attempt)
    raise RuntimeError(last)


def fetch_osm(rawlabels_dir, cities, out_dir, pad_m=200.0, overpass_urls=None, only=None, tile_deg=TILE_DEG):
    """Per-city OSM pull into ``out_dir`` (``<city>_crossings.csv``, ``<city>_service.csv``, ``manifest.json``).

    Each city is queried in ``tile_deg`` tiles that contain at least one CurbRamp/Crosswalk label (padded by
    ``pad_m``); every tile response is cached under ``_tiles/<city>/`` so an interrupted pull resumes.
    """
    import gzip
    import requests
    overpass_urls = overpass_urls or OVERPASS_URLS
    session = requests.Session()
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for cid in sorted(cities):
        if only and cid not in only:
            continue
        if manifest.get(cid, {}).get("status") == "ok":
            continue
        path = os.path.join(rawlabels_dir, f"{cid}.csv")
        if not os.path.exists(path):
            manifest[cid] = {"status": "failed", "error": "no rawLabels file"}
            continue
        lab = pd.read_csv(path, usecols=["label_type", "latitude", "longitude"], low_memory=False)
        lab = lab[lab.label_type.isin(LABEL_TYPES)].dropna()
        if lab.empty:
            manifest[cid] = {"status": "skipped", "error": "no CurbRamp/Crosswalk labels"}
            continue
        tiles = label_tiles(lab.latitude.values, lab.longitude.values, tile_deg)
        tdir = os.path.join(out_dir, "_tiles", cid)
        os.makedirs(tdir, exist_ok=True)
        print(f"{cid}: {len(tiles)} tiles", flush=True)
        elements, used, err = [], set(), None
        for i, j in tiles:
            tpath = os.path.join(tdir, f"{i}_{j}.json.gz")
            if os.path.exists(tpath):
                with gzip.open(tpath, "rt") as f:
                    elements += json.load(f)
                continue
            try:
                el, url = overpass_post(overpass_query(tile_bbox(i, j, tile_deg, pad_m)), overpass_urls,
                                        session=session)
            except Exception as e:
                err = f"tile {i}_{j}: {e}"
                break
            used.add(url)
            with gzip.open(tpath + ".tmp", "wt") as f:
                json.dump(el, f)
            os.replace(tpath + ".tmp", tpath)
            elements += el
            time.sleep(2)
        if err is None:
            crossings, service = parse_overpass(elements)
            crossings.to_csv(os.path.join(out_dir, f"{cid}_crossings.csv"), index=False)
            service.to_csv(os.path.join(out_dir, f"{cid}_service.csv"), index=False)
            manifest[cid] = {"status": "ok", "fetched_on": datetime.date.today().isoformat(),
                             "overpass": sorted(used), "tiles": len(tiles), "crossings": len(crossings),
                             "service_ways": len(service), "driveways": int((service["service"] == "driveway").sum())}
        else:
            manifest[cid] = {"status": "failed", "error": err[:400], "tiles": len(tiles)}
        print(f"{cid}: {manifest[cid]}", flush=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------------------------------

RAW_COLUMNS = ["label_id", "label_type", "latitude", "longitude", "street_edge_id", "osm_way_id", "street_side",
               "centerline_offset_m", "pano_id", "pano_url", "image_capture_date", "validations"]


def load_labels(path, city):
    """CurbRamp / Crosswalk rows of one rawLabels CSV with human vote counts, majority and the AI's cast vote."""
    head = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[c for c in RAW_COLUMNS if c in head], low_memory=False)
    df = df[df.label_type.isin(LABEL_TYPES)].dropna(subset=["latitude", "longitude"]).copy()
    votes = df["validations"].map(human_votes)
    df[["h_agree", "h_dis", "h_unsure", "h_maj", "ai_vote"]] = pd.DataFrame(votes.tolist(), index=df.index)
    df["h_n"] = df.h_agree + df.h_dis
    df.insert(0, "city", city)
    df.insert(1, "label_uid", city + ":" + df.label_id.astype(str))
    return df.drop(columns=["validations"])


def street_features(labels, streets):
    """Add ``d_intersection_m``, ``d_junction3_m``, ``d_deadend_m``, ``own_way_type``, ``own_way_is_service`` to one
    city's labels.

    Distances are straight-line from the label's lat/lng to the nearest street-edge endpoint of degree >= 2 / >= 3
    (the streets API has only edge endpoints, not geometry, so this is not along-street distance). ``d_deadend_m``
    is the distance to the nearest degree-1 endpoint: a dead end, or the edge of the audited area, where a real
    corner ramp can look far from any intersection node. The label's own ``way_type`` comes from its
    ``street_edge_id``.

    >>> s = pd.DataFrame({"street_edge_id": [1, 2, 3], "way_type": ["residential", "service", "residential"],
    ...                   "start_point": ["0,0", "0,0", "0,0"], "end_point": ["0,0.001", "0.001,0", "0,-0.001"]})
    >>> lab = pd.DataFrame({"latitude": [0.0, 0.001], "longitude": [0.0001, 0.0], "street_edge_id": [2, 9]})
    >>> out = street_features(lab, s)
    >>> out.d_intersection_m.round(1).tolist(), out.own_way_type.tolist(), out.own_way_is_service.tolist()
    ([11.1, 111.2], ['service', nan], [1.0, nan])
    """
    out = labels.copy()
    nodes = street_nodes(streets)
    lat0, lng0 = float(out.latitude.mean()), float(out.longitude.mean())
    pxy = np.column_stack(to_xy(out.latitude.values, out.longitude.values, lat0, lng0))
    for col, keep in (("d_intersection_m", nodes.degree >= 2), ("d_junction3_m", nodes.degree >= 3),
                      ("d_deadend_m", nodes.degree == 1)):
        n = nodes[keep]
        out[col] = nearest_distance(pxy, np.column_stack(to_xy(n.lat.values, n.lng.values, lat0, lng0)))
    wt = streets.drop_duplicates("street_edge_id").set_index("street_edge_id")["way_type"]
    out["own_way_type"] = out["street_edge_id"].map(wt)
    out["own_way_is_service"] = np.where(out.own_way_type.isna(), np.nan, (out.own_way_type == "service").astype(float))
    return out


def osm_features(labels, crossings, service):
    """Add ``d_crossing_m``, ``own_way_is_driveway``, ``n_driveways_20m``, ``n_driveways_50m`` to one city's labels.

    ``own_way_is_driveway`` is 1 when the label's ``osm_way_id`` is a ``service=driveway`` way in the pull, 0 when
    it is not (labels sit on streets, so this is almost always 0; it is reported to show that).

    >>> lab = pd.DataFrame({"latitude": [0.0], "longitude": [0.0], "osm_way_id": [7]})
    >>> cr = pd.DataFrame({"osmid": [1], "lat": [0.0], "lng": [0.0002]})
    >>> sv = pd.DataFrame({"osmid": [7, 8], "service": ["driveway", "alley"],
    ...                    "coords": ["[[0.0001, -0.001], [0.0001, 0.001]]", "[[0.0, 0.0001], [0.001, 0.0001]]"]})
    >>> out = osm_features(lab, cr, sv)
    >>> round(float(out.d_crossing_m[0]), 1), float(out.own_way_is_driveway[0]), int(out.n_driveways_20m[0])
    (22.2, 1.0, 1)
    """
    out = labels.copy()
    lat0, lng0 = float(out.latitude.mean()), float(out.longitude.mean())
    pxy = np.column_stack(to_xy(out.latitude.values, out.longitude.values, lat0, lng0))
    out["d_crossing_m"] = nearest_distance(pxy, np.column_stack(to_xy(crossings.lat.values, crossings.lng.values,
                                                                      lat0, lng0)))
    drive = service[service["service"] == "driveway"]
    out["own_way_is_driveway"] = out["osm_way_id"].isin(set(drive.osmid)).astype(float)
    ways = []
    for c in drive["coords"]:
        a = np.asarray(json.loads(c), dtype=float).reshape(-1, 2)
        ways.append(np.column_stack(to_xy(a[:, 1], a[:, 0], lat0, lng0)))
    for r in (20, 50):
        out[f"n_driveways_{r}m"] = count_ways_within(pxy, ways, r)
    return out


def production_assessments(labels_oof, cities):
    """``label_uid, ai_result, ai_accuracy, oof_accuracy`` from ``labels_oof.csv``, keyed ``city_id:label_id``.

    The schema's city part is matched to a ``city_id`` with ``calibration.per_city.match_city_id``; unmatched schemas
    are dropped (they have no rawLabels file here anyway).

    >>> oof = pd.DataFrame({"schema": ["sidewalk_seattle", "sidewalk_zzz"], "label_id": [5, 5],
    ...                     "label_type": ["CurbRamp"] * 2, "ai_result": ["Agree"] * 2,
    ...                     "ai_accuracy": [0.9, 0.9], "oof_accuracy": [0.95, 0.95]})
    >>> production_assessments(oof, {"seattle-wa": "public"}).label_uid.tolist()
    ['seattle-wa:5']
    """
    df = labels_oof[labels_oof.label_type.isin(LABEL_TYPES)]
    sc = df["schema"].str.replace("sidewalk_", "", n=1, regex=False)
    ids = {s: match_city_id(s, cities) for s in sc.unique()}
    cid = sc.map(ids)
    keep = cid.notna()
    out = pd.DataFrame({"label_uid": cid[keep] + ":" + df.loc[keep, "label_id"].astype(str),
                        "ai_result": df.loc[keep, "ai_result"], "ai_accuracy": df.loc[keep, "ai_accuracy"],
                        "oof_accuracy": df.loc[keep, "oof_accuracy"]})
    assert out.label_uid.is_unique, "labels_oof has duplicate (city, label_id) rows"
    return out


def expert_frame(ai_vs_humans, experts):
    """``label_uid`` plus role-keyed ``expert1``, ``expert2``, ... from the private export's expert vote columns.

    ``experts`` lists the export's own column names in order; they are renamed by position, so no name is carried.

    >>> e = pd.DataFrame({"label_uid": ["a:1"], "x": ["Agree"], "y": [None]})
    >>> expert_frame(e, ["x", "y"]).columns.tolist()
    ['label_uid', 'expert1', 'expert2']
    """
    cols = {c: f"expert{i + 1}" for i, c in enumerate(experts)}
    return ai_vs_humans[["label_uid", *cols]].rename(columns=cols)


def build_features(rawlabels_dir, streets_dir, cities_json, osm_dir=None, labels_oof=None, ai_vs_humans=None,
                   experts=(), min_votes=1):
    """The label-level features table for every public city with a rawLabels file (see the module docstring)."""
    urls = public_cities(cities_json)
    frames, report = [], {}
    for cid in sorted(urls):
        path = os.path.join(rawlabels_dir, f"{cid}.csv")
        if not os.path.exists(path):
            report[cid] = "no rawLabels file"
            continue
        lab = load_labels(path, cid)
        lab = lab[lab.h_n >= min_votes]
        if lab.empty:
            report[cid] = "no CurbRamp/Crosswalk labels with human votes"
            continue
        spath = os.path.join(streets_dir, f"{cid}.csv")
        if os.path.exists(spath):
            lab = street_features(lab, pd.read_csv(spath, usecols=["street_edge_id", "way_type", "start_point",
                                                                   "end_point"]))
            lab["streets_available"] = 1
        else:
            for c in STREET_FEATURES + ["own_way_type"]:
                lab[c] = np.nan
            lab["streets_available"] = 0
        cpath = os.path.join(osm_dir or "", f"{cid}_crossings.csv")
        if osm_dir and os.path.exists(cpath):
            lab = osm_features(lab, pd.read_csv(cpath), pd.read_csv(os.path.join(osm_dir, f"{cid}_service.csv")))
            lab["osm_available"] = 1
        else:
            for c in OSM_FEATURES:
                lab[c] = np.nan
            lab["osm_available"] = 0
        frames.append(lab)
        report[cid] = f"{len(lab)} labels, streets={bool(lab.streets_available.iat[0])}, osm={bool(lab.osm_available.iat[0])}"
    df = pd.concat(frames, ignore_index=True)
    assert df.label_uid.is_unique
    cities = load_cities(cities_json)
    if labels_oof is not None:
        prod = production_assessments(labels_oof, cities)
        df = df.merge(prod, on="label_uid", how="left", validate="one_to_one")
    if ai_vs_humans is not None and experts:
        df = df.merge(expert_frame(ai_vs_humans, experts), on="label_uid", how="left", validate="one_to_one")
    return df, report


# ---------------------------------------------------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------------------------------------------------

def analysis_set(df, label_type, min_votes=3):
    """Labels of one type with a strict human majority on >= ``min_votes`` human Agree/Disagree votes; adds ``y``
    (1 = human-majority Disagree)."""
    m = (df.label_type == label_type) & (df.h_n >= min_votes) & df.h_maj.isin(["Agree", "Disagree"])
    out = df[m].copy()
    out["y"] = (out.h_maj == "Disagree").astype(int)
    return out


def auc_ci(y, score, n_boot=500, seed=0):
    """``(auc, lo, hi, n)``: ROC AUC of ``score`` for ``y`` with a percentile bootstrap 95% CI; NaN rows dropped.

    AUC > 0.5 means larger values go with ``y == 1``. NaN when only one class is present.

    >>> a, lo, hi, n = auc_ci(np.array([0, 0, 1, 1]), np.array([1.0, 2.0, 3.0, 4.0]), n_boot=50)
    >>> a, n
    (1.0, 4)
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok].astype(int), s[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, len(y)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(auc), float(lo), float(hi), int(len(y))


def feature_auc_table(df, features, min_city_n=100, n_boot=500):
    """AUC per feature, pooled (``city == "all public cities"``) and per city with ``n >= min_city_n``.

    ``df`` is an ``analysis_set`` (has ``y`` and ``city``, public cities only).
    """
    rows = []
    for f in features:
        groups = [("all public cities", df)] + [(c, g) for c, g in df.groupby("city")]
        for city, g in groups:
            if g[f].notna().sum() < (1 if city == "all public cities" else min_city_n):
                continue
            auc, lo, hi, n = auc_ci(g.y.values, g[f].values, n_boot=n_boot)
            rows.append({"feature": f, "city": city, "n": n, "disagree_rate": float(g.loc[g[f].notna(), "y"].mean()),
                         "auc": auc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def p_agree(ai_result, accuracy):
    """The production model's P(human Agree) from its verdict and the verdict's accuracy.

    >>> p_agree(pd.Series(["Agree", "Disagree"]), pd.Series([0.9, 0.8])).tolist()
    [0.9, 0.19999999999999996]
    """
    return pd.Series(np.where(ai_result == "Agree", accuracy, 1.0 - accuracy), index=ai_result.index)


SUBSET_TAGS = {"all verdicts": "all", "AI Agree only": "agree", "AI Disagree only": "disagree",
               "cast AI Disagree votes": "cast_disagree"}  # lift file names
MODEL_BINS = [0, 5, 15, 25, 40, 100, np.inf]  # the Disagree rate is not monotone in distance (see the binned tables)
MODEL_REFERENCE_BIN = 1  # [5, 15) m, the bulk of real corner ramps, is the reference level


def design(df, cols):
    """Model inputs: each distance as one-hot ``MODEL_BINS`` indicators (reference ``[5, 15)`` m dropped), counts
    as ``log1p``, flags and ``conf_logit`` as is.

    Distances enter as bins, not a single slope, because the Disagree rate rises from 15 to 40 m and falls again
    past 100 m; a log-distance slope would miss that shape.

    >>> design(pd.DataFrame({"d_x": [2.0, 10.0, 30.0], "n_y": [0, 1, 3]}), ["d_x", "n_y"]).columns.tolist()
    ['d_x_0-5', 'd_x_15-25', 'd_x_25-40', 'd_x_40-100', 'd_x_100+', 'n_y']
    """
    X = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[c].astype(float)
        if c.startswith("d_"):
            idx = np.digitize(v.values, MODEL_BINS[1:-1])
            for b in range(len(MODEL_BINS) - 1):
                if b == MODEL_REFERENCE_BIN:
                    continue
                lo, hi = MODEL_BINS[b], MODEL_BINS[b + 1]
                X[f"{c}_{lo:g}{'+' if np.isinf(hi) else f'-{hi:g}'}"] = (idx == b).astype(float)
        elif c.startswith("n_"):
            X[c] = np.log1p(v)
        else:
            X[c] = v
    return X


def loco_predict(df, cols, group="city", C=1.0):
    """Leave-one-city-out out-of-fold P(y = 1) from a standardised logistic regression on ``design(df, cols)``.

    Every city is predicted by a model fitted on the other cities only, so the score says whether the relation
    transfers to an unseen street grid.

    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"city": np.repeat(list("abc"), 50), "d_x": rng.uniform(0, 100, 150)})
    >>> d["y"] = (d.d_x + rng.normal(0, 20, 150) > 50).astype(int)
    >>> p = loco_predict(d, ["d_x"])
    >>> p.shape, bool(p.between(0, 1).all())
    ((150,), True)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = design(df, cols).values
    y = df["y"].values
    out = np.full(len(df), np.nan)
    for g in df[group].unique():
        test = (df[group] == g).values
        if len(np.unique(y[~test])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[~test], y[~test])
        out[test] = m.predict_proba(X[test])[:, 1]
    return pd.Series(out, index=df.index)


def ece(y, p, bins=10):
    """Expected calibration error over equal-width probability bins.

    >>> ece(np.array([0, 1]), np.array([0.0, 1.0]))
    0.0
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def scores(y, p):
    """AUC, log-loss, Brier and ECE of probabilities ``p`` for ``y``."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    one = len(np.unique(y)) < 2
    return {"auc": np.nan if one else roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p), "ece": ece(y, p)}


def paired_delta_ci(y, p0, p1, n_boot=500, seed=0):
    """Bootstrap 95% CIs of AUC(p1) - AUC(p0) and log-loss(p1) - log-loss(p0) over the same resampled labels."""
    from sklearn.metrics import log_loss, roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p0, p1 = (np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6) for p in (p0, p1))
    da, dl = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        da.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
        dl.append(log_loss(y[i], p1[i], labels=[0, 1]) - log_loss(y[i], p0[i], labels=[0, 1]))
    return tuple(np.percentile(da, [2.5, 97.5])), tuple(np.percentile(dl, [2.5, 97.5]))


def lift(df, base_cols, extra_cols, min_city_n=100, n_boot=500):
    """LOCO baseline vs baseline + extra on the rows where every column is present.

    Returns ``(pooled dict, per-city DataFrame, frame with y, p_base, p_full)``.
    """
    cols = base_cols + extra_cols
    d = df.dropna(subset=cols).copy()
    d["p_base"] = loco_predict(d, base_cols)
    d["p_full"] = loco_predict(d, cols)
    d = d.dropna(subset=["p_base", "p_full"])
    s0, s1 = scores(d.y.values, d.p_base.values), scores(d.y.values, d.p_full.values)
    (alo, ahi), (llo, lhi) = paired_delta_ci(d.y.values, d.p_base.values, d.p_full.values, n_boot=n_boot)
    pooled = {"n": len(d), "cities": d.city.nunique(), "disagree_rate": float(d.y.mean()),
              **{f"{k}_base": v for k, v in s0.items()}, **{f"{k}_full": v for k, v in s1.items()},
              "d_auc": s1["auc"] - s0["auc"], "d_auc_lo": alo, "d_auc_hi": ahi,
              "d_log_loss": s1["log_loss"] - s0["log_loss"], "d_log_loss_lo": llo, "d_log_loss_hi": lhi}
    rows = []
    for c, g in d.groupby("city"):
        if len(g) < min_city_n:
            continue
        a, b = scores(g.y.values, g.p_base.values), scores(g.y.values, g.p_full.values)
        rows.append({"city": c, "n": len(g), "disagree_rate": float(g.y.mean()), "auc_base": a["auc"],
                     "auc_full": b["auc"], "d_auc": b["auc"] - a["auc"], "d_log_loss": b["log_loss"] - a["log_loss"]})
    return pooled, pd.DataFrame(rows), d


def binned_rate(d, col, bins=DISTANCE_BINS):
    """Disagree rate with Wilson 95% CI per distance bin of ``col``.

    >>> d = pd.DataFrame({"x": [1.0, 2.0, 30.0, 31.0], "y": [0, 0, 1, 0]})
    >>> binned_rate(d, "x", [0, 10, 100])[["bin", "n", "rate"]].values.tolist()
    [['0-10', 2, 0.0], ['10-100', 2, 0.5]]
    """
    d = d.dropna(subset=[col])
    cut = pd.cut(d[col], bins, right=False)
    rows = []
    for iv, g in d.groupby(cut, observed=True):
        lo, hi = wilson_interval(g.y.mean(), len(g))
        right = "+" if np.isinf(iv.right) else f"-{iv.right:g}"
        rows.append({"bin": f"{iv.left:g}{right}", "lo_m": iv.left, "n": len(g), "rate": float(g.y.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


def _plot_rates(ax, tab, color, label, marker="o"):
    x = np.arange(len(tab))
    ax.errorbar(x, tab.rate, yerr=[tab.rate - tab.ci_lo, tab.ci_hi - tab.rate], color=color, marker=marker,
                linewidth=2, markersize=6, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n:,}" for b, n in zip(tab.bin, tab.n)], fontsize=7)


def plot_pooled(sets, col, path, xlabel):
    """Disagree rate vs distance, one line per label type (``sets`` = ``{label_type: analysis_set}``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calibration._plots import COLORS
    colors = {"CurbRamp": COLORS["observed"], "Crosswalk": COLORS["old"]}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 3.8), squeeze=False)
    for ax, (lt, d) in zip(axes[0], sets.items()):
        tab = binned_rate(d, col)
        if len(tab):
            _plot_rates(ax, tab, colors.get(lt, COLORS["new"]), lt)
        ax.set_title(f"{lt}: human-majority Disagree rate", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Disagree rate (Wilson 95% CI)", fontsize=9)
        ax.set_ylim(0, None)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_by_city(d, col, path, xlabel, min_city_n=100, title=""):
    """Small multiples of Disagree rate vs distance per city with ``n >= min_city_n``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calibration._plots import COLORS
    counts = d.dropna(subset=[col]).city.value_counts()
    cities = list(counts[counts >= min_city_n].index)
    if not cities:
        return
    ncol = 4
    nrow = int(np.ceil(len(cities) / ncol))
    bins = [0, 15, 25, 40, 100, np.inf]
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False, sharey=True)
    for ax, c in zip(axes.flat, cities):
        tab = binned_rate(d[d.city == c], col, bins)
        _plot_rates(ax, tab, COLORS["observed"], c)
        ax.set_title(f"{c} (n={counts[c]:,})", fontsize=9)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(cities):]:
        ax.axis("off")
    fig.suptitle(f"{title} Disagree rate vs {xlabel} (Wilson 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_calibration(frames, path):
    """Reliability of the LOCO models: observed Disagree rate per predicted-probability decile bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calibration._plots import COLORS
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "--", color=COLORS["gate"], linewidth=1, label="perfect")
    styles = [("p_base", "baseline (production confidence)", COLORS["old"], "s"),
              ("p_full", "baseline + road features", COLORS["observed"], "o")]
    for col, label, color, marker in styles:
        b = np.minimum((frames[col] * 10).astype(int), 9)
        g = frames.groupby(b).agg(p=(col, "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 20]
        ax.plot(g.p, g.y, marker=marker, color=color, linewidth=2, markersize=6, label=label)
    ax.set_xlabel("predicted P(human Disagree), LOCO", fontsize=9)
    ax.set_ylabel("observed Disagree rate", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _md(df, floatfmt=3):
    """A pandas frame as a GitHub markdown table (no tabulate dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def public_only(df, cities):
    """Rows whose ``city`` is a public deployment in ``cities`` (the naming convention); the rest are dropped."""
    keep = df.city.map(lambda c: display_city(c, cities) == c)
    return df[keep]


def evaluate(df, cities, out_dir, fig_dir=None, min_votes=3, min_city_n=100, n_boot=500):
    """Every ``evaluate`` table and figure; returns the markdown report (also written to ``out_dir/evaluate.md``)."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = fig_dir or out_dir
    os.makedirs(fig_dir, exist_ok=True)
    df = public_only(df, cities)
    md = [f"# Road prior: evaluate (min human votes {min_votes}, min city n {min_city_n}, bootstrap {n_boot})", ""]
    has_osm = df.osm_available.eq(1).any()
    feats = STREET_FEATURES + (OSM_FEATURES if has_osm else [])
    sets = {lt: analysis_set(df, lt, min_votes) for lt in LABEL_TYPES}
    for lt, d in sets.items():
        md += [f"## {lt}", "", f"Analysis set: {len(d):,} labels with >= {min_votes} human votes and a strict "
               f"majority, {d.city.nunique()} public cities, human-majority Disagree rate {d.y.mean():.3f}; "
               f"{int(d.streets_available.sum()):,} with streets features, {int(d.osm_available.sum()):,} with OSM.", ""]
        t = feature_auc_table(d, feats, min_city_n, n_boot)
        t.to_csv(os.path.join(out_dir, f"feature_auc_{lt.lower()}.csv"), index=False)
        pooled = t[t.city == "all public cities"].drop(columns="city")
        md += ["### Feature alone, pooled (AUC for predicting a human-majority Disagree; > 0.5 = larger value, more "
               "Disagree)", "", _md(pooled), ""]
        wide = t[t.city != "all public cities"].pivot(index="city", columns="feature", values="auc")
        n = t[(t.city != "all public cities") & (t.feature == "d_intersection_m")].set_index("city").n
        wide.insert(0, "n", n)
        md += [f"### Feature alone, per city (AUC; cities with n >= {min_city_n})", "",
               _md(wide.reset_index().sort_values("n", ascending=False)), ""]
        cond = []
        for verdict in ("Agree", "Disagree"):
            v = d[d.ai_result == verdict]
            if len(v) and v.y.nunique() == 2:
                t2 = feature_auc_table(v, [f for f in feats if f != "own_way_is_service"], min_city_n, n_boot)
                t2 = t2[t2.city == "all public cities"].drop(columns="city")
                t2.insert(0, "ai_result", verdict)
                cond.append(t2)
        if cond:
            md += ["### Feature alone within each production verdict (pooled AUC; what the AI has not already used)",
                   "", _md(pd.concat(cond)), ""]
        binned = binned_rate(d, "d_intersection_m")
        md += ["### Disagree rate by distance to the nearest intersection node (pooled)", "", _md(binned), ""]
        if has_osm:
            md += ["### Disagree rate by distance to the nearest OSM crossing (pooled, OSM-covered cities)", "",
                   _md(binned_rate(d, "d_crossing_m")), ""]
        # lift over the production confidence (AI-assessed subset only)
        a = d[d.ai_result.isin(["Agree", "Disagree"]) & d.oof_accuracy.notna()].copy()
        a["conf_logit"] = np.log(np.clip(p_agree(a.ai_result, a.oof_accuracy), 1e-4, 1 - 1e-4)
                                 / np.clip(1 - p_agree(a.ai_result, a.oof_accuracy), 1e-4, 1 - 1e-4))
        md += [f"### Lift over the production confidence, leave-one-city-out", "",
               f"AI-assessed subset with `oof_accuracy`: {len(a):,} labels in {a.city.nunique()} cities. Baseline "
               "= logistic on logit P(Agree) from `ai_result` + `oof_accuracy`; full = baseline + features; each "
               "city predicted by a model fitted on the others. Δ = full - baseline (Δlog-loss < 0 is better).", ""]
        variants = [("streets", STREET_MODEL)]
        if has_osm:
            variants.append(("streets + OSM (OSM-covered cities)", STREET_MODEL + OSM_MODEL))
        prow = []
        subsets = [("all verdicts", a), ("AI Agree only", a[a.ai_result == "Agree"]),
                   ("AI Disagree only", a[a.ai_result == "Disagree"]),
                   ("cast AI Disagree votes", a[a.ai_vote == "Disagree"])]
        for sub_name, sub in subsets:
            for name, cols in variants:
                if len(sub.dropna(subset=cols)) < 50:
                    continue
                pooled_l, per_city, frame = lift(sub, ["conf_logit"], cols, min_city_n, n_boot)
                prow.append({"subset": sub_name, "features": name, **pooled_l})
                tag = f"{lt.lower()}_{name.split()[0]}_{SUBSET_TAGS[sub_name]}"
                per_city.to_csv(os.path.join(out_dir, f"lift_by_city_{tag}.csv"), index=False)
                md += [f"Per held-out city, {sub_name}, {name} (n >= {min_city_n}):", "",
                       _md(per_city.sort_values("n", ascending=False)), ""]
                if name == "streets" and sub_name == "all verdicts":
                    plot_calibration(frame, os.path.join(fig_dir, f"loco_calibration_{lt.lower()}.png"))
        if prow:
            p = pd.DataFrame(prow)
            p.to_csv(os.path.join(out_dir, f"lift_{lt.lower()}.csv"), index=False)
            md += ["Pooled:", "", _md(p[["subset", "features", "n", "cities", "disagree_rate", "auc_base", "auc_full", "d_auc",
                                         "d_auc_lo", "d_auc_hi", "log_loss_base", "log_loss_full", "d_log_loss",
                                         "d_log_loss_lo", "d_log_loss_hi", "ece_base", "ece_full"]], 4), ""]
        # features alone, LOCO, on the whole analysis set (no AI): does the prior transfer by itself?
        alone = []
        for name, cols in variants:
            s = d.dropna(subset=cols).copy()
            if len(s) < 50:
                continue
            s["p"] = loco_predict(s, cols)
            s = s.dropna(subset=["p"])
            per = [scores(g.y.values, g.p.values)["auc"] for _, g in s.groupby("city") if len(g) >= min_city_n]
            alone.append({"features": name, "n": len(s), "cities": s.city.nunique(), **scores(s.y.values, s.p.values),
                          "median_city_auc": float(np.nanmedian(per)) if per else np.nan,
                          "cities_n_ge_min": len(per)})
        if alone:
            md += ["### Road features alone, leave-one-city-out (no AI input, whole analysis set)", "",
                   "`auc` pools every held-out city's predictions into one ranking, so it mixes cities' base rates; "
                   f"`median_city_auc` is the median of the per-city AUCs (cities with n >= {min_city_n}).", "",
                   _md(pd.DataFrame(alone), 4), ""]
        if lt == "CurbRamp":
            plot_by_city(d, "d_intersection_m", os.path.join(fig_dir, "disagree_vs_intersection_by_city.png"),
                         "distance to nearest intersection node (m)", min_city_n, title=lt)
    plot_pooled(sets, "d_intersection_m", os.path.join(fig_dir, "disagree_vs_intersection_pooled.png"),
                "distance to nearest intersection node (m), straight line")
    if has_osm:
        plot_pooled(sets, "d_crossing_m", os.path.join(fig_dir, "disagree_vs_crossing_pooled.png"),
                    "distance to nearest OSM highway=crossing node (m)")
    text = "\n".join(md)
    with open(os.path.join(out_dir, "evaluate.md"), "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------------------------------------------------
# Stratify
# ---------------------------------------------------------------------------------------------------------------------

def production_pools(df, min_votes=3, experts=()):
    """``{pool name: boolean mask}`` over CurbRamp rows with a production verdict (``ai_result``).

    Human-majority pools need ``>= min_votes`` human Agree/Disagree votes and a strict majority; the expert pool is
    AI Agree with any listed expert column voting Disagree.

    >>> d = pd.DataFrame({"label_type": ["CurbRamp"] * 3, "ai_result": ["Agree", "Agree", "Disagree"],
    ...                   "h_n": [3, 3, 0], "h_maj": ["Disagree", "Agree", None], "expert1": [None, "Disagree", None]})
    >>> {k: v.tolist() for k, v in production_pools(d, 3, ["expert1"]).items()}["AI Agree, expert Disagree"]
    [False, True, False]
    """
    cr = (df.label_type == "CurbRamp") & df.ai_result.isin(["Agree", "Disagree"])
    voted = (df.h_n >= min_votes)
    agree, dis = cr & (df.ai_result == "Agree"), cr & (df.ai_result == "Disagree")
    pools = {
        "AI Disagree (all)": dis,
        "AI Disagree, human majority Disagree (AI right)": dis & voted & (df.h_maj == "Disagree"),
        "AI Disagree, human majority Agree (contested)": dis & voted & (df.h_maj == "Agree"),
        "AI Agree (all)": agree,
        "AI Agree, human majority Agree (uncontested)": agree & voted & (df.h_maj == "Agree"),
        "AI Agree, human majority Disagree (contested)": agree & voted & (df.h_maj == "Disagree"),
    }
    if experts:
        ex = pd.Series(False, index=df.index)
        for e in experts:
            ex |= df[e].eq("Disagree")
        pools["AI Agree, expert Disagree"] = agree & ex
    return pools


def far_shares(df, pools, far_int=FAR_INTERSECTION_M, far_cross=FAR_CROSSING_M):
    """Per pool: n, and the share of labels at least each threshold from any intersection / crossing (``>=``, the
    same convention as ``road_context``).

    Intersection shares are over rows with ``d_intersection_m``; crossing shares over rows with ``d_crossing_m``
    (``n_osm``). ``far_both`` = beyond ``far_cross`` from both.

    >>> d = pd.DataFrame({"d_intersection_m": [10.0, 30.0, 50.0, np.nan], "d_crossing_m": [5.0, 40.0, np.nan, 1.0]})
    >>> t = far_shares(d, {"p": pd.Series([True, True, True, True])}, far_int=(25,), far_cross=25)
    >>> [float(v) for v in t.iloc[0, 1:]]  # n, n_streets, far_int_25m, n_osm, far_cross_25m, n_both, far_both_25m
    [4.0, 3.0, 0.6666666666666666, 3.0, 0.3333333333333333, 2.0, 0.5]
    """
    rows = []
    for name, m in pools.items():
        g = df[m.values if hasattr(m, "values") else m]
        s = g.dropna(subset=["d_intersection_m"])
        o = g.dropna(subset=["d_crossing_m"]) if "d_crossing_m" in g else g.iloc[0:0]
        r = {"pool": name, "n": len(g), "n_streets": len(s)}
        for t in far_int:
            r[f"far_int_{t}m"] = float((s.d_intersection_m >= t).mean()) if len(s) else np.nan
        r["n_osm"] = len(o)
        r[f"far_cross_{far_cross}m"] = float((o.d_crossing_m >= far_cross).mean()) if len(o) else np.nan
        both = o.dropna(subset=["d_intersection_m"])
        r["n_both"] = len(both)
        r[f"far_both_{far_cross}m"] = (float(((both.d_intersection_m >= far_cross) & (both.d_crossing_m >= far_cross))
                                             .mean()) if len(both) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


ROUTING_BINS = [0, 15, 25, 40, 100, np.inf]


def agree_error_tables(df, min_votes=3):
    """``(name, rows, distance column)`` for the routing question: among CurbRamp AI Agrees with a human majority
    (all, and those whose ``oof_accuracy`` clears the 0.92 gate, i.e. would be cast under the Phase 0 maps), how
    often the majority says Disagree, by distance. Rows carry ``y``."""
    from calibration._isotonic_map import GATE
    a = df[(df.label_type == "CurbRamp") & (df.ai_result == "Agree") & (df.h_n >= min_votes)
           & df.h_maj.isin(["Agree", "Disagree"])].copy()
    a["y"] = (a.h_maj == "Disagree").astype(int)
    g = a[a.oof_accuracy >= GATE]
    out = []
    for col in ("d_intersection_m", "d_crossing_m"):
        if col in a and a[col].notna().any():
            out += [("AI Agree (all)", a, col), (f"AI Agree, oof_accuracy >= {GATE}", g, col)]
    return out


def road_context(df, far_m=FAR_CROSSING_M):
    """``far_both`` / ``far_one`` / ``near_both``: whether a label is ``>= far_m`` from both the nearest
    intersection node and the nearest OSM crossing, from exactly one, or from neither.

    >>> road_context(pd.DataFrame({"d_intersection_m": [30.0, 30.0, 5.0], "d_crossing_m": [40.0, 5.0, 5.0]})).tolist()
    ['far_both', 'far_one', 'near_both']
    """
    fi, fc = df.d_intersection_m >= far_m, df.d_crossing_m >= far_m
    return pd.Series(np.select([fi & fc, fi ^ fc], ["far_both", "far_one"], "near_both"), index=df.index)


def disagree_context_table(df, min_votes=3, n_conf_bins=4, cast_only=False):
    """Among CurbRamp AI Disagrees with a human majority, the rate at which the majority confirms the Disagree, by
    road context (``road_context``) within ``oof_accuracy`` quantile bins.

    This is the routing question: does road context say anything about an AI Disagree that its own confidence has
    not already said? Rows with no OSM crossing distance are dropped. ``cast_only`` keeps the assessments the AI
    actually cast as a Disagree vote (``ai_vote == "Disagree"``), the ones that can hide a label.

    >>> d = pd.DataFrame({"label_type": "CurbRamp", "ai_result": "Disagree", "h_n": 3,
    ...                   "h_maj": ["Disagree", "Agree"] * 4, "oof_accuracy": [0.2] * 4 + [0.7] * 4,
    ...                   "d_intersection_m": [30.0, 5.0] * 4, "d_crossing_m": [30.0, 5.0] * 4})
    >>> disagree_context_table(d, n_conf_bins=2)[["context", "n", "confirm_rate"]].values.tolist()[:2]
    [['far_both', 2, 1.0], ['near_both', 2, 0.0]]
    """
    v = df[(df.label_type == "CurbRamp") & (df.ai_result == "Disagree") & (df.h_n >= min_votes)
           & df.h_maj.isin(["Agree", "Disagree"])].dropna(subset=["oof_accuracy", "d_intersection_m", "d_crossing_m"])
    if cast_only:
        v = v[v.ai_vote == "Disagree"]
    v = v.assign(context=road_context(v), ok=(v.h_maj == "Disagree"),
                 conf_bin=pd.qcut(v.oof_accuracy, n_conf_bins, duplicates="drop").astype(str))
    rows = []
    for (cb, ctx), g in v.groupby(["conf_bin", "context"]):
        lo, hi = wilson_interval(g.ok.mean(), len(g))
        rows.append({"oof_accuracy_bin": cb, "context": ctx, "n": len(g), "confirm_rate": float(g.ok.mean()),
                     "ci_lo": lo, "ci_hi": hi})
    for ctx, g in v.groupby("context"):
        lo, hi = wilson_interval(g.ok.mean(), len(g))
        rows.append({"oof_accuracy_bin": "all", "context": ctx, "n": len(g), "confirm_rate": float(g.ok.mean()),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Routing: cast CurbRamp Disagree votes by road context
# ---------------------------------------------------------------------------------------------------------------------

ROUTING_THRESHOLDS_M = (15, 25, 40)  # 25 m is the headline; 15 and 40 m are the sensitivity check
ROUTING_MIN_VOTES = (1, 2, 3)  # 3 is the headline; 1 and 2 test the selection of heavily validated labels
DEADEND_NEAR_M = 15


def logit_fit(X, y, names, max_iter=100, tol=1e-10):
    """Unpenalised logistic regression by Newton-Raphson, with Wald standard errors and two-sided p-values.

    ``X`` must already hold an intercept column if one is wanted. Used for the city-fixed-effects models, where
    the coefficient and its p-value are the result (scikit-learn's ``LogisticRegression`` penalises and gives no
    standard errors).

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=4000)
    >>> y = (rng.random(4000) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(float)
    >>> t = logit_fit(np.column_stack([np.ones(4000), x]), y, ["const", "x"])
    >>> bool(abs(t.loc["x", "coef"] - 1.5) < 0.15), bool(t.loc["x", "p"] < 1e-6)
    (True, True)
    """
    from scipy.stats import norm
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-(X @ b)))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    mu = 1 / (1 + np.exp(-(X @ b)))
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    se = np.sqrt(np.diag(cov))
    z = b / se
    return pd.DataFrame({"coef": b, "se": se, "z": z, "p": 2 * norm.sf(np.abs(z))}, index=names)


def cast_disagrees(df, min_votes=3):
    """Public CurbRamp labels the AI cast a Disagree vote on, with ``conf_logit`` (logit of the production
    P(Agree)), ``y`` = 1 when the human majority confirms the Disagree, and ``validated`` (>= ``min_votes`` human
    votes and a strict majority). Rows without both distances are dropped."""
    c = df[(df.label_type == "CurbRamp") & (df.ai_vote == "Disagree") & df.oof_accuracy.notna()]
    c = c.dropna(subset=["d_intersection_m", "d_crossing_m"]).copy()
    pa = np.clip(p_agree(c.ai_result, c.oof_accuracy), 1e-4, 1 - 1e-4)
    c["conf_logit"] = np.log(pa / (1 - pa))
    c["validated"] = (c.h_n >= min_votes) & c.h_maj.isin(["Agree", "Disagree"])
    c["y"] = (c.h_maj == "Disagree").astype(int)
    return c


def context_dummies(d, far_m):
    """``near_both`` and ``far_one`` indicators (reference: far from both), plus the two single axes
    ``near_int`` / ``near_cross``, at threshold ``far_m``.

    >>> context_dummies(pd.DataFrame({"d_intersection_m": [5.0, 30.0, 30.0], "d_crossing_m": [5.0, 5.0, 30.0]}),
    ...                 25).values.tolist()
    [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    """
    ctx = road_context(d, far_m)
    return pd.DataFrame({"near_both": (ctx == "near_both").astype(float), "far_one": (ctx == "far_one").astype(float),
                         "near_int": (d.d_intersection_m < far_m).astype(float),
                         "near_cross": (d.d_crossing_m < far_m).astype(float)}, index=d.index)


def city_fe_models(v, thresholds=ROUTING_THRESHOLDS_M):
    """City-fixed-effects logistic models of ``y`` on ``conf_logit`` plus road context, per threshold.

    Two specifications: context (``near_both``, ``far_one``; reference far from both) and axes (``near_int``,
    ``near_cross``). The city dummies absorb each city's base rate and OSM coverage, so the coefficients compare
    labels within a city at equal confidence.
    """
    rows = []
    cities = pd.get_dummies(v.city, drop_first=True, dtype=float)
    for t in thresholds:
        dm = context_dummies(v, t)
        for spec, cols in (("context", ["near_both", "far_one"]), ("axes", ["near_int", "near_cross"])):
            X = pd.concat([pd.Series(1.0, index=v.index, name="const"), v.conf_logit, dm[cols], cities], axis=1)
            fit = logit_fit(X.values, v.y.values, list(X.columns))
            for term in cols:
                r = fit.loc[term]
                rows.append({"threshold_m": t, "spec": spec, "term": term, "coef": r.coef, "se": r.se, "p": r.p,
                             "share": float(dm[term].mean()), "n": len(v), "cities": v.city.nunique()})
    return pd.DataFrame(rows)


def coarse_context_lift(v, far_m=FAR_CROSSING_M, min_city_n=100, n_boot=500):
    """Leave-one-city-out lift of ``conf_logit`` + the two context dummies over ``conf_logit`` alone.

    This is the coarse counterpart of ``lift``'s binned model: two indicators instead of 17 bin columns.
    """
    d = pd.concat([v, context_dummies(v, far_m)[["near_both", "far_one"]]], axis=1)
    return lift(d, ["conf_logit"], ["near_both", "far_one"], min_city_n, n_boot)


def quartile_context(v):
    """Confirm rate and n by ``oof_accuracy`` quartile and road context (25 m), one row per quartile."""
    q = pd.qcut(v.oof_accuracy, 4, duplicates="drop").astype(str)
    ctx = road_context(v)
    g = v.groupby([q, ctx]).y.agg(["mean", "size"]).unstack()
    out = pd.DataFrame(index=g.index)
    for c in ("far_both", "far_one", "near_both"):
        if ("mean", c) in g:
            out[c] = [f"{m:.2f} ({int(n)})" if n == n else "" for m, n in zip(g[("mean", c)], g[("size", c)])]
    return out.reset_index().rename(columns={"oof_accuracy": "oof_accuracy_quartile"})


def city_routing_table(df, cast, min_n=100, far_m=FAR_CROSSING_M):
    """Per public city: cast Disagrees, their road-context shares, OSM crossing coverage, the validated share and
    the confirm rates near both / far from both. Coverage = share of the city's CurbRamp labels (any vote count)
    within ``far_m`` of a mapped OSM crossing."""
    cr = df[df.label_type == "CurbRamp"].dropna(subset=["d_crossing_m"])
    cover = cr.groupby("city").d_crossing_m.apply(lambda x: float((x < far_m).mean()))
    rows = []
    total = len(cast)
    for city, g in cast.groupby("city"):
        ctx = road_context(g, far_m)
        val = g[g.validated]
        vctx = road_context(val, far_m)
        r = {"city": city, "cast_disagrees": len(g), "share_of_all_cast": len(g) / total,
             "near_both": float((ctx == "near_both").mean()), "far_one": float((ctx == "far_one").mean()),
             "far_both": float((ctx == "far_both").mean()), "crossing_coverage": cover.get(city, np.nan),
             "validated_share": float(g.validated.mean())}
        for c in ("near_both", "far_both"):
            sub = val[vctx == c]
            r[f"confirm_{c}"] = float(sub.y.mean()) if len(sub) else np.nan
            r[f"n_validated_{c}"] = len(sub)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("cast_disagrees", ascending=False)
    return t[t.cast_disagrees >= min_n]


def osm_coverage_lines(df):
    """Markdown lines: how much of the CurbRamp data has a mapped driveway within 50 m / a crossing within 25 m."""
    cr = df[(df.label_type == "CurbRamp") & (df.h_n >= 1)].dropna(subset=["n_driveways_50m"])
    cr3 = analysis_set(df, "CurbRamp", 3).dropna(subset=["n_driveways_50m"])
    return ["## OSM coverage", "",
            f"CurbRamp labels with any mapped `service=driveway` way within 50 m: {(cr3.n_driveways_50m > 0).mean():.1%}"
            f" of the >= 3-vote analysis set (n = {len(cr3):,}), {(cr.n_driveways_50m > 0).mean():.1%} of all "
            f"CurbRamp labels with a human vote (n = {len(cr):,}). Within 25 m of a mapped crossing: "
            f"{(cr3.d_crossing_m < FAR_CROSSING_M).mean():.1%} and {(cr.d_crossing_m < FAR_CROSSING_M).mean():.1%}.",
            ""]


def routing_report(df, cities, n_boot=500, min_city_n=100):
    """Every number behind the routing conclusion of ``docs/road-prior-2026-10.md``, as ``(markdown, tables)``."""
    df = public_only(df, cities)
    tables, md = {}, ["# Road prior: routing (cast CurbRamp Disagree votes, public cities)", ""]
    if "n_driveways_50m" in df:
        md += osm_coverage_lines(df)
    cast = cast_disagrees(df, 3)
    md += [f"Cast CurbRamp Disagree votes with both distances: {len(cast):,} in {cast.city.nunique()} cities. "
           f"Validated (>= 3 human votes, strict majority): {int(cast.validated.sum()):,} "
           f"({cast.validated.mean():.1%}). `y` = the human majority confirms the Disagree.", ""]
    t = city_routing_table(df, cast, min_n=min_city_n)
    tables["routing_by_city"] = t
    md += [f"## Per city (>= {min_city_n} cast Disagrees; context at {FAR_CROSSING_M} m)", "",
           "`crossing_coverage` is the share of the city's CurbRamp labels within 25 m of a mapped OSM crossing. "
           "Where it is low, \"far from a crossing\" mostly means the crossing is not mapped.", "", _md(t), ""]
    ctx_all = road_context(cast)
    md += ["Share of all cast Disagrees by context: " + ", ".join(
        f"{k} {v:.1%}" for k, v in ctx_all.value_counts(normalize=True).sort_index().items()), ""]
    for mv in ROUTING_MIN_VOTES:
        v = cast_disagrees(df, mv)
        v = v[v.validated]
        q = quartile_context(v)
        tables[f"quartile_context_minvotes{mv}"] = q
        md += [f"## Confirm rate by confidence quartile and context, >= {mv} human votes (n = {len(v):,})", "",
               "Cells are confirm rate (n). Quartiles are of `oof_accuracy` within this set.", "", _md(q), ""]
    v3 = cast[cast.validated]
    fe = pd.concat([city_fe_models(v3).assign(min_votes=3), city_fe_models(cast_disagrees(df, 1)
                    .query("validated")).assign(min_votes=1)])
    tables["city_fe"] = fe
    md += ["## City-fixed-effects logistic, y on conf_logit + context + city dummies", "",
           "Coefficients are log-odds of a confirmed Disagree. `context`: near_both and far_one against far from "
           "both. `axes`: near an intersection node, near an OSM crossing.", "",
           _md(fe[["min_votes", "threshold_m", "spec", "term", "coef", "se", "p", "share", "n", "cities"]], 4), ""]
    pooled, per_city, _ = coarse_context_lift(v3, min_city_n=min_city_n, n_boot=n_boot)
    tables["coarse_lift_by_city"] = per_city
    tables["coarse_lift"] = pd.DataFrame([pooled])
    md += ["## Coarse context model, leave-one-city-out (validated, >= 3 votes)", "",
           "Baseline = logistic on conf_logit; full = baseline + near_both + far_one. Same LOCO and paired "
           "bootstrap as `evaluate`.", "",
           _md(pd.DataFrame([pooled])[["n", "cities", "auc_base", "auc_full", "d_auc", "d_auc_lo", "d_auc_hi",
                                        "log_loss_base", "log_loss_full", "d_log_loss", "d_log_loss_lo",
                                        "d_log_loss_hi"]], 4), "",
           f"Per held-out city (n >= {min_city_n}):", "", _md(per_city.sort_values("n", ascending=False)), ""]
    if "d_deadend_m" in v3 and v3.d_deadend_m.notna().any():
        fb = v3[road_context(v3) == "far_both"]
        far_int = df[(df.label_type == "CurbRamp") & (df.d_intersection_m >= FAR_CROSSING_M)].dropna(
            subset=["d_deadend_m"])
        dd = far_int.groupby("city").d_deadend_m.apply(lambda x: float((x < DEADEND_NEAR_M).mean()))
        md += [f"## Audit-area edges: dead ends within {DEADEND_NEAR_M} m", "",
               f"Validated cast Disagrees far from both with a degree-1 node within {DEADEND_NEAR_M} m: "
               f"{float((fb.d_deadend_m < DEADEND_NEAR_M).mean()):.1%} of {len(fb):,}.", "",
               f"Share of CurbRamp labels >= {FAR_CROSSING_M} m from an intersection node that have a dead end "
               f"within {DEADEND_NEAR_M} m, largest cities:", "",
               _md(dd.reindex(far_int.city.value_counts().index[:8]).rename("share").reset_index()), ""]
    return "\n".join(md), tables


# ---------------------------------------------------------------------------------------------------------------------
# Provenance: pin a run's inputs and outputs so a later rerun can be checked against it
# ---------------------------------------------------------------------------------------------------------------------

VERIFIED_EXTS = (".csv", ".md")  # figures are left out: PNG bytes carry matplotlib metadata


def collect_provenance(run_dirs):
    """``{run_dir_name: {"stage", "git_commit", "packages", "inputs": {name: sha256}, "outputs": {file: sha256}}}``.

    Paths are reduced to names, so the record holds no home-directory path or label-level data, only digests.
    Label-level outputs (the features CSV) appear by digest only.
    """
    prov = {}
    for d in run_dirs:
        with open(os.path.join(d, "run_manifest.json")) as f:
            m = json.load(f)
        outs = {}
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if os.path.isfile(full) and name.endswith(VERIFIED_EXTS):
                outs[name] = sha256_file(full)
        for k in ("key_sha256", "packet_sha256"):
            if k in m:
                outs[k] = m[k]
        if "output" in m:
            outs[os.path.basename(m["output"]["path"])] = m["output"]["sha256"]
        prov[os.path.basename(os.path.normpath(d))] = {
            "stage": m["stage"], "git_commit": m["git_commit"], "git_dirty": m["git_dirty"],
            "python": m["python"], "packages": m["packages"],
            "inputs": {k: v["sha256"] for k, v in m["inputs"].items()}, "outputs": outs}
    return prov


def diff_provenance(expected, actual):
    """Human-readable mismatches between two ``collect_provenance`` records (empty list = identical).

    >>> diff_provenance({"r": {"inputs": {"a": "1"}, "outputs": {"x.csv": "2"}}},
    ...                 {"r": {"inputs": {"a": "1"}, "outputs": {"x.csv": "3"}}})
    ['r: output x.csv differs']
    """
    out = []
    for run, e in expected.items():
        a = actual.get(run)
        if a is None:
            out.append(f"{run}: not rerun")
            continue
        for kind in ("inputs", "outputs"):
            for k, v in e.get(kind, {}).items():
                if a.get(kind, {}).get(k) != v:
                    out.append(f"{run}: {kind[:-1]} {k} {'missing' if k not in a.get(kind, {}) else 'differs'}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Hand-coding packet
# ---------------------------------------------------------------------------------------------------------------------

PACKET_BUCKETS = [0, 15, 25, 40, np.inf]
# The rater's sheet carries only the item number and the Street View link. The Project Sidewalk share page shows
# earlier validations to admins and offers a vote, and the label id carries the city, so both live in the key.
PACKET_COLUMNS = ["item", "gsv_link", "code", "notes"]
KEY_COLUMNS = ["item", "label_uid", "ps_label_page", "pool", "reason", "distance_bucket", "d_intersection_m", "d_crossing_m", "ai_result",
               "oof_accuracy", "h_agree", "h_dis"]
RUBRIC = ("driveway", "real_ramp", "other", "cant_tell")


def contested_curbramps(df, min_votes=3, experts=()):
    """CurbRamp labels the AI got arguably wrong: AI Disagree, or AI Agree overturned by the human majority (on
    ``>= min_votes``) or by an expert. Adds ``reason``."""
    cr = df[(df.label_type == "CurbRamp") & df.ai_result.isin(["Agree", "Disagree"])].copy()
    over_h = (cr.ai_result == "Agree") & (cr.h_n >= min_votes) & (cr.h_maj == "Disagree")
    over_e = pd.Series(False, index=cr.index)
    for e in experts:
        over_e |= (cr.ai_result == "Agree") & cr[e].eq("Disagree")
    cr["reason"] = np.select([cr.ai_result == "Disagree", over_e, over_h],
                             ["ai_disagree", "agree_overturned_expert", "agree_overturned_crowd"], "")
    return cr[cr.reason != ""]


def stratified_sample(cand, n_total, seed, buckets=PACKET_BUCKETS, group_col=None):
    """About ``n_total`` rows, equal per stratum (distance bucket, times ``group_col`` if given), spread across
    cities, then shuffled.

    Within a stratum, cities take turns (each city's rows in seeded random order), so no one city fills it. A
    stratum with too few rows gives what it has and the shortfall is filled from the remaining rows in the same
    city-turn order, so the packet still has ``n_total`` items when the pool allows.

    >>> c = pd.DataFrame({"label_uid": [f"a:{i}" for i in range(40)], "city": ["a", "b"] * 20,
    ...                   "d_intersection_m": np.repeat([5.0, 20.0, 30.0, 50.0], 10)})
    >>> s = stratified_sample(c, 8, seed=1)
    >>> len(s), s.distance_bucket.value_counts().tolist(), s.city.value_counts().tolist()
    (8, [2, 2, 2, 2], [4, 4])
    """
    c = cand.dropna(subset=["d_intersection_m"]).sort_values("label_uid").copy()
    c["distance_bucket"] = pd.cut(c.d_intersection_m, buckets, right=False).astype(str)
    strata = ["distance_bucket"] + ([group_col] if group_col else [])
    n_strata = (len(buckets) - 1) * (c[group_col].nunique() if group_col else 1)
    per = int(np.ceil(n_total / max(n_strata, 1)))
    rng = np.random.default_rng(seed)
    c["_r"] = rng.random(len(c))
    c = c.sort_values("_r")
    c["_turn"] = c.groupby(strata + ["city"]).cumcount()
    c = c.sort_values(["_turn", "_r"])
    picked = c.groupby(strata, group_keys=False).head(per)
    if len(picked) < n_total:
        picked = pd.concat([picked, c.drop(picked.index).head(n_total - len(picked))])
    picked = picked.sample(frac=1.0, random_state=int(rng.integers(0, 2 ** 31 - 1))).head(n_total)
    return picked.drop(columns=["_r", "_turn"]).reset_index(drop=True)


def inside_git_checkout(path):
    """True if ``path`` (or its nearest existing parent) is inside any git work tree."""
    import subprocess
    p = os.path.abspath(path)
    while not os.path.isdir(p):
        p = os.path.dirname(p)
    r = subprocess.run(["git", "-C", p, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def build_packet(df, urls, seed, n_total=200, min_votes=3, experts=()):
    """``(packet, key)`` DataFrames: the rater's sheet (no verdicts, no distances) and the unblinding key."""
    cand = contested_curbramps(df, min_votes, experts)
    # the AI-Disagree pool is ~100x the overturned-Agree pool, so the two are sampled as separate halves
    cand["pool"] = np.where(cand.reason == "ai_disagree", "ai_disagree", "agree_overturned")
    s = stratified_sample(cand, n_total, seed, group_col="pool")
    s["item"] = np.arange(1, len(s) + 1)
    s["ps_label_page"] = [f"{urls[c]}/label/{i}" for c, i in zip(s.city, s.label_id)]
    s["gsv_link"] = s.get("pano_url", pd.Series("", index=s.index)).fillna("")
    s["code"] = ""
    s["notes"] = ""
    return s[PACKET_COLUMNS], s[[c for c in KEY_COLUMNS if c in s.columns]]


def sha256_dir(path, pattern_ext=(".csv", ".json", ".gz")):
    """One digest over every data file under ``path`` (relative path + file sha256, sorted), and the file count.

    Used to pin a cache directory (rawLabels, streets, OSM) in a run manifest without listing every file.
    """
    entries = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(pattern_ext):
                full = os.path.join(root, f)
                entries.append(f"{os.path.relpath(full, path)}  {sha256_file(full)}")
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode()).hexdigest(), len(entries)


def run_manifest(args, inputs):
    """Provenance of one stage run: code commit, argv, library versions, and a digest of every input.

    ``inputs`` maps a name to a file or directory path (``None`` entries are skipped). A rerun that reproduces the
    digests and the commit must reproduce the outputs: every random step here has a fixed seed, except the packet,
    whose seed is kept with its key.
    """
    import platform
    import subprocess
    import matplotlib
    import scipy
    import sklearn
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(["git", "-C", here, *a], capture_output=True, text=True).stdout.strip()
    m = {"stage": args.cmd, "argv": sys.argv[1:], "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", here)),
         "python": platform.python_version(),
         "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                      "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
         "inputs": {}}
    for name, path in inputs.items():
        if not path:
            continue
        if os.path.isdir(path):
            digest, n = sha256_dir(path)
            m["inputs"][name] = {"path": path, "kind": "dir", "files": n, "sha256": digest}
        else:
            m["inputs"][name] = {"path": path, "kind": "file", "bytes": os.path.getsize(path),
                                 "sha256": sha256_file(path)}
    return m


def write_run_manifest(out_dir, args, inputs, extra=None):
    """Write ``run_manifest.json`` (see ``run_manifest``) into ``out_dir``."""
    m = run_manifest(args, inputs)
    if extra:
        m.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def cmd_fetch_streets(args):
    cities = public_cities(args.cities_json)
    m = fetch_streets(cities, args.out_dir, timeout=args.timeout)
    bad = {k: v for k, v in m.items() if v.get("status") != "ok"}
    print(f"{len(m) - len(bad)} of {len(m)} cities ok; failed: {json.dumps(bad, indent=1) if bad else 'none'}")
    rows = []
    for cid in sorted(m):
        path = os.path.join(args.out_dir, f"{cid}.csv")
        if m[cid].get("status") == "ok" and os.path.exists(path):
            st = pd.read_csv(path, usecols=["street_edge_id", "way_type", "start_point", "end_point"])
            rows.append({"city": cid, "edges": len(st), **intersection_summary(street_nodes(st)),
                         "service_edge_share": float((st.way_type == "service").mean())})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(args.out_dir, "intersections_summary.csv"), index=False)
        print(_md(t))


def cmd_fetch_osm(args):
    cities = public_cities(args.cities_json)
    only = set(args.only.split(",")) if args.only else None
    m = fetch_osm(args.rawlabels, cities, args.out_dir, pad_m=args.pad_m, overpass_urls=args.overpass_url, only=only)
    bad = {k: v for k, v in m.items() if v.get("status") != "ok"}
    print(f"{len(m) - len(bad)} of {len(m)} cities ok; not ok: {json.dumps(bad, indent=1) if bad else 'none'}")


def _experts(arg):
    return [c.strip() for c in arg.split(",") if c.strip()] if arg else []


def cmd_features(args):
    oof = pd.read_csv(args.labels_oof) if args.labels_oof else None
    avh = pd.read_csv(args.ai_vs_humans, low_memory=False) if args.ai_vs_humans else None
    experts = _experts(args.experts)
    if experts and avh is None:
        sys.exit("--experts needs --ai-vs-humans")
    if avh is not None and sorted(set(experts) - set(avh.columns)):
        sys.exit("some --experts columns are not in --ai-vs-humans")
    df, report = build_features(args.rawlabels, args.streets_dir, args.cities_json, args.osm_dir, oof, avh,
                                experts, min_votes=args.min_votes)
    if inside_git_checkout(args.out):
        sys.exit(f"refusing to write label-level rows inside a git checkout: {args.out}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    write_run_manifest(os.path.dirname(os.path.abspath(args.out)), args,
                       {"cities_json": args.cities_json, "rawlabels": args.rawlabels, "streets_dir": args.streets_dir,
                        "osm_dir": args.osm_dir, "labels_oof": args.labels_oof, "ai_vs_humans": args.ai_vs_humans},
                       extra={"output": {"path": args.out, "rows": len(df), "sha256": sha256_file(args.out)}})
    for cid, r in report.items():
        print(f"  {cid}: {r}")
    print(f"{len(df):,} labels -> {args.out}; by type: {df.label_type.value_counts().to_dict()}")


def _read_features(path):
    return pd.read_csv(path, low_memory=False)


def cmd_evaluate(args):
    df = _read_features(args.features)
    print(evaluate(df, load_cities(args.cities_json), args.out_dir, args.fig_dir, args.min_votes, args.min_city_n,
                   args.n_boot))
    write_run_manifest(args.out_dir, args, {"features": args.features, "cities_json": args.cities_json})


def cmd_stratify(args):
    df = public_only(_read_features(args.features), load_cities(args.cities_json))
    experts = sorted(c for c in df.columns if c.startswith("expert") and c[6:].isdigit())
    t = far_shares(df, production_pools(df, args.min_votes, experts))
    os.makedirs(args.out_dir, exist_ok=True)
    t.to_csv(os.path.join(args.out_dir, "stratify.csv"), index=False)
    text = (f"# Road prior: stratify (CurbRamp with a production verdict, min human votes {args.min_votes})\n\n"
            + _md(t) + "\n")
    for name, sub, col in agree_error_tables(df, args.min_votes):
        b = binned_rate(sub, col, ROUTING_BINS)
        tag = "gated" if "oof_accuracy" in name else "all"
        b.to_csv(os.path.join(args.out_dir, f"agree_error_{tag}_{col}.csv"), index=False)
        text += (f"\n## {name}: human-majority Disagree rate by {col} (n = {len(sub.dropna(subset=[col])):,})\n\n"
                 + _md(b, 4) + "\n")
    if "d_crossing_m" in df and df.d_crossing_m.notna().any():
        for cast, tag, title in ((False, "all", "AI Disagree assessments"), (True, "cast", "cast AI Disagree votes")):
            c = disagree_context_table(df, args.min_votes, cast_only=cast)
            c.to_csv(os.path.join(args.out_dir, f"disagree_context_{tag}.csv"), index=False)
            text += (f"\n## {title}: human-majority confirm rate by road context within oof_accuracy quartiles "
                     f"(far = >= {FAR_CROSSING_M} m from the nearest intersection node and OSM crossing)\n\n"
                     + _md(c.round(3), 3) + "\n")
    with open(os.path.join(args.out_dir, "stratify.md"), "w") as f:
        f.write(text)
    write_run_manifest(args.out_dir, args, {"features": args.features, "cities_json": args.cities_json})
    print(text)


def cmd_routing(args):
    md, tables = routing_report(_read_features(args.features), load_cities(args.cities_json), n_boot=args.n_boot)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(args.out_dir, "routing.md"), "w") as f:
        f.write(md)
    write_run_manifest(args.out_dir, args, {"features": args.features, "cities_json": args.cities_json})
    print(md)


def cmd_provenance(args):
    prov = collect_provenance(args.run_dir)
    with open(args.out, "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(prov)} runs -> {args.out}")


def cmd_verify(args):
    with open(args.provenance) as f:
        expected = json.load(f)
    actual = collect_provenance(args.run_dir)
    problems = diff_provenance({k: v for k, v in expected.items() if k in actual or not args.partial}, actual)
    for k, a in actual.items():
        e = expected.get(k, {})
        if e and (e.get("packages") != a["packages"] or e.get("python") != a["python"]):
            print(f"note: {k} ran with python {a['python']} {a['packages']}; recorded {e.get('python')} "
                  f"{e.get('packages')}. Small float differences can come from library versions.")
    if problems:
        print("MISMATCH:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"all {len(actual)} runs match {args.provenance}")


def cmd_packet(args):
    if inside_git_checkout(args.out_dir) or inside_git_checkout(args.key_dir):
        sys.exit("refusing to write the packet or its key inside a git checkout")
    cities = load_cities(args.cities_json)
    df = public_only(_read_features(args.features), cities)
    experts = sorted(c for c in df.columns if c.startswith("expert") and c[6:].isdigit())
    packet, key = build_packet(df, public_cities(args.cities_json), args.seed, args.n, args.min_votes, experts)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.key_dir, exist_ok=True)
    ppath = os.path.join(args.out_dir, "packet.csv")
    kpath = os.path.join(args.key_dir, "packet_key.csv")
    packet.to_csv(ppath, index=False)
    key.to_csv(kpath, index=False)
    digest = sha256_file(kpath)
    with open(kpath + ".sha256", "w") as f:
        f.write(f"{digest}  packet_key.csv\n")
    # the manifest goes with the key: it records the seed's location, never the seed
    write_run_manifest(args.key_dir, args, {"features": args.features, "cities_json": args.cities_json},
                       extra={"argv": [a if a != str(args.seed) else "<seed>" for a in sys.argv[1:]],
                              "key_sha256": digest, "packet_sha256": sha256_file(ppath)})
    print(f"{len(packet)} items -> {ppath}\nkey -> {kpath}\nsha256 {digest}")
    print("by reason:", key.reason.value_counts().to_dict())
    print("by bucket:", key.distance_bucket.value_counts().sort_index().to_dict())
    print("cities:", key.label_uid.str.split(":").str[0].nunique())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    cj = "saved /v3/api/cities response; only public cities are used or named"

    s = sub.add_parser("fetch-streets", help="download each public city's streets CSV (dated cache dir)")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--out-dir", required=True, help="cache directory, e.g. $D/streets_cache/2026-10-20")
    s.add_argument("--timeout", type=int, default=600, help="per-request timeout in seconds")
    s.set_defaults(func=cmd_fetch_streets)

    s = sub.add_parser("fetch-osm", help="pull OSM crossings and service ways per city (Overpass API)")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--rawlabels", required=True, help="rawLabels cache dir (<city_id>.csv); sets each city's bbox")
    s.add_argument("--out-dir", required=True, help="cache directory, e.g. $D/osm_cache/2026-10-20")
    s.add_argument("--pad-m", type=float, default=200.0, help="padding of each query tile in metres")
    s.add_argument("--overpass-url", action="append", default=None,
                   help="Overpass endpoint(s) tried in order; repeat the flag (default: overpass-api.de, "
                        "overpass.private.coffee, overpass.kumi.systems)")
    s.add_argument("--only", default=None, help="comma list of city_ids to pull (default: all public)")
    s.set_defaults(func=cmd_fetch_osm)

    s = sub.add_parser("features", help="label-level features table (PRIVATE output; keep under ~/Data)")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--rawlabels", required=True, help="rawLabels cache dir (<city_id>.csv)")
    s.add_argument("--streets-dir", required=True, help="fetch-streets output directory")
    s.add_argument("--osm-dir", default=None, help="fetch-osm output directory (optional)")
    s.add_argument("--labels-oof", default=None, help="calibrate_from_production's labels_oof.csv (AI verdicts)")
    s.add_argument("--ai-vs-humans", default=None, help="analysis/analyze.py's ai_vs_humans.csv (expert votes)")
    s.add_argument("--experts", default=None,
                   help="comma list of the export's expert vote columns; renamed expert1, expert2, ... by position")
    s.add_argument("--min-votes", type=int, default=1, help="keep labels with at least this many human votes")
    s.add_argument("--out", required=True, help="output CSV (outside any git checkout)")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("evaluate", help="AUCs, LOCO lift, plots (aggregates; public cities named)")
    s.add_argument("--features", required=True, help="features CSV")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--out-dir", required=True, help="tables (evaluate.md and CSVs)")
    s.add_argument("--fig-dir", default=None, help="figures (default: --out-dir)")
    s.add_argument("--min-votes", type=int, default=3, help="human Agree/Disagree votes for the analysis set")
    s.add_argument("--min-city-n", type=int, default=100, help="minimum labels for a per-city row")
    s.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for CIs")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("stratify", help="far-from-intersection shares of the production verdict pools")
    s.add_argument("--features", required=True, help="features CSV")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--out-dir", required=True, help="stratify.md / stratify.csv")
    s.add_argument("--min-votes", type=int, default=3, help="human votes for the human-majority pools")
    s.set_defaults(func=cmd_stratify)

    s = sub.add_parser("routing", help="cast CurbRamp Disagrees by road context: per city, quartiles, city-FE, LOCO")
    s.add_argument("--features", required=True, help="features CSV")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--out-dir", required=True, help="routing.md, CSVs and run_manifest.json")
    s.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for the LOCO CIs")
    s.set_defaults(func=cmd_routing)

    s = sub.add_parser("provenance", help="record input and output digests of finished runs (no paths, no rows)")
    s.add_argument("--run-dir", action="append", required=True, help="a stage output dir with run_manifest.json")
    s.add_argument("--out", required=True, help="e.g. docs/road-prior-2026-10/provenance.json")
    s.set_defaults(func=cmd_provenance)

    s = sub.add_parser("verify", help="check rerun outputs against a recorded provenance file")
    s.add_argument("--provenance", required=True, help="the recorded provenance.json")
    s.add_argument("--run-dir", action="append", required=True, help="a rerun's stage output dir")
    s.add_argument("--partial", action="store_true", help="only check the runs given, not every recorded run")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("packet", help="blinded hand-coding packet of contested CurbRamp labels")
    s.add_argument("--features", required=True, help="features CSV")
    s.add_argument("--cities-json", required=True, help=cj)
    s.add_argument("--out-dir", required=True, help="where packet.csv goes (outside any git checkout)")
    s.add_argument("--key-dir", required=True, help="where packet_key.csv goes (private; outside any git checkout)")
    s.add_argument("--seed", type=int, required=True, help="secret seed; no default on purpose")
    s.add_argument("--n", type=int, default=200, help="packet size")
    s.add_argument("--min-votes", type=int, default=3, help="human votes for a crowd overturn")
    s.set_defaults(func=cmd_packet)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
