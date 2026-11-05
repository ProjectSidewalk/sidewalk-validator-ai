"""Download rawLabels CSV for every public city into cache/, verifying the last row parses.
Falls back to bbox tiles if the whole-city export truncates.

Usage::

    python analysis/fetch_all.py CACHE_DIR                                   # every label type
    python analysis/fetch_all.py CACHE_DIR --label-types NoSidewalk,Signal   # only these types (issue #42)

``--label-types`` asks the API for each type separately (``/v3/api/rawLabels?filetype=csv&labelType=<type>``, checked
2026-11-05) and writes their rows into the same ``<city_id>.csv`` the full pull writes, so ``data_generation.py
--rawlabels-cache`` and ``split_data.py --rawlabels`` read a type-filtered cache unchanged. ``cities.json`` then
records ``label_types``, and a cache made with one set of types is never topped up with another (a mixed cache would
silently lose rows). Without the flag the requests, files and messages are exactly those of the earlier script.
"""
import argparse, csv, io, json, os, sys, time
from datetime import date

import requests

CITIES_API = "https://sidewalk-teaneck.cs.washington.edu/v3/api/cities"


def ok(text):
    """True if the export is complete: it ends with a newline, parses strictly (no unterminated quoted field), and its
    last rows are as wide as its own header. Returns ``(complete, data_rows)``."""
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error:
        return False, 0
    complete = text.endswith("\n") and len(rows) > 1 and all(len(r) == len(rows[0]) for r in rows[-3:])
    return complete, len(rows) - 1


def parse_label_types(value):
    """``"NoSidewalk, Signal"`` -> ``["NoSidewalk", "Signal"]`` (order kept, duplicates dropped); None or blank -> None.

    >>> parse_label_types(" NoSidewalk,Signal,,NoSidewalk"), parse_label_types("")
    (['NoSidewalk', 'Signal'], None)
    """
    if not value:
        return None
    out = []
    for t in value.split(","):
        t = t.strip()
        if t and t not in out:
            out.append(t)
    return out or None


def type_query(label_type):
    """The query-string suffix for one label type, or ``""`` for the full export.

    >>> type_query(None), type_query("NoSidewalk")
    ('', '&labelType=NoSidewalk')
    """
    return f"&labelType={label_type}" if label_type else ""


def recorded_label_types(cache):
    """``(has_listing, label_types)`` from an existing ``<cache>/cities.json`` (``label_types`` None for a full pull)."""
    path = os.path.join(cache, "cities.json")
    if not os.path.exists(path):
        return False, None
    with open(path) as f:
        return True, json.load(f).get("label_types")


def fetch_city(get, c, label_type):
    """One label type (all types with ``label_type`` None) of one city.

    Returns ``(text, n)`` for a complete whole-city export, else ``(None, (header, rows))`` from a 4x4 bbox tiling of
    the city's bounds (``header`` None when no tile returned data; ``rows`` keyed by ``(city_id, label_id)``)."""
    cid, url = c["city_id"], c["url"].rstrip("/")
    q = type_query(label_type)
    try:
        r = get(f"{url}/v3/api/rawLabels?filetype=csv{q}", timeout=900)
        good, n = ok(r.text)
    except requests.RequestException as e:   # a dropped connection on a big export: try the tiles instead
        print(f"{cid}: whole-city request failed ({e}), tiling", flush=True)
        good, n = False, 0
    if good:
        return r.text, n
    # bbox tile fallback: 4x4 grid over city bounds
    print(f"{cid}: whole-city export truncated ({n} rows), tiling", flush=True)
    # Labels on a tile edge can come back from two tiles, so rows are deduplicated on (city, label_id).
    b = c["bounds"]; header = None; rows = {}
    for i in range(4):
        for j in range(4):
            s = b["south"] + (b["north"]-b["south"])*i/4; nn = b["south"] + (b["north"]-b["south"])*(i+1)/4
            w = b["west"] + (b["east"]-b["west"])*j/4; e = b["west"] + (b["east"]-b["west"])*(j+1)/4
            try:
                rr = get(f"{url}/v3/api/rawLabels?filetype=csv{q}&bbox={w},{s},{e},{nn}", timeout=900)
            except requests.RequestException as ex:
                print(f"  tile {i},{j} failed ({ex}); city incomplete", flush=True); continue
            g, k = ok(rr.text)
            if not g: print(f"  tile {i},{j} still bad ({k})", flush=True)
            tile = list(csv.reader(io.StringIO(rr.text)))
            if not tile: continue
            if header is None: header = tile[0]
            key = header.index("label_id")
            for r in tile[1:]:
                if len(r) == len(header): rows[(cid, r[key])] = r
    return None, (header, rows)


def write_rows(out, header, rows):
    with open(out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)


def main(argv=None, get=None):
    """``get`` is ``requests.get`` unless a test passes a stub."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cache", help="directory to fill with <city_id>.csv and cities.json")
    p.add_argument("--label-types", default=None,
                   help="comma list of label types to pull (the labelType query parameter); default: every type")
    args = p.parse_args(argv)
    get = get or requests.get
    cache = args.cache
    types = parse_label_types(args.label_types)
    has_listing, recorded = recorded_label_types(cache)
    if has_listing and recorded != types and any(n.endswith(".csv") for n in os.listdir(cache)):
        sys.exit(f"{cache} holds CSVs pulled for label types {recorded or 'all'}; this pull asks for "
                 f"{types or 'all'}. Use an empty directory: a cache holds one set of label types.")

    cities = get(CITIES_API, timeout=60).json()["cities"]
    pub = [c for c in cities if c["visibility"] == "public" and c["url"]]
    print(f"{len(pub)} public cities", flush=True)
    # The shape data_generation.py reads and writes as <rawlabels-cache>/cities.json.
    listing = {"fetched_on": date.today().isoformat(), "url": CITIES_API, "cities": pub}
    if types:
        listing["label_types"] = types
    json.dump(listing, open(os.path.join(cache, "cities.json"), "w"), indent=1)

    for c in pub:
        cid = c["city_id"]
        out = os.path.join(cache, f"{cid}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"{cid}: cached", flush=True); continue
        t = time.time()
        if not types:   # the full export, exactly as before --label-types existed
            text, n = fetch_city(get, c, None)
            if text is not None:
                open(out, "w").write(text)
                print(f"{cid}: {n} labels {len(text)/1e6:.1f}MB {time.time()-t:.0f}s", flush=True)
                continue
            header, rows = n
            if header is None:
                print(f"{cid}: no tile returned data; skipped", flush=True); continue
            write_rows(out, header, rows.values())
            print(f"{cid}: tiled, {len(rows)} unique labels {time.time()-t:.0f}s", flush=True)
            continue
        # One export (or tiling) per type; the rows of every type go into one <city_id>.csv under one header.
        header, rows = None, []
        for lt in types:
            text, n = fetch_city(get, c, lt)
            if text is not None:
                part = list(csv.reader(io.StringIO(text)))
                part_header, part_rows = part[0], part[1:]
            else:
                part_header, tiled = n
                if part_header is None:
                    print(f"{cid}: {lt}: no tile returned data", flush=True); continue
                part_rows = list(tiled.values())
            if header is None:
                header = part_header
            elif part_header != header:
                sys.exit(f"{cid}: the {lt} export's header differs from the other types'; not mixing them")
            rows.extend(part_rows)
            print(f"{cid}: {lt}: {len(part_rows)} labels", flush=True)
        if header is None:
            print(f"{cid}: no label type returned data; skipped", flush=True); continue
        write_rows(out, header, rows)
        print(f"{cid}: {len(rows)} labels of {', '.join(types)} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
