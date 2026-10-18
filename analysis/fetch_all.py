"""Download rawLabels CSV for every public city into cache/, verifying the last row parses.
Falls back to bbox tiles if the whole-city export truncates."""
import csv, io, json, os, sys, time, requests
CACHE = sys.argv[1]
cities = requests.get("https://sidewalk-teaneck.cs.washington.edu/v3/api/cities", timeout=60).json()["cities"]
pub = [c for c in cities if c["visibility"] == "public" and c["url"]]
print(f"{len(pub)} public cities", flush=True)
json.dump(pub, open(os.path.join(CACHE, "cities.json"), "w"), indent=1)
def ok(text):
    """True if the export is complete: it ends with a newline, parses strictly (no unterminated quoted field), and its
    last rows are as wide as its own header. Returns ``(complete, data_rows)``."""
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error:
        return False, 0
    complete = text.endswith("\n") and len(rows) > 1 and all(len(r) == len(rows[0]) for r in rows[-3:])
    return complete, len(rows) - 1

for c in pub:
    cid, url = c["city_id"], c["url"].rstrip("/")
    out = os.path.join(CACHE, f"{cid}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"{cid}: cached", flush=True); continue
    t = time.time()
    try:
        r = requests.get(f"{url}/v3/api/rawLabels?filetype=csv", timeout=900)
        good, n = ok(r.text)
    except requests.RequestException as e:   # a dropped connection on a big export: try the tiles instead
        print(f"{cid}: whole-city request failed ({e}), tiling", flush=True)
        good, n = False, 0
    if good:
        open(out, "w").write(r.text)
        print(f"{cid}: {n} labels {len(r.text)/1e6:.1f}MB {time.time()-t:.0f}s", flush=True)
        continue
    # bbox tile fallback: 4x4 grid over city bounds
    print(f"{cid}: whole-city export truncated ({n} rows), tiling", flush=True)
    # Labels on a tile edge can come back from two tiles, so rows are deduplicated on (city, label_id).
    b = c["bounds"]; header = None; rows = {}
    for i in range(4):
        for j in range(4):
            s = b["south"] + (b["north"]-b["south"])*i/4; nn = b["south"] + (b["north"]-b["south"])*(i+1)/4
            w = b["west"] + (b["east"]-b["west"])*j/4; e = b["west"] + (b["east"]-b["west"])*(j+1)/4
            try:
                rr = requests.get(f"{url}/v3/api/rawLabels?filetype=csv&bbox={w},{s},{e},{nn}", timeout=900)
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
    if header is None:
        print(f"{cid}: no tile returned data; skipped", flush=True); continue
    with open(out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(header); wr.writerows(rows.values())
    print(f"{cid}: tiled, {len(rows)} unique labels {time.time()-t:.0f}s", flush=True)
