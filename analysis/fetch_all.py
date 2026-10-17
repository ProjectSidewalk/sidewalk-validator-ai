"""Download rawLabels CSV for every public city into cache/, verifying the last row parses.
Falls back to bbox tiles if the whole-city export truncates."""
import csv, io, json, os, sys, time, requests
CACHE = sys.argv[1]
cities = requests.get("https://sidewalk-teaneck.cs.washington.edu/v3/api/cities", timeout=60).json()["cities"]
pub = [c for c in cities if c["visibility"] == "public" and c["url"]]
print(f"{len(pub)} public cities", flush=True)
json.dump(pub, open(os.path.join(CACHE, "cities.json"), "w"), indent=1)
NCOL = 41

def ok(text):
    rows = list(csv.reader(io.StringIO(text)))
    return len(rows) > 1 and all(len(r) == NCOL for r in rows[-3:]), len(rows) - 1

for c in pub:
    cid, url = c["city_id"], c["url"].rstrip("/")
    out = os.path.join(CACHE, f"{cid}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"{cid}: cached", flush=True); continue
    t = time.time()
    try:
        r = requests.get(f"{url}/v3/api/rawLabels?filetype=csv", timeout=900)
        good, n = ok(r.text)
    except Exception as e:
        print(f"{cid}: ERROR {e}", flush=True); continue
    if good:
        open(out, "w").write(r.text)
        print(f"{cid}: {n} labels {len(r.text)/1e6:.1f}MB {time.time()-t:.0f}s", flush=True)
        continue
    # bbox tile fallback: 4x4 grid over city bounds
    print(f"{cid}: whole-city export truncated ({n} rows), tiling", flush=True)
    b = c["bounds"]; parts = []; header = None
    for i in range(4):
        for j in range(4):
            s = b["south"] + (b["north"]-b["south"])*i/4; nn = b["south"] + (b["north"]-b["south"])*(i+1)/4
            w = b["west"] + (b["east"]-b["west"])*j/4; e = b["west"] + (b["east"]-b["west"])*(j+1)/4
            rr = requests.get(f"{url}/v3/api/rawLabels?filetype=csv&bbox={w},{s},{e},{nn}", timeout=900)
            g, k = ok(rr.text)
            if not g: print(f"  tile {i},{j} still bad ({k})", flush=True)
            lines = rr.text.splitlines(keepends=True)
            if header is None: header = lines[0]
            parts.extend(lines[1:])
    open(out, "w").write(header + "".join(parts))
    print(f"{cid}: tiled, {len(parts)} rows {time.time()-t:.0f}s", flush=True)
