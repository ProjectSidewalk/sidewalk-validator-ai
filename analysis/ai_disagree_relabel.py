"""Masked expert re-label of the production AI validator's Disagree votes, and its severity / tag breakdown.

Production measured the AI's Disagree votes against human majorities only (SidewalkWebpage#5793): 56-71% agreement
depending on how strict the human reference is, against 96-99.7% for its Agree votes. Human votes are noisy, so this
asks an expert to re-label a random sample with the AI and crowd verdicts masked, which is the one measurement that does not rest on the crowd.

**Design.** 300 labels, three groups of 100, each stratified 20 per label type:

``lone_disagree``
    The AI voted Disagree and no human has voted. These labels are hidden on the AI's vote alone, so their
    precision decides whether to restore them.
``voted_disagree``
    The AI voted Disagree and at least two humans voted with a clear majority. Ties the re-label to the crowd
    numbers and measures the crowd's own error against the expert.
``decoy`` (``decoy_agree``, topped up with ``decoy_unsure``)
    The AI voted Agree. Without these every item would be an AI Disagree and the rater would know the AI's verdict.
    A type with too few Agree votes is topped up with labels the AI abstained on (Unsure): Obstacle, whose old map
    never let an Agree clear the gate, has 7. They mask the verdict as well; they are scored apart and give no AI precision.

Labels the rater (``EXPERT1_ID`` in the pull) has already voted on are excluded, so the rater has not seen them.
Within each (group, type) cell candidates are ranked by a seeded hash of ``city:label_id`` and the first ones with a
crop on the server are taken. The rater sees the site's own crop of the label (what Validate shows) with a ring at
the label's position, the label type, and a Street View link to the same pano; nothing about the AI, the crowd, the
city or the labeler. Item ids are random and the order is shuffled with a secret seed.

Subcommands (every file with label-level rows goes OUTSIDE the repo, under ``~/Data/...``):

``build``
    Pool -> candidates -> crop check on the server -> packet (``rate.html``, ``sheet.csv``, ``items/``) and key
    (``key.csv`` plus its sha256 in ``manifest.json``). Needs a live SSH control master to the database host.
``score``
    Unmasks a filled sheet against the key: AI precision per group with Wilson 95% intervals, per type, the
    can't-tell rate, and on ``voted_disagree`` how often the crowd majority matches the expert.
``breakdown``
    From the pool alone: how often humans confirm the AI's Agree and Disagree votes, by label type x severity and by
    label type x tag (cells with at least ``--min-n`` labels), against a majority of at least two humans.

Usage::

    D=~/Data/sidewalk-validator-ai-relabel-2026-10
    ssh makelab1 "EXPERT1_ID=<rater uuid> EXPERT2_ID=<other expert uuid> bash -s" \\
        < analysis/pull_relabel_pool.sh > $D/relabel_pool.csv
    ssh -fNM makelab1        # one master; build makes three calls through it
    python analysis/ai_disagree_relabel.py build --pool $D/relabel_pool.csv --packet-dir $D/packet \\
        --key-dir $D/private-key-do-not-share --seed $SECRET_SEED
    python analysis/ai_disagree_relabel.py score --key $D/private-key-do-not-share/key.csv \\
        --sheet $D/packet/relabel_sheet__jon.csv
    python analysis/ai_disagree_relabel.py breakdown --pool $D/relabel_pool.csv --out $D/breakdown.md
"""
import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.rating_page import render_page  # noqa: E402
from calibration.per_city import wilson_interval  # noqa: E402

LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
GROUPS = ("lone_disagree", "voted_disagree", "decoy_agree", "decoy_unsure")
CELLS = ("lone_disagree", "voted_disagree", "decoy")  # decoy_agree first, topped up with decoy_unsure
PER_CELL = 20          # per (group, type): 3 groups x 5 types x 20 = 300 items
OVERSAMPLE = 4         # candidates checked per needed item, since not every label has a crop
IMAGES_ROOT = "/www/sidewalk/prod/images"
SCHEMA_DIR_OVERRIDES = {"sidewalk_dc": "washington-dc"}
VERDICTS = [("correct", "Correct label", "c"), ("incorrect", "Incorrect label", "i"), ("cant_tell", "Can't tell", "u")]
SHEET_COLUMNS = ["packet_id", "label_type", "verdict", "notes"]
KEY_COLUMNS = ["packet_id", "label_uid", "schema", "label_id", "label_type", "group", "ai_vote", "ai_accuracy",
               "h_agree", "h_dis", "h_unsure", "expert2", "severity", "tags", "marker_x", "marker_y"]
TYPE_HINTS = {
    "CurbRamp": "a curb ramp: a sloped transition from sidewalk to street at a crossing",
    "NoCurbRamp": "a missing curb ramp: a place where pedestrians cross a curb that should have a ramp and does not",
    "Obstacle": "an obstacle in the sidewalk path: something blocking or narrowing the walkway",
    "SurfaceProblem": "a surface problem on the sidewalk: cracks, heaving, uneven or broken surface",
    "Crosswalk": "a marked crosswalk",
}


# ---------------------------------------------------------------------------------------------------------------------
# Pool and sampling (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def load_pool(path):
    """The pull's CSV with ``label_uid`` (``schema:label_id``; label_id restarts per city) and the human tallies.

    Adds ``h_dec`` (decisive human votes, Agree + Disagree) and ``crowd`` (the majority of them, None on a tie or with
    none). Raises if ``label_uid`` is not unique.
    """
    df = pd.read_csv(path, dtype={"tags": str, "expert1": str, "expert2": str}, keep_default_na=True)
    df["label_uid"] = df["schema"] + ":" + df["label_id"].astype(str)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid (schema:label_id) is not unique in the pool")
    df["h_dec"] = df.h_agree + df.h_dis
    df["crowd"] = np.where(df.h_agree > df.h_dis, "Agree", np.where(df.h_dis > df.h_agree, "Disagree", None))
    return df


def assign_group(df):
    """The re-label group of each row (None if it belongs to none).

    >>> d = pd.DataFrame({"ai_vote": ["Disagree", "Disagree", "Disagree", "Agree", "Unsure"],
    ...                   "h_agree": [0, 2, 1, 0, 0], "h_dis": [0, 1, 1, 0, 0], "h_unsure": [0, 0, 0, 0, 0]})
    >>> list(assign_group(d))
    ['lone_disagree', 'voted_disagree', None, 'decoy_agree', 'decoy_unsure']
    """
    n_h = df.h_agree + df.h_dis + df.h_unsure
    dis = df.ai_vote == "Disagree"
    lone = dis & (n_h == 0)
    voted = dis & ((df.h_agree + df.h_dis) >= 2) & (df.h_agree != df.h_dis)
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    for name, mask in zip(GROUPS, (lone, voted, df.ai_vote == "Agree", df.ai_vote == "Unsure")):
        out[mask & out.isna()] = name
    return out


def seeded_rank(seed, label_uid):
    """A deterministic pseudo-random rank for one label: the first 16 hex digits of sha256(seed:label_uid)."""
    return int(hashlib.sha256(f"{seed}:{label_uid}".encode()).hexdigest()[:16], 16)


def candidates(pool, seed, per_cell=PER_CELL, oversample=OVERSAMPLE):
    """The ranked candidates: up to ``per_cell * oversample`` per (cell, label type), in ``rank`` order; in the
    ``decoy`` cell every ``decoy_agree`` row comes before any ``decoy_unsure`` row.

    Only rows with a group, one of ``LABEL_TYPES``, and no vote from the rater (``expert1`` empty).
    """
    d = pool[pool.label_type.isin(LABEL_TYPES) & pool.expert1.isna()].copy()
    d["group"] = assign_group(d)
    d = d[d.group.notna()]
    d["cell"] = d.group.where(~d.group.str.startswith("decoy"), "decoy")
    d["rank"] = [seeded_rank(seed, u) for u in d.label_uid]
    d["fallback"] = d.group == "decoy_unsure"
    d = d.sort_values(["cell", "label_type", "fallback", "rank"])
    return d.groupby(["cell", "label_type"], sort=False).head(per_cell * oversample).reset_index(drop=True)


def pick(cands, available, per_cell=PER_CELL):
    """``(picked, shortfall)``: the first ``per_cell`` candidates per cell, in rank order, whose ``label_uid`` is in
    ``available``; ``shortfall`` maps each cell that came up short to how many it is missing."""
    ok = cands[cands.label_uid.isin(available)]
    picked = ok.groupby(["cell", "label_type"], sort=False).head(per_cell).reset_index(drop=True)
    counts = picked.groupby(["cell", "label_type"]).size()
    shortfall = {f"{g}/{t}": per_cell - int(counts.get((g, t), 0)) for g in CELLS for t in LABEL_TYPES
                 if counts.get((g, t), 0) < per_cell}
    return picked, shortfall


def make_key(picked, seed):
    """The masking key: a random ``r<6 hex>`` id per item and a shuffled order, deterministic in ``seed`` and the
    item set (so the seed stays secret until the rating is done). Same scheme as ``vlm_baseline.packet_key``."""
    rng = np.random.default_rng(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    s = picked.sort_values("label_uid").reset_index(drop=True)
    ids = set()
    while len(ids) < len(s):
        ids.add(f"r{int(rng.integers(0, 16 ** 6)):06x}")
    s = s.assign(packet_id=rng.permutation(sorted(ids)))
    return s.iloc[rng.permutation(len(s))].reset_index(drop=True)


def schema_dir_map(schemas, dirs):
    """``{schema: images directory}`` for the schemas that have one.

    A directory matches a schema when its name, with ``-`` as ``_``, equals the schema's city part or starts with
    it plus ``_`` (``sidewalk_columbus`` -> ``columbus-oh``). A directory that exactly matches another schema is
    never a prefix match (``la-piedad`` is not ``sidewalk_la``'s). Ambiguous prefixes are left out.

    >>> schema_dir_map(["sidewalk_la", "sidewalk_la_piedad", "sidewalk_dc", "sidewalk_zurich"],
    ...                ["la-ca", "la-piedad", "washington-dc", "zurich", "zurich-infra3d"])
    {'sidewalk_la': 'la-ca', 'sidewalk_la_piedad': 'la-piedad', 'sidewalk_dc': 'washington-dc', 'sidewalk_zurich': 'zurich'}
    """
    cities = {s: s[len("sidewalk_"):] for s in schemas}
    norm = {d: d.replace("-", "_") for d in dirs}
    exact_taken = {n for n in norm.values() if n in set(cities.values())}
    out = {}
    for s, c in cities.items():
        if s in SCHEMA_DIR_OVERRIDES and SCHEMA_DIR_OVERRIDES[s] in norm:
            out[s] = SCHEMA_DIR_OVERRIDES[s]
            continue
        exact = [d for d, n in norm.items() if n == c]
        if exact:
            out[s] = exact[0]
            continue
        pre = [d for d, n in norm.items() if n.startswith(c + "_") and n not in exact_taken]
        if len(pre) == 1:
            out[s] = pre[0]
    return out


def crop_path(dir_map, schema, label_type, label_id):
    """The crop's path on the server (``PanoDataService.cropFile``), or None if the schema has no images dir."""
    d = dir_map.get(schema)
    return None if d is None else f"{IMAGES_ROOT}/{d}/{label_type}/crop_{int(label_id)}.png"


def street_view_url(pano_id, heading, pitch):
    """A Google Maps Street View link to the same pano, aimed at the label."""
    return (f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}"
            f"&heading={float(heading):.2f}&pitch={float(pitch):.2f}")


# ---------------------------------------------------------------------------------------------------------------------
# Scoring (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def _rate(k, n):
    lo, hi = wilson_interval(k / n, n) if n else (np.nan, np.nan)
    return {"n": int(n), "k": int(k), "rate": k / n if n else np.nan, "lo": float(lo), "hi": float(hi)}


def unmask(key, sheet):
    """Key rows joined with the rater's verdicts; raises unless the sheet covers exactly the key's items."""
    sheet = sheet[["packet_id", "verdict"]].copy()
    if set(sheet.packet_id) != set(key.packet_id):
        raise ValueError("the sheet's packet_ids do not match the key")
    m = key.merge(sheet, on="packet_id", how="left", validate="one_to_one")
    m["verdict"] = m.verdict.fillna("").astype(str)
    bad = set(m.verdict) - {v for v, _, _ in VERDICTS} - {""}
    if bad:
        raise ValueError(f"unexpected verdicts in the sheet: {sorted(bad)}")
    return m


def ai_right(row_vote, verdict):
    """Whether the expert's verdict confirms the AI's vote (Agree <-> correct, Disagree <-> incorrect). For an
    abstention (Unsure) it is whether the expert called the label correct, reported but not an AI precision."""
    return (row_vote in ("Agree", "Unsure") and verdict == "correct") or \
        (row_vote == "Disagree" and verdict == "incorrect")


def score_table(m):
    """Per group and per (group, type): AI precision against the expert (can't-tell and blank excluded), with the
    can't-tell count; for ``decoy_unsure`` the share the expert called correct. On ``voted_disagree`` also the crowd
    majority's agreement with the expert. Cells with no items are dropped."""
    rows = []
    decided = m[m.verdict.isin(["correct", "incorrect"])]
    for keys, g in [((grp, "all"), decided[decided.group == grp]) for grp in GROUPS] + \
                   [((grp, t), decided[(decided.group == grp) & (decided.label_type == t)])
                    for grp in GROUPS for t in LABEL_TYPES]:
        right = sum(ai_right(v, r) for v, r in zip(g.ai_vote, g.verdict))
        sub = m[(m.group == keys[0]) & ((m.label_type == keys[1]) if keys[1] != "all" else True)]  # incl. can't tell
        row = {"group": keys[0], "label_type": keys[1], "cant_tell": int((sub.verdict == "cant_tell").sum()),
               "unanswered": int((sub.verdict == "").sum()), **_rate(right, len(g))}
        if keys[0] == "voted_disagree":
            crowd_ok = sum((c == "Agree" and r == "correct") or (c == "Disagree" and r == "incorrect")
                           for c, r in zip(g.crowd, g.verdict))
            c = _rate(crowd_ok, len(g))
            row.update(crowd_n=c["n"], crowd_rate=c["rate"], crowd_lo=c["lo"], crowd_hi=c["hi"])
        if len(sub):
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Severity and tag breakdown (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def confirm_rows(pool):
    """AI Agree / Disagree votes with a human majority of at least two decisive votes, and whether it confirms."""
    d = pool[pool.ai_vote.isin(["Agree", "Disagree"]) & (pool.h_dec >= 2) & pool.crowd.notna()
             & pool.label_type.isin(LABEL_TYPES)].copy()
    d["confirmed"] = d.ai_vote == d.crowd
    return d


def breakdown(pool, by, min_n=100):
    """Confirmation rate of the AI's Agree and Disagree votes per label type x ``by`` (``severity`` or ``tag``), with
    Wilson intervals; cells with fewer than ``min_n`` votes of that verdict are left blank. For ``tag`` a label with
    several tags counts once per tag; untagged labels are ``(none)``."""
    d = confirm_rows(pool)
    if by == "tag":
        d = d.assign(tag=d.tags.fillna("").str.split(";")).explode("tag")
        d["tag"] = d.tag.replace("", "(none)")
    else:
        d[by] = d[by].astype("Int64").astype(str).replace("<NA>", "(none)")
    rows = []
    for (t, v), g in d.groupby(["label_type", by]):
        row = {"label_type": t, by: v}
        for vote in ("Agree", "Disagree"):
            s = g[g.ai_vote == vote]
            r = _rate(int(s.confirmed.sum()), len(s))
            ok = r["n"] >= min_n
            row.update({f"{vote}_n": r["n"], f"{vote}_rate": r["rate"] if ok else np.nan,
                        f"{vote}_lo": r["lo"] if ok else np.nan, f"{vote}_hi": r["hi"] if ok else np.nan})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[(out.Agree_n >= min_n) | (out.Disagree_n >= min_n)].reset_index(drop=True)


def lone_counts(pool, by):
    """How many labels the AI's Disagree alone has hidden (no human vote), per label type x ``by``."""
    n_h = pool.h_agree + pool.h_dis + pool.h_unsure
    d = pool[(pool.ai_vote == "Disagree") & (n_h == 0) & pool.label_type.isin(LABEL_TYPES)].copy()
    if by == "tag":
        d = d.assign(tag=d.tags.fillna("").str.split(";")).explode("tag")
        d["tag"] = d.tag.replace("", "(none)")
    else:
        d[by] = d[by].astype("Int64").astype(str).replace("<NA>", "(none)")
    return d.groupby(["label_type", by]).size().rename("lone_disagree").reset_index()


def _pct(r, lo, hi):
    return "" if pd.isna(r) else f"{100 * r:.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def breakdown_markdown(pool, min_n=100):
    """Both breakdowns as Markdown tables (aggregates only: no label ids, no city names)."""
    out = []
    for by in ("severity", "tag"):
        b = breakdown(pool, by, min_n).merge(lone_counts(pool, by), on=["label_type", by], how="left",
                                             validate="one_to_one")
        b["lone_disagree"] = b.lone_disagree.fillna(0).astype(int)
        out.append(f"\n### By {by} (human majority of at least 2; cells under {min_n} votes left blank)\n")
        out.append(f"| Type | {by.capitalize()} | AI Agree confirmed | n | AI Disagree confirmed | n | "
                   "hidden by a lone AI Disagree |")
        out.append("|---|---|---:|---:|---:|---:|---:|")
        for t in LABEL_TYPES:
            for _, r in b[b.label_type == t].iterrows():
                out.append(f"| {t} | {r[by]} | {_pct(r.Agree_rate, r.Agree_lo, r.Agree_hi)} | {r.Agree_n:,} | "
                           f"{_pct(r.Disagree_rate, r.Disagree_lo, r.Disagree_hi)} | {r.Disagree_n:,} | "
                           f"{r.lone_disagree:,} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Server access (thin; through an existing SSH control master)
# ---------------------------------------------------------------------------------------------------------------------

def remote(host, script, binary=False):
    """Run ``script`` on ``host`` with ``bash -s`` (one connection; BatchMode, so a dead master fails fast)."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash -s"], input=script.encode(), capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ssh {host} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-2000:]}")
    return r.stdout if binary else r.stdout.decode()


def crop_status(host, cands):
    """``(dir_map, markers)``: the schema -> images dir map, and for every candidate that has both a crop file and a
    ``label_crop`` row, its ``(marker_x, marker_y)`` keyed by ``label_uid``. Two calls."""
    dirs = remote(host, f"ls {shlex.quote(IMAGES_ROOT)}").split()
    dir_map = schema_dir_map(sorted(cands.schema.unique()), dirs)
    lines = ["export PGOPTIONS='-c default_transaction_read_only=on'",
             'P="psql -X -q -d sidewalk_prod -p 5434 -At -F,"']
    for schema, g in cands.groupby("schema"):
        ids = ",".join(str(int(i)) for i in g.label_id)
        lines.append(f"[ \"$($P -c \"SELECT to_regclass('{schema}.label_crop') IS NOT NULL\")\" = t ] && "
                     f"$P -c \"SELECT '{schema}', label_id, marker_x, marker_y FROM {schema}.label_crop "
                     f"WHERE label_id IN ({ids})\" | sed 's/^/M,/'")
        for _, r in g.iterrows():
            p = crop_path(dir_map, schema, r.label_type, r.label_id)
            if p:
                lines.append(f"[ -f {shlex.quote(p)} ] && echo F,{schema},{int(r.label_id)}")
    have, markers = set(), {}
    for line in remote(host, "\n".join(lines) + "\ntrue\n").splitlines():
        parts = line.split(",")
        if parts[0] == "F":
            have.add(f"{parts[1]}:{parts[2]}")
        elif parts[0] == "M" and len(parts) == 5:
            markers[f"{parts[1]}:{parts[2]}"] = (float(parts[3]), float(parts[4]))
    return dir_map, {u: xy for u, xy in markers.items() if u in have}


def fetch_crops(host, key, dir_map):
    """``{label_uid: PNG bytes}`` for the key's items, in one tar stream."""
    paths = {crop_path(dir_map, r.schema, r.label_type, r.label_id): r.label_uid for _, r in key.iterrows()}
    data = remote(host, "tar -cf - -P " + " ".join(shlex.quote(p) for p in paths), binary=True)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            name = m.name if m.name.startswith("/") else "/" + m.name
            if name in paths:
                out[paths[name]] = tf.extractfile(m).read()
    return out


def marked_image(png, marker_x, marker_y, max_width=1000):
    """The crop as JPEG bytes, downscaled to ``max_width``, with a two-tone ring at the label's position (fractions of
    the width and height), so it shows on any background."""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    x, y, r = marker_x * im.width, marker_y * im.height, max(12, round(0.035 * im.width))
    d = ImageDraw.Draw(im)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], outline=(0, 0, 0), width=7)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 214, 0), width=3)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def inside_git_checkout(path):
    """True if ``path`` (or its nearest existing parent) is inside any git work tree."""
    p = os.path.abspath(path)
    while not os.path.isdir(p):
        p = os.path.dirname(p)
    r = subprocess.run(["git", "-C", p, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def intro_html():
    hints = "".join(f"<li><strong>{t}</strong>: {h}</li>" for t, h in TYPE_HINTS.items())
    return ("<p>Each card shows one Project Sidewalk label: the image a validator sees, with a yellow ring at the "
            "label's position, and its label type. Is it a <strong>correct</strong> label of that type at that "
            "spot? Use the Street View link when the image is not enough. Some of these labels the AI voted Agree "
            "on and some Disagree; nothing on the card says which. Judge severity and tags as irrelevant: only "
            "whether the label type is right.</p><ul>" + hints + "</ul>")


def cmd_build(args):
    for d in (args.packet_dir, args.key_dir):
        if inside_git_checkout(d):
            sys.exit(f"refusing to write the packet or its key inside a git checkout: {d}")
    pool = load_pool(args.pool)
    cands = candidates(pool, args.seed, args.per_cell, args.oversample)
    print(f"{len(cands):,} candidates over {cands.groupby(['cell', 'label_type']).ngroups} cells")
    dir_map, markers = crop_status(args.host, cands)
    picked, shortfall = pick(cands, set(markers), args.per_cell)
    if shortfall:
        print(f"WARNING: cells short of {args.per_cell}: {shortfall}")
    picked["marker_x"] = [markers[u][0] for u in picked.label_uid]
    picked["marker_y"] = [markers[u][1] for u in picked.label_uid]
    key = make_key(picked, args.seed)
    crops = fetch_crops(args.host, key, dir_map)
    missing = set(key.label_uid) - set(crops)
    if missing:
        sys.exit(f"{len(missing)} crops vanished between the check and the fetch; rerun")
    os.makedirs(os.path.join(args.packet_dir, "items"), exist_ok=True)
    os.makedirs(args.key_dir, exist_ok=True)
    items, fixed = [], {}
    for i, r in enumerate(key.itertuples(), 1):
        img = f"items/{r.packet_id}.jpg"
        with open(os.path.join(args.packet_dir, img), "wb") as f:
            f.write(marked_image(crops[r.label_uid], r.marker_x, r.marker_y))
        sv = street_view_url(r.pano_id, r.heading, r.pitch)
        items.append({"id": r.packet_id, "image": img, "alt": f"Item {i}: a {r.label_type} label, ringed in yellow",
                      "caption_html": f"{i}. {r.label_type} &middot; {r.packet_id}",
                      "extra_html": f'<p><a href="{sv}" target="_blank" rel="noopener noreferrer">'
                                    f'Open in Street View</a></p>'})
        fixed[r.packet_id] = {"label_type": r.label_type}
    fields = [{"name": "verdict", "legend": "Is this label correct?", "kind": "radio", "options": VERDICTS},
              {"name": "notes", "legend": "Notes (optional)", "kind": "text"}]
    page = render_page(title="AI Disagree re-label", heading="Re-label: is the label correct?",
                       intro_html=intro_html(), items=items, fields=fields, sheet_columns=SHEET_COLUMNS,
                       fixed=fixed, storage_key="ai_disagree_relabel_2026_10", export_name="relabel_sheet")
    with open(os.path.join(args.packet_dir, "rate.html"), "w") as f:
        f.write(page)
    pd.DataFrame({"packet_id": key.packet_id, "label_type": key.label_type, "verdict": "", "notes": ""}).to_csv(
        os.path.join(args.packet_dir, "sheet.csv"), index=False)
    kpath = os.path.join(args.key_dir, "key.csv")
    key[KEY_COLUMNS + ["crowd", "pano_id"]].to_csv(kpath, index=False)
    manifest = {"pool_sha256": sha256_file(args.pool), "key_sha256": sha256_file(kpath),
                "seed_sha256": hashlib.sha256(str(args.seed).encode()).hexdigest(), "items": len(key),
                "per_cell": args.per_cell, "shortfall": shortfall,
                "groups": {f"{g}/{t}": int(n) for (g, t), n in key.groupby(["group", "label_type"]).size().items()}}
    with open(os.path.join(args.key_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(key)} items -> {args.packet_dir}/rate.html\nkey -> {kpath} (sha256 {manifest['key_sha256']})")


def cmd_score(args):
    key = pd.read_csv(args.key, dtype={"tags": str, "expert2": str})
    t = score_table(unmask(key, pd.read_csv(args.sheet, dtype=str, keep_default_na=False)))
    print("| Group | Type | AI right (expert) | n | can't tell | crowd majority right (expert) |")
    print("|---|---|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        crowd = _pct(r.get("crowd_rate"), r.get("crowd_lo"), r.get("crowd_hi")) if "crowd_rate" in r else ""
        print(f"| {r.group} | {r.label_type} | {_pct(r.rate, r.lo, r.hi)} | {r.n} | {r.cant_tell} | {crowd} |")
    if t.unanswered.sum():
        print(f"\nWARNING: {int(t[t.label_type == 'all'].unanswered.sum())} items unanswered")


def cmd_breakdown(args):
    md = breakdown_markdown(load_pool(args.pool), args.min_n)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"-> {args.out}")
    else:
        print(md)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample, check and fetch crops, write the packet and the key")
    b.add_argument("--pool", required=True)
    b.add_argument("--packet-dir", required=True, help="rate.html, sheet.csv, items/ (outside any git checkout)")
    b.add_argument("--key-dir", required=True, help="key.csv and manifest.json (private; outside any git checkout)")
    b.add_argument("--seed", required=True, help="secret; keep it until the rating is done")
    b.add_argument("--host", default="makelab1")
    b.add_argument("--per-cell", type=int, default=PER_CELL)
    b.add_argument("--oversample", type=int, default=OVERSAMPLE)
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("score", help="unmask a filled sheet against the key")
    s.add_argument("--key", required=True)
    s.add_argument("--sheet", required=True)
    s.set_defaults(func=cmd_score)
    d = sub.add_parser("breakdown", help="AI confirmation rate by severity and by tag, from the pool")
    d.add_argument("--pool", required=True)
    d.add_argument("--min-n", type=int, default=100)
    d.add_argument("--out")
    d.set_defaults(func=cmd_breakdown)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
