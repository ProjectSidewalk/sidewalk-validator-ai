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

Labels the rater (``EXPERT1_ID`` in the pull) has voted on or placed are excluded. Within each Disagree cell
candidates are ranked by a seeded hash of ``city:label_id`` and the first ones with a crop on the server are taken;
the decoys are then drawn city by city in proportion to the Disagree items, so the place (which the imagery and the
Street View link show) does not separate decoys from Disagree items. It can still hint lone from crowd-voted, which
does not reveal the AI's verdict. The rater sees the site's own crop of the label (what Validate shows) with a ring at
the label's position, the label type, and a Street View link to the same pano; nothing about the AI, the crowd or
the labeler. Item ids are random and the order is shuffled with a secret seed.

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
import math
import os
import shlex
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.rating_page import render_page  # noqa: E402
from calibration.per_city import Z95, wilson_interval  # noqa: E402

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


def eligible(pool):
    """Rows of ``LABEL_TYPES`` that the rater has neither voted on (``expert1`` empty) nor placed
    (``expert1_labeled``)."""
    placed = pool.expert1_labeled.astype(str).str.lower().isin(["t", "true"]) if "expert1_labeled" in pool else False
    return pool[pool.label_type.isin(LABEL_TYPES) & pool.expert1.isna() & ~placed]


def candidates(pool, seed, per_cell=PER_CELL, oversample=OVERSAMPLE):
    """The ranked candidates, ``rank`` ascending: up to ``per_cell * oversample`` per Disagree cell and label type,
    and for the ``decoy`` cell up to ``oversample`` per label type and city, only in cities that have Disagree
    candidates of that type (the decoys are matched to the Disagree items on city; :func:`pick`). Within a decoy
    city every ``decoy_agree`` row comes before any ``decoy_unsure`` row. Only :func:`eligible` rows.
    """
    d = eligible(pool).copy()
    d["group"] = assign_group(d)
    d = d[d.group.notna()]
    d["cell"] = d.group.where(~d.group.str.startswith("decoy"), "decoy")
    d["rank"] = [seeded_rank(seed, u) for u in d.label_uid]
    d["fallback"] = d.group == "decoy_unsure"
    d = d.sort_values(["cell", "label_type", "fallback", "rank"])
    dis = d[d.cell != "decoy"].groupby(["cell", "label_type"], sort=False).head(per_cell * oversample)
    dec = d[d.cell == "decoy"].merge(dis[["label_type", "schema"]].drop_duplicates(), on=["label_type", "schema"],
                                     validate="many_to_one")
    dec = dec.sort_values(["label_type", "schema", "fallback", "rank"]).groupby(["label_type", "schema"],
                                                                               sort=False).head(oversample)
    return pd.concat([dis, dec], ignore_index=True)


def pick(cands, available, seed, per_cell=PER_CELL):
    """``(picked, shortfall)``. Each Disagree cell takes its first ``per_cell`` candidates, in rank order, whose
    ``label_uid`` is in ``available``. The decoys of a label type then follow the picked Disagree items' cities:
    those items are put in a seeded order, and each in turn gives one decoy slot in its city, filled with the best
    unused available decoy there, until ``per_cell`` decoys are found. Agree decoys get a first pass over all the
    slots and Unsure decoys fill what is left, so a scarce Agree is used wherever its city has a slot. A city's share
    of the decoys tracks its share of the Disagree items, and the place does not tell the rater which is which.
    ``shortfall`` maps each cell that came up short to how many it is missing."""
    ok = cands[cands.label_uid.isin(available)]
    dis = ok[ok.cell != "decoy"].groupby(["cell", "label_type"], sort=False).head(per_cell)
    dec_pool = ok[ok.cell == "decoy"].sort_values(["fallback", "rank"])
    decoys = []
    for t in LABEL_TYPES:
        slots = dis[dis.label_type == t]
        slots = slots.iloc[np.argsort([seeded_rank(f"{seed}:slot", u) for u in slots.label_uid], kind="stable")]
        used, filled = set(), set()
        for fallback in (False, True):
            for i, schema in enumerate(slots.schema):
                if len(filled) == per_cell:
                    break
                if i in filled:
                    continue
                c = dec_pool[(dec_pool.label_type == t) & (dec_pool.schema == schema) & (dec_pool.fallback == fallback)
                             & ~dec_pool.label_uid.isin(used)]
                if len(c):
                    used.add(c.label_uid.iloc[0])
                    filled.add(i)
                    decoys.append(c.iloc[0])
    picked = pd.concat([dis, pd.DataFrame(decoys)], ignore_index=True) if decoys else dis.reset_index(drop=True)
    counts = picked.groupby(["cell", "label_type"]).size()
    shortfall = {f"{g}/{t}": per_cell - int(counts.get((g, t), 0)) for g in CELLS for t in LABEL_TYPES
                 if counts.get((g, t), 0) < per_cell}
    return picked, shortfall


def population(pool):
    """``{group: {label_type: count}}`` of the eligible labels in each Disagree group: the strata weights that turn
    the per-type precisions into a population estimate (:func:`weighted_rate`)."""
    d = eligible(pool).copy()
    d["group"] = assign_group(d)
    return {g: {t: int(((d.group == g) & (d.label_type == t)).sum()) for t in LABEL_TYPES}
            for g in ("lone_disagree", "voted_disagree")}


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
    it plus ``_`` (``sidewalk_foo`` -> ``foo-xx``). A directory that exactly matches another schema is never a
    prefix match (``foo-bar`` is not ``sidewalk_foo``'s). Ambiguous prefixes are left out.

    >>> schema_dir_map(["sidewalk_foo", "sidewalk_foo_bar", "sidewalk_dc", "sidewalk_baz"],
    ...                ["foo-xx", "foo-bar", "washington-dc", "baz", "baz-qux"])
    {'sidewalk_foo': 'foo-xx', 'sidewalk_foo_bar': 'foo-bar', 'sidewalk_dc': 'washington-dc', 'sidewalk_baz': 'baz'}
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


def weighted_rate(per_type, weights):
    """Population estimate from per-type rates of a stratified sample: ``(rate, lo, hi)``, the strata weighted by
    ``weights`` (population counts), with a normal-approximation 95% interval from the stratified variance
    ``sum(w^2 p~ (1 - p~) / n)``, where ``p~ = (k + 2) / (n + 4)`` (the Agresti-Coull adjustment, so a stratum at
    0 or n still carries variance). ``per_type``: ``{type: (k, n)}``; types with ``n == 0`` are left out and the
    weights renormalised over the rest.

    >>> r, lo, hi = weighted_rate({"A": (5, 10), "B": (9, 10)}, {"A": 300, "B": 100})
    >>> round(r, 3), round(lo, 3), round(hi, 3)
    (0.6, 0.359, 0.841)
    >>> r, lo, hi = weighted_rate({"A": (20, 20), "B": (19, 19)}, {"A": 1, "B": 1})
    >>> r, round(lo, 3)
    (1.0, 0.912)
    """
    ts = [t for t, (k, n) in per_type.items() if n and weights.get(t)]
    if not ts:
        return np.nan, np.nan, np.nan
    tot = sum(weights[t] for t in ts)
    w = {t: weights[t] / tot for t in ts}
    p = {t: per_type[t][0] / per_type[t][1] for t in ts}
    rate = sum(w[t] * p[t] for t in ts)
    pa = {t: (per_type[t][0] + 2) / (per_type[t][1] + 4) for t in ts}
    se = math.sqrt(sum(w[t] ** 2 * pa[t] * (1 - pa[t]) / per_type[t][1] for t in ts))
    return float(rate), max(0.0, rate - Z95 * se), min(1.0, rate + Z95 * se)


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


def score_table(m, population=None):
    """Per group and per (group, type): AI precision against the expert (can't-tell and blank excluded), with the
    can't-tell count; for ``decoy_unsure`` the share the expert called correct. On ``voted_disagree`` also the crowd
    majority's agreement with the expert. Cells with no items are dropped. With ``population`` (:func:`population`),
    each Disagree group's ``all`` row also carries the population-weighted estimate (``w_rate``, ``w_lo``,
    ``w_hi``); the plain ``rate`` there weights every type equally, as the sample does, and is not one."""
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
    out = pd.DataFrame(rows)
    if population:
        for grp, weights in population.items():
            g = decided[decided.group == grp]
            per_type = {t: (sum(ai_right(v, r) for v, r in zip(s.ai_vote, s.verdict)), len(s))
                        for t, s in ((t, g[g.label_type == t]) for t in LABEL_TYPES)}
            i = out.index[(out.group == grp) & (out.label_type == "all")]
            if len(i):
                out.loc[i, ["w_rate", "w_lo", "w_hi"]] = weighted_rate(per_type, weights)
    return out


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
        d[by] = d[by].astype("Int64").astype(str).fillna("(none)").replace("<NA>", "(none)")
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
        d[by] = d[by].astype("Int64").astype(str).fillna("(none)").replace("<NA>", "(none)")
    return d.groupby(["label_type", by]).size().rename("lone_disagree").reset_index()


def _pct(r, lo, hi):
    return "" if pd.isna(r) else f"{100 * r:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def breakdown_markdown(pool, min_n=100):
    """Both breakdowns as Markdown tables (aggregates only: no label ids, no city names)."""
    out = []
    for by in ("severity", "tag"):
        b = breakdown(pool, by, min_n).merge(lone_counts(pool, by), on=["label_type", by], how="outer",
                                             validate="one_to_one")
        b["lone_disagree"] = b.lone_disagree.fillna(0).astype(int)
        for c in ("Agree_n", "Disagree_n"):
            b[c] = b[c].fillna(0).astype(int)
        b = b[(b.Agree_n >= min_n) | (b.Disagree_n >= min_n) | (b.lone_disagree >= min_n)]
        out.append(f"\n### By {by} (human majority of at least 2; rates with under {min_n} votes left blank; rows "
                   f"shown when any column reaches {min_n})\n")
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
            "spot? Use the Street View link when the image is not enough. The AI voted Disagree on some of these "
            "labels and Agree on others, or abstained; nothing on the card says which. Ignore severity and tags: "
            "judge only whether the label type is right.</p><ul>" + hints + "</ul>")


def cmd_build(args):
    for d in (args.packet_dir, args.key_dir):
        if inside_git_checkout(d):
            sys.exit(f"refusing to write the packet or its key inside a git checkout: {d}")
    pool = load_pool(args.pool)
    cands = candidates(pool, args.seed, args.per_cell, args.oversample)
    print(f"{len(cands):,} candidates over {cands.groupby(['cell', 'label_type']).ngroups} cells")
    dir_map, markers = crop_status(args.host, cands)
    picked, shortfall = pick(cands, set(markers), args.seed, args.per_cell)
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
                "per_cell": args.per_cell, "shortfall": shortfall, "population": population(pool),
                "groups": {f"{g}/{t}": int(n) for (g, t), n in key.groupby(["group", "label_type"]).size().items()}}
    with open(os.path.join(args.key_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(key)} items -> {args.packet_dir}/rate.html\nkey -> {kpath} (sha256 {manifest['key_sha256']})")


def cmd_score(args):
    key = pd.read_csv(args.key, dtype={"tags": str, "expert2": str})
    mpath = args.manifest or os.path.join(os.path.dirname(os.path.abspath(args.key)), "manifest.json")
    pop = json.load(open(mpath)).get("population") if os.path.exists(mpath) else None
    t = score_table(unmask(key, pd.read_csv(args.sheet, dtype=str, keep_default_na=False)), pop)
    print("| Group | Type | Expert agrees with the AI (sample) | n | can't tell | population-weighted | "
          "crowd majority agrees with the expert |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        crowd = _pct(r.get("crowd_rate"), r.get("crowd_lo"), r.get("crowd_hi")) if "crowd_rate" in r else ""
        w = _pct(r.get("w_rate"), r.get("w_lo"), r.get("w_hi")) if "w_rate" in r else ""
        print(f"| {r.group} | {r.label_type} | {_pct(r.rate, r.lo, r.hi)} | {r.n} | {r.cant_tell} | {w} | {crowd} |")
    print("\ndecoy_unsure rows are the share the expert called correct (the AI abstained): not an AI precision.")
    if t.unanswered.sum():
        print(f"\nWARNING: {int(t[t.label_type == 'all'].unanswered.sum())} items unanswered")


# ---------------------------------------------------------------------------------------------------------------------
# Rating on the live site: Gallery review lists, and the rater's votes back as a sheet
# ---------------------------------------------------------------------------------------------------------------------

DEFAULT_CITYPARAMS = os.path.expanduser("~/Git/SidewalkWebpage/conf/cityparams.conf")
GALLERY_MASK_CSS = """\
.card-validation-info, .validation-info-content { display: none !important; }
.label-detail__vote-count, .label-detail__vote-icon { display: none !important; }
.label-detail__admin-prev-validations, .label-detail__validator-comments, .label-detail__comments-section,
.label-detail__comments-count, .label-detail__admin-username { display: none !important; }
#ps-tooltip { display: none !important; }
.ai-icon { display: none !important; }
"""
VOTE_TO_VERDICT = {"Agree": "correct", "Disagree": "incorrect", "Unsure": "cant_tell"}


def _conf_block(lines, start):
    """``{key: value}`` of the flat ``key = "value"`` lines in the HOCON block opened at ``lines[start]``."""
    out, depth = {}, 0
    for line in lines[start:]:
        t = line.strip()
        if t.endswith("{"):
            depth += 1
            continue
        if t == "}":
            depth -= 1
            if depth == 0:
                return out
            continue
        if depth == 1 and "=" in t:
            k, v = (x.strip() for x in t.split("=", 1))
            out[k] = v.strip('"') if v.startswith('"') else None
    return out


def cityparams(text):
    """``(schema -> city_id, city_id -> prod URL)`` from SidewalkWebpage's ``conf/cityparams.conf``: the
    ``db-schema`` block and ``landing-page-url.prod``. Unlike ``/v3/api/cities`` this has the URLs of non-public
    deployments too. Only plain string values are read (``staging = ${...}`` references are skipped).

    >>> conf = '''city-params {
    ...   db-schema {
    ...     foo-xx = "sidewalk_foo"
    ...     staging = ${city-params.db-schema.foo-xx}
    ...   }
    ...   landing-page-url {
    ...     prod {
    ...       foo-xx = "https://sidewalk-foo.example.edu"
    ...     }
    ...     test {
    ...       foo-xx = "https://sidewalk-foo-test.example.edu"
    ...     }
    ...   }
    ... }'''
    >>> cityparams(conf)
    ({'sidewalk_foo': 'foo-xx'}, {'foo-xx': 'https://sidewalk-foo.example.edu'})
    """
    lines = text.splitlines()
    idx = {l.strip(): i for i, l in enumerate(lines) if l.strip().endswith("{")}
    schemas = {v: k for k, v in _conf_block(lines, idx["db-schema {"]).items() if v and k != "staging"}
    lp = idx["landing-page-url {"]
    prod = next(i for i in range(lp + 1, len(lines)) if lines[i].strip() == "prod {")
    urls = {k: v.rstrip("/") for k, v in _conf_block(lines, prod).items() if v}
    return schemas, urls


def gallery_lists(key, schema_city, city_url, seed):
    """One Gallery review list per deployment: ``[(city_id, base_url, [label_id, ...]), ...]``.

    ``schema_city`` / ``city_url`` from :func:`cityparams`. Within a deployment the items keep the key's (shuffled)
    order; the deployments are in a seeded order. Raises if a schema has no deployment or no prod URL.

    >>> key = pd.DataFrame({"schema": ["sidewalk_foo", "sidewalk_bar", "sidewalk_foo"], "label_id": [3, 1, 2]})
    >>> lists = gallery_lists(key, {"sidewalk_foo": "foo-xx", "sidewalk_bar": "bar"},
    ...                       {"foo-xx": "https://f", "bar": "https://b"}, "s")
    >>> sorted((c, ids) for c, _, ids in lists)
    [('bar', [1]), ('foo-xx', [3, 2])]
    """
    missing = sorted(s for s in set(key.schema) if city_url.get(schema_city.get(s)) is None)
    if missing:
        raise ValueError(f"no deployment URL for {missing}")
    out = [(schema_city[sch], city_url[schema_city[sch]], [int(i) for i in g.label_id])
           for sch, g in key.groupby("schema", sort=False)]
    return sorted(out, key=lambda r: seeded_rank(f"{seed}:city", r[0]))


def gallery_url(base, label_ids):
    """The Gallery page that shows exactly ``label_ids``, in order (SidewalkWebpage#5444; at most 500)."""
    if len(label_ids) > 500:
        raise ValueError("Gallery review lists hold at most 500 labels")
    return f"{base}/gallery?labelIds={','.join(str(i) for i in label_ids)}"


def gallery_index_html(lists):
    """A local page linking each deployment's review list, with the masking CSS and how to apply it."""
    import html as h
    rows = "".join(f'<li><a href="{h.escape(gallery_url(b, ids))}" target="_blank" rel="noopener">{h.escape(c)}</a> '
                   f'&middot; {len(ids)} label{"s" if len(ids) != 1 else ""}</li>' for c, b, ids in lists)
    total = sum(len(ids) for _, _, ids in lists)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Re-label lists</title><style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1f2328; --muted:#57606a; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 48px; max-width:900px; }}
pre {{ background:var(--panel); padding:12px; overflow-x:auto; white-space:pre-wrap; }} li {{ margin:4px 0; }}
</style></head><body>
<h1>AI Disagree re-label: {total} labels in {len(lists)} deployments</h1>
<ol>
<li>Sign in on any Project Sidewalk site once; the sign-in carries over to every deployment.</li>
<li><strong>Before opening a list</strong>, make the masking CSS below apply on <code>*.cs.washington.edu</code>
(a user-stylesheet extension such as Stylus, so it applies before the page draws). Check it once on a label whose
votes you know: no counts, no vote icons, no tooltips.</li>
<li>For each label, open it, pan around as needed, and vote <strong>Agree</strong> (correct label),
<strong>Disagree</strong> (incorrect) or <strong>Unsure</strong> (can't tell). Your vote is the record.</li>
</ol>
<h2>Review lists</h2><ul>{rows}</ul>
<h2>Masking CSS</h2><pre>{h.escape(GALLERY_MASK_CSS)}</pre>
</body></html>
"""


def votes_script(key, rater_id):
    """Read-only SQL (run on the database host with ``bash -s``) printing ``schema,label_id,vote`` for the rater's
    current vote on each key label."""
    if not all(c in "0123456789abcdef-" for c in rater_id.lower()):
        raise ValueError("rater id must be a user_id UUID")
    lines = ["export PGOPTIONS='-c default_transaction_read_only=on'",
             'P="psql -X -q -d sidewalk_prod -p 5434 -At -F,"', "echo schema,label_id,vote"]
    for schema, g in key.groupby("schema"):
        ids = ",".join(str(int(i)) for i in g.label_id)
        lines.append(f"$P -c \"SELECT DISTINCT ON (label_id) '{schema}', label_id, validation_result::text "
                     f"FROM {schema}.label_validation WHERE user_id = '{rater_id}' AND label_id IN ({ids}) "
                     f"ORDER BY label_id, label_validation_id DESC\"")
    return "\n".join(lines) + "\n"


def votes_to_sheet(key, votes):
    """The rater's prod votes as a filled sheet (``packet_id, label_type, verdict, notes``); unvoted items blank."""
    v = votes.assign(label_uid=votes.schema + ":" + votes.label_id.astype(str))
    if not v.label_uid.is_unique:
        raise ValueError("more than one vote per label")
    m = key[["packet_id", "label_uid", "label_type"]].merge(v[["label_uid", "vote"]], on="label_uid", how="left",
                                                             validate="one_to_one")
    m["verdict"] = m.vote.map(VOTE_TO_VERDICT).fillna("")
    m["notes"] = ""
    return m[SHEET_COLUMNS]


def cmd_gallery(args):
    if inside_git_checkout(args.out):
        sys.exit(f"refusing to write the review lists inside a git checkout: {args.out}")
    key = pd.read_csv(args.key)
    conf_text = open(args.cityparams).read()
    lists = gallery_lists(key, *cityparams(conf_text), args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(gallery_index_html(lists))
    with open(os.path.join(args.out, "gallery-mask.css"), "w") as f:
        f.write(GALLERY_MASK_CSS)
    # Replicability: the exact lists (one row per deployment, in the order shown) and the cities snapshot they were
    # built from, with their sha256 in the key's manifest.
    urls = pd.DataFrame([{"order": i, "city_id": c, "n": len(ids), "label_ids": " ".join(map(str, ids)),
                          "url": gallery_url(b, ids)} for i, (c, b, ids) in enumerate(lists, 1)])
    upath = os.path.join(args.out, "gallery_urls.csv")
    urls.to_csv(upath, index=False)
    cpath = os.path.join(args.out, "cityparams_snapshot.conf")
    with open(cpath, "w") as f:
        f.write(conf_text)
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(args.cityparams)), "log", "-1",
                             "--format=%H %cI", "--", os.path.basename(args.cityparams)],
                            capture_output=True, text=True).stdout.strip()
    mpath = os.path.join(os.path.dirname(os.path.abspath(args.key)), "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        man["gallery"] = {"urls_sha256": sha256_file(upath), "cityparams_sha256": sha256_file(cpath),
                          "cityparams_commit": commit,
                          "lists": len(lists), "labels": int(urls.n.sum()), "css_sha256":
                          hashlib.sha256(GALLERY_MASK_CSS.encode()).hexdigest()}
        with open(mpath, "w") as f:
            json.dump(man, f, indent=1)
    print(f"{int(urls.n.sum())} labels in {len(lists)} lists -> {args.out}/index.html, {upath}")


def cmd_votes(args):
    key = pd.read_csv(args.key)
    out = remote(args.host, votes_script(key, args.rater_id))
    votes = pd.read_csv(io.StringIO(out))
    sheet = votes_to_sheet(key, votes)
    sheet.to_csv(args.out, index=False)
    print(f"{int((sheet.verdict != '').sum())} of {len(sheet)} items voted -> {args.out}")


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
    s.add_argument("--manifest", help="build's manifest.json (default: next to the key), for the population weights")
    s.set_defaults(func=cmd_score)
    g = sub.add_parser("gallery", help="per-deployment Gallery review lists (index.html) for rating on the live site")
    g.add_argument("--key", required=True)
    g.add_argument("--out", required=True, help="directory for index.html (outside any git checkout)")
    g.add_argument("--seed", required=True, help="the build seed (orders the deployments)")
    g.add_argument("--cityparams", default=DEFAULT_CITYPARAMS,
                   help="SidewalkWebpage conf/cityparams.conf (schema -> city and prod URLs; default %(default)s)")
    g.set_defaults(func=cmd_gallery)
    v = sub.add_parser("votes", help="the rater's prod votes on the key's labels, as a filled sheet for score")
    v.add_argument("--key", required=True)
    v.add_argument("--rater-id", required=True, help="the rater's user_id (pass it in; never commit it)")
    v.add_argument("--out", required=True)
    v.add_argument("--host", default="makelab1")
    v.set_defaults(func=cmd_votes)
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
