"""Ground-truth rating rounds: masked expert re-labels of the AI validator's votes, one city per rater with a
shared overlap between raters, designed from a JSON file so the next round is a config change.

Round 1 (``ai_disagree_relabel.py``, PR #66) sampled 300 labels across every city for one rater. This module runs
the rounds after it (round 2 is issue #78: Seattle and Chicago, two raters, 250 items in each rater's own city
plus 50 of the other rater's, so 100 items are rated by both). It reuses round 1's pool pull, masking key, crop
fetch, rating page, Gallery lists, vote pull and sha256 pins, and adds what #74 needs from a gold set:

* **strata** that tell the labels that are rows of the training set under the margin rule from the rest of the
  AI's Disagrees. ``margin_incorrect`` (net human Disagree >= 2: the training target sides with the AI) and
  ``margin_correct`` (net Agree >= 3: the AI contradicts its training target, the hard examples) can score a
  target-cleaning method on exactly the rows it touches; ``lone`` (no human vote; hidden on the AI's vote alone)
  and ``light_voted`` (votes that never met the margin rule) measure the AI itself. ``decoy`` items are AI Agree
  votes (Unsure where a type has no Agree), mixed in so the rater cannot tell the AI's verdict;
* **per-type quotas** and **severity buckets** (1-2 / 3-5 / none) filled in proportion to the eligible pool, so
  the degraded, low-severity cases are not left to chance;
* **shared** items rated by every rater, for inter-rater agreement;
* in the key: ``margin_class`` (the training target under the current tallies), ``round3_split`` (joined to the
  pinned round-3 ``split.csv`` files) and the label in both id forms, ``schema:label_id`` and the deployment's
  API ``city_id:label_id``, the form the training manifests use.

Labels either rater has voted on or placed are excluded everywhere (every ``expert<N>`` column of the pool).

**Design file** (``analysis/gt_rounds/round2.json``; rater keys are first names, rater user_ids never go in it)::

    {"raters": {"jon": "sidewalk_seattle", "mikey": "sidewalk_chicago"},
     "per_type": {"CurbRamp": 42, "Crosswalk": 36, "NoCurbRamp": 34, "Obstacle": 30, "SurfaceProblem": 28},
     "shares": {"margin": 0.5, "lone": 0.35, "light_voted": 0.15},
     "margin_correct_cap": 0.4,
     "decoys_per_type": 16,
     "shared": 50}

``per_type`` is the AI-Disagree quota per type in each city; ``shares`` splits it over the Disagree cells;
``margin_correct_cap`` is the most of the margin cell that ``margin_correct`` rows may take (they are scarce, so
the cell is topped up with ``margin_incorrect``); ``shared`` is the number of each city's items that the other
raters also rate.

**Subcommands** (every file with label-level rows goes outside the repo, under ``~/Data/...``)::

    D=~/Data/sidewalk-validator-ai-relabel-2026-10/round2
    ssh -fNM makelab1                     # one master; the pull and the build go through it
    SCHEMAS="sidewalk_seattle sidewalk_chicago" EXPERT1_ID=<uuid> EXPERT2_ID=<uuid> \\
        ssh makelab1 "SCHEMAS='sidewalk_seattle sidewalk_chicago' EXPERT1_ID=... EXPERT2_ID=... bash -s" \\
        < analysis/pull_relabel_pool.sh > $D/pool.csv
    python analysis/gt_rounds.py build --design analysis/gt_rounds/round2.json --pool $D/pool.csv \\
        --packet-dir $D/packet --key-dir $D/private-key-do-not-share --seed $SECRET_SEED \\
        --round3-splits ~/Data/sidewalk-validator-ai-retrain-2026-10/round3
    python analysis/gt_rounds.py gallery --key $K --rater jon --out $D/gallery_jon
    python analysis/gt_rounds.py votes   --key $K --rater jon --rater-id <uuid> --out $D/packet/sheet_jon.csv
    python analysis/gt_rounds.py score   --key $K --rater jon --sheet $D/packet/sheet_jon.csv
    python analysis/gt_rounds.py agreement --key $K --sheets jon=$D/packet/sheet_jon.csv mikey=$D/packet/sheet_mikey.csv

``build`` needs the crop check and fetch on the database host (two calls through the master). ``gallery`` writes
one page per rater: their own city's list first, then the other raters' shared items, with the masking CSS and the
rating rule. ``votes`` pulls one rater's prod votes on their items and pins the sheet as ``votes_<rater>``.
"""
import argparse
import hashlib
import html
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis.rating_page import render_page  # noqa: E402

LABEL_TYPES = rl.LABEL_TYPES
DISAGREE_STRATA = ("lone", "light_voted", "margin_incorrect", "margin_correct")
STRATA = DISAGREE_STRATA + ("decoy_agree", "decoy_unsure")
CELLS = ("lone", "light_voted", "margin", "decoy")  # margin = margin_correct (capped) topped up with margin_incorrect
SEV_BUCKETS = ("1-2", "3-5", "none")
KEY_COLUMNS = ["packet_id", "label_uid", "label_uid_city", "schema", "city_id", "label_id", "label_type", "rater",
               "shared", "stratum", "cell", "margin_class", "round3_split", "sev_bucket", "ai_vote", "ai_accuracy",
               "h_agree", "h_dis", "h_unsure", "crowd", "severity", "tags", "marker_x", "marker_y", "pano_id"]
SHEET_COLUMNS = rl.SHEET_COLUMNS
RATING_RULE_HTML = (
    "<ol>"
    "<li><strong>Agree</strong> (correct): the feature is really there at that spot, even in poor condition "
    "(faded crosswalk, steep or broken ramp, cobbles). Condition goes in severity and tags, not the vote.</li>"
    "<li><strong>Disagree</strong> (incorrect): the feature is absent, somewhere else, or another type "
    "(a driveway marked as a curb ramp, a road marked as sidewalk surface).</li>"
    "<li><strong>Unsure</strong>: only when the imagery cannot show it.</li>"
    "</ol>")


# ---------------------------------------------------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------------------------------------------------

def load_design(path_or_dict):
    """The design as a dict, checked: every rater has a distinct schema, the shares cover the Disagree cells and
    sum to one, and the quotas are positive integers.

    >>> d = load_design({"raters": {"A": "sidewalk_a", "B": "sidewalk_b"}, "per_type": {t: 10 for t in LABEL_TYPES},
    ...                  "shares": {"margin": 0.5, "lone": 0.3, "light_voted": 0.2}, "margin_correct_cap": 0.4,
    ...                  "decoys_per_type": 4, "shared": 5})
    >>> sorted(d["raters"])
    ['A', 'B']
    >>> load_design({**d, "shares": {"margin": 0.5, "lone": 0.5, "light_voted": 0.5}})
    Traceback (most recent call last):
    ...
    ValueError: shares must sum to 1
    """
    d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else dict(path_or_dict)
    raters = d.get("raters") or {}
    if len(set(raters.values())) != len(raters) or not raters:
        raise ValueError("raters must map distinct roles to distinct schemas")
    if set(d.get("shares", {})) != {"lone", "light_voted", "margin"}:
        raise ValueError("shares must have lone, light_voted and margin")
    if abs(sum(d["shares"].values()) - 1) > 1e-9:
        raise ValueError("shares must sum to 1")
    if set(d.get("per_type", {})) != set(LABEL_TYPES) or any(int(v) <= 0 for v in d["per_type"].values()):
        raise ValueError("per_type needs a positive quota for every label type")
    for k in ("decoys_per_type", "shared"):
        if int(d.get(k, -1)) < 0:
            raise ValueError(f"{k} must be a non-negative integer")
    if not 0 <= float(d.get("margin_correct_cap", -1)) <= 1:
        raise ValueError("margin_correct_cap must be in [0, 1]")
    return d


def largest_remainder(total, shares):
    """Integers summing to ``total`` in proportion to ``shares`` (a dict), by largest remainder; ties go to the
    earlier key.

    >>> largest_remainder(10, {"a": 0.5, "b": 0.35, "c": 0.15})
    {'a': 5, 'b': 4, 'c': 1}
    >>> largest_remainder(0, {"a": 1})
    {'a': 0}
    """
    tot = sum(shares.values())
    raw = {k: total * v / tot for k, v in shares.items()} if tot else {k: 0 for k in shares}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:total - sum(out.values())]:
        out[k] += 1
    return out


def quotas(design):
    """``{(cell, label_type): n}`` for one city: each type's Disagree quota split over the Disagree cells by
    ``shares`` (largest remainder), plus ``decoys_per_type`` decoys.

    >>> q = quotas({"per_type": {"CurbRamp": 10}, "shares": {"margin": 0.5, "lone": 0.3, "light_voted": 0.2},
    ...             "decoys_per_type": 4})
    >>> q[("margin", "CurbRamp")], q[("lone", "CurbRamp")], q[("light_voted", "CurbRamp")], q[("decoy", "CurbRamp")]
    (5, 3, 2, 4)
    """
    out = {}
    for t, n in design["per_type"].items():
        for cell, k in largest_remainder(int(n), design["shares"]).items():
            out[(cell, t)] = k
        out[("decoy", t)] = int(design["decoys_per_type"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Pool, strata and sampling (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def margin_class(df):
    """The training target under the margin rule of ``data_generation.eligible_labels`` (``incorrect`` at net
    human Disagree > 1, ``correct`` at net Agree > 2, else empty), from the pool's human tallies.

    >>> d = pd.DataFrame({"h_agree": [0, 3, 2, 1], "h_dis": [2, 0, 0, 3]})
    >>> list(margin_class(d))
    ['incorrect', 'correct', '', 'incorrect']
    """
    net = df.h_agree - df.h_dis
    return pd.Series(np.where(net > 2, "correct", np.where(-net > 1, "incorrect", "")), index=df.index)


def assign_stratum(df):
    """The stratum of each row (empty string if none): for AI Disagrees ``lone`` (no human vote at all),
    ``margin_incorrect`` / ``margin_correct`` (a training row under the margin rule), ``light_voted`` (at least
    two decisive human votes with a majority, short of the margin rule); for AI Agree / Unsure the decoys.

    >>> d = pd.DataFrame({"ai_vote": ["Disagree"] * 5 + ["Agree", "Unsure"],
    ...                   "h_agree": [0, 0, 3, 2, 0, 0, 0], "h_dis": [0, 2, 0, 1, 0, 0, 0],
    ...                   "h_unsure": [0, 0, 0, 0, 1, 0, 0]})
    >>> list(assign_stratum(d))
    ['lone', 'margin_incorrect', 'margin_correct', 'light_voted', '', 'decoy_agree', 'decoy_unsure']
    """
    dec = df.h_agree + df.h_dis
    n_h = dec + df.h_unsure
    dis = df.ai_vote == "Disagree"
    mc = margin_class(df)
    return pd.Series(np.select(
        [dis & (n_h == 0), dis & (mc == "incorrect"), dis & (mc == "correct"),
         dis & (dec >= 2) & (df.h_agree != df.h_dis), df.ai_vote == "Agree", df.ai_vote == "Unsure"],
        ["lone", "margin_incorrect", "margin_correct", "light_voted", "decoy_agree", "decoy_unsure"], ""),
        index=df.index)


def sev_bucket(severity):
    """``1-2`` / ``3-5`` / ``none`` per label.

    >>> list(sev_bucket(pd.Series([1, 2.0, 3, 5, np.nan])))
    ['1-2', '1-2', '3-5', '3-5', 'none']
    """
    s = pd.to_numeric(severity, errors="coerce")
    return pd.Series(np.where(s.isna(), "none", np.where(s <= 2, "1-2", "3-5")), index=severity.index)


def eligible(pool):
    """Rows of ``LABEL_TYPES`` that no expert in the pool has voted on (every ``expert<N>`` column empty) or placed
    (every ``expert<N>_labeled`` false), and whose author is not excluded (``author_excluded`` false, when the pool
    has that column). Excluding every rater's labels from every city keeps the shared items fair to both raters;
    excluded authors' labels never appear in Gallery, so a rater could not vote on them.

    >>> p = pd.DataFrame({"label_type": ["CurbRamp"] * 5 + ["Other"],
    ...                   "expert1": [None, "Agree", None, None, None, None],
    ...                   "expert2": [None, None, None, "Unsure", None, None],
    ...                   "expert1_labeled": ["f", "f", "t", "f", "f", "f"], "expert2_labeled": ["f"] * 6,
    ...                   "author_excluded": ["f", "f", "f", "f", "t", "f"]})
    >>> list(eligible(p).index)
    [0]
    """
    ok = pool.label_type.isin(LABEL_TYPES)
    if "author_excluded" in pool.columns:
        ok &= ~pool.author_excluded.astype(str).str.lower().isin(["t", "true"])
    for c in pool.columns:
        if c.startswith("expert") and c.endswith("_labeled"):
            ok &= ~pool[c].astype(str).str.lower().isin(["t", "true"])
        elif c.startswith("expert") and c[6:].isdigit():
            ok &= pool[c].isna() | (pool[c].astype(str) == "")
    return pool[ok]


def prepare(pool, schema):
    """The eligible rows of one city with ``stratum``, ``cell``, ``margin_class`` and ``sev_bucket``; rows in no
    stratum dropped."""
    d = eligible(pool[pool.schema == schema]).copy()
    d["stratum"] = assign_stratum(d)
    d = d[d.stratum != ""].copy()
    d["cell"] = d.stratum.where(~d.stratum.str.startswith("decoy"), "decoy")
    d["cell"] = d.cell.where(~d.cell.str.startswith("margin"), "margin")
    d["margin_class"] = margin_class(d)
    d["sev_bucket"] = sev_bucket(d.severity)
    return d


def bucket_shares(prepared):
    """``{(cell, label_type): {bucket: count}}`` of the eligible pool: the severity mix each Disagree cell is filled
    in proportion to."""
    g = prepared[prepared.cell != "decoy"].groupby(["cell", "label_type", "sev_bucket"]).size()
    out = {}
    for (cell, t, b), n in g.items():
        out.setdefault((cell, t), {})[b] = int(n)
    return out


def candidates(prepared, seed, design, oversample=rl.OVERSAMPLE):
    """The ranked candidates of one city, ``rank`` ascending, up to ``quota * oversample`` per cell and type.
    In the ``margin`` cell ``margin_correct`` rows are capped at ``margin_correct_cap * quota`` (times
    ``oversample``) and come first (``prio`` 0); ``margin_incorrect`` rows top the cell up. In ``decoy``,
    ``decoy_agree`` rows come before ``decoy_unsure``."""
    d = prepared.copy()
    d["rank"] = [rl.seeded_rank(seed, u) for u in d.label_uid]
    d["prio"] = np.where(d.stratum.isin(["margin_correct", "decoy_agree"]), 0, 1)
    d = d.sort_values(["cell", "label_type", "prio", "rank"], kind="stable")
    q = quotas(design)
    parts = []
    for (cell, t), g in d.groupby(["cell", "label_type"], sort=False):
        n = q[(cell, t)]
        if cell == "margin":
            cap = margin_correct_cap(design, n)
            parts.append(g[g.stratum == "margin_correct"].head(cap * oversample))
            parts.append(g[g.stratum == "margin_incorrect"].head(n * oversample))
        else:
            parts.append(g.head(n * oversample))
    return pd.concat(parts) if parts else d.iloc[0:0]


def margin_correct_cap(design, quota):
    """The most ``margin_correct`` rows the margin cell of one type may hold.

    >>> margin_correct_cap({"margin_correct_cap": 0.4}, 5), margin_correct_cap({"margin_correct_cap": 0.4}, 0)
    (2, 0)
    """
    return int(np.floor(float(design["margin_correct_cap"]) * quota))


def fill(rows, quota, shares):
    """The first ``quota`` rows (already in priority order) filling severity buckets in proportion to ``shares``
    (``{bucket: count}``), then whatever is left in order: a short bucket spills into the others.

    >>> r = pd.DataFrame({"sev_bucket": ["1-2", "1-2", "3-5", "1-2", "none", "3-5"]})
    >>> list(fill(r, 4, {"1-2": 2, "3-5": 2}).index)
    [0, 1, 2, 5]
    >>> list(fill(r, 3, {"none": 1, "1-2": 1}).index)
    [4, 0, 1]
    """
    want = largest_remainder(quota, shares) if shares else {}
    taken, idx = [], []
    for b, n in want.items():
        sel = rows[(rows.sev_bucket == b) & ~rows.index.isin(idx)].head(n)
        idx.extend(sel.index)
    rest = rows[~rows.index.isin(idx)].head(quota - len(idx))
    idx.extend(rest.index)
    return rows.loc[idx]


def pick(cands, available, design, shares):
    """``(picked, shortfall)`` for one city: each cell and type takes its quota from the candidates whose
    ``label_uid`` is in ``available`` (a crop exists), Disagree cells by :func:`fill` over the severity buckets of
    ``shares`` (:func:`bucket_shares`), the margin cell with at most :func:`margin_correct_cap` ``margin_correct``
    rows and ``margin_incorrect`` rows for the rest, decoys in order. ``shortfall`` maps a short cell to what it is
    missing."""
    ok = cands[cands.label_uid.isin(available)]
    q = quotas(design)
    parts, shortfall = [], {}
    for (cell, t), n in q.items():
        g = ok[(ok.cell == cell) & (ok.label_type == t)]
        sh = shares.get((cell, t), {})
        if cell == "decoy":
            sel = g.head(n)
        elif cell == "margin":
            mc = fill(g[g.stratum == "margin_correct"], margin_correct_cap(design, n), sh)
            sel = pd.concat([mc, fill(g[g.stratum == "margin_incorrect"], n - len(mc), sh)])
        else:
            sel = fill(g, n, sh)
        if len(sel) < n:
            shortfall[f"{cell}/{t}"] = n - len(sel)
        parts.append(sel)
    picked = pd.concat(parts) if parts else ok.iloc[0:0]
    return picked.reset_index(drop=True), shortfall


def mark_shared(picked, seed, n):
    """``shared`` flag on ``n`` of the city's items, spread over (cell, type) in proportion (largest remainder), each
    stratum's items chosen by a seeded rank; decoys included, so the shared items do not reveal the AI's verdict.

    >>> p = pd.DataFrame({"label_uid": [f"s:{i}" for i in range(8)], "cell": ["lone"] * 4 + ["decoy"] * 4,
    ...                   "label_type": ["CurbRamp"] * 8})
    >>> int(mark_shared(p, "s", 4).shared.sum()), mark_shared(p, "s", 4).groupby("cell").shared.sum().to_dict()
    (4, {'decoy': 2, 'lone': 2})
    """
    counts = picked.groupby(["cell", "label_type"]).size()
    want = largest_remainder(min(n, len(picked)), {k: int(v) for k, v in counts.items()})
    shared = pd.Series(False, index=picked.index)
    for (cell, t), k in want.items():
        g = picked[(picked.cell == cell) & (picked.label_type == t)]
        order = sorted(g.index, key=lambda i: rl.seeded_rank(f"{seed}:shared", picked.label_uid[i]))
        shared[order[:k]] = True
    return picked.assign(shared=shared)


def round3_splits(split_dir):
    """``{label_uid_city: split}`` over every label type's pinned ``split.csv`` (``train`` / ``val`` / ``test`` /
    ``test_heldout``), and the files' sha256s."""
    out, shas = {}, {}
    for t in LABEL_TYPES:
        p = os.path.join(split_dir, t, "split.csv")
        s = pd.read_csv(p, usecols=["label_uid", "split"])
        out.update(dict(zip(s.label_uid, s.split)))
        shas[t] = rl.sha256_file(p)
    return out, shas


def city_ids(schemas, cityparams_path):
    """``{schema: city_id}`` for the given schemas from SidewalkWebpage's ``cityparams.conf``; raises on a schema it
    does not list."""
    schema_city, _ = rl.cityparams(open(cityparams_path).read())
    missing = sorted(s for s in schemas if s not in schema_city)
    if missing:
        raise ValueError(f"no city_id in cityparams.conf for {missing}")
    return {s: schema_city[s] for s in schemas}


def rater_items(key, rater):
    """The key rows one rater rates: their own city's items, then the other cities' shared items, in key order."""
    own = key[key.rater == rater]
    other = key[(key.rater != rater) & key.shared.astype(str).str.lower().isin(["true", "1"])]
    return pd.concat([own, other])


# ---------------------------------------------------------------------------------------------------------------------
# Scoring (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def score_table(m, population=None):
    """Per stratum and (stratum, type) of one rater's items: the share of AI votes the expert confirms
    (``cant_tell`` and blank excluded, counted apart); on the margin strata also how often the training target
    (the crowd's margin class) matches the expert. ``population`` (``{stratum: {type: n}}``, the eligible pool of
    the rater's own city) adds a population-weighted row per Disagree stratum."""
    rows = []
    decided = m[m.verdict.isin(["correct", "incorrect"])]
    for s in STRATA:
        for t in ("all",) + tuple(LABEL_TYPES):
            sub = m[(m.stratum == s) & ((m.label_type == t) if t != "all" else True)]
            if not len(sub):
                continue
            g = decided[decided.index.isin(sub.index)]
            right = sum(rl.ai_right(v, r) for v, r in zip(g.ai_vote, g.verdict))
            row = {"stratum": s, "label_type": t, "cant_tell": int((sub.verdict == "cant_tell").sum()),
                   "unanswered": int((sub.verdict == "").sum()), **rl._rate(right, len(g))}
            if s.startswith("margin"):
                ok = sum(c == r for c, r in zip(g.margin_class, g.verdict))
                c = rl._rate(ok, len(g))
                row.update(target_n=c["n"], target_rate=c["rate"], target_lo=c["lo"], target_hi=c["hi"])
            rows.append(row)
    out = pd.DataFrame(rows)
    if population:
        for s, weights in population.items():
            g = decided[decided.stratum == s]
            per_type = {t: (sum(rl.ai_right(v, r) for v, r in zip(x.ai_vote, x.verdict)), len(x))
                        for t, x in ((t, g[g.label_type == t]) for t in LABEL_TYPES)}
            i = out.index[(out.stratum == s) & (out.label_type == "all")]
            if len(i):
                out.loc[i, ["w_rate", "w_lo", "w_hi"]] = rl.weighted_rate(per_type, weights)
    return out


def score_markdown(t):
    cols = ["w_rate" in t.columns, "target_rate" in t.columns]
    head = "| Stratum | Type | Expert confirms the AI | n | can't tell |" + \
        (" population-weighted |" if cols[0] else "") + (" training target matches expert |" if cols[1] else "")
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in t.itertuples():
        line = f"| {r.stratum} | {r.label_type} | {rl._pct(r.rate, r.lo, r.hi)} | {r.n} | {r.cant_tell} |"
        if cols[0]:
            line += (f" {rl._pct(r.w_rate, r.w_lo, r.w_hi)} |" if r.label_type == "all" and not pd.isna(r.w_rate)
                     else " |")
        if cols[1]:
            line += (f" {rl._pct(r.target_rate, r.target_lo, r.target_hi)} ({int(r.target_n)}) |"
                     if not pd.isna(getattr(r, "target_n", np.nan)) else " |")
        out.append(line)
    return "\n".join(out)


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label sequences (nan when either rater used one label only).

    >>> round(cohen_kappa(["c", "c", "i", "i"], ["c", "c", "i", "c"]), 3)
    0.5
    >>> cohen_kappa(["c", "c"], ["c", "c"])
    nan
    """
    a, b = pd.Series(list(a)), pd.Series(list(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    labels = sorted(set(a) | set(b))
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agreement_table(key, sheets):
    """Per type and overall, on the items every rater in ``sheets`` (``{rater: sheet DataFrame}``) answered with
    ``correct`` or ``incorrect``: percent agreement with its Wilson interval, Cohen's kappa, n, and each rater's
    agreement with the AI (Disagree items only). Two raters only."""
    if len(sheets) != 2:
        raise ValueError("agreement needs exactly two raters' sheets")
    (ra, sa), (rb, sb) = sheets.items()
    shared = key[key.shared.astype(str).str.lower().isin(["true", "1"])]
    m = shared.merge(sa[["packet_id", "verdict"]].rename(columns={"verdict": "va"}), on="packet_id", how="left",
                     validate="one_to_one")
    m = m.merge(sb[["packet_id", "verdict"]].rename(columns={"verdict": "vb"}), on="packet_id", how="left",
                validate="one_to_one")
    dec = m[m.va.isin(["correct", "incorrect"]) & m.vb.isin(["correct", "incorrect"])]
    rows = []
    for t in ("all",) + tuple(LABEL_TYPES):
        g = dec if t == "all" else dec[dec.label_type == t]
        if not len(g):
            continue
        r = rl._rate(int((g.va == g.vb).sum()), len(g))
        d = g[g.ai_vote == "Disagree"]
        rows.append({"label_type": t, "n": r["n"], "agree": r["rate"], "lo": r["lo"], "hi": r["hi"],
                     "kappa": cohen_kappa(g.va, g.vb), "n_disagree": len(d),
                     f"{ra}_confirms_ai": float((d.va == "incorrect").mean()) if len(d) else np.nan,
                     f"{rb}_confirms_ai": float((d.vb == "incorrect").mean()) if len(d) else np.nan,
                     "shared_items": int(len(m if t == "all" else m[m.label_type == t]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def _rater_page(key_r, packet_dir, rater, design_name):
    items, fixed = [], {}
    for i, r in enumerate(key_r.itertuples(), 1):
        sv = rl.street_view_url(r.pano_id, r.heading, r.pitch)
        items.append({"id": r.packet_id, "image": f"../items/{r.packet_id}.jpg",
                      "alt": f"Item {i}: a {r.label_type} label, ringed in yellow",
                      "caption_html": f"{i}. {r.label_type} &middot; {r.packet_id}",
                      "extra_html": f'<p><a href="{sv}" target="_blank" rel="noopener noreferrer">'
                                    f'Open in Street View</a></p>'})
        fixed[r.packet_id] = {"label_type": r.label_type}
    fields = [{"name": "verdict", "legend": "Is this label correct?", "kind": "radio", "options": rl.VERDICTS},
              {"name": "notes", "legend": "Notes (optional)", "kind": "text"}]
    page = render_page(title=f"GT round: rater {rater}", heading=f"Rater {rater}: is the label correct?",
                       intro_html=rl.intro_html() + RATING_RULE_HTML, items=items, fields=fields,
                       sheet_columns=SHEET_COLUMNS, fixed=fixed, storage_key=f"gt_{design_name}_{rater}",
                       export_name=f"sheet_{rater}")
    d = os.path.join(packet_dir, f"rater_{rater}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rate.html"), "w") as f:
        f.write(page)
    pd.DataFrame({"packet_id": key_r.packet_id, "label_type": key_r.label_type, "verdict": "", "notes": ""}).to_csv(
        os.path.join(d, "sheet.csv"), index=False)


def cmd_build(args):
    for d in (args.packet_dir, args.key_dir):
        if rl.inside_git_checkout(d):
            sys.exit(f"refusing to write the packet or its key inside a git checkout: {d}")
    design = load_design(args.design)
    design_name = os.path.splitext(os.path.basename(args.design))[0]
    pool = rl.load_pool(args.pool)
    missing = sorted(set(design["raters"].values()) - set(pool.schema))
    if missing:
        sys.exit(f"the pool has no rows for {missing}")
    prepared = {s: prepare(pool, s) for s in design["raters"].values()}
    cands = {s: candidates(p, args.seed, design, args.oversample) for s, p in prepared.items()}
    all_cands = pd.concat(cands.values())
    print(f"{len(all_cands):,} candidates over {len(cands)} cities")
    dir_map, markers = rl.crop_status(args.host, all_cands)
    picked, shortfall, population = [], {}, {}
    for rater, schema in design["raters"].items():
        p, short = pick(cands[schema], set(markers), design, bucket_shares(prepared[schema]))
        shortfall.update({f"{rater}:{k}": v for k, v in short.items()})
        p = mark_shared(p, f"{args.seed}:{schema}", int(design["shared"])).assign(rater=rater)
        picked.append(p)
        e = prepared[schema]
        population[rater] = {s: {t: int(((e.stratum == s) & (e.label_type == t)).sum()) for t in LABEL_TYPES}
                             for s in DISAGREE_STRATA}
    picked = pd.concat(picked, ignore_index=True)
    if shortfall:
        print(f"WARNING: cells short of their quota: {shortfall}")
    picked["marker_x"] = [markers[u][0] for u in picked.label_uid]
    picked["marker_y"] = [markers[u][1] for u in picked.label_uid]
    cid = city_ids(sorted(picked.schema.unique()), args.cityparams)
    picked["city_id"] = picked.schema.map(cid)
    picked["label_uid_city"] = picked.city_id + ":" + picked.label_id.astype(str)
    splits, split_shas = ({}, {})
    if args.round3_splits:
        splits, split_shas = round3_splits(args.round3_splits)
    picked["round3_split"] = picked.label_uid_city.map(splits).fillna("")
    key = rl.make_key(picked, args.seed)
    crops = rl.fetch_crops(args.host, key, dir_map)
    lost = set(key.label_uid) - set(crops)
    if lost:
        sys.exit(f"{len(lost)} crops vanished between the check and the fetch; rerun")
    os.makedirs(os.path.join(args.packet_dir, "items"), exist_ok=True)
    os.makedirs(args.key_dir, exist_ok=True)
    for r in key.itertuples():
        with open(os.path.join(args.packet_dir, "items", f"{r.packet_id}.jpg"), "wb") as f:
            f.write(rl.marked_image(crops[r.label_uid], r.marker_x, r.marker_y))
    for rater in design["raters"]:
        _rater_page(rater_items(key, rater), args.packet_dir, rater, design_name)
    kpath = os.path.join(args.key_dir, "key.csv")
    key[KEY_COLUMNS].to_csv(kpath, index=False)
    counts = key.groupby(["rater", "stratum", "label_type"]).size()
    sev = key[key.cell != "decoy"].groupby(["rater", "sev_bucket"]).size()
    manifest = {"design": design_name, "design_sha256": rl.sha256_file(args.design), "design_json": design,
                "pool_sha256": rl.sha256_file(args.pool), "key_sha256": rl.sha256_file(kpath),
                "seed_sha256": hashlib.sha256(str(args.seed).encode()).hexdigest(), "items": len(key),
                "quotas": {f"{c}/{t}": n for (c, t), n in quotas(design).items()}, "shortfall": shortfall,
                "population": population, "city_ids": cid,
                "counts": {f"{r}/{s}/{t}": int(n) for (r, s, t), n in counts.items()},
                "severity_buckets": {f"{r}/{b}": int(n) for (r, b), n in sev.items()},
                "shared": {r: int(key[(key.rater == r) & key.shared].shape[0]) for r in design["raters"]},
                "round3_in_train": {r: int(((key.rater == r) & (key.round3_split == "train")).sum())
                                    for r in design["raters"]},
                "rating": {"round3_splits": split_shas} if split_shas else {}}
    with open(os.path.join(args.key_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(key)} items ({manifest['shared']} shared) -> {args.packet_dir}/rater_*/rate.html\n"
          f"key -> {kpath} (sha256 {manifest['key_sha256']})")


def gallery_index_html(rater, lists, total):
    rows = "".join(f'<li><a href="{html.escape(rl.gallery_url(b, ids))}" target="_blank" rel="noopener">'
                   f'{html.escape(c)}</a> &middot; {len(ids)} label{"s" if len(ids) != 1 else ""}'
                   f'{" (shared with the other rater)" if shared else ""}</li>' for c, b, ids, shared in lists)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GT round: rater {html.escape(rater)}</title><style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1f2328; --muted:#57606a; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 48px; max-width:900px; }}
pre {{ background:var(--panel); padding:12px; overflow-x:auto; white-space:pre-wrap; }} li {{ margin:4px 0; }}
</style></head><body>
<h1>Rater {html.escape(rater)}: {total} labels in {len(lists)} lists</h1>
{RATING_RULE_HTML}
<p>Mask the votes first: the CSS below in Stylus, or the console snippet on each page.</p>
<h2>Review lists</h2><ul>{rows}</ul>
<h2>Masking CSS (for Stylus; not for the console)</h2><pre>{html.escape(rl.GALLERY_MASK_CSS)}</pre>
<h2>Console snippet (JavaScript; paste into the developer console)</h2><pre>{html.escape(rl.mask_console_js())}</pre>
</body></html>
"""


def cmd_gallery(args):
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write the review lists inside a git checkout: {args.out}")
    key = pd.read_csv(args.key, dtype={"tags": str})
    items = rater_items(key, args.rater)
    if not len(items):
        sys.exit(f"no items for rater {args.rater}")
    conf_text = open(args.cityparams).read()
    schema_city, city_url = rl.cityparams(conf_text)
    lists = []
    for shared_part, part in ((False, items[items.rater == args.rater]), (True, items[items.rater != args.rater])):
        for sch, g in part.groupby("schema", sort=False):
            if city_url.get(schema_city.get(sch)) is None:
                sys.exit(f"no deployment URL for {sch}")
            ids = [int(i) for i in g.label_id]
            for j in range(0, len(ids), 500):
                lists.append((schema_city[sch], city_url[schema_city[sch]], ids[j:j + 500], shared_part))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(gallery_index_html(args.rater, lists, len(items)))
    with open(os.path.join(args.out, "gallery-mask.css"), "w") as f:
        f.write(rl.GALLERY_MASK_CSS)
    urls = pd.DataFrame([{"order": i, "city_id": c, "shared": s, "n": len(ids), "label_ids": " ".join(map(str, ids)),
                          "url": rl.gallery_url(b, ids)} for i, (c, b, ids, s) in enumerate(lists, 1)])
    upath = os.path.join(args.out, "gallery_urls.csv")
    urls.to_csv(upath, index=False)
    cpath = os.path.join(args.out, "cityparams_snapshot.conf")
    with open(cpath, "w") as f:
        f.write(conf_text)
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(args.cityparams)), "log", "-1",
                             "--format=%H %cI", "--", os.path.basename(args.cityparams)],
                            capture_output=True, text=True).stdout.strip()
    mpath = args.manifest or rl.default_manifest(args.key)
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        man.setdefault("gallery", {})[args.rater] = {
            "urls_sha256": rl.sha256_file(upath), "cityparams_sha256": rl.sha256_file(cpath),
            "cityparams_commit": commit, "lists": len(lists), "labels": int(urls.n.sum()),
            "css_sha256": hashlib.sha256(rl.GALLERY_MASK_CSS.encode()).hexdigest()}
        rl.write_manifest(mpath, man)
    print(f"{int(urls.n.sum())} labels in {len(lists)} lists -> {args.out}/index.html, {upath}")


def cmd_votes(args):
    rl.refuse_overwrite(args.out, args.repin)
    key = pd.read_csv(args.key, dtype={"tags": str})
    items = rater_items(key, args.rater)
    out = rl.remote(args.host, rl.votes_script(items, args.rater_id))
    sheet = rl.votes_to_sheet(items, pd.read_csv(io.StringIO(out)))
    sheet.to_csv(args.out, index=False)
    rl.pin(args.manifest or rl.default_manifest(args.key), f"votes_{args.rater}", args.out, len(sheet), args.repin)
    print(f"{int((sheet.verdict != '').sum())} of {len(sheet)} items voted -> {args.out} (pinned)")


def _read_sheet(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cmd_score(args):
    key = pd.read_csv(args.key, dtype={"tags": str})
    mpath = args.manifest or rl.default_manifest(args.key)
    rl.check_pin(mpath, "key", args.key, args.allow_unpinned)
    rl.check_pin(mpath, f"votes_{args.rater}", args.sheet, args.allow_unpinned)
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    items = rater_items(key, args.rater)
    m = rl.unmask(items, _read_sheet(args.sheet))
    own = m[m.rater == args.rater]
    print(f"## Rater {args.rater}, own city ({len(own)} items)\n")
    print(score_markdown(score_table(own, man.get("population", {}).get(args.rater))))
    other = m[m.rater != args.rater]
    if len(other):
        print(f"\n## Rater {args.rater}, shared items from the other city ({len(other)} items)\n")
        print(score_markdown(score_table(other)))
    print("\ndecoy rows are the share the expert called correct (AI Agree, or abstained for decoy_unsure).")
    if (m.verdict == "").sum():
        print(f"\nWARNING: {int((m.verdict == '').sum())} items unanswered")


def cmd_agreement(args):
    key = pd.read_csv(args.key, dtype={"tags": str})
    sheets = {}
    for spec in args.sheets:
        rater, path = spec.split("=", 1)
        sheets[rater] = _read_sheet(path)
    t = agreement_table(key, sheets)
    ra, rb = list(sheets)
    print(f"| Type | shared items | both decided | agreement | kappa | AI Disagrees | {ra} confirms AI | {rb} confirms AI |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for r in t.itertuples():
        print(f"| {r.label_type} | {r.shared_items} | {r.n} | {rl._pct(r.agree, r.lo, r.hi)} | {r.kappa:.2f} | "
              f"{r.n_disagree} | {getattr(r, f'{ra}_confirms_ai'):.0%} | {getattr(r, f'{rb}_confirms_ai'):.0%} |")
    if args.out:
        t.to_csv(args.out, index=False)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample every rater's city, check and fetch crops, write packets and the key")
    b.add_argument("--design", required=True, help="the round's design JSON (analysis/gt_rounds/<round>.json)")
    b.add_argument("--pool", required=True, help="pull_relabel_pool.sh output with every rater's expert columns")
    b.add_argument("--packet-dir", required=True, help="items/ and rater_<R>/rate.html (outside any git checkout)")
    b.add_argument("--key-dir", required=True, help="key.csv and manifest.json (private; outside any git checkout)")
    b.add_argument("--seed", required=True, help="secret; keep it until the rating is done")
    b.add_argument("--round3-splits", help="run dir with <Type>/split.csv, to record each item's round-3 split")
    b.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    b.add_argument("--host", default="makelab1")
    b.add_argument("--oversample", type=int, default=rl.OVERSAMPLE)
    b.set_defaults(func=cmd_build)
    g = sub.add_parser("gallery", help="one rater's Gallery review lists (own city, then the shared items)")
    g.add_argument("--key", required=True)
    g.add_argument("--rater", required=True)
    g.add_argument("--out", required=True, help="directory for index.html (outside any git checkout)")
    g.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    g.add_argument("--manifest")
    g.set_defaults(func=cmd_gallery)
    v = sub.add_parser("votes", help="one rater's prod votes on their items, as a filled sheet (pinned)")
    v.add_argument("--key", required=True)
    v.add_argument("--rater", required=True)
    v.add_argument("--rater-id", required=True, help="the rater's user_id (pass it in; never commit it)")
    v.add_argument("--out", required=True)
    v.add_argument("--host", default="makelab1")
    v.add_argument("--manifest")
    v.add_argument("--repin", action="store_true")
    v.set_defaults(func=cmd_votes)
    s = sub.add_parser("score", help="one rater's sheet against the key")
    s.add_argument("--key", required=True)
    s.add_argument("--rater", required=True)
    s.add_argument("--sheet", required=True)
    s.add_argument("--manifest")
    s.add_argument("--allow-unpinned", action="store_true")
    s.set_defaults(func=cmd_score)
    a = sub.add_parser("agreement", help="inter-rater agreement on the shared items")
    a.add_argument("--key", required=True)
    a.add_argument("--sheets", nargs=2, required=True, metavar="RATER=SHEET")
    a.add_argument("--out", help="CSV of the table")
    a.set_defaults(func=cmd_agreement)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
