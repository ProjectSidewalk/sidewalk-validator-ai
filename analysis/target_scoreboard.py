"""Target-cleaning methods scored against the expert gold set of rounds 1 and 2 (issue #74).

Why
---
Issue #74 asks which automatic way of cleaning the crowd's Disagree targets agrees best with an expert, before any
of them rewrites a training target. The expert gold set (``adjudicator.py gold``: 800 items, rounds 1 and 2 of the
masked re-label) has been complete since 2026-11-02. This module scores the methods that need no GPU against it:
the stricter vote rules of #75 (``symmetric``, ``unanimous3``), confident learning of #76 (``cl-drop``,
``cl-flip``), and the plain vote rules around them. The VLM adjudicator of #77 is scored by ``adjudicator.py agree``
once its klone scores exist; its rows join this scoreboard in the PR that brings those scores back.

What is measured, and what is not
---------------------------------
The gold set is a stratified sample drawn on the same vote tallies the rules read, so a rule's *coverage* on the
gold set restates the sampling design. Coverage and drop shares therefore come from the full AI-Disagree pool
(``relabel_pool.csv``, exact counts, no interval); the gold set is used only for how often a verdict is right.
Two contrasts were fixed in ``docs/target-scoreboard-2026-11.md`` before any number was computed:

* **P1** ``symmetric`` kept against dropped, on the AI Disagrees whose margin target is ``incorrect``: the share
  the expert calls incorrect among the rows the stricter rule keeps, minus among the rows it drops.
* **P2** confident-learning flagged against unflagged, on the decided round-3 train rows: the share of wrong
  targets among flagged rows minus among unflagged rows.

The pooled row of each is the pre-stated estimate (``ROLES``); it is printed after the per-round rows and a
heterogeneity check, which are its check. Each has Wilson intervals per group, a label-level bootstrap interval for
the difference (``N_BOOT`` resamples) and a two-sided Fisher exact p. Every other table is descriptive.

Stages (subcommands)
--------------------
``score``
    Reads the private gold table, the pool, the round-3 splits and their confident-learning outputs, the round-1
    manifest and the round-2 recount; writes the per-label verdicts (private, refused inside a git checkout) and
    ``scoreboard.json`` (counts and rates only) after a privacy check against the live ``/v3/api/cities``.
``report``
    Renders ``scoreboard.json`` into the doc's generated blocks. ``--check`` exits 1 when the doc differs from
    what the JSON renders (no network: it writes nothing).

Keys: ``label_uid`` is the city form ``city_id:label_id`` (label ids restart per deployment); every load asserts it
is unique and every merge passes ``validate=``. No rater id is read; the gold table's ``raters`` column is never
used or written.

Usage (the full commands are in ``docs/target-scoreboard-2026-11.md``, Reproducing)::

    python analysis/target_scoreboard.py score --gold $ADJ/gold.csv --pool $R/relabel_pool.csv \\
        --split-root $DATA --r1-manifest $R/private-key-do-not-share/manifest.before-rating.json \\
        --out-dir ~/Data/sidewalk-validator-ai-target-scoreboard-2026-11
    python analysis/target_scoreboard.py report

Example (the vote rules on one tally each; ``""`` means the rule leaves no target)::

    >>> t = tally_verdicts([0, 2, 0, 1], [2, 0, 3, 4])
    >>> t.loc[:, ["majority", "margin", "symmetric3", "unanimous3"]].values.tolist()
    [['incorrect', 'incorrect', '', ''], ['correct', '', '', ''], ['incorrect', 'incorrect', 'incorrect', 'incorrect'], ['incorrect', 'incorrect', 'incorrect', '']]
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import adjudicator as adj  # noqa: E402
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis import confident_learning as cl  # noqa: E402
from analysis import gt_rounds as gr  # noqa: E402
from analysis import slices as sl  # noqa: E402
import data_generation as dg  # noqa: E402

LABEL_TYPES = tuple(adj.LABEL_TYPES)
DEFAULT_OUT = os.path.expanduser("~/Data/sidewalk-validator-ai-target-scoreboard-2026-11")
DOC = os.path.join(REPO_DIR, "docs", "target-scoreboard-2026-11.md")
PUBLIC_DIR = os.path.join(REPO_DIR, "docs", "target-scoreboard-2026-11")
R2_DESIGN = os.path.join(REPO_DIR, "analysis", "gt_rounds", "round2.json")
R2_RECOUNT = os.path.join(REPO_DIR, "docs", "ai-disagree-relabel-2026-10", "round2_recount.json")
GOLD_SHA256 = "2b472651cb01011a8640b86eb4103e1794ab482186ba3a3471e7d019273edcae"
"""The gold table pinned in ``docs/adjudicator-2026-11.md``; ``score`` records whether its input matches."""
TALLY_RULES = ("majority", "majority2", "symmetric2", "margin", "symmetric3", "unanimous3")
VARIANTS = cl.DEFAULT_VARIANTS  # symmetric, unanimous3, cl-drop, cl-flip
VOTE_CLASSES = ("no decisive vote", "1 vote", "tie", "light", "margin_incorrect", "margin_correct")
N_BOOT = 2000
SLICE_MIN_N = 20
MIN_CELL = gr.MIN_CELL
CONSENSUS = "r2_consensus"
BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/target_scoreboard.py) -->"
END = "<!-- END GENERATED: {name} -->"
BLOCKS = ("gold_counts", "pool", "primary", "weighted", "rules", "detector", "ranking", "slices", "sensitivity",
          "crosscheck", "provenance")
VERDICT_COLUMNS = ["label_uid", "round", "label_type", "stratum", "ai_vote", "expert", "expert_source", "masked",
                   "h_agree", "h_dis", "n_expert_votes"] + [f"rule_{r}" for r in TALLY_RULES] + [
                   "train_row", "target", "wrong", "oof_p_incorrect", "cl_flagged"] + [f"touch_{v}" for v in VARIANTS]


# ---------------------------------------------------------------------------------------------------------------------
# Small statistics, no SciPy (CI does not install it)
# ---------------------------------------------------------------------------------------------------------------------

def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p of the 2x2 table ``[[a, b], [c, d]]``: the summed probability of every table with the
    same margins that is no more likely than the observed one (the rule SciPy's ``fisher_exact`` uses).

    >>> round(fisher_exact(4, 9, 19, 99), 5)
    0.24247
    >>> fisher_exact(5, 5, 5, 5)
    1.0
    >>> round(fisher_exact(10, 0, 0, 10), 6)
    1.1e-05
    """
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0

    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return float(min(1.0, sum(q for q in (p(x) for x in range(lo, hi + 1)) if q <= p0 * (1 + 1e-7))))


def rate(k, n):
    """``{n, k, rate, lo, hi}`` with a Wilson 95% interval (``ai_disagree_relabel._rate``); blank rate when n is 0.

    >>> r = rate(4, 13); (r["n"], r["k"], round(r["rate"], 3), round(r["lo"], 3), round(r["hi"], 3))
    (13, 4, 0.308, 0.127, 0.576)
    """
    r = rl._rate(int(k), int(n))
    return {k_: (None if isinstance(v, float) and math.isnan(v) else v) for k_, v in r.items()}


def diff_boot(ok, in_a, n_boot=N_BOOT, seed=0):
    """``(diff, lo, hi)``: mean of ``ok`` in group A minus in group B, with a 95% percentile interval from resampling
    items with replacement (a resample that leaves either group empty is skipped). ``lo``/``hi`` are None when
    ``n_boot`` is 0 or a group is empty.

    >>> d, lo, hi = diff_boot([1, 1, 0, 0, 0, 1], [True, True, True, False, False, False], n_boot=0)
    >>> round(d, 3), lo, hi
    (0.333, None, None)
    """
    ok, a = np.asarray(ok, float), np.asarray(in_a, bool)
    if not a.any() or a.all():
        return None, None, None
    d = float(ok[a].mean() - ok[~a].mean())
    if not n_boot:
        return d, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ok), size=(n_boot, len(ok)))
    ga, o = a[idx], ok[idx]
    na, nb = ga.sum(1), (~ga).sum(1)
    keep = (na > 0) & (nb > 0)
    vals = (o * ga).sum(1)[keep] / na[keep] - (o * ~ga).sum(1)[keep] / nb[keep]
    return d, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gain_boot(ok, in_a, n_boot=N_BOOT, seed=0):
    """``(gain, lo, hi)``: mean of ``ok`` in group A minus in all items (A and B together), with a 95% percentile
    interval from resampling items with replacement. This is what keeping only A changes against keeping everything;
    it equals the A-minus-B difference times B's share of the items, so it is the same comparison on another scale.
    None as in :func:`diff_boot`.

    >>> g, lo, hi = gain_boot([1, 1, 0, 0, 0, 1], [True, True, True, False, False, False], n_boot=0)
    >>> round(g, 3), lo, hi
    (0.167, None, None)
    """
    ok, a = np.asarray(ok, float), np.asarray(in_a, bool)
    if not a.any() or a.all():
        return None, None, None
    g = float(ok[a].mean() - ok.mean())
    if not n_boot:
        return g, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ok), size=(n_boot, len(ok)))
    ga, o = a[idx], ok[idx]
    na = ga.sum(1)
    keep = (na > 0) & (na < len(ok))
    vals = (o * ga).sum(1)[keep] / na[keep] - o.mean(1)[keep]
    return g, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def did_boot(ok, in_a, group, n_boot=N_BOOT, seed=0):
    """``(did, lo, hi)``: the A-minus-B difference in the first value of ``group`` minus that in the second, with a
    95% interval from resampling items within each group (the heterogeneity check printed beside a pooled row).
    None when an A or a B group of either level has fewer than ``MIN_CELL`` items."""
    ok, a, g = np.asarray(ok, float), np.asarray(in_a, bool), np.asarray(group)
    levels = sorted(set(g))
    if len(levels) != 2:
        return None, None, None
    parts = [(ok[g == lv], a[g == lv]) for lv in levels]
    if any(min(x[1].sum(), (~x[1]).sum()) < MIN_CELL for x in parts):
        return None, None, None
    point = [o[m].mean() - o[~m].mean() for o, m in parts]
    rng = np.random.default_rng(seed)
    sims = []
    for o, m in parts:
        idx = rng.integers(0, len(o), size=(n_boot, len(o)))
        ga, oo = m[idx], o[idx]
        na, nb = ga.sum(1), (~ga).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sims.append((oo * ga).sum(1) / na - (oo * ~ga).sum(1) / nb)
    v = sims[0] - sims[1]
    v = v[np.isfinite(v)]
    return float(point[0] - point[1]), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------------------------------------------------------------
# Vote rules
# ---------------------------------------------------------------------------------------------------------------------

def tally_verdicts(h_agree, h_dis):
    """One column per rule in ``TALLY_RULES``: ``correct`` / ``incorrect`` / ``""`` (the rule leaves no target).

    ``majority`` is ``adjudicator.crowd_class`` (any majority, the #74 baseline); ``majority2``, ``symmetric2``,
    ``margin`` and ``symmetric3`` are ``data_generation.prior_class``; ``unanimous3`` is the margin target kept when
    ``confident_learning.passes_rule(..., "unanimous3")`` passes, the way the trained variant was built. The
    ``symmetric3`` column is derived a second way, as the margin target kept by ``passes_rule(..., "symmetric")``,
    and the two must agree on every row (so this scoreboard is bound to the rule that was trained).
    """
    a = pd.Series(pd.to_numeric(pd.Series(h_agree), errors="coerce").fillna(0).astype(int).to_numpy())
    d = pd.Series(pd.to_numeric(pd.Series(h_dis), errors="coerce").fillna(0).astype(int).to_numpy())
    out = pd.DataFrame(index=a.index)
    out["majority"] = adj.crowd_class(a, d).to_numpy()
    for r in ("majority2", "symmetric2", "margin", "symmetric3"):
        out[r] = dg.prior_class(a.to_numpy(), d.to_numpy(), r)
    m = pd.DataFrame({"label_class": out["margin"], "class_source": "margin", "human_agree": a.astype(str),
                      "human_disagree": d.astype(str)})
    has = m.label_class != ""
    for rule, col in (("unanimous3", "unanimous3"), ("symmetric", "_symmetric_cl")):
        out[col] = np.where(has & cl.passes_rule(m, rule), m.label_class, "")
    if not (out["_symmetric_cl"] == out["symmetric3"]).all():
        raise AssertionError("symmetric3 (data_generation) and symmetric (confident_learning) disagree")
    return out.drop(columns="_symmetric_cl")[list(TALLY_RULES)]


def vote_class(h_agree, h_dis):
    """The pool's vote-count classes (``VOTE_CLASSES``) from decisive human votes (Unsure votes are not counted).

    >>> vote_class([0, 1, 1, 2, 0, 4], [0, 0, 1, 1, 2, 1]).tolist()
    ['no decisive vote', '1 vote', 'tie', 'light', 'margin_incorrect', 'margin_correct']
    """
    a, d = np.asarray(h_agree, int), np.asarray(h_dis, int)
    net, tot = a - d, a + d
    return pd.Series(np.select(
        [tot == 0, tot == 1, net == 0, net <= -2, net >= 3],
        ["no decisive vote", "1 vote", "tie", "margin_incorrect", "margin_correct"], "light"))


def pool_shares(pool):
    """Exact shares on every AI Disagree of the pool, per type and ``all``: the vote classes and, per rule, the share
    given any target (``cover``) and the share given an ``incorrect`` target (``incorrect``). Counts, no ids."""
    p = pool[pool.ai_vote == "Disagree"]
    a = pd.to_numeric(p.h_agree, errors="coerce").fillna(0).astype(int).to_numpy()
    d = pd.to_numeric(p.h_dis, errors="coerce").fillna(0).astype(int).to_numpy()
    v = tally_verdicts(a, d)
    v["label_type"] = p.label_type.to_numpy()
    v["vote_class"] = vote_class(a, d).to_numpy()
    rows = []
    for t in ("all",) + LABEL_TYPES:
        x = v if t == "all" else v[v.label_type == t]
        n = len(x)
        if not n:
            continue
        row = {"label_type": t, "n": int(n),
               "classes": {c: int((x.vote_class == c).sum()) for c in VOTE_CLASSES},
               "rules": {r: {"cover": int((x[r] != "").sum()), "incorrect": int((x[r] == "incorrect").sum())}
                         for r in TALLY_RULES}}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# The per-label frame
# ---------------------------------------------------------------------------------------------------------------------

def load_pool(path):
    """The pool pull as strings, with ``label_uid_schema`` (``schema:label_id``) asserted unique."""
    p = pd.read_csv(path, dtype=str, keep_default_na=False)
    p["label_uid_schema"] = p.schema + ":" + p.label_id
    if not p.label_uid_schema.is_unique:
        raise ValueError(f"{path}: schema:label_id is not unique")
    return p


def remove_expert_votes(gold, pool):
    """``gold`` with every vote of the pool's ``expert1`` / ``expert2`` columns (vote values only; no ids) taken
    out of ``h_agree`` / ``h_dis`` / ``h_unsure``, and ``n_expert_votes``. A rater's own earlier production vote
    must not count as crowd evidence against their verdict. Items not in the pool keep their tallies.

    >>> g = pd.DataFrame({"label_uid": ["c:1", "c:2"], "label_uid_schema": ["s:1", "s:2"], "h_agree": ["1", "0"],
    ...                   "h_dis": ["2", "3"], "h_unsure": ["0", "0"]})
    >>> p = pd.DataFrame({"label_uid_schema": ["s:1"], "expert1": [""], "expert2": ["Disagree"]})
    >>> remove_expert_votes(g, p)[["h_agree", "h_dis", "n_expert_votes"]].values.tolist()
    [[1, 1, 1], [0, 3, 0]]
    """
    cols = [c for c in ("expert1", "expert2") if c in pool.columns]
    m = gold[["label_uid_schema"]].merge(pool[["label_uid_schema"] + cols], on="label_uid_schema", how="left",
                                         validate="one_to_one").fillna("")
    out = gold.copy()
    for c in ("h_agree", "h_dis", "h_unsure"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out["n_expert_votes"] = 0
    for c in cols:
        v = m[c].to_numpy()
        for value, col in (("Agree", "h_agree"), ("Disagree", "h_dis"), ("Unsure", "h_unsure")):
            hit = v == value
            out.loc[hit, col] -= 1
            out.loc[hit, "n_expert_votes"] += 1
    if (out[["h_agree", "h_dis", "h_unsure"]] < 0).any().any():
        raise ValueError("an expert vote is not in the tallies it was subtracted from")
    return out


def touched_rows(split_root, label_type, variant):
    """``{label_uid: touched}`` for one type's train rows: dropped (``train_dropped``) or flipped by ``variant``."""
    base = cl.read_split(os.path.join(split_root, label_type, "split.csv"))
    v = cl.read_split(os.path.join(split_root, label_type, "cl", variant, "split.csv"))
    tr = base[base.split == "train"][["label_uid", "label_class"]]
    m = tr.merge(v[["label_uid", "label_class", "split"]].rename(columns={"label_class": "lc_v", "split": "split_v"}),
                 on="label_uid", how="left", validate="one_to_one")
    if m.split_v.isna().any():
        raise ValueError(f"{label_type}/{variant}: {int(m.split_v.isna().sum())} train rows are missing from the variant")
    t = (m.split_v == cl.TRAIN_DROPPED) | (m.label_class != m.lc_v)
    return dict(zip(m.label_uid, t))


def oof_scores(split_root, label_type):
    """``noisy_targets.csv`` of one type: ``label_uid``, ``oof_p_incorrect`` (out-of-fold) and ``flagged``."""
    n = pd.read_csv(os.path.join(split_root, label_type, "cl", "noisy_targets.csv"), dtype=str, keep_default_na=False)
    if not n.label_uid.is_unique:
        raise ValueError(f"{label_type}: noisy_targets label_uid is not unique")
    return pd.DataFrame({"label_uid": n.label_uid, "oof_p_incorrect": pd.to_numeric(n.p_incorrect),
                         "nt_flagged": n.flagged.isin(["1", "True", "true"])})


def split_files_needed(split_root, label_type):
    """The round-3 files one type needs: its split, its out-of-fold scores and every variant's split."""
    return [os.path.join(split_root, label_type, rel)
            for rel in ["split.csv", "cl/noisy_targets.csv"] + [f"cl/{v}/split.csv" for v in VARIANTS]]


def check_split_root(g, split_root, types=LABEL_TYPES):
    """Raise ``FileNotFoundError`` naming the missing files when a type that has gold train rows lacks any of
    :func:`split_files_needed` (a wrong ``--split-root`` would otherwise skip every type and surface later as a
    misleading disagreement between the flags and the cl-drop split, or pass on a gold set with no flagged row)."""
    missing = [p for t in types if ((g.label_type == t) & g.train_row.astype(bool)).any()
               for p in split_files_needed(split_root, t) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"--split-root {split_root}: {len(missing)} round-3 file(s) missing for types with "
                                f"gold train rows, e.g. {missing[:3]}")


def build_frame(gold, pool, split_root, expert_votes="remove", types=LABEL_TYPES):
    """The decided gold items, one row each, with the rule verdicts from their tallies, the round-3 train target,
    the out-of-fold score and what each split variant did to the row. ``expert_votes`` is ``remove`` (the
    headline) or ``keep`` (the cross-check against the numbers computed before this module existed)."""
    if not gold.label_uid.is_unique:
        raise ValueError("gold label_uid is not unique")
    if expert_votes not in ("remove", "keep"):
        raise ValueError(f"expert_votes must be remove or keep, not {expert_votes!r}")
    raw = gold[gold.decided]
    check_split_root(raw, split_root, types)
    tr = raw.train_row.astype(bool)
    if tr.any() and not raw.cl_flagged[tr].astype(bool).any():
        raise ValueError(f"none of the {int(tr.sum())} decided gold train rows is flagged by confident learning, so "
                         "P2 has no flagged group; check the gold table's cl_flagged and --split-root")
    g = remove_expert_votes(raw, pool)
    if expert_votes == "keep":  # the tallies as pulled; n_expert_votes still counts the votes inside them
        for c in ("h_agree", "h_dis", "h_unsure"):
            g[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(int).to_numpy()
    v = tally_verdicts(g.h_agree, g.h_dis)
    for r in TALLY_RULES:
        g[f"rule_{r}"] = v[r].to_numpy()
    g["masked"] = g.expert_source != CONSENSUS
    g["target"] = np.where(g.train_row, g.label_class_r3, "")
    g["wrong"] = g.train_row & (g.target != g.expert)
    g["oof_p_incorrect"] = np.nan
    for c in VARIANTS:
        g[f"touch_{c}"] = False
    for t in types:
        idx = g.index[(g.label_type == t) & g.train_row]
        if not len(idx):  # check_split_root has made sure every type with train rows has its files
            continue
        o = oof_scores(split_root, t)
        m = g.loc[idx, ["label_uid"]].merge(o, on="label_uid", how="left", validate="one_to_one")
        if m.nt_flagged.isna().any():
            raise ValueError(f"{t}: {int(m.nt_flagged.isna().sum())} gold train rows are not in noisy_targets.csv")
        if not (m.nt_flagged.astype(bool).to_numpy() == g.loc[idx, "cl_flagged"].to_numpy()).all():
            raise AssertionError(f"{t}: gold cl_flagged differs from noisy_targets.csv")
        g.loc[idx, "oof_p_incorrect"] = m.oof_p_incorrect.to_numpy()
        for c in VARIANTS:
            touched = touched_rows(split_root, t, c)
            g.loc[idx, f"touch_{c}"] = [bool(touched[u]) for u in g.loc[idx, "label_uid"]]
    flag_drop = g.train_row & (g.cl_flagged != g["touch_cl-drop"])
    if flag_drop.any():
        raise AssertionError(f"{int(flag_drop.sum())} train rows: the cl-drop split disagrees with the flags")
    g["round"] = g["round"].astype(str)
    return g.reset_index(drop=True)


def check_b(g, split_root, types=LABEL_TYPES):
    """Check B: the detector counts on the train rows equal ``confident_learning.expert_agreement`` summed over
    types, for every variant. Raises on a difference; returns the summed records."""
    gold_cl = g[["label_uid", "expert"]].rename(columns={"expert": "verdict"})
    check_split_root(g, split_root, types)
    tot = {}
    for t in types:
        if not ((g.label_type == t) & g.train_row).any():
            continue
        for r in cl.expert_agreement(split_root, t, VARIANTS, gold_cl):
            s = tot.setdefault(r["variant"], {k: 0 for k in r if k != "variant"})
            for k in s:
                s[k] += r[k]
    tr = g[g.train_row]
    for v, s in tot.items():
        mine = {"n_rated_train": len(tr), "target_matches_expert": int((~tr.wrong).sum()),
                "touched": int(tr[f"touch_{v}"].sum()), "touched_overturned": int((tr[f"touch_{v}"] & tr.wrong).sum()),
                "untouched": int((~tr[f"touch_{v}"]).sum()),
                "untouched_overturned": int((~tr[f"touch_{v}"] & tr.wrong).sum())}
        if mine != s:
            raise AssertionError(f"check B failed for {v}: {mine} != expert_agreement {s}")
    return tot


# ---------------------------------------------------------------------------------------------------------------------
# Contrasts and tables (pure)
# ---------------------------------------------------------------------------------------------------------------------

def contrast(ok, in_a, n_boot=N_BOOT, seed=0):
    """Group A against group B for one boolean outcome: counts, Wilson rates, the difference with its bootstrap
    interval (none when either group has fewer than ``MIN_CELL`` items), and the two-sided Fisher p.

    >>> c = contrast([1, 0, 1, 1, 0, 0], [True, True, True, False, False, False], n_boot=0)
    >>> c["a"]["k"], c["b"]["k"], round(c["diff"], 3), c["lo"], c["fisher_p"]
    (2, 1, 0.333, None, 1.0)
    """
    ok, a = np.asarray(ok, bool), np.asarray(in_a, bool)
    ka, na, kb, nb = int((ok & a).sum()), int(a.sum()), int((ok & ~a).sum()), int((~a).sum())
    d, lo, hi = diff_boot(ok, a, n_boot, seed)
    if min(na, nb) < MIN_CELL:  # a resample of 2 items gives a confident-looking interval that means nothing
        lo = hi = None
    return {"a": rate(ka, na), "b": rate(kb, nb), "diff": d, "lo": lo, "hi": hi,
            "fisher_p": fisher_exact(ka, na - ka, kb, nb - kb)}


def by_round(df, fn, seed_name, seed=0):
    """``fn(frame, seed)`` per round, then pooled; keys ``"1"``, ``"2"``, ``"pooled"``."""
    out = {}
    for r in ("1", "2"):
        x = df[df["round"] == r]
        out[r] = fn(x, sl.slice_seed(seed, f"{seed_name}/{r}"))
    out["pooled"] = fn(df, sl.slice_seed(seed, f"{seed_name}/pooled"))
    return out


def p1_rows(g, rule="symmetric3", n_boot=N_BOOT, seed=0):
    """P1 (``rule="symmetric3"``) and its descriptive twin (``unanimous3``): on decided AI Disagrees whose margin
    target is ``incorrect``, the expert-incorrect share among the rows ``rule`` keeps (A) against the rows it drops
    (B), per round and pooled, plus the heterogeneity check (round 1's difference minus round 2's). Each row also
    carries ``gain``: kept minus all of the margin rule's rows (:func:`gain_boot`), what applying ``rule`` changes."""
    x = g[(g.ai_vote == "Disagree") & (g.rule_margin == "incorrect")]
    ok = (x.expert == "incorrect").to_numpy()
    kept = (x[f"rule_{rule}"] == "incorrect").to_numpy()
    out = by_round(x.assign(_ok=ok, _kept=kept),
                   lambda f, s: contrast(f._ok.to_numpy(), f._kept.to_numpy(), n_boot, s), f"P1/{rule}", seed)

    def gain(f, s):
        k, lo, hi = gain_boot(f._ok.to_numpy(), f._kept.to_numpy(), n_boot, s)
        if min(int(f._kept.sum()), int((~f._kept).sum())) < MIN_CELL:
            lo = hi = None
        return {"gain": k, "lo": lo, "hi": hi}
    for k, v in by_round(x.assign(_ok=ok, _kept=kept), gain, f"P1/{rule}/gain", seed).items():
        out[k]["gain"] = v
    out["heterogeneity"] = dict(zip(("did", "lo", "hi"), did_boot(ok, kept, x["round"].to_numpy(), n_boot,
                                                                   sl.slice_seed(seed, f"P1/{rule}/did"))))
    return out


def p2_rows(g, variant="cl-drop", target_class=None, n_boot=N_BOOT, seed=0):
    """P2 (``variant="cl-drop"``, all target classes) and the descriptive rows: on decided round-3 train rows (of
    one target class when given), the wrong-target share among the rows ``variant`` touched (A) against the rest
    (B), per round and pooled, plus the heterogeneity check."""
    x = g[g.train_row]
    if target_class:
        x = x[x.target == target_class]
    ok, t = x.wrong.to_numpy(bool), x[f"touch_{variant}"].to_numpy(bool)
    name = f"P2/{variant}/{target_class or 'all'}"
    out = by_round(x.assign(_ok=ok, _t=t), lambda f, s: contrast(f._ok.to_numpy(), f._t.to_numpy(), n_boot, s),
                   name, seed)
    for k in ("1", "2", "pooled"):
        f = x if k == "pooled" else x[x["round"] == k]
        w = f.wrong.to_numpy(bool)
        tt = f[f"touch_{variant}"].to_numpy(bool)
        out[k].update(recall=rate(int((w & tt).sum()), int(w.sum())),
                      false_alarm=rate(int((~w & tt).sum()), int((~w).sum())),
                      kept_accuracy=rate(int((~w & ~tt).sum()), int((~tt).sum())))
    out["heterogeneity"] = dict(zip(("did", "lo", "hi"), did_boot(ok, t, x["round"].to_numpy(), n_boot,
                                                                   sl.slice_seed(seed, f"{name}/did"))))
    return out


def rule_rows(g, population="ai_disagree"):
    """Descriptive: per round and pooled, per rule, how often the expert agrees with an ``incorrect`` target and
    with a ``correct`` target the rule gives (Wilson), on the decided AI Disagrees or on the decoys. Coverage is
    not reported here (the sampling design fixes it; see :func:`pool_shares`)."""
    x = g[g.ai_vote == "Disagree"] if population == "ai_disagree" else g[g.ai_vote != "Disagree"]
    out = {}
    for r in ("1", "2", "pooled"):
        f = x if r == "pooled" else x[x["round"] == r]
        rows = {"n": int(len(f)), "expert_incorrect": rate(int((f.expert == "incorrect").sum()), len(f))}
        for rule in TALLY_RULES:
            v = f[f"rule_{rule}"]
            rows[rule] = {c: rate(int(((v == c) & (f.expert == c)).sum()), int((v == c).sum())) for c in cl.CLASSES}
            rows[rule]["agree"] = rate(int(((v != "") & (v == f.expert)).sum()), int((v != "").sum()))
        out[r] = rows
    same = bool((x.rule_majority == x.rule_majority2).all())
    return {"rows": out, "majority_equals_majority2": same}


def ranking_rows(g, n_boot=1000, seed=0):
    """Descriptive, within target class: AUC of a suspicion score against "the target is wrong". ``oof`` (train
    rows): the out-of-fold probability of the class opposite to the target. ``vote margin`` (AI Disagrees with a
    margin target, pool tallies): the smaller the winning side's lead, the more suspect. Panorama bootstrap
    interval only when the smaller class has at least ``slices.MIN_CLASS_FOR_CI`` items."""
    out = []
    tr = g[g.train_row & g.oof_p_incorrect.notna()]
    for c in cl.CLASSES:
        x = tr[tr.target == c]
        score = x.oof_p_incorrect if c == "correct" else 1 - x.oof_p_incorrect
        out.append(_auc_row("oof (train rows)", c, x.wrong.to_numpy(bool), score.to_numpy(float), x.pano_id,
                            n_boot, sl.slice_seed(seed, f"rank/oof/{c}")))
    m = g[(g.ai_vote == "Disagree") & (g.rule_margin != "")]
    for c in cl.CLASSES:
        x = m[m.rule_margin == c]
        lead = (x.h_agree - x.h_dis) if c == "correct" else (x.h_dis - x.h_agree)
        out.append(_auc_row("vote margin (AI Disagrees)", c, (x.expert != c).to_numpy(bool),
                            -lead.to_numpy(float), x.pano_id, n_boot, sl.slice_seed(seed, f"rank/margin/{c}")))
    return out


def _auc_row(score, target_class, y, s, groups, n_boot, seed):
    auc = sl._auc(y, s) if len(y) else np.nan
    lo, hi = sl.bootstrap_auc(y, s, groups, n_boot, seed) if len(y) else (np.nan, np.nan)
    f = (lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
    return {"score": score, "target_class": target_class, "n": int(len(y)), "n_wrong": int(np.sum(y)),
            "auc": f(auc), "lo": f(lo), "hi": f(hi)}


def poststratified(x, cell_cols, weights, domain, success, min_cell=MIN_CELL):
    """Population-weighted share of ``success`` in ``domain``: each cell's rate inside the domain, weighted by the
    cell's population times the domain's share of the cell's decided items (a ratio estimate of the domain's size
    in the cell), via ``ai_disagree_relabel.weighted_rate``. A cell whose domain holds fewer than ``min_cell``
    items carries no weight and is named in ``excluded``.

    >>> x = pd.DataFrame({"t": ["A"] * 10 + ["B"] * 10, "dom": [True] * 6 + [False] * 4 + [True] * 10,
    ...                   "ok": [True] * 6 + [False] * 4 + [False] * 10})
    >>> r = poststratified(x, ["t"], {("A",): 100, ("B",): 100}, x.dom, x.ok)
    >>> round(r["rate"], 3), r["excluded"]
    (0.375, [])
    """
    per, w, excluded = {}, {}, []
    for key, c in x.groupby(cell_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        dom = domain.loc[c.index]
        n_dom = int(dom.sum())
        N = weights.get(key, 0)
        if not N or not n_dom:
            continue
        if n_dom < min_cell:
            excluded.append(f"{'/'.join(map(str, key))}(n={n_dom})")
            continue
        per[key] = (int((success.loc[c.index] & dom).sum()), n_dom)
        w[key] = N * n_dom / len(c)
    if not per:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "excluded": excluded}
    r, lo, hi = rl.weighted_rate(per, w)
    return {"rate": float(r), "lo": float(lo), "hi": float(hi), "n": int(sum(n for _, n in per.values())),
            "excluded": excluded}


def round_weights(r1_manifest, r2_recount, r2_design, schema_city):
    """``{"1": {(group, type): N}, "2": {(city_id, stratum, type): N}}``: round 1's population per (group, type)
    from its manifest; round 2's per (city, stratum, type) from the #85 recount (``after``), with the design's
    rater-to-schema map turned into city ids (no rater name leaves this function)."""
    w1 = {(grp, t): int(n) for grp, d in r1_manifest["population"].items() for t, n in d.items()}
    w2 = {}
    for rater, strata in r2_recount["after"]["population"].items():
        city = schema_city[r2_design["raters"][rater]]
        for s, d in strata.items():
            for t, n in d.items():
                w2[(city, s, t)] = int(n)
    return {"1": w1, "2": w2}


MIN_SHARED_CELLS = 2
"""A population-weighted kept-minus-dropped difference needs at least this many cells in which both groups reach
``MIN_CELL``; with one shared cell the "weighted" difference is that one cell's raw difference."""


def _ac_var(k, n):
    """Variance of a cell rate with the Agresti-Coull adjustment used by ``ai_disagree_relabel.weighted_rate``."""
    p = (k + 2) / (n + 4)
    return p * (1 - p) / n


def shared_cell_diff(x, cell_cols, weights, domain, in_a, success, min_cell=MIN_CELL,
                     min_shared=MIN_SHARED_CELLS):
    """Population-weighted A-minus-B difference of ``success`` inside ``domain``, over the cells where both A and B
    hold at least ``min_cell`` items, so both groups are averaged over the same cells. Each shared cell is weighted
    by its population times the domain's share of its decided items (as :func:`poststratified`), renormalised over
    the shared cells. The interval is ``diff +/- 1.96 se`` with ``se`` from the stratified variance of both groups
    (Agresti-Coull cell rates, as ``weighted_rate``), never clipped, so a cell at 0% or 100% does not shrink it.

    Returns counts always: ``cells`` (cells with weight and any domain item), ``cells_a`` / ``cells_b`` (cells where
    that group reaches ``min_cell``), ``cells_shared`` and ``weight_share`` (the shared cells' share of the domain's
    population weight). The rates and the difference are None, and ``estimable`` False, when fewer than
    ``min_shared`` cells are shared.

    >>> x = pd.DataFrame({"t": ["A"] * 10 + ["B"] * 10, "a": ([True] * 5 + [False] * 5) * 2,
    ...                   "ok": [True] * 5 + [True, True, False, False, False] + [True] * 4 + [False] + [False] * 5})
    >>> r = shared_cell_diff(x, ["t"], {("A",): 100, ("B",): 300}, pd.Series(True, index=x.index), x.a, x.ok,
    ...                      min_shared=2)
    >>> r["estimable"], round(r["diff"], 3), r["cells_shared"]
    (True, 0.75, 2)
    """
    per, w = {}, {}
    out = {"cells": 0, "cells_a": 0, "cells_b": 0, "cells_shared": 0}
    tot_w = 0.0
    for key, c in x.groupby(cell_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        dom = domain.loc[c.index]
        n_dom = int(dom.sum())
        N = weights.get(key, 0)
        if not N or not n_dom:
            continue
        out["cells"] += 1
        wk = N * n_dom / len(c)
        tot_w += wk
        a = dom & in_a.loc[c.index]
        b = dom & ~in_a.loc[c.index]
        s = success.loc[c.index]
        na, nb = int(a.sum()), int(b.sum())
        out["cells_a"] += na >= min_cell
        out["cells_b"] += nb >= min_cell
        if na >= min_cell and nb >= min_cell:
            per[key] = (int((s & a).sum()), na, int((s & b).sum()), nb)
            w[key] = wk
    out["cells_shared"] = len(per)
    out["weight_share"] = (sum(w.values()) / tot_w) if tot_w else None
    out["n_a"] = int(sum(v[1] for v in per.values()))
    out["n_b"] = int(sum(v[3] for v in per.values()))
    out["estimable"] = len(per) >= min_shared
    if not out["estimable"]:
        return {**out, "rate_a": None, "rate_b": None, "diff": None, "se": None, "lo": None, "hi": None}
    tw = sum(w.values())
    wn = {k: v / tw for k, v in w.items()}
    ra = sum(wn[k] * per[k][0] / per[k][1] for k in per)
    rb = sum(wn[k] * per[k][2] / per[k][3] for k in per)
    se = math.sqrt(sum(wn[k] ** 2 * (_ac_var(per[k][0], per[k][1]) + _ac_var(per[k][2], per[k][3])) for k in per))
    d = ra - rb
    return {**out, "rate_a": ra, "rate_b": rb, "diff": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


def weighted_p1(g, weights, rule="symmetric3"):
    """Population-weighted P1 per round. ``all``: the expert-incorrect share of the margin-``incorrect`` AI
    Disagrees (:func:`poststratified`). ``shared``: kept by ``rule`` minus dropped, over only the cells where both
    groups reach ``MIN_CELL`` (:func:`shared_cell_diff`); not estimable when fewer than ``MIN_SHARED_CELLS`` cells
    are shared. Round 1 is weighted within its ``voted_disagree`` group per type, round 2 within its
    ``margin_incorrect`` stratum per city and type."""
    out = {}
    for r in ("1", "2"):
        x = g[(g["round"] == r) & (g.ai_vote == "Disagree")]
        if r == "1":
            x = x[x.group_r1 == "voted_disagree"]
            cells, w = ["group_r1", "label_type"], weights["1"]
        else:
            x = x[x.stratum == "margin_incorrect"]
            cells, w = ["city_id", "stratum", "label_type"], weights["2"]
        mi = x.rule_margin == "incorrect"
        ok = x.expert == "incorrect"
        kept = x[f"rule_{rule}"] == "incorrect"
        out[r] = {"all": poststratified(x, cells, w, mi, ok), "shared": shared_cell_diff(x, cells, w, mi, kept, ok)}
    return out


def slice_rows(g, train, min_n=SLICE_MIN_N):
    """Descriptive slices, pooled over rounds. ``ai_precision``: per type x severity and type x tag, the decided AI
    Disagrees, the share the expert calls incorrect (blank under ``min_n``) and the round-3 train rows of the slice.
    ``p1_severity``: P1's kept and dropped groups by severity, pooled over types, each blank under ``min_n``."""
    out = {"min_n": min_n}
    dis = g[g.ai_vote == "Disagree"]
    for by in sl.BY:
        rows = []
        x = sl.with_slice(dis[["label_uid", "label_type", "severity", "tags", "expert"]], by)
        trc = (sl.with_slice(train, by).groupby(["label_type", "slice"]).size()
               if len(train) else pd.Series(dtype=int))
        for (t, s), f in x.groupby(["label_type", "slice"], sort=True):
            k, n = int((f.expert == "incorrect").sum()), len(f)
            r = rate(k, n) if n >= min_n else {"n": n, "k": k, "rate": None, "lo": None, "hi": None}
            rows.append({"label_type": t, "slice": s, **r, "train": int(trc.get((t, s), 0))})
        out[f"ai_precision_{by}"] = sorted(rows, key=lambda d: (LABEL_TYPES.index(d["label_type"]),
                                                               _slice_key(by, d)))
    x = dis[dis.rule_margin == "incorrect"]
    x = sl.with_slice(x.assign(kept=x.rule_symmetric3 == "incorrect")[["severity", "tags", "expert", "kept"]],
                      "severity")
    rows = []
    for s, f in x.groupby("slice"):
        row = {"slice": s}
        for name, m in (("kept", f.kept), ("dropped", ~f.kept)):
            k, n = int((f.expert[m] == "incorrect").sum()), int(m.sum())
            row[name] = rate(k, n) if n >= min_n else {"n": n, "k": k, "rate": None, "lo": None, "hi": None}
        kp, dr = row["kept"], row["dropped"]
        both = kp["rate"] is not None and dr["rate"] is not None
        row["diff"] = (kp["rate"] - dr["rate"]) if both else None
        row["fisher_p"] = (fisher_exact(kp["k"], kp["n"] - kp["k"], dr["k"], dr["n"] - dr["k"]) if both else None)
        rows.append(row)
    out["p1_severity"] = sorted(rows, key=lambda d: _slice_key("severity", d))
    return out


def _slice_key(by, d):
    if by == "severity":
        return (1, 0) if d["slice"] == sl.NONE else (0, int(d["slice"]))
    return (d["slice"] == sl.NONE, -d["n"], d["slice"])


def gold_counts(gold, g_adj, g_masked, g_keep):
    """Counts only: items, decided, the masked-only subset, AI Disagrees per round and type, train rows, flags."""
    dis = g_masked[g_masked.ai_vote == "Disagree"]
    return {"n_items": int(len(gold)), "n_decided": int(len(g_adj)), "n_decided_masked": int(len(g_masked)),
            "n_consensus_decided": int((g_adj.expert_source == CONSENSUS).sum()),
            "n_ai_disagree_decided": int((g_adj.ai_vote == "Disagree").sum()),
            "n_ai_disagree_masked": int(len(dis)),
            "ai_disagree_masked_by_round": {r: int((dis["round"] == r).sum()) for r in ("1", "2")},
            "ai_disagree_masked_by_type": {t: int((dis.label_type == t).sum()) for t in LABEL_TYPES},
            "n_train_rows": int(g_masked.train_row.sum()),
            "train_rows_by_round": {r: int((g_masked.train_row & (g_masked["round"] == r)).sum()) for r in ("1", "2")},
            "n_flagged": int(g_masked.cl_flagged.sum()), "n_wrong_targets": int(g_masked.wrong.sum()),
            "expert_vote_items": {r: int(((g_keep.n_expert_votes > 0) & (g_keep["round"] == r)).sum())
                                  for r in ("1", "2")},
            "expert_vote_ai_disagree_by_stratum": {s: int(v) for s, v in g_keep[
                (g_keep.n_expert_votes > 0) & (g_keep.ai_vote == "Disagree")].stratum.value_counts().items()},
            "train_target_differs_from_pool_margin": int((g_masked.train_row & (g_masked.target != g_masked.rule_margin)
                                                          ).sum())}


def crosscheck(g_keep_adj, pool_rows):
    """The numbers the plan and its review computed by hand, before this module (adjudicated gold, tallies with the
    expert votes left in), so a test can bind the run to them."""
    dis = g_keep_adj[g_keep_adj.ai_vote == "Disagree"]
    tr = g_keep_adj[g_keep_adj.train_row]
    rules = {}
    for r in TALLY_RULES:
        v = dis[f"rule_{r}"]
        inc = v == "incorrect"
        rules[r] = {"cover": int((v != "").sum()), "agree": int(((v != "") & (v == dis.expert)).sum()),
                    "incorrect": int(inc.sum()), "incorrect_right": int((inc & (dis.expert == "incorrect")).sum())}
    allp = next(p for p in pool_rows if p["label_type"] == "all")
    return {"majority_equals_majority2": bool((dis.rule_majority == dis.rule_majority2).all()),
            "decided": int(len(g_keep_adj)), "ai_disagree_decided": int(len(dis)),
            "train_rows": int(len(tr)), "flagged": int(tr.cl_flagged.sum()),
            "flagged_wrong": int((tr.cl_flagged & tr.wrong).sum()),
            "unflagged": int((~tr.cl_flagged).sum()), "unflagged_wrong": int((~tr.cl_flagged & tr.wrong).sum()),
            "rules": rules,
            "pool_n_ai_disagree": allp["n"],
            "pool_cover_share": {r: round(allp["rules"][r]["cover"] / allp["n"], 3) for r in TALLY_RULES}}


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------

def _p(r):
    """``k/n (rate% [lo, hi])`` or ``k/n`` alone when the rate is blank."""
    if r["rate"] is None:
        return f"{r['k']}/{r['n']}"
    return f"{r['k']}/{r['n']} ({100 * r['rate']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}])"


def _pn(r):
    """:func:`_p`, or ``n <size>`` alone when the rate is blank (a thin cell's k/n invites reading a rate)."""
    return f"n {r['n']}" if r["rate"] is None else _p(r)


def _pp(d, lo, hi):
    if d is None:
        return "n/a"
    return f"{100 * d:+.1f} pts" + ("" if lo is None else f" [{100 * lo:+.1f}, {100 * hi:+.1f}]")


def _ps(x):
    return "n/a" if x is None else f"{x:.3g}"


ROUND_NAMES = {"1": "round 1", "2": "round 2", "pooled": "pooled (sample rate; descriptive)"}
POOLED_NAMES = {"primary": "**pooled (primary estimate)**", "descriptive": ROUND_NAMES["pooled"]}
ROLES = {"P1": "primary", "P2": "primary", "P1_unanimous3": "descriptive"}
"""Which pooled row is the pre-stated estimate. The primaries' pooled rows are the contrasts the plan review fixed
(and computed by hand); the per-round rows above them are its check, beside the heterogeneity check."""


def contrast_table(rows, a_name, b_name, extra=False, role="descriptive"):
    head = f"| Rows | {a_name} | {b_name} | {a_name.split(':')[0]} minus {b_name.split(':')[0]} [95%] | Fisher p |"
    lines = [head + (" Recall | False alarms |" if extra else ""),
             "|---|---|---|---|---:|" + ("---|---|" if extra else "")]
    for k in ("1", "2", "pooled"):
        r = rows[k]
        name = POOLED_NAMES[role] if k == "pooled" else ROUND_NAMES[k]
        line = f"| {name} | {_p(r['a'])} | {_p(r['b'])} | {_pp(r['diff'], r['lo'], r['hi'])} | " \
               f"{_ps(r['fisher_p'])} |"
        if extra:
            line += f" {_p(r['recall'])} | {_p(r['false_alarm'])} |"
        lines.append(line)
    h = rows["heterogeneity"]
    het = _pp(h["did"], h["lo"], h["hi"])
    if h["did"] is None:
        small = min(min(rows[k]["a"]["n"], rows[k]["b"]["n"]) for k in ("1", "2"))
        het = f"not computable (a group in one round has {small} rows; it needs {MIN_CELL})"
    lines += ["", f"Heterogeneity check (round 1's difference minus round 2's, resampled within round): {het}."]
    if all("gain" in rows[k] for k in ("1", "2", "pooled")):
        lines += ["", "Kept minus all rows of the margin rule (what applying the stricter rule changes in the share "
                      "of `incorrect` targets the expert confirms; P1 rescaled by the dropped share): " +
                  "; ".join(f"{ROUND_NAMES[k] if k != 'pooled' else 'pooled'} "
                            f"{_pp(rows[k]['gain']['gain'], rows[k]['gain']['lo'], rows[k]['gain']['hi'])}"
                            for k in ("1", "2", "pooled")) + "."]
    return "\n".join(lines)


def render_gold_counts(j):
    c = j["gold_counts"]
    return "\n".join([
        "| Count | Value |", "|---|---:|",
        f"| Gold items (rounds 1 + 2) | {c['n_items']} |",
        f"| Decided (a correct or incorrect verdict, as `adjudicator.py gold` defines it) | {c['n_decided']} |",
        f"| of which the round-2 meeting reached (`r2_consensus`) | {c['n_consensus_decided']} |",
        f"| Decided, masked-only gold (the headline) | {c['n_decided_masked']} |",
        f"| AI Disagrees, masked-only: round 1 / round 2 | {c['ai_disagree_masked_by_round']['1']} / "
        f"{c['ai_disagree_masked_by_round']['2']} ({c['n_ai_disagree_masked']}) |",
        "| AI Disagrees, masked-only, per type | " + ", ".join(f"{t} {n}" for t, n in
                                                          c["ai_disagree_masked_by_type"].items()) + " |",
        f"| Round-3 train rows: round 1 / round 2 | {c['train_rows_by_round']['1']} / {c['train_rows_by_round']['2']} "
        f"({c['n_train_rows']}) |",
        f"| of which flagged by confident learning / wrong target | {c['n_flagged']} / {c['n_wrong_targets']} |",
        f"| Items whose pool tallies hold a rater's earlier vote: round 1 / round 2 | "
        f"{c['expert_vote_items']['1']} / {c['expert_vote_items']['2']} |",
        f"| Train rows whose round-3 target differs from the margin target of the pool tallies | "
        f"{c['train_target_differs_from_pool_margin']} |"])


def render_pool(j):
    rows = j["pool"]
    lines = ["| Type | AI Disagrees | no decisive vote | 1 vote | tie | light | margin incorrect | margin correct |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        n = r["n"]
        lines.append(f"| {r['label_type']} | {n:,} | " + " | ".join(
            f"{100 * r['classes'][c] / n:.1f}%" for c in VOTE_CLASSES) + " |")
    lines += ["", "| Type | " + " | ".join(f"`{r}`" for r in TALLY_RULES) + " |",
              "|---|" + "---:|" * len(TALLY_RULES)]
    for r in rows:
        n = r["n"]
        lines.append(f"| {r['label_type']} | " + " | ".join(
            f"{100 * r['rules'][x]['cover'] / n:.1f}% ({100 * r['rules'][x]['incorrect'] / n:.1f}%)"
            for x in TALLY_RULES) + " |")
    lines += ["", "Cells: share of the type's AI Disagrees the rule gives any target (in brackets: an `incorrect` "
                  "target). The rest is dropped. Exact counts over the whole pool, so no interval."]
    return "\n".join(lines)


def render_primary(j):
    p = j["primary"]
    roles = p["roles"]
    f1, f2 = p["P2"]["1"]["a"]["n"], p["P2"]["2"]["a"]["n"]
    thin = [f"{f} flagged round-{r} row{'s' if f != 1 else ''}" for r, f in (("1", f1), ("2", f2)) if f < MIN_CELL]
    origin = (f"Of the pooled P2's {f1 + f2} flagged rows, {f1} are from round 1 and {f2} from round 2"
              + (f"; with {' and '.join(thin)} the rounds cannot be compared." if thin else "."))
    return "\n".join([
        "**P1. `symmetric` kept against dropped** (decided AI Disagrees whose margin target is `incorrect`; outcome: "
        f"the expert says incorrect; pooled row: {roles['P1']}):", "",
        contrast_table(p["P1"], "kept: net Disagree >= 3", "dropped: net Disagree = 2", role=roles["P1"]), "",
        "**P2. Confident-learning flag** (decided round-3 train rows, both target classes; outcome: the target "
        f"differs from the expert; pooled row: {roles['P2']}):", "",
        contrast_table(p["P2"], "flagged", "unflagged", extra=True, role=roles["P2"]), "",
        origin, "",
        f"Descriptive twin of P1, `unanimous3` (same rows; pooled row: {roles['P1_unanimous3']}):", "",
        contrast_table(p["P1_unanimous3"], "kept: unanimous", "dropped", role=roles["P1_unanimous3"])])


def render_weighted(j):
    lines = ["| Round | margin-`incorrect` AI Disagrees, expert says incorrect (cells without weight) | cells with "
             f">= {MIN_CELL} kept / >= {MIN_CELL} dropped / both, of all | kept minus dropped over the shared cells "
             "[95%, unclipped normal approx.] |", "|---|---|---|---|"]
    for r in ("1", "2"):
        w = j["weighted"][r]
        a, s = w["all"], w["shared"]
        all_ = ("not estimable" if a["rate"] is None else
                f"{100 * a['rate']:.1f}% [{100 * a['lo']:.1f}, {100 * a['hi']:.1f}] (n {a['n']}; "
                f"{len(a['excluded'])} without weight)")
        cells = f"{s['cells_a']} / {s['cells_b']} / {s['cells_shared']}, of {s['cells']}"
        if s["estimable"]:
            diff = (f"kept {100 * s['rate_a']:.1f}% (n {s['n_a']}), dropped {100 * s['rate_b']:.1f}% (n {s['n_b']}): "
                    f"{_pp(s['diff'], s['lo'], s['hi'])}; shared cells hold {100 * s['weight_share']:.0f}% of the "
                    "population weight")
        else:
            diff = (f"not estimable: {s['cells_shared']} shared cell{'s' if s['cells_shared'] != 1 else ''} "
                    f"(needs {MIN_SHARED_CELLS})")
        lines.append(f"| {r} | {all_} | {cells} | {diff} |")
    lines += ["", f"A kept-minus-dropped difference is computed only over cells where both groups hold at least "
                  f"{MIN_CELL} rows, so the two shares average the same cells; with fewer than {MIN_SHARED_CELLS} "
                  "such cells it is not estimable. Its interval uses the stratified variance of both groups, not "
                  "clipped at 0 or 100%."]
    return "\n".join(lines)


def render_rules(j):
    out = []
    for pop, title in (("ai_disagree", "Decided AI Disagrees"), ("decoys", "Decoys (AI Agree or Unsure)")):
        rr = j["rules"][pop]
        same = rr["majority_equals_majority2"]
        rules = [r for r in TALLY_RULES if not (same and r == "majority2")]
        out += [f"**{title}**" + (" (`majority` and `majority2` give the same verdict on every item here, so they "
                                  "are one row)" if same else ""), "",
                "| Rule | Round | `incorrect` target: expert agrees | `correct` target: expert agrees |",
                "|---|---|---|---|"]
        for r in ("1", "2", "pooled"):
            row = rr["rows"][r]
            out.append(f"| reference: expert says incorrect | {ROUND_NAMES[r]} | {_p(row['expert_incorrect'])} | |")
        for rule in rules:
            name = "`majority` = `majority2`" if (same and rule == "majority") else f"`{rule}`"
            for r in ("1", "2", "pooled"):
                row = rr["rows"][r][rule]
                out.append(f"| {name} | {ROUND_NAMES[r]} | {_p(row['incorrect'])} | {_p(row['correct'])} |")
        out.append("")
    return "\n".join(out).rstrip()


def render_detector(j):
    lines = ["| Variant | Target class | Rows | Touched: wrong target | Untouched: wrong target | touched minus "
             "untouched [95%] | Fisher p |", "|---|---|---|---|---|---|---:|"]
    for v in VARIANTS:
        for c in ("incorrect", "correct", "all"):
            r = j["detector"][v][c]["pooled"]
            n = r["a"]["n"] + r["b"]["n"]
            lines.append(f"| `{v}` | {c}{' (descriptive pool)' if c == 'all' else ''} | {n} | {_p(r['a'])} | "
                         f"{_p(r['b'])} | {_pp(r['diff'], r['lo'], r['hi'])} | {_ps(r['fisher_p'])} |")
    lines += ["", "Rounds pooled (round 1 has " + str(j["gold_counts"]["train_rows_by_round"]["1"]) +
              " train rows); per-round rows of every cell are in `scoreboard.json`. Touched = dropped "
              "(`symmetric`, `unanimous3`, `cl-drop`) or flipped (`cl-flip`)."]
    return "\n".join(lines)


def render_ranking(j):
    lines = ["| Score | Target class | Items | Wrong targets | AUC [95%, panorama bootstrap] |", "|---|---|---:|---:|---|"]
    for r in j["ranking"]:
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}" + (
            "" if r["lo"] is None else f" [{r['lo']:.3f}, {r['hi']:.3f}]")
        lines.append(f"| {r['score']} | {r['target_class']} | {r['n']} | {r['n_wrong']} | {auc} |")
    lines += ["", f"No interval when the smaller class has fewer than {sl.MIN_CLASS_FOR_CI} items."]
    return "\n".join(lines)


def render_slices(j):
    s = j["slices"]
    mn = s["min_n"]
    out = []
    for by in sl.BY:
        rows = s[f"ai_precision_{by}"]
        shown = [r for r in rows if r["n"] >= mn]
        out += [f"**Decided AI Disagrees by type x {by}: the expert says incorrect** (rounds pooled; rate blank "
                f"under {mn} items; train = round-3 train rows of the slice; {len(shown)} of {len(rows)} cells reach "
                f"{mn})", ""]
        if not shown:
            out += [f"_No type x {by} cell reaches {mn} items._", ""]
            continue
        out += ["| Type | Slice | Expert says incorrect | Train rows |", "|---|---|---|---:|"]
        out += [f"| {r['label_type']} | {r['slice']} | {_p(r)} | {r['train']:,} |" for r in shown]
        small = [r for r in rows if r["n"] < mn]
        out += ["", f"Below {mn} items: {len(small)} cells holding {sum(r['n'] for r in small)} items, not shown "
                    "(no rate is read from them; their counts are in `scoreboard.json`).", ""]
    rows = s["p1_severity"]
    out += [f"**P1 by severity** (types pooled; a group under {mn} items shows its size only)", "",
            "| Severity | kept by `symmetric` | dropped | kept minus dropped (Fisher p) |", "|---|---|---|---|"]
    out += [f"| {r['slice']} | {_pn(r['kept'])} | {_pn(r['dropped'])} | " +
            ("" if r["diff"] is None else f"{_pp(r['diff'], None, None)} ({_ps(r['fisher_p'])})") + " |"
            for r in rows]
    if not any(r["kept"]["n"] >= mn and r["dropped"]["n"] >= mn for r in rows):
        out += ["", f"_No severity reaches {mn} items in both groups._"]
    return "\n".join(out).rstrip()


def render_sensitivity(j):
    s = j["sensitivity"]
    lines = ["| Contrast | Gold | Tallies | kept / flagged | dropped / unflagged | difference [95%] |",
             "|---|---|---|---|---|---|"]
    for name, r in s.items():
        p = r["pooled"]
        gold, tallies = name.split("|")
        lines.append(f"| {r['label']} | {gold} | {tallies} | {_p(p['a'])} | {_p(p['b'])} | "
                     f"{_pp(p['diff'], p['lo'], p['hi'])} |")
    lines += ["", "Pooled rows. Masked-only gold leaves out the decided items reached in the round-2 meeting; "
                  "`expert votes removed` takes a rater's earlier production vote out of the pool tallies."]
    return "\n".join(lines)


def render_crosscheck(j):
    c = j["crosscheck"]
    r = c["rules"]
    pct = (lambda k, n: f"{100 * k / n:.1f}%" if n else "n/a")
    lines = ["Adjudicated gold, tallies as pulled (the configuration the plan and its review used by hand):", "",
             f"- decided {c['decided']}, decided AI Disagrees {c['ai_disagree_decided']}, decided train rows "
             f"{c['train_rows']}, flagged {c['flagged']};",
             f"- `margin` covers {r['margin']['cover']}, `incorrect` {r['margin']['incorrect']} at "
             f"{pct(r['margin']['incorrect_right'], r['margin']['incorrect'])}; `symmetric3` covers "
             f"{r['symmetric3']['cover']}, `incorrect` {r['symmetric3']['incorrect']} at "
             f"{pct(r['symmetric3']['incorrect_right'], r['symmetric3']['incorrect'])}; `unanimous3` covers "
             f"{r['unanimous3']['cover']}, `incorrect` {r['unanimous3']['incorrect']} at "
             f"{pct(r['unanimous3']['incorrect_right'], r['unanimous3']['incorrect'])}; `majority` covers "
             f"{r['majority']['cover']} at {pct(r['majority']['agree'], r['majority']['cover'])}, and `majority` "
             f"{'gives the same verdict as `majority2` on every one of these items' if c['majority_equals_majority2'] else 'differs from `majority2` on some of these items'};",
             f"- confident learning: {c['flagged_wrong']} of {c['flagged']} flagged rows have a wrong target, "
             f"{c['unflagged_wrong']} of {c['unflagged']} unflagged;",
             f"- pool ({c['pool_n_ai_disagree']:,} AI Disagrees) given any target: " + ", ".join(
                 f"`{k}` {100 * v:.1f}%" for k, v in c["pool_cover_share"].items()) + "."]
    return "\n".join(lines)


def render_provenance(j):
    p = j["provenance"]
    lines = ["| Item | Pin |", "|---|---|", f"| gold.csv | `{p['gold_sha256']}` (matches the pin in "
             f"`docs/adjudicator-2026-11.md`: {'yes' if p['gold_matches_pin'] else 'NO'}) |",
             f"| relabel_pool.csv | `{p['pool_sha256']}` |",
             f"| round-1 manifest (population) | `{p['r1_manifest_sha256']}` |",
             f"| round-2 recount / design | `{p['r2_recount_sha256']}` / `{p['r2_design_sha256']}` |",
             f"| round-3 splits, noisy targets and variant splits (sha256 of the sorted per-file list) | "
             f"`{p['splits_combined_sha256']}` ({p['n_split_files']} files) |",
             f"| verdicts.csv (private) | `{p['verdicts_sha256']}` |",
             f"| code | commit `{p['git_commit']}`; Python {p['python']}, pandas {p['pandas']}, numpy {p['numpy']} |",
             f"| intervals | {p['n_boot']} label-level resamples (AUC: 1,000 panorama resamples), seed {p['seed']} |",
             f"| public-deployment list | live `/v3/api/cities`, read {p['cities_read_at']} ({p['n_public']} public) |"]
    return "\n".join(lines)


RENDER = {"gold_counts": render_gold_counts, "pool": render_pool, "primary": render_primary,
          "weighted": render_weighted, "rules": render_rules, "detector": render_detector,
          "ranking": render_ranking, "slices": render_slices, "sensitivity": render_sensitivity,
          "crosscheck": render_crosscheck, "provenance": render_provenance}


def render_blocks(j):
    return {name: RENDER[name](j) for name in BLOCKS}


def splice(text, blocks):
    """Replace each generated block; everything else stays byte-identical. A missing block raises.

    >>> t = "a\\n" + BEGIN.format(name="x") + "\\nold\\n" + END.format(name="x") + "\\nb"
    >>> splice(t, {"x": "new"}).splitlines()[3]
    'new'
    """
    for name, body in blocks.items():
        pat = re.compile(re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: BEGIN.format(name=name) + "\n\n" + body + "\n\n" + END.format(name=name), text)
    return text


# ---------------------------------------------------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------------------------------------------------

FORBIDDEN_KEYS = ("label_uid", "raters", "label_id", "user_id", "city_id")


def privacy_gate(text, lists, all_cities):
    """Raise ``ValueError`` when ``text`` names a non-public deployment: ``adjudicator.privacy_check`` (city ids and
    schemas from ``cityparams.conf`` that are not public) and the case- and accent-folded substring test of
    ``publish_to_hf`` over the API's non-public terms (``lists`` is ``publish_to_hf.city_lists()``'s shape), and
    when a forbidden per-label key appears.

    >>> lists = {"public": {"city-a"}, "non_public_terms": {"Hidden Town"}}
    >>> privacy_gate("city-a only", lists, {"city-a": "sidewalk_a"})
    >>> privacy_gate("in hidden town", lists, {})
    Traceback (most recent call last):
    ...
    ValueError: non-public term in the generated text: ['Hidden Town']
    """
    import publish_to_hf as ph
    adj.privacy_check(text, all_cities, lists["public"])
    folded = ph.fold_text(text)
    hits = sorted(t for t in lists["non_public_terms"]
                  if t and (len(t) >= ph.MIN_TERM_LEN or t == t.lower().replace(" ", "")) and ph.fold_text(t) in folded)
    if hits:
        raise ValueError(f"non-public term in the generated text: {hits}")
    bad = [k for k in FORBIDDEN_KEYS if f'"{k}"' in text]
    if bad:
        raise ValueError(f"per-label key in the generated text: {bad}")


def city_lists():
    """The live ``/v3/api/cities`` lists (``publish_to_hf.city_lists``); there is no saved-list option, because
    visibility changes within days. Tests monkeypatch this function."""
    import publish_to_hf as ph
    return ph.city_lists()


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def _git_commit():
    r = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _split_files(split_root):
    files = []
    for t in LABEL_TYPES:
        for rel in ["split.csv", "cl/noisy_targets.csv"] + [f"cl/{v}/split.csv" for v in VARIANTS]:
            p = os.path.join(split_root, t, rel)
            if os.path.exists(p):
                files.append((f"{t}/{rel}", rl.sha256_file(p)))
    return files


def _home(path):
    return os.path.abspath(path).replace(os.path.expanduser("~"), "~", 1)


def score(gold, pool, split_root, weights, n_boot=N_BOOT, seed=0):
    """Everything ``scoreboard.json`` holds except provenance, and the per-label frame (private)."""
    g_adj = build_frame(gold, pool, split_root, "remove")
    g = g_adj[g_adj.masked].reset_index(drop=True)
    g_keep = build_frame(gold, pool, split_root, "keep")
    g_keep_m = g_keep[g_keep.masked].reset_index(drop=True)
    check = check_b(g, split_root)
    train = adj.train_rows_of(split_root)
    pr = pool_shares(pool)
    det = {v: {c or "all": p2_rows(g, v, c, n_boot, seed) for c in ("incorrect", "correct", None)} for v in VARIANTS}
    j = {"gold_counts": gold_counts(gold, g_adj, g, g_keep),
         "pool": pr,
         "primary": {"P1": p1_rows(g, "symmetric3", n_boot, seed), "P2": p2_rows(g, "cl-drop", None, n_boot, seed),
                     "P1_unanimous3": p1_rows(g, "unanimous3", n_boot, seed), "roles": dict(ROLES)},
         "weighted": weighted_p1(g, weights),
         "rules": {"ai_disagree": rule_rows(g, "ai_disagree"), "decoys": rule_rows(g, "decoys")},
         "detector": det,
         "ranking": ranking_rows(g, seed=seed),
         "slices": slice_rows(g, train),
         "sensitivity": {
             "masked-only|expert votes removed": {"label": "P1", **p1_rows(g, "symmetric3", n_boot, seed)},
             "with meeting verdicts|expert votes removed": {"label": "P1",
                                                            **p1_rows(g_adj, "symmetric3", n_boot, seed)},
             "masked-only|tallies as pulled": {"label": "P1", **p1_rows(g_keep_m, "symmetric3", n_boot, seed)},
             "with meeting verdicts|tallies as pulled": {"label": "P1",
                                                         **p1_rows(g_keep, "symmetric3", n_boot, seed)},
             "masked-only|n/a (train targets)": {"label": "P2", **p2_rows(g, "cl-drop", None, n_boot, seed)},
             "with meeting verdicts|n/a (train targets)": {"label": "P2",
                                                           **p2_rows(g_adj, "cl-drop", None, n_boot, seed)}},
         "crosscheck": crosscheck(g_keep, pr),
         "check_b": check,
         "n_boot": n_boot, "seed": seed}
    return j, g_adj


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def dumps(j):
    return json.dumps(j, indent=1, sort_keys=True, default=_json_default) + "\n"


def cmd_score(args):
    adj.refuse_in_checkout(args.out_dir, "the per-label verdicts")
    missing = [f"--{k.replace('_', '-')} {p}" for k, p in (("gold", args.gold), ("pool", args.pool),
                                                         ("r1_manifest", args.r1_manifest),
                                                         ("r2_recount", args.r2_recount),
                                                         ("r2_design", args.r2_design), ("cityparams", args.cityparams))
               if not os.path.isfile(p)] + ([] if os.path.isdir(args.split_root) else [f"--split-root {args.split_root}"])
    if missing:
        sys.exit("input not found: " + "; ".join(missing))
    gold = adj.load_gold(args.gold)
    pool = load_pool(args.pool)
    schema_city = adj.schema_city_map(open(args.cityparams).read())
    r1m, r2r, r2d = (json.load(open(p)) for p in (args.r1_manifest, args.r2_recount, args.r2_design))
    weights = round_weights(r1m, r2r, r2d, schema_city)
    lists = city_lists()
    j, frame = score(gold, pool, args.split_root, weights, args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    vpath = os.path.join(args.out_dir, "verdicts.csv")
    frame[VERDICT_COLUMNS].sort_values("label_uid").to_csv(vpath, index=False, lineterminator="\n")
    vsha = rl.sha256_file(vpath)
    with open(os.path.join(args.out_dir, "verdicts.sha256"), "w") as f:
        f.write(f"{vsha}  verdicts.csv\n")
    files = _split_files(args.split_root)
    combined = hashlib.sha256("".join(f"{s}  {n}\n" for n, s in sorted(files)).encode()).hexdigest()
    gold_sha = rl.sha256_file(args.gold)
    j["provenance"] = {"gold_sha256": gold_sha, "gold_matches_pin": gold_sha == GOLD_SHA256,
                       "pool_sha256": rl.sha256_file(args.pool), "r1_manifest_sha256": rl.sha256_file(args.r1_manifest),
                       "r2_recount_sha256": rl.sha256_file(args.r2_recount),
                       "r2_design_sha256": rl.sha256_file(args.r2_design),
                       "splits_combined_sha256": combined, "n_split_files": len(files), "verdicts_sha256": vsha,
                       "git_commit": _git_commit(), "python": platform.python_version(), "pandas": pd.__version__,
                       "numpy": np.__version__, "n_boot": args.n_boot, "seed": args.seed,
                       "cities_read_at": lists["read_at"], "n_public": len(lists["public"])}
    text = dumps(j)
    all_cities = {c: s for s, c in schema_city.items()}
    try:
        privacy_gate(text + "\n".join(render_blocks(j).values()), lists, all_cities)
    except ValueError as e:
        sys.exit(f"refusing to write scoreboard.json: {e}")
    run = {"argv": sys.argv, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "inputs": {"gold": [args.gold, gold_sha], "pool": [args.pool, j["provenance"]["pool_sha256"]],
                      "r1_manifest": args.r1_manifest, "r2_recount": args.r2_recount, "r2_design": args.r2_design,
                      "cityparams": [args.cityparams, rl.sha256_file(args.cityparams)],
                      "split_root": args.split_root, "split_files": dict(files)},
           "public_cities": sorted(lists["public"]), "cities_source": lists["source"],
           "cities_read_at": lists["read_at"], "provenance": j["provenance"]}
    adj.write_json(os.path.join(args.out_dir, "run.json"), run)
    os.makedirs(args.out, exist_ok=True)
    jpath = os.path.join(args.out, "scoreboard.json")
    with open(jpath, "w") as f:
        f.write(text)
    import publish_to_hf as ph
    try:
        ph.check_no_private_names(args.out, lists["non_public_terms"])
    except ph.PublishError as e:
        os.remove(jpath)
        sys.exit(f"refusing: {e}")
    with open(os.path.join(args.out_dir, "scoreboard.json"), "w") as f:
        f.write(text)
    print(f"verdicts -> {vpath} (sha256 {vsha}); scoreboard -> {jpath}")
    print(render_blocks(j)["primary"])


def cmd_report(args):
    j = json.load(open(args.json))
    blocks = render_blocks(j)
    doc = open(args.doc).read()
    new = splice(doc, blocks)
    if args.check:
        if new != doc:
            sys.exit(f"{args.doc} differs from what {args.json} renders; run report without --check")
        print("doc matches scoreboard.json")
        return
    schema_city = adj.schema_city_map(open(args.cityparams).read())
    try:
        privacy_gate(new, city_lists(), {c: s for s, c in schema_city.items()})
    except ValueError as e:
        sys.exit(f"refusing to write the doc: {e}")
    with open(args.doc, "w") as f:
        f.write(new)
    print(f"spliced {len(blocks)} blocks into {args.doc}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="per-label verdicts (private) and scoreboard.json (aggregates)")
    s.add_argument("--gold", required=True, help="adjudicator.py gold's gold.csv")
    s.add_argument("--pool", required=True, help="round 1's relabel_pool.csv (tallies and expert vote values)")
    s.add_argument("--split-root", required=True, help="dir with <Type>/split.csv and <Type>/cl/ (round 3, #76)")
    s.add_argument("--r1-manifest", required=True, help="round 1's manifest.before-rating.json (population)")
    s.add_argument("--r2-recount", default=R2_RECOUNT)
    s.add_argument("--r2-design", default=R2_DESIGN)
    s.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    s.add_argument("--out-dir", default=DEFAULT_OUT, help="private outputs (refused inside a git checkout)")
    s.add_argument("--out", default=PUBLIC_DIR, help="where scoreboard.json goes")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_score)
    r = sub.add_parser("report", help="splice the generated blocks into the doc")
    r.add_argument("--json", default=os.path.join(PUBLIC_DIR, "scoreboard.json"))
    r.add_argument("--doc", default=DOC)
    r.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    r.add_argument("--check", action="store_true", help="exit 1 if the doc differs (no network, writes nothing)")
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
