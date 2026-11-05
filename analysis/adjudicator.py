"""A VLM adjudicator of the AI's Disagree votes, scored against the expert gold set (issue #77), and the
reliability-weighted crowd target scored the same way (issue #12's scoring step). Part of the Disagree-targets
programme (#74).

Why
---
The crowd margin rule that sets a training target matches an expert only about three times in four on contested
labels (74.5% [64.8, 82.2] on round 1's voted Disagrees, PR #66). Before any automatic target cleaning touches the
training set, it has to beat that crowd on the same items. This module asks an open-weight VLM, served by vLLM,
whether each gold-set label is correct, with the label type's definition and an explicit rating rule, and measures
how often it agrees with the expert gold label, next to how often the crowd majority does. Only a (model, type)
pair that clears a paired bar may relabel or soft-weight contested training rows.

Stages (subcommands)
--------------------
``gold``
    One table of every expert-rated item of rounds 1 and 2 (``label_uid`` in the city form ``city_id:label_id``),
    with its expert verdict, stratum, AI vote, human tallies and crowd majority, and its round-3 training state
    (split, target, confident-learning flag). Writes ``gold.csv`` (private), ``gold.sha256`` and
    ``gold_summary.json`` (counts only).
``crops``
    The label list for the klone crop leg (``ai_labels_eval.py crops``, unchanged) and its pano staging list, for
    the gold items that have no round-3 crop. ``stage-list`` sizes the same for the whole AI-Disagree pool.
``prompt``
    Prints the rendered adjudicator prompts and the template hash.
``score``
    Every gold item through a running vLLM server: the verdict from the first token's top-20 log-probs (as
    ``vlm_vllm.py``), then a one-sentence reason in a second request. Resumable.
``agree``
    Agreement with the expert per model and type, the crowd on the same items, the paired difference with a
    bootstrap interval and McNemar's exact test, strata, severity and tag slices with training counts, and the bar.
``disagreements``
    A masked reading packet of up to 30 items where a model and the expert disagree.
``relabel``
    A model that cleared the bar, on the contested train rows (or the whole pool) of the types it cleared.
``variants``
    ``adj-replace`` and ``adj-soft`` split variants for ``confident_learning.slurm``.
``reliability``
    The reliability-weighted crowd (``validator_reliability.py``'s vote weights) against the expert, in the same
    table as the models.
``report``
    The doc's tables and provenance, spliced between ``<!-- BEGIN GENERATED: name ... -->`` markers.

Every per-label output (``gold.csv``, scores, packets, relabels) refuses a path inside a git checkout; only the
aggregates ``report`` writes go into ``docs/adjudicator-2026-11/``. ``label_uid`` is per city, and every merge
passes ``validate=``.

Usage (the full command list is in ``docs/adjudicator-2026-11.md``, Reproducing)::

    R=~/Data/sidewalk-validator-ai-relabel-2026-10; OUT=~/Data/sidewalk-validator-ai-adjudicator-2026-11
    DATA=~/Data/sidewalk-validator-ai-cl-2026-10/retrain-2026-10
    python analysis/adjudicator.py gold --r1-joined $R/packet/joined_scored.csv \\
        --r2-key $R/round2/private-key-do-not-share/key.csv \\
        --r2-sheets jon=$R/round2/votes_jon.csv mikey=$R/round2/votes_mikey.csv \\
        --r2-gold-shared $R/round2/adjudication/gold_shared.csv --r2-unmasked $R/round2/unmasked.csv \\
        --split-root $DATA --pool $R/relabel_pool.csv --out-dir $OUT
    python analysis/adjudicator.py prompt

Example (the soft-target blend of ``adj-soft``: the VLM counts as ``w`` extra votes)::

    >>> soft_blend(0, 2, 1.0, 2.0)
    0.5
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis import confident_learning as cl  # noqa: E402
from analysis import gt_rounds as gr  # noqa: E402
from analysis import slices as sl  # noqa: E402
from analysis import vlm_baseline as vb  # noqa: E402
from analysis import vlm_vllm as vv  # noqa: E402
from fine_tune_classifier import smoothed_rate  # noqa: E402
import validator_reliability as vr  # noqa: E402

LABEL_TYPES = vb.LABEL_TYPES
DECIDED = ("correct", "incorrect")
EXPERT_VALUES = {"correct", "incorrect", "cant_tell", ""}
DEFAULT_OUT = os.path.expanduser("~/Data/sidewalk-validator-ai-adjudicator-2026-11")
DOC = os.path.join(REPO_DIR, "docs", "adjudicator-2026-11.md")
GOLD_COLUMNS = ["label_uid", "label_uid_schema", "schema", "city_id", "label_type", "severity", "tags", "pano_id",
                "round", "raters", "expert", "expert_source", "unmasked", "decided", "stratum", "group_r1",
                "margin_class", "ai_vote", "ai_accuracy", "h_agree", "h_dis", "h_unsure", "crowd_class",
                "round3_split", "train_row", "label_class_r3", "class_source_r3", "cl_flagged", "crop_path_r3",
                "crop_status_r3"]
ADJ_BACKEND = "adjudicator-vllm"
ADJ_COLUMNS = vv.SCORE_COLUMNS + ["verdict", "reason", "reason_tokens", "prompt_sha256", "prompt_variant"]
R1_REFERENCE = {"label": "crowd majority, round 1 voted Disagrees (PR #66)", "n": 94, "rate": 0.745, "lo": 0.648,
                "hi": 0.822}
"""Printed as a fixed reference row; the bar uses the crowd on the same items instead."""
BAR_MIN_N = 30
SLICE_MIN_N = 10
N_BOOT = 2000
REASON_QUESTION = "In one sentence, why?"
REASON_MAX_TOKENS = 60
PACKET_PER_TYPE = 6
BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/adjudicator.py) -->"
END = "<!-- END GENERATED: {name} -->"
BLOCKS = ("gold_counts", "agreement", "paired", "strata", "slices", "reliability", "bar", "variants", "provenance")
VARIANTS = ("adj-replace", "adj-soft")
GB_PER_PANO = 0.01


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def refuse_in_checkout(path, what):
    """Exit when ``path`` is inside a git checkout: per-label files never go into a repository."""
    if rl.inside_git_checkout(path):
        sys.exit(f"refusing to write {what} inside a git checkout: {path}")


def _truthy(col):
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "t"])


def _num(col):
    return pd.to_numeric(col, errors="coerce")


def crowd_class(h_agree, h_dis):
    """``correct`` / ``incorrect`` / ``""`` from the human Agree and Disagree tallies (the majority; a tie or no
    vote is blank).

    >>> list(crowd_class(pd.Series([2, 0, 1, 0]), pd.Series([1, 3, 1, 0])))
    ['correct', 'incorrect', '', '']
    """
    a, d = _num(h_agree).fillna(0), _num(h_dis).fillna(0)
    return pd.Series(np.where(a > d, "correct", np.where(d > a, "incorrect", "")), index=a.index)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------------
# gold: the unified gold table (pure parts unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def schema_city_map(cityparams_text):
    """``{schema: city_id}`` from SidewalkWebpage's ``cityparams.conf`` (``ai_disagree_relabel.cityparams``)."""
    return rl.cityparams(cityparams_text)[0]


def to_city_form(uid_schema, schema_city):
    """``schema:label_id`` -> ``city_id:label_id``; ``KeyError`` for a schema the conf does not list.

    >>> to_city_form("sidewalk_a:12", {"sidewalk_a": "city-a"})
    'city-a:12'
    >>> to_city_form("sidewalk_z:1", {"sidewalk_a": "city-a"})
    Traceback (most recent call last):
    ...
    KeyError: "schema 'sidewalk_z' is not in cityparams.conf"
    """
    schema, lid = str(uid_schema).rsplit(":", 1)
    if schema not in schema_city:
        raise KeyError(f"schema {schema!r} is not in cityparams.conf")
    return f"{schema_city[schema]}:{lid}"


def check_key_forms(key, schema_city=None):
    """Raise ``ValueError("key form mismatch: ...")`` unless every round-2 key row has ``label_uid_city ==
    city_id + ":" + label_id`` (and, with ``schema_city``, ``city_id`` is its schema's city in the conf)."""
    want = key.city_id.astype(str) + ":" + key.label_id.astype(str)
    bad = key.label_uid_city.astype(str) != want
    if schema_city is not None:
        bad |= key.schema.map(schema_city).astype(str) != key.city_id.astype(str)
    if bad.any():
        raise ValueError(f"key form mismatch: {int(bad.sum())} rows whose label_uid_city is not city_id:label_id")


def _base_rows(df):
    out = pd.DataFrame(index=df.index)
    for c in ("label_type", "severity", "tags", "pano_id", "ai_vote", "ai_accuracy", "h_agree", "h_dis", "h_unsure"):
        out[c] = df[c] if c in df.columns else ""
    return out


def round1_rows(joined, schema_city, rater="jon"):
    """Round 1's rated items (``joined_scored.csv``, schema-form ``label_uid``) as gold rows. The stratum is
    recomputed with ``gt_rounds.assign_stratum`` from the item's tallies, so it is comparable with round 2 (round 1's
    ``voted_disagree`` mixes ``light_voted`` and the margin strata); ``group_r1`` keeps round 1's group."""
    j = joined.copy()
    for c in ("h_agree", "h_dis", "h_unsure"):
        j[c] = _num(j[c]).fillna(0).astype(int)
    out = _base_rows(j)
    out["label_uid_schema"] = j.label_uid.astype(str)
    out["label_uid"] = [to_city_form(u, schema_city) for u in j.label_uid]
    out["schema"] = j.schema
    out["city_id"] = j.schema.map(schema_city)
    out["round"] = 1
    out["raters"] = rater
    out["expert"] = j.verdict.fillna("").astype(str)
    out["expert_source"] = "r1"
    out["unmasked"] = False
    out["stratum"] = gr.assign_stratum(j)
    out["group_r1"] = j.group
    out["margin_class"] = gr.margin_class(j)
    return out.reset_index(drop=True)


def _source_kind(source):
    """``r2_matched`` for a gold label the raters reached independently, ``r2_consensus`` for one from the meeting.

    >>> _source_kind("matched"), _source_kind("both_cant_tell"), _source_kind("consensus_cant_tell")
    ('r2_matched', 'r2_matched', 'r2_consensus')
    """
    return "r2_matched" if source in ("matched", "both_cant_tell") else "r2_consensus"


def round2_rows(key, sheets, gold_shared, unmasked=()):
    """Round 2's items as gold rows: a non-shared item takes its owning rater's verdict (``r2_single``); a shared
    item takes the adjudicated ``gold_shared.gold`` (blank gold is ``cant_tell``). ``unmasked`` packet ids are
    flagged. Raises when a rater has no sheet, a sheet repeats a packet id, or a verdict is not a known value."""
    k = key.copy()
    shared = _truthy(k.shared)
    for c in ("h_agree", "h_dis", "h_unsure"):
        k[c] = _num(k[c]).fillna(0).astype(int)
    k["expert"], k["expert_source"], k["raters"] = "", "", ""
    missing = sorted(set(k.loc[~shared, "rater"]) - set(sheets))
    if missing:
        raise ValueError(f"no votes sheet for rater(s) {missing}")
    for rater, s in sheets.items():
        s = s[["packet_id", "verdict"]].copy()
        s["verdict"] = s.verdict.fillna("").astype(str)
        if not s.packet_id.is_unique:
            raise ValueError(f"{rater}'s sheet repeats packet_ids")
        bad = set(s.verdict) - EXPERT_VALUES
        if bad:
            raise ValueError(f"unexpected verdicts in {rater}'s sheet: {sorted(bad)}")
        own = k.loc[~shared & (k.rater == rater), ["packet_id"]]
        m = own.merge(s, on="packet_id", how="left", validate="one_to_one")
        k.loc[own.index, "expert"] = m.verdict.fillna("").to_numpy()
        k.loc[own.index, "expert_source"] = "r2_single"
        k.loc[own.index, "raters"] = rater
    if shared.any():
        g = gold_shared[["packet_id", "gold", "source"]].copy()
        g["gold"] = g.gold.fillna("").astype(str)
        sh = k.loc[shared, ["packet_id"]].merge(g, on="packet_id", how="left", validate="one_to_one")
        if sh.source.isna().any():
            raise ValueError(f"{int(sh.source.isna().sum())} shared items have no row in gold_shared.csv")
        k.loc[shared, "expert"] = sh.gold.replace("", "cant_tell").to_numpy()
        k.loc[shared, "expert_source"] = [_source_kind(s) for s in sh.source]
        k.loc[shared, "raters"] = "+".join(sorted(sheets))
    out = _base_rows(k)
    out["label_uid"] = k.label_uid_city.astype(str)
    out["label_uid_schema"] = k.label_uid.astype(str)
    out["schema"], out["city_id"] = k.schema, k.city_id
    out["round"] = 2
    out["raters"], out["expert"], out["expert_source"] = k.raters, k.expert, k.expert_source
    out["unmasked"] = k.packet_id.isin(set(unmasked))
    out["stratum"] = k.stratum
    out["group_r1"] = ""
    out["margin_class"] = k.margin_class.fillna("") if "margin_class" in k else gr.margin_class(k)
    return out.reset_index(drop=True)


def combine_rounds(*parts):
    """The rounds stacked, with ``decided`` (a correct / incorrect verdict reached masked) and ``crowd_class``.
    Raises ``ValueError`` when an item is in two rounds."""
    g = pd.concat(parts, ignore_index=True)
    dup = g.label_uid[g.label_uid.duplicated()]
    if len(dup):
        raise ValueError(f"{dup.nunique()} items are in more than one round; a gold item needs one verdict")
    g["unmasked"] = g.unmasked.astype(bool)
    g["decided"] = g.expert.isin(DECIDED) & ~g.unmasked
    g["crowd_class"] = crowd_class(g.h_agree, g.h_dis)
    return g


def attach_pool(gold, pool):
    """``(gold, n_found)``: ``ai_vote``, ``ai_accuracy`` and the human tallies refreshed from the pool pull
    (``ai_disagree_relabel.load_pool``; schema-form ``label_uid``), joined one-to-one, and ``crowd_class``
    recomputed. Items missing from the pool keep their own values. The stratum stays the one the item was sampled
    in."""
    cols = ["ai_vote", "ai_accuracy", "h_agree", "h_dis", "h_unsure"]
    p = pool[["label_uid"] + cols].rename(columns={"label_uid": "label_uid_schema"})
    m = gold[["label_uid_schema"]].merge(p, on="label_uid_schema", how="left", validate="one_to_one",
                                        indicator=True)
    found = (m.pop("_merge") == "both").to_numpy()
    out = gold.copy()
    for c in cols:
        out.loc[found, c] = m.loc[found, c].to_numpy()
    out["crowd_class"] = crowd_class(out.h_agree, out.h_dis)
    return out, int(found.sum())


def attach_training(gold, split_root, cl_dir=None, types=LABEL_TYPES):
    """Each item's round-3 training state, from ``<split_root>/<Type>/split.csv`` and, when present,
    ``<cl_dir>/<Type>/cl/noisy_targets.csv``: ``round3_split``, ``train_row``, ``label_class_r3``,
    ``class_source_r3``, ``crop_path_r3``, ``crop_status_r3`` and ``cl_flagged``. Returns ``(gold, split_shas)``."""
    cl_dir = cl_dir or split_root
    out = gold.copy()
    for c in ("round3_split", "label_class_r3", "class_source_r3", "crop_path_r3", "crop_status_r3"):
        out[c] = ""
    out["cl_flagged"] = False
    shas = {}
    for t in types:
        path = os.path.join(split_root, t, "split.csv")
        if not os.path.exists(path):
            continue
        shas[t] = rl.sha256_file(path)
        s = cl.read_split(path)
        s = s.assign(class_source=s.get("class_source", ""), status=s.get("status", ""))
        s = s[["label_uid", "split", "label_class", "class_source", "crop_path", "status"]].rename(columns={
            "split": "round3_split", "label_class": "label_class_r3", "class_source": "class_source_r3",
            "crop_path": "crop_path_r3", "status": "crop_status_r3"})
        idx = out.index[out.label_type == t]
        m = out.loc[idx, ["label_uid"]].merge(s, on="label_uid", how="left", validate="one_to_one").fillna("")
        for c in s.columns[1:]:
            out.loc[idx, c] = m[c].to_numpy()
        noisy = os.path.join(cl_dir, t, "cl", "noisy_targets.csv")
        if os.path.exists(noisy):
            n = pd.read_csv(noisy, dtype=str, keep_default_na=False)
            flagged = set(n.label_uid[n.flagged.isin(["1", "True", "true"])])
            out.loc[idx, "cl_flagged"] = out.loc[idx, "label_uid"].isin(flagged).to_numpy()
            shas[f"{t}/noisy_targets"] = rl.sha256_file(noisy)
    out["train_row"] = out.round3_split == "train"
    return out, shas


def gold_summary(gold):
    """Counts only (no ids, no city names): per round x stratum x type, total and decided; unmasked, train rows."""
    g = gold.copy()
    cells = g.groupby(["round", "stratum", "label_type"]).agg(n=("label_uid", "size"), decided=("decided", "sum"))
    return {"n_items": int(len(g)), "n_decided": int(g.decided.sum()), "n_unmasked": int(g.unmasked.sum()),
            "n_cant_tell": int((g.expert == "cant_tell").sum()), "n_unanswered": int((g.expert == "").sum()),
            "by_round": {str(k): int(v) for k, v in g.groupby("round").size().items()},
            "train_rows": {str(k): int(v) for k, v in g[g.train_row].groupby("round").size().items()},
            "train_rows_decided": int((g.train_row & g.decided).sum()),
            "cl_flagged": int(g.cl_flagged.sum()),
            "expert_source": {k: int(v) for k, v in g.expert_source.value_counts().items()},
            "cells": [{"round": int(r), "stratum": s or "(none)", "label_type": t, "n": int(x.n),
                       "decided": int(x.decided)} for (r, s, t), x in cells.iterrows()]}


def gold_markdown(summary):
    """The gold-set table: round x stratum rows, ``decided / items`` per type."""
    cells = pd.DataFrame(summary["cells"])
    lines = ["| Round | Stratum | " + " | ".join(LABEL_TYPES) + " | all |", "|---|---|" + "---:|" * 6]
    for (r, s), g in cells.groupby(["round", "stratum"], sort=True):
        row = [f"{int(g[g.label_type == t].decided.sum())} / {int(g[g.label_type == t].n.sum())}" for t in LABEL_TYPES]
        lines.append(f"| {r} | {s} | " + " | ".join(row) + f" | {int(g.decided.sum())} / {int(g.n.sum())} |")
    lines.append(f"| all | all | " + " | ".join(
        f"{int(cells[cells.label_type == t].decided.sum())} / {int(cells[cells.label_type == t].n.sum())}"
        for t in LABEL_TYPES) + f" | {summary['n_decided']} / {summary['n_items']} |")
    lines += ["", f"Cells are decided / items. Decided = a correct or incorrect verdict reached masked; "
                  f"{summary['n_cant_tell']} can't tell, {summary['n_unanswered']} unanswered and "
                  f"{summary['n_unmasked']} unmasked items are not decided. {summary['train_rows_decided']} decided "
                  f"items are round-3 train rows; {summary['cl_flagged']} items were flagged by confident learning."]
    return "\n".join(lines)


def load_gold(path):
    g = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("unmasked", "decided", "train_row", "cl_flagged"):
        if c in g:  # stage-list's pool_items.csv (an input to `crops`) has none of them
            g[c] = _truthy(g[c])
    if not g.label_uid.is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    return g


def _sheets(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            sys.exit(f"expected RATER=PATH, got {p!r}")
        r, path = p.split("=", 1)
        out[r] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return out


def cmd_gold(args):
    refuse_in_checkout(args.out_dir, "the gold table")
    schema_city = schema_city_map(open(args.cityparams).read())
    joined = pd.read_csv(args.r1_joined, dtype={"tags": str})
    key = pd.read_csv(args.r2_key, dtype=str, keep_default_na=False)
    try:
        check_key_forms(key, schema_city)
        r1 = round1_rows(joined, schema_city, args.r1_rater)
    except (KeyError, ValueError) as e:
        sys.exit(str(e.args[0] if e.args else e))
    unmasked = set(pd.read_csv(args.r2_unmasked, dtype=str).packet_id) if args.r2_unmasked else set()
    r2 = round2_rows(key, _sheets(args.r2_sheets), pd.read_csv(args.r2_gold_shared, dtype=str,
                                                                keep_default_na=False), unmasked)
    gold = combine_rounds(r1, r2)
    inputs = {"r1_joined": rl.sha256_file(args.r1_joined), "r2_key": rl.sha256_file(args.r2_key),
              "r2_gold_shared": rl.sha256_file(args.r2_gold_shared), "cityparams": rl.sha256_file(args.cityparams)}
    for p in args.r2_sheets:
        inputs["r2_sheet_" + p.split("=", 1)[0]] = rl.sha256_file(p.split("=", 1)[1])
    if args.r2_unmasked:
        inputs["r2_unmasked"] = rl.sha256_file(args.r2_unmasked)
    n_found = None
    if args.pool:
        gold, n_found = attach_pool(gold, rl.load_pool(args.pool))
        inputs["pool"] = rl.sha256_file(args.pool)
    gold, split_shas = attach_training(gold, args.split_root, args.cl_dir)
    gold = gold.sort_values("label_uid", kind="mergesort")[GOLD_COLUMNS]
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "gold.csv")
    gold.to_csv(path, index=False, lineterminator="\n")
    sha = rl.sha256_file(path)
    with open(os.path.join(args.out_dir, "gold.sha256"), "w") as f:
        f.write(f"{sha}  gold.csv\n")
    # the decided items as `confident_learning.py report --gold` reads them (label_uid, verdict)
    gold[gold.decided][["label_uid", "expert"]].rename(columns={"expert": "verdict"}).to_csv(
        os.path.join(args.out_dir, "gold_for_cl.csv"), index=False, lineterminator="\n")
    summary = gold_summary(gold)
    summary.update(gold_sha256=sha, inputs=inputs, round3_splits=split_shas, pool_rows_found=n_found,
                   built=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write_json(os.path.join(args.out_dir, "gold_summary.json"), summary)
    print(gold_markdown(summary))
    print(f"\n{len(gold)} items -> {path} (sha256 {sha})")


# ---------------------------------------------------------------------------------------------------------------------
# crops: the label list and the staging list for the klone crop leg
# ---------------------------------------------------------------------------------------------------------------------

GEOMETRY = ["pano_x", "pano_y", "pano_width", "pano_height", "camera_pitch", "camera_roll"]


def label_author(user_id):
    """``ai`` for the SidewalkAI account, else ``human`` (``data_generation.label_author``, imported lazily)."""
    from data_generation import label_author as la
    return la(user_id)


def label_geometry(items, rawlabels_dir, label_points=None):
    """``(rows, missing)``: each item with the crop geometry, ``pano_source`` and ``author`` from the rawLabels cache
    (``<rawlabels_dir>/<city_id>.csv``, joined one-to-one on ``label_id`` within its city) or from
    ``label_points`` (``pull_label_points.sh`` output, keyed on ``schema, label_id``) for items the cache lacks
    (private deployments, excluded authors). ``missing`` lists the items found in neither."""
    parts = []
    for city, g in items.groupby("city_id", sort=True):
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        if not os.path.exists(path):
            continue
        r = pd.read_csv(path, usecols=["label_id", "user_id", "pano_source"] + GEOMETRY, dtype={"label_id": str},
                        low_memory=False)
        r["author"] = r.user_id.map(label_author)
        m = g[["label_uid", "label_id"]].astype({"label_id": str}).merge(
            r.drop(columns="user_id"), on="label_id", how="inner", validate="one_to_one")
        parts.append(m.drop(columns="label_id"))
    found = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label_uid"])
    if label_points is not None and len(label_points):
        lp = label_points.copy()
        lp["label_uid_schema"] = lp.schema.astype(str) + ":" + lp.label_id.astype(str)
        lp["author"] = np.where(_truthy(lp.author_is_ai), "ai", "human")
        rest = items[~items.label_uid.isin(found.label_uid)][["label_uid", "label_uid_schema"]]
        m = rest.merge(lp[["label_uid_schema", "pano_source", "author"] + GEOMETRY], on="label_uid_schema",
                       how="inner", validate="one_to_one")
        found = pd.concat([found, m.drop(columns="label_uid_schema")], ignore_index=True)
    rows = items.merge(found, on="label_uid", how="inner", validate="one_to_one")
    return rows, items[~items.label_uid.isin(found.label_uid)]


def crop_labels(rows):
    """``ai_labels_eval.LABEL_COLUMNS`` rows for its ``crops`` leg (columns it does not use left blank)."""
    from analysis import ai_labels_eval as ae
    out = pd.DataFrame({c: "" for c in ae.LABEL_COLUMNS}, index=rows.index)
    out["label_uid"], out["city"], out["label_id"] = rows.label_uid, rows.city_id, rows.label_uid.str.rsplit(
        ":", n=1).str[1]
    for c in ["pano_id", "author", "pano_source", "severity", "tags"] + GEOMETRY:
        out[c] = rows[c] if c in rows else ""
    out["label_class"] = ""  # the crop leg never sees the expert verdict
    return out.reset_index(drop=True)


def stage_files(rows):
    """The pano store paths ``ai_labels_eval.py select`` writes (``<city>/<pano_id[:2]>/<pano_id>.jpg``), sorted.

    >>> stage_files(pd.DataFrame({"city_id": ["c", "c"], "pano_id": ["abc", "abc"]}))
    ['c/ab/abc.jpg']
    """
    return sorted({f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(rows.city_id, rows.pano_id)})


def cmd_crops(args):
    refuse_in_checkout(args.out_dir, "the crop label list")
    items = load_gold(args.gold)
    if "label_id" not in items:
        items["label_id"] = items.label_uid.str.rsplit(":", n=1).str[1]
    for c in ("crop_path_r3", "crop_status_r3"):
        if c not in items:  # pool_items.csv: no round-3 crops recorded
            items[c] = ""
    have = (items.crop_status_r3 == "ok") & (items.crop_path_r3 != "")
    os.makedirs(args.out_dir, exist_ok=True)
    items[have][["label_uid", "label_type", "crop_path_r3"]].to_csv(
        os.path.join(args.out_dir, "existing_crops.csv"), index=False)
    todo = items[~have]
    lp = pd.read_csv(args.label_points, dtype=str, keep_default_na=False) if args.label_points else None
    rows, missing = label_geometry(todo, args.rawlabels, lp)
    crop_labels(rows).to_csv(os.path.join(args.out_dir, "labels.csv"), index=False)
    missing[["label_uid", "label_uid_schema", "label_type"]].to_csv(
        os.path.join(args.out_dir, "geometry_missing.csv"), index=False)
    panos = stage_files(rows)
    with open(os.path.join(args.out_dir, "stage_files.txt"), "w") as f:
        f.write("\n".join(panos) + ("\n" if panos else ""))
    if len(missing):
        req = " ".join(sorted(missing.label_uid_schema))
        with open(os.path.join(args.out_dir, "label_points_request.txt"), "w") as f:
            f.write(f"ssh makelab1 \"LABELS='{req}' bash -s\" < analysis/pull_label_points.sh > label_points.csv\n")
    print(f"{int(have.sum())} items already have a round-3 crop; {len(rows)} to cut from {len(panos)} panoramas; "
          f"{len(missing)} without geometry (not scorable"
          f"{'; label_points_request.txt has the pull' if len(missing) else ''}) -> {args.out_dir}")


def cmd_stage_list(args):
    """The pano count and size a pool run would stage, before anyone rsyncs it."""
    schema_city = schema_city_map(open(args.cityparams).read())
    pool = rl.load_pool(args.pool)
    types = args.types.split(",")
    d = pool[(pool.ai_vote == "Disagree") & pool.label_type.isin(types)].copy()
    d = d[d.schema.isin(schema_city)]
    d["city_id"] = d.schema.map(schema_city)
    d["label_uid_schema"] = d.label_uid
    d["label_uid"] = d.city_id + ":" + d.label_id.astype(str)
    panos = stage_files(d)
    print(f"{len(d):,} pool AI Disagrees of {types} over {len(panos):,} panoramas; about "
          f"{len(panos) * GB_PER_PANO:,.0f} GB to stage at {GB_PER_PANO * 1000:.0f} MB per panorama")
    if args.out_dir:
        refuse_in_checkout(args.out_dir, "the pool item list")
        os.makedirs(args.out_dir, exist_ok=True)
        d[["label_uid", "label_uid_schema", "city_id", "label_id", "label_type", "pano_id", "severity", "tags"]].to_csv(
            os.path.join(args.out_dir, "pool_items.csv"), index=False)
        print(f"-> {args.out_dir}/pool_items.csv (input to `crops --gold`)")


# ---------------------------------------------------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------------------------------------------------

RULE = {  # one paragraph per type: the rating rule under test (#74's first decision box; the round-2 rule, written out)
    "CurbRamp": ("A curb ramp that is present at the labelled point but in poor condition (steep, narrow, no tactile "
                 "warning, damaged, badly aligned) is a CORRECT label; its problems belong in the severity and tags. "
                 "The label is INCORRECT only if there is no curb ramp at the labelled point (for example a driveway, "
                 "an alley, a flush entrance or plain curb), or the ramp is somewhere else in the image."),
    "NoCurbRamp": ("A place where a sidewalk or crossing route meets the street at a raised curb with no ramp is a "
                   "CORRECT label, however minor the curb looks. The label is INCORRECT if a curb ramp is present at "
                   "the labelled point, if no pedestrian route crosses there, or if the labelled point is not at a "
                   "curb."),
    "Obstacle": ("An object in the sidewalk's walking path that blocks it or leaves too little room for a wheelchair "
                 "to pass is a CORRECT label. The label is INCORRECT if the object leaves enough room to pass, if it "
                 "is beside the path rather than in it, or if there is no object at the labelled point."),
    "SurfaceProblem": ("Damage to the walking surface at the labelled point that would make it harder to roll or "
                       "walk (cracks, uplifted or sunken slabs, bumps, broken or missing pavement, gravel, dirt or "
                       "grass) is a CORRECT label, even if it is not severe. The label is INCORRECT if the surface at "
                       "the labelled point is intact, if the point is on the road rather than a walking surface, or "
                       "if the problem is somewhere else in the image."),
    "Crosswalk": ("A marked crossing at the labelled point is a CORRECT label even if its paint is faded, worn or "
                  "partly hidden; its condition belongs in the severity and tags. The label is INCORRECT if there is "
                  "no marked crossing at the labelled point (for example plain road or only a stop line), or the "
                  "crossing is somewhere else in the image."),
}
PROMPT_VERSION = "adjudicator-v1"
PROMPT_VARIANTS = ("no-tags", "with-tags")


def _severity_sentence(severity):
    s = pd.to_numeric(pd.Series([severity]), errors="coerce").iloc[0]
    if pd.isna(s):
        return "The labeler gave no severity."
    return f"The labeler rated its severity {int(s)} on a 1 (minor) to 5 (severe) scale."


def _template(label_type, show_tags):
    """One type's prompt with ``{severity}`` and ``{tags}`` placeholders (what the template hash covers)."""
    parts = [vb.DEFINITIONS[label_type], RULE[label_type], vb.CENTRE_SENTENCE, "{severity}"]
    if show_tags:
        parts.append("{tags}")
    parts.append(f"Is this a correct {vb.TYPE_NAMES[label_type]} label? Answer yes or no.")
    return " ".join(parts)


def build_adjudicator_prompt(label_type, severity, tags=(), show_tags=False):
    """The adjudicator question for one label: the type's definition, the rating rule, the centre sentence, the
    labeler's severity, optionally the tags, and the yes/no question. Nothing about the AI, the other validators or
    the labeler's identity is in it.

    >>> p = build_adjudicator_prompt("Obstacle", 3.0)
    >>> p.endswith("rated its severity 3 on a 1 (minor) to 5 (severe) scale. Is this a correct obstacle label? "
    ...            "Answer yes or no.")
    True
    >>> "Tags: narrow." in build_adjudicator_prompt("Obstacle", None, "narrow", show_tags=True)
    True
    """
    if label_type not in RULE:
        raise ValueError(f"unknown label type {label_type!r}")
    tag_list = sl.parse_tags(tags) if not isinstance(tags, (list, tuple)) else list(tags)
    tag_text = f"Tags: {', '.join(tag_list)}." if tag_list else "Tags: none."
    return _template(label_type, show_tags).replace("{severity}", _severity_sentence(severity)).replace(
        "{tags}", tag_text)


def prompt_template_hash(show_tags=False):
    """sha256 over ``PROMPT_VERSION`` and every type's template (placeholders unrendered)."""
    blob = json.dumps({"version": PROMPT_VERSION, "show_tags": bool(show_tags),
                       "templates": {t: _template(t, show_tags) for t in LABEL_TYPES}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def prompt_hash(text):
    """sha256 of one rendered prompt.

    >>> prompt_hash("a")[:12]
    'ca978112ca1b'
    """
    return hashlib.sha256(text.encode()).hexdigest()


def build_reason_request(served_name, image_url, question, answer_word, prefill=None, template_kwargs=None,
                         max_tokens=REASON_MAX_TOKENS):
    """The follow-up request for a one-sentence reason: the question, the model's own answer as an assistant turn,
    then ``REASON_QUESTION``; greedy, no log-probs. With ``prefill`` (GLM's closed think block) a fourth, assistant
    turn is continued, as in the verdict request.

    >>> b = build_reason_request("m", "data:,", "Q?", "No.", prefill="<think></think>")
    >>> [m["role"] for m in b["messages"]], b["max_tokens"], "logprobs" in b, b["continue_final_message"]
    (['user', 'assistant', 'user', 'assistant'], 60, False, True)
    """
    msgs = vv.build_messages(image_url, question)
    msgs += [{"role": "assistant", "content": answer_word}, {"role": "user", "content": REASON_QUESTION}]
    body = {"model": served_name, "messages": msgs, "max_tokens": int(max_tokens), "temperature": 0.0, "n": 1}
    if prefill is not None:
        msgs.append({"role": "assistant", "content": prefill})
        body.update(continue_final_message=True, add_generation_prompt=False)
    if template_kwargs:
        body["chat_template_kwargs"] = dict(template_kwargs)
    return body


def cmd_prompt(args):
    for t in LABEL_TYPES:
        print(f"## {t}\n\n{build_adjudicator_prompt(t, 2, 'example tag', show_tags=args.show_tags)}\n")
    print(f"prompt version {PROMPT_VERSION}, template sha256 {prompt_template_hash(args.show_tags)}")


# ---------------------------------------------------------------------------------------------------------------------
# score: the gold set (or contested rows) through a vLLM server
# ---------------------------------------------------------------------------------------------------------------------

class AdjudicatorClient(vv.Client):
    """``vlm_vllm.Client`` (the verdict from first-token log-probs) plus :meth:`reason`."""

    def reason(self, image_path, question, answer_word):
        """``(text, completion_tokens)`` of the one-sentence reason; retries like :meth:`score`."""
        body = build_reason_request(self.served_name, vv.image_data_url(image_path), question, answer_word,
                                    self.prefill, self.template_kwargs)
        for attempt in range(self.retries + 1):
            try:
                resp = self.post(self.server + "/v1/chat/completions", body)
                break
            except (urllib.error.URLError, OSError, ValueError) as e:
                if attempt == self.retries:
                    raise
                print(f"  retry {attempt + 1} after {type(e).__name__}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"reason response has no content: {str(resp)[:300]}") from e
        return " ".join(str(text).split()), int(resp.get("usage", {}).get("completion_tokens", -1))


def check_owned(path, meta_path):
    """Refuse to append to a CSV or meta file this stage did not write.

    >>> check_owned("/nonexistent/a.csv", "/nonexistent/a.meta.json")
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != ADJ_COLUMNS:
            raise SystemExit(f"{path} was not written by adjudicator.py (columns {header}); use another --out-dir")
    if os.path.exists(meta_path) and vv._read_json(meta_path).get("backend") != ADJ_BACKEND:
        raise SystemExit(f"{meta_path} was not written by adjudicator.py; use another --out-dir")


def append_rows(path, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    pd.DataFrame(rows, columns=ADJ_COLUMNS).to_csv(path, mode="a", header=new_file, index=False)


def resolve_images(items, crops_root=None, split_root=None):
    """``items`` with an ``image`` column: the staged crop (``<crops_root>/manifest.csv``, status ok) when there is
    one, else the round-3 crop ``<split_root>/<label_type>/<crop_path_r3>``; blank when neither file exists."""
    out = items.copy()
    out["image"] = ""
    if split_root and "crop_path_r3" in out:
        ok = (out.get("crop_status_r3", "ok") == "ok") & (out.crop_path_r3 != "")
        out.loc[ok, "image"] = [os.path.join(split_root, t, p) for t, p in zip(out.label_type[ok], out.crop_path_r3[ok])]
    man = os.path.join(crops_root, "manifest.csv") if crops_root else None
    if man and os.path.exists(man):
        m = pd.read_csv(man, dtype=str, keep_default_na=False)
        m = m[m.status == "ok"]
        staged = dict(zip(m.label_uid, (os.path.join(crops_root, p) for p in m.crop_path)))
        hit = out.label_uid.isin(staged)
        out.loc[hit, "image"] = out.label_uid[hit].map(staged)
    out.loc[~out.image.map(lambda p: bool(p) and os.path.exists(p)), "image"] = ""
    return out


def score_one(client, row, show_tags, with_reason):
    q = build_adjudicator_prompt(row.label_type, row.severity, row.tags, show_tags)
    s = client.score(row.image, q)
    verdict = str(vb.verdict_from_score(s["p_incorrect"]))
    reason, n_tok = "", ""
    if with_reason:
        reason, n_tok = client.reason(row.image, q, "No." if verdict == "incorrect" else "Yes.")
    return {**s, "verdict": verdict, "reason": reason, "reason_tokens": n_tok, "prompt_sha256": prompt_hash(q)}


def score_items(client, todo, path, model, gpu, concurrency, show_tags=False, with_reason=True, chunk=None,
                on_chunk=None):
    """Score ``todo`` (``label_uid, label_type, severity, tags, image``) with ``concurrency`` requests in flight;
    each crop's verdict and reason requests run in the same thread. Appends each chunk to ``path`` in
    ``ADJ_COLUMNS`` order and returns ``[[n, wall_s], ...]``."""
    chunk = chunk or max(4 * concurrency, 16)
    rows, chunks = list(todo.itertuples()), []
    variant = PROMPT_VARIANTS[int(bool(show_tags))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            t0 = time.perf_counter()
            res = list(pool.map(lambda r: score_one(client, r, show_tags, with_reason), part))
            wall = time.perf_counter() - t0
            append_rows(path, [{"label_uid": r.label_uid, "model": model, "variant": PROMPT_VERSION,
                                "batch_size": concurrency, "gpu": gpu, "prompt_variant": variant, **s}
                               for r, s in zip(part, res)])
            chunks.append([len(part), wall])
            if on_chunk:
                on_chunk(start + len(part), len(rows), chunks)
    return chunks


def _meta(args, version, extra):
    """``vlm_vllm._base_meta``-shaped provenance, with this stage's backend and prompt hashes."""
    args.variant = PROMPT_VERSION
    m = vv._base_meta(args, version)
    m.update(backend=ADJ_BACKEND, revision_requested=args.revision, prompt_version=PROMPT_VERSION,
             prompt_template_sha256=prompt_template_hash(args.show_tags), show_tags=bool(args.show_tags),
             reason=not args.no_reason, reason_request={"question": REASON_QUESTION, "max_tokens": REASON_MAX_TOKENS},
             **extra)
    return m


def run_scoring(args, items, path, extra_meta):
    """Resumable scoring of ``items`` into ``path`` (+ ``.meta.json``), as ``vlm_vllm.run_score``."""
    meta_path = path[:-4] + ".meta.json"
    check_owned(path, meta_path)
    client = AdjudicatorClient(args.server, args.served_name or args.repo, args.prefill, args.template_kwargs)
    version = vv._server_version(args.server)
    scorable = items[items.image != ""]
    done = vv.done_uids(path)
    todo = scorable[~scorable.label_uid.isin(done)]
    print(f"{args.model}: {len(todo)} to score, {len(done)} done, {len(items) - len(scorable)} without an image, "
          f"concurrency {args.concurrency}", flush=True)
    meta = {**vv._read_json(meta_path), **_meta(args, version, extra_meta),
            "n_items": int(len(items)), "n_no_image": int(len(items) - len(scorable))}
    if len(scorable) and not meta.get("rendered_prompt_tail"):
        r = scorable.iloc[0]
        try:
            meta["rendered_prompt_tail"] = client.rendered_tail(
                r.image, build_adjudicator_prompt(r.label_type, r.severity, r.tags, args.show_tags))
        except Exception as e:  # noqa: BLE001 -- provenance only
            meta["rendered_prompt_tail"] = {"error": f"{type(e).__name__}: {e}"}
    meta.setdefault("chunks", [])
    vv._write_json(meta_path, meta)
    if len(todo):
        n_prev = len(meta["chunks"])

        def progress(k, n, chunks, _t0=time.time()):
            print(f"  {k}/{n}  {time.time() - _t0:.0f}s  last chunk {chunks[-1][1] / chunks[-1][0]:.3f} s/crop",
                  flush=True)
            meta["chunks"] = meta["chunks"][:n_prev] + chunks
            vv._write_json(meta_path, meta)
        score_items(client, todo, path, args.model, args.gpu, args.concurrency, args.show_tags, not args.no_reason,
                    on_chunk=progress)
    if os.path.exists(path):
        meta.update(vv.summarize_scores(path))
        sp = vv.chunk_s_per_crop(meta["chunks"])
        if sp:
            meta["s_per_crop"] = float(np.median(sp))
        vv._write_json(meta_path, meta)
    return meta


def limit_per_type(df, limit):
    """The first ``limit`` rows per type by ``label_uid`` (a smoke run); all rows when ``limit`` is 0."""
    if not limit:
        return df
    return df.sort_values(["label_type", "label_uid"]).groupby("label_type").head(limit)


def cmd_score(args):
    refuse_in_checkout(args.out_dir, "VLM scores")
    gold = load_gold(args.gold)
    items = limit_per_type(resolve_images(gold, args.crops_root, args.split_root), args.limit)
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    meta = run_scoring(args, items, os.path.join(model_dir, "gold.csv"),
                       {"gold_sha256": rl.sha256_file(args.gold), "rows": "gold"})
    print(json.dumps({k: meta.get(k) for k in ("n_rows", "n_no_image", "s_per_crop", "revision")}))


# ---------------------------------------------------------------------------------------------------------------------
# agree: agreement with the expert (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def load_scores(scores_dir, models, name="gold.csv"):
    """Every model's score CSV stacked (``model``, ``label_uid``, ``p_incorrect``, ``verdict``)."""
    parts = []
    for m in models:
        p = pd.read_csv(os.path.join(scores_dir, m, name), keep_default_na=False)
        parts.append(p[["label_uid", "p_incorrect", "verdict"]].assign(model=m))
    return pd.concat(parts, ignore_index=True)


def merge_scores(gold, scores):
    """Decided gold rows joined to each model's scores (one row per model and item; items a model did not score
    are left out and counted by the caller)."""
    g = gold[gold.decided]
    parts = []
    for model, s in scores.groupby("model", sort=False):
        m = g.merge(s, on="label_uid", how="inner", validate="one_to_one")
        parts.append(m)
    m = pd.concat(parts, ignore_index=True) if parts else g.assign(model="", p_incorrect=np.nan, verdict="")
    m["p_incorrect"] = _num(m.p_incorrect)
    return m


def _groups(m, by):
    """``(model, key, frame)`` per model x value of ``by`` plus ``all``."""
    for model, x in m.groupby("model", sort=False):
        yield model, "all", x
        for k in (LABEL_TYPES if by == "label_type" else sorted(x[by].unique())):
            sub = x[x[by] == k]
            if len(sub):
                yield model, k, sub


def agreement_rows(m, n_boot=N_BOOT, seed=0):
    """Per model x (type | all): ``n``, ``vlm_agree`` [``lo``, ``hi``] (Wilson; share of items where the model's
    verdict is the expert's), the crowd majority on the items that have one (``crowd_n``, ``crowd_agree``
    [``crowd_lo``, ``crowd_hi``]), ``ai_confirmed`` (share of the AI's Disagrees the expert calls incorrect, context)
    and ``auc_vs_expert`` [``auc_lo``, ``auc_hi``] of ``p_incorrect`` against the expert (panorama bootstrap)."""
    rows = []
    for model, t, x in _groups(m, "label_type"):
        v = rl._rate(int((x.verdict == x.expert).sum()), len(x))
        c = x[x.crowd_class != ""]
        cr = rl._rate(int((c.crowd_class == c.expert).sum()), len(c))
        d = x[x.ai_vote == "Disagree"]
        y = (x.expert == "incorrect").to_numpy()
        auc = sl._auc(y, x.p_incorrect.to_numpy()) if len(x) else np.nan
        lo, hi = sl.bootstrap_auc(y, x.p_incorrect, x.pano_id, n_boot, sl.slice_seed(seed, f"{model}/{t}"))
        rows.append({"model": model, "label_type": t, "n": v["n"], "vlm_agree": v["rate"], "lo": v["lo"],
                     "hi": v["hi"], "crowd_n": cr["n"], "crowd_agree": cr["rate"], "crowd_lo": cr["lo"],
                     "crowd_hi": cr["hi"], "ai_disagree_n": int(len(d)),
                     "ai_confirmed": float((d.expert == "incorrect").mean()) if len(d) else np.nan,
                     "auc_vs_expert": auc, "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(rows)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial on the ``b + c`` discordant pairs, p = 1/2), without SciPy.

    >>> mcnemar_exact(0, 0), round(mcnemar_exact(8, 2), 4), round(mcnemar_exact(10, 0), 5)
    (1.0, 0.1094, 0.00195)
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_delta(vlm_ok, crowd_ok, n_boot=N_BOOT, seed=0):
    """``(delta, lo, hi)``: the model's agreement rate minus the crowd's on the same items, with a 95% percentile
    interval from resampling items with replacement (both rates from the same resample)."""
    a, b = np.asarray(vlm_ok, float), np.asarray(crowd_ok, float)
    if not len(a):
        return np.nan, np.nan, np.nan
    delta = float(a.mean() - b.mean())
    if not n_boot:
        return delta, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return delta, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def paired_rows(m, n_boot=N_BOOT, seed=0):
    """Per model x (type | all), on the items with a crowd majority: ``n_pair``, ``vlm_agree``, ``crowd_agree``,
    ``delta`` [``delta_lo``, ``delta_hi``] (paired, label-level bootstrap), ``mcnemar_p`` and the discordant counts
    ``b_vlm_only`` (the model right, the crowd wrong) and ``b_crowd_only``."""
    rows = []
    for model, t, x in _groups(m[m.crowd_class != ""], "label_type"):
        v, c = (x.verdict == x.expert).to_numpy(), (x.crowd_class == x.expert).to_numpy()
        delta, lo, hi = paired_delta(v, c, n_boot, sl.slice_seed(seed, f"{model}/{t}"))
        b, cc = int((v & ~c).sum()), int((~v & c).sum())
        rows.append({"model": model, "label_type": t, "n_pair": int(len(x)), "vlm_agree": float(v.mean()),
                     "crowd_agree": float(c.mean()), "delta": delta, "delta_lo": lo, "delta_hi": hi,
                     "mcnemar_p": mcnemar_exact(b, cc), "b_vlm_only": b, "b_crowd_only": cc})
    return pd.DataFrame(rows)


def stratum_rows(m):
    """Per model x stratum (and all): the model's and, where there is one, the crowd's agreement with the expert."""
    rows = []
    for model, s, x in _groups(m, "stratum"):
        v = rl._rate(int((x.verdict == x.expert).sum()), len(x))
        c = x[x.crowd_class != ""]
        cr = rl._rate(int((c.crowd_class == c.expert).sum()), len(c))
        rows.append({"model": model, "stratum": s, "n": v["n"], "vlm_agree": v["rate"], "lo": v["lo"], "hi": v["hi"],
                     "crowd_n": cr["n"], "crowd_agree": cr["rate"]})
    return pd.DataFrame(rows)


def slice_tables(m, train, min_n=SLICE_MIN_N, n_boot=N_BOOT, seed=0):
    """``{"severity": df, "tag": df}``: ``slices.slice_table`` per model and type of the model's ``p_incorrect``
    against the expert, with ``train`` (the round-3 train rows of each type: ``label_type, severity, tags``) for the
    training counts per slice. ``vlm_agree`` is ``slice_table``'s accuracy at 0.5, the agreement with the expert."""
    out = {}
    for by in sl.BY:
        parts = []
        for (model, t), x in m.groupby(["model", "label_type"], sort=False):
            pred = pd.DataFrame({"label_uid": x.label_uid, "true": x.expert, "p_incorrect": x.p_incorrect,
                                 "severity": x.severity, "tags": x.tags, "pano_id": x.pano_id, "city": x.city_id})
            tr = train[train.label_type == t] if train is not None and len(train) else None
            tab = sl.slice_table(pred, by, train=tr, min_n=min_n, n_boot=n_boot, seed=seed)
            parts.append(tab.assign(model=model, label_type=t, vlm_agree=tab.accuracy))
        cols = ["model", "label_type"]
        out[by] = pd.concat(parts, ignore_index=True).pipe(lambda d: d[cols + [c for c in d.columns if c not in cols]]) \
            if parts else pd.DataFrame()
    return out


def bar_decision(paired_row, all_row, min_n=BAR_MIN_N):
    """Whether a (model, type) clears the bar: ``n_pair >= min_n``, the paired lower bound of (model - crowd)
    agreement with the expert ``> 0``, and the model's agreement on every decided item (the lone ones, which have no
    crowd majority, included) at least the crowd's on the paired items.

    >>> bar_decision({"n_pair": 40, "delta_lo": 0.01, "crowd_agree": 0.7}, {"vlm_agree": 0.8})["clears"]
    True
    >>> bar_decision({"n_pair": 40, "delta_lo": 0.0, "crowd_agree": 0.7}, {"vlm_agree": 0.8})["reason"]
    'paired lower bound 0.000 is not above 0'
    """
    n, lo = int(paired_row.get("n_pair", 0) or 0), paired_row.get("delta_lo", np.nan)
    if n < min_n:
        return {"clears": False, "reason": f"n_pair {n} < {min_n}"}
    if lo is None or pd.isna(lo) or not lo > 0:
        return {"clears": False, "reason": f"paired lower bound {lo:.3f} is not above 0"}
    if not all_row.get("vlm_agree", np.nan) >= paired_row["crowd_agree"]:
        return {"clears": False, "reason": f"agreement on all items {all_row.get('vlm_agree'):.3f} < crowd "
                                           f"{paired_row['crowd_agree']:.3f}"}
    return {"clears": True, "reason": f"n_pair {n}, paired lower bound {lo:.3f} > 0, all-item agreement "
                                      f"{all_row['vlm_agree']:.3f} >= crowd {paired_row['crowd_agree']:.3f}"}


def bar_table(agreement, paired, min_n=BAR_MIN_N):
    """``{model: {type: {"clears", "reason", "delta", "delta_lo"}}}`` for every model x (type | all)."""
    out = {}
    a = agreement.set_index(["model", "label_type"])
    for r in paired.to_dict("records"):
        d = bar_decision(r, a.loc[(r["model"], r["label_type"])].to_dict(), min_n)
        d.update(delta=r["delta"], delta_lo=r["delta_lo"], n_pair=r["n_pair"])
        out.setdefault(r["model"], {})[r["label_type"]] = d
    return out


def _pct(r, lo, hi):
    return "" if r is None or pd.isna(r) else rl._pct(r, lo, hi)


def agreement_markdown(rows):
    """The agreement table (one layout for the models and, with ``model`` named, the reliability-weighted crowd)."""
    lines = ["| Model | Type | n | Agrees with the expert | Crowd majority on its items (n) | AUC vs expert |",
             "|---|---|---:|---|---|---|"]
    for r in rows.to_dict("records"):
        auc = "" if pd.isna(r.get("auc_vs_expert", np.nan)) else f"{r['auc_vs_expert']:.3f}" + (
            "" if pd.isna(r.get("auc_lo", np.nan)) else f" [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]")
        crowd = "" if not r.get("crowd_n") else f"{_pct(r['crowd_agree'], r['crowd_lo'], r['crowd_hi'])} ({r['crowd_n']})"
        lines.append(f"| {r['model']} | {r['label_type']} | {r['n']} | {_pct(r['vlm_agree'], r['lo'], r['hi'])} | "
                     f"{crowd} | {auc} |")
    ref = R1_REFERENCE
    lines.append(f"| reference: {ref['label']} | all | {ref['n']} | {_pct(ref['rate'], ref['lo'], ref['hi'])} | | |")
    return "\n".join(lines)


def paired_markdown(rows):
    lines = ["| Model | Type | Items with a crowd majority | Model agrees | Crowd agrees | Model - crowd [95%] | "
             "Model right, crowd wrong | Crowd right, model wrong | McNemar p |", "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows.to_dict("records"):
        ci = "" if pd.isna(r["delta_lo"]) else f" [{100 * r['delta_lo']:+.1f}, {100 * r['delta_hi']:+.1f}]"
        lines.append(f"| {r['model']} | {r['label_type']} | {r['n_pair']} | {100 * r['vlm_agree']:.1f}% | "
                     f"{100 * r['crowd_agree']:.1f}% | {100 * r['delta']:+.1f}{ci} | {r['b_vlm_only']} | "
                     f"{r['b_crowd_only']} | {r['mcnemar_p']:.3g} |")
    return "\n".join(lines)


def strata_markdown(rows):
    lines = ["| Model | Stratum | n | Model agrees with the expert | Crowd agrees (n) |", "|---|---|---:|---|---|"]
    for r in rows.to_dict("records"):
        crowd = "" if not r["crowd_n"] else f"{100 * r['crowd_agree']:.1f}% ({r['crowd_n']})"
        lines.append(f"| {r['model']} | {r['stratum']} | {r['n']} | {_pct(r['vlm_agree'], r['lo'], r['hi'])} | {crowd} |")
    return "\n".join(lines)


def slices_markdown(tabs):
    out = []
    for by, t in tabs.items():
        out += [f"**By {by}** (agreement blank under {SLICE_MIN_N} items; train = round-3 training rows of the slice)",
                "", "| Model | Type | Slice | n | Agrees | AUC | Train rows |", "|---|---|---|---:|---:|---:|---:|"]
        for r in t.to_dict("records"):
            agree = "" if pd.isna(r["vlm_agree"]) else f"{100 * r['vlm_agree']:.1f}%"
            auc = "" if pd.isna(r["auc"]) else f"{r['auc']:.3f}"
            out.append(f"| {r['model']} | {r['label_type']} | {r['slice']} | {r['n']} | {agree} | {auc} | "
                       f"{r.get('train_n', '')} |")
        out.append("")
    return "\n".join(out).rstrip()


def bar_markdown(bar):
    lines = ["| Model | Type | Clears | Why |", "|---|---|---|---|"]
    for model, types in bar.items():
        for t, d in types.items():
            lines.append(f"| {model} | {t} | {'yes' if d['clears'] else 'no'} | {d['reason']} |")
    return "\n".join(lines)


def train_rows_of(split_root, types=LABEL_TYPES):
    """The round-3 train rows (``label_type, severity, tags``) of each type's ``split.csv``, for training counts."""
    parts = []
    for t in types:
        p = os.path.join(split_root, t, "split.csv")
        if os.path.exists(p):
            s = cl.read_split(p)
            parts.append(s[s.split == "train"][["severity", "tags"]].assign(label_type=t))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label_type", "severity", "tags"])


def cmd_agree(args):
    gold = load_gold(args.gold)
    models = args.models.split(",")
    m = merge_scores(gold, load_scores(args.scores, models))
    unscored = {mod: int(gold.decided.sum() - (m.model == mod).sum()) for mod in models}
    agreement = agreement_rows(m, args.n_boot, args.seed)
    paired = paired_rows(m, args.n_boot, args.seed)
    strata = stratum_rows(m)
    tabs = slice_tables(m, train_rows_of(args.split_root) if args.split_root else None, n_boot=args.n_boot,
                        seed=args.seed)
    bar = bar_table(agreement, paired)
    os.makedirs(args.out_dir, exist_ok=True)
    agreement.to_csv(os.path.join(args.out_dir, "agreement.csv"), index=False)
    paired.to_csv(os.path.join(args.out_dir, "paired.csv"), index=False)
    strata.to_csv(os.path.join(args.out_dir, "strata.csv"), index=False)
    tabs["severity"].to_csv(os.path.join(args.out_dir, "slices_severity.csv"), index=False)
    tabs["tag"].to_csv(os.path.join(args.out_dir, "slices_tag.csv"), index=False)
    write_json(os.path.join(args.out_dir, "bar.json"), {"min_n": BAR_MIN_N, "n_boot": args.n_boot, "seed": args.seed,
                                                       "unscored_decided": unscored, "decisions": bar})
    md = "\n\n".join([agreement_markdown(agreement), paired_markdown(paired), strata_markdown(strata),
                      slices_markdown(tabs), bar_markdown(bar)])
    with open(os.path.join(args.out_dir, "agree.md"), "w") as f:
        f.write(md + "\n")
    print(md)


# ---------------------------------------------------------------------------------------------------------------------
# disagreements: a masked reading packet
# ---------------------------------------------------------------------------------------------------------------------

def pick_disagreements(m, seed, per_type=PACKET_PER_TYPE):
    """Up to ``per_type`` items per type where the model's verdict differs from the expert's, in a seeded order."""
    d = m[m.verdict != m.expert].copy()
    d["rank"] = [rl.seeded_rank(seed, u) for u in d.label_uid]
    return d.sort_values(["label_type", "rank"]).groupby("label_type").head(per_type).drop(columns="rank")


def cmd_disagreements(args):
    for d, what in ((args.out_dir, "the packet"), (args.key_dir, "the packet key")):
        refuse_in_checkout(d, what)
    gold = load_gold(args.gold)
    scores = pd.read_csv(os.path.join(args.scores, args.model, "gold.csv"), keep_default_na=False)
    m = merge_scores(gold, scores[["label_uid", "p_incorrect", "verdict", "reason"]].assign(model=args.model))
    picked = pick_disagreements(m, args.seed)
    picked = resolve_images(picked, args.crops_root, args.split_root)
    key = rl.make_key(picked, args.seed)
    os.makedirs(os.path.join(args.out_dir, "items"), exist_ok=True)
    os.makedirs(args.key_dir, exist_ok=True)
    lines = [f"# {args.model}: reading packet ({len(key)} items)", "",
             "Read each crop and the model's reason. Reading only: nothing here is rated.", "",
             "| Item | Type | Severity | Model verdict | Model reason |", "|---|---|---|---|---|"]
    for r in key.itertuples():
        if r.image:
            shutil.copyfile(r.image, os.path.join(args.out_dir, "items", r.packet_id + os.path.splitext(r.image)[1]))
        sev = _severity_sentence(r.severity).replace("The labeler rated its severity ", "").split(" on ")[0]
        sev = "none" if sev.startswith("The labeler gave") else sev
        lines.append(f"| {r.packet_id} | {r.label_type} | {sev} | {r.verdict} | {str(r.reason).replace('|', '/')} |")
    with open(os.path.join(args.out_dir, "sheet.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    kpath = os.path.join(args.key_dir, "key.csv")
    key.rename(columns={"packet_id": "rid"})[["rid", "label_uid", "expert", "crowd_class", "ai_vote"]].to_csv(
        kpath, index=False)
    write_json(os.path.join(args.out_dir, "packet_manifest.json"),
               {"model": args.model, "n": int(len(key)), "per_type": {k: int(v) for k, v in
                                                                      key.label_type.value_counts().items()},
                "key_sha256": rl.sha256_file(kpath), "seed_sha256": hashlib.sha256(str(args.seed).encode()).hexdigest()})
    print(f"{len(key)} items -> {args.out_dir}/sheet.md; key -> {kpath}")


# ---------------------------------------------------------------------------------------------------------------------
# relabel and variants
# ---------------------------------------------------------------------------------------------------------------------

def pool_city_uids(pool, schema_city, label_type=None, ai_vote="Disagree"):
    """City-form ``label_uid`` of the pool's rows with the given AI vote (and type); schemas the conf lacks are
    skipped."""
    p = pool[(pool.ai_vote == ai_vote) & pool.schema.isin(schema_city)]
    if label_type:
        p = p[p.label_type == label_type]
    return set(p.schema.map(schema_city) + ":" + p.label_id.astype(str))


def contested_rows(split, ai_disagree_uids, noisy=None, mode="both"):
    """The contested train rows of one type's split: ``split == "train"`` rows whose ``label_uid`` is an AI Disagree
    (``ai-disagree``), flagged by confident learning (``cl-flagged``; ``noisy`` = ``noisy_targets.csv``), or either
    (``both``). Adds ``contested_by`` (``ai_disagree``, ``cl_flagged`` or ``both``).

    >>> s = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "split": ["train", "train", "train", "val"]})
    >>> n = pd.DataFrame({"label_uid": ["a:2", "a:3"], "flagged": ["1", "0"]})
    >>> contested_rows(s, {"a:1", "a:4"}, n).contested_by.tolist()
    ['ai_disagree', 'cl_flagged']
    >>> len(contested_rows(s, {"a:1"}, n, mode="ai-disagree"))
    1
    """
    if mode not in ("ai-disagree", "cl-flagged", "both"):
        raise ValueError(f"unknown contested mode {mode!r}")
    tr = split[split.split == "train"].copy()
    ai = tr.label_uid.isin(ai_disagree_uids)
    flagged = set() if noisy is None else set(noisy.label_uid[noisy.flagged.astype(str).isin(["1", "True", "true"])])
    fl = tr.label_uid.isin(flagged)
    use = {"ai-disagree": ai, "cl-flagged": fl, "both": ai | fl}[mode]
    tr["contested_by"] = np.where(ai & fl, "both", np.where(ai, "ai_disagree", "cl_flagged"))
    return tr[use].reset_index(drop=True)


def check_bar(bar, model, label_type, force=False):
    """Exit unless ``bar.json`` says ``model`` clears the bar for ``label_type`` (or ``force``)."""
    d = bar.get("decisions", bar).get(model, {}).get(label_type)
    if d and d.get("clears"):
        return True
    if force:
        print(f"WARNING: {model} / {label_type} does not clear the bar ({(d or {}).get('reason', 'no decision')}); "
              f"--force given, recorded in the meta")
        return False
    sys.exit(f"{model} does not clear the bar for {label_type} ({(d or {}).get('reason', 'no decision in bar.json')}); "
             f"pass --force to run anyway")


def cmd_relabel(args):
    refuse_in_checkout(args.out_dir, "relabel scores")
    bar = json.load(open(args.bar))
    schema_city = schema_city_map(open(args.cityparams).read())
    pool = rl.load_pool(args.pool)
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    for t in args.types.split(","):
        cleared = check_bar(bar, args.model, t, args.force)
        ai = pool_city_uids(pool, schema_city, t)
        if args.rows == "train":
            split = cl.read_split(os.path.join(args.split_root, t, "split.csv"))
            noisy_p = os.path.join(args.split_root, t, "cl", "noisy_targets.csv")
            noisy = pd.read_csv(noisy_p, dtype=str, keep_default_na=False) if os.path.exists(noisy_p) else None
            rows = contested_rows(split, ai, noisy, "both").assign(label_type=t)
            rows = rows.rename(columns={"crop_path": "crop_path_r3", "status": "crop_status_r3"})
            rows.loc[:, ["label_uid", "contested_by"]].to_csv(os.path.join(model_dir, f"contested_{t}.csv"),
                                                             index=False)
        else:
            p = pool[(pool.ai_vote == "Disagree") & (pool.label_type == t) & pool.schema.isin(schema_city)]
            rows = pd.DataFrame({"label_uid": p.schema.map(schema_city) + ":" + p.label_id.astype(str),
                                 "label_type": t, "severity": p.severity, "tags": p.tags.fillna("")})
        rows = resolve_images(rows, args.crops_root, args.split_root)
        extra = {"rows": args.rows, "label_type": t, "pool_sha256": rl.sha256_file(args.pool), "bar_cleared": cleared,
                 "forced": bool(args.force and not cleared), "bar_sha256": rl.sha256_file(args.bar)}
        run_scoring(args, rows, os.path.join(model_dir, f"{args.rows}_{t}.csv"), extra)


def soft_blend(agree, disagree, p_correct, w):
    """The ``adj-soft`` target: the VLM's ``p_correct`` as ``w`` extra votes in the smoothed vote rate of
    ``fine_tune_classifier.soft_targets``, ``(agree + 1 + w * p_correct) / (agree + disagree + 2 + w)``.

    >>> soft_blend(0, 2, 1.0, 2.0), soft_blend(3, 0, 0.0, 0.0) == smoothed_rate(3, 0)
    (0.5, True)
    """
    return (agree + 1 + w * p_correct) / (agree + disagree + 2 + w)


def adj_replace(base, verdicts, contested_uids, spare_expert=True):
    """``adj-replace``: on contested, scored train rows the target becomes the model's verdict; rows whose target
    changes get ``class_source = adjudicator``. Expert-sourced targets are left alone with ``spare_expert``. Every
    val / test / test_heldout row is unchanged (checked)."""
    v = base.copy()
    hit = (v.split == "train") & v.label_uid.isin(contested_uids) & v.label_uid.isin(verdicts.keys())
    if spare_expert and "class_source" in v:
        hit &= v.class_source != "expert"
    new = v.loc[hit, "label_uid"].map(verdicts)
    changed = hit.copy()
    changed[hit] = (new != v.loc[hit, "label_class"]).to_numpy()
    v.loc[changed, "label_class"] = new[changed[hit].to_numpy()].to_numpy()
    if "class_source" in v:
        v.loc[changed, "class_source"] = "adjudicator"
    cl.check_eval_rows_unchanged(base, v)
    return v


def adj_soft(base, p_correct, contested_uids, w=2.0, spare_expert=True):
    """``adj-soft``: a ``soft_target`` column on every train row, the smoothed human vote rate, and on contested,
    scored rows :func:`soft_blend` with the model's ``p_correct``. ``label_class`` is unchanged, so validation stays
    hard; non-train rows get a blank ``soft_target``. Read by ``fine_tune_classifier.soft_targets`` (``--target
    soft``)."""
    v = base.copy()
    tr = v.split == "train"
    a, d = _num(v.human_agree), _num(v.human_disagree)
    target = smoothed_rate(a, d)
    hit = tr & v.label_uid.isin(contested_uids) & v.label_uid.isin(p_correct.keys())
    if spare_expert and "class_source" in v:
        hit &= v.class_source != "expert"
    pc = v.label_uid.map(p_correct)
    target[hit] = soft_blend(a[hit], d[hit], pc[hit].astype(float), w)
    v["soft_target"] = [f"{x:.6f}" if t else "" for x, t in zip(target, tr)]
    cl.check_eval_rows_unchanged(base, v.drop(columns="soft_target"))
    return v


def mark_gold_dropped(variant, gold_uids):
    """``--drop-gold``: gold-set train rows become ``train_dropped`` (kept in the file, never trained on)."""
    v = variant.copy()
    v.loc[(v.split == "train") & v.label_uid.isin(gold_uids), "split"] = cl.TRAIN_DROPPED
    return v


def cmd_variants(args):
    model_dir = os.path.join(args.relabel, args.model)
    gold_uids = set(load_gold(args.gold).label_uid) if args.drop_gold else set()
    for t in args.types.split(","):
        base_path = os.path.join(args.split_root, t, "split.csv")
        base = cl.read_split(base_path)
        scores_path = os.path.join(model_dir, f"train_{t}.csv")
        s = pd.read_csv(scores_path, keep_default_na=False)
        con = pd.read_csv(os.path.join(model_dir, f"contested_{t}.csv"), dtype=str, keep_default_na=False)
        want = {"ai-disagree": ("ai_disagree", "both"), "cl-flagged": ("cl_flagged", "both"),
                "both": ("ai_disagree", "cl_flagged", "both")}[args.contested]
        contested = set(con.label_uid[con.contested_by.isin(want)])
        verdicts = dict(zip(s.label_uid, s.verdict))
        p_correct = dict(zip(s.label_uid, 1 - _num(s.p_incorrect)))
        meta_path = scores_path[:-4] + ".meta.json"
        meta = vv._read_json(meta_path)
        for name, v in (("adj-replace", adj_replace(base, verdicts, contested, not args.no_spare_expert)),
                        ("adj-soft", adj_soft(base, p_correct, contested, args.adjudicator_votes,
                                              not args.no_spare_expert))):
            if args.drop_gold:
                v = mark_gold_dropped(v, gold_uids)
            out = os.path.join(args.split_root, t, "cl", name, "split.csv")
            cl.write_split(v, out)
            tr = base.split == "train"
            changed = int((tr & (v.label_class != base.label_class)).sum()) if name == "adj-replace" else 0
            info = {"variant": name, "label_type": t, "model": args.model, "rows_contested": len(contested),
                    "rows_scored": int(len(contested & set(verdicts))), "train_rows": int(tr.sum()),
                    "changed": changed, "kept": int(tr.sum()) - changed, "contested_mode": args.contested,
                    "spare_expert": not args.no_spare_expert, "adjudicator_votes": args.adjudicator_votes,
                    "drop_gold": bool(args.drop_gold),
                    "gold_train_rows_dropped": int((v.split == cl.TRAIN_DROPPED).sum()) if args.drop_gold else 0,
                    "relabel_sha256": rl.sha256_file(scores_path), "split_sha256": rl.sha256_file(out),
                    "base_split_sha256": rl.sha256_file(base_path), "model_revision": meta.get("revision"),
                    "prompt_template_sha256": meta.get("prompt_template_sha256")}
            if name == "adj-soft":
                st = _num(v.soft_target[tr])
                info["soft_target_mean"] = float(st.mean())
            write_json(os.path.join(os.path.dirname(out), "variant.json"), info)
            print(f"{t} {name}: {json.dumps({k: info[k] for k in ('rows_contested', 'rows_scored', 'changed')})}")


# ---------------------------------------------------------------------------------------------------------------------
# reliability: issue #12's scoring step
# ---------------------------------------------------------------------------------------------------------------------

def reliability_target(w_agree, w_disagree):
    """``correct`` / ``incorrect`` / ``""`` from reliability-weighted votes: the side of 0.5 the smoothed weighted
    rate falls on (blank at exactly 0.5 or with no weights).

    >>> reliability_target(3.2, 0.9), reliability_target(0.4, 1.1), reliability_target(1.0, 1.0)
    ('correct', 'incorrect', '')
    >>> reliability_target(float("nan"), 1.0)
    ''
    """
    if w_agree is None or w_disagree is None or pd.isna(w_agree) or pd.isna(w_disagree):
        return ""
    r = smoothed_rate(float(w_agree), float(w_disagree))
    return "correct" if r > 0.5 else "incorrect" if r < 0.5 else ""


def cached_crowd_target(validations_json):
    """``(target, n_human_votes)``: the unweighted human majority in one rawLabels ``validations`` cell
    (``validator_reliability.human_votes``: human Agree / Disagree only, the AI's votes excluded).

    >>> cached_crowd_target('[{"user_id":"a","validation":"Agree","validator_type":"Human"},'
    ...                     ' {"user_id":"x","validation":"Disagree","validator_type":"AI"}]')
    ('correct', 1)
    """
    votes, _ = vr.human_votes(validations_json)
    a = sum(votes.values())
    d = len(votes) - a
    return ("correct" if a > d else "incorrect" if d > a else ""), len(votes)


def snapshot_check(reliability_run, rawlabels_dir, rated_after):
    """Raise ``ValueError`` when the weights or the rawLabels cache were made on or after ``rated_after`` (an ISO
    date): then a rater's own gold votes could be in the crowd they are compared with."""
    def utc(text):
        t = datetime.fromisoformat(text)
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    cutoff = utc(rated_after)
    created = reliability_run.get("created")
    if not created:
        raise ValueError("reliability_run.json has no created time")
    if utc(created) >= cutoff:
        raise ValueError(f"the vote weights were created {created}, not before the rating ({rated_after})")
    for f in os.listdir(rawlabels_dir):
        if f.endswith(".csv"):
            t = datetime.fromtimestamp(os.path.getmtime(os.path.join(rawlabels_dir, f)), timezone.utc)
            if t >= cutoff:
                raise ValueError(f"rawLabels cache file {f} was written {t.isoformat()}, not before the rating")


def reliability_frame(gold, weights, rawlabels_dir, public):
    """Decided gold rows with ``covered`` (a public deployment, and at least one human vote in the cache),
    ``crowd_pool`` (the pool majority), ``crowd_cache`` (the cache's unweighted majority) and
    ``reliability_weighted`` (:func:`reliability_target`)."""
    g = gold[gold.decided].copy()
    g["label_id"] = g.label_uid.str.rsplit(":", n=1).str[1]
    g["crowd_cache"], g["n_cache_votes"] = "", 0
    for city, x in g.groupby("city_id"):
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        if city not in public or not os.path.exists(path):
            continue
        r = pd.read_csv(path, usecols=["label_id", "validations"], dtype=str)
        m = x[["label_id"]].merge(r, on="label_id", how="left", validate="one_to_one")
        t = [cached_crowd_target(v) for v in m.validations]
        g.loc[x.index, "crowd_cache"] = [a for a, _ in t]
        g.loc[x.index, "n_cache_votes"] = [n for _, n in t]
    w = weights[["label_uid", "w_agree", "w_disagree"]]
    g = g.merge(w, on="label_uid", how="left", validate="one_to_one")
    g["reliability_weighted"] = [reliability_target(a, d) for a, d in zip(g.w_agree, g.w_disagree)]
    g["covered"] = g.city_id.isin(public) & (g.n_cache_votes > 0)
    g["crowd_pool"] = g.crowd_class
    return g


RELIABILITY_MODELS = {"crowd_pool": "crowd majority (pool, 2026-10-27/28)",
                      "crowd_cache": "crowd majority (snapshot)",
                      "reliability_weighted": "reliability-weighted crowd"}


def reliability_rows(g, n_boot=N_BOOT, seed=0):
    """Rows in the agreement table's layout, one per target x (type | all), on the covered items, plus
    ``n_gold_decided``, ``n_covered`` and the paired delta (weighted - unweighted snapshot majority, same items)."""
    rows = []
    for t in ("all",) + tuple(LABEL_TYPES):
        x = g if t == "all" else g[g.label_type == t]
        if not len(x):
            continue
        cov = x[x.covered]
        both = cov[(cov.reliability_weighted != "") & (cov.crowd_cache != "")]
        delta, lo, hi = paired_delta((both.reliability_weighted == both.expert).to_numpy(),
                                     (both.crowd_cache == both.expert).to_numpy(), n_boot,
                                     sl.slice_seed(seed, f"reliability/{t}"))
        for col, name in RELIABILITY_MODELS.items():
            y = cov[cov[col] != ""]
            r = rl._rate(int((y[col] == y.expert).sum()), len(y))
            d = y[y.ai_vote == "Disagree"]
            rows.append({"model": name, "label_type": t, "n": r["n"], "vlm_agree": r["rate"], "lo": r["lo"],
                         "hi": r["hi"], "crowd_n": 0, "crowd_agree": np.nan, "crowd_lo": np.nan, "crowd_hi": np.nan,
                         "ai_disagree_n": int(len(d)),
                         "ai_confirmed": float((d.expert == "incorrect").mean()) if len(d) else np.nan,
                         "auc_vs_expert": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                         "n_gold_decided": int(len(x)), "n_covered": int(len(cov)),
                         "n_paired": int(len(both)), "delta_weighted_minus_unweighted": delta,
                         "delta_lo": lo, "delta_hi": hi})
    return pd.DataFrame(rows)


def reliability_markdown(rows):
    out = [agreement_markdown(rows), "", "| Type | Decided gold items | Covered | Paired | Weighted - unweighted [95%] |",
           "|---|---:|---:|---:|---|"]
    for r in rows[rows.model == RELIABILITY_MODELS["reliability_weighted"]].to_dict("records"):
        d = "" if pd.isna(r["delta_weighted_minus_unweighted"]) else \
            f"{100 * r['delta_weighted_minus_unweighted']:+.1f}" + (
                "" if pd.isna(r["delta_lo"]) else f" [{100 * r['delta_lo']:+.1f}, {100 * r['delta_hi']:+.1f}]")
        out.append(f"| {r['label_type']} | {r['n_gold_decided']} | {r['n_covered']} | {r['n_paired']} | {d} |")
    out += ["", "Covered: a public deployment with at least one human vote in the rawLabels snapshot the weights were "
                "built from; private-deployment items count only as not covered."]
    return "\n".join(out)


def _public(args):
    from analysis import retrain_report as rr
    return rr.public_cities(args.public_cities)[0]


def cmd_reliability(args):
    run_path = os.path.join(os.path.dirname(os.path.abspath(args.vote_weights)), "reliability_run.json")
    try:
        snapshot_check(vv._read_json(run_path), args.rawlabels, args.rated_after)
    except ValueError as e:
        sys.exit(f"refusing: {e}")
    gold = load_gold(args.gold)
    g = reliability_frame(gold, pd.read_csv(args.vote_weights), args.rawlabels, _public(args))
    rows = reliability_rows(g, args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, "reliability.csv"), index=False)
    md = reliability_markdown(rows)
    with open(os.path.join(args.out_dir, "reliability.md"), "w") as f:
        f.write(md + "\n")
    write_json(os.path.join(args.out_dir, "reliability.json"),
               {"vote_weights_sha256": rl.sha256_file(args.vote_weights), "rated_after": args.rated_after,
                "reliability_run": vv._read_json(run_path).get("created")})
    print(md)


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def splice(text, blocks):
    """Replace each generated block in ``text``; everything else stays byte-identical. A missing block raises.

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


def privacy_check(text, all_cities, public):
    """Raise ``ValueError`` when ``text`` names a deployment (``city_id`` or ``sidewalk_<x>`` schema) that is not
    public.

    >>> privacy_check("seattle-wa only", {"seattle-wa": "sidewalk_seattle", "x-y": "sidewalk_x"}, {"seattle-wa"})
    >>> privacy_check("x-y leaked", {"x-y": "sidewalk_x"}, set())
    Traceback (most recent call last):
    ...
    ValueError: private deployment named in the generated text: ['x-y']
    """
    hits = sorted({c for c, schema in all_cities.items() if c not in public
                   and (re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", text)
                        or (schema and re.search(rf"\b{re.escape(schema)}\b", text)))})
    if hits:
        raise ValueError(f"private deployment named in the generated text: {hits}")


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def variants_markdown(split_root, types=LABEL_TYPES):
    lines = ["| Type | Variant | Contested | Scored | Changed | Kept (train) | Mean soft target |",
             "|---|---|---:|---:|---:|---:|---:|"]
    n = 0
    for t in types:
        for v in VARIANTS:
            p = os.path.join(split_root, t, "cl", v, "variant.json") if split_root else ""
            if p and os.path.exists(p):
                j = json.load(open(p))
                mean = "" if "soft_target_mean" not in j else f"{j['soft_target_mean']:.3f}"
                lines.append(f"| {t} | {v} | {j['rows_contested']} | {j['rows_scored']} | {j['changed']} | "
                             f"{j['kept']} | {mean} |")
                n += 1
    return "\n".join(lines) if n else "_No variants yet: `relabel` and `variants` run only for pairs that clear the bar._"


def provenance(agree_dir, scores_dir, gold_summary, split_root):
    """Every pin the doc lists: the gold table and its inputs, each score file and its model revision, prompt
    hashes and job ids, and the variant splits."""
    out = {"gold_sha256": gold_summary.get("gold_sha256"), "gold_inputs": gold_summary.get("inputs", {}),
           "round3_splits": gold_summary.get("round3_splits", {}), "prompt_version": PROMPT_VERSION,
           "prompt_template_sha256": prompt_template_hash(False),
           "prompt_template_sha256_with_tags": prompt_template_hash(True), "scores": {}, "variants": {}}
    if scores_dir and os.path.isdir(scores_dir):
        for model in sorted(os.listdir(scores_dir)):
            for f in sorted(os.listdir(os.path.join(scores_dir, model))) if os.path.isdir(
                    os.path.join(scores_dir, model)) else []:
                if f.endswith(".csv"):
                    p = os.path.join(scores_dir, model, f)
                    meta = vv._read_json(p[:-4] + ".meta.json")
                    out["scores"][f"{model}/{f}"] = {
                        "sha256": rl.sha256_file(p), "repo": meta.get("repo"), "revision": meta.get("revision"),
                        "revision_requested": meta.get("revision_requested"), "slurm_job_id": meta.get("slurm_job_id"),
                        "vllm_version": meta.get("vllm_version"), "image_tag": meta.get("image_tag"),
                        "serve_args": meta.get("serve_args"), "s_per_crop": meta.get("s_per_crop"),
                        "prompt_template_sha256": meta.get("prompt_template_sha256")}
    for name in ("bar.json", "reliability.json"):
        p = os.path.join(agree_dir, name)
        if os.path.exists(p):
            out[name] = rl.sha256_file(p)
    if split_root:
        for t in LABEL_TYPES:
            for v in VARIANTS:
                p = os.path.join(split_root, t, "cl", v, "split.csv")
                if os.path.exists(p):
                    out["variants"][f"{t}/{v}"] = rl.sha256_file(p)
    return out


def provenance_markdown(p):
    lines = ["| Item | Pin |", "|---|---|", f"| gold.csv | `{p['gold_sha256']}` |",
             f"| prompt template ({p['prompt_version']}, no tags) | `{p['prompt_template_sha256']}` |",
             f"| prompt template (with tags) | `{p['prompt_template_sha256_with_tags']}` |"]
    lines += [f"| input {k} | `{v}` |" for k, v in sorted(p["gold_inputs"].items())]
    lines += [f"| round-3 split {k} | `{v}` |" for k, v in sorted(p["round3_splits"].items())]
    for k, v in sorted(p["scores"].items()):
        lines.append(f"| scores {k} | `{v['sha256']}`; {v['repo']} @ `{v['revision'] or v['revision_requested']}`; "
                     f"job {v['slurm_job_id']}; vLLM {v['vllm_version']} |")
    lines += [f"| variant {k} | `{v}` |" for k, v in sorted(p["variants"].items())]
    return "\n".join(lines)


def report_blocks(agree_dir, gold_summary, scores_dir=None, split_root=None):
    """``{block name: Markdown}`` for the doc, from the ``agree`` / ``reliability`` outputs."""
    blocks = {"gold_counts": gold_markdown(gold_summary) + f"\n\n`gold.csv` sha256 `{gold_summary.get('gold_sha256')}`."}
    a, p, s = (_csv(os.path.join(agree_dir, f)) for f in ("agreement.csv", "paired.csv", "strata.csv"))
    pending = "_Not yet run._"
    blocks["agreement"] = agreement_markdown(a) if a is not None else pending
    blocks["paired"] = paired_markdown(p) if p is not None else pending
    blocks["strata"] = strata_markdown(s) if s is not None else pending
    sv, tg = _csv(os.path.join(agree_dir, "slices_severity.csv")), _csv(os.path.join(agree_dir, "slices_tag.csv"))
    blocks["slices"] = slices_markdown({"severity": sv, "tag": tg}) if sv is not None and tg is not None else pending
    r = _csv(os.path.join(agree_dir, "reliability.csv"))
    blocks["reliability"] = reliability_markdown(r) if r is not None else pending
    bp = os.path.join(agree_dir, "bar.json")
    blocks["bar"] = bar_markdown(json.load(open(bp))["decisions"]) if os.path.exists(bp) else pending
    blocks["variants"] = variants_markdown(split_root)
    prov = provenance(agree_dir, scores_dir, gold_summary, split_root)
    blocks["provenance"] = provenance_markdown(prov)
    return blocks, prov


def cmd_report(args):
    summary = json.load(open(args.gold_summary))
    blocks, prov = report_blocks(args.agree_dir, summary, args.scores, args.variants_dir)
    all_cities = {c: s for s, c in schema_city_map(open(args.cityparams).read()).items()}
    public = _public(args)
    text = "\n\n".join(blocks.values()) + json.dumps(prov)
    try:
        privacy_check(text, all_cities, public)
    except ValueError as e:
        sys.exit(f"refusing to write the report: {e}")
    os.makedirs(args.out, exist_ok=True)
    for name, body in blocks.items():
        if name != "provenance":
            with open(os.path.join(args.out, f"{name}.md"), "w") as f:
                f.write(body + "\n")
    write_json(os.path.join(args.out, "provenance.json"), prov)
    if args.doc:
        doc = open(args.doc).read()
        with open(args.doc, "w") as f:
            f.write(splice(doc, blocks))
    print(f"report -> {args.out}" + (f"; spliced into {args.doc}" if args.doc else ""))


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def _server_args(s, concurrency=16):
    s.add_argument("--server", default="http://127.0.0.1:8000")
    s.add_argument("--model", required=True, help="short name (the scores/<model>/ directory)")
    s.add_argument("--repo", required=True, help="HF repo id (also the served name by default)")
    s.add_argument("--revision", default=None, help="the revision the server was started with (recorded)")
    s.add_argument("--served-name", default=None)
    s.add_argument("--concurrency", type=int, default=concurrency)
    s.add_argument("--gpu", required=True, help='label, e.g. "2xL40S"')
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--image-tag", default=None)
    s.add_argument("--prefill", default=None, help="assistant prefill continued by the server, e.g. '<think></think>'")
    s.add_argument("--template-kwargs", type=json.loads, default=None, help="JSON chat_template_kwargs")
    s.add_argument("--thinking-note", default=None)
    s.add_argument("--hf-home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    s.add_argument("--show-tags", action="store_true", help="the with-tags prompt variant")
    s.add_argument("--crops-root", default=None, help="the crops leg's output (manifest.csv + crops/)")
    s.add_argument("--split-root", default=None, help="retrain-2026-10 root with <Type>/split.csv and crops")
    s.add_argument("--out-dir", required=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold", help="the unified gold table of rounds 1 and 2")
    g.add_argument("--r1-joined", required=True, help="round 1 packet/joined_scored.csv")
    g.add_argument("--r1-rater", default="jon")
    g.add_argument("--r2-key", required=True, help="round 2 private key.csv (both id forms)")
    g.add_argument("--r2-sheets", nargs="+", required=True, metavar="RATER=SHEET")
    g.add_argument("--r2-gold-shared", required=True)
    g.add_argument("--r2-unmasked", default=None)
    g.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    g.add_argument("--split-root", required=True, help="dir with <Type>/split.csv (round 3)")
    g.add_argument("--cl-dir", default=None, help="dir with <Type>/cl/noisy_targets.csv (default: --split-root)")
    g.add_argument("--pool", default=None, help="relabel_pool.csv: refresh the AI vote and tallies")
    g.add_argument("--out-dir", default=DEFAULT_OUT)
    g.set_defaults(func=cmd_gold)

    c = sub.add_parser("crops", help="labels.csv and stage_files.txt for the klone crop leg")
    c.add_argument("--gold", required=True, help="gold.csv, or stage-list's pool_items.csv")
    c.add_argument("--rawlabels", required=True, help="rawLabels cache dir (<city_id>.csv)")
    c.add_argument("--label-points", default=None, help="pull_label_points.sh output for the rest")
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_crops)
    sg = sub.add_parser("stage-list", help="panoramas and GB a pool run would stage")
    sg.add_argument("--pool", required=True)
    sg.add_argument("--types", required=True)
    sg.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    sg.add_argument("--out-dir", default=None, help="also write pool_items.csv here (outside the repo)")
    sg.set_defaults(func=cmd_stage_list)

    pr = sub.add_parser("prompt", help="print the rendered prompts and the template hash")
    pr.add_argument("--show-tags", action="store_true")
    pr.set_defaults(func=cmd_prompt)

    s = sub.add_parser("score", help="the gold set through a running vLLM server (resumable)")
    _server_args(s)
    s.add_argument("--gold", required=True)
    s.add_argument("--no-reason", action="store_true", help="skip the one-sentence reason request")
    s.add_argument("--limit", type=int, default=0, help="smoke: first N items per type")
    s.set_defaults(func=cmd_score)

    a = sub.add_parser("agree", help="agreement with the expert, the paired bar, strata and slices")
    a.add_argument("--gold", required=True)
    a.add_argument("--scores", required=True, help="the score stage's --out-dir")
    a.add_argument("--models", required=True, help="comma-separated short names")
    a.add_argument("--split-root", default=None, help="for the training counts per slice")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--n-boot", type=int, default=N_BOOT)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_agree)

    d = sub.add_parser("disagreements", help="a masked reading packet of model-expert disagreements")
    d.add_argument("--gold", required=True)
    d.add_argument("--scores", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--crops-root", default=None)
    d.add_argument("--split-root", default=None)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--key-dir", required=True)
    d.add_argument("--seed", required=True)
    d.set_defaults(func=cmd_disagreements)

    r = sub.add_parser("relabel", help="a model that cleared the bar on contested train rows or the pool")
    _server_args(r, concurrency=32)
    r.add_argument("--bar", required=True)
    r.add_argument("--rows", choices=["train", "pool"], default="train")
    r.add_argument("--types", required=True)
    r.add_argument("--pool", required=True)
    r.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    r.add_argument("--force", action="store_true", help="run a pair that does not clear the bar (recorded)")
    r.add_argument("--reason", dest="no_reason", action="store_false", help="also ask for reasons (10-15x cost)")
    r.set_defaults(func=cmd_relabel, no_reason=True)

    v = sub.add_parser("variants", help="adj-replace and adj-soft split variants")
    v.add_argument("--split-root", required=True)
    v.add_argument("--relabel", required=True, help="the relabel stage's --out-dir")
    v.add_argument("--model", required=True)
    v.add_argument("--types", required=True)
    v.add_argument("--adjudicator-votes", type=float, default=2.0, help="w: the model counts as w votes in adj-soft")
    v.add_argument("--contested", choices=["ai-disagree", "cl-flagged", "both"], default="both")
    v.add_argument("--no-spare-expert", action="store_true", help="also change expert-sourced targets")
    v.add_argument("--drop-gold", action="store_true", help="gold-set train rows -> train_dropped")
    v.add_argument("--gold", default=None, help="gold.csv, for --drop-gold")
    v.set_defaults(func=cmd_variants)

    rel = sub.add_parser("reliability", help="the reliability-weighted crowd against the expert (#12)")
    rel.add_argument("--gold", required=True)
    rel.add_argument("--vote-weights", required=True, help="validator_reliability.py's vote_weights.csv")
    rel.add_argument("--rawlabels", required=True, help="the rawLabels cache the weights were built from")
    rel.add_argument("--rated-after", default="2026-10-27", help="first rating day; the snapshot must predate it")
    rel.add_argument("--public-cities", default=None, help="saved /v3/api/cities or a list (default: live API)")
    rel.add_argument("--out-dir", required=True)
    rel.add_argument("--n-boot", type=int, default=N_BOOT)
    rel.add_argument("--seed", type=int, default=0)
    rel.set_defaults(func=cmd_reliability)

    rp = sub.add_parser("report", help="the doc's tables and provenance (aggregates only)")
    rp.add_argument("--agree-dir", required=True)
    rp.add_argument("--gold-summary", required=True)
    rp.add_argument("--scores", default=None)
    rp.add_argument("--variants-dir", default=None, help="the split root holding <Type>/cl/adj-*/variant.json")
    rp.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    rp.add_argument("--public-cities", default=None)
    rp.add_argument("--out", default=os.path.join(REPO_DIR, "docs", "adjudicator-2026-11"))
    rp.add_argument("--doc", default=DOC, help="splice the blocks into this doc ('' to skip)")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
