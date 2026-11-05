"""Validation as re-typing (issue #13): data, evaluation and report stages for one 8-class DINOv2 over label types.

The five production validators each answer "is this a correct <type> label?". A re-typing model answers "what is
in this crop?" over eight classes (``CLASSES``), and the binary verdict for a label is the probability of its
claimed type, ``p_correct = p(claimed)``. The same output also names the type a wrong label most likely is
(``suggested_type``). Training lives in ``fine_tune_retype.py``; this module builds the data, scores the model and
writes the aggregates for ``docs/retype-2026-11.md``.

Ground truth (``source`` column of ``retype_split.csv``):

- ``validated``: every row of the round-3 ``<Type>/split.csv`` files (and of the extra types made by
  ``data_generation.py --label_type NoSidewalk|Signal|Other|Occlusion``). A ``correct`` row is a positive of its
  type. An ``incorrect`` row says only "not the claimed type": its ``target_type`` is empty and it trains through
  a complement loss (``fine_tune_retype.retype_loss``).
- ``type_edit``: labels whose type a human changed in Project Sidewalk (``/v3/api/labelEdits``), two rows each
  (claimed = new type, correct; claimed = old type, incorrect with ``target_type`` = new type). Evaluation only
  (split ``test_retype``). On 2026-11-05 the 46 public deployments held 4 such edits.
- ``proxy_type`` (a flag, not a source): an ``incorrect`` validated label with a ``correct`` label of exactly one
  other type within ``--colocate-m`` metres, placed by another user. A proxy for "this label was the wrong type"
  (it may be a neighbouring feature instead), used for evaluation only.

Split: every row keeps its round-3 split, except that all rows on one panorama take the highest-priority split any
of them had (``test_heldout > test > val > train``, ``inherit_splits``). Round 3 split each type on its own, so
without this the same panorama could be train for one type and test for another.

Stages (``python retype.py <stage> -h`` for the options)::

    # login node (network): the labelEdits exports of every public deployment
    python retype.py fetch-edits --out OUT/label_edits
    # the type changes, joined to the current rawLabels rows (the crop geometry)
    python retype.py type-edits --edits OUT/label_edits --rawlabels DATA/rawlabels --out OUT/type_edits.csv
    # the numbers of the doc's "What was measured" section (aggregates only)
    python retype.py audit --rawlabels DATA/rawlabels --splits DATA --edits OUT/label_edits --out OUT/audit.json
    # GPU: crops for the type-edit labels, cut as data_generation.py cuts a training crop
    python retype.py crops --labels OUT/type_edits.csv --pano-store STORE --out OUT/edit_crops
    # the multi-class split
    python retype.py assemble --splits DATA --edits OUT/type_edits.csv --edit-crops OUT/edit_crops \\
        --rawlabels DATA/rawlabels --out OUT/retype_split.csv
    # GPU: scores (8 class probabilities per row)
    python retype.py predict --run-dir RUNS/retype/complement --split OUT/retype_split.csv --crop-root DATA \\
        --out OUT/complement/retype_predictions.csv
    # per-type binary predictions for the unchanged evaluate.py and calibration/calibrate.py
    python retype.py export-binary --predictions OUT/complement/retype_predictions.csv \\
        --split OUT/retype_split.csv --out-root DATA --tag retype-complement
    # aggregates for the doc, next to round 3 on the same rows
    python retype.py report --predictions OUT/complement/retype_predictions.csv --split OUT/retype_split.csv \\
        --root DATA --tag retype-complement --docs-dir OUT/docs-complement

Every file with label ids (``type_edits.csv``, ``retype_split.csv``, predictions) belongs outside the repository;
``audit`` and ``report`` write aggregates only and refuse a label-id-like token or a non-public deployment name.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

import data_generation as dg
import split_data as sd

# ---------------------------------------------------------------------------------------------------------------------
# Classes (the order is a serving contract: sidewalk-ai-api reads config.json id2label by index)
# ---------------------------------------------------------------------------------------------------------------------

CLASSES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk", "NoSidewalk", "Signal", "Other")
ID2LABEL = dict(enumerate(CLASSES))
LABEL2ID = {c: i for i, c in ID2LABEL.items()}
VALIDATOR_TYPES = CLASSES[:5]                    # the five with round-3 crops and a per-type baseline
EXTRA_TYPES = ("NoSidewalk", "Signal", "Other")  # crops via data_generation.py --label_type
OTHER = "Other"                                  # Project Sidewalk's Other and Occlusion labels; the "none of these" sink
TYPE_ALIASES = {"Occlusion": "Other"}            # rawLabels label_type -> class

AI_USER_ID = dg.AI_USER_ID
SPLIT_PRIORITY = ("test_heldout", "test", "val", "train")  # the first wins on a shared panorama
TEST_RETYPE = "test_retype"
EVAL_SPLITS = ("val", "test", "test_heldout")
REPORT_SPLITS = ("test", "test_heldout")
COLOCATE_M = 1.5
MIN_EXTRA_INCORRECT = 10  # an extra type with fewer incorrect rows is left out (its class stays in the head)
AUDIT_RADII_M = (1.5, 3.0)
POOLED_PRIVATE = "other (private deployments)"   # as analysis/retrain_report.POOLED_PRIVATE
GATE = 0.92

SCHEMA = ["row_uid", "label_uid", "city", "label_id", "pano_id", "split", "round3_split", "claimed_type",
          "target_type", "is_correct", "source", "proxy_type", "proxy_b_in_train", "severity", "tags",
          "class_source", "author", "pano_source", "human_agree", "human_disagree", "crop_path", "hard"]
PROB_COLUMNS = [f"p_{c}" for c in CLASSES]
PREDICTION_COLUMNS = (["row_uid", "label_uid", "split", "claimed_type", "target_type", "is_correct", "source",
                       "proxy_type", "proxy_b_in_train", "p_claimed"] + PROB_COLUMNS
                      + ["top1", "top3", "suggested_type", "suggested_top3"])
# Columns that would carry per-label identifiers; no aggregate written by ``report`` or ``audit`` may have one.
ID_COLUMNS = ("row_uid", "label_uid", "label_id", "pano_id", "crop_path", "user_id")
LABEL_ID_TOKEN = re.compile(r"\b[a-z][a-z0-9-]*:\d+\b")


def canonical_type(label_type):
    """The class a rawLabels ``label_type`` belongs to (``TYPE_ALIASES`` applied).

    >>> canonical_type("Occlusion"), canonical_type("CurbRamp")
    ('Other', 'CurbRamp')
    """
    return TYPE_ALIASES.get(label_type, label_type)


def file_sha256(path):
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ---------------------------------------------------------------------------------------------------------------------
# Type edits (/v3/api/labelEdits)
# ---------------------------------------------------------------------------------------------------------------------

def type_changes(edits, ai_user_id=AI_USER_ID):
    """The type changes in one deployment's labelEdits export: one row per ``label_id``, its latest change.

    Drops edits that keep the type (tag and severity edits are most of the export) and edits by the SidewalkAI
    account. ``label_id`` must be the per-city integer: a ``label_uid``-form id (``city:123``) means the frame was
    already keyed with its city, and mixing the two forms is how a cross-city join goes wrong, so it raises.

    Returns columns ``label_id`` (str), ``old_label_type``, ``new_label_type``, ``edit_time``, ``source``.

    >>> e = pd.DataFrame({"label_id": [5, 5, 7, 9], "user_id": ["u", "u", "u", AI_USER_ID],
    ...                   "old_label_type": ["CurbRamp", "NoCurbRamp", "Obstacle", "CurbRamp"],
    ...                   "new_label_type": ["NoCurbRamp", "Obstacle", "Obstacle", "Other"],
    ...                   "edit_time": ["2026-10-01T10:00:00-07:00", "2026-10-02T10:00:00-07:00",
    ...                                 "2026-10-01T10:00:00-07:00", "2026-10-01T10:00:00-07:00"]})
    >>> type_changes(e)[["label_id", "old_label_type", "new_label_type"]].values.tolist()
    [['5', 'NoCurbRamp', 'Obstacle']]
    """
    missing = {"label_id", "old_label_type", "new_label_type", "edit_time"} - set(edits.columns)
    if missing:
        raise ValueError(f"labelEdits export lacks column(s) {sorted(missing)}")
    ids = edits["label_id"].astype(str).str.strip()
    if ids.str.contains(":").any():
        raise ValueError("mismatched city key forms: label_id holds label_uid-form ids (city:label_id); pass one "
                         "deployment's raw export and attach the city with attach_city")
    if not ids.str.fullmatch(r"\d+").all():
        bad = ids[~ids.str.fullmatch(r"\d+")].head(3).tolist()
        raise ValueError(f"label_id must be a per-city integer, got e.g. {bad}")
    df = edits.assign(label_id=ids)
    if "user_id" in df.columns:
        df = df[df["user_id"].astype(str) != ai_user_id]
    df = df[df["old_label_type"].astype(str) != df["new_label_type"].astype(str)].copy()
    df["_t"] = pd.to_datetime(df["edit_time"], utc=True, format="ISO8601")
    df = df.sort_values(["label_id", "_t"], kind="mergesort").drop_duplicates("label_id", keep="last")
    df["source"] = df["source"].astype(str) if "source" in df.columns else ""
    df = df.assign(_n=df["label_id"].astype(int)).sort_values("_n")
    return df[["label_id", "old_label_type", "new_label_type", "edit_time", "source"]].reset_index(drop=True)


def attach_city(changes, city):
    """``changes`` with ``city`` and the composite ``label_uid`` (``city:label_id``) added.

    >>> attach_city(pd.DataFrame({"label_id": ["12"]}), "seattle-wa").label_uid.tolist()
    ['seattle-wa:12']
    """
    out = changes.copy()
    out["city"] = city
    out["label_uid"] = city + ":" + out["label_id"].astype(str)
    return out


def with_label_uid(raw):
    """rawLabels rows (with ``city``) plus ``label_id`` as a stripped string and ``label_uid``; asserts uniqueness."""
    raw = raw.copy()
    raw["label_id"] = raw["label_id"].astype(str).str.strip()
    raw["label_uid"] = raw["city"].astype(str) + ":" + raw["label_id"]
    raw = raw.drop_duplicates()
    assert raw["label_uid"].is_unique, "rawLabels has a label_uid twice"
    return raw


def join_current_labels(changes, rawlabels):
    """Type changes joined to each label's current rawLabels row, on ``label_uid``: ``(kept, dropped_counts)``.

    A change is kept when the label still exists, its current ``label_type`` equals ``new_label_type`` (else a
    later edit or a stale export disagrees), both types map to known and different classes, and its pano point is
    usable (``data_generation.valid_geometry``). ``dropped_counts`` counts the rest by reason.

    >>> ch = attach_city(pd.DataFrame({"label_id": ["1", "2"], "old_label_type": ["CurbRamp", "Obstacle"],
    ...                                "new_label_type": ["NoCurbRamp", "Other"]}), "c")
    >>> raw = pd.DataFrame({"city": ["c", "c"], "label_id": [1, 2], "label_type": ["NoCurbRamp", "Obstacle"],
    ...     "pano_source": ["gsv"] * 2, "pano_x": [1] * 2, "pano_y": [1] * 2, "pano_width": [8] * 2,
    ...     "pano_height": [4] * 2})
    >>> kept, why = join_current_labels(ch, raw)
    >>> kept.label_uid.tolist(), why
    (['c:1'], {'type_mismatch': 1})
    """
    raw = with_label_uid(rawlabels)
    keep_raw = [c for c in raw.columns if c not in ("old_label_type", "new_label_type", "edit_time", "source")]
    j = changes.merge(raw[keep_raw].drop(columns=["city", "label_id"]), on="label_uid", how="left",
                      validate="one_to_one", indicator=True)
    why = Counter()
    found = j["_merge"] == "both"
    why["not_found"] += int((~found).sum())
    j = j[found].drop(columns="_merge")
    same = j["label_type"].astype(str) == j["new_label_type"].astype(str)
    why["type_mismatch"] += int((~same).sum())
    j = j[same]
    old_c, new_c = j["old_label_type"].map(canonical_type), j["new_label_type"].map(canonical_type)
    known = old_c.isin(CLASSES) & new_c.isin(CLASSES)
    why["unknown_type"] += int((~known).sum())
    j = j[known]
    differ = j["old_label_type"].map(canonical_type) != j["new_label_type"].map(canonical_type)
    why["same_class_after_alias"] += int((~differ).sum())
    j = j[differ]
    ok, geo = dg.valid_geometry(j)
    why.update(geo)
    kept = j[ok].reset_index(drop=True)
    return kept, {k: v for k, v in why.items() if v}


def edit_rows(kept, crops):
    """Two ``retype_split.csv`` rows per type-edit label (``SCHEMA`` columns), split ``test_retype``: claimed = new
    type (correct) and claimed = old type (incorrect, ``target_type`` = new type). ``crops`` is the ``crops``
    stage's manifest (``label_uid, crop_path`` absolute, ``status``); labels without an ok crop are left out.

    >>> k = pd.DataFrame({"label_uid": ["c:1"], "city": ["c"], "label_id": ["1"], "pano_id": ["p"],
    ...                   "old_label_type": ["CurbRamp"], "new_label_type": ["NoCurbRamp"]})
    >>> r = edit_rows(k, pd.DataFrame({"label_uid": ["c:1"], "crop_path": ["/x.webp"], "status": ["ok"]}))
    >>> r[["row_uid", "claimed_type", "target_type", "is_correct"]].values.tolist()
    [['c:1@NoCurbRamp', 'NoCurbRamp', 'NoCurbRamp', 1], ['c:1@CurbRamp', 'CurbRamp', 'NoCurbRamp', 0]]
    """
    ok = crops[crops["status"].astype(str) == "ok"][["label_uid", "crop_path"]]
    k = kept.merge(ok, on="label_uid", how="inner", validate="one_to_one")
    rows = []
    for r in k.to_dict("records"):
        new, old = canonical_type(r["new_label_type"]), canonical_type(r["old_label_type"])
        base = {c: r.get(c, "") for c in SCHEMA}
        base.update(label_uid=r["label_uid"], city=r["city"], label_id=str(r["label_id"]), pano_id=r.get("pano_id", ""),
                    split=TEST_RETYPE, round3_split="", source="type_edit", proxy_type="", proxy_b_in_train="",
                    hard="0",
                    class_source="type_edit", author=dg.label_author(r.get("user_id", "")),
                    crop_path=r["crop_path"], human_agree="", human_disagree="")
        for claimed, correct in ((new, 1), (old, 0)):
            rows.append({**base, "row_uid": f"{r['label_uid']}@{claimed}", "claimed_type": claimed,
                         "target_type": new, "is_correct": correct})
    return pd.DataFrame(rows, columns=SCHEMA).fillna("")


# ---------------------------------------------------------------------------------------------------------------------
# Validated rows, split inheritance, colocation
# ---------------------------------------------------------------------------------------------------------------------

def read_split(path):
    """A ``split.csv`` as strings (``keep_default_na=False``, so an empty pano_id stays "")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validated_rows(split_df, type_dir):
    """``SCHEMA`` rows from one round-3 ``<type_dir>/split.csv``: ``status == ok`` rows only, ``crop_path`` made
    relative to the data root (``<type_dir>/<crop_path>``), ``target_type`` empty for incorrect rows.

    >>> s = pd.DataFrame({"label_uid": ["c:1", "c:2"], "city": ["c", "c"], "label_id": ["1", "2"],
    ...     "pano_id": ["p", "q"], "label_type": ["Occlusion"] * 2, "label_class": ["correct", "incorrect"],
    ...     "crop_path": ["correct/c_1.webp", "incorrect/c_2.webp"], "split": ["train", "test"], "status": ["ok"] * 2})
    >>> validated_rows(s, "Occlusion")[["row_uid", "target_type", "is_correct", "crop_path"]].values.tolist()
    [['c:1@Other', 'Other', 1, 'Occlusion/correct/c_1.webp'], ['c:2@Other', '', 0, 'Occlusion/incorrect/c_2.webp']]
    """
    df = split_df[split_df["status"] == "ok"] if "status" in split_df.columns else split_df
    if not set(df["label_class"]) <= {"correct", "incorrect"}:
        raise ValueError(f"{type_dir}: label_class must be correct/incorrect")
    claimed = df["label_type"].map(canonical_type)
    if not claimed.isin(CLASSES).all():
        raise ValueError(f"{type_dir}: label_type {sorted(set(df['label_type'][~claimed.isin(CLASSES)]))} is not "
                         f"one of {CLASSES}")
    out = pd.DataFrame({c: df[c] if c in df.columns else "" for c in SCHEMA})
    out["claimed_type"] = claimed.to_numpy()
    out["is_correct"] = (df["label_class"] == "correct").astype(int).to_numpy()
    out["target_type"] = np.where(out["is_correct"] == 1, out["claimed_type"], "")
    out["row_uid"] = out["label_uid"] + "@" + out["claimed_type"]
    out["round3_split"] = df["split"].to_numpy()
    out["source"] = "validated"
    out["proxy_type"] = ""
    out["proxy_b_in_train"] = ""
    out["crop_path"] = [p if os.path.isabs(p) else f"{type_dir}/{p}" for p in df["crop_path"]]
    if "hard" not in df.columns:
        out["hard"] = "0"
    return out.reset_index(drop=True)


def pano_groups(df):
    """One group key per row: its ``pano_id``, or ``uid:<label_uid>`` when it has none (as split_data.py does)."""
    pano = df["pano_id"].astype(str)
    return pano.where(pano.str.len() > 0, "uid:" + df["label_uid"].astype(str))


def inherit_splits(df, groups=None):
    """Give every row of a group the highest-priority split any of its rows had (``SPLIT_PRIORITY``):
    ``(df, moves)``.

    ``groups`` defaults to ``pano_groups(df)``; pass merged groups (``split_data.merge_nearby_groups``) for the 3 m
    pass. Every ``test``/``test_heldout`` row keeps its split, so the comparison with the five round-3 models runs on
    exactly round 3's test rows; the price is train (and val) rows of other types that move up. ``moves`` counts
    them: columns ``type, from, to, n``.

    >>> d = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "pano_id": ["p", "p", "q"],
    ...                   "claimed_type": ["CurbRamp", "Obstacle", "Obstacle"], "split": ["test", "train", "train"]})
    >>> out, moves = inherit_splits(d)
    >>> out.split.tolist(), moves.values.tolist()
    (['test', 'test', 'train'], [['Obstacle', 'train', 'test', 1]])
    """
    rank = {s: i for i, s in enumerate(SPLIT_PRIORITY)}
    unknown = sorted(set(df["split"]) - set(rank))
    if unknown:
        raise ValueError(f"inherit_splits: unknown split(s) {unknown}")
    groups = pano_groups(df) if groups is None else pd.Series(np.asarray(groups), index=df.index)
    best = df["split"].map(rank).groupby(groups).transform("min")
    new = best.map(dict(enumerate(SPLIT_PRIORITY)))
    moved = new != df["split"]
    moves = (pd.DataFrame({"type": df.loc[moved, "claimed_type"], "from": df.loc[moved, "split"],
                           "to": new[moved]})
             .groupby(["type", "from", "to"]).size().reset_index(name="n"))
    out = df.copy()
    out["split"] = new.to_numpy()
    return out, moves


def crowd_classes(raw):
    """``correct``/``incorrect``/``""`` per rawLabels row by the crowd margin rule of
    ``data_generation.eligible_labels`` (human votes only: net Agree > 2, net Disagree > 1).

    >>> v = lambda a, d: json.dumps([{"validation": "Agree", "validator_type": "Human"}] * a
    ...                             + [{"validation": "Disagree", "validator_type": "Human"}] * d)
    >>> crowd_classes(pd.DataFrame({"validations": [v(3, 0), v(0, 2), v(1, 1)]})).tolist()
    ['correct', 'incorrect', '']
    """
    counts = raw["validations"].apply(dg.human_vote_counts)
    net = np.array([a - d for a, d in counts]) if len(counts) else np.array([], dtype=int)
    return pd.Series(np.where(net > 2, "correct", np.where(-net > 1, "incorrect", "")), index=raw.index)


def colocated_targets(incorrect, correct, radius_m=COLOCATE_M):
    """The proxy true type of each incorrect label: the one other type with a correct label within ``radius_m``.

    ``incorrect``: ``label_uid, city, claimed_type, user_id, latitude, longitude``. ``correct``: ``city,
    label_type`` (a class), ``user_id, latitude, longitude``. A correct label counts when its type differs from the
    claimed type and another user placed it; when exactly one type qualifies it is the proxy, otherwise (none, or
    two or more types: a tie) the proxy is ``""``. Distances are ``split_data.haversine_m``. Returns a Series of
    proxies indexed by ``label_uid``.

    >>> inc = pd.DataFrame({"label_uid": ["c:1"], "city": ["c"], "claimed_type": ["NoCurbRamp"], "user_id": ["u"],
    ...                     "latitude": [47.0], "longitude": [-122.0]})
    >>> cor = pd.DataFrame({"city": ["c"], "label_type": ["CurbRamp"], "user_id": ["v"],
    ...                     "latitude": [47.0 + 1 / 111195], "longitude": [-122.0]})
    >>> colocated_targets(inc, cor).to_dict()
    {'c:1': 'CurbRamp'}
    """
    out = pd.Series("", index=incorrect["label_uid"].to_numpy(), dtype=object)
    p = _colocated_pairs(incorrect, correct, radius_m)
    types = p.groupby("uid")["type"].agg(lambda s: sorted(set(s)))
    one = types[types.map(len) == 1]
    out.loc[one.index] = one.map(lambda s: s[0])
    return out


def _colocated_pairs(incorrect, correct, radius_m):
    """Every (incorrect label, qualifying correct label) pair within ``radius_m``: columns ``uid`` (the incorrect
    label), ``type`` and ``pano`` (the correct label's; ``""`` when ``correct`` has no ``pano_id``)."""
    parts = []
    for city, q in incorrect.groupby("city"):
        t = correct[correct["city"] == city]
        q = q[np.isfinite(q["latitude"].astype(float)) & np.isfinite(q["longitude"].astype(float))]
        t = t[np.isfinite(t["latitude"].astype(float)) & np.isfinite(t["longitude"].astype(float))]
        if q.empty or t.empty:
            continue
        i, j = sd.pairs_within(q["latitude"], q["longitude"], t["latitude"], t["longitude"], radius_m)
        if not len(i):
            continue
        tpano = t["pano_id"].fillna("").astype(str).to_numpy() if "pano_id" in t.columns else np.full(len(t), "")
        p = pd.DataFrame({"uid": q["label_uid"].to_numpy()[i], "claimed": q["claimed_type"].to_numpy()[i],
                          "quser": q["user_id"].astype(str).to_numpy()[i],
                          "type": t["label_type"].to_numpy()[j], "tuser": t["user_id"].astype(str).to_numpy()[j],
                          "pano": tpano[j]})
        parts.append(p[(p["type"] != p["claimed"]) & (p["quser"] != p["tuser"])])
    if not parts:
        return pd.DataFrame(columns=["uid", "claimed", "quser", "type", "tuser", "pano"])
    return pd.concat(parts, ignore_index=True)


def colocated_partner_panos(incorrect, correct, radius_m=COLOCATE_M):
    """For each incorrect label with a proxy (``colocated_targets``): the panoramas of the correct labels that made
    it, ``{label_uid: set of pano_id}``. ``assemble`` uses them to flag a proxy whose partner B sits on a train
    panorama, where "suggested = B" could be a memorised location rather than a recognised type.

    >>> inc = pd.DataFrame({"label_uid": ["c:1"], "city": ["c"], "claimed_type": ["NoCurbRamp"], "user_id": ["u"],
    ...                     "latitude": [47.0], "longitude": [-122.0]})
    >>> cor = pd.DataFrame({"city": ["c", "c"], "label_type": ["CurbRamp"] * 2, "user_id": ["v", "w"],
    ...                     "latitude": [47.0 + 1 / 111195] * 2, "longitude": [-122.0] * 2, "pano_id": ["pa", "pb"]})
    >>> {k: sorted(v) for k, v in colocated_partner_panos(inc, cor).items()}
    {'c:1': ['pa', 'pb']}
    """
    proxy = colocated_targets(incorrect, correct, radius_m)
    proxy = proxy[proxy != ""]
    p = _colocated_pairs(incorrect, correct, radius_m)
    p = p[p["uid"].isin(proxy.index)]
    p = p[p["type"] == p["uid"].map(proxy)]
    return {u: set(g["pano"]) - {""} for u, g in p.groupby("uid")}


def load_rawlabels(rawlabels_dir, cities, usecols=None):
    """The cached rawLabels exports ``<dir>/<city>.csv`` of ``cities``, one frame with ``city`` and ``label_uid``.
    A city without an export raises."""
    parts = []
    for city in sorted(set(cities)):
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no rawLabels export for {city} at {path}")
        df = pd.read_csv(path, usecols=usecols, low_memory=False, dtype={"label_id": str, "pano_id": str,
                                                                         "user_id": str})
        df["city"] = city
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=(usecols or []) + ["city", "label_uid"])
    return with_label_uid(pd.concat(parts, ignore_index=True))


def colocation_inputs(raw):
    """``(incorrect, correct)`` frames for ``colocated_targets`` from rawLabels rows: crowd-margin classes, label
    types mapped to ``CLASSES`` (other types dropped)."""
    r = raw.copy()
    r["label_type"] = r["label_type"].map(canonical_type)
    r = r[r["label_type"].isin(CLASSES)]
    r["crowd"] = crowd_classes(r)
    r["latitude"] = pd.to_numeric(r["latitude"], errors="coerce")
    r["longitude"] = pd.to_numeric(r["longitude"], errors="coerce")
    inc = r[r["crowd"] == "incorrect"].rename(columns={"label_type": "claimed_type"})
    return inc, r[r["crowd"] == "correct"]


# ---------------------------------------------------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------------------------------------------------

def class_weights(counts, power=0.5, clip=(0.5, 4.0)):
    """Per-class loss weights for the positive rows: ``(mean / count) ** power``, clipped, then renormalised so the
    weighted mean over positive rows is 1. Classes with no rows get 1.0 (they never occur). ``power = 0`` turns
    weighting off.

    >>> w = class_weights({"CurbRamp": 100, "Crosswalk": 25})
    >>> round(w["Crosswalk"] / w["CurbRamp"], 6), round((100 * w["CurbRamp"] + 25 * w["Crosswalk"]) / 125, 6)
    (2.0, 1.0)
    >>> set(class_weights({"CurbRamp": 100, "Crosswalk": 25}, power=0).values())
    {1.0}
    """
    n = {c: int(counts.get(c, 0)) for c in CLASSES}
    present = [c for c in CLASSES if n[c] > 0]
    if power == 0 or not present:
        return {c: 1.0 for c in CLASSES}
    mean = float(np.mean([n[c] for c in present]))
    w = {c: float(np.clip((mean / n[c]) ** power, clip[0], clip[1])) for c in present}
    norm = sum(w[c] * n[c] for c in present) / sum(n[c] for c in present)
    return {c: (w[c] / norm if c in w else 1.0) for c in CLASSES}


def cap_per_class(df, cap, seed=dg.SEED):
    """``df`` with each class's ``train`` positives cut to ``cap`` rows (the lowest
    ``data_generation.label_rank(seed, row_uid)``, so the kept set is stable when rows join or leave the pool), and
    the number dropped per class. ``cap`` None or 0 keeps everything.

    >>> d = pd.DataFrame({"row_uid": [f"c:{i}@CurbRamp" for i in range(5)], "split": ["train"] * 5,
    ...                   "is_correct": [1] * 5, "target_type": ["CurbRamp"] * 5})
    >>> kept, dropped = cap_per_class(d, 3)
    >>> len(kept), dropped
    (3, {'CurbRamp': 2})
    """
    if not cap:
        return df, {}
    pos = df[(df["split"] == "train") & (df["is_correct"].astype(int) == 1)]
    drop, dropped = [], {}
    for cls, g in pos.groupby("target_type"):
        if len(g) <= cap:
            continue
        ranks = g["row_uid"].map(lambda u: dg.label_rank(seed, u))
        out = ranks.sort_values(kind="mergesort").index[cap:]
        drop.extend(out)
        dropped[cls] = len(out)
    return df.drop(index=drop), dropped


# ---------------------------------------------------------------------------------------------------------------------
# Probabilities -> verdicts and suggestions
# ---------------------------------------------------------------------------------------------------------------------

def claimed_index(claimed):
    """``LABEL2ID`` of each claimed type (raises on an unknown one)."""
    idx = pd.Series(claimed).map(LABEL2ID)
    if idx.isna().any():
        raise ValueError(f"unknown claimed type(s): {sorted(set(pd.Series(claimed)[idx.isna()]))}")
    return idx.astype(int).to_numpy()


def p_claimed_from_probs(probs, claimed_idx):
    """The probability of each row's claimed class: the binary ``p_correct``.

    >>> p_claimed_from_probs(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]), [0, 0]).tolist()
    [0.7, 0.1]
    """
    probs = np.asarray(probs, float)
    return probs[np.arange(len(probs)), np.asarray(claimed_idx, int)]


def suggested_types(probs, claimed_idx, k=1):
    """The ``k`` most probable classes other than the claimed one, per row, as lists of names.

    >>> suggested_types(np.eye(8)[[0, 7]] * 0.5 + 0.0625 * np.arange(8)[::-1] / 7, [0, 1], k=2)
    [['NoCurbRamp', 'Obstacle'], ['Other', 'CurbRamp']]
    """
    p = np.array(probs, float, copy=True)
    p[np.arange(len(p)), np.asarray(claimed_idx, int)] = -np.inf
    order = np.argsort(-p, axis=1, kind="stable")[:, :k]
    return [[CLASSES[i] for i in row] for row in order]


def top_k(probs, k):
    """The ``k`` most probable classes per row (all eight compete), as lists of names."""
    order = np.argsort(-np.asarray(probs, float), axis=1, kind="stable")[:, :k]
    return [[CLASSES[i] for i in row] for row in order]


def api_fields(type_probabilities, claimed):
    """The response fields a retype model adds in sidewalk-ai-api (the contract in ``docs/retype-2026-11.md``), from
    ``{class: p}`` over ``CLASSES`` and the claimed type: the binary verdict by the existing argmax rule
    (``calibration._isotonic_map.predict_verdicts``: incorrect iff ``p_incorrect > p_correct``, score = the max), and
    ``suggested_label_type`` = the most probable class other than the claimed one and ``Other``, ``None`` when
    ``Other`` beats it or the verdict is correct.

    >>> p = dict(zip(CLASSES, [0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.02, 0.04]))
    >>> f = api_fields(p, "CurbRamp")
    >>> f["validation_result"], round(f["validation_score"], 2), f["suggested_label_type"]
    ('incorrect', 0.9, 'NoCurbRamp')
    >>> api_fields(dict(p, Other=0.75, NoCurbRamp=0.0), "CurbRamp")["suggested_label_type"] is None
    True
    """
    if set(type_probabilities) != set(CLASSES):
        raise ValueError(f"type_probabilities must have exactly the keys {CLASSES}")
    p_correct = float(type_probabilities[claimed])
    p_incorrect = 1.0 - p_correct
    result = "incorrect" if p_incorrect > p_correct else "correct"
    out = {"type_probabilities": {c: float(type_probabilities[c]) for c in CLASSES},
           "validation_result": result, "validation_score": max(p_correct, p_incorrect),
           "suggested_label_type": None, "suggested_label_type_confidence": None}
    if result == "correct":
        return out
    rest = {c: p for c, p in type_probabilities.items() if c != claimed}
    best = max(rest, key=lambda c: (rest[c], -LABEL2ID[c]))
    if best != OTHER:
        out["suggested_label_type"], out["suggested_label_type_confidence"] = best, float(rest[best])
    return out


def predictions_frame(rows, probs):
    """``PREDICTION_COLUMNS`` for scored ``retype_split.csv`` rows and their ``n x 8`` softmax ``probs``."""
    probs = np.asarray(probs, float).reshape(len(rows), len(CLASSES))
    ci = claimed_index(rows["claimed_type"])
    out = pd.DataFrame({c: rows[c].to_numpy() for c in ("row_uid", "label_uid", "split", "claimed_type",
                                                         "target_type", "is_correct", "source", "proxy_type")})
    out["proxy_b_in_train"] = rows["proxy_b_in_train"].to_numpy() if "proxy_b_in_train" in rows.columns else ""
    out["p_claimed"] = p_claimed_from_probs(probs, ci)
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i]
    out["top1"] = [r[0] for r in top_k(probs, 1)]
    out["top3"] = ["|".join(r) for r in top_k(probs, 3)]
    sug = suggested_types(probs, ci, 3)
    out["suggested_type"] = [r[0] for r in sug]
    out["suggested_top3"] = ["|".join(r) for r in sug]
    return out[PREDICTION_COLUMNS]


# ---------------------------------------------------------------------------------------------------------------------
# Binary export and comparison with round 3
# ---------------------------------------------------------------------------------------------------------------------

def binary_rows(pred, split_df, label_type):
    """One type's rows of the retype predictions as an evaluate.py predictions frame (``evaluate.PREDICTION_COLUMNS``
    plus nothing else): validated rows claiming ``label_type`` whose split is round 3's own and one of val, test,
    test_heldout. Rows that ``inherit_splits`` moved are left out: round 3 trained on them."""
    meta = split_df[["row_uid", "round3_split"]]
    d = pred.merge(meta, on="row_uid", how="left", validate="one_to_one")
    keep = ((d["source"] == "validated") & (d["claimed_type"] == label_type) & (d["split"] == d["round3_split"])
            & d["split"].isin(EVAL_SPLITS))
    d = d[keep]
    return pd.DataFrame({"label_uid": d["label_uid"].to_numpy(), "split": d["split"].to_numpy(),
                         "true": np.where(d["is_correct"].astype(int) == 1, "correct", "incorrect"),
                         "p_correct": d["p_claimed"].astype(float).to_numpy(),
                         "p_incorrect": 1.0 - d["p_claimed"].astype(float).to_numpy()})


def export_binary(pred, split_df, out_root, tag, types=VALIDATOR_TYPES):
    """Write ``<out_root>/<T>/predictions_<tag>.csv`` for each type (``binary_rows``), so that the unchanged
    ``evaluate.py --label_type T --skip-inference --tag <tag>`` and ``calibration/calibrate.py`` score the re-typing
    verdict. Returns ``{type: {split: n}}``. Never writes the trained model's own ``predictions.csv``."""
    if not tag or not tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--tag must be non-empty letters, digits, '-' or '_' (evaluate.py's rule)")
    counts = {}
    for t in types:
        b = binary_rows(pred, split_df, t)
        os.makedirs(os.path.join(out_root, t), exist_ok=True)
        b.to_csv(os.path.join(out_root, t, f"predictions_{tag}.csv"), index=False, lineterminator="\n")
        counts[t] = b["split"].value_counts().to_dict()
    return counts


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def paired_auc_table(retype, round3, meta, n_boot=2000, seed=0, splits=REPORT_SPLITS, within_min_n=10,
                     levels=(0.92, 0.95)):
    """Per type x split: the retype verdict against round 3's model on the same rows.

    ``retype`` and ``round3``: ``{type: predictions frame}`` (``label_uid, split, true, p_correct, p_incorrect``);
    ``meta``: ``{type: frame with label_uid, city, pano_id}``. Raises ValueError when, in a compared split, the two
    frames do not hold exactly the same ``label_uid``s or disagree on ``true``: a comparison on different rows would
    look paired and not be. Columns: ``type, split, n, n_incorrect, auc_retype, auc_round3, delta, delta_lo,
    delta_hi`` (paired panorama bootstrap, ``analysis.retrain_report.cluster_boot``), within-city AUC of both
    (``retrain_report.within_city_auc``, ``within_min_n``), and recall at precision ``levels`` for each class."""
    from analysis.retrain_report import cluster_boot, within_city_auc
    from evaluate import recall_at_precision
    rows = []
    for t in retype:
        for split in splits:
            a = retype[t][retype[t]["split"] == split]
            b = round3[t][round3[t]["split"] == split]
            if a.empty and b.empty:
                continue
            if set(a["label_uid"]) != set(b["label_uid"]):
                raise ValueError(f"{t} {split}: the retype and round-3 predictions hold different rows "
                                 f"({len(set(a.label_uid) - set(b.label_uid))} only in retype, "
                                 f"{len(set(b.label_uid) - set(a.label_uid))} only in round 3)")
            m = a.merge(b[["label_uid", "true", "p_incorrect"]], on="label_uid", suffixes=("", "_r3"),
                        validate="one_to_one")
            if (m["true"] != m["true_r3"]).any():
                raise ValueError(f"{t} {split}: the two prediction files disagree on the true class")
            m = m.merge(meta[t][["label_uid", "city", "pano_id"]], on="label_uid", how="left", validate="one_to_one")
            y = (m["true"] == "incorrect").astype(int).to_numpy()
            sa, sb = m["p_incorrect"].to_numpy(float), m["p_incorrect_r3"].to_numpy(float)
            row = {"type": t, "split": split, "n": int(len(m)), "n_incorrect": int(y.sum()),
                   "auc_retype": _auc(y, sa), "auc_round3": _auc(y, sb)}
            row["delta"] = row["auc_retype"] - row["auc_round3"]
            row["delta_lo"] = row["delta_hi"] = float("nan")
            if n_boot and 0 < y.sum() < len(y):
                groups = pano_groups(m).to_numpy()
                boot = cluster_boot(y, np.column_stack([sa, sb]), groups, n_boot, seed)
                d = boot[:, 0] - boot[:, 1]
                if np.isfinite(d).any():
                    row["delta_lo"], row["delta_hi"] = (float(v) for v in np.nanpercentile(d, [2.5, 97.5]))
            for name, s in (("retype", sa), ("round3", sb)):
                row[f"within_city_auc_{name}"], row[f"n_cities_within_{name}"] = within_city_auc(
                    y, s, m["city"], within_min_n)
                for lev in levels:
                    row[f"r_at_p{lev:.2f}_incorrect_{name}"] = recall_at_precision(y == 1, s, lev)[0] \
                        if 0 < y.sum() < len(y) else float("nan")
                    row[f"r_at_p{lev:.2f}_correct_{name}"] = recall_at_precision(y == 0, 1 - s, lev)[0] \
                        if 0 < y.sum() < len(y) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Type-head aggregates
# ---------------------------------------------------------------------------------------------------------------------

def confusion_correct(pred, splits=REPORT_SPLITS):
    """8 x 8 counts (true class x top-1 class) over the correct validated rows of ``splits``; index ``true``."""
    d = pred[(pred["source"] == "validated") & (pred["is_correct"].astype(int) == 1) & pred["split"].isin(splits)]
    tab = pd.crosstab(d["target_type"], d["top1"]).reindex(index=list(CLASSES), columns=list(CLASSES),
                                                           fill_value=0)
    tab.index.name = "true"
    return tab.astype(int)


def type_head_table(pred, splits=REPORT_SPLITS):
    """Top-1 and top-3 accuracy of the type head on correct validated rows, per class and overall.

    >>> p = pd.DataFrame({"source": ["validated"] * 2, "is_correct": [1, 1], "split": ["test"] * 2,
    ...                   "target_type": ["CurbRamp", "Obstacle"], "top1": ["CurbRamp", "Other"],
    ...                   "top3": ["CurbRamp|Other|Signal", "Other|Obstacle|Signal"]})
    >>> type_head_table(p)[["class", "n", "top1_acc", "top3_acc"]].values.tolist()[-1]
    ['all', 2, 0.5, 1.0]
    """
    d = pred[(pred["source"] == "validated") & (pred["is_correct"].astype(int) == 1) & pred["split"].isin(splits)]
    hit1 = d["top1"] == d["target_type"]
    hit3 = [t in s.split("|") for t, s in zip(d["target_type"], d["top3"])]
    d = d.assign(hit1=hit1.to_numpy(), hit3=hit3)
    rows = [{"class": c, "n": int(len(g)), "top1_acc": float(g.hit1.mean()) if len(g) else float("nan"),
             "top3_acc": float(g.hit3.mean()) if len(g) else float("nan")}
            for c, g in ((c, d[d["target_type"] == c]) for c in CLASSES)]
    rows.append({"class": "all", "n": int(len(d)), "top1_acc": float(d.hit1.mean()) if len(d) else float("nan"),
                 "top3_acc": float(np.mean(d.hit3)) if len(d) else float("nan")})
    return pd.DataFrame(rows)


def suggested_on_incorrect(pred, splits=REPORT_SPLITS):
    """For the incorrect validated rows of ``splits``: claimed type x suggested type counts (what the model thinks
    the wrong labels are)."""
    d = pred[(pred["source"] == "validated") & (pred["is_correct"].astype(int) == 0) & pred["split"].isin(splits)]
    tab = pd.crosstab(d["claimed_type"], d["suggested_type"]).reindex(index=list(CLASSES), columns=list(CLASSES),
                                                                     fill_value=0)
    tab.index.name = "claimed"
    return tab.astype(int)


def suggestion_eval(rows, target_col, by=None):
    """Top-1 / top-3 accuracy of ``suggested_type`` / ``suggested_top3`` against ``target_col`` on ``rows``
    (incorrect rows whose ``target_col`` is set), overall or per ``by``.

    >>> r = pd.DataFrame({"is_correct": [0, 0, 1], "proxy_type": ["CurbRamp", "Obstacle", ""],
    ...                   "suggested_type": ["CurbRamp", "Signal", "x"],
    ...                   "suggested_top3": ["CurbRamp|a|b", "Signal|Obstacle|a", "x|y|z"], "split": ["test"] * 3})
    >>> suggestion_eval(r, "proxy_type", by="split").values.tolist()
    [['test', 2, 0.5, 1.0]]
    """
    d = rows[(rows["is_correct"].astype(int) == 0) & (rows[target_col].astype(str) != "")]
    d = d.assign(hit1=(d["suggested_type"] == d[target_col]).to_numpy(),
                 hit3=[t in s.split("|") for t, s in zip(d[target_col], d["suggested_top3"])])
    keys = [by] if by else []
    if by:
        g = d.groupby(by)
        out = pd.DataFrame({"n": g.size(), "top1_acc": g["hit1"].mean(), "top3_acc": g["hit3"].mean()}).reset_index()
    else:
        out = pd.DataFrame([{"n": len(d), "top1_acc": float(d.hit1.mean()) if len(d) else float("nan"),
                             "top3_acc": float(d.hit3.mean()) if len(d) else float("nan")}])
    return out[keys + ["n", "top1_acc", "top3_acc"]]


def colocated_eval(pred):
    """``suggestion_eval`` against ``proxy_type`` per split, overall (``proxy_b_in_train = all``) and broken down by
    whether the proxy's partner B sits on a train panorama (``1``/``0``). Rows with B in train carry a location
    leak; test_heldout has none by construction (its cities are not in train), so it is the headline.

    >>> r = pd.DataFrame({"source": ["validated"] * 3, "is_correct": [0, 0, 0], "split": ["test"] * 3,
    ...     "proxy_type": ["CurbRamp"] * 3, "proxy_b_in_train": ["1", "0", "0"],
    ...     "suggested_type": ["CurbRamp", "CurbRamp", "Signal"], "suggested_top3": ["CurbRamp|a|b"] * 3})
    >>> colocated_eval(r).values.tolist()
    [['test', 'all', 3, 0.6666666666666666, 1.0], ['test', '0', 2, 0.5, 1.0], ['test', '1', 1, 1.0, 1.0]]
    """
    v = pred[pred["source"] == "validated"]
    if "proxy_b_in_train" not in v.columns:
        v = v.assign(proxy_b_in_train="")
    overall = suggestion_eval(v, "proxy_type", by="split").assign(proxy_b_in_train="all")
    d = v[v["proxy_type"].astype(str) != ""]
    by_b = [suggestion_eval(g, "proxy_type", by="split").assign(proxy_b_in_train=b)
            for b, g in d.groupby(d["proxy_b_in_train"].astype(str))]
    out = pd.concat([overall] + by_b, ignore_index=True)
    rank = {"all": 0, "0": 1, "1": 2}
    out = out.sort_values(["split", "proxy_b_in_train"], key=lambda c: c.map(rank).fillna(3) if c.name ==
                          "proxy_b_in_train" else c, kind="mergesort").reset_index(drop=True)
    return out[["split", "proxy_b_in_train", "n", "top1_acc", "top3_acc"]]


def check_aggregates(folder, private_terms=()):
    """Refuse an aggregate folder that could identify labels or name a non-public deployment: any CSV with an id
    column (``ID_COLUMNS``), any ``city:123``-like token in a text file, and (``publish_to_hf.check_no_private_names``)
    any ``private_terms`` substring. Raises ValueError; returns the number of files checked."""
    n = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            n += 1
            try:
                text = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            if name.endswith(".csv"):
                cols = set(pd.read_csv(io.StringIO(text), nrows=0).columns) if text.strip() else set()
                bad = cols & set(ID_COLUMNS)
                if bad:
                    raise ValueError(f"{path}: aggregate has id column(s) {sorted(bad)}")
            hit = LABEL_ID_TOKEN.search(text)
            if hit:
                raise ValueError(f"{path}: label-id-like token {hit.group(0)!r}")
    if private_terms:
        from publish_to_hf import PublishError, check_no_private_names
        try:
            check_no_private_names(folder, list(private_terms))
        except PublishError as e:
            raise ValueError(str(e)) from None
    return n


# ---------------------------------------------------------------------------------------------------------------------
# Audit (the doc's measured numbers; aggregates only)
# ---------------------------------------------------------------------------------------------------------------------

def audit_type_changes(edits_by_city, public):
    """Rows and type changes per deployment of labelEdits exports (``{city: frame}``), public deployments by name
    and the rest pooled; plus the change counts per ``old->new`` pair, over all deployments."""
    per_city, pairs = {}, Counter()
    for city, e in sorted(edits_by_city.items()):
        ch = type_changes(e)
        name = city if city in public else POOLED_PRIVATE
        cur = per_city.setdefault(name, {"deployments": 0, "rows": 0, "type_changes": 0})
        cur["deployments"] += 1
        cur["rows"] += int(len(e))
        cur["type_changes"] += int(len(ch))
        pairs.update(f"{o}->{n}" for o, n in zip(ch["old_label_type"], ch["new_label_type"]))
    total = {"deployments": len(edits_by_city), "rows": sum(v["rows"] for v in per_city.values()),
             "type_changes": sum(v["type_changes"] for v in per_city.values())}
    return {"total": total, "per_deployment": per_city, "pairs": dict(sorted(pairs.items()))}


def audit_eligible(raw):
    """Eligible human-authored labels per class (``data_generation.eligible_labels``, aliases applied), before any
    downsampling: ``{class: {"correct": n, "incorrect": n}}``."""
    out = {c: {"correct": 0, "incorrect": 0} for c in CLASSES}
    for lt in sorted(set(raw["label_type"].dropna())):
        c = canonical_type(lt)
        if c not in CLASSES:
            continue
        e, _ = dg.eligible_labels(raw, lt)
        e = e[e["author"] == "human"]
        for k in ("correct", "incorrect"):
            out[c][k] += int((e["label_class"] == k).sum())
    return out


def pano_sharing(validated):
    """Per claimed type: rows, and the share on a panorama that also holds a row of another type."""
    g = pano_groups(validated)
    types_per_pano = validated.groupby(g)["claimed_type"].transform("nunique")
    shared = types_per_pano > 1
    return {t: {"rows": int((validated["claimed_type"] == t).sum()),
                "on_shared_pano": int((shared & (validated["claimed_type"] == t)).sum()),
                "share": float((shared & (validated["claimed_type"] == t)).sum()
                               / max(1, (validated["claimed_type"] == t).sum()))}
            for t in sorted(set(validated["claimed_type"]))}


def audit_colocated(raw, radii=AUDIT_RADII_M):
    """Colocated proxy pairs (``colocated_targets``) per ``A->B`` at each radius, over every city of ``raw``."""
    inc, cor = colocation_inputs(raw)
    out = {}
    for r in radii:
        proxy = colocated_targets(inc, cor, r)
        claimed = inc.set_index("label_uid")["claimed_type"]
        p = proxy[proxy != ""]
        pairs = Counter(f"{claimed[u]}->{t}" for u, t in p.items())
        out[f"{r:g}m"] = {"incorrect_labels": int(len(inc)), "with_proxy": int(len(p)),
                          "pairs": dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])))}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------------------------

def _get_text(url):
    import requests
    for attempt in (1, 2):
        try:
            r = requests.get(url, timeout=dg.CSV_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"{url} failed ({e}); retrying once")
            time.sleep(10)


def cmd_fetch_edits(args, get_text=None, fetch_listing=None):
    """One labelEdits CSV per deployment into ``--out``, and ``fetch.json`` (date, rows and sha256 per city).

    ``--cities-cache`` serves only the listing (its ``cities.json``) when ``--cities`` is not given. With ``--cities``
    it is not passed on: ``resolve_cities`` would otherwise take a bare id with a ``<cache>/<id>.csv`` (a rawLabels
    export, irrelevant here) as its own subdomain, and the labelEdits host is the listing's (``st-louis-mo`` is served
    from ``sidewalk-st-louis``)."""
    cities, source = dg.resolve_cities(args.cities, None if args.cities else args.cities_cache, fetch=fetch_listing)
    os.makedirs(args.out, exist_ok=True)
    record = {"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "city_source": source,
              "cities": {}}
    for city, host in cities.items():
        url = f"{dg.deployment_base_url(host)}/v3/api/labelEdits?filetype=csv"
        text = (get_text or _get_text)(url)
        path = os.path.join(args.out, f"{city}.csv")
        with open(path, "w") as f:
            f.write(text)
        n = len(pd.read_csv(io.StringIO(text))) if text.strip() else 0
        record["cities"][city] = {"rows": n, "sha256": file_sha256(path)}
        print(f"{city}: {n} labelEdits rows")
    write_json(os.path.join(args.out, "fetch.json"), record)
    return record


def read_edits_dir(edits_dir, cities=None):
    """``{city: labelEdits frame}`` from ``<edits_dir>/<city>.csv`` (``fetch.json`` names the cities when present)."""
    if cities is None:
        rec = os.path.join(edits_dir, "fetch.json")
        if os.path.exists(rec):
            cities = sorted(json.load(open(rec))["cities"])
        else:
            cities = sorted(n[:-4] for n in os.listdir(edits_dir) if n.endswith(".csv"))
    out = {}
    for c in cities:
        path = os.path.join(edits_dir, f"{c}.csv")
        out[c] = pd.read_csv(path, dtype={"label_id": str}) if os.path.getsize(path) else pd.DataFrame(
            columns=["label_id", "old_label_type", "new_label_type", "edit_time"])
    return out


def cmd_type_edits(args):
    """The kept type changes of every deployment, joined to the current rawLabels rows, -> ``--out`` (CSV) and
    ``<out>.json`` (counts per reason)."""
    edits = read_edits_dir(args.edits)
    parts, why, per_city = [], Counter(), {}
    for city, e in edits.items():
        ch = attach_city(type_changes(e), city)
        if ch.empty:
            continue
        raw_path = os.path.join(args.rawlabels, f"{city}.csv")
        if not os.path.exists(raw_path):
            why["no_rawlabels_export"] += len(ch)
            continue
        raw = pd.read_csv(raw_path, low_memory=False, dtype={"label_id": str, "pano_id": str, "user_id": str})
        raw["city"] = city
        kept, dropped = join_current_labels(ch, raw)
        why.update(dropped)
        per_city[city] = {"changes": int(len(ch)), "kept": int(len(kept))}
        parts.append(kept)
    kept = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label_uid"])
    assert kept["label_uid"].is_unique
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept.to_csv(args.out, index=False, lineterminator="\n")
    rec = {"kept": int(len(kept)), "dropped": dict(why), "per_city": per_city,
           "pairs": dict(Counter(f"{o}->{n}" for o, n in zip(kept.get("old_label_type", []),
                                                             kept.get("new_label_type", []))))}
    write_json(os.path.splitext(args.out)[0] + ".json", rec)
    print(json.dumps(rec, indent=2))
    return kept, rec


def type_dirs(root, types):
    """``{type_dir: path to split.csv}`` for the requested type directories that exist under ``root``. A missing
    directory of one of the five validator types raises; a missing extra type is skipped with a warning."""
    out = {}
    for t in types:
        path = os.path.join(root, t, "split.csv")
        if os.path.isfile(path):
            out[t] = path
        elif canonical_type(t) in VALIDATOR_TYPES:
            raise FileNotFoundError(f"no {path}: the five round-3 types are required")
        else:
            print(f"WARNING: no {path}; {t} is left out (its class stays in the head)")
    return out


def cmd_audit(args):
    """The doc's measured numbers, as one aggregate JSON (``--out``)."""
    public, public_source = _public(args.public_cities)
    out = {"generated": date.today().isoformat(), "public_cities_source": public_source}
    if args.edits:
        out["type_changes"] = audit_type_changes(read_edits_dir(args.edits), public)
    validated = None
    if args.splits:
        paths = type_dirs(args.splits, args.types.split(","))
        validated = pd.concat([validated_rows(read_split(p), t) for t, p in paths.items()], ignore_index=True)
        out["split_rows"] = {t: int(n) for t, n in validated["claimed_type"].value_counts().sort_index().items()}
        out["pano_sharing"] = pano_sharing(validated)
        _, moves = inherit_splits(validated)
        out["inherit_moves"] = moves.to_dict("records")
    if args.rawlabels:
        cities = sorted(n[:-4] for n in os.listdir(args.rawlabels) if n.endswith(".csv"))
        if validated is not None:
            cities = sorted(set(validated["city"]))
        cols = ["label_id", "user_id", "pano_id", "pano_source", "label_type", "validations", "pano_x", "pano_y",
                "pano_width", "pano_height", "latitude", "longitude"]
        raw = load_rawlabels(args.rawlabels, cities, usecols=cols)
        out["rawlabels_cities"] = {"public": int(sum(c in public for c in cities)),
                                   "private_pooled": int(sum(c not in public for c in cities))}
        out["eligible"] = audit_eligible(raw)
        out["colocated"] = audit_colocated(raw)
    write_json(args.out, out)
    _check_file(args.out, public, args)
    print(json.dumps(out, indent=2)[:4000])
    return out


def _check_file(path, public, args):
    """``check_aggregates`` on a single written file (its own temporary view)."""
    text = open(path).read()
    hit = LABEL_ID_TOKEN.search(text)
    if hit:
        raise ValueError(f"{path}: label-id-like token {hit.group(0)!r}")
    for term in _private_terms(public, args):
        if term and term.lower() in text.lower():
            raise ValueError(f"{path}: names a non-public deployment")


def _private_terms(public, args):
    """City ids seen in the inputs that are not public (never written; used only to scan outputs)."""
    seen = set()
    for d in (getattr(args, "rawlabels", None), getattr(args, "edits", None)):
        if d and os.path.isdir(d):
            seen |= {n[:-4] for n in os.listdir(d) if n.endswith(".csv")}
    if getattr(args, "split", None) and os.path.exists(args.split):
        seen |= set(pd.read_csv(args.split, usecols=["city"], dtype=str)["city"])
    return sorted(seen - set(public))


def _public(spec):
    from analysis.retrain_report import public_cities
    return public_cities(None if spec in (None, "", "live") else spec)


def cmd_crops(args):
    """Crops for the type-edit labels, cut as data_generation.py cuts a training crop: the per-provider pose from
    ``crop_pose.pose_for_row`` (``--tilt-beta`` for GSV), the click latitude offset, depth-sized, WebP. Resumable on
    ``<out>/manifest.csv``; ``crop_path`` there is absolute.

    Mapillary: ``mapillary_rotations`` is None (no Graph API lookup, unlike ``data_generation.py
    --mapillary-token-env``), so a Mapillary label whose DB pose is empty gets the rig-frame fallback here and its
    crop can differ from the one a training run with a token would cut. The manifest's ``pose_source`` says which
    pose each crop used. All four type-edit labels of 2026-11-05 are in GSV deployments; wire the token through
    (with data_generation.py's rotation cache) before this matters."""
    import csv

    import crop_pose
    labels = pd.read_csv(args.labels, dtype={"label_id": str, "pano_id": str, "user_id": str},
                         keep_default_na=False)
    dg.check_pano_store_marker(args.pano_store)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    cols = ["label_uid", "city", "label_id", "pano_id", "pano_source", "pose_source", "tilt_beta",
            "click_lat_offset_deg", "pano_path", "depth_m", "crop_half_px", "clipped", "crop_path", "status"]
    done = set()
    if os.path.exists(manifest_path):
        old = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        old = old[(old["status"] == "ok") & old["crop_path"].map(os.path.exists)]
        done = set(old["label_uid"])
        old.to_csv(manifest_path, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(manifest_path, index=False)
    todo = labels[~labels["label_uid"].isin(done)].sort_values(["city", "pano_id"], kind="mergesort")
    device = dg.resolve_device(args.allow_cpu)
    depth_model = dg.load_depth_model("vitb", device) if len(todo) else None
    last, counts = (None, None, None), Counter()
    with open(manifest_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        for row in todo.to_dict("records"):
            row["author"] = dg.label_author(row.get("user_id", ""))
            for k in ("camera_pitch", "camera_roll"):
                row[k] = pd.to_numeric(row.get(k), errors="coerce")
            pose = crop_pose.pose_for_row(row, args.tilt_beta, None)
            crop_path = os.path.abspath(os.path.join(args.out, "crops", f"{row['city']}_{row['label_id']}.webp"))
            rec = {"label_uid": row["label_uid"], "city": row["city"], "label_id": row["label_id"],
                   "pano_id": row["pano_id"], "pano_source": row.get("pano_source", ""),
                   "pose_source": pose.pose_source, "tilt_beta": "" if pose.beta is None else pose.beta,
                   "click_lat_offset_deg": pose.click_lat_offset_deg or 0.0, "crop_path": crop_path, "status": ""}
            try:
                if last[0] != (row["city"], row["pano_id"]):
                    read, info = dg.read_store_panorama(args.pano_store, row["city"], row["pano_id"])
                    last = ((row["city"], row["pano_id"]), read, info)
                pano_bgr, info = last[1], last[2]
                rec["pano_path"] = info.get("pano_path") or ""
                if isinstance(pano_bgr, str):
                    rec["status"] = pano_bgr
                else:
                    y_norm = (float(row["pano_y"]) / float(row["pano_height"])
                              + (pose.click_lat_offset_deg or 0.0) / 180.0)
                    crop, cinfo = dg.create_perspective_crop(
                        depth_model, np.ascontiguousarray(pano_bgr[..., ::-1]),
                        float(row["pano_x"]) / float(row["pano_width"]), y_norm,
                        camera_pitch=pose.camera_pitch, camera_roll=pose.camera_roll,
                        apply_tilt=pose.beta is not None, beta=pose.beta or 0.0, device=device)
                    dg.save_webp(crop, crop_path)
                    rec.update({k: v for k, v in cinfo.items() if k in cols})
                    rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001 - one bad label must not end the run; the row says what happened
                if dg.is_fatal(e):
                    raise
                traceback.print_exc()
                rec["status"] = f"error:{type(e).__name__}"
            w.writerow(rec)
            f.flush()
            counts[rec["status"]] += 1
    run = {"labels": os.path.abspath(args.labels), "labels_sha256": file_sha256(args.labels),
           "pano_store": os.path.abspath(args.pano_store), "tilt_beta": args.tilt_beta, "depth_encoder": "vitb",
           "renderer": "torch" if device else "numpy", "package_source": dg.package_source(),
           "status_counts_this_run": dict(counts),
           "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(os.path.join(args.out, "crops_run.json"), run)
    print(json.dumps(run, indent=2))
    return run


def assemble(splits_root, types, edits=None, edit_crops=None, raw=None, colocate_m=COLOCATE_M, merge_radius_m=0.0,
             allow_missing_crops=False, crop_root=None, min_extra_incorrect=MIN_EXTRA_INCORRECT):
    """The multi-class split (``SCHEMA``) and its record: ``(df, record)``. See the module docstring.

    ``edits``: the ``type-edits`` stage's frame (or None); ``edit_crops``: the ``crops`` stage's manifest frame;
    ``raw``: rawLabels rows (``load_rawlabels``) for the colocation flag and the optional ``merge_radius_m`` pass
    (0 = off). A type directory other than the five validator types with fewer than ``min_extra_incorrect``
    incorrect rows is left out (recorded under ``too_few_incorrect``). Every check failure raises."""
    paths = type_dirs(splits_root, types)
    parts, too_few = [], {}
    for t, p in paths.items():
        rows = validated_rows(read_split(p), t)
        n_inc = int((rows["is_correct"] == 0).sum())
        if canonical_type(t) not in VALIDATOR_TYPES and t not in VALIDATOR_TYPES and n_inc < min_extra_incorrect:
            print(f"WARNING: {t} has {n_inc} incorrect rows (< {min_extra_incorrect}); left out")
            too_few[t] = n_inc
            continue
        parts.append(rows)
    paths = {t: p for t, p in paths.items() if t not in too_few}
    validated = pd.concat(parts, ignore_index=True)
    groups = None
    if merge_radius_m and merge_radius_m > 0:
        if raw is None:
            raise ValueError("--merge-radius-m needs --rawlabels (label coordinates)")
        coords = raw.set_index("label_uid")[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        c = coords.reindex(validated["label_uid"])
        groups, _ = sd.merge_nearby_groups(pano_groups(validated), c["latitude"].to_numpy(),
                                           c["longitude"].to_numpy(), merge_radius_m)
    df, moves = inherit_splits(validated, groups)
    rec = {"types": {t: {"split_csv": os.path.abspath(p), "sha256": file_sha256(p)} for t, p in paths.items()},
           "missing_types": [t for t in types if t not in paths and t not in too_few],
           "too_few_incorrect": too_few, "moves": moves.to_dict("records"),
           "colocate_m": colocate_m, "merge_radius_m": merge_radius_m}
    if raw is not None:
        inc, cor = colocation_inputs(raw)
        rows = df[(df["source"] == "validated") & (df["is_correct"] == 0)][["label_uid", "city", "claimed_type"]]
        q = rows.merge(raw[["label_uid", "user_id", "latitude", "longitude"]], on="label_uid", how="left",
                       validate="one_to_one")
        q["latitude"] = pd.to_numeric(q["latitude"], errors="coerce")
        q["longitude"] = pd.to_numeric(q["longitude"], errors="coerce")
        proxy = colocated_targets(q, cor, colocate_m)
        df["proxy_type"] = df["label_uid"].map(proxy).where(df["is_correct"] == 0, "").fillna("")
        # The location leak of the proxy: B (the correct label that makes the proxy) on a panorama that is train
        # after inheritance. The model has then seen B's crop as a B positive, so "suggested = B" may be a
        # memorised place. "1"/"0" on proxy rows, "" elsewhere (and "" throughout when rawLabels has no pano_id).
        if "pano_id" in cor.columns:
            partners = colocated_partner_panos(q, cor, colocate_m)
            train_panos = set(df.loc[df["split"] == "train", "pano_id"]) - {""}
            b_train = df["label_uid"].map(lambda u: "1" if partners.get(u, set()) & train_panos else "0")
            df["proxy_b_in_train"] = b_train.where(df["proxy_type"] != "", "")
        has = df["proxy_type"] != ""
        rec["colocated"] = {"n": int(has.sum()),
                            "by_split": {k: int(v) for k, v in df[has]["split"].value_counts().items()},
                            "b_in_train_by_split": {k: int(v) for k, v in
                                                    df[has & (df["proxy_b_in_train"] == "1")]["split"]
                                                    .value_counts().items()}}
    if edits is not None and len(edits):
        if edit_crops is None:
            raise ValueError("type-edit rows need the crops stage's manifest (--edit-crops)")
        in_round3 = edits["label_uid"].isin(set(df["label_uid"]))
        er = edit_rows(edits[~in_round3], edit_crops)
        train_panos = set(df.loc[df["split"].isin(["train", "val"]), "pano_id"]) - {""}
        rec["type_edit"] = {"labels": int(len(edits)), "in_round3_dropped": int(in_round3.sum()),
                            "without_ok_crop": int(len(edits) - in_round3.sum() - er["label_uid"].nunique()),
                            "rows": int(len(er)), "on_train_or_val_pano": int(er.loc[er["is_correct"] == 1, "pano_id"]
                                                                                  .isin(train_panos).sum())}
        df = pd.concat([df, er], ignore_index=True)
    assert df["row_uid"].is_unique, "row_uid is not unique"
    # Belt and braces: inherit_splits already gives every panorama one split, so this cannot fire today (and a
    # mutation test that deletes it survives); it guards a future change to the grouping.
    sd.check_no_shared_panoramas(df[df["split"] != TEST_RETYPE])
    bad = (df["source"] == "validated") & ((df["target_type"] == "") != (df["is_correct"].astype(int) == 0))
    assert not bad.any(), "a validated row's target_type must be empty exactly when it is incorrect"
    bad = (df["source"] == "type_edit") & (df["target_type"] == "")
    assert not bad.any()
    if crop_root is not None:
        full = [p if os.path.isabs(p) else os.path.join(crop_root, p) for p in df["crop_path"]]
        missing = [p for p in full if not os.path.exists(p)]
        rec["missing_crops"] = len(missing)
        if missing and not allow_missing_crops:
            raise FileNotFoundError(f"{len(missing)} crops do not exist (e.g. {missing[:3]}); "
                                    "--allow-missing-crops only lists them")
    df = df[SCHEMA]
    rec["rows"] = int(len(df))
    rec["by_split_class"] = {f"{s}|{c}|{k}": int(n) for (s, c, k), n in
                             df.groupby(["split", "claimed_type", "is_correct"]).size().items()}
    rec["positives_by_class_split"] = {f"{c}|{s}": int(n) for (c, s), n in
                                       df[df["is_correct"].astype(int) == 1].groupby(["target_type", "split"])
                                       .size().items()}
    return df, rec


def cmd_assemble(args):
    types = args.types.split(",")
    edits = pd.read_csv(args.edits, dtype=str, keep_default_na=False) if args.edits else None
    crops = (pd.read_csv(os.path.join(args.edit_crops, "manifest.csv"), dtype=str, keep_default_na=False)
             if args.edit_crops else None)
    raw = None
    if args.rawlabels:
        cities = set()
        for t, p in type_dirs(args.splits, types).items():
            cities |= set(read_split(p)["city"])
        raw = load_rawlabels(args.rawlabels, cities,
                             usecols=["label_id", "user_id", "label_type", "validations", "latitude", "longitude",
                                      "pano_id"])
    df, rec = assemble(args.splits, types, edits, crops, raw, args.colocate_m, args.merge_radius_m,
                       args.allow_missing_crops, crop_root=args.splits, min_extra_incorrect=args.min_extra_incorrect)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    rec.update({"out": os.path.abspath(args.out), "sha256": file_sha256(args.out),
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if args.edits:
        rec["type_edits_csv_sha256"] = file_sha256(args.edits)
    write_json(os.path.join(os.path.dirname(os.path.abspath(args.out)), "assemble.json"), rec)
    print(f"wrote {args.out}: {len(df)} rows; moves {rec['moves']}")
    return df, rec


def cmd_predict(args):
    """Score rows of ``retype_split.csv`` with the run's best checkpoint (softmax over the eight classes)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from evaluate import best_checkpoint_in
    split = read_split(args.split)
    splits = args.splits.split(",")
    keep = split["split"].isin(splits)
    if args.include_train_colocated:
        keep |= (split["split"] == "train") & (split["proxy_type"] != "")
    rows = split[keep].reset_index(drop=True)
    ckpt = args.checkpoint or best_checkpoint_in(args.run_dir)
    print(f"checkpoint: {ckpt}; {len(rows)} rows")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("No CUDA device; --allow-cpu is for smoke tests only.")
    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=False)  # as training and the API
    model = AutoModelForImageClassification.from_pretrained(ckpt).to(device).eval()
    order = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    if tuple(order) != CLASSES:
        raise ValueError(f"checkpoint id2label {order} is not retype.CLASSES {CLASSES}")
    probs = []
    for i, p in enumerate(rows["crop_path"]):
        path = p if os.path.isabs(p) else os.path.join(args.crop_root, p)
        inputs = processor(Image.open(path).convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            probs.append(model(**inputs).logits.softmax(dim=-1).float().cpu().numpy()[0])
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    out = predictions_frame(rows, np.array(probs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    write_json(os.path.splitext(args.out)[0] + ".json",
               {"checkpoint": os.path.abspath(ckpt), "split": os.path.abspath(args.split),
                "split_sha256": file_sha256(args.split), "splits": splits, "rows": int(len(out)),
                "predictions_sha256": file_sha256(args.out)})
    print(f"wrote {args.out}")


def read_predictions(path):
    return pd.read_csv(path, dtype={c: str for c in ("row_uid", "label_uid", "split", "claimed_type", "target_type",
                                                     "source", "proxy_type", "proxy_b_in_train", "top1", "top3",
                                                     "suggested_type", "suggested_top3")}, keep_default_na=False)


def cmd_export_binary(args):
    counts = export_binary(read_predictions(args.predictions), read_split(args.split), args.out_root, args.tag)
    print(json.dumps(counts, indent=2))
    return counts


def gate_what_if(binary, round3, priors_path, splits=REPORT_SPLITS):
    """Per type and eval split: coverage and precision at the 0.92 gate of a map fitted on val under the deployment
    prior (``calibration.calibrate.calibrate``, nothing held back), for the retype verdict and for round 3 refitted
    the same way on its own predictions."""
    import warnings

    from calibration import calibrate as cal
    rows = []
    for t in binary:
        pi, _ = cal.load_prior(priors_path, t)
        for model, frame in (("retype", binary[t]), ("round3", round3[t])):
            for split in splits:
                if not (frame["split"] == split).any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, rep, _ = cal.calibrate(frame, pi, "val", split)
                if rep["eval"] == "n/a":
                    continue
                for v in ("correct", "incorrect"):
                    e = rep["eval"][v]
                    rows.append({"type": t, "split": split, "model": model, "verdict": v, "n": e["n"],
                                 "gate_coverage": e["gate_coverage"], "gate_precision": e["gate_precision"],
                                 "gate_n": e["gate_n"]})
    return pd.DataFrame(rows)


def _md(df, floatfmt="{:.3f}"):
    """A frame as a Markdown table (NaN blank)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    """Aggregates for the doc: the per-type comparison with round 3 on the same rows, slices, the gate what-if, the
    type head and the suggestions. Writes only to ``--docs-dir`` and checks it."""
    from analysis import slices as sl
    pred = read_predictions(args.predictions)
    split = read_split(args.split)
    public, public_source = _public(args.public_cities)
    os.makedirs(args.docs_dir, exist_ok=True)
    binary, round3, meta, train = {}, {}, {}, {}
    for t in VALIDATOR_TYPES:
        binary[t] = binary_rows(pred, split, t)
        round3[t] = pd.read_csv(os.path.join(args.root, t, "predictions.csv" if not args.baseline_tag
                                             else f"predictions_{args.baseline_tag}.csv"), dtype={"label_uid": str})
        r3split = read_split(os.path.join(args.root, t, "split.csv"))
        meta[t] = r3split[["label_uid", "city", "pano_id", "severity", "tags"]]
        train[t] = split[(split["source"] == "validated") & (split["claimed_type"] == t) & (split["split"] == "train")]
    comp = paired_auc_table(binary, round3, meta, args.n_boot, args.seed)
    comp.to_csv(os.path.join(args.docs_dir, "binary_vs_round3.csv"), index=False, lineterminator="\n")
    head = ["type", "split", "n", "n_incorrect", "auc_retype", "auc_round3", "delta", "delta_lo", "delta_hi",
            "within_city_auc_retype", "within_city_auc_round3"]
    with open(os.path.join(args.docs_dir, "binary_vs_round3.md"), "w") as f:
        f.write("# Re-typing verdict p(claimed) against the round-3 per-type models, same rows\n\n")
        f.write(_md(comp[head]) if len(comp) else "No rows.\n")
    parts = ["# Re-typing vs round 3 by severity and tag\n"]
    for t in VALIDATOR_TYPES:
        a = binary[t].merge(meta[t], on="label_uid", how="left", validate="one_to_one")
        b = round3[t]
        for s in REPORT_SPLITS:
            sa = a[a["split"] == s]
            if sa.empty:
                continue
            for by in sl.BY:
                d = sl.paired_delta_table(sa, b[b["split"] == s], by, n_boot=args.slice_boot, seed=args.seed)
                parts.append(sl.delta_markdown(d, by, f"{t}, {s}, by {by}", a="retype", b="round 3"))
                parts.append(sl.slice_markdown(sl.slice_table(sa, by, train=train[t], n_boot=0), by,
                                               f"{t}, {s}, by {by} (retype, with training rows)"))
    with open(os.path.join(args.docs_dir, "slices_vs_round3.md"), "w") as f:
        f.write("\n".join(parts))
    gate = gate_what_if(binary, round3, args.priors or os.path.join(args.root, "deployment_priors.json"))
    gate.to_csv(os.path.join(args.docs_dir, "gate_what_if.csv"), index=False, lineterminator="\n")
    type_head_table(pred).to_csv(os.path.join(args.docs_dir, "type_head.csv"), index=False, lineterminator="\n")
    confusion_correct(pred).to_csv(os.path.join(args.docs_dir, "confusion_correct_test.csv"), lineterminator="\n")
    suggested_on_incorrect(pred).to_csv(os.path.join(args.docs_dir, "suggested_on_incorrect.csv"),
                                        lineterminator="\n")
    suggestion_eval(pred[pred["source"] == "type_edit"], "target_type").to_csv(
        os.path.join(args.docs_dir, "type_edit_eval.csv"), index=False, lineterminator="\n")
    colocated_eval(pred).to_csv(os.path.join(args.docs_dir, "colocated_eval.csv"), index=False,
                                lineterminator="\n")
    prov = {"generated": date.today().isoformat(), "n_boot": args.n_boot, "seed": args.seed,
            "slice_boot": args.slice_boot, "public_cities_source": public_source,
            "sha256": {"retype_split.csv": file_sha256(args.split),
                       "retype_predictions.csv": file_sha256(args.predictions),
                       **{f"{t}/split.csv": file_sha256(os.path.join(args.root, t, "split.csv"))
                          for t in VALIDATOR_TYPES},
                       **{f"{t}/predictions{'_' + args.baseline_tag if args.baseline_tag else ''}.csv":
                          file_sha256(os.path.join(args.root, t, "predictions.csv" if not args.baseline_tag
                                                   else f"predictions_{args.baseline_tag}.csv"))
                          for t in VALIDATOR_TYPES}},
            "job_ids": args.job_ids or None}
    for name in ("training_run.json", "run_key.json"):
        if args.run_dir and os.path.exists(os.path.join(args.run_dir, name)):
            prov["sha256"][name] = file_sha256(os.path.join(args.run_dir, name))
    write_json(os.path.join(args.docs_dir, "provenance.json"), prov)
    n = check_aggregates(args.docs_dir, _private_terms(public, args))
    print(open(os.path.join(args.docs_dir, "binary_vs_round3.md")).read())
    print(f"checked {n} files in {args.docs_dir}: no ids, no non-public deployment names")


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("fetch-edits", help="labelEdits CSV per deployment (network)")
    s.add_argument("--out", required=True)
    s.add_argument("--cities", default=None, help="comma list of city_id[=subdomain] (default: the public listing)")
    s.add_argument("--cities-cache", default=None,
                   help="directory holding a saved cities.json to use instead of the live listing (default: live; "
                        "public/private drifts, so prefer live); ignored with --cities")

    s = sub.add_parser("type-edits", help="type changes joined to the current rawLabels rows")
    s.add_argument("--edits", required=True, help="fetch-edits output directory")
    s.add_argument("--rawlabels", required=True, help="directory of rawLabels exports <city>.csv")
    s.add_argument("--out", required=True, help="type_edits.csv (outside the repo)")

    s = sub.add_parser("audit", help="the doc's measured numbers (aggregates only)")
    s.add_argument("--rawlabels", default=None)
    s.add_argument("--splits", default=None, help="data root with <Type>/split.csv")
    s.add_argument("--types", default=",".join(VALIDATOR_TYPES))
    s.add_argument("--edits", default=None, help="fetch-edits output directory")
    s.add_argument("--public-cities", default="live", help="'live' (the API) or a saved /v3/api/cities response")
    s.add_argument("--out", required=True)

    s = sub.add_parser("crops", help="crops for the type-edit labels (GPU)")
    s.add_argument("--labels", required=True, help="type_edits.csv")
    s.add_argument("--pano-store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tilt-beta", type=float, default=dg.DEFAULT_TILT_BETA)
    s.add_argument("--allow-cpu", action="store_true")

    s = sub.add_parser("assemble", help="the multi-class retype_split.csv")
    s.add_argument("--splits", required=True, help="data root with <Type>/split.csv (crop paths resolve here)")
    s.add_argument("--types", default=",".join(VALIDATOR_TYPES + EXTRA_TYPES + ("Occlusion",)),
                   help="type directories; the five validator types are required, the rest optional")
    s.add_argument("--edits", default=None, help="type_edits.csv")
    s.add_argument("--edit-crops", default=None, help="the crops stage's --out")
    s.add_argument("--rawlabels", default=None, help="for the colocation flag (and --merge-radius-m)")
    s.add_argument("--colocate-m", type=float, default=COLOCATE_M)
    s.add_argument("--merge-radius-m", type=float, default=0.0,
                   help="second inheritance pass over groups merged within this radius (0 = off)")
    s.add_argument("--allow-missing-crops", action="store_true")
    s.add_argument("--min-extra-incorrect", type=int, default=MIN_EXTRA_INCORRECT,
                   help="leave out an extra type with fewer incorrect rows (default %(default)s)")
    s.add_argument("--out", required=True)

    s = sub.add_parser("predict", help="8-class scores (GPU)")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--checkpoint", default=None, help="default: the run's best checkpoint")
    s.add_argument("--split", required=True)
    s.add_argument("--crop-root", required=True)
    s.add_argument("--splits", default=",".join(EVAL_SPLITS + (TEST_RETYPE,)))
    s.add_argument("--include-train-colocated", action="store_true")
    s.add_argument("--allow-cpu", action="store_true")
    s.add_argument("--out", required=True)

    s = sub.add_parser("export-binary", help="per-type predictions_<tag>.csv for evaluate.py / calibrate.py")
    s.add_argument("--predictions", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--tag", required=True)

    s = sub.add_parser("report", help="aggregates for the doc")
    s.add_argument("--predictions", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--root", required=True, help="round-3 data root (<Type>/predictions.csv, split.csv)")
    s.add_argument("--baseline-tag", default="", help="round 3's predictions_<tag>.csv (default predictions.csv)")
    s.add_argument("--priors", default=None, help="default <root>/deployment_priors.json")
    s.add_argument("--run-dir", default=None, help="the training run (training_run.json, run_key.json hashes)")
    s.add_argument("--docs-dir", required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--slice-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=20261105)
    s.add_argument("--public-cities", default="live")
    s.add_argument("--job-ids", default="", help="Slurm job ids, recorded in provenance.json")
    s.add_argument("--rawlabels", default=None, help="only to list non-public city ids for the output scan")
    s.add_argument("--edits", default=None, help="only to list non-public city ids for the output scan")
    return p


STAGES = {"fetch-edits": cmd_fetch_edits, "type-edits": cmd_type_edits, "audit": cmd_audit, "crops": cmd_crops,
          "assemble": cmd_assemble, "predict": cmd_predict, "export-binary": cmd_export_binary,
          "report": cmd_report}


def main(argv=None):
    args = build_parser().parse_args(argv)
    return STAGES[args.stage](args)


if __name__ == "__main__":
    main()
