"""The round-4 recipe on round-3 data (issue #74): gold-free training splits, the rule variants, the gold scoring
manifests and the bookkeeping of the klone runs.

Why
---
Every model so far trained on the expert gold set's own labels **and on the rest of their scenes** (other train rows
on the same panorama, or within the split's 3 m merge), so the gold set could not judge a Disagree map. This module
builds a mirror of the round-3 data with those rows out of training, the ``cl-drop`` variant on top of it, the
rebalanced rule test of #75 / #56 (built, tested, launched only in stage 2), the training count of every severity
and tag slice under each variant, and the manifests and scoring that put a model's score on every gold item. The
decision module that reads the results is ``analysis/disagree_gate.py``; the doc is ``docs/round4-2026-11.md``.

Layout on klone (``analysis/round4_submit.sh`` sets these)::

    DATA = .../retrain-2026-10              # round 3, read-only: <Type>/split.csv, crops, cl/, dinov2/
    R4   = $WORK/data                        # the mirror: <Type>/split.csv (gold-free), <Type>/cl/<variant>/split.csv,
                                             # symlinks to round 3's crop folders, and everything evaluate.py writes
    RUNS = $WORK/runs                        # <Type>/<TAG>/{training_run.json, run_key.json, best/}
    KEEP = /gscratch/makelab/.../round4-disagree-2026-11-keep   # small artifacts only (collect)

Subcommands (CPU unless noted):

- ``gold-uids``: the four columns of the gold table the klone runs need (``label_uid, label_type, pano_id,
  crop_path_r3``), never a verdict, vote or stratum. Run on the Mac; the file is private (it names private
  deployments' labels) and is refused inside a git checkout.
- ``prep``: the gold-free base split per type (``gold_item`` and ``gold_scene`` train rows set to
  ``train_dropped``), crop symlinks, ``r4_prep.json``; ``--round3-ref`` adds round 3's own outputs as the reference
  row ``round3``.
- ``variants``: ``cl-drop`` (the round-3 flags on the gold-free base) and, for stage 2, ``symmetric-rb`` /
  ``matched-rb`` (one per seed).
- ``slice-train``: training rows per severity and tag slice under each variant.
- ``gold-manifest``: one scoring manifest per type, ``r3crops`` (round 3's crops; descriptive only) or ``uniform``
  (crops cut by the stage-2 leg).
- ``round3-ckpt``: the round-3 checkpoint directory of a type, resolved next to ``trainer_state.json`` (the path
  that file records is the one at training time) and checked for the files inference needs.
- ``score-gold`` (GPU): score a manifest with a checkpoint; skips when the same checkpoint and manifest were scored.
- ``storage``: parse ``hyakstorage`` output and refuse when a filesystem is nearly full.
- ``verify-smoke``: the smoke run's checks.
- ``collect``: the run manifest, the untagged baseline copies ``confident_learning.py report`` reads, and the copy
  of small artifacts to ``KEEP``.

Usage (klone; the launcher runs these)::

    python analysis/round4.py gold-uids --gold ~/Data/.../gold.csv --out gold_uids.csv          # on the Mac
    python analysis/round4.py prep --data $DATA --r4 $R4 --gold-uids $WORK/gold_uids.csv \\
        --types Crosswalk,CurbRamp,NoCurbRamp,Obstacle,SurfaceProblem --rawlabels $RAWLABELS --round3-ref
    python analysis/round4.py variants --r4 $R4 --data $DATA --types CurbRamp,Crosswalk,NoCurbRamp,Obstacle \\
        --which cl-drop
    python analysis/round4.py gold-manifest --gold-uids $WORK/gold_uids.csv --split-root $DATA --mode r3crops \\
        --types Crosswalk --out $R4/gold/r3crops
    python analysis/round4.py score-gold --manifest $R4/gold/r3crops/Crosswalk.csv \\
        --checkpoint $RUNS/Crosswalk/baseline/best --tag baseline \\
        --out $R4/Crosswalk/gold_predictions_r3crops_baseline.csv
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
import data_generation as dg  # noqa: E402
import split_data  # noqa: E402
from analysis import confident_learning as cl  # noqa: E402
from analysis import slices as sl  # noqa: E402
from fine_tune_classifier import git_state, is_complete_checkpoint, slurm_state  # noqa: E402

LABEL_TYPES = ("Crosswalk", "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem")
CL_DROP_TYPES = ("Crosswalk", "CurbRamp", "NoCurbRamp", "Obstacle")  # SurfaceProblem's recipe is the baseline
GOLD_UID_COLUMNS = ["label_uid", "label_type", "pano_id", "crop_path_r3"]
FORBIDDEN_ROOTS = ("/tmp", "/gscratch/makelab")  # R4 holds splits and evaluation outputs: scrubbed, never these
SEEDS = (42, 43, 44)
BASE_SEED = 42
GOLD_SPLIT = "gold"
MANIFEST_COLUMNS = ["label_uid", "label_class", "split", "crop_path"]
WEIGHT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
SCRUBBED_PURGE_DAYS = 21
# A5: the 3 m merge is transitive, so one chain of nearby labels can drop far more than the gold panoramas hold.
SCENE_CEILING_FACTOR = 2.0
SCENE_CEILING_SHARE = 0.03
# collect: what goes to KEEP (small files only) and its budget.
KEEP_MAX_BYTES = 1 << 30
KEEP_MAX_FILES = 600
KEEP_DATA_PATTERNS = ("r4_prep.json", "metrics_*.json", "predictions_*.csv", "report_*.txt", "slices_*.md",
                      "per_city_*.md", "gold_predictions_*.csv", "gold_predictions_*_provenance.json",
                      "cl/*/split.csv", "cl/*/variant.json")
KEEP_RUN_FILES = ("training_run.json", "run_key.json")
PRIVATE_MODE = 0o600  # gold-derived files in KEEP: Jon only


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 22):
    """Hex sha256 of a file, read in chunks (the weights are 1.2 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (set, tuple)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def split_list(value, default=()):
    """A comma (or colon) separated list, as the Slurm files pass it."""
    if value is None:
        return list(default)
    return [x.strip() for x in re.split(r"[,:]", value) if x.strip()]


def city_of(uid):
    """The deployment of a ``label_uid`` (``city:label_id``); ``label_id`` is per-city, so the pair is the key."""
    return uid.rsplit(":", 1)[0]


def under(path, root):
    """True if ``path`` is ``root`` or inside it (both made absolute; symlinks resolved as well)."""
    for p, r in ((os.path.abspath(path), os.path.abspath(root)), (os.path.realpath(path), os.path.realpath(root))):
        if p == r or p.startswith(r.rstrip("/") + "/"):
            return True
    return False


def refuse_location(r4, data, forbidden=FORBIDDEN_ROOTS):
    """Raise ``SystemExit`` when ``r4`` equals ``data`` (round 3 is read-only) or lies under a forbidden root (``/tmp``
    is node-local on klone; ``/gscratch/makelab`` has no room). Tests pass their own ``forbidden``.

    >>> refuse_location("/scratch/r4", "/data", forbidden=("/tmp",))
    >>> refuse_location("/tmp/x/r4", "/data", forbidden=("/tmp",))
    Traceback (most recent call last):
    ...
    SystemExit: refusing R4 /tmp/x/r4: under /tmp
    """
    if os.path.realpath(r4) == os.path.realpath(data):
        raise SystemExit(f"refusing R4 {r4}: it is DATA (round 3 is read-only)")
    for root in forbidden:
        if under(r4, root):
            raise SystemExit(f"refusing R4 {r4}: under {root}")


def read_gold_uids(path):
    """The uid file (``gold-uids``): strings, ``label_uid`` unique, exactly ``GOLD_UID_COLUMNS``."""
    g = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(g.columns) != GOLD_UID_COLUMNS:
        raise ValueError(f"{path}: columns {list(g.columns)}, expected {GOLD_UID_COLUMNS}")
    if not g["label_uid"].is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    return g


# ---------------------------------------------------------------------------------------------------------------------
# gold-uids (Mac)
# ---------------------------------------------------------------------------------------------------------------------

def gold_uid_table(gold):
    """The four columns klone needs, from ``adjudicator.py gold``'s table. ``crop_path_r3`` is kept only where the
    round-3 crop was ``ok``. No verdict, vote, tally, stratum or round leaves the Mac.

    >>> g = pd.DataFrame({"label_uid": ["b:2", "a:1"], "label_type": ["Obstacle"] * 2, "pano_id": ["p", "q"],
    ...                   "crop_path_r3": ["correct/b_2.webp", "x"], "crop_status_r3": ["ok", "no_pano"],
    ...                   "expert": ["incorrect", "correct"]})
    >>> gold_uid_table(g).values.tolist()
    [['a:1', 'Obstacle', 'q', ''], ['b:2', 'Obstacle', 'p', 'correct/b_2.webp']]
    """
    g = gold.copy()
    if "crop_status_r3" in g:
        g.loc[g["crop_status_r3"] != "ok", "crop_path_r3"] = ""
    for c in GOLD_UID_COLUMNS:
        if c not in g:
            g[c] = ""
    out = g[GOLD_UID_COLUMNS].sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    if not out["label_uid"].is_unique:
        raise ValueError("gold table: label_uid is not unique")
    return out


def cmd_gold_uids(args):
    from analysis import ai_disagree_relabel as rl
    if rl.inside_git_checkout(args.out):
        raise SystemExit(f"refusing to write the gold uid file inside a git checkout: {args.out}")
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    out = gold_uid_table(gold)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    os.chmod(args.out, PRIVATE_MODE)
    print(f"wrote {args.out} ({len(out)} items) sha256 {sha256_file(args.out)}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# prep: the gold-free base split
# ---------------------------------------------------------------------------------------------------------------------

def _coords(frame, rawlabels, strict):
    """``latitude``/``longitude`` per row of ``frame`` (``label_uid, city, label_id``), indexed like ``frame``.
    ``strict``: ``split_data.load_coordinates`` (a missing city export raises). Otherwise cities without an export
    get NaN and are returned in the second value."""
    f = frame.assign(status="ok")
    if strict or f.empty:
        c = split_data.load_coordinates(f, rawlabels) if not f.empty else pd.DataFrame(columns=["latitude", "longitude"])
        return c.reindex(f["label_uid"]).set_axis(f.index), []
    have = f["city"].map(lambda c: os.path.isfile(os.path.join(rawlabels, f"{c}.csv")))
    out = pd.DataFrame({"latitude": np.nan, "longitude": np.nan}, index=f.index)
    if have.any():
        c = split_data.load_coordinates(f[have], rawlabels)
        out.loc[have, ["latitude", "longitude"]] = c.reindex(f.loc[have, "label_uid"]).to_numpy()
    return out, sorted(f.loc[~have, "label_uid"])


def scene_groups(frame, lat, lon, merge_radius_m):
    """Scene group per row: ``city|pano_id`` (a row without a panorama is its own group), then merged within
    ``merge_radius_m`` metres (``split_data.merge_nearby_groups``; rows with unknown coordinates stay per panorama).
    The city is part of the key so the same ``pano_id`` string in two deployments is never one scene."""
    pano = frame["pano_id"].astype(str)
    group = pd.Series(np.where(pano.str.len() > 0, frame["city"].astype(str) + "|" + pano,
                               "uid:" + frame["label_uid"].astype(str)), index=frame.index)
    if merge_radius_m > 0:
        group, _ = split_data.merge_nearby_groups(group, lat, lon, merge_radius_m)
    return group


def gold_free(base, gold_t, rawlabels=None, merge_radius_m=3.0, ceiling=True):
    """``(variant, stats)``: ``base`` (one type's ``split.csv``) with every train row that is a gold item
    (``gold_item``) or shares a scene with one (``gold_scene``) set to ``train_dropped``. Gold items that are not
    train rows (val / test rows, or not in the split at all) enter as pseudo-rows so their scenes count too. Val,
    test and test_heldout rows are never touched.

    With ``rawlabels`` the scene is the 3 m merge of the split and the fold assignment; train rows need their
    coordinates (a missing export raises, as ``split_data`` does), gold pseudo-rows from a deployment without an
    export fall back to the panorama and are counted (A5). Without ``rawlabels`` scenes are panoramas only.

    Ceiling (A5): the 3 m merge is transitive, so when the ``gold_scene`` drop exceeds
    ``max(2 x the panorama-only drop, 3% of the type's train rows)`` this raises ``ValueError`` naming the largest
    merged group it would drop. Jon decides whether to accept a larger drop (``ceiling=False``).
    """
    df = base.copy()
    train = df[df["split"] == "train"]
    gold_uids = set(gold_t["label_uid"])
    in_split = gold_uids & set(df["label_uid"])
    pseudo_src = gold_t[~gold_t["label_uid"].isin(set(train["label_uid"]))]
    split_pano = dict(zip(df["label_uid"], df["pano_id"]))
    pseudo = pd.DataFrame({
        "label_uid": pseudo_src["label_uid"].to_numpy(),
        "city": [city_of(u) for u in pseudo_src["label_uid"]],
        "label_id": [u.rsplit(":", 1)[1] for u in pseudo_src["label_uid"]],
        "pano_id": [split_pano.get(u) or p for u, p in zip(pseudo_src["label_uid"], pseudo_src["pano_id"])]})
    frame = pd.concat([train[["label_uid", "city", "label_id", "pano_id"]].assign(_pseudo=False),
                       pseudo.assign(_pseudo=True)], ignore_index=True)
    if not frame["label_uid"].is_unique:
        raise AssertionError("a gold pseudo-row duplicates a train row")
    is_gold = frame["label_uid"].isin(gold_uids).to_numpy()
    no_export = []
    if rawlabels and merge_radius_m > 0:
        ct, _ = _coords(frame[~frame["_pseudo"]], rawlabels, strict=True)
        cp, no_export = _coords(frame[frame["_pseudo"]], rawlabels, strict=False)
        xy = pd.concat([ct, cp]).loc[frame.index]
        lat, lon = xy["latitude"].to_numpy(dtype=float), xy["longitude"].to_numpy(dtype=float)
        mode = f"scene: panorama + {merge_radius_m:g} m merge (rawLabels coordinates)"
    else:
        lat = lon = np.full(len(frame), np.nan)
        mode = "panorama only (no rawLabels given)"
    radius = merge_radius_m if (rawlabels and merge_radius_m > 0) else 0.0
    group = scene_groups(frame, lat, lon, radius)
    pano_group = scene_groups(frame, lat, lon, 0.0)
    gold_groups, gold_panos = set(group[is_gold]), set(pano_group[is_gold])
    is_train = ~frame["_pseudo"].to_numpy()
    drop_item = is_train & is_gold
    drop_scene = is_train & ~is_gold & group.isin(gold_groups).to_numpy()
    pano_only = int((is_train & ~is_gold & pano_group.isin(gold_panos).to_numpy()).sum())
    n_train = int(is_train.sum())
    limit = max(SCENE_CEILING_FACTOR * pano_only, SCENE_CEILING_SHARE * n_train)
    dropped_groups = group[drop_item | drop_scene]
    largest = dropped_groups.value_counts()
    largest_info = ({"group": "(withheld)", "rows": int(largest.iloc[0]),
                     "panoramas": int(pano_group[group == largest.index[0]].nunique())} if len(largest) else None)
    stats = {"n_train_before": n_train, "gold_items_of_type": int(len(gold_t)),
             "gold_items_in_split": int(len(in_split)), "gold_items_not_in_split": int(len(gold_uids - in_split)),
             "gold_items_train_rows": int(drop_item.sum()),
             "gold_items_without_panorama": int((pseudo["pano_id"].astype(str) == "").sum()),
             "gold_items_without_coordinates": int(np.isnan(lat[is_gold]).sum()) if radius else None,
             "gold_pseudo_rows_without_rawlabels_export": len(no_export),
             "grouping": mode, "merge_radius_m": radius,
             "dropped": {"gold_item": int(drop_item.sum()), "gold_scene": int(drop_scene.sum())},
             "gold_scene_panorama_only": pano_only,
             "gold_scene_ceiling": {"limit": float(limit), "factor": SCENE_CEILING_FACTOR,
                                    "share_of_train": SCENE_CEILING_SHARE, "applied": bool(ceiling)},
             "largest_dropped_group": largest_info}
    if ceiling and drop_scene.sum() > limit:
        raise ValueError(f"gold_scene drop {int(drop_scene.sum())} exceeds the ceiling {limit:.0f} "
                         f"(max of {SCENE_CEILING_FACTOR:g} x the panorama-only {pano_only} and "
                         f"{SCENE_CEILING_SHARE:.0%} of {n_train} train rows); largest merged group dropped: "
                         f"{largest_info}. Jon decides whether to accept it (prep --no-scene-ceiling).")
    uid_reason = dict(zip(frame.loc[drop_item, "label_uid"], ["gold_item"] * int(drop_item.sum())))
    uid_reason.update(zip(frame.loc[drop_scene, "label_uid"], ["gold_scene"] * int(drop_scene.sum())))
    hit = df["label_uid"].isin(uid_reason) & (df["split"] == "train")
    df.loc[hit, "split"] = cl.TRAIN_DROPPED
    reason = df["label_uid"].map(uid_reason)
    by = (df[hit].assign(reason=reason[hit]).groupby(["reason", "label_class"]).size())
    stats["dropped_by_reason_and_class"] = {f"{r}/{c}": int(n) for (r, c), n in by.items()}
    stats["n_train_after"] = int((df["split"] == "train").sum())
    cl.check_eval_rows_unchanged(base, df)
    return df, stats


def link_crops(base, r4_type_dir, data_type_dir):
    """Symlink each first path component of the relative ``crop_path``s (``correct``, ``incorrect``) from the mirror
    to round 3's folder, so ``fine_tune_classifier.py`` and ``evaluate.py`` resolve the crops unchanged."""
    firsts = sorted({p.split("/", 1)[0] for p in base["crop_path"] if p and not os.path.isabs(p) and "/" in p})
    made = []
    for name in firsts:
        link, target = os.path.join(r4_type_dir, name), os.path.join(data_type_dir, name)
        if os.path.islink(link):
            if os.path.realpath(link) != os.path.realpath(target):
                raise SystemExit(f"{link} points to {os.readlink(link)}, not {target}")
            continue
        if os.path.exists(link):
            raise SystemExit(f"{link} exists and is not a symlink")
        os.symlink(target, link)
        made.append(name)
    return firsts


def cmd_prep(args):
    refuse_location(args.r4, args.data, tuple(args.forbidden_root) if args.forbidden_root else FORBIDDEN_ROOTS)
    gold = read_gold_uids(args.gold_uids)
    uid_sha = sha256_file(args.gold_uids)
    git = git_state(REPO_DIR)
    for t in split_list(args.types):
        src = os.path.join(args.data, t, "split.csv")
        base = cl.read_split(src)
        gold_t = gold[gold["label_type"] == t]
        variant, stats = gold_free(base, gold_t, args.rawlabels, args.merge_radius_m, not args.no_scene_ceiling)
        tdir = os.path.join(args.r4, t)
        os.makedirs(tdir, exist_ok=True)
        out = cl.write_split(variant, os.path.join(tdir, "split.csv"))
        crops = link_crops(base, tdir, os.path.join(args.data, t))
        copied = []
        for name in ("generation_run.json",):
            if os.path.exists(os.path.join(args.data, t, name)):
                shutil.copy2(os.path.join(args.data, t, name), os.path.join(tdir, name))
                copied.append(name)
        ref = {}
        if args.round3_ref:
            for a, b in (("metrics.json", "metrics_round3.json"), ("predictions.csv", "predictions_round3.csv")):
                p = os.path.join(args.data, t, a)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(tdir, b))
                    ref[b] = sha256_file(p)
            ref["cl/round3/split.csv"] = sha256_file(cl.write_split(base, os.path.join(tdir, "cl", "round3",
                                                                                      "split.csv")))
        stats.update({"label_type": t, "source_split": os.path.abspath(src), "source_sha256": sha256_file(src),
                      "gold_uids": os.path.abspath(args.gold_uids), "gold_uids_sha256": uid_sha,
                      "split_sha256": sha256_file(out), "crop_links": crops, "copied": copied,
                      "round3_reference": ref or None, **git,
                      "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        write_json(stats, os.path.join(tdir, "r4_prep.json"))
        print(f"{t}: train {stats['n_train_before']} -> {stats['n_train_after']} "
              f"(gold_item {stats['dropped']['gold_item']}, gold_scene {stats['dropped']['gold_scene']}, "
              f"panorama-only {stats['gold_scene_panorama_only']}); {stats['grouping']}")
    root_prior = os.path.join(args.data, "deployment_priors.json")
    if os.path.exists(root_prior):
        shutil.copy2(root_prior, os.path.join(args.r4, "deployment_priors.json"))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------------------------------------

def check_flags_match(flag_summary, round3_split_sha):
    """Raise unless the confident-learning flags were computed on the round-3 split with this sha256.

    >>> check_flags_match({"source_sha256": "ab"}, "ab")
    >>> check_flags_match({"source_sha256": "ab"}, "cd")
    Traceback (most recent call last):
    ...
    ValueError: the flags were computed on split sha256 ab, not on round 3's cd
    """
    if flag_summary.get("source_sha256") != round3_split_sha:
        raise ValueError(f"the flags were computed on split sha256 {flag_summary.get('source_sha256')}, "
                         f"not on round 3's {round3_split_sha}")


def cl_drop(base, flagged_uids):
    """``cl-drop`` on the gold-free base: flagged train rows left out (``confident_learning.cl_variants``). A gold row
    that is also flagged is already ``train_dropped`` and stays so (dropped once)."""
    return cl.cl_variants(base, set(flagged_uids))["cl-drop"]


def train_counts(df):
    t = df[df["split"] == "train"]["label_class"].value_counts()
    return {c: int(t.get(c, 0)) for c in cl.CLASSES}


def _rank(seed, uids):
    return pd.Series([dg.label_rank(seed, u) for u in uids], index=uids.index)


def rebalance(df, ratio, seed):
    """Set surplus ``correct`` train rows to ``train_dropped`` so correct : incorrect is ``ratio`` (rounded to the
    nearest row), keeping the rows with the lowest ``data_generation.label_rank(seed, label_uid)``. Rows are only
    ever dropped, never added.

    >>> d = pd.DataFrame({"label_uid": [f"a:{i}" for i in range(10)], "split": "train",
    ...                   "label_class": ["correct"] * 8 + ["incorrect"] * 2})
    >>> train_counts(rebalance(d, 2.0, 42))
    {'correct': 4, 'incorrect': 2}
    """
    out = df.copy()
    c = train_counts(out)
    target = int(round(ratio * c["incorrect"]))
    surplus = c["correct"] - target
    if surplus <= 0:
        return out
    cand = out[(out["split"] == "train") & (out["label_class"] == "correct")]
    order = _rank(seed, cand["label_uid"]).sort_values(kind="mergesort")
    out.loc[order.index[-surplus:], "split"] = cl.TRAIN_DROPPED
    return out


def base_ratio(base):
    c = train_counts(base)
    return c["correct"] / c["incorrect"] if c["incorrect"] else float("nan")


def symmetric_rb(base, seed):
    """``symmetric`` (net Disagree > 2 for ``incorrect``) on the train rows, then rebalanced to the base's ratio."""
    return rebalance(cl.rule_variant(base, "symmetric"), base_ratio(base), seed)


def matched_rb(base, sym, seed):
    """The size-matched control of ``symmetric-rb`` (``sym``): drop as many crowd-sourced ``incorrect`` train rows
    as ``symmetric`` did, picked by ``label_rank(seed, …)`` without looking at the vote margin (expert-sourced rows
    spared), then as many ``correct`` rows as ``sym`` dropped. Train counts per class equal ``sym``'s."""
    out = base.copy()
    was_train = base["split"] == "train"
    sym_drop = was_train & (sym["split"].to_numpy() == cl.TRAIN_DROPPED)
    n_inc = int((sym_drop & (base["label_class"] == "incorrect")).sum())
    n_cor = int((sym_drop & (base["label_class"] == "correct")).sum())
    expert = base["class_source"].eq("expert") if "class_source" in base else pd.Series(False, index=base.index)
    for cls, n, spare in (("incorrect", n_inc, True), ("correct", n_cor, False)):
        if not n:
            continue
        pool = out[(out["split"] == "train") & (out["label_class"] == cls) & ~(expert & spare)]
        if len(pool) < n:
            raise ValueError(f"matched-rb: {n} {cls} rows to drop, {len(pool)} eligible")
        order = _rank(seed, pool["label_uid"]).sort_values(kind="mergesort")
        out.loc[order.index[-n:], "split"] = cl.TRAIN_DROPPED
    if train_counts(out) != train_counts(sym):
        raise AssertionError(f"matched-rb {train_counts(out)} != symmetric-rb {train_counts(sym)}")
    return out


def variant_name(variant, seed):
    """``matched-rb`` for seed 42, ``matched-rb-s43`` otherwise (the Slurm ``TAG`` convention ``seed_runs`` reads)."""
    return variant if seed == BASE_SEED else f"{variant}-s{seed}"


def variant_meta(base, v, name, seed, extra=None):
    """Train counts before and after (by class, vote margin, class source), sha256s and the seed."""
    was = base["split"] == "train"
    kept = v["split"].to_numpy() == "train"
    margin = (pd.to_numeric(base.get("human_agree", 0), errors="coerce").fillna(0)
              - pd.to_numeric(base.get("human_disagree", 0), errors="coerce").fillna(0)).astype(int)
    margin_bucket = np.where(base["label_class"] == "incorrect",
                             np.where(-margin <= 2, "net Disagree <= 2", "net Disagree >= 3"),
                             np.where(margin <= 3, "net Agree <= 3", "net Agree >= 4"))
    source = base["class_source"] if "class_source" in base else pd.Series("", index=base.index)
    margin_bucket = np.where(source == "expert", "expert", margin_bucket)
    tab = pd.DataFrame({"class": base["label_class"], "margin": margin_bucket, "source": source,
                        "before": was.to_numpy(), "after": kept & was.to_numpy()})
    def by(cols):
        return {"/".join(map(str, k if isinstance(k, tuple) else (k,))):
                {"before": int(g["before"].sum()), "after": int(g["after"].sum())} for k, g in tab.groupby(cols)}

    return {"variant": name, "seed": seed, "train_before": train_counts(base), "train_after": train_counts(v),
            "by_class": by("class"), "by_margin": by(["class", "margin"]), "by_class_source": by("source"),
            **(extra or {})}


def cmd_variants(args):
    which = split_list(args.which)
    unknown = set(which) - {"cl-drop", "symmetric-rb", "matched-rb"}
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}")
    seeds = [int(s) for s in split_list(args.seeds)]
    for t in split_list(args.types):
        bpath = os.path.join(args.r4, t, "split.csv")
        base = cl.read_split(bpath)
        r3_path = os.path.join(args.data, t, "split.csv")
        common = {"base_split": os.path.abspath(bpath), "base_sha256": sha256_file(bpath),
                  "round3_split_sha256": sha256_file(r3_path), **git_state(REPO_DIR)}
        made = {}
        if "cl-drop" in which:
            flags_path = os.path.join(args.data, t, "cl", "flag_summary.json")
            with open(flags_path) as f:
                check_flags_match(json.load(f), common["round3_split_sha256"])
            noisy_path = os.path.join(args.data, t, "cl", "noisy_targets.csv")
            noisy = pd.read_csv(noisy_path, dtype=str, keep_default_na=False)
            flagged = set(noisy.loc[noisy["flagged"].isin(["1", "True", "true"]), "label_uid"])
            made["cl-drop"] = (cl_drop(base, flagged), None,
                               {"noisy_targets_sha256": sha256_file(noisy_path), "flagged": len(flagged)})
        for seed in seeds:
            if "symmetric-rb" in which or "matched-rb" in which:
                sym = symmetric_rb(base, seed)
                if "symmetric-rb" in which:
                    made[variant_name("symmetric-rb", seed)] = (sym, seed, {"ratio": base_ratio(base)})
                if "matched-rb" in which:
                    made[variant_name("matched-rb", seed)] = (matched_rb(base, sym, seed), seed,
                                                              {"ratio": base_ratio(base)})
        for name, (v, seed, extra) in made.items():
            cl.check_eval_rows_unchanged(base, v)
            path = cl.write_split(v, os.path.join(args.r4, t, "cl", name, "split.csv"))
            meta = variant_meta(base, v, name, seed, {**common, **extra, "split_sha256": sha256_file(path)})
            write_json(meta, os.path.join(args.r4, t, "cl", name, "variant.json"))
            print(f"{t} {name}: train {meta['train_before']} -> {meta['train_after']}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# slice-train
# ---------------------------------------------------------------------------------------------------------------------

def variant_split_path(r4, t, variant):
    return os.path.join(r4, t, "split.csv") if variant == "baseline" else os.path.join(r4, t, "cl", variant,
                                                                                         "split.csv")


def slice_train_counts(df, by):
    """``{slice: {"correct": n, "incorrect": n, "train_n": n, "needs_data": bool}}`` over the train rows of ``df``."""
    tr = sl.with_slice(df[df["split"] == "train"], by)
    out = {}
    for name, g in tr.groupby("slice"):
        c = g["label_class"].value_counts()
        n = int(len(g))
        out[str(name)] = {"correct": int(c.get("correct", 0)), "incorrect": int(c.get("incorrect", 0)),
                          "train_n": n, "needs_data": n < sl.THIN_TRAIN}
    return out


def slice_train(r4, types, variants):
    """Per type, slice dimension and variant: the training count of each severity and tag slice, from that
    variant's own ``split.csv`` (``evaluate.py``'s ``slices_<tag>.md`` counts the base split whatever the variant).
    A slice a variant empties shows ``train_n`` 0 there."""
    res = {}
    for t in types:
        res[t] = {}
        frames = {v: cl.read_split(variant_split_path(r4, t, v)) for v in variants
                  if os.path.exists(variant_split_path(r4, t, v))}
        for by in sl.BY:
            per = {v: slice_train_counts(f, by) for v, f in frames.items()}
            names = sorted({s for d in per.values() for s in d}, key=lambda s: (s == sl.NONE, s))
            empty = {"correct": 0, "incorrect": 0, "train_n": 0, "needs_data": True}
            res[t][by] = {s: {v: per[v].get(s, empty) for v in frames} for s in names}
    return res


def slice_train_markdown(res):
    lines = ["# Training rows per slice and variant", "",
             f"Train rows (`split == train`) of each severity and tag slice in each variant's own split file; "
             f"`*` marks a slice under {sl.THIN_TRAIN} rows (`slices.THIN_TRAIN`).", ""]
    for t, dims in res.items():
        for by, rows in dims.items():
            variants = sorted({v for r in rows.values() for v in r}, key=lambda v: (v != "round3", v != "baseline", v))
            lines += [f"## {t}, by {by}", "", "| slice | " + " | ".join(variants) + " |",
                      "|---|" + "---:|" * len(variants)]
            for s, r in rows.items():
                cells = [f"{r[v]['train_n']:,}{'*' if r[v]['needs_data'] else ''}" if v in r else "" for v in variants]
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def cmd_slice_train(args):
    res = slice_train(args.r4, split_list(args.types), split_list(args.variants))
    os.makedirs(args.out, exist_ok=True)
    write_json(res, os.path.join(args.out, "slice_train.json"))
    with open(os.path.join(args.out, "slice_train.md"), "w") as f:
        f.write(slice_train_markdown(res))
    print(f"wrote {args.out}/slice_train.json and slice_train.md")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# gold-manifest and score-gold
# ---------------------------------------------------------------------------------------------------------------------

def gold_manifest(uids, label_type, mode, split_root=None, crops_root=None):
    """``(manifest, sidecar)``: one row per gold item of ``label_type`` with a crop file (``label_uid, label_class``
    (blank), ``split = gold``, absolute ``crop_path``). ``r3crops``: round 3's crop ``<split_root>/<Type>/
    <crop_path_r3>``. ``uniform``: only crops the stage-2 leg cut (``<crops_root>/manifest.csv``, status ok), never
    a round-3 crop. Items without a file are left out and counted, so inference never opens a blank path."""
    g = uids[uids["label_type"] == label_type]
    if mode == "r3crops":
        paths = [os.path.join(split_root, label_type, p) if p else "" for p in g["crop_path_r3"]]
    elif mode == "uniform":
        man = pd.read_csv(os.path.join(crops_root, "manifest.csv"), dtype=str, keep_default_na=False)
        man = man[man["status"] == "ok"]
        if not man["label_uid"].is_unique:
            raise ValueError(f"{crops_root}/manifest.csv: label_uid is not unique")
        staged = dict(zip(man["label_uid"], (os.path.join(crops_root, p) for p in man["crop_path"])))
        paths = [staged.get(u, "") for u in g["label_uid"]]
    else:
        raise ValueError(f"unknown mode {mode!r} (r3crops or uniform)")
    paths = [os.path.abspath(p) if p and os.path.isfile(p) else "" for p in paths]
    m = pd.DataFrame({"label_uid": g["label_uid"].to_numpy(), "label_class": "", "split": GOLD_SPLIT,
                      "crop_path": paths}, columns=MANIFEST_COLUMNS)
    have = m["crop_path"] != ""
    side = {"label_type": label_type, "mode": mode, "n_items": int(len(m)), "n_with_crop": int(have.sum()),
            "n_no_crop": int((~have).sum())}
    return m[have].sort_values("label_uid", kind="mergesort").reset_index(drop=True), side


def cmd_gold_manifest(args):
    uids = read_gold_uids(args.gold_uids)
    os.makedirs(args.out, exist_ok=True)
    for t in split_list(args.types):
        m, side = gold_manifest(uids, t, args.mode, args.split_root, args.crops_root)
        path = os.path.join(args.out, f"{t}.csv")
        m.to_csv(path, index=False, lineterminator="\n")
        side.update({"gold_uids_sha256": sha256_file(args.gold_uids), "manifest_sha256": sha256_file(path)})
        write_json(side, os.path.join(args.out, f"{t}.json"))
        print(f"{t} {args.mode}: {side['n_with_crop']} of {side['n_items']} gold items have a crop -> {path}")
    return 0


def checkpoint_dirs(base):
    """Complete ``checkpoint-N`` directories under ``base``, by step."""
    dirs = [d for d in os.listdir(base) if is_complete_checkpoint(os.path.join(base, d))]
    return sorted(dirs, key=lambda d: int(d.split("-")[1]))


def resolve_round3_checkpoint(dinov2_dir):
    """The best checkpoint of a round-3 ``dinov2/`` directory, resolved **next to** ``trainer_state.json``: that file
    records ``best_model_checkpoint`` as it was written at training time (on klone a path relative to the training
    job's working directory), so only its basename is used (A3). Raises unless ``config.json``,
    ``model.safetensors`` and ``preprocessor_config.json`` are there. Reads JSON only (cheap on a login node)."""
    dirs = checkpoint_dirs(dinov2_dir)
    if not dirs:
        raise FileNotFoundError(f"no complete checkpoint-* under {dinov2_dir}")
    with open(os.path.join(dinov2_dir, dirs[-1], "trainer_state.json")) as f:
        best = json.load(f).get("best_model_checkpoint")
    if not best:
        raise ValueError(f"{dinov2_dir}/{dirs[-1]}/trainer_state.json has no best_model_checkpoint")
    path = os.path.join(os.path.abspath(dinov2_dir), os.path.basename(os.path.normpath(best)))
    missing = [f for f in WEIGHT_FILES if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"{path} lacks {missing} (trainer_state.json said {best})")
    return path


def cmd_round3_ckpt(args):
    print(resolve_round3_checkpoint(args.dinov2))
    return 0


def hf_cache_match(weights_sha, label_type, hf_home=None):
    """Descriptive: whether a published validator revision in the local Hugging Face cache has these exact weights
    (an LFS blob is stored under its sha256). None when the repo is not cached."""
    hf_home = hf_home or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    blobs = os.path.join(hf_home, "hub", f"models--projectsidewalk--sidewalk-validator-ai-{label_type.lower()}",
                         "blobs")
    if not os.path.isdir(blobs):
        return None
    return os.path.exists(os.path.join(blobs, weights_sha))


def provenance_path(out):
    return re.sub(r"\.csv$", "", out) + "_provenance.json"


def score_is_current(out, ckpt_sha, manifest_sha):
    """True when ``out`` exists and its provenance records the same weights and manifest (the requeue skip rule)."""
    p = provenance_path(out)
    if not (os.path.exists(out) and os.path.exists(p)):
        return False
    with open(p) as f:
        prov = json.load(f)
    return prov.get("model_safetensors_sha256") == ckpt_sha and prov.get("manifest_sha256") == manifest_sha


def score_gold(manifest_path, checkpoint, tag, out, label_type=None, infer=None):
    """Score the gold manifest with ``checkpoint`` (``evaluate.run_inference``, split ``gold``) into ``out``
    (``label_uid, p_correct, p_incorrect``) and ``<out>_provenance.json``. Returns ``"skipped"``, ``"empty"`` or
    ``"scored"``. ``infer`` replaces ``run_inference`` in tests."""
    missing = [f for f in WEIGHT_FILES if not os.path.isfile(os.path.join(checkpoint, f))]
    if missing:
        raise FileNotFoundError(f"{checkpoint} lacks {missing}")
    ckpt_sha = sha256_file(os.path.join(checkpoint, "model.safetensors"))
    man_sha = sha256_file(manifest_path)
    if score_is_current(out, ckpt_sha, man_sha):
        print(f"{out}: already scored with these weights and this manifest; skipping")
        return "skipped"
    man = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    side_path = re.sub(r"\.csv$", ".json", manifest_path)
    side = {}
    if os.path.exists(side_path):
        with open(side_path) as f:
            side = json.load(f)
    cols = ["label_uid", "p_correct", "p_incorrect"]
    if man.empty:
        pred = pd.DataFrame(columns=cols)
        status = "empty"
    else:
        if infer is None:
            import evaluate
            infer = evaluate.run_inference
        pred = infer(man, checkpoint, splits=(GOLD_SPLIT,))[cols]
        if len(pred) != len(man):
            raise RuntimeError(f"scored {len(pred)} of {len(man)} manifest rows")
        status = "scored"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    pred.sort_values("label_uid", kind="mergesort").to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, out)
    prov = {"tag": tag, "checkpoint": os.path.abspath(checkpoint), "model_safetensors_sha256": ckpt_sha,
            "manifest": os.path.abspath(manifest_path), "manifest_sha256": man_sha, "n_scored": int(len(pred)),
            "n_left_out_no_crop": side.get("n_no_crop"), "crops": side.get("mode"),
            "hf_cache_published_match": hf_cache_match(ckpt_sha, label_type) if label_type else None,
            "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **git_state(REPO_DIR), **slurm_state()}
    write_json(prov, provenance_path(out))
    print(f"{status} {len(pred)} gold items with {checkpoint} -> {out}")
    return status


def cmd_score_gold(args):
    score_gold(args.manifest, args.checkpoint, args.tag, args.out, args.label_type)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# storage (login-node preflight)
# ---------------------------------------------------------------------------------------------------------------------

def parse_hyakstorage(text):
    """``(used_gb, total_gb, used_files, total_files)`` of the ``Total:`` row of ``hyakstorage <fs>``.

    >>> parse_hyakstorage("│ Total:  │ 969GB / 1024GB  │ 974033 / 1000000 files │")
    (969.0, 1024.0, 974033, 1000000)
    """
    for line in text.splitlines():
        if "Total:" in line:
            d = re.search(r"([\d.]+)\s*GB\s*/\s*([\d.]+)\s*GB", line)
            n = re.search(r"(\d+)\s*/\s*(\d+)", line[d.end():] if d else line)
            if d and n:
                return float(d[1]), float(d[2]), int(n[1]), int(n[2])
    raise ValueError("no 'Total:' row with GB and file counts in the hyakstorage output")


def cmd_storage(args):
    used, total, files, max_files = parse_hyakstorage(sys.stdin.read())
    free_gb, free_files = total - used, max_files - files
    print(f"{args.name}: {free_gb:.0f} GB and {free_files:,} files free")
    if free_gb < args.min_gb or free_files < args.min_files:
        raise SystemExit(f"{args.name}: under {args.min_gb} GB or {args.min_files:,} files free; not submitting")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# verify-smoke
# ---------------------------------------------------------------------------------------------------------------------

def verify_smoke(smoke, label_type="Crosswalk", tag="smoke", crops="r3crops"):
    """Check the smoke run; returns the list of ``(check, ok, detail)`` and stops at the first failure."""
    r4, runs = os.path.join(smoke, "data"), os.path.join(smoke, "runs")
    run = os.path.join(runs, label_type, tag)
    tdir = os.path.join(r4, label_type)
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        return bool(ok)

    prep_path = os.path.join(tdir, "r4_prep.json")
    if not check("prep job wrote r4_prep.json (round4_cpu.slurm STEP=prep)", os.path.exists(prep_path), prep_path):
        return results
    prep = json.load(open(prep_path))
    if not check("prep recorded drops by reason", "dropped" in prep and prep.get("n_train_after", 0) > 0,
                 str(prep.get("dropped"))):
        return results
    tr_path = os.path.join(run, "training_run.json")
    if not check("training_run.json", os.path.exists(tr_path), tr_path):
        return results
    tr = json.load(open(tr_path))
    if not check("clean git state at training", tr.get("git_dirty") is False and tr.get("git_commit"),
                 f"{tr.get('git_commit')} dirty={tr.get('git_dirty')}"):
        return results
    split_path = tr.get("split_path") or ""
    if not check("training split sha256 matches the variant file", os.path.exists(split_path)
                 and sha256_file(split_path) == tr.get("split_sha256"), split_path):
        return results
    for name in (f"report_{tag}.txt", f"metrics_{tag}.json"):
        if not check(name, os.path.exists(os.path.join(tdir, name))):
            return results
    best = os.path.join(run, "best")
    for f in WEIGHT_FILES:
        if not check(f"best/{f}", os.path.isfile(os.path.join(best, f))):
            return results
    if not check("no checkpoint-* left", not glob.glob(os.path.join(run, "checkpoint-*"))):
        return results
    pred = os.path.join(tdir, f"gold_predictions_{crops}_{tag}.csv")
    if not check("gold prediction file non-empty", os.path.exists(pred) and len(pd.read_csv(pred)) > 0, pred):
        return results
    prov = json.load(open(provenance_path(pred)))
    if not check("gold provenance sha256 = best/model.safetensors",
                 prov.get("model_safetensors_sha256") == sha256_file(os.path.join(best, "model.safetensors"))):
        return results
    ref = os.path.join(tdir, f"gold_predictions_{crops}_round3.csv")
    if not check("round-3 reference scored", os.path.exists(ref) and len(pd.read_csv(ref)) > 0, ref):
        return results
    try:
        cl.check_eval_rows_unchanged(cl.read_split(prep["source_split"]), cl.read_split(os.path.join(tdir,
                                                                                                    "split.csv")))
        ok = True
    except AssertionError:
        ok = False
    check("mirror eval rows equal round 3's", ok)
    return results


def cmd_verify_smoke(args):
    results = verify_smoke(args.smoke, args.label_type, args.tag)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return 0 if results and all(ok for _, ok, _ in results) else 1


# ---------------------------------------------------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------------------------------------------------

def parse_expect(specs, types):
    """``["baseline:42,43,44", "cl-drop:42,43,44:CurbRamp,Crosswalk"]`` -> ``[(type, variant, seed, tag)]``. The
    optional third field limits the types; the tag follows the Slurm convention (``variant_name``).

    >>> parse_expect(["cl-drop:42,43:Crosswalk"], ["Crosswalk", "Obstacle"])
    [('Crosswalk', 'cl-drop', 42, 'cl-drop'), ('Crosswalk', 'cl-drop', 43, 'cl-drop-s43')]
    """
    out = []
    for spec in specs:
        parts = spec.split(":")
        variant, seeds = parts[0], [int(s) for s in parts[1].split(",")]
        only = parts[2].split(",") if len(parts) > 2 else types
        for t in types:
            if t in only:
                out += [(t, variant, s, variant_name(variant, s)) for s in seeds]
    return out


def untagged_copies(tdir):
    """Write ``metrics.json`` / ``predictions.csv`` (what ``confident_learning.py report`` and ``seed_runs`` read as
    the seed-42 baseline) from ``metrics_baseline.json`` / ``predictions_baseline.csv`` **every time** (A2): a stale
    untagged file from an earlier collect, a manual copy or an older evaluation is replaced, never kept. Returns the
    sha256 of the source and of whatever the untagged name held before."""
    out = {}
    for src, dst in (("metrics_baseline.json", "metrics.json"), ("predictions_baseline.csv", "predictions.csv")):
        s, d = os.path.join(tdir, src), os.path.join(tdir, dst)
        if not os.path.exists(s):
            out[dst] = {"source": None}
            continue
        before = sha256_file(d) if os.path.exists(d) else None
        shutil.copyfile(s, d + ".tmp")
        os.replace(d + ".tmp", d)
        out[dst] = {"source_sha256": sha256_file(s), "previous_sha256": before, "written_sha256": sha256_file(d),
                    "replaced_different": before is not None and before != sha256_file(s)}
    return out


def read_jobs(path):
    """``{tag-ish job name: id}`` is not recoverable from ids alone; return the raw lines of ``round4_jobs.txt``."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def keep_plan(r4, runs, types, expected, extra=()):
    """``[(source, relative destination)]`` of the small artifacts that go to ``KEEP``."""
    plan = []
    for t in types:
        tdir = os.path.join(r4, t)
        seen = set()
        for pat in KEEP_DATA_PATTERNS + ("split.csv", "generation_run.json"):
            for p in sorted(glob.glob(os.path.join(tdir, pat))):
                if os.path.isfile(p) and p not in seen:
                    seen.add(p)
                    plan.append((p, os.path.join("data", t, os.path.relpath(p, tdir))))
    for t, _, _, tag in expected:
        for name in KEEP_RUN_FILES:
            p = os.path.join(runs, t, tag, name)
            if os.path.isfile(p):
                plan.append((p, os.path.join("runs", t, tag, name)))
    for p in extra:
        if p and os.path.isfile(p):
            plan.append((p, os.path.relpath(p, os.path.dirname(r4)) if under(p, os.path.dirname(r4))
                         else os.path.basename(p)))
    return plan


def is_gold_derived(rel):
    name = os.path.basename(rel)
    return name.startswith("gold_") or "gold_uids" in name


def cmd_collect(args):
    types = split_list(args.types)
    expected = parse_expect(args.expect, types)
    now = datetime.now(timezone.utc)
    runs_out, missing = [], []
    for t, variant, seed, tag in expected:
        tdir = os.path.join(args.r4, t)
        files = {"metrics": os.path.join(tdir, f"metrics_{tag}.json"),
                 "predictions": os.path.join(tdir, f"predictions_{tag}.csv"),
                 "weights": os.path.join(args.runs, t, tag, "best", "model.safetensors")}
        rec = {"label_type": t, "variant": variant, "seed": seed, "tag": tag,
               "present": all(os.path.exists(p) for p in files.values())}
        for k, p in files.items():
            rec[f"{k}_path"] = p
            rec[f"{k}_sha256"] = sha256_file(p) if os.path.exists(p) else None
        if os.path.exists(files["weights"]):
            mtime = datetime.fromtimestamp(os.path.getmtime(files["weights"]), timezone.utc)
            rec["weights_purge_after"] = (mtime + timedelta(days=SCRUBBED_PURGE_DAYS)).strftime("%Y-%m-%d")
        if not rec["present"]:
            missing.append(f"{t}/{tag}")
        runs_out.append(rec)
    copies = {t: untagged_copies(os.path.join(args.r4, t)) for t in types
              if any(e[0] == t and e[3] == "baseline" for e in expected)}
    manifest = {"written_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "r4": os.path.abspath(args.r4),
                "runs_dir": os.path.abspath(args.runs), "expected": len(expected), "missing": missing,
                "runs": runs_out, "untagged_baseline_copies": copies, "jobs": read_jobs(args.jobs),
                **git_state(REPO_DIR), **slurm_state()}
    man_path = write_json(manifest, os.path.join(args.r4, "round4_manifest.json"))
    plan = keep_plan(args.r4, args.runs, types, expected,
                     [man_path, args.jobs, args.gold_uids] + [p for p in glob.glob(os.path.join(args.r4, "gold", "*",
                                                                                            "*.json"))]
                     + glob.glob(os.path.join(args.r4, "slice_train", "*")) + list(args.extra or []))
    total = sum(os.path.getsize(p) for p, _ in plan)
    status = 0
    if total > args.max_bytes or len(plan) > args.max_files:
        print(f"refusing to copy to {args.keep}: {len(plan)} files, {total / 1e6:.1f} MB "
              f"(budget {args.max_files} files, {args.max_bytes / 1e6:.0f} MB)", file=sys.stderr)
        status = 3
    elif args.keep:
        for src, rel in plan:
            dst = os.path.join(args.keep, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            if is_gold_derived(rel):
                os.chmod(dst, PRIVATE_MODE)
        print(f"copied {len(plan)} files ({total / 1e6:.1f} MB) to {args.keep}")
    if missing:
        print(f"missing runs: {', '.join(missing)}", file=sys.stderr)
        status = status or 4
    print(f"wrote {man_path}: {len(expected) - len(missing)} of {len(expected)} expected runs present")
    return status


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    types_default = ",".join(LABEL_TYPES)

    g = sub.add_parser("gold-uids", help="the four-column gold uid file (Mac)")
    g.add_argument("--gold", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gold_uids)

    p = sub.add_parser("prep", help="gold-free base splits, crop links, r4_prep.json")
    p.add_argument("--data", required=True)
    p.add_argument("--r4", required=True)
    p.add_argument("--gold-uids", required=True)
    p.add_argument("--types", default=types_default)
    p.add_argument("--rawlabels", default=None)
    p.add_argument("--merge-radius-m", type=float, default=3.0)
    p.add_argument("--round3-ref", action="store_true")
    p.add_argument("--no-scene-ceiling", action="store_true", help="accept a gold_scene drop over the ceiling (Jon)")
    p.add_argument("--forbidden-root", action="append", default=None,
                   help=f"refuse an R4 under this root (repeatable; default {FORBIDDEN_ROOTS})")
    p.set_defaults(func=cmd_prep)

    v = sub.add_parser("variants", help="cl-drop and the rebalanced rule variants on the gold-free base")
    v.add_argument("--r4", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--types", default=",".join(CL_DROP_TYPES))
    v.add_argument("--which", default="cl-drop")
    v.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    v.set_defaults(func=cmd_variants)

    s = sub.add_parser("slice-train", help="training rows per severity / tag slice under each variant")
    s.add_argument("--r4", required=True)
    s.add_argument("--types", default=types_default)
    s.add_argument("--variants", default="round3,baseline,cl-drop")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_slice_train)

    m = sub.add_parser("gold-manifest", help="gold scoring manifests per type")
    m.add_argument("--gold-uids", required=True)
    m.add_argument("--split-root", default=None)
    m.add_argument("--crops-root", default=None)
    m.add_argument("--mode", choices=["r3crops", "uniform"], required=True)
    m.add_argument("--types", default=types_default)
    m.add_argument("--out", required=True, help="directory: <Type>.csv and <Type>.json")
    m.set_defaults(func=cmd_gold_manifest)

    r = sub.add_parser("round3-ckpt", help="print the resolved round-3 best checkpoint of a dinov2/ directory")
    r.add_argument("--dinov2", required=True)
    r.set_defaults(func=cmd_round3_ckpt)

    c = sub.add_parser("score-gold", help="score a gold manifest with a checkpoint (GPU)")
    c.add_argument("--manifest", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--label-type", default=None, help="for the descriptive published-revision match")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_score_gold)

    st = sub.add_parser("storage", help="parse hyakstorage output on stdin; refuse when nearly full")
    st.add_argument("--name", default="filesystem")
    st.add_argument("--min-gb", type=float, default=5.0)
    st.add_argument("--min-files", type=int, default=2000)
    st.set_defaults(func=cmd_storage)

    w = sub.add_parser("verify-smoke", help="the smoke run's checks")
    w.add_argument("--smoke", required=True)
    w.add_argument("--label-type", default="Crosswalk")
    w.add_argument("--tag", default="smoke")
    w.set_defaults(func=cmd_verify_smoke)

    k = sub.add_parser("collect", help="run manifest, untagged baseline copies, small artifacts to KEEP")
    k.add_argument("--r4", required=True)
    k.add_argument("--runs", required=True)
    k.add_argument("--keep", default=None)
    k.add_argument("--types", default=types_default)
    k.add_argument("--expect", nargs="+", required=True, help="variant:seeds[:types], e.g. baseline:42,43,44")
    k.add_argument("--jobs", default=None, help="round4_jobs.txt")
    k.add_argument("--gold-uids", default=None)
    k.add_argument("--extra", nargs="*", default=None, help="more small files to keep (e.g. submitted job files)")
    k.add_argument("--max-bytes", type=int, default=KEEP_MAX_BYTES)
    k.add_argument("--max-files", type=int, default=KEEP_MAX_FILES)
    k.set_defaults(func=cmd_collect)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
