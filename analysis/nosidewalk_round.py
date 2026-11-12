"""NoSidewalk step (b): a masked expert round outside Seattle, scored first (issue #42, ``docs/nosidewalk-2026-11.md``).

Step (a) (``nosidewalk_gate.py``) met the serving bar only in aggregate and only on labels with a settled verdict.
This round draws labels from the **deployment population** of other cities instead: every human-placed NoSidewalk
label with a block face, whatever its votes. One expert rates about 250 of them in prod Gallery with the votes
masked, and the gate is read on those ratings.

Two-phase sampling, so the model can choose what is rated without biasing the estimates:

1. ``pool``: the eligible labels of each city in the design (never placed or voted on by the rater, author not
   excluded, usable pano point, no face or panorama among the rows the fold models trained on) and a seeded
   simple random sample of them, **phase 1**, to crop and score on klone (``ai_labels_eval.py crops
   --fixed-crop-half-px 1024`` and ``score``, one prediction file per fold model; ``nosidewalk_round.slurm``).
2. ``build``: every phase-1 label gets each fold model's served accuracy under the map fitted on all pooled
   out-of-fold predictions of the gate run (as the gate's held-out table). Strata: ``pass`` (at least
   ``pass_models`` of the models give Agree at served accuracy >= the gate), ``agree_below`` (not pass, median
   ``p_correct`` >= 0.5) and ``model_incorrect`` (the rest). **Phase 2** is a seeded sample per (city, stratum)
   cell, at most one label per face; a short cell is topped up from the same stratum of other cities. The key
   carries each item's inclusion weight, ``N_city / n_scored_city * M_cell / m_cell``, and is written in the
   ``gt_rounds.py`` form, so ``gt_rounds.py gallery`` / ``votes`` / ``comments`` run on it unchanged.
3. ``report``: the rater's pinned votes against the key. The bar is read on the ``pass`` stratum as it was
   sampled: correct / decided with a Wilson 95 % lower bound >= the gate on >= 100 decided labels (Unsure votes
   leave the denominator and are counted). That rate is for the design's city mix (the per-city quotas), not the
   cities' population; the population-weighted rate is reported beside it and has a much smaller effective n. Beside it: the same rate reweighted to the cities' population, per
   city, per fold model (weighted precision, coverage and AUC), the population share of correct labels against
   the deployment prior, severity and tag slices, and the crowd's majority against the expert.

Nothing with label-level rows may be written inside a git checkout; the rater's user_id comes from the
``RATER_ID`` environment variable and is never written anywhere.

Usage (``D`` outside the repo; the full run order is in ``docs/nosidewalk-2026-11.md``)::

    D=~/Data/sidewalk-validator-ai-nosidewalk-2026-11/round-b
    python analysis/nosidewalk_round.py pull --design analysis/gt_rounds/nosidewalk_b.json --out $D/rawlabels
    ssh makelab1 "SCHEMAS='sidewalk_chicago ...' bash -s" < analysis/pull_author_excluded.sh > $D/author_excluded.csv
    RATER_ID=<uuid> python analysis/nosidewalk_round.py pool --design analysis/gt_rounds/nosidewalk_b.json \\
        --rawlabels $D/rawlabels --author-excluded $D/author_excluded.csv --pilot-split <whole split.csv> \\
        --seed $SEED --out $D/pool
    # klone: stage, crops, score (nosidewalk_round.slurm), then copy manifest.csv and predictions_fold<k>.csv back
    python analysis/nosidewalk_round.py build --design analysis/gt_rounds/nosidewalk_b.json --pool $D/pool \\
        --manifest $D/scored/manifest.csv --predictions $D/scored --gate <gate2 dir> --priors <deployment_priors.json> \\
        --seed $SEED --key-dir $D/private-key-do-not-share
    python analysis/nosidewalk_round.py gallery --key $D/private-key-do-not-share/key.csv --out $D/gallery_jon
    python analysis/gt_rounds.py votes --key ... --rater jon --rater-id <uuid> --out $D/votes_jon.csv
    python analysis/nosidewalk_round.py report --key ... --sheet $D/votes_jon.csv --scores .../scores.csv \\
        --out-json docs/nosidewalk-2026-11/round_b.json --out-md docs/nosidewalk-2026-11/round_b.md \\
        --doc docs/nosidewalk-2026-11.md
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generation as dg  # noqa: E402
import nosidewalk_faces  # noqa: E402
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis import ai_labels_eval as ale  # noqa: E402
from analysis import nosidewalk_gate as ng  # noqa: E402
from analysis import slices  # noqa: E402
from analysis.nosidewalk_pilot import without_nan  # noqa: E402
from calibration import _isotonic_map as im  # noqa: E402
from calibration import calibrate as cal  # noqa: E402

LABEL_TYPE = "NoSidewalk"
STRATA = ("pass", "agree_below", "model_incorrect")
POOLED = ("train", "val", "test")
START, END = "<!-- round-b-results:start -->", "<!-- round-b-results:end -->"
MIN_DECIDED = 100  # the bar's count of decided labels in the pass stratum
KEY_COLUMNS = ["packet_id", "label_uid", "label_uid_city", "schema", "city_id", "label_id", "label_type", "rater",
               "shared", "stratum", "weight", "face_id", "n_pass_models", "median_p_correct", "h_agree", "h_dis",
               "severity", "tags", "pano_id"]
RATING_RULE_HTML = (
    "<ol>"
    "<li><strong>Agree</strong> (correct): there is no sidewalk on the labeled side of the street at that spot.</li>"
    "<li><strong>Disagree</strong> (incorrect): a sidewalk is there, or the label marks something else.</li>"
    "<li><strong>Unsure</strong>: only when the imagery cannot show it.</li>"
    "</ol>")


# ---------------------------------------------------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------------------------------------------------

def load_design(path_or_dict):
    """The round's design, checked. ``cities`` maps city_id to its phase-1 and phase-2 sizes; ``shares`` splits each
    city's phase 2 over the strata.

    >>> d = load_design({"rater": "jon", "cities": {"a": {"phase1": 40, "phase2": 10}}, "gate": 0.92,
    ...                  "shares": {"pass": 0.6, "agree_below": 0.2, "model_incorrect": 0.2}, "pass_models": 3})
    >>> quotas(d)
    {('a', 'pass'): 6, ('a', 'agree_below'): 2, ('a', 'model_incorrect'): 2}
    """
    d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
    if set(d["shares"]) != set(STRATA) or abs(sum(d["shares"].values()) - 1) > 1e-9:
        raise ValueError(f"shares must name exactly {STRATA} and sum to 1")
    for c, n in d["cities"].items():
        if not 0 < int(n["phase2"]) <= int(n["phase1"]):
            raise ValueError(f"{c}: need 0 < phase2 <= phase1")
    if not (0 < float(d["gate"]) < 1 and int(d["pass_models"]) >= 1 and d["rater"]):
        raise ValueError("gate, pass_models and rater are required")
    return d


def largest_remainder(total, shares):
    """``total`` split over ``shares`` (name -> fraction) as integers that sum to ``total``.

    >>> largest_remainder(55, {"pass": 0.6, "agree_below": 0.2, "model_incorrect": 0.2})
    {'pass': 33, 'agree_below': 11, 'model_incorrect': 11}
    """
    raw = {k: total * v for k, v in shares.items()}
    out = {k: int(np.floor(v + 1e-9)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), list(shares).index(k)))[:total - sum(out.values())]:
        out[k] += 1
    return out


def quotas(design):
    """``(city, stratum) -> n`` for phase 2."""
    out = {}
    for c, n in design["cities"].items():
        for s, k in largest_remainder(int(n["phase2"]), {s: design["shares"][s] for s in STRATA}).items():
            out[(c, s)] = k
    return out


# ---------------------------------------------------------------------------------------------------------------------
# pull and pool (phase 1)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_pull(args):
    """A fresh NoSidewalk-only rawLabels export per design city (``fetch_all.fetch_city``), refusing a city the live
    listing does not show as public: per-label files of a private deployment must not be mixed in unnoticed."""
    import requests
    from analysis import fetch_all as fa
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write rawLabels inside a git checkout: {args.out}")
    design = load_design(args.design)
    listing = requests.get(fa.CITIES_API, timeout=60).json()["cities"]
    public = {c["city_id"]: c for c in listing if c["visibility"] == "public" and c["url"]}
    missing = sorted(set(design["cities"]) - set(public))
    if missing:
        sys.exit(f"not public in the live listing: {missing}")
    os.makedirs(args.out, exist_ok=True)
    for cid in design["cities"]:
        text, n = fa.fetch_city(requests.get, public[cid], LABEL_TYPE)
        if text is None:
            sys.exit(f"{cid}: the whole-city export did not come back complete; rerun")
        with open(os.path.join(args.out, f"{cid}.csv"), "w") as f:
            f.write(text)
        print(f"{cid}: {n} {LABEL_TYPE} labels")
    json.dump({"fetched_on": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "url": fa.CITIES_API,
               "label_types": [LABEL_TYPE], "cities": [public[c] for c in design["cities"]]},
              open(os.path.join(args.out, "cities.json"), "w"), indent=1)


def rater_touched(df, rater_id):
    """``(placed, voted)`` masks: labels the rater placed, and labels the rater cast any vote on (Unsure included,
    whatever the vote's ``validator_type``).

    >>> d = pd.DataFrame({"user_id": ["me", "x", "x"], "validations": [
    ...     "[]", '[{"user_id":"me","validation":"Unsure","validator_type":"Human"}]', float("nan")]})
    >>> [m.tolist() for m in rater_touched(d, "me")]
    [[True, False, False], [False, True, False]]
    """
    def voted(cell):
        return isinstance(cell, str) and bool(cell) and any(str(v.get("user_id")) == rater_id for v in json.loads(cell))
    return (df["user_id"].astype(str) == rater_id), df["validations"].map(voted)


def eligible_pool(raw, excluded, trained, rater_id):
    """The eligible labels of one city's rawLabels rows (``city`` and ``schema`` columns set) and the count each
    rule removed, applied in this order (a label is counted under the first rule that removes it).

    ``excluded`` is the set of ``schema:label_id`` whose author is excluded; ``trained`` a ``(faces, panos)`` pair of
    sets from the rows the fold models trained on.
    """
    df = raw[raw["label_type"] == LABEL_TYPE].copy()
    df["label_uid"] = df["city"] + ":" + df["label_id"].astype(str)
    if not df["label_uid"].is_unique:
        raise ValueError("duplicate labels in the rawLabels export")
    df["face_id"] = nosidewalk_faces.face_id(df)
    df["author"] = df["user_id"].map(dg.label_author)
    geometry_ok, _ = dg.valid_geometry(df)
    placed, voted = rater_touched(df, rater_id)
    faces, panos = trained
    rules = [("ai_placed", df["author"] != "human"), ("no_face", df["face_id"] == ""),
             ("bad_geometry", ~geometry_ok),
             ("author_excluded", (df["schema"] + ":" + df["label_id"].astype(str)).isin(excluded)),
             ("rater_placed", placed), ("rater_voted", voted),
             ("trained_face_or_pano", df["face_id"].isin(faces) | df["pano_id"].astype(str).isin(panos))]
    keep, dropped = pd.Series(True, index=df.index), {"labels": int(len(df))}
    for name, bad in rules:
        hit = keep & bad
        dropped[name] = int(hit.sum())
        keep &= ~bad
    dropped["eligible"] = int(keep.sum())
    return df[keep].reset_index(drop=True), dropped


def ranked(df, seed):
    """``df`` ordered by the seeded rank of ``label_uid`` (``rl.seeded_rank``): any prefix is a simple random sample."""
    rank = df["label_uid"].map(lambda u: rl.seeded_rank(seed, u))
    return df.assign(rank=rank).sort_values("rank", kind="mergesort").reset_index(drop=True)


def cmd_pool(args):
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write the pool inside a git checkout: {args.out}")
    rater_id = os.environ.get("RATER_ID", "")
    rl._check_uuid(rater_id)
    design = load_design(args.design)
    schema_city, _ = rl.cityparams(open(args.cityparams).read())
    city_schema = {c: s for s, c in schema_city.items()}
    ex = pd.read_csv(args.author_excluded, dtype=str)
    excluded = set((ex["schema"] + ":" + ex["label_id"])[ex["author_excluded"] == "t"])
    split = pd.read_csv(args.pilot_split, dtype=str, usecols=["face_id", "pano_id", "split"])
    seen = split[split["split"].isin(POOLED)]
    trained = (set(seen["face_id"].dropna()) - {""}, set(seen["pano_id"]))
    frames, counts, inputs = [], {}, {}
    for city, n in design["cities"].items():
        path = os.path.join(args.rawlabels, f"{city}.csv")
        inputs[city] = rl.sha256_file(path)
        raw = pd.read_csv(path, low_memory=False, dtype={"pano_id": str, "label_id": str})
        if city_schema[city] not in set(ex["schema"]):
            sys.exit(f"{args.author_excluded} has no rows for {city_schema[city]}")
        raw = raw.assign(city=city, schema=city_schema[city])
        elig, dropped = eligible_pool(raw, excluded, trained, rater_id)
        elig = ranked(elig, args.seed)
        elig["phase1"] = (np.arange(len(elig)) < int(n["phase1"])).astype(int)
        dropped["phase1"] = int(elig["phase1"].sum())
        counts[city] = dropped
        frames.append(elig)
    pool = pd.concat(frames, ignore_index=True)
    votes = pool["validations"].map(dg.human_vote_counts)
    pool["human_agree"], pool["human_disagree"] = [v[0] for v in votes], [v[1] for v in votes]
    pool["n_votes"] = pool["human_agree"] + pool["human_disagree"]
    for col in ale.LABEL_COLUMNS:  # the columns ai_labels_eval.py crops reads; the unknown class stays empty
        if col not in pool:
            pool[col] = ""
    os.makedirs(args.out, exist_ok=True)
    cols = ale.LABEL_COLUMNS + ["schema", "face_id", "rank"]
    pool[["label_uid", "city", "schema", "face_id", "rank", "phase1"]].to_csv(
        os.path.join(args.out, "eligible.csv"), index=False)
    labels_path = os.path.join(args.out, "labels.csv")
    p1 = pool[pool["phase1"] == 1]
    p1[cols].to_csv(labels_path, index=False)
    panos = sorted({f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(p1["city"], p1["pano_id"])})
    with open(os.path.join(args.out, "stage_files.txt"), "w") as f:
        f.write("\n".join(panos) + "\n")
    summary = {"design": design, "design_sha256": rl.sha256_file(args.design), "rawlabels_sha256": inputs,
               "author_excluded_sha256": rl.sha256_file(args.author_excluded),
               "pilot_split_sha256": rl.sha256_file(args.pilot_split),
               "seed_sha256": hashlib.sha256(str(args.seed).encode()).hexdigest(), "cities": counts,
               "labels_sha256": rl.sha256_file(labels_path),
               "eligible_sha256": rl.sha256_file(os.path.join(args.out, "eligible.csv")), "panos": len(panos)}
    rl.write_manifest(os.path.join(args.out, "pool.json"), summary)
    print(json.dumps({"cities": counts, "panos": len(panos)}, indent=1))


# ---------------------------------------------------------------------------------------------------------------------
# build (phase 2)
# ---------------------------------------------------------------------------------------------------------------------

def score_table(labels, preds, api_map, gate, pass_models):
    """One row per scored phase-1 label: each model's ``p_correct`` and gate pass, the count of passing models, the
    median ``p_correct`` and the stratum. ``preds`` is ``{model: predictions frame}``; a label must be scored by
    every model.

    >>> lab = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "city": "a"})
    >>> one = {"correct": {"0.0": 0.5, "0.8": 0.95}, "incorrect": {"0.0": 0.5}}
    >>> p = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "p_correct": [0.9, 0.6, 0.2]})
    >>> p["p_incorrect"] = 1 - p.p_correct
    >>> score_table(lab, {0: p, 1: p}, one, 0.92, 2).stratum.tolist()
    ['pass', 'agree_below', 'model_incorrect']
    """
    out = labels.copy()
    for k, pred in sorted(preds.items()):
        if not pred["label_uid"].is_unique:
            raise ValueError(f"model {k}: a label was scored twice")
        s = ng.served_with_map(pred[["label_uid", "p_correct", "p_incorrect"]].assign(true=""), api_map)  # class unknown
        s[f"pass_{k}"] = ((s["verdict"] == "correct") & (s["served"] >= gate)).astype(int)
        s = s.rename(columns={"p_correct": f"p_correct_{k}", "served": f"served_{k}"})
        out = out.merge(s[["label_uid", f"p_correct_{k}", f"served_{k}", f"pass_{k}"]], on="label_uid", how="inner",
                        validate="one_to_one")
    ks = sorted(preds)
    out["n_pass_models"] = out[[f"pass_{k}" for k in ks]].sum(axis=1)
    out["median_p_correct"] = out[[f"p_correct_{k}" for k in ks]].median(axis=1)
    out["stratum"] = np.where(out["n_pass_models"] >= pass_models, "pass",
                              np.where(out["median_p_correct"] >= 0.5, "agree_below", "model_incorrect"))
    return out


def one_per_face(ordered):
    """The rows of ``ordered`` (already in sampling order) that are the first of their face."""
    return ordered[~ordered["face_id"].duplicated()]


def sample_phase2(scores, design, used_faces=()):
    """Phase 2: per (city, stratum) cell, the first ``quota`` labels in rank order with at most one per face over the
    whole sample; a short cell's shortfall goes to the same stratum of the cities with the most labels to spare.
    Returns ``(picked, shortfall)``; ``picked`` carries ``cell_scored`` (M) and ``cell_picked`` (m).

    >>> d = load_design({"rater": "jon", "gate": 0.92, "pass_models": 1, "cities": {
    ...     "a": {"phase1": 9, "phase2": 5}, "b": {"phase1": 9, "phase2": 5}},
    ...     "shares": {"pass": 0.6, "agree_below": 0.2, "model_incorrect": 0.2}})
    >>> s = pd.DataFrame({"city": ["a"] * 9 + ["b"] * 9, "stratum": (["pass"] * 1 + ["agree_below"] * 4
    ...     + ["model_incorrect"] * 4) + (["pass"] * 7 + ["agree_below"] + ["model_incorrect"]), "rank": range(18)})
    >>> s["label_uid"], s["face_id"] = [f"u{i}" for i in range(18)], [f"f{i}" for i in range(18)]
    >>> p, short = sample_phase2(s, d)
    >>> p.groupby(["city", "stratum"]).size().to_dict(), short
    ({('a', 'agree_below'): 1, ('a', 'model_incorrect'): 1, ('a', 'pass'): 1, ('b', 'agree_below'): 1, ('b', 'model_incorrect'): 1, ('b', 'pass'): 5}, {'a/pass': 2})
    """
    want = quotas(design)
    s = scores.sort_values("rank", kind="mergesort")
    s = s[~s["face_id"].isin(set(used_faces))]
    s = one_per_face(s)  # a face is used once, in the stratum of its first-ranked label
    cells = {k: g for k, g in s.groupby(["city", "stratum"], sort=False)}
    take, shortfall = {}, {}
    for key, q in want.items():
        have = len(cells.get(key, ()))
        take[key] = min(q, have)
        if have < q:
            shortfall[f"{key[0]}/{key[1]}"] = q - have
    for stratum in STRATA:  # top up from the same stratum elsewhere, one label at a time from the largest spare
        need = sum(v for k, v in shortfall.items() if k.endswith("/" + stratum))
        while need > 0:
            spare = {k: len(cells[k]) - take[k] for k in take if k[1] == stratum and k in cells
                     and len(cells[k]) > take[k]}
            if not spare:
                break
            best = max(spare, key=lambda k: (spare[k], k[0]))
            take[best] += 1
            need -= 1
    picked = [cells[k].head(n).assign(cell_picked=n) for k, n in take.items() if n]
    picked = pd.concat(picked, ignore_index=True)
    scored = scores.groupby(["city", "stratum"]).size()
    picked["cell_scored"] = [int(scored[(c, t)]) for c, t in zip(picked["city"], picked["stratum"])]
    return picked, shortfall


def cmd_build(args):
    if rl.inside_git_checkout(args.key_dir):
        sys.exit(f"refusing to write the key inside a git checkout: {args.key_dir}")
    kpath = os.path.join(args.key_dir, "key.csv")
    if os.path.exists(kpath):
        sys.exit(f"{kpath} exists: rebuilding would replace the key and its rating pins; build into a new key dir")
    design = load_design(args.design)
    pool_json = json.load(open(os.path.join(args.pool, "pool.json")))
    if pool_json["seed_sha256"] != hashlib.sha256(str(args.seed).encode()).hexdigest():
        sys.exit("--seed is not the seed the pool was drawn with")
    labels_path = os.path.join(args.pool, "labels.csv")
    if rl.sha256_file(labels_path) != pool_json["labels_sha256"]:
        sys.exit(f"{labels_path} is not the file pool.json pins")
    labels = pd.read_csv(labels_path, dtype={"label_id": str, "pano_id": str, "tags": str}, low_memory=False)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    ok = set(manifest.loc[manifest["status"] == "ok", "label_uid"])
    pooled, _, n_folds, missing = ng.load_predictions(args.gate)
    if missing:
        sys.exit(f"the gate run is incomplete: {missing}")
    pi, _ = cal.load_prior(args.priors, LABEL_TYPE)
    api_map = ng.fit_map(pooled, pi)
    preds, pred_shas = {}, {}
    for k in range(n_folds):
        path = os.path.join(args.predictions, f"predictions_fold{k}.csv")
        preds[k], pred_shas[k] = pd.read_csv(path, dtype={"label_uid": str}), rl.sha256_file(path)
        if set(preds[k]["label_uid"]) != ok:
            sys.exit(f"{path} does not cover exactly the manifest's ok crops")
    scores = score_table(labels[labels["label_uid"].isin(ok)], preds, api_map, float(design["gate"]),
                         int(design["pass_models"]))
    picked, shortfall = sample_phase2(scores, design)
    eligible_n = {c: v["eligible"] for c, v in pool_json["cities"].items()}
    scored_n = scores.groupby("city").size().to_dict()
    picked["weight"] = [eligible_n[c] / scored_n[c] * m / n
                        for c, m, n in zip(picked["city"], picked["cell_scored"], picked["cell_picked"])]
    cid = picked["city"]
    picked = picked.assign(label_uid_city=picked["label_uid"], city_id=cid, label_type=LABEL_TYPE,
                           label_uid=picked["schema"] + ":" + picked["label_id"].astype(str),
                           rater=design["rater"], shared=False, h_agree=picked["human_agree"],
                           h_dis=picked["human_disagree"])
    key = rl.make_key(picked, args.seed)
    os.makedirs(args.key_dir, exist_ok=True)
    key[KEY_COLUMNS].to_csv(kpath, index=False)
    spath = os.path.join(args.key_dir, "scores.csv")
    scores.drop(columns=[c for c in ("validations", "comments", "description", "user_id") if c in scores]).to_csv(
        spath, index=False)
    cells = {f"{c}/{s}": {"scored": int(((scores["city"] == c) & (scores["stratum"] == s)).sum()),
                          "picked": int(((key["city_id"] == c) & (key["stratum"] == s)).sum()),
                          "quota": q} for (c, s), q in quotas(design).items()}
    man = {"design": os.path.splitext(os.path.basename(args.design))[0], "design_json": design,
           "design_sha256": rl.sha256_file(args.design), "pool_json_sha256": rl.sha256_file(
               os.path.join(args.pool, "pool.json")), "manifest_sha256": rl.sha256_file(args.manifest),
           "predictions_sha256": pred_shas, "gate_predictions_sha256": {
               k: rl.sha256_file(ng.predictions_path(args.gate, k)) for k in range(n_folds)},
           "prior": pi, "seed_sha256": pool_json["seed_sha256"], "key_sha256": rl.sha256_file(kpath),
           "scores_sha256": rl.sha256_file(spath), "items": int(len(key)), "eligible": eligible_n,
           "scored": {c: int(n) for c, n in scored_n.items()}, "phase1": {c: v["phase1"] for c, v in pool_json["cities"].items()},
           "cells": cells, "shortfall": shortfall,
           "weight_sum": {c: float(key.loc[key["city_id"] == c, "weight"].sum()) for c in design["cities"]}}
    rl.write_manifest(os.path.join(args.key_dir, "manifest.json"), man)
    print(json.dumps({k: man[k] for k in ("items", "eligible", "scored", "cells", "shortfall")}, indent=1))
    print(f"key -> {kpath} (sha256 {man['key_sha256']})")


def cmd_gallery(args):
    """The rater's Gallery review lists, exactly as ``gt_rounds.py gallery`` writes and pins them, with this round's
    rating rule on the index page in place of the five-type rule."""
    from analysis import gt_rounds as gr
    man = json.load(open(args.manifest or rl.default_manifest(args.key)))
    old, gr.RATING_RULE_HTML = gr.RATING_RULE_HTML, RATING_RULE_HTML
    try:
        gr.cmd_gallery(argparse.Namespace(key=args.key, rater=man["design_json"]["rater"], out=args.out,
                                          cityparams=args.cityparams, manifest=args.manifest, repin=args.repin))
    finally:
        gr.RATING_RULE_HTML = old


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def rate(k, n):
    """``{k, n, rate, wilson95_lower}`` for ``k`` correct of ``n`` decided.

    >>> r = rate(147, 150); r["rate"], round(r["wilson95_lower"], 3)
    (0.98, 0.943)
    >>> rate(0, 0)["rate"] is None
    True
    """
    if not n:
        return {"k": 0, "n": 0, "rate": None, "wilson95_lower": None}
    return {"k": int(k), "n": int(n), "rate": k / n, "wilson95_lower": float(im.wilson_lower(k / n, n, z=ng.Z95))}


def weighted_share(df):
    """The weighted share of correct among the decided rows of ``df`` (NaN when none)."""
    d = df[df["verdict"].isin(["correct", "incorrect"])]
    return float(np.average(d["verdict"] == "correct", weights=d["weight"])) if len(d) else float("nan")


def boot_ci(df, stat, n_boot, seed):
    """95 % interval of ``stat(df)`` from resampling rows within each (city, stratum) cell, the sampling unit."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in df.groupby(["city_id", "stratum"])]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(stat(df.loc[idx]))
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [None, None]


def weighted_auc(d, score_col):
    """AUC of ``score_col`` for separating incorrect from correct on decided rows, weighted to the population."""
    y = (d["verdict"] == "incorrect").astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, 1 - d[score_col], sample_weight=d["weight"]))


def within_city_weighted_auc(d, score_col, min_n=10):
    """Per-city weighted AUCs averaged with weights (weighted positives x weighted negatives): ``(auc, cities)``."""
    aucs, ws = [], []
    for _, g in d.groupby("city_id"):
        y = (g["verdict"] == "incorrect")
        if len(g) < min_n or y.nunique() < 2:
            continue
        aucs.append(weighted_auc(g, score_col))
        ws.append(g.loc[y, "weight"].sum() * g.loc[~y, "weight"].sum())
    return (float(np.average(aucs, weights=ws)), len(aucs)) if aucs else (float("nan"), 0)


def crowd_majority(a, d):
    return "correct" if a > d else "incorrect" if d > a else ""


def build_report(m, design, prior_correct, n_boot=1000, seed=0, train=None):
    """Every table of the round from ``m``: the key joined with the rater's verdict and the per-model scores."""
    gate = float(design["gate"])
    decided = m[m["verdict"].isin(["correct", "incorrect"])]
    passed = decided[decided["stratum"] == "pass"]
    primary = rate(int((passed["verdict"] == "correct").sum()), len(passed))
    primary["bar_met"] = bool(primary["n"] >= MIN_DECIDED and primary["wilson95_lower"] >= gate)
    n_unsure = int(((m["stratum"] == "pass") & (m["verdict"] == "cant_tell")).sum())
    primary["unsure_as_incorrect"] = rate(primary["k"], primary["n"] + n_unsure)  # sensitivity, not the bar
    primary["weighted"] = weighted_share(passed)
    primary["weighted_ci95"] = boot_ci(passed, weighted_share, n_boot, seed) if len(passed) else [None, None]
    rep = {"items": int(len(m)), "unanswered": int((m["verdict"] == "").sum()),
           "cant_tell": int((m["verdict"] == "cant_tell").sum()), "gate": gate, "primary": primary,
           "prior_correct": prior_correct,
           "population_correct": {"share": weighted_share(decided),
                                  "ci95": boot_ci(decided, weighted_share, n_boot, seed) if len(decided) else [None, None]}}
    rep["strata"] = {s: dict(rate(int(((decided["stratum"] == s) & (decided["verdict"] == "correct")).sum()),
                                  int((decided["stratum"] == s).sum())),
                             items=int((m["stratum"] == s).sum()),
                             cant_tell=int(((m["stratum"] == s) & (m["verdict"] == "cant_tell")).sum()))
                     for s in STRATA}
    rep["cities"] = {}
    for c in design["cities"]:
        g, gp = decided[decided["city_id"] == c], passed[passed["city_id"] == c]
        rep["cities"][c] = {"pass": rate(int((gp["verdict"] == "correct").sum()), len(gp)),
                            "population_correct": weighted_share(g), "decided": int(len(g)),
                            "incorrect": int((g["verdict"] == "incorrect").sum())}
    rep["models"] = {}
    for k in sorted(int(c.split("_")[1]) for c in m.columns if c.startswith("pass_")):
        gk = decided[decided[f"pass_{k}"] == 1]
        auc_w, n_c = within_city_weighted_auc(decided, f"p_correct_{k}")
        rep["models"][str(k)] = dict(
            rate(int((gk["verdict"] == "correct").sum()), len(gk)), precision_weighted=weighted_share(gk),
            coverage_weighted=float(gk["weight"].sum() / decided["weight"].sum()) if len(decided) else float("nan"),
            auc_weighted=weighted_auc(decided, f"p_correct_{k}"), within_city_auc_weighted=auc_w, auc_cities=n_c)
    rep["slices"] = {}
    for by in slices.BY:
        d = slices.with_slice(decided, by)
        col = "severity" if by == "severity" else "tags"
        tr = (slices.with_slice(train, by)["slice"].value_counts().to_dict()
              if train is not None and col in train else {})
        rep["slices"][by] = {str(name): dict(rate(int(((g["stratum"] == "pass") & (g["verdict"] == "correct")).sum()),
                                                  int((g["stratum"] == "pass").sum())),
                                             decided=int(len(g)), population_correct=weighted_share(g),
                                             train_rows=int(tr.get(name, 0)))
                             for name, g in d.groupby("slice")}
    crowd = [crowd_majority(a, d) for a, d in zip(decided["h_agree"], decided["h_dis"])]
    tab = pd.crosstab(pd.Series(crowd, name="crowd").replace("", "none or tie"), decided["verdict"].to_numpy())
    rep["crowd_vs_expert"] = {r: {c: int(tab.loc[r, c]) for c in tab.columns} for r in tab.index}
    return rep


def _f(x, digits=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def _rate_cell(r):
    return "n/a" if not r["n"] else f"{r['k']} / {r['n']} = {_f(r['rate'])} ({_f(r['wilson95_lower'])})"


def markdown(rep):
    p, pc = rep["primary"], rep["population_correct"]
    lines = [
        f"{rep['items']} items rated ({rep['cant_tell']} Unsure, {rep['unanswered']} unanswered). Precision cells are "
        f"correct / decided = rate (Wilson 95 % lower bound); Unsure votes are outside the denominator.", "",
        f"**Bar:** in the pass stratum, {_rate_cell(p)}; a lower bound >= {rep['gate']} on >= {MIN_DECIDED} decided "
        f"labels is {'met' if p['bar_met'] else 'not met'}. **{'PASSES' if p['bar_met'] else 'DOES NOT PASS'}.** "
        f"Reweighted to the cities' population: {_f(p['weighted'])} [{_f(p['weighted_ci95'][0])}, "
        f"{_f(p['weighted_ci95'][1])}] (the interval holds the weights fixed, so it is somewhat narrow). With every "
        f"Unsure in the pass stratum counted as incorrect: {_rate_cell(p['unsure_as_incorrect'])}.", "",
        f"Share of correct NoSidewalk labels in these cities' eligible population: {_f(pc['share'])} "
        f"[{_f(pc['ci95'][0])}, {_f(pc['ci95'][1])}] (the deployment prior assumes {_f(rep['prior_correct'])} over "
        f"every label with a face; this population leaves out the rater's own labels and trained faces).", "",
        "| stratum | items | Unsure | correct / decided (Wilson 95 % lower) |", "|---|---|---|---|"]
    lines += [f"| {s} | {r['items']} | {r['cant_tell']} | {_rate_cell(r)} |" for s, r in rep["strata"].items()]
    lines += ["", "| city | decided (incorrect) | pass stratum (Wilson 95 % lower) | population share correct |",
              "|---|---|---|---|"]
    lines += [f"| {c} | {r['decided']} ({r['incorrect']}) | {_rate_cell(r['pass'])} | {_f(r['population_correct'])} |"
              for c, r in rep["cities"].items()]
    lines += ["", "Per fold model, on its own gate verdicts (weights are the items' inverse inclusion probabilities):",
              "", "| model | gate passes, rated items unweighted (Wilson 95 % lower) | precision, weighted | coverage, weighted | AUC, "
              "weighted | within-city AUC, weighted (cities) |", "|---|---|---|---|---|---|"]
    lines += [f"| fold {k} | {_rate_cell(r)} | {_f(r['precision_weighted'])} | {_f(r['coverage_weighted'])} | "
              f"{_f(r['auc_weighted'])} | {_f(r['within_city_auc_weighted'])} ({r['auc_cities']}) |"
              for k, r in rep["models"].items()]
    for by, title in (("severity", "severity"), ("tag", "tag")):
        lines += ["", f"| {title} | decided | pass stratum (Wilson 95 % lower) | population share correct | "
                  "training rows |", "|---|---|---|---|---|"]
        lines += [f"| {name} | {r['decided']} | {_rate_cell(r)} | {_f(r['population_correct'])} | {r['train_rows']} |"
                  for name, r in rep["slices"][by].items()]
    lines += ["", "Crowd majority (human votes at the pool pull) against the expert, decided items:", "",
              "| crowd | expert correct | expert incorrect |", "|---|---|---|"]
    lines += [f"| {r} | {v.get('correct', 0)} | {v.get('incorrect', 0)} |" for r, v in rep["crowd_vs_expert"].items()]
    return "\n".join(lines) + "\n"


def splice(doc_text, block):
    """Replace what lies between the round markers of ``doc_text`` with ``block``; ValueError without markers.

    >>> splice("a\\n<!-- round-b-results:start -->\\nold\\n<!-- round-b-results:end -->\\nb", "new\\n")
    'a\\n<!-- round-b-results:start -->\\nnew\\n<!-- round-b-results:end -->\\nb'
    """
    i, j = doc_text.find(START), doc_text.find(END)
    if i < 0 or j < i:
        raise ValueError(f"no {START} ... {END} markers in the document")
    return doc_text[:i + len(START)] + "\n" + block + doc_text[j:]


def cmd_report(args):
    mpath = args.manifest or rl.default_manifest(args.key)
    man = json.load(open(mpath))
    rl.check_pin(mpath, "key", args.key, args.allow_unpinned)
    rl.check_pin(mpath, f"votes_{man['design_json']['rater']}", args.sheet, args.allow_unpinned)
    if rl.sha256_file(args.scores) != man["scores_sha256"] and not args.allow_unpinned:
        sys.exit(f"{args.scores} is not the scores file the manifest pins")
    key = pd.read_csv(args.key, dtype={"tags": str, "label_id": str})
    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    m = rl.unmask(key, sheet)
    scores = pd.read_csv(args.scores, dtype={"label_uid": str}, low_memory=False)
    model_cols = [c for c in scores.columns if c.startswith(("pass_", "p_correct_", "served_"))]
    m = m.merge(scores[["label_uid"] + model_cols].rename(columns={"label_uid": "label_uid_city"}),
                on="label_uid_city", how="left", validate="one_to_one")
    train = None
    if args.pilot_split:
        sp = pd.read_csv(args.pilot_split, dtype=str, keep_default_na=False)
        train = sp[sp["split"].isin(POOLED)]
    rep = build_report(m, man["design_json"], float(man["prior"]["correct"]), args.n_boot, args.seed, train)
    rep["provenance"] = {"key_sha256": man["key_sha256"], "sheet_sha256": rl.sha256_file(args.sheet),
                         "scores_sha256": man["scores_sha256"], "n_boot": args.n_boot, "seed": args.seed}
    md = markdown(rep)
    for path, text in ((args.out_json, json.dumps(without_nan(rep), indent=1) + "\n"), (args.out_md, md)):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
    if args.doc:
        with open(args.doc) as f:
            doc = f.read()
        with open(args.doc, "w") as f:
            f.write(splice(doc, md))
    print(md)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("pull", help="fresh NoSidewalk rawLabels of the design's cities (public API)")
    u.add_argument("--design", required=True)
    u.add_argument("--out", required=True)
    u.set_defaults(func=cmd_pull)
    o = sub.add_parser("pool", help="eligible labels and the phase-1 sample to score (RATER_ID in the environment)")
    o.add_argument("--design", required=True)
    o.add_argument("--rawlabels", required=True, help="directory of <city_id>.csv from `pull`")
    o.add_argument("--author-excluded", required=True, help="pull_author_excluded.sh output for the cities' schemas")
    o.add_argument("--pilot-split", required=True, help="the split.csv the gate's folds were made from")
    o.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    o.add_argument("--seed", required=True, help="secret until the rating is done")
    o.add_argument("--out", required=True, help="outside any git checkout")
    o.set_defaults(func=cmd_pool)
    b = sub.add_parser("build", help="strata from the fold models' scores, the phase-2 sample, key and manifest")
    b.add_argument("--design", required=True)
    b.add_argument("--pool", required=True, help="the directory `pool` wrote")
    b.add_argument("--manifest", required=True, help="manifest.csv of ai_labels_eval.py crops")
    b.add_argument("--predictions", required=True, help="directory of predictions_fold<k>.csv (ai_labels_eval.py score)")
    b.add_argument("--gate", required=True, help="the gate run whose fold models scored the labels")
    b.add_argument("--priors", required=True, help="deployment_priors.json of the gate run")
    b.add_argument("--seed", required=True)
    b.add_argument("--key-dir", required=True, help="private; outside any git checkout")
    b.set_defaults(func=cmd_build)
    g = sub.add_parser("gallery", help="the rater's Gallery lists (gt_rounds.py gallery) with this round's rating rule")
    g.add_argument("--key", required=True)
    g.add_argument("--out", required=True, help="outside any git checkout")
    g.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    g.add_argument("--manifest")
    g.add_argument("--repin", action="store_true")
    g.set_defaults(func=cmd_gallery)
    r = sub.add_parser("report", help="the rater's pinned votes against the key: the bar and every table")
    r.add_argument("--key", required=True)
    r.add_argument("--sheet", required=True, help="gt_rounds.py votes output (pinned)")
    r.add_argument("--scores", required=True, help="scores.csv beside the key")
    r.add_argument("--pilot-split", help="the gate's split.csv, for training counts per severity and tag")
    r.add_argument("--manifest")
    r.add_argument("--allow-unpinned", action="store_true")
    r.add_argument("--out-json")
    r.add_argument("--out-md")
    r.add_argument("--doc")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
