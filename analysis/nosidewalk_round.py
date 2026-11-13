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


def crop_labels(rows):
    """``rows`` (rawLabels rows with ``label_uid``, ``city`` and ``author``) with the columns ``ai_labels_eval.py
    crops`` reads: the human vote counts, and every other ``ale.LABEL_COLUMNS`` column empty (the class is unknown).
    Returns a new frame."""
    rows = rows.copy()
    votes = rows["validations"].map(dg.human_vote_counts)
    rows["human_agree"], rows["human_disagree"] = [v[0] for v in votes], [v[1] for v in votes]
    rows["n_votes"] = rows["human_agree"] + rows["human_disagree"]
    for col in ale.LABEL_COLUMNS:
        if col not in rows:
            rows[col] = ""
    return rows


def write_stage_files(rows, path):
    """The store-relative panorama paths of ``rows`` for ``ai_labels_stage.slurm``, sorted and unique."""
    panos = sorted({f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(rows["city"], rows["pano_id"])})
    with open(path, "w") as f:
        f.write("\n".join(panos) + "\n")
    return panos


def read_city_inputs(args, design):
    """What ``pool`` and ``neighbours`` both read: ``(city -> schema, the schema:label_id set of labels whose author
    is excluded)``. Stops when the author-excluded pull has no rows for a city of the design."""
    schema_city, _ = rl.cityparams(open(args.cityparams).read())
    city_schema = {c: s for s, c in schema_city.items()}
    ex = pd.read_csv(args.author_excluded, dtype=str)
    for city in design["cities"]:
        if city_schema[city] not in set(ex["schema"]):
            sys.exit(f"{args.author_excluded} has no rows for {city_schema[city]}")
    excluded = set((ex["schema"] + ":" + ex["label_id"])[ex["author_excluded"] == "t"])
    return city_schema, excluded


def cmd_pool(args):
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write the pool inside a git checkout: {args.out}")
    rater_id = os.environ.get("RATER_ID", "")
    rl._check_uuid(rater_id)
    design = load_design(args.design)
    city_schema, excluded = read_city_inputs(args, design)
    split = pd.read_csv(args.pilot_split, dtype=str, usecols=["face_id", "pano_id", "split"])
    seen = split[split["split"].isin(POOLED)]
    trained = (set(seen["face_id"].dropna()) - {""}, set(seen["pano_id"]))
    frames, counts, inputs = [], {}, {}
    for city, n in design["cities"].items():
        path = os.path.join(args.rawlabels, f"{city}.csv")
        inputs[city] = rl.sha256_file(path)
        raw = pd.read_csv(path, low_memory=False, dtype={"pano_id": str, "label_id": str})
        raw = raw.assign(city=city, schema=city_schema[city])
        elig, dropped = eligible_pool(raw, excluded, trained, rater_id)
        elig = ranked(elig, args.seed)
        elig["phase1"] = (np.arange(len(elig)) < int(n["phase1"])).astype(int)
        dropped["phase1"] = int(elig["phase1"].sum())
        counts[city] = dropped
        frames.append(elig)
    pool = crop_labels(pd.concat(frames, ignore_index=True))
    os.makedirs(args.out, exist_ok=True)
    cols = ale.LABEL_COLUMNS + ["schema", "face_id", "rank"]
    pool[["label_uid", "city", "schema", "face_id", "rank", "phase1"]].to_csv(
        os.path.join(args.out, "eligible.csv"), index=False)
    labels_path = os.path.join(args.out, "labels.csv")
    p1 = pool[pool["phase1"] == 1]
    p1[cols].to_csv(labels_path, index=False)
    panos = write_stage_files(p1, os.path.join(args.out, "stage_files.txt"))
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


# ---------------------------------------------------------------------------------------------------------------------
# neighbours: every other label on the rated labels' faces (issue #102)
# ---------------------------------------------------------------------------------------------------------------------

NEIGHBOUR_FLAGS = ("rater_placed", "rater_voted", "author_excluded", "trained_pano", "pilot_heldout")
FACE_COLUMNS = ["label_uid", "city", "schema", "label_id", "face_id", "pano_id", "rated", "scored", "new", "ai_placed",
                "bad_geometry", *NEIGHBOUR_FLAGS, "same_labeler", "labeler_idx", "time_created", "latitude",
                "longitude", "severity", "tags"]


def face_neighbours(raw, rated_uids, scored_uids, excluded, trained_panos, heldout_uids, rater_id):
    """One city's NoSidewalk labels on the block faces of the rated labels, the rated labels included, one row each.

    Nothing is dropped for the eligibility rules of ``pool``: these labels are scored, not rated, so each rule is a
    flag column instead. ``new`` marks the labels to crop and score now (not scored in phase 1, human-placed, usable
    pano point). ``same_labeler`` is 1 when the label's author placed the face's rated label; ``labeler_idx`` numbers
    the face's labelers by their first label there, so a labeler's run on a face (issue #102) is
    ``(face_id, labeler_idx)`` and no ``user_id`` has to be written.

    >>> v = "[]"
    >>> raw = pd.DataFrame({"label_id": ["1", "2", "3", "4", "5"], "user_id": ["u1", "u2", "u1", "u1", "u3"],
    ...     "label_type": "NoSidewalk", "city": "a", "schema": "s", "street_edge_id": [7, 7, 7, 8, 7],
    ...     "street_side": ["left", "left", "left", "left", "right"], "pano_id": list("pqrst"), "validations": v,
    ...     "time_created": ["2026-01-03T00:00:00Z", "2026-01-01T00:00:00.5Z", "2026-01-01T00:00:00.12Z",
    ...                      "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "pano_source": "gsv", "pano_x": 1.0, "pano_y": 1.0, "pano_width": 9.0,
    ...     "pano_height": 9.0})
    >>> f = face_neighbours(raw, {"a:1"}, {"a:1"}, set(), {"q"}, set(), "me").set_index("label_uid")
    >>> f.index.tolist(), f["new"].tolist(), f["same_labeler"].tolist(), f["labeler_idx"].tolist()
    (['a:3', 'a:2', 'a:1'], [1, 1, 0], [1, 0, 1], [0, 1, 0])
    >>> f["trained_pano"].tolist()
    [0, 1, 0]
    """
    df = raw[raw["label_type"] == LABEL_TYPE].copy()
    df["label_uid"] = df["city"] + ":" + df["label_id"].astype(str)
    if not df["label_uid"].is_unique:
        raise ValueError("duplicate labels in the rawLabels export")
    df["face_id"] = nosidewalk_faces.face_id(df)
    df["rated"] = df["label_uid"].isin(rated_uids).astype(int)
    rated = df.loc[df["rated"] == 1, ["face_id", "user_id"]]
    if (rated["face_id"] == "").any() or rated["face_id"].duplicated().any():
        raise ValueError("a rated label has no face, or two rated labels share one")
    df = df[df["face_id"].isin(set(rated["face_id"]))].copy()
    if df["user_id"].isna().any():
        raise ValueError("a label on a rated face has no user_id: its labeler cannot be numbered")
    df["author"] = df["user_id"].map(dg.label_author)
    geometry_ok, _ = dg.valid_geometry(df)
    placed, voted = rater_touched(df, rater_id)
    df["scored"] = df["label_uid"].isin(scored_uids).astype(int)
    df["ai_placed"] = (df["author"] != "human").astype(int)
    df["bad_geometry"] = (~geometry_ok).astype(int)
    df["rater_placed"], df["rater_voted"] = placed.astype(int), voted.astype(int)
    df["author_excluded"] = (df["schema"] + ":" + df["label_id"].astype(str)).isin(excluded).astype(int)
    df["trained_pano"] = df["pano_id"].astype(str).isin(trained_panos).astype(int)
    df["pilot_heldout"] = df["label_uid"].isin(heldout_uids).astype(int)
    df["new"] = ((df["scored"] == 0) & (df["ai_placed"] == 0) & (df["bad_geometry"] == 0)).astype(int)
    rated_user = dict(zip(rated["face_id"], rated["user_id"].astype(str)))
    df["same_labeler"] = (df["user_id"].astype(str) == df["face_id"].map(rated_user)).astype(int)
    # Parsed, not as text: the export writes 0 to 3 fractional digits, and "Z" sorts after a digit.
    df["_t"] = pd.to_datetime(df["time_created"], utc=True, format="ISO8601")
    df["_id"] = pd.to_numeric(df["label_id"])
    df = df.sort_values(["face_id", "_t", "_id"], kind="mergesort").drop(columns=["_t", "_id"]).reset_index(drop=True)
    first = df.drop_duplicates(["face_id", "user_id"])
    idx = first.assign(labeler_idx=first.groupby("face_id").cumcount())[["face_id", "user_id", "labeler_idx"]]
    return df.merge(idx, on=["face_id", "user_id"], how="left", validate="many_to_one")


def cmd_neighbours(args):
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write label rows inside a git checkout: {args.out}")
    rater_id = os.environ.get("RATER_ID", "")
    rl._check_uuid(rater_id)
    design = load_design(args.design)
    city_schema, excluded = read_city_inputs(args, design)
    kman = json.load(open(os.path.join(args.key_dir, "manifest.json")))
    for name, want in (("key.csv", kman["key_sha256"]), ("scores.csv", kman["scores_sha256"])):
        if rl.sha256_file(os.path.join(args.key_dir, name)) != want:
            sys.exit(f"{name} is not the file the key's manifest pins")
    ppath = os.path.join(args.pool, "pool.json")  # the same pulls the round was drawn from, or faces and votes drift
    if rl.sha256_file(ppath) != kman["pool_json_sha256"]:
        sys.exit(f"{ppath} is not the pool the key was built from")
    pool_json = json.load(open(ppath))
    for path, want in ((args.author_excluded, pool_json["author_excluded_sha256"]),
                       (args.pilot_split, pool_json["pilot_split_sha256"]),
                       *((os.path.join(args.rawlabels, f"{c}.csv"), pool_json["rawlabels_sha256"][c])
                         for c in design["cities"])):
        if rl.sha256_file(path) != want:
            sys.exit(f"{path} is not the file the pool was drawn from")
    key = pd.read_csv(os.path.join(args.key_dir, "key.csv"), dtype=str)
    scores = pd.read_csv(os.path.join(args.key_dir, "scores.csv"), dtype=str, usecols=["label_uid"])
    split = pd.read_csv(args.pilot_split, dtype=str, usecols=["label_uid", "pano_id", "split"])
    trained_panos = set(split.loc[split["split"].isin(POOLED), "pano_id"])
    heldout = set(split.loc[split["split"] == "test_heldout", "label_uid"])
    frames, counts, inputs = [], {}, {}
    for city in design["cities"]:
        path = os.path.join(args.rawlabels, f"{city}.csv")
        inputs[city] = rl.sha256_file(path)
        raw = pd.read_csv(path, low_memory=False, dtype={"pano_id": str, "label_id": str})
        raw = raw.assign(city=city, schema=city_schema[city])
        rated = set(key.loc[key["city_id"] == city, "label_uid_city"])  # city:label_id; the key's label_uid is by schema
        f = face_neighbours(raw, rated, set(scores["label_uid"]), excluded, trained_panos, heldout, rater_id)
        if int(f["rated"].sum()) != len(rated):
            sys.exit(f"{city}: {len(rated) - int(f['rated'].sum())} rated labels are not in {path}")
        counts[city] = {"rated_faces": int(f["face_id"].nunique()), "face_labels": int(len(f)),
                        "scored": int(f["scored"].sum()),
                        "ai_placed": int(((f["scored"] == 0) & (f["ai_placed"] == 1)).sum()),
                        "bad_geometry": int(((f["scored"] == 0) & (f["ai_placed"] == 0)
                                             & (f["bad_geometry"] == 1)).sum()),
                        "new": int(f["new"].sum()), "same_labeler": int(f["same_labeler"].sum()),
                        "runs": int(len(f.drop_duplicates(["face_id", "labeler_idx"]))),
                        **{f"new_{flag}": int(((f["new"] == 1) & (f[flag] == 1)).sum()) for flag in NEIGHBOUR_FLAGS}}
        frames.append(f)
    faces = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out, exist_ok=True)
    fpath, lpath = os.path.join(args.out, "faces.csv"), os.path.join(args.out, "labels.csv")
    faces[FACE_COLUMNS].to_csv(fpath, index=False)
    new = crop_labels(faces[faces["new"] == 1].copy())
    new[ale.LABEL_COLUMNS + ["schema", "face_id"]].to_csv(lpath, index=False)
    panos = write_stage_files(new, os.path.join(args.out, "stage_files.txt"))
    summary = {"design_sha256": rl.sha256_file(args.design), "rawlabels_sha256": inputs,
               "author_excluded_sha256": rl.sha256_file(args.author_excluded),
               "pilot_split_sha256": rl.sha256_file(args.pilot_split), "key_sha256": kman["key_sha256"],
               "scores_sha256": kman["scores_sha256"], "cities": counts, "faces_sha256": rl.sha256_file(fpath),
               "labels_sha256": rl.sha256_file(lpath), "labels": int(len(new)), "panos": len(panos)}
    rl.write_manifest(os.path.join(args.out, "neighbours.json"), summary)
    print(json.dumps({"cities": counts, "labels": int(len(new)), "panos": len(panos)}, indent=1))


def read_fold_predictions(directory, n_folds, pins=None):
    """``{k: frame}`` of ``predictions_fold<k>.csv`` under ``directory`` and their sha256s; with ``pins`` (fold ->
    sha256, keys as strings) a file that is not the pinned one stops the run."""
    preds, shas = {}, {}
    for k in range(n_folds):
        path = os.path.join(directory, f"predictions_fold{k}.csv")
        preds[k], shas[str(k)] = pd.read_csv(path, dtype={"label_uid": str}), rl.sha256_file(path)
        if pins is not None and shas[str(k)] != pins[str(k)]:
            sys.exit(f"{path} is not the file the key's manifest pins")
    return preds, shas


def model_shas(directory, n_folds):
    """``{fold: sha256 of the weights}`` from the ``predictions_fold<k>_provenance.json`` files ``ai_labels_eval.py
    score`` writes beside each prediction file."""
    out = {}
    for k in range(n_folds):
        path = os.path.join(directory, f"predictions_fold{k}_provenance.json")
        if not os.path.exists(path):
            sys.exit(f"{path} is missing: cannot tell which weights made predictions_fold{k}.csv")
        out[str(k)] = json.load(open(path))["model_sha256"]
    return out


def pass_summary(df):
    """Counts and gate-pass shares of scored labels: by label, and as the mean over faces of each face's share (a few
    long faces hold a large part of the labels, so the two differ).

    >>> pass_summary(pd.DataFrame({"face_id": list("aaab"), "stratum": ["pass", "pass", "pass", "agree_below"]}))
    {'labels': 4, 'faces': 2, 'pass': 3, 'pass_share': 0.75, 'pass_share_per_face': 0.5}
    """
    if not len(df):
        return {"labels": 0, "faces": 0, "pass": 0, "pass_share": None, "pass_share_per_face": None}
    p = df["stratum"] == "pass"
    return {"labels": int(len(df)), "faces": int(df["face_id"].nunique()), "pass": int(p.sum()),
            "pass_share": float(p.mean()), "pass_share_per_face": float(p.groupby(df["face_id"]).mean().mean())}


def neighbour_report(face_scores, min_n=slices.MIN_N):
    """Aggregates of the scored labels on the rated faces (no label rows): per city, per flag, per severity and tag,
    and how often the model's own gate verdict is the same for every scored label of a face. That last table is
    about the **model's** verdict only; it says nothing on whether the labels of a face are right or wrong together
    (issue #102 item 1), which needs several rated labels per face."""
    s = face_scores[face_scores["stratum"].notna() & (face_scores["stratum"] != "")]
    rep = {"all": pass_summary(s), "cities": {c: pass_summary(g) for c, g in s.groupby("city")},
           "by": {"rated": pass_summary(s[s["rated"] == 1]), "not_rated": pass_summary(s[s["rated"] == 0]),
                  "same_labeler": pass_summary(s[(s["rated"] == 0) & (s["same_labeler"] == 1)]),
                  "other_labeler": pass_summary(s[(s["rated"] == 0) & (s["same_labeler"] == 0)]),
                  **{flag: pass_summary(s[s[flag] == 1]) for flag in NEIGHBOUR_FLAGS}},
           "unscored": {c: int(n) for c, n in face_scores[~face_scores.index.isin(s.index)].groupby("city").size().items()}}
    # The faces were chosen through their rated label, 60 % of them from the pass stratum, so the shares above are
    # of this sample and not of any population; the split by the rated label's stratum shows how far they follow it.
    rated_stratum = s[s["rated"] == 1].set_index("face_id")["stratum"]
    others = s[s["rated"] == 0]
    rep["not_rated_by_rated_stratum"] = {
        st: pass_summary(others[others["face_id"].map(rated_stratum) == st]) for st in STRATA}
    is_pass = (s["stratum"] == "pass").astype(int)
    multi = pd.DataFrame({"n": is_pass.groupby(s["face_id"]).size(), "passing": is_pass.groupby(s["face_id"]).sum(),
                          "city": s.groupby("face_id")["city"].first()})
    multi = multi[multi["n"] >= 2]
    mixed = multi[(multi["passing"] > 0) & (multi["passing"] < multi["n"])]
    minority = np.minimum(mixed["passing"], mixed["n"] - mixed["passing"]) / mixed["n"]
    p_city = multi["city"].map(is_pass.groupby(s["city"]).mean())  # the benchmark: verdicts independent within a city
    rep["faces_with_2plus_scored"] = {
        "faces": int(len(multi)), "all_pass": int((multi["passing"] == multi["n"]).sum()),
        "none_pass": int((multi["passing"] == 0).sum()), "mixed": int(len(mixed)),
        "mixed_expected_if_independent": float((1 - p_city ** multi["n"] - (1 - p_city) ** multi["n"]).sum()),
        "mixed_minority_at_most_20pct": int((minority <= 0.2).sum()),
        "median_labels_mixed": float(mixed["n"].median()) if len(mixed) else None,
        "two_label_faces": int((multi["n"] == 2).sum()), "two_label_faces_mixed": int((mixed["n"] == 2).sum())}
    rep["slices"] = {}
    for by in slices.BY:
        d = slices.with_slice(s, by)
        rep["slices"][by] = {name: pass_summary(d[d["slice"] == name]) for name in slices.slice_names(s, by, min_n)}
    return rep


def rater_touched_report(rated, face_scores, gate):
    """The round's rates on rated labels whose face carries a label the rater placed or voted on, and on the rest.
    The pool keeps the rater's own labels and votes out of the round, but not the faces they sit on, so the rater
    may know such a face. ``rated`` is the unmasked key (``face_id``, ``stratum``, ``verdict``); Unsure votes leave
    the denominator, as in the round's report.

    >>> r = pd.DataFrame({"face_id": list("abcd"), "stratum": ["pass"] * 3 + ["agree_below"],
    ...                   "verdict": ["correct", "incorrect", "correct", "cant_tell"]})
    >>> f = pd.DataFrame({"face_id": list("aabcd"), "rater_placed": [0, 1, 0, 0, 0], "rater_voted": 0})
    >>> t = rater_touched_report(r, f, 0.92)
    >>> t["faces_touched"], t["touched"]["pass"]["n"], (t["untouched"]["pass"]["k"], t["untouched"]["pass"]["n"])
    (1, 1, (1, 2))
    """
    touched_faces = set(face_scores.loc[(face_scores["rater_placed"] == 1) | (face_scores["rater_voted"] == 1),
                                        "face_id"])
    decided = rated[rated["verdict"].isin(["correct", "incorrect"])]
    is_touched = decided["face_id"].isin(touched_faces)

    def rates(d):
        return {st: rate(int((d.loc[d["stratum"] == st, "verdict"] == "correct").sum()),
                         int((d["stratum"] == st).sum())) for st in STRATA}
    out = {"faces_touched": int(rated["face_id"].isin(touched_faces).sum()), "faces": int(len(rated)),
           "touched": rates(decided[is_touched]), "untouched": rates(decided[~is_touched])}
    bar = out["untouched"]["pass"]
    out["bar_met_without_touched"] = bool(bar["n"] >= MIN_DECIDED and bar["wilson95_lower"] >= gate)
    return out


def neighbour_markdown(rep):
    def row(name, v):
        share = "" if v["pass_share"] is None else f"{v['pass_share']:.3f}"
        face = "" if v["pass_share_per_face"] is None else f"{v['pass_share_per_face']:.3f}"
        return f"| {name} | {v['labels']} | {v['faces']} | {v['pass']} | {share} | {face} |"
    head = "| {} | Scored labels | Faces | Pass the gate | Share | Mean share per face |\n|---|---|---|---|---|---|"
    out = [head.format("City"), *[row(c, v) for c, v in rep["cities"].items()], row("all", rep["all"]), "",
           head.format("Group"), *[row(k.replace("_", " "), v) for k, v in rep["by"].items()], ""]
    out += ["Labels beside the rated one, by the stratum of the face's rated label (the faces were chosen through that "
            "label, 60 % of them from the pass stratum, so every share on this page describes this sample only):", "",
            head.format("Rated label's stratum"),
            *[row(k.replace("_", " "), v) for k, v in rep["not_rated_by_rated_stratum"].items()], ""]
    f = rep["faces_with_2plus_scored"]
    out += [f"Faces with at least two scored labels: {f['faces']}. The model's gate verdict is pass for every label on "
            f"{f['all_pass']}, for none on {f['none_pass']}, and mixed on {f['mixed']}; independent verdicts at each "
            f"city's pass share would leave about {f['mixed_expected_if_independent']:.0f} mixed, so the verdict "
            f"clusters by face. Mixed faces are the long ones (median {f['median_labels_mixed']:.0f} labels); on "
            f"{f['mixed_minority_at_most_20pct']} of them at most a fifth of the labels differ from the rest. Of the "
            f"{f['two_label_faces']} two-label faces, {f['two_label_faces_mixed']} are mixed.", ""]
    if "rater_touched_faces" in rep:
        t = rep["rater_touched_faces"]
        out += [f"{t['faces_touched']} of the {t['faces']} rated faces carry a label the rater placed or voted on. "
                "Correct / decided = rate (Wilson 95 % lower bound) on the rated labels:", "",
                "| Stratum | Faces the rater had touched | Other faces |", "|---|---|---|",
                *[f"| {st.replace('_', ' ')} | {_rate_cell(t['touched'][st])} | {_rate_cell(t['untouched'][st])} |"
                  for st in STRATA], "",
                "Without the touched faces the bar on the pass stratum is "
                + ("still met." if t["bar_met_without_touched"] else "**not met**."), ""]
    for by, table in rep["slices"].items():
        out += [head.format(by.capitalize()), *[row(k, v) for k, v in table.items()], ""]
    return "\n".join(out)


def cmd_neighbour_scores(args):
    if rl.inside_git_checkout(args.neighbours):
        sys.exit(f"refusing to write label rows inside a git checkout: {args.neighbours}")
    design = load_design(args.design)
    gate, n_pass = float(design["gate"]), int(design["pass_models"])
    kman = json.load(open(os.path.join(args.key_dir, "manifest.json")))
    nman = json.load(open(os.path.join(args.neighbours, "neighbours.json")))
    if (nman["key_sha256"], nman["scores_sha256"]) != (kman["key_sha256"], kman["scores_sha256"]):
        sys.exit("neighbours.json was made from a different key or scores file")
    pooled, _, n_folds, missing = ng.load_predictions(args.gate)
    if missing:
        sys.exit(f"the gate run is incomplete: {missing}")
    for k in range(n_folds):  # the data directory can hold several gate runs; only the key's one fits its map
        if rl.sha256_file(ng.predictions_path(args.gate, k)) != kman["gate_predictions_sha256"][str(k)]:
            sys.exit(f"{args.gate} is not the gate run the key was built with (fold {k})")
    pi, _ = cal.load_prior(args.priors, LABEL_TYPE)
    if pi != kman["prior"]:
        sys.exit("--priors is not the prior the key was built with")
    api_map = ng.fit_map(pooled, pi)
    spath = os.path.join(args.key_dir, "scores.csv")
    if rl.sha256_file(spath) != kman["scores_sha256"]:
        sys.exit(f"{spath} is not the file the key's manifest pins")
    pinned = pd.read_csv(spath, dtype={"label_uid": str, "label_id": str}, low_memory=False)
    old_preds, _ = read_fold_predictions(args.pinned_predictions, n_folds, kman["predictions_sha256"])
    redo = score_table(pinned[["label_uid"]], old_preds, api_map, gate, n_pass)
    both = pinned.merge(redo, on="label_uid", validate="one_to_one", suffixes=("", "_redo"))
    same = len(both) == len(pinned) and (both["stratum"] == both["stratum_redo"]).all() and all(
        np.allclose(both[f"served_{k}"], both[f"served_{k}_redo"], rtol=0, atol=1e-12)
        and (both[f"pass_{k}"] == both[f"pass_{k}_redo"]).all() for k in range(n_folds))
    if not same:
        sys.exit("the refitted map does not reproduce the pinned scores.csv: not the map the round was rated under")
    lpath = os.path.join(args.neighbours, "labels.csv")
    if rl.sha256_file(lpath) != nman["labels_sha256"]:
        sys.exit(f"{lpath} is not the file neighbours.json pins")
    labels = pd.read_csv(lpath, dtype={"label_uid": str, "label_id": str, "pano_id": str, "tags": str},
                         low_memory=False)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    if set(manifest["label_uid"]) != set(labels["label_uid"]):
        sys.exit(f"{args.manifest} does not cover exactly the labels of {lpath}")
    ok = set(manifest.loc[manifest["status"] == "ok", "label_uid"])
    preds, pred_shas = read_fold_predictions(args.predictions, n_folds)
    weights = model_shas(args.pinned_predictions, n_folds)
    if model_shas(args.predictions, n_folds) != weights:  # a retrained fold model is a different model
        sys.exit("the neighbours were not scored with the weights that scored the round")
    for k, pred in preds.items():
        if set(pred["label_uid"]) != ok:
            sys.exit(f"predictions_fold{k}.csv does not cover exactly the manifest's ok crops")
    new = score_table(labels.loc[labels["label_uid"].isin(ok), ["label_uid"]], preds, api_map, gate, n_pass)
    score_cols = [c for c in new.columns if c != "label_uid"]
    scored = pd.concat([pinned[["label_uid", *score_cols]], new], ignore_index=True)
    faces = pd.read_csv(os.path.join(args.neighbours, "faces.csv"), dtype={"label_uid": str, "label_id": str,
                                                                          "pano_id": str, "tags": str})
    if rl.sha256_file(os.path.join(args.neighbours, "faces.csv")) != nman["faces_sha256"]:
        sys.exit("faces.csv is not the file neighbours.json pins")
    out = faces.merge(scored, on="label_uid", how="left", validate="one_to_one")
    status = dict(zip(manifest["label_uid"], manifest["status"]))
    out["crop_status"] = [status.get(u, "phase1" if s == 1 else "not_cropped")
                          for u, s in zip(out["label_uid"], out["scored"])]
    opath = os.path.join(args.neighbours, "face_scores.csv")
    out.to_csv(opath, index=False)
    rep = neighbour_report(out)
    rep["pins"] = {"neighbours_json_sha256": rl.sha256_file(os.path.join(args.neighbours, "neighbours.json")),
                   "manifest_sha256": rl.sha256_file(args.manifest), "predictions_sha256": pred_shas,
                   "face_scores_sha256": rl.sha256_file(opath), "key_sha256": kman["key_sha256"],
                   "gate_predictions_sha256": kman["gate_predictions_sha256"], "model_sha256": weights}
    if args.sheet:
        rater = kman["design_json"]["rater"]
        rl.check_pin(os.path.join(args.key_dir, "manifest.json"), f"votes_{rater}", args.sheet, False)
        key = pd.read_csv(os.path.join(args.key_dir, "key.csv"), dtype={"tags": str, "label_id": str})
        rated = rl.unmask(key, pd.read_csv(args.sheet, dtype=str, keep_default_na=False))
        rep["rater_touched_faces"] = rater_touched_report(rated, out, gate)
        rep["pins"]["sheet_sha256"] = rl.sha256_file(args.sheet)
    rep["crops"] = {c: {s: int(n) for s, n in g["status"].value_counts().items()}
                    for c, g in manifest.groupby("city")}
    rep["selection"] = nman["cities"]
    rep["gate"], rep["pass_models"] = gate, n_pass
    rep = without_nan(rep)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=1)
            f.write("\n")
    md = neighbour_markdown(rep)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(md + "\n")
    print(md)
    print(f"face scores -> {opath} (sha256 {rep['pins']['face_scores_sha256']})")


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
    # The bootstrap cannot vary a cell with no misses, so the bound printed beside the weighted rate is Wilson's at
    # the Kish effective sample size of the weights.
    w = passed["weight"].astype(float)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum()) if len(passed) else 0.0
    primary["weighted_n_eff"] = n_eff
    primary["weighted_wilson95_lower_n_eff"] = (ale.wilson(primary["weighted"] * n_eff, n_eff)[0] if n_eff
                                                else float("nan"))
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
        f"{rep['items']} items, {rep['items'] - rep['unanswered']} rated ({rep['cant_tell']} Unsure, "
        f"{rep['unanswered']} unanswered). Precision cells are "
        f"correct / decided = rate (Wilson 95 % lower bound); Unsure votes are outside the denominator.", "",
        f"**Bar:** in the pass stratum, {_rate_cell(p)}; a lower bound >= {rep['gate']} on >= {MIN_DECIDED} decided "
        f"labels is {'met' if p['bar_met'] else 'not met'}. **{'PASSES' if p['bar_met'] else 'DOES NOT PASS'}.** "
        f"Reweighted to the cities' population: {_f(p['weighted'])} (Wilson 95 % lower bound "
        f"{_f(p['weighted_wilson95_lower_n_eff'])} at the weights' effective sample size of "
        f"{p['weighted_n_eff']:.0f}; no bootstrap interval is given, because resampling cannot vary a cell with no "
        f"misses). With every Unsure in the pass stratum counted as incorrect: "
        f"{_rate_cell(p['unsure_as_incorrect'])}.", "",
        f"Share of correct NoSidewalk labels in these cities' eligible population: {_f(pc['share'])} "
        f"[{_f(pc['ci95'][0])}, {_f(pc['ci95'][1])}] (the deployment prior assumes {_f(rep['prior_correct'])} over "
        f"every label with a face; this population leaves out the rater's own labels and trained faces. The interval "
        f"resamples within (city, stratum) cells with the weights fixed, so it is too narrow where a cell has no "
        f"misses, and Unsure and unanswered items drop out with their weight).", "",
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
    lines += ["", "Slice population shares rest on few items with unequal weights and carry no interval; read them as "
              "direction only.",
              "", "Crowd majority (human votes at the pool pull) against the expert, decided items:", "",
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
    n = sub.add_parser("neighbours", help="every other label on the rated labels' faces, to crop and score (#102)")
    n.add_argument("--design", required=True)
    n.add_argument("--rawlabels", required=True, help="the directory the pool was drawn from")
    n.add_argument("--key-dir", required=True, help="holds key.csv, scores.csv and manifest.json")
    n.add_argument("--pool", required=True, help="the directory `pool` wrote; its pool.json pins the inputs")
    n.add_argument("--author-excluded", required=True)
    n.add_argument("--pilot-split", required=True)
    n.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    n.add_argument("--out", required=True, help="outside any git checkout")
    n.set_defaults(func=cmd_neighbours)
    ns = sub.add_parser("neighbour-scores", help="the fold models' gate verdicts on the neighbours, under the round's map")
    ns.add_argument("--design", required=True)
    ns.add_argument("--key-dir", required=True)
    ns.add_argument("--gate", required=True, help="the gate run the key was built with")
    ns.add_argument("--priors", required=True)
    ns.add_argument("--pinned-predictions", required=True, help="the round's own predictions_fold<k>.csv directory")
    ns.add_argument("--neighbours", required=True, help="the directory `neighbours` wrote; face_scores.csv goes here")
    ns.add_argument("--manifest", required=True, help="manifest.csv of the neighbour crops")
    ns.add_argument("--predictions", required=True, help="directory of the neighbours' predictions_fold<k>.csv")
    ns.add_argument("--sheet", help="the rater's pinned votes: adds the round's rates with and without the faces "
                                    "that carry a label the rater placed or voted on")
    ns.add_argument("--out-json", help="aggregates only (safe for the repo)")
    ns.add_argument("--out-md")
    ns.set_defaults(func=cmd_neighbour_scores)
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
