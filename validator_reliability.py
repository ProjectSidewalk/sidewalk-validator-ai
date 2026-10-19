"""Per-validator reliability and per-label reliability-weighted vote counts, from cached rawLabels exports.

Part of https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12 (soft targets and per-validator weighting;
Duan et al. ASSETS'22, https://doi.org/10.1145/3517428.3550381, section 4).

Reads every ``<rawlabels>/<city_id>.csv`` (the same public rawLabels exports ``split_data.py --rawlabels`` reads; the
city_id is the file name stem) and uses only each label's ``validations`` column, a JSON list of
``{user_id, validation, validator_type, ...}``. As in ``data_generation.human_vote_counts``, only votes whose
``validator_type`` is exactly ``"Human"`` count (fail closed: the AI account's votes, and any vote with a missing or new
type, are ignored), and only ``Agree``/``Disagree`` votes count (``Unsure`` is ignored everywhere). If one validator
appears twice on a label, the last vote in the list is kept (the 2026-10-17 exports have none; the count is printed).

**Reliability** (leave-one-out). For each human Agree/Disagree vote, the *other* human Agree/Disagree votes on the same
label are counted. The vote is scored only if there are at least ``--min-other-votes`` (default 3) of them and they do
not tie; it is a hit if it matches their majority. ``user_id`` comes from the shared login schema, so a validator's
votes are pooled across all cities. With ``hits`` hits out of ``n`` scored votes, the reported reliability is the
posterior mean under a Beta prior centred on the pooled hit rate ``m`` of all scored votes, with strength
``k = --prior-strength`` (default 10 pseudo-votes)::

    reliability = (hits + k * m) / (n + k)

so a validator with few scored votes sits near ``m`` and only a long, consistent record reaches 0.95. A validator with
no scored votes gets ``m``. The raw rate ``hits / n`` is reported alongside.

**Vote weights.** For every label with at least one human Agree/Disagree vote, each vote is weighted by its caster's
reliability: ``w_agree`` / ``w_disagree`` are the summed reliabilities of the Agree / Disagree voters, ``n_votes`` the
number of votes, ``n_gold`` how many came from validators with reliability ``>= --gold-min-reliability`` (default
0.95), and ``min_reliability`` the lowest reliability among the label's voters (so a "gold" label, every vote from a
validator at or above any threshold ``x``, is ``min_reliability >= x``; evaluate.py uses it). Note the reliability of
a validator is computed from all their votes, including the one on the label being weighted; with ``>= 3`` other votes
required per scored vote this leak is small, and it is the same for every label.

Outputs, in ``--out`` (a data directory, never the repository: the files hold pseudonymous user ids):

- ``validator_reliability.json``: ``{user_id: {"reliability", "n", "hits", "raw_rate"}}``
- ``vote_weights.csv``: ``label_uid, w_agree, w_disagree, n_votes, n_gold, min_reliability``, sorted by ``label_uid``
  (``<city_id>:<label_id>``, exactly as data_generation.py builds it)
- ``reliability_run.json``: the settings, the prior mean, the source files and aggregate counts

Usage::

    python validator_reliability.py --rawlabels ~/data/rawlabels --out ~/data/reliability
    python validator_reliability.py --rawlabels ~/data/rawlabels --out ~/data/reliability \\
        --min-other-votes 3 --prior-strength 10 --gold-min-reliability 0.95

Then ``fine_tune_classifier.py --target weighted --vote-weights ~/data/reliability/vote_weights.csv`` and
``evaluate.py --vote-weights ~/data/reliability/vote_weights.csv``.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MIN_OTHER_VOTES = 3
PRIOR_STRENGTH = 10.0
GOLD_MIN_RELIABILITY = 0.95
VOTE_WEIGHT_COLUMNS = ["label_uid", "w_agree", "w_disagree", "n_votes", "n_gold", "min_reliability"]


def human_votes(validations_json):
    """``({user_id: True for Agree / False for Disagree}, n_duplicates)`` from one ``validations`` cell.

    Human votes only (``validator_type == "Human"``, fail closed), Agree/Disagree only; a validator listed twice keeps
    their last vote.

    >>> human_votes('[{"user_id":"a","validation":"Agree","validator_type":"AI"},'
    ...             ' {"user_id":"b","validation":"Agree","validator_type":"Human"},'
    ...             ' {"user_id":"c","validation":"Unsure","validator_type":"Human"},'
    ...             ' {"user_id":"d","validation":"Disagree"},'
    ...             ' {"user_id":"e","validation":"Disagree","validator_type":"Human"}]')
    ({'b': True, 'e': False}, 0)
    >>> human_votes(float("nan"))
    ({}, 0)
    """
    if not isinstance(validations_json, str) or not validations_json:
        return {}, 0
    votes, dup = {}, 0
    for v in json.loads(validations_json):
        if v.get("validator_type") != "Human" or v.get("validation") not in ("Agree", "Disagree"):
            continue
        uid = v.get("user_id")
        if uid in votes:
            dup += 1
            del votes[uid]  # re-insert so dict order is the order of the kept votes
        votes[uid] = v["validation"] == "Agree"
    return votes, dup


def reliability_counts(label_votes, min_other_votes=MIN_OTHER_VOTES):
    """``{user_id: [hits, n]}`` from an iterable of per-label ``{user_id: agree}`` dicts (leave-one-out majority).

    A vote is scored only when the label has at least ``min_other_votes`` other votes and they do not tie. Validators
    with votes but no scored vote appear with ``[0, 0]``.

    >>> c = reliability_counts([{"a": True, "b": True, "c": True, "d": False}])
    >>> c["a"], c["d"]     # a agrees with the other three's 2-1 majority; d's others are 3-0 Agree
    ([1, 1], [0, 1])
    >>> reliability_counts([{"a": True, "b": True, "c": False}])["a"]   # only 2 others: not scored
    [0, 0]
    >>> reliability_counts([{"a": True, "b": True, "c": False, "d": False, "e": True}])["e"]   # others tie 2-2
    [0, 0]
    """
    counts = defaultdict(lambda: [0, 0])
    for votes in label_votes:
        agree = sum(votes.values())
        disagree = len(votes) - agree
        for uid, a in votes.items():
            oa, od = agree - a, disagree - (not a)
            c = counts[uid]
            if oa + od >= min_other_votes and oa != od:
                c[0] += int(a == (oa > od))
                c[1] += 1
    return dict(counts)


def smooth(counts, prior_strength=PRIOR_STRENGTH):
    """``(table, prior_mean)``: ``{user_id: {"reliability", "n", "hits", "raw_rate"}}`` under a Beta prior centred on
    the pooled hit rate with ``prior_strength`` pseudo-votes.

    >>> t, m = smooth({"a": [9, 10], "b": [1, 2], "c": [0, 0]}, prior_strength=4)
    >>> m   # 10 hits in 12 scored votes
    0.8333333333333334
    >>> round(t["a"]["reliability"], 4), round(t["b"]["reliability"], 4), t["c"]["reliability"] == m, t["c"]["raw_rate"]
    (0.881, 0.7222, True, None)
    """
    hits = sum(h for h, _ in counts.values())
    n = sum(k for _, k in counts.values())
    if n == 0:
        raise ValueError("no scored votes: no label has enough other human Agree/Disagree votes")
    m = hits / n
    table = {uid: {"reliability": (h + prior_strength * m) / (k + prior_strength), "n": k, "hits": h,
                   "raw_rate": h / k if k else None}
             for uid, (h, k) in sorted(counts.items())}
    return table, m


def vote_weights(labels, reliability, gold_min_reliability=GOLD_MIN_RELIABILITY):
    """The per-label weighted counts (``VOTE_WEIGHT_COLUMNS``) for labels with at least one vote, by ``label_uid``.

    Args:
        labels: iterable of ``(label_uid, {user_id: agree})``.
        reliability: ``{user_id: float}``; every voter must be in it.

    >>> vote_weights([("x:1", {"a": True, "b": False}), ("x:2", {})], {"a": 0.96, "b": 0.5}).to_dict("records")
    [{'label_uid': 'x:1', 'w_agree': 0.96, 'w_disagree': 0.5, 'n_votes': 2, 'n_gold': 1, 'min_reliability': 0.5}]
    """
    rows = []
    for uid, votes in labels:
        if not votes:
            continue
        r = [(reliability[u], a) for u, a in votes.items()]
        rows.append((uid, sum(w for w, a in r if a), sum(w for w, a in r if not a), len(r),
                     sum(1 for w, _ in r if w >= gold_min_reliability), min(w for w, _ in r)))
    df = pd.DataFrame(rows, columns=VOTE_WEIGHT_COLUMNS)
    df = df.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid is not unique in the rawLabels exports")
    return df


def load_votes(rawlabels_dir):
    """``(labels, files, n_duplicates)``: ``labels`` is a list of ``(label_uid, {user_id: agree})`` for every row of
    every ``<rawlabels_dir>/<city_id>.csv``; ``label_uid`` is ``<city_id>:<label_id>``."""
    files = sorted(glob.glob(os.path.join(rawlabels_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no <city_id>.csv files in {rawlabels_dir}")
    labels, dup = [], 0
    for path in files:
        city = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, usecols=["label_id", "validations"], dtype={"label_id": str, "validations": str})
        for label_id, cell in zip(df["label_id"], df["validations"]):
            votes, d = human_votes(cell)
            dup += d
            labels.append((f"{city}:{label_id}", votes))
    return labels, files, dup


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rawlabels", required=True, help="directory of <city_id>.csv rawLabels exports")
    p.add_argument("--out", required=True, help="output directory (a data directory, not the repository)")
    p.add_argument("--min-other-votes", type=int, default=MIN_OTHER_VOTES,
                   help="other human Agree/Disagree votes a label needs before a vote on it is scored (default 3)")
    p.add_argument("--prior-strength", type=float, default=PRIOR_STRENGTH,
                   help="Beta prior strength, in pseudo-votes at the pooled hit rate (default 10)")
    p.add_argument("--gold-min-reliability", type=float, default=GOLD_MIN_RELIABILITY,
                   help="reliability at or above which a vote counts toward n_gold (default 0.95)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    labels, files, dup = load_votes(args.rawlabels)
    counts = reliability_counts((v for _, v in labels), args.min_other_votes)
    table, prior_mean = smooth(counts, args.prior_strength)
    weights = vote_weights(labels, {u: t["reliability"] for u, t in table.items()}, args.gold_min_reliability)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "validator_reliability.json"), "w") as f:
        json.dump(table, f, indent=1)
        f.write("\n")
    weights.to_csv(os.path.join(args.out, "vote_weights.csv"), index=False)

    rel = np.array([t["reliability"] for t in table.values()])
    scored = np.array([t["n"] > 0 for t in table.values()])
    gold = int(((weights.n_gold == weights.n_votes) & (weights.n_votes >= 1)).sum())
    summary = {
        "n_validators": len(table), "n_validators_scored": int(scored.sum()),
        "median_reliability_scored": float(np.median(rel[scored])) if scored.any() else None,
        "share_scored_at_or_above_gold": float((rel[scored] >= args.gold_min_reliability).mean()) if scored.any() else None,
        "n_scored_votes": int(sum(t["n"] for t in table.values())),
        "n_labels_with_votes": int(len(weights)), "n_labels_all_gold": gold, "n_duplicate_votes_dropped": dup,
    }
    run = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rawlabels": os.path.abspath(args.rawlabels), "files": [os.path.basename(f) for f in files],
           "min_other_votes": args.min_other_votes, "prior_strength": args.prior_strength, "prior_mean": prior_mean,
           "gold_min_reliability": args.gold_min_reliability, "summary": summary}
    with open(os.path.join(args.out, "reliability_run.json"), "w") as f:
        json.dump(run, f, indent=2)
        f.write("\n")
    print(json.dumps({"prior_mean": prior_mean, **summary}, indent=2))
    print(f"wrote validator_reliability.json, vote_weights.csv and reliability_run.json to {args.out}")


if __name__ == "__main__":
    main()
