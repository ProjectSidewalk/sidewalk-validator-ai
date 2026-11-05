"""Is a label type ready for an automatic validator? The training-data census for any type, from a rawLabels cache.

Issue #42 (NoSidewalk and Signal). For each ``--label-type`` it counts, from a cache of rawLabels exports
(``analysis/fetch_all.py``, optionally ``--label-types``-filtered):

- labels, cities, providers, human-authored labels, labels with >= 1 / 2 / 3 human votes;
- the label rule of ``data_generation.eligible_labels`` (margin only, and margin + ``EXPERTS``): class x
  ``class_source``, GSV-only counts, the cities with the most ``incorrect`` labels, cities with >= 20;
- the (human agree, human disagree) vote-pair histogram of voted labels;
- block faces (``nosidewalk_faces``): labels with a face, faces among the label-rule labels and faces whose
  label-rule labels carry both classes;
- the face rule as the 2026-11-05 census counted it (``count="votes"``, every label on a decided face) and as
  training applies it (``nosidewalk_faces.add_face_verdicts``: distinct validators, only labels the label rule left
  undecided, then the per-face cap): faces and labels per class, incorrect labels per city, labels per incorrect
  face, the single-labeler share;
- tags x class of the label-rule labels;
- the face-size deployment prior (``calibration/nosidewalk_prior.py``), for NoSidewalk.

Only aggregates are written (``--out-json`` and ``--out-md``); no label id or user id leaves this script. Cities are
named only when the deployment listing says they are public (``visibility == "public"`` with a url); any other city's
counts are pooled under ``private (pooled)``. The listing is fetched live by default (``--listing live``), or read
from a saved ``cities.json`` (tests, offline runs; check it is current before publishing).

Expert validators come from the environment, ``EXPERTS="<name>=<user_id>,..."`` (as everywhere in this repo), and
never appear in the output (only how many there were).

Usage::

    EXPERTS=... python analysis/type_readiness.py --cache rawlabels-ns --label-type NoSidewalk --label-type Signal \\
        --out-json docs/nosidewalk-2026-11/readiness.json --out-md docs/nosidewalk-2026-11/readiness.md
    # a cache whose files are <city>__<type>.csv (the 2026-11-05 scratch pull):
    ... --pattern '{city}__{label_type}.csv'
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generation as dg  # noqa: E402
import nosidewalk_faces as nf  # noqa: E402
from calibration import nosidewalk_prior  # noqa: E402

PRIVATE = "private (pooled)"
TOP_CITIES = 12
MIN_INCORRECT_CITY = 20  # "cities with >= 20 incorrect": enough negatives in one city for a within-city AUC


def load_cache(cache, label_type, pattern="{city}.csv"):
    """Every cached city's rows of ``label_type`` as one frame with a ``city`` column; ``(frame, city_ids)``.

    Cities come from ``<cache>/cities.json`` when present (its order), else from the file names matching
    ``pattern``. A listed city without a file is skipped (a type-filtered pull writes no file for a city with no
    such label).
    """
    listing_path = os.path.join(cache, dg.CITIES_CACHE_FILE)
    if os.path.exists(listing_path):
        with open(listing_path) as f:
            ids = [c["city_id"] for c in dg.listing_entries(json.load(f))]
    else:
        suffix = pattern.split("{city}", 1)[1].format(label_type=label_type)
        ids = sorted(n[:-len(suffix)] for n in os.listdir(cache) if n.endswith(suffix))
    frames, present = [], []
    for city in ids:
        path = os.path.join(cache, pattern.format(city=city, label_type=label_type))
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False, dtype=dg.PANO_ID_DTYPE)
        df["city"] = city
        frames.append(df)
        present.append(city)
    if not frames:
        raise SystemExit(f"no rawLabels files for {label_type} in {cache} (pattern {pattern})")
    df = pd.concat(frames, ignore_index=True)
    return df[df["label_type"] == label_type].copy(), present


def public_cities(listing):
    """The city_ids a ``/v3/api/cities`` listing marks public with a url (the ones that may be named)."""
    return {c["city_id"] for c in dg.listing_entries(listing) if c.get("visibility") == "public" and c.get("url")}


def city_name(city, public):
    return city if city in public else PRIVATE


def _ints(d):
    return {str(k): int(v) for k, v in d.items()}


def class_by_source(eligible):
    """``{label_class: {class_source: n}}`` for human-authored rows."""
    t = eligible.groupby(["label_class", "class_source"]).size()
    out = {}
    for (cls, src), n in t.items():
        out.setdefault(cls, {})[src] = int(n)
    return out


def per_city(frame, public, top=TOP_CITIES):
    """Correct / incorrect per city (private pooled), sorted by incorrect then name; ``(top rows, n with >= 20)``."""
    names = frame["city"].map(lambda c: city_name(c, public))
    t = pd.crosstab(names, frame["label_class"]).reindex(columns=["correct", "incorrect"], fill_value=0)
    t = t.reset_index().rename(columns={"city": "city"})
    t.columns = ["city", "correct", "incorrect"]
    t = t.sort_values(["incorrect", "city"], ascending=[False, True], kind="mergesort")
    rows = [{"city": r.city, "correct": int(r.correct), "incorrect": int(r.incorrect)} for r in t.itertuples()]
    return rows[:top], int((t["incorrect"] >= MIN_INCORRECT_CITY).sum())


def vote_pairs(valid):
    """``[{"agree", "disagree", "n"}]`` over labels with >= 1 human vote, most frequent first."""
    counts = valid["validations"].apply(dg.human_vote_counts)
    pairs = Counter((a, d) for a, d in counts if a + d > 0)
    return [{"agree": a, "disagree": d, "n": n}
            for (a, d), n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))]


def label_rule(rows, label_type, experts, public):
    """The label rule's counts (human-authored labels only, as the census and the prior count them)."""
    eligible, dropped = dg.eligible_labels(rows, label_type, experts)
    human = eligible[eligible["author"] == "human"]
    gsv = human[human["pano_source"].astype(str).str.lower() == "gsv"]
    top, n20 = per_city(human, public)
    return eligible, {
        "dropped": _ints(dropped),
        "class_by_source": class_by_source(human),
        "total": _ints(human["label_class"].value_counts()),
        "gsv_only": _ints(gsv["label_class"].value_counts()),
        "cities_with_20_incorrect": n20,
        "top_cities_by_incorrect": top,
    }


def census_face_rule(valid, experts, public):
    """The face union rule exactly as the 2026-11-05 census counted it: votes summed over the face's labels, and
    every label on a decided face counted (not only the undecided ones)."""
    votes = nf.face_votes(valid, experts)
    cls = nf.face_classes(votes, bool(experts), count="votes")
    faces = votes.join(cls)
    out = {}
    for c in ("correct", "incorrect"):
        f = faces[faces["face_class"] == c]
        out[c] = {"faces": int(len(f)), "labels": int(f["n_labels"].sum()),
                  "single_labeler_share": round(float((f["labelers"] == 1).mean()), 2) if len(f) else None}
    inc = faces[faces["face_class"] == "incorrect"]
    by_city = inc.reset_index()
    by_city["city"] = by_city["face_id"].str.split(":").str[0].map(lambda c: city_name(c, public))
    top = by_city.groupby("city")["n_labels"].sum().sort_values(ascending=False, kind="mergesort")
    out["incorrect_labels_by_city"] = [{"city": c, "labels": int(n)} for c, n in top.head(8).items()]
    q = inc["n_labels"].quantile([0.5, 0.9]) if len(inc) else pd.Series([np.nan, np.nan], index=[0.5, 0.9])
    out["labels_per_incorrect_face"] = {"median": float(q[0.5]), "p90": float(q[0.9]),
                                        "max": int(inc["n_labels"].max()) if len(inc) else None}
    return out


def pipeline_face_rule(rows, eligible, label_type, experts, public, cap, seed):
    """What ``data_generation.py --verdict-unit face`` selects from: label-rule rows plus face-rule rows, before and
    after the per-face cap (human-authored only)."""
    enlarged = nf.add_face_verdicts(rows, eligible, label_type, experts)
    capped = nf.cap_labels_per_face(enlarged, cap, seed)
    out = {}
    for name, frame in (("before_cap", enlarged), ("after_cap", capped)):
        human = frame[frame["author"] == "human"]
        top, n20 = per_city(human, public)
        face_rows = human[human["face_id"] != ""]
        out[name] = {"class_by_source": class_by_source(human), "total": _ints(human["label_class"].value_counts()),
                     "faces": _ints(face_rows.groupby("label_class")["face_id"].nunique()),
                     "cities_with_20_incorrect": n20, "top_cities_by_incorrect": top}
    out["cap"] = cap
    return out


def tags_by_class(eligible):
    """``{tag: {class: n}}`` over label-rule rows; a label counts once per tag, ``<none>`` for no tag."""
    out = {}
    for cls, tags in zip(eligible["label_class"], eligible["tags"].fillna("[]")):
        for t in (json.loads(tags) or ["<none>"]):
            out.setdefault(t, {}).setdefault(cls, 0)
            out[t][cls] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1].get("incorrect", 0), kv[0])))


def readiness(rows, label_type, experts, public, cap=nf.DEFAULT_MAX_LABELS_PER_FACE, seed=dg.SEED):
    """Every aggregate for one label type, as a JSON-ready dict (see the module docstring)."""
    counts = rows["validations"].apply(dg.human_vote_counts)
    n_votes = pd.Series([a + d for a, d in counts], index=rows.index)
    valid = nf.type_rows(rows, label_type)
    out = {
        "label_type": label_type,
        "labels": int(len(rows)),
        "cities": int(rows["city"].nunique()),
        "pano_source": _ints(rows["pano_source"].value_counts()),
        "human_authored": int((rows["user_id"].map(dg.label_author) == "human").sum()),
        "with_votes": {"ge1": int((n_votes >= 1).sum()), "ge2": int((n_votes >= 2).sum()),
                       "ge3": int((n_votes >= 3).sum())},
        "n_experts": len(experts),
        "vote_pairs": vote_pairs(valid),
    }
    _, out["label_rule_margin"] = label_rule(rows, label_type, frozenset(), public)
    eligible, out["label_rule_experts"] = label_rule(rows, label_type, experts, public)
    with_face = valid[valid["face_id"] != ""]
    elig_faces = nf.face_id(eligible)
    on_face = eligible[elig_faces != ""].assign(face_id=elig_faces[elig_faces != ""])
    mixed = on_face.groupby("face_id")["label_class"].nunique()
    out["faces"] = {"labels_with_face": int(len(with_face)), "valid_labels": int(len(valid)),
                    "faces": int(with_face["face_id"].nunique()),
                    "faces_among_label_rule": int(len(mixed)), "label_rule_labels_on_a_face": int(len(on_face)),
                    "faces_with_both_classes": int((mixed > 1).sum())}
    out["face_rule_census"] = census_face_rule(valid, experts, public)
    out["face_rule_pipeline"] = pipeline_face_rule(rows, eligible, label_type, experts, public, cap, seed)
    out["tags_by_class"] = tags_by_class(eligible[eligible["author"] == "human"])
    if label_type == "NoSidewalk":
        e = nosidewalk_prior.prior_entry(valid, sorted(set(valid["city"])), measured_on=None)
        out["face_size_prior"] = {k: e[k] for k in ("p_correct", "face_size_shares", "face_size_precision",
                                                     "labels_with_face", "labels_without_face")}
    return out


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _f(n):
    return f"{n:,}" if isinstance(n, (int, np.integer)) else str(n)


def markdown(report):
    """The tables of ``docs/nosidewalk-2026-11.md`` from a :func:`main` report dict."""
    parts = [f"<!-- generated by analysis/type_readiness.py from {report['cache_name']} "
             f"(cache {report['cache_date']}); do not edit by hand -->"]
    for t, r in report["types"].items():
        lm, le = r["label_rule_margin"], r["label_rule_experts"]
        cen, pipe = r["face_rule_census"], r["face_rule_pipeline"]
        parts.append(f"\n### {t}\n")
        parts.append(f"{_f(r['labels'])} labels in {r['cities']} cities; {_f(r['human_authored'])} human-authored; "
                     f"providers {r['pano_source']}. Labels with >= 1 / 2 / 3 human votes: "
                     f"{_f(r['with_votes']['ge1'])} / {_f(r['with_votes']['ge2'])} / {_f(r['with_votes']['ge3'])}.\n")
        rows = [["label margin only", _f(lm["total"].get("correct", 0)), _f(lm["total"].get("incorrect", 0)),
                 lm["cities_with_20_incorrect"]],
                [f"label margin + expert ({r['n_experts']} experts)", _f(le["total"].get("correct", 0)),
                 _f(le["total"].get("incorrect", 0)), le["cities_with_20_incorrect"]]]
        if t == "NoSidewalk":
            rows.append(["face rule, census count (vote sums, every label on a decided face)",
                         f"{_f(cen['correct']['labels'])} on {_f(cen['correct']['faces'])} faces",
                         f"{_f(cen['incorrect']['labels'])} on {_f(cen['incorrect']['faces'])} faces", "n/a"])
            for k, name in (("before_cap", "training rule (label rule, then distinct-validator face rule)"),
                            ("after_cap", f"training rule after the cap of {pipe['cap']} per face and class")):
                p = pipe[k]
                rows.append([name, f"{_f(p['total'].get('correct', 0))} on {_f(p['faces'].get('correct', 0))} faces",
                             f"{_f(p['total'].get('incorrect', 0))} on {_f(p['faces'].get('incorrect', 0))} faces",
                             p["cities_with_20_incorrect"]])
        parts.append(_table(["rule", "correct", "incorrect", "cities with >= 20 incorrect"], rows))
        if t == "NoSidewalk":
            parts.append("\nTraining rule by source (before the cap):\n")
            src = pipe["before_cap"]["class_by_source"]
            sources = ["margin", "expert", "face_margin", "face_expert"]
            parts.append(_table(["class"] + sources,
                                [[c] + [_f(src.get(c, {}).get(s, 0)) for s in sources] for c in ("correct", "incorrect")]))
            parts.append("\nIncorrect labels by city, face rule (census count): "
                         + ", ".join(f"{d['city']} {d['labels']}" for d in cen["incorrect_labels_by_city"]) + ".")
            q = cen["labels_per_incorrect_face"]
            parts.append(f"Labels per incorrect face: median {q['median']:g}, p90 {q['p90']:g}, max {q['max']}. "
                         f"Single-labeler faces: {cen['incorrect']['single_labeler_share']} of incorrect faces, "
                         f"{cen['correct']['single_labeler_share']} of correct faces.")
            fc = r["faces"]
            parts.append(f"Faces: {_f(fc['labels_with_face'])} of {_f(fc['valid_labels'])} labels with usable geometry "
                         f"have a face ({_f(fc['faces'])} faces); the {_f(fc['label_rule_labels_on_a_face'])} label-rule "
                         f"labels (with experts) on a face sit on {_f(fc['faces_among_label_rule'])} faces, "
                         f"{fc['faces_with_both_classes']} of which carry both classes.\n")
            fp = r["face_size_prior"]
            sh = fp["face_size_shares"]
            parts.append(f"Face-size deployment prior: P(correct) = {fp['p_correct']:.4f}; labels on faces with 1 / 2 / "
                         f"3+ NoSidewalk labels: {sh['1']:.3f} / {sh['2']:.3f} / {sh['3']:.3f}.\n")
            parts.append("Vote pairs (human agree, human disagree) of voted labels, the 14 most frequent:\n")
            parts.append(_table(["agree", "disagree", "labels"],
                                [[p["agree"], p["disagree"], _f(p["n"])] for p in r["vote_pairs"][:14]]))
        parts.append(f"\nLabel rule with experts, cities by incorrect (top {TOP_CITIES}):\n")
        parts.append(_table(["city", "correct", "incorrect"],
                            [[d["city"], _f(d["correct"]), _f(d["incorrect"])] for d in le["top_cities_by_incorrect"]]))
    return "\n".join(parts) + "\n"


def main(argv=None, fetch=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", required=True, help="rawLabels cache directory (analysis/fetch_all.py)")
    p.add_argument("--label-type", action="append", required=True, help="repeatable")
    p.add_argument("--pattern", default="{city}.csv",
                   help="file name inside the cache, with {city} and optionally {label_type}")
    p.add_argument("--listing", default="live",
                   help="'live' (fetch /v3/api/cities now) or a saved cities.json: decides which cities are named")
    p.add_argument("--cache-date", default=None, help="the pull date to record (default: the cache's cities.json)")
    p.add_argument("--max-labels-per-face", type=int, default=nf.DEFAULT_MAX_LABELS_PER_FACE)
    p.add_argument("--seed", type=int, default=dg.SEED)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", default=None)
    args = p.parse_args(argv)
    experts = dg.parse_experts(os.environ.get(dg.EXPERTS_ENV))
    if args.listing == "live":
        listing = dg.fetch_city_listing(fetch=fetch)
    else:
        with open(args.listing) as f:
            listing = json.load(f)
    public = public_cities(listing)
    cache_date = args.cache_date
    listing_path = os.path.join(args.cache, dg.CITIES_CACHE_FILE)
    if cache_date is None and os.path.exists(listing_path):
        with open(listing_path) as f:
            cache_date = json.load(f).get("fetched_on")
    report = {"cache_name": os.path.basename(os.path.normpath(args.cache)), "cache_date": cache_date,
              "listing": "live" if args.listing == "live" else "saved", "n_experts": len(experts),
              "max_labels_per_face": args.max_labels_per_face, "seed": args.seed, "types": {}}
    for t in args.label_type:
        rows, _ = load_cache(args.cache, t, args.pattern)
        private = sorted(set(rows["city"]) - public)
        if private:
            print(f"{t}: {len(private)} cities are not public in the listing; their counts are pooled", file=sys.stderr)
        report["types"][t] = readiness(rows, t, experts, public, args.max_labels_per_face, args.seed)
        print(f"{t}: {report['types'][t]['labels']:,} labels in {report['types'][t]['cities']} cities")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out_json}")
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(markdown(report))
        print(f"wrote {args.out_md}")
    return report


if __name__ == "__main__":
    main()
