"""Print every number in docs/calibration-2026-10.md that is not in calibrate_from_production's summary.md.

Reads the ``labels_oof.csv`` that ``python -m calibration.calibrate_from_production`` writes next to its maps (one row
per assessed label: verdict, date, confidence, old / shipped / out-of-fold served value, outcome, human votes). That
file and this script's output contain private-city data: keep both outside the repo and quote only aggregates.

City names are printed only for public deployments, read from a saved copy of the platform's city list
(``curl https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > cities.json``); everything else is reported as a
private deployment. Without ``--cities-json`` no city is named.

Usage::

    python analysis/calibration_doc_numbers.py ~/Data/<pull>/phase0/labels_oof.csv --cities-json ~/Data/<pull>/cities.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calibration import _isotonic_map as im  # noqa: E402

GATE = im.GATE
TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]


def public_names(cities_json):
    """``{schema_city: display}`` for public deployments; schema cities that match nothing count as private."""
    if not cities_json:
        return {}
    cities = json.load(open(cities_json))
    cities = cities.get("cities", cities)
    return {c["city_id"]: c["city_id"] for c in cities if c.get("visibility") == "public"}


def city_label(schema_city, public):
    slug = schema_city.replace("_", "-")
    for cid in public:
        if cid == slug or cid.startswith(slug + "-"):
            return schema_city
    return None   # private or unknown: never named


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("labels_oof")
    p.add_argument("--cities-json", default=None)
    args = p.parse_args(argv)
    d = pd.read_csv(args.labels_oof, low_memory=False, true_values=["True"], false_values=["False"])
    d["city"] = d.schema.str.replace("sidewalk_", "", regex=False)
    public = public_names(args.cities_json)
    d["shown"] = d.city.map(lambda c: city_label(c, public))
    lab = d[d.labelled]

    print("== Confidence (all assessed) ==")
    print(f"  median {d.ai_confidence.median():.5f}; share above 0.9999: {(d.ai_confidence > 0.9999).mean():.1%}")

    print("\n== Majority-oracle vote share (labelled rows): max(h_agree, h_dis) / (h_agree + h_dis) ==")
    o = lab.assign(oracle=np.maximum(lab.h_agree, lab.h_dis) / (lab.h_agree + lab.h_dis))
    print(o.groupby("label_type").oracle.mean().round(3).to_string())

    print("\n== Assessments 2025-08-16 .. 2025-08-31 (model-swap window) ==")
    print(f"  {d.ai_date.between('2025-08-16', '2025-08-31').sum():,}")

    print("\n== Pooled calibration error (|served - observed| by confidence bin, weighted by n; labelled rows) ==")
    for col, name in (("ai_accuracy", "old maps"), ("oof_accuracy", "new maps, out-of-fold")):
        tabs = [im.reliability_table(g.ai_confidence, g.correct, predicted=g[col])
                for _, g in lab.groupby(["label_type", "ai_result"])]
        t = pd.concat(tabs)
        print(f"  {name}: {np.average((t.predicted - t.observed).abs(), weights=t.n):.3f}")

    print("\n== Disagree verdicts: majority agrees (ties 0.5) by number of human votes ==")
    x = d[(d.ai_result == "Disagree")].copy()
    x["h_n"] = x.h_agree + x.h_dis
    x = x[x.h_n.between(1, 3)]
    x["right"] = np.where(x.h_agree == x.h_dis, 0.5, (x.h_dis > x.h_agree).astype(float))
    print(x.groupby(["label_type", "h_n"]).right.mean().unstack().round(2).to_string())

    print("\n== NoCurbRamp Agree: labels the new map drops vs keeps (labelled rows) ==")
    n = lab[(lab.label_type == "NoCurbRamp") & (lab.ai_result == "Agree")]
    for col, name in (("oof_accuracy", "out-of-fold"), ("new_accuracy", "shipped map")):
        drop = n[(n.ai_accuracy >= GATE) & (n[col] < GATE)]
        keep = n[n[col] >= GATE]
        print(f"  {name}: dropped {len(drop):,} at {drop.correct.mean():.3f}; kept {len(keep):,} at "
              f"{keep.correct.mean():.4f}")
    print(f"  old gate pooled: {n[n.ai_accuracy >= GATE].correct.mean():.3f}")

    print("\n== Vote share outcome on the shipped maps (Agree classes; share of human votes matching) ==")
    for t in TYPES:
        g = lab[(lab.label_type == t) & (lab.ai_result == "Agree") & (lab.new_accuracy >= GATE)]
        if len(g):
            share = g.h_agree / (g.h_agree + g.h_dis)
            print(f"  {t}: {share.mean():.3f} (n={len(g):,})")

    print("\n== City mix: share of labelled vs share of assessed ==")
    mix = pd.DataFrame({"assessed": d.groupby(d.shown.fillna("(private)")).size() / len(d),
                        "labelled": lab.groupby(lab.shown.fillna("(private)")).size() / len(lab)}).fillna(0)
    n_private = d[d.shown.isna()].city.nunique()
    print(f"  ({n_private} private deployments pooled as '(private)')")
    mix["ratio"] = mix.labelled / mix.assessed
    print(mix.sort_values("assessed", ascending=False).head(12).round(3).to_string())
    top_priv = [c for c in lab[lab.shown.isna()].city.value_counts().index[:2]]
    for i, c in enumerate(top_priv, 1):
        print(f"  private deployment #{i} by labelled count: labelled {(lab.city == c).mean():.1%}, "
              f"assessed {(d.city == c).mean():.1%}")

    print("\n== New votes (out-of-fold) per city with >= 500 labelled labels ==")
    nv = lab[lab.oof_accuracy >= GATE]
    big = lab.groupby("city").size()
    big = big[big >= 500].index
    pc = nv[nv.city.isin(big)].groupby("city").correct.agg(["size", "mean"])
    print(f"  cities: {len(big)}; below {GATE}: {(pc['mean'] < GATE).sum()}")
    pc["name"] = [city_label(c, public) or "(a private deployment)" for c in pc.index]
    print(pc.sort_values("mean").head(5).round(4).to_string(index=False))
    st = nv.groupby(["city", "label_type"]).correct.agg(["size", "mean"]).reset_index()
    st = st[(st["size"] >= 300) & (st["mean"] < GATE)]
    st["city"] = [city_label(c, public) or "(a private deployment)" for c in st.city]
    print(f"  city x type strata with >= 300 new votes below {GATE}:")
    print(st.round(3).to_string(index=False))

    print("\n== Triage what-if (SidewalkWebpage ValidationQueuePolicy; only the AI's vote changes) ==")
    for name, acc in (("old", d.ai_accuracy), ("new", d.new_accuracy)):
        vote = np.where(acc >= GATE, d.ai_result, "Unsure")
        A = d.h_agree + (vote == "Agree")
        D = d.h_dis + (vote == "Disagree")
        U = d.h_unsure + (vote == "Unsure")
        margin, tot = (A - D).abs(), A + D + U
        capped = (margin < 2) & (tot >= 5)
        unsure = (margin < 2) & (U >= 2) & (U >= A + D)
        contested = (margin < 2) & (((vote == "Agree") & (D > A - 1)) | ((vote == "Disagree") & (A > D - 1)))
        if name == "old":
            old = (capped, unsure, contested, capped | unsure | contested)
        else:
            new = (capped, unsure, contested, capped | unsure | contested)
    for i, k in enumerate(["crowdCappedOut", "unsureHeavy", "aiContested", "triage (any)"]):
        print(f"  {k:15s} {old[i].sum():7,} -> {new[i].sum():7,}  (newly in {(new[i] & ~old[i]).sum():,}, "
              f"out {(old[i] & ~new[i]).sum():,})")


if __name__ == "__main__":
    main()
