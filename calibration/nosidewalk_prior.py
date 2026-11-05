"""NoSidewalk's deployment prior from the block-face structure of production labels and a measured precision per
face size (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42).

Why not the margin-rule count. For the five validated types ``data_generation.deployment_prior_entry`` uses the
labels the crowd margin decided, as the production class mix. For NoSidewalk that sample is 79 correct / 80 incorrect
(2026-11-05) and it is biased by the queue: SidewalkWebpage serves NoSidewalk per block face and prioritises faces
with a single labeler, which is where the false calls concentrate, so the labels that collect enough votes are the
doubtful ones.

The method. SidewalkWebpage #5222 (``SidewalkPresenceBasis.scala``) measured, against Seattle's sidewalk inventory,
how often a face's NoSidewalk call is right by how many NoSidewalk labels the face carries: one label 69 %, two 80 %,
three or more 84 %. The prior is that precision weighted by the share of production NoSidewalk labels on faces of
each size, counted from rawLabels:

    P(correct) = sum_k share_k * precision_k,   k in {1, 2, 3+}

written as pseudo-counts (``correct = round(P * N)``, ``incorrect = N - correct``, ``N`` = labels with a face) into
``deployment_priors.json`` with ``data_generation.update_deployment_priors``, so ``calibration/calibrate.py`` reads it
like any other prior. Only the ratio matters to calibration. Caveats, also in ``docs/decisions.md``: the precisions
are from one city, measured per face rather than per label, and the inventory has its own errors. Revisit with
production outcomes once the validator votes on NoSidewalk.

Run it after ``data_generation.py`` (which writes a margin-only placeholder for the type) and before calibration::

    python -m calibration.nosidewalk_prior --cache rawlabels-ns --priors out/deployment_priors.json
"""
import argparse
import json
import os
from datetime import date

import data_generation as dg
import nosidewalk_faces as nf

#: SidewalkWebpage #5222 (SidewalkPresenceBasis.scala): share of NoSidewalk calls the Seattle sidewalk inventory
#: confirms, by NoSidewalk labels on the face (3 means 3 or more).
FACE_SIZE_PRECISION = {1: 0.69, 2: 0.80, 3: 0.84}
SOURCE_URL = "https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5222"


def face_size_shares(rows):
    """Share of labels on faces with 1, 2 and 3+ labels, and the counts behind it: ``(shares, n_with_face,
    n_without_face)``. ``rows`` needs ``face_id`` (``nosidewalk_faces.face_id``).

    >>> import pandas as pd
    >>> s, n, none = face_size_shares(pd.DataFrame({"face_id": ["a", "b", "b", "c", "c", "c", "c", ""]}))
    >>> s, n, none
    ({1: 0.142857, 2: 0.285714, 3: 0.571429}, 7, 1)
    """
    has = rows["face_id"].astype(str) != ""
    size = rows[has].groupby("face_id")["face_id"].transform("size").clip(upper=3)
    n = int(has.sum())
    shares = {k: (round(float((size == k).sum()) / n, 6) if n else 0.0) for k in (1, 2, 3)}
    return shares, n, int((~has).sum())


def prior_entry(rows, cities, measured_on, precision=FACE_SIZE_PRECISION):
    """The ``deployment_priors.json`` entry for NoSidewalk from its rawLabels rows (human-authored only).

    >>> import pandas as pd
    >>> e = prior_entry(pd.DataFrame({"face_id": ["a", "b", "b"], "user_id": ["u"] * 3}), ["x"], "2026-11-05")
    >>> e["correct"], e["incorrect"], e["p_correct"]
    (2, 1, 0.763333)
    """
    human = rows[rows["user_id"].map(dg.label_author) == "human"] if "user_id" in rows else rows
    shares, n, without = face_size_shares(human)
    if n == 0:
        raise ValueError("no NoSidewalk label with a block face: cannot weight the face-size precisions")
    p = sum(shares[k] * precision[k] for k in (1, 2, 3))
    correct = int(round(p * n))
    return {"correct": correct, "incorrect": n - correct, "p_correct": round(p, 6),
            "measured_on": measured_on,
            "source": (f"face-size method (issue #42): Seattle sidewalk-inventory precision of NoSidewalk by labels on "
                       f"the face, 1: {precision[1]}, 2: {precision[2]}, 3+: {precision[3]} ({SOURCE_URL}), weighted "
                       f"by the share of human-authored labels on faces of each size in rawLabels of "
                       f"{len(cities)} cities; pseudo-counts over the labels with a face"),
            "method": "face_size", "face_size_shares": {str(k): v for k, v in shares.items()},
            "face_size_precision": {str(k): v for k, v in precision.items()},
            "labels_with_face": n, "labels_without_face": without, "cities": list(cities), "n_rows": int(len(rows))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", required=True, help="rawLabels cache (analysis/fetch_all.py, or data_generation's)")
    p.add_argument("--pattern", default="{city}.csv", help="file name inside the cache ({city}, {label_type})")
    p.add_argument("--label-type", default="NoSidewalk")
    p.add_argument("--priors", required=True, help="deployment_priors.json to update (other types are kept)")
    p.add_argument("--measured-on", default=None, help="default: the cache's cities.json fetched_on, else today")
    args = p.parse_args(argv)
    from analysis.type_readiness import load_cache
    rows, cities = load_cache(args.cache, args.label_type, args.pattern)
    rows = nf.type_rows(rows, args.label_type)
    measured_on = args.measured_on
    listing = os.path.join(args.cache, dg.CITIES_CACHE_FILE)
    if measured_on is None and os.path.exists(listing):
        with open(listing) as f:
            measured_on = json.load(f).get("fetched_on")
    entry = prior_entry(rows, sorted(set(rows["city"])), measured_on or date.today().isoformat())
    dg.update_deployment_priors(args.priors, args.label_type, entry)
    print(f"{args.label_type} prior ({args.priors}): P(correct) {entry['p_correct']:.4f} from face-size shares "
          f"{entry['face_size_shares']} over {entry['labels_with_face']:,} labels with a face "
          f"({entry['labels_without_face']:,} without); pseudo-counts {entry['correct']:,} / {entry['incorrect']:,}")
    return entry


if __name__ == "__main__":
    main()
