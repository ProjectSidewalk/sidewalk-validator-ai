"""NoSidewalk runs: validate the run, not the label? (issue #102, ``docs/nosidewalk-runs-2026-11.md``).

NoSidewalk labels come in sequences: one labeler drops a label every panorama or so along a block face that has no
sidewalk. A **run** is one labeler's NoSidewalk labels on one block face, ``(city, street_edge_id, street_side,
user_id)``, contiguous or not. The validator scores each label alone. This module asks the four questions of the
issue on the data that already exists, and changes nothing that is served:

1. ``agreement`` (item 1, descriptive): do the labels of a run get the same **expert** verdict more often than
   chance, and is that the run, the labeler, or the expert's own sitting? Each label's own expert votes are read from
   ``validations`` (never ``label_class``, which the face rule copies across a face). Nulls permute verdicts within
   a city and within a labeler; pairs are split by the gap between the expert's two votes (``pull_expert_vote_times.sh``),
   the page the vote was cast on, and same versus different expert.
2. ``pooling`` (item 2): an image score pooled over the run or the face (mean logit) against the single-label score,
   on step (b) (masked, one rated label per face; primary) and step (a) (out-of-fold, secondary: its truth is
   face-copied or from same-sitting votes). A pooled score is served through a map fitted on pooled scores of the
   same group definition (:func:`pooled_map`), never through the single-label map.
3. ``features`` (item 3): neighbour features that need no image (run length, labelers on the face, distance to the
   nearest NoSidewalk label, contradicting CurbRamp / Obstacle / SurfaceProblem labels on the face) in a second-stage
   logistic model, fitted on step (a) label-level truth and transferred, frozen, to step (b).
4. ``report``: the census of runs for every type, the tables of 1-3, and the recommendation rule of the plan
   (``docs/nosidewalk-runs-2026-11.md``), spliced between ``<!-- nosidewalk-runs-results:start -->`` and
   ``<!-- nosidewalk-runs-results:end -->``.

``step-a-neighbours`` lists every NoSidewalk label on the step (a) faces (out of fold and held out) for one klone
scoring job with the kept gate2 weights (``nosidewalk_round.slurm``), so step (a) can be pooled over whole faces as
serving would.

Keys: labels are ``label_uid = "<city>:<label_id>"`` (per-city serials), faces ``"<city>:<street_edge_id>:<side>"``.
User ids are read in memory only: outputs carry ``labeler_idx`` (dense per city) and ``expert_set`` ("1", "2",
"1+2", by position in ``EXPERTS``). Rows with label ids are written outside the repository only; the repository
gets aggregates. Cities that are not public in the day's ``/v3/api/cities`` snapshot are dropped before any statistic.

Usage (``D=~/Data/sidewalk-validator-ai-nosidewalk-2026-11``; the full run order is on the doc page)::

    EXPERTS="<name>=<uuid>,<name>=<uuid>" python analysis/nosidewalk_runs.py step-a-neighbours \\
        --pull $D/data/rawlabels-ns-2026-11-05 --gate $D/gate2 --cities-json $D/runs/cities_<date>.json \\
        --out $D/runs/step-a
    EXPERTS=... python analysis/nosidewalk_runs.py agreement --ns-pull ... --cityparams ... --cities-json ... \\
        --vote-times ... --out-json $D/runs/agreement.json
    python analysis/nosidewalk_runs.py report --agreement $D/runs/agreement.json --pooling $D/runs/pooling.json \\
        --features $D/runs/features.json --cities-json ... --out-json docs/nosidewalk-runs-2026-11/runs.json \\
        --out-md docs/nosidewalk-runs-2026-11/runs.md --doc docs/nosidewalk-runs-2026-11.md
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generation as dg  # noqa: E402
import nosidewalk_faces  # noqa: E402
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis import nosidewalk_gate as ng  # noqa: E402
from analysis import nosidewalk_round as nr  # noqa: E402
from analysis import slices  # noqa: E402
from analysis.nosidewalk_pilot import without_nan  # noqa: E402
from calibration import calibrate as cal  # noqa: E402

LABEL_TYPE = "NoSidewalk"
#: The types of item 1's like-for-like comparison (all on one pull).
ITEM1_TYPES = ("NoSidewalk", "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk", "Signal")
#: Labels of these types imply a sidewalk on the face (item 3).
CONTRA_TYPES = ("CurbRamp", "Obstacle", "SurfaceProblem")
#: Cities named in item 1's per-city table; every other public city is pooled as "other".
NAMED_CITIES = ("seattle-wa", "chicago-il", "oradell-nj")
#: A city enters the macro mean of item 1 with at least this many multi-label runs.
MIN_RUNS = 10
#: Vote-gap strata, in hours: [0, 1), [1, 24), [24, inf).
GAP_EDGES = (1.0, 24.0)
GAP_LABELS = ("< 1 h", "1 h - 1 day", ">= 1 day")
#: Pooled groups of item 2, in report order; "single" is the label alone.
GROUPS = ("single", "run", "face", "face_loo", "run_pano", "near50")
GROUP_TEXT = {"single": "the label alone", "run": "the labeler's run on the face (R, primary)",
              "face": "the whole face (F)", "face_loo": "the face without the label itself",
              "run_pano": "the run, one vote per panorama (P)", "near50": "labels within 50 m on the face (D)"}
NEAR_M = 50.0
START, END = "<!-- nosidewalk-runs-results:start -->", "<!-- nosidewalk-runs-results:end -->"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
#: ``label_validation.source`` values grouped for the source stratum.
SOURCE_GROUPS = {"Validate": "Validate", "ExpertValidate": "Validate", "ValidateMobile": "Validate",
                 "AdminUserDashboard": "AdminUserDashboard", "AdminContributionsTab": "AdminUserDashboard",
                 "AdminLabelSearchTab": "AdminUserDashboard", "LabelMap": "LabelMap",
                 "Old data, unknown source": "Old data"}
STEP_A_FACE_COLUMNS = ["label_uid", "city", "label_id", "face_id", "pano_id", "fold_of_face", "heldout", "in_split",
                       "expert_placed", "ai_placed", "bad_geometry", "scored_new", *[f"trained_pano_{k}" for k in
                                                                                       range(5)],
                       "labeler_idx", "time_created", "latitude", "longitude", "severity", "tags",
                       "own_expert_verdict", "expert_set"]
STEP_A_OUTPUTS = ("step_a_faces.csv", "labels.csv", "stage_files.txt", "step_a.json")


# ---------------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------------

def public_cities(cities_json):
    """The set of ``city_id``s with ``visibility == "public"`` in a ``/v3/api/cities`` snapshot (a path or the parsed
    JSON, either the API's ``{"status", "cities": [...]}`` or a bare list).

    >>> sorted(public_cities({"status": "OK", "cities": [{"city_id": "a", "visibility": "public"},
    ...                                                  {"city_id": "b", "visibility": "private"}]}))
    ['a']
    """
    data = json.load(open(cities_json)) if isinstance(cities_json, str) else cities_json
    rows = data["cities"] if isinstance(data, dict) else data
    return {c["city_id"] for c in rows if c.get("visibility") == "public"}


def expert_order(value):
    """Expert user_ids from an ``EXPERTS`` value in the order given, so ``expert_set`` can number them (1, 2, ...).
    Validated by ``data_generation.parse_experts``; the ids stay in memory.

    >>> expert_order("a=u1, b=u2")
    ('u1', 'u2')
    >>> expert_order("")
    ()
    """
    ids = dg.parse_experts(value)
    order = [kv.split("=", 1)[1].strip() for kv in (value or "").split(",") if "=" in kv and kv.split("=", 1)[1].strip()]
    assert set(order) == set(ids)
    return tuple(dict.fromkeys(order))


def sha_sums(paths):
    """``{basename: sha256}`` of ``paths`` and the sha256 of their sorted ``shasum -a 256`` listing (the form of a
    ``SHA256SUMS`` file), so one value pins a whole pull."""
    shas = {os.path.basename(p): rl.sha256_file(p) for p in paths}
    listing = "".join(f"{h}  {n}\n" for n, h in sorted(shas.items()))
    import hashlib
    return shas, hashlib.sha256(listing.encode()).hexdigest()


def load_pull(directory, label_types, cities, usecols=None):
    """Concatenate the per-city rawLabels CSVs ``<city>.csv`` of ``directory`` for ``cities`` (those present), keeping
    ``label_types``; adds ``city``, ``label_uid`` and ``face_id`` and asserts ``label_uid`` is unique. Returns
    ``(frame, {file: sha256})``. ``usecols`` limits the columns read (the 2026-10-18 all-type pull is about 1 GB)."""
    frames, shas = [], {}
    need = {"label_id", "label_type", "street_edge_id", "street_side"}
    cols = None if usecols is None else sorted(set(usecols) | need)
    for city in sorted(cities):
        path = os.path.join(directory, f"{city}.csv")
        if not os.path.exists(path):
            continue
        shas[f"{city}.csv"] = rl.sha256_file(path)
        df = pd.read_csv(path, low_memory=False, usecols=cols,
                         dtype={"label_id": str, "pano_id": str, "user_id": str, "street_side": str})
        df = df[df["label_type"].isin(label_types)].copy()
        if len(df):
            df["city"] = city
            frames.append(df)
    if not frames:
        raise SystemExit(f"no <city>.csv for {len(cities)} cities under {directory}")
    df = pd.concat(frames, ignore_index=True)
    df["label_uid"] = df["city"] + ":" + df["label_id"].astype(str)
    if not df["label_uid"].is_unique:
        raise ValueError(f"{directory}: duplicate label_uid")
    df["face_id"] = nosidewalk_faces.face_id(df)
    return df, shas


def run_ids(face_id, label_type, labeler_idx):
    """The one run-id format, ``<face_id>:<label_type>:L<labeler_idx>``, empty without a face (array-like inputs).

    >>> run_ids(pd.Series(["a:1:left", ""]), "NoSidewalk", pd.Series([3, 0])).tolist()
    ['a:1:left:NoSidewalk:L3', '']
    """
    face = pd.Series(face_id).astype(str).reset_index(drop=True)
    typ = pd.Series(label_type, index=face.index).astype(str) if np.ndim(label_type) == 0 else \
        pd.Series(label_type).astype(str).reset_index(drop=True)
    idx = pd.Series(labeler_idx).reset_index(drop=True).astype(int).astype(str)
    return pd.Series(np.where(face != "", face + ":" + typ + ":L" + idx, ""), dtype=object)


def add_runs(df):
    """``labeler_idx`` (dense per city, in order of each labeler's first label) and ``run_id`` =
    ``<face_id>:L<labeler_idx>`` (empty without a face). A run is one labeler's labels of one type on one face; the
    ``user_id`` never leaves memory. Rows are kept in their order.

    >>> d = pd.DataFrame({"city": ["a", "a", "a", "b"], "user_id": ["u", "v", "u", "u"],
    ...                   "face_id": ["a:1:left", "a:1:left", "", "b:1:left"], "label_type": "NoSidewalk",
    ...                   "time_created": ["2026-01-02", "2026-01-01", "2026-01-03", "2026-01-01"],
    ...                   "label_id": ["1", "2", "3", "1"]})
    >>> add_runs(d)[["labeler_idx", "run_id"]].values.tolist()
    [[1, 'a:1:left:NoSidewalk:L1'], [0, 'a:1:left:NoSidewalk:L0'], [1, ''], [0, 'b:1:left:NoSidewalk:L0']]
    """
    df = df.drop(columns=[c for c in ("labeler_idx", "run_id") if c in df])
    if df["user_id"].isna().any():
        raise ValueError("a label has no user_id: its labeler cannot be numbered")
    t = pd.to_datetime(df["time_created"], utc=True, format="ISO8601")
    order = (df.assign(_t=t, _id=pd.to_numeric(df["label_id"]))
             .sort_values(["city", "_t", "_id"], kind="mergesort"))
    first = order.drop_duplicates(["city", "user_id"])
    idx = first.assign(labeler_idx=first.groupby("city").cumcount())[["city", "user_id", "labeler_idx"]]
    df = df.merge(idx, on=["city", "user_id"], how="left", validate="many_to_one")
    df["run_id"] = run_ids(df["face_id"], df["label_type"], df["labeler_idx"]).to_numpy()
    return df


def own_votes(validations_json):
    """``{user_id: "Agree"|"Disagree"}`` of the human votes on one label (Unsure and AI votes ignored)."""
    return {u: v for u, v in nosidewalk_faces._human_votes(validations_json) if u is not None}


def expert_label_verdicts(raw, experts, rule="experts_only"):
    """One row per label with an expert verdict **on that label itself**, and counts of what was dropped.

    ``experts`` is the ordered tuple of :func:`expert_order`. ``rule``:

    - ``experts_only`` (primary): every expert Agree/Disagree on the label points the same way; crowd votes ignored;
    - ``dg_strict``: ``data_generation.expert_verdict`` (a contradicting crowd vote drops the label; sensitivity).

    Labels placed by an expert or by SidewalkAI are dropped and counted. Never reads ``correct``, ``label_class``,
    crowd margins or face verdicts. ``expert_set`` numbers the experts who voted ("1", "2", "1+2").

    >>> v = lambda *vs: json.dumps([{"user_id": u, "validation": x, "validator_type": "Human"} for u, x in vs])
    >>> raw = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4", "a:5"], "city": "a",
    ...     "user_id": ["x", "x", "e1", "x", "x"], "validations": [v(("e1", "Agree"), ("c", "Disagree")),
    ...     v(("e2", "Disagree")), v(("e2", "Agree")), v(("e1", "Agree"), ("e2", "Disagree")), "[]"]})
    >>> d, n = expert_label_verdicts(raw, ("e1", "e2"))
    >>> d[["label_uid", "verdict", "expert_set"]].values.tolist(), n["expert_placed"], n["experts_split"]
    ([['a:1', 'correct', '1'], ['a:2', 'incorrect', '2']], 1, 1)
    >>> expert_label_verdicts(raw, ("e1", "e2"), "dg_strict")[0]["label_uid"].tolist()
    ['a:2']
    """
    if rule not in ("experts_only", "dg_strict"):
        raise ValueError(rule)
    if not experts:
        raise SystemExit("EXPERTS is empty: item 1 needs the expert accounts (EXPERTS=\"<name>=<uuid>,...\")")
    pos = {e: i + 1 for i, e in enumerate(experts)}
    author = raw["user_id"].astype(str)
    expert_placed = author.isin(pos)
    ai_placed = author.map(dg.label_author).eq("ai")
    keep = ~expert_placed & ~ai_placed
    verdicts, sets, n_split, n_strict_drop = [], [], 0, 0
    for vj in raw.loc[keep, "validations"]:
        ev = {u: x for u, x in own_votes(vj).items() if u in pos}
        values = set(ev.values())
        s = "+".join(str(pos[u]) for u in sorted(ev, key=pos.get))
        if len(values) != 1:
            n_split += len(values) > 1
            verdicts.append("")
        else:
            verdict = "correct" if values == {"Agree"} else "incorrect"
            if rule == "dg_strict" and dg.expert_verdict(vj, frozenset(pos)) != verdict:
                n_strict_drop += 1
                verdict = ""
            verdicts.append(verdict)
        sets.append(s)
    d = raw.loc[keep].copy()
    d["verdict"], d["expert_set"] = verdicts, sets
    counts = {"labels": int(len(raw)), "expert_placed": int(expert_placed.sum()),
              "ai_placed": int((ai_placed & ~expert_placed).sum()), "experts_split": int(n_split),
              "crowd_contradicts_dropped": int(n_strict_drop)}
    d = d[d["verdict"] != ""].reset_index(drop=True)
    counts["with_verdict"] = int(len(d))
    return d, counts


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (straight line; a proxy for distance along the street).

    >>> round(haversine_m(47.0, -122.0, 47.0, -122.001))
    76
    """
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------------------------------------------------
# step-a-neighbours (R3 inputs)
# ---------------------------------------------------------------------------------------------------------------------

def gate_split(gate):
    """The gate run's rows: fold 0's ``split.csv`` (every fold carries every row) with ``fold`` from ``folds.csv``
    (NaN for ``test_heldout``), and per fold the panoramas that fold's model trained or validated on."""
    with open(os.path.join(gate, "folds.json")) as f:
        n_folds = json.load(f)["folds"]
    split = pd.read_csv(os.path.join(gate, "fold0", "split.csv"), dtype=str, keep_default_na=False)
    folds = pd.read_csv(os.path.join(gate, "folds.csv"), dtype={"label_uid": str})
    split = split.merge(folds, on="label_uid", how="left", validate="one_to_one")
    trained = {}
    for k in range(n_folds):
        s = pd.read_csv(os.path.join(gate, f"fold{k}", "split.csv"), dtype=str, keep_default_na=False,
                        usecols=["label_uid", "pano_id", "face_id", "split"])
        trained[k] = set(s.loc[s["split"].isin(["train", "val"]), "pano_id"])
        own = set(s.loc[s["split"] == "test", "face_id"]) - {""}
        if own & set(s.loc[s["split"].isin(["train", "val"]), "face_id"]):
            raise ValueError(f"fold {k}: a face of its test rows is also in its train or val rows")
        if set(s.loc[s["split"] == "test", "label_uid"]) != set(folds.loc[folds["fold"] == k, "label_uid"]):
            raise ValueError(f"fold {k}: split.csv test rows are not folds.csv's fold {k}")
    return split, trained, n_folds


def step_a_faces(pull, split, trained, experts):
    """Every NoSidewalk label of ``pull`` (11-05) on a step (a) face: the out-of-fold faces (``fold_of_face``) and the
    held-out faces (``heldout``). Asserts that the pull reproduces every split row's face and that no face straddles
    folds. ``scored_new`` marks what to crop: human-placed, usable pano point (split rows included, so both arms of
    item 2 share one crop path). ``own_expert_verdict`` is :func:`expert_label_verdicts` (``experts_only``)."""
    split = split.copy()
    in_pull = pull.set_index("label_uid")["face_id"]
    missing = sorted(set(split["label_uid"]) - set(in_pull.index))
    if missing:
        raise ValueError(f"{len(missing)} split rows are not in the pull (e.g. {missing[:3]})")
    moved = split[split["face_id"] != split["label_uid"].map(in_pull)]
    if len(moved):
        raise ValueError(f"{len(moved)} split rows sit on another face in this pull (e.g. "
                         f"{moved['label_uid'].tolist()[:3]}): faces must come from the pull that defined them")
    s = split[split["face_id"] != ""]
    fold_of = s[s["split"] != "test_heldout"].groupby("face_id")["fold"].agg(lambda x: set(x.astype(int)))
    if (fold_of.map(len) > 1).any():
        raise ValueError("a face straddles folds")
    fold_of = fold_of.map(lambda x: next(iter(x)))
    heldout_faces = set(s.loc[s["split"] == "test_heldout", "face_id"])
    if heldout_faces & set(fold_of.index):
        raise ValueError("a face is both out of fold and held out")
    df = pull[pull["face_id"].isin(set(fold_of.index) | heldout_faces)].copy()
    df = add_runs(df)
    df["fold_of_face"] = df["face_id"].map(fold_of)
    df["heldout"] = df["face_id"].isin(heldout_faces).astype(int)
    df["in_split"] = df["label_uid"].isin(set(split["label_uid"])).astype(int)
    df["expert_placed"] = df["user_id"].astype(str).isin(set(experts)).astype(int)
    df["ai_placed"] = df["user_id"].map(dg.label_author).eq("ai").astype(int)
    ok, _ = dg.valid_geometry(df)
    df["bad_geometry"] = (~ok).astype(int)
    df["scored_new"] = ((df["ai_placed"] == 0) & (df["bad_geometry"] == 0)).astype(int)
    for k, panos in trained.items():
        df[f"trained_pano_{k}"] = df["pano_id"].astype(str).isin(panos).astype(int)
    v, _ = expert_label_verdicts(df, experts)
    df = df.merge(v[["label_uid", "verdict", "expert_set"]].rename(columns={"verdict": "own_expert_verdict"}),
                  on="label_uid", how="left", validate="one_to_one")
    df[["own_expert_verdict", "expert_set"]] = df[["own_expert_verdict", "expert_set"]].fillna("")
    lost = set(split.loc[split["face_id"] != "", "label_uid"]) - set(df["label_uid"])
    if lost:
        raise ValueError(f"{len(lost)} split rows with a face are missing from the listing")
    return df.sort_values(["face_id", "label_uid"], kind="mergesort").reset_index(drop=True)


def cmd_step_a_neighbours(args):
    if rl.inside_git_checkout(args.out):
        sys.exit(f"refusing to write label rows inside a git checkout: {args.out}")
    experts = expert_order(os.environ.get("EXPERTS", ""))
    check_cities(args.cities_json, args.allow_unpinned)
    pull_sums(args.pull, "ns_pull_2026_11_05_sums", args.allow_unpinned)
    split, trained, n_folds = gate_split(args.gate)
    check_gate_pins(args.gate, n_folds, args.allow_unpinned)
    public = public_cities(args.cities_json)
    cities = sorted(set(split["city"]))
    private = sorted(set(cities) - public)
    if private:
        sys.exit(f"{len(private)} step (a) cities are not public in {args.cities_json}")
    pull, shas = load_pull(args.pull, [LABEL_TYPE], cities)
    faces = step_a_faces(pull, split, trained, experts)
    os.makedirs(args.out, exist_ok=True)
    new = nr.crop_labels(faces[faces["scored_new"] == 1].assign(author="human"))
    with nr.guarded_outputs(args.out, STEP_A_OUTPUTS, args.force) as tmp:
        faces[STEP_A_FACE_COLUMNS].to_csv(tmp["step_a_faces.csv"], index=False, lineterminator="\n")
        new[nr.ale.LABEL_COLUMNS + ["face_id"]].to_csv(tmp["labels.csv"], index=False, lineterminator="\n")
        panos = nr.write_stage_files(new, tmp["stage_files.txt"])
        oof = faces[faces["heldout"] == 0]
        summary = {"pull_sha256": shas, "pull_sums_sha256": sha_sums([os.path.join(args.pull, n) for n in shas])[1],
                   "gate_folds_json_sha256": rl.sha256_file(os.path.join(args.gate, "folds.json")),
                   "gate_folds_csv_sha256": rl.sha256_file(os.path.join(args.gate, "folds.csv")),
                   "gate_split_sha256": {str(k): rl.sha256_file(os.path.join(args.gate, f"fold{k}", "split.csv"))
                                         for k in range(n_folds)},
                   "cities_json_sha256": rl.sha256_file(args.cities_json),
                   "faces_oof": int(oof["face_id"].nunique()), "faces_heldout": int(faces.loc[faces["heldout"] == 1,
                                                                                               "face_id"].nunique()),
                   "labels": int(len(faces)), "labels_oof": int(len(oof)), "in_split": int(faces["in_split"].sum()),
                   "expert_placed": int(faces["expert_placed"].sum()), "ai_placed": int(faces["ai_placed"].sum()),
                   "bad_geometry": int(faces["bad_geometry"].sum()), "to_score": int(len(new)), "panos": len(panos),
                   "own_expert_verdict": int((faces["own_expert_verdict"] != "").sum()),
                   "own_expert_verdict_not_in_split": int(((faces["own_expert_verdict"] != "")
                                                           & (faces["in_split"] == 0)).sum()),
                   "step_a_faces_sha256": rl.sha256_file(tmp["step_a_faces.csv"]),
                   "labels_sha256": rl.sha256_file(tmp["labels.csv"])}
        with open(tmp["step_a.json"], "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# item 1: within-run agreement of expert verdicts (descriptive)
# ---------------------------------------------------------------------------------------------------------------------

def group_pairs(y, codes, n_groups=None):
    """Per group: ``(pairs, same)``, the number of within-group label pairs and of those with the same verdict, from
    0/1 verdicts ``y`` and group codes ``codes`` (0..G-1). Counts only, so no pair is enumerated.

    >>> p, s = group_pairs(np.array([1, 1, 0, 1, 1]), np.array([0, 0, 0, 1, 1]))
    >>> p.tolist(), s.tolist()
    ([3.0, 1.0], [1.0, 1.0])
    """
    n = np.bincount(codes, minlength=n_groups or 0).astype(float)
    c = np.bincount(codes, weights=np.asarray(y, float), minlength=n_groups or 0)
    return n * (n - 1) / 2, c * (c - 1) / 2 + (n - c) * (n - c - 1) / 2


def pair_agreement(d, group="run_id", weight="pair"):
    """Share of within-group label pairs whose two verdicts agree. ``weight="pair"`` pools the pairs;
    ``weight="run"`` averages each multi-label group's own share. NaN without pairs.

    >>> d = pd.DataFrame({"run_id": list("aaabb"), "verdict": ["correct", "correct", "incorrect", "correct", "correct"]})
    >>> round(pair_agreement(d), 3), round(pair_agreement(d, weight="run"), 3)
    (0.5, 0.667)
    """
    d = d[d[group] != ""]
    if d.empty:
        return float("nan")
    codes, _ = pd.factorize(d[group])
    pairs, same = group_pairs((d["verdict"] == "correct").to_numpy(), codes)
    if not pairs.sum():
        return float("nan")
    if weight == "pair":
        return float(same.sum() / pairs.sum())
    m = pairs > 0
    return float(np.mean(same[m] / pairs[m]))


def homogeneous_share(d, group="run_id"):
    """Share of groups with 2+ labels whose verdicts are all the same (each group counts once).

    >>> homogeneous_share(pd.DataFrame({"run_id": list("aabbc"), "verdict": ["correct", "correct", "correct",
    ...                                                                        "incorrect", "correct"]}))
    0.5
    """
    d = d[d[group] != ""]
    g = d.groupby(group)["verdict"].agg(["size", "nunique"])
    g = g[g["size"] >= 2]
    return float((g["nunique"] == 1).mean()) if len(g) else float("nan")


def icc_binary(d, group="run_id"):
    """One-way ANOVA ICC(1) of a 0/1 verdict over groups of unequal size (groups with 2+ labels only), with the usual
    ``n0 = (N - sum n_i^2 / N) / (k - 1)``. Call it within one city: pooled over cities, base-rate differences
    between cities would read as run effect. NaN with fewer than two groups or no variance.

    >>> d = pd.DataFrame({"run_id": list("aabbcc"), "verdict": ["correct"] * 2 + ["incorrect"] * 2 + ["correct"] * 2})
    >>> icc_binary(d)
    1.0
    >>> d2 = pd.DataFrame({"run_id": list("aabb"), "verdict": ["correct", "incorrect"] * 2})
    >>> round(icc_binary(d2), 3)
    -1.0
    """
    d = d[d[group] != ""]
    size = d.groupby(group)[group].transform("size")
    d = d[size >= 2]
    y = (d["verdict"] == "correct").astype(float).to_numpy()
    codes, _ = pd.factorize(d[group])
    k, n_total = codes.max() + 1 if len(codes) else 0, len(y)
    if k < 2:
        return float("nan")
    n = np.bincount(codes).astype(float)
    means = np.bincount(codes, weights=y) / n
    grand = y.mean()
    msb = float(np.sum(n * (means - grand) ** 2) / (k - 1))
    msw = float(np.sum((y - means[codes]) ** 2) / (n_total - k))
    n0 = (n_total - np.sum(n ** 2) / n_total) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def permute_within(strata, rng):
    """A permutation of positions that only moves each position within its stratum (``strata`` are codes).

    >>> s = np.array([0, 1, 0, 1, 1])
    >>> p = permute_within(s, np.random.default_rng(0))
    >>> bool((s[p] == s).all()), sorted(p.tolist())
    (True, [0, 1, 2, 3, 4])
    """
    strata = np.asarray(strata)
    home = np.argsort(strata, kind="stable")
    shuffled = np.lexsort((rng.random(len(strata)), strata))
    out = np.empty(len(strata), dtype=int)
    out[home] = shuffled
    return out


def permutation_null(d, strata_cols, n_perm=1000, seed=0, group="run_id"):
    """The within-run pair agreement under verdicts permuted within ``strata_cols`` (``["city"]``: null 1, the
    city's base rate; ``["city", "labeler_idx"]``: null 2, the labeler across their faces), keeping every run's size.
    Returns the ``n_perm`` null values (pair-weighted)."""
    d = d[d[group] != ""]
    codes, _ = pd.factorize(d[group])
    strata, _ = pd.factorize(pd.MultiIndex.from_frame(d[strata_cols].astype(str)))
    y = (d["verdict"] == "correct").to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for i in range(n_perm):
        pairs, same = group_pairs(y[permute_within(strata, rng)], codes)
        out[i] = same.sum() / pairs.sum() if pairs.sum() else np.nan
    return out


def null_summary(observed, null):
    """``{mean, sd, p}``: the one-sided permutation p-value ``(1 + #(null >= observed)) / (1 + n)``."""
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(observed):
        return {"mean": None, "sd": None, "p": None}
    return {"mean": float(null.mean()), "sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
            "p": float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))}


def reference_agreements(d):
    """Pair agreement of three reference sets, from counts: the same labeler on **another** face of the city, the
    same face but **another** labeler, and any two labels of the city (``p^2 + (1-p)^2`` up to the finite-sample
    term)."""
    y = (d["verdict"] == "correct").to_numpy()

    def tot(key):
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(d[key].astype(str)) if len(key) > 1 else d[key[0]])
        p, s = group_pairs(y, codes)
        return p.sum(), s.sum()

    with_face = d["face_id"] != ""
    run_p, run_s = tot(["run_id"]) if with_face.all() else (0.0, 0.0)
    lab_p, lab_s = tot(["city", "labeler_idx"])
    face_p, face_s = tot(["face_id"]) if with_face.all() else (0.0, 0.0)
    city_p, city_s = tot(["city"])
    share = lambda s, p: float(s / p) if p else float("nan")  # noqa: E731
    return {"same_labeler_other_face": share(lab_s - run_s, lab_p - run_p),
            "same_labeler_other_face_pairs": int(lab_p - run_p),
            "same_face_other_labeler": share(face_s - run_s, face_p - run_p),
            "same_face_other_labeler_pairs": int(face_p - run_p),
            "city_any": share(city_s, city_p)}


def run_boot(d, n_boot, seed, group="run_id"):
    """``n_boot`` pair agreements from resampling multi-label runs with replacement within each city."""
    d = d[d[group] != ""]
    codes, uniq = pd.factorize(d[group])
    pairs, same = group_pairs((d["verdict"] == "correct").to_numpy(), codes)
    city = d.groupby(codes)["city"].first().to_numpy()
    m = pairs > 0
    rng = np.random.default_rng(seed)
    blocks = [(pairs[m & (city == c)], same[m & (city == c)]) for c in np.unique(city[m])]
    out = np.empty(n_boot)
    for b in range(n_boot):
        p = s = 0.0
        for bp, bs in blocks:
            i = rng.integers(0, len(bp), len(bp))
            p, s = p + bp[i].sum(), s + bs[i].sum()
        out[b] = s / p if p else np.nan
    return out


def ci95(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def agreement_block(d, n_perm, n_boot, seed):
    """Every item-1 statistic of one set of labels (one city, or several pooled): counts, homogeneous share, pair
    agreement (pair- and run-weighted) with a run bootstrap CI, ICC (NaN across cities: use it per city), both
    nulls and the reference agreements."""
    d = d[d["run_id"] != ""]
    size = d.groupby("run_id")["run_id"].transform("size")
    obs = pair_agreement(d)
    out = {"labels": int(len(d)), "labels_in_multi_runs": int((size >= 2).sum()),
           "multi_runs": int(d.loc[size >= 2, "run_id"].nunique()),
           "labelers": int(d[["city", "labeler_idx"]].drop_duplicates().shape[0]),
           "pairs": int(group_pairs((d["verdict"] == "correct").to_numpy(), pd.factorize(d["run_id"])[0])[0].sum()),
           "share_correct": float((d["verdict"] == "correct").mean()) if len(d) else None,
           "pair_agreement": obs, "pair_agreement_run_weighted": pair_agreement(d, weight="run"),
           "homogeneous_share": homogeneous_share(d),
           "icc": icc_binary(d) if d["city"].nunique() == 1 else None}
    out["pair_agreement_ci95"] = ci95(run_boot(d, n_boot, seed)) if out["pairs"] else [None, None]
    out["null_city"] = null_summary(obs, permutation_null(d, ["city"], n_perm, seed)) if out["pairs"] else None
    out["null_labeler"] = (null_summary(obs, permutation_null(d, ["city", "labeler_idx"], n_perm, seed))
                           if out["pairs"] else None)
    out.update(reference_agreements(d))
    return out


def macro_block(d, n_perm, n_boot, seed, min_runs=MIN_RUNS):
    """The macro mean over cities with >= ``min_runs`` multi-label runs of each city's pair agreement, ICC, nulls and
    references, so no city's base rate or size dominates. The nulls and the CI are recomputed per permutation /
    resample and averaged over the same cities."""
    cities = []
    for c, g in d[d["run_id"] != ""].groupby("city"):
        size = g.groupby("run_id")["run_id"].transform("size")
        if g.loc[size >= 2, "run_id"].nunique() >= min_runs:
            cities.append(c)
    if not cities:
        return {"cities": [], "n_cities": 0}
    per = {c: d[d["city"] == c] for c in cities}
    obs = np.array([pair_agreement(g) for g in per.values()])
    n1 = np.vstack([permutation_null(g, ["city"], n_perm, seed) for g in per.values()]).mean(axis=0)
    n2 = np.vstack([permutation_null(g, ["city", "labeler_idx"], n_perm, seed) for g in per.values()]).mean(axis=0)
    boot = np.vstack([run_boot(g, n_boot, seed) for g in per.values()]).mean(axis=0)
    refs = [reference_agreements(g) for g in per.values()]
    mean = lambda xs: float(np.nanmean(xs)) if np.isfinite(np.asarray(xs, float)).any() else None  # noqa: E731
    return {"cities": sorted(cities), "n_cities": len(cities), "pair_agreement": float(obs.mean()),
            "pair_agreement_ci95": ci95(boot), "icc": mean([icc_binary(g) for g in per.values()]),
            "homogeneous_share": mean([homogeneous_share(g) for g in per.values()]),
            "null_city": null_summary(float(obs.mean()), n1), "null_labeler": null_summary(float(obs.mean()), n2),
            "same_labeler_other_face": mean([r["same_labeler_other_face"] for r in refs]),
            "same_face_other_labeler": mean([r["same_face_other_labeler"] for r in refs]),
            "city_any": mean([r["city_any"] for r in refs]),
            "labels": int(sum(len(g) for g in per.values())),
            "pairs": int(sum(group_pairs((g["verdict"] == "correct").to_numpy(),
                                         pd.factorize(g["run_id"])[0])[0].sum() for g in per.values()))}


def vote_times(path, schema_city, as_of):
    """``{(city, label_id, expert_idx): (end_timestamp, result, source)}`` from ``pull_expert_vote_times.sh``: each
    expert's latest vote on a label before ``as_of`` (the pull's date; a later vote is not in the pull). Cities
    outside ``schema_city`` are dropped."""
    t = pd.read_csv(path, dtype={"label_id": str, "schema": str})
    t["city"] = t["schema"].map(schema_city)
    t = t[t["city"].notna()].copy()
    t["end"] = pd.to_datetime(t["end_timestamp"], utc=True, format="ISO8601")
    t = t[t["end"] < pd.Timestamp(as_of, tz="UTC")]
    t = t.sort_values("end", kind="mergesort").drop_duplicates(["city", "label_id", "expert_idx"], keep="last")
    return {(c, i, int(e)): (end, r, s) for c, i, e, end, r, s in
            zip(t["city"], t["label_id"], t["expert_idx"], t["end"], t["validation_result"], t["source"])}


def gap_stratum(hours):
    """``GAP_LABELS`` stratum of a gap in hours, ``"unknown"`` when NaN.

    >>> [gap_stratum(h) for h in (0.2, 3, 30, float("nan"))]
    ['< 1 h', '1 h - 1 day', '>= 1 day', 'unknown']
    """
    if not np.isfinite(hours):
        return "unknown"
    return GAP_LABELS[int(np.searchsorted(GAP_EDGES, hours, side="right"))]


def source_group(source):
    s = str(source)
    if s.startswith("Gallery"):
        return "Gallery"
    return SOURCE_GROUPS.get(s, "other")


def pair_table(d, times=None, group="run_id"):
    """Every within-group pair of labels (``d`` from :func:`expert_label_verdicts` with ``e<i>`` vote columns and
    ``run_id``): ``same_verdict``, ``same_expert`` (an expert voted on both), ``gap_hours`` (the smallest gap between
    one shared expert's two votes, from ``times`` of :func:`vote_times`; NaN without), ``gap`` (stratum),
    ``source_pair`` (the page group of that expert's two votes, ``mixed`` when they differ), the two labels' tags and
    severities. Pair counts grow with the square of run length; NoSidewalk's are in the tens of thousands.

    >>> d = pd.DataFrame({"run_id": ["r", "r", "r"], "city": "a", "label_id": ["1", "2", "3"],
    ...                   "verdict": ["correct", "correct", "incorrect"], "e1": ["Agree", "Agree", ""],
    ...                   "e2": ["", "", "Disagree"], "tags": "[]", "severity": None})
    >>> t = {("a", "1", 1): (pd.Timestamp("2026-01-01T00:00Z"), "Agree", "Validate"),
    ...      ("a", "2", 1): (pd.Timestamp("2026-01-01T00:30Z"), "Agree", "LabelMap")}
    >>> p = pair_table(d, t)
    >>> p[["same_verdict", "same_expert", "gap", "source_pair"]].values.tolist()
    [[True, True, '< 1 h', 'mixed'], [False, False, 'different expert', 'different expert'], \
[False, False, 'different expert', 'different expert']]
    """
    ecols = sorted(c for c in d.columns if re.fullmatch(r"e\d+", c))
    rows = []
    for _, g in d[d[group] != ""].groupby(group, sort=True):
        if len(g) < 2:
            continue
        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                shared = [int(c[1:]) for c in ecols if a[c] and b[c]]
                gap, src = float("nan"), "different expert"
                if shared:
                    src = "unknown"
                    for e in shared:
                        ta = (times or {}).get((a["city"], str(a["label_id"]), e))
                        tb = (times or {}).get((b["city"], str(b["label_id"]), e))
                        if ta is None or tb is None or ta[1] != a[f"e{e}"] or tb[1] != b[f"e{e}"]:
                            continue  # no time, or a re-vote after the pull: the pull's vote has no time
                        h = abs((ta[0] - tb[0]).total_seconds()) / 3600
                        if not np.isfinite(gap) or h < gap:
                            sa, sb = source_group(ta[2]), source_group(tb[2])
                            gap, src = h, sa if sa == sb else "mixed"
                rows.append({"group": a[group], "city": a["city"], "same_verdict": a["verdict"] == b["verdict"],
                             "same_expert": bool(shared), "gap_hours": gap,
                             "gap": gap_stratum(gap) if shared else "different expert", "source_pair": src,
                             "tags_a": a["tags"], "tags_b": b["tags"], "severity_a": a["severity"],
                             "severity_b": b["severity"]})
    return pd.DataFrame(rows, columns=["group", "city", "same_verdict", "same_expert", "gap_hours", "gap",
                                       "source_pair", "tags_a", "tags_b", "severity_a", "severity_b"])


def time_match_counts(d, times):
    """How each expert vote of the pull (``e<i>`` columns) matches the timed votes: ``matched`` (the latest timed vote
    before the pull has the same value), ``revote`` (it has another value: the expert re-voted around the pull) and
    ``no_time`` (no timed vote before the pull).

    >>> d = pd.DataFrame({"city": "a", "label_id": ["1", "2", "3"], "e1": ["Agree", "Agree", ""]})
    >>> t = {("a", "1", 1): (None, "Agree", "Validate"), ("a", "2", 1): (None, "Disagree", "Validate")}
    >>> time_match_counts(d, t)
    {'matched': 1, 'revote': 1, 'no_time': 0}
    """
    out = {"matched": 0, "revote": 0, "no_time": 0}
    for c in sorted(c for c in d.columns if re.fullmatch(r"e\d+", c)):
        e = int(c[1:])
        for city, lid, v in zip(d["city"], d["label_id"].astype(str), d[c]):
            if not v:
                continue
            t = (times or {}).get((city, lid, e))
            out["no_time" if t is None else "matched" if t[1] == v else "revote"] += 1
    return out


def strata_table(pairs, col, min_n=slices.MIN_N):
    """``{stratum: {pairs, agreement}}`` of a pair table; agreement blank under ``min_n`` pairs."""
    out = {}
    for name, g in pairs.groupby(col):
        out[str(name)] = {"pairs": int(len(g)), "agreement": float(g["same_verdict"].mean()) if len(g) >= min_n
                          else None}
    return out


def pair_slices(pairs, by, min_n=slices.MIN_N):
    """Pairs whose two labels share a tag (or a severity) counted under it: ``{slice: {pairs, agreement}}``."""
    out = {}
    if by == "tag":
        both = [set(slices.parse_tags(a)) & set(slices.parse_tags(b)) for a, b in zip(pairs["tags_a"], pairs["tags_b"])]
    else:
        sa, sb = pd.to_numeric(pairs["severity_a"], errors="coerce"), pd.to_numeric(pairs["severity_b"], errors="coerce")
        both = [{str(int(a))} if np.isfinite(a) and a == b else set() for a, b in zip(sa, sb)]
    names = sorted({t for s in both for t in s})
    same = pairs["same_verdict"].to_numpy()
    for t in names:
        m = np.array([t in s for s in both])
        out[t] = {"pairs": int(m.sum()), "agreement": float(same[m].mean()) if m.sum() >= min_n else None}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["pairs"]))


def item1_labels(raw, experts, rule="experts_only"):
    """``(labels with an own expert verdict, a run and the expert vote columns e1, e2, ...; counts)``."""
    raw = add_runs(raw)
    d, counts = expert_label_verdicts(raw, experts, rule)
    pos = {e: i + 1 for i, e in enumerate(experts)}
    for e, i in pos.items():
        d[f"e{i}"] = [own_votes(v).get(e, "") for v in d["validations"]]
    counts["without_face"] = int((d["run_id"] == "").sum())
    d = d[d["run_id"] != ""].reset_index(drop=True)
    keep = ["label_uid", "city", "label_id", "label_type", "face_id", "labeler_idx", "run_id", "verdict", "expert_set",
            "tags", "severity", *[f"e{i}" for i in pos.values()]]
    return d[[c for c in keep if c in d.columns]], counts


def agreement_report(d, counts, times, n_perm, n_boot, seed, named=NAMED_CITIES):
    """Item 1 for one type: per named city, the other cities pooled as ``other``, the macro mean, and the pair
    strata (vote gap, page, same/different expert), plus tag and severity slices."""
    out = {"counts": counts, "per_city": {}}
    for c in named:
        g = d[d["city"] == c]
        if len(g):
            out["per_city"][c] = agreement_block(g, n_perm, n_boot, seed)
    other = d[~d["city"].isin(named)]
    if len(other):
        out["per_city"]["other"] = agreement_block(other, n_perm, n_boot, seed)
        out["per_city"]["other"]["n_cities"] = int(other["city"].nunique())
    out["macro"] = macro_block(d, n_perm, n_boot, seed)
    pairs = pair_table(d, times)
    out["pairs_total"] = int(len(pairs))
    out["vote_times_match"] = time_match_counts(d, times) if times is not None else None
    out["strata"] = {}
    for scope, p in (("all", pairs), *((c, pairs[pairs["city"] == c]) for c in named)):
        if len(p):
            out["strata"][scope] = {"expert": strata_table(p.assign(e=np.where(p["same_expert"], "same expert",
                                                                                  "different expert")), "e"),
                                    "gap": strata_table(p, "gap"), "source": strata_table(p[p["same_expert"]],
                                                                                          "source_pair")}
    out["slices"] = {"tag": pair_slices(pairs, "tag"), "severity": pair_slices(pairs, "severity")}
    return out


def run_census(raw, pooled=True):
    """The issue's run table per city (``raw`` of one type with ``face_id`` and ``user_id``): labels, share in a run
    of 2+ and 3+, the run length of the median label, and the share of multi-label faces with one labeler. A label
    without a face is a run of one.

    >>> raw = pd.DataFrame({"city": "a", "user_id": ["u", "u", "u", "v", "u"],
    ...                     "face_id": ["a:1:left"] * 4 + [""]})
    >>> run_census(raw, pooled=False)["a"]
    {'labels': 5, 'in_run_2': 0.6, 'in_run_3': 0.6, 'median_run_length': 3.0, 'multi_faces_one_labeler': 0.0, 'multi_faces': 1}
    """
    out = {}
    for city, g in raw.groupby("city"):
        has = g["face_id"] != ""
        run_len = pd.Series(1, index=g.index)
        run_len[has] = g[has].groupby(["face_id", "user_id"])["face_id"].transform("size")
        faces = g[has].groupby("face_id")["user_id"].agg(["size", "nunique"])
        multi = faces[faces["size"] >= 2]
        out[city] = {"labels": int(len(g)), "in_run_2": float((run_len >= 2).mean()),
                     "in_run_3": float((run_len >= 3).mean()), "median_run_length": float(run_len.median()),
                     "multi_faces_one_labeler": float((multi["nunique"] == 1).mean()) if len(multi) else None,
                     "multi_faces": int(len(multi))}
    if pooled and len(out) > 1:
        out["all"] = run_census(raw.assign(city="all"), pooled=False)["all"]
    elif pooled and out:
        out["all"] = dict(next(iter(out.values())))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# item 2: pooled image scores
# ---------------------------------------------------------------------------------------------------------------------

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def pooled_scores(df, how, score_col="logit", members=None):
    """The pooled logit of every row of ``df`` under ``how`` (one of ``GROUPS``), as a Series on ``df.index``.

    Group members are the rows where ``members`` is True (all rows by default) and the row itself, always: a filter
    on the neighbours never removes the label being judged. ``df`` needs ``face_id``, ``run_id``, ``pano_id`` and,
    for ``near50``, ``latitude``/``longitude``; rows without a face keep their own score.

    - ``single``: the row's own logit; ``run`` / ``face``: the mean over the group;
    - ``face_loo``: the face without the row itself (a row alone on its face keeps its own score);
    - ``run_pano``: the run with one value per panorama (each panorama's mean, then the mean of those);
    - ``near50``: the face's rows within ``NEAR_M`` metres in a straight line (a row without coordinates keeps its
      own score).

    >>> d = pd.DataFrame({"face_id": ["f", "f", "f", ""], "run_id": ["r", "r", "s", ""], "pano_id": ["p", "p", "q", "z"],
    ...                   "logit": [1.0, 3.0, 5.0, 7.0], "latitude": [0, 0, 1, 0], "longitude": 0.0})
    >>> [pooled_scores(d, h).tolist() for h in ("run", "face", "face_loo")]
    [[2.0, 2.0, 5.0, 7.0], [3.0, 3.0, 3.0, 7.0], [4.0, 3.0, 2.0, 7.0]]
    >>> pooled_scores(d, "near50").tolist(), pooled_scores(d, "face", members=pd.Series([True, True, False, True])).tolist()
    ([2.0, 2.0, 5.0, 7.0], [2.0, 2.0, 3.0, 7.0])
    """
    if how not in GROUPS:
        raise ValueError(f"how must be one of {GROUPS}")
    s = df[score_col].astype(float)
    if how == "single":
        return s.copy()
    m = pd.Series(True, index=df.index) if members is None else members.reindex(df.index).fillna(False).astype(bool)
    out = s.copy()
    if how == "near50":
        for _, g in df[df["face_id"] != ""].groupby("face_id"):
            if len(g) < 2:
                continue
            lat, lon = g["latitude"].to_numpy(float), g["longitude"].to_numpy(float)
            dist = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
            eye = np.eye(len(g), dtype=bool)  # the row itself always counts, even without coordinates
            near = ((dist <= NEAR_M) | eye) & (m.loc[g.index].to_numpy()[None, :] | eye)
            out.loc[g.index] = (near * s.loc[g.index].to_numpy()[None, :]).sum(1) / near.sum(1)
        return out
    key = "run_id" if how in ("run", "run_pano") else "face_id"
    has = df[key] != ""
    for _, g in df[has].groupby(key):
        if len(g) < 2:
            continue
        mm, v = m.loc[g.index].to_numpy(), s.loc[g.index].to_numpy()
        if how == "run_pano":
            panos = g["pano_id"].astype(str).to_numpy()
            for i, idx in enumerate(g.index):
                keep = mm.copy()
                keep[i] = True
                pm = pd.Series(v[keep]).groupby(panos[keep]).mean()
                out.loc[idx] = pm.mean()
            continue
        tot, cnt = (v * mm).sum(), mm.sum()
        for i, idx in enumerate(g.index):
            t, c = (tot, cnt) if mm[i] else (tot + v[i], cnt + 1)
            if how == "face_loo":
                t, c = t - v[i], c - 1
                out.loc[idx] = t / c if c else v[i]
            else:
                out.loc[idx] = t / c
    return out


def pooled_map(fit_rows, pi_deploy, group):
    """The served map for scores pooled by ``group``, fitted (Disagree held back, as the gate) on ``fit_rows``: rows
    with ``pooled`` (a pooled logit of the same group definition) and ``true``. Returned with its group tag: a pooled
    score must never go through a map fitted on single-label scores, whose confidence it does not share."""
    rows = fit_rows.assign(p_correct=sigmoid(fit_rows["pooled"]), p_incorrect=1 - sigmoid(fit_rows["pooled"]))
    return {"pooled_group": group, "map": ng.fit_map(rows, pi_deploy)}


def serve_pooled(rows, pmap, group):
    """``verdict`` / ``served`` for rows with a ``pooled`` logit through ``pmap``; refuses a map of another group."""
    tag = pmap.get("pooled_group") if isinstance(pmap, dict) else None
    if tag != group:
        raise ValueError(f"a {group!r} pooled score needs a map fitted on {group!r} pooled scores "
                         f"(got {'an untagged map' if tag is None else repr(tag)})")
    p = sigmoid(rows["pooled"])
    rows = rows.assign(p_correct=p, p_incorrect=1 - p)
    if "true" not in rows:  # class unknown (step (b) neighbours): cal.verdicts still wants the column
        rows = rows.assign(true="")
    return ng.served_with_map(rows, pmap["map"])


def lofo_served(fit_rows, eval_rows, pi_deploy, group):
    """Leave-one-fold-out: rows of fold ``k`` in ``eval_rows`` are served by a map fitted on ``fit_rows`` of the
    other folds (both carry ``fold`` and ``pooled``)."""
    parts = []
    for k in sorted(eval_rows["fold"].unique()):
        pmap = pooled_map(fit_rows[fit_rows["fold"] != k], pi_deploy, group)
        parts.append(serve_pooled(eval_rows[eval_rows["fold"] == k], pmap, group))
    return pd.concat(parts).loc[eval_rows.index]


def read_predictions(dirs, n_folds):
    """``{k: frame}`` of ``predictions_fold<k>.csv`` concatenated over the run directories ``dirs`` (the R3 job was
    split by city), with each file's sha256; a label scored twice by one model stops the run."""
    preds, shas = {}, {}
    for k in range(n_folds):
        frames = []
        for d in dirs:
            path = os.path.join(d, f"predictions_fold{k}.csv")
            frames.append(pd.read_csv(path, dtype={"label_uid": str}))
            shas[f"{os.path.basename(os.path.normpath(d))}/predictions_fold{k}.csv"] = rl.sha256_file(path)
        preds[k] = pd.concat(frames, ignore_index=True)
        if not preds[k]["label_uid"].is_unique:
            raise ValueError(f"fold {k}: a label was scored twice")
    return preds, shas


def step_a_table(faces, preds, split, gate_preds):
    """One row per NoSidewalk label on a step (a) face with its re-scored logit: an out-of-fold face's labels by that
    fold's model only, a held-out face's labels by the mean logit of all models. Adds the split truth (``true``,
    ``class_source``), the label-level truth set (``truth``/``truth_source``: the split's margin/expert class, else
    the label's own expert verdict), ``fold`` (-1 held out), ``trained_pano`` (for the scoring model) and gate2's
    own logit of split rows (``logit_gate2``, for the crop-parity check: the out-of-fold model's for an out-of-fold
    row, the five-model mean for a held-out row, as the re-scored ``logit``). Raises when a face's rows sit in two
    folds."""
    t = faces.copy()
    t["face_id"] = t["face_id"].astype(str)
    folds = t.loc[t["heldout"] == 0].groupby("face_id")["fold_of_face"].nunique()
    if (folds > 1).any():
        raise ValueError("a face's rows sit in two folds")
    t["fold"] = np.where(t["heldout"] == 1, -1, pd.to_numeric(t["fold_of_face"], errors="coerce").fillna(-1)).astype(int)
    t["run_id"] = run_ids(t["face_id"], LABEL_TYPE, t["labeler_idx"]).to_numpy()
    n = len(preds)
    lg = np.full((len(t), n), np.nan)
    for k, p in preds.items():
        m = t["label_uid"].map(p.set_index("label_uid")["p_correct"])
        lg[:, k] = np.where(m.notna(), logit(m.fillna(0.5)), np.nan)
    fold = t["fold"].to_numpy()
    own = np.full(len(t), np.nan)
    oof = fold >= 0
    own[oof] = lg[np.flatnonzero(oof), fold[oof]]
    held = ~oof & ~np.isnan(lg).any(axis=1)
    own[held] = lg[held].mean(axis=1)
    t["logit"] = own
    for k in range(n):
        t[f"logit_{k}"] = lg[:, k]
    t["scored"] = np.isfinite(t["logit"]).astype(int)
    t["trained_pano"] = np.where(t["fold"] >= 0, [t.at[i, f"trained_pano_{k}"] if k >= 0 else 0
                                                    for i, k in zip(t.index, t["fold"])],
                                 t[[f"trained_pano_{k}" for k in range(n)]].max(axis=1))
    s = split[["label_uid", "label_class", "class_source", "split"]]
    t = t.merge(s, on="label_uid", how="left", validate="one_to_one")
    t["true"] = t["label_class"].fillna("")
    lvl = t["class_source"].isin(["margin", "expert"])
    t["truth"] = np.where(lvl, t["true"], np.where(t["in_split"] == 0, t["own_expert_verdict"].fillna(""), ""))
    t["truth_source"] = np.where(lvl, "split_label", np.where((t["in_split"] == 0) & (t["truth"] != ""),
                                                              "own_expert", ""))
    g = pd.concat([p for p in gate_preds.values()], ignore_index=True)
    g = g[g["split"] == "test"].set_index("label_uid")["p_correct"]
    t["logit_gate2"] = logit(t["label_uid"].map(g))
    t.loc[t["label_uid"].map(g).isna(), "logit_gate2"] = np.nan
    hl = pd.DataFrame({k: p[p["split"] == "test_heldout"].set_index("label_uid")["p_correct"]
                       for k, p in gate_preds.items()})
    held_mean = pd.Series(logit(hl.to_numpy()).mean(axis=1), index=hl.index)[hl.notna().all(axis=1)]
    is_held = t["fold"] < 0
    t.loc[is_held, "logit_gate2"] = t.loc[is_held, "label_uid"].map(held_mean)
    return t


def auc_row(y_incorrect, score_incorrect, groups, city, n_boot, seed):
    """AUC with a group-bootstrap CI and the within-city AUC (incorrect is the positive class)."""
    from analysis.retrain_report import cluster_boot, within_city_auc
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_incorrect, int)
    s = np.asarray(score_incorrect, float)
    out = {"n": int(len(y)), "n_incorrect": int(y.sum()), "faces": int(pd.Series(groups).nunique())}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, s))
        b = cluster_boot(y, s, groups, n_boot, seed)[:, 0] if n_boot else np.array([])
        out["auc_ci95"] = ci95(b) if len(b) else [None, None]
    else:
        out["auc"], out["auc_ci95"] = None, [None, None]
    out["within_city_auc"], out["within_city_n_cities"] = within_city_auc(y, s, city)
    return out


def paired_delta(y_incorrect, s_a, s_b, groups, n_boot, seed):
    """AUC(a) - AUC(b) with a 95 % CI from resampling whole groups (faces), the same resample for both."""
    from analysis.retrain_report import cluster_boot
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_incorrect, int)
    if not 0 < y.sum() < len(y):
        return {"delta": None, "delta_ci95": [None, None]}
    d = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    b = cluster_boot(y, np.column_stack([s_a, s_b]), groups, n_boot, seed) if n_boot else np.empty((0, 2))
    return {"delta": d, "delta_ci95": ci95(b[:, 0] - b[:, 1]) if len(b) else [None, None]}


def one_per_face(df, seed):
    """One row per face: a seeded choice among the face's rows (``rl.seeded_rank`` of ``label_uid``)."""
    r = df.assign(_r=df["label_uid"].map(lambda u: rl.seeded_rank(seed, u)))
    return r.sort_values(["face_id", "_r"], kind="mergesort").drop_duplicates("face_id").drop(columns="_r")


def eval_sets(oof, seed):
    """Item 2's step (a) evaluation sets on scored out-of-fold rows with a face (``truth``/``true`` as in
    :func:`step_a_table`): ``ii_split`` / ``ii_extra`` / ``ii_all`` (label-level truth: the split's margin and expert
    rows, the other labels' own expert verdicts, both), ``iii`` (one label-level row per face), ``i_circular`` (every
    split row, face-copied truth included) and ``iii_circular`` (one split row per face)."""
    split_rows = oof[oof["in_split"] == 1].assign(y=lambda x: x["true"])
    ii = oof[oof["truth"] != ""].assign(y=lambda x: x["truth"])
    return {"ii_all": ii, "ii_split": ii[ii["truth_source"] == "split_label"],
            "ii_extra": ii[ii["truth_source"] == "own_expert"], "iii": one_per_face(ii, seed),
            "i_circular": split_rows, "iii_circular": one_per_face(split_rows, seed)}


def _plain_auc(y_incorrect, score_incorrect):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_incorrect, int)
    return float(roc_auc_score(y, score_incorrect)) if 0 < y.sum() < len(y) else None


def per_city_aucs(rows, scores, min_rows=10):
    """Per city with at least ``min_rows`` rows of ``rows`` (``y``, ``city``): rows, incorrect and the AUC of each
    pooled logit in ``scores`` (``{group: Series on rows' index}``; incorrect is the positive class, so the score is
    the negated logit). The AUC is None where the city has one class only.

    >>> r = pd.DataFrame({"city": ["a"] * 10 + ["b"] * 3, "y": ["incorrect", "correct"] * 6 + ["correct"]})
    >>> s = pd.Series(np.arange(13.0))
    >>> per_city_aucs(r, {"single": s})
    {'a': {'rows': 10, 'incorrect': 5, 'single': 0.6}}
    """
    out = {}
    for city, sub in rows.groupby("city"):
        if len(sub) < min_rows:
            continue
        y = (sub["y"] == "incorrect").to_numpy()
        out[city] = {"rows": int(len(sub)), "incorrect": int(y.sum())}
        for g, s in scores.items():
            out[city][g] = _plain_auc(y, -s.loc[sub.index].to_numpy())
    return out


def member_masks(t):
    """Neighbour filters of the step (a) sensitivities (the judged row itself always stays in its group)."""
    base = t["scored"] == 1
    return {"all": base, "no_expert_placed": base & (t["expert_placed"] == 0),
            "no_trained_pano": base & (t["trained_pano"] == 0)}


def step_a_pooling(t, pi_deploy, n_boot=1000, seed=0, groups=GROUPS, train=None):
    """Item 2 on step (a) (secondary): for each group, the pooled logit of every scored label, served through a
    leave-one-fold-out map fitted on the split rows' pooled scores of the same group; AUC, within-city AUC, the
    paired AUC delta against the single score and the Agree gate on each evaluation set; held-out faces
    (descriptive) through a map fitted on all split rows. Tag slices on ``ii_all``. Raises when a face's rows sit
    in two folds."""
    oof_all = t[(t["fold"] >= 0)]
    if (oof_all.groupby("face_id")["fold"].nunique() > 1).any():
        raise ValueError("a face's rows sit in two folds")
    out = {"counts": {"oof_labels": int(len(oof_all)), "oof_scored": int(oof_all["scored"].sum()),
                      "oof_faces": int(oof_all["face_id"].nunique()),
                      "split_rows_scored": int(((oof_all["in_split"] == 1) & (oof_all["scored"] == 1)).sum())},
           "groups": {}, "sensitivity": {}, "heldout": {}}
    masks = member_masks(t)
    pooled = {(g, mname): pooled_scores(t[t["scored"] == 1], g, members=m[t["scored"] == 1])
              for g in groups for mname, m in masks.items()}
    scored = t[t["scored"] == 1]
    oof = scored[scored["fold"] >= 0]
    held = scored[scored["fold"] < 0]
    sets = eval_sets(oof, seed)
    out["counts"].update({f"set_{k}": int(len(v)) for k, v in sets.items()})
    out["counts"]["set_ii_all_incorrect"] = int((sets["ii_all"]["y"] == "incorrect").sum())
    for mname in masks:
        res = {}
        for g in groups:
            sc = scored.assign(pooled=pooled[(g, mname)])
            fit = sc[(sc["fold"] >= 0) & (sc["in_split"] == 1)].assign(true=lambda x: x["true"])
            served = lofo_served(fit, sc[sc["fold"] >= 0], pi_deploy, g)
            block = {}
            for name, rows in sets.items():
                r = served.loc[rows.index].assign(true=rows["y"])
                y = (rows["y"] == "incorrect").to_numpy()
                single = -scored.loc[rows.index, "logit"].to_numpy()
                b = auc_row(y, -sc.loc[rows.index, "pooled"].to_numpy(), rows["face_id"].to_numpy(), rows["city"],
                            n_boot if mname == "all" else 0, seed)
                b["agree_gate"] = ng.gate_stats(r, pi_deploy)
                if g != "single":
                    b["delta_vs_single"] = paired_delta(y, -sc.loc[rows.index, "pooled"].to_numpy(), single,
                                                        rows["face_id"].to_numpy(), n_boot if mname == "all" else 0,
                                                        seed)
                block[name] = b
            res[g] = block
            if mname == "all" and len(held):
                hp = held.assign(pooled=pooled[(g, mname)].loc[held.index])
                hp = hp[hp["truth"] != ""]
                if len(hp):
                    pmap = pooled_map(fit, pi_deploy, g)
                    hs = serve_pooled(hp, pmap, g).assign(true=hp["truth"])
                    hb = auc_row((hp["truth"] == "incorrect").to_numpy(), -hp["pooled"].to_numpy(),
                                 hp["face_id"].to_numpy(), hp["city"], n_boot, seed)
                    hb["agree_gate"] = ng.gate_stats(hs, pi_deploy)
                    out["heldout"][g] = hb
        if mname == "all":
            out["groups"] = res
        else:
            out["sensitivity"][mname] = {g: {k: {"auc": v["auc"], "within_city_auc": v["within_city_auc"],
                                                 "n_pass": v["agree_gate"]["n_pass"],
                                                 "precision": v["agree_gate"]["precision"]}
                                             for k, v in res[g].items() if k in ("ii_all", "iii")} for g in groups}
    ii = sets["ii_all"]
    out["per_city"] = per_city_aucs(ii, {g: pooled[(g, "all")] for g in ("single", "run", "face") if g in groups})
    pred = lambda col: pd.DataFrame({"label_uid": ii["label_uid"], "true": ii["y"], "pano_id": ii["face_id"],  # noqa
                                     "city": ii["city"], "tags": ii["tags"], "severity": ii["severity"],
                                     "p_incorrect": sigmoid(-col.loc[ii.index].to_numpy())})
    single = pred(scored["logit"])
    out["slices"] = {"tag": {"single": slices.slice_table(single, "tag", train, n_boot=n_boot, seed=seed)
                             .astype(object).where(lambda x: x.notna(), None).to_dict("records")}}
    for g in ("run", "face"):
        if g in groups:
            out["slices"]["tag"][g] = (slices.paired_delta_table(pred(pooled[(g, "all")]), single, "tag",
                                                                 n_boot=n_boot, seed=seed)
                                       .astype(object).where(lambda x: x.notna(), None).to_dict("records"))
    out["severity_note"] = "NoSidewalk labels carry no severity, so there is no severity slice"
    return out


def crop_parity(t, gate_preds, pi_deploy, gate=ng.GATE, split=None):
    """The R3 re-scored split rows against gate2's own predictions (same models, other crop path), for the
    out-of-fold rows (their own fold's model) and the held-out rows (five-model mean logit): median and max |delta
    logit|, verdict flips at 0.5 and gate-pass flips under the gate2 map. ``split`` adds the count of split rows
    without a face, which step (a) never lists and so never re-scores."""
    pooled = pd.concat([p for p in gate_preds.values()], ignore_index=True)
    api_map = ng.fit_map(pooled[pooled["split"] == "test"], pi_deploy)

    def block(r):
        d = (r["logit"] - r["logit_gate2"]).abs()
        a = ng.served_with_map(r.assign(p_correct=sigmoid(r["logit"]), p_incorrect=1 - sigmoid(r["logit"])), api_map)
        b = ng.served_with_map(r.assign(p_correct=sigmoid(r["logit_gate2"]),
                                        p_incorrect=1 - sigmoid(r["logit_gate2"])), api_map)
        pa = (a["verdict"] == "correct") & (a["served"] >= gate)
        pb = (b["verdict"] == "correct") & (b["served"] >= gate)
        return {"rows": int(len(r)), "median_abs_delta_logit": float(d.median()) if len(d) else None,
                "p90_abs_delta_logit": float(d.quantile(0.9)) if len(d) else None,
                "max_abs_delta_logit": float(d.max()) if len(d) else None,
                "verdict_flips": int(((r["logit"] >= 0) != (r["logit_gate2"] >= 0)).sum()),
                "pass_flips": int((pa != pb).sum()), "pass_gate2": int(pb.sum()), "pass_rescored": int(pa.sum())}

    r = t[(t["in_split"] == 1) & (t["scored"] == 1) & t["logit_gate2"].notna()]
    out = block(r[r["fold"] >= 0])
    out["heldout"] = block(r[r["fold"] < 0])
    if split is not None:
        out["split_rows_without_face"] = int((split["face_id"] == "").sum())
    return out


def gate_run_stability(gate_dir, split, seed=0, n_boot=0):
    """The step (a) ``gate/`` run (not kept) as one line: single against face-mean AUC on the label-level split rows,
    pooled over split rows only (only they were scored)."""
    preds = {k: pd.read_csv(ng.predictions_path(gate_dir, k), dtype={"label_uid": str}) for k in range(5)}
    p = pd.concat(preds.values(), ignore_index=True)
    p = p[p["split"] == "test"].merge(split[["label_uid", "face_id", "class_source", "city", "pano_id"]],
                                      on="label_uid", how="left", validate="one_to_one")
    p = p[p["face_id"] != ""].copy()
    p = p.assign(logit=logit(p["p_correct"]), run_id="", fold=0)
    p["face_pooled"] = pooled_scores(p, "face")
    ii = p[p["class_source"].isin(["margin", "expert"])]
    y = (ii["true"] == "incorrect").to_numpy()
    return {"rows": int(len(ii)), "auc_single": auc_row(y, -ii["logit"], ii["face_id"], ii["city"], 0, seed)["auc"],
            "auc_face_split_rows": auc_row(y, -ii["face_pooled"], ii["face_id"], ii["city"], 0, seed)["auc"],
            "files_sha256": {str(k): rl.sha256_file(ng.predictions_path(gate_dir, k)) for k in range(5)}}


def join_key_scores(key, face_scores):
    """The rated rows of ``face_scores`` joined to the key. The key's ``label_uid`` is ``<schema>:<label_id>``; the
    city form is ``label_uid_city``, which is what ``face_scores.label_uid`` holds. Raises unless every key row finds
    its scores, so a join on the wrong column can never return zero or NaN rows quietly."""
    col = "label_uid_city"
    if col not in key:
        raise ValueError("the key has no label_uid_city column")
    fs = face_scores[face_scores["rated"].astype(int) == 1]
    m = key.merge(fs, left_on=col, right_on="label_uid", how="inner", validate="one_to_one",
                  suffixes=("_key", ""))
    if len(m) != len(key):
        raise ValueError(f"{len(key) - len(m)} key rows have no rated face_scores row: join on label_uid_city")
    return m


def step_b_frame(key, sheet, face_scores):
    """The rated labels with the rater's verdict, the key's weight, stratum and city, and the face's scored labels:
    ``(rated, neighbours)``. ``sheet`` is ``gt_rounds.py votes`` output (``packet_id, verdict``)."""
    k = key.merge(sheet[["packet_id", "verdict"]], on="packet_id", how="left", validate="one_to_one")
    k["verdict"] = k["verdict"].fillna("")
    rated = join_key_scores(k, face_scores)
    rated["weight"] = rated["weight"].astype(float)
    fs = face_scores.copy()
    fs["run_id"] = run_ids(fs["face_id"].fillna(""), LABEL_TYPE, fs["labeler_idx"]).to_numpy()
    return rated, fs


def step_b_pooled(rated, fs, how, n_models=5, members=None):
    """Per rated label and model k, the pooled logit over its group among the face's scored labels (``members`` a
    mask on ``fs``; the rated label itself always counts): columns ``pooled_<k>``."""
    out = pd.DataFrame(index=rated.index)
    scored = fs[fs["p_correct_0"].notna()].copy()
    mm = scored["p_correct_0"].notna() if members is None else members.reindex(scored.index).fillna(False)
    for k in range(n_models):
        scored["logit"] = logit(scored[f"p_correct_{k}"])
        p = pooled_scores(scored, how, members=mm)
        out[f"pooled_{k}"] = rated["label_uid"].map(pd.Series(p.to_numpy(), index=scored["label_uid"])).to_numpy()
    return out


def vote_pooled(rated, fs, how, members=None, strict=False):
    """Vote pooling: Agree when at least half of the group's scored labels pass the round's own rule
    (``n_pass_models >= 3``); ``strict`` also needs the label itself to pass."""
    scored = fs[fs["p_correct_0"].notna()].copy()
    scored["logit"] = (scored["stratum"] == "pass").astype(float)  # pooled share of passing labels
    mm = None if members is None else members.reindex(scored.index).fillna(False)
    share = pooled_scores(scored, how, members=mm)
    s = rated["label_uid"].map(pd.Series(share.to_numpy(), index=scored["label_uid"])).to_numpy()
    passed = s >= 0.5
    if strict:
        passed &= (rated["stratum"] == "pass").to_numpy()
    return passed


def kish_bound(d):
    """Weighted precision of ``d`` (decided passers, ``verdict``/``weight``) with the Wilson bound at the Kish n."""
    if not len(d):
        return {"weighted": None, "n_eff": 0.0, "wilson95_lower_n_eff": None}
    w = d["weight"].astype(float)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    p = nr.weighted_share(d)
    return {"weighted": p, "n_eff": n_eff, "wilson95_lower_n_eff": nr.ale.wilson(p * n_eff, n_eff)[0]}


def gate_block(decided, passed, single, n_boot, seed):
    """Every number of one pooled gate on step (b): transition counts and weights, raw precision with Wilson,
    weighted coverage and precision (bootstrap CI within city x stratum, Wilson at the Kish n), what it adds
    outside the pass stratum and what it lets in."""
    d = decided.assign(pooled_pass=passed, single_pass=single)
    tr = (d.groupby(["stratum", "single_pass", "pooled_pass", "verdict"])["weight"].agg(["size", "sum"])
          .reset_index().rename(columns={"size": "n", "sum": "weight"}))
    p = d[d["pooled_pass"]]
    s = d[d["single_pass"]]
    out = {"transition": tr.to_dict("records"),
           "raw": nr.rate(int((p["verdict"] == "correct").sum()), len(p)),
           "raw_single": nr.rate(int((s["verdict"] == "correct").sum()), len(s)),
           "added_outside_pass": int((p["stratum"] != "pass").sum()),
           "added_outside_pass_incorrect": int(((p["stratum"] != "pass") & (p["verdict"] == "incorrect")).sum()),
           "newly_admitted": int((d["pooled_pass"] & ~d["single_pass"]).sum()),
           "newly_admitted_incorrect": int((d["pooled_pass"] & ~d["single_pass"] & (d["verdict"] == "incorrect")).sum()),
           "dropped": int((~d["pooled_pass"] & d["single_pass"]).sum()),
           "dropped_incorrect": int((~d["pooled_pass"] & d["single_pass"] & (d["verdict"] == "incorrect")).sum()),
           "coverage_weighted": float(p["weight"].sum() / d["weight"].sum()) if len(d) else None,
           "precision_weighted": kish_bound(p)}
    out["precision_weighted"]["ci95"] = nr.boot_ci(p, nr.weighted_share, n_boot, seed) if len(p) else [None, None]
    return out


def recommendation(block, min_added=10, max_new_incorrect=1, max_precision_drop=0.01, refit_single=None):
    """The plan's rule on the primary step (b) gate (group R, score pooling): at least ``min_added`` passing decided
    labels outside the pass stratum, at most ``max_new_incorrect`` newly admitted incorrect labels, and raw
    precision among all passers within ``max_precision_drop`` of the single-label rule's.

    The rule compares the run pool, served through a map refitted on re-scored step (a) rows, with the round's own
    rule, so it moves with the change of map as well as with pooling. ``refit_single`` (the ``single`` group's
    score block under its own refitted map) gives the same three counts without pooling; ``run_minus_single`` is
    what pooling adds on top of the change of map. The rule itself is unchanged.

    >>> b = {"added_outside_pass": 12, "newly_admitted_incorrect": 1, "raw": {"rate": 0.98}, "raw_single": {"rate": 0.986}}
    >>> recommendation(b)["study_further"]
    True
    >>> recommendation({**b, "added_outside_pass": 9})["study_further"]
    False
    >>> r = recommendation(b, refit_single={**b, "raw": {"rate": 0.985}})
    >>> r["study_further"], r["refit_single_meets_rule"], r["run_minus_single"]["added_outside_pass"]
    (True, True, 0)
    """
    def three(blk):
        added_ok = blk["added_outside_pass"] >= min_added
        incorrect_ok = blk["newly_admitted_incorrect"] <= max_new_incorrect
        rp, rs = blk["raw"]["rate"], blk["raw_single"]["rate"]
        precision_ok = rp is not None and rs is not None and rp >= rs - max_precision_drop - 1e-12
        return added_ok, incorrect_ok, precision_ok

    added_ok, incorrect_ok, precision_ok = three(block)
    out = {"added_ok": bool(added_ok), "incorrect_ok": bool(incorrect_ok), "precision_ok": bool(precision_ok),
           "study_further": bool(added_ok and incorrect_ok and precision_ok),
           "rule": f">= {min_added} added outside the pass stratum, <= {max_new_incorrect} newly admitted incorrect, "
                   f"raw precision within {max_precision_drop} of the single rule"}
    if refit_single is not None:
        rp, sp = block["raw"]["rate"], refit_single["raw"]["rate"]
        out["refit_single"] = {"added_outside_pass": int(refit_single["added_outside_pass"]),
                               "newly_admitted_incorrect": int(refit_single["newly_admitted_incorrect"]),
                               "raw_rate": sp}
        out["refit_single_meets_rule"] = bool(all(three(refit_single)))
        out["run_minus_single"] = {
            "added_outside_pass": int(block["added_outside_pass"] - refit_single["added_outside_pass"]),
            "newly_admitted_incorrect": int(block["newly_admitted_incorrect"] - refit_single["newly_admitted_incorrect"]),
            "raw_rate": None if rp is None or sp is None else float(rp - sp)}
    return out


def step_b_pooling(rated, fs, maps, pi_deploy, gate=ng.GATE, pass_models=3, n_boot=1000, seed=0, groups=GROUPS,
                   members=None, n_models=5):
    """Item 2 on step (b) (primary): per group, score pooling through ``maps[group]`` (a :func:`pooled_map` of the
    same group, fitted on step (a)) and vote pooling (plain and strict), against the round's own single-label rule.
    Nothing is recalibrated on step (b). ``members`` filters the neighbours (sensitivities)."""
    decided = rated[rated["verdict"].isin(["correct", "incorrect"])].copy()
    single = (decided["stratum"] == "pass").to_numpy()
    out = {"decided": int(len(decided)), "incorrect": int((decided["verdict"] == "incorrect").sum()),
           "single_rule": nr.rate(int(((decided["stratum"] == "pass") & (decided["verdict"] == "correct")).sum()),
                                  int(single.sum())), "groups": {}}
    out["single_block"] = gate_block(decided, single, single, n_boot, seed)
    out["single_auc_weighted"] = nr.weighted_auc(decided, "median_p_correct")
    out["single_within_city_auc_weighted"] = nr.within_city_weighted_auc(decided, "median_p_correct")
    for g in groups:
        pooled = step_b_pooled(decided, fs, g, n_models, members)
        passes = np.zeros(len(decided), int)
        for k in range(n_models):
            r = decided.assign(pooled=pooled[f"pooled_{k}"].to_numpy())
            s = serve_pooled(r, maps[g], g)
            passes += ((s["verdict"] == "correct") & (s["served"] >= gate)).to_numpy().astype(int)
        score_pass = passes >= pass_models
        med = np.median(sigmoid(pooled[[f"pooled_{k}" for k in range(n_models)]].to_numpy()), axis=1)
        dd = decided.assign(pooled_median_p=med)
        block = {"score": gate_block(decided, score_pass, single, n_boot, seed)}
        block["score"]["auc_weighted"] = nr.weighted_auc(dd, "pooled_median_p")
        block["score"]["within_city_auc_weighted"] = nr.within_city_weighted_auc(dd, "pooled_median_p")
        if g != "single":
            block["vote"] = gate_block(decided, vote_pooled(decided, fs, g, members), single, n_boot, seed)
            block["vote_strict"] = gate_block(decided, vote_pooled(decided, fs, g, members, strict=True), single,
                                              n_boot, seed)
        block["tags"] = tag_rates(decided, score_pass, single)
        if g != "single":
            block["tags_vote"] = tag_rates(decided, vote_pooled(decided, fs, g, members), single)
            block["tags_vote_strict"] = tag_rates(decided, vote_pooled(decided, fs, g, members, strict=True), single)
        out["groups"][g] = block
    return out


def tag_rates(decided, passed, single, min_n=slices.MIN_N):
    """Per tag with at least ``min_n`` decided labels: raw precision of the pooled gate ``passed`` and of the single
    rule ``single`` among that tag's labels (NoSidewalk has no severity)."""
    tags = {}
    for t, sub in slices.with_slice(decided.assign(_pass=passed, _single=single), "tag").groupby("slice"):
        if len(sub) >= min_n:
            tags[t] = {"decided": int(len(sub)), "incorrect": int((sub["verdict"] == "incorrect").sum()),
                       "pooled": nr.rate(int((sub["_pass"] & (sub["verdict"] == "correct")).sum()),
                                         int(sub["_pass"].sum())),
                       "single": nr.rate(int((sub["_single"] & (sub["verdict"] == "correct")).sum()),
                                         int(sub["_single"].sum()))}
    return tags


def singleton_runs(rated, fs):
    """Decided rated labels with no other scored label in their run (pooling cannot change them)."""
    scored = fs[fs["p_correct_0"].notna()]
    n = scored.groupby("run_id").size()
    d = rated[rated["verdict"].isin(["correct", "incorrect"])]
    run = pd.Series(run_ids(d["face_id"].fillna(""), LABEL_TYPE, d["labeler_idx"]).to_numpy(), index=d.index)
    return int((run.map(n).fillna(1) <= 1).sum())


# ---------------------------------------------------------------------------------------------------------------------
# item 3: neighbour features without the image, second stage
# ---------------------------------------------------------------------------------------------------------------------

#: Columns a feature computation must never see: votes, verdicts and classes would leak the truth.
LEAKY = ("validations", "correct", "agree_count", "disagree_count", "unsure_count", "label_class", "class_source",
         "own_expert_verdict", "truth", "true", "verdict", "comments")
FEATURES = ("log_n_ns_face", "log_run_len", "log_n_labelers_face", "dist_nearest_ns_m", "no_other_ns",
            "log_n_contra_face", "log_n_contra_same_labeler")
DIST_CAP_M = 200.0


def neighbour_features(targets, ns, contra, exclude_experts=False):
    """Image-free features of each target label from label tables only.

    ``targets``: the labels to describe (``label_uid``, ``face_id``, ``user_key``, ``latitude``, ``longitude``).
    ``ns``: every NoSidewalk label of the defining pull on those faces (``label_uid``, ``face_id``, ``user_key``,
    ``latitude``, ``longitude``, ``expert_placed``). ``contra``: CurbRamp / Obstacle / SurfaceProblem labels of the
    fresh pull (``face_id``, ``user_key``, ``expert_placed``). ``user_key`` is any per-city labeler key (a user id in
    memory, or ``labeler_idx``); it is used to match and never returned. With ``exclude_experts`` the labels the
    experts (or the rater) placed are left out of every count: an expert labelling a face is a proxy for the truth.
    Votes, verdicts and classes are dropped before anything is computed.

    >>> ns = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "face_id": "a:1:left", "user_key": ["u", "u", "e"],
    ...                    "latitude": [0.0, 0.0, 0.001], "longitude": 0.0, "expert_placed": [0, 0, 1]})
    >>> contra = pd.DataFrame({"face_id": ["a:1:left"], "user_key": ["u"], "expert_placed": [0]})
    >>> f = neighbour_features(ns.iloc[:1], ns, contra)
    >>> f[["n_ns_face", "run_len", "n_labelers_face", "dist_nearest_ns_m", "n_contra_face"]].values.tolist()
    [[3.0, 2.0, 2.0, 0.0, 1.0]]
    >>> neighbour_features(ns.iloc[:1], ns, contra, exclude_experts=True)["n_ns_face"].tolist()
    [2.0]
    """
    targets = targets.drop(columns=[c for c in LEAKY if c in targets])
    ns = ns.drop(columns=[c for c in LEAKY if c in ns])
    contra = contra.drop(columns=[c for c in LEAKY if c in contra])
    if exclude_experts:
        ns = ns[ns["expert_placed"].astype(int) == 0]
        contra = contra[contra["expert_placed"].astype(int) == 0]
    rows = []
    by_face = {f: g for f, g in ns.groupby("face_id")}
    contra_face = contra.groupby("face_id").size()
    contra_lab = contra.groupby(["face_id", "user_key"]).size()
    for uid, face, user, lat, lon in zip(targets["label_uid"], targets["face_id"], targets["user_key"],
                                         targets["latitude"], targets["longitude"]):
        g = by_face.get(face)
        others = g[g["label_uid"] != uid] if g is not None else ns.iloc[:0]
        n_face = len(others) + 1
        run_len = int((others["user_key"] == user).sum()) + 1
        labelers = len(set(others["user_key"]) | {user})
        if len(others):
            dist = float(np.min(haversine_m(lat, lon, others["latitude"].to_numpy(float),
                                            others["longitude"].to_numpy(float))))
        else:
            dist = float("nan")
        rows.append({"label_uid": uid, "n_ns_face": float(n_face), "run_len": float(run_len),
                     "n_labelers_face": float(labelers),
                     "dist_nearest_ns_m": min(dist, DIST_CAP_M) if np.isfinite(dist) else DIST_CAP_M,
                     "no_other_ns": float(not len(others)),
                     "n_contra_face": float(contra_face.get(face, 0)),
                     "n_contra_same_labeler": float(contra_lab.get((face, user), 0))})
    f = pd.DataFrame(rows)
    for c in ("n_ns_face", "run_len", "n_labelers_face", "n_contra_face", "n_contra_same_labeler"):
        f[f"log_{c}"] = np.log1p(f[c])
    return f


def face_check(defining, fresh):
    """Labels whose NoSidewalk row sits on the same face in the fresh pull as in the pull that defined them; the
    rest (moved or gone) get no contradicting-label features and are counted. Returns ``(kept uids, counts)``.

    >>> a = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3"], "face_id": ["c:1:left", "c:2:left", "c:3:left"]})
    >>> b = pd.DataFrame({"label_uid": ["c:1", "c:2"], "face_id": ["c:1:left", "c:9:left"]})
    >>> face_check(a, b)
    ({'c:1'}, {'same_face': 1, 'moved': 1, 'missing': 1})
    """
    m = defining[["label_uid", "face_id"]].merge(fresh[["label_uid", "face_id"]], on="label_uid", how="left",
                                                 validate="one_to_one", suffixes=("", "_fresh"))
    same = m["face_id"] == m["face_id_fresh"]
    return set(m.loc[same, "label_uid"]), {"same_face": int(same.sum()),
                                            "moved": int((m["face_id_fresh"].notna() & ~same).sum()),
                                            "missing": int(m["face_id_fresh"].isna().sum())}


def fit_second_stage(X, y, w):
    """Standardised logistic regression (``C=1.0``, no tuning) with sample weights; returns the fitted pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(X, y, logisticregression__sample_weight=w)
    return model


MODEL_COLUMNS = {"a_image": ("image_logit",), "b_image_features": ("image_logit", *FEATURES), "c_features": FEATURES}


def face_weights(df):
    """Weights that make every face sum to 1."""
    return 1.0 / df.groupby("face_id")["face_id"].transform("size").to_numpy(float)


def second_stage(train, test_sets, n_boot, seed):
    """Leave-one-fold-out over the step (a) folds: models a (image logit), b (image + features), c (features) fitted
    on ``train`` (label-level truth, faces weighted equally) and scored on each held-back fold; per evaluation set
    in ``test_sets`` (uids) the AUC, within-city AUC and the paired face-bootstrap delta b - a."""
    preds = {m: pd.Series(np.nan, index=train.index) for m in MODEL_COLUMNS}
    for k in sorted(train["fold"].unique()):
        tr, te = train[train["fold"] != k], train[train["fold"] == k]
        if set(tr["face_id"]) & set(te["face_id"]):
            raise ValueError("a face on both sides of a fold")
        y = (tr["truth"] == "incorrect").astype(int).to_numpy()
        for m, cols in MODEL_COLUMNS.items():
            model = fit_second_stage(tr[list(cols)].to_numpy(float), y, face_weights(tr))
            preds[m].loc[te.index] = model.predict_proba(te[list(cols)].to_numpy(float))[:, 1]
    out = {}
    for name, uids in test_sets.items():
        d = train[train["label_uid"].isin(uids)]
        y = (d["truth"] == "incorrect").to_numpy()
        block = {m: auc_row(y, preds[m].loc[d.index], d["face_id"].to_numpy(), d["city"], n_boot, seed)
                 for m in MODEL_COLUMNS}
        block["delta_b_minus_a"] = paired_delta(y, preds["b_image_features"].loc[d.index],
                                                preds["a_image"].loc[d.index], d["face_id"].to_numpy(), n_boot, seed)
        block["delta_c_minus_a"] = paired_delta(y, preds["c_features"].loc[d.index], preds["a_image"].loc[d.index],
                                                d["face_id"].to_numpy(), n_boot, seed)
        out[name] = block
    if "ii_all" in test_sets and "tags" in train:
        d = train[train["label_uid"].isin(test_sets["ii_all"])]
        d = d.assign(**{f"_p_{m}": preds[m].loc[d.index].to_numpy() for m in MODEL_COLUMNS})
        out["tags_ii_all"] = {}
        for t, sub in slices.with_slice(d, "tag").groupby("slice"):
            if len(sub) >= slices.MIN_N:
                y = (sub["truth"] == "incorrect").to_numpy()
                out["tags_ii_all"][t] = {"rows": int(len(sub)), "incorrect": int(y.sum()),
                                         **{m: _plain_auc(y, sub[f"_p_{m}"].to_numpy()) for m in MODEL_COLUMNS}}
    return out


def transfer(train, rated, n_models, n_boot, seed):
    """Step (b) transfer (primary for item 3): models fitted once on all of ``train``, frozen, applied to each model
    k's image logit of the decided rated labels and averaged over k. Weighted AUC and within-city weighted AUC
    against model a, with a paired bootstrap of the deltas within (city, stratum)."""
    y = (train["truth"] == "incorrect").astype(int).to_numpy()
    fitted = {m: fit_second_stage(train[list(c)].to_numpy(float), y, face_weights(train)) for m, c in
              MODEL_COLUMNS.items()}
    d = rated.copy()
    for m, cols in MODEL_COLUMNS.items():
        ps = []
        for k in range(n_models):
            X = d.assign(image_logit=logit(d[f"p_correct_{k}"]))[list(cols)].to_numpy(float)
            ps.append(fitted[m].predict_proba(X)[:, 1])
        d[f"pc_{m}"] = 1 - np.mean(ps, axis=0)  # p(correct), the form nr.weighted_auc reads
    out = {"decided": int(len(d)), "models": {}}
    for m in MODEL_COLUMNS:
        out["models"][m] = {"auc_weighted": nr.weighted_auc(d, f"pc_{m}"),
                            "within_city_auc_weighted": nr.within_city_weighted_auc(d, f"pc_{m}")[0]}
    for other in ("b_image_features", "c_features"):
        for stat, fn in (("auc_weighted", lambda x, c: nr.weighted_auc(x, c)),
                         ("within_city_auc_weighted", lambda x, c: nr.within_city_weighted_auc(x, c)[0])):
            delta = lambda x: fn(x, f"pc_{other}") - fn(x, "pc_a_image")  # noqa: E731
            out[f"delta_{other}_{stat}"] = {"delta": delta(d), "ci95": nr.boot_ci(d, delta, n_boot, seed)}
    if "tags" in d:
        out["tags"] = {}
        for t, sub in slices.with_slice(d, "tag").groupby("slice"):
            if len(sub) >= slices.MIN_N:
                out["tags"][t] = {"decided": int(len(sub)), "incorrect": int((sub["verdict"] == "incorrect").sum()),
                                  **{m: nr.weighted_auc(sub, f"pc_{m}") for m in MODEL_COLUMNS}}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------------------------------------------------

#: sha256 of the inputs of the published run (2026-11-17); ``--allow-unpinned`` skips the check (tests only). Keys
#: name the input, never the file's basename, so a copy saved under another name is still checked. Pull directories
#: are pinned by the sha256 of their sorted ``SHA256SUMS``-form listing (:func:`pull_sums`); the 2026-10-18 pull by
#: the form ``deployment_gate.py`` records (:func:`provenance_sums`), so it can be compared with that page's pin.
PINS = {
    "ns_pull_2026_11_05_sums": "0907a87e52e04dec27d11828b59109c2e4bfa2a924cba2f2774d9bb5b1fec178",
    "round_b_pull_2026_11_11_sums": "71ce176114ed555531948661a6ca5dc2179413f0341bad0688cfaa8ace025c4d",
    "contra_pull_2026_11_17_sums": "6dd1f0c62cf6c08704351c913ecd1798b7452c1686d2524fdb23618ba730c527",
    # = docs/deployment-gate-2026-11/provenance.json slices.rawlabels_files_sha256 (the same 54 files)
    "all_pull_2026_10_18_files": "99e286e784c01f5a940716d8ee0e4aef91c0b057989bcedc280e19e298525e7b",
    "key.csv": "d2d30d05912e1a12cd75f56822a32413cb250dd048a5c17ff5bc13ae949172a8",
    "votes_jon.csv": "09a2624f93ff5ca203dfd8f0a4038129252d12334d863046986931e99ce957c9",
    "face_scores.csv": "6da73f9e4e3c93de1102e8657b18e1e457c15b3d14c6ebf8de4b796775c54bd3",
    "deployment_priors.json": "e33be7279bcf8d10a8a4b346c3633f7999b65b923073e6a6d783c8b99392ee2f",
    "gate2_folds.csv": "895d00b4f5fb5567cf5b2b0253db5dba986ac42085c4827fe6123e3101df25ac",
    "gate2_folds.json": "e74d4455f199bfe005217b41e72cee69422ad77e4e99ba384d8c0d87bb773afb",
    # gate2's out-of-fold predictions (as pinned in docs/nosidewalk-2026-11/neighbours.json) and splits
    "gate2_predictions_fold0": "b44a555fb2e063d330bd11cd54d2b15a49bafd20cf6a2802b972f793a2af8e1c",
    "gate2_predictions_fold1": "ed0c74735a7347e84a352128a9d6b799631fadcadf01b1cd265168536e3127a4",
    "gate2_predictions_fold2": "c6306f54914bca59daae87892665c630c8dcd64a46457ca49f9842219c7d4ede",
    "gate2_predictions_fold3": "a2a9270f17b3ad26627794ee81e12cc111b84d15bf5e497bc454656cbc1a70e8",
    "gate2_predictions_fold4": "f36ed206947435bfbcc4acdad8fd15675f2efdc35db4161436724bf13d217b4f",
    "gate2_split_fold0": "d2615554686963b76fd605a2a36d774de812b3c7f4bf7e513ab23d80b50a30a6",
    "gate2_split_fold1": "7e78497d221cc9e88c7537bfc63bb2ed5d235801a1aabe7fadfac00848251fc6",
    "gate2_split_fold2": "57a68ce9f2789812848bdd5507348a99532c24224aaac09439caf238b3948b83",
    "gate2_split_fold3": "8286f3b4f999b40b6a060941e569e74aa38c52922743867b8c1b7801ca398800",
    "gate2_split_fold4": "bf834900701141ef5ff6f322858e8d86bb0cec7a46197a8e0d595c2102399ec7",
    "step_a_faces.csv": "e571f6acd0dd7cdf0430e9be7677d8dedee4b77d124c00752db8e0def8d8cfff",
    "expert_vote_times.csv": "6662148df20f55ae728883c9b9324a15946ca002c8772037f12878146cf413ad",
    "cities_json": "8d09c714b45602ad9358da578b5b57c2f8d9aa84be070e85c91d6732aabf064c",
    "cityparams_snapshot.conf": "eed92cfd7658762f981c082f135fa1474041d688c7b3ef63f21278263026a899",
    # TODO(#102 results run): the R3 step (a) predictions, keyed "r3_<a|b>_predictions_fold<k>", once copied back
    # from klone; until then read_predictions records their sha256 without a check.
}


def check_pin(name, sha, allow_unpinned):
    """Stop when an input pinned in ``PINS`` has another sha256 (unless ``allow_unpinned``); returns ``sha``."""
    want = PINS.get(name)
    if want is not None and sha != want and not allow_unpinned:
        sys.exit(f"{name}: sha256 {sha[:12]}... is not the pinned {want[:12]}... (pass --allow-unpinned only in tests)")
    return sha


def pull_sums(directory, name, allow_unpinned):
    """The sha256 of a pull's sorted listing (as its ``SHA256SUMS``), checked against ``PINS[name]``."""
    paths = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv"))
    shas, total = sha_sums(paths)
    return check_pin(name, total, allow_unpinned), shas


def provenance_sums(directory, name, allow_unpinned):
    """The sha256 ``deployment_gate.py`` records for a rawLabels directory (``rawlabels_files_sha256``: the
    ``<sha256>  <file>`` lines of every ``.csv``, sorted, joined by newlines without a trailing one), checked
    against ``PINS[name]``. Returns ``(sha256, number of files)``."""
    import hashlib
    files = sorted(n for n in os.listdir(directory) if n.endswith(".csv"))
    text = "\n".join(f"{rl.sha256_file(os.path.join(directory, n))}  {n}" for n in files)
    return check_pin(name, hashlib.sha256(text.encode()).hexdigest(), allow_unpinned), len(files)


def check_gate_pins(gate, n_folds, allow_unpinned, predictions=False):
    """Check gate2's ``folds.csv``, ``folds.json``, every ``fold<k>/split.csv`` and (``predictions``) every fold's
    predictions against ``PINS``; returns ``{name: sha256}``."""
    out = {"gate2_folds.csv": check_pin("gate2_folds.csv", rl.sha256_file(os.path.join(gate, "folds.csv")),
                                        allow_unpinned),
           "gate2_folds.json": check_pin("gate2_folds.json", rl.sha256_file(os.path.join(gate, "folds.json")),
                                         allow_unpinned)}
    for k in range(n_folds):
        out[f"gate2_split_fold{k}"] = check_pin(f"gate2_split_fold{k}", rl.sha256_file(
            os.path.join(gate, f"fold{k}", "split.csv")), allow_unpinned)
        if predictions:
            out[f"gate2_predictions_fold{k}"] = check_pin(f"gate2_predictions_fold{k}", rl.sha256_file(
                ng.predictions_path(gate, k)), allow_unpinned)
    return out


def check_cities(path, allow_unpinned):
    return check_pin("cities_json", rl.sha256_file(path), allow_unpinned)


def schema_map(cityparams):
    sc, _ = rl.cityparams(open(cityparams).read())
    return sc


def write_json(path, obj, force=None):
    """Write ``obj`` as JSON. With ``force`` not None the file is written through ``nr.guarded_outputs``: an existing
    file with other bytes stops the run unless ``force`` is True (the doc pins these outputs by sha256)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    if force is None:
        with open(path, "w") as f:
            json.dump(without_nan(obj), f, indent=1, sort_keys=False, default=str)
            f.write("\n")
        return
    name = os.path.basename(path)
    with nr.guarded_outputs(directory, [name], force) as tmp:
        with open(tmp[name], "w") as f:
            json.dump(without_nan(obj), f, indent=1, sort_keys=False, default=str)
            f.write("\n")


def cmd_agreement(args):
    experts = expert_order(os.environ.get("EXPERTS", ""))
    public = public_cities(args.cities_json)
    inputs = {"cities_json": check_cities(args.cities_json, args.allow_unpinned),
              "cityparams": check_pin("cityparams_snapshot.conf", rl.sha256_file(args.cityparams), args.allow_unpinned)}
    sc = schema_map(args.cityparams)
    times_ns = times_all = None
    if args.vote_times:
        inputs["vote_times"] = check_pin("expert_vote_times.csv", rl.sha256_file(args.vote_times), args.allow_unpinned)
        times_ns = vote_times(args.vote_times, sc, args.ns_as_of)
        times_all = vote_times(args.vote_times, sc, args.all_as_of)
    inputs["ns_pull_sums"], _ = pull_sums(args.ns_pull, "ns_pull_2026_11_05_sums", args.allow_unpinned)
    cols = ["label_id", "user_id", "label_type", "street_edge_id", "street_side", "validations", "time_created",
            "tags", "severity"]
    ns, _ = load_pull(args.ns_pull, [LABEL_TYPE], public, cols)
    out = {"inputs": inputs, "experts": len(experts), "public_cities": len(public),
           "settings": {"n_perm": args.n_perm, "n_boot": args.n_boot, "seed": args.seed, "min_runs": MIN_RUNS,
                        "gap_edges_hours": list(GAP_EDGES), "ns_as_of": args.ns_as_of, "all_as_of": args.all_as_of}}
    d, counts = item1_labels(ns, experts)
    out["nosidewalk"] = agreement_report(d, counts, times_ns, args.n_perm, args.n_boot, args.seed)
    sens = {}
    ds, cs = item1_labels(ns, experts, "dg_strict")
    sens["dg_strict"] = {"counts": cs, "macro": macro_block(ds, args.n_perm, args.n_boot, args.seed)}
    sens["without_oradell"] = {"macro": macro_block(d[d["city"] != "oradell-nj"], args.n_perm, args.n_boot,
                                                    args.seed)}
    out["census"] = {"NoSidewalk_2026_11_05": run_census(add_runs(ns))}
    if args.all_pull:
        paths = sorted(os.path.join(args.all_pull, f"{c}.csv") for c in public
                       if os.path.exists(os.path.join(args.all_pull, f"{c}.csv")))
        shas, total = sha_sums(paths)
        inputs["all_pull_public_files_sums"] = total
        inputs["all_pull_public_files"] = len(paths)
        inputs["all_pull_files_sha256"], inputs["all_pull_files"] = provenance_sums(
            args.all_pull, "all_pull_2026_10_18_files", args.allow_unpinned)
        inputs["all_pull_same_as"] = ("docs/deployment-gate-2026-11/provenance.json slices.rawlabels_files_sha256"
                                      if inputs["all_pull_files_sha256"] == PINS["all_pull_2026_10_18_files"] else "")
        allp, _ = load_pull(args.all_pull, list(ITEM1_TYPES), public, cols)
        out["types"] = {}
        for t in ITEM1_TYPES:
            raw = allp[allp["label_type"] == t]
            dt, ct = item1_labels(raw, experts)
            rep = agreement_report(dt, ct, times_all, args.n_perm, args.n_boot, args.seed)
            out["types"][t] = {k: rep[k] for k in ("counts", "macro", "strata", "slices", "pairs_total")}
            out["types"][t]["per_city_named"] = {c: {k: v for k, v in b.items() if not isinstance(v, dict)}
                                                 for c, b in rep["per_city"].items()}
            out["census"][f"{t}_2026_10_18"] = run_census(add_runs(raw))
            if t == LABEL_TYPE:
                sens["october_pull"] = {"counts": ct, "macro": rep["macro"]}
    out["sensitivity"] = sens
    write_json(args.out_json, out, args.force)
    print(f"wrote {args.out_json}")
    return 0


def load_step_a(args):
    faces_path = os.path.join(args.step_a, "step_a_faces.csv")
    check_pin("step_a_faces.csv", rl.sha256_file(faces_path), args.allow_unpinned)
    faces = pd.read_csv(faces_path, dtype={"label_uid": str, "pano_id": str, "label_id": str}, keep_default_na=False)
    for c in ("latitude", "longitude"):
        faces[c] = pd.to_numeric(faces[c])
    faces["fold_of_face"] = pd.to_numeric(faces["fold_of_face"].replace("", np.nan))
    split, trained, n_folds = gate_split(args.gate)
    gate_pins = check_gate_pins(args.gate, n_folds, args.allow_unpinned, predictions=True)
    preds, shas = read_predictions(args.scores, n_folds)
    for name, sha in shas.items():  # "<a|b>/predictions_fold<k>.csv" -> "r3_<a|b>_predictions_fold<k>"
        check_pin("r3_" + name.replace("/", "_").removesuffix(".csv"), sha, args.allow_unpinned)
    gate_preds = {k: pd.read_csv(ng.predictions_path(args.gate, k), dtype={"label_uid": str}) for k in range(n_folds)}
    t = step_a_table(faces, preds, split, gate_preds)
    inputs = {"step_a_faces": rl.sha256_file(faces_path), "r3_predictions": shas, "gate2": gate_pins}
    return t, split, gate_preds, inputs, n_folds


def load_step_b(args):
    key = pd.read_csv(args.key, dtype=str, keep_default_na=False)
    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    fs = pd.read_csv(args.face_scores, dtype={"label_uid": str, "label_id": str, "pano_id": str, "face_id": str})
    for name, path in (("key.csv", args.key), ("votes_jon.csv", args.sheet), ("face_scores.csv", args.face_scores)):
        check_pin(name, rl.sha256_file(path), args.allow_unpinned)
    rated, fs = step_b_frame(key, sheet, fs)
    if (rated["stratum"] != rated["stratum_key"]).any():
        raise ValueError("face_scores.csv and the key disagree on a rated label's stratum")
    inputs = {"key": rl.sha256_file(args.key), "sheet": rl.sha256_file(args.sheet),
              "face_scores": rl.sha256_file(args.face_scores)}
    return rated, fs, inputs


def step_b_members(fs, rated):
    """Neighbour filters of the step (b) sensitivities, masks on ``fs``."""
    scored = fs["p_correct_0"].notna()
    touched = fs.groupby("face_id")[["rater_placed", "rater_voted"]].transform("max").max(axis=1) == 1
    return {"no_rater_placed_or_voted": scored & (fs["rater_placed"] == 0) & (fs["rater_voted"] == 0),
            "no_trained_pano_or_pilot_heldout": scored & (fs["trained_pano"] == 0) & (fs["pilot_heldout"] == 0),
            "_touched_faces": set(fs.loc[touched, "face_id"])}


def cmd_pooling(args):
    t, split, gate_preds, inputs, n_folds = load_step_a(args)
    pi_deploy, _ = cal.load_prior(args.priors, LABEL_TYPE)
    inputs["priors"] = check_pin("deployment_priors.json", rl.sha256_file(args.priors), args.allow_unpinned)
    out = {"inputs": inputs, "gate": ng.GATE, "prior": pi_deploy,
           "settings": {"n_boot": args.n_boot, "seed": args.seed, "near_m": NEAR_M, "groups": list(GROUPS)}}
    out["crop_parity"] = crop_parity(t, gate_preds, pi_deploy, split=split)
    out["r3_coverage"] = {"labels": int(len(t)), "scored": int(t["scored"].sum()),
                          "share": float(t["scored"].mean())}
    train = split[split["split"].isin(["train", "val", "test"])][["label_uid", "tags", "severity"]]
    out["step_a"] = step_a_pooling(t, pi_deploy, args.n_boot, args.seed, train=train)
    if args.gate1:
        out["step_a_gate1"] = gate_run_stability(args.gate1, split, args.seed)
    rated, fs, b_inputs = load_step_b(args)
    out["inputs"].update(b_inputs)
    scored = t[(t["scored"] == 1)]
    fit = scored[(scored["fold"] >= 0) & (scored["in_split"] == 1)]
    maps = {}
    for g in GROUPS:
        p = pooled_scores(scored, g).loc[fit.index]
        maps[g] = pooled_map(fit.assign(pooled=p), pi_deploy, g)
    out["step_b"] = step_b_pooling(rated, fs, maps, pi_deploy, n_boot=args.n_boot, seed=args.seed)
    out["step_b"]["singleton_runs"] = singleton_runs(rated, fs)
    out["recommendation"] = recommendation(out["step_b"]["groups"]["run"]["score"],
                                           refit_single=out["step_b"]["groups"]["single"]["score"])
    mem = step_b_members(fs, rated)
    sens = {}
    for name in ("no_rater_placed_or_voted", "no_trained_pano_or_pilot_heldout"):
        sens[name] = step_b_pooling(rated, fs, maps, pi_deploy, n_boot=0, seed=args.seed, groups=("run",),
                                    members=mem[name])["groups"]["run"]["score"]
    untouched = rated[~rated["face_id"].isin(mem["_touched_faces"])]
    sens["without_touched_faces"] = step_b_pooling(untouched, fs, maps, pi_deploy, n_boot=0, seed=args.seed,
                                                   groups=("run",))["groups"]["run"]["score"]
    sens["touched_faces_rated"] = int(rated["face_id"].isin(mem["_touched_faces"]).sum())
    sens["per_city"] = {c: step_b_pooling(rated[rated["city_id"] == c], fs, maps, pi_deploy, n_boot=0,
                                          seed=args.seed, groups=("run",))["groups"]["run"]["score"]
                        for c in sorted(rated["city_id"].unique())}
    out["step_b"]["sensitivity"] = sens
    if args.out_dir:
        if rl.inside_git_checkout(args.out_dir):
            sys.exit(f"refusing to write label rows inside a git checkout: {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        cols = ["label_uid", "city", "face_id", "fold", "in_split", "scored", "truth", "truth_source", "logit",
                "logit_gate2"]
        with nr.guarded_outputs(args.out_dir, ["step_a_scores.csv"], args.force) as tmp:
            t[cols].to_csv(tmp["step_a_scores.csv"], index=False, lineterminator="\n")
    write_json(args.out_json, out, args.force)
    print(f"wrote {args.out_json}")
    return 0


def features_frames(t, args, experts):
    """Targets and label universes of item 3: step (a) labels (from the 11-05 pull, with in-memory user ids), step
    (b) rated labels (from the 11-11 pulls), and the contradicting labels of the fresh pull."""
    public = public_cities(args.cities_json)
    cols = ["label_id", "user_id", "label_type", "street_edge_id", "street_side", "latitude", "longitude"]
    ns, _ = load_pull(args.ns_pull, [LABEL_TYPE], sorted(set(t["city"])), cols)
    fresh, fresh_shas = load_pull(args.contra_pull, [LABEL_TYPE, *CONTRA_TYPES],
                                  sorted(public & (set(t["city"]) | set(args.step_b_cities))), cols)
    rb, _ = load_pull(args.round_b_pull, [LABEL_TYPE], args.step_b_cities, cols)
    for df in (ns, fresh, rb):
        df["user_key"] = df["user_id"].astype(str)
        df["expert_placed"] = df["user_id"].astype(str).isin(set(experts)).astype(int)
    return ns, fresh, rb, fresh_shas


def cmd_features(args):
    experts = expert_order(os.environ.get("EXPERTS", ""))
    t, split, gate_preds, inputs, n_folds = load_step_a(args)
    inputs["cities_json"] = check_cities(args.cities_json, args.allow_unpinned)
    inputs["ns_pull_sums"], _ = pull_sums(args.ns_pull, "ns_pull_2026_11_05_sums", args.allow_unpinned)
    inputs["round_b_pull_sums"], _ = pull_sums(args.round_b_pull, "round_b_pull_2026_11_11_sums", args.allow_unpinned)
    inputs["contra_pull_sums"], _ = pull_sums(args.contra_pull, "contra_pull_2026_11_17_sums", args.allow_unpinned)
    ns, fresh, rb, fresh_shas = features_frames(t, args, experts)
    rated, fs, b_inputs = load_step_b(args)
    inputs.update(b_inputs)
    out = {"inputs": inputs, "settings": {"n_boot": args.n_boot, "seed": args.seed, "dist_cap_m": DIST_CAP_M,
                                          "C": 1.0, "features": list(FEATURES)}}
    contra = fresh[fresh["label_type"].isin(CONTRA_TYPES) & (fresh["face_id"] != "")]
    fresh_ns = fresh[fresh["label_type"] == LABEL_TYPE]
    # step (a): the labels the second stage learns from (label-level truth, out of fold, scored)
    a_t = t[(t["fold"] >= 0) & (t["scored"] == 1) & (t["truth"] != "")].copy()
    a_ns = ns[ns["face_id"].isin(set(t["face_id"]))]
    a_t = a_t.merge(a_ns[["label_uid", "user_key"]], on="label_uid", how="left", validate="one_to_one")
    kept_a, check_a = face_check(a_ns[a_ns["label_uid"].isin(set(a_t["label_uid"]))], fresh_ns)
    # step (b): the decided rated labels, their faces' labels from the 11-11 pulls
    dec = rated[rated["verdict"].isin(["correct", "incorrect"])].copy()
    b_ns = rb[rb["face_id"].isin(set(dec["face_id"]))]
    dec = dec.merge(b_ns[["label_uid", "user_key"]], on="label_uid", how="left", validate="one_to_one")
    kept_b, check_b = face_check(b_ns[b_ns["label_uid"].isin(set(dec["label_uid"]))], fresh_ns)
    out["face_check"] = {"step_a": check_a, "step_b": check_b}
    a_t = a_t[a_t["label_uid"].isin(kept_a)]
    dec = dec[dec["label_uid"].isin(kept_b)]
    out["variants"] = {}
    for variant, excl in (("with_expert_placed", False), ("without_expert_placed", True)):
        fa = neighbour_features(a_t, a_ns, contra, excl)
        da = a_t.merge(fa, on="label_uid", how="inner", validate="one_to_one").assign(image_logit=lambda x: x["logit"])
        da = da.reset_index(drop=True)
        sets = {"ii_all": set(da["label_uid"]), "iii": set(one_per_face(da, args.seed)["label_uid"]),
                "ii_split": set(da.loc[da["truth_source"] == "split_label", "label_uid"]),
                "ii_extra": set(da.loc[da["truth_source"] == "own_expert", "label_uid"])}
        block = {"step_a_rows": int(len(da)), "step_a_incorrect": int((da["truth"] == "incorrect").sum()),
                 "step_a": second_stage(da, sets, args.n_boot, args.seed)}
        fb = neighbour_features(dec, b_ns.assign(expert_placed=b_ns["expert_placed"]), contra, excl)
        db = dec.merge(fb, on="label_uid", how="inner", validate="one_to_one")
        block["step_b"] = transfer(da, db, n_folds, args.n_boot, args.seed)
        block["feature_means"] = {"step_a": {c: float(da[c].mean()) for c in FEATURES},
                                  "step_b": {c: float(db[c].mean()) for c in FEATURES}}
        out["variants"][variant] = block
    out["circular_sensitivity_note"] = ("the all-rows fit (face-copied truth) is not run: plan 5.4 makes it a "
                                        "labelled sensitivity only")
    write_json(args.out_json, out, args.force)
    print(f"wrote {args.out_json}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _f(x, digits=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def _ci(c):
    return "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"


def _null(n):
    return "n/a" if not n or n.get("mean") is None else f"{n['mean']:.3f} (p {n['p']:.3f})"


def _rate(r):
    return "0" if not r or not r.get("n") else f"{r['k']} / {r['n']} = {_f(r['rate'])} ({_f(r['wilson95_lower'])})"


def _signed(x, digits=3):
    return "n/a" if x is None else f"{x:+.{digits}f}"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def item1_markdown(a):
    ns = a["nosidewalk"]
    lines = ["#### Item 1: do the labels of a run get the same expert verdict?", "",
             "Descriptive only. NoSidewalk on the 2026-11-05 pull; a label's verdict is its **own** expert votes "
             f"(every expert vote on it the same way, crowd ignored): {ns['counts']['with_verdict']:,} labels "
             f"({ns['counts']['expert_placed']:,} placed by an expert and {ns['counts']['ai_placed']} by SidewalkAI "
             f"left out, {ns['counts']['experts_split']} where the experts split, {ns['counts']['without_face']} "
             "without a face). Pair agreement is the share of within-run label pairs with the same verdict, with a 95 % "
             "CI from resampling runs within city. Null 1 permutes verdicts within the city, null 2 within the labeler "
             "(across their faces); p is one-sided over "
             f"{a['settings']['n_perm']:,} permutations.", "",
             "| city | labels | runs of 2+ | pairs | pair agreement [95 % CI] | all-same runs | ICC | null 1: city "
             "(p) | null 2: labeler (p) | same labeler, other face | same face, other labeler | any two in the city |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, b in ns["per_city"].items():
        name = f"other ({b.get('n_cities')} cities, pooled)" if c == "other" else c
        lines.append(f"| {name} | {b['labels']:,} | {b['multi_runs']} | {b['pairs']:,} | "
                     f"{_f(b['pair_agreement'])}{_ci(b['pair_agreement_ci95'])} | {_f(b['homogeneous_share'])} | "
                     f"{_f(b['icc'])} | {_null(b['null_city'])} | {_null(b['null_labeler'])} | "
                     f"{_f(b['same_labeler_other_face'])} | {_f(b['same_face_other_labeler'])} | {_f(b['city_any'])} |")
    m = ns["macro"]
    if m.get("n_cities"):
        lines.append(f"| **macro mean**, {m['n_cities']} cities with >= {MIN_RUNS} runs of 2+ ({', '.join(m['cities'])}) "
                     f"| {m['labels']:,} | | {m['pairs']:,} | {_f(m['pair_agreement'])}{_ci(m['pair_agreement_ci95'])} "
                     f"| {_f(m['homogeneous_share'])} | {_f(m['icc'])} | {_null(m['null_city'])} | "
                     f"{_null(m['null_labeler'])} | {_f(m['same_labeler_other_face'])} | "
                     f"{_f(m['same_face_other_labeler'])} | {_f(m['city_any'])} |")
    def scope_name(scope, strata):
        """``all`` is a pair-weighted pool, so the city with the most pairs sets it; say how much that is."""
        if scope != "all":
            return scope
        total = lambda s: sum(v["pairs"] for v in s["expert"].values())  # noqa: E731
        big = max((c for c in strata if c != "all"), key=lambda c: total(strata[c]), default=None)
        share = f"; {_pct(total(strata[big]) / total(strata['all']))} of them {big}" if big else ""
        return f"all cities (pair-weighted pool, not a macro mean{share})"

    tm = ns.get("vote_times_match") or {}
    lines += ["", "Within-run pairs by the time between the expert's two votes, and by the page they were cast on "
              f"(`pull_expert_vote_times.sh`, each expert's latest vote before the pull; cells under {slices.MIN_N} "
              f"pairs blank). Of the pull's expert votes on these labels {tm.get('matched', 0):,} match a timed vote, "
              f"{tm.get('revote', 0)} were changed around the pull and {tm.get('no_time', 0)} have no timed vote; a "
              "pair with either is \"time unknown\".", "",
              "| scope | different expert | " + " | ".join(f"same expert, {g}" for g in GAP_LABELS) +
              " | same expert, time unknown |", "|---|---|---|---|---|---|"]
    for scope, s in ns["strata"].items():
        cells = [s["gap"].get(k) for k in ("different expert", *GAP_LABELS, "unknown")]
        lines.append(f"| {scope_name(scope, ns['strata'])} | " + " | ".join(
            "0" if c is None else f"{c['pairs']:,}: {_f(c['agreement'])}" for c in cells) + " |")
    srcs = sorted({k for s in ns["strata"].values() for k in s["source"]})
    lines += ["", "| scope (same-expert pairs) | " + " | ".join(srcs) + " |", "|---" * (len(srcs) + 1) + "|"]
    for scope, s in ns["strata"].items():
        lines.append(f"| {scope_name(scope, ns['strata'])} | " + " | ".join(
            f"{s['source'][k]['pairs']:,}: {_f(s['source'][k]['agreement'])}" if k in s["source"] else "0"
            for k in srcs) + " |")
    tags = [(t, v) for t, v in ns["slices"]["tag"].items() if v["agreement"] is not None][:8]
    if tags:
        lines += ["", "By tag (pairs where both labels carry it, pair-weighted over all cities): " + "; ".join(
            f"{t} {v['pairs']:,} pairs, {_f(v['agreement'])}" for t, v in tags) + ". NoSidewalk has no severity."]
    sens = a.get("sensitivity", {})
    bits = []
    for name, text in (("dg_strict", "`data_generation.expert_verdict` (a contradicting crowd vote drops the label)"),
                       ("without_oradell", "without Oradell"), ("october_pull", "the 2026-10-18 pull")):
        mm = sens.get(name, {}).get("macro", {})
        if mm.get("n_cities"):
            bits.append(f"{text}: {_f(mm['pair_agreement'])} over {mm['n_cities']} cities (null 1 "
                        f"{_null(mm['null_city'])}, null 2 {_null(mm['null_labeler'])})")
    if bits:
        lines += ["", "Sensitivities of the macro mean: " + "; ".join(bits) + "."]
    if a.get("types"):
        lines += ["", "All seven types on the 2026-10-18 pull (macro mean over cities with >= "
                  f"{MIN_RUNS} runs of 2+; same rules):", "",
                  "| type | labels with own verdict | pairs | cities | pair agreement [95 % CI] | null 1: city (p) | "
                  "null 2: labeler (p) | same labeler, other face | same face, other labeler | share of same-expert "
                  "pairs voted < 1 h apart |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, b in a["types"].items():
            m = b["macro"]
            g = b["strata"].get("all", {}).get("gap", {})
            same = sum(v["pairs"] for k, v in g.items() if k != "different expert")
            fast = g.get(GAP_LABELS[0], {}).get("pairs", 0)
            share = _pct(fast / same) if same else "n/a"
            if not m.get("n_cities"):
                lines.append(f"| {t} | {b['counts']['with_verdict']:,} | {b['pairs_total']:,} | 0 | n/a | | | | | "
                             f"{share} |")
                continue
            lines.append(f"| {t} | {b['counts']['with_verdict']:,} | {b['pairs_total']:,} | {m['n_cities']} | "
                         f"{_f(m['pair_agreement'])}{_ci(m['pair_agreement_ci95'])} | {_null(m['null_city'])} | "
                         f"{_null(m['null_labeler'])} | {_f(m['same_labeler_other_face'])} | "
                         f"{_f(m['same_face_other_labeler'])} | {share} |")
        lines += ["", "Type x severity and type x tag (within-run pairs where both labels carry the same severity or "
                  "the tag, a pair-weighted pool over cities, not a macro mean, so the cities with the most pairs set "
                  f"it; blank under {slices.MIN_N} pairs; top three tags by pairs):", "",
                  "| type | severity 1 | severity 2 | severity 3 | tags |", "|---|---|---|---|---|"]
        cell = lambda v: "0" if v is None else f"{v['pairs']:,}: {_f(v['agreement'])}"  # noqa: E731
        for t, b in a["types"].items():
            sev = b["slices"]["severity"]
            tg = "; ".join(f"{k} {cell(v)}" for k, v in list(b["slices"]["tag"].items())[:3]) or "none"
            sevs = " | ".join(cell(sev.get(str(i))) for i in (1, 2, 3)) if sev else "no severity | | "
            lines.append(f"| {t} | {sevs} | {tg} |")
    return lines


def _gate_row(name, gate, b, auc=True):
    pw = b["precision_weighted"]
    row = (f"| {name} | {gate} | {_rate(b['raw'])} | {b['added_outside_pass']} ({b['added_outside_pass_incorrect']}) | "
           f"{b['newly_admitted']} ({b['newly_admitted_incorrect']}) | {b['dropped']} ({b['dropped_incorrect']}) | "
           f"{_f(b['coverage_weighted'])} | {_f(pw['weighted'])}{_ci(pw.get('ci95'))}; Kish n {_f(pw['n_eff'], 1)}, "
           f"{_f(pw['wilson95_lower_n_eff'])} |")
    if auc:
        row += f" {_f(b.get('auc_weighted'))} | {_f((b.get('within_city_auc_weighted') or [None])[0])} |"
    else:
        row += " | |"
    return row


GATE_HEAD = ("| group | gate | passing: correct / decided (Wilson 95 % lower) | added outside the pass stratum "
             "(incorrect) | newly admitted (incorrect) | dropped (incorrect) | weighted coverage | weighted precision "
             "[95 % CI]; Wilson at the Kish n | weighted AUC | within-city weighted AUC |")


def step_b_markdown(p):
    b = p["step_b"]
    rec = p["recommendation"]
    lines = ["#### Item 2 on step (b), the masked round (primary; exploratory, post hoc)", "",
             "Exploratory, post hoc; the step (b) bar is as published above and is not re-read here. "
             f"{b['decided']} decided ratings ({b['incorrect']} incorrect), each label's inclusion weight from the "
             "key. A pooled score is the mean logit of one fold model's `p_correct` over the group (the rated label "
             "included), served through a map fitted on step (a)'s pooled out-of-fold scores of the **same** group "
             "(all five folds; Disagree held back), then 3 of 5 models at >= 0.92 as in the round. Vote pooling: "
             "Agree when at least half of the group's scored labels pass the round's own rule; strict also needs the "
             f"label itself to pass. The single rule is the round's pass stratum: {_rate(b['single_rule'])}. "
             f"{b['singleton_runs']} decided labels have no other scored label in their run, so a run pool cannot "
             "change them. Nothing is refitted on step (b).", "", GATE_HEAD, "|---|---|---|---|---|---|---|---|---|---|"]
    lines.append(_gate_row("single (the round's rule)", "as served in round b",
                           {**b["single_block"], "auc_weighted": b["single_auc_weighted"],
                            "within_city_auc_weighted": b["single_within_city_auc_weighted"]}))
    for g, block in b["groups"].items():
        name = "single, step (a) re-scored map" if g == "single" else GROUP_TEXT[g]
        lines.append(_gate_row(name, "score", block["score"]))
        for v in ("vote", "vote_strict"):
            if v in block:
                lines.append(_gate_row(name, v.replace("_", " "), block[v], auc=False))
    prim = b["groups"]["run"]["score"]
    lines += ["", f"**Rule (fixed before any pooled number was computed):** {rec['rule']}, on the run, score "
              f"pooling. Added outside the pass stratum: {prim['added_outside_pass']} "
              f"({'met' if rec['added_ok'] else 'not met'}); newly admitted incorrect: {prim['newly_admitted_incorrect']} "
              f"({'met' if rec['incorrect_ok'] else 'not met'}); raw precision {_f(prim['raw']['rate'])} against "
              f"{_f(prim['raw_single']['rate'])} ({'met' if rec['precision_ok'] else 'not met'}). "
              f"**{'Study a per-run gate further' if rec['study_further'] else 'Do not pursue a per-run gate on this evidence'}.**"]
    if rec.get("refit_single"):
        rs, dm = rec["refit_single"], rec["run_minus_single"]
        lines[-1] += (
            " The rule compares the run pool under its refitted map with the round's own rule, so part of any change "
            "is the change of map. Without pooling, the single label under its own refitted map: added outside the "
            f"pass stratum {rs['added_outside_pass']}, newly admitted incorrect {rs['newly_admitted_incorrect']}, raw "
            f"precision {_f(rs['raw_rate'])} (it {'meets' if rec['refit_single_meets_rule'] else 'does not meet'} the "
            f"rule on its own). **What pooling adds (run minus single, both refitted):** "
            f"{dm['added_outside_pass']:+d} added outside the pass stratum, {dm['newly_admitted_incorrect']:+d} newly "
            f"admitted incorrect, raw precision {_signed(dm['raw_rate'])}.")
    lines += ["", "Transitions of the primary gate (stratum, single pass, pooled pass, verdict: labels, summed weight):",
              "", "| stratum | single passes | pooled passes | verdict | labels | weight |", "|---|---|---|---|---|---|"]
    for r in prim["transition"]:
        lines.append(f"| {r['stratum']} | {'yes' if r['single_pass'] else 'no'} | {'yes' if r['pooled_pass'] else 'no'} "
                     f"| {r['verdict']} | {r['n']} | {r['weight']:.1f} |")
    s = b["sensitivity"]
    lines += ["", "Sensitivities of the primary gate (run, score pooling):", "", GATE_HEAD,
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, text in (("no_rater_placed_or_voted", "without neighbours the rater placed or voted on"),
                       ("no_trained_pano_or_pilot_heldout", "without neighbours on trained panoramas or pilot "
                                                            "held-out rows"),
                       ("without_touched_faces", f"without the {s['touched_faces_rated']} faces the rater had touched")):
        lines.append(_gate_row(text, "score", s[name]))
    for c, blk in s["per_city"].items():
        lines.append(_gate_row(c, "score", blk))
    run = b["groups"]["run"]
    for key, text in (("tags", "score pooling"), ("tags_vote", "vote pooling"), ("tags_vote_strict", "strict vote "
                                                                                                    "pooling")):
        tags = run.get(key, {})
        if tags:
            lines += ["", f"By tag, run, {text} (>= {slices.MIN_N} decided; raw precision of passers, Wilson 95 % "
                      "lower; NoSidewalk has no severity): " + "; ".join(
                          f"{t} ({v['decided']} decided, {v.get('incorrect', 'n/a')} incorrect): pooled "
                          f"{_rate(v['pooled'])}, single {_rate(v['single'])}" for t, v in tags.items()) + "."]
    return lines


def step_a_markdown(p):
    a = p["step_a"]
    cp, cov = p["crop_parity"], p["r3_coverage"]
    lines = ["#### Item 2 on step (a), out of fold (secondary; exploratory)", "",
             f"Every NoSidewalk label on the {a['counts']['oof_faces']} out-of-fold faces and the held-out faces was "
             f"re-cropped and scored with the kept gate2 weights ({cov['scored']:,} of {cov['labels']:,} labels "
             f"scored), each out-of-fold face by its own fold's model only. Both arms use these re-scored values; "
             f"against gate2's own predictions for the {cp['rows']} split rows (other crop path) the median |delta "
             f"logit| is {_f(cp['median_abs_delta_logit'])} (90th percentile {_f(cp['p90_abs_delta_logit'])}, max "
             f"{_f(cp['max_abs_delta_logit'])}), {cp['verdict_flips']} verdicts flip at 0.5 and {cp['pass_flips']} "
             f"gate passes flip under the gate2 map"
             + (f"; for the {cp['heldout']['rows']} held-out split rows (five-model mean) the median is "
                f"{_f(cp['heldout']['median_abs_delta_logit'])} (max {_f(cp['heldout']['max_abs_delta_logit'])}), "
                f"{cp['heldout']['verdict_flips']} verdicts and {cp['heldout']['pass_flips']} gate passes flip"
                if cp.get("heldout", {}).get("rows") else "")
             + (f". The {cp['split_rows_without_face']} split rows without a face are on no step (a) face, so they "
                "are neither re-scored nor pooled" if cp.get("split_rows_without_face") else "")
             + ". Each pooled score's map is refitted leave-one-fold-out on the "
             "split rows' pooled scores of the same group. **The label-level expert truth on one face comes from the "
             "same sittings as item 1, so this comparison favours pooling; step (b) is the primary test.**", "",
             "| set | group | rows (incorrect) | AUC [95 % CI, faces] | within-city AUC | delta AUC vs single "
             "[95 % CI] | Agree gate: passing (precision, unweighted) |", "|---|---|---|---|---|---|---|"]
    names = {"ii_all": "(ii) label-level truth, all", "ii_split": "(ii) split rows (margin / expert)",
             "ii_extra": "(ii) other labels' own expert verdicts", "iii": "(iii) one label-level row per face",
             "i_circular": "(i) all split rows (circular)", "iii_circular": "(iii) one split row per face (circular)"}
    for s, text in names.items():
        for g, blocks in a["groups"].items():
            b = blocks[s]
            d = b.get("delta_vs_single")
            delta = "" if d is None else f"{_f(d['delta'])}{_ci(d['delta_ci95'])}"
            gs = b["agree_gate"]
            lines.append(f"| {text} | {g} | {b['n']} ({b['n_incorrect']}) | {_f(b['auc'])}{_ci(b['auc_ci95'])} | "
                         f"{_f(b['within_city_auc'])} | {delta} | {gs['n_pass']} ({_f(gs['precision'])}) |")
    if a.get("per_city"):
        groups = [g for g in ("single", "run", "face") if any(g in v for v in a["per_city"].values())]
        lines += ["", "Per city, set (ii) label-level truth, cities with >= 10 rows (AUC; n/a with one class):", "",
                  "| city | rows (incorrect) | " + " | ".join(f"AUC {g}" for g in groups) + " |",
                  "|---|---|" + "---|" * len(groups)]
        for c, v in a["per_city"].items():
            lines.append(f"| {c} | {v['rows']} ({v['incorrect']}) | " + " | ".join(_f(v.get(g)) for g in groups) + " |")
    if a.get("heldout"):
        lines += ["", "Held-out faces (descriptive; five-model mean logit, map fitted on all split rows):", "",
                  "| group | rows (incorrect) | AUC | within-city AUC | passing (precision) |", "|---|---|---|---|---|"]
        for g, b in a["heldout"].items():
            lines.append(f"| {g} | {b['n']} ({b['n_incorrect']}) | {_f(b['auc'])} | {_f(b['within_city_auc'])} | "
                         f"{b['agree_gate']['n_pass']} ({_f(b['agree_gate']['precision'])}) |")
    sens = a.get("sensitivity", {})
    for name, text in (("no_expert_placed", "without expert-placed neighbours"),
                       ("no_trained_pano", "without neighbours on the scoring model's trained panoramas")):
        if name in sens:
            lines += ["", f"Sensitivity, {text}: " + "; ".join(
                f"{g} AUC {_f(v['ii_all']['auc'])} (one per face {_f(v['iii']['auc'])})" for g, v in sens[name].items())
                + "."]
    g1 = p.get("step_a_gate1")
    if g1:
        lines += ["", f"Stability, the step (a) `gate/` run (its own predictions, split rows only): on {g1['rows']} "
                  f"label-level rows single AUC {_f(g1['auc_single'])}, face mean over split rows "
                  f"{_f(g1['auc_face_split_rows'])}."]
    tag = a.get("slices", {}).get("tag", {})
    if tag.get("run"):
        rows = [r for r in tag["run"] if r.get("delta") is not None]
        if rows:
            lines += ["", "Tag slices on (ii), run pooled against single (training rows are the pooled split rows "
                      "the fold models draw from; NoSidewalk has no severity):", "",
                      "| tag | rows (incorrect) | single AUC | run AUC | delta [95 % CI] |", "|---|---|---|---|---|"]
            for r in rows:
                lines.append(f"| {r['slice']} | {r['n']} ({r['n_incorrect']}) | {_f(r['auc_b'])} | {_f(r['auc_a'])} "
                             f"| {_f(r['delta'])}{_ci([r['delta_lo'], r['delta_hi']])} |")
    return lines


def features_markdown(f):
    lines = ["#### Item 3: neighbour features without the image", "",
             "Logistic regression (standardised, C = 1, no tuning): a = image logit, b = image logit + features, "
             "c = features only. Fitted on step (a) label-level truth (faces weighted equally), leave-one-fold-out; "
             "then fitted on all of it, frozen, and applied to each fold model's logit of the step (b) ratings "
             "(averaged over the five). Contradicting labels come from a fresh public pull; labels whose own row "
             f"moved face or vanished there are dropped (step (a): {f['face_check']['step_a']}; step (b): "
             f"{f['face_check']['step_b']}). The face-size prior already uses the face's label count at "
             "calibration, so `n_ns_face` is not new to the served map.", "",
             "| variant | step (a) set | rows | AUC a | AUC b | AUC c | delta b - a [95 % CI] | within-city a / b |",
             "|---|---|---|---|---|---|---|---|"]
    for v, blk in f["variants"].items():
        for s, b in blk["step_a"].items():
            if s.startswith("tags"):
                continue
            d = b["delta_b_minus_a"]
            lines.append(f"| {v.replace('_', ' ')} | {s} | {b['a_image']['n']} | {_f(b['a_image']['auc'])} | "
                         f"{_f(b['b_image_features']['auc'])} | {_f(b['c_features']['auc'])} | "
                         f"{_f(d['delta'])}{_ci(d['delta_ci95'])} | {_f(b['a_image']['within_city_auc'])} / "
                         f"{_f(b['b_image_features']['within_city_auc'])} |")
    lines += ["", "| variant | step (b) decided | weighted AUC a / b / c | within-city weighted AUC a / b / c | "
              "delta b - a, within-city weighted [95 % CI] |", "|---|---|---|---|---|"]
    for v, blk in f["variants"].items():
        t = blk["step_b"]
        m = t["models"]
        d = t["delta_b_image_features_within_city_auc_weighted"]
        lines.append(f"| {v.replace('_', ' ')} | {t['decided']} | " + " / ".join(_f(m[k]['auc_weighted']) for k in m)
                     + " | " + " / ".join(_f(m[k]['within_city_auc_weighted']) for k in m)
                     + f" | {_f(d['delta'])}{_ci(d['ci95'])} |")
    for v, blk in f["variants"].items():
        ta, tb = blk["step_a"].get("tags_ii_all", {}), blk["step_b"].get("tags", {})
        if ta or tb:
            lines += ["", f"By tag, {v.replace('_', ' ')} (>= {slices.MIN_N} rows; AUC a / b / c; NoSidewalk has no "
                      "severity): step (a) (ii) " + ("; ".join(
                          f"{t} {x['rows']} ({x['incorrect']}): " + " / ".join(_f(x[m]) for m in MODEL_COLUMNS)
                          for t, x in ta.items()) or "none") + "; step (b), weighted " + ("; ".join(
                          f"{t} {x['decided']} ({x['incorrect']}): " + " / ".join(_f(x[m]) for m in MODEL_COLUMNS)
                          for t, x in tb.items()) or "none") + "."]
    w = f["variants"].get("without_expert_placed", {}).get("step_b", {})
    d = w.get("delta_b_image_features_within_city_auc_weighted", {})
    adds = d.get("ci95") and d["ci95"][0] is not None and d["ci95"][0] > 0
    lines += ["", f"**Rule:** the features count as adding only if the step (b) within-city weighted delta (b - a) "
              f"without expert-placed labels has a 95 % CI above 0: **{'they add' if adds else 'they do not add'}**."]
    return lines


def census_markdown(census):
    lines = ["#### Runs of every type (the issue's table)", "",
             "A run is one labeler's labels of one type on one block face; a label without a face is a run of one. "
             "Public cities only.", "",
             "| type (pull) | cities | labels | in a run of 2+ | of 3+ | run length of the median label | "
             "multi-label faces with one labeler | seattle-wa: labels, in 2+, median run |",
             "|---|---|---|---|---|---|---|---|"]
    for name, per in census.items():
        allc = per.get("all")
        sea = per.get("seattle-wa")
        if not allc:
            continue
        lines.append(f"| {name.replace('_', ' ', 1).replace('_', '-')} | {len(per) - 1} | {allc['labels']:,} | "
                     f"{_pct(allc['in_run_2'])} | {_pct(allc['in_run_3'])} | {_f(allc['median_run_length'], 0)} | "
                     f"{_pct(allc['multi_faces_one_labeler'])} | "
                     + (f"{sea['labels']:,}, {_pct(sea['in_run_2'])}, {_f(sea['median_run_length'], 0)} |" if sea
                        else "n/a |"))
    ns = census.get("NoSidewalk_2026_11_05", {})
    big = sorted(((c, v) for c, v in ns.items() if c != "all" and v["labels"] >= 1000), key=lambda kv: -kv[1]["labels"])
    if big:
        lines += ["", "NoSidewalk on the 2026-11-05 pull, cities with >= 1,000 labels:", "",
                  "| city | labels | in a run of 2+ | of 3+ | run length of the median label | multi-label faces with "
                  "one labeler |", "|---|---|---|---|---|---|"]
        for c, v in big:
            lines.append(f"| {c} | {v['labels']:,} | {_pct(v['in_run_2'])} | {_pct(v['in_run_3'])} | "
                         f"{_f(v['median_run_length'], 0)} | {_pct(v['multi_faces_one_labeler'])} |")
    return lines


def markdown(rep):
    lines = []
    missing = [text for key, text in (("pooling", "item 2 (pooled scores), item 4 (the recommendation)"),
                                      ("features", "item 3 (neighbour features)")) if not rep.get(key)]
    if missing:
        lines += [f"**Not yet computed: {'; '.join(missing)}.** They need the step (a) labels re-scored with the "
                  "kept gate2 weights on klone; until those scores are copied back and pinned, this page reports item "
                  "1 and the run census only, and no statement below is about pooling or neighbour features.", ""]
    if rep.get("agreement"):
        lines += item1_markdown(rep["agreement"]) + [""]
    if rep.get("pooling"):
        lines += step_b_markdown(rep["pooling"]) + [""] + step_a_markdown(rep["pooling"]) + [""]
    if rep.get("features"):
        lines += features_markdown(rep["features"]) + [""]
    if rep.get("agreement", {}).get("census"):
        lines += census_markdown(rep["agreement"]["census"])
    return "\n".join(lines).rstrip() + "\n"


def check_public_text(text, public, all_ids):
    """Raise if ``text`` names a city id that is not public in the snapshot."""
    bad = sorted(c for c in set(all_ids) - set(public) if re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", text))
    if bad:
        raise ValueError(f"{len(bad)} non-public city ids in the report text")


def check_no_ids(text):
    if UUID_RE.search(text):
        raise ValueError("a UUID-shaped string in the report output")


def cmd_report(args):
    check_cities(args.cities_json, args.allow_unpinned)
    rep = {}
    for name in ("agreement", "pooling", "features"):
        path = getattr(args, name)
        if path:
            rep[name] = json.load(open(path))
            rep.setdefault("inputs_sha256", {})[name] = rl.sha256_file(path)
    data = json.load(open(args.cities_json))
    rows = data["cities"] if isinstance(data, dict) else data
    md = markdown(rep)
    text = json.dumps(rep) + md
    check_public_text(text, public_cities(data), [c["city_id"] for c in rows])
    check_no_ids(text)
    write_json(args.out_json, rep)
    with open(args.out_md, "w") as f:
        f.write(md)
    if args.doc:
        doc = open(args.doc).read()
        with open(args.doc, "w") as f:
            f.write(nr.splice(doc, md, start=START, end=END))
    print(f"wrote {args.out_json}, {args.out_md}" + (f" and spliced {args.doc}" if args.doc else ""))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def _common(s, step_a=True, step_b=True):
    if step_a:
        s.add_argument("--step-a", required=True, help="step-a-neighbours output (step_a_faces.csv)")
        s.add_argument("--scores", nargs="+", required=True,
                       help="the R3 run directories holding predictions_fold<k>.csv (copied back from klone)")
        s.add_argument("--gate", required=True, help="the gate2 run (folds.json, folds.csv, fold<k>/)")
    if step_b:
        s.add_argument("--key", required=True, help="round b's private key.csv")
        s.add_argument("--sheet", required=True, help="round b's votes_jon.csv")
        s.add_argument("--face-scores", required=True, help="round b's neighbours/face_scores.csv")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-unpinned", action="store_true", help="skip the PINS check (tests only)")
    s.add_argument("--force", action="store_true", help="replace outputs whose bytes would change")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("step-a-neighbours", help="list every NoSidewalk label on the step (a) faces for one scoring job")
    s.add_argument("--pull", required=True, help="the 2026-11-05 NoSidewalk rawLabels pull (one <city>.csv each)")
    s.add_argument("--gate", required=True, help="the gate2 run: folds.json, folds.csv, fold<k>/split.csv")
    s.add_argument("--cities-json", required=True, help="the day's /v3/api/cities snapshot")
    s.add_argument("--out", required=True, help="outside the repo: step_a_faces.csv, labels.csv, stage_files.txt")
    s.add_argument("--force", action="store_true", help="replace outputs whose bytes would change")
    s.add_argument("--allow-unpinned", action="store_true", help="skip the PINS check (tests only)")
    s.set_defaults(func=cmd_step_a_neighbours)
    a = sub.add_parser("agreement", help="item 1 (within-run expert agreement) and the run census; aggregates only")
    a.add_argument("--ns-pull", required=True, help="the 2026-11-05 NoSidewalk pull")
    a.add_argument("--all-pull", default=None, help="the 2026-10-18 all-type pull (the type comparison)")
    a.add_argument("--vote-times", default=None, help="pull_expert_vote_times.sh output")
    a.add_argument("--cityparams", required=True, help="a cityparams.conf snapshot (schema -> city)")
    a.add_argument("--cities-json", required=True, help="the day's /v3/api/cities snapshot")
    a.add_argument("--ns-as-of", default="2026-11-06", help="votes before this UTC time are in the NoSidewalk pull")
    a.add_argument("--all-as-of", default="2026-10-19", help="votes before this UTC time are in the all-type pull")
    a.add_argument("--n-perm", type=int, default=1000)
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--allow-unpinned", action="store_true", help="skip the PINS check (tests only)")
    a.add_argument("--force", action="store_true", help="replace an --out-json whose bytes would change")
    a.add_argument("--out-json", required=True)
    a.set_defaults(func=cmd_agreement)
    o = sub.add_parser("pooling", help="item 2: pooled against single scores on step (b) and step (a)")
    _common(o)
    o.add_argument("--gate1", default=None, help="the step (a) gate/ run, for the one-line stability check")
    o.add_argument("--priors", required=True, help="the pilot's whole/deployment_priors.json")
    o.add_argument("--out-json", required=True)
    o.add_argument("--out-dir", default=None, help="outside the repo: per-label step (a) scores")
    o.set_defaults(func=cmd_pooling)
    f = sub.add_parser("features", help="item 3: neighbour features in a second stage")
    _common(f)
    f.add_argument("--ns-pull", required=True, help="the 2026-11-05 NoSidewalk pull (step (a) faces)")
    f.add_argument("--round-b-pull", required=True, help="round b's 2026-11-11 rawlabels (step (b) faces)")
    f.add_argument("--step-b-cities", nargs="+", default=["cdmx", "chicago-il", "oradell-nj", "tucson-az"])
    f.add_argument("--contra-pull", required=True, help="a fresh public pull of CurbRamp,Obstacle,SurfaceProblem,"
                                                         "NoSidewalk (analysis/fetch_all.py --label-types)")
    f.add_argument("--cities-json", required=True)
    f.add_argument("--out-json", required=True)
    f.set_defaults(func=cmd_features)
    r = sub.add_parser("report", help="combine the three JSONs into runs.json / runs.md and splice the doc")
    r.add_argument("--agreement", default=None)
    r.add_argument("--pooling", default=None)
    r.add_argument("--features", default=None)
    r.add_argument("--cities-json", required=True, help="the day's /v3/api/cities snapshot (non-public names check)")
    r.add_argument("--out-json", required=True)
    r.add_argument("--out-md", required=True)
    r.add_argument("--doc", default=None, help="splice the markdown between the nosidewalk-runs-results markers")
    r.add_argument("--allow-unpinned", action="store_true", help="skip the PINS check (tests only)")
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
