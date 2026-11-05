"""Block-face training verdicts for NoSidewalk (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42).

Why a face, not a label. SidewalkWebpage validates NoSidewalk per **block face**, ``(street_edge_id, street_side)``
(SidewalkWebpage #5285, ``app/models/validation/ValidationQueuePolicy.scala``), and treats a face as settled at two
agreeing human votes (``SettledMargin = 2``, ``FaceSettledSupport = 2``). So the third net Agree vote that
``data_generation.eligible_labels`` needs almost never arrives: on the 2026-11-05 public rawLabels 1,579 NoSidewalk
labels sit at exactly (2 agree, 0 disagree), and the label margin rule finds 79 correct / 80 incorrect labels in 237,291.
The website's votes are also spread over a face's labels: a face with ten labels and two Agree votes on two different
labels is settled for the website, while every one of its labels is unusable label by label.

The rule (``docs/nosidewalk-2026-11.md``, ``docs/decisions.md``). Label-level verdicts (margin, then expert) are
computed first, exactly as for every other type, and are never overridden. Then each label the label rule leaves
undecided takes its face's verdict, from the pooled human votes on every label of the face:

- ``support`` = distinct human validators who voted Agree on any label of the face, ``opposition`` = distinct human
  validators who voted Disagree on any label of the face. Distinct people, not votes: one validator agreeing with
  three labels of a face is one piece of evidence.
- ``correct`` if support >= 2 and opposition == 0, ``incorrect`` if opposition >= 2 and support == 0
  (``class_source = face_margin``);
- else, with ``EXPERTS``: an expert voted on the face, all experts the same way, and no human voted the other way
  anywhere on the face: the expert's verdict (``class_source = face_expert``);
- a contradicted face (support > 0 and opposition > 0) decides nothing beyond the label rule.

Only ``validator_type == "Human"`` votes count (the SidewalkAI account's votes never do), as in
``data_generation.human_vote_counts``. A label without a usable ``street_edge_id`` / ``street_side`` has no face and
keeps the label rule only. Label and face verdicts cannot disagree: a label-level ``incorrect`` needs a Disagree vote,
which makes the face's opposition positive, so the face cannot be ``correct`` (and the other way round).

``cap_labels_per_face`` then keeps at most ``n`` labels per (face, class), the ``n`` with the lowest
``data_generation.label_rank`` (stable under a changing pool), so a 30-label face is not 5 % of the negatives.

``data_generation.py --verdict-unit face`` calls :func:`apply_verdict_unit` once, right after ``eligible_labels``.
``data_generation`` is imported inside the functions, not at the top, because it imports this module.
"""
import json

import numpy as np
import pandas as pd

#: The two values of ``street_side`` that name a face; anything else (empty, NaN) means the label has no face.
FACE_SIDES = ("left", "right")
#: The ``class_source`` values this module adds (``data_generation`` sets ``margin`` and ``expert``).
FACE_SOURCES = ("face_margin", "face_expert")
#: ``--verdict-unit`` choices; ``label`` is every type's rule and the default.
VERDICT_UNITS = ("label", "face")
#: Default ``--max-labels-per-face`` (per face and class), plan decision 2.
DEFAULT_MAX_LABELS_PER_FACE = 5


def face_id(df):
    """The block face of each row, ``"<city>:<street_edge_id>:<street_side>"``, or ``""`` when it has none.

    ``street_edge_id`` is per city (a serial in each city schema), so the city is part of the key.

    >>> face_id(pd.DataFrame({"city": ["a", "a", "b", "b"], "street_edge_id": [12.0, 12, None, 7],
    ...                       "street_side": ["left", "Right ", "left", None]})).tolist()
    ['a:12:left', 'a:12:right', '', '']
    """
    edge = pd.to_numeric(df["street_edge_id"], errors="coerce") if "street_edge_id" in df else \
        pd.Series(np.nan, index=df.index)
    side = df["street_side"].astype(str).str.strip().str.lower() if "street_side" in df else \
        pd.Series("", index=df.index)
    ok = np.isfinite(edge) & side.isin(FACE_SIDES)
    out = pd.Series("", index=df.index, dtype=object)
    out[ok] = (df.loc[ok, "city"].astype(str) + ":" + edge[ok].astype("int64").astype(str) + ":" + side[ok])
    return out


def _human_votes(validations_json):
    """``[(user_id, "Agree"|"Disagree"), ...]`` of the human votes in one ``validations`` cell.

    A vote without a ``user_id`` cannot be told apart from another person's, so it gets ``None`` and
    :func:`face_votes` counts it as its own validator.

    >>> _human_votes('[{"user_id":"a","validation":"Agree","validator_type":"Human"},'
    ...              ' {"user_id":"b","validation":"Unsure","validator_type":"Human"},'
    ...              ' {"user_id":"c","validation":"Disagree","validator_type":"AI"}]')
    [('a', 'Agree')]
    """
    if not isinstance(validations_json, str) or not validations_json:
        return []
    out = []
    for v in json.loads(validations_json):
        if v.get("validator_type") == "Human" and v.get("validation") in ("Agree", "Disagree"):
            uid = v.get("user_id")
            out.append((None if uid is None else str(uid), v["validation"]))
    return out


def face_votes(df, experts=frozenset()):
    """Pooled human votes per face, one row per non-empty ``face_id`` in ``df``.

    ``df`` needs ``face_id`` (:func:`face_id`), ``validations`` and, for ``labelers``, ``user_id``. Columns:

    - ``n_labels``: labels of ``df`` on the face; ``labelers``: distinct label authors;
    - ``support`` / ``opposition``: distinct human validators who voted Agree / Disagree on any label of the face;
    - ``agree_votes`` / ``disagree_votes``: the same as vote counts (one person on three labels counts three times;
      what the 2026-11-05 census summed);
    - ``expert_agree`` / ``expert_disagree``: distinct validators in ``experts`` voting each way;
    - ``contradicted``: support > 0 and opposition > 0.

    >>> v = lambda *vs: json.dumps([{"user_id": u, "validation": x, "validator_type": "Human"} for u, x in vs])
    >>> d = pd.DataFrame({"face_id": ["f", "f", "g"], "user_id": ["l1", "l2", "l1"],
    ...                   "validations": [v(("a", "Agree")), v(("a", "Agree"), ("e", "Agree")), "[]"]})
    >>> face_votes(d, {"e"})[["n_labels", "support", "agree_votes", "expert_agree", "contradicted"]].to_dict("index")
    {'f': {'n_labels': 2, 'support': 2, 'agree_votes': 3, 'expert_agree': 1, 'contradicted': False}, \
'g': {'n_labels': 1, 'support': 0, 'agree_votes': 0, 'expert_agree': 0, 'contradicted': False}}
    """
    with_face = df[df["face_id"].astype(str) != ""]
    rows = {}
    anonymous = 0
    for fid, validations in zip(with_face["face_id"], with_face["validations"]):
        r = rows.setdefault(fid, {"agree": set(), "disagree": set(), "agree_votes": 0, "disagree_votes": 0})
        for user, vote in _human_votes(validations):
            if user is None:  # cannot be de-duplicated: its own validator
                anonymous += 1
                user = f"<anonymous {anonymous}>"
            if vote == "Agree":
                r["agree"].add(user)
                r["agree_votes"] += 1
            else:
                r["disagree"].add(user)
                r["disagree_votes"] += 1
    experts = frozenset(str(e) for e in experts)
    n_labels = with_face.groupby("face_id").size()
    if "user_id" in with_face:
        labelers = with_face.groupby("face_id")["user_id"].nunique()
    else:
        labelers = pd.Series(0, index=n_labels.index)
    out = pd.DataFrame({
        "n_labels": n_labels,
        "labelers": labelers,
        "support": pd.Series({f: len(r["agree"]) for f, r in rows.items()}, dtype="int64"),
        "opposition": pd.Series({f: len(r["disagree"]) for f, r in rows.items()}, dtype="int64"),
        "agree_votes": pd.Series({f: r["agree_votes"] for f, r in rows.items()}, dtype="int64"),
        "disagree_votes": pd.Series({f: r["disagree_votes"] for f, r in rows.items()}, dtype="int64"),
        "expert_agree": pd.Series({f: len(r["agree"] & experts) for f, r in rows.items()}, dtype="int64"),
        "expert_disagree": pd.Series({f: len(r["disagree"] & experts) for f, r in rows.items()}, dtype="int64"),
    }).fillna(0).astype("int64")
    out["contradicted"] = (out["support"] > 0) & (out["opposition"] > 0)
    out.index.name = "face_id"
    return out


def face_classes(votes, use_experts, agree_min=2, disagree_min=2, count="validators"):
    """``face_class`` (``correct`` / ``incorrect`` / ``""``) and ``face_source`` (``face_margin`` / ``face_expert`` /
    ``""``) for each row of :func:`face_votes`.

    ``count="validators"`` is the training rule (distinct people). ``count="votes"`` sums votes instead; it is how
    the 2026-11-05 census counted (``analysis/type_readiness.py`` reproduces that census with it) and is not used for
    training. The expert branch is the same in both: an expert voted, experts all one way, and nobody voted the other
    way anywhere on the face.

    >>> votes = pd.DataFrame({"support": [2, 1, 0, 2, 0], "opposition": [0, 0, 2, 1, 0],
    ...                       "agree_votes": [2, 2, 0, 2, 0], "disagree_votes": [0, 0, 2, 1, 0],
    ...                       "expert_agree": [0, 0, 0, 1, 0], "expert_disagree": [0, 0, 0, 0, 0]},
    ...                      index=["two", "one-twice", "neg", "contradicted", "none"])
    >>> face_classes(votes, False)["face_class"].tolist()
    ['correct', '', 'incorrect', '', '']
    >>> face_classes(votes, False, count="votes")["face_class"].tolist()
    ['correct', 'correct', 'incorrect', '', '']
    """
    if count == "validators":
        a, d = votes["support"], votes["opposition"]
    elif count == "votes":
        a, d = votes["agree_votes"], votes["disagree_votes"]
    else:
        raise ValueError(f"count must be 'validators' or 'votes', got {count!r}")
    cls = np.where((a >= agree_min) & (d == 0), "correct", np.where((d >= disagree_min) & (a == 0), "incorrect", ""))
    src = np.where(cls != "", "face_margin", "")
    if use_experts:
        ea, ed = votes["expert_agree"], votes["expert_disagree"]
        exp = np.where((ea > 0) & (ed == 0) & (d == 0), "correct",
                       np.where((ed > 0) & (ea == 0) & (a == 0), "incorrect", ""))
        by_expert = (cls == "") & (exp != "")
        cls = np.where(by_expert, exp, cls)
        src = np.where(by_expert, "face_expert", src)
    return pd.DataFrame({"face_class": cls, "face_source": src}, index=votes.index)


def _label_uid(df):
    return df["city"].astype(str) + ":" + df["label_id"].astype(str)


def type_rows(all_labels, label_type):
    """The rows of ``label_type`` with a usable pano point (``data_generation.valid_geometry``), with ``face_id``.

    The face's votes are pooled over exactly the labels ``eligible_labels`` looks at, so a label dropped for its
    geometry neither gets a face verdict nor lends its votes to one.
    """
    import data_generation as dg
    df = all_labels[all_labels["label_type"] == label_type]
    keep, _ = dg.valid_geometry(df)
    df = df[keep].copy()
    df["face_id"] = face_id(df)
    return df


def add_face_verdicts(all_labels, eligible, label_type, experts=frozenset(), agree_min=2, disagree_min=2):
    """``eligible`` (``data_generation.eligible_labels``'s frame) plus the labels the face rule decides, with a
    ``face_id`` column on every row.

    ``all_labels`` is the full rawLabels frame (every type, every city): ``eligible_labels`` drops the undecided
    labels, and the face rule needs them. Rows already in ``eligible`` keep their verdict and ``class_source``; an
    added row gets the same derived columns ``eligible_labels`` adds (``author``, ``human_agree``,
    ``human_disagree``) and ``class_source`` ``face_margin`` or ``face_expert``. Keys are ``label_uid``
    (``city:label_id``), never ``label_id`` alone.
    """
    import data_generation as dg
    rows = type_rows(all_labels, label_type)
    rows_uid = _label_uid(rows)
    if not rows_uid.is_unique:
        raise ValueError(f"label_uid (city:label_id) is not unique among the {label_type} rows")
    classes = face_classes(face_votes(rows, experts), bool(experts), agree_min, disagree_min)

    eligible = eligible.copy()
    eligible["face_id"] = face_id(eligible)
    decided = set(_label_uid(eligible))
    undecided = rows[~rows_uid.isin(decided) & (rows["face_id"] != "")].copy()
    undecided = undecided.merge(classes, left_on="face_id", right_index=True, how="left", validate="many_to_one")
    added = undecided[undecided["face_class"].fillna("") != ""].copy()
    added["author"] = added["user_id"].map(dg.label_author) if "user_id" in added else "human"
    counts = added["validations"].apply(dg.human_vote_counts)
    added["human_agree"] = [a for a, _ in counts]
    added["human_disagree"] = [d for _, d in counts]
    added["label_class"] = added["face_class"]
    added["class_source"] = added["face_source"]
    added = added.drop(columns=["face_class", "face_source"])

    # The two rules cannot disagree (module docstring); check it rather than trust it.
    on_face = eligible[eligible["face_id"] != ""].merge(classes, left_on="face_id", right_index=True, how="left",
                                                        validate="many_to_one")
    clash = (on_face["face_class"].fillna("") != "") & (on_face["face_class"] != on_face["label_class"])
    if clash.any():
        raise AssertionError(f"{int(clash.sum())} label-level verdicts contradict their face's verdict, e.g. "
                             f"{_label_uid(on_face[clash]).iloc[0]}")
    out = pd.concat([eligible, added], ignore_index=True)
    assert _label_uid(out).is_unique, "label_uid is not unique after adding face verdicts"
    return out


def cap_labels_per_face(df, n, seed):
    """Keep at most ``n`` rows per (``face_id``, ``label_class``): the ``n`` with the lowest
    ``data_generation.label_rank(seed, label_uid)``. Rows without a face are all kept; the row order is kept.
    ``n`` of None or 0 means no cap.

    The rank is each label's own seeded hash, so when a label enters or leaves a face the kept set changes by at most
    one label in and one out (the same stability argument as ``select_training_labels``).

    >>> d = pd.DataFrame({"city": ["c"] * 4, "label_id": [1, 2, 3, 4], "face_id": ["f", "f", "f", ""],
    ...                   "label_class": ["correct"] * 4})
    >>> len(cap_labels_per_face(d, 2, seed=42)), cap_labels_per_face(d, 2, seed=42)["label_id"].tolist()[-1]
    (3, 4)
    """
    if not n:
        return df
    if n < 0:
        raise ValueError(f"max labels per face must be >= 0, got {n}")
    import data_generation as dg
    rank = pd.Series([dg.label_rank(seed, u) for u in _label_uid(df)], index=df.index)
    has_face = df["face_id"].astype(str) != ""
    order = pd.DataFrame({"face_id": df["face_id"], "label_class": df["label_class"], "rank": rank})[has_face]
    order = order.sort_values(["face_id", "label_class", "rank"], kind="mergesort")
    kept = order.groupby(["face_id", "label_class"], sort=False).cumcount() < n
    keep = ~has_face
    keep[kept.index[kept.to_numpy()]] = True
    return df[keep]


def apply_verdict_unit(all_labels, eligible, label_type, experts, unit, max_labels_per_face, seed):
    """``data_generation.main``'s one hook: ``eligible`` unchanged for ``unit == "label"``; for ``"face"``,
    :func:`add_face_verdicts` then :func:`cap_labels_per_face`, with a printed summary.

    The deployment prior (``data_generation.deployment_prior_entry``) still counts ``class_source == "margin"`` rows
    only, so face rows never enter it. It is computed after this hook, so under ``face`` it counts the margin labels
    that survive the cap; that entry is a placeholder in any case: NoSidewalk's prior is written over it by
    ``calibration/nosidewalk_prior.py`` (the pilot runs it after ``data_generation.py``), because the margin-rule
    count is a queue-biased sample (``docs/nosidewalk-2026-11.md``).
    """
    if unit == "label":
        return eligible
    if unit != "face":
        raise ValueError(f"verdict unit must be one of {VERDICT_UNITS}, got {unit!r}")
    before = len(eligible)
    enlarged = add_face_verdicts(all_labels, eligible, label_type, experts)
    capped = cap_labels_per_face(enlarged, max_labels_per_face, seed)
    by = enlarged.groupby(["class_source", "label_class"]).size().to_dict()
    print(f"Face verdicts ({label_type}, --verdict-unit face): {before} label-rule labels + "
          f"{len(enlarged) - before} face-rule labels; by source and class {by}; "
          f"cap {max_labels_per_face or 'off'} per face and class dropped {len(enlarged) - len(capped)}")
    return capped.reset_index(drop=True)


def resume_settings(unit, max_labels_per_face):
    """The ``generation_run.json`` settings for a verdict unit: none for ``label`` (so earlier records are unchanged),
    ``verdict_unit`` and ``max_labels_per_face`` for ``face`` (``data_generation.OPTIONAL_SETTINGS``, so switching
    either way, or changing the cap, refuses a resume without ``--force-reselect``).

    >>> resume_settings("label", 5), resume_settings("face", 5)
    ({}, {'verdict_unit': 'face', 'max_labels_per_face': 5})
    """
    if unit == "label":
        return {}
    return {"verdict_unit": unit, "max_labels_per_face": max_labels_per_face}
