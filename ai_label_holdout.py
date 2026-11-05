"""Hold out AI-placed labels for evaluation before training selection (issue #79).

The round-3 CurbRamp retrain ran with ``--include-ai-labels``, and :func:`data_generation.select_training_labels`
keeps every ``incorrect`` row, so it consumed every lead-rejected Vancouver AI label and left none to measure the
validator on. This module carves a seeded, pool-stable share of the eligible AI-authored labels out of the pool
BEFORE selection, so the same labels are never trained on and can be scored later (``analysis/ai_labels_eval.py
select --holdout``). It also records the policy (``generation_run.json["ai_labels"]``).

``data_generation.main`` calls it in exactly two places, each a single line::

    eligible, ai_holdout = ai_label_holdout.apply(eligible, args)       # between eligible_labels and selection
    ai_labels_record = ai_label_holdout.record(ai_holdout, eligible, selection, base_dir, args)  # after makedirs

**Membership.** A label is held out when ``holdout_fraction(seed, label_uid) < frac``: a per-label hash, so adding or
removing one label (new votes) moves at most that label in or out, and the held-out set of a later pull is the earlier
one plus or minus the labels that changed. The hash stream is salted (``HOLDOUT_SALT``), so it is independent of the
correct-class downsample rank (:func:`data_generation.label_rank`). Every AI-authored row is a candidate, rejected ones
included.

**Both policies carve.** The held-out set is carved with or without ``--include-ai-labels``. Under ``excluded`` the
removed rows would have been dropped by the selection anyway, so an existing run's selection hash does not move; and
the two arms of the with/without ablation then score the identical held-out labels.

**Tier.** ``CAMPAIGNS``/``TIERS``/:func:`tier_of` live here (moved from ``analysis/ai_labels_eval.py``, which
re-imports them) because ``data_generation`` cannot import that script: it imports ``data_generation``.

Needs only numpy/pandas; tested in tests/test_ai_label_holdout.py.
"""
import argparse
import hashlib
import os

import pandas as pd

HOLDOUT_FILE = "ai_label_holdout.csv"
#: Separates the holdout hash stream from the correct-class downsample's ``sha256(f"{seed}:{label_uid}")``.
HOLDOUT_SALT = "ai-holdout"
#: Share of eligible AI-authored labels held out. 0.5 is a judgement: the with/without ablation is the point of the
#: held-out set, so evaluation power is weighted equally with training rows (docs/ai-label-holdout-2026-11.md).
DEFAULT_FRAC = 0.5
POLICIES = ("excluded", "included_tagged")
HOLDOUT_COLUMNS = ["label_uid", "city", "label_id", "label_class", "class_source", "pano_source", "tier", "stratum",
                   "holdout_rank"]

#: Submission campaigns per deployment: (first UTC date, tier). A label's tier is that of the latest campaign whose date
#: is <= its ``time_created`` date. From sidewalk-auto-labeler ``runs/<city>/*.submission.json`` (richmond, laurens) and
#: its docs/server-agree-check.md (Vancouver: one campaign at the 0.55 operating point, 2025-09-18/19, before the band
#: existed). Richmond 2026-09-24 re-submitted 14 repositioned sequences at 0.30 (``results.posfix3seq.raw.jsonl``), so
#: those labels mix both tiers.
CAMPAIGNS = {
    "vancouver-wa": [("2025-09-18", "core")],
    "richmond-va": [("2026-09-05", "core"), ("2026-09-22", "band"), ("2026-09-24", "mixed")],
    "laurens-ia": [("2026-09-17", "core"), ("2026-09-22", "band")],
}
TIERS = ("core", "band", "mixed")


def tier_of(city, time_created, campaigns=CAMPAIGNS):
    """The confidence tier of a label from its deployment's campaign dates; ``""`` when unknown.

    >>> tier_of("richmond-va", "2026-09-05T01:10:00Z"), tier_of("richmond-va", "2026-09-22 19:03:45.1")
    ('core', 'band')
    >>> tier_of("richmond-va", "2026-09-30"), tier_of("laurens-ia", "2026-09-01"), tier_of("nowhere", "2026-09-05")
    ('mixed', '', '')

    Campaign days are UTC dates; a timestamp with an offset is converted first, a naive one is taken as UTC (what the
    rawLabels export writes):

    >>> tier_of("laurens-ia", "2026-09-21T20:00:00-07:00"), tier_of("laurens-ia", "not a date")
    ('band', '')
    """
    try:
        ts = pd.to_datetime(time_created, utc=True)
    except (ValueError, TypeError):
        return ""
    if pd.isna(ts):
        return ""
    day = ts.strftime("%Y-%m-%d")
    tier = ""
    for first_day, name in campaigns.get(city, []):
        if day >= first_day:
            tier = name
    return tier


def holdout_fraction(seed, label_uid, salt=HOLDOUT_SALT):
    """A label's own uniform draw in ``[0, 1)``: the first 12 hex digits of ``sha256(f"{salt}:{seed}:{label_uid}")``.

    >>> holdout_fraction(42, "toy-city:4") == holdout_fraction(42, "toy-city:4")
    True
    >>> 0 <= holdout_fraction(1, "x:1") < 1
    True
    """
    return int(hashlib.sha256(f"{salt}:{seed}:{label_uid}".encode()).hexdigest()[:12], 16) / 16 ** 12


def check_frac(frac):
    """``frac`` as a float, or ``ValueError`` unless ``0 < frac < 1``."""
    frac = float(frac)
    if not 0 < frac < 1:
        raise ValueError(f"the AI-label holdout fraction must be strictly between 0 and 1, got {frac}")
    return frac


def frac_arg(value):
    """argparse ``type=`` for ``--ai-holdout-frac``: a usage error, not a traceback, for a bad value."""
    try:
        return check_frac(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def policy(include_ai_labels):
    """``included_tagged`` when AI-authored labels may enter training (tagged ``author = ai``), else ``excluded``.

    >>> policy(True), policy(False)
    ('included_tagged', 'excluded')
    """
    return "included_tagged" if include_ai_labels else "excluded"


def _uids(df):
    return df["city"].astype(str) + ":" + df["label_id"].astype(str)


def carve(eligible, seed, frac=DEFAULT_FRAC):
    """Split ``eligible`` (an :func:`data_generation.eligible_labels` frame) into ``(remaining, holdout)``.

    ``holdout`` holds the AI-authored rows whose :func:`holdout_fraction` is below ``frac``, with
    :data:`HOLDOUT_COLUMNS` (``tier`` from :func:`tier_of`, ``""`` when ``time_created`` is absent or the deployment
    has no campaign table; ``stratum`` = ``city/label_class/tier``; ``holdout_rank`` = the draw), sorted by
    ``label_uid``. ``remaining`` is every other row, columns unchanged, index reset. Human-authored rows never move.

    >>> e = pd.DataFrame({"city": ["c"] * 4, "label_id": [1, 2, 3, 4], "author": ["human", "ai", "ai", "ai"],
    ...                   "label_class": ["correct", "incorrect", "correct", "incorrect"], "class_source": "margin"})
    >>> rest, held = carve(e, seed=42, frac=0.5)
    >>> len(rest) + len(held) == 4 and int((rest.author == "human").sum()) == 1
    True
    """
    frac = check_frac(frac)
    uid = _uids(eligible)
    ai = (eligible["author"] == "ai").to_numpy()
    draw = pd.Series([holdout_fraction(seed, u) if a else 1.0 for u, a in zip(uid, ai)], index=eligible.index)
    held = ai & (draw < frac).to_numpy()
    holdout = eligible[held].copy()
    holdout["label_uid"] = uid[held]
    for col in ("class_source", "pano_source"):
        if col not in holdout:
            holdout[col] = ""
    times = holdout["time_created"] if "time_created" in holdout else pd.Series("", index=holdout.index)
    holdout["tier"] = [tier_of(c, t) for c, t in zip(holdout["city"], times)]
    holdout["stratum"] = holdout["city"] + "/" + holdout["label_class"] + "/" + holdout["tier"]
    holdout["holdout_rank"] = draw[held]
    holdout = holdout[HOLDOUT_COLUMNS].sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    return eligible[~held].reset_index(drop=True), holdout


def counts_by_city_class(df):
    """``{city: {"correct": n, "incorrect": n}}`` over the rows of ``df``; cities with no row are absent.

    >>> counts_by_city_class(pd.DataFrame({"city": ["b", "a", "a"], "label_class": ["correct", "incorrect", "correct"]}))
    {'a': {'correct': 1, 'incorrect': 1}, 'b': {'correct': 1, 'incorrect': 0}}
    """
    out = {}
    for city, g in df.groupby("city", sort=True):
        out[str(city)] = {cls: int((g["label_class"] == cls).sum()) for cls in ("correct", "incorrect")}
    return out


def summary_line(holdout):
    """One printable line: the held-out AI labels per deployment and class.

    >>> summary_line(pd.DataFrame(columns=HOLDOUT_COLUMNS))
    'AI-label holdout (issue #79): no AI-authored label held out'
    """
    if not len(holdout):
        return "AI-label holdout (issue #79): no AI-authored label held out"
    per = "; ".join(f"{c} {v['correct']} correct / {v['incorrect']} incorrect"
                    for c, v in counts_by_city_class(holdout).items())
    return f"AI-label holdout (issue #79): {len(holdout)} AI-authored labels held out of training ({per})"


def apply(eligible, args):
    """The selection hook: ``(remaining, holdout)`` = :func:`carve` of ``eligible`` with ``args.seed`` and
    ``args.ai_holdout_frac``, carved under both policies (module docstring), with the summary printed."""
    frac = getattr(args, "ai_holdout_frac", DEFAULT_FRAC)
    remaining, holdout = carve(eligible, args.seed, frac)
    n_ai = int((eligible["author"] == "ai").sum())
    print(f"{summary_line(holdout)}; {n_ai} eligible AI-authored, frac {frac}, policy "
          f"{policy(getattr(args, 'include_ai_labels', False))}")
    return remaining, holdout


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(holdout, remaining, selection, base_dir, args):
    """Write ``<base_dir>/ai_label_holdout.csv`` and return the ``generation_run.json["ai_labels"]`` entry.

    The CSV is written on every run, empty when no AI-authored label was held out, so its absence means a run from
    before issue #79. ``remaining`` is the pool after :func:`apply` and ``selection`` the selected ``label_uid``s
    (a mapping or iterable, taken before ``--limit``), so ``included_counts`` counts the AI-authored labels training
    actually got (``{}`` under ``excluded``).

    Returns ``{"policy", "holdout_frac", "holdout_salt", "holdout_seed", "holdout_file", "holdout_sha256",
    "holdout_counts", "included_counts", "eligible_ai_counts"}``; counts are ``{city: {"correct": n, "incorrect":
    n}}`` and ``eligible_ai_counts`` is before carving.
    """
    include = bool(getattr(args, "include_ai_labels", False))
    path = os.path.join(base_dir, HOLDOUT_FILE)
    out = holdout[HOLDOUT_COLUMNS].sort_values("label_uid", kind="mergesort")
    if not out["label_uid"].is_unique:
        raise ValueError("label_uid (city:label_id) is not unique in the AI-label holdout")
    out.to_csv(path, index=False, lineterminator="\n")
    ai_left = remaining[remaining["author"] == "ai"].copy()
    ai_left["label_uid"] = _uids(ai_left)
    chosen = set(selection)
    included = ai_left[ai_left["label_uid"].isin(chosen)] if include else ai_left.iloc[0:0]
    eligible_ai = pd.concat([ai_left[["city", "label_class"]], holdout[["city", "label_class"]]], ignore_index=True)
    return {"policy": policy(include), "holdout_frac": getattr(args, "ai_holdout_frac", DEFAULT_FRAC),
            "holdout_salt": HOLDOUT_SALT, "holdout_seed": args.seed, "holdout_file": HOLDOUT_FILE,
            "holdout_sha256": sha256_file(path), "holdout_counts": counts_by_city_class(holdout),
            "included_counts": counts_by_city_class(included), "eligible_ai_counts": counts_by_city_class(eligible_ai)}


def load_holdout(path):
    """A holdout CSV written by :func:`record`, every column as text; ``ValueError`` on a duplicate ``label_uid``."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"label_uid", "label_class"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if not df["label_uid"].is_unique:
        dup = df["label_uid"][df["label_uid"].duplicated()].head(3).tolist()
        raise ValueError(f"{path}: duplicate label_uid {dup}")
    return df
