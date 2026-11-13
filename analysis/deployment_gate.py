"""Per-deployment status of the AI's gated votes, and a replay of the rule on the previous models (issue #52).

Why
---
Issue #52 option 1: a deployment's AI votes should count only once the AI has been checked there, per label type and
per verdict. SidewalkWebpage#5832 proposes a per city, type and verdict mode (``off`` / ``shadow`` / ``vote``) and
needs a check that says when ``shadow`` may become ``vote``. This module is that check, as a **status report**. It
does not decide a serving mode for any existing deployment: whether existing deployments keep voting while they build
evidence is an open decision, so the public output carries no mode at all. The private output records the mode each
stratum would get under each candidate policy, for comparison only; nothing applies it.

The rule (defaults; every value is a flag and is recorded in the output)
--------------------------------------------------------------------------
* **Stratum**: one deployment x label type x verdict (Agree, Disagree) x evidence regime, in one model era.
* **Gated assessment**: the AI's latest assessment of a label, verdict Agree or Disagree, served accuracy (the
  recorded ``ai_accuracy``) at least ``gate`` (0.92, ``calibration._isotonic_map.GATE``).
* **Reference** (``--reference``, default ``majority2``): a label is *decided* when it has at least two human Agree
  or Disagree votes and they are not tied; the AI's own vote is excluded (in the pull's SQL), Unsure votes are not
  counted. ``data_generation.prior_class`` gives the class of the LABEL: ``correct`` (humans agree with the label) or
  ``incorrect``. An AI **Agree** is right when the class is ``correct``; an AI **Disagree** is right when it is
  ``incorrect``. Two sensitivity references are reported beside it, never pooled into it: ``tie_half`` (at least
  two votes, a tie counts half right, what ``per_city.py`` uses) and ``any_vote`` (at least ONE human vote, strict
  majority). ``any_vote`` matters because in vote mode the AI's own Agree plus one human Agree settles a label
  (SidewalkWebpage ``ValidationQueuePolicy``: margin 2 includes the AI's vote), so a label confirmed by one human
  never reaches two votes and ``majority2`` never counts it, while a contested one is routed to more votes.
* **Checkpoints** (``--schedule``, default 50, 100, 200, 400, 800, 1600, then doubling): a stratum's status is
  evaluated on its first C decided rows, in assessment order (``ai_date``, then ``label_id``, within one schema),
  where C is the largest checkpoint not above its decided count. Evaluating only at fixed checkpoints, not at every
  re-run, is what keeps the chance of a lucky pass bounded; ``power`` gives that chance exactly.
* **Status** at checkpoint C with k right (Wilson interval at ``z``, default 1.96):

  ======================  =================================================================================
  ``pass``                C >= ``min_n`` (50) and the Wilson lower bound >= ``gate``
  ``fail``                C >= ``min_n`` and the Wilson upper bound < ``gate``
  ``undecided``           C >= ``min_n`` and the interval contains ``gate``
  ``insufficient``        at least one gated assessment, but fewer decided rows than the first checkpoint
  ``no_gated``            assessments of this verdict, none at or above the gate (every held-back Disagree)
  ``never_votes``         the type's policy says this verdict never votes (NoSidewalk Disagree)
  ``not_served``          production evidence for a type the API does not serve
  ======================  =================================================================================

* **Era**: with a ``validator_revision`` column (pulls after this PR's change to ``pull_ai_assessments.sh``), the
  current era is each model's newest revision and the previous era every other row. Without it (the 2026-11-11
  pull), the era is a DATE split: current = ``ai_date >= --since`` (2026-10-28, the first full day of the round-3
  revisions), previous = before 2026-10-27, the changeover day in neither. Status uses the current era only.
* **Evidence regime**: with an ``ai_vote_cast`` column, a row is ``vote`` (the AI's vote was cast) or ``shadow``
  (scored, no vote); the two are separate strata, never pooled. Without it every row is ``unrecorded``.

Evidence beyond production votes
--------------------------------
Every stratum carries ``evidence`` (``production_votes`` or ``masked_expert_round``), ``evidence_ref``, ``basis``
(``raw`` or ``prior_weighted``), ``min_n`` and ``model_revision``, and the report's ``types`` block lists every type
with ``served`` and, per verdict, ``eligible`` or ``never``. :func:`expert_round_stratum` builds a stratum from a
masked expert round's counts (one look, all rows), which is how NoSidewalk step (c) would enter its evidence. This
module computes production evidence only.

Subcommands
-----------
``gate``      the status report: public ``deployment_status.json`` (public deployments by ``city_id``; every figure,
              pooled ones included, over PUBLIC deployments only, so nothing about a non-public deployment can be
              recovered by subtraction) and, under ``--private-out``, ``deployment_status_all.json`` / ``.csv`` with
              every deployment, the pooled figures that include non-public ones, and the mode each stratum would get
              under each policy. Also the acceptance check (every deployment's current-era decided gated Agree counts
              against ``--expect``; only its outcome is public).
``backtest``  count-based sequential replay of the rule on the previous era, split by evidence regime (votes as of the
              pull's end; no dates, no waiting times), with the items of masked re-label rounds excluded; public
              deployments only in the public file.
``power``     exact (no simulation, no seed): errors allowed per n, single-look pass probability, and the chance of
              EVER passing under a look schedule at a given true precision; the z that holds it to 5%.
``slices``    type x severity and type x tag (tags joined from cached public rawLabels), gated Agree, current era
              against the previous one, with training counts; and severity within each pass/undecided stratum.
``doc``       fills the generated blocks of the doc from the public JSON files; ``--check`` fails on drift.
``privacy-check``  word-boundary, case- and accent-insensitive search of files for every non-public term the live
              listing gives (plus the pull's private schema names and an optional extra-terms file); prints the
              file and offset of a hit, never the term.

Naming: the listing comes from the live ``/v3/api/cities`` at run time (``data_generation.fetch_city_listing``); there
is no option to read it from a saved file. A schema is named by its ``city_id`` only when
``calibration.per_city.public_name`` resolves it to a public deployment. Every public file is checked against the
non-public terms before it is written.

Usage (the run in ``docs/deployment-gate-2026-11.md``)::

    D=~/Data/sidewalk-validator-ai-deployment-gate-2026-11
    A=~/Data/sidewalk-validator-ai-prior-rule-2026-11/ai_assessments_2026-11-11.csv
    X="$HOME/Data/sidewalk-validator-ai-relabel-2026-10/private-key-do-not-share/key.csv ..."   # re-label keys
    O=docs/deployment-gate-2026-11
    python analysis/deployment_gate.py gate --assessments $A --exclude-items $X --private-out $D/run \\
        --public-out $O --expect 885,834,129,199,226
    python analysis/deployment_gate.py backtest --assessments $A --exclude-items $X --private-out $D/run --public-out $O
    python analysis/deployment_gate.py power --public-out $O
    python analysis/deployment_gate.py slices --assessments $A --public-out $O \\
        --rawlabels-cache ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels \\
        --train-root ~/Data/sidewalk-validator-ai-retrain-2026-10/round3
    python analysis/deployment_gate.py doc --public-out $O --doc docs/deployment-gate-2026-11.md

Example (the status of one stratum's counts)::

    >>> r = Rule()
    >>> [status_from_counts(k, n, r)["status"] for k, n in [(45, 45), (50, 50), (49, 50), (40, 50)]]
    ['insufficient', 'pass', 'undecided', 'fail']
    >>> status_from_counts(120, 137, r)["checkpoint"]   # evaluated on the first 100 decided rows
    100
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_generation as dg  # noqa: E402
import regions  # noqa: E402
from analysis import prior_rules as pr  # noqa: E402
from analysis import slices as sl  # noqa: E402
from calibration import _isotonic_map as im  # noqa: E402
from calibration import per_city as pc  # noqa: E402

SCHEMA_VERSION = 1
SERVED_TYPES = pr.LABEL_TYPES
VERDICTS = ("Agree", "Disagree")
CLASS_FOR_VERDICT = {"Agree": "correct", "Disagree": "incorrect"}  # the LABEL class that makes the AI verdict right
REFERENCES = ("majority2", "tie_half", "any_vote")
EVIDENCE = ("production_votes", "masked_expert_round")
BASES = ("raw", "prior_weighted")
REGIMES = ("vote", "shadow", "unrecorded")
STATUSES = ("pass", "undecided", "fail", "insufficient", "no_gated", "never_votes", "not_served")
MODES = ("off", "shadow", "vote")  # SidewalkWebpage#5832's vocabulary; private decision file only
POLICIES = ("strict", "grandfather")
EXCLUDED_SCHEMAS = pc.EXCLUDED_SCHEMAS  # the internal test deployment
SINCE, BEFORE = pr.ROUND3_FROM, pr.OLDER_BEFORE
CAP_MORE_NEEDED = 25_600
MODEL_PREFIX = "sidewalk-validator-ai-"
PUBLIC_FILES = ("deployment_status.json", "backtest.json", "power.json", "slices.json", "provenance.json")
NOT_EXPORTED = "(not exported)"

# Per type: whether the API serves it, and per verdict whether it may ever vote. A type missing here is unserved.
TYPE_POLICY = {t: {"served": True, "verdicts": {"Agree": "eligible", "Disagree": "eligible"}} for t in SERVED_TYPES}
TYPE_POLICY["NoSidewalk"] = {"served": False, "verdicts": {"Agree": "eligible", "Disagree": "never"}}


@dataclass(frozen=True)
class Rule:
    """The bar a stratum must clear. ``schedule`` is the list of checkpoints (decided-row counts) at which a status is
    evaluated; past its last entry it keeps doubling."""
    gate: float = im.GATE
    min_n: int = 50
    z: float = pc.Z95
    schedule: tuple = (50, 100, 200, 400, 800, 1600)
    reference: str = "majority2"

    def __post_init__(self):
        s = tuple(int(x) for x in self.schedule)
        if not s or list(s) != sorted(set(s)) or s[0] < 1:
            raise ValueError(f"schedule must be strictly ascending positive counts, got {self.schedule}")
        if s[0] < self.min_n:
            raise ValueError(f"the first checkpoint ({s[0]}) is below min_n ({self.min_n}): a look there could never "
                             "be judged")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}")
        object.__setattr__(self, "schedule", s)


def checkpoints(rule, up_to):
    """Every checkpoint of ``rule`` not above ``up_to``, the schedule extended by doubling.

    >>> checkpoints(Rule(), 4000)
    [50, 100, 200, 400, 800, 1600, 3200]
    """
    out, c = [], list(rule.schedule)
    while c[-1] <= up_to:
        c.append(c[-1] * 2)
    return [x for x in c if x <= up_to]


def checkpoint_of(n, rule):
    """The largest checkpoint not above ``n``, or ``None`` before the first one."""
    cps = checkpoints(rule, n)
    return cps[-1] if cps else None


def wilson(k, n, z=pc.Z95):
    """``(lo, hi)`` of the Wilson interval of ``k`` right (may be fractional) in ``n``; NaNs for ``n == 0``."""
    if n == 0:
        return float("nan"), float("nan")
    lo, hi = pc.wilson_interval(k / n, n, z)
    return float(lo), float(hi)


def classify(k, n, rule):
    """``pass`` / ``fail`` / ``undecided`` for ``k`` right of ``n`` (``n >= rule.min_n``), else ``insufficient``."""
    if n < rule.min_n:
        return "insufficient"
    lo, hi = wilson(k, n, rule.z)
    return "pass" if lo >= rule.gate else ("fail" if hi < rule.gate else "undecided")


def more_needed_if_perfect(outcomes, rule):
    """The fewest further decided rows, all right, that would make the stratum pass at a checkpoint, given its decided
    ``outcomes`` in order; 0 when it already passes, ``None`` past ``CAP_MORE_NEEDED``.

    >>> more_needed_if_perfect([], Rule()), more_needed_if_perfect([1.0] * 49 + [0.0], Rule())
    (50, 50)
    """
    n = len(outcomes)
    k_all = float(np.sum(outcomes)) if n else 0.0
    c = checkpoint_of(n, rule)
    if c is not None and classify(float(np.sum(outcomes[:c])), c, rule) == "pass":
        return 0
    for cp in checkpoints(rule, CAP_MORE_NEEDED):
        if cp <= n:
            continue
        if classify(k_all + (cp - n), cp, rule) == "pass":
            return cp - n
    return None


def first_pass_checkpoint(outcomes, rule):
    """The first checkpoint at which the decided ``outcomes`` (in order) pass, or ``None``. This is the look a sticky
    rule (a pass is kept for good) would act on; :func:`ever_pass` gives its false-pass rate.

    >>> first_pass_checkpoint([1.0] * 50 + [1.0] * 42 + [0.0] * 8, Rule())   # passes at 50, not at 100
    50
    """
    o = np.asarray(outcomes, dtype=float)
    for c in checkpoints(rule, len(o)):
        if c >= rule.min_n and classify(float(o[:c].sum()), c, rule) == "pass":
            return c
    return None


def status_from_outcomes(outcomes, rule):
    """Status of one stratum from its decided outcomes in assessment order (1 right, 0 wrong, 0.5 a tie under
    ``tie_half``): ``{"n", "k", "precision_all", "ci_all_lo", "ci_all_hi", "checkpoint", "k_checkpoint", "precision",
    "ci_lo", "ci_hi", "status", "first_pass_checkpoint"}``. ``status`` and its interval are those of the first
    ``checkpoint`` rows (the largest checkpoint, re-judged at every run); ``first_pass_checkpoint`` is the first
    checkpoint that passed, which a sticky rule would keep even when ``status`` is no longer ``pass``; ``k``,
    ``precision_all`` and its interval cover all decided rows."""
    o = np.asarray(outcomes, dtype=float)
    n = len(o)
    c = checkpoint_of(n, rule) if n >= rule.min_n else None
    k = float(o.sum()) if n else 0.0
    lo_all, hi_all = wilson(k, n, rule.z) if n else (None, None)
    rec = {"n": int(n), "k": k, "precision_all": float(o.mean()) if n else None, "ci_all_lo": lo_all,
           "ci_all_hi": hi_all, "checkpoint": c, "k_checkpoint": None, "precision": None, "ci_lo": None,
           "ci_hi": None, "status": "insufficient", "first_pass_checkpoint": first_pass_checkpoint(o, rule)}
    if c is not None:
        kc = float(o[:c].sum())
        lo, hi = wilson(kc, c, rule.z)
        rec.update(k_checkpoint=kc, precision=kc / c, ci_lo=lo, ci_hi=hi, status=classify(kc, c, rule))
    return rec


def status_from_counts(k, n, rule):
    """:func:`status_from_outcomes` for ``k`` right in ``n`` with the errors last (an order-free summary for tests and
    for evidence that has no order)."""
    return status_from_outcomes([1.0] * int(k) + [0.0] * (int(n) - int(k)), rule)


# ---------------------------------------------------------------------------------------------------------------------
# Loading the pull
# ---------------------------------------------------------------------------------------------------------------------

def _truthy(s):
    return s.astype(str).str.lower().isin(["t", "true", "1"])


def read_pull(path):
    """The raw pull CSV, every column a string where it matters (no filtering)."""
    return pd.read_csv(path, dtype={"schema": str, "label_type": str, "model_id": str, "ai_result": str,
                                    "deleted": str, "validator_revision": str, "ai_vote_cast": str},
                       low_memory=False)


def load_pull(path_or_frame, since=SINCE, before=BEFORE, unrevisioned_before=None):
    """The pull (``analysis/pull_ai_assessments.sh``) as one row per AI-assessed label, with ``label_uid``
    (``schema:label_id``, asserted unique), ``era`` (``current`` / ``previous`` / ``changeover`` by date; ``current``
    / ``previous`` / ``unknown`` by revision, see :func:`assign_era`), ``model_revision`` and ``regime``. Dropped, with counts: backup schemas (``_bak_``), the internal ``sidewalk_validation`` schema,
    deleted labels, and rows whose ``model_id`` is not ``sidewalk-validator-ai-<label type>`` (a label retyped after
    its assessment). Returns ``(df, counts)``.

    ``label_id`` is per schema, so ``(schema, label_id)`` is the key; the same ``label_id`` in two schemas is two
    labels.
    """
    if before > since:
        raise ValueError(f"the previous era must end ({before}) on or before the current one starts ({since})")
    d = path_or_frame.copy() if isinstance(path_or_frame, pd.DataFrame) else read_pull(path_or_frame)
    counts = {"rows": int(len(d))}
    bak = d["schema"].str.contains("_bak_", regex=False)
    internal = d["schema"].isin(EXCLUDED_SCHEMAS)
    counts["backup_schema_rows"], counts["internal_schema_rows"] = int(bak.sum()), int(internal.sum())
    d = d[~bak & ~internal]
    deleted = _truthy(d["deleted"])
    counts["deleted_rows"] = int(deleted.sum())
    d = d[~deleted].copy()
    d["label_id"] = d["label_id"].astype(int)
    d["label_uid"] = d["schema"] + ":" + d["label_id"].astype(str)
    if not d["label_uid"].is_unique:
        raise ValueError(f"(schema, label_id) is not unique: {int(d['label_uid'].duplicated().sum())} repeats")
    ok = d["model_id"].astype(str).str.lower() == (MODEL_PREFIX + d["label_type"].astype(str).str.lower())
    counts["model_type_mismatch"] = int((~ok).sum())
    d = d[ok].copy()
    for c in ("h_agree", "h_dis", "h_unsure"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    d["ai_accuracy"] = pd.to_numeric(d["ai_accuracy"], errors="coerce")
    d["ai_date"] = pd.to_datetime(d["ai_date"]).dt.strftime("%Y-%m-%d")
    assign_era(d, since, before, counts, unrevisioned_before)
    # Regime = whether AI validation was ON for the label's deployment when it was assessed. ``ai_vote_cast`` is
    # ``label_validation_id IS NOT NULL``: in vote mode the website casts a vote for EVERY assessment, the verdict when
    # the served accuracy reaches the gate and an Unsure below it, so for a gated row ``vote`` also means the verdict
    # was counted, and for an ungated row it means an Unsure was cast. Known limit: SidewalkWebpage evolution 355
    # nulled the link of AI votes it voided (the off-target-marker repair), so those vote-mode rows read ``shadow``.
    if "ai_vote_cast" in d.columns and d["ai_vote_cast"].notna().any():
        d["regime"] = np.where(_truthy(d["ai_vote_cast"]), "vote", "shadow")
        counts["regime_source"] = "ai_vote_cast"
    else:
        d["regime"] = "unrecorded"
        counts["regime_source"] = "none (the pull has no ai_vote_cast column)"
    counts["eras"] = {k: int(v) for k, v in d["era"].value_counts().items()}
    counts["n_schemas"] = int(d["schema"].nunique())
    counts["ai_date_min"], counts["ai_date_max"] = str(d["ai_date"].min()), str(d["ai_date"].max())
    counts["rows_kept"] = int(len(d))
    return d.reset_index(drop=True), counts


def assign_era(d, since=SINCE, before=BEFORE, counts=None, unrevisioned_before=None):
    """Set ``era`` and ``model_revision`` in place. With a populated ``validator_revision`` column, each model's
    current revision is the one of its newest assessment and the era follows the revision: ``current`` for that
    revision, ``previous`` for any other recorded revision, and ``unknown`` for an EMPTY revision, which is excluded
    from both eras and counted (``unknown_revision_rows``). An empty revision is never read as the previous era: it is
    what an assessment before API 1.0.2 looks like, but also what a schema whose column is missing looks like, and the
    two cannot be told apart from the row. ``unrevisioned_before`` (a date) is the explicit exception: empty-revision
    rows assessed before it are ``previous`` (pass the date revisions began to be recorded). Without the column, the
    date split.
    """
    counts = {} if counts is None else counts
    if "validator_revision" in d.columns and d["validator_revision"].notna().any():
        rev = d["validator_revision"].astype("string").str.strip()
        rev = rev.mask(rev == "")
        has = rev.notna().to_numpy(dtype=bool)
        dated = d.assign(_rev=rev).dropna(subset=["_rev"]).sort_values(["ai_date", "label_id"], kind="mergesort")
        current = dated.groupby("model_id")["_rev"].last().to_dict()
        is_cur = (rev == d["model_id"].map(current)).fillna(False).to_numpy(dtype=bool) & has
        era = np.where(is_cur, "current", np.where(has, "previous", "unknown"))
        if unrevisioned_before:
            era = np.where(~has & (d["ai_date"].to_numpy() < unrevisioned_before), "previous", era)
        d["era"] = era
        d["model_revision"] = rev.fillna("unrecorded").astype(str)
        counts["era_key"] = "validator_revision" + (f"; empty revision before {unrevisioned_before} = previous"
                                                    if unrevisioned_before else "; empty revision = unknown (excluded)")
        counts["current_revisions"] = {str(k): str(v) for k, v in sorted(current.items())}
        counts["unknown_revision_rows"] = int((era == "unknown").sum())
    else:
        d["era"] = np.where(d["ai_date"] >= since, "current", np.where(d["ai_date"] < before, "previous", "changeover"))
        d["model_revision"] = np.where(d["era"] == "current", f"unrecorded; current era by date (ai_date >= {since})",
                                       f"unrecorded; previous era by date (ai_date < {before})")
        counts["era_key"] = f"date (no validator_revision in the pull): current >= {since}, previous < {before}"
    return d


def add_outcomes(d):
    """Per row and per reference: ``out_<ref>`` = 1 if the human reference agrees with the AI verdict, 0 if not, 0.5
    for a ``tie_half`` tie, NaN when undecided (or the verdict is Unsure)."""
    a, x = d["h_agree"].to_numpy(), d["h_dis"].to_numpy()
    verdict = d["ai_result"].astype(str).to_numpy()
    want = np.select([verdict == "Agree", verdict == "Disagree"], ["correct", "incorrect"], "")
    cls = {"majority2": dg.prior_class(a, x, "majority2")}
    one = (a + x) >= 1
    cls["any_vote"] = np.where(one & (a > x), "correct", np.where(one & (x > a), "incorrect", ""))
    for ref in ("majority2", "any_vote"):
        c = cls[ref]
        d[f"out_{ref}"] = np.where((want == "") | (c == ""), np.nan, (c == want).astype(float))
    tie = ((a + x) >= 2) & (a == x)
    d["out_tie_half"] = np.where((want != "") & tie, 0.5, d["out_majority2"])
    if {"h_agree_excl", "h_dis_excl"} <= set(d.columns):  # pulls after #52: votes of excluded validators removed
        ae = a - pd.to_numeric(d["h_agree_excl"], errors="coerce").fillna(0).astype(int).to_numpy()
        xe = x - pd.to_numeric(d["h_dis_excl"], errors="coerce").fillna(0).astype(int).to_numpy()
        c = dg.prior_class(ae, xe, "majority2")
        d["out_majority2_no_excluded"] = np.where((want == "") | (c == ""), np.nan, (c == want).astype(float))
    return d


def label_class(d, reference):
    """The human class of each LABEL (``correct`` / ``incorrect`` / ``""``) under ``reference``; ``tie_half`` uses the
    ``majority2`` class (its ties are undecided as a class)."""
    a, x = d["h_agree"].to_numpy(), d["h_dis"].to_numpy()
    if reference == "any_vote":
        one = (a + x) >= 1
        return np.where(one & (a > x), "correct", np.where(one & (x > a), "incorrect", ""))
    return dg.prior_class(a, x, "majority2")


def item_exclusions(paths, d, cities, pub=None):
    """``label_uid`` (schema form) of every item in the masked re-label round keys at ``paths``. A key has ``schema``
    and ``label_id`` columns, or a ``label_uid`` in either the schema form (``sidewalk_x:12``) or the API form
    (``city_id:12``), which is mapped to its schema through the listing (:func:`schema_city_map`). Returns
    ``(set, per_file_counts)``; the counts are of rows and of matched pull rows (all, and in the ``pub`` schemas), no
    ids."""
    to_schema = {cid: s for s, cid in schema_city_map(d["schema"].unique(), cities).items() if cid}
    uids, per_file = set(), []
    for p in paths or []:
        k = pd.read_csv(p, dtype=str)
        if {"schema", "label_id"} <= set(k.columns):
            u = k["schema"] + ":" + k["label_id"].astype(int).astype(str)
        else:
            parts = k["label_uid"].str.rsplit(":", n=1, expand=True)
            sch = parts[0].where(parts[0].str.startswith("sidewalk_"), parts[0].map(to_schema))
            u = (sch + ":" + parts[1]).dropna()
        u = set(u)
        hit = d["label_uid"].isin(u)
        per_file.append({"file": _short_path(p), "sha256": pr.sha256_of(p), "items": len(u),
                         "in_pull": int(hit.sum()),
                         "in_pull_public": int((hit & d["schema"].isin(pub or {})).sum())})
        uids |= u
    return uids, per_file


# ---------------------------------------------------------------------------------------------------------------------
# Naming and privacy
# ---------------------------------------------------------------------------------------------------------------------

def _short_path(p):
    """A private input's path for provenance: relative to ``~/Data`` when under it, else the basename."""
    a = os.path.abspath(os.path.expanduser(p))
    root = os.path.join(os.path.expanduser("~"), "Data") + os.sep
    return a[len(root):] if a.startswith(root) else os.path.basename(a)


def listing_cities(listing):
    """``{city_id: {"visibility", "country_id"}}`` from a ``/v3/api/cities`` response."""
    return {str(c["city_id"]): {"visibility": c.get("visibility"), "country_id": c.get("country_id")}
            for c in regions._entries(listing)}


def schema_city_map(schemas, cities):
    """``{schema: city_id or None}``: the listed ``city_id`` each schema's city part identifies
    (``per_city.match_city_id``), the bridge between the pull's ``schema:label_id`` and the ``city_id:label_id`` that
    rawLabels and ``split.csv`` use. Raises if two schemas identify the same ``city_id``.

    >>> schema_city_map(["sidewalk_seattle", "sidewalk_x"], {"seattle-wa": "public"})
    {'sidewalk_seattle': 'seattle-wa', 'sidewalk_x': None}
    """
    out = {s: pc.match_city_id(s.replace("sidewalk_", "", 1), cities) for s in sorted(set(schemas))}
    ids = [v for v in out.values() if v]
    if len(ids) != len(set(ids)):
        raise ValueError("two schemas identify the same city_id; the schema -> city_id mapping is not one to one")
    return out


def public_schemas(schemas, cities):
    """``{schema: public city_id}`` for the schemas ``per_city.public_name`` names; every other schema is private."""
    return {s: n for s in sorted(set(schemas)) if (n := pc.public_name(s.replace("sidewalk_", "", 1), cities))}


def private_terms(listing, schemas=(), extra=()):
    """Every string that must not appear in public text: ``regions.non_public_terms`` (non-public ``city_id``, display
    names, non-public-only countries), the hostname of every non-public deployment's url, every schema of the pull
    that is not a public deployment (the internal ``sidewalk_validation`` schema aside) and its city part (both
    spellings), and ``extra``."""
    terms = set(regions.non_public_terms(listing))
    for c in regions._entries(listing):
        if c.get("visibility") != "public" and c.get("url"):
            host = re.sub(r"^https?://", "", str(c["url"])).split("/")[0]
            if host:
                terms.add(host)
    pub = public_schemas(schemas, listing_cities(listing))
    for s in schemas:
        if s not in pub and s not in EXCLUDED_SCHEMAS:  # the internal test schema is not a deployment
            part = s.replace("sidewalk_", "", 1)
            terms |= {s, part, part.replace("_", "-"), part.replace("_", " ")}
    terms |= {t.strip() for t in extra if t and t.strip()}
    return {t for t in terms if t}


def _fold(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).lower()


def find_private_terms(text, terms):
    """Character offsets (in the folded text) of every word-bounded, case- and accent-insensitive hit of a term.
    Never returns the term itself, so the result can be printed.

    >>> find_private_terms("Seattle and Mönchtal", {"monchtal"}), find_private_terms("a monchtalx b", {"monchtal"})
    ([12], [])
    """
    t = _fold(text)
    hits = []
    for term in terms:
        ft = _fold(term)
        if not ft:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(ft) + r"(?![a-z0-9])", t):
            hits.append(m.start())
    return sorted(hits)


class PrivacyError(RuntimeError):
    pass


def assert_public(text, terms, where):
    """Raise :class:`PrivacyError` naming ``where`` and the offsets (not the terms) when ``text`` holds a term."""
    hits = find_private_terms(text, terms)
    if hits:
        raise PrivacyError(f"{where}: {len(hits)} non-public term hit(s) at offset(s) {hits[:10]}; nothing written")


def write_public_json(obj, path, terms):
    """Write ``obj`` as JSON after checking its text against ``terms``."""
    text = json.dumps(_clean(obj), indent=2) + "\n"
    assert_public(text, terms, os.path.basename(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _clean(obj):
    """JSON-safe copy: NaN to None, numpy scalars to Python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def fetch_listing(fetch=None):
    """The live ``/v3/api/cities`` listing (``data_generation.fetch_city_listing``); raises on failure."""
    return dg.fetch_city_listing(fetch=fetch)


# ---------------------------------------------------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------------------------------------------------

def type_policy(label_type):
    return TYPE_POLICY.get(label_type, {"served": False, "verdicts": {"Agree": "eligible", "Disagree": "eligible"}})


def types_block(label_types):
    """The report's ``types`` block: every policy type and every type seen, with ``served`` and per-verdict
    ``eligible`` / ``never``. An unserved type reads ``off`` on the website, not ``shadow``: nothing scores it."""
    return {t: type_policy(t) for t in sorted(set(TYPE_POLICY) | set(label_types))}


def stratum_record(g, rule, evidence_ref, base, label_type, verdict):
    """One production-evidence stratum from its rows ``g`` (one schema, type, verdict, regime, era), ``base`` being
    that schema and type's decided rows of the era (any verdict) for the base rate."""
    pol = type_policy(label_type)
    gated = g[g["ai_accuracy"] >= rule.gate].sort_values(["ai_date", "label_id"], kind="mergesort")
    ref = rule.reference
    dec = gated[gated[f"out_{ref}"].notna()]
    rec = {"evidence": "production_votes", "evidence_ref": evidence_ref, "basis": "raw", "reference": ref,
           "min_n": rule.min_n, "model_revision": "; ".join(sorted(g["model_revision"].astype(str).unique())),
           "n_assessed": int(len(g)), "n_gated": int(len(gated))}
    rec.update(status_from_outcomes(dec[f"out_{ref}"].to_numpy(), rule))
    if pol["verdicts"].get(verdict) == "never":
        rec["status"] = "never_votes"
    elif not pol["served"]:
        rec["status"] = "not_served"
    elif len(gated) == 0:
        rec["status"] = "no_gated"
    # None where no number of decided rows can help: the verdict never votes, the type is not served, or no
    # assessment reaches the gate (no_gated).
    rec["more_needed_if_perfect"] = (None if rec["status"] in ("never_votes", "not_served", "no_gated")
                                     else more_needed_if_perfect(dec[f"out_{ref}"].to_numpy(), rule))
    rec["sensitivity"] = {}
    for other in REFERENCES + ("majority2_no_excluded",):
        if other == ref or f"out_{other}" not in g.columns:
            continue
        o = gated[gated[f"out_{other}"].notna()][f"out_{other}"].to_numpy()
        s = status_from_outcomes(o, rule)
        if rec["status"] in ("no_gated", "never_votes", "not_served"):
            s["status"] = rec["status"]
        rec["sensitivity"][other] = {k: s[k] for k in ("n", "k", "checkpoint", "precision", "ci_lo", "ci_hi", "status")}
    want = CLASS_FOR_VERDICT[verdict]
    cls = base["_class"]
    nb = int(cls.isin(["correct", "incorrect"]).sum())
    rec["base_rate"] = float((cls == want).sum() / nb) if nb else None
    rec["base_n"] = nb
    rec["lift"] = (rec["precision_all"] - rec["base_rate"]) if (rec["precision_all"] is not None and nb) else None
    return rec


def compute_strata(d, rule, evidence_ref):
    """``{(schema, type, verdict, regime): record}`` for the current era, plus the previous era's status per
    ``(schema, type, verdict, regime)`` as context (``previous``) and ``existing`` (any previous-era assessment of
    that schema and type)."""
    d = d[d["ai_result"].isin(VERDICTS)]
    d = d.assign(_class=label_class(d, rule.reference))
    out = {}
    prev_keys = set(map(tuple, d[d["era"] == "previous"][["schema", "label_type"]].drop_duplicates().to_numpy()))
    eras = {}
    for era in ("current", "previous"):
        e = d[d["era"] == era]
        bases = {k: g for k, g in e.groupby(["schema", "label_type"])}
        eras[era] = {}
        for (schema, t, v, regime), g in e.groupby(["schema", "label_type", "ai_result", "regime"], sort=True):
            eras[era][(schema, t, v, regime)] = stratum_record(g, rule, evidence_ref, bases[(schema, t)], t, v)
    for key, rec in eras["current"].items():
        p = eras["previous"].get(key)
        rec["previous_era"] = None if p is None else {k: p[k] for k in ("n", "checkpoint", "precision", "ci_lo",
                                                                        "ci_hi", "status")}
        rec["existing"] = (key[0], key[1]) in prev_keys
        out[key] = rec
    return out, eras["previous"]


def recommended_mode(rec, label_type, verdict, policy):
    """What a stratum's mode would be under ``policy`` (private decision file only; nothing applies it):
    ``strict`` votes only on ``pass``; ``grandfather`` also keeps an existing deployment voting unless it ``fail``s.
    Unserved types and never-vote verdicts are ``off``."""
    pol = type_policy(label_type)
    if not pol["served"] or pol["verdicts"].get(verdict) == "never":
        return "off"
    if rec["status"] == "pass":
        return "vote"
    if policy == "grandfather" and rec.get("existing") and rec["status"] != "fail":
        return "vote"
    return "shadow"


def acceptance_counts(d, rule):
    """Pooled current-era decided gated Agree rows per served type under ``majority2`` and the share confirmed: the
    numbers ``docs/deployment-prior-2026-11/observed.md`` reports for the same pull (885 / 834 / 129 / 199 / 226)."""
    g = d[(d["era"] == "current") & (d["ai_result"] == "Agree") & (d["ai_accuracy"] >= rule.gate)]
    out = {}
    for t in SERVED_TYPES:
        o = g.loc[g["label_type"] == t, "out_majority2"].dropna()
        out[t] = {"decided": int(len(o)), "confirmed": int(o.sum()),
                  "share_confirmed": float(o.mean()) if len(o) else None}
    return out


PRIVATE_NOTE = ("Non-public deployments are left out of every figure in this file, pooled figures included. Their "
                "strata, and every figure that includes them, are only in the private file kept outside the "
                "repository.")


def public_private_block():
    """The public report's ``private`` block. Its keys and text are FIXED: nothing in it depends on which non-public
    deployments exist, which types they were assessed on, or how they fared."""
    return {"note": PRIVATE_NOTE}


def private_cells(strata_priv):
    """Pooled counts for the non-public strata per type x verdict (PRIVATE file only): contributing deployments,
    strata by status and decided rows."""
    cells = {}
    for (schema, t, v, regime), rec in strata_priv.items():
        c = cells.setdefault(t, {}).setdefault(v, {"schemas": set(), "by_status": {}, "n": 0})
        c["schemas"].add(schema)
        c["by_status"][rec["status"]] = c["by_status"].get(rec["status"], 0) + 1
        c["n"] += rec["n"]
    return {t: {v: {"n_deployments": len(c["schemas"]), "strata_by_status": dict(sorted(c["by_status"].items())),
                    "n_decided": c["n"]} for v, c in sorted(by_v.items())} for t, by_v in sorted(cells.items())}


def status_summary(strata):
    """Strata by status per type x verdict, over the given strata."""
    out = {}
    for (_, t, v, _), rec in strata.items():
        c = out.setdefault(t, {}).setdefault(v, {})
        c[rec["status"]] = c.get(rec["status"], 0) + 1
    return {t: {v: dict(sorted(c.items())) for v, c in sorted(bv.items())} for t, bv in sorted(out.items())}


def pooled_summary(d, strata, rule, names):
    """``summary`` over ``strata`` and the rows of ``d`` (both already restricted to the deployments to pool):
    strata by status, status changes under each sensitivity reference, the largest one-deployment share of the
    decided gated rows (named by ``names`` when listed there) and the acceptance counts."""
    sens = {}
    for rec in strata.values():
        for other, srec in rec["sensitivity"].items():
            sens[other] = sens.get(other, 0) + int(srec["status"] != rec["status"])
    largest = None
    cur = d[(d["era"] == "current") & d["ai_result"].isin(VERDICTS) & (d["ai_accuracy"] >= rule.gate)]
    cur_dec = cur[cur[f"out_{rule.reference}"].notna()]
    if len(cur_dec):
        share = cur_dec["schema"].value_counts(normalize=True)
        top = share.index[0]
        largest = {"share": float(share.iloc[0]), "deployment": names.get(top, top),
                   "n_deployments_with_decided": int(cur_dec["schema"].nunique())}
    return {"strata": len(strata), "by_status": status_summary(strata), "status_differs_under": sens,
            "largest_deployment_share_of_decided_gated": largest, "acceptance": acceptance_counts(d, rule)}


def build_report(d, counts, listing, rule, evidence_ref, pull_meta, exclusions_meta):
    """``(public_report, private_report, strata)`` for the current era. The public report has no ``mode`` anywhere,
    and every figure in it, pooled ones included, is computed over PUBLIC deployments only, so no public number minus
    the listed public strata leaves anything about a non-public deployment. Pooled figures that include non-public
    deployments go to the private report (``summary_all``, ``private_cells``)."""
    cities = listing_cities(listing)
    pub = public_schemas(d["schema"].unique(), cities)
    strata, _ = compute_strata(d, rule, evidence_ref)
    deployments, priv, pub_strata = {}, {}, {}
    for key, rec in strata.items():
        schema, t, v, regime = key
        if schema in pub:
            deployments.setdefault(pub[schema], {}).setdefault(t, {}).setdefault(v, {})[regime] = rec
            pub_strata[key] = rec
        else:
            priv[key] = rec
    d_pub = d[d["schema"].isin(pub)]
    common = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.date.today().isoformat(),
        "pull": pull_meta,
        "exclusions": exclusions_meta,
        "rule": {**asdict(rule), "schedule": list(rule.schedule), "served": "ai_accuracy",
                 "decided": {"majority2": ">= 2 human Agree/Disagree votes, strict majority, ties out",
                             "tie_half": ">= 2 votes, a tie counts half right",
                             "any_vote": ">= 1 human Agree/Disagree vote, strict majority"}[rule.reference],
                 "status_at": "the largest checkpoint not above n, re-judged at every run; first_pass_checkpoint "
                              "is the first checkpoint that passed (what a sticky rule would keep)",
                 "era_key": counts["era_key"], "regime_source": counts["regime_source"]},
    }
    public = {
        "kind": "deployment_status_report",
        "note": ("A status report: per public deployment, type, verdict and evidence regime, how the AI's gated "
                 "assessments fared against the human reference in the current model era. It carries no serving "
                 "decision. Every figure, pooled ones included, covers public deployments only."),
        **common,
        "types": types_block(d_pub["label_type"].unique()),
        "deployments": {c: deployments[c] for c in sorted(deployments)},
        "private": public_private_block(),
        "summary": pooled_summary(d_pub, pub_strata, rule, pub),
    }
    rows = []
    decisions = {p: {} for p in POLICIES}
    for key, rec in sorted(strata.items()):
        schema, t, v, regime = key
        row = {"schema": schema, "city_id": schema_city_map([schema], cities)[schema], "public": schema in pub,
               "label_type": t, "verdict": v, "regime": regime,
               **{k: rec[k] for k in ("status", "first_pass_checkpoint", "n_assessed", "n_gated", "n", "k",
                                      "checkpoint", "precision", "ci_lo", "ci_hi", "precision_all", "ci_all_lo",
                                      "ci_all_hi", "more_needed_if_perfect", "base_rate", "lift", "existing",
                                      "model_revision")},
               **{f"status_{o}": s["status"] for o, s in rec["sensitivity"].items()},
               "previous_status": (rec["previous_era"] or {}).get("status")}
        for p in POLICIES:
            row[f"mode_{p}"] = recommended_mode(rec, t, v, p)
            decisions[p].setdefault(schema, {}).setdefault(t, {})[v] = row[f"mode_{p}"]
        rows.append(row)
    private = {"kind": "deployment_status_private", **common, "types": types_block(d["label_type"].unique()),
               "note": ("PRIVATE: every deployment by schema. The modes are what each policy WOULD give, for "
                        "comparison only; a person applies any change to the website's city-params. Absent "
                        "deployment or type = no evidence; an unserved type is off, not shadow."),
               "policies": {"strict": "vote only on pass (status re-judged at the largest checkpoint)",
                            "grandfather": "existing deployments keep voting unless fail; new ones as strict"},
               "summary_all": pooled_summary(d, strata, rule, pub),
               "private_cells": private_cells(priv),
               "decisions": decisions, "strata": rows}
    return public, private, strata


def expert_round_stratum(k, n, rule, evidence_ref, model_revision, label_type, verdict, basis="raw", min_n=None):
    """A stratum from a masked expert round (NoSidewalk step (c)): ``k`` right of ``n`` decided ratings, evaluated
    ONCE on all rows (a round is one look, so no checkpoint), at ``min_n`` (default ``rule.min_n``). ``basis`` says
    whether ``k`` is a raw count or a prior-weighted one (then pass the effective counts).

    >>> r = expert_round_stratum(51, 51, Rule(), "docs/x.json@abc", "fold ensemble", "NoSidewalk", "Agree")
    >>> r["status"], r["evidence"]
    ('pass', 'masked_expert_round')
    >>> expert_round_stratum(0, 0, Rule(), "x", "y", "NoSidewalk", "Disagree")["status"]
    'never_votes'
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    m = rule.min_n if min_n is None else int(min_n)
    lo, hi = wilson(k, n, rule.z)
    status = "insufficient" if n < m else ("pass" if lo >= rule.gate else ("fail" if hi < rule.gate else "undecided"))
    if type_policy(label_type)["verdicts"].get(verdict) == "never":
        status = "never_votes"
    return {"evidence": "masked_expert_round", "evidence_ref": evidence_ref, "basis": basis, "reference": "expert",
            "min_n": m, "model_revision": model_revision, "n_assessed": None, "n_gated": None, "n": n, "k": k,
            "precision_all": (k / n) if n else None, "ci_all_lo": lo if n else None, "ci_all_hi": hi if n else None,
            "first_pass_checkpoint": None, "checkpoint": None, "k_checkpoint": None,
            "precision": (k / n) if n else None, "ci_lo": lo if n else None, "ci_hi": hi if n else None,
            "status": status, "more_needed_if_perfect": None, "sensitivity": {}, "base_rate": None, "base_n": 0,
            "lift": None, "previous_era": None, "existing": None}


def add_stratum(report, city_id, label_type, verdict, regime, rec):
    """Put ``rec`` into ``report["deployments"]`` (the way step (c) adds its masked-round strata); the regime key of an
    expert round is ``"expert_round"``. Adds the type to ``types`` if new."""
    report["deployments"].setdefault(city_id, {}).setdefault(label_type, {}).setdefault(verdict, {})[regime] = rec
    report["types"].setdefault(label_type, type_policy(label_type))
    return report


REQUIRED_STRATUM = ("evidence", "evidence_ref", "basis", "min_n", "model_revision", "status", "n", "k", "precision",
                    "ci_lo", "ci_hi")


def validate_report(report, gate=None):
    """Raise ``ValueError`` unless ``report`` is a well-formed public status report: required keys, known vocabulary,
    no ``mode`` key anywhere, every ``pass`` with n at or above its ``min_n`` and a lower bound at the gate, and a
    ``types`` entry for every type."""
    for k in ("schema_version", "kind", "rule", "types", "deployments", "private", "summary"):
        if k not in report:
            raise ValueError(f"missing {k}")
    if re.search(r'"(mode|recommended_mode|mode_[a-z_]+)"\s*:', json.dumps(report)):
        raise ValueError("a public status report must not carry a serving mode")
    g = report["rule"]["gate"] if gate is None else gate
    for city, by_t in report["deployments"].items():
        for t, by_v in by_t.items():
            if t not in report["types"]:
                raise ValueError(f"type {t} has no types entry")
            for v, by_r in by_v.items():
                if v not in VERDICTS:
                    raise ValueError(f"unknown verdict {v}")
                for regime, rec in by_r.items():
                    for k in REQUIRED_STRATUM:
                        if k not in rec:
                            raise ValueError(f"{city}/{t}/{v}/{regime}: missing {k}")
                    if rec["evidence"] not in EVIDENCE or rec["basis"] not in BASES or rec["status"] not in STATUSES:
                        raise ValueError(f"{city}/{t}/{v}/{regime}: bad evidence, basis or status")
                    if rec["status"] == "pass":
                        n_eval = rec["checkpoint"] if rec.get("checkpoint") else rec["n"]
                        if n_eval < rec["min_n"] or rec["ci_lo"] < g:
                            raise ValueError(f"{city}/{t}/{v}/{regime}: pass without the bar")
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Back-test
# ---------------------------------------------------------------------------------------------------------------------

def replay_stratum(outcomes, rule):
    """Sequential replay of one stratum's decided outcomes (in assessment order) at the rule's checkpoints:
    ``{"n", "graduated_at" (checkpoint or None), "final_status", "withheld": (k, n), "kept": (k, n)}``. A stratum
    graduates at its first ``pass`` and stays graduated (the sticky rule). Withheld = rows up to and including the
    graduation checkpoint (all rows if it never graduates); kept = the rest."""
    o = np.asarray(outcomes, dtype=float)
    n = len(o)
    grad = first_pass_checkpoint(o, rule)
    cut = n if grad is None else grad
    return {"n": n, "graduated_at": grad, "final_status": status_from_outcomes(o, rule)["status"],
            "withheld": (float(o[:cut].sum()), int(cut)), "kept": (float(o[cut:].sum()), int(n - cut))}


def _prec(k, n, z):
    lo, hi = wilson(k, n, z)
    return {"n": int(n), "k": float(k), "precision": (k / n) if n else None, "ci_lo": lo if n else None,
            "ci_hi": hi if n else None}


def _pool_replays(replays, rule):
    """Pool ``[(label_type, verdict, regime, replay, total_k)]`` into ``{regime: {type: {verdict: cell}}}``."""
    pooled = {}
    for t, v, regime, r, k_all in replays:
        p = pooled.setdefault(regime, {}).setdefault(t, {}).setdefault(v, {
            "strata": 0, "strata_ge_min_n": 0, "graduated": 0, "grad_n": [], "final": {}, "withheld": [0.0, 0],
            "kept": [0.0, 0], "never": [0.0, 0], "small": [0.0, 0], "total": [0.0, 0]})
        p["strata"] += 1
        p["total"][0] += k_all
        p["total"][1] += r["n"]
        if r["n"] < rule.min_n:
            p["small"][0] += r["withheld"][0]
            p["small"][1] += r["n"]
            continue
        p["strata_ge_min_n"] += 1
        p["final"][r["final_status"]] = p["final"].get(r["final_status"], 0) + 1
        slot = "never"
        if r["graduated_at"]:
            p["graduated"] += 1
            p["grad_n"].append(r["graduated_at"])
            p["kept"][0] += r["kept"][0]
            p["kept"][1] += r["kept"][1]
            slot = "withheld"
        p[slot][0] += r["withheld"][0]
        p[slot][1] += r["withheld"][1]
    out = {}
    for regime, by_t in sorted(pooled.items()):
        for t, by_v in sorted(by_t.items()):
            for v, p in sorted(by_v.items()):
                gn = sorted(p["grad_n"])
                out.setdefault(regime, {}).setdefault(t, {})[v] = {
                    "strata": p["strata"], "strata_ge_min_n": p["strata_ge_min_n"], "graduated": p["graduated"],
                    "graduation_n": {"median": float(np.median(gn)) if gn else None, "min": gn[0] if gn else None,
                                     "max": gn[-1] if gn else None,
                                     "counts": {str(c): gn.count(c) for c in sorted(set(gn))}},
                    "final_status": dict(sorted(p["final"].items())),
                    "withheld_in_graduated": _prec(*p["withheld"], rule.z), "kept": _prec(*p["kept"], rule.z),
                    "never_graduated": _prec(*p["never"], rule.z), "strata_under_min_n": _prec(*p["small"], rule.z),
                    "all": _prec(*p["total"], rule.z)}
    return out


def backtest(d, listing, rule):
    """Replay the rule on the previous era (served types, Agree and Disagree), one stratum per deployment, type,
    verdict and evidence REGIME, so vote-mode and shadow-mode evidence are never pooled. Returns
    ``(public, private)``: the public part pools and lists PUBLIC deployments only; the private part has every
    stratum by schema and the pooled cells over all deployments."""
    cities = listing_cities(listing)
    pub = public_schemas(d["schema"].unique(), cities)
    e = d[(d["era"] == "previous") & d["ai_result"].isin(VERDICTS) & d["label_type"].isin(SERVED_TYPES)
          & (d["ai_accuracy"] >= rule.gate)]
    ref = f"out_{rule.reference}"
    e = e[e[ref].notna()].sort_values(["schema", "ai_date", "label_id"], kind="mergesort")
    per_dep, private_rows, replays_pub, replays_all = {}, [], [], []
    for (schema, t, v, regime), g in e.groupby(["schema", "label_type", "ai_result", "regime"], sort=True):
        r = replay_stratum(g[ref].to_numpy(), rule)
        item = (t, v, regime, r, float(g[ref].sum()))
        replays_all.append(item)
        row = {"label_type": t, "verdict": v, "regime": regime, "n": r["n"], "graduated_at": r["graduated_at"],
               "final_status": r["final_status"], "kept": _prec(*r["kept"], rule.z)}
        if schema in pub:
            replays_pub.append(item)
            per_dep.setdefault(pub[schema], {}).setdefault(t, {}).setdefault(v, {})[regime] = row
        private_rows.append({"schema": schema, **row})
    public = {"kind": "backtest",
              "label": "votes as of the end of the pull, public deployments only; count-based replay",
              "rule": {**asdict(rule), "schedule": list(rule.schedule)},
              "pooled": _pool_replays(replays_pub, rule),
              "public_deployments": {c: per_dep[c] for c in sorted(per_dep)},
              "private": public_private_block(),
              "limits": [
                  "Whether a row is decided, and its majority, uses votes as of the pull's end: an early checkpoint "
                  "here sees votes that did not exist yet. Contested labels are decided later, so this understates "
                  "early false graduation.",
                  "Strata are split by evidence regime (vote / shadow / unrecorded) and never pooled across it. The "
                  "2026-11-11 pull has no ai_vote_cast column, so every row is unrecorded; its previous era was "
                  "collected in vote mode, so this says nothing about how evidence accrues in shadow mode.",
                  "A stratum graduates at its first pass and stays graduated (the sticky rule).",
                  "No dates and no waiting times: the pull has no vote timestamps.",
                  "The previous era mixes model revisions (releases before API 1.0.2 loaded the hub's main branch)."]}
    return public, {"kind": "backtest_private", "pooled_all": _pool_replays(replays_all, rule),
                    "strata": private_rows}


# ---------------------------------------------------------------------------------------------------------------------
# Power (exact)
# ---------------------------------------------------------------------------------------------------------------------

def pass_threshold(n, rule):
    """The fewest right of ``n`` that pass (Wilson lower >= gate), or ``None``.

    >>> [pass_threshold(n, Rule()) for n in (45, 67, 87, 106)]
    [45, 66, 85, 103]
    """
    k = np.arange(n + 1)
    ok = im.wilson_lower(k / n, n, rule.z) >= rule.gate
    return int(k[ok][0]) if ok.any() else None


def binom_pmf(n, p):
    """The Binomial(n, p) probabilities of 0..n (no scipy)."""
    k = np.arange(n + 1)
    if p <= 0:
        return (k == 0).astype(float)
    if p >= 1:
        return (k == n).astype(float)
    lg = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    return np.exp(lg + k * math.log(p) + (n - k) * math.log1p(-p))


def single_look_pass(p, n, rule):
    """P(pass at one look on ``n`` rows | true precision ``p``)."""
    t = pass_threshold(n, rule)
    return 0.0 if t is None else float(binom_pmf(n, p)[t:].sum())


def ever_pass(p, looks, rule):
    """P(pass at any of ``looks`` (ascending row counts) | true precision ``p``), exactly: the distribution of the
    number right is carried from look to look, and the mass that passes at a look is absorbed.

    >>> round(ever_pass(0.92, [50], Rule()), 4) == round(single_look_pass(0.92, 50, Rule()), 4)
    True
    """
    looks = [int(x) for x in looks if x >= rule.min_n]
    if not looks:
        return 0.0
    state = np.array([1.0])  # P(k right so far, not yet passed), k = 0..n
    n_prev, passed = 0, 0.0
    for n in looks:
        state = np.convolve(state, binom_pmf(n - n_prev, p))
        t = pass_threshold(n, rule)
        if t is not None:
            passed += float(state[t:].sum())
            state[t:] = 0.0
        n_prev = n
    return passed


def look_schedules(rule, up_to):
    return {"checkpoints": checkpoints(rule, up_to),
            "every 50": list(range(rule.min_n, up_to + 1, 50)),
            "every 10": list(range(rule.min_n, up_to + 1, 10)),
            "every row": list(range(rule.min_n, up_to + 1))}


def z_for(rule, looks, target=0.05, grid=None):
    """The smallest ``z`` on ``grid`` whose P(ever pass | p = gate) under ``looks`` is at most ``target``."""
    grid = np.round(np.arange(1.96, 4.01, 0.01), 2) if grid is None else grid
    for z in grid:
        r = Rule(rule.gate, rule.min_n, float(z), rule.schedule, rule.reference)
        if ever_pass(rule.gate, looks, r) <= target:
            return float(z)
    return None


def power_tables(rule, up_to=2000, ps=(0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0),
                 ns=(50, 67, 87, 100, 200, 400, 800, 1600)):
    allowed = {str(n): (None if pass_threshold(n, rule) is None else n - pass_threshold(n, rule)) for n in ns}
    single = {str(p): {str(n): single_look_pass(p, n, rule) for n in ns} for p in ps}
    scheds = look_schedules(rule, up_to)
    ever = {name: {str(p): ever_pass(p, looks, rule) for p in ps if p <= 0.96} for name, looks in scheds.items()}
    max_single = max(single_look_pass(rule.gate, n, rule) for n in range(rule.min_n, up_to + 1))
    longer = {str(u): ever_pass(rule.gate, checkpoints(rule, u), rule) for u in (6400, 25600) if u > up_to}
    return {"kind": "power", "rule": {**asdict(rule), "schedule": list(rule.schedule)}, "up_to": up_to,
            "notes": [
                "ever_pass is the chance of passing at ANY look, i.e. of a sticky rule (a pass is kept). A rule that "
                "re-judges each stratum at its current checkpoint can lose a pass later; ever_pass bounds how often "
                "it passes at some point, not how often it is passing at the end.",
                "The rates assume NESTED looks: the rows judged at one checkpoint are the first rows of those judged "
                "at the next. Real re-runs are not nested. The rule takes the first C decided rows ordered by "
                "(ai_date, label_id) within a deployment, which is deterministic for one pull, but a label assessed "
                "early and decided late (contested labels get extra votes) enters that prefix in a later pull and "
                "pushes another row out. Over real re-runs the false-pass rate is therefore at least the stated "
                "rate, not exactly it: each figure here is a lower bound.",
                f"ever_pass is by n = {up_to:,}; the checkpoints keep doubling past it and the rate keeps growing "
                "(ever_pass_checkpoints_at_gate_longer)."],
            "ever_pass_checkpoints_at_gate_longer": longer,
            "errors_allowed": allowed, "single_look_pass": single, "max_single_look_pass_at_gate": max_single,
            "ever_pass": ever, "schedules": {k: (v if len(v) < 20 else f"{v[0]}, {v[1]}, ... {v[-1]}")
                                             for k, v in scheds.items()},
            "z_for_5pct": {name: z_for(rule, looks) for name, looks in scheds.items()}}


# ---------------------------------------------------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------------------------------------------------

def join_tags(d, cache_dir, cities):
    """``d`` with ``tags`` from the cached public rawLabels (``<cache_dir>/<city_id>.csv``), joined on the API key
    ``label_uid_city = city_id:label_id``, one to one. Only public deployments are joined; every other row, and every
    public row without an export, gets ``exported = False``. Returns ``(df, coverage)``."""
    smap = schema_city_map(d["schema"].unique(), cities)
    pub = public_schemas(d["schema"].unique(), cities)
    d = d.copy()
    d["city_id"] = d["schema"].map(smap)
    d["label_uid_city"] = d["city_id"] + ":" + d["label_id"].astype(str)
    frames = []
    for s, cid in sorted(pub.items()):
        path = os.path.join(cache_dir, f"{cid}.csv")
        if os.path.exists(path):
            r = pd.read_csv(path, usecols=["label_id", "tags"], dtype={"tags": str})
            frames.append(pd.DataFrame({"label_uid_city": cid + ":" + r["label_id"].astype(int).astype(str),
                                        "tags": r["tags"]}))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["label_uid_city", "tags"])
    if not raw["label_uid_city"].is_unique:
        raise ValueError("rawLabels cache: label_uid_city is not unique")
    sub = d[d["schema"].isin(pub)]
    if not sub["label_uid_city"].is_unique:
        raise ValueError("pull: label_uid_city is not unique")
    m = sub[["label_uid"]].join(sub["label_uid_city"]).merge(raw, on="label_uid_city", how="left",
                                                              validate="one_to_one", indicator=True)
    d = d.merge(m[["label_uid", "tags", "_merge"]], on="label_uid", how="left", validate="one_to_one")
    d["exported"] = (d["_merge"] == "both").to_numpy(dtype=bool)
    d = d.drop(columns="_merge")
    cov = {"rows": int(len(d)), "public_rows": int(len(sub)), "exported": int(d["exported"].sum())}
    return d, cov


def confirmed_slices(g, by, ref, min_n=sl.MIN_N, train=None):
    """Per slice of the gated Agree rows ``g``: decided n, share confirmed (blank under ``min_n``), the largest
    one-deployment share and, with ``train`` (``split.csv`` train rows), the training rows by class."""
    if by == "tag" and "exported" in g.columns:
        exp = g[g["exported"]]
        d = sl.with_slice(exp, "tag")
        d = pd.concat([d, g[~g["exported"]].assign(slice=NOT_EXPORTED)], ignore_index=True)
    else:
        d = sl.with_slice(g if by in g.columns or by == "tag" else g.assign(severity=None), by)
    tr = sl.with_slice(train, by).groupby(["slice", "label_class"]).size() if train is not None else None
    rows = []
    for name, s in d.groupby("slice"):
        o = s[ref].dropna()
        lo, hi = wilson(float(o.sum()), len(o))
        row = {"slice": name, "n": int(len(o)),
               "confirmed": float(o.mean()) if len(o) >= min_n else None,
               "ci_lo": lo if len(o) >= min_n else None, "ci_hi": hi if len(o) >= min_n else None,
               "max_deployment_share": float(s.loc[o.index, "schema"].value_counts(normalize=True).iloc[0])
               if len(o) else None}
        if tr is not None:
            row["train_correct"] = int(tr.get((name, "correct"), 0))
            row["train_incorrect"] = int(tr.get((name, "incorrect"), 0))
        rows.append(row)
    if not rows:
        return []
    out = pd.DataFrame(rows)
    return sl._order(out, by).to_dict("records")


def slices_report(d, listing, rule, cache_dir=None, train_root=None, public_only=True):
    """Type x severity and type x tag tables of the gated Agree rows, current era against the previous one, and the
    severity mix inside each public pass or undecided stratum. With ``public_only`` (the public file) every table
    pools PUBLIC deployments only; ``public_only=False`` pools every deployment, for the private file."""
    cities = listing_cities(listing)
    ref = f"out_{rule.reference}"
    pub = public_schemas(d["schema"].unique(), cities)
    if public_only:
        d = d[d["schema"].isin(pub)]
    cov = None
    if cache_dir:
        d, cov = join_tags(d, cache_dir, cities)
        if public_only:
            cov = {k: cov[k] for k in ("public_rows", "exported")}
    g = d[(d["ai_result"] == "Agree") & (d["ai_accuracy"] >= rule.gate) & d["label_type"].isin(SERVED_TYPES)]
    g = g[g[ref].notna()]
    out = {"kind": "slices", "reference": rule.reference, "min_n": sl.MIN_N,
           "deployments": "public only" if public_only else "all (PRIVATE)", "tag_coverage": cov, "types": {},
           "train_sha256": {}, "within_strata": {}}
    for t in SERVED_TYPES:
        train = None
        if train_root and os.path.exists(os.path.join(train_root, t, "split.csv")):
            p = os.path.join(train_root, t, "split.csv")
            sp = pd.read_csv(p, dtype=str, keep_default_na=False).replace({"": None})
            train = sp[sp["split"] == "train"]
            out["train_sha256"][t] = pr.sha256_of(p)
        out["types"][t] = {}
        for by in sl.BY:
            if by == "tag" and not cache_dir:
                continue
            out["types"][t][by] = {era: confirmed_slices(g[(g.label_type == t) & (g.era == era)], by, ref,
                                                         train=train)
                                   for era in ("current", "previous")}
    cur = g[g["era"] == "current"]
    for (schema, t, regime), s in cur.groupby(["schema", "label_type", "regime"]):
        if schema not in pub:
            continue
        st = status_from_outcomes(s.sort_values(["ai_date", "label_id"], kind="mergesort")[ref].to_numpy(),
                                  rule)["status"]
        if st in ("pass", "undecided"):
            out["within_strata"].setdefault(pub[schema], {}).setdefault(t, {})[regime] = {
                "status": st, "severity": confirmed_slices(s, "severity", ref)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Rendering (the doc's generated blocks)
# ---------------------------------------------------------------------------------------------------------------------

MARK_BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/deployment_gate.py) -->"
MARK_END = "<!-- END GENERATED: {name} -->"
BLOCKS = ("headline", "status_counts", "public_table", "private_counts", "sensitivity", "backtest", "power",
          "slices", "provenance")


def _f(v, fmt="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v)


def _ci(r):
    if r.get("precision") is None:
        return ""
    return f"{r['precision']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def _load(out_dir, name):
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _ci_all(r):
    if r.get("precision_all") is None or r.get("ci_all_lo") is None:
        return ""
    return f"{r['precision_all']:.3f} [{r['ci_all_lo']:.3f}, {r['ci_all_hi']:.3f}]"


def render_headline(rep):
    if rep is None:
        return "_Not yet run._"
    s = rep["summary"]
    tot = {}
    for by_v in s["by_status"].values():
        for c in by_v.values():
            for k, n in c.items():
                tot[k] = tot.get(k, 0) + n
    big = s.get("largest_deployment_share_of_decided_gated") or {}
    acc = s["acceptance"]
    chk = s.get("acceptance_check")
    p = rep["pull"]
    lines = [f"- Pull `{p['file']}` (sha256 `{p['sha256'][:16]}`), AI dates of the public deployments' rows "
             f"{p['ai_date_min']} .. {p['ai_date_max']}; era key: {rep['rule']['era_key']}.",
             "- Every figure below and in the generated files covers **public deployments only**, pooled figures "
             "included; non-public deployments are left out (their figures are only in the private file).",
             f"- Current-era strata (public deployment x type x verdict x regime): {s['strata']}; by status: "
             + ", ".join(f"{k} {v}" for k, v in sorted(tot.items(), key=lambda kv: STATUSES.index(kv[0]))) + ".",
             f"- {100 * big.get('share', 0):.1f}% of the public deployments' current-era decided gated rows come from "
             f"one deployment ({big.get('deployment')}); {big.get('n_deployments_with_decided', 0)} public "
             "deployments have any.",
             "- Pooled current-era decided gated Agree (majority2) of the public deployments after the re-label "
             "exclusions, share confirmed in brackets: "
             + ", ".join(f"{t} {a['decided']:,} ({_f(a['share_confirmed'])})" for t, a in acc.items()) + "."]
    if p.get("unknown_revision_rows"):
        lines.append(f"- Rows with an empty model revision, excluded from both eras: {p['unknown_revision_rows']:,}.")
    if chk:
        lines.append("- Acceptance check (every deployment's counts before the exclusions, computed privately, "
                     f"against `{chk['against']}`): " + ("passes." if chk["ok"] else "FAILS."))
    return "\n".join(lines)


def render_status_counts(rep):
    if rep is None:
        return "_Not yet run._"
    lines = ["Public deployments only.", "",
             "| Type | Verdict | " + " | ".join(STATUSES[:5]) + " |", "|---|---|" + "---:|" * 5]
    for t, by_v in rep["summary"]["by_status"].items():
        for v, c in by_v.items():
            lines.append(f"| {t} | {v} | " + " | ".join(str(c.get(k, 0)) for k in STATUSES[:5]) + " |")
    return "\n".join(lines)


def _agree_rows(rep):
    for city, by_t in rep["deployments"].items():
        for t, by_v in sorted(by_t.items()):
            for regime, r in sorted((by_v.get("Agree") or {}).items()):
                yield city, t, regime, r


def _type_name(t, regime):
    return t if regime == "unrecorded" else f"{t} ({regime})"


def render_public_table(rep):
    if rep is None:
        return "_Not yet run._"
    rows = list(_agree_rows(rep))
    shown = [x for x in rows if x[3]["n"] > 0]
    zero = [x for x in rows if x[3]["n"] == 0]
    zero_by = {}
    for _, _, _, r in zero:
        zero_by[r["status"]] = zero_by.get(r["status"], 0) + 1
    dis = [r for by_t in rep["deployments"].values() for by_v in by_t.values()
           for r in (by_v.get("Disagree") or {}).values()]
    dis_txt = ("Disagree is `no_gated` in every public deployment (the round-3 Disagree maps are capped below 0.92), "
               "so it is not shown." if dis and all(r["status"] == "no_gated" for r in dis)
               else "Disagree strata are in `deployment_status.json`.")
    lines = ["Public deployments' Agree strata with at least one decided row. " + dis_txt + " n = decided rows "
             "(checkpoint the status is evaluated at); at checkpoint = precision [Wilson 95%] of the first "
             "`checkpoint` decided rows, which sets the status; all decided = precision [Wilson 95%] of every decided "
             "row, including those past the checkpoint; first pass = the first checkpoint that passed (a sticky rule "
             "would keep it; the status is re-judged at the current checkpoint); base rate = share of the "
             "deployment's decided AI-assessed labels of the type that humans call correct; more needed = further "
             "decided rows, all right, that would pass.", "",
             "| Deployment | Type | Status | first pass | n (checkpoint) | at checkpoint [95%] | all decided [95%] | "
             "base rate | more needed | previous era |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for city, t, regime, r in shown:
        prev = (r.get("previous_era") or {}).get("status", "")
        more = "" if r.get("more_needed_if_perfect") is None else r["more_needed_if_perfect"]
        fp = r.get("first_pass_checkpoint") or ""
        lines.append(f"| {city} | {_type_name(t, regime)} | {r['status']} | {fp} | {r['n']:,} "
                     f"({r['checkpoint'] or '-'}) | {_ci(r)} | {_ci_all(r)} | {_f(r.get('base_rate'))} | {more} | "
                     f"{prev} |")
    deps = len({c for c, _, _, _ in zero})
    lines += ["", f"Another {len(zero)} public Agree strata in {deps} deployments have current-era assessments but no "
              "decided row (" + ", ".join(f"{k} {v}" for k, v in sorted(zero_by.items())) + "); each needs at least "
              f"{rep['rule']['schedule'][0]} decided rows. Their previous-era status is in `deployment_status.json`."]
    return "\n".join(lines)


def render_private(rep):
    if rep is None:
        return "_Not yet run._"
    return rep["private"]["note"]


def render_sensitivity(rep):
    if rep is None:
        return "_Not yet run._"
    lines = [f"Strata whose status differs from the binding `{rep['rule']['reference']}` reference: "
             + ", ".join(f"`{k}` {v}" for k, v in rep["summary"]["status_differs_under"].items()) + ". Public Agree "
             "strata where any reference has at least one decided row; cell: status (decided n), and for `any_vote` "
             "the precision at the checkpoint.", "",
             "| Deployment | Type | majority2 | tie_half | any_vote |", "|---|---|---|---|---|"]
    for city, t, regime, r in _agree_rows(rep):
        th, av = r["sensitivity"].get("tie_half", {}), r["sensitivity"].get("any_vote", {})
        if max(r["n"], th.get("n", 0), av.get("n", 0)) == 0:
            continue
        p = _f(av.get("precision"))
        lines.append(f"| {city} | {_type_name(t, regime)} | {r['status']} ({r['n']:,}) | {th.get('status')} "
                     f"({th.get('n', 0):,}) | "
                     f"{av.get('status')} ({av.get('n', 0):,}{', ' + p if p else ''}) |")
    return "\n".join(lines)


def render_backtest(bt):
    if bt is None:
        return "_Not yet run._"
    lines = [f"Previous era, {bt['label']}. Checkpoints {', '.join(map(str, bt['rule']['schedule']))} then doubling; "
             "a stratum graduates at its first pass and stays graduated. Strata are split by evidence regime and never "
             "pooled across it. Cells: decided rows, precision [Wilson 95%].", "",
             "| Type | Verdict | Regime | strata (n >= min) | final pass / undecided / fail | graduated | n at "
             "graduation (median, range) | withheld in graduated | kept after graduation | never graduated | strata "
             "under min | all decided |",
             "|---|---|---|---:|---|---:|---|---|---|---|---|---|"]

    def cell(x):
        return f"{x['n']:,}" + (f", {_ci(x)}" if x["n"] else "")
    for regime, by_t in bt["pooled"].items():
        for t, by_v in by_t.items():
            for v, p in by_v.items():
                fs = p["final_status"]
                gn = p["graduation_n"]
                gtxt = "" if gn["median"] is None else f"{gn['median']:.0f} ({gn['min']}-{gn['max']})"
                lines.append(f"| {t} | {v} | {regime} | {p['strata']} ({p['strata_ge_min_n']}) | {fs.get('pass', 0)} "
                             f"/ {fs.get('undecided', 0)} / {fs.get('fail', 0)} | {p['graduated']} | {gtxt} | "
                             f"{cell(p['withheld_in_graduated'])} | {cell(p['kept'])} | {cell(p['never_graduated'])} | "
                             f"{cell(p['strata_under_min_n'])} | {cell(p['all'])} |")
    return "\n".join(lines)


def render_power(pw):
    if pw is None:
        return "_Not yet run._"
    r = pw["rule"]
    lines = [f"Rule: gate {r['gate']}, z {r['z']}, min n {r['min_n']}, checkpoints {', '.join(map(str, r['schedule']))}"
             " then doubling. Exact binomial, no simulation.", "",
             "Errors allowed at n: " + ", ".join(f"{n}: {e}" for n, e in pw["errors_allowed"].items()) + ".", "",
             f"Single look: the largest P(pass | true precision = gate) over n = {r['min_n']}..{pw['up_to']} is "
             f"{pw['max_single_look_pass_at_gate']:.4f}.", "",
             "| true precision | " + " | ".join(f"n={n}" for n in next(iter(pw["single_look_pass"].values()))) + " |",
             "|---|" + "---:|" * len(next(iter(pw["single_look_pass"].values())))]
    for p, by_n in pw["single_look_pass"].items():
        lines.append(f"| {p} | " + " | ".join(f"{v:.3f}" for v in by_n.values()) + " |")
    lines += ["", f"P(EVER pass by n = {pw['up_to']:,}) under each look schedule:", "",
              "| looks | " + " | ".join(next(iter(pw["ever_pass"].values()))) + " | z holding p = gate to 5% |",
              "|---|" + "---:|" * (len(next(iter(pw["ever_pass"].values()))) + 1)]
    for name, by_p in pw["ever_pass"].items():
        z = pw["z_for_5pct"].get(name)
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in by_p.values()) + f" | {_f(z, '{:.2f}')} |")
    longer = pw.get("ever_pass_checkpoints_at_gate_longer") or {}
    if longer:
        lines += ["", "Past n = {:,} the checkpoints keep doubling and the rate keeps growing: at true precision = "
                  "gate, P(ever pass) under the checkpoints is ".format(pw["up_to"])
                  + ", ".join(f"{v:.3f} by n = {int(u):,}" for u, v in longer.items()) + "."]
    if pw.get("notes"):
        lines += [""] + [f"- {n}" for n in pw["notes"]]
    return "\n".join(lines)


def render_slices(s):
    if s is None:
        return "_Not yet run._"
    lines = [f"Gated Agree rows decided under `{s['reference']}`, pooled over public deployments only (non-public "
             "deployments are left out of every table); confirmed share blank under "
             f"{s['min_n']} decided; train = round-3 `split.csv` training rows (correct / incorrect); largest = the "
             "largest one-deployment share of the slice."]
    if s.get("tag_coverage"):
        c = s["tag_coverage"]
        lines.append(f"Tags joined from cached public rawLabels: {c['exported']:,} of {c['public_rows']:,} public-"
                     f"deployment rows have an export (private deployments are never joined; rows without one are "
                     f"`{NOT_EXPORTED}`).")
    else:
        lines.append("Tags: not joined (no rawLabels cache given).")
    for t, by in s["types"].items():
        for b, eras in by.items():
            prev = {r["slice"]: r for r in eras["previous"]}
            names = [r["slice"] for r in eras["current"]] + [k for k in prev if k not in
                                                             {r["slice"] for r in eras["current"]}]
            cur = {r["slice"]: r for r in eras["current"]}
            if b == "tag":
                names = [n for n in names if (cur.get(n, {}).get("n", 0) >= s["min_n"]
                                              or prev.get(n, {}).get("n", 0) >= s["min_n"])]
            lines += ["", f"**{t} by {b}**" + (" (slices with at least min n in either era)" if b == "tag" else ""), "",
                      "| slice | current n | current confirmed | previous n | previous confirmed (baseline) | "
                      "largest (current) | train |", "|---|---:|---:|---:|---:|---:|---:|"]
            for n in names:
                c, p = cur.get(n, {}), prev.get(n, {})
                tr = c or p
                trs = f"{tr.get('train_correct', '')} / {tr.get('train_incorrect', '')}" if "train_correct" in tr else ""
                lines.append(f"| {n} | {c.get('n', 0):,} | {_f(c.get('confirmed'))} | {p.get('n', 0):,} | "
                             f"{_f(p.get('confirmed'))} | {_f(c.get('max_deployment_share'), '{:.0%}')} | {trs} |")
    if s.get("within_strata"):
        lines += ["", "**Severity within each current-era pass or undecided stratum** (public deployments)", "",
                  "| Deployment | Type | status | severity: n (confirmed) |", "|---|---|---|---|"]
        for city, by_t in s["within_strata"].items():
            for t, by_r in by_t.items():
                for regime, r in by_r.items():
                    cells = "; ".join(f"{x['slice']}: {x['n']} ({_f(x['confirmed'])})" for x in r["severity"])
                    name = t if regime == "unrecorded" else f"{t} ({regime})"
                    lines.append(f"| {city} | {name} | {r['status']} | {cells} |")
    return "\n".join(lines)


def render_provenance(pv):
    if pv is None:
        return "_Not yet run._"
    lines = [f"Versions: " + ", ".join(f"{k} {v}" for k, v in pv.get("versions", {}).items()) + "."]
    first = next((pv[k] for k in ("gate", "backtest", "slices") if k in pv), None)
    if first:
        p, li = first["pull"], first["listing"]
        lines += [f"Pull `{p['file']}`, sha256 `{p['sha256']}`. Public deployments only: {p['rows_public']:,} rows, {p['rows_kept_public']:,} kept after the drops and the "
                  f"exclusions ({', '.join(f'{k} {v:,}' for k, v in p['dropped_public'].items())}); eras "
                  f"{', '.join(f'{k} {v:,}' for k, v in p['eras_public'].items())}.",
                  f"Listing: live `{li['source']}`, sha256 `{li['sha256'][:16]}`, {li['n']} deployments, "
                  f"{li['n_public']} public. Every public file is checked against the non-public terms before it is "
                  "written.",
                  "Excluded re-label keys (rows of public deployments in the pull): " + "; ".join(
                      f"`{e['file']}` sha256 `{e['sha256'][:16]}` ({e['in_pull_public']})"
                      for e in first["exclusions"]["files"])
                  + f"; {first['exclusions']['rows_removed_public']} rows removed."]
    if "slices" in pv:
        sv = pv["slices"]
        lines.append(f"rawLabels cache: {sv.get('rawlabels_n_files')} files, sha256 of the sorted file hashes "
                     f"`{(sv.get('rawlabels_files_sha256') or '')[:16]}`; split.csv sha256: " + ", ".join(
                         f"{t} `{h[:12]}`" for t, h in sv.get("train_split_sha256", {}).items()) + ".")
    lines += ["", "| command | run on | code (commit; dirty = uncommitted changes) | flags |", "|---|---|---|---|"]
    for sec in ("gate", "backtest", "power", "slices"):
        if sec not in pv:
            continue
        f = {k: v for k, v in pv[sec]["flags"].items() if k != "command"}
        flags = " ".join(f"--{k.replace('_', '-')} {' '.join(map(str, v)) if isinstance(v, list) else v}"
                         for k, v in f.items() if v not in (None, [], False))
        lines.append(f"| {sec} | {pv[sec]['run_on']} | `{(pv[sec].get('git_commit') or '')[:12]}`"
                     f"{' dirty' if pv[sec].get('git_dirty') else ''} | `{flags}` |")
    return "\n".join(lines)


def render_blocks(out_dir):
    rep = _load(out_dir, "deployment_status.json")
    return {"headline": render_headline(rep), "status_counts": render_status_counts(rep),
            "public_table": render_public_table(rep), "private_counts": render_private(rep),
            "sensitivity": render_sensitivity(rep), "backtest": render_backtest(_load(out_dir, "backtest.json")),
            "power": render_power(_load(out_dir, "power.json")), "slices": render_slices(_load(out_dir, "slices.json")),
            "provenance": render_provenance(_load(out_dir, "provenance.json"))}


def splice(text, blocks):
    for name, body in blocks.items():
        pat = re.compile(re.escape(MARK_BEGIN.format(name=name)) + r".*?" + re.escape(MARK_END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: MARK_BEGIN.format(name=name) + "\n\n" + body + "\n\n" + MARK_END.format(name=name),
                       text)
    return text


def check_doc(doc_path, out_dir):
    """``[]`` when the doc's generated blocks equal what the committed JSON files render; else the drifted blocks."""
    with open(doc_path) as f:
        text = f.read()
    blocks = render_blocks(out_dir)
    return [b for b in BLOCKS if splice(text, {b: blocks[b]}) != text]


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def rule_from_args(args):
    return Rule(gate=args.gate, min_n=args.min_n, z=args.z,
                schedule=tuple(int(x) for x in str(args.schedule).split(",")), reference=args.reference)


def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    except Exception:
        return None


def _git_commit():
    return _git("rev-parse", "HEAD")


def _git_dirty():
    """True when the code (Python and shell files of ``analysis/`` and ``calibration/``, the two top-level modules)
    differs from HEAD or is untracked; READMEs and the generated docs do not count."""
    out = _git("status", "--porcelain", "--", "analysis/*.py", "analysis/*.sh", "calibration/*.py",
               "data_generation.py", "regions.py")
    return bool(out)


def _read_terms(path):
    if not path:
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def update_provenance(out_dir, section, payload, terms):
    path = os.path.join(out_dir, "provenance.json")
    pv = _load(out_dir, "provenance.json") or {}
    import platform
    pv["versions"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    pv[section] = {**payload, "git_commit": _git_commit(), "git_dirty": _git_dirty(),
                   "run_on": datetime.date.today().isoformat()}
    write_public_json(dict(sorted(pv.items())), path, terms)


def _public_pull_counts(raw, d, pub, revision_mode=False):
    """The pull's counts over the PUBLIC deployments' schemas only (``raw`` = the CSV as read, ``d`` = loaded and
    after the exclusions)."""
    rp = raw[raw["schema"].isin(pub)]
    deleted = _truthy(rp["deleted"])
    nd = rp[~deleted]
    mismatch = nd["model_id"].astype(str).str.lower() != (MODEL_PREFIX + nd["label_type"].astype(str).str.lower())
    dp = d[d["schema"].isin(pub)]
    out = {"rows_public": int(len(rp)), "rows_kept_public": int(len(dp)),
           "ai_date_min": str(dp["ai_date"].min()) if len(dp) else None,
           "ai_date_max": str(dp["ai_date"].max()) if len(dp) else None,
           "dropped_public": {"deleted_rows": int(deleted.sum()), "model_type_mismatch": int(mismatch.sum())},
           "eras_public": {k: int(v) for k, v in dp["era"].value_counts().sort_index().items()}}
    if revision_mode:
        out["unknown_revision_rows"] = int((dp["era"] == "unknown").sum())
    return out


def _prepare(args, fetch=None):
    """Fetch the listing (live), load the pull, apply exclusions. Returns ``(d, counts, listing, terms, meta,
    meta_all, raw)``: ``meta`` is the PUBLIC metadata (counts over public deployments only), ``meta_all`` the private
    one (every deployment), ``raw`` the pull as read (for the acceptance check)."""
    listing = fetch_listing(fetch)
    raw = read_pull(args.assessments)
    d, counts = load_pull(raw, args.since, args.before, args.unrevisioned_before)
    d = add_outcomes(d)
    cities = listing_cities(listing)
    # The term list comes from the UNFILTERED pull, so a private schema whose rows are all dropped or excluded is
    # still checked for.
    terms = private_terms(listing, raw["schema"].dropna().unique(), _read_terms(getattr(args, "extra_terms", None)))
    pub = public_schemas(d["schema"].unique(), cities)
    excl, per_file = item_exclusions(getattr(args, "exclude_items", None), d, cities, pub)
    n0, n0_pub = len(d), int(d["schema"].isin(pub).sum())
    if excl:
        d = d[~d["label_uid"].isin(excl)].reset_index(drop=True)
    n_pub = int(d["schema"].isin(pub).sum())
    entries = regions._entries(listing)
    listing_text = json.dumps(listing, sort_keys=True)
    sha = pr.sha256_of(args.assessments)
    listing_meta = {"source": dg.CITIES_API, "sha256": pr.sha256_of_text(listing_text), "n": len(entries),
                    "n_public": sum(1 for c in entries if c.get("visibility") == "public")}
    pull_pub = {"file": os.path.basename(args.assessments), "sha256": sha,
                **_public_pull_counts(raw, d, pub, "unknown_revision_rows" in counts)}
    meta = {"pull": pull_pub,
            "exclusions": {"files": [{k: e[k] for k in ("file", "sha256", "in_pull_public")} for e in per_file],
                           "rows_removed_public": int(n0_pub - n_pub)},
            "listing": listing_meta}
    meta_all = {"pull": {"file": pull_pub["file"], "sha256": sha, "rows": counts["rows"],
                         "rows_kept": counts["rows_kept"], "n_schemas": counts["n_schemas"],
                         "ai_date_min": counts["ai_date_min"], "ai_date_max": counts["ai_date_max"],
                         "dropped": {k: counts[k] for k in ("backup_schema_rows", "internal_schema_rows",
                                                            "deleted_rows", "model_type_mismatch")},
                         "eras": counts["eras"], "unknown_revision_rows": counts.get("unknown_revision_rows")},
                "exclusions": {"files": per_file, "rows_removed": int(n0 - len(d))},
                "listing": {**listing_meta, "n_non_public_terms": len(terms)}}
    if getattr(args, "private_out", None):
        os.makedirs(args.private_out, exist_ok=True)
        with open(os.path.join(args.private_out, "cities_listing.json"), "w") as f:
            f.write(listing_text)
    if counts.get("unknown_revision_rows"):
        print(f"WARNING: {counts['unknown_revision_rows']} rows have an empty model revision and are excluded from "
              "both eras (see --unrevisioned-before)", file=sys.stderr)
    return d, counts, listing, terms, meta, meta_all, raw


def _write_private_provenance(args, section, payload):
    """The private twin of ``provenance.json`` (every deployment's counts), under ``--private-out``."""
    if not getattr(args, "private_out", None):
        return
    path = os.path.join(args.private_out, "provenance_private.json")
    pv = {}
    if os.path.exists(path):
        with open(path) as f:
            pv = json.load(f)
    pv[section] = {**payload, "git_commit": _git_commit(), "git_dirty": _git_dirty(),
                   "run_on": datetime.date.today().isoformat()}
    pr.write_json(dict(sorted(pv.items())), path)


def _flags(args):
    """The run's flags with paths reduced to basenames (no home paths in public files)."""
    out = {}
    for k, v in sorted(vars(args).items()):
        if k == "func":
            continue
        if isinstance(v, list):
            v = [_short_path(str(x)) for x in v]
        elif isinstance(v, str) and (os.sep in v or v.startswith("~")):
            v = os.path.basename(v.rstrip(os.sep))
        out[k] = v
    return out


ACCEPTANCE_SOURCE = "docs/deployment-prior-2026-11/observed.md"


def cmd_gate(args, fetch=None):
    rule = rule_from_args(args)
    try:
        d, counts, listing, terms, meta, meta_all, raw = _prepare(args, fetch)
    except dg.ListingError as e:
        print(f"listing fetch failed, nothing written: {e}", file=sys.stderr)
        return 2
    ref = f"{meta['pull']['file']}@{meta['pull']['sha256'][:16]}"
    public, private, strata = build_report(d, counts, listing, rule, ref, meta["pull"], meta["exclusions"])
    private["pull"], private["exclusions"] = meta_all["pull"], meta_all["exclusions"]
    validate_report(public)
    # The acceptance check compares EVERY deployment's counts before the exclusions (a private-inclusive total) with
    # the ones observed.md states for the same pull. Only its outcome goes into the public file; the counts go to the
    # private one.
    acc = None
    if args.expect:
        d0, _ = load_pull(raw, args.since, args.before, args.unrevisioned_before)
        a0 = acceptance_counts(add_outcomes(d0), rule)
        want = [int(x) for x in args.expect.split(",")]
        got = [a0[t]["decided"] for t in SERVED_TYPES]
        acc = {"expected": dict(zip(SERVED_TYPES, want)), "got": dict(zip(SERVED_TYPES, got)), "ok": got == want}
        public["summary"]["acceptance_check"] = {"ok": acc["ok"], "against": ACCEPTANCE_SOURCE}
        private["acceptance_check"] = {**acc, "against": ACCEPTANCE_SOURCE}
    os.makedirs(args.public_out, exist_ok=True)
    write_public_json(public, os.path.join(args.public_out, "deployment_status.json"), terms)
    update_provenance(args.public_out, "gate", {"flags": _flags(args), **meta}, terms)
    if args.private_out:
        os.makedirs(args.private_out, exist_ok=True)
        pr.write_json(private, os.path.join(args.private_out, "deployment_status_all.json"))
        pd.DataFrame(private["strata"]).to_csv(os.path.join(args.private_out, "deployment_status_all.csv"), index=False)
        _write_private_provenance(args, "gate", {"flags": _flags(args), **meta_all})
    text = render_headline(public) + "\n\n" + render_status_counts(public)
    assert_public(text, terms, "stdout")
    print(text)
    if acc is not None and not acc["ok"]:
        print("ACCEPTANCE CHECK FAILED: the counts differ from " + ACCEPTANCE_SOURCE
              + (f" (see {args.private_out}/deployment_status_all.json)" if args.private_out else ""), file=sys.stderr)
        return 1
    return 0


def cmd_backtest(args, fetch=None):
    rule = rule_from_args(args)
    try:
        d, counts, listing, terms, meta, meta_all, _ = _prepare(args, fetch)
    except dg.ListingError as e:
        print(f"listing fetch failed, nothing written: {e}", file=sys.stderr)
        return 2
    public, private = backtest(d, listing, rule)
    public["exclusions"] = meta["exclusions"]
    write_public_json(public, os.path.join(args.public_out, "backtest.json"), terms)
    update_provenance(args.public_out, "backtest", {"flags": _flags(args), **meta}, terms)
    if args.private_out:
        private["exclusions"] = meta_all["exclusions"]
        pr.write_json(private, os.path.join(args.private_out, "backtest_all.json"))
        _write_private_provenance(args, "backtest", {"flags": _flags(args), **meta_all})
    text = render_backtest(public)
    assert_public(text, terms, "stdout")
    print(text)
    return 0


def cmd_power(args, fetch=None):
    rule = rule_from_args(args)
    pw = power_tables(rule, args.up_to)
    write_public_json(pw, os.path.join(args.public_out, "power.json"), set())
    update_provenance(args.public_out, "power", {"flags": _flags(args)}, set())
    print(render_power(pw))
    return 0


def cmd_slices(args, fetch=None):
    rule = rule_from_args(args)
    try:
        d, counts, listing, terms, meta, meta_all, _ = _prepare(args, fetch)
    except dg.ListingError as e:
        print(f"listing fetch failed, nothing written: {e}", file=sys.stderr)
        return 2
    s = slices_report(d, listing, rule, args.rawlabels_cache, args.train_root, public_only=True)
    if args.private_out:
        os.makedirs(args.private_out, exist_ok=True)
        pr.write_json(slices_report(d, listing, rule, args.rawlabels_cache, args.train_root, public_only=False),
                      os.path.join(args.private_out, "slices_all.json"))
        _write_private_provenance(args, "slices", {"flags": _flags(args), **meta_all})
    extra = {}
    if args.rawlabels_cache:
        files = sorted(n for n in os.listdir(args.rawlabels_cache) if n.endswith(".csv"))
        extra["rawlabels_files_sha256"] = pr.sha256_of_text("\n".join(
            f"{pr.sha256_of(os.path.join(args.rawlabels_cache, n))}  {n}" for n in files))
        extra["rawlabels_n_files"] = len(files)
    write_public_json(s, os.path.join(args.public_out, "slices.json"), terms)
    update_provenance(args.public_out, "slices", {"flags": _flags(args), **meta, **extra,
                                                  "train_split_sha256": s["train_sha256"]}, terms)
    text = render_slices(s)
    assert_public(text, terms, "stdout")
    print(text)
    return 0


def cmd_doc(args, fetch=None):
    if args.check:
        drift = check_doc(args.doc, args.public_out)
        if drift:
            print(f"{args.doc} drifted from {args.public_out} in block(s): {', '.join(drift)}", file=sys.stderr)
            return 1
        print(f"{args.doc} matches {args.public_out}")
        return 0
    with open(args.doc) as f:
        text = f.read()
    new = splice(text, render_blocks(args.public_out))   # render fully before opening the doc for writing
    with open(args.doc, "w") as f:
        f.write(new)
    print(f"wrote the generated blocks of {args.doc}")
    return 0


def cmd_privacy_check(args, fetch=None):
    """Check files (``-`` = stdin) for non-public terms; exit 1 on any hit, printing file and offsets only."""
    listing = fetch_listing(fetch)
    schemas = []
    if args.assessments:
        schemas = pd.read_csv(args.assessments, usecols=["schema"], dtype=str)["schema"].unique().tolist()
    terms = private_terms(listing, schemas, _read_terms(args.extra_terms))
    bad = 0
    for p in args.files:
        text = sys.stdin.read() if p == "-" else open(p, encoding="utf-8", errors="replace").read()
        hits = find_private_terms(text, terms) + find_private_terms(os.path.basename(p), terms)
        if hits:
            bad += 1
            print(f"HIT {p}: {len(hits)} at offset(s) {hits[:10]}")
    print(f"privacy-check: {len(args.files)} file(s), {len(terms)} non-public terms, {bad} file(s) with a hit")
    return 1 if bad else 0


def _rule_flags(p):
    p.add_argument("--since", default=SINCE, help="first day of the current era when the pull has no revision column")
    p.add_argument("--before", default=BEFORE,
                   help="the previous era is assessments BEFORE this day when the pull has no revision column; days "
                        "from --before up to --since are the changeover, in neither era")
    p.add_argument("--unrevisioned-before", default=None,
                   help="with a revision column: rows with an EMPTY revision assessed before this day are the "
                        "previous era (pass the day revisions began to be recorded); every other empty-revision row "
                        "is excluded as unknown")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--gate", type=float, default=im.GATE)
    p.add_argument("--z", type=float, default=pc.Z95, help="Wilson z (1.96 = 95%%)")
    p.add_argument("--schedule", default="50,100,200,400,800,1600", help="checkpoints; doubling after the last")
    p.add_argument("--reference", choices=REFERENCES, default="majority2")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, hlp in (("gate", cmd_gate, "the status report"), ("backtest", cmd_backtest, "replay, previous era"),
                          ("slices", cmd_slices, "severity and tag slices")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--assessments", required=True, help="CSV from analysis/pull_ai_assessments.sh (PRIVATE)")
        s.add_argument("--public-out", required=True, help="directory for the public JSON files (in the repo)")
        s.add_argument("--private-out", default=None, help="directory for the private files (under ~/Data)")
        s.add_argument("--exclude-items", nargs="*", default=[], help="masked re-label round keys to drop")
        s.add_argument("--extra-terms", default=None, help="file of extra non-public terms, one per line (private)")
        _rule_flags(s)
        s.set_defaults(func=fn)
        if name == "gate":
            s.add_argument("--expect", default=None,
                           help="comma-separated pooled current-era decided gated Agree counts per served type "
                                f"({','.join(SERVED_TYPES)}), checked before exclusions; exit 1 on a mismatch")
        if name == "slices":
            s.add_argument("--rawlabels-cache", default=None, help="cached public rawLabels (<city_id>.csv)")
            s.add_argument("--train-root", default=None, help="root holding <type>/split.csv (round 3)")
    w = sub.add_parser("power", help="exact pass probabilities and the look-schedule false-pass rate")
    w.add_argument("--public-out", required=True)
    w.add_argument("--up-to", type=int, default=2000)
    _rule_flags(w)
    w.set_defaults(func=cmd_power)
    c = sub.add_parser("doc", help="fill or --check the doc's generated blocks")
    c.add_argument("--public-out", required=True)
    c.add_argument("--doc", required=True)
    c.add_argument("--check", action="store_true")
    c.set_defaults(func=cmd_doc)
    k = sub.add_parser("privacy-check", help="search files for non-public terms (live listing)")
    k.add_argument("files", nargs="+")
    k.add_argument("--assessments", default=None, help="pull whose private schema names to add to the terms")
    k.add_argument("--extra-terms", default=None)
    k.set_defaults(func=cmd_privacy_check)
    return p


def main(argv=None, fetch=None):
    args = build_parser().parse_args(argv)
    return args.func(args, fetch=fetch)


if __name__ == "__main__":
    sys.exit(main())
