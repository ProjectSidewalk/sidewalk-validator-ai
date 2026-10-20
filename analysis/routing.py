"""Human-AI routing by calibrated accuracy, simulated on production outcomes (issue #18).

Today the AI's verdict becomes a vote only when the served accuracy is >= 0.92; otherwise the AI casts an Unsure and
the label waits for humans. That is a fixed gate. A *routing policy* asks a different question: for which labels can
the AI's verdict stand in for a human decision, and what does each choice of threshold cost (labels sent to humans)
and buy (accuracy)? Tohme (Hara et al., UIST'14) answered it for curb-ramp detection; this script answers it for
validation, from the production pull that ``calibration/calibrate_from_production.py`` was fitted on.

**Input.** ``labels_oof.csv`` written by ``python -m calibration.calibrate_from_production`` (one row per AI-assessed
label: raw verdict ``ai_result``, raw softmax ``ai_confidence``, the old served value ``ai_accuracy``, the Phase 0 map
``new_accuracy`` fitted on all rows, its out-of-fold value ``oof_accuracy``, and the human votes ``h_agree`` /
``h_dis`` / ``h_unsure`` with the AI's own vote excluded). Optionally ``ai_assessments.csv`` from
``analysis/pull_ai_assessments.sh`` for its two expert-validator columns. Both are private (they include private
cities): keep them and this script's outputs outside the repo and quote aggregates only.

**Definitions (identical to calibrate_from_production).** A label is *labelled* when ``h_agree + h_dis >=
--min-human-votes`` (default 2). The outcome is ``tie_half``: 1 if the strict human majority matches the AI's verdict,
0 if it opposes it, 0.5 on a tie. Labels are keyed on ``schema:label_id`` (``label_id`` is per city).

**Policies, per label type, on the labelled subset.**

* *Single threshold*: the AI decides a label when ``score >= t``; otherwise the label is routed to humans. Scores:
  ``oof_accuracy`` (calibrated, out-of-fold, the honest stand-in for the Phase 0 maps), ``ai_confidence`` (raw
  softmax) and ``ai_accuracy`` (the old shipped map). ``t`` sweeps the empirical quantiles of the score plus 0.92.
* *Per verdict*: the AI decides an Agree verdict when ``oof_accuracy >= t_agree`` and a Disagree verdict when
  ``oof_accuracy >= t_disagree`` (today's gate is applied per verdict map). A grid over both, and its Pareto front.

**Metrics per operating point.** ``routed`` = fraction of labelled labels sent to humans (the cost); ``n_ai`` = AI
decisions; ``ai_precision`` = mean outcome of the AI's decisions against the human majority; overall accuracy under
three human models for the routed labels:

* ``acc_oracle``: humans resolve every routed label correctly (an upper bound on what routing can reach; the right
  model when a routed label gets a full crowd, whose majority *is* the reference).
* ``acc_human``: one human of accuracy ``p_h`` decides each routed label. ``p_h`` is ``--human-accuracy``; the
  default ``auto`` derives it per type from the vote counts (below). The references differ: the AI is scored
  against all ``n`` votes, the human against ``n - 1``.
* ``acc_one_human`` (like for like): on the labels with ``>= --min-human-votes + 1`` votes only, the AI's decisions
  and a random human's are both scored against the same ``n - 1`` other votes (``loo_ai_agreement`` /
  ``loo_agreement``), so ``p_h`` is in effect measured on the routed labels themselves. Its x axis is
  ``routed_matched``, the share of *that* subset routed. Humans agree with each other much less on the labels the
  AI is unsure about, so a constant ``p_h`` flatters routing.

**Single-human accuracy from vote counts.** On a label with ``a`` Agree and ``d`` Disagree human votes, a randomly
chosen voter's vote is compared with the majority of the *other* votes (tie_half): an Agree voter scores 1 if
``a-1 > d``, 0.5 if equal, else 0, and symmetrically for Disagree. Averaged over that label's voters, this is the
leave-one-vote-out agreement of a single human with the crowd; it needs ``a + d >= --min-human-votes + 1`` so that
the "others" are themselves a labelled crowd. The arrival order of votes is not exported, so this is the agreement
of a random voter, not of the first one.

**Also reported.** AUC of each score for predicting an AI error (``correct == 0``; ties excluded) per type and per
verdict, with a bootstrap CI of the calibrated-minus-raw difference; the 0.92 operating points (old maps
``ai_accuracy >= 0.92``, Phase 0 ``new_accuracy >= 0.92`` in-sample and ``oof_accuracy >= 0.92`` out-of-fold) and
how each compares with the curve; complementarity with two expert validators (``--expert-cols``; on labels where the
AI and the crowd excluding that expert disagree, which side the expert takes); and precision against the
>= 3-unanimous-human gold subset.

Outputs in ``--out-dir``: ``curves.csv``, ``per_verdict_grid.csv``, ``operating_points.csv``, ``auc.csv``,
``complementarity.csv``, ``gold.csv``, ``human_accuracy.csv``, ``routing_<type>.png`` and ``summary.md``. Only
public deployments are named (``--cities-json``); expert columns are role-keyed ``expert1``, ``expert2`` whatever the
header says.

Usage::

    python analysis/routing.py --labels-oof ~/Data/<pull>/phase0/labels_oof.csv \\
        --assessments ~/Data/<pull>/ai_assessments.csv --cities-json ~/Data/<pull>/cities.json \\
        --out-dir ~/Data/<pull>/routing

Example of the core sweep::

    >>> s = np.array([0.6, 0.8, 0.95, 0.99]); y = np.array([0.0, 1.0, 1.0, 1.0])
    >>> c = sweep(s, y, thresholds=[0.9], p_h=0.8)
    >>> float(c.routed[0]), float(c.ai_precision[0]), float(c.acc_human[0])
    (0.5, 1.0, 0.9)

Needs only numpy, pandas and matplotlib (plus the repo's ``calibration`` package for ``add_outcomes``).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calibration.calibrate_from_production import add_outcomes  # noqa: E402
from calibration.per_city import load_cities, public_name  # noqa: E402

GATE = 0.92
TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]
VERDICTS = ["Agree", "Disagree"]
SCORES = {"oof_accuracy": "calibrated (Phase 0, out-of-fold)",
          "ai_confidence": "raw softmax confidence",
          "ai_accuracy": "old shipped map"}
OPERATING_POINTS = {"old maps @0.92": "ai_accuracy",
                    "Phase 0 @0.92 (in-sample)": "new_accuracy",
                    "Phase 0 @0.92 (out-of-fold)": "oof_accuracy"}
EXPERT_VOTES = ("Agree", "Disagree")


# ---------------------------------------------------------------- deployments

def public_names(cities_json):
    """Every listed deployment from a saved ``/v3/api/cities`` response (``calibration.per_city.load_cities``); ``{}``
    without one, in which case nothing is named."""
    return load_cities(cities_json) if cities_json else {}


def city_label(schema_city, cities):
    """The schema city to print if it is a public deployment, else ``None`` (private, unlisted or ambiguous: never
    named). The rule is ``calibration.per_city.match_city_id``, shared with ``calibration_doc_numbers.py``: an exact
    ``city_id`` match decides alone, and a prefix match counts only when exactly one listed id is the slug plus a
    region suffix.

    >>> city_label("x", {"x-yy": "public", "x-zz": "private"}) is None
    True
    """
    return schema_city if public_name(schema_city, cities) else None


def deployments_line(df, cities):
    """One printable line naming the deployments the public-only rule allows, and how many others were counted
    (private, ambiguous or not in the city list; never named)."""
    schemas = sorted(df.schema.str.replace("sidewalk_", "", regex=False).unique())
    shown = [c for c in schemas if city_label(c, cities)]
    return (f"{len(schemas)} deployments: {', '.join(shown) or '(none named)'}; plus {len(schemas) - len(shown)} "
            "not named (private, ambiguous or not in the city list)")


# ---------------------------------------------------------------- loading

def parse_expert_cols(arg, columns):
    """Map assessment-file header names to role names ``expert1``, ``expert2``, ...

    ``arg`` is ``"<header>=<role>,..."`` or ``None``; by default the last two columns are taken by position (that is
    where ``pull_ai_assessments.sh`` writes them). Header names are only read, never written anywhere.

    >>> parse_expert_cols(None, ["schema", "h_unsure", "x", "y"])
    {'x': 'expert1', 'y': 'expert2'}
    >>> parse_expert_cols("y=expert1", ["schema", "x", "y"])
    {'y': 'expert1'}
    """
    if not arg:
        return {c: f"expert{i + 1}" for i, c in enumerate(list(columns)[-2:])}
    out = {}
    for kv in arg.split(","):
        if "=" not in kv:
            raise ValueError(f"--expert-cols wants header=role pairs, got {kv!r}")
        k, v = (s.strip() for s in kv.split("=", 1))
        if k not in columns:
            raise ValueError("an --expert-cols header name is not in the assessments file")
        out[k] = v
    return out


def load(labels_oof, assessments=None, expert_cols=None, min_human_votes=2):
    """Read ``labels_oof.csv``, recompute the tie_half outcome at ``min_human_votes`` and attach expert votes.

    Returns every assessed row with ``label_uid``, ``h_n``, ``labelled``, ``tie``, ``correct`` and, when
    ``assessments`` is given, role-keyed expert columns (``None`` where that expert did not vote).
    """
    df = pd.read_csv(labels_oof, low_memory=False, true_values=["True"], false_values=["False"])
    df["label_uid"] = df.schema + ":" + df.label_id.astype(str)
    assert df.label_uid.is_unique, "(schema, label_id) is not unique in labels_oof"
    df["h_n"] = df.h_agree + df.h_dis
    if min_human_votes == 2 and "correct" in df:
        old = df[df.labelled]
        df = add_outcomes(df, min_human_votes, outcome="tie_half")
        # Same rows, same outcome as the calibration run: the definitions match.
        assert df.labelled.sum() == len(old) and np.allclose(df.loc[df.labelled, "correct"], old.correct)
    else:
        df = add_outcomes(df, min_human_votes, outcome="tie_half")
    if assessments:
        a = pd.read_csv(assessments, low_memory=False)
        mapping = parse_expert_cols(expert_cols, a.columns)
        a = a[["schema", "label_id", *mapping]].rename(columns=mapping)
        df = df.merge(a, on=["schema", "label_id"], how="left", validate="one_to_one")
    return df


# ---------------------------------------------------------------- human accuracy

def _side(x, y):
    """tie_half score of siding with the side that has ``x`` votes against ``y``: 1 ahead, 0.5 tied, 0 behind."""
    return np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))


def loo_agreement(h_agree, h_dis):
    """Leave-one-vote-out agreement of a random human voter with the majority of the others (tie_half), per label.

    NaN where there are no votes. Example: 2 Agree, 1 Disagree -> each Agree voter faces (1, 1), a tie (0.5); the
    Disagree voter faces (2, 0) and loses (0); mean (0.5 + 0.5 + 0) / 3.

    >>> round(float(loo_agreement(np.array([2]), np.array([1]))[0]), 4)
    0.3333
    """
    a, d = np.asarray(h_agree, float), np.asarray(h_dis, float)
    n = a + d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (a * _side(a - 1, d) + d * _side(d - 1, a)) / n, np.nan)


def loo_ai_agreement(h_agree, h_dis, ai_result):
    """The AI's agreement with the same "others" a random left-out human is scored against (tie_half), per label.

    This makes the AI-vs-single-human comparison like for like: both are scored against the majority of the same
    ``n - 1`` human votes. Example: 2 Agree, 1 Disagree, AI says Agree -> leaving out an Agree (2/3) leaves a tie
    (0.5); leaving out the Disagree (1/3) leaves 2-0 (1); mean 2/3.

    >>> round(float(loo_ai_agreement(np.array([2]), np.array([1]), np.array(["Agree"]))[0]), 4)
    0.6667
    """
    a, d = np.asarray(h_agree, float), np.asarray(h_dis, float)
    n = a + d
    agree = np.asarray(ai_result) == "Agree"
    drop_a = np.where(agree, _side(a - 1, d), _side(d, a - 1))     # others after leaving out one Agree vote
    drop_d = np.where(agree, _side(a, d - 1), _side(d - 1, a))     # ... one Disagree vote
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (a * drop_a + d * drop_d) / n, np.nan)


def human_accuracy_table(df, min_human_votes, experts=()):
    """Per type, on labels with ``>= min_human_votes + 1`` votes (so the "others" are a labelled crowd): the
    leave-one-vote-out agreement ``p_h`` of one human with the others, split by whether the calibrated score clears
    the gate (humans find the AI's unsure labels harder too), and the AI's agreement with the same others."""
    x = df[df.h_n >= min_human_votes + 1].assign(loo=lambda f: loo_agreement(f.h_agree, f.h_dis),
                                                 loo_ai=lambda f: loo_ai_agreement(f.h_agree, f.h_dis, f.ai_result))
    rows = []
    for t in TYPES + ["all"]:
        g = x if t == "all" else x[x.label_type == t]
        hi = g.oof_accuracy >= GATE
        rows.append({"label_type": t, "n": len(g), "p_h": g.loo.mean(),
                     "p_h_ai_below_gate": g.loo[~hi].mean(), "p_h_ai_above_gate": g.loo[hi].mean(),
                     "p_ai_same_others": g.loo_ai.mean(), "p_ai_same_others_below_gate": g.loo_ai[~hi].mean(),
                     "p_ai_same_others_above_gate": g.loo_ai[hi].mean()})
    out = pd.DataFrame(rows)
    if experts:
        # The same with every expert's vote removed from the counts (experts are not random humans).
        a, d = df.h_agree.copy(), df.h_dis.copy()
        for e in experts:
            a -= (df[e] == "Agree").astype(int)
            d -= (df[e] == "Disagree").astype(int)
        y = df[a + d >= min_human_votes + 1]
        ya, yd = a[y.index], d[y.index]
        y = y.assign(loo=loo_agreement(ya, yd), loo_ai=loo_ai_agreement(ya, yd, y.ai_result))
        ne = {t: (y if t == "all" else y[y.label_type == t]) for t in TYPES + ["all"]}
        out["n_no_experts"] = [len(ne[t]) for t in out.label_type]
        out["p_h_no_experts"] = [ne[t].loo.mean() for t in out.label_type]
        out["p_ai_no_experts"] = [ne[t].loo_ai.mean() for t in out.label_type]
    return out


def by_vote_count(df):
    """Per number of human Agree/Disagree votes on a labelled label: how many, the AI's mean tie_half outcome, the
    share of ties and the share of AI Disagree verdicts (which labels attract more votes)."""
    x = df[df.labelled].assign(votes=lambda f: f.h_n.clip(upper=6).astype(int))
    g = x.groupby("votes")
    out = pd.DataFrame({"n": g.size(), "ai_outcome": g.correct.mean(), "tie_share": g.tie.mean(),
                        "disagree_share": g.ai_result.apply(lambda r: (r == "Disagree").mean())}).reset_index()
    out["votes"] = out.votes.map(lambda v: f"{v}+" if v == 6 else str(v))
    return out


# ---------------------------------------------------------------- sweeps

def sweep(score, correct, thresholds, p_h=np.nan, loo=None, loo_ai=None):
    """Metrics of the policy "AI decides when ``score >= t``" for each ``t`` in ``thresholds``.

    ``correct`` is the tie_half outcome per label (against the majority of all its human votes). NaN scores are never
    decided by the AI. Columns: ``routed``, ``n_ai``, ``ai_precision``, ``ai_errors``, ``acc_oracle`` and
    ``acc_human`` (constant ``p_h``) on all rows given.

    ``loo`` and ``loo_ai`` (optional, together) are the per-label :func:`loo_agreement` and :func:`loo_ai_agreement`,
    NaN where undefined (too few votes). They add the like-for-like one-human model, computed on the rows where both
    are defined: the AI is scored with ``loo_ai`` on the labels it decides and the human with ``loo`` on the routed
    ones, so both are compared with the same ``n - 1`` other votes. ``n_matched`` is that subset's size,
    ``routed_matched`` the share of it routed and ``acc_one_human`` its overall accuracy.
    """
    s = np.asarray(score, float)
    y = np.asarray(correct, float)
    s = np.where(np.isnan(s), -np.inf, s)
    order = np.argsort(s, kind="stable")
    s, y = s[order], y[order]
    n = len(s)
    cy = np.concatenate([[0.0], np.cumsum(y)])             # cy[k] = sum of the k lowest scores' outcomes
    th = np.asarray(sorted(set(float(t) for t in thresholds)), float)
    k = np.searchsorted(s, th, side="left")                # rows below t -> routed
    n_ai = n - k
    sum_ai = cy[-1] - cy[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(n_ai > 0, sum_ai / np.maximum(n_ai, 1), np.nan)
        routed = k / n if n else np.full(len(th), np.nan)
        out = pd.DataFrame({"threshold": th, "n": n, "n_ai": n_ai, "routed": routed, "ai_precision": prec,
                            "ai_errors": n_ai - sum_ai,
                            "acc_oracle": (sum_ai + k) / n if n else np.nan,
                            "acc_human": (sum_ai + p_h * k) / n if n else np.nan})
        if loo is not None and loo_ai is not None:
            lh = np.asarray(loo, float)[order]
            la = np.asarray(loo_ai, float)[order]
            ok = ~np.isnan(lh) & ~np.isnan(la)
            cn = np.concatenate([[0], np.cumsum(ok)])
            ch = np.concatenate([[0.0], np.cumsum(np.where(ok, lh, 0.0))])
            ca = np.concatenate([[0.0], np.cumsum(np.where(ok, la, 0.0))])
            m = cn[-1]
            out["n_matched"] = m
            out["routed_matched"] = cn[k] / m if m else np.nan
            out["acc_one_human"] = (ch[k] + ca[-1] - ca[k]) / m if m else np.nan
    return out


def quantile_thresholds(score, n_quantiles=200, extra=(GATE,)):
    """The distinct empirical quantiles of ``score`` (0..1 in ``n_quantiles`` steps), plus ``extra`` values and
    ``inf`` (the AI decides nothing)."""
    s = np.asarray(score, float)
    s = s[~np.isnan(s)]
    q = np.quantile(s, np.linspace(0, 1, n_quantiles + 1)) if len(s) else np.array([])
    return np.unique(np.concatenate([q, np.asarray(extra, float), [np.inf]]))


def pareto_mask(cost, accuracy):
    """True for points no other point beats (lower or equal cost with strictly higher accuracy, or strictly lower cost
    with equal or higher accuracy).

    >>> pareto_mask(np.array([0.1, 0.2, 0.3, 0.2]), np.array([0.8, 0.9, 0.85, 0.7])).tolist()
    [True, True, False, False]
    """
    cost, accuracy = np.asarray(cost, float), np.asarray(accuracy, float)
    order = np.lexsort((-accuracy, cost))                  # cost ascending, then accuracy descending
    keep = np.zeros(len(cost), bool)
    best = -np.inf
    for i in order:
        if accuracy[i] > best:
            keep[i] = True
            best = accuracy[i]
    return keep


def per_verdict_grid(g, p_h, n_quantiles=200, score="oof_accuracy"):
    """Every (t_agree, t_disagree) pair on a quantile grid of ``score`` within each verdict, for one label type.

    Returns one row per pair with the same metrics as :func:`sweep` (without the one-human model) and Pareto flags
    for the oracle and the constant-``p_h`` human model.
    """
    parts = {}
    for v in VERDICTS:
        x = g[g.ai_result == v]
        th = quantile_thresholds(x[score], n_quantiles)
        parts[v] = sweep(x[score], x.correct, th, p_h=p_h)
    a, d = parts["Agree"], parts["Disagree"]
    n = len(g)
    n_ai = a.n_ai.values[:, None] + d.n_ai.values[None, :]
    err = a.ai_errors.values[:, None] + d.ai_errors.values[None, :]
    routed_n = n - n_ai
    out = pd.DataFrame({"t_agree": np.repeat(a.threshold.values, len(d)),
                        "t_disagree": np.tile(d.threshold.values, len(a)),
                        "n": n, "n_ai": n_ai.ravel(), "routed": (routed_n / n).ravel(),
                        "ai_errors": err.ravel()})
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ai_precision"] = np.where(out.n_ai > 0, 1 - out.ai_errors / out.n_ai.clip(lower=1), np.nan)
    out["acc_oracle"] = (out.n_ai - out.ai_errors + (out.n - out.n_ai)) / n
    out["acc_human"] = (out.n_ai - out.ai_errors + p_h * (out.n - out.n_ai)) / n
    out["pareto_oracle"] = pareto_mask(out.routed, out.acc_oracle)
    out["pareto_human"] = pareto_mask(out.routed, out.acc_human)
    return out


def operating_point(g, mask, p_h, loo=None, loo_ai=None):
    """Metrics (as :func:`sweep`) of an arbitrary AI-decides mask on one type's labelled rows."""
    s = np.where(np.asarray(mask, bool), 1.0, 0.0)
    return sweep(s, g.correct.values, [0.5], p_h=p_h, loo=loo, loo_ai=loo_ai).iloc[0]


def compare_to_curve(op, curve, col, routed_col="routed"):
    """How an operating point sits against a curve: the best ``col`` accuracy reachable with no more routing
    (``routed_col``), and the least routing that reaches at least the operating point's accuracy.

    >>> c = pd.DataFrame({"routed": [0.0, 0.2, 0.5, 1.0], "acc": [0.80, 0.90, 0.85, 0.70]})
    >>> {k: float(v) for k, v in compare_to_curve(pd.Series({"routed": 0.5, "acc": 0.84}), c, "acc").items()}
    {'curve_acc_at_same_routing': 0.9, 'curve_routing_at_same_acc': 0.2}
    """
    same_cost = curve[curve[routed_col] <= op[routed_col] + 1e-12]
    same_acc = curve[curve[col] >= op[col] - 1e-12]
    return {"curve_acc_at_same_routing": same_cost[col].max() if len(same_cost) else np.nan,
            "curve_routing_at_same_acc": same_acc[routed_col].min() if len(same_acc) else np.nan}


def _select(x, label_type, verdict):
    """Row mask for one type (``"all"`` = every type) and one verdict (``"both"`` = either)."""
    m = pd.Series(True, index=x.index)
    if label_type != "all":
        m &= x.label_type == label_type
    if verdict != "both":
        m &= x.ai_result == verdict
    return m


# ---------------------------------------------------------------- error prediction

def auc(score, positive):
    """Probability that a random positive scores higher than a random negative (ties count half): Mann-Whitney.

    >>> auc(np.array([0.9, 0.8, 0.3, 0.3]), np.array([1, 1, 0, 1]))
    0.8333333333333334
    """
    s = np.asarray(score, float)
    p = np.asarray(positive, bool)
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").values
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_table(df, n_boot=100, seed=42):
    """AUC of each score for ranking the AI's right verdicts above its wrong ones (ties excluded), per type, per
    verdict and pooled over verdicts.

    Besides the three ``SCORES`` it reports the shipped Phase 0 map (``new_accuracy``, in-sample) and two
    differences against raw confidence: ``diff_new_minus_raw`` (the shipped map is monotone in confidence within a
    verdict, so within a verdict this is only the effect of its flat blocks) and ``diff_oof_within_fold`` (the mean
    over folds of the calibrated-minus-raw AUC computed inside one fold, where a single map applies). The pooled
    out-of-fold difference ``diff_calib_minus_raw`` also mixes five per-fold maps. For pooled-verdict rows only, a
    percentile bootstrap 95% CI of ``diff_calib_minus_raw`` resamples labels (not clustered by city or panorama).
    """
    rng = np.random.default_rng(seed)
    x = df[df.labelled & (df.correct != 0.5)]
    rows = []
    for t in TYPES:
        for v in VERDICTS + ["both"]:
            g = x[_select(x, t, v)]
            right = (g.correct == 1).values
            row = {"label_type": t, "verdict": v, "n": len(g), "errors": int((~right).sum())}
            for s in list(SCORES) + ["new_accuracy"]:
                row[f"auc_{s}"] = auc(g[s].values, right)
            row["diff_calib_minus_raw"] = row["auc_oof_accuracy"] - row["auc_ai_confidence"]
            row["diff_new_minus_raw"] = row["auc_new_accuracy"] - row["auc_ai_confidence"]
            if "fold" in g:
                per_fold = [auc(f.oof_accuracy.values, (f.correct == 1).values)
                            - auc(f.ai_confidence.values, (f.correct == 1).values) for _, f in g.groupby("fold")]
                row["diff_oof_within_fold"] = np.nanmean(per_fold) if per_fold else np.nan
            diffs = []
            if v == "both" and n_boot and len(g) and 0 < right.sum() < len(g):
                a, c = g.oof_accuracy.values, g.ai_confidence.values
                for _ in range(n_boot):
                    i = rng.integers(0, len(g), len(g))
                    diffs.append(auc(a[i], right[i]) - auc(c[i], right[i]))
            row["diff_lo"], row["diff_hi"] = (np.nanpercentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- complementarity and gold

def _expert_view(df, e, min_crowd):
    """Labels expert ``e`` voted Agree/Disagree on, with the crowd **excluding that expert** (``h_agree``/``h_dis``
    include the expert's own vote: the pull excludes only the AI). Kept only when that crowd has >= ``min_crowd``
    votes and a strict majority; adds ``crowd_vs_ai`` (agree/disagree), ``expert_with_ai`` and ``gated_now``
    (calibrated out-of-fold >= 0.92). The other expert's vote, if any, stays in the crowd."""
    x = df[df[e].isin(EXPERT_VOTES)].copy()
    ca = x.h_agree - (x[e] == "Agree")
    cd = x.h_dis - (x[e] == "Disagree")
    keep = (ca + cd >= min_crowd) & (ca != cd)
    x, ca, cd = x[keep].copy(), ca[keep], cd[keep]
    crowd = np.where(ca > cd, "Agree", "Disagree")
    x["crowd_vs_ai"] = np.where(crowd == x.ai_result, "agree", "disagree")
    x["expert_with_ai"] = x[e] == x.ai_result
    x["gated_now"] = x.oof_accuracy >= GATE
    return x


def complementarity(df, experts, min_crowd=2):
    """On labels an expert voted Agree/Disagree on, compare the AI's verdict with the crowd **excluding that expert**.

    For each type, verdict and expert: ``n_disagree`` labels where the crowd (>= ``min_crowd`` other human
    Agree/Disagree votes, strict majority) opposes the AI, and the share of those where the expert sides with the AI;
    ``n_agree`` where crowd and AI agree, and the share where the expert sides with both.
    """
    rows = []
    for e in experts:
        x = _expert_view(df, e, min_crowd)
        for t in TYPES + ["all"]:
            for v in VERDICTS + ["both"]:
                g = x[_select(x, t, v)]
                dis, agr = g[g.crowd_vs_ai == "disagree"], g[g.crowd_vs_ai == "agree"]
                rows.append({"expert": e, "label_type": t, "verdict": v,
                             "n_disagree": len(dis), "expert_sides_with_ai": dis.expert_with_ai.mean(),
                             "n_disagree_above_gate": int(dis.gated_now.sum()),
                             "expert_sides_with_ai_above_gate": dis[dis.gated_now].expert_with_ai.mean(),
                             "n_agree": len(agr), "expert_with_both": agr.expert_with_ai.mean()})
    return pd.DataFrame(rows)


def contested_gated_labels(df, experts, min_crowd=2):
    """Distinct ``label_uid`` s where the calibrated score clears 0.92 and at least one expert's crowd opposes the AI
    (the union over experts of the ``n_disagree_above_gate`` labels, which can overlap)."""
    uids = set()
    for e in experts:
        x = _expert_view(df, e, min_crowd)
        uids |= set(x.label_uid[(x.crowd_vs_ai == "disagree") & x.gated_now])
    return uids


def gold_table(df, ops, min_votes=3):
    """AI precision against the gold subset (>= ``min_votes`` human Agree/Disagree votes, all on one side) for all
    verdicts and for each operating point's AI votes, per type and verdict; alongside the same on the majority."""
    gold = (df.h_n >= min_votes) & ((df.h_agree == 0) | (df.h_dis == 0))
    rows = []
    for t in TYPES:
        for v in VERDICTS:
            g = df[(df.label_type == t) & (df.ai_result == v)]
            gg = g[gold[g.index]]
            for name, mask in [("all verdicts", pd.Series(True, index=g.index))] + \
                              [(k, g[c] >= GATE) for k, c in ops.items()]:
                lab = g[g.labelled & mask]
                gm = gg[mask[gg.index]]
                rows.append({"label_type": t, "verdict": v, "policy": name,
                             "n_majority": len(lab), "precision_majority": lab.correct.mean(),
                             "n_gold": len(gm), "precision_gold": gm.correct.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots and summary

def plot_curves(curves, ops, label_type, path, min_human_votes=2):
    """Two panels: left, overall accuracy under the like-for-like one-human model against the share of those labels
    routed (labels with enough votes for it); right, the precision of the AI's own decisions against the share of
    labelled labels routed. One line per score, the 0.92 operating points marked, a legend on each panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=80)
    # The old map is drawn last and thicker so that it is visible where it runs on top of the raw line.
    styles = {"oof_accuracy": ("#1f77b4", "-", 1.8, 3), "ai_confidence": ("#ff7f0e", "--", 1.8, 2),
              "ai_accuracy": ("#9467bd", ":", 2.4, 4)}
    markers = {"old maps @0.92": ("s", "#d62728"), "Phase 0 @0.92 (in-sample)": ("^", "#2ca02c"),
               "Phase 0 @0.92 (out-of-fold)": ("o", "#1f77b4")}
    c = curves[curves.label_type == label_type]
    o = ops[ops.label_type == label_type]
    panels = ((axes[0], "routed_matched", "acc_one_human",
               f"fraction routed (labels with >= {min_human_votes + 1} human votes)",
               "overall accuracy, one human per routed label\n(AI and human vs the same other votes)"),
              (axes[1], "routed", "ai_precision", "fraction of labelled labels routed to humans",
               "precision of the AI's decisions\nvs human majority (tie_half)"))
    for ax, xcol, ycol, xlabel, ylabel in panels:
        for s, label in SCORES.items():
            x = c[(c.score == s) & (c.n_ai > 0)].sort_values(xcol)
            colr, ls, lw, z = styles[s]
            ax.plot(x[xcol], x[ycol], color=colr, ls=ls, lw=lw, zorder=z, label=label)
        for _, r in o.iterrows():
            m, colr = markers[r.operating_point]
            ax.scatter([r[xcol]], [r[ycol]], marker=m, s=60, color=colr, edgecolor="black", zorder=5,
                       label=r.operating_point)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[1].axhline(GATE, color="black", lw=0.8, ls="-.")
    # One legend for both panels (same lines and markers), below the axes so it never covers a curve or a marker.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{label_type}: the AI decides when score >= t, humans decide the rest", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def envelope_gap(curves, label_type, a="oof_accuracy", b="ai_confidence", n_points=101):
    """Min and max over routing levels r in [0, 1] of (best one-human accuracy score ``a`` reaches with
    ``routed_matched <= r``) minus the same for score ``b``: a min >= 0 means ``a``'s curve dominates.

    >>> c = pd.DataFrame({"label_type": "T", "score": ["a", "a", "b", "b"], "routed_matched": [0, 1, 0, 1],
    ...                   "acc_one_human": [0.9, 0.8, 0.85, 0.8]})
    >>> [round(x, 3) for x in envelope_gap(c, "T", "a", "b")]
    [0.05, 0.05]
    """
    c = curves[curves.label_type == label_type]
    env = []
    for s in (a, b):
        x = c[(c.score == s) & c.acc_one_human.notna()].sort_values("routed_matched")
        if not len(x):
            return np.nan, np.nan
        best = x.acc_one_human.cummax().values
        r = x.routed_matched.values
        grid = np.linspace(0, 1, n_points)
        idx = np.searchsorted(r, grid + 1e-12, side="right") - 1
        env.append(np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan))
    d = np.round(env[0] - env[1], 9) + 0.0                   # float noise at equal points must not print as -0.000
    return float(np.nanmin(d)), float(np.nanmax(d))


def _f(x, fmt=".3f"):
    return "—" if pd.isna(x) else format(x, fmt)


def summary_markdown(meta, hum, votes, curves, ops, grid_front, aucs, comp, gold, n_contested_union=None):
    """The markdown tables the doc copies from."""
    k = meta["min_human_votes"] + 1
    L = [f"# Routing simulation ({meta['inputs']})", "", meta["deployments"], "",
         f"Labelled = >= {meta['min_human_votes']} human Agree/Disagree votes; outcome tie_half; "
         f"{meta['n_labelled']:,} labelled of {meta['n_assessed']:,} assessed.", "",
         f"## Single-human accuracy (leave-one-vote-out, labels with >= {k} votes)", "",
         "AI (same others) = the AI's raw verdict scored against the same n - 1 votes as the left-out human. "
         "'no experts' = the same with both expert validators' votes removed from the counts first.", "",
         "| type | n | p_h | p_h, calibrated < 0.92 | p_h, >= 0.92 | AI (same others) | AI, < 0.92 | AI, >= 0.92 "
         "| n, no experts | p_h, no experts | AI, no experts | p_h used |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in hum.iterrows():
        L.append(f"| {r.label_type} | {r.n:,} | {_f(r.p_h)} | {_f(r.p_h_ai_below_gate)} | {_f(r.p_h_ai_above_gate)} "
                 f"| {_f(r.p_ai_same_others)} | {_f(r.p_ai_same_others_below_gate)} "
                 f"| {_f(r.p_ai_same_others_above_gate)} | {_f(r.get('n_no_experts', np.nan), ',.0f')} "
                 f"| {_f(r.get('p_h_no_experts', np.nan))} | {_f(r.get('p_ai_no_experts', np.nan))} "
                 f"| {_f(meta['p_h'].get(r.label_type, np.nan))} |")
    L += ["", "## Labelled labels by number of human votes (6 = 6 or more)", "",
          "| votes | labels | AI outcome (tie_half) | tie share | AI Disagree share |", "|---|---:|---:|---:|---:|"]
    for _, r in votes.iterrows():
        L.append(f"| {r.votes} | {r.n:,} | {_f(r.ai_outcome)} | {_f(r.tie_share)} | {_f(r.disagree_share)} |")
    if "disagree_at_099" in meta:
        L += ["", "## Raw Disagree verdicts at confidence >= 0.99: precision vs the majority (labelled)", "",
              "| type | n | precision |", "|---|---:|---:|"]
        for t, (n, p) in meta["disagree_at_099"].items():
            L.append(f"| {t} | {n:,} | {_f(p)} |")
    L += ["", "## The 0.92 operating points", "",
          "routed = share of labelled labels the AI does not decide; acc oracle = humans right on every routed label; "
          "acc const p_h = one human of accuracy p_h (AI scored against all n votes, human against n - 1: the "
          f"references differ); one-human columns = like-for-like model on labels with >= {k} votes (AI and human "
          "both scored against the same n - 1 other votes). Curve = calibrated out-of-fold single threshold, compared "
          "under the one-human model.", "",
          "| type | operating point | routed (labelled) | routed (all assessed) | AI decisions | AI precision | "
          f"acc oracle | acc const p_h | routed (>= {k} votes) | acc one human | curve acc at same routing | "
          "curve routing at same acc |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in ops.iterrows():
        L.append(f"| {r.label_type} | {r.operating_point} | {_f(r.routed)} | {_f(r.routed_all_assessed)} | "
                 f"{int(r.n_ai):,} | {_f(r.ai_precision)} | {_f(r.acc_oracle)} | {_f(r.acc_human)} | "
                 f"{_f(r.routed_matched)} | {_f(r.acc_one_human)} | "
                 f"{_f(r.curve_acc_at_same_routing)} | {_f(r.curve_routing_at_same_acc)} |")
    L += ["", "## Accuracy-maximizing single threshold per score, one-human model (picked in-sample on the same curve)",
          "", f"| type | score | threshold | routed (>= {k} votes) | acc one human | acc, every label routed |",
          "|---|---|---:|---:|---:|---:|"]
    for t in TYPES:
        for s in SCORES:
            c = curves[(curves.label_type == t) & (curves.score == s)]
            if not len(c) or c.acc_one_human.isna().all():
                continue
            b = c.loc[c.acc_one_human.idxmax()]
            allh = c[c.n_ai == 0].acc_one_human
            L.append(f"| {t} | {s} | {_f(b.threshold, '.4f')} | {_f(b.routed_matched)} | {_f(b.acc_one_human)} | "
                     f"{_f(allh.iloc[0] if len(allh) else np.nan)} |")
    L += ["", "Calibrated minus raw-confidence curve, best one-human accuracy at each routing level (min, max over "
          "routing): " + "; ".join(f"{t} {lo:+.3f}, {hi:+.3f}" for t in TYPES
                                    for lo, hi in [envelope_gap(curves, t)]
                                    if (curves.label_type == t).any()) + "."]
    L += ["", "## Calibrated per-verdict policy: points on the Pareto front at fixed AI-precision targets", "",
          "For each target, the least-routing (t_agree, t_disagree) pair whose pooled AI precision is >= the target "
          f"(oracle-human Pareto front of a {meta['grid_quantiles']}-quantile grid per verdict, out-of-fold score). "
          "inf = that verdict never decides; a target no point reaches is listed with routed 1.", "",
          "| type | target precision | t_agree | t_disagree | routed | AI decisions | AI precision |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in grid_front.iterrows():
        L.append(f"| {r.label_type} | {r.target:.2f} | {_f(r.t_agree, '.4f')} | {_f(r.t_disagree, '.4f')} | "
                 f"{_f(r.routed)} | {int(r.n_ai):,} | {_f(r.ai_precision)} |")
    L += ["", "## Error prediction: AUC of each score (right verdicts ranked above wrong; ties excluded)", "",
          "shipped = Phase 0 map fitted on all rows (in-sample, monotone in confidence within a verdict); "
          "within-fold = mean over folds of AUC(oof) - AUC(raw) inside one fold. CI: label bootstrap, pooled rows "
          "only, not clustered by city or panorama.", "",
          "| type | verdict | n | errors | calibrated (oof) | shipped map | raw confidence | old map | shipped − raw "
          "| oof − raw | oof − raw within fold | oof − raw 95% CI |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in aucs.iterrows():
        ci = f"[{_f(r.diff_lo, '+.3f')}, {_f(r.diff_hi, '+.3f')}]" if not pd.isna(r.diff_lo) else "—"
        L.append(f"| {r.label_type} | {r.verdict} | {r.n:,} | {r.errors:,} | {_f(r.auc_oof_accuracy)} | "
                 f"{_f(r.auc_new_accuracy)} | {_f(r.auc_ai_confidence)} | {_f(r.auc_ai_accuracy)} | "
                 f"{_f(r.diff_new_minus_raw, '+.4f')} | {_f(r.diff_calib_minus_raw, '+.4f')} | "
                 f"{_f(r.get('diff_oof_within_fold', np.nan), '+.4f')} | {ci} |")
    if comp is not None and len(comp):
        L += ["", "## Complementarity: expert vs AI where the AI and the crowd (excluding the expert) disagree", "",
              "The other expert's vote, if any, stays in the crowd.", "",
              "| expert | type | verdict | AI-vs-crowd disagreements | expert sides with AI | of those above 0.92 | "
              "expert sides with AI there | AI-crowd agreements | expert sides with both |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in comp[comp.verdict != "both"].iterrows():
            L.append(f"| {r.expert} | {r.label_type} | {r.verdict} | {r.n_disagree:,} | {_f(r.expert_sides_with_ai)} | "
                     f"{r.n_disagree_above_gate:,} | {_f(r.expert_sides_with_ai_above_gate)} | {r.n_agree:,} | "
                     f"{_f(r.expert_with_both)} |")
        for _, r in comp[(comp.label_type == "all") & (comp.verdict == "both")].iterrows():
            L.append(f"\n{r.expert}, all types and verdicts: {r.n_disagree:,} disagreements, sides with AI "
                     f"{_f(r.expert_sides_with_ai)}; {r.n_agree:,} agreements, sides with both {_f(r.expert_with_both)}.")
        if n_contested_union is not None:
            L.append(f"\nDistinct labels contested above 0.92 (union over experts): {n_contested_union:,}.")
    L += ["", "## Precision against the >= 3-unanimous gold subset vs the tie_half majority", "",
          "| type | verdict | policy | n (majority) | precision (majority) | n (gold) | precision (gold) |",
          "|---|---|---|---:|---:|---:|---:|"]
    for _, r in gold.iterrows():
        L.append(f"| {r.label_type} | {r.verdict} | {r.policy} | {r.n_majority:,} | {_f(r.precision_majority)} | "
                 f"{r.n_gold:,} | {_f(r.precision_gold)} |")
    return "\n".join(L) + "\n"


def front_at_targets(grid, targets=(0.90, 0.92, 0.95)):
    """Least-routing Pareto (oracle) grid point per type whose AI precision reaches each target; a target no point
    reaches gets a row with ``inf`` thresholds, routed 1 and no AI decisions.

    >>> g = pd.DataFrame({"label_type": "T", "t_agree": [0.5, 0.8, 0.9], "t_disagree": np.inf, "n_ai": [10, 6, 2],
    ...                   "routed": [0.0, 0.4, 0.8], "ai_precision": [0.85, 0.93, 0.97], "pareto_oracle": True})
    >>> front_at_targets(g, targets=(0.92, 0.99))[["target", "t_agree", "routed"]].values.tolist()
    [[0.92, 0.8, 0.4], [0.99, inf, 1.0]]
    """
    rows = []
    for t, g in grid.groupby("label_type", sort=False):
        f = g[g.pareto_oracle & (g.n_ai > 0)]
        for tg in targets:
            ok = f[f.ai_precision >= tg]
            if len(ok):
                r = ok.loc[ok.routed.idxmin()]
                rows.append({"label_type": t, "target": tg, **r[["t_agree", "t_disagree", "routed", "n_ai",
                                                                  "ai_precision"]].to_dict()})
            else:
                rows.append({"label_type": t, "target": tg, "t_agree": np.inf, "t_disagree": np.inf, "routed": 1.0,
                             "n_ai": 0, "ai_precision": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def run(df, out_dir, human_accuracy="auto", min_human_votes=2, seed=42, n_boot=100, n_quantiles=200,
        grid_quantiles=200, public=None, experts=None, plots=True, inputs=""):
    """Compute every table, write them to ``out_dir`` and return them as a dict of DataFrames."""
    os.makedirs(out_dir, exist_ok=True)
    experts = experts if experts is not None else sorted(c for c in df.columns if re.fullmatch(r"expert\d+", c))
    lab = df[df.labelled].copy()
    enough = lab.h_n >= min_human_votes + 1
    lab["loo"] = np.where(enough, loo_agreement(lab.h_agree, lab.h_dis), np.nan)
    lab["loo_ai"] = np.where(enough, loo_ai_agreement(lab.h_agree, lab.h_dis, lab.ai_result), np.nan)
    hum = human_accuracy_table(df, min_human_votes, experts)
    if human_accuracy == "auto":
        p_h = dict(zip(hum.label_type, hum.p_h))
    else:
        p_h = {t: float(human_accuracy) for t in TYPES + ["all"]}

    curves, ops, grids = [], [], []
    for t in TYPES:
        g = lab[lab.label_type == t]
        if not len(g):
            continue
        for s in SCORES:
            c = sweep(g[s], g.correct, quantile_thresholds(g[s], n_quantiles), p_h=p_h[t], loo=g.loo,
                      loo_ai=g.loo_ai)
            curves.append(c.assign(label_type=t, score=s))
        ref = curves[-len(SCORES)]                       # the oof_accuracy curve for this type
        a = df[df.label_type == t]
        for name, col in OPERATING_POINTS.items():
            r = operating_point(g, g[col] >= GATE, p_h[t], loo=g.loo, loo_ai=g.loo_ai)
            cmp = compare_to_curve(r, ref, "acc_one_human", "routed_matched")
            # Coverage over every assessed label needs a served value on unlabelled rows: out-of-fold has none.
            # A missing value is routed (never AI-decided), as in sweep().
            routed_all = (~(a[col] >= GATE)).mean() if col != "oof_accuracy" else np.nan
            ops.append({"label_type": t, "operating_point": name, **r.to_dict(), **cmp,
                        "routed_all_assessed": routed_all})
        grids.append(per_verdict_grid(g, p_h[t], n_quantiles=grid_quantiles).assign(label_type=t))
    curves = pd.concat(curves, ignore_index=True)
    ops = pd.DataFrame(ops)
    grid = pd.concat(grids, ignore_index=True)
    front = front_at_targets(grid)
    aucs = auc_table(df, n_boot=n_boot, seed=seed)
    comp = complementarity(df, experts, min_crowd=min_human_votes) if experts else pd.DataFrame()
    n_union = len(contested_gated_labels(df, experts, min_human_votes)) if experts else None
    gold = gold_table(df, OPERATING_POINTS, min_votes=max(3, min_human_votes))
    votes = by_vote_count(df)

    curves.to_csv(os.path.join(out_dir, "curves.csv"), index=False)
    grid.to_csv(os.path.join(out_dir, "per_verdict_grid.csv"), index=False)
    ops.to_csv(os.path.join(out_dir, "operating_points.csv"), index=False)
    aucs.to_csv(os.path.join(out_dir, "auc.csv"), index=False)
    comp.to_csv(os.path.join(out_dir, "complementarity.csv"), index=False)
    gold.to_csv(os.path.join(out_dir, "gold.csv"), index=False)
    votes.to_csv(os.path.join(out_dir, "by_vote_count.csv"), index=False)
    hum.assign(p_h_used=hum.label_type.map(p_h)).to_csv(os.path.join(out_dir, "human_accuracy.csv"), index=False)
    if plots:
        for t in curves.label_type.unique():
            plot_curves(curves, ops, t, os.path.join(out_dir, f"routing_{t.lower()}.png"), min_human_votes)
    meta = {"inputs": inputs, "deployments": deployments_line(df, public or {}), "min_human_votes": min_human_votes,
            "n_labelled": len(lab), "n_assessed": len(df), "p_h": p_h, "grid_quantiles": grid_quantiles,
            "disagree_at_099": {t: (len(x), x.correct.mean()) for t in TYPES
                                for x in [lab[(lab.label_type == t) & (lab.ai_result == "Disagree")
                                              & (lab.ai_confidence >= 0.99)]]}}
    md = summary_markdown(meta, hum, votes, curves, ops, front, aucs, comp, gold, n_union)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    return {"curves": curves, "ops": ops, "grid": grid, "front": front, "auc": aucs, "complementarity": comp,
            "gold": gold, "human": hum, "votes": votes, "summary": md, "meta": meta, "n_contested_union": n_union}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-oof", required=True,
                   help="labels_oof.csv from python -m calibration.calibrate_from_production (private; outside the repo)")
    p.add_argument("--assessments", default=None,
                   help="ai_assessments.csv from analysis/pull_ai_assessments.sh, read only for its expert columns")
    p.add_argument("--cities-json", default=None,
                   help="saved /v3/api/cities response; only deployments marked public there are named")
    p.add_argument("--out-dir", required=True, help="where the CSVs, PNGs and summary.md go (keep outside the repo)")
    p.add_argument("--human-accuracy", default="auto",
                   help="accuracy p_h of one human deciding a routed label: 'auto' (default) derives it per type from "
                        "leave-one-vote-out agreement in the vote counts; a number in [0, 1] uses it for every type")
    p.add_argument("--expert-cols", default=None,
                   help="'<header>=expert1,<header>=expert2': which --assessments columns are expert votes; default: "
                        "the last two columns, by position. Names are never written to the outputs")
    p.add_argument("--min-human-votes", type=int, default=2,
                   help="human Agree/Disagree votes for a label to count as labelled (default 2, as the maps)")
    p.add_argument("--seed", type=int, default=42, help="seed for the AUC bootstrap")
    p.add_argument("--n-boot", type=int, default=100, help="bootstrap resamples for the AUC difference CI (0 = none)")
    p.add_argument("--no-plots", action="store_true", help="skip the PNGs")
    args = p.parse_args(argv)
    if args.human_accuracy != "auto":
        v = float(args.human_accuracy)
        if not 0 <= v <= 1:
            p.error("--human-accuracy must be 'auto' or in [0, 1]")
    if args.min_human_votes < 1:
        p.error("--min-human-votes must be >= 1")

    df = load(args.labels_oof, args.assessments, args.expert_cols, args.min_human_votes)
    public = public_names(args.cities_json)
    print("counted:", deployments_line(df, public))
    res = run(df, args.out_dir, args.human_accuracy, args.min_human_votes, args.seed, args.n_boot,
              public=public, plots=not args.no_plots,
              inputs=f"{os.path.basename(args.labels_oof)}, min_human_votes={args.min_human_votes}")
    print(res["summary"])
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
