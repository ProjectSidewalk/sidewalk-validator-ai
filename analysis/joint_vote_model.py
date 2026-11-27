"""An asymmetric Dawid-Skene model of the crowd's votes, scored against the expert gold set (#12, #74).

Why
---
Issue #12's per-validator reliability (one smoothed hit rate per validator, ``validator_reliability.py``) changes
almost nothing: its weighted targets correlate 0.999 with plain soft targets, and it reverses 0 of 142,275 Seattle
majorities (``docs/adjudicator-2026-11.md``). One number per validator cannot tell a validator who agrees with
everything from one who is careful, and the Disagree side is where the crowd's targets go wrong. This module fits the
standard latent-truth alternative (Dawid and Skene 1979) with two rates per validator, so a validator's Agree and
Disagree votes can carry different weight, and asks whether its verdict matches the expert gold set of rounds 1 and 2
more often than ``majority2`` does. The pre-registered contrast and the rule are in
``docs/joint-vote-model-2026-11.md``; this module computes them.

The model (one fit per label type, over every label of the gold deployments)
---------------------------------------------------------------------------
* Latent truth ``z_i`` in {correct, incorrect} for each label ``i``.
* Validator ``u`` (``user_id``, the cross-city key) has sensitivity ``s_u = P(Agree | correct)`` and specificity
  ``p_u = P(Disagree | incorrect)``, each with a Beta prior of strength ``kappa`` centred on the population rate of
  the type (``mu_s``, ``mu_p``, re-estimated every iteration), so a validator with few votes sits near the population.
* Class prior ``pi_c = P(correct)`` per deployment, shrunk towards the type's rate with ``DEPLOY_PRIOR_STRENGTH``
  pseudo-labels.
* Arm ``labeler`` (secondary) replaces the deployment prior of a label by its labeler's precision ``lambda_l``
  (``P(correct)`` of that person's labels of the type), shrunk towards the deployment priors of their labels with
  ``LABELER_PRIOR_STRENGTH`` pseudo-labels.
* Votes are Agree / Disagree by ``validator_type == "Human"`` only; the AI account's votes and the expert accounts'
  votes (``EXPERTS``) never enter the fit; Unsure votes are dropped (the doc says why).

Fitting is EM with pseudo-count M-steps (the MAP estimate under the Beta priors):
``s_u = (sum_i q_i [Agree] + kappa mu_s) / (sum_i q_i + kappa)``, ``p_u`` likewise with ``1 - q_i`` and Disagree, and
the E-step ``logit q_i = logit prior_i + sum_votes log-likelihood ratio``. It starts from the smoothed vote share, so
the fit cannot land on the label-switched solution unnoticed; :func:`fit_ds` raises if it does
(``mu_s + mu_p <= 1``). The EM never reads a gold verdict. ``kappa`` is the only quantity chosen on gold, by the
nested cross-validation of :func:`nested_cv`: each outer fold's items are scored with the ``kappa`` that minimises
the Brier score on the other folds' items.

Stages
------
``score``
    Reads the rawLabels snapshot (refused if any file postdates the first rating day), the private gold table, the
    pool and the target scoreboard's ``verdicts.csv``; fits every ``kappa`` for both arms; writes the per-label
    posteriors of the gold items (private, refused inside a git checkout) and ``results.json`` (aggregates only, after
    the privacy gate against the live ``/v3/api/cities``).
``report``
    Renders ``results.json`` into the doc's generated blocks; ``--check`` exits 1 when the doc differs.

Keys: ``label_uid`` is ``city_id:label_id`` with a uniqueness assert after every build, and every merge passes
``validate=``. No user id or rater id is written to the repository.

Usage::

    EXPERTS="<name>=<user_id>,<name>=<user_id>" python analysis/joint_vote_model.py score \\
        --rawlabels ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels \\
        --gold ~/Data/sidewalk-validator-ai-adjudicator-2026-11/gold.csv \\
        --pool ~/Data/sidewalk-validator-ai-relabel-2026-10/relabel_pool.csv \\
        --scoreboard-verdicts ~/Data/sidewalk-validator-ai-target-scoreboard-2026-11/verdicts.csv
    python analysis/joint_vote_model.py report

Example (synthetic votes where a third of the validators agree with everything; on the labels the majority
decides, the model's verdict is right more often)::

    >>> d = synthetic(n_labels=4000, n_validators=80, votes_per_label=(2, 5), seed=1)
    >>> f = fit_ds(d["label"], d["validator"], d["agree"], d["n_labels"], d["n_validators"], d["group"], 1, kappa=5)
    >>> a = np.bincount(d["label"], d["agree"], d["n_labels"]); n = np.bincount(d["label"], minlength=d["n_labels"])
    >>> dec = 2 * a != n
    >>> round(float(((2 * a > n) == d["truth"])[dec].mean()), 3), round(float(((f["q"] > 0.5) == d["truth"])[dec].mean()), 3)
    (0.929, 0.947)
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import adjudicator as adj  # noqa: E402
from analysis import ai_disagree_relabel as rl  # noqa: E402
from analysis import slices as sl  # noqa: E402
from analysis import target_scoreboard as ts  # noqa: E402
import data_generation as dg  # noqa: E402

LABEL_TYPES = tuple(adj.LABEL_TYPES)
KAPPA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0)
"""Validator prior strengths (pseudo-votes) the nested cross-validation chooses among."""
DEPLOY_PRIOR_STRENGTH = 10.0
LABELER_PRIOR_STRENGTH = 10.0
ARMS = ("votes", "labeler")
PRIMARY_ARM = "votes"
N_FOLDS = 5
N_BOOT = 2000
RULE_MIN_GAIN = 0.04
RATED_AFTER = "2026-10-27"
SLICE_MIN_N = 20
CONF_THRESHOLDS = (0.8, 0.9, 0.95)
CLIP = 1e-4
CONSENSUS = "r2_consensus"
DEFAULT_RAWLABELS = os.path.expanduser("~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels")
DEFAULT_GOLD = os.path.expanduser("~/Data/sidewalk-validator-ai-adjudicator-2026-11/gold.csv")
DEFAULT_POOL = os.path.expanduser("~/Data/sidewalk-validator-ai-relabel-2026-10/relabel_pool.csv")
DEFAULT_VERDICTS = os.path.expanduser("~/Data/sidewalk-validator-ai-target-scoreboard-2026-11/verdicts.csv")
DEFAULT_OUT = os.path.expanduser("~/Data/sidewalk-validator-ai-joint-vote-2026-11")
DOC = os.path.join(REPO_DIR, "docs", "joint-vote-model-2026-11.md")
PUBLIC_DIR = os.path.join(REPO_DIR, "docs", "joint-vote-model-2026-11")
BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/joint_vote_model.py) -->"
END = "<!-- END GENERATED: {name} -->"
BLOCKS = ("counts", "primary", "secondary", "types", "slices", "coverage", "fit", "provenance")


# ---------------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------------

def _logit(x):
    x = np.clip(x, CLIP, 1 - CLIP)
    return np.log(x) - np.log1p(-x)


def fit_ds(label, validator, agree, n_labels, n_validators, group, n_groups, kappa,
           deploy_strength=DEPLOY_PRIOR_STRENGTH, labeler=None, n_labelers=0, labeler_strength=LABELER_PRIOR_STRENGTH,
           max_iter=1000, tol=1e-7):
    """Fit the asymmetric Dawid-Skene model by EM; returns a dict of arrays and population rates.

    ``label``, ``validator`` and ``agree`` are one entry per vote (``agree`` 1 for Agree, 0 for Disagree); ``group``
    is one entry per label (its deployment, ``0 .. n_groups - 1``). ``labeler`` (optional) is one entry per label,
    ``-1`` for a label whose labeler gets no parameter. Labels without a vote keep their prior as ``q``.

    Returned: ``q`` (posterior P(correct) per label), ``prior`` (per label), ``s`` / ``p`` (per validator),
    ``n_votes`` (per validator), ``mu_s`` / ``mu_p``, ``pi`` (per group), ``iters``, ``converged``.

    >>> f = fit_ds(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 1, 0]), 2, 2, np.array([0, 0]), 1, 10)
    >>> bool(f["q"][0] > 0.5 > f["q"][1]), f["converged"]
    (True, True)
    """
    label, validator = np.asarray(label, int), np.asarray(validator, int)
    agree = np.asarray(agree, float)
    group = np.asarray(group, int)
    if len(group) != n_labels:
        raise ValueError("group must have one entry per label")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if not len(label):
        raise ValueError("no votes to fit")
    n_lab = np.bincount(label, minlength=n_labels)
    voted = n_lab > 0
    a_lab = np.bincount(label, weights=agree, minlength=n_labels)
    q = (a_lab + 0.5) / (n_lab + 1.0)
    use_labeler = labeler is not None
    if use_labeler:
        labeler = np.asarray(labeler, int)
        has_l = labeler >= 0
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        qv = q[label]
        sa = np.bincount(validator, weights=qv * agree, minlength=n_validators)
        sc = np.bincount(validator, weights=qv, minlength=n_validators)
        pdis = np.bincount(validator, weights=(1 - qv) * (1 - agree), minlength=n_validators)
        pc = np.bincount(validator, weights=1 - qv, minlength=n_validators)
        mu_s, mu_p = sa.sum() / sc.sum(), pdis.sum() / pc.sum()
        s = np.clip((sa + kappa * mu_s) / (sc + kappa), CLIP, 1 - CLIP)
        p = np.clip((pdis + kappa * mu_p) / (pc + kappa), CLIP, 1 - CLIP)
        m = q[voted].mean()
        gq = np.bincount(group[voted], weights=q[voted], minlength=n_groups)
        gn = np.bincount(group[voted], minlength=n_groups)
        pi = (gq + deploy_strength * m) / (gn + deploy_strength)
        prior = pi[group]
        if use_labeler:
            sel = voted & has_l
            lq = np.bincount(labeler[sel], weights=q[sel], minlength=n_labelers)
            ln = np.bincount(labeler[sel], minlength=n_labelers)
            lm = np.bincount(labeler[sel], weights=prior[sel], minlength=n_labelers)
            lam = (lq + labeler_strength * np.where(ln > 0, lm / np.maximum(ln, 1), m)) / (ln + labeler_strength)
            prior = np.where(has_l, lam[np.where(has_l, labeler, 0)], prior)
        sv, pv = s[validator], p[validator]
        llr = np.where(agree == 1, np.log(sv) - np.log1p(-pv), np.log1p(-sv) - np.log(pv))
        logit_q = _logit(prior) + np.bincount(label, weights=llr, minlength=n_labels)
        q_new = 1 / (1 + np.exp(-logit_q))
        delta = np.max(np.abs(q_new - q)) if n_labels else 0.0
        q = q_new
        if delta < tol:
            converged = True
            break
    if mu_s + mu_p <= 1:
        raise ValueError(f"label-switched fit: mu_s {mu_s:.3f} + mu_p {mu_p:.3f} <= 1")
    return {"q": q, "prior": prior, "s": s, "p": p, "n_votes": np.bincount(validator, minlength=n_validators),
            "mu_s": float(mu_s), "mu_p": float(mu_p), "pi": pi, "iters": it, "converged": converged}


def synthetic(n_labels=2000, n_validators=50, votes_per_label=(1, 6), p_correct=0.7, seed=0):
    """A dataset with known truth for testing :func:`fit_ds`: a third of the validators agree with almost everything
    (sensitivity 0.97, specificity 0.25), the rest are careful (0.9, 0.85); activity is heavy-tailed, as in the
    crowd. Returns arrays ``label``, ``validator``, ``agree``, ``truth`` (per label, True = correct), ``group``
    (all 0), the true ``s`` / ``p`` and the sizes."""
    rng = np.random.default_rng(seed)
    truth = rng.random(n_labels) < p_correct
    lax = np.arange(n_validators) % 3 == 0
    s = np.where(lax, 0.97, 0.9)
    p = np.where(lax, 0.25, 0.85)
    act = rng.pareto(1.2, n_validators) + 1
    act /= act.sum()
    lab, val = [], []
    for i in range(n_labels):
        k = rng.integers(votes_per_label[0], votes_per_label[1] + 1)
        vs = rng.choice(n_validators, size=min(k, n_validators), replace=False, p=act)
        lab += [i] * len(vs)
        val += list(vs)
    lab, val = np.array(lab), np.array(val)
    pa = np.where(truth[lab], s[val], 1 - p[val])
    agree = (rng.random(len(lab)) < pa).astype(int)
    return {"label": lab, "validator": val, "agree": agree, "truth": truth, "group": np.zeros(n_labels, int),
            "s": s, "p": p, "n_labels": n_labels, "n_validators": n_validators}


# ---------------------------------------------------------------------------------------------------------------------
# Votes from the rawLabels snapshot
# ---------------------------------------------------------------------------------------------------------------------

def parse_votes(validations_json, experts):
    """``(votes, counts)`` for one ``validations`` cell: ``votes`` is ``{user_id: 1 Agree / 0 Disagree}`` over human
    votes by non-experts (a validator listed twice keeps the last vote, as ``validator_reliability.human_votes``);
    ``counts`` tallies what was left out: ``ai``, ``expert``, ``unsure``, ``duplicate``.

    >>> v = ('[{"user_id":"x","validation":"Disagree","validator_type":"AI"},'
    ...      ' {"user_id":"e","validation":"Agree","validator_type":"Human"},'
    ...      ' {"user_id":"a","validation":"Unsure","validator_type":"Human"},'
    ...      ' {"user_id":"b","validation":"Agree","validator_type":"Human"},'
    ...      ' {"user_id":"b","validation":"Disagree","validator_type":"Human"}]')
    >>> parse_votes(v, {"e"})
    ({'b': 0}, {'ai': 1, 'expert': 1, 'unsure': 1, 'duplicate': 1})
    """
    counts = {"ai": 0, "expert": 0, "unsure": 0, "duplicate": 0}
    if not isinstance(validations_json, str) or not validations_json.strip():
        return {}, counts
    votes = {}
    for v in json.loads(validations_json):
        if v.get("validator_type") != "Human":
            counts["ai"] += 1
            continue
        uid = v.get("user_id")
        if uid in experts:
            counts["expert"] += 1
            continue
        val = v.get("validation")
        if val == "Unsure":
            counts["unsure"] += 1
            continue
        if val not in ("Agree", "Disagree"):
            continue
        if uid in votes:
            counts["duplicate"] += 1
            del votes[uid]
        votes[uid] = int(val == "Agree")
    return votes, counts


def read_snapshot(rawlabels_dir, cities, types, experts):
    """``(labels, votes, counts)`` from ``<rawlabels_dir>/<city>.csv`` for ``cities``.

    ``labels``: one row per label of ``types`` with ``label_uid`` (``city:label_id``, asserted unique), ``city``,
    ``label_type``, ``labeler`` (``""`` when the labeler is an expert account), ``a`` / ``d`` (kept Agree /
    Disagree votes). ``votes``: ``label_uid``, ``user_id``, ``agree``. ``counts``: what was left out, summed."""
    lab_rows, vote_rows = [], []
    counts = {"ai": 0, "expert": 0, "unsure": 0, "duplicate": 0, "human_ad": 0}
    for city in cities:
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         usecols=["label_id", "user_id", "label_type", "validations"])
        df = df[df.label_type.isin(types)]
        for lid, labeler, lt, vj in zip(df.label_id, df.user_id, df.label_type, df.validations):
            uid = f"{city}:{lid}"
            votes, c = parse_votes(vj, experts)
            for k, v in c.items():
                counts[k] += v
            a = sum(votes.values())
            lab_rows.append((uid, city, lt, "" if labeler in experts else labeler, a, len(votes) - a))
            for u, ag in votes.items():
                vote_rows.append((uid, u, ag))
    labels = pd.DataFrame(lab_rows, columns=["label_uid", "city", "label_type", "labeler", "a", "d"])
    if not labels.label_uid.is_unique:
        raise ValueError("label_uid (city:label_id) is not unique in the snapshot")
    votes = pd.DataFrame(vote_rows, columns=["label_uid", "user_id", "agree"])
    counts["human_ad"] = int(len(votes))
    if experts and votes.user_id.isin(experts).any():
        raise AssertionError("an expert vote reached the fit")
    return labels, votes, counts


def snapshot_files_ok(rawlabels_dir, cities, rated_after=RATED_AFTER):
    """Raise ``ValueError`` when a snapshot file was written on or after ``rated_after`` (the first rating day), so
    no vote cast after an expert saw an item can be in the fit."""
    cutoff = datetime.fromisoformat(rated_after).replace(tzinfo=timezone.utc)
    for c in cities:
        t = datetime.fromtimestamp(os.path.getmtime(os.path.join(rawlabels_dir, f"{c}.csv")), timezone.utc)
        if t >= cutoff:
            raise ValueError(f"rawLabels file {c}.csv was written {t.date()}, not before the first rating day")


def fit_type(labels, votes, label_type, kappa, arm):
    """Fit one type; returns ``(fit, label_uids)`` with ``fit["q"]`` aligned to ``label_uids``."""
    lt = labels[labels.label_type == label_type].reset_index(drop=True)
    idx = pd.Series(np.arange(len(lt)), index=lt.label_uid)
    v = votes[votes.label_uid.isin(idx.index)]
    vcodes, vuniq = pd.factorize(v.user_id)
    gcodes, guniq = pd.factorize(lt.city)
    kw = {}
    if arm == "labeler":
        lc, luniq = pd.factorize(lt.labeler.where(lt.labeler != "", None))
        kw = {"labeler": lc, "n_labelers": len(luniq)}
    elif arm != "votes":
        raise ValueError(f"unknown arm {arm!r}")
    f = fit_ds(idx[v.label_uid].to_numpy(), vcodes, v.agree.to_numpy(), len(lt), len(vuniq), gcodes, len(guniq),
               kappa, **kw)
    return f, lt.label_uid.to_numpy()


def validator_summary(f, min_votes=20):
    """Population rates and the spread of the fitted per-validator rates (validators with ``>= min_votes`` votes);
    counts and quantiles only, no ids."""
    keep = f["n_votes"] >= min_votes
    q = lambda x: [float(np.quantile(x, t)) for t in (0.1, 0.5, 0.9)] if len(x) else None  # noqa: E731
    return {"mu_s": f["mu_s"], "mu_p": f["mu_p"], "n_validators": int((f["n_votes"] > 0).sum()),
            "n_validators_min_votes": int(keep.sum()), "min_votes": min_votes,
            "s_p10_p50_p90": q(f["s"][keep]), "p_p10_p50_p90": q(f["p"][keep]),
            "iters": int(f["iters"]), "converged": bool(f["converged"])}


# ---------------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------------

def fold_of(pano_id, n_folds=N_FOLDS):
    """The outer fold of a gold item, from its panorama (so a panorama's items share a fold).

    >>> fold_of("abc") == fold_of("abc"), 0 <= fold_of("abc") < N_FOLDS
    (True, True)
    """
    return int(hashlib.sha256(f"joint-vote:{pano_id}".encode()).hexdigest()[:12], 16) % n_folds


def brier(q, y):
    return float(np.mean((np.asarray(q, float) - np.asarray(y, float)) ** 2))


def nested_cv(q_by_kappa, y, fold, in_select, kappas=KAPPA_GRID):
    """``(q_cv, chosen)``: each item's posterior under the ``kappa`` that minimises the Brier score over the
    ``in_select`` items of the *other* folds (ties go to the larger ``kappa``, the stronger shrinkage); ``chosen``
    maps fold to ``kappa``. ``q_by_kappa`` maps each ``kappa`` to an array aligned with ``y``.

    >>> qk = {1.0: np.array([0.9, 0.2, 0.6, 0.4]), 10.0: np.array([0.6, 0.4, 0.9, 0.1])}
    >>> q, ch = nested_cv(qk, np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]), np.ones(4, bool), (1.0, 10.0))
    >>> ch, q.tolist()
    ({0: 10.0, 1: 1.0}, [0.6, 0.4, 0.6, 0.4])
    """
    y, fold, sel = np.asarray(y, float), np.asarray(fold), np.asarray(in_select, bool)
    q_cv = np.full(len(y), np.nan)
    chosen = {}
    for k in sorted(set(fold.tolist())):
        inner = (fold != k) & sel
        scores = [(brier(q_by_kappa[kap][inner], y[inner]), -kap) for kap in kappas]
        best = -min(scores)[1]
        chosen[int(k)] = float(best)
        q_cv[fold == k] = q_by_kappa[best][fold == k]
    return q_cv, chosen


def select_kappa(q_by_kappa, y, in_select, kappas=KAPPA_GRID):
    """The ``kappa`` minimising the Brier score over all ``in_select`` items (used only for the pool coverage, the
    value a training arm would use; never to score gold)."""
    y, sel = np.asarray(y, float), np.asarray(in_select, bool)
    return float(-min((brier(q_by_kappa[k][sel], y[sel]), -k) for k in kappas)[1])


def paired_boot(x, y, groups=None, n_boot=N_BOOT, seed=0):
    """``(diff, lo, hi)``: mean of ``x - y`` with a 95% percentile interval from ``n_boot`` resamples, of items
    (``groups`` None) or of whole groups (panorama clusters: each resample draws groups with replacement and takes
    the item mean). None for the interval when ``n_boot`` is 0.

    >>> paired_boot([1, 1, 0, 1], [0, 1, 0, 0], n_boot=0)
    (0.5, None, None)
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    if not len(d):
        return None, None, None
    point = float(d.mean())
    if not n_boot:
        return point, None, None
    rng = np.random.default_rng(seed)
    if groups is None:
        vals = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    else:
        codes, _ = pd.factorize(pd.Series(groups))
        g = codes.max() + 1
        dsum = np.bincount(codes, weights=d, minlength=g)
        cnt = np.bincount(codes, minlength=g)
        idx = rng.integers(0, g, size=(n_boot, g))
        vals = dsum[idx].sum(1) / cnt[idx].sum(1)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _resample_weights(n, groups, n_boot, rng):
    """``(n_boot, n)`` multiplicities: items drawn with replacement (``groups`` None) or whole groups."""
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        return np.bincount((np.arange(n_boot)[:, None] * n + idx).ravel(), minlength=n_boot * n).reshape(n_boot, n)
    codes, _ = pd.factorize(pd.Series(groups))
    g = codes.max() + 1
    gidx = rng.integers(0, g, size=(n_boot, g))
    wg = np.bincount((np.arange(n_boot)[:, None] * g + gidx).ravel(), minlength=n_boot * g).reshape(n_boot, g)
    return wg[:, codes]


def nested_boot(q_by_kappa, y, fold, select, ok_ref, pops, groups=None, kappas=KAPPA_GRID, n_boot=N_BOOT, seed=0):
    """95% percentile intervals of ``mean(model right - reference right)`` over each population in ``pops`` (name ->
    bool mask), with the nested ``kappa`` selection of :func:`nested_cv` redone inside every resample, so the
    interval carries the selection's own variability. Items (or ``groups``) are resampled jointly for every
    population. Returns ``{name: (lo, hi)}``."""
    if not n_boot:
        return {name: (None, None) for name in pops}
    y = np.asarray(y, float)
    fold, sel, ref = np.asarray(fold), np.asarray(select, bool), np.asarray(ok_ref, float)
    n, k_n = len(y), len(kappas)
    err = np.stack([(np.asarray(q_by_kappa[k], float) - y) ** 2 for k in kappas])           # K x n
    ok = np.stack([(np.asarray(q_by_kappa[k], float) > 0.5) == (y == 1) for k in kappas])   # K x n
    ok &= np.stack([np.asarray(q_by_kappa[k], float) != 0.5 for k in kappas])
    w = _resample_weights(n, groups, n_boot, np.random.default_rng(seed)).astype(float)
    choice = np.zeros((n_boot, n), int)
    for f in sorted(set(fold.tolist())):
        inner = (fold != f) & sel
        s = w[:, inner] @ err[:, inner].T                       # n_boot x K, summed squared error
        best = k_n - 1 - np.argmin(s[:, ::-1], axis=1)          # ties to the larger kappa
        choice[:, fold == f] = best[:, None]
    model_ok = ok[choice, np.arange(n)[None, :]].astype(float)
    out = {}
    for name, mask in pops.items():
        m = np.asarray(mask, bool)
        num = (w[:, m] * (model_ok[:, m] - ref[m][None, :])).sum(1)
        den = w[:, m].sum(1)
        vals = num[den > 0] / den[den > 0]
        out[name] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def mcnemar_p(b, c):
    """Two-sided exact McNemar p from the discordant counts ``b`` and ``c``.

    >>> mcnemar_p(0, 0), round(mcnemar_p(10, 2), 4)
    (1.0, 0.0386)
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def contrast(ok_a, ok_b, pano, n_boot=N_BOOT, seed=0):
    """Paired agreement contrast A minus B: rates, discordant counts, label-level and panorama-cluster intervals."""
    a, b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    d, lo, hi = paired_boot(a, b, None, n_boot, seed)
    _, clo, chi = paired_boot(a, b, pano, n_boot, seed)
    return {"n": int(len(a)), "a": ts.rate(a.sum(), len(a)), "b": ts.rate(b.sum(), len(b)),
            "a_only": int((a & ~b).sum()), "b_only": int((~a & b).sum()), "diff": d, "lo": lo, "hi": hi,
            "cluster_lo": clo, "cluster_hi": chi, "n_panos": int(pd.Series(pano).nunique()),
            "mcnemar_p": mcnemar_p(int((a & ~b).sum()), int((~a & b).sum()))}


def brier_contrast(q_a, q_b, y, pano, n_boot=N_BOOT, seed=0):
    """Paired Brier A minus B (negative = A closer to the expert), both intervals."""
    ea = (np.asarray(q_a, float) - y) ** 2
    eb = (np.asarray(q_b, float) - y) ** 2
    d, lo, hi = paired_boot(ea, eb, None, n_boot, seed)
    _, clo, chi = paired_boot(ea, eb, pano, n_boot, seed)
    return {"n": int(len(y)), "a": float(ea.mean()) if len(y) else None, "b": float(eb.mean()) if len(y) else None,
            "diff": d, "lo": lo, "hi": hi, "cluster_lo": clo, "cluster_hi": chi}


def verdict(q):
    """``correct`` above 0.5, ``incorrect`` below, ``""`` at exactly 0.5."""
    q = np.asarray(q, float)
    return np.where(q > 0.5, "correct", np.where(q < 0.5, "incorrect", ""))


def rule_met(primary, decisive, min_gain=RULE_MIN_GAIN):
    """The pre-registered rule: gain >= ``min_gain``, both 95% intervals above 0, and not worse on decisive items
    (point estimate >= 0). Returns ``{"met": bool, "parts": {...}}``."""
    parts = {"gain_at_least": primary["diff"] is not None and primary["diff"] >= min_gain - 1e-12,
             "label_interval_excludes_0": primary["lo"] is not None and primary["lo"] > 0,
             "cluster_interval_excludes_0": primary["cluster_lo"] is not None and primary["cluster_lo"] > 0,
             "not_worse_on_decisive": decisive["diff"] is not None and decisive["diff"] >= 0}
    return {"met": all(parts.values()), "parts": parts}


# ---------------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------------

def gold_frame(gold, labels):
    """Decided, masked-only gold items joined to the snapshot's non-expert tallies; adds ``in_snapshot``,
    ``majority2``, ``margin``, ``soft``, ``fold``, ``y`` and the populations ``primary`` (AI Disagree, majority2
    decides) and ``decisive`` (any decided item the margin rule decides)."""
    g = gold[gold.decided & (gold.expert_source != CONSENSUS)].copy()
    if not g.label_uid.is_unique:
        raise ValueError("gold label_uid is not unique")
    m = g.merge(labels[["label_uid", "a", "d"]].rename(columns={"a": "s_a", "d": "s_d"}), on="label_uid", how="left",
                validate="one_to_one", indicator=True)
    m["in_snapshot"] = m.pop("_merge") == "both"
    m["s_a"] = m.s_a.fillna(0).astype(int)
    m["s_d"] = m.s_d.fillna(0).astype(int)
    m["majority2"] = dg.prior_class(m.s_a.to_numpy(), m.s_d.to_numpy(), "majority2")
    m["margin"] = dg.prior_class(m.s_a.to_numpy(), m.s_d.to_numpy(), "margin")
    m["soft"] = (m.s_a + 1) / (m.s_a + m.s_d + 2)
    m["fold"] = [fold_of(p) for p in m.pano_id]
    m["y"] = (m.expert == "correct").astype(int)
    m["ai_disagree"] = m.ai_vote == "Disagree"
    m["primary"] = m.in_snapshot & m.ai_disagree & (m.majority2 != "")
    m["decisive"] = m.in_snapshot & (m.margin != "")
    m["round"] = m["round"].astype(str)
    return m.reset_index(drop=True)


def slice_rows(g, ok_a, ok_b, min_n=SLICE_MIN_N):
    """Type x severity and type x tag on the primary population: counts always, rates and the paired difference
    only at ``>= min_n`` items."""
    d = g.assign(ok_a=ok_a, ok_b=ok_b)
    out = {}
    for by in sl.BY:
        rows = []
        for t in LABEL_TYPES:
            x = sl.with_slice(d[d.label_type == t], by)
            for name in sl.slice_names(x, by, min_n=1):
                c = x[x.slice == name]
                r = {"label_type": t, "slice": name, "n": int(len(c))}
                if len(c) >= min_n:
                    r.update({"a": ts.rate(c.ok_a.sum(), len(c)), "b": ts.rate(c.ok_b.sum(), len(c)),
                              "diff": float(c.ok_a.mean() - c.ok_b.mean())})
                rows.append(r)
        out[by] = rows
    return out


def coverage(pool, labels, q_uid, q, schema_city):
    """Exact shares over the pool's AI Disagrees in the fitted deployments that the snapshot holds: majority2 on the
    snapshot tallies (experts removed) against the model's confident verdicts, and how often the model reverses a
    majority2 verdict or decides a tie or a single vote. Counts only."""
    p = pool[(pool.ai_vote == "Disagree") & pool.schema.isin(schema_city)].copy()
    p["label_uid"] = [f"{schema_city[s]}:{i}" for s, i in zip(p.schema, p.label_id)]
    if not p.label_uid.is_unique:
        raise ValueError("pool label_uid is not unique")
    qs = pd.DataFrame({"label_uid": q_uid, "q": q})
    m = p[["label_uid", "label_type"]].merge(labels[["label_uid", "a", "d"]], on="label_uid", how="inner",
                                             validate="one_to_one")
    m = m.merge(qs, on="label_uid", how="inner", validate="one_to_one")
    rows = []
    for t in ("all",) + LABEL_TYPES:
        x = m if t == "all" else m[m.label_type == t]
        if not len(x):
            continue
        m2 = dg.prior_class(x.a.to_numpy(), x.d.to_numpy(), "majority2")
        v = verdict(x.q.to_numpy())
        tot, net = (x.a + x.d).to_numpy(), (x.a - x.d).to_numpy()
        conf = np.maximum(x.q, 1 - x.q).to_numpy()
        rows.append({"label_type": t, "n": int(len(x)),
                     "majority2": int((m2 != "").sum()), "majority2_incorrect": int((m2 == "incorrect").sum()),
                     "model_at": {str(c): int((conf >= c).sum()) for c in CONF_THRESHOLDS},
                     "model_incorrect_at": {str(c): int(((conf >= c) & (v == "incorrect")).sum())
                                            for c in CONF_THRESHOLDS},
                     "reverses_majority2": int(((m2 != "") & (v != "") & (v != m2)).sum()),
                     "ties": int(((tot > 0) & (net == 0)).sum()),
                     "ties_at_0.8": int(((tot > 0) & (net == 0) & (conf >= 0.8)).sum()),
                     "one_vote": int((tot == 1).sum()), "one_vote_at_0.8": int(((tot == 1) & (conf >= 0.8)).sum()),
                     "no_vote": int((tot == 0).sum())})
    return {"n_pool_ai_disagree": int(len(p)), "n_in_snapshot": int(len(m)), "rows": rows}


def score(gold, labels, votes, pool, schema_city, sb_verdicts=None, kappas=KAPPA_GRID, n_boot=N_BOOT, seed=0):
    """Everything ``results.json`` holds except provenance, and the private per-label frame."""
    g = gold_frame(gold, labels)
    fits, q_by = {}, {arm: {} for arm in ARMS}
    for arm in ARMS:
        for kap in kappas:
            q_col = np.full(len(g), np.nan)
            summ = {}
            all_q = []
            for t in LABEL_TYPES:
                if not (labels.label_type == t).any():
                    continue
                f, uids = fit_type(labels, votes, t, kap, arm)
                s = pd.Series(f["q"], index=uids)
                sel = (g.label_type == t).to_numpy()
                q_col[sel] = s.reindex(g.label_uid[sel]).to_numpy()
                summ[t] = validator_summary(f)
                all_q.append((uids, f["q"]))
            # a gold item not in the snapshot has no fitted label; it is outside every scored population
            q_by[arm][kap] = q_col
            fits[(arm, kap)] = {"summary": summ, "q": all_q}
    y = g.y.to_numpy()
    prim = g.primary.to_numpy()
    q_cv, chosen = {}, {}
    for arm in ARMS:
        qk = {k: np.nan_to_num(v, nan=0.5) for k, v in q_by[arm].items()}
        q_cv[arm], chosen[arm] = nested_cv(qk, y, g.fold.to_numpy(), prim, kappas)
        g[f"q_{arm}"] = np.where(g.in_snapshot, q_cv[arm], np.nan)
        g[f"v_{arm}"] = np.where(g.in_snapshot, verdict(q_cv[arm]), "")
        for k in kappas:
            g[f"q_{arm}_k{k:g}"] = q_by[arm][k]
    gp = g[g.primary].reset_index(drop=True)
    gd = g[g.decisive].reset_index(drop=True)
    ok_m2 = (gp.majority2 == gp.expert).to_numpy()
    out = {"arms": {}, "counts": counts_block(gold, g), "kappas": list(kappas)}
    for arm in ARMS:
        ok = (gp[f"v_{arm}"] == gp.expert).to_numpy()
        prim_c = contrast(ok, ok_m2, gp.pano_id, n_boot, seed)
        dec_c = contrast((gd[f"v_{arm}"] == gd.expert).to_numpy(), (gd.majority2 == gd.expert).to_numpy(),
                         gd.pano_id, n_boot, seed)
        ins = g.in_snapshot.to_numpy()
        gs = g[ins].reset_index(drop=True)
        qk = {k: q_by[arm][k][ins] for k in kappas}
        ref = (gs.majority2 == gs.expert).to_numpy()
        pops = {"primary": gs.primary.to_numpy(), "decisive": gs.decisive.to_numpy()}
        for groups, lo_k, hi_k in ((None, "lo", "hi"), (gs.pano_id, "cluster_lo", "cluster_hi")):
            iv = nested_boot(qk, gs.y, gs.fold, gs.primary, ref, pops, groups, kappas, n_boot, seed)
            prim_c[lo_k], prim_c[hi_k] = iv["primary"]
            dec_c[lo_k], dec_c[hi_k] = iv["decisive"]
        prim_c["interval"] = dec_c["interval"] = "kappa selection redone in every resample"
        per_round = {r: contrast(ok[gp["round"] == r], ok_m2[gp["round"] == r], gp.pano_id[gp["round"] == r],
                                 n_boot, seed) for r in ("1", "2")}
        per_type = {t: contrast(ok[gp.label_type == t], ok_m2[gp.label_type == t], gp.pano_id[gp.label_type == t],
                                n_boot, seed) for t in LABEL_TYPES}
        yp = gp.y.to_numpy()
        ad = g[g.in_snapshot & g.ai_disagree & (g.majority2 == "")]
        per_kappa = {f"{k:g}": {"agree": ts.rate((verdict(q_by[arm][k][g.primary.to_numpy()]) == gp.expert).sum(),
                                                 len(gp)),
                                "brier": brier(q_by[arm][k][g.primary.to_numpy()], yp)} for k in kappas}
        out["arms"][arm] = {
            "primary": prim_c, "decisive": dec_c, "rule": rule_met(prim_c, dec_c),
            "per_round": per_round, "per_type": per_type,
            "brier_vs_soft": brier_contrast(gp[f"q_{arm}"], gp.soft, yp, gp.pano_id, n_boot, seed),
            "brier_majority2": float(1 - ok_m2.mean()) if len(ok_m2) else None,
            "undecided_by_majority2": {"n": int(len(ad)),
                                       "model": ts.rate((ad[f"v_{arm}"] == ad.expert).sum(), len(ad))},
            "slices": slice_rows(gp, ok, ok_m2),
            "chosen_kappa": {str(k): v for k, v in chosen[arm].items()},
            "per_kappa_descriptive": per_kappa,
            "fit": {f"{k:g}": fits[(arm, k)]["summary"] for k in kappas}}
        kap_all = select_kappa({k: np.nan_to_num(v, nan=0.5) for k, v in q_by[arm].items()}, y, prim, kappas)
        uids = np.concatenate([u for u, _ in fits[(arm, kap_all)]["q"]])
        qs = np.concatenate([q for _, q in fits[(arm, kap_all)]["q"]])
        out["arms"][arm]["coverage"] = {"kappa": kap_all, **coverage(pool, labels, uids, qs, schema_city)}
    if sb_verdicts is not None:
        sb = gp[["label_uid", "expert"]].merge(sb_verdicts[["label_uid", "rule_majority2", "h_agree", "h_dis"]],
                                               on="label_uid", how="left", validate="one_to_one")
        tallies_differ = int(((sb.h_agree.astype(float) != gp.s_a) | (sb.h_dis.astype(float) != gp.s_d)).sum())
        out["scoreboard_check"] = {"n": int(len(sb)), "pool_majority2": ts.rate(
            (sb.rule_majority2 == sb.expert).sum(), int((sb.rule_majority2 != "").sum())),
            "tallies_differ": tallies_differ}
    out["n_boot"], out["seed"] = n_boot, seed
    return out, g


def counts_block(gold, g):
    d = gold[gold.decided]
    return {"gold_items": int(len(gold)), "decided": int(len(d)),
            "decided_masked": int(len(g)), "in_snapshot": int(g.in_snapshot.sum()),
            "ai_disagree_in_snapshot": int((g.in_snapshot & g.ai_disagree).sum()),
            "primary": int(g.primary.sum()),
            "primary_by_round": {r: int((g.primary & (g["round"] == r)).sum()) for r in ("1", "2")},
            "primary_by_type": {t: int((g.primary & (g.label_type == t)).sum()) for t in LABEL_TYPES},
            "decisive": int(g.decisive.sum()),
            "n_panos_primary": int(g[g.primary].pano_id.nunique())}


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------

def _pts(d, lo=None, hi=None):
    if d is None:
        return "n/a"
    return f"{100 * d:+.1f} pts" + ("" if lo is None else f" [{100 * lo:+.1f}, {100 * hi:+.1f}]")


def _r(r):
    return ts._p(r)


def render_counts(j):
    c, s = j["counts"], j["snapshot"]
    lines = ["| Count | Value |", "|---|---:|",
             f"| Gold items / decided / decided masked-only | {c['gold_items']} / {c['decided']} / "
             f"{c['decided_masked']} |",
             f"| Decided masked-only items found in the snapshot | {c['in_snapshot']} |",
             f"| of which AI Disagrees | {c['ai_disagree_in_snapshot']} |",
             f"| **Primary population** (AI Disagrees that majority2 decides, snapshot tallies, experts removed) | "
             f"**{c['primary']}** on {c['n_panos_primary']} panoramas |",
             f"| primary, round 1 / round 2 | {c['primary_by_round']['1']} / {c['primary_by_round']['2']} |",
             "| primary, per type | " + ", ".join(f"{t} {n}" for t, n in c["primary_by_type"].items()) + " |",
             f"| Decisive items (any decided item the margin rule decides) | {c['decisive']} |",
             f"| Deployments fitted / labels of the five types / with a kept vote | {s['n_deployments']} / "
             f"{s['n_labels']:,} / {s['n_labels_voted']:,} |",
             f"| Kept votes (human Agree / Disagree, experts removed) | {s['human_ad']:,} |",
             f"| Left out: AI votes / expert-account votes / human Unsure / duplicate | {s['ai']:,} / {s['expert']:,} / "
             f"{s['unsure']:,} ({100 * s['unsure'] / max(1, s['unsure'] + s['human_ad']):.1f}% of non-expert human "
             f"votes) / {s['duplicate']:,} |"]
    if "scoreboard_check" in j:
        sc = j["scoreboard_check"]
        lines.append(f"| Check: majority2 on the 2026-10-27 pool tallies (the scoreboard's), same items | "
                     f"{_r(sc['pool_majority2'])}; tallies differ from the snapshot's on {sc['tallies_differ']} |")
    return "\n".join(lines)


def _contrast_row(name, c):
    return (f"| {name} | {c['n']} | {_r(c['a'])} | {_r(c['b'])} | {c['a_only']} / {c['b_only']} | "
            f"{_pts(c['diff'], c['lo'], c['hi'])} | {_pts(c['diff'], c['cluster_lo'], c['cluster_hi'])} | "
            f"{c['mcnemar_p']:.3g} |")


HEAD = ("| Rows | n | model agrees with expert | majority2 agrees | model only / majority2 only right | "
        "model minus majority2 [95%, label-level] | [95%, panorama-cluster] | McNemar p |\n"
        "|---|---:|---|---|---|---|---|---:|")


def render_primary(j):
    out = []
    for arm in ARMS:
        a = j["arms"][arm]
        role = "primary" if arm == PRIMARY_ARM else "secondary, descriptive"
        out.append(f"**Arm `{arm}`** ({role}; kappa chosen per outer fold: "
                   + ", ".join(f"fold {k} {v:g}" for k, v in sorted(a["chosen_kappa"].items())) + ")")
        out.append("")
        out.append(HEAD)
        out.append(_contrast_row("round 1 (check)", a["per_round"]["1"]))
        out.append(_contrast_row("round 2 (check)", a["per_round"]["2"]))
        out.append(_contrast_row("**pooled, primary population**" if arm == PRIMARY_ARM else "pooled, primary population",
                                 a["primary"]))
        out.append(_contrast_row("decisive items (the not-worse part of the rule)", a["decisive"]))
        r = a["rule"]
        parts = "; ".join(f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" for k, v in r["parts"].items())
        out.append("")
        out.append(f"Rule ({'applies' if arm == PRIMARY_ARM else 'reported only; this arm cannot license a training arm'}): "
                   f"**{'met' if r['met'] else 'not met'}** ({parts}).")
        out.append("")
    return "\n".join(out).rstrip()


def render_secondary(j):
    lines = ["| Arm | n | Brier, model | Brier, soft target | model minus soft [95%, label] | [95%, cluster] | "
             "majority2 error rate | model on AI Disagrees majority2 leaves undecided |",
             "|---|---:|---:|---:|---|---|---:|---|"]
    for arm in ARMS:
        a = j["arms"][arm]
        b = a["brier_vs_soft"]
        u = a["undecided_by_majority2"]
        lines.append(f"| `{arm}` | {b['n']} | {b['a']:.4f} | {b['b']:.4f} | {b['diff']:+.4f} "
                     f"[{b['lo']:+.4f}, {b['hi']:+.4f}] | [{b['cluster_lo']:+.4f}, {b['cluster_hi']:+.4f}] | "
                     f"{a['brier_majority2']:.4f} | {_r(u['model'])} |")
    lines.append("")
    lines.append("Brier against the expert verdict (1 = correct), lower is better; the soft target is "
                 "`(agree + 1) / (agree + disagree + 2)` on the same snapshot tallies, the `--target soft` training "
                 "target. majority2's error rate is its Brier score as a 0/1 forecast. The last column is "
                 "descriptive: those items are outside the primary population.")
    lines.append("")
    lines.append("Every kappa on the primary population (descriptive: this is the grid the nested selection "
                 "chooses from, read on all its items, so no row here is an estimate of anything):")
    lines.append("")
    lines.append("| kappa | `votes`: agrees with expert | `votes`: Brier | `labeler`: agrees | `labeler`: Brier |")
    lines.append("|---:|---|---:|---|---:|")
    for k in j["kappas"]:
        kk = f"{k:g}"
        v, lab = j["arms"]["votes"]["per_kappa_descriptive"][kk], j["arms"]["labeler"]["per_kappa_descriptive"][kk]
        lines.append(f"| {kk} | {_r(v['agree'])} | {v['brier']:.4f} | {_r(lab['agree'])} | {lab['brier']:.4f} |")
    return "\n".join(lines)


def render_types(j):
    lines = [HEAD.replace("| Rows |", "| Arm, type |")]
    for arm in ARMS:
        for t in LABEL_TYPES:
            c = j["arms"][arm]["per_type"][t]
            lines.append(_contrast_row(f"`{arm}`, {t}", c))
    lines.append("")
    lines.append("Descriptive. Each type is fitted on its own; a per-type interval is wide at these sizes.")
    return "\n".join(lines)


def render_slices(j):
    out = []
    for by, title in (("severity", "type x severity"), ("tag", "type x tag")):
        rows = j["arms"][PRIMARY_ARM]["slices"][by]
        shown = [r for r in rows if "a" in r]
        hidden = [r for r in rows if "a" not in r]
        out.append(f"**Arm `{PRIMARY_ARM}`, {title}** (primary population; rates at >= {SLICE_MIN_N} items; "
                   f"{len(shown)} of {len(rows)} cells reach it)")
        out.append("")
        out.append("| Type | Slice | n | model agrees | majority2 agrees | model minus majority2 |")
        out.append("|---|---|---:|---|---|---|")
        for r in shown:
            out.append(f"| {r['label_type']} | {r['slice']} | {r['n']} | {_r(r['a'])} | {_r(r['b'])} | "
                       f"{_pts(r['diff'])} |")
        out.append("")
        out.append(f"Below {SLICE_MIN_N} items: {len(hidden)} cells holding {sum(r['n'] for r in hidden)} "
                   "item-slices (counts in `results.json`).")
        out.append("")
    return "\n".join(out).rstrip()


def render_coverage(j):
    out = []
    for arm in ARMS:
        cv = j["arms"][arm]["coverage"]
        out.append(f"**Arm `{arm}`** (kappa {cv['kappa']:g}, the value the selection picks on all primary items; "
                   f"{cv['n_in_snapshot']:,} of the pool's {cv['n_pool_ai_disagree']:,} AI Disagrees in the fitted "
                   "deployments are in the snapshot)")
        out.append("")
        out.append("| Type | AI Disagrees | majority2 decides (incorrect) | model >= 0.8 (incorrect) | model >= 0.9 | "
                   "model >= 0.95 | model reverses majority2 | ties / model >= 0.8 on them | one vote / model >= 0.8 |")
        out.append("|---|---:|---|---|---|---|---:|---|---|")
        for r in cv["rows"]:
            n = r["n"]
            pc = lambda k: f"{100 * k / n:.1f}%"  # noqa: E731
            out.append(f"| {r['label_type']} | {n:,} | {pc(r['majority2'])} ({pc(r['majority2_incorrect'])}) | "
                       f"{pc(r['model_at']['0.8'])} ({pc(r['model_incorrect_at']['0.8'])}) | "
                       f"{pc(r['model_at']['0.9'])} | {pc(r['model_at']['0.95'])} | {r['reverses_majority2']:,} | "
                       f"{r['ties']:,} / {r['ties_at_0.8']:,} | {r['one_vote']:,} / {r['one_vote_at_0.8']:,} |")
        out.append("")
    out.append("Shares of the type's AI Disagrees; model >= t means max(q, 1 - q) >= t. Exact counts over the pool, "
               "no interval, no gold verdict.")
    return "\n".join(out)


def render_fit(j):
    lines = ["| Arm | Type | kappa | validators (>= 20 votes) | mu_s | mu_p | s p10 / p50 / p90 | p p10 / p50 / p90 | "
             "EM iterations |", "|---|---|---:|---|---:|---:|---|---|---:|"]
    for arm in ARMS:
        kap = j["arms"][arm]["coverage"]["kappa"]
        for t in LABEL_TYPES:
            f = j["arms"][arm]["fit"][f"{kap:g}"].get(t)
            if f is None:
                continue
            fq = lambda x: "n/a" if x is None else " / ".join(f"{v:.2f}" for v in x)  # noqa: E731
            lines.append(f"| `{arm}` | {t} | {kap:g} | {f['n_validators']:,} ({f['n_validators_min_votes']:,}) | "
                         f"{f['mu_s']:.3f} | {f['mu_p']:.3f} | {fq(f['s_p10_p50_p90'])} | {fq(f['p_p10_p50_p90'])} | "
                         f"{f['iters']}{'' if f['converged'] else ' (not converged)'} |")
    lines.append("")
    lines.append("mu_s = population P(Agree | correct), mu_p = population P(Disagree | incorrect); the quantiles are "
                 "over validators with at least 20 votes of the type.")
    return "\n".join(lines)


def render_provenance(j):
    p = j["provenance"]
    return "\n".join(["| Item | Pin |", "|---|---|",
                      f"| gold.csv | `{p['gold_sha256']}` (matches the pin in `docs/adjudicator-2026-11.md`: "
                      f"{'yes' if p['gold_matches_pin'] else 'NO'}) |",
                      f"| rawLabels snapshot (round 3's, sha256 of the sorted per-file list) | "
                      f"`{p['snapshot_sha256']}` ({p['n_snapshot_files']} files, every one written before "
                      f"{RATED_AFTER}) |",
                      f"| relabel_pool.csv | `{p['pool_sha256']}` |",
                      f"| scoreboard verdicts.csv (private) | `{p['verdicts_sha256']}` |",
                      f"| gold_posteriors.csv (private) | `{p['posteriors_sha256']}` |",
                      f"| expert accounts left out | {p['n_experts']} (from `EXPERTS`; ids not recorded) |",
                      f"| code | commit `{p['git_commit']}`; Python {p['python']}, pandas {p['pandas']}, "
                      f"numpy {p['numpy']} |",
                      f"| intervals | {j['n_boot']} resamples (labels, and panorama clusters), seed {j['seed']} |",
                      f"| public-deployment list | live `/v3/api/cities`, read {p['cities_read_at']} "
                      f"({p['n_public']} public) |"])


RENDER = {"counts": render_counts, "primary": render_primary, "secondary": render_secondary, "types": render_types,
          "slices": render_slices, "coverage": render_coverage, "fit": render_fit, "provenance": render_provenance}


def render_blocks(j):
    return {name: RENDER[name](j) for name in BLOCKS}


def splice(text, blocks):
    """Replace each generated block; a missing block raises.

    >>> t = "a\\n" + BEGIN.format(name="x") + "\\nold\\n" + END.format(name="x") + "\\nb"
    >>> splice(t, {"x": "new"}).splitlines()[3]
    'new'
    """
    for name, body in blocks.items():
        pat = re.compile(re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: BEGIN.format(name=name) + "\n\n" + body + "\n\n" + END.format(name=name), text)
    return text


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def _dir_sha(d, files):
    lines = "".join(f"{rl.sha256_file(os.path.join(d, f))}  {f}\n" for f in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def cmd_score(args):
    ts.refuse_private_out(args.out_dir, "the per-label posteriors")
    experts = dg.parse_experts(os.environ.get("EXPERTS"))
    if len(experts) < 2:
        sys.exit("EXPERTS must name both expert accounts (EXPERTS=\"<name>=<user_id>,<name>=<user_id>\"): their votes "
                 "are left out of every fit, and without them the gold raters' own votes would be crowd evidence")
    gold = adj.load_gold(args.gold)
    cities = sorted(set(gold.city_id))
    snapshot_files_ok(args.rawlabels, cities)
    schema_city = dict(zip(gold.schema, gold.city_id))
    if len(schema_city) != len(set(gold.city_id)):
        raise ValueError("gold schema and city_id do not map one to one")
    labels, votes, counts = read_snapshot(args.rawlabels, cities, LABEL_TYPES, experts)
    pool = ts.load_pool(args.pool)
    sbv = pd.read_csv(args.scoreboard_verdicts, dtype=str, keep_default_na=False) if args.scoreboard_verdicts else None
    lists = ts.city_lists()
    j, g = score(gold, labels, votes, pool, schema_city, sbv, KAPPA_GRID, args.n_boot, args.seed)
    j["snapshot"] = {**counts, "n_deployments": len(cities), "n_labels": int(len(labels)),
                     "n_labels_voted": int(((labels.a + labels.d) > 0).sum())}
    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["label_uid", "round", "label_type", "stratum", "ai_vote", "expert", "in_snapshot", "s_a", "s_d",
            "majority2", "margin", "soft", "fold", "primary", "decisive"] + [c for c in g.columns if c.startswith(("q_", "v_"))]
    ppath = os.path.join(args.out_dir, "gold_posteriors.csv")
    g[cols].sort_values("label_uid").to_csv(ppath, index=False, lineterminator="\n")
    files = [f"{c}.csv" for c in cities]
    gold_sha = rl.sha256_file(args.gold)
    j["provenance"] = {"gold_sha256": gold_sha, "gold_matches_pin": gold_sha == ts.GOLD_SHA256,
                       "snapshot_sha256": _dir_sha(args.rawlabels, files), "n_snapshot_files": len(files),
                       "pool_sha256": rl.sha256_file(args.pool),
                       "verdicts_sha256": rl.sha256_file(args.scoreboard_verdicts) if args.scoreboard_verdicts else None,
                       "posteriors_sha256": rl.sha256_file(ppath), "n_experts": len(experts),
                       "git_commit": ts._git_commit(), "python": platform.python_version(), "pandas": pd.__version__,
                       "numpy": np.__version__, "cities_read_at": lists["read_at"], "n_public": len(lists["public"])}
    text = ts.dumps(j)
    all_cities = {c: s for s, c in rl.cityparams(open(args.cityparams).read())[0].items()} \
        if os.path.exists(args.cityparams) else {c: s for s, c in schema_city.items()}
    try:
        ts.privacy_gate(text + "\n".join(render_blocks(j).values()), lists, all_cities)
    except ValueError as e:
        sys.exit(f"refusing to write results.json: {e}")
    if any(u in text for u in experts):
        sys.exit("refusing to write results.json: an expert id is in it")
    run = {"argv": [ts._home(a) if os.path.exists(a) else a for a in sys.argv],
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "inputs": {"rawlabels": ts._home(args.rawlabels), "gold": [ts._home(args.gold), gold_sha],
                      "pool": [ts._home(args.pool), j["provenance"]["pool_sha256"]]},
           "public_cities": sorted(lists["public"]), "provenance": j["provenance"]}
    adj.write_json(os.path.join(args.out_dir, "run.json"), run)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        f.write(text)
    print(render_blocks(j)["primary"])


def cmd_report(args):
    j = json.load(open(args.json))
    doc = open(args.doc).read()
    new = splice(doc, render_blocks(j))
    if args.check:
        if new != doc:
            sys.exit(f"{args.doc} differs from what {args.json} renders; run report without --check")
        print("doc matches results.json")
        return
    if os.path.exists(args.cityparams):
        all_cities = {c: s for s, c in rl.cityparams(open(args.cityparams).read())[0].items()}
        try:
            ts.privacy_gate(new, ts.city_lists(), all_cities)
        except ValueError as e:
            sys.exit(f"refusing to write the doc: {e}")
    with open(args.doc, "w") as f:
        f.write(new)
    print(f"spliced {len(BLOCKS)} blocks into {args.doc}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="fit, score against gold, write results.json (aggregates) and private posteriors")
    s.add_argument("--rawlabels", default=DEFAULT_RAWLABELS, help="rawLabels snapshot dir (<city_id>.csv)")
    s.add_argument("--gold", default=DEFAULT_GOLD)
    s.add_argument("--pool", default=DEFAULT_POOL)
    s.add_argument("--scoreboard-verdicts", default=DEFAULT_VERDICTS,
                   help="target_scoreboard's verdicts.csv (the pool-tally majority2 check); '' to skip")
    s.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    s.add_argument("--out-dir", default=DEFAULT_OUT, help="private outputs (refused inside a git checkout)")
    s.add_argument("--out", default=PUBLIC_DIR, help="where results.json goes")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_score)
    r = sub.add_parser("report", help="splice the generated blocks into the doc")
    r.add_argument("--json", default=os.path.join(PUBLIC_DIR, "results.json"))
    r.add_argument("--doc", default=DOC)
    r.add_argument("--cityparams", default=rl.DEFAULT_CITYPARAMS)
    r.add_argument("--check", action="store_true")
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
