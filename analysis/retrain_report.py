"""Tables and figures for ``docs/retrain-2026-10.md``, the round-3 retrain report.

Reads, for each label type, the ``evaluate.py`` outputs of one training run (``predictions.csv``, ``metrics.json``)
next to its ``split.csv`` (for the city, panorama and class of every row), the calibration maps that
``python -m calibration.calibrate`` wrote under ``<Type>/calibration/`` (the shipped maps, and the ``uncapped/``
variants), and ``deployment_priors.json``, and writes to ``--out-dir``:

* ``headline.csv``: per type and split (``test`` = the training cities' test share, ``test_heldout`` = cities the
  model never saw): n, AUC with a 95% interval from resampling panoramas, accuracy at 0.5, the within-city AUC and
  the test minus held-out AUC difference with its interval.
* ``per_city.csv``: per type, split and deployment: n, AUC with a panorama-resampled interval (blank when the
  smaller class has fewer than ``--min-minority`` labels). Public deployments are named by ``city_id``; private
  test deployments are pooled into one row; private held-out deployments are called "private city A (outside the
  US)", "private city B (US)", and so on.
* ``class_share.csv``: the incorrect share per deployment in ``train`` and in the held-out split, the city-class
  confound the report discusses.
* ``recall_at_precision.csv`` and ``hard.csv`` (with a class-stratified label bootstrap interval for the hard
  slice): from ``metrics.json`` and ``predictions.csv``.
* ``production_comparison.csv``: round 3 against the production validators on the 1,988-crop held-out sample
  (``--sample`` and ``--production-scores``, from ``analysis/vlm_baseline.py baseline``): paired bootstraps by
  label (within class) and by panorama, and within-city AUC for both.
* ``calibration.csv``: per type, the 0.92-gate coverage and precision of the shipped Agree maps on ``test`` and
  ``test_heldout`` under the file prior, read from the calibration reports, next to the same precision under each
  deployment's own class mix (below), what the uncapped Disagree maps would do, and the ``uncapped/fit-test``
  variant (maps fitted on ``test`` instead of ``val``).
* ``alt_prior.csv``: the same maps refitted under ``--alt-prior`` (a second deployment prior, for sensitivity),
  with ``calibration.calibrate``.
* ``gate_per_city.csv``: the shipped Agree maps applied per deployment: gated votes, coverage, and precision three
  ways (pooled prior, the deployment's own mix, raw), under the file prior and under ``--alt-prior``.
* ``saturation.csv``: per type, the confidence at which the Agree map clears the gate and the share of test Agree
  verdicts at confidence exactly 1.0 (float32 softmax saturation).
* ``auc_by_type.png``, ``heldout_per_city.png``, ``tables.md`` (every table as Markdown) and ``provenance.json``.

**Naming.** A deployment is named only if it is public. By default the public list is read from the live
``/v3/api/cities`` (``CITIES_API``) at run time, because visibility changes; ``--public-cities`` overrides it with
a saved ``/v3/api/cities`` response or a text file of one ``city_id`` per line. The list used is recorded in
``provenance.json``.

**Two weightings of the gate precision.** ``calibration.calibrate`` weights every eval row by
``prior(class) / split_mix(class)`` so the pooled gate statistics hold under the deployment prior. Applied per
deployment, that weighting erases the deployment's own class mix, which is the question "what precision would
this deployment see". Because ``data_generation.select_training_labels`` keeps correct labels by one global
hash-rank cutoff, every deployment lost the same share of its correct labels, so weighting every correct row by
``prior(correct) / selected_mix(correct)`` (over the whole ``split.csv``) and every incorrect row by the incorrect
counterpart restores each deployment's own mix under the margin rule. ``precision_city_own`` is the gated
precision under that weighting; ``precision_pooled`` is ``calibrate.py``'s; ``precision_raw`` is unweighted.

Usage::

    python analysis/retrain_report.py --run-dir ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 \\
        --sample ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/sample.csv \\
        --production-scores ~/Data/sidewalk-validator-ai-vlm-2026-10/scores/production.csv \\
        --alt-prior Crosswalk=0.04,CurbRamp=0.06,NoCurbRamp=0.20,Obstacle=0.22,SurfaceProblem=0.15 \\
        --out-dir docs/retrain-2026-10

Example (the within-city AUC)::

    >>> y = [0, 1] * 10 + [1] * 10; s = list(range(20)) + [100] * 10; c = ["a"] * 20 + ["b"] * 10
    >>> within_city_auc(y, s, c)   # city b has one class, so only city a counts
    (0.55, 1)
"""
import argparse
import datetime
import json
import os
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import _isotonic_map as im  # noqa: E402
from calibration import calibrate as cal  # noqa: E402

TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]
SPLITS = ["test", "test_heldout"]
PRECISION_LEVELS = ["0.92", "0.95"]
CITIES_API = "https://sidewalk-teaneck.cs.washington.edu/v3/api/cities"   # as data_generation.CITIES_API
POOLED_PRIVATE = "other (private deployments)"


def public_cities(path=None):
    """``(set_of_public_city_ids, source)``: from a saved ``/v3/api/cities`` response, a text file of ids, or the
    live API when ``path`` is None."""
    if path is None:
        with urllib.request.urlopen(CITIES_API, timeout=60) as r:
            d = json.load(r)
        return {c["city_id"] for c in d["cities"] if c.get("visibility") == "public"}, CITIES_API
    with open(path) as f:
        text = f.read()
    try:
        d = json.loads(text)
        return {c["city_id"] for c in d["cities"] if c.get("visibility") == "public"}, path
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}, path


def heldout_names(heldout_cities, public, us_cities):
    """Display names for the held-out deployments: public ones by ``city_id``; private ones lettered, those outside
    the US first, so that with one of each they are "private city A (outside the US)" and "private city B (US)".

    >>> heldout_names(["x", "y", "z"], {"x"}, {"x", "y"})
    {'x': 'x', 'z': 'private city A (outside the US)', 'y': 'private city B (US)'}
    """
    names = {c: c for c in heldout_cities if c in public}
    private = sorted([c for c in heldout_cities if c not in public], key=lambda c: (c in us_cities, c))
    for i, c in enumerate(private):
        names[c] = f"private city {chr(ord('A') + i)} ({'US' if c in us_cities else 'outside the US'})"
    return names


def load_type(run_dir, label_type):
    pred = pd.read_csv(f"{run_dir}/{label_type}/predictions.csv")
    split = pd.read_csv(f"{run_dir}/{label_type}/split.csv", usecols=[
        "label_uid", "city", "pano_id", "label_class", "class_source", "hard", "split"])
    assert pred.label_uid.is_unique and split.label_uid.is_unique
    df = pred.merge(split.drop(columns=["split"]), on="label_uid", how="left", validate="one_to_one")
    assert df.city.notna().all(), f"{label_type}: predictions without a split row"
    assert (df.true == df.label_class).all(), f"{label_type}: predictions.csv and split.csv disagree on the class"
    df["y"] = (df.true == "incorrect").astype(int)
    df["verdict"], df["confidence"] = im.predict_verdicts(df.p_correct.values, df.p_incorrect.values)
    df["hit"] = (df.verdict == df.true).astype(int)
    return df, split


def within_city_auc(y, score, city, min_n=10):
    """City-stratified AUC (as in ``analysis/vlm_baseline.py``): per-city AUCs averaged with weights n_pos x n_neg,
    over cities with both classes and >= ``min_n`` rows. ``(auc, n_cities)``; NaN if none."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(score, float), "c": np.asarray(city)})
    aucs, ws = [], []
    for _, g in d.groupby("c"):
        if len(g) < min_n or g.y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(g.y, g.s))
        ws.append(int(g.y.sum()) * int((1 - g.y).sum()))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=ws)), len(aucs)


def cluster_boot(y, scores, groups, n_boot, seed, max_tries=20):
    """AUC of each score column in ``scores`` (``n x k``) on resamples of whole ``groups`` (panoramas), as an
    ``n_boot x k`` array; draws with one class are redrawn."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    names, inverse = np.unique(np.asarray(groups), return_inverse=True)
    index = [np.flatnonzero(inverse == i) for i in range(len(names))]
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, scores.shape[1]), np.nan)
    for b in range(n_boot):
        for _ in range(max_tries):
            idx = np.concatenate([index[i] for i in rng.integers(0, len(names), len(names))])
            if 0 < y[idx].sum() < len(idx):
                out[b] = [roc_auc_score(y[idx], scores[idx, k]) for k in range(scores.shape[1])]
                break
    return out


def label_boot(y, scores, n_boot, seed):
    """Paired, class-stratified label bootstrap of the AUC of each score column (as ``vlm_baseline.paired_bootstrap_auc``)."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, scores.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out[b] = [roc_auc_score(y[idx], scores[idx, k]) for k in range(scores.shape[1])]
    return out


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def display_test(city_series, public):
    return city_series.where(city_series.isin(public), POOLED_PRIVATE)


def headline_rows(df, label_type, args):
    rows, boots = [], {}
    for split in SPLITS:
        g = df[df.split == split]
        b = cluster_boot(g.y, g.p_incorrect, g.pano_id, args.n_boot, args.seed)[:, 0]
        boots[split] = b
        lo, hi = ci(b)
        wc, nc = within_city_auc(g.y, g.p_incorrect, g.city, args.within_min_n)
        pred = (g.p_incorrect >= 0.5).astype(int)
        tpr, tnr = float((pred[g.y == 1] == 1).mean()), float((pred[g.y == 0] == 0).mean())
        rows.append({"label_type": label_type, "split": split, "n": len(g),
                     "n_correct": int((g.y == 0).sum()), "n_incorrect": int(g.y.sum()),
                     "n_panos": g.pano_id.nunique(), "n_cities": g.city.nunique(),
                     "auc": roc_auc_score(g.y, g.p_incorrect), "auc_lo": lo, "auc_hi": hi,
                     "accuracy": float((pred == g.y).mean()), "balanced_accuracy": (tpr + tnr) / 2,
                     "within_city_auc": wc, "within_city_n_cities": nc})
    dlo, dhi = ci(boots["test"] - boots["test_heldout"])
    for r in rows:
        r["diff_test_minus_heldout"] = rows[0]["auc"] - rows[1]["auc"]
        r["diff_lo"], r["diff_hi"] = dlo, dhi
    return rows


def per_city_rows(df, label_type, public, names, args):
    rows = []
    for split in SPLITS:
        g = df[df.split == split]
        disp = display_test(g.city, public) if split == "test" else g.city.map(names)
        g = g.assign(display=disp.values)
        for city, h in g.groupby("display"):
            pooled = city == POOLED_PRIVATE
            if split == "test" and not pooled and len(h) < args.min_city_n:
                continue
            n_pos, n_neg = int(h.y.sum()), int((h.y == 0).sum())
            row = {"label_type": label_type, "split": split, "city": city, "n": len(h),
                   "n_correct": n_neg, "n_incorrect": n_pos, "n_panos": h.pano_id.nunique(),
                   "n_deployments": h.city.nunique() if pooled else 1,
                   "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                   "accuracy": float(((h.p_incorrect >= 0.5).astype(int) == h.y).mean())}
            if n_pos and n_neg:
                row["auc"] = roc_auc_score(h.y, h.p_incorrect)
                if min(n_pos, n_neg) >= args.min_minority and len(h) >= args.min_city_n:
                    row["auc_lo"], row["auc_hi"] = ci(cluster_boot(h.y, h.p_incorrect, h.pano_id,
                                                                   args.n_boot, args.seed)[:, 0])
            rows.append(row)
    return rows


def class_share_rows(split, label_type, public, names, min_n):
    rows = []
    for sp in ["train", "test_heldout"]:
        g = split[split.split == sp]
        disp = display_test(g.city, public) if sp == "train" else g.city.map(names)
        g = g.assign(display=disp.values)
        for city, h in g.groupby("display"):
            if sp == "train" and city != POOLED_PRIVATE and len(h) < min_n:
                continue
            rows.append({"label_type": label_type, "split": sp, "city": city, "n": len(h),
                         "share_of_split": len(h) / len(g),
                         "incorrect_share": float((h.label_class == "incorrect").mean())})
    return rows


def metrics_rows(run_dir, label_type, df, args):
    with open(f"{run_dir}/{label_type}/metrics.json") as f:
        m = json.load(f)
    rp = [{"label_type": label_type, "split": split, "class": cls, "precision": float(lvl),
           "recall": m[split]["recall_at_precision"][cls][lvl]["recall"],
           "threshold": m[split]["recall_at_precision"][cls][lvl]["threshold"]}
          for split in SPLITS for cls in ["correct", "incorrect"] for lvl in PRECISION_LEVELS]
    hard = []
    h = m.get("hard")
    if h:
        hs = df[(df.hard == 1) & df.split.isin(SPLITS)]
        assert len(hs) == h["n"], f"{label_type}: hard slice {len(hs)} rows here, {h['n']} in metrics.json"
        lo, hi = (np.nan, np.nan)
        if hs.y.nunique() == 2 and min(int(hs.y.sum()), int((hs.y == 0).sum())) >= args.min_minority:
            lo, hi = ci(label_boot(hs.y, hs.p_incorrect, args.n_boot, args.seed)[:, 0])
        hard.append({"label_type": label_type, "n": h["n"], "n_correct": h["n_by_class"]["correct"],
                     "n_incorrect": h["n_by_class"]["incorrect"], "auc": h.get("auc"), "auc_lo": lo, "auc_hi": hi,
                     "accuracy": h["accuracy"], "balanced_accuracy": h["balanced_accuracy"]})
    return rp, hard


def production_rows(preds, sample_path, scores_path, args):
    sample = pd.read_csv(sample_path, usecols=["label_uid", "city", "pano_id", "label_type", "split"])
    prod = pd.read_csv(scores_path, usecols=["label_uid", "p_incorrect"]).rename(columns={"p_incorrect": "p_prod"})
    r3 = pd.concat([p[["label_uid", "p_incorrect", "y"]] for p in preds.values()]).rename(columns={"p_incorrect": "r3"})
    df = sample.merge(prod, on="label_uid", validate="one_to_one").merge(r3, on="label_uid", validate="one_to_one")
    assert len(df) == len(sample), f"production comparison: {len(sample) - len(df)} sample rows without both scores"
    rows = []
    groups = [(t, s, g) for (t, s), g in df.groupby(["label_type", "split"])]
    groups += [("all (pooled)", s, g) for s, g in df.groupby("split")]
    for t, s, g in groups:
        sc = g[["r3", "p_prod"]].to_numpy()
        lb = label_boot(g.y, sc, args.n_boot, args.seed)
        pb = cluster_boot(g.y, sc, g.pano_id, args.n_boot, args.seed)
        w3, _ = within_city_auc(g.y, g.r3, g.city, args.within_min_n)
        wp, nc = within_city_auc(g.y, g.p_prod, g.city, args.within_min_n)
        a3, ap = roc_auc_score(g.y, g.r3), roc_auc_score(g.y, g.p_prod)
        rows.append({"label_type": t, "split": s, "n": len(g), "n_panos": g.pano_id.nunique(),
                     "n_deployments": g.city.nunique(), "auc_round3": a3, "auc_production": ap, "diff": a3 - ap,
                     "diff_label_lo": ci(lb[:, 0] - lb[:, 1])[0], "diff_label_hi": ci(lb[:, 0] - lb[:, 1])[1],
                     "diff_pano_lo": ci(pb[:, 0] - pb[:, 1])[0], "diff_pano_hi": ci(pb[:, 0] - pb[:, 1])[1],
                     "within_city_round3": w3, "within_city_production": wp, "within_city_n_cities": nc})
    return rows


# ---------------------------------------------------------------- calibration: maps, priors, gate per deployment

def parse_prior(spec):
    """``"CurbRamp=0.06,Crosswalk=0.04"`` -> ``{"CurbRamp": {"correct": 0.94, "incorrect": 0.06}, ...}``.

    >>> parse_prior("CurbRamp=0.06")
    {'CurbRamp': {'correct': 0.94, 'incorrect': 0.06}}
    """
    out = {}
    for part in spec.split(","):
        t, s = part.split("=")
        s = float(s)
        out[t.strip()] = {"correct": 1 - s, "incorrect": s}
    return out


def load_map(run_dir, label_type, variant=""):
    d = os.path.join(run_dir, label_type, "calibration", variant)
    fn = [x for x in os.listdir(d) if x.endswith(".json") and "report" not in x and "provenance" not in x]
    assert len(fn) == 1, f"{d}: expected one map, found {fn}"
    with open(os.path.join(d, fn[0])) as f:
        amap = json.load(f)
    with open(os.path.join(d, "calibration_report.json")) as f:
        rep = json.load(f)
    return amap, rep


def clears_at(class_map, gate=im.GATE):
    """Lowest grid confidence whose map value reaches the gate, as the map's string key, or ``"never"``."""
    keys = sorted(class_map, key=float)
    above = [k for k in keys if class_map[k] >= gate]
    return above[0] if above else "never"


def city_own_weights(split_all, true, pi):
    """Weights that restore each deployment's own class mix under the margin rule: ``pi(class) / selected_mix(class)``
    with the mix taken over the whole selected set (``split.csv``, every split), since the correct-class downsample
    used one global cutoff (see the module docstring)."""
    mix = split_all.label_class.value_counts(normalize=True)
    return np.asarray(true.map(lambda c: pi[c] / mix[c]), float)


def served_for(df, amap):
    served = np.zeros(len(df))
    for v in ("correct", "incorrect"):
        m = (df.verdict == v).values
        served[m] = im.api_lookup_many(amap[v], df.confidence.values[m])
    return served


def gate_rows(df, split_all, amap, pi, prior_name, label_type, public, names, min_n):
    """Agree gate statistics per deployment under ``amap``: pooled-prior weighting (``calibrate.py``'s), the
    deployment's own mix, and raw."""
    rows = []
    for split in SPLITS:
        g = df[df.split == split].copy()
        g["w_pooled"] = np.asarray(cal.prior_weights(g["true"], pi))
        g["w_own"] = city_own_weights(split_all, g["true"], pi)
        g["served"] = served_for(g, amap)
        g["gated"] = g.served >= im.GATE
        g["display"] = (display_test(g.city, public) if split == "test" else g.city.map(names)).values
        for city, h in g.groupby("display"):
            a = h[h.verdict == "correct"]
            if split == "test" and city != POOLED_PRIVATE and len(a) < min_n:
                continue
            p = a[a.gated]
            rows.append({"label_type": label_type, "prior": prior_name, "split": split, "city": city,
                         "city_incorrect_share": float(h.y.mean()), "n_agree": len(a), "gated": len(p),
                         "n_wrong": int((1 - p.hit).sum()) if len(p) else 0,
                         "coverage_pooled": float(p.w_pooled.sum() / a.w_pooled.sum()) if len(a) else np.nan,
                         "coverage_city_own": float(p.w_own.sum() / a.w_own.sum()) if len(a) else np.nan,
                         "precision_pooled": float(np.average(p.hit, weights=p.w_pooled)) if len(p) else np.nan,
                         "precision_city_own": float(np.average(p.hit, weights=p.w_own)) if len(p) else np.nan,
                         "precision_raw": float(p.hit.mean()) if len(p) else np.nan})
    return rows


def pooled_city_own(df, split_all, amap, pi):
    """Pooled Agree coverage / precision per split under the deployment's-own-mix weighting."""
    out = {}
    for split in SPLITS:
        g = df[df.split == split].copy()
        w = city_own_weights(split_all, g["true"], pi)
        served = served_for(g, amap)
        a = (g.verdict == "correct").values
        s = im.gate_summary(served[a], g.hit.values[a], w[a])
        out[split] = (s["coverage"], s["precision"], s["n_pass"])
    return out


def calibration_rows(run_dir, label_type, df, split_all, pi):
    """One row per type from the calibration reports (shipped and uncapped) plus the city-own pooled precision."""
    amap, rep = load_map(run_dir, label_type)
    _, rep_h = load_map(run_dir, label_type, "eval-heldout")
    umap, rep_u = load_map(run_dir, label_type, "uncapped")
    _, rep_uh = load_map(run_dir, label_type, "uncapped/eval-heldout")
    _, rep_ft = load_map(run_dir, label_type, "uncapped/fit-test")
    own = pooled_city_own(df, split_all, amap, pi)

    def e(r, v):
        x = r["eval"][v]
        return x["gate_coverage"], x["gate_precision"], x["gate_n"]
    row = {"label_type": label_type, "prior_incorrect_share": rep["prior_deploy"]["incorrect"],
           "agree_clears_at": clears_at(amap["correct"]),
           "disagree_map_max_shipped": max(amap["incorrect"].values()),
           "disagree_map_max_uncapped": max(umap["incorrect"].values()),
           "disagree_clears_at_uncapped": clears_at(umap["incorrect"])}
    for name, r in [("test", rep), ("heldout", rep_h)]:
        c, p, n = e(r, "correct")
        row[f"agree_{name}_coverage"], row[f"agree_{name}_precision"], row[f"agree_{name}_n"] = c, p, n
        row[f"agree_{name}_ece"] = r["eval"]["correct"]["ece"]
        oc, op, on = own["test" if name == "test" else "test_heldout"]
        row[f"agree_{name}_precision_city_own"], row[f"agree_{name}_coverage_city_own"] = op, oc
    for name, r in [("test", rep_u), ("heldout", rep_uh)]:
        c, p, n = e(r, "incorrect")
        row[f"disagree_uncapped_{name}_coverage"], row[f"disagree_uncapped_{name}_precision"], row[f"disagree_uncapped_{name}_n"] = c, p, n
    for v, name in [("correct", "agree"), ("incorrect", "disagree")]:
        c, p, n = e(rep_ft, v)
        row[f"{name}_fit_test_heldout_coverage"], row[f"{name}_fit_test_heldout_precision"], row[f"{name}_fit_test_heldout_n"] = c, p, n
    return row


def alt_prior_rows(df, label_type, pi_alt):
    """Refit the maps under another prior with ``calibration.calibrate`` and report the Agree / Disagree gate on
    both eval splits. Returns ``(rows, api_map_fit_on_val)``."""
    pred = df[["label_uid", "split", "true", "p_correct", "p_incorrect"]]
    rows, amap = [], None
    for split in SPLITS:
        m, rep, _ = cal.calibrate(pred, pi_alt, "val", split)
        amap = amap or m
        row = {"label_type": label_type, "eval_split": split, "prior_incorrect_share": pi_alt["incorrect"]}
        for v, name in [("correct", "agree"), ("incorrect", "disagree")]:
            x = rep["eval"][v]
            row[f"{name}_coverage"], row[f"{name}_precision"], row[f"{name}_n"] = x["gate_coverage"], x["gate_precision"], x["gate_n"]
            row[f"{name}_clears_at"] = clears_at(m[v])
        rows.append(row)
    return rows, amap


def saturation_row(df, label_type, amap):
    g = df[(df.split == "test") & (df.verdict == "correct")]
    d = df[(df.split == "test") & (df.verdict == "incorrect")]
    return {"label_type": label_type, "agree_clears_at": clears_at(amap["correct"]),
            "map_value_at_0.999999": amap["correct"].get("0.999999"),
            "n_test_agree": len(g), "share_agree_at_1": float((g.confidence >= 1.0).mean()),
            "share_agree_ge_0.999999": float((g.confidence >= 0.999999).mean()),
            "share_disagree_at_1": float((d.confidence >= 1.0).mean())}


# ---------------------------------------------------------------- output

def fmt_ci(lo, hi, digits=3):
    return "—" if pd.isna(lo) else f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def md(df, cols, floatfmt="{:.3f}"):
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if df[c].dtype.kind in "fi" else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(("—" if pd.isna(v) else floatfmt.format(v)) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figures(headline, per_city, out_dir):
    h = headline.set_index(["label_type", "split"])
    x = np.arange(len(TYPES))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (split, color, label) in enumerate([("test", "#2a6f97", "test (training cities)"),
                                               ("test_heldout", "#d1495b", "test_heldout (unseen cities)")]):
        auc = [h.loc[(t, split), "auc"] for t in TYPES]
        lo = [h.loc[(t, split), "auc_lo"] for t in TYPES]
        hi = [h.loc[(t, split), "auc_hi"] for t in TYPES]
        wc = [h.loc[(t, split), "within_city_auc"] for t in TYPES]
        off = (i - 0.5) * 0.25
        ax.errorbar(x + off, auc, yerr=[np.clip(np.subtract(auc, lo), 0, None), np.clip(np.subtract(hi, auc), 0, None)], fmt="o", color=color,
                    capsize=3, label=f"{label}, pooled AUC")
        ax.scatter(x + off, wc, marker="_", s=220, color=color, label=f"{label}, within-city AUC")
    ax.set_xticks(x, TYPES)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (incorrect)")
    ax.set_title("Round 3: pooled AUC with 95% panorama-bootstrap intervals, and within-city AUC")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/auc_by_type.png", dpi=150)
    plt.close(fig)

    p = per_city[per_city.split == "test_heldout"]
    cities = sorted(p.city.unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in enumerate(cities):
        q = p[p.city == c].set_index("label_type").reindex(TYPES)
        off = (i - (len(cities) - 1) / 2) * 0.22
        yerr = [np.clip(np.nan_to_num(q.auc - q.auc_lo), 0, None), np.clip(np.nan_to_num(q.auc_hi - q.auc), 0, None)]
        ax.errorbar(x + off, q.auc, yerr=yerr, fmt="o", capsize=3, label=c)
    ax.set_xticks(x, TYPES)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (incorrect)")
    ax.set_title("Round 3 on the held-out deployments, per type (no interval when a class has < 8 labels)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/heldout_per_city.png", dpi=150)
    plt.close(fig)


def write_tables(out_dir, headline, per_city, share, rp, hard, prod, calib, alt, gate, sat):
    out = ["## headline", ""]
    h = headline.assign(**{"AUC [95% CI]": [f"{r.auc:.3f} {fmt_ci(r.auc_lo, r.auc_hi)}" for r in headline.itertuples()],
                           "test − held-out [95% CI]": [f"{r.diff_test_minus_heldout:+.3f} {fmt_ci(r.diff_lo, r.diff_hi)}"
                                                       for r in headline.itertuples()]})
    out.append(md(h, ["label_type", "split", "n", "n_correct", "n_incorrect", "n_panos", "n_cities", "AUC [95% CI]",
                      "within_city_auc", "within_city_n_cities", "accuracy", "balanced_accuracy", "test − held-out [95% CI]"]))
    out += ["", "## per city", ""]
    p = per_city.assign(**{"AUC [95% CI]": [("—" if pd.isna(r.auc) else f"{r.auc:.3f}") + " " + fmt_ci(r.auc_lo, r.auc_hi)
                                            for r in per_city.itertuples()]})
    for t in TYPES:
        for s in SPLITS:
            q = p[(p.label_type == t) & (p.split == s)].sort_values("n", ascending=False)
            out += [f"### {t} / {s}", "", md(q, ["city", "n", "n_correct", "n_incorrect", "n_panos", "AUC [95% CI]", "accuracy"]), ""]
    out += ["## class share", ""]
    for t in TYPES:
        q = share[share.label_type == t].sort_values(["split", "n"], ascending=[True, False])
        out += [f"### {t}", "", md(q, ["split", "city", "n", "share_of_split", "incorrect_share"]), ""]
    out += ["## recall at precision", "", md(rp, ["label_type", "split", "class", "precision", "recall", "threshold"]), ""]
    hh = hard.assign(**{"AUC [95% CI]": [f"{r.auc:.3f} {fmt_ci(r.auc_lo, r.auc_hi)}" for r in hard.itertuples()]})
    out += ["## hard", "", md(hh, ["label_type", "n", "n_correct", "n_incorrect", "AUC [95% CI]", "accuracy", "balanced_accuracy"]), ""]
    if prod is not None:
        q = prod.assign(**{"Δ [label CI]": [f"{r.diff:+.3f} {fmt_ci(r.diff_label_lo, r.diff_label_hi)}" for r in prod.itertuples()],
                           "Δ [pano CI]": [f"{r.diff:+.3f} {fmt_ci(r.diff_pano_lo, r.diff_pano_hi)}" for r in prod.itertuples()]})
        out += ["## production comparison", "",
                md(q, ["label_type", "split", "n", "n_panos", "n_deployments", "auc_round3", "auc_production",
                       "Δ [label CI]", "Δ [pano CI]", "within_city_round3", "within_city_production", "within_city_n_cities"]), ""]
    out += ["## calibration (file prior)", "", md(calib, list(calib.columns), "{:.4g}"), ""]
    if alt is not None:
        out += ["## alternative prior", "", md(alt, list(alt.columns), "{:.4g}"), ""]
    out += ["## gate per deployment", "", md(gate, list(gate.columns)), ""]
    out += ["## saturation", "", md(sat, list(sat.columns)), ""]
    with open(f"{out_dir}/tables.md", "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--priors", help="deployment_priors.json (default: <run-dir>/deployment_priors.json)")
    ap.add_argument("--public-cities", help="saved /v3/api/cities response or one city_id per line; default: the live API")
    ap.add_argument("--sample")
    ap.add_argument("--production-scores")
    ap.add_argument("--alt-prior", help="e.g. CurbRamp=0.06,Crosswalk=0.04 (incorrect shares); refits the maps for sensitivity")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--min-city-n", type=int, default=50, help="rows a named test deployment needs for its own row")
    ap.add_argument("--min-minority", type=int, default=8, help="labels of the smaller class a row needs for an interval")
    ap.add_argument("--within-min-n", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    priors_path = args.priors or os.path.join(args.run_dir, "deployment_priors.json")
    public, public_source = public_cities(args.public_cities)
    # country for the held-out lettering: a US deployment's city_id ends in a state code, like seattle-wa
    us = {c for c in public} | set()

    preds, splits = {}, {}
    for t in TYPES:
        preds[t], splits[t] = load_type(args.run_dir, t)
    heldout = sorted(set().union(*[set(p[p.split == "test_heldout"].city) for p in preds.values()]))
    us_heldout = {c for c in heldout if len(c.split("-")[-1]) == 2 and c.split("-")[-1].isalpha()}
    names = heldout_names(heldout, public, us_heldout)
    alt = parse_prior(args.alt_prior) if args.alt_prior else None

    headline, per_city, share, rp, hard, calib, altrows, gate, sat = [], [], [], [], [], [], [], [], []
    for t in TYPES:
        df, split_all = preds[t], splits[t]
        pi, _ = cal.load_prior(priors_path, t)
        headline += headline_rows(df, t, args)
        per_city += per_city_rows(df, t, public, names, args)
        share += class_share_rows(split_all, t, public, names, args.min_city_n)
        a, b = metrics_rows(args.run_dir, t, df, args)
        rp += a
        hard += b
        amap, _ = load_map(args.run_dir, t)
        calib.append(calibration_rows(args.run_dir, t, df, split_all, pi))
        gate += gate_rows(df, split_all, amap, pi, "file", t, public, names, args.min_city_n)
        sat.append(saturation_row(df, t, amap))
        if alt:
            rows, amap_alt = alt_prior_rows(df, t, alt[t])
            altrows += rows
            gate += gate_rows(df, split_all, amap_alt, alt[t], "alt", t, public, names, args.min_city_n)
    frames = {"headline": pd.DataFrame(headline), "per_city": pd.DataFrame(per_city), "class_share": pd.DataFrame(share),
              "recall_at_precision": pd.DataFrame(rp), "hard": pd.DataFrame(hard), "calibration": pd.DataFrame(calib),
              "gate_per_city": pd.DataFrame(gate), "saturation": pd.DataFrame(sat)}
    if alt:
        frames["alt_prior"] = pd.DataFrame(altrows)
    prod = None
    if args.sample and args.production_scores:
        prod = pd.DataFrame(production_rows(preds, args.sample, args.production_scores, args))
        frames["production_comparison"] = prod
    for name, df in frames.items():
        df.to_csv(f"{args.out_dir}/{name}.csv", index=False)
    figures(frames["headline"], frames["per_city"], args.out_dir)
    write_tables(args.out_dir, frames["headline"], frames["per_city"], frames["class_share"], frames["recall_at_precision"],
                 frames["hard"], prod, frames["calibration"], frames.get("alt_prior"), frames["gate_per_city"], frames["saturation"])
    with open(f"{args.out_dir}/provenance.json", "w") as f:
        json.dump({"generated": datetime.date.today().isoformat(), "run_dir": os.path.basename(args.run_dir.rstrip("/")),
                   "priors": os.path.basename(priors_path), "public_cities_source": public_source,
                   "public_cities": sorted(public), "n_boot": args.n_boot, "seed": args.seed,
                   "min_city_n": args.min_city_n, "min_minority": args.min_minority, "within_min_n": args.within_min_n,
                   "alt_prior": alt, "heldout_display_names": sorted(names.values())}, f, indent=1)
    print(open(f"{args.out_dir}/tables.md").read())


if __name__ == "__main__":
    main()
