"""Does the Gemini 3.8 Flash score carry information the round-3 classifier lacks? A stacking check on existing scores.

Reads only score files that already exist (the staging directory of ``vlm_scoreboard.py`` and the round-3 run's
``split.csv`` files); it calls no model and no network. On the crops scored by both legs it reports:

1. **AUC per label type and macro** (``incorrect`` is the positive class, each type equal weight in the macro) for
   round 3 alone, Flash alone, and a stacked model: logistic regression on ``logit(p_incorrect)`` of the two legs
   (clipped, see ``CLIP``), fitted per label type and, separately, pooled over the types. The stack is scored on
   out-of-fold predictions from ``N_FOLDS`` folds **grouped by city** (``GroupKFold``; no city is in both the train
   and the test side of a fold, asserted). The out-of-fold score is ``logit(round 3) + w * logit(Flash)``, with
   ``w`` the ratio of the two fitted coefficients: each fold's intercept and scale are dropped, so the scores of
   different fold models are on round 3's own scale and ``w = 0`` is round 3 exactly. Three more rows bound the
   result. ``stack_type_raw`` pools each fold model's raw decision values instead, and ``round3_cv_raw`` does the
   same for a logistic regression on the round-3 score alone: fold models differ in intercept and scale, which moves
   a pooled AUC with no change in ranking inside a fold, so the raw stack is read against that control and not
   against round 3. ``stack_insample`` is the per-type stack fitted and scored on the same crops (optimistic).
2. **Paired intervals for stacked minus round 3** (macro AUC, 95% percentile): a class-stratified crop bootstrap
   (``vlm_scoreboard.boot_auc``, the same draws for every model) and a city bootstrap (whole cities resampled with
   replacement, one draw for all types). Both resample the fixed out-of-fold scores; the stack is not refitted per
   resample (a refitted bootstrap is one of the sensitivity analyses, item 5).
3. **Error correlation** at ``THRESHOLD`` (0.5 on ``p_incorrect`` for both legs, the scoreboard's argmax; the
   calibrated 0.92 gate of the deployed validator is not applied here): among the crops one leg gets wrong, how often
   the other is right, next to the rate expected if the two legs' errors were independent given label type and class
   (each crop weighted by the other leg's accuracy on its type x class cell), and each leg's TPR and TNR.
4. **Hold-out check.** Every crop is looked up in the round-3 ``split.csv`` (the manifest the classifier trained
   from). A crop whose split is ``train`` or ``val``, or whose panorama is shared with a ``train``/``val`` row of the
   same label type (each type has its own classifier), counts as trained on; if any exist, every number is redone
   without them (``*_clean.csv``). Crops sharing a panorama only with a ``train``/``val`` row of another type are
   counted, not excluded.
5. **Sensitivity analyses, not the primary result.** The verdict is read from item 2 only. These rows show how much
   the estimate moves under choices that were equally defensible: the round-3 logit clipped at 1e-6 (the first
   version of this script, superseded), round 3 replaced by the normal scores of its rank within type, leave one
   city out, ``len(FOLD_SEEDS)`` shuffled ``StratifiedGroupKFold(N_FOLDS)`` fold assignments, and a bootstrap that
   refits the stack inside each of ``REFIT_BOOT`` resamples (crop and city).

**Change from the first version.** Commit ``2d1c37a`` clipped round 3's ``p_incorrect`` at 1e-6 before the logit.
Round 3's stored scores run from about 1e-11 to 1 - 1e-7 (plus exact 1.0), so that clip tied 2,964 crops in its
tails, which the stack could re-rank only by Flash, while the reference was the unclipped round 3. ``CLIP["p_r3"]``
is now 1e-12, which only keeps the exact 1.0 values finite. Nothing else in the primary analysis changed.

**Keys.** A Project Sidewalk ``label_id`` is unique only within a city, so every join is on ``label_uid``
(``city:label_id``), rebuilt here from ``city`` and ``label_id``, asserted unique, with ``validate=`` on every merge.

**Privacy.** Per-crop data stays outside the repo (as ``vlm_scoreboard.py``): the outputs are aggregates only, with
no ``label_uid`` and no deployment name.

Usage::

    python analysis/vlm_stacking.py --scores-dir D:/Data/sidewalk-validator-ai-vlm-gemini/staging \\
        --run-dir D:/Data/sidewalk-validator-ai-vlm-gemini/round3

writes ``docs/vlm-stacking-2026-11.md`` and ``docs/vlm-stacking-2026-11/*.{csv,json}``.

Example (the logit is clipped, so a self-reported 0 or 1 stays finite)::

    >>> [round(float(v), 3) for v in logit(np.array([0.0, 0.5, 1.0]), 0.01)]
    [-4.595, 0.0, 4.595]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "analysis"))
import vlm_scoreboard as sb  # noqa: E402

LABEL_TYPES = sb.LABEL_TYPES
CLASSIFIER, VLM = "round3", "gemini-3.8-flash"
OUT_DIR = os.path.join(REPO_DIR, "docs", "vlm-stacking-2026-11")
DOC = os.path.join(REPO_DIR, "docs", "vlm-stacking-2026-11.md")
SEED, N_BOOT, N_FOLDS = 20261115, 2000, 5
# Round 3 has no exact 0 and its smallest score is about 1e-11, so 1e-12 only keeps its exact 1.0 values finite (a
# larger clip ties its tails; see "Change from the first version"). Flash reports an integer confidence, so 0.01 is
# its resolution.
CLIP = {"p_r3": 1e-12, "p_vlm": 0.01}
FIRST_VERSION_CLIP = 1e-6              # the superseded round-3 clip, kept as a sensitivity row
THRESHOLD = 0.5
MIN_GAIN = 0.005                       # the bar; no record of it predates the first run (see BAR_ORIGIN)
FOLD_SEEDS = tuple(range(20))          # sensitivity: shuffled StratifiedGroupKFold assignments
REFIT_BOOT = 300                       # sensitivity: resamples of the refitted bootstrap
MODELS = ("round3", "flash", "stack_type", "stack_pooled", "stack_type_raw", "round3_cv_raw", "stack_insample")
NAMES = {"round3": "round 3", "flash": "Flash", "stack_type": "stack, per type", "stack_pooled": "stack, pooled",
         "stack_type_raw": "stack, per type, raw fold scores", "round3_cv_raw": "round 3, raw fold scores (control)",
         "stack_insample": "stack, in-sample (optimistic)"}


def logit(p, eps):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------------------------------------

def with_uid(df):
    """``df`` with ``label_uid`` rebuilt as ``city:label_id`` and asserted unique (a bare ``label_id`` repeats across
    cities). ``label_id`` is taken from the existing ``label_uid`` when the frame has no such column.

    >>> with_uid(pd.DataFrame({"city": ["a", "b"], "label_id": [1, 1]})).label_uid.tolist()
    ['a:1', 'b:1']
    """
    df = df.copy()
    if "label_id" not in df:
        df["label_id"] = df.label_uid.str.rsplit(":", n=1).str[1]
    uid = df.city.astype(str) + ":" + df.label_id.astype(str)
    if "label_uid" in df and not (uid == df.label_uid).all():
        raise ValueError("label_uid is not city:label_id")
    if uid.duplicated().any():
        raise ValueError(f"{int(uid.duplicated().sum())} duplicate (city, label_id) keys")
    return df.assign(label_uid=uid)


def load_both(scores_dir):
    """One row per crop scored by both legs: ``label_uid, city, label_type, split, pano_id, y, p_r3, p_vlm``."""
    splits = with_uid(sb.load_splits(scores_dir))
    legs = {}
    for leg in (CLASSIFIER, VLM):
        d, unscored = sb.load_leg(os.path.join(scores_dir, "scores", leg))
        if d is None:
            raise SystemExit(f"missing score files: {os.path.join(scores_dir, 'scores', leg)}/<Type>.csv")
        legs[leg] = (d, unscored)
    m = legs[VLM][0].rename(columns={"p_incorrect": "p_vlm"}).merge(
        legs[CLASSIFIER][0].rename(columns={"p_incorrect": "p_r3"}), on="label_uid", how="inner",
        validate="one_to_one")
    m = m.merge(splits, on="label_uid", how="left", validate="one_to_one")
    if m.label_type.isna().any():
        raise ValueError(f"{int(m.label_type.isna().sum())} scored crops are not in the held-out splits")
    counts = {"n_vlm": len(legs[VLM][0]), "n_classifier": len(legs[CLASSIFIER][0]), "n_both": len(m),
              "vlm_unscored": int(sum(legs[VLM][1].values()))}
    return with_uid(m).reset_index(drop=True), counts


def load_run_splits(run_dir):
    """Every row of the round-3 ``<Type>/split.csv`` manifests (train, val, test, test_heldout)."""
    cols = ["label_uid", "city", "label_id", "label_type", "pano_id", "split"]
    frames = []
    for t in LABEL_TYPES:
        p = os.path.join(run_dir, t, "split.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing split definition: {p}")
        frames.append(pd.read_csv(p, usecols=cols))
    return with_uid(pd.concat(frames, ignore_index=True))


def holdout_check(df, run):
    """``(rows, trained_on)``: the crops by their split in the round-3 manifest, and a boolean mask of the crops the
    classifier trained on: split ``train``/``val``, or a panorama shared with a ``train``/``val`` row of the same
    label type. A crop missing from the manifest raises."""
    j = df[["label_uid", "label_type", "pano_id"]].merge(
        run[["label_uid", "split"]].rename(columns={"split": "run_split"}), on="label_uid", how="left",
        validate="one_to_one")
    if j.run_split.isna().any():
        raise ValueError(f"{int(j.run_split.isna().sum())} crops are not in the round-3 split.csv")
    fit = run[run.split.isin(["train", "val"]) & run.pano_id.notna()]
    fit = fit[["label_type", "pano_id"]].drop_duplicates().assign(shared_pano=True)
    j = j.merge(fit, on=["label_type", "pano_id"], how="left", validate="many_to_one")
    shared = j.shared_pano.fillna(False).astype(bool).to_numpy()
    in_fit = j.run_split.isin(["train", "val"]).to_numpy()
    rows = [{"check": f"round-3 split = {s}", "n": int((j.run_split == s).sum())}
            for s in ("train", "val", "test", "test_heldout")]
    rows.append({"check": "panorama shared with a train/val row of the same label type", "n": int(shared.sum())})
    rows.append({"check": "trained on (either)", "n": int((shared | in_fit).sum())})
    fit_types = (run[run.split.isin(["train", "val"]) & run.pano_id.notna()]
                 .groupby("pano_id").label_type.agg(lambda v: frozenset(v)).to_dict())
    other = np.array([bool(fit_types.get(pano, frozenset()) - {t}) for pano, t in zip(j.pano_id, j.label_type)])
    rows.append({"check": "panorama shared with a train/val row of a different label type (not a leak: each type has "
                          "its own classifier)", "n": int(other.sum())})
    rows.append({"check": "round-3 manifest rows: " + ", ".join(
        f"{s} {int((run.split == s).sum()):,}" for s in ("train", "val", "test", "test_heldout")), "n": len(run)})
    return pd.DataFrame(rows), shared | in_fit


# --------------------------------------------------------------------------------------------------------------------
# The stack
# --------------------------------------------------------------------------------------------------------------------

def _fit_predict(xtr, ytr, xte, raw=False):
    """Scores of ``xte`` from a logistic regression fitted on ``xtr``: the decision values when ``raw``, else
    ``x[:, 0] + sum(w_k * x[:, k])`` with ``w_k = coef_k / coef_0`` (no intercept, the first column's scale). Also
    returns the weights ``w`` (empty for one column)."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=1000).fit(xtr, ytr)
    b = m.coef_[0]
    if b[0] <= 0:
        raise ValueError("the first score has a non-positive coefficient; the ratio form is undefined")
    return (m.decision_function(xte) if raw else xte @ (b / b[0])), b[1:] / b[0]


def oof_scores(x, y, city, raw=False, n_folds=N_FOLDS, splitter=None):
    """``(scores, weights per fold)``: out-of-fold scores of a logistic regression on ``x`` (see ``_fit_predict``),
    folds grouped by ``city`` (no city on both sides of a fold). ``splitter`` is any scikit-learn group splitter;
    the default, ``GroupKFold(n_folds)``, is deterministic and is the primary analysis.

    >>> x = np.arange(12.0)[:, None]; y = np.tile([0, 1], 6); city = np.repeat(list("abcdef"), 2)
    >>> bool((oof_scores(x, y, city, n_folds=3)[0] == x[:, 0]).all())     # one column: the score itself
    True
    """
    from sklearn.model_selection import GroupKFold
    x, y, city = np.asarray(x, float), np.asarray(y, int), np.asarray(city)
    out, weights = np.full(len(y), np.nan), []
    for tr, te in (splitter or GroupKFold(n_splits=n_folds)).split(x, y, city):
        assert not set(city[tr]) & set(city[te]), "a city is on both sides of a fold"
        out[te], w = _fit_predict(x[tr], y[tr], x[te], raw)
        weights.append(w)
    assert not np.isnan(out).any()
    return out, np.array(weights)


def add_models(df):
    """``(df, weights)``: ``df`` with one score column per entry of ``MODELS`` (higher = more likely ``incorrect``),
    and the Flash weight ``w`` of each stack (in-sample, and the range over the folds)."""
    df = df.copy()
    x = np.column_stack([logit(df.p_r3, CLIP["p_r3"]), logit(df.p_vlm, CLIP["p_vlm"])])
    y, city = df.y.to_numpy(), df.city.to_numpy()
    df["round3"], df["flash"] = df.p_r3, df.p_vlm
    df["stack_pooled"], w = oof_scores(x, y, city)
    weights = [{"stack": "pooled", "w_insample": float(_fit_predict(x, y, x)[1][0]), "w_fold_min": float(w.min()),
                "w_fold_max": float(w.max())}]
    for col in MODELS[4:]:
        df[col] = np.nan
    for t in LABEL_TYPES:
        i = np.flatnonzero((df.label_type == t).to_numpy())
        if len(i):
            rows = df.index[i]
            df.loc[rows, "stack_type"], w = oof_scores(x[i], y[i], city[i])
            df.loc[rows, "stack_type_raw"] = oof_scores(x[i], y[i], city[i], raw=True)[0]
            df.loc[rows, "round3_cv_raw"] = oof_scores(x[i, :1], y[i], city[i], raw=True)[0]
            df.loc[rows, "stack_insample"], wi = _fit_predict(x[i], y[i], x[i])
            weights.append({"stack": t, "w_insample": float(wi[0]), "w_fold_min": float(w.min()),
                            "w_fold_max": float(w.max())})
    return df, pd.DataFrame(weights)


# --------------------------------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------------------------------

def crop_draws(df, n_boot, seed, cols=MODELS):
    """``{Type: n_boot x len(cols)}`` bootstrap AUCs, crops resampled within class, the same draws for every model."""
    return {t: sb.boot_auc(g.y, g[list(cols)].to_numpy(), n_boot, sb._type_rng(seed, t))
            for t, g in ((t, df[df.label_type == t]) for t in LABEL_TYPES) if len(g)}


def city_draws(df, n_boot, seed, cols=MODELS):
    """``n_boot x n_types x len(cols)`` AUCs with whole cities resampled with replacement (one draw of cities per
    replicate, shared by every type and model). NaN where a type's resample has one class only."""
    rng = np.random.default_rng([seed, 99])
    cities = np.sort(df.city.unique())
    types = [t for t in LABEL_TYPES if (df.label_type == t).any()]
    rows = {(t, c): np.flatnonzero(((df.label_type == t) & (df.city == c)).to_numpy()) for t in types for c in cities}
    y, s = df.y.to_numpy(), df[list(cols)].to_numpy()
    out = np.full((n_boot, len(types), len(cols)), np.nan)
    for b in range(n_boot):
        draw = cities[rng.integers(0, len(cities), len(cities))]
        for k, t in enumerate(types):
            idx = np.concatenate([rows[t, c] for c in draw])
            if len(idx) and 0 < y[idx].sum() < len(idx):
                out[b, k] = sb._rank_auc(np.repeat(y[idx][None, :], len(cols), 0), s[idx].T)
    return out


def auc_rows(df, draws):
    """One row per (label type or macro) x model: n, AUC and its crop-bootstrap interval."""
    rows, point, boot = [], {m: [] for m in MODELS}, {m: [] for m in MODELS}
    for t in LABEL_TYPES:
        g = df[df.label_type == t]
        if g.empty:
            continue
        for k, m in enumerate(MODELS):
            lo, hi = sb.ci(draws[t][:, k])
            a = sb.auc(g.y, g[m])
            rows.append({"label_type": t, "model": m, "n": len(g), "n_incorrect": int(g.y.sum()),
                         "n_cities": g.city.nunique(), "auc": a, "auc_lo": lo, "auc_hi": hi})
            point[m].append(a)
            boot[m].append(draws[t][:, k])
    for m in MODELS:
        lo, hi = sb.ci(np.mean(boot[m], axis=0))
        rows.append({"label_type": "macro", "model": m, "n": len(df), "n_incorrect": int(df.y.sum()),
                     "n_cities": df.city.nunique(), "auc": float(np.mean(point[m])), "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(rows)


PAIRS = (("stack_type", "round3"), ("stack_pooled", "round3"), ("stack_insample", "round3"),
         ("stack_type_raw", "round3_cv_raw"), ("stack_type_raw", "round3"), ("flash", "round3"))


def paired_rows(df, draws, cdraws):
    """Model minus reference: per type and macro, with the crop-bootstrap and city-bootstrap intervals."""
    types = [t for t in LABEL_TYPES if t in draws]
    rows = []
    for a, b in PAIRS:
        ia, ib = MODELS.index(a), MODELS.index(b)
        pts = []
        for k, t in enumerate(types):
            g = df[df.label_type == t]
            d = sb.auc(g.y, g[a]) - sb.auc(g.y, g[b])
            pts.append(d)
            lo, hi = sb.ci(draws[t][:, ia] - draws[t][:, ib])
            clo, chi = sb.ci(cdraws[:, k, ia] - cdraws[:, k, ib])
            rows.append({"model": a, "reference": b, "label_type": t, "diff": d, "crop_lo": lo, "crop_hi": hi,
                         "city_lo": clo, "city_hi": chi})
        lo, hi = sb.ci(np.mean([draws[t][:, ia] - draws[t][:, ib] for t in types], axis=0))
        cd = cdraws[:, :, ia] - cdraws[:, :, ib]
        ok = ~np.isnan(cd).any(axis=1)      # a macro needs every type scoreable in the city resample
        clo, chi = sb.ci(cd[ok].mean(axis=1))
        rows.append({"model": a, "reference": b, "label_type": "macro", "diff": float(np.mean(pts)), "crop_lo": lo,
                     "crop_hi": hi, "city_lo": clo, "city_hi": chi, "city_resamples_dropped": int((~ok).sum())})
    return pd.DataFrame(rows)


def error_rows(df, threshold=THRESHOLD):
    """Per type and over all crops: each leg's accuracy at ``threshold``, and how often one leg is right on the
    crops the other gets wrong."""
    d = df.assign(r_ok=((df.p_r3 >= threshold).astype(int) == df.y),
                  v_ok=((df.p_vlm >= threshold).astype(int) == df.y))
    # each leg's accuracy on the crop's (label type, class) cell: what the other leg's errors would see if the two
    # legs' errors were independent given type and class
    cell = d.groupby(["label_type", "y"])
    d = d.assign(r_cell=cell.r_ok.transform("mean"), v_cell=cell.v_ok.transform("mean"))
    rows = []
    for t in list(LABEL_TYPES) + ["all"]:
        g = d if t == "all" else d[d.label_type == t]
        if g.empty:
            continue
        r_ok, v_ok, pos = g.r_ok.to_numpy(), g.v_ok.to_numpy(), (g.y == 1).to_numpy()
        rows.append({"label_type": t, "n": len(g), "round3_accuracy": r_ok.mean(), "flash_accuracy": v_ok.mean(),
                     "n_round3_wrong": int((~r_ok).sum()), "flash_right_when_round3_wrong": v_ok[~r_ok].mean(),
                     "flash_right_if_independent": g.v_cell.to_numpy()[~r_ok].mean(),
                     "n_flash_wrong": int((~v_ok).sum()), "round3_right_when_flash_wrong": r_ok[~v_ok].mean(),
                     "round3_right_if_independent": g.r_cell.to_numpy()[~v_ok].mean(),
                     "n_both_wrong": int((~r_ok & ~v_ok).sum()),
                     "error_phi": float(np.corrcoef(~r_ok, ~v_ok)[0, 1]),
                     "round3_tpr": r_ok[pos].mean(), "round3_tnr": r_ok[~pos].mean(),
                     "flash_tpr": v_ok[pos].mean(), "flash_tnr": v_ok[~pos].mean()})
    return pd.DataFrame(rows)


def analyse(df, n_boot=N_BOOT, seed=SEED):
    """Every table of the primary analysis for one set of crops: ``{"auc", "paired", "errors", "weights"}``."""
    df, weights = add_models(df)
    draws = crop_draws(df, n_boot, seed)
    return {"auc": auc_rows(df, draws), "paired": paired_rows(df, draws, city_draws(df, n_boot, seed)),
            "errors": error_rows(df), "weights": weights}


def verdict(paired):
    """``(yes, row)`` for the pre-stated rule on the per-type stack: macro gain over round 3 of at least
    ``MIN_GAIN`` with a crop-bootstrap interval that excludes zero."""
    r = paired[(paired.model == "stack_type") & (paired.reference == "round3") & (paired.label_type == "macro")].iloc[0]
    return bool(r["diff"] >= MIN_GAIN and r.crop_lo > 0), r


# --------------------------------------------------------------------------------------------------------------------
# Sensitivity analyses (labelled as such; the verdict is read from the primary analysis only)
# --------------------------------------------------------------------------------------------------------------------

def normal_rank(df):
    """Round 3 replaced by the normal scores of its rank within each label type (label-free, ties averaged).

    >>> [round(float(v), 3) for v in normal_rank(pd.DataFrame({"label_type": ["a"] * 3, "p_r3": [0.2, 0.9, 0.5]}))]
    [-0.967, 0.967, 0.0]
    """
    from scipy.stats import norm, rankdata
    out = np.full(len(df), np.nan)
    for _, i in df.groupby("label_type").indices.items():
        out[i] = norm.ppf((rankdata(df.p_r3.to_numpy()[i]) - 0.5) / len(i))
    return out


def type_stack(df, x0, splitter=None):
    """Out-of-fold ratio scores of the per-type stack on ``[x0, logit(Flash)]`` (as ``stack_type``)."""
    x = np.column_stack([x0, logit(df.p_vlm, CLIP["p_vlm"])])
    y, city, out = df.y.to_numpy(), df.city.to_numpy(), np.full(len(df), np.nan)
    for _, i in df.groupby("label_type").indices.items():
        out[i] = oof_scores(x[i], y[i], city[i], splitter=splitter)[0]
    return out


def paired_rows_for(df, a, b, n_boot, seed, city=True):
    """Per type and macro, ``AUC(a) - AUC(b)`` for two score columns of ``df`` with crop (and, if ``city``, city)
    bootstrap intervals. The crop draws depend only on ``y`` and the seed, so they are the primary's resamples."""
    cols = (a, b)
    draws = crop_draws(df, n_boot, seed, cols)
    types = [t for t in LABEL_TYPES if t in draws]
    cd = city_draws(df, n_boot, seed, cols) if city else np.full((n_boot, len(types), 2), np.nan)
    rows = []
    for k, t in enumerate(types):
        g = df[df.label_type == t]
        lo, hi = sb.ci(draws[t][:, 0] - draws[t][:, 1])
        clo, chi = sb.ci(cd[:, k, 0] - cd[:, k, 1])
        rows.append({"label_type": t, "diff": sb.auc(g.y, g[a]) - sb.auc(g.y, g[b]), "crop_lo": lo, "crop_hi": hi,
                     "city_lo": clo, "city_hi": chi})
    lo, hi = sb.ci(np.mean([draws[t][:, 0] - draws[t][:, 1] for t in types], axis=0))
    c = cd[:, :, 0] - cd[:, :, 1]
    c = c[~np.isnan(c).any(axis=1)]
    clo, chi = sb.ci(c.mean(axis=1)) if len(c) else (np.nan, np.nan)
    rows.append({"label_type": "macro", "diff": float(np.mean([r["diff"] for r in rows])), "crop_lo": lo,
                 "crop_hi": hi, "city_lo": clo, "city_hi": chi})
    return pd.DataFrame(rows)


def refit_bootstrap(df, n_rep=REFIT_BOOT, seed=SEED):
    """``{"crop": draws, "city": draws}`` of the macro gain (per-type stack minus round 3) with the stack REFITTED,
    GroupKFold by city, inside each resample. Crop: class-stratified within type. City: one draw of cities per
    replicate; copies of a city share a group, so a city is never on both sides of a fold. A replicate where a type
    has fewer than ``N_FOLDS`` cities or one class is dropped."""
    from sklearn.model_selection import GroupKFold
    x = np.column_stack([logit(df.p_r3, CLIP["p_r3"]), logit(df.p_vlm, CLIP["p_vlm"])])
    y, city, p = df.y.to_numpy(), df.city.to_numpy(), df.p_r3.to_numpy()
    by_type = {t: np.flatnonzero((df.label_type == t).to_numpy()) for t in LABEL_TYPES if (df.label_type == t).any()}

    def gain(idx):
        if len(np.unique(city[idx])) < N_FOLDS or not 0 < y[idx].sum() < len(idx):
            return np.nan
        s = oof_scores(x[idx], y[idx], city[idx], splitter=GroupKFold(N_FOLDS))[0]
        return sb.auc(y[idx], s) - sb.auc(y[idx], p[idx])

    crop = np.full((n_rep, len(by_type)), np.nan)
    for k, (t, i) in enumerate(by_type.items()):
        rng = sb._type_rng(seed, t, 3)
        pos, neg = i[y[i] == 1], i[y[i] == 0]
        for r in range(n_rep):
            crop[r, k] = gain(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))
    rng = np.random.default_rng([seed, 98])
    cities = np.sort(df.city.unique())
    rows = {(t, c): i[city[i] == c] for t, i in by_type.items() for c in cities}
    cty = np.full((n_rep, len(by_type)), np.nan)
    for r in range(n_rep):
        draw = cities[rng.integers(0, len(cities), len(cities))]
        for k, t in enumerate(by_type):
            cty[r, k] = gain(np.concatenate([rows[t, c] for c in draw]))
    return {"crop": crop[~np.isnan(crop).any(axis=1)].mean(axis=1),
            "city": cty[~np.isnan(cty).any(axis=1)].mean(axis=1)}


def sensitivity(df, n_boot=N_BOOT, seed=SEED, fold_seeds=FOLD_SEEDS, n_refit=REFIT_BOOT):
    """``(variants, seeds)``: one row per sensitivity variant (macro gain over round 3, intervals, per-type gains,
    the gain of the variant's round-3 input alone, and whether it would meet the rule), and one row per shuffled fold
    seed (crop interval only). None of these is the primary result."""
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
    d = df[["city", "label_type", "y", "p_r3", "p_vlm"]].reset_index(drop=True)
    d["round3"] = d.p_r3
    x_primary = logit(d.p_r3, CLIP["p_r3"])
    specs = (("clip_1e-6", f"round-3 logit clipped at {FIRST_VERSION_CLIP:g} (first version, superseded)",
              logit(d.p_r3, FIRST_VERSION_CLIP), None),
             ("normal_rank", "round 3 as normal scores of its rank within type", normal_rank(d), None),
             ("loco", "leave one city out", x_primary, LeaveOneGroupOut()))
    rows = []
    for key, label, x0, splitter in specs:
        d["x0"], d["stack"] = x0, type_stack(d, x0, splitter)
        per = paired_rows_for(d, "stack", "round3", n_boot, seed)
        m = per.iloc[-1]
        alone = paired_rows_for(d, "x0", "round3", 10, seed, city=False)
        row = {"variant": key, "label": label, "diff": m["diff"], "crop_lo": m.crop_lo, "crop_hi": m.crop_hi,
               "city_lo": m.city_lo, "city_hi": m.city_hi,
               "meets_rule": bool(m["diff"] >= MIN_GAIN and m.crop_lo > 0),
               "x0_alone_diff": float(alone.iloc[-1]["diff"])}
        for _, r in per.iloc[:-1].iterrows():
            row[f"diff_{r.label_type}"] = r["diff"]
        for _, r in alone.iloc[:-1].iterrows():
            row[f"x0_alone_{r.label_type}"] = r["diff"]
        if key == "clip_1e-6":
            for t in LABEL_TYPES:
                g = d[d.label_type == t]
                row[f"n_tied_{t}"] = int(((g.p_r3 < FIRST_VERSION_CLIP) | (g.p_r3 > 1 - FIRST_VERSION_CLIP)).sum())
        rows.append(row)
    if n_refit:
        r = refit_bootstrap(d, n_refit, seed)
        (lo, hi), (clo, chi) = sb.ci(r["crop"]), sb.ci(r["city"])
        d["stack"] = type_stack(d, x_primary)
        m = paired_rows_for(d, "stack", "round3", 10, seed, city=False).iloc[-1]
        rows.append({"variant": "refit_bootstrap", "label": f"primary stack, refitted inside each resample "
                     f"({len(r['crop'])} crop and {len(r['city'])} city resamples usable of {n_refit})",
                     "diff": m["diff"], "crop_lo": lo, "crop_hi": hi, "city_lo": clo, "city_hi": chi,
                     "meets_rule": bool(m["diff"] >= MIN_GAIN and lo > 0)})
    seeds = []
    for s in fold_seeds:
        d["stack"] = type_stack(d, x_primary, StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=s))
        m = paired_rows_for(d, "stack", "round3", n_boot, seed, city=False).iloc[-1]
        seeds.append({"fold_seed": s, "diff": m["diff"], "crop_lo": m.crop_lo, "crop_hi": m.crop_hi,
                      "meets_rule": bool(m["diff"] >= MIN_GAIN and m.crop_lo > 0)})
    return pd.DataFrame(rows), pd.DataFrame(seeds)


# --------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------

def _iv(v, lo, hi, signed=False):
    f = "%+.3f" if signed else "%.3f"
    return "" if pd.isna(v) else f"{f % v} [{f % lo}, {f % hi}]"


def _code_state():
    """``(commit, dirty)``: the repository's HEAD and whether the analysis code differs from it (None if no git)."""
    import subprocess
    try:
        run = lambda *a: subprocess.run(["git", "-C", REPO_DIR, *a], capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--", "analysis/vlm_stacking.py",
                                                   "analysis/vlm_scoreboard.py"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _versions():
    import platform
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def _excl(lo, hi, x=0.0):
    return "excludes" if lo > x or hi < x else "includes"


def _bar_clause(v):
    """How the crop interval of the primary gain sits against ``MIN_GAIN``."""
    if v.crop_hi < MIN_GAIN:
        return f"lies wholly below {MIN_GAIN}, so a gain of {MIN_GAIN} is ruled out"
    if v.crop_lo > MIN_GAIN:
        return f"lies wholly above {MIN_GAIN}"
    return (f"contains {MIN_GAIN}, so the data neither establish a gain of {MIN_GAIN} nor rule one out")


BAR_ORIGIN = (
    "**Where the 0.005 bar comes from.** The figure first appears in the Planning repository's options report, "
    "`reports/2026-11-ai-module-options.md`, section 3 F: \"Expect a small effect. 25%: >= 0.005 AUC.\" The report "
    "says its sections 1-5 were written before the results, but its earliest timestamped copy is Planning commit "
    "`9d22e08` (2026-11-15 07:30:59 -0800), about 19 minutes after this analysis was first committed (`2d1c37a`, "
    "07:11:32 -0800). Neither [Planning#37](https://github.com/ProjectSidewalk/Planning/issues/37) nor "
    "[Planning#38](https://github.com/ProjectSidewalk/Planning/issues/38) states it. Nothing on record shows the bar "
    "before the run, so it is treated as **post hoc**. The report also gives 0.005 as a forecast (a 25% chance of a "
    "gain at least that large), not as a decision rule; the clause that the crop interval exclude zero appears first "
    "in this script.")


def render(counts, hold, res, clean, n_trained, sens=None, seeds=None, prov=None):
    """The results page (aggregates only)."""
    a, p, e = res["auc"], res["paired"], res["errors"]
    yes, v = verdict(p)
    st = p[(p.model == "stack_type") & (p.reference == "round3")].set_index("label_type")
    top = st.drop("macro")["diff"].idxmax()
    ts = st.loc[top]
    ins = p[(p.model == "stack_insample") & (p.label_type == "macro")].iloc[0]
    out = ["# Does Gemini 3.8 Flash add information to round 3? A stacking check (2026-11)", "",
           "Generated by `analysis/vlm_stacking.py` from existing score files; no model was called. Methods are in "
           "the script's docstring. `incorrect` is the positive class; intervals are 95% percentile intervals from "
           f"{N_BOOT:,} resamples (seed {SEED}).", "",
           f"**Verdict under the pre-stated rule: {'YES' if yes else 'NO'}.** Adding Flash to round 3 (the per-type "
           f"stack, city-grouped {N_FOLDS}-fold CV) changes macro AUC by **{v['diff']:+.4f}**, crop bootstrap "
           f"[{v.crop_lo:+.4f}, {v.crop_hi:+.4f}], city bootstrap [{v.city_lo:+.4f}, {v.city_hi:+.4f}]. The crop "
           f"interval {_excl(v.crop_lo, v.crop_hi)} zero and the city interval {_excl(v.city_lo, v.city_hi)} zero. "
           f"The point estimate is {'at or above' if v['diff'] >= MIN_GAIN else 'below'} the {MIN_GAIN} bar, and "
           f"the crop interval {_bar_clause(v)}. The rule was a macro gain of at least {MIN_GAIN} with a "
           "crop-bootstrap interval excluding zero.", "",
           f"**Where the gain is.** {top} carries most of it: {ts['diff']:+.4f}, crop [{ts.crop_lo:+.4f}, "
           f"{ts.crop_hi:+.4f}], city [{ts.city_lo:+.4f}, {ts.city_hi:+.4f}]. The city interval "
           f"{_excl(ts.city_lo, ts.city_hi)} zero"
           + (", so the gain is uneven across cities." if ts.city_lo <= 0 <= ts.city_hi else ".")
           + " The other types: " + "; ".join(f"{t} {r['diff']:+.4f} [{r.crop_lo:+.4f}, {r.crop_hi:+.4f}]"
                                             for t, r in st.drop(["macro", top]).iterrows())
           + f". The in-sample per-type stack, an optimistic bound, gains {ins['diff']:+.4f}.", ""]
    if sens is not None and (sens.variant == "clip_1e-6").any():
        c = sens[sens.variant == "clip_1e-6"].iloc[0]
        n_tied = sum(int(c[f"n_tied_{t}"]) for t in LABEL_TYPES if f"n_tied_{t}" in c)
        out += [f"**Change from the first version (commit `2d1c37a`).** That version clipped round 3's "
                f"`p_incorrect` at {FIRST_VERSION_CLIP:g} before the logit, which tied {n_tied:,} crops in round 3's "
                "tails; the stack could re-rank them only by Flash, while the reference was the unclipped round 3. "
                f"The clipped round-3 logit alone scores {c.x0_alone_diff:+.4f} macro against round 3, and the "
                f"first version's stack gain was {c['diff']:+.4f} [{c.crop_lo:+.4f}, {c.crop_hi:+.4f}]. Its "
                f"Obstacle result, {c['diff_Obstacle']:+.4f}, came from the clip, not from Flash: the clip alone "
                f"costs Obstacle {c['x0_alone_Obstacle']:+.4f} ({int(c['n_tied_Obstacle'])} crops tied), and "
                f"unclipped the Obstacle gain is {st.loc['Obstacle', 'diff']:+.4f} "
                f"[{st.loc['Obstacle', 'crop_lo']:+.4f}, {st.loc['Obstacle', 'crop_hi']:+.4f}]. The clip is now "
                f"{CLIP['p_r3']:g}, which only keeps round 3's exact 1.0 values finite; nothing else in the primary "
                "analysis changed. The first version is kept as a sensitivity row (section 5).", ""]
    out += [BAR_ORIGIN, "",
            f"Crops: {counts['n_both']:,} scored by both legs (Flash {counts['n_vlm']:,}, "
            f"{counts['vlm_unscored']} unscored; round 3 {counts['n_classifier']:,}), "
            f"{int(a[a.label_type == 'macro'].n_cities.iloc[0])} deployments, all public.", "",
            "## 1. AUC, crop-bootstrap intervals", "",
            "| Label type | n (incorrect) | cities | " + " | ".join(NAMES[m] for m in MODELS[:4]) + " |",
            "|---|---|---|" + "---|" * 4]
    for t in list(LABEL_TYPES) + ["macro"]:
        g = a[a.label_type == t].set_index("model")
        if g.empty:
            continue
        r0 = g.loc["round3"]
        out.append(f"| {t} | {int(r0.n):,} ({int(r0.n_incorrect):,}) | {int(r0.n_cities)} | " + " | ".join(
            _iv(g.loc[m].auc, g.loc[m].auc_lo, g.loc[m].auc_hi) for m in MODELS[:4]) + " |")
    g = a[a.label_type == "macro"].set_index("model")
    out += ["", "Bounds on the macro: " + "; ".join(
        f"{NAMES[m]} {_iv(g.loc[m].auc, g.loc[m].auc_lo, g.loc[m].auc_hi)}" for m in MODELS[4:]) + ". Pooling raw "
        "fold scores lowers round 3's own AUC with no change in ranking inside a fold, so the raw stack is compared "
        "with that control.", "",
        "Flash weight `w` in `logit(round 3) + w * logit(Flash)`, in-sample (range over the folds): " + "; ".join(
            f"{r.stack} {r.w_insample:.2f} ({r.w_fold_min:.2f} to {r.w_fold_max:.2f})"
            for _, r in res["weights"].iterrows()) + ".", "",
        "## 2. Paired differences (macro AUC unless a type is named)", "",
        "| Difference | Label type | Δ AUC | crop bootstrap | city bootstrap |", "|---|---|---|---|---|"]
    for _, r in p.iterrows():
        if r.label_type == "macro" or (r.model == "stack_type" and r.reference == "round3"):
            out.append(f"| {NAMES[r.model]} − {NAMES[r.reference]} | {r.label_type} | {r['diff']:+.4f} | "
                       f"[{r.crop_lo:+.4f}, {r.crop_hi:+.4f}] | [{r.city_lo:+.4f}, {r.city_hi:+.4f}] |")
    al = e[e.label_type == "all"].iloc[0]
    pe = e[e.label_type != "all"]
    out += ["", f"## 3. Error correlation at p_incorrect ≥ {THRESHOLD} (both legs)", "",
            f"Round 3 is wrong on {int(al.n_round3_wrong):,} crops. Flash is right on "
            f"{al.flash_right_when_round3_wrong:.1%} of them; if the two legs' errors were independent given label "
            f"type and class it would be right on {al.flash_right_if_independent:.1%}. Flash is wrong on "
            f"{int(al.n_flash_wrong):,}; round 3 is right on {al.round3_right_when_flash_wrong:.1%} of those, against "
            f"{al.round3_right_if_independent:.1%} expected. The errors are strongly correlated (φ "
            f"{al.error_phi:.2f}), which fits the small stack gain.", "",
            "| Label type | n | round 3 acc. | Flash acc. | round 3 wrong | Flash right on those | expected if "
            "independent | Flash wrong | round 3 right on those | expected if independent | both wrong | φ of "
            "errors |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in e.iterrows():
        out.append(f"| {r.label_type} | {int(r.n):,} | {r.round3_accuracy:.3f} | {r.flash_accuracy:.3f} | "
                   f"{int(r.n_round3_wrong):,} | {r.flash_right_when_round3_wrong:.3f} | "
                   f"{r.flash_right_if_independent:.3f} | {int(r.n_flash_wrong):,} | "
                   f"{r.round3_right_when_flash_wrong:.3f} | {r.round3_right_if_independent:.3f} | "
                   f"{int(r.n_both_wrong):,} | {r.error_phi:.2f} |")
    out += ["", f"Caveat: at {THRESHOLD} the two legs sit at different operating points. Per type, Flash's TPR is "
                f"{pe.flash_tpr.min():.2f} to {pe.flash_tpr.max():.2f} and its TNR {pe.flash_tnr.min():.2f} to "
                f"{pe.flash_tnr.max():.2f}; round 3's are {pe.round3_tpr.min():.2f} to {pe.round3_tpr.max():.2f} and "
                f"{pe.round3_tnr.min():.2f} to {pe.round3_tnr.max():.2f}. So this table compares unmatched operating "
                "points and does not bear on the AUC verdict. (`incorrect` is the positive class.)"]
    out += ["", "## 4. Were these crops held out from round 3's training?", "",
            "Each crop was looked up by `city:label_id` in the round-3 run's `<Type>/split.csv`, the manifest "
            "`fine_tune_classifier.py` trains from (`split_data.py`: grouped by panorama, panoramas within 3 m "
            "merged, three deployments held out whole as `test_heldout`).", "", "| Check | n |", "|---|---|"]
    out += [f"| {r.check} | {int(r.n):,} |" for _, r in hold.iterrows()]
    cross = hold[hold.check.str.startswith("panorama shared with a train/val row of a different")]
    if len(cross):
        out += ["", f"{int(cross.n.iloc[0]):,} of the {counts['n_both']:,} crops share a panorama with a train/val "
                    "row of a different label type. Each label type has its own classifier, so that is not a leak "
                    "for these per-type scores."]
    if n_trained and clean is not None:
        yes2, v2 = verdict(clean["paired"])
        out += ["", f"{n_trained:,} crops were trained on. Without them: stack − round 3 macro "
                    f"{_iv(v2['diff'], v2.crop_lo, v2.crop_hi, True)} (crop), [{v2.city_lo:+.3f}, {v2.city_hi:+.3f}] "
                    f"(city); verdict {'YES' if yes2 else 'NO'}. Tables in `*_clean.csv`."]
    else:
        out += ["", "None was trained on, so no number is redone. The `test` crops are from deployments round 3 "
                    "trained on (other panoramas); only the `test_heldout` crops are from an unseen deployment."]
    if sens is not None:
        out += ["", "## 5. Sensitivity analyses (not the primary result)", "",
                "The verdict above is read from the primary analysis only. These rows show how far the estimate moves "
                "under other defensible choices; none replaces it. \"Meets rule\" applies the same rule to the row, "
                "for information only. \"Input alone\" is the macro AUC of the row's round-3 input minus round 3's.",
                "", "| Variant | macro Δ | crop bootstrap | city bootstrap | meets rule | input alone | "
                + " | ".join(LABEL_TYPES) + " |", "|---|---|---|---|---|---|" + "---|" * len(LABEL_TYPES),
                f"| primary (clip {CLIP['p_r3']:g}, GroupKFold) | {v['diff']:+.4f} | [{v.crop_lo:+.4f}, "
                f"{v.crop_hi:+.4f}] | [{v.city_lo:+.4f}, {v.city_hi:+.4f}] | {'yes' if yes else 'no'} | +0.0000 | "
                + " | ".join(f"{st.loc[t, 'diff']:+.4f}" for t in LABEL_TYPES) + " |"]
        for _, r in sens.iterrows():
            types = " | ".join(f"{r[f'diff_{t}']:+.4f}" if pd.notna(r.get(f"diff_{t}")) else "" for t in LABEL_TYPES)
            alone = f"{r.x0_alone_diff:+.4f}" if pd.notna(r.get("x0_alone_diff")) else ""
            out.append(f"| {r.label} | {r['diff']:+.4f} | [{r.crop_lo:+.4f}, {r.crop_hi:+.4f}] | "
                       f"[{r.city_lo:+.4f}, {r.city_hi:+.4f}] | {'yes' if r.meets_rule else 'no'} | {alone} | "
                       f"{types} |")
        if seeds is not None and len(seeds):
            out += ["", f"**Fold assignment.** Over {len(seeds)} shuffled `StratifiedGroupKFold({N_FOLDS})` "
                        f"assignments (seeds {min(seeds.fold_seed)} to {max(seeds.fold_seed)}, grouped by city, "
                        f"same stacker and crop bootstrap), the macro gain runs from {seeds['diff'].min():+.4f} to "
                        f"{seeds['diff'].max():+.4f}, mean {seeds['diff'].mean():+.4f}. "
                        f"{int((seeds['diff'] >= MIN_GAIN).sum())} of {len(seeds)} reach {MIN_GAIN}, and "
                        f"{int(seeds.meets_rule.sum())} of {len(seeds)} would meet the rule (gain ≥ {MIN_GAIN} and "
                        f"crop interval above zero); {int((seeds.crop_lo > 0).sum())} of {len(seeds)} have a crop "
                        f"interval above zero. The spread over fold assignments "
                        f"({seeds['diff'].max() - seeds['diff'].min():.4f}) is "
                        f"{'larger' if seeds['diff'].max() - seeds['diff'].min() > abs(MIN_GAIN - v['diff']) else 'smaller'}"
                        f" than the gap between the primary estimate and the bar ({abs(MIN_GAIN - v['diff']):.4f}), "
                        "and this spread is not in the bootstrap intervals. Per seed: `sensitivity_seeds.csv`."]
    if prov is not None:
        commit = prov.get("commit") or "unknown"
        state = (" (the analysis code had uncommitted changes)" if prov.get("code_dirty")
                 else "" if prov.get("code_dirty") is False else " (git state unknown)")
        out += ["", "## 6. Replication", "",
                "Inputs are local and not in the repository (per-crop data stays outside it): the "
                "`vlm_scoreboard.py` staging directory and the round-3 run's `<Type>/split.csv` files. With them in "
                "place:", "", "```bash",
                "python analysis/vlm_stacking.py --scores-dir D:/Data/sidewalk-validator-ai-vlm-gemini/staging \\",
                "    --run-dir D:/Data/sidewalk-validator-ai-vlm-gemini/round3", "```", "",
                f"This page was generated from commit `{commit}`{state}, with "
                + ", ".join(f"{k} {ver}" for k, ver in prov["versions"].items())
                + ". The script rewrites this page and `docs/vlm-stacking-2026-11/*.csv`; a run on the same inputs "
                  "should reproduce them exactly. Expected input hashes (SHA-256, also in `provenance.json`):", "",
                "| Input | SHA-256 |", "|---|---|"]
        out += [f"| `{k}` | `{h}` |" for k, h in prov["inputs"].items()]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores-dir", required=True, help="vlm_scoreboard staging directory (read only)")
    ap.add_argument("--run-dir", required=True, help="round-3 run root with <Type>/split.csv (read only)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--doc", default=DOC)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    df, counts = load_both(args.scores_dir)
    counts["n_r3_clipped"] = int(((df.p_r3 < CLIP["p_r3"]) | (df.p_r3 > 1 - CLIP["p_r3"])).sum())
    hold, trained = holdout_check(df, load_run_splits(args.run_dir))
    res = analyse(df, args.n_boot)
    clean = analyse(df[~trained].reset_index(drop=True), args.n_boot) if trained.any() else None
    sens, seeds = sensitivity(df, args.n_boot)
    os.makedirs(args.out_dir, exist_ok=True)
    for suffix, r in (("", res), ("_clean", clean)):
        for name, table in (r or {}).items():
            table.to_csv(os.path.join(args.out_dir, f"{name}{suffix}.csv"), index=False, float_format=sb.FLOAT)
    sens.to_csv(os.path.join(args.out_dir, "sensitivity.csv"), index=False, float_format=sb.FLOAT)
    seeds.to_csv(os.path.join(args.out_dir, "sensitivity_seeds.csv"), index=False, float_format=sb.FLOAT)
    hold.to_csv(os.path.join(args.out_dir, "holdout.csv"), index=False)
    files = [os.path.join(args.scores_dir, d, leg, f"{t}.csv") for d, leg in
             (("scores", CLASSIFIER), ("scores", VLM), ("splits", ""))for t in LABEL_TYPES]
    files += [os.path.join(args.run_dir, t, "split.csv") for t in LABEL_TYPES]
    commit, dirty = _code_state()
    prov = {"seed": SEED, "n_boot": args.n_boot, "n_folds": N_FOLDS, "clip": CLIP, "threshold": THRESHOLD,
            "min_gain": MIN_GAIN, "sensitivity": {"first_version_clip": FIRST_VERSION_CLIP,
                                                  "fold_seeds": list(FOLD_SEEDS), "refit_boot": REFIT_BOOT},
            "counts": counts, "n_trained_on": int(trained.sum()), "commit": commit, "code_dirty": dirty,
            "versions": _versions(),
            "inputs": {"/".join(os.path.normpath(f).split(os.sep)[-3:]): sb.sha256_of(f)
                       for f in files if os.path.exists(f)}}
    with open(os.path.join(args.out_dir, "provenance.json"), "w", newline="\n") as f:
        json.dump(prov, f, indent=1)
    with open(args.doc, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(counts, hold, res, clean, int(trained.sum()), sens, seeds, prov))
    yes, v = verdict(res["paired"])
    print(f"stack - round 3, macro AUC: {v['diff']:+.4f} crop [{v.crop_lo:+.4f}, {v.crop_hi:+.4f}] "
          f"city [{v.city_lo:+.4f}, {v.city_hi:+.4f}] -> {'YES' if yes else 'NO'}")


if __name__ == "__main__":
    main()
