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
   resample.
3. **Error correlation** at ``THRESHOLD`` (0.5 on ``p_incorrect`` for both legs, the scoreboard's argmax; the
   calibrated 0.92 gate of the deployed validator is not applied here): among the crops one leg gets wrong, how often
   the other is right, next to the other leg's accuracy on all crops; and the AUC of each leg inside round 3's
   uncertain band, which is the middle 20% of its scores by rank within each type (``BAND``, the 40th to 60th
   percentile), with the band ``0.4 <= p_incorrect <= 0.6`` as a second definition.
4. **Hold-out check.** Every crop is looked up in the round-3 ``split.csv`` (the manifest the classifier trained
   from). A crop whose split is ``train`` or ``val``, or whose panorama is shared with a ``train``/``val`` row of the
   same label type (each type has its own classifier), counts as trained on; if any exist, every number is redone
   without them (``*_clean.csv``).

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
CLIP = {"p_r3": 1e-6, "p_vlm": 0.01}   # Flash reports an integer confidence, so 0.01 is its resolution
THRESHOLD = 0.5
BAND = (0.40, 0.60)                    # round 3's uncertain band, as quantiles of its score within a label type
P_BAND = (0.40, 0.60)                  # the second definition, on p_incorrect itself
MIN_GAIN = 0.005
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


def oof_scores(x, y, city, raw=False, n_folds=N_FOLDS):
    """``(scores, weights per fold)``: out-of-fold scores of a logistic regression on ``x`` (see ``_fit_predict``),
    folds grouped by ``city`` (no city on both sides of a fold; ``GroupKFold`` is deterministic).

    >>> x = np.arange(12.0)[:, None]; y = np.tile([0, 1], 6); city = np.repeat(list("abcdef"), 2)
    >>> bool((oof_scores(x, y, city, n_folds=3)[0] == x[:, 0]).all())     # one column: the score itself
    True
    """
    from sklearn.model_selection import GroupKFold
    x, y, city = np.asarray(x, float), np.asarray(y, int), np.asarray(city)
    out, weights = np.full(len(y), np.nan), []
    for tr, te in GroupKFold(n_splits=n_folds).split(x, y, city):
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
    rows = []
    for t in list(LABEL_TYPES) + ["all"]:
        g = df if t == "all" else df[df.label_type == t]
        if g.empty:
            continue
        r_ok = ((g.p_r3 >= threshold).astype(int) == g.y).to_numpy()
        v_ok = ((g.p_vlm >= threshold).astype(int) == g.y).to_numpy()
        rows.append({"label_type": t, "n": len(g), "round3_accuracy": r_ok.mean(), "flash_accuracy": v_ok.mean(),
                     "n_round3_wrong": int((~r_ok).sum()), "flash_right_when_round3_wrong": v_ok[~r_ok].mean(),
                     "n_flash_wrong": int((~v_ok).sum()), "round3_right_when_flash_wrong": r_ok[~v_ok].mean(),
                     "n_both_wrong": int((~r_ok & ~v_ok).sum()),
                     "error_phi": float(np.corrcoef(~r_ok, ~v_ok)[0, 1])})
    return pd.DataFrame(rows)


def band_mask(df, kind):
    """Round 3's uncertain band: ``quantile`` = the ``BAND`` quantiles of its score within each label type;
    ``p`` = ``P_BAND`` on ``p_incorrect``."""
    if kind == "p":
        return df.p_r3.between(*P_BAND).to_numpy()
    lo = df.groupby("label_type").p_r3.transform(lambda s: s.quantile(BAND[0]))
    hi = df.groupby("label_type").p_r3.transform(lambda s: s.quantile(BAND[1]))
    return ((df.p_r3 >= lo) & (df.p_r3 <= hi)).to_numpy()


def band_rows(df, n_boot, seed):
    """AUC of round 3, Flash and the per-type stack inside round 3's uncertain band, both definitions."""
    cols, rows = ("round3", "flash", "stack_type"), []
    for kind in ("quantile", "p"):
        b = df[band_mask(df, kind)]
        pts, boots = {c: [] for c in cols}, {c: [] for c in cols}
        for t in LABEL_TYPES:
            g = b[b.label_type == t]
            row = {"band": kind, "label_type": t, "n": len(g), "n_incorrect": int(g.y.sum()),
                   "p_r3_min": g.p_r3.min(), "p_r3_max": g.p_r3.max()}
            if len(g) and 0 < g.y.sum() < len(g):
                d = sb.boot_auc(g.y, g[list(cols)].to_numpy(), n_boot, sb._type_rng(seed, t, 2))
                for k, c in enumerate(cols):
                    row[f"auc_{c}"] = sb.auc(g.y, g[c])
                    row[f"auc_{c}_lo"], row[f"auc_{c}_hi"] = sb.ci(d[:, k])
                    if min(g.y.sum(), len(g) - g.y.sum()) >= 10:    # a macro over types with 10+ of each class
                        pts[c].append(row[f"auc_{c}"])
                        boots[c].append(d[:, k])
            rows.append(row)
        row = {"band": kind, "label_type": f"macro ({len(pts[cols[0]])} types with 10+ of each class)", "n": len(b),
               "n_incorrect": int(b.y.sum()), "p_r3_min": b.p_r3.min(), "p_r3_max": b.p_r3.max()}
        if pts[cols[0]]:
            for c in cols:
                row[f"auc_{c}"] = float(np.mean(pts[c]))
                row[f"auc_{c}_lo"], row[f"auc_{c}_hi"] = sb.ci(np.mean(boots[c], axis=0))
        rows.append(row)
    return pd.DataFrame(rows)


def analyse(df, n_boot=N_BOOT, seed=SEED):
    """Every table for one set of crops: ``{"auc", "paired", "errors", "band", "weights"}``."""
    df, weights = add_models(df)
    draws = crop_draws(df, n_boot, seed)
    return {"auc": auc_rows(df, draws), "paired": paired_rows(df, draws, city_draws(df, n_boot, seed)),
            "errors": error_rows(df), "band": band_rows(df, n_boot, seed), "weights": weights}


def verdict(paired):
    """``(yes, row)`` for the pre-stated rule on the per-type stack: macro gain over round 3 of at least
    ``MIN_GAIN`` with a crop-bootstrap interval that excludes zero."""
    r = paired[(paired.model == "stack_type") & (paired.reference == "round3") & (paired.label_type == "macro")].iloc[0]
    return bool(r["diff"] >= MIN_GAIN and r.crop_lo > 0), r


# --------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------

def _iv(v, lo, hi, signed=False):
    f = "%+.3f" if signed else "%.3f"
    return "" if pd.isna(v) else f"{f % v} [{f % lo}, {f % hi}]"


def render(counts, hold, res, clean, n_trained):
    """The results page (aggregates only)."""
    a, p, e, b = res["auc"], res["paired"], res["errors"], res["band"]
    yes, v = verdict(p)
    best = p[(p.label_type == "macro") & (p.model != "flash") & (p.reference == "round3")]["diff"].max()
    city_excl = v.city_lo > 0 or v.city_hi < 0
    out = ["# Does Gemini 3.8 Flash add information to round 3? A stacking check (2026-11)", "",
           "Generated by `analysis/vlm_stacking.py` from existing score files; no model was called. Methods are in "
           "the script's docstring. `incorrect` is the positive class; intervals are 95% percentile intervals from "
           f"{N_BOOT:,} resamples (seed {SEED}).", "",
           f"**Verdict: {'YES' if yes else 'NO'}.** The per-type stack (city-grouped {N_FOLDS}-fold CV) changes "
           f"macro AUC by {_iv(v['diff'], v.crop_lo, v.crop_hi, True)} over round 3 (crop bootstrap); the city "
           f"bootstrap gives [{v.city_lo:+.3f}, {v.city_hi:+.3f}], which "
           f"{'excludes' if city_excl else 'includes'} zero. The rule was a gain of at least {MIN_GAIN} with a "
           "crop-bootstrap interval excluding zero. Against round 3 itself, the largest macro gain of any stack "
           f"variant below, the in-sample one included, is {best:+.4f}.", "",
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
    out += ["", f"## 3. Error correlation at p_incorrect ≥ {THRESHOLD} (both legs)", "",
            "| Label type | n | round 3 acc. | Flash acc. | round 3 wrong | Flash right on those | Flash wrong | "
            "round 3 right on those | both wrong | φ of errors |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in e.iterrows():
        out.append(f"| {r.label_type} | {int(r.n):,} | {r.round3_accuracy:.3f} | {r.flash_accuracy:.3f} | "
                   f"{int(r.n_round3_wrong):,} | {r.flash_right_when_round3_wrong:.3f} | {int(r.n_flash_wrong):,} | "
                   f"{r.round3_right_when_flash_wrong:.3f} | {int(r.n_both_wrong):,} | {r.error_phi:.2f} |")
    for kind, title in (("quantile", f"the {int(BAND[0] * 100)}th to {int(BAND[1] * 100)}th percentile of its score "
                                     "within each label type (the middle 20% by rank)"),
                        ("p", f"{P_BAND[0]} ≤ p_incorrect ≤ {P_BAND[1]}")):
        out += ["", f"Round 3's uncertain band = {title}:", "",
                "| Label type | n (incorrect) | round-3 p range | round 3 AUC | Flash AUC | stack AUC |",
                "|---|---|---|---|---|---|"]
        for _, r in b[b.band == kind].iterrows():
            rng = "" if not r.n else f"{r.p_r3_min:.2g} to {r.p_r3_max:.2g}"
            cells = [_iv(r.get(f"auc_{c}"), r.get(f"auc_{c}_lo"), r.get(f"auc_{c}_hi"))
                     for c in ("round3", "flash", "stack_type")]
            out.append(f"| {r.label_type} | {int(r.n):,} ({int(r.n_incorrect):,}) | {rng} | " + " | ".join(cells) + " |")
    out += ["", "## 4. Were these crops held out from round 3's training?", "",
            "Each crop was looked up by `city:label_id` in the round-3 run's `<Type>/split.csv`, the manifest "
            "`fine_tune_classifier.py` trains from (`split_data.py`: grouped by panorama, panoramas within 3 m "
            "merged, three deployments held out whole as `test_heldout`).", "", "| Check | n |", "|---|---|"]
    out += [f"| {r.check} | {int(r.n):,} |" for _, r in hold.iterrows()]
    if n_trained and clean is not None:
        yes2, v2 = verdict(clean["paired"])
        out += ["", f"{n_trained:,} crops were trained on. Without them: stack − round 3 macro "
                    f"{_iv(v2['diff'], v2.crop_lo, v2.crop_hi, True)} (crop), [{v2.city_lo:+.3f}, {v2.city_hi:+.3f}] "
                    f"(city); verdict {'YES' if yes2 else 'NO'}. Tables in `*_clean.csv`."]
    else:
        out += ["", "None was trained on, so no number is redone. The `test` crops are from deployments round 3 "
                    "trained on (other panoramas); only the `test_heldout` crops are from an unseen deployment."]
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
    hold, trained = holdout_check(df, load_run_splits(args.run_dir))
    res = analyse(df, args.n_boot)
    clean = analyse(df[~trained].reset_index(drop=True), args.n_boot) if trained.any() else None
    os.makedirs(args.out_dir, exist_ok=True)
    for suffix, r in (("", res), ("_clean", clean)):
        for name, table in (r or {}).items():
            table.to_csv(os.path.join(args.out_dir, f"{name}{suffix}.csv"), index=False, float_format=sb.FLOAT)
    hold.to_csv(os.path.join(args.out_dir, "holdout.csv"), index=False)
    files = [os.path.join(args.scores_dir, d, leg, f"{t}.csv") for d, leg in
             (("scores", CLASSIFIER), ("scores", VLM), ("splits", ""))for t in LABEL_TYPES]
    files += [os.path.join(args.run_dir, t, "split.csv") for t in LABEL_TYPES]
    prov = {"seed": SEED, "n_boot": args.n_boot, "n_folds": N_FOLDS, "clip": CLIP, "threshold": THRESHOLD,
            "band": BAND, "p_band": P_BAND, "counts": counts, "n_trained_on": int(trained.sum()),
            "inputs": {"/".join(os.path.normpath(f).split(os.sep)[-3:]): sb.sha256_of(f)
                       for f in files if os.path.exists(f)}}
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    with open(args.doc, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(counts, hold, res, clean, int(trained.sum())))
    yes, v = verdict(res["paired"])
    print(f"stack - round 3, macro AUC: {v['diff']:+.4f} crop [{v.crop_lo:+.4f}, {v.crop_hi:+.4f}] "
          f"city [{v.city_lo:+.4f}, {v.city_hi:+.4f}] -> {'YES' if yes else 'NO'}")


if __name__ == "__main__":
    main()
