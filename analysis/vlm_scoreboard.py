"""The open-weight VLM scoreboard (issue #63): every leg on the round-3 held-out split, one generated page.

Rows are models (the round-3 checkpoints, production, every open-weight VLM leg, the 8-shot arm, the 4-bit Mac legs
of #16); columns are macro-AUC over the five label types (each type equal weight), per-type AUC with a 95% label
bootstrap interval, test vs held-out-deployment AUC, within-city AUC, balanced accuracy at the argmax, recall of
``incorrect`` at precision 0.95, s/crop with the GPU, and coverage. ``incorrect`` is the positive class and the
score is ``p_incorrect`` everywhere. Writes ``docs/vlm-scoreboard.md`` (generated blocks only; the prose around them
is hand-written) and ``docs/vlm-scoreboard/*.{csv,png,json}``.

**Per-crop data never enters the repo.** Score CSVs and splits carry ``label_uid`` (``city:label_id``), which names
private deployments, so they live in a staging directory outside the checkout (``--scores-dir``, default
``~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10``)::

    <scores-dir>/
      splits/<Type>.csv                # label_uid, city, label_type, label_class, split (test + test_heldout rows)
      scores/<leg>/<Type>.csv          # label_uid, p_incorrect, ... (vlm_baseline score CSV, or round3/production)
      scores/<leg>/<Type>.meta.json    # optional: repo, revision, gpu, batch_size, ...
      scores/timing*.csv               # model, gpu, batch_size, n_crops, median_s_per_crop, p90_s_per_crop, source
      scores/notes.json                # optional: {leg: {"*" | Type: "why this cell is missing or partial"}}

The repo gets only aggregates (no ``label_uid``, no private deployment id), figures, and ``provenance.json`` with the
sha256 and row count of every staged file and the resolved public-deployment list. ``stage`` fills the staging
directory from what is on this Mac: the held-out rows of the round-3 ``split.csv`` files, the round-3
``predictions.csv`` (held-out rows), and the three 4-bit MLX legs of #16 (``<model>-mlx4``).

**Naming.** A deployment is named only if it is public in the live ``/v3/api/cities`` at build time
(``retrain_report.public_cities``; ``--public-cities`` overrides with a saved response or a list). The two private
held-out deployments are lettered as in ``docs/retrain-2026-10.md``; every other private deployment is pooled.

**Ordering.** Legs that scored all held-out crops ("full") are ranked by macro-AUC. Partial legs sit below a rule,
sorted by coverage then macro-AUC, and are never ranked against the full ones (their AUCs are on other crops).

**Drift check.** ``build --check`` has two modes. With the staging directory present it regenerates every aggregate
CSV into a temporary directory and compares bytes with the committed ones, then checks the doc. Without it (CI) it
renders the doc's generated blocks from the committed aggregate CSVs and checks them and the prose counts
(``PROSE_RULES``, as RampNet's ``scoreboard_render.py``) against the doc. Both modes read the public-deployment list
from the live API (10 s timeout) and fail if a deployment the committed ``per_deployment.csv`` names has gone
private since the build; offline they fall back to the list frozen in ``provenance.json``.

Usage::

    python analysis/vlm_scoreboard.py stage --run-dir ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 \\
        --mac-dir ~/Data/sidewalk-validator-ai-vlm-2026-10
    python analysis/vlm_scoreboard.py build                  # aggregates, doc blocks, figures, provenance
    python analysis/vlm_scoreboard.py build --check          # exit 1 on any drift

Example (the macro-AUC is the mean of the per-type AUCs)::

    >>> macro({"CurbRamp": 0.9, "Obstacle": 0.7})
    0.8
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "analysis"))
import evaluate  # noqa: E402
import vlm_baseline as vb  # noqa: E402
from retrain_report import CITIES_API, heldout_names, public_cities  # noqa: E402

LABEL_TYPES = vb.LABEL_TYPES
SPLITS = ("test", "test_heldout")
DEFAULT_SCORES_DIR = os.path.expanduser("~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10")
DOC = os.path.join(REPO_DIR, "docs", "vlm-scoreboard.md")
OUT_DIR = os.path.join(REPO_DIR, "docs", "vlm-scoreboard")
N_BOOT, SEED = 1000, 20261021
REFERENCE = "round3"
FLOAT = "%.4f"

PURPOSE, CHAT, POINTING = "purpose-trained", "chat VLM", "pointing model"

LEGS = {
    # leg -> (display, class, why it is on the board). Order is the coverage table's order for legs not yet run.
    "round3": ("Round 3 (DINOv2-L, 2026-10)", PURPOSE, "the retrained validators"),
    "production": ("Production (DINOv2-L, 2025-08)", PURPOSE, "the deployed validators"),
    "qwen3.8-27b": ("Qwen3.8-27B", CHAT, ""),
    "gemma4-31b": ("Gemma 4 31B", CHAT, ""),
    "gemma4-12b-bf16": ("Gemma 4 12B", CHAT, ""),
    "molmo2-8b": ("Molmo2-8B", POINTING, ""),
    "minicpm-v4.7-35b-a3b": ("MiniCPM-V 4.7 35B-A3B", CHAT, ""),
    "muse-glimmer-30b": ("Muse Glimmer 30B", CHAT, ""),
    "nemotron3.5-super-vl-120b": ("Nemotron 3.5 Super VL 120B-A12B", CHAT, ""),
    "glm5.3-flash": ("GLM-5.3-Flash (FP8, vLLM)", CHAT, ""),
    "qwen3vl-8b-bf16": ("Qwen3-VL-8B", CHAT, ""),
    "qwen3vl-32b-bf16": ("Qwen3-VL-32B", CHAT, ""),
    "phi4-rv-15b": ("Phi-4-reasoning-vision-15B", CHAT, "optional"),
    "qwen3vl-2b-mlx4": ("Qwen3-VL-2B (4-bit, Mac)", CHAT, ""),
    "qwen3vl-8b-mlx4": ("Qwen3-VL-8B (4-bit, Mac)", CHAT, ""),
    "gemma4-12b-mlx4": ("Gemma 4 12B (4-bit, Mac)", CHAT, ""),
}
MAC_LEGS = {"qwen3vl-2b": "qwen3vl-2b-mlx4", "qwen3vl-8b": "qwen3vl-8b-mlx4", "gemma4-12b": "gemma4-12b-mlx4"}
DEFAULT_WHY = {
    "phi4-rv-15b": "optional leg; not run",
    "production": "not staged yet (the full-split production run is on klone)",
    "fewshot8": "the 8-shot arm runs on the best zero-shot leg once the board names it",
}
for _leg in MAC_LEGS.values():
    DEFAULT_WHY[_leg] = "4-bit, Mac, sample (#16's 1,988-crop sample or its 500-crop subset)"


def leg_info(leg):
    """``(display, class)`` of a leg; ``<model>-fewshot8`` is the 8-shot arm of ``<model>``; unknown legs are chat VLMs.

    >>> leg_info("gemma4-31b-fewshot8")
    ('Gemma 4 31B, 8-shot', 'chat VLM')
    """
    if leg.endswith("-fewshot8"):
        d, c = leg_info(leg[:-len("-fewshot8")])
        return f"{d}, 8-shot", c
    d, c, _ = LEGS.get(leg, (leg, CHAT, ""))
    return d, c


def macro(per_type):
    """Mean over the types present (each type equal weight), or NaN when none are."""
    vals = [v for v in per_type.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------------------------------------

def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_splits(scores_dir):
    """The held-out rows of every type, one frame; ``y`` = 1 for ``incorrect``. Unique ``label_uid``."""
    frames = []
    for t in LABEL_TYPES:
        p = os.path.join(scores_dir, "splits", f"{t}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=["label_uid", "city", "label_type", "label_class", "split"]))
    if not frames:
        raise SystemExit(f"no splits under {scores_dir}/splits; run `stage` first")
    s = pd.concat(frames, ignore_index=True)
    s = s[s.split.isin(SPLITS)]
    if s.label_uid.duplicated().any():
        raise ValueError("splits hold duplicate label_uid")
    return s.assign(y=(s.label_class == "incorrect").astype(int)).reset_index(drop=True)


UNSCORED_WHY = "{n} rows unscored: neither yes nor no in the top-20 log-probs"


def load_leg(leg_dir):
    """All ``<Type>.csv`` of one leg as ``(frame, unscored)``: the frame is ``label_uid, p_incorrect`` (definition
    variant only); ``unscored`` is ``{Type: n}`` for rows dropped because the vLLM client found neither answer in the
    top 20 (``answer_floor == "both"``, scored 0.5 as a placeholder; the column is absent for other backends). Exact
    duplicate rows (a resumed run that re-wrote a chunk) are dropped; conflicting duplicates raise. ``(None, {})``
    when the leg has no CSV."""
    frames, unscored = [], {}
    for t in LABEL_TYPES:
        p = os.path.join(leg_dir, f"{t}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "variant" in d and d.variant.nunique() > 1:
                raise ValueError(f"{p}: more than one prompt variant ({sorted(d.variant.unique())})")
            if "answer_floor" in d:
                both = d.answer_floor.astype(str).eq("both")
                if both.any():
                    unscored[t] = int(both.sum())
                    d = d[~both]
            frames.append(d[["label_uid", "p_incorrect"]])
    if not frames:
        return None, {}
    d = pd.concat(frames, ignore_index=True).drop_duplicates()
    if d.label_uid.duplicated().any():
        raise ValueError(f"{leg_dir}: conflicting duplicate scores for one label")
    return d, unscored


def load_scores(scores_dir, splits):
    """``({leg: per-crop frame joined to the splits}, {leg: {Type: n unscored}})``. A score for a crop that is not in
    the splits raises."""
    root = os.path.join(scores_dir, "scores")
    legs, unscored = {}, {}
    for leg in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not os.path.isdir(os.path.join(root, leg)):
            continue
        d, u = load_leg(os.path.join(root, leg))
        if d is None:
            continue
        m = d.merge(splits, on="label_uid", how="left", validate="one_to_one")
        if m.label_type.isna().any():
            raise ValueError(f"{leg}: {int(m.label_type.isna().sum())} scores for crops not in the held-out splits")
        legs[leg] = m
        if u:
            unscored[leg] = u
    return legs, unscored


def load_timing(scores_dir):
    paths = sorted(f for f in os.listdir(os.path.join(scores_dir, "scores")) if re.match(r"timing.*\.csv$", f)) \
        if os.path.isdir(os.path.join(scores_dir, "scores")) else []
    if not paths:
        return pd.DataFrame(columns=["model", "gpu", "batch_size", "n_crops", "median_s_per_crop", "p90_s_per_crop",
                                     "source"])
    return pd.concat([pd.read_csv(os.path.join(scores_dir, "scores", f)) for f in paths], ignore_index=True)


def load_notes(scores_dir):
    p = os.path.join(scores_dir, "scores", "notes.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# --------------------------------------------------------------------------------------------------------------------
# Statistics -- pure, unit-tested
# --------------------------------------------------------------------------------------------------------------------

def _rank_auc(y_rows, s_rows):
    """Row-wise AUC (Mann-Whitney with tie-averaged ranks, so equal to sklearn's ``roc_auc_score``)."""
    from scipy.stats import rankdata
    r = rankdata(s_rows, axis=1)
    npos = y_rows.sum(axis=1)
    nneg = y_rows.shape[1] - npos
    return ((r * y_rows).sum(axis=1) - npos * (npos + 1) / 2) / (npos * nneg)


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.min() == y.max():
        return float("nan")
    return float(_rank_auc(y[None, :], s[None, :])[0])


def boot_auc(y, scores, n_boot=N_BOOT, rng=None):
    """Class-stratified label bootstrap of the AUC of each column of ``scores`` (``n`` or ``n x k``), with the SAME
    draws for every column (so differences are paired). Returns ``n_boot x k``; NaN rows when a class is missing.

    >>> y = np.array([0, 0, 1, 1]); b = boot_auc(y, np.array([0.1, 0.2, 0.8, 0.9]), 50, np.random.default_rng(0))
    >>> b.shape, float(b.min())
    ((50, 1), 1.0)
    """
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    s = s[:, None] if s.ndim == 1 else s
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.full((n_boot, s.shape[1]), np.nan)
    rng = rng or np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))], axis=1)
    yr = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])[None, :].repeat(n_boot, 0)
    return np.stack([_rank_auc(yr, s[idx, k]) for k in range(s.shape[1])], axis=1)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan"), float("nan")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def _type_rng(seed, t, salt=0):
    return np.random.default_rng([seed, salt, LABEL_TYPES.index(t)])


def per_type_rows(leg, df, n_boot, seed, within_min_n=10):
    """One row per type for one leg: n, AUC with its label-bootstrap CI, per-split AUC, within-city AUC, balanced
    accuracy at the argmax and recall of incorrect at precision 0.95. Also returns the bootstrap draws per type."""
    rows, draws = [], {}
    for t in LABEL_TYPES:
        g = df[df.label_type == t]
        if g.empty:
            continue
        b = boot_auc(g.y, g.p_incorrect, n_boot, _type_rng(seed, t))[:, 0]
        draws[t] = b
        lo, hi = ci(b)
        m = vb.metrics_row(g.assign(true=g.label_class, p_correct=1 - g.p_incorrect))
        wc, nc = vb.within_city_auc(g.y, g.p_incorrect, g.city, within_min_n)
        row = {"leg": leg, "label_type": t, "n": len(g), "n_incorrect": int(g.y.sum()), "auc": auc(g.y, g.p_incorrect),
               "auc_lo": lo, "auc_hi": hi}
        for sp in SPLITS:
            gs = g[g.split == sp]
            row[f"auc_{sp}"] = auc(gs.y, gs.p_incorrect) if len(gs) else float("nan")
            row[f"n_{sp}"] = len(gs)
        row.update(within_city_auc=wc, n_cities=nc, balanced_accuracy=m["balanced_accuracy"],
                   r_incorrect_p95=m["r_incorrect_p95"])
        rows.append(row)
    return rows, draws


def paired_rows(leg, df, ref, n_boot, seed):
    """Leg minus the reference on the crops both scored: per type and macro, paired label bootstrap (95%)."""
    m = df[["label_uid", "label_type", "y", "p_incorrect"]].merge(
        ref[["label_uid", "p_incorrect"]], on="label_uid", suffixes=("", "_ref"), validate="one_to_one")
    rows, diffs, a_all, b_all = [], [], [], []
    for t in LABEL_TYPES:
        g = m[m.label_type == t]
        if g.empty or g.y.nunique() < 2:
            continue
        b = boot_auc(g.y, g[["p_incorrect", "p_incorrect_ref"]].to_numpy(), n_boot, _type_rng(seed, t, 1))
        a, r = auc(g.y, g.p_incorrect), auc(g.y, g.p_incorrect_ref)
        lo, hi = ci(b[:, 0] - b[:, 1])
        rows.append({"leg": leg, "reference": REFERENCE, "label_type": t, "n": len(g), "auc_leg": a, "auc_ref": r,
                     "diff": a - r, "diff_lo": lo, "diff_hi": hi})
        diffs.append(b[:, 0] - b[:, 1])
        a_all.append(a)
        b_all.append(r)
    if rows:
        lo, hi = ci(np.mean(diffs, axis=0))
        rows.append({"leg": leg, "reference": REFERENCE, "label_type": f"macro ({len(rows)} types)",
                     "n": int(sum(r["n"] for r in rows)), "auc_leg": float(np.mean(a_all)),
                     "auc_ref": float(np.mean(b_all)), "diff": float(np.mean(a_all) - np.mean(b_all)),
                     "diff_lo": lo, "diff_hi": hi})
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Deployments
# --------------------------------------------------------------------------------------------------------------------

def is_us(city):
    """A US deployment's id ends in a two-letter state code (``seattle-wa``), as ``retrain_report.main`` assumes."""
    tail = str(city).split("-")[-1]
    return "-" in str(city) and len(tail) == 2 and tail.isalpha()


def deployment_names(splits, public):
    """``{city: display}``: public ids as is; the private held-out deployments lettered (``retrain_report``); every
    other private deployment pooled into ``other (N private deployments)``.

    >>> s = pd.DataFrame({"city": ["a-wa", "b", "c-or", "d", "e"], "split": ["test", "test", "test_heldout",
    ...                   "test_heldout", "test"]})
    >>> deployment_names(s, {"a-wa"})["d"], deployment_names(s, {"a-wa"})["e"]
    ('private city A (outside the US)', 'other (2 private deployments)')
    """
    heldout = sorted(set(splits[splits.split == "test_heldout"].city))
    names = heldout_names(heldout, public, {c for c in heldout if is_us(c)})
    rest = sorted(set(splits.city) - set(names))
    private = [c for c in rest if c not in public]
    for c in rest:
        names[c] = c if c in public else vb.private_label(len(private))
    return names


def per_deployment_rows(leg, df, names, public, min_n=30):
    """Per displayed deployment (all five types): n, pooled AUC and the mean within-type AUC. Public deployments
    below ``min_n`` rows are left out; private rows are kept at any size (pooled or lettered)."""
    d = df.assign(deployment=df.city.map(names))
    rows = []
    for dep, g in d.groupby("deployment"):
        pub = dep in public
        if pub and len(g) < min_n:
            continue
        mwt, nt = vb.mean_within_type_auc(g.assign(true=g.label_class))
        rows.append({"leg": leg, "deployment": dep, "public": pub,
                     "held_out": bool((g.split == "test_heldout").all()), "n": len(g), "n_incorrect": int(g.y.sum()),
                     "auc_pooled": auc(g.y, g.p_incorrect), "mean_within_type_auc": mwt, "n_types": nt})
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------------------------------------------------

def timing_rows(timing):
    """The staged timing rows plus GPU-hours per 1,000 labels and per nightly 800-label city (``n_gpus`` x s/crop)."""
    if timing.empty:
        return timing
    t = timing.copy()
    if "n_gpus" not in t:
        t["n_gpus"] = np.nan
    # a missing count is read off the GPU name ("4xH200", "2 x L40S"), else 1
    parsed = t.gpu.astype(str).str.extract(r"^\s*(\d+)\s*[x×]", expand=False).astype(float)
    t["n_gpus"] = t.n_gpus.fillna(parsed).fillna(1).astype(int)
    t["gpu_hours_per_1000"] = t.median_s_per_crop * t.n_gpus * 1000 / 3600
    t["gpu_hours_per_800"] = t.median_s_per_crop * t.n_gpus * 800 / 3600
    t["display"] = t.model.map(lambda m: leg_info(m)[0])
    cols = ["model", "display", "gpu", "n_gpus", "batch_size", "n_crops", "median_s_per_crop", "p90_s_per_crop",
            "gpu_hours_per_1000", "gpu_hours_per_800", "source"]
    return t[[c for c in cols if c in t]].sort_values(["model", "batch_size"]).reset_index(drop=True)


def run_speed(timing, leg):
    """``(s/crop, gpu, batch)`` at the largest batch timed for a leg, or Nones. Among rows tied on the batch size, a
    full-run row (``source == "full run"``) beats a timing run, then the row over more crops wins.

    >>> t = pd.DataFrame({"model": "m", "gpu": ["A", "B", "C"], "batch_size": 1, "n_crops": [50, 900, 9000],
    ...                   "median_s_per_crop": [0.9, 0.3, 0.2], "source": ["timing", "full run", "full run"]})
    >>> run_speed(t.iloc[::-1], "m")
    (0.2, 'C', 1)
    """
    t = timing[timing.model == leg] if len(timing) else timing
    if t is None or t.empty:
        return None, None, None
    full = t.source.astype(str).eq("full run") if "source" in t else pd.Series(False, index=t.index)
    n = t.n_crops if "n_crops" in t else pd.Series(0, index=t.index)
    r = t.assign(_full=full, _n=n).sort_values(["batch_size", "_full", "_n"], kind="mergesort").iloc[-1]
    return float(r.median_s_per_crop), str(r.gpu), int(r.batch_size)


def coverage_rows(legs, splits, notes, unscored=None):
    """One row per (leg, type), including roster legs that are not staged yet, each with the reason it is partial or
    missing (rows the leg could not score, ``UNSCORED_WHY``, first; then ``notes.json``, then ``DEFAULT_WHY``)."""
    unscored = unscored or {}
    totals = splits.groupby("label_type").size()
    names = list(legs) + [k for k in LEGS if k not in legs] + (["fewshot8"] if not any(
        k.endswith("-fewshot8") for k in legs) else [])
    rows = []
    for leg in names:
        df = legs.get(leg)
        note = notes.get(leg, {})
        for t in LABEL_TYPES:
            n = 0 if df is None else int((df.label_type == t).sum())
            tot = int(totals.get(t, 0))
            why = note.get(t) or note.get("*") or ""
            nu = unscored.get(leg, {}).get(t, 0)
            if n + nu < tot and not why:
                why = DEFAULT_WHY.get(leg, "not run yet" if n == 0 else f"partial: run incomplete ({n + nu} of {tot})")
            if nu:
                why = "; ".join(w for w in (UNSCORED_WHY.format(n=nu), why) if w)
            rows.append({"leg": leg, "display": "8-shot arm" if leg == "fewshot8" else leg_info(leg)[0],
                         "label_type": t, "n": n, "total": tot, "complete": n == tot, "why": "" if n == tot else why})
    return rows


def build_aggregates(scores_dir, public, n_boot=N_BOOT, seed=SEED, min_city_n=30):
    """Every committed CSV, as ``{name: DataFrame}``, plus the per-crop frames the ROC figure needs."""
    splits = load_splits(scores_dir)
    legs, unscored = load_scores(scores_dir, splits)
    timing = load_timing(scores_dir)
    notes = load_notes(scores_dir)
    names = deployment_names(splits, public)
    n_total = len(splits)
    per_type, paired, per_dep, board = [], [], [], []
    for leg, df in legs.items():
        rows, draws = per_type_rows(leg, df, n_boot, seed)
        per_type += rows
        full = len(df) == n_total
        r = {x["label_type"]: x for x in rows}
        mac_draws = np.nanmean([draws[t] for t in draws], axis=0) if draws else np.array([np.nan])
        lo, hi = ci(mac_draws)
        disp, cls = leg_info(leg)
        sp, gpu, bs = run_speed(timing, leg)
        row = {"leg": leg, "display": disp, "model_class": cls, "n": len(df), "n_total": n_total, "full": full,
               "n_types": len(rows), "macro_auc": macro({t: x["auc"] for t, x in r.items()}),
               "macro_auc_lo": lo, "macro_auc_hi": hi}
        row.update({f"auc_{t}": r[t]["auc"] if t in r else float("nan") for t in LABEL_TYPES})
        row.update(auc_test=macro({t: x["auc_test"] for t, x in r.items()}),
                   auc_heldout=macro({t: x["auc_test_heldout"] for t, x in r.items()}),
                   within_city_auc=macro({t: x["within_city_auc"] for t, x in r.items()}),
                   balanced_accuracy=macro({t: x["balanced_accuracy"] for t, x in r.items()}),
                   r_incorrect_p95=macro({t: x["r_incorrect_p95"] for t, x in r.items()}),
                   s_per_crop=sp, gpu=gpu, batch_size=bs, diff_vs_round3=None, diff_lo=None, diff_hi=None)
        if leg != REFERENCE and REFERENCE in legs:
            pr = paired_rows(leg, df, legs[REFERENCE], n_boot, seed)
            paired += pr
            mrow = pr[-1] if pr else {}
            row.update(diff_vs_round3=mrow.get("diff"), diff_lo=mrow.get("diff_lo"), diff_hi=mrow.get("diff_hi"))
        board.append(row)
        if full:
            per_dep += per_deployment_rows(leg, df, names, public, min_city_n)
    board = order_board(pd.DataFrame(board))
    out = {"board": board, "per_type": pd.DataFrame(per_type), "paired": pd.DataFrame(paired),
           "per_deployment": pd.DataFrame(per_dep),
           "coverage": pd.DataFrame(coverage_rows(legs, splits, notes, unscored)),
           "timing": timing_rows(timing)}
    if len(out["per_type"]):
        out["per_type"] = out["per_type"].assign(order=out["per_type"].leg.map(
            {k: i for i, k in enumerate(board.leg)})).sort_values(["order", "label_type"]).drop(columns="order")
    return out, legs, splits, names


def order_board(board):
    """Full legs by macro-AUC (descending), then partial legs by coverage then macro-AUC; ``rank`` for full only.

    >>> b = pd.DataFrame({"leg": ["a", "b", "c"], "full": [False, True, True], "n": [5, 10, 10],
    ...                   "macro_auc": [0.99, 0.7, 0.8]})
    >>> order_board(b)[["leg", "rank"]].values.tolist()
    [['c', 1], ['b', 2], ['a', '']]
    """
    if board.empty:
        return board
    f = board[board.full].sort_values(["macro_auc", "leg"], ascending=[False, True])
    p = board[~board.full].sort_values(["n", "macro_auc", "leg"], ascending=[False, False, True])
    out = pd.concat([f, p], ignore_index=True)
    out["rank"] = [i + 1 for i in range(len(f))] + [""] * len(p)
    return out


def write_aggregates(aggs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in aggs.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, float_format=FLOAT)


def read_aggregates(out_dir):
    out = {}
    for name in ("board", "per_type", "paired", "per_deployment", "coverage", "timing"):
        p = os.path.join(out_dir, f"{name}.csv")
        try:
            out[name] = pd.read_csv(p, keep_default_na=False, na_values=[""])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            out[name] = pd.DataFrame()
    return out


# --------------------------------------------------------------------------------------------------------------------
# The doc -- generated blocks and prose rules (rendered from the committed aggregate CSVs only)
# --------------------------------------------------------------------------------------------------------------------

BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/vlm_scoreboard.py) -->"
END = "<!-- END GENERATED: {name} -->"


def f3(v):
    return "—" if v is None or v == "" or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}"


def interval(lo, hi):
    return "" if f3(lo) == "—" else f" [{f3(lo)}, {f3(hi)}]"


def best_vlm(board):
    """The non-purpose-trained leg with the highest macro-AUC among full legs, else among all; ``(row, is_full)``."""
    if board.empty:
        return None, False
    v = board[board.model_class != PURPOSE]
    for pool, full in ((v[v.full.astype(bool)], True), (v, False)):
        if len(pool):
            return pool.sort_values(["macro_auc", "leg"], ascending=[False, True]).iloc[0], full
    return None, False


def facts(aggs):
    """The numbers the hand-written prose quotes, read off the committed board (``PROSE_RULES``)."""
    b = aggs["board"]
    best, best_full = best_vlm(b)
    # the paired gap on the crops both scored (for a full leg, exactly round 3's macro-AUC minus the leg's)
    gap = float(-best.diff_vs_round3) if best is not None and f3(best.get("diff_vs_round3")) != "—" else float("nan")
    return {"legs": len(b), "full_legs": int(b.full.astype(bool).sum()) if len(b) else 0,
            "n_total": int(b.n_total.iloc[0]) if len(b) else 0, "best_vlm": best.display if best is not None else "",
            "gap": gap}


_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
_NUM = r"\b(\d+|(?i:" + "|".join(_NUMBER_WORDS) + r"))\b"
_DEC = r"(-?\d\.\d{3})"

PROSE_RULES = (
    # (name, regex, [fact per group], required)
    ("intro: legs", r"\bThe board has " + _NUM + r" legs\b", ["legs"], True),
    ("intro: full legs", r"\b" + _NUM + r" of them scored all ([\d,]+) held-out crops\b", ["full_legs", "n_total"],
     True),
    ("intro: headline gap", r"\bRound 3 leads the best VLM \((.+?)\) by " + _DEC + r" macro-AUC on the crops both "
     r"scored\b",
     ["best_vlm", "gap"], True),
    ("anywhere: N legs", r"\b" + _NUM + r" model legs\b", ["legs"], False),
)


def _agrees(word, value):
    if isinstance(value, float):
        return word == f"{value:.3f}"
    if isinstance(value, int):
        w = word.replace(",", "")
        return (int(w) if w.isdigit() else _NUMBER_WORDS.index(w.lower()) if w.lower() in _NUMBER_WORDS
                else None) == value
    return word == value


def prose_problems(text, aggs):
    """Every count or value in the hand-written prose that disagrees with the committed board; [] when current."""
    prose = re.sub(r"<!-- BEGIN GENERATED: (\S+) .*?<!-- END GENERATED: \1 -->", "", text, flags=re.S)
    prose = re.sub(r"\s+", " ", prose)
    fx = facts(aggs)
    problems = []
    for name, pattern, keys, required in PROSE_RULES:
        found = list(re.finditer(pattern, prose))
        if required and not found:
            problems.append(f"{name}: no sentence states it any more")
        for m in found:
            for word, key in zip(m.groups(), keys):
                if not _agrees(word, fx[key]):
                    shown = f"{fx[key]:.3f}" if isinstance(fx[key], float) else fx[key]
                    problems.append(f'{name}: "{m.group(0)}" says {word}, the board has {shown} ({key})')
    return sorted(set(problems))


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _int(v):
    return f"{int(v):,}"


def render_blocks(aggs):
    """``{block name: markdown}`` from the aggregate CSVs (no per-crop data needed)."""
    b, pt, pr, dep, cov, tm = (aggs[k] for k in ("board", "per_type", "paired", "per_deployment", "coverage",
                                                  "timing"))
    blocks = {}
    best, best_full = best_vlm(b)
    ref = b[b.leg == REFERENCE].iloc[0] if len(b) and (b.leg == REFERENCE).any() else None
    lines = []
    if ref is not None:
        lines.append(f"- **{ref.display}**: macro-AUC {f3(ref.macro_auc)}{interval(ref.macro_auc_lo, ref.macro_auc_hi)}"
                     f" on {'all ' if ref.n == ref.n_total else ''}{_int(ref.n)}"
                     f"{'' if ref.n == ref.n_total else ' of ' + _int(ref.n_total)} held-out crops.")
    if best is not None:
        scope = "the best VLM with full coverage" if best_full else \
            "the best VLM so far (no VLM leg has full coverage yet; this one is partial)"
        lines.append(f"- **{best.display}**, {scope}: macro-AUC {f3(best.macro_auc)}"
                     f"{interval(best.macro_auc_lo, best.macro_auc_hi)} on {_int(best.n)} crops.")
        if ref is not None and f3(best.get("diff_vs_round3")) != "—":
            lines.append(f"- Paired on the crops both scored, {best.display} minus round 3: "
                         f"{f3(best.diff_vs_round3)}{interval(best.diff_lo, best.diff_hi)} macro-AUC.")
    if len(b):
        prod = b[b.leg == "production"]
        if len(prod):
            p = prod.iloc[0]
            lines.append(f"- Production (optimistic, see caveats): macro-AUC {f3(p.macro_auc)}"
                         f"{interval(p.macro_auc_lo, p.macro_auc_hi)} on {_int(p.n)} crops.")
    blocks["short_answer"] = "\n".join(lines) if lines else "(no legs staged)"

    head = ["#", "model", "class", "macro-AUC [95% CI]"] + list(LABEL_TYPES) + [
        "test", "held-out", "within-city", "bal. acc.", "R@P95", "Δ vs round 3 [95% CI]", "s/crop (GPU, batch)",
        "coverage"]
    rows, ruled = [], False
    for r in b.itertuples():
        if not bool(r.full) and not ruled and len(rows):
            rows.append(["", "*partial coverage, not ranked against the rows above*"] + [""] * (len(head) - 2))
            ruled = True
        speed = "—" if f3(r.s_per_crop) == "—" else f"{float(r.s_per_crop):.3g} ({r.gpu}, {int(r.batch_size)})"
        d = "—" if r.leg == REFERENCE else f"{f3(r.diff_vs_round3)}{interval(r.diff_lo, r.diff_hi)}"
        name = f"**{r.display}**" if r.leg == REFERENCE else r.display
        rows.append(["" if f3(r.rank) == "—" else int(float(r.rank)), name, r.model_class, f"{f3(r.macro_auc)}{interval(r.macro_auc_lo, r.macro_auc_hi)}"]
                    + [f3(getattr(r, f"auc_{t}")) for t in LABEL_TYPES]
                    + [f3(r.auc_test), f3(r.auc_heldout), f3(r.within_city_auc), f3(r.balanced_accuracy),
                       f3(r.r_incorrect_p95), d, speed, f"{_int(r.n)} / {_int(r.n_total)}"])
    blocks["board"] = _table(head, rows) if rows else "(no legs staged)"

    rows = []
    disp = dict(zip(b.leg, b.display)) if len(b) else {}
    for r in pt.itertuples():
        rows.append([disp.get(r.leg, r.leg), r.label_type, _int(r.n), _int(r.n_incorrect),
                     f"{f3(r.auc)}{interval(r.auc_lo, r.auc_hi)}", f3(r.auc_test), f3(r.auc_test_heldout),
                     f"{f3(r.within_city_auc)} ({int(r.n_cities)})", f3(r.balanced_accuracy), f3(r.r_incorrect_p95)])
    blocks["per_type"] = _table(["model", "type", "n", "incorrect", "AUC [95% CI]", "test", "held-out",
                                 "within-city (cities)", "bal. acc.", "R@P95"], rows) if rows else "(none)"

    rows = [[disp.get(r.leg, r.leg), r.label_type, _int(r.n), f3(r.auc_leg), f3(r.auc_ref),
             f"{f3(r.diff)}{interval(r.diff_lo, r.diff_hi)}"] for r in pr.itertuples()] if len(pr) else []
    blocks["paired"] = _table(["model", "type", "shared crops", "AUC (model)", "AUC (round 3)",
                               "difference [95% CI]"], rows) if rows else "(none)"

    rows = []
    if len(cov):
        for leg, g in cov.groupby("leg", sort=False):
            cells = [f"{_int(x.n)} / {_int(x.total)}" for x in g.itertuples()]
            whys = sorted({x.why for x in g.itertuples() if isinstance(x.why, str) and x.why})
            rows.append([g.display.iloc[0]] + cells + ["; ".join(whys)])
    blocks["coverage"] = _table(["model"] + list(LABEL_TYPES) + ["why not complete"], rows) if rows else "(none)"

    rows = []
    if len(dep) and best is not None:
        keep = [REFERENCE, "production", best.leg]
        d = dep[dep.leg.isin(keep)]
        piv = d.pivot_table(index=["deployment", "held_out"], columns="leg", values="mean_within_type_auc",
                            aggfunc="first")
        ns = d.groupby(["deployment", "held_out"]).n.max()
        for (dname, ho), r in piv.sort_index(level=[1, 0]).iterrows():
            rows.append([dname, "held-out" if bool(ho) else "test", _int(ns.loc[(dname, ho)])]
                        + [f3(r.get(k)) for k in keep if k in piv.columns])
        hdr = ["deployment", "split", "n"] + [disp.get(k, k) for k in keep if k in piv.columns]
    blocks["per_deployment"] = _table(hdr, rows) if rows else "(none)"

    rows = [[r.display, r.gpu, int(r.n_gpus), int(r.batch_size), _int(r.n_crops), f"{r.median_s_per_crop:.3g}",
             f"{r.p90_s_per_crop:.3g}" if f3(r.p90_s_per_crop) != "—" else "—", f"{r.gpu_hours_per_1000:.3g}",
             f"{r.gpu_hours_per_800:.3g}", r.source] for r in tm.itertuples()] if len(tm) else []
    blocks["timing"] = _table(["model", "GPU", "GPUs", "batch", "crops timed", "median s/crop", "p90 s/crop",
                               "GPU-h / 1,000 labels", "GPU-h / 800-label night", "source"], rows) if rows else \
        "(no timing staged yet)"
    return blocks


def splice(text, blocks):
    """Replace each generated block in ``text``; everything else stays byte-identical. A missing block raises."""
    for name, body in blocks.items():
        pat = re.compile(re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: BEGIN.format(name=name) + "\n\n" + body + "\n\n" + END.format(name=name), text)
    return text


# --------------------------------------------------------------------------------------------------------------------
# Figures -- matplotlib, Agg, dpi 150; palette from the dataviz reference (as RampNet's scoreboard figures)
# --------------------------------------------------------------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MUTED_FILL, EMPTY = "#c9c8c2", "#f3f2ee"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
FOOTNOTE = ("Production may have trained on some held-out labels (optimistic). One run per VLM, thinking off, "
            "prompts not parallel across types; 4-bit Mac legs cover a sample only.")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK2, labelsize=8)


def _titles(ax, title, subtitle):
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=24)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def fig_leaderboard(b, path):
    plt = _plt()
    n = len(b)
    n_full = int(b.full.astype(bool).sum())
    gap = 0 < n_full < n
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.42 * max(n + gap, 1)), facecolor=SURFACE)
    _style(ax)
    ys = -np.array([i + (1 if gap and i >= n_full else 0) for i in range(n)], float)
    for y, r in zip(ys, b.itertuples()):
        col = BLUE if r.leg == REFERENCE else MUTED_FILL
        hatch = None if bool(r.full) else "////"
        ax.barh(y, r.macro_auc, color=col if bool(r.full) else EMPTY, edgecolor=MUTED if hatch else col,
                hatch=hatch, height=0.62, lw=0.6)
        ax.text(r.macro_auc + 0.004, y, f"{r.macro_auc:.3f}" + ("" if bool(r.full) else f"  ({r.n:,} crops)"),
                va="center", fontsize=8.5, color=INK, fontweight="bold" if r.leg == REFERENCE else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.display}  ·  {r.model_class}" for r in b.itertuples()], fontsize=8.5)
    for lab, r in zip(ax.get_yticklabels(), b.itertuples()):
        lab.set_color(INK if r.leg == REFERENCE else INK2)
        if r.leg == REFERENCE:
            lab.set_fontweight("bold")
    if gap:
        ax.axhline(-n_full + 0.45, color=INK2, lw=0.8)
        ax.text(0.505, -n_full, "Partial coverage: on fewer crops, not ranked against the rows above",
                fontsize=8, color=INK2, style="italic", va="center", ha="left")
    ax.set_xlim(0.5, 1.0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("macro-AUC over the five label types (incorrect = positive)", fontsize=9, color=INK2)
    best, _ = best_vlm(b)
    sub = ""
    ref = b[b.leg == REFERENCE]
    if best is not None and len(ref):
        sub = f"Round 3 {ref.macro_auc.iloc[0]:.3f}; best VLM {best.display} {best.macro_auc:.3f}."
    _titles(ax, "VLM scoreboard: macro-AUC on the round-3 held-out crops", sub)
    fig.text(0.01, 0.005, FOOTNOTE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_by_type(b, path):
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cols = list(LABEL_TYPES) + ["macro", "test", "held-out"]
    keys = [f"auc_{t}" for t in LABEL_TYPES] + ["macro_auc", "auc_test", "auc_heldout"]
    n = len(b)
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.42 * max(n, 1)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap, norm = ListedColormap(SEQ), BoundaryNorm(np.linspace(0.5, 1.0, len(SEQ) + 1), len(SEQ))
    xpos = [0, 1, 2, 3, 4, 5.4, 6.8, 7.8]
    for i, r in enumerate(b.itertuples()):
        for j, (x, k) in enumerate(zip(xpos, keys)):
            v = getattr(r, k)
            ok = v is not None and not (isinstance(v, float) and np.isnan(v))
            c = cmap(norm(max(min(v, 0.999), 0.5))) if ok else EMPTY
            ax.add_patch(plt.Rectangle((x - 0.46, i - 0.44), 0.92, 0.88, color=c, lw=0))
            txt = (f"{v:.3f}" if k == "macro_auc" else f"{v:.2f}") if ok else "—"
            light = ok and norm(max(min(v, 0.999), 0.5)) >= 7
            ax.text(x, i, txt, ha="center", va="center", fontsize=8, color="#ffffff" if light else INK,
                    fontweight="bold" if k == "macro_auc" else None)
    n_full = int(b.full.astype(bool).sum())
    if 0 < n_full < n:
        ax.axhline(n_full - 0.5, color=INK2, lw=0.8)
    ax.set_xlim(-0.6, 8.4)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cols, fontsize=8.5, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([("" if bool(r.full) else "(partial) ") + r.display for r in b.itertuples()], fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2)
    _titles(ax, "AUC by model and label type", "Macro = mean of the five type AUCs; test = trained deployments, "
            "held-out = unseen ones; — = not scored.")
    fig.text(0.01, 0.035, "Partial rows (below the rule) are on the crops they scored, not comparable row-for-row "
             "with the full rows.", fontsize=7, color=MUTED)
    fig.text(0.01, 0.005, FOOTNOTE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_roc(legs, b, path):
    """ROC per type: round 3 (blue) and production (ink) solid, the best VLM orange dashed, other VLM legs muted."""
    plt = _plt()
    best, _ = best_vlm(b)
    fig, axes = plt.subplots(1, len(LABEL_TYPES), figsize=(16, 4.0), sharey=True, facecolor=SURFACE)
    order = [leg for leg in b.leg if leg in legs]
    for ax, t in zip(axes, LABEL_TYPES):
        _style(ax)
        for leg in sorted(order, key=lambda k: (k == REFERENCE) * 3 + (k == "production") * 2
                          + (best is not None and k == best.leg)):
            g = legs[leg][legs[leg].label_type == t]
            if g.empty or g.y.nunique() < 2:
                continue
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(g.y, g.p_incorrect)
            a = auc(g.y, g.p_incorrect)
            if leg == REFERENCE:
                kw = dict(color=BLUE, lw=2, ls="-", label=f"round 3 ({a:.2f})")
            elif leg == "production":
                kw = dict(color=INK, lw=1.6, ls="-", label=f"production ({a:.2f})")
            elif best is not None and leg == best.leg:
                kw = dict(color=ORANGE, lw=2, ls="--", label=f"{leg_info(leg)[0]} ({a:.2f})")
            else:
                kw = dict(color=MUTED, lw=0.9, ls="--", alpha=0.7, label=None)
            ax.plot(fpr, tpr, **kw)
        ax.plot([0, 1], [0, 1], color=BASELINE, lw=0.8)
        ax.plot([], [], color=MUTED, lw=0.9, ls="--", label="other VLM legs")
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel("false positive rate", fontsize=8, color=INK2)
        ax.legend(fontsize=6.8, loc="lower right", frameon=False)
        ax.set_aspect("equal")
    axes[0].set_ylabel("true positive rate (incorrect)", fontsize=8, color=INK2)
    fig.suptitle("ROC per label type on each leg's scored crops (AUC in the legend)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.01, FOOTNOTE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_per_deployment(dep, b, path):
    plt = _plt()
    best, _ = best_vlm(b)
    keep = [k for k in (REFERENCE, best.leg if best is not None else None) if k]
    d = dep[dep.leg.isin(keep)] if len(dep) else dep
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 2), facecolor=SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, "no full-coverage legs staged", ha="center", color=INK2)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        return
    piv = d.pivot_table(index=["held_out", "deployment"], columns="leg", values="mean_within_type_auc",
                        aggfunc="first").sort_index(level=[0, 1], ascending=[False, True])
    n = len(piv)
    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.3 * n), facecolor=SURFACE)
    _style(ax)
    ys = np.arange(n)
    for leg, col, mk in zip(keep, (BLUE, ORANGE), ("o", "s")):
        if leg in piv:
            ax.scatter(piv[leg], ys, color=col, marker=mk, s=36, zorder=3, edgecolor=SURFACE, lw=1.2,
                       label=leg_info(leg)[0])
    if len(keep) == 2 and all(k in piv for k in keep):
        for y, (a, c) in enumerate(zip(piv[keep[0]], piv[keep[1]])):
            ax.plot([a, c], [y, y], color=BASELINE, lw=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([("held-out: " if ho else "") + dn for ho, dn in piv.index], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("mean within-type AUC (types with >= 10 rows and both classes)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _titles(ax, "Per deployment: round 3 vs the best VLM", "Public deployments with >= 30 crops named; private\n"
            "held-out deployments lettered, all other private deployments pooled.")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_latency(tm, path):
    plt = _plt()
    if tm.empty:
        fig, ax = plt.subplots(figsize=(8, 2), facecolor=SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, "no timing staged yet", ha="center", color=INK2)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        return
    t = tm.sort_values("median_s_per_crop").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.34 * len(t)), facecolor=SURFACE)
    _style(ax)
    ys = np.arange(len(t))
    for y, r in zip(ys, t.itertuples()):
        col = BLUE if r.model in (REFERENCE, "production") else INK2
        ax.scatter(r.median_s_per_crop, y, color=col, s=34, zorder=3, edgecolor=SURFACE, lw=1.2)
        if f3(r.p90_s_per_crop) != "—":
            ax.plot([r.median_s_per_crop, r.p90_s_per_crop], [y, y], color=BASELINE, lw=1.2, zorder=1)
        ax.text(r.median_s_per_crop * 1.12, y, f"{r.median_s_per_crop:.3g} s", fontsize=7.5, va="center", color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    lo, hi = t.median_s_per_crop.min(), t[["median_s_per_crop", "p90_s_per_crop"]].max().max()
    ax.set_xlim(10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi * 1.5)))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.display}  ({r.gpu}{'' if int(r.n_gpus) == 1 else f' x{int(r.n_gpus)}'}, batch "
                        f"{int(r.batch_size)})" for r in t.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("median seconds per crop (log scale; line to p90)", fontsize=9, color=INK2)
    _titles(ax, "Seconds per crop, GPU named", "Batch-1 and run-batch timings are separate rows; production and "
            "round 3 in blue.")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def render_figures(aggs, legs, out_dir):
    b = aggs["board"]
    fig_leaderboard(b, os.path.join(out_dir, "leaderboard.png"))
    fig_by_type(b, os.path.join(out_dir, "by_type.png"))
    fig_roc(legs, b, os.path.join(out_dir, "roc.png"))
    fig_per_deployment(aggs["per_deployment"], b, os.path.join(out_dir, "per_deployment.png"))
    fig_latency(aggs["timing"], os.path.join(out_dir, "latency.png"))


# --------------------------------------------------------------------------------------------------------------------
# Provenance and privacy
# --------------------------------------------------------------------------------------------------------------------

UID_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*:\d+\b")
DEPLOYMENT_COLUMNS = ("deployment", "city")


def privacy_problems(out_dir, doc, public):
    """Committed-artifact scan: any ``<city>:<id>`` string, and any deployment column value that is neither public
    nor a lettered / pooled private label. [] when clean."""
    problems = []
    paths = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith((".csv", ".json", ".md"))] \
        if os.path.isdir(out_dir) else []
    for p in paths + ([doc] if os.path.exists(doc) else []):
        text = open(p, encoding="utf-8").read()
        for m in UID_PATTERN.finditer(text):
            if not m.group(0).startswith(("http:", "https:")):
                problems.append(f"{os.path.basename(p)}: label key {m.group(0)!r}")
        if p.endswith(".csv") and text.strip():
            df = pd.read_csv(p)
            for col in DEPLOYMENT_COLUMNS:
                if col in df:
                    bad = sorted({str(v) for v in df[col].dropna()
                                  if v not in public and not str(v).startswith(("private city ", "other ("))})
                    problems += [f"{os.path.basename(p)}: {col} {v!r} is not public" for v in bad]
    return problems


def provenance(scores_dir, public, public_source, args):
    files = {}
    for root, _, names in os.walk(scores_dir):
        for f in sorted(names):
            if f.endswith((".csv", ".json")):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, scores_dir)
                entry = {"sha256": sha256_of(p)}
                if f.endswith(".csv"):
                    entry["rows"] = int(sum(1 for _ in open(p)) - 1)
                files[rel] = entry
    import matplotlib
    import sklearn
    return {"staged_files": dict(sorted(files.items())), "public_cities": sorted(public),
            "public_cities_source": public_source if public_source.startswith("http") else
            os.path.basename(public_source), "n_boot": args.n_boot, "seed": args.seed, "min_city_n": args.min_city_n,
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                         "matplotlib": matplotlib.__version__}}


# --------------------------------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------------------------------

def run_stage(args):
    """Fill the staging directory from the Mac: held-out splits and round-3 scores from ``--run-dir``, the 4-bit MLX
    legs and their timing from ``--mac-dir``. Refuses a staging directory inside this repo."""
    if vb.inside_this_repo(os.path.join(args.scores_dir, "x")):
        raise SystemExit("--scores-dir must be outside the repo (it holds per-label data)")
    os.makedirs(os.path.join(args.scores_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(args.scores_dir, "scores", REFERENCE), exist_ok=True)
    held = []
    for t in LABEL_TYPES:
        s = vb.load_split_rows(args.run_dir, t, SPLITS)    # status == "ok" only, as the scoring legs select
        keep = [c for c in ("label_uid", "city", "label_type", "label_class", "split", "pano_id") if c in s]
        s[keep].to_csv(os.path.join(args.scores_dir, "splits", f"{t}.csv"), index=False)
        p = pd.read_csv(os.path.join(args.run_dir, t, "predictions.csv"))
        p = p[p.label_uid.isin(set(s.label_uid))].sort_values("label_uid")
        if len(p) != len(s):
            raise SystemExit(f"{t}: predictions.csv covers {len(p)} of {len(s)} held-out rows")
        p[["label_uid", "p_incorrect"]].assign(model=REFERENCE).to_csv(
            os.path.join(args.scores_dir, "scores", REFERENCE, f"{t}.csv"), index=False)
        held.append(s[["label_uid", "label_type"]])
        print(f"{t}: {len(s)} held-out rows staged")
    held = pd.concat(held)
    if not args.mac_dir:
        return
    hw = "Apple silicon"
    prov = os.path.join(REPO_DIR, "docs", "vlm-baseline-2026-10", "provenance.json")
    if os.path.exists(prov):
        hw = json.load(open(prov)).get("hardware", hw).split(",")[0].replace("arm64 ", "")
    timing = []
    for model, leg in MAC_LEGS.items():
        src = vb.score_path(args.mac_dir, model)
        if not os.path.exists(src):
            continue
        d = pd.read_csv(src)
        d = d[d.variant == "definition"].merge(held, on="label_uid", how="inner", validate="one_to_one")
        out = os.path.join(args.scores_dir, "scores", leg)
        os.makedirs(out, exist_ok=True)
        for t, g in d.groupby("label_type"):
            g.drop(columns="label_type").assign(model=leg).to_csv(os.path.join(out, f"{t}.csv"), index=False)
        meta = {"repo": vb.MODELS[model], "quantisation": "4-bit (MLX)", "hardware": hw, "source": "issue #16"}
        for t in LABEL_TYPES:
            with open(os.path.join(out, f"{t}.meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
        tp = os.path.join(args.mac_dir, "timing", f"{model}.csv")
        if os.path.exists(tp):
            tt = pd.read_csv(tp)
            timing.append({"model": leg, "gpu": f"{hw} (MLX)", "n_gpus": 1, "batch_size": 1, "n_crops": len(tt),
                           "median_s_per_crop": tt.latency_s.median(), "p90_s_per_crop": tt.latency_s.quantile(0.9),
                           "source": "vlm_baseline timing (#16), idle Mac"})
        print(f"{leg}: {len(d)} crops staged")
    if timing:
        pd.DataFrame(timing).to_csv(os.path.join(args.scores_dir, "scores", "timing_mac.csv"), index=False,
                                    float_format=FLOAT)


def _public(args):
    if args.public_cities:
        return public_cities(args.public_cities)
    return public_cities(None)


LIVE_TIMEOUT_S = 10


def _live_public():
    """The public deployments in the live ``/v3/api/cities`` (short timeout: ``--check`` runs in CI)."""
    return public_cities(None, timeout=LIVE_TIMEOUT_S)


def _check_public(args, out_dir):
    """``(public, frozen)`` for ``--check``: ``--public-cities`` if given, else the live API, else (offline) the list
    frozen in the committed ``provenance.json``. ``frozen`` is that committed list either way."""
    pp = os.path.join(out_dir, "provenance.json")
    frozen = set(json.load(open(pp))["public_cities"]) if os.path.exists(pp) else set()
    if args.public_cities:
        return public_cities(args.public_cities)[0], frozen
    try:
        return _live_public()[0], frozen
    except (OSError, ValueError, KeyError) as e:   # URLError and socket timeouts are OSErrors
        print(f"{CITIES_API} unavailable ({type(e).__name__}): checking against the public list frozen in "
              f"provenance.json")
        return frozen, frozen


def no_longer_public(out_dir, public, frozen):
    """Deployments named in the committed ``per_deployment.csv`` as public (or public at build time, ``frozen``) that
    are not in ``public`` now. [] when none.

    >>> d = tempfile.mkdtemp()
    >>> pd.DataFrame({"deployment": ["a-wa", "b-or", "other (2 private deployments)"], "public": [True, True, False]}
    ...              ).to_csv(os.path.join(d, "per_deployment.csv"), index=False)
    >>> no_longer_public(d, {"a-wa"}, {"a-wa", "b-or"})
    ["per_deployment.csv: deployment 'b-or' is no longer public (re-run `build`)"]
    """
    p = os.path.join(out_dir, "per_deployment.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    dep = pd.read_csv(p)
    if "deployment" not in dep:
        return []
    named = dep.deployment.astype(str)
    was = dep.public.astype(str).str.lower().eq("true") if "public" in dep else named.isin(frozen)
    return [f"per_deployment.csv: deployment {c!r} is no longer public (re-run `build`)"
            for c in sorted(set(named[was | named.isin(frozen)]) - set(public))]


def run_build(args):
    out_dir, doc = args.out_dir, args.doc
    staged = os.path.isdir(os.path.join(args.scores_dir, "splits"))
    if args.check:
        problems = []
        committed = read_aggregates(out_dir)
        public, frozen = _check_public(args, out_dir)
        problems += no_longer_public(out_dir, public, frozen)
        if staged:
            aggs, *_ = build_aggregates(args.scores_dir, public, args.n_boot, args.seed, args.min_city_n)
            with tempfile.TemporaryDirectory() as tmp:
                write_aggregates(aggs, tmp)
                for name in aggs:
                    a, c = os.path.join(tmp, f"{name}.csv"), os.path.join(out_dir, f"{name}.csv")
                    if not os.path.exists(c) or open(a, "rb").read() != open(c, "rb").read():
                        problems.append(f"{name}.csv: stale (re-run `build`)")
        else:
            print(f"{args.scores_dir} not found: checking the doc against the committed aggregates only")
        if not os.path.exists(doc):
            problems.append(f"{doc}: missing")
        else:
            text = open(doc, encoding="utf-8").read()
            try:
                if splice(text, render_blocks(committed)) != text:
                    problems.append(f"{os.path.basename(doc)}: generated blocks are stale (re-run `build`)")
            except ValueError as e:
                problems.append(f"{os.path.basename(doc)}: {e}")
            problems += [f"{os.path.basename(doc)}: prose is stale -- {p}" for p in prose_problems(text, committed)]
        problems += privacy_problems(out_dir, doc, public)
        if problems:
            print("\n".join(problems))
            raise SystemExit(1)
        print("vlm-scoreboard: current")
        return
    if not staged:
        raise SystemExit(f"no staging directory at {args.scores_dir}; run `stage` or pass --scores-dir")
    public, src = _public(args)
    aggs, legs, _splits, _names = build_aggregates(args.scores_dir, public, args.n_boot, args.seed, args.min_city_n)
    write_aggregates(aggs, out_dir)
    committed = read_aggregates(out_dir)        # render from what was written, exactly as --check will
    if os.path.exists(doc):
        text = open(doc, encoding="utf-8").read()
        new = splice(text, render_blocks(committed))
        if new != text:
            with open(doc, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"updated {doc}")
        for p in prose_problems(new, committed):
            print(f"prose needs an edit: {p}")
    else:
        print(f"{doc} not found: write the prose with the generated-block markers first")
    if not args.no_figures:
        render_figures(committed, legs, out_dir)
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(provenance(args.scores_dir, public, src, args), f, indent=1, sort_keys=True)
        f.write("\n")
    problems = privacy_problems(out_dir, doc, public)
    if problems:
        print("\n".join(problems))
        raise SystemExit("privacy check failed: fix before committing")
    print(f"wrote {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stage", help="fill the staging dir from the round-3 run dir and the #16 Mac outputs")
    s.add_argument("--run-dir", required=True, help="round-3 run root with <Type>/{split,predictions}.csv")
    s.add_argument("--mac-dir", default=None, help="#16's private output dir (scores/, timing/)")
    s.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR)
    b = sub.add_parser("build", help="aggregates, doc blocks, figures and provenance (or --check)")
    b.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR, help="staging dir, outside the repo")
    b.add_argument("--out-dir", default=OUT_DIR)
    b.add_argument("--doc", default=DOC)
    b.add_argument("--public-cities", default=None,
                   help="saved /v3/api/cities response or one city_id per line (default: the live API; --check "
                        "falls back to the list in the committed provenance.json when offline)")
    b.add_argument("--n-boot", type=int, default=N_BOOT)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--min-city-n", type=int, default=30)
    b.add_argument("--no-figures", action="store_true")
    b.add_argument("--check", action="store_true", help="write nothing; exit 1 on drift or a privacy problem")
    args = ap.parse_args(argv)
    if args.command == "stage":
        run_stage(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
