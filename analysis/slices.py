"""Performance by label type x severity and label type x tag: the standing slice analysis.

Project Sidewalk's label types are coarse; the severity a labeler chose and the tags they added say far more about
what is in the crop (a CurbRamp tagged ``missing tactile warning``, a Crosswalk tagged ``paint fading``). Every
performance study reports these slices next to its baseline comparison, so that a severity or tag the model handles
badly shows up even when the type-level number looks fine, and so that thin slices point to where more training
data is needed.

A label with several tags counts once in each of its tag slices; an untagged label is in ``(none)``, and so is a
label with no severity. Rates on fewer than ``min_n`` labels are left blank (the count is still reported), and an
AUC interval is left blank when the smaller class has fewer than ``MIN_CLASS_FOR_CI`` labels.

Used by ``evaluate.py`` (``slices`` in ``metrics.json`` and ``slices.md``, on by default) and
``analysis/retrain_report.py`` (a model against the production baseline, per slice, on the same rows).

Example::

    >>> pred = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "pano_id": ["p", "p", "q", "q"],
    ...                      "true": ["correct", "incorrect", "correct", "incorrect"],
    ...                      "p_incorrect": [0.1, 0.9, 0.4, 0.6], "severity": [1, 1, 3, None],
    ...                      "tags": ['["steep"]', '["steep","narrow"]', "[]", None]})
    >>> t = slice_table(pred, "tag", min_n=1, n_boot=0)
    >>> list(zip(t.slice, t.n))
    [('(none)', 2), ('steep', 2), ('narrow', 1)]
"""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

BY = ("severity", "tag")
NONE = "(none)"
MIN_N = 30            # rates on fewer labels are left blank
MIN_CLASS_FOR_CI = 8  # same rule as retrain_report: no interval when the smaller class has fewer labels
THIN_TRAIN = 100      # a slice with fewer training rows is flagged as needing data
N_BOOT = 1000


def parse_tags(value):
    """The tag list of one label, from any of the encodings in use: a JSON array (``split.csv``, rawLabels), a
    ``;``-joined string (the database pull), a list, or missing.

    >>> parse_tags('["narrow","pole"]'), parse_tags("narrow;pole"), parse_tags("[]"), parse_tags(None)
    (['narrow', 'pole'], ['narrow', 'pole'], [], [])
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value if str(t)]
    s = str(value).strip()
    if s.startswith("["):
        return [str(t) for t in json.loads(s) if str(t)]
    return [t for t in s.split(";") if t]


def with_slice(df, by):
    """``df`` with a ``slice`` column for ``by`` (``"severity"`` or ``"tag"``); tags explode to one row per tag."""
    if by not in BY:
        raise ValueError(f"by must be one of {BY}")
    d = df.copy()
    if by == "severity":
        sev = pd.to_numeric(d["severity"], errors="coerce")
        d["slice"] = [NONE if pd.isna(v) else str(int(v)) for v in sev]
        return d
    d["slice"] = [parse_tags(v) or [NONE] for v in d["tags"]]
    return d.explode("slice").reset_index(drop=True)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def slice_seed(seed, name):
    """A per-slice seed, so a slice's interval does not change when another slice is added or removed."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16)


def boot_indices(y, groups, n_boot, rng, max_tries=20):
    """``n_boot`` row-index arrays, each a resample of whole groups (panoramas) with replacement. A resample with only
    one class is redrawn, up to ``max_tries`` times, as ``retrain_report.cluster_boot`` does."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        for _ in range(max_tries):
            idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
            if 0 < y[idx].sum() < len(idx):
                out.append(idx)
                break
    if len(out) < n_boot:
        warnings.warn(f"{n_boot - len(out)} of {n_boot} resamples had one class after {max_tries} tries; the "
                      "interval rests on the rest")
    return out


def bootstrap_auc(y, score, groups, n_boot=N_BOOT, seed=0):
    """95% percentile interval of the AUC from ``n_boot`` resamples of whole groups (panoramas), or ``(nan, nan)``
    when the smaller class is under ``MIN_CLASS_FOR_CI`` or ``n_boot`` is 0."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if n_boot == 0 or min(y.sum(), (~y).sum()) < MIN_CLASS_FOR_CI:
        return np.nan, np.nan
    vals = [_auc(y[i], score[i]) for i in boot_indices(y, groups, n_boot, np.random.default_rng(seed))]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def _order(out, by):
    """Severity in its own order (1, 2, 3, ..., then ``(none)``); tags by descending n, then name."""
    if by == "severity":
        key = out.slice.map(lambda v: (1, 0) if v == NONE else (0, int(v)))
        return out.assign(_k=key).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return out.sort_values(["n", "slice"], ascending=[False, True]).reset_index(drop=True)


def slice_table(pred, by, train=None, min_n=MIN_N, n_boot=N_BOOT, seed=0):
    """One row per slice of ``pred`` (columns ``label_uid, true, p_incorrect, severity, tags``, ``pano_id`` for the
    panorama bootstrap and, if present, ``city``): ``n``, ``n_correct``, ``n_incorrect``, ``auc`` [``auc_lo``,
    ``auc_hi``] of the incorrect score, ``accuracy`` at 0.5, ``max_city_share`` (the largest share of the slice from
    one deployment: near 1 means the slice is one city's labels) and, with ``train`` (the training rows, same
    columns), ``train_n`` and ``needs_data`` (``train_n < THIN_TRAIN``). Accuracy is NaN under ``min_n``; the AUC is
    NaN under ``min_n`` or when the smaller class has fewer than ``MIN_CLASS_FOR_CI`` labels."""
    d = with_slice(pred, by)
    tr = with_slice(train, by).groupby("slice").size() if train is not None else None
    groups_col = "pano_id" if "pano_id" in d.columns else "label_uid"
    rows = []
    for name, g in d.groupby("slice"):
        y = (g["true"] == "incorrect").to_numpy()
        row = {"slice": name, "n": int(len(g)), "n_correct": int((~y).sum()), "n_incorrect": int(y.sum()),
               "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan, "accuracy": np.nan}
        if "city" in g.columns:
            row["max_city_share"] = float(g.city.value_counts(normalize=True).iloc[0])
        if len(g) >= min_n:
            row["accuracy"] = float(((g.p_incorrect.to_numpy() >= 0.5) == y).mean())
            if min(y.sum(), (~y).sum()) >= MIN_CLASS_FOR_CI:
                row["auc"] = _auc(y, g.p_incorrect.to_numpy())
                row["auc_lo"], row["auc_hi"] = bootstrap_auc(y, g.p_incorrect, g[groups_col], n_boot,
                                                             slice_seed(seed, name))
        if tr is not None:
            row["train_n"] = int(tr.get(name, 0))
            row["needs_data"] = row["train_n"] < THIN_TRAIN
        rows.append(row)
    return _order(pd.DataFrame(rows), by)


def paired_delta_table(pred_a, pred_b, by, min_n=MIN_N, n_boot=N_BOOT, seed=0):
    """Model A against model B (e.g. a retrain against production) on the same rows, per slice: ``auc_a``,
    ``auc_b`` and ``delta`` (A - B) with a 95% interval from resampling whole panoramas, the same resample applied
    to both models. ``pred_a`` carries the slice columns and decides the rows; ``pred_b`` needs only ``label_uid,
    p_incorrect`` and must score every one of ``pred_a``'s labels (raises otherwise, so no slice silently loses
    rows); its other rows, e.g. another split, are ignored. Blank under ``min_n`` or when
    the smaller class has fewer than ``MIN_CLASS_FOR_CI`` labels. The intervals are not corrected for the number of
    slices: at 95%, about one slice in twenty excludes zero by chance."""
    missing = set(pred_a.label_uid) - set(pred_b.label_uid)
    if missing:
        raise ValueError(f"the baseline has no score for {len(missing)} of the model's labels (e.g. "
                         f"{sorted(missing)[:3]}); score both on the same split")
    cols = ["label_uid", "p_incorrect"] + (["true"] if "true" in pred_b.columns else [])
    m = pred_a.merge(pred_b[cols], on="label_uid", how="left", suffixes=("", "_b"), validate="one_to_one")
    if "true_b" in m.columns and (m["true"] != m["true_b"]).any():
        raise ValueError("the two prediction files disagree on the true class of some labels")
    d = with_slice(m, by)
    groups_col = "pano_id" if "pano_id" in d.columns else "label_uid"
    rows = []
    for name, g in d.groupby("slice"):
        y = (g["true"] == "incorrect").to_numpy()
        row = {"slice": name, "n": int(len(g)), "n_incorrect": int(y.sum()), "auc_a": np.nan, "auc_b": np.nan,
               "delta": np.nan, "delta_lo": np.nan, "delta_hi": np.nan}
        if "city" in g.columns:
            row["max_city_share"] = float(g.city.value_counts(normalize=True).iloc[0])
        if len(g) >= min_n and min(y.sum(), (~y).sum()) >= MIN_CLASS_FOR_CI:
            a, b = g.p_incorrect.to_numpy(), g.p_incorrect_b.to_numpy()
            row.update(auc_a=_auc(y, a), auc_b=_auc(y, b))
            row["delta"] = row["auc_a"] - row["auc_b"]
            if n_boot:
                idxs = boot_indices(y, g[groups_col], n_boot, np.random.default_rng(slice_seed(seed, name)))
                vals = [_auc(y[i], a[i]) - _auc(y[i], b[i]) for i in idxs]
                if vals:
                    row["delta_lo"], row["delta_hi"] = np.percentile(vals, [2.5, 97.5]).tolist()
        rows.append(row)
    return _order(pd.DataFrame(rows), by)


def slice_report(pred, train=None, splits=("test", "test_heldout"), min_n=MIN_N, n_boot=N_BOOT, seed=0):
    """``{split: {by: [row, ...]}}`` for ``evaluate.py``'s ``metrics.json``: :func:`slice_table` per split and per
    ``by``, NaN written as None."""
    out = {}
    for split in splits:
        p = pred[pred.split == split]
        if len(p):
            out[split] = {by: slice_table(p, by, train, min_n, n_boot, seed).astype(object).where(lambda x: x.notna(), None)
                          .to_dict("records") for by in BY}
    return out


def report_markdown(report, label_type, min_n=MIN_N):
    """:func:`slice_report` output as Markdown, one table per split and ``by``."""
    parts = [f"# {label_type}: performance by severity and tag", "",
             f"AUC of the incorrect score with 95% panorama-bootstrap intervals; accuracy blank under {min_n} "
             f"labels, AUC also blank when the smaller class has fewer than {MIN_CLASS_FOR_CI}; ⚠ thin = fewer than "
             f"{THIN_TRAIN} training rows in the slice; intervals are not corrected for the number of slices.", ""]
    for split, tables in report.items():
        for by, rows in tables.items():
            parts.append(slice_markdown(pd.DataFrame(rows).astype({"auc": float, "auc_lo": float, "auc_hi": float,
                                                                   "accuracy": float}),
                                        by, f"{split}, by {by}"))
    return "\n".join(parts)


def _f(v, fmt="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _ci(lo, hi):
    return "" if _f(lo) == "" else f" [{lo:.3f}, {hi:.3f}]"


def slice_markdown(table, by, title):
    """``slice_table`` output as a Markdown table under a ``###`` heading."""
    has_train = "train_n" in table.columns
    has_city = "max_city_share" in table.columns
    head = f"| {by.capitalize()} | n | correct / incorrect | AUC [95% CI] | accuracy at 0.5 |"
    sep = "|---|---:|---:|---:|---:|"
    if has_city:
        head += " largest one-deployment share |"
        sep += "---:|"
    if has_train:
        head += " training rows |"
        sep += "---:|"
    lines = [f"### {title}", "", head, sep]
    for r in table.itertuples():
        line = (f"| {r.slice} | {r.n:,} | {r.n_correct:,} / {r.n_incorrect:,} | {_f(r.auc)}{_ci(r.auc_lo, r.auc_hi)} | "
                f"{_f(r.accuracy)} |")
        if has_city:
            line += f" {_f(r.max_city_share, '{:.0%}')} |"
        if has_train:
            line += f" {r.train_n:,}{' ⚠ thin' if r.needs_data else ''} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def delta_markdown(table, by, title, a="model", b="baseline"):
    """``paired_delta_table`` output as a Markdown table under a ``###`` heading."""
    has_city = "max_city_share" in table.columns
    lines = [f"### {title}", "", f"| {by.capitalize()} | n | incorrect | AUC, {a} | AUC, {b} | Δ [95% CI] |"
             + (" largest one-deployment share |" if has_city else ""),
             "|---|---:|---:|---:|---:|---:|" + ("---:|" if has_city else "")]
    for r in table.itertuples():
        d = "" if np.isnan(r.delta) else f"{r.delta:+.3f}" + (
            "" if np.isnan(r.delta_lo) else f" [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]")
        line = f"| {r.slice} | {r.n:,} | {r.n_incorrect:,} | {_f(r.auc_a)} | {_f(r.auc_b)} | {d} |"
        lines.append(line + (f" {_f(r.max_city_share, '{:.0%}')} |" if has_city else ""))
    return "\n".join(lines) + "\n"
