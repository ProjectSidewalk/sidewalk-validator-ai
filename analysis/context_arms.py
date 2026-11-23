"""Issue #14 report: the three "more context" arms (``context_model.py``) against the round-3 model, on the same rows.

For each label type, reads ``evaluate.py``'s outputs in ``<data-dir>/<label_type>/``: the round-3 baseline
(``metrics.json``/``predictions.csv``) and each arm's tagged run (``metrics_<arm>.json``/``predictions_<arm>.csv``,
written by ``context_model.py predict`` + ``evaluate.py --skip-inference --tag <arm> --per-city``), plus seed
replicates named ``<run>-s<seed>`` (the baseline's ``baseline-s43``/``-s44`` from the confident-learning batch). Writes
to ``--out``:

* ``arms.md`` / ``arms.json``: per type x arm, test and held-out AUC with the paired panorama-bootstrap difference
  against the baseline on the same rows, the within-city AUC (round-3 definition), macro F1 at the model's verdict,
  and incorrect/correct recall at precision 0.92 and 0.95; plus the held-out deployments one by one (private ones
  lettered, never named).
* ``seeds.md``: every seed replicate against the baseline of the *same* seed (``confident_learning.py``'s layout).
* ``slices.md``: type x severity and type x tag, each arm against the baseline on the same rows
  (``analysis/slices.paired_delta_table``), with the slice's training rows and the thin flag.
* ``clipping.md``: how often the wide crop is the whole view (``wide_clipped``, ``wide_half_px`` quantiles), per type,
  from ``<label_type>/context/index.csv``.
* ``provenance.json``: sha256 of every file read.

The table helpers are ``analysis/confident_learning.py``'s (same within-city definition, same bootstrap), imported,
not copied.

Usage::

    python analysis/context_arms.py report --data-dir $DATA --label-types NoCurbRamp,Obstacle,SurfaceProblem \\
        --public-cities cities.json --out docs/context-arms-2026-11
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import confident_learning as cl  # noqa: E402
from analysis import slices as sl  # noqa: E402

ARMS = ("two_scale", "view_heatmap", "view_heatmap-r448", "side_info", "side_info_nohq")  # report --arms default
TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
LEVELS = ("0.92", "0.95")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(m, split):
    """Mean of the two classes' F1 in ``evaluate.py``'s ``metrics.json`` for ``split`` (at the model's verdict), or
    None.

    >>> macro_f1({"test": {"per_class": {"correct": {"f1-score": 0.8}, "incorrect": {"f1-score": 0.6}}}}, "test")
    0.7
    """
    pc = (m.get(split) or {}).get("per_class") or {}
    f = [pc[c]["f1-score"] for c in ("correct", "incorrect") if c in pc]
    return round(float(np.mean(f)), 10) if len(f) == 2 else None


def arm_row(data_dir, label_type, name, preds, within_min_n=10):
    """``confident_learning.metric_row`` plus macro F1 and recall at precision 0.92 and 0.95 for both classes."""
    r = cl.metric_row(data_dir, label_type, name, preds, within_min_n)
    if "missing" in r:
        return r
    with open(cl.metrics_path(data_dir, label_type, name)) as f:
        m = json.load(f)
    for split, key in (("test", "test"), ("test_heldout", "heldout")):
        r[f"macro_f1_{key}"] = macro_f1(m, split)
        for cls in ("incorrect", "correct"):
            for lvl in LEVELS:
                r[f"{cls}_recall_p{lvl[2:]}_{key}"] = cl._metric(m, split, f"recall_at_precision:{cls}:{lvl}")
    return r


def arms_markdown(label_type, rows, heldout=None, within_min_n=10):
    """One table per type: the baseline and each arm, Δ against the baseline on the same rows."""
    lines = [f"### {label_type}", "",
             "| run | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | "
             "within-city held-out | macro F1 test / held-out | incorrect R@P0.92 / 0.95 held-out | "
             "correct R@P0.92 / 0.95 held-out |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = cl._fmt
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['variant']} | not yet run | | | | | | | | |")
            continue
        d_test, d_held = (cl._delta_cell({"delta": r[f"{k}_delta"], "delta_lo": r.get(f"{k}_delta_lo"),
                                          "delta_hi": r.get(f"{k}_delta_hi")} if f"{k}_delta" in r else None)
                          for k in ("test", "heldout"))
        lines.append("| " + " | ".join([
            r["variant"], f(r["test_auc"]), d_test, f"{f(r['within_city_test'])} ({r['cities_test']})",
            f(r["heldout_auc"]), d_held, f"{f(r['within_city_heldout'])} ({r['cities_heldout']})",
            f"{f(r['macro_f1_test'])} / {f(r['macro_f1_heldout'])}",
            f"{f(r['incorrect_recall_p92_heldout'])} / {f(r['incorrect_recall_p95_heldout'])}",
            f"{f(r['correct_recall_p92_heldout'])} / {f(r['correct_recall_p95_heldout'])}"]) + " |")
    if heldout:
        models = [k for k in heldout[0] if k not in ("deployment", "n", "n_incorrect")]
        lines += ["", f"Held-out deployments, AUC per run (blank under {within_min_n} labels or with one class):", "",
                  "| deployment | n | incorrect | " + " | ".join(models) + " |", "|---|---:|---:|" + "---:|" * len(models)]
        for h in heldout:
            lines.append(f"| {h['deployment']} | {h['n']:,} | {h['n_incorrect']:,} | "
                         + " | ".join(f(h[k]) for k in models) + " |")
    return "\n".join(lines) + "\n"


def slice_deltas(preds, arms, train, min_n=None, n_boot=None, seed=0):
    """Severity and tag slices of each arm against the baseline on the same rows, test and held-out
    (``slices.paired_delta_table``), each slice with its training rows (``train``: split.csv's train rows) and
    ``needs_data`` (fewer than ``slices.THIN_TRAIN``). Records with ``split``, ``by``, ``variant`` and the table's
    columns; empty without baseline predictions."""
    base = preds.get("baseline")
    if base is None:
        return []
    min_n = sl.MIN_N if min_n is None else min_n
    n_boot = sl.N_BOOT if n_boot is None else n_boot
    counts = {by: sl.with_slice(train, by).groupby("slice").size().to_dict() for by in sl.BY}
    out = []
    for name in arms:
        p = preds.get(name)
        if p is None:
            continue
        for split in ("test", "test_heldout"):
            part = p[p["split"] == split]
            if part.empty:
                continue
            for by in sl.BY:
                t = sl.paired_delta_table(part, base, by, min_n, n_boot, seed)
                for r in t.to_dict("records"):
                    n_train = int(counts[by].get(r["slice"], 0))
                    out.append({"split": split, "by": by, "variant": name, **r, "train_n": n_train,
                                "needs_data": n_train < sl.THIN_TRAIN})
    return out


def slices_markdown(label_type, records, arms):
    """``slice_deltas`` records as one table per (split, by): slice, rows, training rows (thin flagged), baseline AUC,
    and each arm's Δ with its interval."""
    lines = []
    for split in ("test", "test_heldout"):
        for by in sl.BY:
            rs = [r for r in records if r["split"] == split and r["by"] == by]
            if not rs:
                continue
            lines += [f"### {label_type} / {split} / by {by}", "",
                      f"| {by.capitalize()} | n | incorrect | training rows | baseline AUC | "
                      + " | ".join(f"{a} Δ [95% CI]" for a in arms) + " |",
                      "|---|---:|---:|---:|---:|" + "---:|" * len(arms)]
            for name in dict.fromkeys(r["slice"] for r in rs):
                by_arm = {r["variant"]: r for r in rs if r["slice"] == name}
                first = next(iter(by_arm.values()))
                train = f"{first['train_n']:,}" + (" ⚠ thin" if first["needs_data"] else "")
                cells = [str(name), f"{first['n']:,}", f"{first['n_incorrect']:,}", train, cl._fmt(first.get("auc_b"))]
                cells += [cl._delta_cell(by_arm.get(a)) for a in arms]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def clipping_row(index):
    """How the wide crops of one type came out: ``n``, the share whose requested context did not fit
    (``wide_clipped``), the share that is the whole view, and ``wide_half_px`` quantiles.

    >>> r = clipping_row(pd.DataFrame({"wide_half_px": [400, 1024, 1024, 800], "wide_clipped": [0, 1, 0, 0]}))
    >>> r["n"], r["clipped_share"], r["whole_view_share"], r["wide_half_px_q50"]
    (4, 0.25, 0.5, 912.0)
    """
    half = index["wide_half_px"].astype(float)
    row = {"n": int(len(index)), "clipped_share": float(index["wide_clipped"].astype(int).mean()),
           "whole_view_share": float((half >= 1024).mean())}
    row.update({f"wide_half_px_q{int(q * 100)}": float(half.quantile(q)) for q in QUANTILES})
    return row


def clipping_markdown(rows):
    lines = ["# How much context the wide crop adds", "",
             "The wide crop asks for 4x the depth-sized half-width, capped at the whole 2048 px view (half-width "
             "1024). `clipped` = the requested context did not fit; `whole view` = the wide crop is the view "
             "itself. Quantiles of the wide half-width in view px.", "",
             "| type | labels | clipped | whole view | " + " | ".join(f"q{int(q * 100)}" for q in QUANTILES) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(QUANTILES)]
    for lt, r in rows.items():
        if r is None:
            lines.append(f"| {lt} | no context index yet | | |" + " |" * len(QUANTILES))
            continue
        lines.append(f"| {lt} | {r['n']:,} | {r['clipped_share']:.0%} | {r['whole_view_share']:.0%} | "
                     + " | ".join(f"{r[f'wide_half_px_q{int(q * 100)}']:.0f}" for q in QUANTILES) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# The pre-registered decision (docs/context-arms-2026-11.md, "Pre-registered design and decision rule")
# ---------------------------------------------------------------------------------------------------------------------

#: The rule's constants, written verbatim into decision.json; the doc's fenced json block must equal it (tested).
DECISION_RULE = {
    "min_delta": 0.02, "level": 0.95, "min_positive_deployments": 2, "all_seeds_above_baseline_mean": True,
    "test_no_harm_lower": -0.02, "regression_test_upper": -0.01, "prior_rule": "majority2", "gate": "report_only",
    "baseline_variant": "baseline-rerun",
    # five confirmatory seeds: Jon's decision J5 (2026-11-20), taken from the power table on October's trio after the
    # pilot's trainings had finished and before stage A was submitted (the doc's Deviations entry)
    "seeds_confirmatory": [42, 43, 44, 45, 46], "seeds_baseline": [42, 43, 44, 45, 46],
}
CONFIRMATORY = ("two_scale", "view_heatmap")
EXPLORATORY = ("view_heatmap-r448", "side_info", "side_info_nohq")
OCTOBER = "baseline"                 # October's round-3 model and its seed-43/44 replicates (the drift row)
TARGET_TYPES = ("NoCurbRamp", "Obstacle", "SurfaceProblem")
#: October's within-city held-out AUC sample sd over seeds 42/43/44 (docs/confident-learning-2026-10/seeds.md).
OCTOBER_SEED_SD = {"NoCurbRamp": 0.024, "Obstacle": 0.009, "SurfaceProblem": 0.021}
SHARES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "deployment-prior-2026-11",
                      "shares.json")
MAX_REDRAWS = 20
#: The fewest kept joint resamples (each of test and held-out) a verdict may rest on: below it the bootstrap term of
#: se is missing or noise, and leaving it out narrows the interval toward a false WIN. ``report`` refuses a
#: ``--decision-boot`` below it.
MIN_DECISION_BOOT = 200
#: The round-3 recipe's epochs; a run record with another value (a smoke run, a hand-set EPOCHS) never counts as a seed.
RECIPE_EPOCHS = 25
#: The training knobs ``analysis/context_arms.slurm`` pins for every run (``--best-metric macro_f1 --target hard
#: --batch-size 8``), as ``run_key.json`` records them. ``fine_tune_classifier`` writes no ``target`` for the hard
#: target and no ``batch_size`` for its default batch (``DEFAULT_BATCH_SIZE``), so an absent key reads as the value in
#: ``RECIPE_ABSENT``. A hand submission with another value (``BATCH=4``) never counts as a seed.
RECIPE_SETTINGS = {"best_metric": "macro_f1", "target": "hard", "batch_size": 8}
RECIPE_ABSENT = {"target": "hard", "batch_size": 8}
VERDICTS = ("WIN", "NO GAIN", "REGRESSION", "INCOMPLETE")
NO_GAIN_WORDING = "no gain detected at this power"


def _rank_auc(y, scores):
    """AUC of every column of ``scores`` (``n x k``) against the 0/1 vector ``y`` (ties get half credit, as
    ``sklearn.metrics.roc_auc_score``); NaN when ``y`` has one class.

    >>> _rank_auc(np.array([0, 0, 1, 1]), np.array([[0.1, 0.4], [0.2, 0.3], [0.8, 0.2], [0.9, 0.1]])).round(3).tolist()
    [1.0, 0.0]
    """
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, float).reshape(len(y), -1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    r = rankdata(scores, axis=0)
    return (r[y].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def aligned_scores(preds, split, names):
    """One frame of the ``split`` rows shared by every run in ``names`` (``preds``: run -> joined predictions with
    ``label_uid``, ``city``, ``pano_id``, ``true``, ``p_incorrect``): ``label_uid``, ``city``, ``pano_id``, ``y``
    and one score column per run, merged on ``label_uid`` one to one. Raises when the runs do not score the same rows
    (a paired comparison on different rows is not paired)."""
    base = None
    for n in names:
        p = preds[n]
        p = p[p["split"] == split][["label_uid", "city", "pano_id", "true", "p_incorrect"]]
        if base is None:
            base = p.rename(columns={"p_incorrect": n})
            continue
        m = base.merge(p[["label_uid", "p_incorrect"]].rename(columns={"p_incorrect": n}), on="label_uid",
                       how="outer", validate="one_to_one", indicator=True)
        if (m["_merge"] != "both").any():
            raise ValueError(f"run {n} scores {int((m['_merge'] != 'both').sum())} rows of {split} the others do "
                             "not (or the reverse); the comparison needs the same rows")
        base = m.drop(columns="_merge")
    if base is None:
        return None
    base["y"] = (base["true"] == "incorrect").astype(int)
    return base.reset_index(drop=True)


def within_city_boot(preds, split, names, min_n=10, n_boot=1000, seed=0, max_redraws=MAX_REDRAWS):
    """Within-city AUC of every run in ``names`` on ``split``, its joint panorama bootstrap, and the per-deployment
    AUCs: one loop over resample ``b``, in which every deployment's panoramas are resampled with replacement
    (stratified by deployment). A deployment whose resample lost a class is redrawn alone, up to ``max_redraws``
    times; if it still has one class, resample ``b`` is dropped for everyone. The same resample ``b`` is applied to
    every run, so differences between runs are paired.

    Deployments: those with at least ``min_n`` labels and both classes in the full split (``retrain_report.within_city_auc``'s
    rule); weights n_incorrect x n_correct of each resample. Returns ``{"names", "deployments", "point" (per run),
    "per_deployment" ({deployment: {run: auc}}), "boot" (n_kept x n_runs), "redraws" ({deployment: n}),
    "dropped"}``; ``boot`` is empty when ``n_boot`` is 0.
    """
    df = aligned_scores(preds, split, names)
    out = {"names": list(names), "deployments": [], "point": {n: None for n in names}, "per_deployment": {},
           "boot": np.empty((0, len(names))), "redraws": {}, "dropped": 0, "n_boot": int(n_boot)}
    if df is None or df.empty:
        return out
    groups = []
    for city, g in df.groupby("city", sort=True):
        if len(g) >= min_n and 0 < g["y"].sum() < len(g):
            panos = g["pano_id"].astype(str).to_numpy()
            uniq, inv = np.unique(panos, return_inverse=True)
            groups.append((city, g.index.to_numpy(), [np.flatnonzero(inv == k) for k in range(len(uniq))]))
    if not groups:
        return out
    y = df["y"].to_numpy()
    s = df[list(names)].to_numpy(float)

    def wc(rows_by_city):
        aucs, ws = [], []
        for rows in rows_by_city:
            yy = y[rows]
            aucs.append(_rank_auc(yy, s[rows]))
            ws.append(float(yy.sum()) * float(len(yy) - yy.sum()))
        return np.average(np.vstack(aucs), axis=0, weights=ws), aucs

    point, per = wc([rows for _, rows, _ in groups])
    out["deployments"] = [c for c, _, _ in groups]
    out["point"] = {n: float(v) for n, v in zip(names, point)}
    out["per_deployment"] = {c: {n: float(v) for n, v in zip(names, a)} for (c, _, _), a in zip(groups, per)}
    out["redraws"] = {c: 0 for c, _, _ in groups}
    rng = np.random.default_rng(seed)
    rows_b = []
    for _ in range(int(n_boot)):
        picked, ok = [], True
        for city, rows, members in groups:
            for attempt in range(max_redraws + 1):
                idx = rows[np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])]
                if 0 < y[idx].sum() < len(idx):
                    break
                out["redraws"][city] += 1
            else:
                ok = False
            picked.append(idx)
        if not ok:
            out["dropped"] += 1
            continue
        rows_b.append(wc(picked)[0])
    if rows_b:
        out["boot"] = np.vstack(rows_b)
    return out


def welch_df(v_a, n_a, v_b, n_b):
    """Welch-Satterthwaite degrees of freedom of ``v_a / n_a + v_b / n_b`` (a group of one seed contributes no
    variance and no degrees of freedom); None when neither group has two seeds.

    >>> round(welch_df(0.024 ** 2, 3, 0.024 ** 2, 5), 2)
    4.34
    """
    terms = [(v / n, n - 1) for v, n in ((v_a, n_a), (v_b, n_b)) if n > 1 and v is not None]
    if not terms:
        return None
    num = sum(t for t, _ in terms) ** 2
    den = sum(t ** 2 / d for t, d in terms)
    return float(num / den) if den > 0 else float(sum(d for _, d in terms))


def seed_mean_delta(arm_vals, base_vals, var_boot=0.0, level=0.95):
    """``D`` = mean(arm seeds) - mean(baseline seeds) with the interval ``D +/- t(level, nu) * se``,
    ``se^2 = s_arm^2 / n_arm + s_base^2 / n_base + var_boot`` (sample variances across seeds; ``var_boot`` the
    bootstrap variance of ``D`` on the evaluation rows), ``nu`` the Welch-Satterthwaite degrees of freedom of the two
    seed terms. A single arm seed has no seed variance to add (``single_arm_seed``: the interval then understates the
    spread; exploratory rows only). ``n_above``: arm seeds above the baseline's seed mean.

    >>> r = seed_mean_delta([0.92, 0.93, 0.94], [0.88, 0.89, 0.90, 0.89, 0.89], var_boot=0.00005)
    >>> round(r["delta"], 3), r["n_above"], r["lo"] > 0, round(r["nu"], 1)
    (0.04, 3, True, 3.2)
    """
    from scipy.stats import t as student_t
    a = [float(v) for v in arm_vals if v is not None and np.isfinite(v)]
    b = [float(v) for v in base_vals if v is not None and np.isfinite(v)]
    out = {"n_arm": len(a), "n_base": len(b), "arm_values": a, "base_values": b, "delta": None, "se": None,
           "nu": None, "t": None, "lo": None, "hi": None, "n_above": None, "var_boot": var_boot,
           "single_arm_seed": len(a) == 1}
    if not a or not b:
        return out
    va = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    base_mean = float(np.mean(b))
    out.update(delta=float(np.mean(a) - base_mean), s_arm=va ** 0.5, s_base=vb ** 0.5, base_mean=base_mean,
               n_above=int(sum(v > base_mean for v in a)))
    nu = welch_df(va, len(a), vb, len(b))
    if nu is None:
        return out
    se = (va / len(a) + vb / len(b) + (var_boot or 0.0)) ** 0.5
    t = float(student_t.ppf(0.5 + level / 2, nu))
    out.update(se=se, nu=nu, t=t, lo=out["delta"] - t * se, hi=out["delta"] + t * se)
    return out


def boot_var_of_delta(boot, names, arm_runs, base_runs):
    """Variance over the bootstrap resamples of mean(arm runs) - mean(baseline runs) (columns of ``boot``)."""
    if boot is None or len(boot) < 2:
        return None
    ia = [names.index(n) for n in arm_runs]
    ib = [names.index(n) for n in base_runs]
    d = boot[:, ia].mean(axis=1) - boot[:, ib].mean(axis=1)
    return float(np.var(d, ddof=1))


def apply_rule(primary, test, deployment_diffs, rule=DECISION_RULE):
    """The verdict of one type x confirmatory arm from its two intervals (``seed_mean_delta`` records for the held-out
    and the test within-city AUC) and its per-deployment seed-mean differences. Returns ``(verdict, criteria)``;
    ``criteria`` lists every criterion with its value and pass/fail. Missing inputs are the caller's INCOMPLETE.

    >>> win = {"delta": 0.04, "lo": 0.01, "hi": 0.07, "n_arm": 3, "n_above": 3}
    >>> apply_rule(win, {"delta": 0.0, "lo": -0.01, "hi": 0.01}, [0.03, 0.05, -0.01])[0]
    'WIN'
    >>> apply_rule({**win, "lo": -0.01}, {"delta": 0.0, "lo": -0.01, "hi": 0.01}, [0.03, 0.05, -0.01])[0]
    'NO GAIN'
    >>> apply_rule({**win, "delta": -0.05, "lo": -0.08, "hi": -0.02}, {"delta": 0, "lo": -0.01, "hi": 0.01}, [])[0]
    'REGRESSION'
    """
    q = len(deployment_diffs)
    need = rule["min_positive_deployments"] if q >= 3 else q
    positive = sum(d > 0 for d in deployment_diffs)
    crit = [
        {"criterion": "1. D >= min_delta", "value": primary["delta"], "pass": primary["delta"] >= rule["min_delta"]},
        {"criterion": "2. interval lower bound > 0", "value": primary["lo"], "pass": primary["lo"] > 0},
        {"criterion": "3. every arm seed above the baseline seed mean", "value": f"{primary['n_above']} of "
         f"{primary['n_arm']}", "pass": (not rule["all_seeds_above_baseline_mean"])
         or primary["n_above"] == primary["n_arm"]},
        {"criterion": f"4. positive in >= {need} of {q} held-out deployments", "value": positive,
         "pass": q > 0 and positive >= need},
        {"criterion": f"5. test lower bound > {rule['test_no_harm_lower']}", "value": test["lo"],
         "pass": test["lo"] > rule["test_no_harm_lower"]},
    ]
    regression = [{"criterion": "REGRESSION: held-out upper bound < 0", "value": primary["hi"],
                   "pass": primary["hi"] < 0},
                  {"criterion": f"REGRESSION: test upper bound < {rule['regression_test_upper']}", "value": test["hi"],
                   "pass": test["hi"] < rule["regression_test_upper"]}]
    if any(c["pass"] for c in regression):
        return "REGRESSION", crit + regression
    if all(c["pass"] for c in crit):
        return "WIN", crit + regression
    return "NO GAIN", crit + regression


def simulate_rule(effect, sd_seed, sd_boot, n_arm, n_base, rule=DECISION_RULE, n_sim=4000, seed=0):
    """Share of simulated experiments the rule calls ``WIN`` and ``REGRESSION`` at a true held-out effect ``effect``.

    Each draw: baseline seed values ``N(0, sd_seed)``, arm seed values ``N(effect, sd_seed)`` shifted together by one
    draw of the evaluation-sample noise of the difference ``N(0, sd_boot)``; then criteria 1 to 3 and the held-out
    REGRESSION test exactly as ``apply_rule`` (``seed_mean_delta``'s interval with ``var_boot = sd_boot^2`` and its
    Welch t quantile). Criteria 4 (deployments) and 5 (test no-harm) are not simulated, so the WIN rate is an upper
    bound on the full rule's.
    """
    from scipy.stats import t as student_t
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, sd_seed, (n_sim, n_base))
    a = rng.normal(effect, sd_seed, (n_sim, n_arm)) + rng.normal(0.0, sd_boot, (n_sim, 1))
    base_mean = b.mean(axis=1)
    d = a.mean(axis=1) - base_mean
    va = a.var(axis=1, ddof=1) if n_arm > 1 else np.zeros(n_sim)
    vb = b.var(axis=1, ddof=1)
    ta, tb = va / n_arm, vb / n_base
    if n_arm > 1:
        nu = (ta + tb) ** 2 / (ta ** 2 / (n_arm - 1) + tb ** 2 / (n_base - 1))
    else:
        nu = np.full(n_sim, n_base - 1.0)
    t = student_t.ppf(0.5 + rule["level"] / 2, nu)
    se = np.sqrt(ta + tb + sd_boot ** 2)
    lo, hi = d - t * se, d + t * se
    win = (d >= rule["min_delta"]) & (lo > 0)
    if rule["all_seeds_above_baseline_mean"]:
        win &= (a > base_mean[:, None]).all(axis=1)
    reg = hi < 0
    return {"win": float((win & ~reg).mean()), "regression": float(reg.mean())}


def operating_characteristics(sd_seed, sd_boot, n_arm=3, n_base=5, rule=DECISION_RULE, n_sim=4000, seed=0,
                              effects=(0.0, 0.02, 0.03, 0.05), grid=None):
    """The rule's false-WIN rate at a true effect of 0, its power at ``effects``, its REGRESSION rate at 0, and the
    smallest effect (on ``grid``) detected at 80% power, from ``simulate_rule``.

    >>> oc = operating_characteristics(0.01, 0.005, n_sim=2000)
    >>> oc["false_win"] < 0.05 and oc["power"]["0.05"] > 0.8
    True
    """
    grid = np.round(np.arange(0.0, 0.1501, 0.005), 3) if grid is None else grid
    out = {"sd_seed": sd_seed, "sd_boot": sd_boot, "n_arm": n_arm, "n_base": n_base, "n_sim": n_sim,
           "power": {f"{e:.2f}": simulate_rule(e, sd_seed, sd_boot, n_arm, n_base, rule, n_sim, seed)["win"]
                     for e in effects}}
    null = simulate_rule(0.0, sd_seed, sd_boot, n_arm, n_base, rule, n_sim, seed)
    out["false_win"], out["null_regression"] = null["win"], null["regression"]
    out["mde80"] = next((float(e) for e in grid
                         if simulate_rule(float(e), sd_seed, sd_boot, n_arm, n_base, rule, n_sim, seed)["win"] >= 0.8),
                        None)
    return out


def gate_block(pred, split_df, prior_entry, shares, label_type, rules=("majority2", "margin")):
    """The report-only gate statistics of one run (``prior_rules.gate_rows`` under each prior rule in ``rules``):
    per eval split and verdict the gated n, coverage, precision and its Wilson lower bound at
    ``retrain_report.WILSON_Z`` (1.96, the round-3 report's), the confidence where the val-fitted uncapped map first
    clears the gate (``retrain_report.clears_at``), the share of test verdicts at exactly 1.0
    (``retrain_report.saturation_row``) and, on ``test_heldout``, the Agree precision at the gate under the
    deployments' own class mix (``retrain_report.correct_weight``)."""
    from analysis import prior_rules as pr
    from analysis import retrain_report as rr
    from calibration import calibrate as cal
    from calibration import _isotonic_map as im
    from calibration.per_city import wilson_interval
    out = {}
    df = cal.verdicts(pred)
    k_correct = None
    if prior_entry is not None and {"author", "class_source"} <= set(split_df.columns):
        try:
            k_correct = rr.correct_weight(split_df, prior_entry)
        except (AssertionError, ValueError, KeyError):
            k_correct = None
    for rule in rules:
        pi = pr.rule_prior(shares["types"], label_type, rule)
        rows = pr.gate_rows(pred, pi)
        amap = cal.calibrate(pred, pi, "val", "test")[0]   # no hold-back: the uncapped, val-fitted map
        for split, block in rows.items():
            for v, g in block.items():
                if isinstance(g, dict) and g.get("gate_n") is not None and g.get("gate_precision") is not None \
                        and np.isfinite(g["gate_precision"]):
                    g["wilson_lo"] = float(wilson_interval(g["gate_precision"], g["gate_n"], rr.WILSON_Z)[0])
        sat = rr.saturation_row(df, label_type, amap)
        entry = {"prior": pi, "splits": rows, "clears_at": {v: rr.clears_at(amap[v]) for v in cal.CLASSES},
                 "share_agree_at_1": sat["share_agree_at_1"], "share_disagree_at_1": sat["share_disagree_at_1"]}
        held = df[df["split"] == "test_heldout"]
        if k_correct is not None and len(held):
            served = rr.served_for(held, amap)
            m = (held["verdict"] == "correct").to_numpy() & (served >= im.GATE)
            w = rr.own_weights(held["true"], k_correct)[m]
            entry["heldout_agree_own_mix_precision"] = (float(np.average(held["correct"].to_numpy()[m], weights=w))
                                                        if m.any() else None)
            entry["heldout_agree_own_mix_n"] = int(m.sum())
        out[rule] = entry
    return out


def boot_summary(b, names=None):
    """The bootstrap's bookkeeping for the report: deployments used, resamples kept and dropped, and the redraws per
    deployment (held-out: named through ``names``, private ones lettered) or pooled (test: many deployments)."""
    if not b:
        return None
    redraws = b["redraws"]
    return {"deployments": len(b["deployments"]), "n_boot": b["n_boot"], "kept": int(len(b["boot"])),
            "dropped": b["dropped"],
            "redraws": ({names.get(c, c): n for c, n in redraws.items()} if names is not None
                        else int(sum(redraws.values())))}


def _variant_runs(data_dir, label_type, variants, base_seed=42):
    """``{variant: {seed: run name}}`` for ``variants``: every ``<variant>-s<seed>`` replicate plus the base-seed run
    (``metrics_<variant>.json``, or ``metrics.json`` for October's ``baseline``) when its metrics file exists."""
    reps = cl.seed_runs(data_dir, label_type, base_seed)
    out = {v: {} for v in variants}
    for (v, s), name in reps.items():
        if v in out:
            out[v][s] = name
    for v in variants:
        if base_seed not in out[v] and os.path.exists(cl.metrics_path(data_dir, label_type, v)):
            out[v][base_seed] = v
    return {v: dict(sorted(r.items())) for v, r in out.items()}


def rule_seeds(variant, kind, rule=DECISION_RULE):
    """The seeds the rule reads for ``variant`` (``kind``: "baseline", "confirmatory", "exploratory", "october"):
    exactly ``seeds_baseline`` / ``seeds_confirmatory``, or None (every seed on disk: exploratory arms and October's
    trio are descriptive)."""
    return {"baseline": rule["seeds_baseline"], "confirmatory": rule["seeds_confirmatory"]}.get(kind)


def select_seeds(on_disk, seeds):
    """``(used, not_used)`` of ``{seed: run}``: the runs at ``seeds`` (all of them when ``seeds`` is None) and the
    seeds on disk the rule does not read. A seed trained beyond the rule's list never enters D, its interval or a
    criterion, so a partly finished or extended stage cannot move a verdict (no optional stopping).

    >>> select_seeds({42: "a", 43: "a-s43", 45: "a-s45"}, [42, 43, 44])
    ({42: 'a', 43: 'a-s43'}, [45])
    """
    if seeds is None:
        return dict(on_disk), []
    return {s: n for s, n in on_disk.items() if s in seeds}, sorted(s for s in on_disk if s not in seeds)


def expected_settings(variant):
    """What a run of ``variant`` must have recorded in its ``run_key.json``: ``context_arm`` (None for
    ``baseline-rerun``: ``fine_tune_classifier.context_settings`` writes no arm for the tight-crop model) and
    ``context_view_res`` (448 for ``<arm>-r448``, else 224; None without an arm). None for a name that is not this
    chain's.

    >>> expected_settings("view_heatmap-r448"), expected_settings("baseline-rerun")
    ({'context_arm': 'view_heatmap', 'context_view_res': 448}, {'context_arm': None, 'context_view_res': None})
    """
    if variant == "baseline-rerun":
        return {"context_arm": None, "context_view_res": None}
    arm, res = (variant[:-len("-r448")], 448) if variant.endswith("-r448") else (variant, 224)
    if arm not in CONFIRMATORY + ("side_info", "side_info_nohq"):
        return None
    return {"context_arm": arm, "context_view_res": res}


def recipe_problems(rec):
    """The pinned training knobs (``RECIPE_SETTINGS``) that the run record ``rec`` records with another value, one
    message each; an absent ``target`` or ``batch_size`` reads as the default the trainer leaves out.

    >>> recipe_problems({"best_metric": "macro_f1"}), recipe_problems({"best_metric": "macro_f1", "batch_size": 4})
    ([], ["record batch_size 4, the recipe's is 8"])
    """
    out = []
    for k, want in RECIPE_SETTINGS.items():
        got = rec.get(k)
        if (RECIPE_ABSENT.get(k) if got is None else got) != want:
            out.append(f"record {k} {got}, the recipe's is {want}")
    return out


def record_problem(rec, variant, seed, split_sha, epochs=RECIPE_EPOCHS):
    """Why the run record ``rec`` (``run_record``) does not belong to ``variant`` at ``seed`` on this split, or None:
    the record's status, seed, split hash, context arm, view resolution, (when ``epochs`` is not None) epochs, and the
    knobs the job file pins (``recipe_problems``: best metric, target, per-device batch).

    >>> rec = {"status": "ok", "seed": 43, "split_sha256": "h", "context_arm": "view_heatmap", "context_view_res": 448,
    ...        "epochs": 25, "best_metric": "macro_f1"}
    >>> record_problem(rec, "view_heatmap", 43, "h")
    'record context_view_res 448, view_heatmap needs 224'
    """
    if rec["status"] != "ok":
        return f"run record {rec['status']}"
    if rec["seed"] != seed:
        return f"record seed {rec['seed']} for tag seed {seed}"
    if rec["split_sha256"] != split_sha:
        return "record split hash differs from this split.csv"
    for k, v in (expected_settings(variant) or {}).items():
        if rec.get(k) != v:
            return f"record {k} {rec.get(k)}, {variant} needs {v}"
    if epochs is not None and rec.get("epochs") != epochs:
        return f"record epochs {rec.get('epochs')}, the recipe's is {epochs}"
    recipe = recipe_problems(rec)
    return recipe[0] if recipe else None


def _mean_sd(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)


def slice_seed_rows(preds, split_df, runs, base_variant, train, min_n=None):
    """Type x severity and type x tag slices across seeds (descriptive): per slice and split, rows, incorrect rows,
    training rows (thin flagged), the AUC of October's round-3 trio and of the baseline rerun (seed means), and per
    variant the seed-mean AUC difference against the baseline rerun's seed mean, its min and max over seeds and how
    many seeds are above that mean."""
    min_n = sl.MIN_N if min_n is None else min_n
    counts = {by: sl.with_slice(train, by).groupby("slice").size().to_dict() for by in sl.BY}
    out = []
    for split in ("test", "test_heldout"):
        names = [n for r in runs.values() for n in r.values() if n in preds]
        if not names:
            continue
        df = aligned_scores(preds, split, names)
        if df is None or df.empty:
            continue
        df = df.merge(split_df[["label_uid", "severity", "tags"]], on="label_uid", how="left", validate="one_to_one")
        for by in sl.BY:
            d = sl.with_slice(df, by)
            for name in sl.slice_names(df, by, min_n):
                g = d[d["slice"] == name]
                if len(g) < min_n or not 0 < g["y"].sum() < len(g):
                    continue
                auc = dict(zip(names, _rank_auc(g["y"].to_numpy(), g[names].to_numpy(float))))
                base = [auc[n] for n in runs.get(base_variant, {}).values() if n in auc]
                base_mean = float(np.mean(base)) if base else None
                n_train = int(counts[by].get(name, 0))
                row = {"split": split, "by": by, "slice": name, "n": int(len(g)), "n_incorrect": int(g["y"].sum()),
                       "train_n": n_train, "needs_data": n_train < sl.THIN_TRAIN,
                       "october_auc": _mean_sd([auc[n] for n in runs.get(OCTOBER, {}).values() if n in auc])[0],
                       "baseline_rerun_auc": base_mean, "variants": {}}
                for v, r in runs.items():
                    if v in (base_variant, OCTOBER) or base_mean is None:
                        continue
                    vals = [auc[n] - base_mean for n in r.values() if n in auc]
                    if vals:
                        row["variants"][v] = {"delta_mean": float(np.mean(vals)), "delta_min": float(min(vals)),
                                              "delta_max": float(max(vals)), "n_seeds": len(vals),
                                              "k_above": int(sum(x > 0 for x in vals))}
                out.append(row)
    return out


def heldout_display_names(preds, public):
    """``retrain_report.heldout_names`` over every deployment in any run's ``test_heldout`` rows (qualifying for the
    within-city AUC or not), as ``confident_learning.heldout_rows`` letters them for ``arms.md``, so a private
    deployment carries the same letter in every table."""
    from analysis import retrain_report as rr
    cities = sorted(set().union(*[set(p.loc[p["split"] == "test_heldout", "city"]) for p in preds.values()])
                    ) if preds else []
    us = {c for c in cities if len(c.split("-")[-1]) == 2 and c.split("-")[-1].isalpha()}
    return rr.heldout_names(cities, public, us)


def decide_type(data_dir, label_type, split_df, arms, public, shares, prior_entry, subset, base_variant, n_boot,
                boot_seed, within_min_n=10, rule=DECISION_RULE, base_seed=42, min_boot=MIN_DECISION_BOOT):
    """Everything the decision needs for one type: the runs, their records, the joint bootstrap on test and held-out,
    one row per arm (confirmatory: a verdict; exploratory: never WIN), the drift row against October's trio, the gate
    block per run, the operating characteristics, and the seed slices. A plain dict, JSON-ready.

    Only the rule's seeds are read (``select_seeds``): the baseline at ``seeds_baseline``, a confirmatory arm at
    ``seeds_confirmatory``; any other seed on disk is listed under ``not_used`` and enters nothing. A verdict also
    needs at least ``min_boot`` kept joint resamples on each of test and held-out (the bootstrap term of se)."""
    variants = [base_variant, OCTOBER, *arms]
    kinds = {v: ("baseline" if v == base_variant else "october" if v == OCTOBER
                 else "confirmatory" if v in CONFIRMATORY else "exploratory") for v in variants}
    on_disk = _variant_runs(data_dir, label_type, variants, base_seed)
    runs, not_used = {}, {}
    for v, r in on_disk.items():
        runs[v], unused = select_seeds(r, rule_seeds(v, kinds[v], rule))
        if unused:
            not_used[v] = unused
    preds = {}
    for r in runs.values():
        for name in r.values():
            p = cl.load_predictions(data_dir, label_type, name, split_df)
            if p is not None:
                preds[name] = p
    runs = {v: {s: n for s, n in r.items() if n in preds} for v, r in runs.items()}
    split_sha = sha256_file(os.path.join(data_dir, label_type, "split.csv"))
    records = {}
    for v, r in runs.items():
        for s, name in r.items():
            rec = run_record(data_dir, label_type, name)
            # this chain's runs must carry a matching record; October's are shown, never required
            problem = None if v == OCTOBER else record_problem(rec, v, s, split_sha)
            records[name] = {**rec, "variant": v, "seed": s, "problem": problem}
    names = [n for r in runs.values() for n in r.values()]
    boot = {split: within_city_boot(preds, split, names, within_min_n, n_boot, boot_seed) if names else None
            for split in ("test_heldout", "test")}
    kept = {split: (len(b["boot"]) if b else 0) for split, b in boot.items()}
    held_names = heldout_display_names(preds, public)
    gates = {}
    for name in names:
        try:
            gates[name] = gate_block(preds[name], split_df, prior_entry, shares, label_type)
        except Exception as e:  # recorded; a missing gate block makes the verdict INCOMPLETE
            gates[name] = {"error": f"{type(e).__name__}: {e}"}

    def interval(split, arm_names, base_names):
        """``seed_mean_delta`` with the bootstrap variance of D; without one (``boot_var_of_delta`` None) the record
        keeps D but no bounds, never an interval silently narrowed to the seed terms."""
        b = boot[split]
        if not b or any(b["point"].get(n) is None for n in arm_names + base_names):
            return None
        vb = boot_var_of_delta(b["boot"], b["names"], arm_names, base_names)
        r = seed_mean_delta([b["point"][n] for n in arm_names], [b["point"][n] for n in base_names],
                            vb or 0.0, rule["level"])
        if vb is None:
            r.update(se=None, t=None, lo=None, hi=None, var_boot=None)
        return r

    base_runs = runs.get(base_variant, {})
    base_names = list(base_runs.values())
    rows = []
    for arm in arms:
        a_runs = runs.get(arm, {})
        a_names = list(a_runs.values())
        row = {"arm": arm, "kind": kinds[arm], "seeds": sorted(a_runs), "baseline_seeds": sorted(base_runs),
               "seeds_not_used": not_used.get(arm, []), "baseline_seeds_not_used": not_used.get(base_variant, [])}
        if a_names and base_names:
            row["heldout"] = interval("test_heldout", a_names, base_names)
            row["test"] = interval("test", a_names, base_names)
            per = (boot["test_heldout"] or {}).get("per_deployment", {})
            row["deployments"] = {held_names.get(c, c): float(np.mean([per[c][n] for n in a_names])
                                                               - np.mean([per[c][n] for n in base_names]))
                                  for c in per}
        reasons = []
        if subset:
            reasons.append("subset shadow")
        if not a_names:
            reasons.append("not run")
        else:
            if row["kind"] == "confirmatory":
                miss = sorted(set(rule["seeds_confirmatory"]) - set(a_runs))
                if miss:
                    reasons.append(f"missing arm seed(s) {miss}")
            miss_b = sorted(set(rule["seeds_baseline"]) - set(base_runs))
            if miss_b:
                reasons.append(f"missing {base_variant} seed(s) {miss_b}")
            bad = [f"{n}: {records[n]['problem']}" for n in a_names + base_names if records[n]["problem"]]
            if bad:
                reasons.append("; ".join(bad))
            no_gate = [n for n in a_names + base_names if "error" in gates.get(n, {"error": 1})]
            if no_gate:
                reasons.append(f"gate block missing for {no_gate}")
            short = [f"{s} {kept[s]}" for s in ("test_heldout", "test") if kept[s] < min_boot]
            if short:
                reasons.append(f"joint bootstrap kept fewer than {min_boot} resamples ({', '.join(short)})")
            if not row.get("heldout") or row["heldout"].get("lo") is None or not row.get("test") \
                    or row["test"].get("lo") is None:
                reasons.append("no interval (too few seeds, no bootstrap variance or no qualifying deployment)")
        if row["kind"] == "exploratory":
            row["verdict"] = "exploratory"
            row["reason"] = "; ".join(reasons) or "exploratory arms never produce a WIN"
        elif reasons:
            row["verdict"], row["reason"] = "INCOMPLETE", "; ".join(reasons)
        else:
            row["verdict"], row["criteria"] = apply_rule(row["heldout"], row["test"], list(row["deployments"].values()),
                                                         rule)
            row["reason"] = NO_GAIN_WORDING if row["verdict"] == "NO GAIN" else ""
        rows.append(row)
    oct_runs = list(runs.get(OCTOBER, {}).values())
    drift = {"october_seeds": sorted(runs.get(OCTOBER, {})), "rerun_seeds": sorted(base_runs)}
    for split, key in (("test_heldout", "heldout"), ("test", "test")):
        b = boot[split] or {"point": {}}
        rm, rsd = _mean_sd([b["point"].get(n) for n in base_names])
        om, osd = _mean_sd([b["point"].get(n) for n in oct_runs])
        drift[key] = {"rerun_mean": rm, "rerun_sd": rsd, "october_mean": om, "october_sd": osd,
                      "difference": (rm - om) if rm is not None and om is not None else None}
    drift["flag"] = drift_flag(drift["heldout"])
    sd_seed = drift["heldout"]["rerun_sd"] if len(base_names) >= 3 else OCTOBER_SEED_SD.get(label_type)
    sd_source = f"{base_variant}, {len(base_names)} seeds" if len(base_names) >= 3 else "October, 3 seeds"
    pair = (base_names + oct_runs)[:2]
    vb = boot_var_of_delta((boot["test_heldout"] or {}).get("boot"), (boot["test_heldout"] or {}).get("names", []),
                           pair[:1], pair[1:]) if len(pair) == 2 else None
    oc = None
    if sd_seed is not None and vb is not None:
        oc = {"sd_seed_source": sd_source, "sd_boot_source": f"paired bootstrap of {pair[0]} - {pair[1]}",
              "arm_seeds_3": operating_characteristics(sd_seed, vb ** 0.5, 3, rule=rule),
              "arm_seeds_5": operating_characteristics(sd_seed, vb ** 0.5, 5, rule=rule)}
    train = split_df[split_df["split"] == "train"]
    return {"runs": {v: {str(s): n for s, n in r.items()} for v, r in runs.items()}, "not_used": not_used,
            "records": records, "rows": rows, "drift": drift, "gates": gates, "oc": oc,
            "boot": {k: boot_summary(v, held_names if k == "test_heldout" else None) for k, v in boot.items()},
            "slices": slice_seed_rows(preds, split_df, runs, base_variant, train)}


def drift_flag(heldout):
    """The rule's drift flag: the rerun's held-out seed mean differs from October's by more than two October seed sds.

    >>> drift_flag({"difference": 0.05, "october_sd": 0.02}), drift_flag({"difference": 0.03, "october_sd": 0.02})
    (True, False)
    """
    return bool(heldout["difference"] is not None and heldout["october_sd"]
                and abs(heldout["difference"]) > 2 * heldout["october_sd"])


def _f3(v, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def _iv(r):
    if not r or r.get("delta") is None:
        return "—"
    if r.get("lo") is None:
        return f"{r['delta']:+.3f}"
    return f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def rule_row_note(rule=DECISION_RULE):
    """One sentence under an operating-characteristics table naming the row the rule uses (the tables print both the
    three- and the five-arm-seed rows; the rule reads ``len(rule["seeds_confirmatory"])``).

    >>> rule_row_note({"seeds_confirmatory": [42, 43, 44, 45, 46]}).startswith("The rule reads 5 confirmatory")
    True
    """
    n = len(rule["seeds_confirmatory"])
    which = (f"so the {n}-arm-seed row is the one that applies; the other row is for comparison" if n in (3, 5)
             else "so neither row applies")
    return f"The rule reads {n} confirmatory arm seeds (`seeds_confirmatory`), {which}."


def decision_markdown(decisions, rule=DECISION_RULE):
    lines = ["# Context arms: pre-registered decision", "",
             "Applied mechanically from `DECISION_RULE` (below and in `decision.json`; "
             "docs/context-arms-2026-11.md has the rule in words). D = seed-mean minus the "
             f"`{rule['baseline_variant']}` seed-mean of the within-city AUC; interval D ± t·se with "
             "se² = s_arm²/n_arm + s_base²/n_base + var_boot (joint panorama bootstrap within deployment) and the "
             f"Welch t at {rule['level']:.0%}. `NO GAIN` means {NO_GAIN_WORDING}; read it next to the power table. "
             "Exploratory arms never produce a `WIN`; a single seed's interval omits that arm's seed variance. Only the "
             "rule's seeds are read (`seeds_baseline`, `seeds_confirmatory`); any other seed on disk is listed as not "
             "used and enters nothing.", ""]
    flagged = [lt for lt, d in decisions.items() if d["drift"]["flag"]]
    if flagged:
        lines += [f"**Drift flag: {', '.join(flagged)}.** The `{rule['baseline_variant']}` held-out seed mean differs "
                  "from October's round-3 seed mean by more than two October seed sds (drift row in each section); "
                  "the decision still uses the rerun.", ""]
    lines += ["```json", json.dumps(rule, sort_keys=True), "```", ""]
    for lt, d in decisions.items():
        lines += [f"## {lt}", "", "| arm | kind | seeds | verdict | D held-out within-city [CI] | D test within-city [CI] "
                  "| arm seeds above baseline mean | held-out deployments (D) | reason |",
                  "|---|---|---|---|---:|---:|---:|---|---|"]
        for r in d["rows"]:
            dep = ", ".join(f"{k} {v:+.3f}" for k, v in sorted((r.get("deployments") or {}).items()))
            h = r.get("heldout") or {}
            lines.append(f"| {r['arm']} | {r['kind']} | {', '.join(map(str, r['seeds'])) or '—'} | {r['verdict']} | "
                         f"{_iv(r.get('heldout'))} | {_iv(r.get('test'))} | "
                         f"{h.get('n_above', '—') if h else '—'} of {h.get('n_arm', '—') if h else '—'} | "
                         f"{dep or '—'} | {r.get('reason', '')} |")
        unused = {v: s for v, s in (d.get("not_used") or {}).items() if v != OCTOBER}
        if unused:
            lines += ["", f"{lt}: seeds on disk, not used by the rule: "
                      + "; ".join(f"{v} {s}" for v, s in sorted(unused.items())) + "."]
        for r in d["rows"]:
            if r.get("criteria"):
                lines += ["", f"{lt} / {r['arm']}: " + "; ".join(
                    f"{c['criterion']}: {c['value'] if not isinstance(c['value'], float) else round(c['value'], 4)} "
                    f"({'pass' if c['pass'] else 'fail'})" for c in r["criteria"])]
        dr = d["drift"]
        lines += ["", f"Drift row (`{rule['baseline_variant']}` seeds {dr['rerun_seeds']} vs October's round-3 seeds "
                  f"{dr['october_seeds']}):", "", "| endpoint | rerun mean ± sd | October mean ± sd | rerun − October |",
                  "|---|---:|---:|---:|"]
        for key, label in (("heldout", "within-city held-out"), ("test", "within-city test")):
            x = dr[key]
            lines.append(f"| {label} | {_f3(x['rerun_mean'])} ± {_f3(x['rerun_sd'])} | {_f3(x['october_mean'])} ± "
                         f"{_f3(x['october_sd'])} | {_f3(x['difference'], True)} |")
        if dr["flag"]:
            lines += ["", f"**{lt}: the rerun's held-out seed mean differs from October's by more than two October "
                      "seed sds; the decision still uses the rerun.**"]
        oc = d.get("oc")
        if oc:
            lines += ["", f"Operating characteristics (seed sd from {oc['sd_seed_source']}, "
                      f"{oc['arm_seeds_3']['sd_seed']:.3f}; bootstrap sd of D from {oc['sd_boot_source']}, "
                      f"{oc['arm_seeds_3']['sd_boot']:.3f}; criteria 1-3 simulated, so WIN rates are upper bounds):",
                      "", "| arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | +0.03 | +0.05 | "
                      "80% power at |", "|---:|---:|---:|---:|---:|---:|---:|"]
            for k in ("arm_seeds_3", "arm_seeds_5"):
                o = oc[k]
                lines.append(f"| {o['n_arm']} | {o['false_win']:.3f} | {o['null_regression']:.3f} | "
                             f"{o['power']['0.02']:.2f} | {o['power']['0.03']:.2f} | {o['power']['0.05']:.2f} | "
                             f"{'+%.3f' % o['mde80'] if o['mde80'] is not None else 'above +0.15'} |")
            lines += ["", rule_row_note(rule)]
        lines.append("")
    return "\n".join(lines) + "\n"


def gate_markdown(decisions):
    lines = ["# Gate block (report-only)", "",
             "Per run: the val-fitted, uncapped map under the **majority2** deployment prior (the margin prior "
             "beside it), applied at the 0.92 gate. gated n, coverage and precision per verdict; Wilson lower bound "
             "at z = 1.96 (`retrain_report.WILSON_Z`, as the round-3 report); the confidence where each map first "
             "clears the gate; the share of test verdicts at exactly 1.0 (a new head's float saturation moves "
             "coverage without any change in discrimination); on held-out, the Agree precision at the gate under "
             "the deployments' own class mix. It never changes a verdict.", ""]
    for lt, d in decisions.items():
        lines += [f"## {lt}", "", "| run | prior | split | verdict | gated n | coverage | precision | Wilson lo | "
                  "clears at | at 1.0 | held-out own-mix Agree precision |", "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
        for name, g in d["gates"].items():
            if "error" in g:
                lines.append(f"| {name} | — | — | — | gate block missing: {g['error']} | | | | | | |")
                continue
            for rule, e in g.items():
                for split, block in e["splits"].items():
                    for v in ("correct", "incorrect"):
                        x = block.get(v) or {}
                        lines.append(f"| {name} | {rule} | {split} | {'Agree' if v == 'correct' else 'Disagree'} | "
                                     f"{x.get('gate_n', '—')} | {_f3(x.get('gate_coverage'))} | "
                                     f"{_f3(x.get('gate_precision'))} | {_f3(x.get('wilson_lo'))} | "
                                     f"{e['clears_at'][v]} | "
                                     f"{_f3(e['share_agree_at_1'] if v == 'correct' else e['share_disagree_at_1'])} | "
                                     f"{_f3(e.get('heldout_agree_own_mix_precision')) if v == 'correct' and split == 'test_heldout' else ''} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def slice_seeds_markdown(decisions):
    lines = ["# Severity and tag slices across seeds (descriptive)", "",
             "AUC of the incorrect score per slice; the baseline rerun's and October's round-3 seed means, and per "
             "variant the seed-mean difference against the rerun's seed mean with its range over seeds and how many "
             f"seeds are above it. ⚠ thin = fewer than {sl.THIN_TRAIN} training rows. Not corrected for the number "
             "of slices; no verdict reads this table.", ""]
    for lt, d in decisions.items():
        variants = sorted({v for r in d["slices"] for v in r["variants"]})
        for split in ("test", "test_heldout"):
            for by in sl.BY:
                rs = [r for r in d["slices"] if r["split"] == split and r["by"] == by]
                if not rs:
                    continue
                lines += [f"## {lt} / {split} / by {by}", "",
                          f"| {by} | n | incorrect | training rows | round-3 (Oct) AUC | baseline rerun AUC | "
                          + " | ".join(f"{v} Δ mean [min, max] (k/n above)" for v in variants) + " |",
                          "|---|---:|---:|---:|---:|---:|" + "---:|" * len(variants)]
                for r in rs:
                    cells = []
                    for v in variants:
                        x = r["variants"].get(v)
                        cells.append("—" if not x else f"{x['delta_mean']:+.3f} [{x['delta_min']:+.3f}, "
                                     f"{x['delta_max']:+.3f}] ({x['k_above']}/{x['n_seeds']})")
                    lines.append(f"| {r['slice']} | {r['n']:,} | {r['n_incorrect']:,} | {r['train_n']:,}"
                                 f"{' ⚠ thin' if r['needs_data'] else ''} | {_f3(r['october_auc'])} | "
                                 f"{_f3(r['baseline_rerun_auc'])} | " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines) + "\n"


def power_table(data_dir, label_type, variant=OCTOBER, n_boot=1000, seed=0, within_min_n=10, n_sim=4000,
                rule=DECISION_RULE):
    """The operating characteristics of the rule for one type from the seeds of ``variant`` (October's ``baseline``
    trio before the pilot, ``baseline-rerun`` after it): the within-city held-out AUC's sample sd over seeds, the
    bootstrap sd of a difference between two of its seeds (the joint panorama bootstrap), and
    ``operating_characteristics`` at 3 and 5 arm seeds against the five baseline seeds of the rule."""
    split_df = cl.read_split(os.path.join(data_dir, label_type, "split.csv"))
    runs = _variant_runs(data_dir, label_type, [variant])[variant]
    preds = {n: cl.load_predictions(data_dir, label_type, n, split_df) for n in runs.values()}
    preds = {n: p for n, p in preds.items() if p is not None}
    names = [n for n in runs.values() if n in preds]
    if len(names) < 2:
        return {"variant": variant, "seeds": sorted(runs), "error": "fewer than two seeds with predictions"}
    b = within_city_boot(preds, "test_heldout", names, within_min_n, n_boot, seed)
    sd_seed = _mean_sd([b["point"][n] for n in names])[1]
    sd_boot = boot_var_of_delta(b["boot"], names, names[1:2], names[2:3] or names[:1]) ** 0.5
    # one pair at one bootstrap seed drives the table; the other pairs show how much that choice moves it
    all_pairs = {f"{x} - {y}": boot_var_of_delta(b["boot"], names, [x], [y]) ** 0.5
                 for i, x in enumerate(names) for y in names[i + 1:]}
    n_base = len(rule["seeds_baseline"])
    return {"variant": variant, "seeds": sorted(runs), "within_city_heldout": {n: b["point"][n] for n in names},
            "sd_seed": sd_seed, "sd_boot": sd_boot, "sd_boot_pair": [names[1], (names[2:3] or names[:1])[0]],
            "sd_boot_all_pairs": all_pairs,
            "n_boot": n_boot, "kept": int(len(b["boot"])),
            "arm_seeds_3": operating_characteristics(sd_seed, sd_boot, 3, n_base, rule, n_sim, seed),
            "arm_seeds_5": operating_characteristics(sd_seed, sd_boot, 5, n_base, rule, n_sim, seed)}


def power_markdown(tables):
    """The doc's operating-characteristics table; the recommendation follows A7 of the sign-off: five arm seeds for
    a type whose power at +0.03 with three is below 0.5."""
    lines = ["| type | seed sd | bootstrap sd of D | arm seeds | false WIN at 0 | REGRESSION at 0 | power +0.02 | "
             "+0.03 | +0.05 | effect at 80% power | recommended arm seeds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lt, t in tables.items():
        if "error" in t:
            lines.append(f"| {lt} | {t['error']} | | | | | | | | | |")
            continue
        rec = 5 if t["arm_seeds_3"]["power"]["0.03"] < 0.5 else 3
        for k in ("arm_seeds_3", "arm_seeds_5"):
            o = t[k]
            mde = f"+{o['mde80']:.3f}" if o["mde80"] is not None else "above +0.15"
            lines.append(f"| {lt} | {t['sd_seed']:.3f} | {t['sd_boot']:.3f} | {o['n_arm']} | {o['false_win']:.3f} | "
                         f"{o['null_regression']:.3f} | {o['power']['0.02']:.2f} | {o['power']['0.03']:.2f} | "
                         f"{o['power']['0.05']:.2f} | {mde} | {rec if k == 'arm_seeds_3' else ''} |")
    lines += ["", rule_row_note() + " The recommendation column is the sign-off's rule, read from the three-seed row."]
    spread = [f"{lt} {min(t['sd_boot_all_pairs'].values()):.4f} to {max(t['sd_boot_all_pairs'].values()):.4f} "
              f"over {len(t['sd_boot_all_pairs'])} pairs" for lt, t in tables.items() if t.get("sd_boot_all_pairs")]
    if spread:
        lines += ["", "The bootstrap sd of D is from one seed pair at one bootstrap seed; over every pair of seeds it "
                  "ranges " + "; ".join(spread) + "."]
    return "\n".join(lines) + "\n"


def cmd_power(args):
    tables = {lt: power_table(args.data_dir, lt, args.variant, args.n_boot, args.seed, args.within_min_n, args.n_sim)
              for lt in _list(args.label_types)}
    text = power_markdown(tables)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"power_{args.variant}.json"), "w") as f:
            json.dump({"rule": DECISION_RULE, "data_dir": args.data_dir, "tables": tables}, f, indent=2, default=float)
            f.write("\n")
        with open(os.path.join(args.out, f"power_{args.variant}.md"), "w") as f:
            f.write(text)
    return 0


def report(args):
    from analysis import retrain_report as rr
    if args.decision_boot < MIN_DECISION_BOOT:
        raise SystemExit(f"--decision-boot {args.decision_boot}: a verdict needs at least {MIN_DECISION_BOOT} joint "
                         "resamples (the bootstrap term of the interval)")
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    public, public_source = rr.public_cities(args.public_cities)
    os.makedirs(args.out, exist_ok=True)
    everything = {"_public_cities_source": public_source, "_arms": arms}
    prov = {}
    tables, slice_tables, seed_tables, clip = [], [], [], {}
    decisions, run_provenance = {}, {}
    shadow_rec = read_shadow(args.data_dir)
    subset = bool(shadow_rec and shadow_rec.get("subset"))
    with open(args.shares) as f:
        shares = json.load(f)
    priors_path = os.path.join(args.data_dir, "deployment_priors.json")
    priors = {}
    if os.path.exists(priors_path):
        with open(priors_path) as f:
            priors = json.load(f)

    def note(path):
        if os.path.exists(path):
            prov[os.path.relpath(path, args.data_dir)] = sha256_file(path)

    for lt in label_types:
        split_path = os.path.join(args.data_dir, lt, "split.csv")
        if not os.path.exists(split_path):
            tables.append(f"### {lt}\n\nnot yet run (no {lt}/split.csv)\n")
            clip[lt] = None
            continue
        note(split_path)
        split_df = cl.read_split(split_path)
        names = tuple(dict.fromkeys(("baseline", args.baseline_variant, *arms)))
        preds = {n: cl.load_predictions(args.data_dir, lt, n, split_df) for n in names}
        preds = {k: v for k, v in preds.items() if v is not None}
        for n in names:
            note(cl.metrics_path(args.data_dir, lt, n))
            note(os.path.join(args.data_dir, lt, "predictions.csv" if n == "baseline" else f"predictions_{n}.csv"))
        rows = [arm_row(args.data_dir, lt, n, preds, args.within_min_n) for n in names]
        deltas = cl.overall_deltas(preds, [n for n in names if n != "baseline"], args.n_boot, args.seed)
        for r in rows:
            for split, key in (("test", "test"), ("test_heldout", "heldout")):
                d = deltas.get((r["variant"], split))
                if d:
                    r[f"{key}_delta"], r[f"{key}_delta_lo"], r[f"{key}_delta_hi"] = d["delta"], d["delta_lo"], d["delta_hi"]
        heldout = cl.heldout_rows(preds, public, args.within_min_n) if preds else []
        train = split_df[split_df["split"] == "train"]
        slices = slice_deltas(preds, [n for n in names if n != "baseline"], train, args.slice_min_n, args.n_boot,
                              args.seed)
        everything[lt] = {"arms": rows, "heldout": heldout, "slices": slices}
        tables.append(arms_markdown(lt, rows, heldout, args.within_min_n))
        if slices:
            slice_tables.append(slices_markdown(lt, slices, [a for a in names if a in preds and a != "baseline"]))
        runs = {k: v for k, v in cl.seed_runs(args.data_dir, lt, args.base_seed).items() if k[0] in names}
        if runs:
            for name in runs.values():
                note(cl.metrics_path(args.data_dir, lt, name))
                note(os.path.join(args.data_dir, lt, f"predictions_{name}.csv"))
            srows = cl.seed_rows(args.data_dir, lt, runs, split_df, args.within_min_n, args.n_boot, args.seed)
            summary = cl.seed_summary(srows)
            everything[lt]["seeds"] = {"rows": srows, "summary": summary}
            seed_tables.append(cl.seeds_markdown(lt, srows, summary))
        index_path = os.path.join(args.data_dir, lt, "context", "index.csv")
        if os.path.exists(index_path):
            note(index_path)
            clip[lt] = clipping_row(pd.read_csv(index_path))
        else:
            clip[lt] = None
        everything[lt]["clipping"] = clip[lt]
        decisions[lt] = decide_type(args.data_dir, lt, split_df, arms, public, shares, priors.get(lt), subset,
                                    args.baseline_variant, args.decision_boot, args.seed, args.within_min_n)
        everything[lt]["decision"] = decisions[lt]
        for name, rec in decisions[lt]["records"].items():
            run_provenance[f"{lt}/{name}"] = {k: rec[k] for k in ("variant", "seed", "status", "path",
                                                                  "split_sha256", "files_sha256", "problem")}
    with open(os.path.join(args.out, "decision.md"), "w") as f:
        f.write(decision_markdown(decisions))
    with open(os.path.join(args.out, "decision.json"), "w") as f:
        json.dump({"rule": DECISION_RULE, "subset_shadow": subset,
                   "types": {lt: {"rows": d["rows"], "drift": d["drift"], "oc": d["oc"], "boot": d["boot"]}
                             for lt, d in decisions.items()}}, f, indent=2, default=float)
        f.write("\n")
    with open(os.path.join(args.out, "gate.md"), "w") as f:
        f.write(gate_markdown(decisions))
    with open(os.path.join(args.out, "slices_seeds.md"), "w") as f:
        f.write(slice_seeds_markdown(decisions))
    with open(os.path.join(args.out, "arms.md"), "w") as f:
        f.write("# Context arms against the round-3 model\n\n"
                "Each arm is trained with the round-3 recipe on the round-3 split and scored on the same rows; Δ = run "
                "− October's seed-42 round-3 model (`baseline`), descriptive only (the decision is in `decision.md`, "
                "against the baseline rerun's seed mean), with a 95% interval from resampling whole panoramas of the "
                "evaluation split, the same resample for both models. Within-city AUC: per-deployment AUCs weighted by n_incorrect x n_correct "
                f"(deployments with at least {args.within_min_n} labels and both classes; count in brackets). Private "
                "deployments are lettered.\n\n" + "\n".join(tables))
    with open(os.path.join(args.out, "arms.json"), "w") as f:
        json.dump(everything, f, indent=2, default=float)
        f.write("\n")
    if slice_tables:
        with open(os.path.join(args.out, "slices.md"), "w") as f:
            f.write("# Severity and tag slices: each arm against the round-3 baseline\n\n"
                    "AUC of the incorrect score on the same rows; Δ = arm − baseline with a 95% interval from "
                    "resampling whole panoramas (`analysis/slices.py`). Blank under the minimum rows or when the "
                    f"smaller class has fewer than {sl.MIN_CLASS_FOR_CI}; ⚠ thin = fewer than {sl.THIN_TRAIN} "
                    "training rows. Not corrected for the number of slices: at 95%, about one cell in twenty "
                    "excludes zero by chance.\n\n" + "\n".join(slice_tables))
    if seed_tables:
        with open(os.path.join(args.out, "seeds.md"), "w") as f:
            f.write("# Seed replicates\n\nEvery run repeated with another training seed (`TAG=<run>-s<seed>`), the "
                    f"chain's own run counted as seed {args.base_seed}; each compared with the baseline of the same "
                    "seed.\n\n" + "\n".join(seed_tables))
    with open(os.path.join(args.out, "clipping.md"), "w") as f:
        f.write(clipping_markdown(clip))
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump({"files_sha256": dict(sorted(prov.items())), "public_cities_source": public_source,
                   "arguments": {k: v for k, v in vars(args).items() if k != "func"},
                   "bootstrap": {"seed": args.seed, "n_boot": args.n_boot, "decision_boot": args.decision_boot},
                   "decision_rule": DECISION_RULE, **_git_state(),
                   "shares_sha256": sha256_file(args.shares),
                   "shadow_run_sha256": (sha256_file(os.path.join(args.data_dir, SHADOW_FILE))
                                         if shadow_rec is not None else None),
                   "deployment_priors_sha256": sha256_file(priors_path) if os.path.exists(priors_path) else None,
                   "runs": dict(sorted(run_provenance.items()))}, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("\n".join(tables))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Launch side: the shadow data directory, preflight and the per-run check (login node: pandas only, never torch)
# ---------------------------------------------------------------------------------------------------------------------

SHADOW_FILE = "shadow_run.json"
#: Copied into the shadow for every type; split.csv byte-identical, so run_key.json's split hash is the baseline's.
SHADOW_COPY = ("split.csv", "manifest.csv", "generation_run.json")
#: October's round-3 outputs, copied when present (the drift row reads them; nothing writes to them).
OCTOBER_FILES = ("metrics.json", "predictions.csv", "metrics_baseline-s43.json", "metrics_baseline-s44.json",
                 "predictions_baseline-s43.csv", "predictions_baseline-s44.csv")
CROP_DIRS = ("correct", "incorrect")
EVAL_SPLITS = ("val", "test", "test_heldout")   # evaluate.EVAL_SPLITS (not imported: evaluate pulls in sklearn)
PANO_MB = 11.4               # mean staged panorama, measured on klone 2026-11-19 (400-file sample)
CONTEXT_KB_PER_LABEL = 100   # a 448 px view (~40 KB) and a wide crop (~60 KB) at WebP q95
WRITE_TEST_BYTES = 8 << 20   # scrubbed has refused real writes while a 0-byte touch succeeded
BACKBONE_CACHE = "hub/models--facebook--dinov2-large/snapshots/*/config.json"
SIDE_SHARE_COLUMNS = ("severity", "image_capture_date", "depth_m")
#: split.csv columns every run reads (fine_tune_classifier.load_split_frames, evaluate.load_split).
BASE_COLUMNS = ("label_uid", "city", "pano_id", "label_class", "crop_path", "split")
#: The run record a training job copies next to its outputs (``<data>/<type>/runs/<tag>/``), before any delete.
RECORD_FILES = ("training_run.json", "run_key.json", "trainer_state.json", "config.json")


def _real_inside(path, root):
    """True when ``path`` is ``root`` or below it, after resolving symlinks."""
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _git_state():
    from fine_tune_classifier import git_state  # torch-free at import (tests/test_context_shadow.py checks)
    return git_state()


def read_shadow(data_dir):
    """``<data_dir>/shadow_run.json`` as a dict, or None when ``data_dir`` is not a shadow."""
    path = os.path.join(data_dir, SHADOW_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def staged_subset(split_df, manifest_df, store):
    """The rows of ``split_df`` (and the manifest rows with the same ``label_uid``) whose panorama file is in the pano
    store ``store`` (``data_generation.pano_store_path``), plus ``{split: {class: n}}`` of what is kept. Refuses
    (``SystemExit``) when any split of ``split_df`` would be left with fewer than two classes."""
    import data_generation as dg
    have = [os.path.exists(dg.pano_store_path(store, c, p)) for c, p in zip(split_df["city"], split_df["pano_id"])]
    kept = split_df[np.asarray(have, dtype=bool)]
    counts = {s: {c: int(n) for c, n in g["label_class"].value_counts().sort_index().items()}
              for s, g in kept.groupby("split")}
    lost = [s for s in sorted(split_df["split"].unique()) if len(counts.get(s, {})) < 2]
    if lost:
        raise SystemExit(f"--only-staged {store}: split(s) {lost} would keep fewer than two classes "
                         f"(kept {counts}); a one-class split cannot be scored")
    return kept, manifest_df[manifest_df["label_uid"].isin(set(kept["label_uid"]))], counts


def shadow(src, dest, types, only_staged=None, today=None):
    """Build (or extend) the shadow data directory ``dest`` of the round-3 dataset ``src`` for ``types``.

    Per type, ``<dest>/<type>/`` gets symlinks to ``<src>/<type>/correct`` and ``incorrect`` (the crops, never copied
    or written), byte-identical copies of ``SHADOW_COPY`` and of any ``OCTOBER_FILES`` present; ``<dest>`` gets a copy
    of ``deployment_priors.json`` and ``shadow_run.json`` (source path, the rawLabels snapshot ``<src>/../rawlabels``
    the backfill reads, date, git state, the sha256 of every source file copied). Every job then runs with
    ``DATA=<dest>``, so evaluation outputs, context images and run records land on scratch and ``src`` is only read.

    ``only_staged`` (a pano store): keep only the rows whose panorama is in it (``staged_subset``), October's outputs
    not copied (they cover rows the subset lacks), ``"subset": true`` recorded; ``report`` never decides on it.

    Refuses a ``dest`` inside ``src``, an existing shadow of another source or ``only_staged``, and a type already in
    the shadow whose source files have changed since (``SystemExit``). Rerunning with the same inputs is a no-op
    apart from the date.

    Example::

        python analysis/context_arms.py shadow --src $SRC --dest $WORK/data \\
            --types NoCurbRamp,Obstacle,SurfaceProblem
    """
    src, dest = os.path.abspath(src), os.path.abspath(dest)
    only_staged = os.path.abspath(only_staged) if only_staged else None
    today = today or date.today().isoformat()
    if _real_inside(dest, src):
        raise SystemExit(f"--dest {dest} is inside --src {src}: the shadow exists so that nothing writes there")
    if only_staged:
        import data_generation as dg
        dg.check_pano_store_marker(only_staged)
    old = read_shadow(dest)
    if old is not None and (old.get("src") != src or old.get("only_staged") != only_staged):
        raise SystemExit(f"{os.path.join(dest, SHADOW_FILE)} is a shadow of {old.get('src')} "
                         f"(only_staged {old.get('only_staged')}); use another --dest")
    entries = dict((old or {}).get("types") or {})
    for lt in types:
        s_dir, d_dir = os.path.join(src, lt), os.path.join(dest, lt)
        missing = [n for n in SHADOW_COPY if not os.path.isfile(os.path.join(s_dir, n))]
        if missing:
            raise SystemExit(f"{s_dir} has no {missing}")
        names = list(SHADOW_COPY) + ([] if only_staged else [n for n in OCTOBER_FILES
                                                             if os.path.isfile(os.path.join(s_dir, n))])
        sources = {n: sha256_file(os.path.join(s_dir, n)) for n in names}
        if lt in entries and entries[lt].get("sources") != sources:
            changed = sorted(set(sources.items()) ^ set((entries[lt].get("sources") or {}).items()))
            raise SystemExit(f"{lt}: the shadow records different source hashes than {s_dir} has now "
                             f"({[c[0] for c in changed]}); make a new shadow")
        os.makedirs(d_dir, exist_ok=True)
        for c in CROP_DIRS:
            target, link = os.path.join(s_dir, c), os.path.join(d_dir, c)
            if os.path.islink(link):
                if os.readlink(link) != target:
                    raise SystemExit(f"{link} points at {os.readlink(link)}, not {target}")
            elif os.path.exists(link):
                raise SystemExit(f"{link} exists and is not a symlink to {target}")
            elif os.path.isdir(target):
                os.symlink(target, link)
        entry = {"sources": sources, "subset": bool(only_staged)}
        if only_staged:
            split = pd.read_csv(os.path.join(s_dir, "split.csv"), dtype=str, keep_default_na=False)  # pano_id a str
            manifest = pd.read_csv(os.path.join(s_dir, "manifest.csv"), dtype=str, keep_default_na=False)
            kept, kept_manifest, counts = staged_subset(split, manifest, only_staged)
            for df, name in ((kept, "split.csv"), (kept_manifest, "manifest.csv")):
                df.to_csv(os.path.join(d_dir, name), index=False, lineterminator="\n")
            shutil.copyfile(os.path.join(s_dir, "generation_run.json"), os.path.join(d_dir, "generation_run.json"))
            entry.update(kept=counts, rows_source=int(len(split)), rows_kept=int(len(kept)))
            print(f"{lt}: kept {len(kept):,} of {len(split):,} rows whose panorama is in {only_staged}: {counts}")
        else:
            for n in names:
                shutil.copyfile(os.path.join(s_dir, n), os.path.join(d_dir, n))
            print(f"{lt}: shadowed {len(names)} files and the crop directories")
        entries[lt] = entry
    priors = os.path.join(src, "deployment_priors.json")
    record = {"src": src, "dest": dest, "rawlabels": os.path.join(os.path.dirname(src), "rawlabels"),
              "only_staged": only_staged, "subset": bool(only_staged),
              "created": (old or {}).get("created", today), "updated": today, **_git_state(),
              "deployment_priors_sha256": sha256_file(priors) if os.path.exists(priors) else None,
              "types": entries}
    if os.path.exists(priors):
        shutil.copyfile(priors, os.path.join(dest, "deployment_priors.json"))
    _write_json(record, os.path.join(dest, SHADOW_FILE))
    print(f"wrote {os.path.join(dest, SHADOW_FILE)}")
    return record


def environment_pin(name, path=None):
    """The ``name==X`` pin in ``environment.yml`` (the klone env's), or None."""
    import re
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "environment.yml")
    with open(path) as f:
        m = re.search(rf"^\s*-\s*{re.escape(name)}==([\w.+-]+)\s*$", f.read(), re.M)
    return m.group(1) if m else None


def installed_versions(names=("pandas", "transformers")):
    """``{name: version or None}`` read with ``importlib.metadata`` (no import: torch must not load on a login
    node)."""
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def missing_shares(split_df, columns=SIDE_SHARE_COLUMNS):
    """Share of rows with no usable value per column (absent column -> 1.0): blank cells, and for ``depth_m`` also a
    non-positive depth (the side arms treat both as "no depth").

    >>> missing_shares(pd.DataFrame({"severity": ["3", ""], "depth_m": ["0", "7.5"]}))
    {'severity': 0.5, 'image_capture_date': 1.0, 'depth_m': 0.5}
    """
    out = {}
    for c in columns:
        if c not in split_df.columns:
            out[c] = 1.0
            continue
        v = split_df[c].astype(str).str.strip()
        miss = v.isin(["", "nan", "NaN", "None"])
        if c == "depth_m":
            miss |= ~(pd.to_numeric(v, errors="coerce") > 0)
        out[c] = round(float(miss.mean()), 4) if len(v) else 1.0
    return out


def write_probe(directory, nbytes=WRITE_TEST_BYTES):
    """Write ``nbytes`` to a new file in ``directory`` (created if needed), fsync, delete. None when it worked, else
    the error text: a 0-byte ``touch`` can succeed where a real write fails."""
    path = os.path.join(directory, f".context-arms-write-probe-{os.getpid()}")
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\0" * nbytes)
            f.flush()
            os.fsync(f.fileno())
        os.remove(path)
        return None
    except OSError as e:
        try:
            os.remove(path)
        except OSError:
            pass
        return str(e)


def preflight(data, runs, types, arms, stage="", keep=None, pano_store=None, backfill=False, skip_backfill=False,
              pair_october=False, hf_home=None, versions=None, transformers_pin=None, write_test=True, out=print,
              confirmatory_seeds=None, baseline_seeds=None, rule=DECISION_RULE):
    """Everything a stage needs before it may take a GPU, checked on the login node in seconds; returns the list of
    problems (empty: go). Never imports torch or transformers.

    - the shadow (``shadow_run.json``; each type's source hashes unchanged; ``SHADOW_COPY`` present; 20 sampled crops
      resolve through the symlinks);
    - each type's ``split.csv`` header has ``BASE_COLUMNS`` and every arm's ``context_model.FRAME_COLUMNS`` (a missing
      column would raise ``KeyError`` after the job holds its GPU); the share of rows missing severity, imagery date
      and depth is printed (a side arm without depth is silently weaker);
    - ``pair_october``: October's ``metrics.json`` / ``predictions.csv`` (with val rows) and the seed-43/44 files;
    - ``backfill``: the pano store's marker and every ok manifest row's panorama (``write_stage_files``: whenever a
      store path is given, marked or not, and a missing store counts every panorama as missing, it writes
      ``<data>/<type>/stage_files.txt`` with the count and a size estimate and names the restage commands), what the
      backfill will write, and a ``<city>.csv`` in the shadow's rawLabels snapshot for every manifest city;
      ``skip_backfill``: ``context/index.csv`` covers every ok manifest row;
    - ``confirmatory_seeds`` / ``baseline_seeds`` (a stage preset's seed lists): equal to ``rule``'s
      ``seeds_confirmatory`` / ``seeds_baseline``, so a stage never trains a seed list the decision does not read
      (the rule reads five confirmatory seeds, so stages A and B need ``A5=1``; the three-seed preset is refused);
    - ``versions`` (default ``installed_versions()``): pandas 2.x, transformers at ``environment.yml``'s pin;
    - the backbone in the HF cache (``hf_home``; a path check, not a load);
    - a real 8 MiB write to ``runs`` and ``data``;
    - ``keep``: free space and files on its filesystem (and ``hyakstorage`` when installed), printed only.
    """
    import glob
    import context_model as cm
    problems = []
    for got, key in ((confirmatory_seeds, "seeds_confirmatory"), (baseline_seeds, "seeds_baseline")):
        if got is not None and sorted(set(got)) != sorted(rule[key]):
            hint = (f"stages A and B: launch with A5=1, which trains the rule's {len(rule[key])} seeds; otherwise "
                    if key == "seeds_confirmatory" else "")
            problems.append(f"stage {stage} trains seeds {sorted(set(got))} where DECISION_RULE[{key!r}] reads "
                            f"{rule[key]}: {hint}change the rule (and the doc's json block) before the stage, or the "
                            "stage, never after")
    rec = read_shadow(data)
    if rec is None:
        problems.append(f"{data} has no {SHADOW_FILE}: make a shadow (analysis/context_arms.py shadow)")
    for lt in types:
        d = os.path.join(data, lt)
        entry = ((rec or {}).get("types") or {}).get(lt)
        if rec is not None and entry is None:
            problems.append(f"{lt}: not in {SHADOW_FILE}")
        elif entry is not None:
            for name, digest in entry.get("sources", {}).items():
                p = os.path.join(rec["src"], lt, name)
                if not os.path.exists(p) or sha256_file(p) != digest:
                    problems.append(f"{lt}: source {p} changed or vanished since the shadow was made")
        absent = [n for n in SHADOW_COPY if not os.path.isfile(os.path.join(d, n))]
        if absent:
            problems.append(f"{lt}: {d} has no {absent}")
            continue
        split = pd.read_csv(os.path.join(d, "split.csv"), dtype=str, keep_default_na=False)
        need = set(BASE_COLUMNS)
        for arm in arms:
            if arm != "none":
                need |= set(cm.FRAME_COLUMNS[arm])
        lacking = sorted(need - set(split.columns))
        if lacking:
            problems.append(f"{lt}: split.csv lacks {lacking}, which arms {list(arms)} read")
        shares = missing_shares(split)
        out(f"{lt}: {len(split):,} rows; missing share " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
        if "crop_path" in split.columns and len(split):
            sample = split.sample(min(20, len(split)), random_state=0)["crop_path"]
            bad = [p for p in sample if not os.path.exists(p if os.path.isabs(p) else os.path.join(d, p))]
            if bad:
                problems.append(f"{lt}: {len(bad)} of {len(sample)} sampled crops do not resolve under {d} "
                                f"(e.g. {bad[:2]})")
        if pair_october:
            for n in OCTOBER_FILES:
                if not os.path.isfile(os.path.join(d, n)):
                    problems.append(f"{lt}: October's {n} is missing from the shadow (the drift row reads it)")
            p = os.path.join(d, "predictions.csv")
            if os.path.isfile(p) and not (pd.read_csv(p, usecols=["split"])["split"] == "val").any():
                problems.append(f"{lt}: October's predictions.csv has no val rows")
        manifest = pd.read_csv(os.path.join(d, "manifest.csv"), dtype=str, keep_default_na=False)
        ok = manifest[manifest["status"] == "ok"] if "status" in manifest.columns else manifest
        if skip_backfill:
            index = os.path.join(d, "context", "index.csv")
            have = set(pd.read_csv(index, dtype=str)["label_uid"]) if os.path.exists(index) else set()
            short = int((~ok["label_uid"].isin(have)).sum())
            if short:
                problems.append(f"{lt}: SKIP_BACKFILL but {index} lacks {short:,} of {len(ok):,} ok manifest rows")
        if backfill:
            n_files = 2 * len(ok)
            out(f"{lt}: the backfill writes {n_files:,} files, about {len(ok) * CONTEXT_KB_PER_LABEL / 1e6:.1f} GB, "
                f"under {os.path.join(d, 'context')}")
            marked = bool(pano_store) and os.path.isfile(os.path.join(pano_store, ".pano-store"))
            if not marked:
                problems.append(f"{lt}: pano store {pano_store} has no .pano-store marker")
            if pano_store:
                listing, gone, total = write_stage_files(d, ok, pano_store)
                if gone:
                    problems.append(
                        f"{lt}: {len(gone):,} of {total:,} panoramas are not in {pano_store} (about "
                        f"{len(gone) * PANO_MB / 1e3:.0f} GB at {PANO_MB} MB each); list in {listing}. Restage (Jon's "
                        "call; makelab2): one deduplicated list for every type, `analysis/context_arms.py stage-list "
                        f"--data {data} --types <types> --pano-store {pano_store} --out <dir>`, then `RUN=<dir> "
                        + (f"DEST=<a new store, not {pano_store}: the stage job removes .pano-store at its start>"
                           if marked else f"DEST={pano_store}")
                        + " sbatch analysis/ai_labels_stage.slurm` (docs/context-arms-2026-11.md)")
            raw = (rec or {}).get("rawlabels")
            if not raw or not os.path.isdir(raw):
                problems.append(f"{lt}: the shadow's rawLabels snapshot {raw} is not a directory")
            else:
                cities = sorted(set(ok["city"]))
                no_csv = [c for c in cities if not os.path.isfile(os.path.join(raw, f"{c}.csv"))]
                if no_csv:
                    problems.append(f"{lt}: {raw} has no <city>.csv for {len(no_csv)} manifest cities")
    versions = installed_versions() if versions is None else versions
    pandas_v = versions.get("pandas")
    if not pandas_v or pandas_v.split(".")[0] != "2":
        problems.append(f"pandas {pandas_v} in the job env; the klone env must keep pandas 2.x")
    pin = transformers_pin if transformers_pin is not None else environment_pin("transformers")
    if versions.get("transformers") != pin:
        problems.append(f"transformers {versions.get('transformers')} in the job env, environment.yml pins {pin}")
    if not hf_home or not glob.glob(os.path.join(hf_home, BACKBONE_CACHE)):
        problems.append(f"facebook/dinov2-large is not in the HF cache {hf_home} (jobs run with HF_HUB_OFFLINE=1)")
    if write_test:
        for directory in (runs, data):
            err = write_probe(directory)
            if err:
                problems.append(f"a real {WRITE_TEST_BYTES >> 20} MiB write to {directory} failed: {err}")
    if keep:
        probe = keep if os.path.isdir(keep) else os.path.dirname(os.path.abspath(keep))
        try:
            st = os.statvfs(probe)
            out(f"KEEP {keep}: {st.f_bavail * st.f_frsize / 1e9:.0f} GB and {st.f_favail:,} files free "
                "(filesystem view; group quotas may be tighter)")
        except OSError as e:
            out(f"KEEP {keep}: statvfs failed ({e})")
        if shutil.which("hyakstorage"):
            import subprocess
            try:
                r = subprocess.run(["hyakstorage", probe], capture_output=True, text=True, timeout=60)
                out(r.stdout.strip())
            except (OSError, subprocess.SubprocessError) as e:
                out(f"hyakstorage failed ({e}); skipped")
        else:
            out("hyakstorage not installed here; quota not checked")
    for p in problems:
        out(f"PREFLIGHT FAILED: {p}")
    out(f"preflight {stage or ''}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return problems


def write_stage_files(type_dir, ok, pano_store):
    """Write ``<type_dir>/stage_files.txt`` (``adjudicator.stage_files`` format, ``<city>/<pano_id[:2]>/<pano_id>.jpg``)
    listing the panoramas of the manifest rows ``ok`` that are not in ``pano_store``; a store that does not exist yet
    lacks all of them. Returns ``(path, missing (city, pano_id) pairs, unique panoramas)``."""
    import data_generation as dg
    pairs = sorted(set(zip(ok["city"], ok["pano_id"])))
    gone = [(c, p) for c, p in pairs if not os.path.exists(dg.pano_store_path(pano_store, c, p))]
    listing = os.path.join(type_dir, "stage_files.txt")
    with open(listing, "w") as f:
        f.writelines(f"{c}/{p[:2]}/{p}.jpg\n" for c, p in gone)
    return listing, gone, len(pairs)


def stage_list(data, types, pano_store, out_dir, out=print):
    """The restage list of a stage: per type, ``write_stage_files`` into ``<data>/<type>/``; then one deduplicated
    ``<out_dir>/stage_files.txt`` (``out_dir`` created) for a single ``analysis/ai_labels_stage.slurm`` job with
    ``RUN=<out_dir>``, so the makelab hosts see one sequential pull, not one per type. No GPU, no network; sizes the
    restage before any launcher run. Returns ``{type: missing}`` and the combined count under ``"_all"``.

    Example::

        python analysis/context_arms.py stage-list --data $WORK/data --types NoCurbRamp,Obstacle,SurfaceProblem \
            --pano-store /gscratch/scrubbed/$USER/pano-stage/context-arms --out $WORK/stage
    """
    counts, everything = {}, set()
    for lt in types:
        d = os.path.join(data, lt)
        manifest = pd.read_csv(os.path.join(d, "manifest.csv"), dtype=str, keep_default_na=False)
        ok = manifest[manifest["status"] == "ok"] if "status" in manifest.columns else manifest
        listing, gone, total = write_stage_files(d, ok, pano_store)
        counts[lt] = len(gone)
        everything.update(gone)
        out(f"{lt}: {len(gone):,} of {total:,} panoramas not in {pano_store}; {listing}")
    os.makedirs(out_dir, exist_ok=True)
    combined = os.path.join(out_dir, "stage_files.txt")
    with open(combined, "w") as f:
        f.writelines(f"{c}/{p[:2]}/{p}.jpg\n" for c, p in sorted(everything))
    counts["_all"] = len(everything)
    out(f"all types, deduplicated: {len(everything):,} panoramas, about {len(everything) * PANO_MB / 1e3:.0f} GB at "
        f"{PANO_MB} MB each; {combined}")
    return counts


def run_record(data_dir, label_type, tag):
    """What a run's durable record says: ``{"status": "ok" | "missing" | "incomplete", "seed", "split_sha256",
    "context_arm", "context_view_res", "epochs", "best_metric", "target", "batch_size", "files_sha256", "path"}`` (the
    settings from ``run_key.json``; a tight-crop run records no arm or resolution, a default run no target or batch). Read from ``<data>/<type>/runs/<tag>/`` (this chain's layout) or, failing that,
    ``training_run_<tag>.json`` beside the metrics (the backbone stream's layout)."""
    d = os.path.join(data_dir, label_type, "runs", tag)
    flat = os.path.join(data_dir, label_type, f"training_run_{tag}.json")
    if os.path.isdir(d):
        files = {n: sha256_file(os.path.join(d, n)) for n in RECORD_FILES if os.path.isfile(os.path.join(d, n))}
        key = {}
        if "run_key.json" in files:
            with open(os.path.join(d, "run_key.json")) as f:
                key = json.load(f)
        status = "ok" if {"training_run.json", "run_key.json"} <= set(files) else "incomplete"
        return {"status": status, "path": os.path.relpath(d, data_dir), **_record_keys(key), "files_sha256": files}
    if os.path.isfile(flat):
        with open(flat) as f:
            run = json.load(f)
        return {"status": "ok", "path": os.path.relpath(flat, data_dir), **_record_keys(run),
                "files_sha256": {os.path.basename(flat): sha256_file(flat)}}
    return {"status": "missing", "path": None, **_record_keys({}), "files_sha256": {}}


def _record_keys(key):
    return {k: key.get(k) for k in ("seed", "split_sha256", "context_arm", "context_view_res", "epochs",
                                    *RECIPE_SETTINGS)}


def check_run(data_dir, label_type, tag, seed=None, epochs=None):
    """The checks a finished training job must pass (the smoke stage's criterion): ``predictions_<tag>.csv`` holds
    exactly the split's ``EVAL_SPLITS`` rows, one per ``label_uid``, each in its own split; ``metrics_<tag>.json``
    and ``report_<tag>.txt`` exist; the run record is complete, with this ``seed`` (when given), the hash of this
    ``split.csv``, the context arm and view resolution the tag stands for (``expected_settings``), ``epochs``
    (when given) and the knobs the job file pins (``recipe_problems``). Returns the list of problems (empty: pass)."""
    d = os.path.join(data_dir, label_type)
    problems = []
    split = pd.read_csv(os.path.join(d, "split.csv"), dtype=str, keep_default_na=False)
    pred_path = os.path.join(d, f"predictions_{tag}.csv")
    if not os.path.exists(pred_path):
        return [f"{label_type}/{tag}: no predictions_{tag}.csv"]
    pred = pd.read_csv(pred_path, dtype={"label_uid": str, "split": str})
    if not pred["label_uid"].is_unique:
        problems.append(f"{label_type}/{tag}: {int(pred['label_uid'].duplicated().sum())} duplicate label_uids")
    want = split[split["split"].isin(EVAL_SPLITS)]
    if set(pred["label_uid"]) != set(want["label_uid"]) or len(pred) != len(want):
        problems.append(f"{label_type}/{tag}: {len(pred):,} predictions for {len(want):,} val/test/held-out rows "
                        f"({len(set(want['label_uid']) - set(pred['label_uid'])):,} missing, "
                        f"{len(set(pred['label_uid']) - set(want['label_uid'])):,} extra)")
    m = pred.merge(want[["label_uid", "split"]], on="label_uid", how="inner", suffixes=("", "_split"),
                   validate="one_to_one") if pred["label_uid"].is_unique else None
    if m is not None and (m["split"] != m["split_split"]).any():
        problems.append(f"{label_type}/{tag}: {int((m['split'] != m['split_split']).sum())} rows in the wrong split")
    for n in (f"metrics_{tag}.json", f"report_{tag}.txt"):
        if not os.path.exists(os.path.join(d, n)):
            problems.append(f"{label_type}/{tag}: no {n}")
    rec = run_record(data_dir, label_type, tag)
    if rec["status"] != "ok":
        problems.append(f"{label_type}/{tag}: run record {rec['status']}")
    else:
        if seed is not None and rec["seed"] != seed:
            problems.append(f"{label_type}/{tag}: run record seed {rec['seed']}, expected {seed}")
        if rec["split_sha256"] != sha256_file(os.path.join(d, "split.csv")):
            problems.append(f"{label_type}/{tag}: run record split hash differs from this split.csv")
        variant = variant_of_tag(tag)
        for k, v in (expected_settings(variant) or {}).items():
            if rec.get(k) != v:
                problems.append(f"{label_type}/{tag}: run record {k} {rec.get(k)}, {variant} needs {v}")
        if epochs is not None and rec.get("epochs") != epochs:
            problems.append(f"{label_type}/{tag}: run record epochs {rec.get('epochs')}, expected {epochs}")
        problems.extend(f"{label_type}/{tag}: run {p}" for p in recipe_problems(rec))
    return problems


def cmd_shadow(args):
    shadow(args.src, args.dest, _list(args.types), args.only_staged)
    return 0


def cmd_preflight(args):
    seeds = {k: ([int(x) for x in _list(v)] if v else None)
             for k, v in (("c", args.confirmatory_seeds), ("b", args.baseline_seeds))}
    problems = preflight(args.data, args.runs, _list(args.types), _list(args.arms), args.stage, args.keep,
                         args.pano_store, args.backfill, args.skip_backfill, args.pair_october, args.hf_home,
                         write_test=not args.no_write_test, confirmatory_seeds=seeds["c"], baseline_seeds=seeds["b"])
    return 1 if problems else 0


def cmd_stage_list(args):
    stage_list(args.data, _list(args.types), args.pano_store, args.out)
    return 0


def cmd_check_run(args):
    problems = []
    for tag in _list(args.tags):
        p = check_run(args.data, args.label_type, tag, seed_of_tag(tag) if args.seed_from_tag else None, args.epochs)
        print(f"{args.label_type}/{tag}: " + ("ok" if not p else "; ".join(p)))
        problems += p
    return 1 if problems else 0


def _list(text):
    return [t.strip() for t in str(text).replace(" ", ",").split(",") if t.strip()]


def variant_of_tag(tag):
    """The variant a tag stands for, without its ``-s<seed>``.

    >>> variant_of_tag("view_heatmap-r448-s43"), variant_of_tag("baseline-rerun")
    ('view_heatmap-r448', 'baseline-rerun')
    """
    m = cl.SEED_RUN_RE.match(f"metrics_{tag}.json")
    return m["variant"] if m else tag


def seed_of_tag(tag, base_seed=42):
    """The training seed a tag stands for: ``<variant>-s<seed>``, else ``base_seed``.

    >>> seed_of_tag("view_heatmap-r448-s43"), seed_of_tag("baseline-rerun")
    (43, 42)
    """
    m = cl.SEED_RUN_RE.match(f"metrics_{tag}.json")
    return int(m["seed"]) if m else base_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shadow", help="a shadow data directory of the round-3 dataset on scratch (see shadow())")
    s.add_argument("--src", required=True, help="the round-3 data directory (read only)")
    s.add_argument("--dest", required=True, help="the shadow to create or extend (not inside --src)")
    s.add_argument("--types", required=True, help="comma list of label types")
    s.add_argument("--only-staged", default=None,
                   help="a pano store: keep only rows whose panorama is in it (smoke tests; never decidable)")
    s.set_defaults(func=cmd_shadow)
    f = sub.add_parser("preflight", help="check a stage before it takes a GPU (login node; see preflight())")
    f.add_argument("--data", required=True, help="the shadow data directory the jobs use")
    f.add_argument("--runs", required=True, help="the checkpoint root the jobs write")
    f.add_argument("--types", required=True, help="comma list of label types")
    f.add_argument("--arms", required=True, help="comma list of the stage's arms (none = the baseline rerun)")
    f.add_argument("--stage", default="", help="the stage name, printed only (smoke, pilot, A, B)")
    f.add_argument("--keep", default=None, help="the durable KEEP directory: its free space is printed")
    f.add_argument("--pano-store", default=None,
                   help="with --backfill: the store the backfill reads; missing panoramas go to stage_files.txt")
    f.add_argument("--confirmatory-seeds", default=None,
                   help="the seeds the stage trains a confirmatory arm at; must equal DECISION_RULE's")
    f.add_argument("--baseline-seeds", default=None,
                   help="the seeds the stage trains baseline-rerun at; must equal DECISION_RULE's")
    f.add_argument("--backfill", action="store_true", help="the stage submits a backfill")
    f.add_argument("--skip-backfill", action="store_true", help="image arms run on an existing context index")
    f.add_argument("--pair-october", action="store_true", help="require October's baseline files (drift row)")
    f.add_argument("--hf-home", default=os.environ.get("HF_HOME"), help="the HF cache the jobs read (default $HF_HOME)")
    f.add_argument("--no-write-test", action="store_true", help="skip the 8 MiB write probes (tests only)")
    f.set_defaults(func=cmd_preflight)
    g = sub.add_parser("stage-list", help="the deduplicated restage list of a stage's types (see stage_list())")
    g.add_argument("--data", required=True, help="the shadow data directory")
    g.add_argument("--types", required=True, help="comma list of label types")
    g.add_argument("--pano-store", required=True, help="the store to check; one not staged yet lacks every panorama")
    g.add_argument("--out", required=True, help="directory for the combined stage_files.txt (RUN= of the stage job)")
    g.set_defaults(func=cmd_stage_list)
    w = sub.add_parser("power", help="the rule's operating characteristics from a variant's seed spread")
    w.add_argument("--data-dir", required=True)
    w.add_argument("--label-types", default=",".join(TARGET_TYPES))
    w.add_argument("--variant", default=OCTOBER, help="baseline (October's trio) or baseline-rerun (the pilot's)")
    w.add_argument("--n-boot", type=int, default=1000)
    w.add_argument("--n-sim", type=int, default=4000)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--within-min-n", type=int, default=10)
    w.add_argument("--out", default=None)
    w.set_defaults(func=cmd_power)
    c = sub.add_parser("check-run", help="did a finished training job write everything (smoke criterion)")
    c.add_argument("--data", required=True)
    c.add_argument("--label-type", required=True)
    c.add_argument("--tags", required=True, help="comma list of evaluate.py tags")
    c.add_argument("--seed-from-tag", action="store_true", help="also check the record's seed (-s<seed>, else 42)")
    c.add_argument("--epochs", type=int, default=None,
                   help=f"also check the record's epochs (the recipe's {RECIPE_EPOCHS}; the smoke stage runs 1)")
    c.set_defaults(func=cmd_check_run)
    r = sub.add_parser("report", help="decision.md/json, gate.md, slices_seeds.md, arms.md/json, seeds.md, "
                                      "slices.md, clipping.md, provenance.json")
    r.add_argument("--data-dir", required=True,
                   help="the shadow data directory the jobs wrote ($WORK/data; analysis/context_arms.py shadow)")
    r.add_argument("--label-types", default=",".join(TYPES))
    r.add_argument("--arms", default=",".join(ARMS),
                   help="evaluate.py tags to compare with the baseline (e.g. add view_heatmap-r448)")
    r.add_argument("--public-cities", default=None,
                   help="a saved /v3/api/cities response or a file of public city_ids; default: the live API. Only "
                        "public deployments are named")
    r.add_argument("--within-min-n", type=int, default=10)
    r.add_argument("--slice-min-n", type=int, default=None)
    r.add_argument("--n-boot", type=int, default=None, help="panorama resamples (slices.py default)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--base-seed", type=int, default=42)
    r.add_argument("--baseline-variant", default=DECISION_RULE["baseline_variant"],
                   help="the run the decision compares against (October's trio is always the drift row)")
    r.add_argument("--shares", default=SHARES, help="prior_rules.py shares.json (the majority2 deployment prior)")
    r.add_argument("--decision-boot", type=int, default=1000,
                   help=f"joint panorama resamples for the decision's intervals (at least {MIN_DECISION_BOOT})")
    r.add_argument("--out", required=True)
    r.set_defaults(func=report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
