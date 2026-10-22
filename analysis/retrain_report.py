"""Tables and figures for ``docs/retrain-2026-10.md``, the round-3 retrain report.

Reads, for each label type, the ``evaluate.py`` outputs of one training run (``predictions.csv``, ``metrics.json``)
next to its ``split.csv`` (for the city, panorama and class of every row), and writes to ``--out-dir``:

* ``headline.csv``: per type and split (``test`` = the training cities' test share, ``test_heldout`` = cities the
  model never saw): n, AUC with a 95% interval from resampling panoramas, accuracy at 0.5, the within-city AUC and
  the test minus held-out AUC difference with its interval.
* ``per_city.csv``: per type, split and deployment: n, AUC with a panorama-resampled interval (blank below
  ``--min-city-n`` rows). Public deployments are named by ``city_id``; private test deployments are pooled into one
  row; private held-out deployments are called "private city A (outside the US)", "private city B (US)", and so on.
* ``class_share.csv``: the incorrect share per deployment in ``train`` and in the held-out split, the city-class
  confound the report discusses.
* ``recall_at_precision.csv`` and ``hard.csv``: straight from ``metrics.json``.
* ``production_comparison.csv``: round 3 against the production validators on the 1,988-crop held-out sample
  (``--sample`` and ``--production-scores``, from ``analysis/vlm_baseline.py baseline``): paired bootstraps by
  label (within class) and by panorama, and within-city AUC for both.
* ``auc_by_type.png`` and ``heldout_per_city.png``.
* ``tables.md``: every table above as Markdown, for the report.

Naming follows ``analysis/vlm_baseline.py``: a deployment is named only if ``--cities-json`` (a saved
``/v3/api/cities`` response) lists it as public. Intervals are 95% percentile intervals from ``--n-boot`` resamples
of whole panoramas (a label's crop and its neighbours on the same panorama are not independent), seed ``--seed``.
Within-city AUC is the per-city AUC averaged with weights n_incorrect x n_correct, over cities with both classes and
at least ``--within-min-n`` rows; it cannot be raised by a score that only separates incorrect-heavy cities from
correct-heavy ones.

Usage::

    python analysis/retrain_report.py --run-dir ~/Data/sidewalk-validator-ai-retrain-2026-10/round3 \\
        --cities-json ~/Data/sidewalk-validator-ai-prod-2026-10-17/cities.json \\
        --sample ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/sample.csv \\
        --production-scores ~/Data/sidewalk-validator-ai-vlm-2026-10/scores/production.csv \\
        --out-dir docs/retrain-2026-10

Example (the within-city AUC)::

    >>> y = [0, 1] * 10 + [1] * 10; s = list(range(20)) + [100] * 10; c = ["a"] * 20 + ["b"] * 10
    >>> within_city_auc(y, s, c)   # city b has one class, so only city a counts
    (0.55, 1)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]
SPLITS = ["test", "test_heldout"]
PRECISION_LEVELS = ["0.92", "0.95"]


def load_cities(path):
    """``{city_id: (visibility, country_id)}`` from a saved ``/v3/api/cities`` response."""
    with open(path) as f:
        d = json.load(f)
    return {c["city_id"]: (c.get("visibility"), c.get("country_id")) for c in d["cities"]}


def is_public(city, cities):
    return cities.get(city, (None, None))[0] == "public"


def heldout_names(heldout_cities, cities):
    """Display names for the held-out deployments: public ones by ``city_id``; private ones lettered, those outside
    the US first, so that with one of each they are "private city A (outside the US)" and "private city B (US)".

    >>> heldout_names(["x", "y", "z"], {"x": ("public", "usa"), "y": ("private", "usa"), "z": ("private", "ch")})
    {'x': 'x', 'z': 'private city A (outside the US)', 'y': 'private city B (US)'}
    """
    names = {c: c for c in heldout_cities if is_public(c, cities)}
    private = [c for c in heldout_cities if not is_public(c, cities)]
    private.sort(key=lambda c: (cities.get(c, (None, "usa"))[1] == "usa", c))
    for i, c in enumerate(private):
        where = "US" if cities.get(c, (None, "usa"))[1] == "usa" else "outside the US"
        names[c] = f"private city {chr(ord('A') + i)} ({where})"
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
    return df


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
    """AUC of each score column in ``scores`` (an array ``n x k``) on resamples of whole ``groups`` (panoramas).
    Returns the ``n_boot x k`` array of resampled AUCs; draws with one class are redrawn."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    groups = np.asarray(groups)
    names, inverse = np.unique(groups, return_inverse=True)
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


def headline_rows(df, label_type, args):
    rows, boots = [], {}
    for split in SPLITS:
        g = df[df.split == split]
        b = cluster_boot(g.y, g.p_incorrect, g.pano_id, args.n_boot, args.seed)[:, 0]
        boots[split] = b
        lo, hi = ci(b)
        wc, nc = within_city_auc(g.y, g.p_incorrect, g.city, args.within_min_n)
        pred = (g.p_incorrect >= 0.5).astype(int)
        tpr = float((pred[g.y == 1] == 1).mean())
        tnr = float((pred[g.y == 0] == 0).mean())
        rows.append({"label_type": label_type, "split": split, "n": len(g),
                     "n_correct": int((g.y == 0).sum()), "n_incorrect": int(g.y.sum()),
                     "n_panos": g.pano_id.nunique(), "n_cities": g.city.nunique(),
                     "auc": roc_auc_score(g.y, g.p_incorrect), "auc_lo": lo, "auc_hi": hi,
                     "accuracy": float((pred == g.y).mean()), "balanced_accuracy": (tpr + tnr) / 2,
                     "within_city_auc": wc, "within_city_n_cities": nc})
    d = boots["test"] - boots["test_heldout"]
    dlo, dhi = ci(d)
    for r in rows:
        r["diff_test_minus_heldout"] = rows[0]["auc"] - rows[1]["auc"]
        r["diff_lo"], r["diff_hi"] = dlo, dhi
    return rows


def per_city_rows(df, label_type, cities, names, args):
    rows = []
    for split in SPLITS:
        g = df[df.split == split]
        if split == "test":
            disp = g.city.where(g.city.map(lambda c: is_public(c, cities)), "other (private deployments)")
        else:
            disp = g.city.map(names)
        g = g.assign(display=disp.values)
        for city, h in g.groupby("display"):
            public = city != "other (private deployments)" and not city.startswith("private city")
            n_private = h.city.nunique() if city == "other (private deployments)" else 0
            if public and split == "test" and len(h) < args.min_city_n:
                continue
            row = {"label_type": label_type, "split": split, "city": city, "n": len(h),
                   "n_correct": int((h.y == 0).sum()), "n_incorrect": int(h.y.sum()),
                   "n_panos": h.pano_id.nunique(), "n_deployments": n_private or 1,
                   "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                   "accuracy": float(((h.p_incorrect >= 0.5).astype(int) == h.y).mean())}
            if h.y.nunique() == 2:
                row["auc"] = roc_auc_score(h.y, h.p_incorrect)
                if len(h) >= args.min_city_n:
                    row["auc_lo"], row["auc_hi"] = ci(cluster_boot(h.y, h.p_incorrect, h.pano_id,
                                                                   args.n_boot, args.seed)[:, 0])
            rows.append(row)
    return rows



def class_share_from_split(run_dir, label_type, cities, names, min_n):
    split = pd.read_csv(f"{run_dir}/{label_type}/split.csv", usecols=["city", "label_class", "split"])
    rows = []
    for sp in ["train", "test_heldout"]:
        g = split[split.split == sp]
        if sp == "train":
            disp = g.city.where(g.city.map(lambda c: is_public(c, cities)), "other (private deployments)")
        else:
            disp = g.city.map(names)
        g = g.assign(display=disp.values)
        for city, h in g.groupby("display"):
            if sp == "train" and city != "other (private deployments)" and len(h) < min_n:
                continue
            rows.append({"label_type": label_type, "split": sp, "city": city, "n": len(h),
                         "share_of_split": len(h) / len(g),
                         "incorrect_share": float((h.label_class == "incorrect").mean())})
    return rows


def metrics_rows(run_dir, label_type):
    with open(f"{run_dir}/{label_type}/metrics.json") as f:
        m = json.load(f)
    rp, hard = [], []
    for split in SPLITS:
        for cls in ["correct", "incorrect"]:
            for lvl in PRECISION_LEVELS:
                r = m[split]["recall_at_precision"][cls][lvl]
                rp.append({"label_type": label_type, "split": split, "class": cls, "precision": float(lvl),
                           "recall": r["recall"], "threshold": r["threshold"]})
    h = m.get("hard")
    if h:
        hard.append({"label_type": label_type, "n": h["n"], "n_correct": h["n_by_class"]["correct"],
                     "n_incorrect": h["n_by_class"]["incorrect"], "auc": h.get("auc"),
                     "accuracy": h["accuracy"], "balanced_accuracy": h["balanced_accuracy"]})
    return rp, hard


def production_rows(preds, sample_path, scores_path, cities, args):
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
        rows.append({"label_type": t, "split": s, "n": len(g), "n_panos": g.pano_id.nunique(),
                     "n_deployments": g.city.nunique(),
                     "auc_round3": roc_auc_score(g.y, g.r3), "auc_production": roc_auc_score(g.y, g.p_prod),
                     "diff": roc_auc_score(g.y, g.r3) - roc_auc_score(g.y, g.p_prod),
                     "diff_label_lo": ci(lb[:, 0] - lb[:, 1])[0], "diff_label_hi": ci(lb[:, 0] - lb[:, 1])[1],
                     "diff_pano_lo": ci(pb[:, 0] - pb[:, 1])[0], "diff_pano_hi": ci(pb[:, 0] - pb[:, 1])[1],
                     "within_city_round3": w3, "within_city_production": wp, "within_city_n_cities": nc})
    return rows


def fmt_ci(lo, hi, digits=3):
    if pd.isna(lo):
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def md(df, cols, floatfmt="{:.3f}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if df[c].dtype.kind in "fi" else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
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
        ax.errorbar(x + off, auc, yerr=[np.subtract(auc, lo), np.subtract(hi, auc)], fmt="o", color=color,
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
        yerr = [np.nan_to_num(q.auc - q.auc_lo), np.nan_to_num(q.auc_hi - q.auc)]
        ax.errorbar(x + off, q.auc, yerr=yerr, fmt="o", capsize=3, label=c)
    ax.set_xticks(x, TYPES)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (incorrect)")
    ax.set_title("Round 3 on the held-out deployments, per type (interval blank below the minimum n)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/heldout_per_city.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cities-json", required=True)
    ap.add_argument("--sample")
    ap.add_argument("--production-scores")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--min-city-n", type=int, default=50)
    ap.add_argument("--within-min-n", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cities = load_cities(args.cities_json)

    preds = {t: load_type(args.run_dir, t) for t in TYPES}
    heldout = sorted(set().union(*[set(p[p.split == "test_heldout"].city) for p in preds.values()]))
    names = heldout_names(heldout, cities)

    headline, per_city, share, rp, hard = [], [], [], [], []
    for t in TYPES:
        headline += headline_rows(preds[t], t, args)
        per_city += per_city_rows(preds[t], t, cities, names, args)
        share += class_share_from_split(args.run_dir, t, cities, names, args.min_city_n)
        a, b = metrics_rows(args.run_dir, t)
        rp += a
        hard += b
    headline, per_city, share = pd.DataFrame(headline), pd.DataFrame(per_city), pd.DataFrame(share)
    rp, hard = pd.DataFrame(rp), pd.DataFrame(hard)
    for name, df in [("headline", headline), ("per_city", per_city), ("class_share", share),
                     ("recall_at_precision", rp), ("hard", hard)]:
        df.to_csv(f"{args.out_dir}/{name}.csv", index=False)

    prod = None
    if args.sample and args.production_scores:
        prod = pd.DataFrame(production_rows(preds, args.sample, args.production_scores, cities, args))
        prod.to_csv(f"{args.out_dir}/production_comparison.csv", index=False)

    figures(headline, per_city, args.out_dir)

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
    out += ["## hard", "", md(hard, ["label_type", "n", "n_correct", "n_incorrect", "auc", "accuracy", "balanced_accuracy"]), ""]
    if prod is not None:
        q = prod.assign(**{"Δ [label CI]": [f"{r.diff:+.3f} {fmt_ci(r.diff_label_lo, r.diff_label_hi)}" for r in prod.itertuples()],
                           "Δ [pano CI]": [f"{r.diff:+.3f} {fmt_ci(r.diff_pano_lo, r.diff_pano_hi)}" for r in prod.itertuples()]})
        out += ["## production comparison", "",
                md(q, ["label_type", "split", "n", "n_panos", "n_deployments", "auc_round3", "auc_production",
                       "Δ [label CI]", "Δ [pano CI]", "within_city_round3", "within_city_production", "within_city_n_cities"]), ""]
    with open(f"{args.out_dir}/tables.md", "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
