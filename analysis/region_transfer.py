"""Cross-region transfer: leave-one-country-out (LOCO) variants of the round-3 splits, local fine-tuning, a
region-conditioned validator and per-country calibration (issues #33, #52 option 3 and #15).

Why
---
Round 3 holds out three deployments, only one of them outside the US (``docs/retrain-2026-10.md``). That is one data
point for the question a new deployment asks: does one global validator transfer to a country it never saw, and if
not, what helps most? This module builds the variants and reads the runs for four arms, each scored on the same rows:

(a) ``global``: the round-3 recipe trained without one country (its rows held out).
(b) ``ft``: the global model fine-tuned on a nested fraction of that country's own former train rows (the "new
    deployment with some local labels" curve).
(c) ``cond``: a region-conditioned model (``region_model.py``: a learned country embedding in the head, the unknown
    row trained by region dropout) without the country; the held-out country is scored as unknown.
(d) per-country isotonic calibration of the global model, no retraining (``calibrate``).

Plus ``condall``: the region-conditioned model on the round-3 split itself, against the round-3 baseline, to see
whether telling the model the country helps in distribution.

Stages
------
- ``variants`` (CPU, seconds): per country, ``<out>/<country>/split.csv``: that country's ``train`` rows become
  ``pool``, its ``val`` and ``test`` rows ``test_heldout_country``; every other row is byte-identical (round-3
  ``test_heldout`` deployments stay ``test_heldout``). Then ``<out>/<country>/ft-<pct>/split.csv`` for each fraction:
  nested, panorama-grouped (3 m merge, as the split) subsets of the pool as ``train``, a fixed local ``val``, and
  every other country's ``train``/``val`` as ``unused``. ``variants.json`` records counts, eligibility and hashes.
- ``list``: the eligible countries and fractions of a ``variants.json`` (one line each), for the launcher.
- ``predict`` (GPU): score a variant split with a plain or a region-conditioned checkpoint.
- ``calibrate`` (CPU): arm (d), pooled vs local maps on the held-out country, and per-country maps in distribution.
- ``report`` (CPU, on the Mac from a pulled copy): tables per label type into ``docs/region-transfer-2026-11/`` and
  the generated blocks of ``docs/region-transfer-2026-11.md``. ``report --check`` renders the blocks from the
  committed ``results.json`` and fails on drift (it passes while the doc says "not yet run").

Naming: a country is named only if one of its deployments is public; the others are "private country A", "B", ...
(``regions.display_names``). Every committed string is checked against the non-public term set (``redact``).

Layout (``$DATA`` = the round-3 data directory; nothing under ``$DATA/<LT>/`` outside ``region/`` is written)::

    $DATA/<LT>/region/variants.json, <country>/split.csv, <country>/ft-<pct>/split.csv, calibration.json
    $DATA/<LT>/region/eval/<LT>/split.csv -> $DATA/<LT>/split.csv (link), predictions_<tag>.csv, metrics_<tag>.json

Usage::

    python analysis/region_transfer.py variants --split $DATA/CurbRamp/split.csv --cities-json $DATA/rawlabels/cities.json \\
        --rawlabels $DATA/rawlabels --out $DATA/CurbRamp/region
    python analysis/region_transfer.py predict --split-file $DATA/CurbRamp/region/mexico/split.csv --data-dir $DATA \\
        --label-type CurbRamp --run-dir $RUNS/CurbRamp/global-mexico-s42 \\
        --out $DATA/CurbRamp/region/eval/CurbRamp/predictions_global-mexico-s42.csv
    python analysis/region_transfer.py calibrate --data-dir $DATA --label-type CurbRamp \\
        --priors $DATA/deployment_priors.json --cities-json $DATA/rawlabels/cities.json
    python analysis/region_transfer.py report --data-dir $DATA --cities-json $DATA/rawlabels/cities.json \\
        --out docs/region-transfer-2026-11 --doc docs/region-transfer-2026-11.md
    python analysis/region_transfer.py report --check --out docs/region-transfer-2026-11 --doc docs/region-transfer-2026-11.md
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confident_learning as cl  # noqa: E402
import regions  # noqa: E402
import split_data  # noqa: E402

CLASSES = ("correct", "incorrect")
HELDOUT_COUNTRY = "test_heldout_country"  # the held-out country's former val + test rows: every arm is scored here
POOL = "pool"                             # its former train rows: scored zero-shot, the fine-tune source
UNUSED = "unused"                         # other countries' train/val rows in a fine-tune split
ARMS = ("global", "ft", "cond", "condall")
DEFAULT_FRACTIONS = (0.10, 0.25, 0.50, 1.0)
SCORED_SPLITS = ("val", "test", "test_heldout", HELDOUT_COUNTRY, POOL)
FT_SCORED_SPLITS = ("test", "test_heldout", HELDOUT_COUNTRY)

# Minimum-n rules (repeated in the doc).
MIN_EVAL_CLASS = 30   # a LOCO cell exists at >= 30 held-out-country eval rows of each class
MIN_POOL = 100        # fine-tune fractions need a pool of at least this many rows
MIN_FRAC_CLASS = 20   # a fraction point needs >= 20 train rows of each class
MAX_TRAIN_SHARE = 0.5  # a country holding more of the train rows is not held out (most of the data would go)
WITHIN_MIN_N = 10     # within-city AUC: deployments with >= 10 labels of both classes
MIN_GATED = 50        # a gate precision is shown at >= 50 gated votes (calibration/per_city.py)
MIN_JUDGE_N = 100     # a country is judged only with n_eval >= 100 and a paired interval
MIN_VAL_OWN_MAP = 50  # in distribution: a per-country map needs >= 50 val rows
MIN_COUNTRY_CLASS = 30  # condall per-country Δ: >= 30 test rows of each class
GATE = 0.92

MARK_BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/region_transfer.py) -->"
MARK_END = "<!-- END GENERATED: {name} -->"
BLOCKS = ("status", "summary", "loco_table", "results_loco", "results_curves", "results_conditioned",
          "results_calibration", "provenance")
NOT_RUN = "_Not yet run: the report stage fills this block._"


# ---------------------------------------------------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------------------------------------------------

def pct_of(fraction):
    """``int(round(100 * fraction))``: the fraction's name in paths and tags.

    >>> pct_of(0.1), pct_of(0.25), pct_of(1.0)
    (10, 25, 100)
    """
    return int(round(100 * float(fraction)))


def tag_global(country, seed):
    return f"global-{country}-s{seed}"


def tag_cond(country, seed):
    return f"cond-{country}-s{seed}"


def tag_ft(country, pct, seed):
    return f"ft-{country}-f{pct}-s{seed}"


def tag_condall(level, seed):
    return f"condall-{level}-s{seed}"


def region_dir(data_dir, label_type):
    return os.path.join(data_dir, label_type, "region")


def eval_dir(data_dir, label_type):
    """``<data>/<LT>/region/eval``: evaluate.py's ``--data-dir`` for every run of this study, so its outputs never
    land next to the round-3 files (where ``confident_learning.py report`` scans for seed replicates)."""
    return os.path.join(region_dir(data_dir, label_type), "eval")


def pred_path(data_dir, label_type, tag):
    return os.path.join(eval_dir(data_dir, label_type), label_type, f"predictions_{tag}.csv")


def baseline_name(seed, base_seed=42):
    """The round-3 baseline run of a seed as ``cl.load_predictions`` names it: ``baseline`` (``predictions.csv``)
    for the chain's own seed, ``baseline-s<seed>`` for the confident-learning seed replicates.

    >>> baseline_name(42), baseline_name(43)
    ('baseline', 'baseline-s43')
    """
    return "baseline" if int(seed) == int(base_seed) else f"baseline-s{seed}"


# ---------------------------------------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------------------------------------

def class_counts(df):
    """``{"correct": n, "incorrect": n}`` of a frame's ``label_class``."""
    vc = df["label_class"].value_counts()
    return {c: int(vc.get(c, 0)) for c in CLASSES}


def loco_split(base, country_col, country):
    """``base`` with ``country``'s ``train`` rows as ``pool`` and its ``val``/``test`` rows as
    ``test_heldout_country``; every other row (round-3 ``test_heldout`` included) unchanged.

    >>> b = pd.DataFrame({"label_uid": list("abcde"), "split": ["train", "val", "test", "test_heldout", "train"]})
    >>> loco_split(b, pd.Series(["x", "x", "x", "x", "y"]), "x")["split"].tolist()
    ['pool', 'test_heldout_country', 'test_heldout_country', 'test_heldout', 'train']
    """
    df = base.copy()
    mine = (pd.Series(np.asarray(country_col), index=df.index) == country)
    df.loc[mine & (df["split"] == "train"), "split"] = POOL
    df.loc[mine & df["split"].isin(["val", "test"]), "split"] = HELDOUT_COUNTRY
    return df


def check_rows_unchanged(base, variant, untouched):
    """Raise unless every row of ``base`` where ``untouched`` is true is in ``variant`` with every cell equal."""
    a = base[np.asarray(untouched, dtype=bool)].sort_values("label_uid").reset_index(drop=True)
    b = variant[variant["label_uid"].isin(a["label_uid"])].sort_values("label_uid").reset_index(drop=True)
    if len(a) != len(b) or not a.equals(b[a.columns]):
        raise AssertionError("a variant changed rows it must leave exactly as they were")


def check_no_straddle(df):
    """No ``pano_id`` in two of train / val / test / test_heldout_country (``split_data.check_no_shared_panoramas``
    on those rows)."""
    split_data.check_no_shared_panoramas(df[df["split"].isin(["train", "val", "test", HELDOUT_COUNTRY])])


def eligible(base, country_col, country, min_eval_class=MIN_EVAL_CLASS, max_train_share=MAX_TRAIN_SHARE):
    """``(ok, reason, counts)`` for holding ``country`` out. ``counts`` has ``pool``/``eval``/``heldout`` class
    counts (its train rows, its val + test rows, its round-3 test_heldout rows). Not eligible when every row is
    already ``test_heldout``, when the country holds more than ``max_train_share`` of the train rows, or when the
    eval rows have fewer than ``min_eval_class`` of a class (reason e.g. ``"eval incorrect 5 < 30"``)."""
    mine = base[np.asarray(country_col) == country]
    counts = {"pool": class_counts(mine[mine["split"] == "train"]),
              "eval": class_counts(mine[mine["split"].isin(["val", "test"])]),
              "heldout": class_counts(mine[mine["split"] == "test_heldout"])}
    n_pool, n_eval = sum(counts["pool"].values()), sum(counts["eval"].values())
    if n_pool + n_eval == 0:
        return False, "all rows already test_heldout", counts
    share = n_pool / max(int((base["split"] == "train").sum()), 1)
    if share > max_train_share:
        return False, f"holds {share:.0%} of the train rows", counts
    for c in CLASSES:
        if counts["eval"][c] < min_eval_class:
            return False, f"eval {c} {counts['eval'][c]} < {min_eval_class}", counts
    return True, "", counts


def pool_fractions(pool_rows, groups, fractions, seed, val_share=0.2, min_frac_class=MIN_FRAC_CLASS):
    """Nested, grouped training subsets of one country's pool.

    The groups (panoramas merged within 3 m, ``cl.train_groups``) are shuffled once with ``seed``. The first
    ``val_share`` of the rows, by whole groups, is the local ``val`` for every fraction; fraction ``f`` takes the
    first ``f`` share of the remaining rows, so each subset contains every smaller one. Returns ``{pct: {"train":
    [uids], "val": [uids], "train_counts", "val_counts", "groups", "skipped": reason or None}}``; a fraction with
    fewer than ``min_frac_class`` train rows of a class is skipped (its uids are still listed for the record).

    >>> p = pd.DataFrame({"label_uid": [f"a:{i}" for i in range(10)], "label_class": ["correct", "incorrect"] * 5})
    >>> out = pool_fractions(p, pd.Series([f"g{i}" for i in range(10)]), [0.5, 1.0], 0, 0.2, 1)
    >>> set(out[50]["train"]) < set(out[100]["train"]), len(out[100]["val"]), len(out[100]["train"])
    (True, 2, 8)
    """
    pool_rows = pool_rows.reset_index(drop=True)
    g = pd.Series(np.asarray(groups), index=pool_rows.index)
    sizes = g.value_counts().sort_index()
    order = sizes.index.to_numpy()[np.random.default_rng(seed).permutation(len(sizes))]
    n = len(pool_rows)
    val_groups, rest = [], []
    taken = 0
    for gid in order:
        if taken < val_share * n:
            val_groups.append(gid)
            taken += int(sizes[gid])
        else:
            rest.append(gid)
    val_uids = pool_rows.loc[g.isin(val_groups), "label_uid"].tolist()
    m = sum(int(sizes[x]) for x in rest)
    out = {}
    for f in sorted(float(x) for x in fractions):
        chosen, cum = [], 0
        for gid in rest:
            if cum < f * m - 1e-9:
                chosen.append(gid)
                cum += int(sizes[gid])
        train = pool_rows[g.isin(chosen)]
        val = pool_rows[g.isin(val_groups)]
        tc = class_counts(train)
        low = [c for c in CLASSES if tc[c] < min_frac_class]
        out[pct_of(f)] = {"train": sorted(train["label_uid"]), "val": sorted(val_uids), "train_counts": tc,
                          "val_counts": class_counts(val), "groups": len(chosen),
                          "skipped": (f"train {low[0]} {tc[low[0]]} < {min_frac_class}" if low else None)}
    return out


def ft_split(loco, train_uids, val_uids):
    """The fine-tune split: pool rows in ``train_uids`` -> ``train``, in ``val_uids`` -> ``val``, the rest of the
    pool stays ``pool``; every other ``train``/``val`` row -> ``unused`` (so ``fine_tune_classifier.py`` sees only
    local rows); ``test``, ``test_heldout`` and ``test_heldout_country`` unchanged.

    >>> l = pd.DataFrame({"label_uid": list("abcdef"), "split": ["pool", "pool", "pool", "train", "val", "test"]})
    >>> ft_split(l, ["a"], ["b"])["split"].tolist()
    ['train', 'val', 'pool', 'unused', 'unused', 'test']
    """
    df = loco.copy()
    pool = df["split"] == POOL
    other = df["split"].isin(["train", "val"])
    df.loc[other, "split"] = UNUSED
    df.loc[pool & df["label_uid"].isin(set(train_uids)), "split"] = "train"
    df.loc[pool & df["label_uid"].isin(set(val_uids)), "split"] = "val"
    return df


def cmd_variants(args):
    base = cl.read_split(args.split)
    table = regions.country_table(regions.load_listing(args.cities_json))
    country = regions.assign(base[["city"]], table)["country"]
    wanted = [c.strip() for c in (args.countries or "").split(",") if c.strip()] or sorted(country.unique())
    fractions = [float(x) for x in args.fractions.split(",")]
    os.makedirs(args.out, exist_ok=True)
    meta = {"source_split": os.path.abspath(args.split), "source_sha256": cl.sha256_of(args.split),
            "cities_json_sha256": cl.sha256_of(args.cities_json), "seed": args.seed,
            "merge_radius_m": args.merge_radius_m, "fractions": [pct_of(f) for f in fractions],
            "min_eval_class": args.min_eval_class, "min_pool": args.min_pool, "min_frac_class": args.min_frac_class,
            "max_train_share": args.max_train_share, "pool_val_share": args.pool_val_share, "countries": {}}
    lines = []
    for c in wanted:
        ok, reason, counts = eligible(base, country, c, args.min_eval_class, args.max_train_share)
        rec = {"eligible": ok, "reason": reason, "counts": counts,
               "deployments": int(base.loc[(country == c).to_numpy(), "city"].nunique()), "fractions": {}}
        if ok:
            loco = loco_split(base, country, c)
            untouched = (country != c).to_numpy() | (base["split"] == "test_heldout").to_numpy()
            check_rows_unchanged(base, loco, untouched)
            check_no_straddle(loco)
            path = cl.write_split(loco, os.path.join(args.out, c, "split.csv"))
            rec["split_sha256"] = cl.sha256_of(path)
            pool_rows = loco[loco["split"] == POOL]
            if len(pool_rows) < args.min_pool:
                rec["fractions_reason"] = f"pool {len(pool_rows)} < {args.min_pool}"
            else:
                groups = cl.train_groups(pool_rows, args.rawlabels, args.merge_radius_m)
                for pct, fr in pool_fractions(pool_rows, groups, fractions, args.seed, args.pool_val_share,
                                              args.min_frac_class).items():
                    entry = {"train": fr["train_counts"], "val": fr["val_counts"], "groups": fr["groups"],
                             "skipped": fr["skipped"]}
                    if not fr["skipped"]:
                        variant = ft_split(loco, fr["train"], fr["val"])
                        check_rows_unchanged(loco, variant, loco["split"].isin(["test", "test_heldout",
                                                                                 HELDOUT_COUNTRY]).to_numpy())
                        check_no_straddle(variant)
                        fpath = cl.write_split(variant, os.path.join(args.out, c, f"ft-{pct}", "split.csv"))
                        entry["split_sha256"] = cl.sha256_of(fpath)
                    rec["fractions"][str(pct)] = entry
        meta["countries"][c] = rec
        p, e, h = counts["pool"], counts["eval"], counts["heldout"]
        lines.append(f"{c:16s} deployments {rec['deployments']:3d}  pool {p['correct']:5d}/{p['incorrect']:<5d} "
                     f"eval {e['correct']:5d}/{e['incorrect']:<5d} round-3 held-out {h['correct']:5d}/{h['incorrect']:<5d} "
                     + ("eligible, fractions " + ",".join(k for k, v in rec["fractions"].items() if not v["skipped"])
                        if ok else f"not eligible: {reason}"))
    with open(os.path.join(args.out, "variants.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("country          (counts correct/incorrect)")
    print("\n".join(lines))
    print(f"wrote {os.path.join(args.out, 'variants.json')}")
    return 0


def eligible_list(meta):
    """``[(country, [pct, ...])]`` of a ``variants.json``: eligible countries and their non-skipped fractions."""
    out = []
    for c, rec in sorted(meta["countries"].items()):
        if rec["eligible"]:
            out.append((c, sorted(int(p) for p, v in rec["fractions"].items() if not v.get("skipped"))))
    return out


def cmd_list(args):
    with open(args.variants) as f:
        meta = json.load(f)
    for c, pcts in eligible_list(meta):
        print(" ".join([c] + [str(p) for p in pcts]))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# predict (GPU)
# ---------------------------------------------------------------------------------------------------------------------

def load_variant(path, data_dir, label_type):
    """A variant split with ``crop_path`` resolved as ``evaluate.load_split`` does: under ``<data_dir>/<LT>/`` when
    the file exists there, else as written."""
    df = cl.read_split(path)
    base = os.path.normpath(os.path.join(data_dir, label_type))

    def resolve(p):
        if os.path.isabs(p):
            return p
        under = os.path.join(base, p)
        return under if os.path.exists(under) else p
    df["crop_path"] = df["crop_path"].map(resolve)
    return df


def prediction_complete(path, expected_uids):
    """True when ``path`` exists and scores exactly ``expected_uids`` (the requeue guard)."""
    if not os.path.exists(path):
        return False
    got = pd.read_csv(path, dtype={"label_uid": str}, usecols=["label_uid"])["label_uid"]
    return set(got) == set(expected_uids) and got.is_unique


def cmd_predict(args):
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    manifest = load_variant(args.split_file, args.data_dir, args.label_type)
    expected = manifest.loc[manifest["split"].isin(splits), "label_uid"]
    if prediction_complete(args.out, expected):
        print(f"{args.out} already scores all {len(expected)} rows; nothing to do")
        return 0
    import evaluate
    checkpoint = args.checkpoint or evaluate.best_checkpoint_in(args.run_dir)
    print(f"{len(expected)} rows of {splits}, checkpoint {checkpoint}")
    if args.conditioned:
        import region_model
        table = regions.country_table(regions.load_listing(args.cities_json))
        pred = region_model.run_inference(manifest, checkpoint, splits, table, force_unk=args.force_unk)
    else:
        pred = evaluate.run_inference(manifest, checkpoint, splits=splits)
    pred = pred.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    pred.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({len(pred)} rows)")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# calibrate (CPU): arm (d)
# ---------------------------------------------------------------------------------------------------------------------

def gate_records(fit_df, eval_df, pi_deploy, min_gated=MIN_GATED):
    """Fit the served map on ``fit_df`` (``calibration.calibrate.fit_curves`` under ``pi_deploy``) and apply it to
    ``eval_df``: per verdict, ``n``, ``gated_n`` (served >= 0.92), ``coverage`` (gated share of the verdict's rows),
    ``precision_raw`` under the eval rows' own class mix with its 95% Wilson interval (``calibration/per_city.py``;
    shown only at >= ``min_gated``) and ``precision_prior`` (weighted to the file prior, as ``evaluate_map``).
    Both frames carry ``true, p_correct, p_incorrect``. None when ``fit_df`` lacks a class (no map can be fitted)."""
    from calibration import _isotonic_map as im
    from calibration import calibrate as cal
    from calibration.per_city import wilson_interval
    fit = cal.verdicts(fit_df)
    if set(fit["true"]) != set(CLASSES):
        return None
    curves, _ = cal.fit_curves(fit, pi_deploy)
    api = im.build_api_map(curves)
    ev = cal.verdicts(eval_df)
    weights = cal.prior_weights(ev["true"], pi_deploy) if set(ev["true"]) == set(CLASSES) else None
    out = []
    for v in CLASSES:
        m = (ev["verdict"] == v).to_numpy()
        served = im.api_lookup_many(api[v], ev["confidence"].to_numpy()[m])
        correct = ev["correct"].to_numpy()[m]
        passed = served >= im.GATE
        n_g = int(passed.sum())
        rec = {"verdict": v, "n": int(m.sum()), "gated_n": n_g,
               "coverage": float(n_g / m.sum()) if m.any() else None,
               "precision_raw": None, "wilson_lo": None, "wilson_hi": None, "precision_prior": None}
        if n_g >= min_gated:
            p = float(correct[passed].mean())
            lo, hi = wilson_interval(p, n_g)
            rec.update(precision_raw=p, wilson_lo=float(lo), wilson_hi=float(hi))
            if weights is not None:
                rec["precision_prior"] = im.gate_summary(served, correct, weights[m])["precision"]
        out.append(rec)
    return out


def loco_calibration(pred, ft_train_uids, pi_deploy, min_gated=MIN_GATED):
    """Arm (d) for one country and seed, from the global model's predictions ``pred``: the pooled map (fitted on the
    other countries' ``val`` rows) and a local map per fraction (fitted on that fraction's ``train`` rows, scored by
    the global model as ``pool``), each applied to the ``test_heldout_country`` rows. ``ft_train_uids`` is ``{pct:
    uids}``. Records with ``map`` = ``pooled`` or ``local-<pct>`` and ``fit_n``."""
    ev = pred[pred["split"] == HELDOUT_COUNTRY]
    fits = {"pooled": pred[pred["split"] == "val"]}
    for pct, uids in sorted(ft_train_uids.items()):
        fits[f"local-{pct}"] = pred[(pred["split"] == POOL) & pred["label_uid"].isin(set(uids))]
    out = []
    for name, fit in fits.items():
        recs = gate_records(fit, ev, pi_deploy, min_gated)
        for r in recs or [{"verdict": v, "skipped": "fit rows lack a class"} for v in CLASSES]:
            out.append({"map": name, "fit_n": int(len(fit)), **r})
    return out


def in_distribution_calibration(pred, country_of_uid, pi_deploy, min_val=MIN_VAL_OWN_MAP, min_gated=MIN_GATED):
    """Per country with >= ``min_val`` val rows: the pooled map (all ``val`` rows) and the country's own map (its
    ``val`` rows), both applied to that country's ``test`` rows (round-3 predictions). Records with ``country``,
    ``map`` (``pooled`` / ``own``) and ``fit_n``."""
    p = pred.assign(country=pred["label_uid"].map(country_of_uid))
    val = p[p["split"] == "val"]
    out = []
    for c in sorted(val["country"].dropna().unique()):
        own = val[val["country"] == c]
        test = p[(p["split"] == "test") & (p["country"] == c)]
        if len(own) < min_val or test.empty:
            continue
        for name, fit in (("pooled", val), ("own", own)):
            recs = gate_records(fit, test, pi_deploy, min_gated)
            for r in recs or [{"verdict": v, "skipped": "fit rows lack a class"} for v in CLASSES]:
                out.append({"country": c, "map": name, "fit_n": int(len(fit)), **r})
    return out


def cmd_calibrate(args):
    from calibration import calibrate as cal
    lt = args.label_type
    pi, _ = cal.load_prior(args.priors, lt)
    rdir = region_dir(args.data_dir, lt)
    with open(os.path.join(rdir, "variants.json")) as f:
        meta = json.load(f)
    seeds = [int(s) for s in args.seeds.split(",")]
    records, missing = [], []
    for c, pcts in eligible_list(meta):
        uids = {}
        for pct in pcts:
            v = cl.read_split(os.path.join(rdir, c, f"ft-{pct}", "split.csv"))
            uids[pct] = v.loc[v["split"] == "train", "label_uid"].tolist()
        for s in seeds:
            path = pred_path(args.data_dir, lt, tag_global(c, s))
            if not os.path.exists(path):
                missing.append(path)
                continue
            pred = pd.read_csv(path, dtype={"label_uid": str})
            records += [{"kind": "loco", "country": c, "seed": s, **r} for r in loco_calibration(pred, uids, pi)]
    split_df = cl.read_split(os.path.join(args.data_dir, lt, "split.csv"))
    table = regions.country_table(regions.load_listing(args.cities_json))
    country_of_uid = dict(zip(split_df["label_uid"], regions.assign(split_df[["city"]], table)["country"]))
    for s in seeds:
        name = baseline_name(s, args.base_seed)
        path = os.path.join(args.data_dir, lt, "predictions.csv" if name == "baseline" else f"predictions_{name}.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        pred = pd.read_csv(path, dtype={"label_uid": str})
        records += [{"kind": "in_dist", "seed": s, **r} for r in in_distribution_calibration(pred, country_of_uid, pi)]
    out = args.out or os.path.join(rdir, "calibration.json")
    with open(out, "w") as f:
        json.dump({"label_type": lt, "seeds": seeds, "pi_deploy": pi, "priors_sha256": cl.sha256_of(args.priors),
                   "records": records, "missing": missing}, f, indent=2, default=float)
        f.write("\n")
    print(f"wrote {out} ({len(records)} records; {len(missing)} prediction files missing)")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report: pure functions
# ---------------------------------------------------------------------------------------------------------------------

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def auc_on(pred, splits):
    """AUC of ``p_incorrect`` on the rows of ``pred`` whose split is in ``splits`` (None with one class)."""
    p = pred[pred["split"].isin([splits] if isinstance(splits, str) else splits)]
    return _auc(p["true"] == "incorrect", p["p_incorrect"]) if len(p) else None


def within_on(pred, split, min_n=WITHIN_MIN_N):
    """``retrain_report.within_city_auc`` on one split: ``(auc or None, n_deployments)``."""
    from analysis import retrain_report as rr
    p = pred[pred["split"] == split]
    if p.empty:
        return None, 0
    auc, n = rr.within_city_auc(p["true"] == "incorrect", p["p_incorrect"], p["city"], min_n)
    return (None if np.isnan(auc) else float(auc)), int(n)


def recall_at_p(pred, split, cls, level=GATE):
    """Recall of verdict ``cls`` at precision ``level`` on one split (``evaluate.recall_at_precision``)."""
    from evaluate import recall_at_precision
    p = pred[pred["split"] == split]
    y = (p["true"] == cls).to_numpy()
    if not 0 < y.sum() < len(y):
        return None
    return recall_at_precision(y, p[f"p_{cls}"].to_numpy(), level)[0]


def paired_delta(pred_a, pred_b, split, n_boot=1000, seed=0, name=""):
    """AUC of ``pred_a`` minus ``pred_b`` on ``pred_a``'s rows of ``split``, with a 95% interval from resampling whole
    panoramas (the same resample for both; ``analysis/slices.py``). ``pred_b`` must score every one of those rows
    (its other rows are ignored). The interval is None when the smaller class has fewer than
    ``slices.MIN_CLASS_FOR_CI`` rows or ``n_boot`` is 0. ``{"delta", "delta_lo", "delta_hi", "n"}`` or None."""
    from analysis import slices as sl
    a = pred_a[pred_a["split"] == split]
    if a.empty:
        return None
    m = a.merge(pred_b[["label_uid", "p_incorrect"]], on="label_uid", how="left", suffixes=("", "_b"),
                validate="one_to_one")
    if m["p_incorrect_b"].isna().any():
        raise ValueError(f"the reference has no score for {int(m['p_incorrect_b'].isna().sum())} rows of {split}")
    y = (m["true"] == "incorrect").to_numpy()
    x, z = m["p_incorrect"].to_numpy(), m["p_incorrect_b"].to_numpy()
    if not 0 < y.sum() < len(y):
        return None
    rec = {"delta": _auc(y, x) - _auc(y, z), "delta_lo": None, "delta_hi": None, "n": int(len(m))}
    if n_boot and min(y.sum(), (~y).sum()) >= sl.MIN_CLASS_FOR_CI:
        idxs = sl.boot_indices(y, m["pano_id"].to_numpy(), n_boot,
                               np.random.default_rng(sl.slice_seed(seed, f"{name}:{split}")))
        vals = [_auc(y[i], x[i]) - _auc(y[i], z[i]) for i in idxs]
        if vals:
            rec["delta_lo"], rec["delta_hi"] = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return rec


def judge(delta, n_eval, min_n=MIN_JUDGE_N):
    """The verdict column: ``better``/``worse`` when the interval excludes 0, ``no clear difference`` otherwise,
    ``n too small`` without an interval or under ``min_n`` eval rows.

    >>> judge({"delta": 0.03, "delta_lo": 0.01, "delta_hi": 0.05}, 150), judge({"delta": 0.03}, 150)
    ('better', 'n too small')
    """
    if not delta or delta.get("delta_lo") is None or n_eval < min_n:
        return "n too small"
    if delta["delta_lo"] > 0:
        return "better"
    if delta["delta_hi"] < 0:
        return "worse"
    return "no clear difference"


def _put_delta(row, key, d):
    row[f"{key}_delta"] = d["delta"] if d else None
    row[f"{key}_delta_lo"] = d["delta_lo"] if d else None
    row[f"{key}_delta_hi"] = d["delta_hi"] if d else None


def arm_metrics(pred, split, prefix, within_min_n=WITHIN_MIN_N):
    """``{<prefix>_auc, _within, _within_n, _rp92_incorrect, _rp92_correct}`` of one model on one split."""
    w, n = within_on(pred, split, within_min_n)
    return {f"{prefix}_auc": auc_on(pred, split), f"{prefix}_within": w, f"{prefix}_within_n": n,
            f"{prefix}_rp92_incorrect": recall_at_p(pred, split, "incorrect"),
            f"{prefix}_rp92_correct": recall_at_p(pred, split, "correct")}


def country_rows(preds, base_preds, countries, seeds, n_boot=1000, seed=0, within_min_n=WITHIN_MIN_N):
    """The LOCO table, one record per country x seed with a ``global`` run: the held-out country's eval rows
    (``n_eval``, ``n_incorrect``), the global and conditioned models' AUC, within-city AUC and recall at precision
    0.92 there, the conditioned model's paired Δ vs the same-seed global and its ``verdict``; the global model's
    AUC on all of the country's rows (pool + eval, zero-shot), and its paired Δ vs the same-seed round-3 baseline on
    ``test`` and ``test_heldout`` (does dropping the country move the rest). ``preds`` is ``{tag: joined
    predictions}``, ``base_preds`` ``{seed: joined baseline predictions}``."""
    rows = []
    for c in countries:
        for s in seeds:
            g = preds.get(tag_global(c, s))
            if g is None:
                continue
            ev = g[g["split"] == HELDOUT_COUNTRY]
            row = {"country": c, "seed": s, "n_eval": int(len(ev)), "n_incorrect": int((ev["true"] == "incorrect").sum()),
                   **arm_metrics(g, HELDOUT_COUNTRY, "global", within_min_n),
                   "global_all_auc": auc_on(g, [POOL, HELDOUT_COUNTRY])}
            cond = preds.get(tag_cond(c, s))
            if cond is not None:
                row.update(arm_metrics(cond, HELDOUT_COUNTRY, "cond", within_min_n))
                d = paired_delta(cond, g, HELDOUT_COUNTRY, n_boot, seed, f"cond:{c}:{s}")
                _put_delta(row, "cond", d)
                row["verdict"] = judge(d, row["n_eval"])
            else:
                row["verdict"] = "not run"
            base = base_preds.get(s)
            if base is not None:
                for split, key in (("test", "drop_test"), ("test_heldout", "drop_heldout")):
                    _put_delta(row, key, paired_delta(g, base, split, n_boot, seed, f"drop:{c}:{s}"))
            rows.append(row)
    return rows


def curve_rows(preds, meta, calib, countries, seeds, n_boot=1000, seed=0):
    """The data-size curve, one record per country x seed x fraction (fraction 0 = the global model zero-shot):
    the local train rows used, the fine-tuned AUC on the held-out country and its paired Δ vs the same-seed global,
    the AUC on ``test`` and its Δ vs global (forgetting), and arm (d) at the same fraction: the Agree verdict's
    gated n, coverage and raw precision with its Wilson interval (``calib`` = the ``loco`` calibration records)."""
    rows = []
    for c in countries:
        fr = meta["countries"][c]["fractions"]
        pcts = sorted(int(p) for p, v in fr.items() if not v.get("skipped"))
        for s in seeds:
            g = preds.get(tag_global(c, s))
            if g is None:
                continue
            cal_by = {r["map"]: r for r in calib if r.get("country") == c and r.get("seed") == s
                      and r.get("verdict") == "correct"}
            for pct in [0] + pcts:
                row = {"country": c, "seed": s, "pct": pct,
                       "train_correct": fr[str(pct)]["train"]["correct"] if pct else 0,
                       "train_incorrect": fr[str(pct)]["train"]["incorrect"] if pct else 0}
                model = g if pct == 0 else preds.get(tag_ft(c, pct, s))
                if model is not None:
                    row["auc"] = auc_on(model, HELDOUT_COUNTRY)
                    row["test_auc"] = auc_on(model, "test")
                    if pct:
                        _put_delta(row, "heldout", paired_delta(model, g, HELDOUT_COUNTRY, n_boot, seed, f"ft:{c}:{pct}:{s}"))
                        _put_delta(row, "test", paired_delta(model, g, "test", n_boot, seed, f"ftt:{c}:{pct}:{s}"))
                else:
                    row["auc"] = None
                r = cal_by.get("pooled" if pct == 0 else f"local-{pct}")
                if r:
                    row.update({f"agree_{k}": r.get(k) for k in ("gated_n", "coverage", "precision_raw",
                                                                  "wilson_lo", "wilson_hi")})
                rows.append(row)
    return rows


def condall_rows(preds, base_preds, country_of_uid, levels, seeds, n_boot=1000, seed=0,
                 min_class=MIN_COUNTRY_CLASS, within_min_n=WITHIN_MIN_N):
    """In distribution: ``condall-<level>-s<seed>`` (and its ``-unk`` scoring, every row as unknown) vs the round-3
    baseline of the same seed on ``test``: pooled AUC, within-city AUC and paired Δ; and per country with at least
    ``min_class`` test rows of each class, the paired Δ (``scope`` = the country)."""
    rows = []
    for level in levels:
        for s in seeds:
            base = base_preds.get(s)
            for suffix in ("", "-unk"):
                model = preds.get(tag_condall(level, s) + suffix)
                if model is None or base is None:
                    continue
                arm = f"{level}{' (scored as unknown)' if suffix else ''}"
                w, n = within_on(model, "test", within_min_n)
                bw, _ = within_on(base, "test", within_min_n)
                row = {"level": arm, "seed": s, "scope": "all", "n": int((model["split"] == "test").sum()),
                       "auc": auc_on(model, "test"), "base_auc": auc_on(base, "test"), "within": w,
                       "base_within": bw, "within_n": n}
                _put_delta(row, "test", paired_delta(model, base, "test", n_boot, seed, f"condall:{arm}:{s}"))
                rows.append(row)
                t = model[model["split"] == "test"].assign(country=lambda d: d["label_uid"].map(country_of_uid))
                for c in sorted(t["country"].dropna().unique()):
                    part = t[t["country"] == c]
                    k = class_counts(part.rename(columns={"true": "label_class"}))
                    r = {"level": arm, "seed": s, "scope": c, "n": int(len(part)), "n_incorrect": k["incorrect"]}
                    if min(k.values()) >= min_class:
                        r.update(auc=auc_on(part, "test"), base_auc=auc_on(base[base["label_uid"].isin(part["label_uid"])], "test"))
                        _put_delta(r, "test", paired_delta(part, base, "test", n_boot, seed, f"condall:{arm}:{c}:{s}"))
                    else:
                        r["note"] = "n too small"
                    rows.append(r)
    return rows


def slice_rows(preds, train_rows, countries, seed_of_run, n_boot=1000, seed=0, min_n=None):
    """Severity and tag slices of each arm against the same-seed global model on the held-out country's eval rows
    (``slices.paired_delta_table``), with the training rows of the slice from the arm's **own** variant split
    (``train_rows``: ``{(arm, country): train frame}``), flagged thin under ``slices.THIN_TRAIN``. Arms: ``cond``
    and every ``ft-<pct>`` present."""
    from analysis import slices as sl
    min_n = sl.MIN_N if min_n is None else min_n
    out = []
    for c in countries:
        g = preds.get(tag_global(c, seed_of_run))
        if g is None:
            continue
        arms = [("cond", tag_cond(c, seed_of_run))]
        pcts = sorted(int(a[3:]) for a, cc in train_rows if cc == c and a.startswith("ft-"))
        arms += [(f"ft-{pct}", tag_ft(c, pct, seed_of_run)) for pct in pcts]
        for arm, tag in arms:
            p = preds.get(tag)
            if p is None:
                continue
            part = p[p["split"] == HELDOUT_COUNTRY]
            train = train_rows.get((arm, c))
            for by in sl.BY:
                t = sl.paired_delta_table(part, g, by, min_n, n_boot, seed)
                counts = sl.with_slice(train, by).groupby("slice").size() if train is not None else pd.Series(dtype=int)
                for r in t.to_dict("records"):
                    tn = int(counts.get(r["slice"], 0))
                    out.append({"country": c, "arm": arm, "by": by, **r, "train_n": tn,
                                "thin": tn < sl.THIN_TRAIN})
    return out


def seed_rows(preds, countries, seeds):
    """Per (arm in global/cond) x country x seed: AUC on ``test`` and on the held-out country, within-city and
    incorrect recall at 0.92 on the held-out country, and for ``cond`` the Δ vs the global model of the same seed
    (no interval; the per-seed intervals are in the LOCO table). ``variant`` = ``<arm>|<country>``, the shape
    ``confident_learning.seed_summary`` reads."""
    rows = []
    for c in countries:
        for s in seeds:
            g = preds.get(tag_global(c, s))
            for arm, p in (("global", g), ("cond", preds.get(tag_cond(c, s)))):
                if p is None:
                    continue
                w, _ = within_on(p, HELDOUT_COUNTRY)
                r = {"variant": f"{arm}|{c}", "seed": s, "test_auc": auc_on(p, "test"),
                     "heldout_auc": auc_on(p, HELDOUT_COUNTRY), "within_city_heldout": w,
                     "incorrect_recall_p92_heldout": recall_at_p(p, HELDOUT_COUNTRY, "incorrect")}
                if arm == "cond" and g is not None:
                    for k, split in (("heldout", HELDOUT_COUNTRY), ("test", "test")):
                        a, b = auc_on(p, split), auc_on(g, split)
                        if a is not None and b is not None:
                            r[f"{k}_delta"] = a - b
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# report: naming, redaction, rendering
# ---------------------------------------------------------------------------------------------------------------------

def rename(records, names, keys=("country", "scope")):
    """Copies of ``records`` with every country id in ``keys`` replaced by its display name."""
    out = []
    for r in records:
        r = dict(r)
        for k in keys:
            if k in r and r[k] in names:
                r[k] = names[r[k]]
        out.append(r)
    return out


def redact(obj, terms):
    """Raise ``ValueError`` when the JSON of ``obj`` contains any of ``terms`` (case-insensitive, as a whole token
    between non-alphanumeric characters). Returns ``obj`` unchanged otherwise.

    >>> redact({"a": "private country A"}, {"atlantis"})
    {'a': 'private country A'}
    """
    text = json.dumps(obj, default=str).lower()
    hits = sorted(t for t in terms if t and re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", text))
    if hits:
        raise ValueError(f"{len(hits)} non-public term(s) would be committed; fix the naming before writing")
    return obj


def _f(v, fmt="{:.3f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return fmt.format(v)


def _d(r, key):
    """``Δ [lo, hi]`` from ``<key>_delta`` (``delta`` when ``key`` is empty) and its interval keys."""
    k = f"{key}_delta" if key else "delta"
    d = r.get(k)
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "—"
    cell = f"{d:+.3f}"
    lo = r.get(f"{k}_lo")
    if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
        cell += f" [{lo:+.3f}, {r[f'{k}_hi']:+.3f}]"
    return cell


def _table(head, rows, n_left=2):
    """A Markdown table; the first ``n_left`` columns (names) are left-aligned, the rest (numbers) right-aligned."""
    align = "|" + "|".join("---" if i < n_left else "---:" for i in range(len(head))) + "|"
    return "\n".join(["| " + " | ".join(head) + " |", align] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def loco_table_md(per_type):
    """The LOCO table: per type and country, deployments, pool / eval / round-3 held-out rows by class, eligible."""
    rows = []
    for lt, t in per_type.items():
        for r in t["loco_table"]:
            p, e, h = r["counts"]["pool"], r["counts"]["eval"], r["counts"]["heldout"]
            rows.append([lt, r["country"], r["deployments"], f"{p['correct']:,} / {p['incorrect']:,}",
                         f"{e['correct']:,} / {e['incorrect']:,}", f"{h['correct']:,} / {h['incorrect']:,}",
                         "yes" if r["eligible"] else f"no ({r['reason']})", r.get("fractions") or "—"])
    return _table(["type", "country", "deployments", "pool (correct / incorrect)", "eval (correct / incorrect)",
                   "round-3 held-out", "eligible", "fine-tune fractions (%)"], rows)


def loco_md(per_type):
    rows = []
    for lt, t in per_type.items():
        for r in t["loco"]:
            rows.append([lt, r["country"], r["seed"], f"{r['n_eval']:,} ({r['n_incorrect']:,})",
                         _f(r.get("global_auc")), _f(r.get("cond_auc")), _d(r, "cond"), r["verdict"],
                         f"{_f(r.get('global_within'))} / {_f(r.get('cond_within'))} ({r.get('global_within_n', 0)})",
                         f"{_f(r.get('global_rp92_incorrect'))} / {_f(r.get('cond_rp92_incorrect'))}",
                         f"{_f(r.get('global_rp92_correct'))} / {_f(r.get('cond_rp92_correct'))}",
                         _f(r.get("global_all_auc")), _d(r, "drop_test"), _d(r, "drop_heldout")])
    return _table(["type", "country", "seed", "eval n (incorrect)", "global AUC", "cond AUC",
                   "cond − global [95% CI]", "verdict", "within-city global / cond (deployments)",
                   "incorrect R@P0.92 global / cond", "correct R@P0.92 global / cond", "global AUC, all country rows",
                   "global − round 3, test", "global − round 3, test_heldout"], rows)


def curves_md(per_type):
    rows = []
    for lt, t in per_type.items():
        for r in t["curves"]:
            prec = _f(r.get("agree_precision_raw"))
            if r.get("agree_wilson_lo") is not None:
                prec += f" [{r['agree_wilson_lo']:.3f}, {r['agree_wilson_hi']:.3f}]"
            rows.append([lt, r["country"], r["seed"], f"{r['pct']}%" if r["pct"] else "0 (global)",
                         f"{r['train_correct']:,} / {r['train_incorrect']:,}", _f(r.get("auc")), _d(r, "heldout"),
                         _f(r.get("test_auc")), _d(r, "test"), _f(r.get("agree_gated_n"), "{:,}"),
                         _f(r.get("agree_coverage"), "{:.1%}"), prec])
    return _table(["type", "country", "seed", "local fraction", "local train rows (correct / incorrect)",
                   "AUC, held-out country", "Δ vs global [95% CI]", "AUC, test", "Δ test vs global (forgetting)",
                   "Agree gated n (map at this fraction)", "Agree coverage", "Agree raw precision [Wilson 95%]"], rows,
                  n_left=4)


def condall_md(per_type):
    rows = []
    for lt, t in per_type.items():
        for r in t["condall"]:
            rows.append([lt, r["level"], r["seed"], r["scope"], f"{r['n']:,}", _f(r.get("base_auc")), _f(r.get("auc")),
                         r.get("note") or _d(r, "test"),
                         f"{_f(r.get('base_within'))} / {_f(r.get('within'))}" if r["scope"] == "all" else ""])
    return _table(["type", "conditioning", "seed", "scope", "test rows", "round-3 AUC", "conditioned AUC",
                   "Δ [95% CI]", "within-city round 3 / conditioned"], rows, n_left=4)


def calibration_md(per_type):
    rows = []
    for lt, t in per_type.items():
        for r in t["calibration"]:
            if r.get("skipped"):
                continue
            prec = _f(r.get("precision_raw"))
            if r.get("wilson_lo") is not None:
                prec += f" [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"
            rows.append([lt, "held out" if r["kind"] == "loco" else "in distribution", r["country"], r["seed"],
                         r["map"], f"{r['fit_n']:,}", "Agree" if r["verdict"] == "correct" else "Disagree",
                         f"{r['n']:,}", f"{r['gated_n']:,}", _f(r.get("coverage"), "{:.1%}"), prec,
                         _f(r.get("precision_prior"))])
    return _table(["type", "setting", "country", "seed", "map", "fit rows", "verdict", "verdict rows", "gated",
                   "coverage", "raw precision [Wilson 95%]", "precision under the file prior"], rows, n_left=5)


def summary_md(per_type):
    """One bullet per type: how many countries were held out and the conditioned model's verdicts."""
    lines = []
    for lt, t in per_type.items():
        loco = t["loco"]
        if not loco:
            lines.append(f"- {lt}: no held-out country was run.")
            continue
        parts = [f"{r['country']} (seed {r['seed']}): global {_f(r.get('global_auc'))}, cond − global "
                 f"{_d(r, 'cond')}, {r['verdict']}" for r in loco]
        lines.append(f"- {lt}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


def render_blocks(results):
    """``{block: markdown}`` from a ``results.json`` dict (None = not yet run)."""
    if results is None:
        return {b: ("**Status: not yet run.** No GPU job has run; the report stage replaces this line."
                    if b == "status" else NOT_RUN) for b in BLOCKS}
    pt = results["per_type"]
    prov = results["provenance"]
    return {
        "status": (f"**Status: run.** Report generated on {results['generated_on']} for "
                   f"{', '.join(results['label_types'])}; seeds {', '.join(str(s) for s in results['seeds'])}."),
        "summary": summary_md(pt),
        "loco_table": loco_table_md(pt),
        "results_loco": loco_md(pt),
        "results_curves": curves_md(pt),
        "results_conditioned": condall_md(pt),
        "results_calibration": calibration_md(pt),
        "provenance": "```json\n" + json.dumps(prov, indent=2, sort_keys=True) + "\n```",
    }


def splice(text, blocks):
    """Replace each generated block of ``text``; everything else stays byte-identical. A missing block raises."""
    for name, body in blocks.items():
        pat = re.compile(re.escape(MARK_BEGIN.format(name=name)) + r".*?" + re.escape(MARK_END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: MARK_BEGIN.format(name=name) + "\n\n" + body + "\n\n" + MARK_END.format(name=name),
                       text)
    return text


def check_doc(doc_path, out_dir):
    """``[]`` when the doc's generated blocks equal what the committed ``results.json`` renders (or, without one,
    the "not yet run" placeholders); otherwise the names of the blocks that drifted."""
    path = os.path.join(out_dir, "results.json")
    results = None
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    with open(doc_path) as f:
        text = f.read()
    expected = splice(text, render_blocks(results))
    if expected == text:
        return []
    return [b for b in BLOCKS if splice(text, {b: render_blocks(results)[b]}) != text]


# ---------------------------------------------------------------------------------------------------------------------
# report: I/O
# ---------------------------------------------------------------------------------------------------------------------

def join_predictions(path, split_df):
    """A predictions file joined with the split's city, panorama, severity and tags (``cl.PRED_JOIN_COLUMNS``)."""
    pred = pd.read_csv(path, dtype={"label_uid": str, "split": str, "true": str})
    m = pred.merge(split_df[cl.PRED_JOIN_COLUMNS], on="label_uid", how="left", validate="one_to_one")
    if m["city"].isna().any():
        raise ValueError(f"{path}: {int(m['city'].isna().sum())} predictions have no row in split.csv")
    return m


TAG_RE = re.compile(r"^predictions_(?P<tag>(global|cond|ft|condall)-.+)\.csv$")


def load_type(data_dir, lt, seeds, base_seed):
    """Everything the report reads for one label type: ``(split_df, meta, preds, base_preds, calib, hashes)``."""
    split_path = os.path.join(data_dir, lt, "split.csv")
    split_df = cl.read_split(split_path)
    rdir = region_dir(data_dir, lt)
    with open(os.path.join(rdir, "variants.json")) as f:
        meta = json.load(f)
    hashes = {"split": cl.sha256_of(split_path), "variants_json": cl.sha256_of(os.path.join(rdir, "variants.json"))}
    preds = {}
    for path in sorted(glob.glob(os.path.join(eval_dir(data_dir, lt), lt, "predictions_*.csv"))):
        m = TAG_RE.match(os.path.basename(path))
        if m:
            preds[m["tag"]] = join_predictions(path, split_df)
            hashes[f"predictions:{m['tag']}"] = cl.sha256_of(path)
    base_preds = {}
    for s in seeds:
        name = baseline_name(s, base_seed)
        p = cl.load_predictions(data_dir, lt, name, split_df)
        if p is not None:
            base_preds[s] = p
            fn = "predictions.csv" if name == "baseline" else f"predictions_{name}.csv"
            hashes[f"baseline:{s}"] = cl.sha256_of(os.path.join(data_dir, lt, fn))
    calib = []
    cpath = os.path.join(rdir, "calibration.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            calib = json.load(f)["records"]
        hashes["calibration_json"] = cl.sha256_of(cpath)
    return split_df, meta, preds, base_preds, calib, hashes


def rename_hash_keys(hashes, names):
    """Provenance keys carry run tags, which contain country ids; replace each id by its display name."""
    out = {}
    for k, v in hashes.items():
        for c in sorted(names, key=len, reverse=True):
            k = re.sub(r"(?<=-)" + re.escape(c) + r"(?=-)", names[c].replace(" ", "_"), k)
        out[k] = v
    return out


def train_frames(data_dir, lt, meta):
    """``{(arm, country): train rows}`` of each arm's own split: ``cond`` (and ``global``) the LOCO split,
    ``ft-<pct>`` that fraction's split."""
    out = {}
    rdir = region_dir(data_dir, lt)
    for c, pcts in eligible_list(meta):
        loco = cl.read_split(os.path.join(rdir, c, "split.csv"))
        out[("cond", c)] = loco[loco["split"] == "train"]
        for pct in pcts:
            v = cl.read_split(os.path.join(rdir, c, f"ft-{pct}", "split.csv"))
            out[(f"ft-{pct}", c)] = v[v["split"] == "train"]
    return out


def seeds_md(per_type):
    lines = []
    for lt, t in per_type.items():
        if not t["seed_summary"]:
            continue
        lines += [f"### {lt}", "", _table(
            ["arm", "country", "seeds", "AUC held-out country (mean ± sd)", "AUC test (mean ± sd)",
             "Δ cond − global, held-out country: mean [min, max], seeds > 0"], n_left=3,
            rows=[[s["variant"].split("|")[0], s["country"], ", ".join(str(x) for x in s["seeds"]),
              cl._pm(s["heldout_auc_mean"], s["heldout_auc_sd"]), cl._pm(s["test_auc_mean"], s["test_auc_sd"]),
              cl._delta_range(s, "heldout")] for s in t["seed_summary"]]), ""]
    return "\n".join(lines)


def slices_md(per_type):
    lines = []
    for lt, t in per_type.items():
        if not t["slices"]:
            continue
        for (c, arm, by), grp in pd.DataFrame(t["slices"]).groupby(["country", "arm", "by"], sort=False):
            lines += [f"### {lt} / {c} / {arm} vs global / by {by}", "", _table(
                [by.capitalize(), "n", "incorrect", "global AUC", f"{arm} AUC", "Δ [95% CI]", "training rows"],
                [[r["slice"], f"{r['n']:,}", f"{r['n_incorrect']:,}", _f(r["auc_b"]), _f(r["auc_a"]),
                  _d(r, ""), f"{r['train_n']:,}{' ⚠ thin' if r['thin'] else ''}"]
                 for r in grp.to_dict("records")], n_left=1), ""]
    return "\n".join(lines)


def _clean(obj):
    """NaN -> None, numpy scalars -> Python, so the JSON is stable and valid."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def job_ids(path):
    """Numeric Slurm job ids from ``region_jobs.txt`` (nothing else of its lines is kept)."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return sorted(set(re.findall(r"(?<![\w-])\d{5,}(?![\w-])", f.read())), key=int)


def cmd_report(args):
    if args.check:
        drift = check_doc(args.doc, args.out)
        if drift:
            print(f"{args.doc} drifted from {args.out}/results.json in block(s): {', '.join(drift)}", file=sys.stderr)
            return 1
        print(f"{args.doc} matches {args.out}/results.json")
        return 0
    if not args.data_dir or not args.cities_json:
        raise SystemExit("report needs --data-dir and --cities-json (or --check)")
    seeds = [int(s) for s in args.seeds.split(",")]
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    run_listing = regions.load_listing(args.cities_json)
    table = regions.country_table(run_listing)
    naming = regions.load_listing(args.public_cities) if args.public_cities else regions.load_listing(None)
    terms = regions.non_public_terms(naming)
    per_type, hashes, all_countries = {}, {}, set()
    raw = {}
    for lt in label_types:
        if not os.path.exists(os.path.join(region_dir(args.data_dir, lt), "variants.json")):
            print(f"{lt}: no region/variants.json, skipped", file=sys.stderr)
            continue
        split_df, meta, preds, base_preds, calib, h = load_type(args.data_dir, lt, seeds, args.base_seed)
        countries = [c for c, _ in eligible_list(meta)]
        all_countries |= set(meta["countries"])
        country_of_uid = dict(zip(split_df["label_uid"], regions.assign(split_df[["city"]], table)["country"]))
        loco_tab = [{"country": c, "deployments": r["deployments"], "counts": r["counts"], "eligible": r["eligible"],
                     "reason": r["reason"],
                     "fractions": ", ".join(p for p, v in r["fractions"].items() if not v.get("skipped"))}
                    for c, r in sorted(meta["countries"].items())]
        loco_calib = [r for r in calib if r.get("kind") == "loco"]
        srows = seed_rows(preds, countries, seeds)
        summary = cl.seed_summary(srows)
        for s in summary:
            s["country"] = s["variant"].split("|")[1]
        raw[lt] = {
            "loco_table": loco_tab,
            "loco": country_rows(preds, base_preds, countries, seeds, args.n_boot, args.seed, args.within_min_n),
            "curves": curve_rows(preds, meta, loco_calib, countries, seeds, args.n_boot, args.seed),
            "condall": condall_rows(preds, base_preds, country_of_uid, ("country", "region"), seeds, args.n_boot,
                                    args.seed, within_min_n=args.within_min_n),
            "calibration": calib,
            "slices": slice_rows(preds, train_frames(args.data_dir, lt, meta), countries, args.base_seed,
                                 args.n_boot, args.seed),
            "seed_summary": summary,
        }
        for r in raw[lt]["condall"] + calib:
            for k in ("scope", "country"):
                if k in r and r[k] not in ("all", None):
                    all_countries.add(r[k])
        hashes[lt] = h
    names = regions.display_names(all_countries, regions.public_countries(naming))
    for lt, t in raw.items():
        per_type[lt] = {k: rename(v, names) for k, v in t.items()}
        for s in per_type[lt]["seed_summary"]:
            s["variant"] = s["variant"].split("|")[0] + "|" + s["country"]
        hashes[lt] = rename_hash_keys(hashes[lt], names)
    results = _clean({
        "generated_on": datetime.date.today().isoformat(), "label_types": list(per_type), "seeds": seeds,
        "per_type": per_type,
        "provenance": {"inputs_sha256": hashes, "cities_json_sha256": cl.sha256_of(args.cities_json),
                       "naming_source": os.path.basename(args.public_cities) if args.public_cities else "live /v3/api/cities",
                       "public_countries": sorted(regions.public_countries(naming)), "n_boot": args.n_boot,
                       "bootstrap_seed": args.seed, "seeds": seeds, "base_seed": args.base_seed,
                       "within_min_n": args.within_min_n, "job_ids": job_ids(args.jobs_file),
                       "min_n_rules": {"min_eval_class": MIN_EVAL_CLASS, "min_frac_class": MIN_FRAC_CLASS,
                                       "min_gated": MIN_GATED, "min_judge_n": MIN_JUDGE_N,
                                       "min_country_class": MIN_COUNTRY_CLASS}},
    })
    redact(results, terms)
    os.makedirs(args.out, exist_ok=True)
    blocks = render_blocks(results)
    texts = {"results.md": "\n\n".join(f"## {b}\n\n{blocks[b]}" for b in BLOCKS if b != "provenance") + "\n",
             "slices.md": ("# Severity and tag slices on the held-out country: each arm against the same-seed global "
                           "model\n\nPaired panorama bootstrap (`analysis/slices.py`); training rows from the arm's "
                           "own split.\n\n" + slices_md(per_type)),
             "seeds.md": "# Seed replicates per arm and country\n\n" + seeds_md(per_type),
             "results.json": json.dumps(results, indent=2, sort_keys=True) + "\n"}
    for name, text in texts.items():
        redact(text, terms)
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        f.write(json.dumps(results["provenance"], indent=2, sort_keys=True) + "\n")
    if args.doc:
        with open(args.doc) as f:
            doc = f.read()
        doc = splice(doc, blocks)
        redact(doc, terms)
        with open(args.doc, "w") as f:
            f.write(doc)
    print(blocks["summary"])
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("variants", help="LOCO and fine-tune split files for one label type")
    v.add_argument("--split", required=True, help="the round-3 split.csv")
    v.add_argument("--cities-json", required=True, help="a saved /v3/api/cities listing (country_id per deployment)")
    v.add_argument("--rawlabels", default=None,
                   help="cached rawLabels exports <city>.csv (needed for the 3 m merge: the round-3 split.csv has no "
                        "coordinates)")
    v.add_argument("--merge-radius-m", type=float, default=3.0)
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--countries", default=None, help="comma list (default: every country in the split)")
    v.add_argument("--min-eval-class", type=int, default=MIN_EVAL_CLASS)
    v.add_argument("--min-pool", type=int, default=MIN_POOL)
    v.add_argument("--max-train-share", type=float, default=MAX_TRAIN_SHARE)
    v.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    v.add_argument("--min-frac-class", type=int, default=MIN_FRAC_CLASS)
    v.add_argument("--pool-val-share", type=float, default=0.2)
    v.add_argument("--out", required=True, help="$DATA/<LT>/region")
    v.set_defaults(func=cmd_variants)

    li = sub.add_parser("list", help="eligible countries and fractions of a variants.json")
    li.add_argument("--variants", required=True)
    li.set_defaults(func=cmd_list)

    r = sub.add_parser("predict", help="score a variant split (GPU)")
    r.add_argument("--split-file", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--label-type", required=True)
    r.add_argument("--run-dir", default=None, help="fine_tune_classifier.py --output-dir (its best checkpoint)")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--splits", default=",".join(SCORED_SPLITS))
    r.add_argument("--conditioned", action="store_true", help="a region_model.py checkpoint (needs --cities-json)")
    r.add_argument("--cities-json", default=None)
    r.add_argument("--force-unk", action="store_true", help="score every row as the unknown region")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_predict)

    c = sub.add_parser("calibrate", help="arm (d): pooled vs local maps (CPU)")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--label-type", required=True)
    c.add_argument("--priors", required=True, help="deployment_priors.json")
    c.add_argument("--cities-json", required=True)
    c.add_argument("--seeds", default="42")
    c.add_argument("--base-seed", type=int, default=42)
    c.add_argument("--out", default=None, help="default $DATA/<LT>/region/calibration.json")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("report", help="tables and the doc's generated blocks (CPU)")
    t.add_argument("--data-dir", default=None)
    t.add_argument("--label-types", default="CurbRamp,Crosswalk,NoCurbRamp,Obstacle,SurfaceProblem")
    t.add_argument("--cities-json", default=None, help="the listing the run used (countries)")
    t.add_argument("--public-cities", default=None,
                   help="a saved /v3/api/cities response for naming (default: the live API at report time)")
    t.add_argument("--seeds", default="42,43,44")
    t.add_argument("--base-seed", type=int, default=42)
    t.add_argument("--within-min-n", type=int, default=WITHIN_MIN_N)
    t.add_argument("--n-boot", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--jobs-file", default=None, help="region_jobs.txt from analysis/region_submit.sh (job ids)")
    t.add_argument("--out", default="docs/region-transfer-2026-11")
    t.add_argument("--doc", default="docs/region-transfer-2026-11.md")
    t.add_argument("--check", action="store_true", help="fail when the doc differs from the committed results.json")
    t.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "predict":
        if not args.run_dir and not args.checkpoint:
            raise SystemExit("predict needs --run-dir or --checkpoint")
        if args.conditioned and not args.cities_json:
            raise SystemExit("predict --conditioned needs --cities-json")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
