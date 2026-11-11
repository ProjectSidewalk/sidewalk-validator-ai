"""The NoSidewalk serving bar's gate half: 5-fold out-of-fold Agree-gate precision (issue #42, step (a) of
``docs/nosidewalk-2026-11.md``).

The pilot could not evaluate the Agree verdict at the 0.92 gate: its calibration maps were fitted on about 70 Agree
verdicts of a 111-row val split, so every isotonic block was thin and the guard (``calibration/_isotonic_map.py``,
``THIN_BLOCK_N``) held the served value under the gate. This module pools the pilot's whole-view crops (train, val and
test: 758 rows in 450 face groups) into K face-grouped folds and scores every row out of fold, twice over:

1. ``folds``: fold ``k`` becomes ``test``, fold ``k+1`` becomes ``val`` (checkpoint selection) and the rest ``train``;
   ``test_heldout`` (the held-out cities) is left exactly as it was in every fold's split file. Faces never straddle
   folds (``analysis/confident_learning.assign_folds``, grouped by ``face_id``).
2. ``predict`` (GPU, one job per fold): the fold's model scores its ``test`` rows and ``test_heldout``.
3. ``report``: the pooled out-of-fold predictions get a leave-one-fold-out calibration: the map that serves fold
   ``k`` is fitted on the other folds' out-of-fold predictions under the face-size prior, with the Disagree verdict
   held back as the pilot did. So each row's verdict and served accuracy are both out of sample, for the model and
   for the map. The bar is then read on all pooled rows: within-city AUC >= 0.85 and the Agree verdict's gate
   precision, reweighted to the deployment prior as the map is, with a Wilson 95 % lower bound (on the Kish
   effective n) >= 0.92 on a raw count of >= 100 passing labels; the unweighted precision is reported beside it.
   ``test_heldout`` is reported per fold model under a map fitted on all pooled rows, plus the 5-model mean
   (descriptive only). A missing fold fails the bar whatever the numbers say.

What the number is: the map serving fold k is fitted on predictions from four *other* fold models, so the estimate is
for the recipe "a model of this kind plus a pooled out-of-fold map", not for one deployed model with its own map; a
final model trained on every row is sharper than the 60 % fold models; the pooled AUC mixes five score scales; and
folds are grouped by face, so a panorama carrying two faces can straddle folds (as the pilot split allows).

Only aggregates are written. ``--doc`` splices the markdown between the ``<!-- gate-results:start -->`` and
``<!-- gate-results:end -->`` markers of the study page.

Usage (klone; ``nosidewalk_gate_submit.sh`` runs the chain)::

    python analysis/nosidewalk_gate.py folds --split $DATA/whole/NoSidewalk/split.csv --out $GATE
    python analysis/nosidewalk_gate.py predict --gate $GATE --fold 0 --run-dir $RUNS/gate/fold0
    python analysis/nosidewalk_gate.py report --gate $GATE --priors $DATA/whole/deployment_priors.json \\
        --out-json $GATE/report/gate.json --out-md $GATE/report/gate.md [--doc docs/nosidewalk-2026-11.md]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import _isotonic_map as im  # noqa: E402
from calibration import calibrate as cal  # noqa: E402
from analysis.confident_learning import assign_folds, read_split, write_split  # noqa: E402
from analysis.nosidewalk_pilot import without_nan  # noqa: E402
from analysis.retrain_report import cluster_boot, within_city_auc  # noqa: E402

LABEL_TYPE = "NoSidewalk"
POOLED = ("train", "val", "test")
HELDOUT = "test_heldout"
GATE = im.GATE
Z95 = 1.96
BAR_AUC, BAR_PRECISION, BAR_N = 0.85, 0.92, 100
START, END = "<!-- gate-results:start -->", "<!-- gate-results:end -->"
PREDICTION_COLUMNS = ["label_uid", "fold", "split", "true", "p_correct", "p_incorrect"]
HOLD_BACK_REASON = ("NoSidewalk gate (issue 42): Disagree held back as for the five shipped types (issue 57)")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------------------------------------

def face_groups(df):
    """Resampling / fold group per row: ``face_id``, else ``pano:<pano_id>``, else the label itself."""
    face = df["face_id"].astype(str) if "face_id" in df else pd.Series("", index=df.index)
    pano = df["pano_id"].astype(str)
    group = face.where(face != "", "pano:" + pano)
    return group.where(pano != "", "uid:" + df["label_uid"])


def fold_split(base, fold_of, k, n_folds):
    """The split file of fold ``k``: pooled rows of fold ``k`` are ``test``, of fold ``(k + 1) % K`` ``val``, the
    rest ``train``; every other row (``test_heldout``) is unchanged.

    >>> b = pd.DataFrame({"label_uid": list("abcdef"), "split": ["train"] * 5 + ["test_heldout"]})
    >>> f = pd.Series([0, 1, 2, 0, 1], index=list("abcde"))
    >>> fold_split(b, f, 0, 3)["split"].tolist()
    ['test', 'val', 'train', 'test', 'val', 'test_heldout']
    """
    df = base.copy()
    pooled = df["split"].isin(POOLED)
    fold = df["label_uid"].map(fold_of)
    df.loc[pooled & fold.eq(k), "split"] = "test"
    df.loc[pooled & fold.eq((k + 1) % n_folds), "split"] = "val"
    df.loc[pooled & ~fold.isin([k, (k + 1) % n_folds]), "split"] = "train"
    return df


def cmd_folds(args):
    base = read_split(args.split)
    pooled = base[base["split"].isin(POOLED)]
    if pooled.empty:
        raise SystemExit(f"{args.split} has no train/val/test rows")
    if args.folds < 3:
        raise SystemExit("--folds must be at least 3 (one test fold, one val fold, at least one train fold)")
    group = face_groups(pooled)
    fold = assign_folds(pooled, group, args.folds, args.seed)
    straddle = pd.DataFrame({"g": group.to_numpy(), "f": fold.to_numpy()}).groupby("g")["f"].nunique()
    assert (straddle == 1).all(), "a face straddles folds"
    fold_of = pd.Series(fold.to_numpy(), index=pooled["label_uid"].to_numpy())
    # Absolute crop paths: the fold split files live outside the data dir, and predict resolves them as written.
    base_dir = os.path.dirname(os.path.abspath(args.split))
    base = base.assign(crop_path=[p if os.path.isabs(p) else os.path.join(base_dir, p) for p in base["crop_path"]])
    os.makedirs(args.out, exist_ok=True)
    folds = pd.DataFrame({"label_uid": fold_of.index, "fold": fold_of.to_numpy()}).sort_values("label_uid")
    folds_path = os.path.join(args.out, "folds.csv")
    folds.to_csv(folds_path, index=False, lineterminator="\n")
    per_fold = {}
    for k in range(args.folds):
        variant = fold_split(base, fold_of, k, args.folds)
        write_split(variant, os.path.join(args.out, f"fold{k}", "split.csv"))
        counts = variant.groupby(["split", "label_class"]).size()
        per_fold[k] = {f"{s}_{c}": int(n) for (s, c), n in counts.items()}
    meta = {"source_split": os.path.abspath(args.split), "source_sha256": sha256_of(args.split),
            "folds": args.folds, "seed": args.seed, "pooled_rows": int(len(pooled)),
            "pooled_faces": int(group.nunique()), "heldout_rows": int((base["split"] == HELDOUT).sum()),
            "rows_per_fold": per_fold, "folds_csv_sha256": sha256_of(folds_path)}
    with open(os.path.join(args.out, "folds.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {folds_path} ({len(pooled)} pooled rows on {meta['pooled_faces']} faces, {args.folds} folds) and "
          f"{args.folds} split files under {args.out}/")
    print(pd.DataFrame(per_fold).T.to_string())
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# predict (GPU)
# ---------------------------------------------------------------------------------------------------------------------

def predictions_path(gate, k):
    return os.path.join(gate, f"fold{k}", "predictions.csv")


def cmd_predict(args):
    import evaluate  # torch/transformers only when a model is loaded

    split = read_split(os.path.join(args.gate, f"fold{args.fold}", "split.csv"))
    rows = split[split["split"].isin(["test", HELDOUT])]
    if rows.empty:
        raise SystemExit(f"fold {args.fold} has no test rows")
    checkpoint = args.checkpoint or evaluate.best_checkpoint_in(args.run_dir)
    print(f"fold {args.fold}: {len(rows)} rows, checkpoint {checkpoint}")
    pred = evaluate.run_inference(rows, checkpoint, splits=("test", HELDOUT))
    out = pd.DataFrame({"label_uid": pred["label_uid"], "fold": args.fold, "split": pred["split"],
                        "true": pred["true"], "p_correct": pred["p_correct"], "p_incorrect": pred["p_incorrect"]},
                       columns=PREDICTION_COLUMNS)
    out = out.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    path = predictions_path(args.gate, args.fold)
    out.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path} ({len(out)} rows)")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def load_predictions(gate):
    """Every fold's predictions, joined with ``city`` and the face group from the fold's split file. Returns
    ``(pooled, heldout, n_folds, missing)``: pooled = out-of-fold ``test`` rows (one per label), heldout = every
    fold model's ``test_heldout`` rows."""
    with open(os.path.join(gate, "folds.json")) as f:
        n_folds = json.load(f)["folds"]
    frames, missing = [], []
    for k in range(n_folds):
        path = predictions_path(gate, k)
        if not os.path.exists(path):
            missing.append(path)
            continue
        pred = pd.read_csv(path, dtype={"label_uid": str})
        split = read_split(os.path.join(gate, f"fold{k}", "split.csv"))
        split = split.assign(group=face_groups(split))
        df = pred.merge(split[["label_uid", "city", "group"]], on="label_uid", how="left", validate="one_to_one")
        if df["city"].isna().any():
            raise ValueError(f"{path}: predictions without a split.csv row")
        frames.append(df)
    if not frames:
        return None, None, n_folds, missing
    df = pd.concat(frames, ignore_index=True)
    pooled = df[df["split"] == "test"].reset_index(drop=True)
    if not pooled["label_uid"].is_unique:
        raise ValueError("a label was scored out of fold more than once")
    return pooled, df[df["split"] == HELDOUT].reset_index(drop=True), n_folds, missing


def fit_map(fit_df, pi_deploy):
    """The served map fitted on ``fit_df`` (rows with ``p_correct``/``p_incorrect``/``true``) under ``pi_deploy``,
    Disagree held back as the pilot did (``cal.hold_back``)."""
    curves, _ = cal.fit_curves(cal.verdicts(fit_df), pi_deploy)
    api_map = cal.hold_back(im.build_api_map(curves), ["incorrect"])
    im.validate_api_map(api_map)
    return api_map


def served_out_of_fold(pooled, pi_deploy):
    """Served accuracy per pooled row from a map fitted on the OTHER folds (leave-one-fold-out), with the verdict.

    >>> rng = np.random.default_rng(0)
    >>> y = np.array(["correct"] * 200 + ["incorrect"] * 100)
    >>> p = np.clip(np.where(y == "correct", 0.9, 0.2) + rng.normal(0, 0.08, 300), 0.01, 0.99)
    >>> d = pd.DataFrame({"p_correct": p, "p_incorrect": 1 - p, "true": y, "fold": np.arange(300) % 3})
    >>> s = served_out_of_fold(d, {"correct": 0.83, "incorrect": 0.17})
    >>> sorted(s.columns.tolist())[:3], len(s)
    (['confidence', 'correct', 'fold'], 300)
    """
    df = cal.verdicts(pooled)
    served = np.zeros(len(df))
    for k in sorted(df["fold"].unique()):
        fit_df, mask = df[df["fold"] != k], (df["fold"] == k).to_numpy()
        api_map = fit_map(fit_df, pi_deploy)
        for v in im.API_CLASSES:
            m = mask & (df["verdict"] == v).to_numpy()
            served[m] = im.api_lookup_many(api_map[v], df["confidence"].to_numpy()[m])
    return df.assign(served=served)


def kish_n(w):
    """Kish effective sample size of weights ``w`` (``(sum w)^2 / sum w^2``); 0 for no weights.

    >>> kish_n(np.array([1.0, 1.0, 1.0])), round(kish_n(np.array([1.0, 3.0])), 3)
    (3.0, 1.6)
    """
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def gate_stats(df, pi_deploy=None, gate=GATE):
    """The Agree verdict at ``gate`` on rows that carry ``verdict`` and ``served``: raw count, coverage and
    precision on the rows as they are, and, with ``pi_deploy``, the precision reweighted to the deployment prior
    (``calibrate.prior_weights``, the weighting the map itself was fitted under) with its Wilson 95 % lower bound on
    the Kish effective n. The bar reads the weighted bound and the raw count. The weighted fields are NaN when the
    rows lack a class (a city with only correct labels cannot be reweighted).

    >>> d = pd.DataFrame({"verdict": ["correct"] * 4, "served": [0.95, 0.93, 0.5, 0.99],
    ...                   "true": ["correct", "incorrect", "correct", "correct"]})
    >>> g = gate_stats(d, {"correct": 0.9, "incorrect": 0.1})
    >>> g["n_pass"], round(g["precision"], 3), round(g["precision_weighted"], 3), round(g["kish_n"], 1)
    (3, 0.667, 0.857, 2.6)
    """
    passed = ((df["verdict"] == "correct") & (df["served"] >= gate)).to_numpy()
    correct = (df["true"].to_numpy() == "correct").astype(float)
    n_pass = int(passed.sum())
    out = {"n": int(len(df)), "n_agree_verdicts": int((df["verdict"] == "correct").sum()), "n_pass": n_pass,
           "coverage": n_pass / len(df) if len(df) else float("nan"),
           "precision": float(correct[passed].mean()) if n_pass else float("nan"),
           "precision_wilson95_lower": float(im.wilson_lower(correct[passed].mean(), n_pass, z=Z95)) if n_pass
           else float("nan"),
           "precision_weighted": float("nan"), "kish_n": float("nan"), "precision_weighted_wilson95_lower": float("nan")}
    if pi_deploy is not None and n_pass:
        try:
            w = cal.prior_weights(df["true"], pi_deploy)
        except ValueError:  # a class is absent from these rows
            return out
        out["precision_weighted"] = float(np.average(correct[passed], weights=w[passed]))
        out["kish_n"] = kish_n(w[passed])
        out["precision_weighted_wilson95_lower"] = float(im.wilson_lower(out["precision_weighted"], out["kish_n"],
                                                                         z=Z95))
    return out


def served_with_map(df, api_map):
    """``verdict``, ``confidence`` and ``served`` (from ``api_map``) for rows with ``p_correct``/``p_incorrect``."""
    df = cal.verdicts(df)
    served = np.zeros(len(df))
    for v in im.API_CLASSES:
        m = (df["verdict"] == v).to_numpy()
        served[m] = im.api_lookup_many(api_map[v], df["confidence"].to_numpy()[m])
    return df.assign(served=served)


def auc_block(df, n_boot, seed):
    y = (df["true"] == "incorrect").astype(int).to_numpy()
    out = {"n": int(len(df)), "n_incorrect": int(y.sum()), "faces": int(df["group"].nunique())}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, df["p_incorrect"]))
        boot = cluster_boot(y, df["p_incorrect"].to_numpy(), df["group"].to_numpy(), n_boot, seed)[:, 0]
        boot = boot[np.isfinite(boot)]
        out["auc_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else None
    else:
        out["auc"], out["auc_ci95"] = float("nan"), None
    out["within_city_auc"], out["within_city_n_cities"] = within_city_auc(y, df["p_incorrect"], df["city"])
    return out


def bar_verdict(block, complete=True):
    """The bar on a pooled block: within-city AUC >= ``BAR_AUC``, and the Agree gate's prior-weighted Wilson lower
    bound >= ``BAR_PRECISION`` on a raw count of >= ``BAR_N`` passing labels. ``complete=False`` (a fold's
    predictions missing) fails it whatever the numbers say.

    >>> g = {"n_pass": 120, "precision_weighted_wilson95_lower": 0.93}
    >>> bar_verdict({"within_city_auc": 0.9, "agree_gate": g})["passes"]
    True
    >>> bar_verdict({"within_city_auc": 0.9, "agree_gate": {**g, "n_pass": 99}})["gate_ok"]
    False
    >>> bar_verdict({"within_city_auc": 0.9, "agree_gate": g}, complete=False)["passes"]
    False
    """
    g = block["agree_gate"]
    auc = block["within_city_auc"]
    lower = g["precision_weighted_wilson95_lower"]
    out = {"within_city_auc_min": BAR_AUC, "gate_precision_min": BAR_PRECISION, "gate_n_min": BAR_N,
           "precision_basis": "prior-weighted (calibrate.prior_weights), Wilson 95 % on the Kish effective n",
           "complete": bool(complete),
           "auc_ok": bool(auc >= BAR_AUC) if auc is not None and np.isfinite(auc) else False,
           "gate_ok": bool(g["n_pass"] >= BAR_N and lower is not None and np.isfinite(lower)
                           and lower >= BAR_PRECISION)}
    out["passes"] = bool(complete and out["auc_ok"] and out["gate_ok"])
    return out


def pooled_report(pooled, pi_deploy, n_boot=1000, seed=0, min_city_n=10, complete=True):
    """The bar on the pooled out-of-fold rows: AUCs, the leave-one-fold-out gate, per city and per fold."""
    served = served_out_of_fold(pooled, pi_deploy)
    out = auc_block(pooled, n_boot, seed)
    out["agree_gate"] = gate_stats(served, pi_deploy)
    out["per_fold"] = {int(k): gate_stats(served[served["fold"] == k], pi_deploy)
                       for k in sorted(served["fold"].unique())}
    per_city = {}
    for city, part in served.groupby("city"):
        if len(part) >= min_city_n:
            per_city[city] = gate_stats(part, pi_deploy)
    out["per_city_gate"] = per_city
    out["bar"] = bar_verdict(out, complete)
    return out


def heldout_report(pooled, heldout, pi_deploy, n_boot=1000, seed=0):
    """Each fold model on the held-out cities under a map fitted on all pooled rows, plus the mean of the fold
    models' probabilities. The mean row is descriptive only: averaging compresses confidence, so a map fitted on
    single-model confidences is mis-specified for it."""
    api_map = fit_map(pooled, pi_deploy)
    out = {"map_fit_n": int(len(pooled)), "per_fold": {}}
    for k, part in heldout.groupby("fold"):
        block = auc_block(part, n_boot, seed)
        block["agree_gate"] = gate_stats(served_with_map(part, api_map), pi_deploy)
        out["per_fold"][int(k)] = block
    mean = (heldout.groupby(["label_uid", "city", "group", "true"], as_index=False)[["p_correct", "p_incorrect"]]
            .mean())
    block = auc_block(mean, n_boot, seed)
    block["agree_gate"] = gate_stats(served_with_map(mean, api_map), pi_deploy)
    block["models"] = int(heldout["fold"].nunique())
    block["note"] = "descriptive: the single-model map is mis-specified for averaged probabilities"
    out["mean_of_models"] = block
    return out


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


GATE_HEAD = ("Agree gate n (coverage) | precision, prior-weighted (Wilson 95 % lower; Kish n) | "
             "precision, unweighted (Wilson 95 % lower)")


def _gate_cell(g):
    """Three cells: raw count (coverage), the weighted precision with its bound and Kish n, the unweighted one."""
    if not g["n_pass"]:
        return "0 | n/a | n/a"
    weighted = (f"{_fmt(g['precision_weighted'])} ({_fmt(g['precision_weighted_wilson95_lower'])}; "
                f"{_fmt(g['kish_n'], 0)})") if np.isfinite(g.get("precision_weighted", float("nan"))) else "n/a (one class)"
    return (f"{g['n_pass']} ({_fmt(g['coverage'])}) | {weighted} | "
            f"{_fmt(g['precision'])} ({_fmt(g['precision_wilson95_lower'])})")


def markdown(rep):
    p, h = rep["pooled"], rep["heldout"]
    bar, g = p["bar"], p["agree_gate"]
    lines = [f"Pooled out-of-fold rows: {p['n']} ({p['n_incorrect']} incorrect) in {p['faces']} face groups, "
             f"{rep['folds']} folds; the map serving each fold is fitted on the other folds' out-of-fold predictions "
             f"under the face-size prior (P(correct) = {rep['prior']['correct']:.3f}), Disagree held back, thin-block "
             "guard on. Precision is reported both reweighted to that prior (the basis the map is fitted under, "
             "comparable with `calibration_report.json`) and unweighted on the pooled rows' own class mix.", "",
             f"| rows | AUC [95 % CI, face resampling] | within-city AUC (cities) | {GATE_HEAD} |",
             "|---|---|---|---|---|---|"]
    ci = f" [{p['auc_ci95'][0]:.3f}, {p['auc_ci95'][1]:.3f}]" if p.get("auc_ci95") else ""
    lines.append(f"| pooled | {_fmt(p['auc'])}{ci} | {_fmt(p['within_city_auc'])} ({p['within_city_n_cities']}) | "
                 f"{_gate_cell(g)} |")
    lines += ["", f"**Bar:** within-city AUC >= {BAR_AUC} is {'met' if bar['auc_ok'] else 'NOT met'}; the Agree "
              f"gate's prior-weighted Wilson 95 % lower bound >= {BAR_PRECISION} on >= {BAR_N} passing labels is "
              f"{'met' if bar['gate_ok'] else 'NOT met'} ({g['n_pass']} labels pass of {g['n_agree_verdicts']} Agree "
              f"verdicts; weighted lower bound {_fmt(g['precision_weighted_wilson95_lower'])}, unweighted "
              f"{_fmt(g['precision_wilson95_lower'])})."
              + ("" if bar["complete"] else " **Incomplete: a fold's predictions are missing, so the bar fails.**")
              + f" **{'PASSES' if bar['passes'] else 'DOES NOT PASS'}.**", "",
              "Per fold (each fold's map fitted on the other folds):", "",
              f"| fold | rows | {GATE_HEAD} |", "|---|---|---|---|---|"]
    for k, gk in p["per_fold"].items():
        lines.append(f"| {k} | {gk['n']} | {_gate_cell(gk)} |")
    lines += ["", "Per city (>= 10 pooled rows):", "", f"| city | rows | {GATE_HEAD} |", "|---|---|---|---|---|"]
    for city, gc in sorted(p["per_city_gate"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {city} | {gc['n']} | {_gate_cell(gc)} |")
    lines += ["", f"Held-out cities (`test_heldout`), each fold model under a map fitted on all {h['map_fit_n']} "
              "pooled rows; the mean-of-models row is descriptive only (averaged probabilities are not what the map "
              "was fitted on):", "",
              f"| model | rows (incorrect) | AUC | within-city AUC | {GATE_HEAD} |", "|---|---|---|---|---|---|---|"]
    for k, b in h["per_fold"].items():
        lines.append(f"| fold {k} | {b['n']} ({b['n_incorrect']}) | {_fmt(b['auc'])} | {_fmt(b['within_city_auc'])} | "
                     f"{_gate_cell(b['agree_gate'])} |")
    b = h["mean_of_models"]
    lines.append(f"| mean of {b['models']} models | {b['n']} ({b['n_incorrect']}) | {_fmt(b['auc'])} | "
                 f"{_fmt(b['within_city_auc'])} | {_gate_cell(b['agree_gate'])} |")
    if rep["missing"]:
        lines += ["", "Missing fold predictions: " + ", ".join(rep["missing"])]
    return "\n".join(lines) + "\n"


def splice(doc_text, block):
    """Replace what lies between the gate markers of ``doc_text`` with ``block``; ValueError without markers.

    >>> splice("a\\n<!-- gate-results:start -->\\nold\\n<!-- gate-results:end -->\\nb", "new\\n")
    'a\\n<!-- gate-results:start -->\\nnew\\n<!-- gate-results:end -->\\nb'
    """
    i, j = doc_text.find(START), doc_text.find(END)
    if i < 0 or j < i:
        raise ValueError(f"no {START} ... {END} markers in the document")
    return doc_text[:i + len(START)] + "\n" + block + doc_text[j:]


def cmd_report(args):
    pooled, heldout, n_folds, missing = load_predictions(args.gate)
    if pooled is None:
        raise SystemExit(f"no fold under {args.gate} has predictions (looked for {', '.join(missing)})")
    folds = pd.read_csv(os.path.join(args.gate, "folds.csv"), dtype={"label_uid": str})
    unscored = sorted(set(folds["label_uid"]) - set(pooled["label_uid"]))
    complete = not missing and not unscored
    if not complete:
        print(f"WARNING: {len(missing)} fold(s) without predictions and {len(unscored)} pooled rows unscored; "
              "the bar cannot pass on an incomplete pool", file=sys.stderr)
    pi_deploy, entry = cal.load_prior(args.priors, args.label_type)
    rep = {"label_type": args.label_type, "folds": n_folds, "gate": GATE, "prior": pi_deploy,
           "prior_source": entry.get("source"), "hold_back": {"verdicts": ["incorrect"], "reason": HOLD_BACK_REASON},
           "pooled": pooled_report(pooled, pi_deploy, args.n_boot, args.seed, complete=complete),
           "heldout": heldout_report(pooled, heldout, pi_deploy, args.n_boot, args.seed)
           if heldout is not None and len(heldout) else None,
           "missing": missing, "unscored_pooled_rows": len(unscored)}
    for path in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(without_nan(rep), f, indent=1)
        f.write("\n")
    md = markdown(rep) if rep["heldout"] else markdown({**rep, "heldout": {"map_fit_n": len(pooled), "per_fold": {},
                                                        "mean_of_models": {"models": 0, "n": 0, "n_incorrect": 0,
                                                                           "auc": float("nan"),
                                                                           "within_city_auc": float("nan"),
                                                                           "agree_gate": {"n_pass": 0}}}})
    with open(args.out_md, "w") as f:
        f.write(md)
    if args.doc:
        with open(args.doc) as f:
            text = f.read()
        with open(args.doc, "w") as f:
            f.write(splice(text, md))
    print(md)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("folds", help="K face-grouped folds over the pooled train/val/test rows, one split file each")
    f.add_argument("--split", required=True, help="the pilot's whole-view NoSidewalk/split.csv")
    f.add_argument("--out", required=True, help="the gate directory: folds.csv, folds.json, fold<k>/split.csv")
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(func=cmd_folds)
    r = sub.add_parser("predict", help="score one fold's test and test_heldout rows with its model (GPU)")
    r.add_argument("--gate", required=True)
    r.add_argument("--fold", type=int, required=True)
    r.add_argument("--run-dir", default=None, help="fine_tune_classifier.py --output-dir of this fold's model")
    r.add_argument("--checkpoint", default=None, help="a checkpoint directory instead of --run-dir's best")
    r.set_defaults(func=cmd_predict)
    s = sub.add_parser("report", help="pool the folds, leave-one-fold-out calibration, the bar, markdown")
    s.add_argument("--gate", required=True)
    s.add_argument("--priors", required=True, help="the pilot's whole/deployment_priors.json (face-size prior)")
    s.add_argument("--label-type", default=LABEL_TYPE)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-json", required=True)
    s.add_argument("--out-md", required=True)
    s.add_argument("--doc", default=None, help="splice the markdown into this page between the gate markers")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "predict" and not (args.run_dir or args.checkpoint):
        build_parser().error("predict needs --run-dir or --checkpoint")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
