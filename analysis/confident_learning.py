"""Wrong crowd targets by out-of-fold contradiction (confident learning), and the stricter-target rules, as split
variants of one label type's ``split.csv``.

Why
---
The crowd margin rule that sets a training target (``incorrect`` at net Disagree > 1, ``correct`` at net Agree > 2,
``data_generation.eligible_labels``) matches an expert only about three times in four on contested labels
(https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66), so the models learn the crowd's Disagree habits.
This module makes variants of a split that change only its **train** rows (val, test and test_heldout stay byte for
byte the same, so every variant is scored on the same rows as the round-3 baseline):

- ``folds``: k panorama-grouped folds over the train rows, one split file per fold with that fold held out
  (``split = fold_out``), so k models trained on them give every train row an **out-of-fold** score from a model
  that never saw it (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76).
- ``predict``: score one fold's held-out train rows with the model trained on the other folds.
- ``flag``: the confident joint of Northcutt, Jiang and Chuang (JAIR 2021, doi:10.1613/jair.1.12125) over the
  out-of-fold scores, in NumPy: per class ``c`` a threshold ``t_c`` = the mean out-of-fold ``p(c)`` over rows whose
  target is ``c``; a row counts for class ``j`` when ``p(j) >= t_j`` (the largest such ``p`` if both classes
  qualify); the off-diagonal cells of that 2x2 count are the estimated number of wrong targets per (given, likely)
  pair, and the rows flagged are, per cell, the ``count`` rows with that given target and the highest ``p(likely)``
  (the ``prune_by_noise_rate`` ranking of cleanlab, on the **uncalibrated** joint: cleanlab first rescales each
  row of the joint to the given class's count, which here would flag 1-9% more rows; not done, so the flag counts
  are a lower bound of its). Writes ``noisy_targets.csv`` and two more variants: ``cl-drop`` (flagged train rows
  left out) and ``cl-flip`` (their target flipped).
- ``rules``: the stricter crowd rules of https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/75 applied
  to the train rows: ``symmetric`` (``incorrect`` needs net Disagree > 2, as ``correct`` needs net Agree > 2) and
  ``unanimous3`` (at least three votes, none the other way; Unsure votes are not counted in the manifest, so they
  are ignored). Rows that fail the rule are kept in the file as ``split = train_dropped`` so the counts are
  auditable; ``fine_tune_classifier.py`` trains on ``split == train`` only. Expert-sourced targets
  (``class_source == expert``) are not touched by either rule.
- ``prior``: what each rule does to the deployment prior (``data_generation.deployment_prior_entry``: the
  pre-downsampling class counts of human-authored labels meeting the rule, from the cached rawLabels), for the
  second question of https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56.
- ``report``: one table per label type from the ``evaluate.py`` outputs of the baseline (``metrics.json``,
  ``predictions.csv``) and of each variant (``metrics_<variant>.json``, ``predictions_<variant>.csv``): test and
  held-out AUC, the within-city AUC of ``docs/retrain-2026-10.md`` (from the predictions, via
  ``analysis/retrain_report.py``), the hard slice, recall at precision 0.92; the AUC per held-out deployment
  (private ones lettered as in the round-3 report); the severity x tag slices of every variant against the
  baseline (``analysis/slices.py`` paired panorama bootstrap, ``slices.md``); the train-row accounting of every
  variant; and, when a private expert-rated file is given, how the dropped or flagged rows agree with the expert
  (aggregates only; the per-row file is never written into the repo). Seed replicates (``metrics_<variant>-s<seed>.json``
  from the Slurm script's ``TAG``) go to ``seeds.md``: each paired with the baseline of the same seed, and the
  across-seed mean, sd and range.

Every variant split is written by the same writer, keeps the manifest columns as strings exactly as read, and is
deterministic (sorted by ``label_uid``; the fold shuffle is seeded). Crop paths are left relative, so a variant
file under ``<label_type>/cl/<variant>/`` shares the crops with the production split (``fine_tune_classifier.py
--split`` resolves them under ``<data-dir>/<label_type>/``).

Usage (paths as on the October 2026 retrain, ``DATA=retrain-2026-10``)::

    python analysis/confident_learning.py folds --split $DATA/CurbRamp/split.csv --rawlabels rawlabels \\
        --folds 5 --seed 42 --merge-radius-m 3 --out $DATA/CurbRamp/cl
    python analysis/confident_learning.py rules --split $DATA/CurbRamp/split.csv --rule symmetric \\
        --out $DATA/CurbRamp/cl
    # on a GPU, once fold k's model has trained with --split $DATA/CurbRamp/cl/fold<k>/split.csv:
    python analysis/confident_learning.py predict --split $DATA/CurbRamp/split.csv --folds-file \\
        $DATA/CurbRamp/cl/folds.csv --fold 0 --run-dir $SCRATCH/CurbRamp/fold0 --out $DATA/CurbRamp/cl/oof_fold0.csv
    python analysis/confident_learning.py flag --split $DATA/CurbRamp/split.csv \\
        --oof $DATA/CurbRamp/cl/oof_fold*.csv --out $DATA/CurbRamp/cl
    python analysis/confident_learning.py report --data-dir $DATA --label-types CurbRamp,Crosswalk \\
        --variants symmetric,unanimous3,cl-drop,cl-flip --out docs/confident-learning-2026-10
    python analysis/confident_learning.py prior --rawlabels-cache rawlabels --cities "$(cat rawlabels/cities.txt)" \\
        --out $DATA/cl_priors.json
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import split_data  # noqa: E402

CLASSES = ("correct", "incorrect")
FOLD_OUT = "fold_out"          # the split value of a fold's held-out train rows in that fold's split file
TRAIN_DROPPED = "train_dropped"  # train rows a rule or the flag left out (kept in the file, never trained on)
RULES = ("symmetric", "unanimous3")
CL_VARIANTS = ("cl-drop", "cl-flip")
DEFAULT_VARIANTS = RULES + CL_VARIANTS
OOF_COLUMNS = ["label_uid", "fold", "label_class", "p_correct", "p_incorrect"]
NOISY_COLUMNS = ["label_uid", "city", "label_class", "class_source", "author", "human_agree", "human_disagree",
                 "fold", "p_incorrect", "likely_class", "flagged"]


# ---------------------------------------------------------------------------------------------------------------------
# Split files
# ---------------------------------------------------------------------------------------------------------------------

def read_split(path):
    """``split.csv`` as strings, exactly as written (every cell is written back unchanged)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("label_uid", "pano_id", "city", "label_class", "split", "crop_path"):
        if col not in df.columns:
            raise ValueError(f"{path} has no {col} column")
    if not df["label_uid"].is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    return df


def write_split(df, path):
    """Write a variant split: sorted by ``label_uid``, LF line ends, via a temporary file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = df.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)
    return path


def check_eval_rows_unchanged(base, variant):
    """Raise unless every non-train row of ``base`` is in ``variant`` unchanged (the whole point of a variant)."""
    keep = ~base["split"].isin(["train"])
    a = base[keep].sort_values("label_uid").reset_index(drop=True)
    b = variant[variant["label_uid"].isin(a["label_uid"])].sort_values("label_uid").reset_index(drop=True)
    if len(a) != len(b) or not a.equals(b[a.columns]):
        raise AssertionError("a variant must leave every val/test/test_heldout row exactly as it was")


def sha256_of_text(text):
    """sha256 of a string (utf-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------------------------------------

def train_groups(df, rawlabels=None, merge_radius_m=0.0):
    """Group key per row of ``df`` (train rows), the way ``split_data.assign_splits`` groups: by ``pano_id`` (a row
    without one is its own group), then merged within ``merge_radius_m`` metres using the coordinates
    ``split_data.load_coordinates`` finds (the manifest's own columns or the rawLabels cache)."""
    pano = df["pano_id"].astype(str)
    group = pano.where(pano.str.len() > 0, "uid:" + df["label_uid"])
    if merge_radius_m > 0:
        coords = split_data.load_coordinates(df.assign(status="ok"), rawlabels)
        xy = coords.loc[df["label_uid"]]
        group, _ = split_data.merge_nearby_groups(group, xy["latitude"].to_numpy(), xy["longitude"].to_numpy(),
                                                  merge_radius_m)
    return pd.Series(np.asarray(group), index=df.index)


def assign_folds(train, group, k, seed):
    """Fold (0..k-1) per train row: groups never straddle folds; within each class-ratio bin
    (``round(4 * share_incorrect)``, as the split does) groups are shuffled with the seed and each goes to the fold
    with the largest deficit against an equal share of that bin's rows. Returns a Series aligned with ``train``.

    >>> t = pd.DataFrame({"label_uid": list("abcdef"), "label_class": ["correct"] * 3 + ["incorrect"] * 3})
    >>> g = pd.Series(["p1", "p1", "p2", "p3", "p3", "p4"], index=t.index)
    >>> f = assign_folds(t, g, 2, 0)
    >>> (f[g == "p1"].nunique(), f[g == "p3"].nunique(), sorted(int(x) for x in f.unique()))
    (1, 1, [0, 1])
    """
    if k < 2:
        raise ValueError("--folds must be at least 2")
    g = pd.DataFrame({"group": group.to_numpy(), "incorrect": (train["label_class"] == "incorrect").astype(int)})
    groups = g.groupby("group", sort=True).agg(n=("incorrect", "size"), inc=("incorrect", "sum"))
    groups["bin"] = np.rint(split_data.N_BINS * groups["inc"] / groups["n"]).astype(int)
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, members in groups.groupby("bin", sort=True):
        order = members.index.to_numpy()[rng.permutation(len(members))]
        total = members["n"].sum()
        filled = np.zeros(k, dtype=int)
        for gid in order:
            fold = int(np.argmax(total / k - filled))  # the first fold on ties, so the run is deterministic
            assignment[gid] = fold
            filled[fold] += members.at[gid, "n"]
    return pd.Series([assignment[x] for x in g["group"]], index=train.index, dtype=int)


def fold_split(base, folds, k):
    """The split file for fold ``k``: train rows of that fold become ``fold_out``; everything else is unchanged."""
    df = base.copy()
    out = df["label_uid"].map(folds).eq(k) & (df["split"] == "train")
    df.loc[out, "split"] = FOLD_OUT
    return df


def cmd_folds(args):
    base = read_split(args.split)
    train = base[base["split"] == "train"]
    if train.empty:
        raise SystemExit(f"{args.split} has no train rows")
    group = train_groups(train, args.rawlabels, args.merge_radius_m)
    fold = assign_folds(train, group, args.folds, args.seed)
    folds = pd.DataFrame({"label_uid": train["label_uid"].to_numpy(), "fold": fold.to_numpy()})
    folds = folds.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "folds.csv")
    folds.to_csv(path, index=False, lineterminator="\n")
    by_fold = train.assign(fold=fold.to_numpy()).groupby("fold")["label_class"].value_counts().unstack(fill_value=0)
    # No panorama (or merged group) may straddle folds.
    straddle = train.assign(fold=fold.to_numpy(), group=group.to_numpy()).groupby("group")["fold"].nunique()
    assert (straddle == 1).all(), "a group straddles folds"
    mapping = folds.set_index("label_uid")["fold"]
    written = []
    for k in range(args.folds):
        variant = fold_split(base, mapping, k)
        check_eval_rows_unchanged(base, variant)
        written.append(write_split(variant, os.path.join(args.out, f"fold{k}", "split.csv")))
    meta = {"source_split": os.path.abspath(args.split), "source_sha256": sha256_of(args.split),
            "folds": args.folds, "seed": args.seed, "merge_radius_m": args.merge_radius_m,
            "rows_per_fold": {int(k): {c: int(v) for c, v in row.items()} for k, row in by_fold.iterrows()},
            "groups": int(group.nunique()), "folds_csv_sha256": sha256_of(path)}
    with open(os.path.join(args.out, "folds.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {path} ({len(folds)} train rows, {meta['groups']} groups, {args.folds} folds) and "
          f"{len(written)} fold split files under {args.out}/")
    print(by_fold.to_string())
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# predict (GPU)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_predict(args):
    import evaluate  # torch/transformers only when a model is loaded

    base = read_split(args.split)
    folds = pd.read_csv(args.folds_file, dtype={"label_uid": str})
    uids = set(folds.loc[folds["fold"] == args.fold, "label_uid"])
    rows = base[(base["split"] == "train") & base["label_uid"].isin(uids)].copy()
    if rows.empty:
        raise SystemExit(f"no train rows in fold {args.fold}")
    base_dir = os.path.dirname(os.path.abspath(args.split))
    rows["crop_path"] = [p if os.path.isabs(p) else os.path.join(base_dir, p) for p in rows["crop_path"]]
    checkpoint = args.checkpoint or evaluate.best_checkpoint_in(args.run_dir)
    print(f"fold {args.fold}: {len(rows)} rows, checkpoint {checkpoint}")
    pred = evaluate.run_inference(rows, checkpoint, splits=("train",))
    out = pd.DataFrame({"label_uid": pred["label_uid"], "fold": args.fold, "label_class": pred["true"],
                        "p_correct": pred["p_correct"], "p_incorrect": pred["p_incorrect"]}, columns=OOF_COLUMNS)
    out = out.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out} ({len(out)} rows)")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# flag: the confident joint
# ---------------------------------------------------------------------------------------------------------------------

def confident_joint(y, p, classes=CLASSES):
    """``(joint, thresholds)``: the confident joint (a ``len(classes)`` square array, rows = given class, columns =
    likely class) and the per-class thresholds ``t_c = mean(p[:, c] | y == c)``.

    A row counts for class ``j`` when ``p[i, j] >= t_j``; with several qualifying classes, the one with the largest
    ``p``; with none, the row is not counted at all.

    >>> y = np.array(["correct", "correct", "correct", "incorrect"])
    >>> p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])
    >>> joint, t = confident_joint(y, p)
    >>> joint.tolist(), [round(float(x), 4) for x in t]
    ([[2, 1], [0, 1]], [0.6333, 0.7])
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    thresholds = np.array([p[y == c, i].mean() if (y == c).any() else np.inf for i, c in enumerate(classes)])
    qualifies = p >= thresholds[None, :]
    masked = np.where(qualifies, p, -np.inf)
    likely = masked.argmax(axis=1)
    counted = qualifies.any(axis=1)
    joint = np.zeros((len(classes), len(classes)), dtype=int)
    given = np.array([classes.index(c) for c in y])
    for g, l in zip(given[counted], likely[counted]):
        joint[g, l] += 1
    return joint, thresholds


def flag_rows(y, p, classes=CLASSES):
    """``(flagged, likely)``: per row, whether it is a label issue and the class it most likely belongs to.

    For every off-diagonal cell ``(given, likely)`` of the confident joint with count ``n``, the ``n`` rows with that
    given class and the highest ``p[:, likely]`` are flagged (ties broken by row order, so the result is
    deterministic). ``likely`` is ``classes[argmax p]`` for flagged rows and the given class otherwise.

    >>> y = np.array(["correct", "correct", "correct", "incorrect"])
    >>> p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])
    >>> f, l = flag_rows(y, p)
    >>> f.tolist(), l.tolist()
    ([False, False, True, False], ['correct', 'correct', 'incorrect', 'incorrect'])
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    joint, _ = confident_joint(y, p, classes)
    flagged = np.zeros(len(y), dtype=bool)
    likely = y.astype(object).copy()
    for g, given in enumerate(classes):
        for l, other in enumerate(classes):
            if g == l or joint[g, l] == 0:
                continue
            idx = np.flatnonzero(y == given)
            order = idx[np.argsort(-p[idx, l], kind="stable")][: joint[g, l]]
            flagged[order] = True
            likely[order] = other
    return flagged, np.asarray(likely, dtype=object)


def cl_variants(base, flagged_uids):
    """``{"cl-drop": df, "cl-flip": df}``: train rows in ``flagged_uids`` left out (``train_dropped``) or with their
    ``label_class`` flipped (``class_source`` becomes ``cl_flip``); crop paths are unchanged."""
    hit = (base["split"] == "train") & base["label_uid"].isin(flagged_uids)
    drop = base.copy()
    drop.loc[hit, "split"] = TRAIN_DROPPED
    flip = base.copy()
    flip.loc[hit, "label_class"] = flip.loc[hit, "label_class"].map({"correct": "incorrect", "incorrect": "correct"})
    if "class_source" in flip.columns:
        flip.loc[hit, "class_source"] = "cl_flip"
    return {"cl-drop": drop, "cl-flip": flip}


def _shares(df, by):
    """Flagged count and share per value of ``by`` (a DataFrame with ``n``, ``flagged``, ``share``)."""
    g = df.groupby(by)["flagged"].agg(n="size", flagged="sum")
    g["share"] = (g["flagged"] / g["n"]).round(4)
    return g.reset_index()


def cmd_flag(args):
    base = read_split(args.split)
    paths = sorted(p for pattern in args.oof for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no out-of-fold files match {args.oof}")
    oof = pd.concat([pd.read_csv(p, dtype={"label_uid": str}) for p in paths], ignore_index=True)
    if not oof["label_uid"].is_unique:
        raise SystemExit("a label_uid appears in more than one out-of-fold file")
    train = base[base["split"] == "train"]
    missing = set(train["label_uid"]) - set(oof["label_uid"])
    extra = set(oof["label_uid"]) - set(train["label_uid"])
    if missing or extra:
        raise SystemExit(f"out-of-fold rows do not match the train rows: {len(missing)} train rows unscored, "
                         f"{len(extra)} scored rows not in train")
    df = train.merge(oof[["label_uid", "fold", "p_correct", "p_incorrect"]], on="label_uid", validate="one_to_one")
    y = df["label_class"].to_numpy()
    p = df[["p_correct", "p_incorrect"]].to_numpy(dtype=float)
    joint, thresholds = confident_joint(y, p)
    flagged, likely = flag_rows(y, p)
    df["likely_class"] = likely
    df["flagged"] = flagged.astype(int)
    df["p_incorrect"] = df["p_incorrect"].astype(float).round(6)
    for col in ("class_source", "author", "human_agree", "human_disagree"):
        if col not in df.columns:
            df[col] = ""
    os.makedirs(args.out, exist_ok=True)
    noisy = df[NOISY_COLUMNS].sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    noisy_path = os.path.join(args.out, "noisy_targets.csv")
    noisy.to_csv(noisy_path, index=False, lineterminator="\n")

    # Accounting that the report reads: by class, deployment, target source, author, vote margin, hard flag.
    df["net_disagree"] = pd.to_numeric(df["human_disagree"], errors="coerce").fillna(0).astype(int) \
        - pd.to_numeric(df["human_agree"], errors="coerce").fillna(0).astype(int)
    df["margin_bucket"] = np.where(df["label_class"] == "incorrect",
                                   np.where(df["net_disagree"] <= 2, "net Disagree 2", "net Disagree >= 3"),
                                   np.where(-df["net_disagree"] <= 3, "net Agree 3", "net Agree >= 4"))
    df.loc[df["class_source"] == "expert", "margin_bucket"] = "expert"
    summary = {
        "source_split": os.path.abspath(args.split), "source_sha256": sha256_of(args.split),
        "oof_files": [os.path.abspath(p) for p in paths], "n_train": int(len(df)),
        "thresholds": {c: float(t) for c, t in zip(CLASSES, thresholds)},
        "confident_joint": {g: {l: int(joint[i, j]) for j, l in enumerate(CLASSES)} for i, g in enumerate(CLASSES)},
        "flagged": int(flagged.sum()), "flagged_share": float(flagged.mean()),
        "argmax_disagrees": int((np.argmax(p, axis=1) != np.array([CLASSES.index(c) for c in y])).sum()),
        "by_class": _shares(df, "label_class").to_dict("records"),
        "by_city": _shares(df, "city").sort_values("share", ascending=False).to_dict("records"),
        "by_class_source": _shares(df, "class_source").to_dict("records"),
        "by_author": _shares(df, "author").to_dict("records"),
        "by_margin": _shares(df, ["label_class", "margin_bucket"]).to_dict("records"),
        "by_hard": _shares(df, "hard").to_dict("records") if "hard" in df.columns else [],
        "noisy_targets_sha256": sha256_of(noisy_path),
    }
    variants = cl_variants(base, set(noisy.loc[noisy["flagged"] == 1, "label_uid"]))
    for name, variant in variants.items():
        check_eval_rows_unchanged(base, variant)
        path = write_split(variant, os.path.join(args.out, name, "split.csv"))
        summary[f"{name}_sha256"] = sha256_of(path)
    with open(os.path.join(args.out, "flag_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"{len(df)} train rows; thresholds {summary['thresholds']}; confident joint {joint.tolist()}; "
          f"flagged {int(flagged.sum())} ({flagged.mean():.1%}); argmax disagrees {summary['argmax_disagrees']}")
    print(pd.DataFrame(summary["by_class"]).to_string(index=False))
    print(pd.DataFrame(summary["by_margin"]).to_string(index=False))
    print(f"wrote {noisy_path} and variants {', '.join(variants)} under {args.out}/")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# rules (issue #75)
# ---------------------------------------------------------------------------------------------------------------------

def passes_rule(df, rule):
    """Boolean Series: which rows keep their target under ``rule`` (``symmetric`` or ``unanimous3``). Expert-sourced
    targets always pass; the rules read the manifest's ``human_agree`` / ``human_disagree`` counts.

    >>> d = pd.DataFrame({"label_class": ["incorrect", "incorrect", "correct", "correct", "incorrect"],
    ...                   "class_source": ["margin", "margin", "margin", "margin", "expert"],
    ...                   "human_agree": ["0", "1", "3", "4", "0"], "human_disagree": ["2", "4", "0", "1", "0"]})
    >>> passes_rule(d, "symmetric").tolist(), passes_rule(d, "unanimous3").tolist()
    ([False, True, True, True, True], [False, False, True, False, True])
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    agree = pd.to_numeric(df["human_agree"], errors="coerce").fillna(0).astype(int)
    disagree = pd.to_numeric(df["human_disagree"], errors="coerce").fillna(0).astype(int)
    incorrect = df["label_class"] == "incorrect"
    if rule == "symmetric":
        ok = np.where(incorrect, disagree - agree > 2, agree - disagree > 2)
    else:
        ok = np.where(incorrect, (disagree >= 3) & (agree == 0), (agree >= 3) & (disagree == 0))
    expert = df["class_source"].eq("expert") if "class_source" in df.columns else pd.Series(False, index=df.index)
    return pd.Series(ok, index=df.index) | expert


def rule_variant(base, rule):
    """``base`` with the train rows that fail ``rule`` set to ``train_dropped``."""
    df = base.copy()
    train = df["split"] == "train"
    df.loc[train & ~passes_rule(df, rule), "split"] = TRAIN_DROPPED
    return df


def cmd_rules(args):
    base = read_split(args.split)
    for col in ("human_agree", "human_disagree"):
        if col not in base.columns:
            raise SystemExit(f"{args.split} has no {col} column; the rules need the manifest's vote counts")
    rules = [r.strip() for r in args.rule.split(",") if r.strip()]
    for rule in rules:
        variant = rule_variant(base, rule)
        check_eval_rows_unchanged(base, variant)
        path = write_split(variant, os.path.join(args.out, rule, "split.csv"))
        train = base["split"] == "train"
        kept = variant["split"] == "train"
        before = base.loc[train, "label_class"].value_counts()
        after = variant.loc[kept, "label_class"].value_counts()
        by_city = (base.loc[train].assign(kept=kept[train].to_numpy())
                   .groupby(["city", "label_class"])["kept"].agg(n="size", kept="sum").reset_index())
        meta = {"rule": rule, "source_split": os.path.abspath(args.split), "source_sha256": sha256_of(args.split),
                "train_before": {c: int(before.get(c, 0)) for c in CLASSES},
                "train_after": {c: int(after.get(c, 0)) for c in CLASSES},
                "dropped": int(train.sum() - kept.sum()),
                "by_city": by_city.to_dict("records"), "split_sha256": sha256_of(path)}
        with open(os.path.join(args.out, rule, "rule.json"), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"{rule}: train {meta['train_before']} -> {meta['train_after']} ({meta['dropped']} dropped); "
              f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _metric(m, split, key):
    s = m.get(split) or {}
    if key == "auc":
        return s.get("auc")
    if key.startswith("recall_at_precision"):
        _, cls, level = key.split(":")
        return ((s.get("recall_at_precision") or {}).get(cls) or {}).get(level, {}).get("recall")
    return s.get(key)


def within_city_auc(m, split):
    """Within-city AUC of ``split`` from evaluate.py's ``--per-city`` block (cities with at least ``--min-city-n``
    rows), and the number of cities: the per-city AUCs averaged with weights n_incorrect x n_correct, the
    definition of docs/retrain-2026-10.md, which a score that only separates deployments cannot raise."""
    cities = ((m.get("per_city") or {}).get(split) or {}).get("cities") or {}
    aucs, weights = [], []
    for v in cities.values():
        if not isinstance(v, dict) or v.get("auc") is None:
            continue
        n = v.get("n_by_class") or {}
        w = n.get("incorrect", 0) * n.get("correct", 0)
        if w > 0:
            aucs.append(v["auc"])
            weights.append(w)
    return (float(np.average(aucs, weights=weights)) if aucs else None), len(aucs)


def redact_cities(records, public, key="city", other="other (private deployments)"):
    """Pool the ``records`` (dicts with ``key``, ``n``, ``flagged``) of cities not in ``public`` into one row, so a
    published table names only deployments the live /v3/api/cities listed as public when the report ran."""
    kept, pooled = [], {key: other, "n": 0, "flagged": 0}
    for r in records:
        if r[key] in public:
            kept.append(dict(r))
        else:
            pooled["n"] += int(r["n"])
            pooled["flagged"] += int(r["flagged"])
    if pooled["n"]:
        pooled["share"] = round(pooled["flagged"] / pooled["n"], 4)
        kept.append(pooled)
    return sorted(kept, key=lambda r: -r["share"])


PRED_JOIN_COLUMNS = ["label_uid", "city", "pano_id", "severity", "tags"]


def load_predictions(data_dir, label_type, name, split_df):
    """``evaluate.py``'s ``predictions.csv`` (baseline) or ``predictions_<name>.csv`` joined with the split's city,
    panorama and slice columns (so ``analysis/slices.py`` can slice it); None when the file is absent."""
    suffix = "" if name == "baseline" else f"_{name}"
    path = os.path.join(data_dir, label_type, f"predictions{suffix}.csv")
    if not os.path.exists(path):
        return None
    pred = pd.read_csv(path, dtype={"label_uid": str, "split": str, "true": str})
    m = pred.merge(split_df[PRED_JOIN_COLUMNS], on="label_uid", how="left", validate="one_to_one")
    if m["city"].isna().any():
        raise ValueError(f"{path}: {int(m['city'].isna().sum())} predictions have no row in split.csv")
    return m


def within_city_from_predictions(pred, split, min_n=10):
    """The within-city AUC of ``docs/retrain-2026-10.md`` on one split of a joined predictions frame: per-deployment
    AUCs averaged with weights n_incorrect x n_correct over deployments with both classes and at least ``min_n``
    labels (``analysis/retrain_report.within_city_auc``). ``(auc or None, n_cities)``."""
    from analysis import retrain_report as rr  # imported here: it pulls in matplotlib, which the GPU steps never need
    p = pred[pred["split"] == split]
    if p.empty:
        return None, 0
    auc, n = rr.within_city_auc(p["true"] == "incorrect", p["p_incorrect"], p["city"], min_n)
    return (None if np.isnan(auc) else float(auc)), int(n)


def heldout_rows(preds, public, min_n=10):
    """AUC per held-out deployment for every model in ``preds`` (``{name: joined predictions}``). Deployments are
    named as in the round-3 report: public ones by ``city_id``, private ones lettered (``retrain_report.heldout_names``,
    a US deployment's ``city_id`` ends in a state code). One record per deployment: ``deployment``, ``n``,
    ``n_incorrect`` and one AUC per model (None under ``min_n`` labels or with one class)."""
    from analysis import retrain_report as rr
    from sklearn.metrics import roc_auc_score
    held = {name: p[p["split"] == "test_heldout"] for name, p in preds.items()}
    cities = sorted(set().union(*[set(h["city"]) for h in held.values()])) if held else []
    us = {c for c in cities if len(c.split("-")[-1]) == 2 and c.split("-")[-1].isalpha()}
    names = rr.heldout_names(cities, public, us)
    rows = []
    for c in sorted(cities, key=lambda c: names[c]):
        row = {"deployment": names[c]}
        for name, h in held.items():
            g = h[h["city"] == c]
            y = (g["true"] == "incorrect").to_numpy()
            row.setdefault("n", int(len(g)))
            row.setdefault("n_incorrect", int(y.sum()))
            ok = len(g) >= min_n and 0 < y.sum() < len(g)
            row[name] = float(roc_auc_score(y, g["p_incorrect"])) if ok else None
        rows.append(row)
    return rows


def slice_deltas(preds, variants, min_n=None, n_boot=None, seed=0):
    """Severity and tag slices of each variant against the baseline on the same rows, test and held-out:
    ``analysis/slices.paired_delta_table`` (AUC of both, the difference, and its 95% interval from resampling whole
    panoramas, the same resample for both models). Records with ``split``, ``by``, ``variant`` and the table's
    columns; empty without baseline predictions."""
    from analysis import slices as sl
    base = preds.get("baseline")
    out = []
    if base is None:
        return out
    min_n = sl.MIN_N if min_n is None else min_n
    n_boot = sl.N_BOOT if n_boot is None else n_boot
    for name in variants:
        p = preds.get(name)
        if p is None:
            continue
        for split in ("test", "test_heldout"):
            part = p[p["split"] == split]
            if part.empty:
                continue
            for by in sl.BY:
                t = sl.paired_delta_table(part, base, by, min_n, n_boot, seed)
                out += [{"split": split, "by": by, "variant": name, **r} for r in t.to_dict("records")]
    return out


def overall_deltas(preds, variants, n_boot=None, seed=0):
    """Each variant's AUC minus the baseline's on the whole test and held-out splits, with the paired panorama
    bootstrap of the slices (``analysis/slices.boot_indices``, the same resample scored by both models).
    ``{(variant, split): {"delta", "delta_lo", "delta_hi"}}``; empty without baseline predictions."""
    from analysis import slices as sl
    from sklearn.metrics import roc_auc_score
    base = preds.get("baseline")
    out = {}
    if base is None:
        return out
    n_boot = sl.N_BOOT if n_boot is None else n_boot
    for name in variants:
        p = preds.get(name)
        if p is None:
            continue
        for split in ("test", "test_heldout"):
            part = p[p["split"] == split]
            if part.empty:
                continue
            m = part.merge(base[["label_uid", "p_incorrect"]], on="label_uid", suffixes=("", "_b"), validate="one_to_one")
            y = (m["true"] == "incorrect").to_numpy()
            a, b = m["p_incorrect"].to_numpy(), m["p_incorrect_b"].to_numpy()
            rec = {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "delta_lo": None, "delta_hi": None}
            if n_boot:
                idxs = sl.boot_indices(y, m["pano_id"], n_boot, np.random.default_rng(sl.slice_seed(seed, f"{name}:{split}")))
                vals = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idxs]
                if vals:
                    rec["delta_lo"], rec["delta_hi"] = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
            out[(name, split)] = rec
    return out


def _delta_cell(r):
    if r is None or r.get("delta") is None or np.isnan(r["delta"]):
        return ""
    cell = f"{r['delta']:+.3f}"
    if r.get("delta_lo") is not None and not np.isnan(r["delta_lo"]):
        cell += f" [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
    return cell


def slices_markdown(label_type, records, variants):
    """``slice_deltas`` records as one table per (split, by): the slice, its rows, the baseline AUC and each
    variant's AUC difference with its interval."""
    lines = []
    for split in ("test", "test_heldout"):
        for by in ("severity", "tag"):
            rs = [r for r in records if r["split"] == split and r["by"] == by]
            if not rs:
                continue
            order = list(dict.fromkeys(r["slice"] for r in rs))
            lines += [f"### {label_type} / {split} / by {by}", "",
                      f"| {by.capitalize()} | n | incorrect | baseline AUC | "
                      + " | ".join(f"{v} Δ [95% CI]" for v in variants) + " |",
                      "|---|---:|---:|---:|" + "---:|" * len(variants)]
            for name in order:
                by_variant = {r["variant"]: r for r in rs if r["slice"] == name}
                first = next(iter(by_variant.values()))
                cells = [str(name), f"{first['n']:,}", f"{first['n_incorrect']:,}", _fmt(first.get("auc_b"))]
                cells += [_delta_cell(by_variant.get(v)) for v in variants]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def metrics_path(data_dir, label_type, name):
    """``evaluate.py``'s metrics file of a run: ``metrics.json`` for the baseline, ``metrics_<name>.json`` otherwise."""
    suffix = "" if name == "baseline" else f"_{name}"
    return os.path.join(data_dir, label_type, f"metrics{suffix}.json")


def metric_row(data_dir, label_type, name, preds=None, within_min_n=10):
    """One row of the comparison table: the run's test / held-out / hard AUCs and recalls from its metrics file.
    The within-city AUC comes from the joined predictions in ``preds`` (the round-3 definition, cities with at least
    ``within_min_n`` labels) when the run has them, else from the ``--per-city`` block of ``metrics.json`` (cities
    with at least its ``min_city_n`` rows, a coarser set); ``within_city_source`` says which. ``{"variant": name,
    "missing": path}`` without a metrics file."""
    preds = preds or {}
    path = metrics_path(data_dir, label_type, name)
    if not os.path.exists(path):
        return {"variant": name, "missing": path}
    with open(path) as f:
        m = json.load(f)
    if name in preds:
        wc_t, n_t = within_city_from_predictions(preds[name], "test", within_min_n)
        wc_h, n_h = within_city_from_predictions(preds[name], "test_heldout", within_min_n)
        source = f"predictions (cities with >= {within_min_n} labels)"
    else:
        wc_t, n_t = within_city_auc(m, "test")
        wc_h, n_h = within_city_auc(m, "test_heldout")
        source = "metrics.json per_city block"
    return {"variant": name, "test_auc": _metric(m, "test", "auc"),
            "heldout_auc": _metric(m, "test_heldout", "auc"),
            "within_city_test": wc_t, "within_city_heldout": wc_h, "cities_test": n_t, "cities_heldout": n_h,
            "within_city_source": source,
            "hard_auc": _metric(m, "hard", "auc"), "hard_n": (m.get("hard") or {}).get("n"),
            "incorrect_recall_p92_test": _metric(m, "test", "recall_at_precision:incorrect:0.92"),
            "incorrect_recall_p92_heldout": _metric(m, "test_heldout", "recall_at_precision:incorrect:0.92"),
            "correct_recall_p92_test": _metric(m, "test", "recall_at_precision:correct:0.92")}


def variant_rows(data_dir, label_type, variants, preds=None, within_min_n=10):
    """Rows of the comparison table for one label type: the baseline and each variant (``metric_row``)."""
    return [metric_row(data_dir, label_type, name, preds, within_min_n) for name in ("baseline", *variants)]


# A replicate's run name is <variant>-s<seed>, so a variant's own name must not end in -s<digits>.
SEED_RUN_RE = re.compile(r"^metrics_(?P<variant>.+)-s(?P<seed>\d+)\.json$")


def seed_runs(data_dir, label_type, base_seed=42):
    """The seed replicates of one label type: every ``metrics_<variant>-s<seed>.json`` (the Slurm script's
    ``TAG=<variant>-s<seed>``), plus the chain's own single-seed run of each such variant (seed ``base_seed``:
    the baseline's ``metrics.json``, a variant's ``metrics_<variant>.json``) when it exists. ``{(variant, seed):
    run name}`` with the run name as ``load_predictions`` / ``metric_row`` take it; empty without replicates."""
    runs = {}
    d = os.path.join(data_dir, label_type)
    for fn in os.listdir(d) if os.path.isdir(d) else []:
        m = SEED_RUN_RE.match(fn)
        if m:
            runs[(m["variant"], int(m["seed"]))] = f"{m['variant']}-s{m['seed']}"
    for v in {v for v, _ in runs}:
        if (v, base_seed) not in runs and os.path.exists(metrics_path(data_dir, label_type, v)):
            runs[(v, base_seed)] = v
    return runs


def seed_rows(data_dir, label_type, runs, split_df, within_min_n=10, n_boot=None, seed=0):
    """``metric_row`` of every run in ``runs`` (``seed_runs``), with ``seed`` and ``variant`` set to the plain
    variant name, and for each seed that has both the baseline and the variant, the variant's paired AUC
    difference against the baseline *of the same seed* on test and held-out (``overall_deltas``: whole-panorama
    bootstrap, both models scored on the same resample). Sorted by variant then seed, baseline first."""
    preds = {name: load_predictions(data_dir, label_type, name, split_df) for name in runs.values()}
    preds = {k: v for k, v in preds.items() if v is not None}
    rows = []
    for (v, s), name in sorted(runs.items(), key=lambda kv: (kv[0][0] != "baseline", kv[0])):
        r = metric_row(data_dir, label_type, name, preds, within_min_n)
        r.update({"variant": v, "seed": s, "run": name})
        base = runs.get(("baseline", s))
        if v != "baseline" and "missing" not in r:
            if base in preds and name in preds:
                d = overall_deltas({"baseline": preds[base], v: preds[name]}, [v], n_boot, seed)
                for split, key in (("test", "test"), ("test_heldout", "heldout")):
                    if (v, split) in d:
                        dd = d[(v, split)]
                        r[f"{key}_delta"], r[f"{key}_delta_lo"], r[f"{key}_delta_hi"] = dd["delta"], dd["delta_lo"], dd["delta_hi"]
            else:  # no same-seed baseline, or one side has no predictions file: the Δ cells say so
                r["delta_missing"] = "no same-seed baseline" if base is None else "no predictions"
                print(f"{label_type} {name}: {r['delta_missing']}, no Δ against the baseline", file=sys.stderr)
        rows.append(r)
    return rows


def _mean_sd(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def seed_summary(rows):
    """Per variant over its seeds: mean and sample sd of the test and held-out AUC and of the within-city
    held-out AUC, and for the paired differences against the same-seed baseline their mean, min, max and how many
    seeds were positive. One record per variant, in the order of ``rows``.

    >>> rows = [{"variant": "baseline", "seed": 42, "test_auc": 0.90, "heldout_auc": 0.80, "within_city_heldout": 0.7},
    ...         {"variant": "baseline", "seed": 43, "test_auc": 0.92, "heldout_auc": 0.82, "within_city_heldout": 0.7},
    ...         {"variant": "x", "seed": 42, "test_auc": 0.91, "heldout_auc": 0.83, "within_city_heldout": 0.7,
    ...          "test_delta": 0.01, "heldout_delta": 0.03},
    ...         {"variant": "x", "seed": 43, "test_auc": 0.91, "heldout_auc": 0.81, "within_city_heldout": 0.7,
    ...          "test_delta": -0.01, "heldout_delta": -0.01}]
    >>> s = {r["variant"]: r for r in seed_summary(rows)}
    >>> (s["baseline"]["n_seeds"], round(s["baseline"]["test_auc_mean"], 3), round(s["baseline"]["test_auc_sd"], 4))
    (2, 0.91, 0.0141)
    >>> (s["x"]["n_deltas"], round(s["x"]["heldout_delta_mean"], 3), s["x"]["heldout_delta_positive"])
    (2, 0.01, 1)
    """
    out = []
    for v in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == v and "missing" not in r]
        rec = {"variant": v, "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs)}
        for key in ("test_auc", "heldout_auc", "within_city_heldout", "incorrect_recall_p92_heldout"):
            rec[f"{key}_mean"], rec[f"{key}_sd"], _ = _mean_sd(r.get(key) for r in rs)
        deltas = {k: [r[f"{k}_delta"] for r in rs if r.get(f"{k}_delta") is not None] for k in ("test", "heldout")}
        rec["n_deltas"] = len(deltas["heldout"])  # kept for readers of older files; the per-split counts follow
        for k, ds in deltas.items():
            rec[f"n_deltas_{k}"] = len(ds)
            rec[f"{k}_delta_mean"], rec[f"{k}_delta_sd"], _ = _mean_sd(ds)
            rec[f"{k}_delta_min"] = min(ds) if ds else None
            rec[f"{k}_delta_max"] = max(ds) if ds else None
            rec[f"{k}_delta_positive"] = sum(d > 0 for d in ds)
        out.append(rec)
    return out


def _pm(mean, sd):
    if mean is None:
        return "—"
    return f"{mean:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")


def _delta_range(rec, key):
    if rec.get(f"{key}_delta_mean") is None:
        return "—"
    n = rec[f"n_deltas_{key}"]
    return (f"{rec[f'{key}_delta_mean']:+.3f} [{rec[f'{key}_delta_min']:+.3f}, {rec[f'{key}_delta_max']:+.3f}], "
            f"{rec[f'{key}_delta_positive']}/{n} > 0")


def seeds_markdown(label_type, rows, summary):
    """The per-seed table (every run, with the paired difference against the same-seed baseline) and the
    across-seed summary (mean ± sd; differences as mean [min, max] and the count of positive seeds)."""
    lines = [f"### {label_type}", "",
             "| variant | seed | test AUC | Δ test vs same-seed baseline [95% CI] | within-city test | held-out AUC | "
             "Δ held-out vs same-seed baseline [95% CI] | within-city held-out | incorrect R@P0.92 held-out |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['variant']} | {r['seed']} | not run | | | | | | |")
            continue
        d_test = _delta_cell({"delta": r.get("test_delta"), "delta_lo": r.get("test_delta_lo"),
                              "delta_hi": r.get("test_delta_hi")} if "test_delta" in r else None)
        d_held = _delta_cell({"delta": r.get("heldout_delta"), "delta_lo": r.get("heldout_delta_lo"),
                              "delta_hi": r.get("heldout_delta_hi")} if "heldout_delta" in r else None)
        if r.get("delta_missing"):
            d_test = d_held = r["delta_missing"]
        lines.append("| " + " | ".join([
            r["variant"], str(r["seed"]), _fmt(r["test_auc"]), d_test,
            f"{_fmt(r['within_city_test'])} ({r['cities_test']})", _fmt(r["heldout_auc"]), d_held,
            f"{_fmt(r['within_city_heldout'])} ({r['cities_heldout']})",
            _fmt(r["incorrect_recall_p92_heldout"])]) + " |")
    lines += ["", "Across seeds (mean ± sample sd; Δ against the same-seed baseline as mean [min, max] and the "
              "number of seeds with Δ > 0):", "",
              "| variant | seeds | test AUC | Δ test | held-out AUC | Δ held-out | within-city held-out | "
              "incorrect R@P0.92 held-out |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append("| " + " | ".join([
            s["variant"], ", ".join(str(x) for x in s["seeds"]), _pm(s["test_auc_mean"], s["test_auc_sd"]),
            _delta_range(s, "test"), _pm(s["heldout_auc_mean"], s["heldout_auc_sd"]), _delta_range(s, "heldout"),
            _pm(s["within_city_heldout_mean"], s["within_city_heldout_sd"]),
            _pm(s["incorrect_recall_p92_heldout_mean"], s["incorrect_recall_p92_heldout_sd"])]) + " |")
    return "\n".join(lines) + "\n"


def train_accounting(data_dir, label_type, variants):
    """Train rows kept / dropped / flipped per variant, from the variant split files under ``<label_type>/cl/``."""
    base_path = os.path.join(data_dir, label_type, "split.csv")
    base = read_split(base_path)
    n_train = int((base["split"] == "train").sum())
    rows = [{"variant": "baseline", "train": n_train, "dropped": 0, "flipped": 0}]
    for name in variants:
        path = os.path.join(data_dir, label_type, "cl", name, "split.csv")
        if not os.path.exists(path):
            rows.append({"variant": name, "missing": path})
            continue
        v = read_split(path)
        merged = base[["label_uid", "label_class", "split"]].merge(
            v[["label_uid", "label_class", "split"]], on="label_uid", suffixes=("", "_v"), validate="one_to_one")
        was_train = merged["split"] == "train"
        rows.append({"variant": name, "train": int((merged["split_v"] == "train").sum()),
                     "dropped": int((was_train & (merged["split_v"] == TRAIN_DROPPED)).sum()),
                     "flipped": int((was_train & (merged["label_class"] != merged["label_class_v"])).sum())})
    return rows


def adjudicate_gold(gold):
    """One verdict per ``label_uid``: an item rated by several raters (the #78 round-2 packets overlap by design)
    keeps its verdict only when every rater gave the same one; otherwise it becomes ``cant_tell`` and counts as
    unrated. Rows are reduced to ``label_uid, verdict``.

    >>> g = pd.DataFrame({"label_uid": ["a:1", "a:1", "a:2", "a:2", "a:3"],
    ...                   "verdict": ["correct", "correct", "correct", "incorrect", "incorrect"]})
    >>> adjudicate_gold(g).sort_values("label_uid").verdict.tolist()
    ['correct', 'cant_tell', 'incorrect']
    """
    def one(v):
        u = set(v)
        return next(iter(u)) if len(u) == 1 else "cant_tell"
    out = gold.groupby("label_uid", sort=False)["verdict"].agg(one).reset_index()
    assert out["label_uid"].is_unique
    return out


def expert_agreement(data_dir, label_type, variants, gold):
    """How the rows a variant dropped or flipped agree with an expert verdict (``gold``: ``label_uid``, ``verdict``
    in ``correct``/``incorrect``). Returns one record per variant with counts only (no ids), so it can be published:
    ``n_rated_train`` rated rows among the train rows, ``target_matches_expert`` of those, and for the rows the
    variant touched, how many the expert overturned (target != verdict), against the untouched rows."""
    base = read_split(os.path.join(data_dir, label_type, "split.csv"))
    train = base[base["split"] == "train"][["label_uid", "label_class"]]
    rated = train.merge(gold[["label_uid", "verdict"]], on="label_uid", validate="one_to_one")
    rated = rated[rated["verdict"].isin(CLASSES)]
    out = []
    for name in variants:
        path = os.path.join(data_dir, label_type, "cl", name, "split.csv")
        if not os.path.exists(path) or rated.empty:
            continue
        v = read_split(path)[["label_uid", "label_class", "split"]].rename(
            columns={"label_class": "label_class_v", "split": "split_v"})
        m = rated.merge(v, on="label_uid", validate="one_to_one")
        touched = (m["split_v"] == TRAIN_DROPPED) | (m["label_class"] != m["label_class_v"])
        overturned = m["label_class"] != m["verdict"]
        out.append({"variant": name, "n_rated_train": int(len(m)),
                    "target_matches_expert": int((~overturned).sum()),
                    "touched": int(touched.sum()), "touched_overturned": int((touched & overturned).sum()),
                    "untouched": int((~touched).sum()), "untouched_overturned": int((~touched & overturned).sum())})
    return out


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def report_markdown(label_type, metric_rows, train_rows, agreement_rows=None, heldout=None, within_min_n=10):
    lines = [f"### {label_type}", "",
             "| variant | train rows | dropped | flipped | test AUC | Δ test [95% CI] | within-city test | "
             "held-out AUC | Δ held-out [95% CI] | within-city held-out | hard AUC (n) | incorrect R@P0.92 test | "
             "incorrect R@P0.92 held-out | correct R@P0.92 test |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    train_by = {r["variant"]: r for r in train_rows}
    for r in metric_rows:
        t = train_by.get(r["variant"], {})
        if "missing" in r:
            lines.append(f"| {r['variant']} | {_fmt(t.get('train'))} | {_fmt(t.get('dropped'))} | "
                         f"{_fmt(t.get('flipped'))} | not run | | | | | | | | | |")
            continue
        hard = f"{_fmt(r['hard_auc'])} ({r['hard_n']})" if r.get("hard_n") else "—"
        d_test = _delta_cell({"delta": r.get("test_delta"), "delta_lo": r.get("test_delta_lo"),
                              "delta_hi": r.get("test_delta_hi")} if "test_delta" in r else None)
        d_held = _delta_cell({"delta": r.get("heldout_delta"), "delta_lo": r.get("heldout_delta_lo"),
                              "delta_hi": r.get("heldout_delta_hi")} if "heldout_delta" in r else None)
        lines.append("| " + " | ".join([
            r["variant"], _fmt(t.get("train")), _fmt(t.get("dropped")), _fmt(t.get("flipped")),
            _fmt(r["test_auc"]), d_test, f"{_fmt(r['within_city_test'])} ({r['cities_test']})",
            _fmt(r["heldout_auc"]), d_held, f"{_fmt(r['within_city_heldout'])} ({r['cities_heldout']})", hard,
            _fmt(r["incorrect_recall_p92_test"]), _fmt(r["incorrect_recall_p92_heldout"]),
            _fmt(r["correct_recall_p92_test"])]) + " |")
    if heldout:
        models = [k for k in heldout[0] if k not in ("deployment", "n", "n_incorrect")]
        lines += ["", f"Held-out deployments, AUC per model (blank under {within_min_n} labels or with one class):", "",
                  "| deployment | n | incorrect | " + " | ".join(models) + " |", "|---|---:|---:|" + "---:|" * len(models)]
        for h in heldout:
            lines.append(f"| {h['deployment']} | {h['n']:,} | {h['n_incorrect']:,} | "
                         + " | ".join(_fmt(h[k]) for k in models) + " |")
    if agreement_rows:
        lines += ["", "Expert-rated train rows (aggregates; the rated file stays private):", "",
                  "| variant | rated train rows | target = expert | rows touched | of which expert overturns | "
                  "rows untouched | of which expert overturns |", "|---|---:|---:|---:|---:|---:|---:|"]
        for a in agreement_rows:
            lines.append(f"| {a['variant']} | {a['n_rated_train']} | {a['target_matches_expert']} | {a['touched']} | "
                         f"{a['touched_overturned']} | {a['untouched']} | {a['untouched_overturned']} |")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    gold = None
    if args.gold:
        gold = pd.read_csv(args.gold, dtype=str)
        if not {"label_uid", "verdict"} <= set(gold.columns):
            raise SystemExit("--gold needs label_uid and verdict columns")
        gold = adjudicate_gold(gold)
    public = set(c.strip() for c in (args.public_cities or "").split(",") if c.strip())
    os.makedirs(args.out, exist_ok=True)
    tables, slice_tables, seed_tables, everything = [], [], [], {"_public_cities": sorted(public)}
    for lt in label_types:
        split_df = read_split(os.path.join(args.data_dir, lt, "split.csv"))
        preds = {name: load_predictions(args.data_dir, lt, name, split_df) for name in ("baseline", *variants)}
        preds = {k: v for k, v in preds.items() if v is not None}
        metric_rows = variant_rows(args.data_dir, lt, variants, preds, args.within_min_n)
        deltas = overall_deltas(preds, variants, args.n_boot, args.seed)
        for r in metric_rows:
            for split, key in (("test", "test"), ("test_heldout", "heldout")):
                d = deltas.get((r["variant"], split))
                if d:
                    r[f"{key}_delta"], r[f"{key}_delta_lo"], r[f"{key}_delta_hi"] = d["delta"], d["delta_lo"], d["delta_hi"]
        train_rows = train_accounting(args.data_dir, lt, variants)
        agreement = expert_agreement(args.data_dir, lt, variants, gold) if gold is not None else None
        heldout = heldout_rows(preds, public, args.within_min_n) if preds else []
        slices = slice_deltas(preds, variants, args.slice_min_n, args.n_boot, args.seed)
        everything[lt] = {"metrics": metric_rows, "train": train_rows, "expert": agreement, "heldout": heldout,
                          "slices": slices}
        tables.append(report_markdown(lt, metric_rows, train_rows, agreement, heldout, args.within_min_n))
        runs = seed_runs(args.data_dir, lt, args.base_seed)
        if runs:
            rows = seed_rows(args.data_dir, lt, runs, split_df, args.within_min_n, args.n_boot, args.seed)
            summary = seed_summary(rows)
            everything[lt]["seeds"] = {"rows": rows, "summary": summary}
            seed_tables.append(seeds_markdown(lt, rows, summary))
        if slices:
            slice_tables.append(slices_markdown(lt, slices, [v for v in variants if v in preds]))
        flag_path = os.path.join(args.data_dir, lt, "cl", "flag_summary.json")
        if os.path.exists(flag_path):
            with open(flag_path) as f:
                flags = json.load(f)
            flags["by_city"] = redact_cities(flags.get("by_city", []), public)
            for k in ("source_split", "oof_files"):  # paths, not needed in the published copy
                flags.pop(k, None)
            everything[lt]["flag_summary"] = flags
    text = "\n".join(tables)
    with open(os.path.join(args.out, "variants.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "variants.json"), "w") as f:
        json.dump(everything, f, indent=2, default=float)
        f.write("\n")
    if slice_tables:
        from analysis import slices as sl
        min_n = sl.MIN_N if args.slice_min_n is None else args.slice_min_n
        n_boot = sl.N_BOOT if args.n_boot is None else args.n_boot
        with open(os.path.join(args.out, "slices.md"), "w") as f:
            f.write("# Severity and tag slices: each variant against the round-3 baseline\n\n"
                    "AUC of the incorrect score on the same rows; Δ = variant − baseline with a 95% interval from "
                    f"{n_boot:,} resamples of whole panoramas, the same resample applied to both models "
                    f"(`analysis/slices.py`). Blank under {min_n} labels or when the smaller class has fewer than "
                    f"{sl.MIN_CLASS_FOR_CI}. The intervals are not corrected for the number of slices: at 95%, about "
                    "one cell in twenty excludes zero by chance.\n\n" + "\n".join(slice_tables))
    if seed_tables:
        with open(os.path.join(args.out, "seeds.md"), "w") as f:
            f.write("# Seed replicates: the training seed varied, everything else fixed\n\n"
                    "Every run of a variant that was repeated with another training seed (the Slurm script's "
                    "`TAG=<variant>-s<seed>`), the chain's own run counted as seed "
                    f"{args.base_seed}. Each replicate is compared with the baseline trained with the *same* seed, "
                    "so Δ holds the seed fixed on both sides; its interval resamples whole panoramas of the "
                    "evaluation split (`analysis/slices.py`), the same resample for both models. The across-seed "
                    "rows give the spread of the seed itself (sample sd over the seeds) next to the spread of the "
                    "difference.\n\n" + "\n".join(seed_tables))
    print(text)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# prior (issue #56, second box)
# ---------------------------------------------------------------------------------------------------------------------

LABEL_TYPES = ("CurbRamp", "Crosswalk", "NoCurbRamp", "Obstacle", "SurfaceProblem")


def rule_priors(all_labels, label_types=LABEL_TYPES, rules=("margin",) + RULES):
    """``{rule: {label_type: {"correct": n, "incorrect": n, "share_incorrect": p}}}``: the deployment prior counts
    under the margin rule (what ``data_generation.py`` records) and under each stricter rule. Human-authored,
    margin-sourced labels only, as ``deployment_prior_entry`` counts them; a stricter rule keeps a subset.

    See ``tests/test_confident_learning.py`` for a worked example."""
    import data_generation as dg
    out = {rule: {} for rule in rules}
    for lt in label_types:
        eligible, _ = dg.eligible_labels(all_labels, lt)
        human = eligible[(eligible["author"] == "human") & (eligible["class_source"] == "margin")]
        for rule in rules:
            keep = human if rule == "margin" else human[passes_rule(human, rule)]
            counts = keep["label_class"].value_counts()
            c, i = int(counts.get("correct", 0)), int(counts.get("incorrect", 0))
            out[rule][lt] = {"correct": c, "incorrect": i, "share_incorrect": (i / (c + i)) if c + i else None}
    return out


def cmd_prior(args):
    import data_generation as dg
    cities = dg.parse_cities(args.cities)
    frames = [dg.load_city(c, s, args.rawlabels_cache) for c, s in cities.items()]
    all_labels = pd.concat(frames, ignore_index=True)
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    priors = rule_priors(all_labels, label_types)
    # Provenance without deployment names (the list includes private deployments) or machine paths, so the file
    # can be committed: the number of deployments and a hash of their sorted ids.
    priors["_source"] = {"n_cities": len(cities), "cities_sha256": sha256_of_text("\n".join(sorted(cities))),
                         "rawlabels_cache": os.path.basename(os.path.abspath(args.rawlabels_cache)),
                         "rows": int(len(all_labels))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(priors, f, indent=2)
        f.write("\n")
    for rule in ("margin",) + RULES:
        print(rule, {lt: (v["correct"], v["incorrect"], round(v["share_incorrect"] or 0, 4))
                     for lt, v in priors[rule].items()})
    print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("folds", help="k panorama-grouped folds over the train rows, one split file per fold")
    f.add_argument("--split", required=True, help="the production split.csv (split_data.py output)")
    f.add_argument("--rawlabels", default=None, help="cached rawLabels exports for coordinates (--merge-radius-m)")
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--merge-radius-m", type=float, default=3.0,
                   help="merge groups within this distance, as split_data.py did (default 3; 0 = panoramas only)")
    f.add_argument("--out", required=True, help="directory for folds.csv, folds.json and fold<k>/split.csv")
    f.set_defaults(func=cmd_folds)

    r = sub.add_parser("predict", help="out-of-fold scores for one fold's held-out train rows (GPU)")
    r.add_argument("--split", required=True)
    r.add_argument("--folds-file", required=True, help="folds.csv from the folds command")
    r.add_argument("--fold", type=int, required=True)
    r.add_argument("--run-dir", default=None, help="fine_tune_classifier.py --output-dir of this fold's model")
    r.add_argument("--checkpoint", default=None, help="a checkpoint directory instead of --run-dir's best")
    r.add_argument("--out", required=True, help="oof_fold<k>.csv")
    r.set_defaults(func=cmd_predict)

    g = sub.add_parser("flag", help="confident joint over the out-of-fold scores; noisy_targets.csv and cl-* variants")
    g.add_argument("--split", required=True)
    g.add_argument("--oof", nargs="+", required=True, help="oof_fold*.csv files or globs, one row per train row")
    g.add_argument("--out", required=True, help="the cl/ directory: noisy_targets.csv, flag_summary.json, cl-*/")
    g.set_defaults(func=cmd_flag)

    s = sub.add_parser("rules", help="stricter crowd rules (issue #75) as split variants")
    s.add_argument("--split", required=True)
    s.add_argument("--rule", default=",".join(RULES), help=f"comma list from {RULES} (default all)")
    s.add_argument("--out", required=True, help="the cl/ directory: <rule>/split.csv and <rule>/rule.json")
    s.set_defaults(func=cmd_rules)

    t = sub.add_parser("report", help="comparison tables from evaluate.py outputs")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--label-types", default="CurbRamp,Crosswalk,NoCurbRamp,Obstacle,SurfaceProblem")
    t.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    t.add_argument("--gold", default=None,
                   help="PRIVATE csv with label_uid (city:label_id) and verdict (correct/incorrect) from an expert "
                        "re-label; only counts are written")
    t.add_argument("--public-cities", default="",
                   help="comma list of the deployments the live /v3/api/cities lists as public at run time; every "
                        "other deployment is pooled as 'other (private deployments)' in the published flag shares")
    t.add_argument("--within-min-n", type=int, default=10,
                   help="labels a deployment needs to count in the within-city AUC and the held-out table")
    t.add_argument("--slice-min-n", type=int, default=None, help="labels a severity or tag slice needs (slices.py default)")
    t.add_argument("--n-boot", type=int, default=None, help="panorama resamples per slice interval (slices.py default)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--base-seed", type=int, default=42,
                   help="training seed of the chain's own runs (metrics.json, metrics_<variant>.json), for seeds.md")
    t.add_argument("--out", required=True, help="directory for variants.md, variants.json, slices.md and seeds.md")
    t.set_defaults(func=cmd_report)

    q = sub.add_parser("prior", help="deployment prior counts under the margin rule and each stricter rule (#56)")
    q.add_argument("--rawlabels-cache", required=True, help="directory of cached <city_id>.csv rawLabels exports")
    q.add_argument("--cities", required=True, help="comma list of city_id or city_id=subdomain (data_generation.py)")
    q.add_argument("--label-types", default=",".join(LABEL_TYPES))
    q.add_argument("--out", required=True, help="JSON path")
    q.set_defaults(func=cmd_prior)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "run_dir", None) is None and getattr(args, "checkpoint", None) is None \
            and args.command == "predict":
        raise SystemExit("predict needs --run-dir or --checkpoint")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
