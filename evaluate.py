"""Evaluate one label type's best checkpoint: predictions for every held-out row, then metrics.

Replaces ``get_stats.py``. Two stages, kept separate so the metrics are testable without a GPU:

(a) **Inference** (``run_inference``, needs torch + transformers): loads the best checkpoint named in the latest
    ``<label_type>/dinov2/checkpoint-*/trainer_state.json``, preprocesses each crop with the checkpoint's
    ``AutoImageProcessor`` and writes ``<label_type>/predictions.csv`` with columns
    ``label_uid,split,true,p_correct,p_incorrect`` for every row whose split is ``val``, ``test`` or
    ``test_heldout``. ``true`` is the class name (``correct`` / ``incorrect``).
(b) **Metrics** (``report``, pure pandas/scikit-learn): AUC, plain and balanced accuracy, per-class
    precision/recall/F1, and recall at precision 0.90..0.99 with the matching score threshold, for the ``test``
    split, for ``test_heldout`` if present, and for the ``hard == 1`` slice of the two (``--per-city`` adds the same
    metrics per city of ``test`` and ``test_heldout``, in ``metrics.json`` and ``per_city.md``). Written to
    ``<label_type>/metrics.json`` and ``<label_type>/report.txt``, with ``pvr.png`` and ``report.png``. ``--checkpoint``
    scores another model (such as the one the API serves) on the same split; ``--tag`` suffixes all its outputs.
    ``--vote-weights`` (validator_reliability.py's ``vote_weights.csv``) adds a ``gold`` slice of the two test
    splits: labels with at least 3 human votes, every one from a validator with reliability
    ``>= --gold-min-reliability`` (default 0.85; at 0.95 no trainable label qualifies on the 2026-10-17 data;
    https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12).

The confidence-to-accuracy map the API serves is no longer produced here; it is fitted on the val split under the
deployment prior by ``calibration/calibrate.py`` (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4).
This file also fixes the metric defects in https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/6.

Input: ``<label_type>/split.csv`` with at least the columns
``label_uid,city,label_id,pano_id,label_type,label_class,crop_path,split,hard``, written by ``split_data.py`` from
``data_generation.py``'s manifest (``data_generation.py`` also writes the ``deployment_priors.json`` that
``calibration/calibrate.py`` needs; issues #3 and #10). ``crop_path`` may be absolute or
relative to ``<label_type>/`` (tried first) or the working directory. If ``split.csv`` is absent the old
``<label_type>/ds/{train,val,test}/{correct,incorrect}/`` image-folder layout is read instead, with a warning.

Usage::

    python evaluate.py --label_type CurbRamp                 # inference + metrics
    python evaluate.py --label_type CurbRamp --skip-inference # metrics from an existing predictions.csv
    # Per-city metrics for test and test_heldout (cities with >= 50 rows), also written to per_city.md:
    python evaluate.py --label_type CurbRamp --skip-inference --per-city --min-city-n 50
    # Another model on the same split, e.g. the one the API serves; outputs get a _<tag> suffix:
    python evaluate.py --label_type CurbRamp --checkpoint projectsidewalk/sidewalk-validator-ai-curbramp \
        --revision 969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a --tag prod
    # Add the all-reliable-validators "gold" slice:
    python evaluate.py --label_type CurbRamp --skip-inference --vote-weights reliability/vote_weights.csv
"""
import argparse
import glob
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from calibration._isotonic_map import predict_verdicts

CLASSES = ("correct", "incorrect")
EVAL_SPLITS = ("val", "test", "test_heldout")
PRECISION_LEVELS = (0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
SPLIT_COLUMNS = ["label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "crop_path", "split", "hard"]
PREDICTION_COLUMNS = ["label_uid", "split", "true", "p_correct", "p_incorrect"]


# ---------------------------------------------------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------------------------------------------------

def load_split(label_type, data_dir="."):
    """Return the split manifest as a DataFrame with ``SPLIT_COLUMNS`` and resolved ``crop_path``s.

    Reads ``<data_dir>/<label_type>/split.csv`` (the layout data_generation.py's ``--out`` writes).
    """
    base = os.path.normpath(os.path.join(data_dir, label_type))
    path = os.path.join(base, "split.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = set(SPLIT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        if not df.label_uid.is_unique:
            raise ValueError(f"{path}: label_uid is not unique")

        def resolve(p):
            if os.path.isabs(p):
                return p
            under = os.path.join(base, p)
            return under if os.path.exists(under) else p
        df["crop_path"] = df.crop_path.map(resolve)
        return df
    warnings.warn(f"{path} not found; falling back to the old image-folder layout under {base}/ds/. "
                  "That layout has no panorama grouping, no test_heldout split and no hard-example flag.")
    return load_imagefolder(label_type, data_dir)


def load_imagefolder(label_type, data_dir="."):
    """Build a manifest from ``<label_type>/ds/<split>/<class>/<city>_<label_id>.<ext>`` (split_data.py output)."""
    rows = []
    for split_dir in ("train", "val", "validation", "test"):
        for cls in CLASSES:
            for f in sorted(glob.glob(os.path.join(data_dir, label_type, "ds", split_dir, cls, "*"))):
                stem = os.path.splitext(os.path.basename(f))[0]
                city, _, label_id = stem.rpartition("_")
                rows.append({"label_uid": f"{city}:{label_id}", "city": city, "label_id": label_id, "pano_id": None,
                             "label_type": label_type, "label_class": cls, "crop_path": f,
                             "split": "val" if split_dir == "validation" else split_dir, "hard": 0})
    if not rows:
        raise FileNotFoundError(f"no split.csv and no images under {label_type}/ds/")
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def find_best_checkpoint(label_type, data_dir="."):
    """Path of ``best_model_checkpoint`` from the latest checkpoint's ``trainer_state.json``."""
    base = os.path.join(data_dir, label_type, "dinov2")
    from fine_tune_classifier import is_complete_checkpoint
    dirs = [d for d in os.listdir(base) if is_complete_checkpoint(os.path.join(base, d))]  # skips one cut off mid-save
    if not dirs:
        raise FileNotFoundError(f"no complete checkpoint-* directories under {base}")
    latest = max(dirs, key=lambda d: int(d.split("-")[1]))
    with open(os.path.join(base, latest, "trainer_state.json")) as f:
        best = json.load(f).get("best_model_checkpoint")
    if not best:
        raise ValueError("best_model_checkpoint not found in trainer_state.json")
    return best


# ---------------------------------------------------------------------------------------------------------------------
# (a) Inference -- GPU; not covered by the unit tests
# ---------------------------------------------------------------------------------------------------------------------

def run_inference(manifest, checkpoint_dir, revision=None):
    """Score every row of ``manifest`` whose split is in ``EVAL_SPLITS``; return a predictions DataFrame.

    ``checkpoint_dir`` is a local checkpoint or a Hugging Face model id, with ``revision`` pinning the latter."""
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Slow (PIL) BitImageProcessor, matching fine_tune_classifier.py and sidewalk-ai-api's backend="pil".
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir, revision=revision, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(checkpoint_dir, revision=revision).to(device).eval()
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    if set(label2id) != set(CLASSES):
        raise ValueError(f"unexpected id2label {model.config.id2label}")

    rows = manifest[manifest.split.isin(EVAL_SPLITS)]
    probs = []
    for path in tqdm(rows.crop_path, desc="inference"):
        image = Image.open(path).convert("RGB")
        inputs = image_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            probs.append(model(**inputs).logits.softmax(dim=-1).cpu().numpy()[0])
    probs = np.array(probs).reshape(-1, len(CLASSES))
    return pd.DataFrame({"label_uid": rows.label_uid.values, "split": rows.split.values,
                         "true": rows.label_class.values,
                         "p_correct": probs[:, label2id["correct"]], "p_incorrect": probs[:, label2id["incorrect"]]},
                        columns=PREDICTION_COLUMNS)


# ---------------------------------------------------------------------------------------------------------------------
# (b) Metrics -- pure functions, unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def recall_at_precision(y_true, scores, level):
    """Largest recall achievable with precision >= ``level``, and the score threshold that achieves it.

    ``precision_recall_curve`` returns precision that is NOT monotone in the threshold, so this takes
    ``max(recall[precision >= level])`` rather than a binary search (issue #6). ``thresholds`` has one element fewer
    than ``precision``: the final (precision 1, recall 0) point has no threshold, so it is reported as ``None``.
    Ties in recall resolve to the highest such threshold (same recall, fewest false positives).

    Returns:
        ``(recall, threshold)``; ``(0.0, None)`` when no operating point reaches ``level``.

    Example::

        >>> recall_at_precision([1, 1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.55, 0.4], 0.5)
        (1.0, 0.6)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    assert len(thresholds) == len(precision) - 1
    ok = np.flatnonzero(precision >= level)
    if len(ok) == 0:
        return 0.0, None
    best = recall[ok].max()
    idx = ok[recall[ok] == best].max()   # the highest threshold reaching the best recall
    threshold = float(thresholds[idx]) if idx < len(thresholds) else None
    return float(recall[idx]), threshold


def slice_metrics(df, levels=PRECISION_LEVELS):
    """Metrics for one slice of a predictions frame (columns ``true, p_correct, p_incorrect``)."""
    y = df["true"].astype(str).values
    if set(np.unique(y)) - set(CLASSES):
        raise ValueError(f"'true' must be one of {CLASSES}")
    pred, _ = predict_verdicts(df.p_correct.values, df.p_incorrect.values)
    out = {"n": int(len(df)), "n_by_class": {c: int((y == c).sum()) for c in CLASSES}}
    if len(df) == 0:
        return out
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    both = len(np.unique(y)) == 2
    out["auc"] = float(roc_auc_score(y == "incorrect", df.p_incorrect.values)) if both else None
    rep = classification_report(y, pred, labels=list(CLASSES), output_dict=True, zero_division=0)
    out["per_class"] = {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                        for c in CLASSES}
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=list(CLASSES)).tolist()
    out["recall_at_precision"] = {}
    for c in CLASSES:
        if not (y == c).any():
            continue
        out["recall_at_precision"][c] = {f"{p:.2f}": dict(zip(("recall", "threshold"),
                                                              recall_at_precision(y == c, df[f"p_{c}"].values, p)))
                                         for p in levels}
    return out


GOLD_MIN_RELIABILITY = 0.85
GOLD_MIN_VOTES = 3


def gold_labels(vote_weights, min_reliability=GOLD_MIN_RELIABILITY, min_votes=GOLD_MIN_VOTES):
    """Boolean Series by ``label_uid``: True where the label has at least ``min_votes`` human votes and every one came
    from a validator with reliability ``>= min_reliability`` (``vote_weights`` from validator_reliability.py).

    >>> w = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "n_votes": [3, 2, 4],
    ...                   "min_reliability": [0.97, 0.99, 0.6]})
    >>> gold_labels(w, 0.95).to_dict()
    {'a:1': True, 'a:2': False, 'a:3': False}
    """
    missing = {"label_uid", "n_votes", "min_reliability"} - set(vote_weights.columns)
    if missing:
        raise ValueError(f"vote weights have no {sorted(missing)} column(s); rebuild with validator_reliability.py")
    gold = (vote_weights["n_votes"] >= min_votes) & (vote_weights["min_reliability"] >= min_reliability)
    return pd.Series(gold.to_numpy(), index=vote_weights["label_uid"].to_numpy())


def per_city_metrics(df, levels=PRECISION_LEVELS, min_city_n=50):
    """``slice_metrics`` for every city of one split with at least ``min_city_n`` rows (issue #15).

    Returns:
        ``{"min_city_n": ..., "cities": {city: slice_metrics(...)}, "below_min_n": {city: n}}``, cities by
        descending n (ties by name).
    """
    counts = df.groupby("city").size().reset_index(name="n").sort_values(["n", "city"], ascending=[False, True])
    big = counts[counts.n >= min_city_n]
    small = counts[counts.n < min_city_n]
    return {"min_city_n": int(min_city_n),
            "cities": {c: slice_metrics(df[df.city == c], levels) for c in big.city},
            "below_min_n": {c: int(n) for c, n in zip(small.city, small.n)}}


def report(predictions_df, levels=PRECISION_LEVELS, per_city=False, min_city_n=50):
    """Metrics for ``test``, ``test_heldout`` (if present) and the ``hard == 1`` slice of the two, plus their
    ``gold`` slice when ``predictions_df`` has a ``gold`` column (see ``gold_labels``).

    Args:
        predictions_df: columns ``label_uid, split, true, p_correct, p_incorrect`` and, optionally, ``hard``,
            ``gold`` and ``city``.
        per_city: also add ``per_city``: ``{split: per_city_metrics(...)}`` for ``test`` and ``test_heldout``
            (needs ``city``); render with ``format_per_city``.

    Returns:
        ``{slice_name: slice_metrics(...)}``; render with ``format_report``.
    """
    out = {}
    for split in ("test", "test_heldout"):
        s = predictions_df[predictions_df.split == split]
        if len(s):
            out[split] = slice_metrics(s, levels)
    if "hard" in predictions_df.columns:
        h = predictions_df[predictions_df.split.isin(["test", "test_heldout"]) & (predictions_df.hard == 1)]
        if len(h):
            out["hard"] = slice_metrics(h, levels)
    if per_city:
        if "city" not in predictions_df.columns:
            raise ValueError("per_city needs a 'city' column (from split.csv)")
        out["per_city"] = {split: per_city_metrics(predictions_df[predictions_df.split == split], levels, min_city_n)
                           for split in ("test", "test_heldout") if (predictions_df.split == split).any()}
    if "gold" in predictions_df.columns:
        g = predictions_df[predictions_df.split.isin(["test", "test_heldout"]) & predictions_df.gold.astype(bool)]
        out["gold"] = slice_metrics(g, levels)  # reported even when empty, so a missing slice is visible
    return out


def format_report(metrics):
    """Plain-text rendering of ``report()``'s output."""
    lines = []
    for name, m in metrics.items():
        if name == "per_city":
            continue  # rendered by format_per_city
        lines.append(f"== {name}: n={m['n']} ({', '.join(f'{c} {k}' for c, k in m['n_by_class'].items())}) ==")
        if m["n"] == 0:
            continue
        auc = "n/a (one class)" if m["auc"] is None else f"{m['auc']:.4f}"
        lines.append(f"  accuracy          {m['accuracy']:.4f}")
        lines.append(f"  balanced accuracy {m['balanced_accuracy']:.4f}")
        lines.append(f"  AUC (incorrect)   {auc}")
        lines.append(f"  {'class':10s} {'precision':>9s} {'recall':>7s} {'f1':>7s} {'support':>8s}")
        for c, r in m["per_class"].items():
            lines.append(f"  {c:10s} {r['precision']:9.4f} {r['recall']:7.4f} {r['f1-score']:7.4f} {int(r['support']):8d}")
        for c, table in m["recall_at_precision"].items():
            lines.append(f"  recall at precision, class '{c}':")
            for p, r in table.items():
                th = "n/a" if r["threshold"] is None else f"{r['threshold']:.4f}"
                lines.append(f"    precision >= {p}: recall {r['recall']:.4f} at score >= {th}")
    return "\n".join(lines) + "\n"


def format_per_city(per_city):
    """Markdown tables of ``report(..., per_city=True)["per_city"]``: one per split, one row per city.

    Precision and recall are at the argmax verdict (the model's own decision, before any served-accuracy gate);
    AUC is shown only when the city has both classes, and a class with no rows in that city shows "—" for its
    precision and recall (scikit-learn's zero_division would print 0.000).
    """
    def pr(r):
        return ("—", "—") if r["support"] == 0 else (f"{r['precision']:.3f}", f"{r['recall']:.3f}")

    lines = []
    for split, pc in per_city.items():
        lines += [f"### {split}: per city (n >= {pc['min_city_n']})", ""]
        if pc["cities"]:
            lines += ["| city | n (correct / incorrect) | accuracy | AUC | precision correct | recall correct "
                      "| precision incorrect | recall incorrect |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
        else:
            lines.append("No city reaches the minimum.")
        for city, m in pc["cities"].items():
            auc = "—" if m["auc"] is None else f"{m['auc']:.3f}"
            (pc_p, pc_r), (pi_p, pi_r) = pr(m["per_class"]["correct"]), pr(m["per_class"]["incorrect"])
            lines.append(f"| {city} | {m['n']:,} ({m['n_by_class']['correct']:,} / {m['n_by_class']['incorrect']:,}) "
                         f"| {m['accuracy']:.3f} | {auc} | {pc_p} | {pc_r} | {pi_p} | {pi_r} |")
        if pc["below_min_n"]:
            lines += ["", "Below the minimum: " + ", ".join(f"{c} (n={n})" for c, n in pc["below_min_n"].items())
                      + "."]
        lines.append("")
    return "\n".join(lines)


def save_figures(predictions_df, text, label_type, out_dir, suffix=""):
    """``pvr<suffix>.png`` (PR curve per class on test) and ``report<suffix>.png`` (curve plus the text report)."""
    test = predictions_df[predictions_df.split == "test"]
    if len(test) == 0:
        return

    def draw(ax):
        for c in CLASSES:
            p, r, _ = precision_recall_curve(test["true"] == c, test[f"p_{c}"])
            ax.plot(r, p, linewidth=2, label=c)
        ax.axhline(0.9, color="#52514e", linestyle="--", linewidth=1, label="90% precision")
        ax.axhline(0.95, color="#52514e", linestyle=":", linewidth=1, label="95% precision")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision vs recall, test split ({label_type})")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(7, 5))
    draw(ax)
    fig.savefig(os.path.join(out_dir, f"pvr{suffix}.png"), dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 16))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    draw(fig.add_subplot(gs[0]))
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.text(0.01, 1.0, text, fontsize=8, va="top", ha="left", family="monospace", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"report{suffix}.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one label type's best checkpoint.")
    parser.add_argument("--label_type", required=True)
    parser.add_argument("--data-dir", default=".",
                        help="root holding <label_type>/split.csv, <label_type>/dinov2/ and deployment_priors.json "
                             "(data_generation.py's --out); outputs go to <data-dir>/<label_type>/")
    parser.add_argument("--skip-inference", action="store_true", help="reuse <label_type>/predictions.csv")
    parser.add_argument("--checkpoint", default=None,
                        help="score this model instead of the best checkpoint under <label_type>/dinov2/: a local "
                             "checkpoint or a Hugging Face model id (needs --tag)")
    parser.add_argument("--revision", default=None,
                        help="Hugging Face revision for --checkpoint; under HF_HUB_OFFLINE=1 a full commit sha whose "
                             "snapshot is already cached (download it on a login node first)")
    parser.add_argument("--tag", default=None,
                        help="suffix for every output (predictions_<tag>.csv, metrics_<tag>.json, ...), so scoring "
                             "another model never overwrites this run's")
    parser.add_argument("--per-city", action="store_true",
                        help="add per-city metrics for test and test_heldout to metrics.json, and write them as "
                             "markdown to per_city.md (issue #15)")
    parser.add_argument("--min-city-n", type=int, default=50,
                        help="rows a city needs in a split to get its own --per-city row (default 50); the same "
                             "minimum applies to test and test_heldout")
    parser.add_argument("--vote-weights", default=None,
                        help="validator_reliability.py's vote_weights.csv: adds the 'gold' slice (labels with >= 3 "
                             "human votes, every one from a validator with reliability >= --gold-min-reliability)")
    parser.add_argument("--gold-min-reliability", type=float, default=None,
                        help=f"reliability threshold for the gold slice (default {GOLD_MIN_RELIABILITY}; needs "
                             "--vote-weights)")
    args = parser.parse_args(argv)
    if args.gold_min_reliability is not None and not args.vote_weights:
        parser.error("--gold-min-reliability needs --vote-weights")
    if args.checkpoint and not args.tag:
        parser.error("--checkpoint needs --tag, so its outputs do not overwrite the trained model's")
    if args.revision and not args.checkpoint:
        parser.error("--revision applies only to --checkpoint")
    if args.checkpoint and args.skip_inference:
        parser.error("--skip-inference reads predictions_<tag>.csv; --checkpoint would be ignored")
    if args.tag and not args.tag.replace("-", "").replace("_", "").isalnum():
        parser.error("--tag must be letters, digits, '-' or '_' (it becomes part of file names)")
    label_type = args.label_type
    base = os.path.join(args.data_dir, label_type)
    suffix = f"_{args.tag}" if args.tag else ""
    pred_path = os.path.join(base, f"predictions{suffix}.csv")

    manifest = load_split(label_type, args.data_dir)
    if args.skip_inference:
        predictions = pd.read_csv(pred_path)
    else:
        checkpoint = args.checkpoint or find_best_checkpoint(label_type, args.data_dir)
        print(f"{'checkpoint' if args.checkpoint else 'best checkpoint'}: {checkpoint}"
              + (f" @ {args.revision}" if args.revision else ""))
        predictions = run_inference(manifest, checkpoint, args.revision)
        predictions.to_csv(pred_path, index=False)
        print(f"wrote {pred_path} ({len(predictions)} rows)")

    joined = ["label_uid", "hard"] + (["city"] if args.per_city else [])
    with_hard = predictions.merge(manifest[joined], on="label_uid", how="left", validate="one_to_one")
    if args.vote_weights:
        gold_min = GOLD_MIN_RELIABILITY if args.gold_min_reliability is None else args.gold_min_reliability
        gold = gold_labels(pd.read_csv(args.vote_weights, dtype={"label_uid": str}), gold_min)
        absent = ~with_hard.label_uid.isin(gold.index)
        if absent.any():
            raise SystemExit(f"{int(absent.sum())} predictions have no row in {args.vote_weights} (e.g. "
                             f"{with_hard.label_uid[absent].head(3).tolist()})")
        with_hard["gold"] = with_hard.label_uid.map(gold).astype(bool)
        print(f"gold (>= {GOLD_MIN_VOTES} human votes, every voter's reliability >= {gold_min}): "
              + ", ".join(f"{s} {int(with_hard.gold[with_hard.split == s].sum())}/{int((with_hard.split == s).sum())}"
                          for s in EVAL_SPLITS if (with_hard.split == s).any()))
    metrics = report(with_hard, per_city=args.per_city, min_city_n=args.min_city_n)
    if "gold" in metrics:
        metrics["gold"].update(gold_min_reliability=gold_min, gold_min_votes=GOLD_MIN_VOTES)
    text = format_report(metrics)
    print(text)
    if args.per_city:
        per_city_text = format_per_city(metrics.get("per_city", {}))
        print(per_city_text)
        with open(os.path.join(base, f"per_city{suffix}.md"), "w") as f:
            f.write(per_city_text)
    with open(os.path.join(base, f"metrics{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(base, f"report{suffix}.txt"), "w") as f:
        f.write(text)
    save_figures(predictions, text, label_type, base, suffix)
    if args.tag:
        return  # tagged outputs are a comparison model's: the served map is fitted for the trained checkpoint only
    priors = os.path.join(args.data_dir, "deployment_priors.json")
    print(f"next (fit the served map under the deployment prior data_generation.py wrote to {priors}):")
    print(f"  python -m calibration.calibrate --predictions {pred_path} --priors {priors} "
          f"--label-type {label_type} --out {os.path.join(base, 'calibration', label_type.lower() + '.json')}")


if __name__ == "__main__":
    main()
