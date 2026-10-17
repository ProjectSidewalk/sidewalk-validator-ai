"""Fit the served accuracy map for a RETRAINED model from its predictions, under the deployment class prior.

The training data are downsampled to a fixed correct:incorrect ratio (55:45), but production sees roughly 90:10
(issue #4). Precision depends on the prior, so each fit-split example is weighted by

    w(y) = pi_deploy(y) / pi_fit(y)

where ``y`` is its true class, ``pi_fit`` the class frequency in the fit split and ``pi_deploy`` the frequency in
``deployment_priors.json``. This is importance weighting for label shift: it assumes ``p(image | class)`` is the same
in training and production, so only the class mix changes. Then, per predicted verdict (argmax; confidence = max
softmax), a non-decreasing isotonic curve of P(verdict correct) vs confidence is fitted on the weighted examples and
written in sidewalk-ai-api's format (``_isotonic_map``). The eval split is reweighted the same way (with its own
``pi_eval``) and used for the ECE, a reliability diagram, and coverage + precision at the 0.92 gate per verdict,
printed and saved as ``calibration_report.json`` next to the map.

``deployment_priors.json`` (written by the data-generation step)::

    {"CurbRamp": {"correct": 3969, "incorrect": 417, "measured_on": "2026-10-17", "source": "..."}, ...}

i.e. pre-downsampling counts of labels meeting the training vote thresholds on human votes alone.

Usage::

    python -m calibration.calibrate --predictions CurbRamp/predictions.csv --priors deployment_priors.json \\
        --label-type CurbRamp --out CurbRamp/calibration/curbramp.json [--fit-split val --eval-split test]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration import _isotonic_map as im

CLASSES = im.API_CLASSES


def load_prior(priors_path, label_type):
    """Return ``(pi, entry)``: the deployment class proportions and the raw JSON entry for ``label_type``."""
    with open(priors_path) as f:
        priors = json.load(f)
    if label_type not in priors:
        raise KeyError(f"{label_type} not in {priors_path} (has {sorted(priors)})")
    entry = priors[label_type]
    counts = np.array([float(entry[c]) for c in CLASSES])
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError(f"bad prior counts for {label_type}: {entry}")
    return dict(zip(CLASSES, counts / counts.sum())), entry


def verdicts(df):
    """Add ``verdict`` (argmax class), ``confidence`` (max softmax) and ``correct`` (verdict == true)."""
    df = df.copy()
    df["verdict"] = np.where(df.p_incorrect.values > df.p_correct.values, "incorrect", "correct")
    df["confidence"] = np.maximum(df.p_correct.values, df.p_incorrect.values)
    df["correct"] = (df.verdict == df["true"].astype(str)).astype(int)
    return df


def prior_weights(true, pi_target):
    """Per-example weights ``pi_target(y) / pi_sample(y)``; ``pi_sample`` is the class mix of ``true`` itself."""
    true = pd.Series(true).astype(str)
    pi_sample = true.value_counts(normalize=True)
    missing = [c for c in CLASSES if pi_target.get(c, 0) > 0 and c not in pi_sample]
    if missing:
        raise ValueError(f"no examples of class {missing} to reweight")
    return true.map(lambda y: pi_target[y] / pi_sample[y]).values.astype(float)


def fit_curves(fit_df, pi_deploy):
    """Fit one weighted isotonic curve per verdict. Returns ``({verdict: iso}, weights)``."""
    w = prior_weights(fit_df["true"], pi_deploy)
    curves = {}
    for v in CLASSES:
        m = (fit_df.verdict == v).values
        curves[v] = im.fit_isotonic(fit_df.confidence.values[m], fit_df.correct.values[m], w[m])
    return curves, w


def evaluate_map(api_map, eval_df, pi_deploy):
    """Reliability, ECE and gate stats of ``api_map`` on ``eval_df`` reweighted to ``pi_deploy``."""
    w = prior_weights(eval_df["true"], pi_deploy)
    out, tables = {}, {}
    served = np.zeros(len(eval_df))
    for v in CLASSES:
        m = (eval_df.verdict == v).values
        served[m] = im.api_lookup_many(api_map[v], eval_df.confidence.values[m])
    for v in CLASSES:
        m = (eval_df.verdict == v).values
        tab = im.reliability_table(eval_df.confidence.values[m], eval_df.correct.values[m], served[m], w[m])
        tables[v] = tab
        g = im.gate_summary(served[m], eval_df.correct.values[m], w[m])
        out[v] = {"n": int(m.sum()), "weighted_share": float(w[m].sum() / w.sum()),
                  "precision": float(np.average(eval_df.correct.values[m], weights=w[m])) if m.any() else None,
                  "ece": im.expected_calibration_error(tab),
                  "gate_coverage": g["coverage"], "gate_precision": g["precision"], "gate_n": g["n_pass"],
                  "reliability": tab.to_dict(orient="records")}
    all_tab = im.reliability_table(eval_df.confidence.values, eval_df.correct.values, served, w)
    out["overall_ece"] = im.expected_calibration_error(all_tab)
    return out, tables


def plot_reliability(tables, label_type, path):
    """Reliability diagram on the eval split: served (map) value vs observed precision per verdict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, v in zip(axes, CLASSES):
        tab = tables[v]
        ax.axhline(im.GATE, color="#52514e", linestyle="--", linewidth=1, label=f"gate {im.GATE}")
        if len(tab):
            x = np.arange(len(tab))
            ax.plot(x, tab.observed, "o-", color="#2a78d6", linewidth=2, markersize=8, label="observed (reweighted)")
            ax.plot(x, tab.predicted, "^-", color="#1baf7a", linewidth=2, markersize=8, label="map value")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lo:g}\nn={n}" for lo, n in zip(tab.bin_lo, tab.n)], fontsize=7)
        ax.set_title(f"{label_type}: verdict {v}")
        ax.set_xlabel("confidence, bin lower edge (eval split)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("precision of the verdict")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def calibrate(predictions, pi_deploy, fit_split="val", eval_split="test"):
    """Fit on ``fit_split`` and evaluate on ``eval_split``. Returns ``(api_map, report_dict, tables)``."""
    df = verdicts(predictions)
    fit_df, eval_df = df[df.split == fit_split], df[df.split == eval_split]
    if len(fit_df) == 0:
        raise ValueError(f"no rows with split == {fit_split!r}")
    curves, _ = fit_curves(fit_df, pi_deploy)
    api_map = im.build_api_map(curves)
    im.validate_api_map(api_map)
    rep = {"prior_deploy": pi_deploy, "fit_split": fit_split, "eval_split": eval_split,
           "prior_fit": fit_df["true"].value_counts(normalize=True).to_dict(), "n_fit": int(len(fit_df))}
    tables = {}
    if len(eval_df):
        rep["prior_eval"] = eval_df["true"].value_counts(normalize=True).to_dict()
        rep["n_eval"] = int(len(eval_df))
        rep["eval"], tables = evaluate_map(api_map, eval_df, pi_deploy)
    return api_map, rep, tables


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--predictions", required=True, help="<label_type>/predictions.csv from evaluate.py")
    p.add_argument("--priors", required=True, help="deployment_priors.json")
    p.add_argument("--label-type", required=True)
    p.add_argument("--out", required=True, help="path of the map JSON to write (drop-in for sidewalk-ai-api)")
    p.add_argument("--fit-split", default="val")
    p.add_argument("--eval-split", default="test")
    args = p.parse_args(argv)

    pi_deploy, entry = load_prior(args.priors, args.label_type)
    api_map, rep, tables = calibrate(pd.read_csv(args.predictions), pi_deploy, args.fit_split, args.eval_split)
    rep.update({"label_type": args.label_type, "prior_source": entry})

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(api_map, f, indent=4)
        f.write("\n")
    with open(os.path.join(out_dir, "calibration_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    if tables:
        plot_reliability(tables, args.label_type, os.path.join(out_dir, "reliability.png"))

    print(f"{args.label_type}: deployment prior {pi_deploy} (from {entry.get('source', '?')}, "
          f"{entry.get('measured_on', '?')}); fit on {rep['n_fit']} {args.fit_split} rows")
    if "eval" in rep:
        e = rep["eval"]
        print(f"eval on {rep['n_eval']} {args.eval_split} rows, reweighted: overall ECE {e['overall_ece']:.4f}")
        for v in CLASSES:
            r = e[v]
            gp = "n/a" if np.isnan(r["gate_precision"]) else f"{r['gate_precision']:.3f}"
            print(f"  verdict {v:9s} n={r['n']:5d} precision {r['precision'] if r['precision'] is not None else float('nan'):.3f} "
                  f"ECE {r['ece']:.4f} | gate {im.GATE}: coverage {r['gate_coverage']:.3f}, precision {gp}")
    print(f"wrote {args.out} and calibration_report.json")


if __name__ == "__main__":
    main()
