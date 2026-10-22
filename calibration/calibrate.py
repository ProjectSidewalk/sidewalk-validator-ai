"""Fit the served accuracy map for a RETRAINED model from its predictions, under the deployment class prior.

The training data are downsampled to a fixed correct:incorrect ratio (55:45), but production sees roughly 90:10
(issue #4; that figure is a proxy, the Teaneck labels on which the human crowd was decisive). Precision depends on the prior, so each fit-split example is weighted by

    w(y) = pi_deploy(y) / pi_fit(y)

where ``y`` is its true class, ``pi_fit`` the class frequency in the fit split and ``pi_deploy`` the frequency in
``deployment_priors.json``. This is importance weighting for label shift: it assumes ``p(image | class)`` is the same
in training and production, so only the class mix changes. Then, per predicted verdict (argmax; confidence = max
softmax), a non-decreasing isotonic curve of P(verdict correct) vs confidence is fitted on the weighted examples and
written in sidewalk-ai-api's format (``_isotonic_map``). The eval split is reweighted the same way (with its own
``pi_eval``) and used for the ECE, a reliability diagram, and coverage + precision at the 0.92 gate per verdict,
printed and saved as ``calibration_report.json`` next to the map.

**Hold-back.** ``--hold-back incorrect`` (any subset of the verdicts) caps that verdict's map at
``_isotonic_map.HOLD_BACK_CAP`` (0.9199), the same cap and JSON shape ``calibrate_from_production.py`` uses for a class
whose out-of-fold gate precision is not established, so the API and the website treat the verdict exactly as a
held-back production class: it is served, with its accuracy, but never reaches the 0.92 gate and never becomes a vote.
The report records the held-back verdicts, the cap, the reason (``--hold-back-reason``, required with
``--hold-back``), and the gate statistics the uncapped map would have had, so the published card is truthful about
what is served. The CLI default is no hold-back; the first ship of the October 2026 retrain held ``incorrect`` back
(issue #56) until the deployment prior is settled and the revision has production outcomes to fit on.

**Fit split.** The default fits on ``val``, which is also the split ``fine_tune_classifier.py`` uses to choose the
checkpoint, so the fit is mildly optimistic. With a ``test_heldout`` split available, prefer
``--fit-split test --eval-split test_heldout``.

**Inputs.** ``predictions.csv`` comes from ``evaluate.py``, which reads ``<label_type>/split.csv`` (written by
``split_data.py`` from ``data_generation.py``'s manifest); ``deployment_priors.json`` is written by
``data_generation.py`` from the pre-downsampling class counts (issues #3 and #10).

``deployment_priors.json``::

    {"CurbRamp": {"correct": 3969, "incorrect": 417, "measured_on": "2026-10-17", "source": "..."}, ...}

i.e. pre-downsampling counts of labels meeting the training vote thresholds on human votes alone.

Usage::

    python -m calibration.calibrate --predictions CurbRamp/predictions.csv --priors deployment_priors.json \\
        --label-type CurbRamp --out CurbRamp/calibration/curbramp.json [--fit-split val --eval-split test] \\
        [--hold-back incorrect --hold-back-reason "..."]
"""
import argparse
import json
import os

import warnings

import numpy as np
import pandas as pd

from calibration import _isotonic_map as im
from calibration._plots import plot_reliability

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
    df["verdict"], df["confidence"] = im.predict_verdicts(df.p_correct.values, df.p_incorrect.values)
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


def hold_back(api_map, verdicts_to_hold, cap=im.HOLD_BACK_CAP):
    """A copy of ``api_map`` with every verdict in ``verdicts_to_hold`` capped at ``cap`` (``cap_class_map``).

    Example: hold_back(api_map, ["incorrect"])["incorrect"] never reaches 0.92; ["correct"] is unchanged.
    """
    unknown = sorted(set(verdicts_to_hold) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown verdict(s) to hold back: {unknown}; choose from {list(CLASSES)}")
    return {v: (im.cap_class_map(m, cap) if v in verdicts_to_hold else dict(m)) for v, m in api_map.items()}


def calibrate(predictions, pi_deploy, fit_split="val", eval_split="test", hold_back_verdicts=(),
              hold_back_reason=None):
    """Fit on ``fit_split``, optionally hold verdicts back, evaluate on ``eval_split``.

    Returns ``(api_map, report_dict, tables)``; ``api_map`` is the shipped (capped) map, and the report's
    ``hold_back`` entry lists the held-back verdicts, the cap, the reason and the uncapped gate statistics. A
    hold-back needs a ``hold_back_reason`` (it is published on the model card); ``ValueError`` without one.
    """
    df = verdicts(predictions)
    fit_df, eval_df = df[df.split == fit_split], df[df.split == eval_split]
    if len(fit_df) == 0:
        raise ValueError(f"no rows with split == {fit_split!r}")
    held = [v for v in CLASSES if v in set(hold_back_verdicts)]
    if held and not (hold_back_reason or "").strip():
        raise ValueError(f"holding back {held} needs a hold_back_reason (it goes on the model card)")
    curves, _ = fit_curves(fit_df, pi_deploy)
    uncapped = im.build_api_map(curves)
    im.validate_api_map(uncapped)
    api_map = hold_back(uncapped, held)
    im.validate_api_map(api_map)
    rep = {"prior_deploy": pi_deploy, "fit_split": fit_split, "eval_split": eval_split,
           "prior_fit": fit_df["true"].value_counts(normalize=True).to_dict(), "n_fit": int(len(fit_df)),
           "fit_range": {v: im.fit_range(curves[v]) for v in CLASSES},
           "hold_back": {"verdicts": held, "cap": im.HOLD_BACK_CAP if held else None,
                         "reason": hold_back_reason if held else None,
                         "uncapped_reaches_gate": {v: bool(max(uncapped[v].values()) >= im.GATE) for v in CLASSES}}}
    tables = {}
    if len(eval_df):
        rep["prior_eval"] = eval_df["true"].value_counts(normalize=True).to_dict()
        rep["n_eval"] = int(len(eval_df))
        try:
            rep["eval"], tables = evaluate_map(api_map, eval_df, pi_deploy)
            if held:    # what the held-back verdicts would have done at the gate, for the record
                unc, _ = evaluate_map(uncapped, eval_df, pi_deploy)
                rep["hold_back"]["uncapped_eval"] = {v: {k: unc[v][k] for k in ("gate_coverage", "gate_precision",
                                                                                "gate_n")} for v in held}
            for v in CLASSES:
                rep["eval"][v]["held_back"] = v in held
        except ValueError as e:      # e.g. the eval split has no examples of one class: cannot reweight it
            warnings.warn(f"evaluation on {eval_split!r} skipped: {e}")
            rep["eval"] = "n/a"
            rep["eval_error"] = str(e)
    else:
        warnings.warn(f"no rows with split == {eval_split!r}; evaluation skipped")
        rep["eval"] = "n/a"
    return api_map, rep, tables


def without_nan(obj):
    """``obj`` with every NaN float (numpy or Python) replaced by None, recursively, so the JSON is strict (``null``
    rather than the bare ``NaN`` Python would write).

    Example: without_nan({"a": float("nan"), "b": [1.0, np.nan]}) -> {"a": None, "b": [1.0, None]}
    """
    if isinstance(obj, dict):
        return {k: without_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [without_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--predictions", required=True, help="<label_type>/predictions.csv from evaluate.py")
    p.add_argument("--priors", required=True, help="deployment_priors.json")
    p.add_argument("--label-type", required=True)
    p.add_argument("--out", required=True, help="path of the map JSON to write (drop-in for sidewalk-ai-api)")
    p.add_argument("--fit-split", default="val")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--hold-back", action="extend", nargs="+", default=[], choices=CLASSES, metavar="VERDICT",
                   help=f"verdict(s) whose map is capped at {im.HOLD_BACK_CAP} so they never reach the gate "
                        f"(choose from {', '.join(CLASSES)}; repeatable, e.g. --hold-back incorrect for a first "
                        f"ship); requires --hold-back-reason")
    p.add_argument("--hold-back-reason", default=None,
                   help="why the verdict(s) are held back; recorded in calibration_report.json and printed on the "
                        "model card. Required with --hold-back")
    args = p.parse_args(argv)
    if args.hold_back and not (args.hold_back_reason or "").strip():
        p.error("--hold-back requires --hold-back-reason (it is published on the model card)")

    pi_deploy, entry = load_prior(args.priors, args.label_type)
    api_map, rep, tables = calibrate(pd.read_csv(args.predictions), pi_deploy, args.fit_split, args.eval_split,
                                     args.hold_back, args.hold_back_reason)
    rep.update({"label_type": args.label_type, "prior_source": entry})

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(api_map, f, indent=4)
        f.write("\n")
    with open(os.path.join(out_dir, "calibration_report.json"), "w") as f:
        json.dump(without_nan(rep), f, indent=2, default=float)
    if tables:
        plot_reliability([(f"{args.label_type}: verdict {v}", tables[v]) for v in CLASSES],
                         os.path.join(out_dir, "reliability.png"),
                         xlabel=f"confidence, bin lower edge ({args.eval_split} split)",
                         observed_label="observed (reweighted)", new_label="map value")

    print(f"{args.label_type}: deployment prior {pi_deploy} (from {entry.get('source', '?')}, "
          f"{entry.get('measured_on', '?')}); fit on {rep['n_fit']} {args.fit_split} rows")
    if rep["eval"] == "n/a":
        print(f"eval: n/a ({rep.get('eval_error', 'no eval rows')})")
    else:
        e = rep["eval"]
        print(f"eval on {rep['n_eval']} {args.eval_split} rows, reweighted: overall ECE {e['overall_ece']:.4f}")
        for v in CLASSES:
            r = e[v]
            gp = "n/a" if np.isnan(r["gate_precision"]) else f"{r['gate_precision']:.3f}"
            prec = r["precision"] if r["precision"] is not None else float("nan")
            print(f"  verdict {v:9s} n={r['n']:5d} precision {prec:.3f} ECE {r['ece']:.4f} | gate {im.GATE}: "
                  f"coverage {r['gate_coverage']:.3f}, precision {gp}" + (" | HELD BACK" if r.get("held_back") else ""))
        for v, u in (rep["hold_back"].get("uncapped_eval") or {}).items():
            gp = "n/a" if u["gate_precision"] is None or np.isnan(u["gate_precision"]) else f"{u['gate_precision']:.3f}"
            print(f"  (uncapped {v}: coverage {u['gate_coverage']:.3f}, precision {gp}, n {u['gate_n']})")
    hb = rep["hold_back"]
    if hb["verdicts"]:
        print(f"held back at {im.HOLD_BACK_CAP}: {', '.join(hb['verdicts'])} ({hb['reason']})")
    print(f"wrote {args.out} and calibration_report.json")


if __name__ == "__main__":
    main()
