"""Accuracy maps for the CURRENTLY DEPLOYED validator models, calibrated on production outcomes.

Input is one row per label the AI assessed, as pulled from every city's ``label_ai_assessment`` joined to its human
votes (``analysis/pull_ai_assessments.sh``). Columns used: ``schema, label_id, label_type, ai_result, ai_accuracy,
ai_confidence, ai_date, model_id, h_agree, h_dis``. ``ai_result`` is the raw verdict (Agree / Disagree) whether or
not it passed the 0.92 gate, ``ai_confidence`` the raw softmax confidence, ``ai_accuracy`` the value the current maps
served, and ``h_agree``/``h_dis`` the human votes on that label **excluding the AI's own vote**. ``label_id`` is
per-city, so labels are keyed on ``schema:label_id`` throughout.

For each label type and raw verdict (Agree -> the map's ``correct`` class, Disagree -> ``incorrect``) we keep labels
with at least ``--min-human-votes`` human Agree/Disagree votes and a strict human majority, define the outcome as
"the majority agrees with the AI's verdict", fit a non-decreasing isotonic curve of that outcome against
``ai_confidence`` and write it in sidewalk-ai-api's format (``_isotonic_map``). The shipped maps are fitted on ALL
labelled rows. With ``--holdout-fraction`` a second, check-only curve is fitted on a hash-selected part of the labels
and evaluated on the rest; that check drives the reported gate precision, the per-bin acceptance table and the
hold-back rule.

**Hold-back rule.** A (label type, verdict) class may clear the 0.92 gate only if the held-out precision of the
verdicts its check map lets through, minus 1.96 standard errors, is >= 0.92. Otherwise its shipped map is capped at
0.9199 everywhere (it never becomes a vote) and the summary lists it as held back with the numbers that decided it.
With ``--holdout-fraction 0`` the rule is applied to in-sample numbers, and every such number is labelled in-sample.

Outputs under ``--out-dir``: ``accuracy_mappings/<type>.json`` (drop-in for sidewalk-ai-api), ``reliability_<type>.png``,
``summary.csv``, ``holdout_bins.csv``, ``summary.md`` and ``calibration_report.json`` (fitted confidence range,
hold-back decision and gate numbers per class).

Caveats, in order of importance:

* **These maps are only valid for the model revisions that produced the data.** As of 2026-10-17 those are the
  revisions pinned in sidewalk-ai-api's ``sidewalk_ai_api/model_map.py`` (training_date 08-16-2025):
  sidewalk-validator-ai-curbramp ``969ad5a``, -nocurbramp ``9f3f3bb``, -obstacle ``8f0fe8f``, -surfaceproblem
  ``0e1439d``, -crosswalk ``b7936e5``. The assessment table records ``model_id`` but not the revision, so the script
  prints the ``model_id`` values it saw and a ``--since`` date can drop assessments made before a model swap. A
  retrained model needs its own map (``calibration/calibrate.py``).
* **Selection bias.** Humans do not vote on a random sample of AI-assessed labels: SidewalkWebpage's Triage queue
  routes labels the AI contested to humans first, so labels with a Disagree verdict and >= 2 human votes are
  over-represented among the hard cases. The fitted Disagree curve is therefore pessimistic, which is the safe
  direction for a gate that casts votes against crowd labels. Agree verdicts are less affected.
* **The hold-out is split per label, not per panorama** (the export has no ``pano_id``), so neighbouring labels on
  one panorama can fall on both sides and the check is somewhat optimistic.
* The human majority is a noisy reference (measured expert-vs-crowd agreement is 86-91%), so observed precision
  near 1.0 is bounded by crowd error, not only by the model.

Usage::

    python -m calibration.calibrate_from_production --assessments ai_assessments.csv --out-dir phase0/ \\
        --min-human-votes 2 --holdout-fraction 0.5 --seed 42 --old-maps-dir ../sidewalk-ai-api/accuracy_mappings
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from calibration import _isotonic_map as im
from calibration._plots import plot_reliability

LABEL_TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]
VERDICT_TO_CLASS = {"Agree": "correct", "Disagree": "incorrect"}
EXCLUDED_SCHEMAS = {"sidewalk_validation"}  # internal test deployment
ACCEPT_TOLERANCE = 0.01
ACCEPT_MIN_N = 200


def load_assessments(path, since=None):
    """Read the assessments CSV, drop test schemas and non-binary verdicts, and add ``label_uid`` and ``h_n``."""
    df = pd.read_csv(path, low_memory=False)
    df = df[~df.schema.isin(EXCLUDED_SCHEMAS)]
    df = df[df.ai_result.isin(VERDICT_TO_CLASS) & df.label_type.isin(LABEL_TYPES)].copy()
    if since is not None:
        df = df[pd.to_datetime(df.ai_date) >= pd.Timestamp(since)].copy()
    df["label_uid"] = df.schema + ":" + df.label_id.astype(str)
    if not df.label_uid.is_unique:
        raise ValueError("(schema, label_id) is not unique in the assessments file")
    df["h_n"] = df.h_agree + df.h_dis
    return df.reset_index(drop=True)


def holdout_mask(label_uids, fraction, seed):
    """Deterministic split: True for labels in the held-out part, selected by a hash of ``seed:label_uid``."""
    if fraction <= 0:
        return np.zeros(len(label_uids), dtype=bool)
    u = np.array([int(hashlib.md5(f"{seed}:{x}".encode()).hexdigest()[:12], 16) / 16 ** 12 for x in label_uids])
    return u < fraction


def add_outcomes(df, min_human_votes):
    """Add ``labelled`` (enough human votes and a strict majority) and ``correct`` (majority agrees with the AI)."""
    df = df.copy()
    df["labelled"] = (df.h_n >= min_human_votes) & (df.h_agree != df.h_dis)
    majority = np.where(df.h_agree > df.h_dis, "Agree", "Disagree")
    df["correct"] = (majority == df.ai_result).astype(int)
    return df


def check_old_map(rows, api_map, rounding=5e-5):
    """How well ``api_lookup`` on a served map reproduces the recorded ``ai_accuracy``.

    The export rounds ``ai_confidence`` to 4 decimals while the API looked up the unrounded value, so a row is
    *consistent* if its recorded accuracy is one the map could serve for some confidence within ``±rounding``.

    Returns:
        dict with ``exact`` (fraction reproduced from the rounded confidence) and ``consistent``.
    """
    if len(rows) == 0:
        return {"exact": float("nan"), "consistent": float("nan")}
    exact = consistent = 0
    for verdict, cls in VERDICT_TO_CLASS.items():
        class_map = api_map[cls]
        th = np.array([float(k) for k in class_map])
        vals = np.array(list(class_map.values()), dtype=float)
        r = rows[rows.ai_result == verdict]
        served = im.api_lookup_many(class_map, r.ai_confidence.values)
        exact += int(np.sum(np.abs(served - r.ai_accuracy.values) <= 1e-4))
        for c, g in r.groupby("ai_confidence"):
            cand = np.r_[vals[(th >= c - rounding) & (th <= c + rounding)], im.api_lookup(class_map, c - rounding)]
            consistent += int(np.sum(np.min(np.abs(g.ai_accuracy.values[:, None] - cand[None, :]), axis=1) <= 1e-4))
    return {"exact": exact / len(rows), "consistent": consistent / len(rows)}


def hold_back_decision(map_reaches_gate, gate_precision, gate_n, gate=im.GATE):
    """Apply the hold-back rule to one class. Returns ``(decision, se, lower_bound)``.

    ``decision`` is ``"below gate"`` when the map never reaches the gate (nothing to hold back), ``"clears"`` when
    ``precision - 1.96 * se >= gate`` on the check set, and ``"held back"`` otherwise (including no evidence at all).
    """
    if gate_n > 0:
        se = float(np.sqrt(gate_precision * (1 - gate_precision) / gate_n))
        lower = gate_precision - 1.96 * se
    else:
        se = lower = float("nan")
    if not map_reaches_gate:
        return "below gate", se, lower
    if gate_n > 0 and lower >= gate:
        return "clears", se, lower
    return "held back", se, lower


def calibrate(df, min_human_votes=2, holdout_fraction=0.5, seed=42, old_maps=None):
    """Fit the maps, apply the hold-back rule and compute every table. Pure: no file I/O.

    Args:
        df: output of ``load_assessments``.
        old_maps: optional ``{label_type: api_map}`` of the currently served maps, used only to check that they
            reproduce ``ai_accuracy`` (a test of ``api_lookup`` against production).

    Returns:
        dict with ``maps`` ({label_type: shipped api_map}), ``summary`` and ``holdout_bins`` (DataFrames),
        ``reliability`` ({(label_type, verdict): DataFrame}), ``classes`` (per-class report: fitted range, gate
        numbers, decision), ``df`` (input plus ``new_accuracy``, ``heldout``, ``labelled``, ``correct``),
        ``old_map_match`` ({label_type: ``check_old_map`` result}) and ``in_sample`` (True without a hold-out).
    """
    df = add_outcomes(df, min_human_votes)
    df["heldout"] = holdout_mask(df.label_uid.values, holdout_fraction, seed)
    in_sample = not df.heldout.any()
    df["new_accuracy"] = np.nan
    maps, rows, bins, reliability, classes, old_map_match = {}, [], [], {}, {}, {}
    for t in LABEL_TYPES:
        ship_curves, check_curves = {}, {}
        for verdict, cls in VERDICT_TO_CLASS.items():
            lab = df[(df.label_type == t) & (df.ai_result == verdict) & df.labelled]
            ship_curves[cls] = im.fit_isotonic(lab.ai_confidence.values, lab.correct.values)
            fit = lab if in_sample else lab[~lab.heldout]
            check_curves[cls] = im.fit_isotonic(fit.ai_confidence.values, fit.correct.values)
        ship_map, check_map = im.build_api_map(ship_curves), im.build_api_map(check_curves)
        if old_maps is not None and t in old_maps:
            old_map_match[t] = check_old_map(df[df.label_type == t], old_maps[t])

        for verdict, cls in VERDICT_TO_CLASS.items():
            sel = (df.label_type == t) & (df.ai_result == verdict)
            s = df[sel]
            ev = s[s.labelled & (s.heldout if not in_sample else True)]
            check_served = im.api_lookup_many(check_map[cls], ev.ai_confidence.values)
            g = im.gate_summary(check_served, ev.correct)
            decision, se, lower = hold_back_decision(max(ship_map[cls].values()) >= im.GATE, g["precision"], g["n_pass"])
            if decision == "held back":
                ship_map[cls] = im.cap_class_map(ship_map[cls])
                check_map[cls] = im.cap_class_map(check_map[cls])
                check_served = np.minimum(check_served, im.HOLD_BACK_CAP)
            classes[(t, verdict)] = {"label_type": t, "verdict": verdict, "decision": decision,
                                     "check_gate_n": g["n_pass"], "check_gate_precision": g["precision"],
                                     "check_gate_se": se, "check_gate_lower_95": lower,
                                     "fit_range": im.fit_range(ship_curves[cls]),
                                     "check_fit_range": im.fit_range(check_curves[cls])}

            df.loc[sel, "new_accuracy"] = im.api_lookup_many(ship_map[cls], s.ai_confidence.values)
            s = df[sel]
            old_all, new_all = im.gate_summary(s.ai_accuracy, None), im.gate_summary(s.new_accuracy, None)
            old_ev = im.gate_summary(ev.ai_accuracy, ev.correct)
            new_ev = im.gate_summary(check_served, ev.correct)
            lab = s[s.labelled]
            tab = im.reliability_table(ev.ai_confidence, ev.correct, predicted=check_served)
            tab["old_predicted"] = im.reliability_table(ev.ai_confidence, ev.correct, predicted=ev.ai_accuracy).predicted.values
            reliability[(t, verdict)] = tab
            rows.append({"label_type": t, "verdict": verdict, "n_assessed": len(s), "n_labelled": len(lab),
                         "observed_precision": lab.correct.mean() if len(lab) else np.nan,
                         "old_coverage": old_all["coverage"], "old_gate_precision": old_ev["precision"],
                         "old_gate_n": old_ev["n_pass"],
                         "new_coverage": new_all["coverage"], "new_gate_precision": new_ev["precision"],
                         "new_gate_n": new_ev["n_pass"], "n_eval": len(ev), "decision": decision,
                         "old_ece": im.expected_calibration_error(tab.assign(predicted=tab.old_predicted)),
                         "new_ece": im.expected_calibration_error(tab)})
            for _, r in tab.iterrows():
                diff = r.predicted - r.observed
                bins.append({"label_type": t, "verdict": verdict, "bin_lo": r.bin_lo, "bin_hi": r.bin_hi, "n": r.n,
                             "new_map": r.predicted, "old_map": r.old_predicted, "observed": r.observed,
                             "diff": diff, "se": r.se, "checked": r.n >= ACCEPT_MIN_N,
                             "within_tol": bool(abs(diff) <= ACCEPT_TOLERANCE) if r.n >= ACCEPT_MIN_N else None,
                             "within_1.96se": bool(abs(diff) <= 1.96 * r.se)})
        im.validate_api_map(ship_map)
        maps[t] = ship_map
    return {"maps": maps, "summary": pd.DataFrame(rows), "holdout_bins": pd.DataFrame(bins),
            "reliability": reliability, "classes": classes, "df": df, "old_map_match": old_map_match,
            "in_sample": in_sample}


def summary_markdown(summary, holdout_bins, classes, min_human_votes, in_sample):
    """Render the summary, the hold-back decisions and the held-out acceptance check as Markdown tables."""
    def pct(x):
        return "—" if pd.isna(x) else f"{x:.1%}"
    where = "in-sample (no hold-out: these numbers are optimistic)" if in_sample else "held-out"
    lines = [f"Labels with >= {min_human_votes} human votes and a strict majority. Coverage is over all assessed "
             "labels of that type and verdict, using the shipped map (fitted on all labelled rows, after the "
             f"hold-back rule). Gate precision is {where}: the check map on the "
             f"{'same' if in_sample else 'held-out'} labelled rows.", "",
             f"| type | verdict | assessed | labelled | observed precision | old coverage | old gate precision ({where.split(' ')[0]}) | "
             f"new coverage | new gate precision ({where.split(' ')[0]}) | decision |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.label_type} | {r.verdict} | {r.n_assessed:,} | {r.n_labelled:,} | {pct(r.observed_precision)} "
                     f"| {pct(r.old_coverage)} | {pct(r.old_gate_precision)} (n={r.old_gate_n:,}) "
                     f"| {pct(r.new_coverage)} | {pct(r.new_gate_precision)} (n={r.new_gate_n:,}) | {r.decision} |")
    lines += ["", f"Hold-back rule: a class may clear {im.GATE} only if its {where.split(' ')[0]} gate precision − 1.96·SE "
              f">= {im.GATE}; otherwise its map is capped at {im.HOLD_BACK_CAP}. Numbers are for the uncapped check map.", "",
              "| type | verdict | gate n | gate precision | SE | precision − 1.96·SE | decision |",
              "|---|---|---:|---:|---:|---:|---|"]
    for c in classes.values():
        se = "—" if pd.isna(c["check_gate_se"]) else f"{c['check_gate_se']:.4f}"
        lo = "—" if pd.isna(c["check_gate_lower_95"]) else f"{c['check_gate_lower_95']:.4f}"
        p = "—" if pd.isna(c["check_gate_precision"]) else f"{c['check_gate_precision']:.4f}"
        lines.append(f"| {c['label_type']} | {c['verdict']} | {c['check_gate_n']:,} | {p} | {se} | {lo} | "
                     f"**{c['decision']}** |")
    hb = holdout_bins[holdout_bins.checked] if len(holdout_bins) else holdout_bins
    lines += ["", f"{'In-sample' if in_sample else 'Held-out'} check: |new map − observed| <= {ACCEPT_TOLERANCE} "
              f"on bins with n >= {ACCEPT_MIN_N} (check map, before any hold-back cap)."]
    if len(hb) == 0:
        lines.append(f"No bin has n >= {ACCEPT_MIN_N} labels, so nothing was checked.")
    else:
        lines += ["", "| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | median 1.96·SE |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for (t, v), g in hb.groupby(["label_type", "verdict"], sort=False):
            lines.append(f"| {t} | {v} | {len(g)} | {int(g.within_tol.sum())} | {int(g['within_1.96se'].sum())} "
                         f"| {g['diff'].abs().max():.3f} | {1.96 * g.se.median():.3f} |")
        lines.append(f"| **all** | | {len(hb)} | {int(hb.within_tol.sum())} | {int(hb['within_1.96se'].sum())} "
                     f"| {hb['diff'].abs().max():.3f} | {1.96 * hb.se.median():.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--assessments", required=True, help="CSV from analysis/pull_ai_assessments.sh")
    p.add_argument("--out-dir", required=True, help="where to write maps, figures and tables (keep it outside the repo)")
    p.add_argument("--min-human-votes", type=int, default=2)
    p.add_argument("--holdout-fraction", type=float, default=0.5, help="0 disables the held-out check")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--since", default=None, help="drop assessments before this date (YYYY-MM-DD), e.g. a model swap")
    p.add_argument("--old-maps-dir", default=None, help="sidewalk-ai-api/accuracy_mappings, to check the lookup")
    args = p.parse_args(argv)

    df = load_assessments(args.assessments, args.since)
    print(f"{len(df):,} Agree/Disagree assessments in {df.schema.nunique()} schemas, "
          f"{df.ai_date.min()} .. {df.ai_date.max()}; model_id values: {sorted(df.model_id.dropna().unique())}")
    old_maps = None
    if args.old_maps_dir:
        old_maps = {t: json.load(open(os.path.join(args.old_maps_dir, f"{t.lower()}.json"))) for t in LABEL_TYPES}
    res = calibrate(df, args.min_human_votes, args.holdout_fraction, args.seed, old_maps)

    os.makedirs(os.path.join(args.out_dir, "accuracy_mappings"), exist_ok=True)
    for t, api_map in res["maps"].items():
        with open(os.path.join(args.out_dir, "accuracy_mappings", f"{t.lower()}.json"), "w") as f:
            json.dump(api_map, f, indent=4)
            f.write("\n")
        where = "in-sample" if res["in_sample"] else "held-out"
        plot_reliability([(f"{t}: AI says {v} ({VERDICT_TO_CLASS[v]})", res["reliability"][(t, v)])
                          for v in VERDICT_TO_CLASS],
                         os.path.join(args.out_dir, f"reliability_{t.lower()}.png"),
                         xlabel=f"raw confidence, bin lower edge ({where} labels with a human majority)",
                         observed_label="observed (humans)",
                         new_label="new map (all rows)" if res["in_sample"] else "new map (check fit)")
    res["summary"].to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    res["holdout_bins"].to_csv(os.path.join(args.out_dir, "holdout_bins.csv"), index=False)
    md = summary_markdown(res["summary"], res["holdout_bins"], res["classes"], args.min_human_votes, res["in_sample"])
    report = {"in_sample": res["in_sample"], "min_human_votes": args.min_human_votes,
              "holdout_fraction": args.holdout_fraction, "seed": args.seed, "since": args.since,
              "model_ids": sorted(df.model_id.dropna().unique()), "gate": im.GATE, "hold_back_cap": im.HOLD_BACK_CAP,
              "old_map_match": res["old_map_match"],
              "classes": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()}
                          for c in res["classes"].values()]}
    with open(os.path.join(args.out_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(md)
    if res["old_map_match"]:
        print("old maps reproduce served ai_accuracy (exact / consistent within ±5e-5 of the recorded confidence): " + ", ".join(f"{t} {v['exact']:.2%} / {v['consistent']:.2%}"
                                                   for t, v in res["old_map_match"].items()))
    print(md)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
