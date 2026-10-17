"""Accuracy maps for the CURRENTLY DEPLOYED validator models, calibrated on production outcomes.

Input is one row per label the AI assessed, as pulled from every city's ``label_ai_assessment`` joined to its human
votes (``analysis/pull_ai_assessments.sh``). Columns used: ``schema, label_id, label_type, ai_result, ai_accuracy,
ai_confidence, ai_date, model_id, h_agree, h_dis``. ``ai_result`` is the raw verdict (Agree / Disagree) whether or
not it passed the 0.92 gate, ``ai_confidence`` the raw softmax confidence, ``ai_accuracy`` the value the current maps
served, and ``h_agree``/``h_dis`` the human votes on that label **excluding the AI's own vote**. ``label_id`` is
per-city, so labels are keyed on ``schema:label_id`` throughout.

**Outcome.** For each label type and raw verdict (Agree -> the map's ``correct`` class, Disagree -> ``incorrect``)
we keep labels with at least ``--min-human-votes`` human Agree/Disagree votes. Three definitions (``--outcome``):

* ``tie_half`` (default): 1 if the strict human majority matches the verdict, 0 if it opposes it, 0.5 for a tie.
  Ties count as half right instead of being dropped.
* ``strict``: the same, with tied labels dropped. At two votes that keeps only unanimous pairs and flatters the
  marginal classes (out-of-fold, CurbRamp's gated precision is 0.988 strict vs 0.966 tie_half).
* ``share``: the share of the human votes that match the verdict (1-1 -> 0.5, 2-1 -> 0.667, 3-0 -> 1). This
  measures agreement with one randomly chosen human voter, not with the crowd's verdict, so it is capped by how
  often humans disagree with each other: a model that always sided with the majority would score only 0.85 on
  NoCurbRamp, SurfaceProblem and Obstacle (0.93 CurbRamp, 0.97 Crosswalk), below the 0.92 gate by construction.

The two definitions not used for the maps are run as a sensitivity check. A non-decreasing isotonic curve of the outcome against ``ai_confidence`` is fitted and written in
sidewalk-ai-api's format (``_isotonic_map``): each map value is the isotonic block mean, except that blocks with
fewer than 300 labels are served at their 90% Wilson lower bound (thin-block guard).

**Shipped maps** are fitted on ALL labelled rows.

**Hold-back rule (5-fold cross-fitting).** Labels are split into ``--folds`` folds by a hash of ``seed:schema:label_id``.
Each labelled row gets an out-of-fold served value from a map fitted without its fold. Pooling every row whose
out-of-fold value clears 0.92, a class may clear the gate only if the Wilson 95% lower bound of their mean outcome is
>= 0.92; otherwise its shipped map is capped at 0.9199 everywhere (it never becomes a vote). The same out-of-fold
values drive the reported gate precision and the per-bin reliability check, and the decision at a 0.90 bound is
reported for context. With ``--folds 1`` there is no cross-fitting and every such number is labelled in-sample.

Outputs under ``--out-dir``: ``accuracy_mappings/<type>.json`` (drop-in for sidewalk-ai-api), ``reliability_<type>.png``,
``summary.csv``, ``holdout_bins.csv``, ``summary.md``, ``calibration_report.json`` (fitted confidence range, gate
numbers and decision per class) and ``labels_oof.csv`` (one row per assessed label with old, new and out-of-fold
served values; private-city data, keep it outside the repo).

Caveats, in order of importance:

* **These maps are only valid for the model revisions that produced the data.** As of 2026-10-17 those are the
  revisions pinned in sidewalk-ai-api's ``sidewalk_ai_api/model_map.py`` (training_date 08-16-2025):
  sidewalk-validator-ai-curbramp ``969ad5a``, -nocurbramp ``9f3f3bb``, -obstacle ``8f0fe8f``, -surfaceproblem
  ``0e1439d``, -crosswalk ``b7936e5``. The assessment table records ``model_id`` but not the revision, and the API
  pinned revisions only on 2026-10-15, so a ``--since`` date is the only guard against a model swap.
* **Selection.** Humans do not vote on a random sample of AI-assessed labels (the Triage queue routes AI-contested
  labels to them, and labels collect votes at different rates). Disagree precision rises with the number of human
  votes, so the >= 2-vote subset is not obviously pessimistic for Disagree; the direction of the bias cannot be
  identified from vote counts alone because the order in which votes arrived is not exported.
* **Folds are split per label, not per panorama** (the export has no ``pano_id``), so neighbouring labels on one
  panorama can fall in different folds and the out-of-fold check is somewhat optimistic.
* The human vote share is a noisy reference (measured expert-vs-crowd agreement is 86-91%).

Usage::

    python -m calibration.calibrate_from_production --assessments ~/Data/<pull>/ai_assessments.csv \\
        --out-dir ~/Data/<pull>/phase0 \\
        --min-human-votes 2 --folds 5 --seed 42 --since 2025-08-16 --old-maps-dir ../sidewalk-ai-api/accuracy_mappings
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
GATE_Z = 1.96          # Wilson 95% lower bound for the hold-back rule
CONTEXT_BOUND = 0.90   # the decision at this bound is reported for context only
OUTCOMES = ("tie_half", "strict", "share")
OUTCOME_TEXT = {"tie_half": "the strict human majority matches the verdict (1), opposes it (0), or ties (0.5)",
                "strict": "the strict human majority matches the verdict (1/0), ties dropped",
                "share": "share of human Agree/Disagree votes that match the verdict"}


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


def fold_ids(label_uids, folds, seed):
    """Deterministic fold per label (0..folds-1) from a hash of ``seed:label_uid``; all zeros when ``folds < 2``."""
    if folds < 2:
        return np.zeros(len(label_uids), dtype=int)
    return np.array([int(hashlib.md5(f"{seed}:{x}".encode()).hexdigest()[:12], 16) % folds for x in label_uids])


def add_outcomes(df, min_human_votes, outcome="share"):
    """Add ``labelled``, ``tie`` and ``correct`` (the outcome in [0, 1]) for each assessed label.

    Labelled when ``h_agree + h_dis >= min_human_votes`` (``strict`` additionally drops ties). ``correct``:
    ``tie_half`` 1 / 0 by strict majority and 0.5 on a tie; ``strict`` 1 / 0 by strict majority; ``share`` the share
    of human votes matching the AI's verdict.
    """
    if min_human_votes < 1:
        raise ValueError("min_human_votes must be >= 1 (0 would score unvoted labels as ties)")
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    df = df.copy()
    enough = df.h_n >= min_human_votes
    df["tie"] = enough & (df.h_agree == df.h_dis)
    matching = np.where(df.ai_result == "Agree", df.h_agree, df.h_dis)
    majority_matches = (np.where(df.h_agree > df.h_dis, "Agree", "Disagree") == df.ai_result).astype(float)
    if outcome == "share":
        df["labelled"] = enough
        df["correct"] = np.where(df.h_n > 0, matching / df.h_n.where(df.h_n > 0, 1), np.nan)
    elif outcome == "tie_half":
        df["labelled"] = enough
        df["correct"] = np.where(df.tie, 0.5, majority_matches)
    else:
        df["labelled"] = enough & ~df.tie
        df["correct"] = majority_matches
    return df


def check_old_map(rows, api_map, rounding=5e-5):
    """How well ``api_lookup`` on a served map reproduces the recorded ``ai_accuracy``.

    A row is *consistent* if its recorded accuracy is one the map could serve for some confidence within
    ``±rounding`` of the recorded one (an older export rounded confidence to 4 decimals; the current one does not).

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
        hit = np.abs(served - r.ai_accuracy.values) <= 1e-4
        exact += int(hit.sum())
        consistent += int(hit.sum())
        for c, g in r[~hit].groupby("ai_confidence"):   # only rows the exact lookup missed
            cand = np.r_[vals[(th >= c - rounding) & (th <= c + rounding)], im.api_lookup(class_map, c - rounding)]
            consistent += int(np.sum(np.min(np.abs(g.ai_accuracy.values[:, None] - cand[None, :]), axis=1) <= 1e-4))
    return {"exact": exact / len(rows), "consistent": consistent / len(rows)}


def hold_back_decision(map_reaches_gate, precision, n, bound=im.GATE, z=GATE_Z):
    """Apply the hold-back rule to one class. Returns ``(decision, wilson_lower_bound)``.

    ``"below gate"`` when the shipped map never reaches the gate (nothing to hold back); ``"clears"`` when the
    Wilson lower bound of the pooled out-of-fold gate precision is >= ``bound``; ``"held back"`` otherwise,
    including when no out-of-fold label clears the gate.
    """
    lower = float(im.wilson_lower(precision, n, z)) if n > 0 else float("nan")
    if not map_reaches_gate:
        return "below gate", lower
    if n > 0 and lower >= bound:
        return "clears", lower
    return "held back", lower


def calibrate(df, min_human_votes=2, folds=5, seed=42, outcome="tie_half", old_maps=None):
    """Fit the maps, apply the cross-fitted hold-back rule and compute every table. Pure: no file I/O.

    Args:
        df: output of ``load_assessments``.
        old_maps: optional ``{label_type: api_map}`` of the currently served maps, used only to check that they
            reproduce ``ai_accuracy`` (a test of ``api_lookup`` against production).

    Returns:
        dict with ``maps`` ({label_type: shipped api_map}), ``summary`` and ``holdout_bins`` (DataFrames),
        ``reliability`` ({(label_type, verdict): DataFrame}), ``classes`` (per-class report), ``df`` (input plus
        ``labelled``, ``correct``, ``fold``, ``new_accuracy``, ``oof_accuracy``), ``old_map_match`` and
        ``in_sample`` (True without cross-fitting).
    """
    df = add_outcomes(df, min_human_votes, outcome)
    df["fold"] = fold_ids(df.label_uid.values, folds, seed)
    in_sample = folds < 2
    df["new_accuracy"] = np.nan
    df["oof_accuracy"] = np.nan
    maps, rows, bins, reliability, classes, old_map_match = {}, [], [], {}, {}, {}
    for t in LABEL_TYPES:
        ship_map = {}
        if old_maps is not None and t in old_maps:
            old_map_match[t] = check_old_map(df[df.label_type == t], old_maps[t])
        for verdict, cls in VERDICT_TO_CLASS.items():
            sel = ((df.label_type == t) & (df.ai_result == verdict)).values
            lab_idx = np.flatnonzero(sel & df.labelled.values)
            conf = df.ai_confidence.values[lab_idx]
            y = df.correct.values[lab_idx]
            fold = df.fold.values[lab_idx]
            curve = im.fit_isotonic(conf, y)
            ship_map[cls] = im.curve_to_class_map(curve)
            oof = np.zeros(len(lab_idx))
            if in_sample:
                oof = im.api_lookup_many(ship_map[cls], conf)
            else:
                for k in range(folds):
                    m = fold == k
                    fit = ~m
                    if m.any():
                        k_map = im.curve_to_class_map(im.fit_isotonic(conf[fit], y[fit]))
                        oof[m] = im.api_lookup_many(k_map, conf[m])
            passed = oof >= im.GATE
            n_pass = int(passed.sum())
            prec = float(y[passed].mean()) if n_pass else float("nan")
            reaches = max(ship_map[cls].values()) >= im.GATE
            decision, lower = hold_back_decision(reaches, prec, n_pass)
            context_decision, _ = hold_back_decision(reaches, prec, n_pass, bound=CONTEXT_BOUND)
            if decision == "held back":
                ship_map[cls] = im.cap_class_map(ship_map[cls])
                oof = np.minimum(oof, im.HOLD_BACK_CAP)
            df.loc[df.index[lab_idx], "oof_accuracy"] = oof
            all_idx = np.flatnonzero(sel)
            new_acc = im.api_lookup_many(ship_map[cls], df.ai_confidence.values[all_idx])
            df.loc[df.index[all_idx], "new_accuracy"] = new_acc
            old_acc = df.ai_accuracy.values[all_idx]
            classes[(t, verdict)] = {"label_type": t, "verdict": verdict, "decision": decision,
                                     "decision_at_0.90": context_decision, "oof_gate_n": n_pass,
                                     "oof_gate_precision": prec, "oof_gate_wilson95_lower": lower,
                                     "fit_range": im.fit_range(curve),
                                     "blocks": 0 if curve is None else int(len(curve.block_n)),
                                     "thin_blocks": 0 if curve is None else int(curve.thin().sum()),
                                     "guard_mean_shift": float(np.mean(curve.lower(conf) - curve.value(conf)))
                                     if curve is not None else float("nan")}
            old_pass_lab = df.ai_accuracy.values[lab_idx] >= im.GATE
            new_pass_lab = oof >= im.GATE
            old_pass, new_pass = old_acc >= im.GATE, new_acc >= im.GATE
            ship_pass_lab = im.api_lookup_many(ship_map[cls], conf) >= im.GATE   # in-sample, the map that ships
            tab = im.reliability_table(conf, y, predicted=oof)
            tab["old_predicted"] = im.reliability_table(conf, y, predicted=df.ai_accuracy.values[lab_idx]).predicted.values
            reliability[(t, verdict)] = tab
            rows.append({"label_type": t, "verdict": verdict, "n_assessed": len(all_idx), "n_labelled": len(lab_idx),
                         "tie_share": float(df.tie.values[lab_idx].mean()) if len(lab_idx) else np.nan,
                         "observed_precision": float(y.mean()) if len(y) else np.nan,
                         "old_coverage": float(old_pass.mean()) if len(all_idx) else np.nan,
                         "old_gate_precision": float(y[old_pass_lab].mean()) if old_pass_lab.any() else np.nan,
                         "old_gate_n": int(old_pass_lab.sum()),
                         "new_coverage": float(new_pass.mean()) if len(all_idx) else np.nan,
                         "new_gate_precision": float(y[new_pass_lab].mean()) if new_pass_lab.any() else np.nan,
                         "new_gate_n": int(new_pass_lab.sum()),
                         "shipped_gate_n": int(ship_pass_lab.sum()),
                         "shipped_gate_precision": float(y[ship_pass_lab].mean()) if ship_pass_lab.any() else np.nan,
                         "old_votes": int(old_pass.sum()), "new_votes": int(new_pass.sum()),
                         "votes_lost": int((old_pass & ~new_pass).sum()),
                         "votes_gained": int((new_pass & ~old_pass).sum()),
                         "decision": decision,
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
            "in_sample": in_sample, "outcome": outcome}


def _pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def _num(x, fmt=".4f"):
    return "—" if pd.isna(x) else format(x, fmt)


def summary_markdown(res, min_human_votes, sensitivity=()):
    """Render the summary, vote effects, hold-back decisions, sensitivity and reliability check as Markdown."""
    summary, holdout_bins, classes, in_sample = res["summary"], res["holdout_bins"], res["classes"], res["in_sample"]
    where = "in-sample" if in_sample else "out-of-fold"
    outcome = OUTCOME_TEXT[res["outcome"]]
    lines = [f"Outcome: {outcome}; labels with >= {min_human_votes} human votes. Coverage is over all assessed labels "
             "of that type and verdict, under the shipped map (all labelled rows, block means with the thin-block guard, "
             f"after the hold-back rule). Gate precision is the mean outcome of labelled rows whose {where} served "
             f"value clears {im.GATE}{' (optimistic: no cross-fitting)' if in_sample else ''}."
             + ("" if in_sample else " The out-of-fold check validates the procedure, not this exact map; the "
                "shipped-map column is the same quantity in-sample for the map that ships (its gate set differs from "
                "the pooled out-of-fold one at the margin)."), "",
             f"| type | verdict | assessed | labelled (ties) | precision | old coverage | old gate precision "
             f"| new coverage | new gate precision ({where}) | shipped-map gate precision (in-sample) | decision |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.label_type} | {r.verdict} | {r.n_assessed:,} | {r.n_labelled:,} ({_pct(r.tie_share)}) "
                     f"| {_pct(r.observed_precision)} | {_pct(r.old_coverage)} | {_pct(r.old_gate_precision)} "
                     f"(n={r.old_gate_n:,}) | {_pct(r.new_coverage)} | {_pct(r.new_gate_precision)} "
                     f"(n={r.new_gate_n:,}) | {_pct(r.shipped_gate_precision)} (n={r.shipped_gate_n:,}) | {r.decision} |")
    lines += ["", "Votes cast over all assessed labels (Disagree verdicts that no longer clear the gate are submitted "
              "as Unsure):", "", "| type | verdict | old votes | new votes | lost (→ Unsure) | gained |",
              "|---|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.label_type} | {r.verdict} | {r.old_votes:,} | {r.new_votes:,} | {r.votes_lost:,} "
                     f"| {r.votes_gained:,} |")
    tot = summary.groupby("verdict")[["old_votes", "new_votes", "votes_lost", "votes_gained"]].sum()
    for v, r in tot.iterrows():
        lines.append(f"| **all** | {v} | {r.old_votes:,} | {r.new_votes:,} | {r.votes_lost:,} | {r.votes_gained:,} |")
    lines += ["", f"Hold-back rule ({where}): a class may clear {im.GATE} only if the Wilson 95% lower bound of the "
              f"pooled {where} gate precision is >= {im.GATE}; otherwise its map is capped at {im.HOLD_BACK_CAP}. "
              f"Numbers are before the cap. The last column is the decision at a {CONTEXT_BOUND} bound, for "
              "context only.", "",
              "| type | verdict | pooled n | precision | Wilson 95% lower | decision | at 0.90 |",
              "|---|---|---:|---:|---:|---|---|"]
    for c in classes.values():
        lines.append(f"| {c['label_type']} | {c['verdict']} | {c['oof_gate_n']:,} | {_num(c['oof_gate_precision'])} "
                     f"| {_num(c['oof_gate_wilson95_lower'])} | **{c['decision']}** | {c['decision_at_0.90']} |")
    lines += ["", f"Thin-block guard: blocks with < {im.THIN_BLOCK_N} labels served at their 90% Wilson lower bound. "
              "Mean shift = mean over labelled rows of (served − block mean).", "",
              "| type | verdict | blocks | thin blocks | mean shift |", "|---|---|---:|---:|---:|"]
    for c in classes.values():
        lines.append(f"| {c['label_type']} | {c['verdict']} | {c['blocks']} | {c['thin_blocks']} "
                     f"| {_num(c['guard_mean_shift'], '+.4f')} |")
    for sens in sensitivity:
        lines += ["", f"Sensitivity, `{sens['outcome']}` outcome ({OUTCOME_TEXT[sens['outcome']]}). Not used for the "
                  "shipped maps.", "",
                  "| type | verdict | pooled n | precision | Wilson 95% lower | decision |", "|---|---|---:|---:|---:|---|"]
        for c in sens["classes"].values():
            lines.append(f"| {c['label_type']} | {c['verdict']} | {c['oof_gate_n']:,} | {_num(c['oof_gate_precision'])} "
                         f"| {_num(c['oof_gate_wilson95_lower'])} | {c['decision']} |")
    hb = holdout_bins[holdout_bins.checked] if len(holdout_bins) else holdout_bins
    lines += ["", f"Reliability check ({where} served value vs observed mean outcome, bins with n >= {ACCEPT_MIN_N}, "
              f"before any cap): |diff| <= {ACCEPT_TOLERANCE}, and within 1.96 binomial SE."]
    if len(hb) == 0:
        lines.append(f"No bin has n >= {ACCEPT_MIN_N} labels, so nothing was checked.")
    else:
        lines += ["", "| type | verdict | bins checked | within ±0.01 | within 1.96·SE | max abs diff | mean diff |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for (t, v), g in hb.groupby(["label_type", "verdict"], sort=False):
            lines.append(f"| {t} | {v} | {len(g)} | {int(g.within_tol.sum())} | {int(g['within_1.96se'].sum())} "
                         f"| {g['diff'].abs().max():.3f} | {g['diff'].mean():+.3f} |")
        lines.append(f"| **all** | | {len(hb)} | {int(hb.within_tol.sum())} | {int(hb['within_1.96se'].sum())} "
                     f"| {hb['diff'].abs().max():.3f} | {hb['diff'].mean():+.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--assessments", required=True, help="CSV from analysis/pull_ai_assessments.sh")
    p.add_argument("--out-dir", required=True, help="where to write maps, figures and tables (keep it outside the repo)")
    p.add_argument("--min-human-votes", type=int, default=2)
    p.add_argument("--outcome", choices=OUTCOMES, default="tie_half")
    p.add_argument("--folds", type=int, default=5, help="cross-fitting folds; 1 disables (in-sample numbers)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--since", default=None, help="drop assessments before this date (YYYY-MM-DD), e.g. a model swap")
    p.add_argument("--old-maps-dir", default=None, help="sidewalk-ai-api/accuracy_mappings, to check the lookup")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the runs with the other outcome definitions")
    args = p.parse_args(argv)
    if args.min_human_votes < 1:
        p.error("--min-human-votes must be >= 1 (0 would score unvoted labels as ties)")

    df = load_assessments(args.assessments, args.since)
    print(f"{len(df):,} Agree/Disagree assessments in {df.schema.nunique()} schemas, "
          f"{df.ai_date.min()} .. {df.ai_date.max()}; model_id values: {sorted(df.model_id.dropna().unique())}")
    old_maps = None
    if args.old_maps_dir:
        old_maps = {t: json.load(open(os.path.join(args.old_maps_dir, f"{t.lower()}.json"))) for t in LABEL_TYPES}
    res = calibrate(df, args.min_human_votes, args.folds, args.seed, args.outcome, old_maps)
    sens = [] if args.no_sensitivity else [calibrate(df, args.min_human_votes, args.folds, args.seed, o)
                                           for o in OUTCOMES if o != args.outcome]

    os.makedirs(os.path.join(args.out_dir, "accuracy_mappings"), exist_ok=True)
    where = "in-sample" if res["in_sample"] else "out-of-fold"
    for t, api_map in res["maps"].items():
        with open(os.path.join(args.out_dir, "accuracy_mappings", f"{t.lower()}.json"), "w") as f:
            json.dump(api_map, f, indent=4)
            f.write("\n")
        plot_reliability([(f"{t}: AI says {v} ({VERDICT_TO_CLASS[v]})", res["reliability"][(t, v)])
                          for v in VERDICT_TO_CLASS],
                         os.path.join(args.out_dir, f"reliability_{t.lower()}.png"),
                         xlabel=f"raw confidence, bin lower edge (labelled rows, {where} map)",
                         observed_label="observed (mean outcome)", new_label=f"new map ({where})")
    res["summary"].to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    res["holdout_bins"].to_csv(os.path.join(args.out_dir, "holdout_bins.csv"), index=False)
    res["df"][["schema", "label_id", "label_type", "ai_result", "ai_date", "ai_confidence", "ai_accuracy",
               "new_accuracy", "oof_accuracy", "labelled", "tie", "correct", "h_agree", "h_dis", "h_unsure",
               "fold"]].to_csv(
        os.path.join(args.out_dir, "labels_oof.csv"), index=False)
    md = summary_markdown(res, args.min_human_votes, sens)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(md)

    def clean(c):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()}
    report = {"outcome": args.outcome, "in_sample": res["in_sample"], "folds": args.folds, "seed": args.seed,
              "min_human_votes": args.min_human_votes, "since": args.since,
              "model_ids": sorted(df.model_id.dropna().unique()), "gate": im.GATE, "hold_back_cap": im.HOLD_BACK_CAP,
              "map_values": f"isotonic block mean; blocks with < {im.THIN_BLOCK_N} labels at their 90% Wilson lower bound",
              "old_map_match": res["old_map_match"], "classes": [clean(c) for c in res["classes"].values()],
              "sensitivity": [{"outcome": r["outcome"], "classes": [clean(c) for c in r["classes"].values()]}
                              for r in sens]}
    with open(os.path.join(args.out_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    if res["old_map_match"]:
        print("old maps reproduce served ai_accuracy (exact / consistent within ±5e-5 of the recorded confidence): "
              + ", ".join(f"{t} {v['exact']:.2%} / {v['consistent']:.2%}" for t, v in res["old_map_match"].items()))
    print(md)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
