"""The deployment prior under every vote rule, and what each rule does to the 0.92 gate (issue #56).

Why
---
``deployment_priors.json`` was counted under the TRAINING margin rule, which is asymmetric (``incorrect`` at net
human Disagree > 1, ``correct`` only at net Agree > 2), so it overstated the incorrect share by 10-15 points for
NoCurbRamp, Obstacle and SurfaceProblem. ``data_generation.py`` now records the counts under every rule in
``data_generation.PRIOR_RULES`` and puts ``majority2`` (at least two decisive human votes, majority, ties out) at the
top level. This module measures the
four rules on ONE population and what the choice does to the served maps, without refitting or publishing anything:

- ``shares``: per label type, correct/incorrect counts under each rule over the human-authored, geometry-valid
  labels of the cached rawLabels (``data_generation.voted_labels``), restricted to the cities a
  ``deployment_priors.json`` entry lists. Hard check: the ``margin`` counts must equal that file's counts (the
  file's ``rules.margin`` when it has one, else its top level), or the command exits non-zero. Also the incorrect
  share per rule by severity and by tag (``analysis/slices.py`` slices; rates blank under ``MIN_N`` labels).
- ``gate``: per label type and rule, ``calibration.calibrate.calibrate`` fitted on ``val`` and evaluated on ``test``
  and ``test_heldout``, nothing held back: Agree and Disagree coverage, precision and gated n at 0.92. With
  ``--check-reports``, the margin rows must reproduce each type's existing ``calibration/calibration_report.json``.
- ``slices``: Agree gate coverage and precision per type x severity and type x tag under the ``margin`` map (the
  baseline) and the ``majority2`` map, with the training counts of each slice (``split.csv`` rows with
  ``split == train``). Predictions join ``split.csv`` on ``label_uid`` (``city:label_id``) with
  ``validate="one_to_one"``.
- ``observed``: from a production pull (``analysis/pull_ai_assessments.sh``), the AI Agree votes at served accuracy
  >= 0.92 and the share of them that the human majority under each rule confirms, for the round-3 revisions and for
  the older models separately. Backup schemas (``_bak_`` in the name) and deleted labels are dropped; labels are keyed
  by ``(schema, label_id)``. Also the four rules' incorrect shares over every AI-assessed label of the pull.

Every output is aggregate: per-type and per-slice counts, rates and the number of cities with a sha256 of their sorted
ids. No city name, label id or user id is written, so ``--out`` can be a directory in the repo; the inputs (rawLabels,
predictions, the production pull) hold private-city rows and stay outside it.

Usage (paths as in ``docs/deployment-prior-2026-11.md``)::

    R=~/Data/sidewalk-validator-ai-retrain-2026-10/round3
    python analysis/prior_rules.py shares --rawlabels-cache $R/rawlabels --priors $R/deployment_priors.json \\
        --out docs/deployment-prior-2026-11
    python analysis/prior_rules.py gate --data-dir $R --shares docs/deployment-prior-2026-11/shares.json \\
        --check-reports --out docs/deployment-prior-2026-11
    python analysis/prior_rules.py slices --data-dir $R --shares docs/deployment-prior-2026-11/shares.json \\
        --out docs/deployment-prior-2026-11
    python analysis/prior_rules.py observed --assessments ~/Data/.../ai_assessments_2026-11-11.csv \\
        --out docs/deployment-prior-2026-11
"""
import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_generation as dg  # noqa: E402
from analysis import slices as sl  # noqa: E402
from calibration import _isotonic_map as im  # noqa: E402
from calibration import calibrate as cal  # noqa: E402
from calibration.per_city import wilson_interval  # noqa: E402

LABEL_TYPES = ("Crosswalk", "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem")
RULES = dg.PRIOR_RULES
BASELINE_RULE, NEW_RULE = "margin", dg.DEFAULT_PRIOR_RULE
REPLAY_RULES = (BASELINE_RULE, NEW_RULE)
EVAL_SPLITS = ("test", "test_heldout")
GATE = im.GATE
# Round-3 revisions went live in production on 2026-10-27 at about 12:36 PDT (sidewalk-ai-api 1.2.0). The pull stores
# the assessment's DATE only, so that day holds both models' votes and is left out of both groups.
ROUND3_FROM = "2026-10-28"
OLDER_BEFORE = "2026-10-27"
REPORT_TOL = 1e-9


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(obj, path):
    """Write ``obj`` as indented JSON. The text is built before the file is opened, so a value that cannot be
    serialised leaves an existing file untouched instead of truncated (some outputs are read back on the next run)."""
    text = json.dumps(cal.without_nan(obj), indent=2, default=float) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def share(c, i):
    return i / (c + i) if c + i else float("nan")


def _f(v, fmt="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


# ---------------------------------------------------------------------------------------------------------------------
# shares
# ---------------------------------------------------------------------------------------------------------------------

def rule_classes(voted):
    """``voted`` (human-authored rows only) with one ``class_<rule>`` column per rule."""
    h = voted[voted["author"] == "human"].copy()
    for r in RULES:
        h[f"class_{r}"] = dg.prior_class(h["human_agree"], h["human_disagree"], r)
    return h


def share_slices(h, by, min_n=sl.MIN_N):
    """Per slice of ``h`` (from :func:`rule_classes`, with ``severity``/``tags``): decided n and incorrect share per
    rule; the share is blank (NaN) under ``min_n`` decided labels."""
    col = "severity" if by == "severity" else "tags"
    d = sl.with_slice(h if col in h.columns else h.assign(**{col: None}), by)
    rows = []
    for name, g in d.groupby("slice"):
        row = {"slice": name, "n_labels": int(len(g))}
        for r in RULES:
            c, i = int((g[f"class_{r}"] == "correct").sum()), int((g[f"class_{r}"] == "incorrect").sum())
            row[f"n_{r}"] = c + i
            row[f"incorrect_{r}"] = share(c, i) if c + i >= min_n else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    return sl._order(out.rename(columns={"n_labels": "n"}), by).rename(columns={"n": "n_labels"})


def margin_reference(entry):
    """The margin-rule counts a ``deployment_priors.json`` entry records: ``rules.margin``, else the top level of a
    pre-#56 entry."""
    if "margin" in (entry.get("rules") or {}):
        return {k: int(entry["rules"]["margin"][k]) for k in ("correct", "incorrect")}
    if entry.get("rule", "margin") != "margin":
        raise ValueError(f"entry is under rule {entry.get('rule')!r} and records no margin counts")
    return {k: int(entry[k]) for k in ("correct", "incorrect")}


def compute_shares(all_labels, label_types, priors=None, min_n=sl.MIN_N):
    """``{"types": {type: {"n_rows", "n_voted_human", "rules": {rule: {correct, incorrect, share_incorrect}},
    "check": ...}}, "slices": {type: {by: [rows]}}}``. With ``priors`` (a loaded ``deployment_priors.json``), each
    type's margin counts are compared with it (``check.ok``)."""
    out = {"types": {}, "slices": {}}
    for t in label_types:
        voted, dropped = dg.voted_labels(all_labels, t)
        counts = dg.prior_rule_counts(voted)
        rec = {"n_rows": int((all_labels["label_type"] == t).sum()), "dropped_geometry": dict(dropped),
               "n_voted_human": int((voted["author"] == "human").sum()),
               "rules": {r: {**c, "share_incorrect": share(c["correct"], c["incorrect"])} for r, c in counts.items()}}
        if priors is not None and t in priors:
            ref = margin_reference(priors[t])
            rec["check"] = {"file_margin": ref, "ok": ref == counts["margin"],
                            "file_n_rows": priors[t].get("n_rows"),
                            "n_rows_ok": priors[t].get("n_rows") in (None, rec["n_rows"])}
        out["types"][t] = rec
        h = rule_classes(voted)
        out["slices"][t] = {by: share_slices(h, by, min_n).to_dict("records") for by in sl.BY}
    return out


def shares_markdown(res, min_n=sl.MIN_N):
    parts = ["# Deployment prior under four vote rules (issue #56)", "",
             f"Human-authored, geometry-valid labels of the cached rawLabels ({res['_source']['n_cities']} cities). "
             "Cell: incorrect share (correct / incorrect counts).", "",
             "| Rule | " + " | ".join(res["types"]) + " |", "|---|" + "---:|" * len(res["types"])]
    for r in RULES:
        cells = []
        for t, rec in res["types"].items():
            c = rec["rules"][r]
            cells.append(f"{100 * c['share_incorrect']:.1f}% ({c['correct']:,} / {c['incorrect']:,})")
        parts.append(f"| {r} | " + " | ".join(cells) + " |")
    parts += ["", "Hard check against the deployment_priors.json margin counts: "
              + ", ".join(f"{t} {'ok' if rec.get('check', {}).get('ok') else 'MISMATCH'}"
                          for t, rec in res["types"].items()), ""]
    for t, tabs in res["slices"].items():
        for by, rows in tabs.items():
            parts += [f"## {t}, incorrect share by {by}", "",
                      f"Blank under {min_n} decided labels. n = labels in the slice; n_rule = decided under the rule.", "",
                      "| " + by + " | n | " + " | ".join(f"{r} (n)" for r in RULES) + " |",
                      "|---|---:|" + "---:|" * len(RULES)]
            for row in rows:
                parts.append(f"| {row['slice']} | {row['n_labels']:,} | "
                             + " | ".join(f"{_f(row[f'incorrect_{r}'])} ({row[f'n_{r}']:,})" for r in RULES) + " |")
            parts.append("")
    return "\n".join(parts)


def cmd_shares(args):
    with open(args.priors) as f:
        priors = json.load(f)
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    cities = None
    for t in label_types:
        c = priors[t].get("cities")
        if cities is None:
            cities = c
        elif c != cities:
            sys.exit(f"{args.priors}: the label types list different cities; run one type at a time")
    if not cities:
        sys.exit(f"{args.priors} lists no cities")
    frames = [dg.load_city(c, None, args.rawlabels_cache) for c in cities]
    all_labels = pd.concat(frames, ignore_index=True)
    res = compute_shares(all_labels, label_types, priors, args.min_n)
    # One hash over the inputs' contents without naming them: sha256 of the sorted "<file sha256>  <city>.csv" lines.
    files_hash = sha256_of_text("\n".join(sorted(
        f"{sha256_of(os.path.join(args.rawlabels_cache, c + '.csv'))}  {c}.csv" for c in cities)))
    res["_source"] = {"n_cities": len(cities), "cities_sha256": sha256_of_text("\n".join(sorted(cities))),
                      "rawlabels_files_sha256": files_hash,
                      "rawlabels_cache": os.path.basename(os.path.abspath(args.rawlabels_cache)),
                      "priors_file": os.path.basename(args.priors), "priors_sha256": sha256_of(args.priors),
                      "rows": int(len(all_labels)), "min_n": args.min_n}
    write_json(res, os.path.join(args.out, "shares.json"))
    with open(os.path.join(args.out, "shares.md"), "w") as f:
        f.write(shares_markdown(res, args.min_n) + "\n")
    bad = [t for t, r in res["types"].items() if not (r.get("check", {}).get("ok") and r["check"]["n_rows_ok"])]
    for t, r in res["types"].items():
        print(t, {k: (v["correct"], v["incorrect"], round(v["share_incorrect"], 4)) for k, v in r["rules"].items()},
              "check", r.get("check"))
    if bad:
        print(f"HARD CHECK FAILED: margin counts or n_rows differ from {args.priors} for {', '.join(bad)}")
        return 1
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------------------------------------------------

def rule_prior(shares_types, t, rule):
    c = shares_types[t]["rules"][rule]
    n = c["correct"] + c["incorrect"]
    return {"correct": c["correct"] / n, "incorrect": c["incorrect"] / n}


def gate_rows(pred, pi, splits=EVAL_SPLITS):
    """Per eval split: gate stats of the val-fitted, uncapped map under prior ``pi`` (``calibrate.calibrate``)."""
    out = {}
    for split in splits:
        if not (pred["split"] == split).any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, rep, _ = cal.calibrate(pred, pi, "val", split)
        if rep["eval"] == "n/a":
            out[split] = {"error": rep.get("eval_error", "n/a")}
            continue
        out[split] = {v: {k: rep["eval"][v][k] for k in ("n", "gate_coverage", "gate_precision", "gate_n")}
                      for v in cal.CLASSES}
    return out


def _missing(v):
    """None or NaN: a gate statistic with nothing behind it (no verdict passed)."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def check_report(gate_margin, report, split):
    """Compare margin-rule gate rows with an existing ``calibration_report.json`` (Agree as served; Disagree from
    ``hold_back.uncapped_eval`` when it was held back). Returns a list of mismatch strings."""
    bad = []
    if report.get("eval_split") != split or not isinstance(report.get("eval"), dict):
        return [f"report is for split {report.get('eval_split')!r}"]
    for v in cal.CLASSES:
        ref = report["eval"][v]
        if v in (report.get("hold_back") or {}).get("verdicts", []):
            ref = report["hold_back"]["uncapped_eval"][v]
        mine = gate_margin[v]
        for k in ("gate_coverage", "gate_precision", "gate_n"):
            a, b = mine[k], ref[k]
            a_missing, b_missing = _missing(a), _missing(b)
            if a_missing and b_missing:
                continue
            if a_missing != b_missing or abs(float(a) - float(b)) > REPORT_TOL:
                bad.append(f"{v} {k}: {a} vs report {b}")
    return bad


def cmd_gate(args):
    with open(args.shares) as f:
        shares = json.load(f)
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    res, failures = {"types": {}}, []
    for t in label_types:
        pred_path = os.path.join(args.data_dir, t, "predictions.csv")
        pred = pd.read_csv(pred_path, dtype={"label_uid": str})
        rec = {"predictions_sha256": sha256_of(pred_path), "rules": {}}
        for r in RULES:
            pi = rule_prior(shares["types"], t, r)
            rec["rules"][r] = {"prior_incorrect": pi["incorrect"], **gate_rows(pred, pi)}
        if args.check_reports:
            for split, sub in (("test", "calibration"), ("test_heldout", os.path.join("calibration", "eval-heldout"))):
                path = os.path.join(args.data_dir, t, sub, "calibration_report.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    report = json.load(f)
                bad = check_report(rec["rules"][BASELINE_RULE][split], report, split)
                rec.setdefault("report_check", {})[split] = {"ok": not bad, "mismatches": bad}
                failures += [f"{t} {split}: {b}" for b in bad]
        res["types"][t] = rec
    res["_source"] = {"shares": os.path.basename(args.shares), "fit_split": "val", "eval_splits": list(EVAL_SPLITS),
                      "gate": GATE, "hold_back": "none (uncapped maps)"}
    write_json(res, os.path.join(args.out, "gate.json"))
    with open(os.path.join(args.out, "gate.md"), "w") as f:
        f.write(gate_markdown(res) + "\n")
    print(gate_markdown(res))
    if failures:
        print("REPORT CHECK FAILED:\n  " + "\n  ".join(failures))
        return 1
    return 0


def _gate_cell(g):
    if not g or g.get("gate_coverage") is None:
        return ""
    if g["gate_n"] == 0:
        return "0 (never clears)"
    return f"{g['gate_coverage']:.3f} / {_f(g['gate_precision'])} ({g['gate_n']})"


def gate_markdown(res):
    parts = ["# The 0.92 gate under each prior rule (uncapped maps, fit on val)", "",
             "Cell: coverage / precision (gated n), reweighted to the rule's prior.", ""]
    for split in EVAL_SPLITS:
        for v, name in (("correct", "Agree"), ("incorrect", "Disagree")):
            parts += [f"## {name}, {split}", "", "| Type | " + " | ".join(RULES) + " |", "|---|" + "---:|" * len(RULES)]
            for t, rec in res["types"].items():
                parts.append(f"| {t} | " + " | ".join(_gate_cell((rec["rules"][r].get(split) or {}).get(v))
                                                       for r in RULES) + " |")
            parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# slices
# ---------------------------------------------------------------------------------------------------------------------

SPLIT_COLS = ["label_uid", "severity", "tags", "label_class", "split"]


def join_split(pred, split):
    """``pred`` with ``severity`` and ``tags`` from ``split`` (``split.csv``), joined on ``label_uid`` one to one; a
    duplicate ``label_uid`` on either side raises (``pandas.errors.MergeError``), a prediction without a split row
    raises ``ValueError``."""
    s = split[["label_uid", "severity", "tags"]]
    m = pred.merge(s, on="label_uid", how="left", validate="one_to_one", indicator=True)
    missing = int((m["_merge"] != "both").sum())
    if missing:
        raise ValueError(f"{missing} predictions have no split.csv row")
    return m.drop(columns="_merge")


def val_map(pred, pi):
    """The uncapped map ``calibrate.calibrate`` fits on ``val`` under prior ``pi``."""
    df = cal.verdicts(pred)
    curves, _ = cal.fit_curves(df[df.split == "val"], pi)
    return im.build_api_map(curves)


def served_and_weights(pred, pi):
    """Val-fitted uncapped map under ``pi``, then for every non-val row: ``verdict``, ``confidence``, ``correct``,
    ``served`` and the per-class weight ``w`` (``calibrate.prior_weights`` over that row's eval split)."""
    df = cal.verdicts(pred)
    amap = val_map(pred, pi)
    out = []
    for split in EVAL_SPLITS:
        e = df[df.split == split].copy()
        if not len(e):
            continue
        e["w"] = cal.prior_weights(e["true"], pi)
        e["served"] = 0.0
        for v in cal.CLASSES:
            m = (e.verdict == v).values
            e.loc[m, "served"] = im.api_lookup_many(amap[v], e.confidence.values[m])
        out.append(e)
    return pd.concat(out, ignore_index=True)


def agree_gate_slices(scored, train, by, min_n=sl.MIN_N):
    """Per slice of the Agree verdicts in ``scored``: n, weighted coverage and precision at the gate (blank under
    ``min_n`` Agree verdicts), and the training rows of the slice by class."""
    d = sl.with_slice(scored[scored.verdict == "correct"], by)
    tr = sl.with_slice(train, by)
    tr_counts = tr.groupby(["slice", "label_class"]).size()
    rows = []
    for name, g in d.groupby("slice"):
        row = {"slice": name, "n": int(len(g)), "coverage": float("nan"), "precision": float("nan"), "gate_n": 0}
        if len(g) >= min_n:
            s = im.gate_summary(g.served.values, g.correct.values, g.w.values)
            row.update(coverage=s["coverage"], precision=s["precision"], gate_n=s["n_pass"])
        row["train_correct"] = int(tr_counts.get((name, "correct"), 0))
        row["train_incorrect"] = int(tr_counts.get((name, "incorrect"), 0))
        rows.append(row)
    return sl._order(pd.DataFrame(rows), by)


def cmd_slices(args):
    with open(args.shares) as f:
        shares = json.load(f)
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    res = {"types": {}}
    for t in label_types:
        pred = pd.read_csv(os.path.join(args.data_dir, t, "predictions.csv"), dtype={"label_uid": str})
        split_path = os.path.join(args.data_dir, t, "split.csv")
        split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
        split = split.replace({"": None})
        train = split[split["split"] == "train"]
        res.setdefault("split_sha256", {})[t] = sha256_of(split_path)
        res["types"][t] = {}
        for r in (BASELINE_RULE, NEW_RULE):
            scored = join_split(served_and_weights(pred, rule_prior(shares["types"], t, r)), split)
            res["types"][t][r] = {sp: {by: agree_gate_slices(scored[scored.split == sp], train, by, args.min_n)
                                       .to_dict("records") for by in sl.BY}
                                  for sp in EVAL_SPLITS if (scored.split == sp).any()}
    res["_source"] = {"shares": os.path.basename(args.shares), "min_n": args.min_n, "rules": [BASELINE_RULE, NEW_RULE]}
    write_json(res, os.path.join(args.out, "slices.json"))
    with open(os.path.join(args.out, "slices.md"), "w") as f:
        f.write(slices_markdown(res, args.min_n) + "\n")
    print(f"wrote {os.path.join(args.out, 'slices.md')}")
    return 0


def slices_markdown(res, min_n=sl.MIN_N):
    parts = ["# Agree at the 0.92 gate by severity and tag: margin prior (baseline) vs majority2 prior", "",
             f"Uncapped maps fit on val; cells are coverage / precision, reweighted to the rule's prior, blank under "
             f"{min_n} Agree verdicts in the slice. n = Agree verdicts in the slice; train = training rows of the slice "
             f"(correct / incorrect). A label with several tags counts in each.", ""]
    for t, rules in res["types"].items():
        for sp in EVAL_SPLITS:
            if sp not in rules[BASELINE_RULE]:
                continue
            for by in sl.BY:
                a = {row["slice"]: row for row in rules[BASELINE_RULE][sp][by]}
                b = {row["slice"]: row for row in rules[NEW_RULE][sp][by]}
                parts += [f"## {t}, {sp}, by {by}", "",
                          f"| {by} | n | {BASELINE_RULE} cov / prec | {NEW_RULE} cov / prec | Δ coverage | Δ precision "
                          f"| train (c / i) |", "|---|---:|---:|---:|---:|---:|---:|"]
                for name, ra in a.items():
                    rb = b.get(name, {})
                    dc = rb.get("coverage", np.nan) - ra["coverage"] if rb else np.nan
                    dp = rb.get("precision", np.nan) - ra["precision"] if rb else np.nan
                    parts.append(f"| {name} | {ra['n']:,} | {_f(ra['coverage'])} / {_f(ra['precision'])} | "
                                 f"{_f(rb.get('coverage'))} / {_f(rb.get('precision'))} | {_f(dc, '{:+.3f}')} | "
                                 f"{_f(dp, '{:+.3f}')} | {ra['train_correct']:,} / {ra['train_incorrect']:,} |")
                parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# observed (production outcomes)
# ---------------------------------------------------------------------------------------------------------------------

TYPE_BY_MODEL = {f"sidewalk-validator-ai-{t.lower()}": t for t in LABEL_TYPES}


def load_assessments(path):
    """The pull as a DataFrame: backup schemas and deleted labels dropped, ``(schema, label_id)`` asserted unique,
    ``period`` set to ``round3`` / ``older`` / ``changeover`` from ``ai_date`` (see :data:`ROUND3_FROM`). Returns
    ``(df, counts)`` where ``counts`` records what was dropped."""
    d = pd.read_csv(path, dtype={"schema": str, "label_type": str, "model_id": str, "ai_result": str, "deleted": str})
    counts = {"rows": int(len(d))}
    bak = d["schema"].str.contains("_bak_", regex=False)
    counts["backup_schema_rows"] = int(bak.sum())
    d = d[~bak]
    deleted = d["deleted"].astype(str).str.lower().isin(["t", "true"])
    counts["deleted_rows"] = int(deleted.sum())
    d = d[~deleted].copy()
    d["label_uid"] = d["schema"] + ":" + d["label_id"].astype(str)
    if not d["label_uid"].is_unique:
        raise ValueError(f"(schema, label_id) is not unique in {path}: {int(d['label_uid'].duplicated().sum())} repeats")
    # The model a vote came from is named by model_id (one per type, no revision), so the type is checked against it.
    d["model_type"] = d["model_id"].map(TYPE_BY_MODEL)
    mismatch = d["model_type"] != d["label_type"]
    counts["model_type_mismatch"] = int(mismatch.sum())  # e.g. a label retyped after it was assessed; dropped
    d = d[~mismatch].copy()
    date = pd.to_datetime(d["ai_date"]).dt.strftime("%Y-%m-%d")
    d["period"] = np.where(date >= ROUND3_FROM, "round3", np.where(date < OLDER_BEFORE, "older", "changeover"))
    counts["periods"] = d["period"].value_counts().to_dict()
    counts["n_schemas"] = int(d["schema"].nunique())
    counts["ai_date_min"], counts["ai_date_max"] = str(date.min()), str(date.max())
    return d, counts


def wilson(k, n):
    """95% Wilson score interval of ``k`` successes in ``n`` (``calibration.per_city.wilson_interval``); NaNs for
    ``n == 0``.

    >>> [round(v, 3) for v in wilson(199, 199)]
    [0.981, 1.0]
    """
    if n == 0:
        return float("nan"), float("nan")
    lo, hi = wilson_interval(k / n, n)
    return float(lo), float(hi)


def confirmed_by_rule(g):
    """Per rule: labels the human votes decide, the share decided ``correct`` (the AI Agree confirmed) and its 95%
    Wilson interval."""
    out = {"n": int(len(g)), "n_with_human_vote": int(((g.h_agree + g.h_dis) > 0).sum())}  # Agree/Disagree only
    for r in RULES:
        cls = dg.prior_class(g["h_agree"].to_numpy(), g["h_dis"].to_numpy(), r)
        c, i = int((cls == "correct").sum()), int((cls == "incorrect").sum())
        lo, hi = wilson(c, c + i)
        out[r] = {"decided": c + i, "confirmed": c, "share_confirmed": (c / (c + i)) if c + i else float("nan"),
                  "ci_lo": lo, "ci_hi": hi}
    return out


def observed_tables(d, gate=GATE):
    """``{"gated_agree": {period: {type: confirmed_by_rule}}, "population": {type: {rule: counts}}}``. The population
    table counts every AI-assessed label (any verdict, any period) under each rule, as a second population for the
    rule shares."""
    gated = d[(d["ai_result"] == "Agree") & (d["ai_accuracy"] >= gate)]
    out = {"gated_agree": {}, "gated_agree_by_severity": {}, "population": {}}
    for period in ("round3", "older"):
        p = gated[gated["period"] == period]
        out["gated_agree"][period] = {t: confirmed_by_rule(p[p.label_type == t]) for t in LABEL_TYPES}
    r3 = gated[gated["period"] == "round3"]
    sev = sl.with_slice(r3 if "severity" in r3.columns else r3.assign(severity=None), "severity")
    out["gated_agree_by_severity"]["round3"] = {
        t: {name: confirmed_by_rule(g) for name, g in sev[sev.label_type == t].groupby("slice")} for t in LABEL_TYPES}
    for t in LABEL_TYPES:
        g = d[d.label_type == t]
        rec = {}
        for r in RULES:
            cls = dg.prior_class(g["h_agree"].to_numpy(), g["h_dis"].to_numpy(), r)
            c, i = int((cls == "correct").sum()), int((cls == "incorrect").sum())
            rec[r] = {"correct": c, "incorrect": i, "share_incorrect": share(c, i)}
        out["population"][t] = rec
    return out


def replay_maps(d, maps, gate=GATE, tol=1e-6):
    """Serve the round-3 production Agree confidences through each rule's map: ``{type: {"served_check": ...,
    rule: {"gated": n, **confirmed_by_rule}}}``.

    ``maps`` is ``{type: {rule: api_map}}``. ``served_check`` compares the margin map's lookup with the served
    ``ai_accuracy`` (the round-3 served Agree map is the margin map, uncapped for Agree), which both checks the replay
    and confirms that the round-3 period holds round-3 votes: ``share_equal`` near 1 means it does."""
    out = {}
    r3 = d[(d["period"] == "round3") & (d["ai_result"] == "Agree")]
    for t, by_rule in maps.items():
        g = r3[r3.label_type == t]
        rec = {}
        if BASELINE_RULE in by_rule and len(g):
            served = im.api_lookup_many(by_rule[BASELINE_RULE]["correct"], g["ai_confidence"].to_numpy())
            rec["served_check"] = {"n": int(len(g)),
                                   "share_equal": float((np.abs(served - g["ai_accuracy"].to_numpy()) <= tol).mean())}
        for r, amap in by_rule.items():
            served = im.api_lookup_many(amap["correct"], g["ai_confidence"].to_numpy())
            rec[r] = {"gated": int((served >= gate).sum()), **confirmed_by_rule(g[served >= gate])}
        out[t] = rec
    return out


def _obs_cell(r, min_n):
    if r["decided"] < min_n:
        return f"({r['decided']:,})"
    return f"{r['share_confirmed']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] ({r['decided']:,})"


def observed_markdown(res, min_n=sl.MIN_N):
    src = res["_source"]
    parts = ["# Observed: AI Agree votes at served accuracy >= 0.92, confirmed by the human majority", "",
             f"Production pull `{src['assessments']}` (sha256 `{src['assessments_sha256'][:16]}`), "
             f"{src['counts']['n_schemas']} schemas, AI dates {src['counts']['ai_date_min']} .. "
             f"{src['counts']['ai_date_max']}. Round 3 = assessments dated {ROUND3_FROM} or later; older = before "
             f"{OLDER_BEFORE}; {OLDER_BEFORE} itself (the changeover day) is in neither. Cell: share confirmed "
             f"[95% Wilson interval] (decided n); share blank under {min_n} decided.", ""]
    for period in ("round3", "older"):
        parts += [f"## {period}", "", "| Type | gated Agree | with a decisive human vote | " + " | ".join(RULES) + " |",
                  "|---|---:|---:|" + "---:|" * len(RULES)]
        for t, rec in res["gated_agree"][period].items():
            parts.append(f"| {t} | {rec['n']:,} | {rec['n_with_human_vote']:,} | " + " | ".join(
                _obs_cell(rec[r], min_n) for r in RULES) + " |")
        parts.append("")
    parts += ["## Round 3 by severity (majority2)", "", "| Type | severity | gated Agree | confirmed (decided) |",
              "|---|---|---:|---:|"]
    for t, by in res["gated_agree_by_severity"]["round3"].items():
        for name, rec in by.items():
            r = rec[NEW_RULE]
            parts.append(f"| {t} | {name} | {rec['n']:,} | "
                         f"{_f(r['share_confirmed'] if r['decided'] >= min_n else float('nan'))} ({r['decided']:,}) |")
    if res.get("replay"):
        parts += ["", "## Round 3 replayed through each prior's map (Agree verdicts)", "",
                  "The round-3 production Agree confidences served through the val-fitted, uncapped Agree map of each "
                  "prior: how many would clear 0.92, and the share of those the human majority2 rule confirms. "
                  "served = share of rows where the margin map's lookup equals the served accuracy (checks the "
                  "replay and the round-3 date split).", "",
                  "| Type | Agree rows | served | " + " | ".join(f"{r}: gated, confirmed" for r in REPLAY_RULES) + " |",
                  "|---|---:|---:|" + "---:|" * len(REPLAY_RULES)]
        for t, rec in res["replay"].items():
            chk = rec.get("served_check", {})
            parts.append(f"| {t} | {chk.get('n', 0):,} | {_f(chk.get('share_equal'))} | " + " | ".join(
                f"{rec[r]['gated']:,}, {_obs_cell(rec[r][NEW_RULE], min_n)}" for r in REPLAY_RULES) + " |")
    parts += ["", "## Every AI-assessed label of the pull: incorrect share under each rule", "",
              "| Type | " + " | ".join(RULES) + " |", "|---|" + "---:|" * len(RULES)]
    for t, rec in res["population"].items():
        parts.append(f"| {t} | " + " | ".join(
            f"{100 * rec[r]['share_incorrect']:.1f}% ({rec[r]['correct'] + rec[r]['incorrect']:,})" for r in RULES)
            + " |")
    return "\n".join(parts) + "\n"


def cmd_observed(args):
    d, counts = load_assessments(args.assessments)
    res = observed_tables(d)
    if args.data_dir and args.shares:
        with open(args.shares) as f:
            shares = json.load(f)
        maps = {}
        for t in LABEL_TYPES:
            pred = pd.read_csv(os.path.join(args.data_dir, t, "predictions.csv"), dtype={"label_uid": str})
            maps[t] = {r: val_map(pred, rule_prior(shares["types"], t, r)) for r in REPLAY_RULES}
        res["replay"] = replay_maps(d, maps)
    res["_source"] = {"assessments": os.path.basename(args.assessments), "assessments_sha256": sha256_of(args.assessments),
                      "counts": counts, "round3_from": ROUND3_FROM, "older_before": OLDER_BEFORE, "gate": GATE,
                      "min_n": args.min_n}
    write_json(res, os.path.join(args.out, "observed.json"))
    text = observed_markdown(res, args.min_n)
    with open(os.path.join(args.out, "observed.md"), "w") as f:
        f.write(text)
    print(text)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    types = ",".join(LABEL_TYPES)

    s = sub.add_parser("shares", help="counts under every rule on one population; hard check of the margin counts")
    s.add_argument("--rawlabels-cache", required=True, help="directory of cached <city_id>.csv rawLabels exports")
    s.add_argument("--priors", required=True, help="deployment_priors.json whose cities and margin counts to check")
    s.add_argument("--label-types", default=types)
    s.add_argument("--min-n", type=int, default=sl.MIN_N)
    s.add_argument("--out", required=True, help="directory for shares.json and shares.md")
    s.set_defaults(func=cmd_shares)

    g = sub.add_parser("gate", help="gate coverage/precision under each rule's prior (uncapped, fit on val)")
    g.add_argument("--data-dir", required=True, help="root holding <label_type>/predictions.csv")
    g.add_argument("--shares", required=True, help="shares.json from the shares command")
    g.add_argument("--label-types", default=types)
    g.add_argument("--check-reports", action="store_true",
                   help="the margin rows must match <type>/calibration/calibration_report.json (test) and "
                        "<type>/calibration/eval-heldout/calibration_report.json (test_heldout) where present")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gate)

    c = sub.add_parser("slices", help="Agree gate by severity and tag, margin vs majority2 prior, with training counts")
    c.add_argument("--data-dir", required=True, help="root holding <label_type>/predictions.csv and split.csv")
    c.add_argument("--shares", required=True)
    c.add_argument("--label-types", default=types)
    c.add_argument("--min-n", type=int, default=sl.MIN_N)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_slices)

    o = sub.add_parser("observed", help="production AI Agree votes at >= 0.92 confirmed by the human majority")
    o.add_argument("--assessments", required=True, help="CSV from analysis/pull_ai_assessments.sh (PRIVATE)")
    o.add_argument("--data-dir", default=None,
                   help="with --shares: replay the round-3 Agree confidences through each prior's map "
                        "(<label_type>/predictions.csv under this root)")
    o.add_argument("--shares", default=None, help="shares.json from the shares command, for the replay")
    o.add_argument("--min-n", type=int, default=sl.MIN_N)
    o.add_argument("--out", required=True)
    o.set_defaults(func=cmd_observed)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
