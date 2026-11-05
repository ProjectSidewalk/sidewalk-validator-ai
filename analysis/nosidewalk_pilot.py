"""The NoSidewalk pilot's results table: crop arms x seeds (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/42).

Reads what ``nosidewalk_submit.sh`` leaves under one data root, per arm (``depth``: depth-sized crops, the default;
``whole``: ``--fixed-crop-half-px 1024``, the whole 90 degree view) and seed:

- ``<data>/<arm>/NoSidewalk/split.csv`` (``split_data.py --group-col face_id``, with ``face_id`` and ``city``),
- ``<data>/<arm>/NoSidewalk/predictions_s<seed>.csv`` (``evaluate.py --tag s<seed>``),
- ``<data>/<arm>/NoSidewalk/calibration/s<seed>/nosidewalk.json`` (``calibration/calibrate.py``; optional).

For ``test`` and ``test_heldout`` it reports, per run: rows and incorrect rows, AUC with a 95 % interval from
resampling whole block faces (labels on one face are not independent; a row without a face resamples with its
panorama), within-city AUC (``retrain_report.within_city_auc``: the incorrect share differs by city, so a pooled AUC
partly measures the city), and the Agree verdict at the 0.92 gate: coverage (share of the split's rows the gate lets
through as Agree), precision and its Wilson 95 % lower bound. The gate uses the run's calibration map as
sidewalk-ai-api serves it; without a map, the raw confidence (labelled ``raw``). These are single-split numbers on
the split's own class mix: the serving bar of ``docs/nosidewalk-2026-11.md`` (5-fold out-of-fold gate precision) is a
separate phase-B step. Then the seeds are summarised per arm (mean, and min-max).

Only aggregates are written. ``--doc`` splices the markdown table between the ``<!-- pilot-results:start -->`` and
``<!-- pilot-results:end -->`` markers of the study page.

Usage::

    python analysis/nosidewalk_pilot.py --data /gscratch/.../nosidewalk-2026-11 --out-json report/pilot.json \\
        --out-md report/pilot.md [--doc docs/nosidewalk-2026-11.md]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import _isotonic_map as im  # noqa: E402
from analysis.retrain_report import cluster_boot, within_city_auc  # noqa: E402

LABEL_TYPE = "NoSidewalk"
ARMS = ("depth", "whole")
SEEDS = (42, 43, 44)
SPLITS = ("test", "test_heldout")
GATE = im.GATE
Z95 = 1.96
START, END = "<!-- pilot-results:start -->", "<!-- pilot-results:end -->"


def run_paths(data, arm, seed, label_type=LABEL_TYPE):
    """``(split.csv, predictions_s<seed>.csv, calibration map)`` of one run."""
    base = os.path.join(data, arm, label_type)
    return (os.path.join(base, "split.csv"), os.path.join(base, f"predictions_s{seed}.csv"),
            os.path.join(base, "calibration", f"s{seed}", f"{label_type.lower()}.json"))


def join_predictions(split, predictions):
    """Predictions with ``city`` and a resampling group (``face_id``, else ``pano:<pano_id>``), joined on
    ``label_uid`` (one row per prediction)."""
    cols = ["label_uid", "city", "pano_id"] + (["face_id"] if "face_id" in split else [])
    df = predictions.merge(split[cols], on="label_uid", how="left", validate="one_to_one")
    if df["city"].isna().any():
        raise ValueError(f"{int(df['city'].isna().sum())} predictions have no split.csv row")
    face = df["face_id"].fillna("").astype(str) if "face_id" in df else pd.Series("", index=df.index)
    df["group"] = face.where(face != "", "pano:" + df["pano_id"].astype(str))
    return df


def agree_gate(df, api_map=None, gate=GATE):
    """Coverage, precision and Wilson 95 % lower bound of the Agree verdict at ``gate``.

    >>> d = pd.DataFrame({"p_correct": [0.99, 0.97, 0.6, 0.1], "p_incorrect": [0.01, 0.03, 0.4, 0.9],
    ...                   "true": ["correct", "incorrect", "correct", "incorrect"]})
    >>> g = agree_gate(d)
    >>> g["rule"], g["n_pass"], g["coverage"], g["precision"]
    ('raw', 2, 0.5, 0.5)
    """
    verdict, conf = im.predict_verdicts(df["p_correct"].to_numpy(), df["p_incorrect"].to_numpy())
    if api_map is not None:
        served = im.api_lookup_many(api_map["correct"], conf)
        rule = "map"
    else:
        served = conf
        rule = "raw"
    passed = (verdict == "correct") & (served >= gate)
    n_pass = int(passed.sum())
    precision = float((df["true"].to_numpy()[passed] == "correct").mean()) if n_pass else float("nan")
    lower = float(im.wilson_lower(precision, n_pass, z=Z95)) if n_pass else float("nan")
    return {"rule": rule, "n_pass": n_pass, "coverage": n_pass / len(df) if len(df) else float("nan"),
            "precision": precision, "precision_wilson95_lower": lower}


def split_metrics(df, api_map=None, n_boot=1000, seed=0):
    """Every number of one run on one split (``df`` from :func:`join_predictions`, one split)."""
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
    out["agree_gate"] = agree_gate(df, api_map)
    return out


def collect(data, arms=ARMS, seeds=SEEDS, n_boot=1000, label_type=LABEL_TYPE):
    """``[{arm, seed, split, ...split_metrics}]`` for every run whose predictions exist; missing runs are listed
    under ``missing``."""
    rows, missing = [], []
    for arm in arms:
        for seed in seeds:
            split_path, pred_path, map_path = run_paths(data, arm, seed, label_type)
            if not (os.path.exists(split_path) and os.path.exists(pred_path)):
                missing.append(f"{arm}/s{seed}")
                continue
            split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
            df = join_predictions(split, pd.read_csv(pred_path))
            api_map = None
            if os.path.exists(map_path):
                with open(map_path) as f:
                    api_map = json.load(f)
            for s in SPLITS:
                part = df[df["split"] == s]
                if len(part):
                    rows.append({"arm": arm, "seed": seed, "split": s, **split_metrics(part, api_map, n_boot, seed)})
    return rows, missing


def summarise(rows):
    """Per (arm, split): runs, mean and min-max of AUC, within-city AUC, Agree gate coverage and precision."""
    out = []
    keys = sorted({(r["arm"], r["split"]) for r in rows}, key=lambda k: (ARMS.index(k[0]) if k[0] in ARMS else 9, k[1]))
    for arm, split in keys:
        part = [r for r in rows if r["arm"] == arm and r["split"] == split]
        def stat(get):
            v = np.array([get(r) for r in part], dtype=float)
            v = v[np.isfinite(v)]
            return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())} if len(v) else None
        out.append({"arm": arm, "split": split, "runs": len(part), "n": part[0]["n"],
                    "n_incorrect": part[0]["n_incorrect"],
                    "auc": stat(lambda r: r["auc"]), "within_city_auc": stat(lambda r: r["within_city_auc"]),
                    "gate_coverage": stat(lambda r: r["agree_gate"]["coverage"]),
                    "gate_precision": stat(lambda r: r["agree_gate"]["precision"]),
                    "gate_rule": ",".join(sorted({r["agree_gate"]["rule"] for r in part}))})
    return out


def _fmt(s, digits=3):
    if s is None:
        return "n/a"
    if s["min"] == s["max"]:
        return f"{s['mean']:.{digits}f}"
    return f"{s['mean']:.{digits}f} ({s['min']:.{digits}f}–{s['max']:.{digits}f})"


def markdown(summary, rows, missing):
    """The results table (per arm and split, mean and min–max over seeds) and the per-run AUC intervals."""
    lines = ["| arm | split | runs | rows (incorrect) | AUC | within-city AUC | Agree gate coverage | Agree gate precision |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['arm']} | {s['split']} | {s['runs']} | {s['n']:,} ({s['n_incorrect']:,}) | {_fmt(s['auc'])} "
                     f"| {_fmt(s['within_city_auc'])} | {_fmt(s['gate_coverage'])} | {_fmt(s['gate_precision'])} "
                     f"({s['gate_rule']}) |")
    lines += ["", "Per run, AUC with a 95 % interval from resampling block faces; Agree gate precision with its Wilson "
              "95 % lower bound:", "", "| arm | seed | split | AUC [95 % CI] | gate n | gate precision (lower) |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        ci = r["auc_ci95"]
        g = r["agree_gate"]
        auc = f"{r['auc']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        prec = "n/a" if not g["n_pass"] else f"{g['precision']:.3f} ({g['precision_wilson95_lower']:.3f})"
        lines.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {auc} | {g['n_pass']} | {prec} |")
    if missing:
        lines += ["", f"Runs without predictions (not included): {', '.join(missing)}."]
    return "\n".join(lines) + "\n"


def splice(doc_text, block):
    """Replace what lies between the result markers of ``doc_text`` with ``block``; ValueError without markers.

    >>> splice("a\\n<!-- pilot-results:start -->\\nold\\n<!-- pilot-results:end -->\\nb", "new\\n")
    'a\\n<!-- pilot-results:start -->\\nnew\\n<!-- pilot-results:end -->\\nb'
    """
    i, j = doc_text.find(START), doc_text.find(END)
    if i < 0 or j < i:
        raise ValueError(f"no {START} ... {END} markers in the document")
    return doc_text[:i + len(START)] + "\n" + block + doc_text[j:]


def without_nan(obj):
    if isinstance(obj, dict):
        return {k: without_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [without_nan(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="the pilot's data root (<data>/<arm>/NoSidewalk/...)")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--doc", default=None, help="splice the table into this page between the pilot-results markers")
    args = p.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows, missing = collect(args.data, arms, seeds, args.n_boot)
    if not rows:
        raise SystemExit(f"no run under {args.data} has predictions (looked for {', '.join(missing)})")
    summary = summarise(rows)
    for path in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(without_nan({"summary": summary, "runs": rows, "missing": missing}), f, indent=1)
        f.write("\n")
    md = markdown(summary, rows, missing)
    with open(args.out_md, "w") as f:
        f.write(md)
    if args.doc:
        with open(args.doc) as f:
            text = f.read()
        with open(args.doc, "w") as f:
            f.write(splice(text, md))
    print(md)
    return summary


if __name__ == "__main__":
    main()
