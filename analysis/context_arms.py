"""Issue #14 report: the three "more context" arms (``context_model.py``) against the round-3 model, on the same rows.

For each label type, reads ``evaluate.py``'s outputs in ``<data-dir>/<label_type>/``: the round-3 baseline
(``metrics.json``/``predictions.csv``) and each arm's tagged run (``metrics_<arm>.json``/``predictions_<arm>.csv``,
written by ``context_model.py predict`` + ``evaluate.py --skip-inference --tag <arm> --per-city``), plus seed
replicates named ``<run>-s<seed>`` (the baseline's ``baseline-s43``/``-s44`` from the confident-learning batch). Writes
to ``--out``:

* ``arms.md`` / ``arms.json``: per type x arm, test and held-out AUC with the paired panorama-bootstrap difference
  against the baseline on the same rows, the within-city AUC (round-3 definition), macro F1 at the model's verdict,
  and incorrect/correct recall at precision 0.92 and 0.95; plus the held-out deployments one by one (private ones
  lettered, never named).
* ``seeds.md``: every seed replicate against the baseline of the *same* seed (``confident_learning.py``'s layout).
* ``slices.md``: type x severity and type x tag, each arm against the baseline on the same rows
  (``analysis/slices.paired_delta_table``), with the slice's training rows and the thin flag.
* ``clipping.md``: how often the wide crop is the whole view (``wide_clipped``, ``wide_half_px`` quantiles), per type,
  from ``<label_type>/context/index.csv``.
* ``provenance.json``: sha256 of every file read.

The table helpers are ``analysis/confident_learning.py``'s (same within-city definition, same bootstrap), imported,
not copied.

Usage::

    python analysis/context_arms.py report --data-dir $DATA --label-types NoCurbRamp,Obstacle,SurfaceProblem \\
        --public-cities cities.json --out docs/context-arms-2026-11
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import confident_learning as cl  # noqa: E402
from analysis import slices as sl  # noqa: E402

ARMS = ("two_scale", "view_heatmap", "side_info")
TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
LEVELS = ("0.92", "0.95")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(m, split):
    """Mean of the two classes' F1 in ``evaluate.py``'s ``metrics.json`` for ``split`` (at the model's verdict), or
    None.

    >>> macro_f1({"test": {"per_class": {"correct": {"f1-score": 0.8}, "incorrect": {"f1-score": 0.6}}}}, "test")
    0.7
    """
    pc = (m.get(split) or {}).get("per_class") or {}
    f = [pc[c]["f1-score"] for c in ("correct", "incorrect") if c in pc]
    return round(float(np.mean(f)), 10) if len(f) == 2 else None


def arm_row(data_dir, label_type, name, preds, within_min_n=10):
    """``confident_learning.metric_row`` plus macro F1 and recall at precision 0.92 and 0.95 for both classes."""
    r = cl.metric_row(data_dir, label_type, name, preds, within_min_n)
    if "missing" in r:
        return r
    with open(cl.metrics_path(data_dir, label_type, name)) as f:
        m = json.load(f)
    for split, key in (("test", "test"), ("test_heldout", "heldout")):
        r[f"macro_f1_{key}"] = macro_f1(m, split)
        for cls in ("incorrect", "correct"):
            for lvl in LEVELS:
                r[f"{cls}_recall_p{lvl[2:]}_{key}"] = cl._metric(m, split, f"recall_at_precision:{cls}:{lvl}")
    return r


def arms_markdown(label_type, rows, heldout=None, within_min_n=10):
    """One table per type: the baseline and each arm, Δ against the baseline on the same rows."""
    lines = [f"### {label_type}", "",
             "| run | test AUC | Δ test [95% CI] | within-city test | held-out AUC | Δ held-out [95% CI] | "
             "within-city held-out | macro F1 test / held-out | incorrect R@P0.92 / 0.95 held-out | "
             "correct R@P0.92 / 0.95 held-out |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = cl._fmt
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['variant']} | not yet run | | | | | | | | |")
            continue
        d_test, d_held = (cl._delta_cell({"delta": r[f"{k}_delta"], "delta_lo": r.get(f"{k}_delta_lo"),
                                          "delta_hi": r.get(f"{k}_delta_hi")} if f"{k}_delta" in r else None)
                          for k in ("test", "heldout"))
        lines.append("| " + " | ".join([
            r["variant"], f(r["test_auc"]), d_test, f"{f(r['within_city_test'])} ({r['cities_test']})",
            f(r["heldout_auc"]), d_held, f"{f(r['within_city_heldout'])} ({r['cities_heldout']})",
            f"{f(r['macro_f1_test'])} / {f(r['macro_f1_heldout'])}",
            f"{f(r['incorrect_recall_p92_heldout'])} / {f(r['incorrect_recall_p95_heldout'])}",
            f"{f(r['correct_recall_p92_heldout'])} / {f(r['correct_recall_p95_heldout'])}"]) + " |")
    if heldout:
        models = [k for k in heldout[0] if k not in ("deployment", "n", "n_incorrect")]
        lines += ["", f"Held-out deployments, AUC per run (blank under {within_min_n} labels or with one class):", "",
                  "| deployment | n | incorrect | " + " | ".join(models) + " |", "|---|---:|---:|" + "---:|" * len(models)]
        for h in heldout:
            lines.append(f"| {h['deployment']} | {h['n']:,} | {h['n_incorrect']:,} | "
                         + " | ".join(f(h[k]) for k in models) + " |")
    return "\n".join(lines) + "\n"


def slice_deltas(preds, arms, train, min_n=None, n_boot=None, seed=0):
    """Severity and tag slices of each arm against the baseline on the same rows, test and held-out
    (``slices.paired_delta_table``), each slice with its training rows (``train``: split.csv's train rows) and
    ``needs_data`` (fewer than ``slices.THIN_TRAIN``). Records with ``split``, ``by``, ``variant`` and the table's
    columns; empty without baseline predictions."""
    base = preds.get("baseline")
    if base is None:
        return []
    min_n = sl.MIN_N if min_n is None else min_n
    n_boot = sl.N_BOOT if n_boot is None else n_boot
    counts = {by: sl.with_slice(train, by).groupby("slice").size().to_dict() for by in sl.BY}
    out = []
    for name in arms:
        p = preds.get(name)
        if p is None:
            continue
        for split in ("test", "test_heldout"):
            part = p[p["split"] == split]
            if part.empty:
                continue
            for by in sl.BY:
                t = sl.paired_delta_table(part, base, by, min_n, n_boot, seed)
                for r in t.to_dict("records"):
                    n_train = int(counts[by].get(r["slice"], 0))
                    out.append({"split": split, "by": by, "variant": name, **r, "train_n": n_train,
                                "needs_data": n_train < sl.THIN_TRAIN})
    return out


def slices_markdown(label_type, records, arms):
    """``slice_deltas`` records as one table per (split, by): slice, rows, training rows (thin flagged), baseline AUC,
    and each arm's Δ with its interval."""
    lines = []
    for split in ("test", "test_heldout"):
        for by in sl.BY:
            rs = [r for r in records if r["split"] == split and r["by"] == by]
            if not rs:
                continue
            lines += [f"### {label_type} / {split} / by {by}", "",
                      f"| {by.capitalize()} | n | incorrect | training rows | baseline AUC | "
                      + " | ".join(f"{a} Δ [95% CI]" for a in arms) + " |",
                      "|---|---:|---:|---:|---:|" + "---:|" * len(arms)]
            for name in dict.fromkeys(r["slice"] for r in rs):
                by_arm = {r["variant"]: r for r in rs if r["slice"] == name}
                first = next(iter(by_arm.values()))
                train = f"{first['train_n']:,}" + (" ⚠ thin" if first["needs_data"] else "")
                cells = [str(name), f"{first['n']:,}", f"{first['n_incorrect']:,}", train, cl._fmt(first.get("auc_b"))]
                cells += [cl._delta_cell(by_arm.get(a)) for a in arms]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def clipping_row(index):
    """How the wide crops of one type came out: ``n``, the share whose requested context did not fit
    (``wide_clipped``), the share that is the whole view, and ``wide_half_px`` quantiles.

    >>> r = clipping_row(pd.DataFrame({"wide_half_px": [400, 1024, 1024, 800], "wide_clipped": [0, 1, 0, 0]}))
    >>> r["n"], r["clipped_share"], r["whole_view_share"], r["wide_half_px_q50"]
    (4, 0.25, 0.5, 912.0)
    """
    half = index["wide_half_px"].astype(float)
    row = {"n": int(len(index)), "clipped_share": float(index["wide_clipped"].astype(int).mean()),
           "whole_view_share": float((half >= 1024).mean())}
    row.update({f"wide_half_px_q{int(q * 100)}": float(half.quantile(q)) for q in QUANTILES})
    return row


def clipping_markdown(rows):
    lines = ["# How much context the wide crop adds", "",
             "The wide crop asks for 4x the depth-sized half-width, capped at the whole 2048 px view (half-width "
             "1024). `clipped` = the requested context did not fit; `whole view` = the wide crop is the view "
             "itself. Quantiles of the wide half-width in view px.", "",
             "| type | labels | clipped | whole view | " + " | ".join(f"q{int(q * 100)}" for q in QUANTILES) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(QUANTILES)]
    for lt, r in rows.items():
        if r is None:
            lines.append(f"| {lt} | no context index yet | | |" + " |" * len(QUANTILES))
            continue
        lines.append(f"| {lt} | {r['n']:,} | {r['clipped_share']:.0%} | {r['whole_view_share']:.0%} | "
                     + " | ".join(f"{r[f'wide_half_px_q{int(q * 100)}']:.0f}" for q in QUANTILES) + " |")
    return "\n".join(lines) + "\n"


def report(args):
    from analysis import retrain_report as rr
    label_types = [t.strip() for t in args.label_types.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    public, public_source = rr.public_cities(args.public_cities)
    os.makedirs(args.out, exist_ok=True)
    everything = {"_public_cities_source": public_source, "_arms": arms}
    prov = {}
    tables, slice_tables, seed_tables, clip = [], [], [], {}

    def note(path):
        if os.path.exists(path):
            prov[os.path.relpath(path, args.data_dir)] = sha256_file(path)

    for lt in label_types:
        split_path = os.path.join(args.data_dir, lt, "split.csv")
        if not os.path.exists(split_path):
            tables.append(f"### {lt}\n\nnot yet run (no {lt}/split.csv)\n")
            clip[lt] = None
            continue
        note(split_path)
        split_df = cl.read_split(split_path)
        names = ("baseline", *arms)
        preds = {n: cl.load_predictions(args.data_dir, lt, n, split_df) for n in names}
        preds = {k: v for k, v in preds.items() if v is not None}
        for n in names:
            note(cl.metrics_path(args.data_dir, lt, n))
            note(os.path.join(args.data_dir, lt, "predictions.csv" if n == "baseline" else f"predictions_{n}.csv"))
        rows = [arm_row(args.data_dir, lt, n, preds, args.within_min_n) for n in names]
        deltas = cl.overall_deltas(preds, arms, args.n_boot, args.seed)
        for r in rows:
            for split, key in (("test", "test"), ("test_heldout", "heldout")):
                d = deltas.get((r["variant"], split))
                if d:
                    r[f"{key}_delta"], r[f"{key}_delta_lo"], r[f"{key}_delta_hi"] = d["delta"], d["delta_lo"], d["delta_hi"]
        heldout = cl.heldout_rows(preds, public, args.within_min_n) if preds else []
        train = split_df[split_df["split"] == "train"]
        slices = slice_deltas(preds, arms, train, args.slice_min_n, args.n_boot, args.seed)
        everything[lt] = {"arms": rows, "heldout": heldout, "slices": slices}
        tables.append(arms_markdown(lt, rows, heldout, args.within_min_n))
        if slices:
            slice_tables.append(slices_markdown(lt, slices, [a for a in arms if a in preds]))
        runs = {k: v for k, v in cl.seed_runs(args.data_dir, lt, args.base_seed).items() if k[0] in names}
        if runs:
            for name in runs.values():
                note(cl.metrics_path(args.data_dir, lt, name))
                note(os.path.join(args.data_dir, lt, f"predictions_{name}.csv"))
            srows = cl.seed_rows(args.data_dir, lt, runs, split_df, args.within_min_n, args.n_boot, args.seed)
            summary = cl.seed_summary(srows)
            everything[lt]["seeds"] = {"rows": srows, "summary": summary}
            seed_tables.append(cl.seeds_markdown(lt, srows, summary))
        index_path = os.path.join(args.data_dir, lt, "context", "index.csv")
        if os.path.exists(index_path):
            note(index_path)
            clip[lt] = clipping_row(pd.read_csv(index_path))
        else:
            clip[lt] = None
        everything[lt]["clipping"] = clip[lt]
    with open(os.path.join(args.out, "arms.md"), "w") as f:
        f.write("# Context arms against the round-3 model\n\n"
                "Each arm is trained with the round-3 recipe on the round-3 split and scored on the same rows; Δ = arm "
                "− baseline AUC with a 95% interval from resampling whole panoramas of the evaluation split, the same "
                "resample for both models. Within-city AUC: per-deployment AUCs weighted by n_incorrect x n_correct "
                f"(deployments with at least {args.within_min_n} labels and both classes; count in brackets). Private "
                "deployments are lettered.\n\n" + "\n".join(tables))
    with open(os.path.join(args.out, "arms.json"), "w") as f:
        json.dump(everything, f, indent=2, default=float)
        f.write("\n")
    if slice_tables:
        with open(os.path.join(args.out, "slices.md"), "w") as f:
            f.write("# Severity and tag slices: each arm against the round-3 baseline\n\n"
                    "AUC of the incorrect score on the same rows; Δ = arm − baseline with a 95% interval from "
                    "resampling whole panoramas (`analysis/slices.py`). Blank under the minimum rows or when the "
                    f"smaller class has fewer than {sl.MIN_CLASS_FOR_CI}; ⚠ thin = fewer than {sl.THIN_TRAIN} "
                    "training rows. Not corrected for the number of slices: at 95%, about one cell in twenty "
                    "excludes zero by chance.\n\n" + "\n".join(slice_tables))
    if seed_tables:
        with open(os.path.join(args.out, "seeds.md"), "w") as f:
            f.write("# Seed replicates\n\nEvery run repeated with another training seed (`TAG=<run>-s<seed>`), the "
                    f"chain's own run counted as seed {args.base_seed}; each compared with the baseline of the same "
                    "seed.\n\n" + "\n".join(seed_tables))
    with open(os.path.join(args.out, "clipping.md"), "w") as f:
        f.write(clipping_markdown(clip))
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump({"files_sha256": dict(sorted(prov.items())), "public_cities_source": public_source}, f, indent=2)
        f.write("\n")
    print("\n".join(tables))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="arms.md, arms.json, seeds.md, slices.md, clipping.md, provenance.json")
    r.add_argument("--data-dir", required=True, help="data_generation.py's --out (round 3: retrain-2026-10)")
    r.add_argument("--label-types", default=",".join(TYPES))
    r.add_argument("--arms", default=",".join(ARMS),
                   help="evaluate.py tags to compare with the baseline (e.g. add view_heatmap-r448)")
    r.add_argument("--public-cities", default=None,
                   help="a saved /v3/api/cities response or a file of public city_ids; default: the live API. Only "
                        "public deployments are named")
    r.add_argument("--within-min-n", type=int, default=10)
    r.add_argument("--slice-min-n", type=int, default=None)
    r.add_argument("--n-boot", type=int, default=None, help="panorama resamples (slices.py default)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--base-seed", type=int, default=42)
    r.add_argument("--out", required=True)
    r.set_defaults(func=report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
