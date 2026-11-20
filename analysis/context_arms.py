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
import shutil
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import confident_learning as cl  # noqa: E402
from analysis import slices as sl  # noqa: E402

ARMS = ("two_scale", "view_heatmap", "view_heatmap-r448", "side_info", "side_info_nohq")  # report --arms default
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


# ---------------------------------------------------------------------------------------------------------------------
# Launch side: the shadow data directory, preflight and the per-run check (login node: pandas only, never torch)
# ---------------------------------------------------------------------------------------------------------------------

SHADOW_FILE = "shadow_run.json"
#: Copied into the shadow for every type; split.csv byte-identical, so run_key.json's split hash is the baseline's.
SHADOW_COPY = ("split.csv", "manifest.csv", "generation_run.json")
#: October's round-3 outputs, copied when present (the drift row reads them; nothing writes to them).
OCTOBER_FILES = ("metrics.json", "predictions.csv", "metrics_baseline-s43.json", "metrics_baseline-s44.json",
                 "predictions_baseline-s43.csv", "predictions_baseline-s44.csv")
CROP_DIRS = ("correct", "incorrect")
EVAL_SPLITS = ("val", "test", "test_heldout")   # evaluate.EVAL_SPLITS (not imported: evaluate pulls in sklearn)
PANO_MB = 11.4               # mean staged panorama, measured on klone 2026-11-19 (400-file sample)
CONTEXT_KB_PER_LABEL = 100   # a 448 px view (~40 KB) and a wide crop (~60 KB) at WebP q95
WRITE_TEST_BYTES = 8 << 20   # scrubbed has refused real writes while a 0-byte touch succeeded
BACKBONE_CACHE = "hub/models--facebook--dinov2-large/snapshots/*/config.json"
SIDE_SHARE_COLUMNS = ("severity", "image_capture_date", "depth_m")
#: split.csv columns every run reads (fine_tune_classifier.load_split_frames, evaluate.load_split).
BASE_COLUMNS = ("label_uid", "city", "pano_id", "label_class", "crop_path", "split")
#: The run record a training job copies next to its outputs (``<data>/<type>/runs/<tag>/``), before any delete.
RECORD_FILES = ("training_run.json", "run_key.json", "trainer_state.json", "config.json")


def _real_inside(path, root):
    """True when ``path`` is ``root`` or below it, after resolving symlinks."""
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _git_state():
    from fine_tune_classifier import git_state  # torch-free at import (tests/test_context_shadow.py checks)
    return git_state()


def read_shadow(data_dir):
    """``<data_dir>/shadow_run.json`` as a dict, or None when ``data_dir`` is not a shadow."""
    path = os.path.join(data_dir, SHADOW_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def staged_subset(split_df, manifest_df, store):
    """The rows of ``split_df`` (and the manifest rows with the same ``label_uid``) whose panorama file is in the pano
    store ``store`` (``data_generation.pano_store_path``), plus ``{split: {class: n}}`` of what is kept. Refuses
    (``SystemExit``) when any split of ``split_df`` would be left with fewer than two classes."""
    import data_generation as dg
    have = [os.path.exists(dg.pano_store_path(store, c, p)) for c, p in zip(split_df["city"], split_df["pano_id"])]
    kept = split_df[np.asarray(have, dtype=bool)]
    counts = {s: {c: int(n) for c, n in g["label_class"].value_counts().sort_index().items()}
              for s, g in kept.groupby("split")}
    lost = [s for s in sorted(split_df["split"].unique()) if len(counts.get(s, {})) < 2]
    if lost:
        raise SystemExit(f"--only-staged {store}: split(s) {lost} would keep fewer than two classes "
                         f"(kept {counts}); a one-class split cannot be scored")
    return kept, manifest_df[manifest_df["label_uid"].isin(set(kept["label_uid"]))], counts


def shadow(src, dest, types, only_staged=None, today=None):
    """Build (or extend) the shadow data directory ``dest`` of the round-3 dataset ``src`` for ``types``.

    Per type, ``<dest>/<type>/`` gets symlinks to ``<src>/<type>/correct`` and ``incorrect`` (the crops, never copied
    or written), byte-identical copies of ``SHADOW_COPY`` and of any ``OCTOBER_FILES`` present; ``<dest>`` gets a copy
    of ``deployment_priors.json`` and ``shadow_run.json`` (source path, the rawLabels snapshot ``<src>/../rawlabels``
    the backfill reads, date, git state, the sha256 of every source file copied). Every job then runs with
    ``DATA=<dest>``, so evaluation outputs, context images and run records land on scratch and ``src`` is only read.

    ``only_staged`` (a pano store): keep only the rows whose panorama is in it (``staged_subset``), October's outputs
    not copied (they cover rows the subset lacks), ``"subset": true`` recorded; ``report`` never decides on it.

    Refuses a ``dest`` inside ``src``, an existing shadow of another source or ``only_staged``, and a type already in
    the shadow whose source files have changed since (``SystemExit``). Rerunning with the same inputs is a no-op
    apart from the date.

    Example::

        python analysis/context_arms.py shadow --src $SRC --dest $WORK/data \\
            --types NoCurbRamp,Obstacle,SurfaceProblem
    """
    src, dest = os.path.abspath(src), os.path.abspath(dest)
    only_staged = os.path.abspath(only_staged) if only_staged else None
    today = today or date.today().isoformat()
    if _real_inside(dest, src):
        raise SystemExit(f"--dest {dest} is inside --src {src}: the shadow exists so that nothing writes there")
    if only_staged:
        import data_generation as dg
        dg.check_pano_store_marker(only_staged)
    old = read_shadow(dest)
    if old is not None and (old.get("src") != src or old.get("only_staged") != only_staged):
        raise SystemExit(f"{os.path.join(dest, SHADOW_FILE)} is a shadow of {old.get('src')} "
                         f"(only_staged {old.get('only_staged')}); use another --dest")
    entries = dict((old or {}).get("types") or {})
    for lt in types:
        s_dir, d_dir = os.path.join(src, lt), os.path.join(dest, lt)
        missing = [n for n in SHADOW_COPY if not os.path.isfile(os.path.join(s_dir, n))]
        if missing:
            raise SystemExit(f"{s_dir} has no {missing}")
        names = list(SHADOW_COPY) + ([] if only_staged else [n for n in OCTOBER_FILES
                                                             if os.path.isfile(os.path.join(s_dir, n))])
        sources = {n: sha256_file(os.path.join(s_dir, n)) for n in names}
        if lt in entries and entries[lt].get("sources") != sources:
            changed = sorted(set(sources.items()) ^ set((entries[lt].get("sources") or {}).items()))
            raise SystemExit(f"{lt}: the shadow records different source hashes than {s_dir} has now "
                             f"({[c[0] for c in changed]}); make a new shadow")
        os.makedirs(d_dir, exist_ok=True)
        for c in CROP_DIRS:
            target, link = os.path.join(s_dir, c), os.path.join(d_dir, c)
            if os.path.islink(link):
                if os.readlink(link) != target:
                    raise SystemExit(f"{link} points at {os.readlink(link)}, not {target}")
            elif os.path.exists(link):
                raise SystemExit(f"{link} exists and is not a symlink to {target}")
            elif os.path.isdir(target):
                os.symlink(target, link)
        entry = {"sources": sources, "subset": bool(only_staged)}
        if only_staged:
            split = pd.read_csv(os.path.join(s_dir, "split.csv"), dtype=str, keep_default_na=False)  # pano_id a str
            manifest = pd.read_csv(os.path.join(s_dir, "manifest.csv"), dtype=str, keep_default_na=False)
            kept, kept_manifest, counts = staged_subset(split, manifest, only_staged)
            for df, name in ((kept, "split.csv"), (kept_manifest, "manifest.csv")):
                df.to_csv(os.path.join(d_dir, name), index=False, lineterminator="\n")
            shutil.copyfile(os.path.join(s_dir, "generation_run.json"), os.path.join(d_dir, "generation_run.json"))
            entry.update(kept=counts, rows_source=int(len(split)), rows_kept=int(len(kept)))
            print(f"{lt}: kept {len(kept):,} of {len(split):,} rows whose panorama is in {only_staged}: {counts}")
        else:
            for n in names:
                shutil.copyfile(os.path.join(s_dir, n), os.path.join(d_dir, n))
            print(f"{lt}: shadowed {len(names)} files and the crop directories")
        entries[lt] = entry
    priors = os.path.join(src, "deployment_priors.json")
    record = {"src": src, "dest": dest, "rawlabels": os.path.join(os.path.dirname(src), "rawlabels"),
              "only_staged": only_staged, "subset": bool(only_staged),
              "created": (old or {}).get("created", today), "updated": today, **_git_state(),
              "deployment_priors_sha256": sha256_file(priors) if os.path.exists(priors) else None,
              "types": entries}
    if os.path.exists(priors):
        shutil.copyfile(priors, os.path.join(dest, "deployment_priors.json"))
    _write_json(record, os.path.join(dest, SHADOW_FILE))
    print(f"wrote {os.path.join(dest, SHADOW_FILE)}")
    return record


def environment_pin(name, path=None):
    """The ``name==X`` pin in ``environment.yml`` (the klone env's), or None."""
    import re
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "environment.yml")
    with open(path) as f:
        m = re.search(rf"^\s*-\s*{re.escape(name)}==([\w.+-]+)\s*$", f.read(), re.M)
    return m.group(1) if m else None


def installed_versions(names=("pandas", "transformers")):
    """``{name: version or None}`` read with ``importlib.metadata`` (no import: torch must not load on a login
    node)."""
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def missing_shares(split_df, columns=SIDE_SHARE_COLUMNS):
    """Share of rows with no usable value per column (absent column -> 1.0): blank cells, and for ``depth_m`` also a
    non-positive depth (the side arms treat both as "no depth").

    >>> missing_shares(pd.DataFrame({"severity": ["3", ""], "depth_m": ["0", "7.5"]}))
    {'severity': 0.5, 'image_capture_date': 1.0, 'depth_m': 0.5}
    """
    out = {}
    for c in columns:
        if c not in split_df.columns:
            out[c] = 1.0
            continue
        v = split_df[c].astype(str).str.strip()
        miss = v.isin(["", "nan", "NaN", "None"])
        if c == "depth_m":
            miss |= ~(pd.to_numeric(v, errors="coerce") > 0)
        out[c] = round(float(miss.mean()), 4) if len(v) else 1.0
    return out


def write_probe(directory, nbytes=WRITE_TEST_BYTES):
    """Write ``nbytes`` to a new file in ``directory`` (created if needed), fsync, delete. None when it worked, else
    the error text: a 0-byte ``touch`` can succeed where a real write fails."""
    path = os.path.join(directory, f".context-arms-write-probe-{os.getpid()}")
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\0" * nbytes)
            f.flush()
            os.fsync(f.fileno())
        os.remove(path)
        return None
    except OSError as e:
        try:
            os.remove(path)
        except OSError:
            pass
        return str(e)


def preflight(data, runs, types, arms, stage="", keep=None, pano_store=None, backfill=False, skip_backfill=False,
              pair_october=False, hf_home=None, versions=None, transformers_pin=None, write_test=True, out=print):
    """Everything a stage needs before it may take a GPU, checked on the login node in seconds; returns the list of
    problems (empty: go). Never imports torch or transformers.

    - the shadow (``shadow_run.json``; each type's source hashes unchanged; ``SHADOW_COPY`` present; 20 sampled crops
      resolve through the symlinks);
    - each type's ``split.csv`` header has ``BASE_COLUMNS`` and every arm's ``context_model.FRAME_COLUMNS`` (a missing
      column would raise ``KeyError`` after the job holds its GPU); the share of rows missing severity, imagery date
      and depth is printed (a side arm without depth is silently weaker);
    - ``pair_october``: October's ``metrics.json`` / ``predictions.csv`` (with val rows) and the seed-43/44 files;
    - ``backfill``: the pano store's marker and every ok manifest row's panorama (if any is missing, writes
      ``<data>/<type>/stage_files.txt`` in ``adjudicator.stage_files`` format with the count and a size estimate,
      and prints the restage command), what the backfill will write, and a ``<city>.csv`` in the shadow's rawLabels
      snapshot for every manifest city; ``skip_backfill``: ``context/index.csv`` covers every ok manifest row;
    - ``versions`` (default ``installed_versions()``): pandas 2.x, transformers at ``environment.yml``'s pin;
    - the backbone in the HF cache (``hf_home``; a path check, not a load);
    - a real 8 MiB write to ``runs`` and ``data``;
    - ``keep``: free space and files on its filesystem (and ``hyakstorage`` when installed), printed only.
    """
    import glob
    import context_model as cm
    problems = []
    rec = read_shadow(data)
    if rec is None:
        problems.append(f"{data} has no {SHADOW_FILE}: make a shadow (analysis/context_arms.py shadow)")
    for lt in types:
        d = os.path.join(data, lt)
        entry = ((rec or {}).get("types") or {}).get(lt)
        if rec is not None and entry is None:
            problems.append(f"{lt}: not in {SHADOW_FILE}")
        elif entry is not None:
            for name, digest in entry.get("sources", {}).items():
                p = os.path.join(rec["src"], lt, name)
                if not os.path.exists(p) or sha256_file(p) != digest:
                    problems.append(f"{lt}: source {p} changed or vanished since the shadow was made")
        absent = [n for n in SHADOW_COPY if not os.path.isfile(os.path.join(d, n))]
        if absent:
            problems.append(f"{lt}: {d} has no {absent}")
            continue
        split = pd.read_csv(os.path.join(d, "split.csv"), dtype=str, keep_default_na=False)
        need = set(BASE_COLUMNS)
        for arm in arms:
            if arm != "none":
                need |= set(cm.FRAME_COLUMNS[arm])
        lacking = sorted(need - set(split.columns))
        if lacking:
            problems.append(f"{lt}: split.csv lacks {lacking}, which arms {list(arms)} read")
        shares = missing_shares(split)
        out(f"{lt}: {len(split):,} rows; missing share " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
        if "crop_path" in split.columns and len(split):
            sample = split.sample(min(20, len(split)), random_state=0)["crop_path"]
            bad = [p for p in sample if not os.path.exists(p if os.path.isabs(p) else os.path.join(d, p))]
            if bad:
                problems.append(f"{lt}: {len(bad)} of {len(sample)} sampled crops do not resolve under {d} "
                                f"(e.g. {bad[:2]})")
        if pair_october:
            for n in OCTOBER_FILES:
                if not os.path.isfile(os.path.join(d, n)):
                    problems.append(f"{lt}: October's {n} is missing from the shadow (the drift row reads it)")
            p = os.path.join(d, "predictions.csv")
            if os.path.isfile(p) and not (pd.read_csv(p, usecols=["split"])["split"] == "val").any():
                problems.append(f"{lt}: October's predictions.csv has no val rows")
        manifest = pd.read_csv(os.path.join(d, "manifest.csv"), dtype=str, keep_default_na=False)
        ok = manifest[manifest["status"] == "ok"] if "status" in manifest.columns else manifest
        if skip_backfill:
            index = os.path.join(d, "context", "index.csv")
            have = set(pd.read_csv(index, dtype=str)["label_uid"]) if os.path.exists(index) else set()
            short = int((~ok["label_uid"].isin(have)).sum())
            if short:
                problems.append(f"{lt}: SKIP_BACKFILL but {index} lacks {short:,} of {len(ok):,} ok manifest rows")
        if backfill:
            n_files = 2 * len(ok)
            out(f"{lt}: the backfill writes {n_files:,} files, about {len(ok) * CONTEXT_KB_PER_LABEL / 1e6:.1f} GB, "
                f"under {os.path.join(d, 'context')}")
            if not pano_store or not os.path.isfile(os.path.join(pano_store, ".pano-store")):
                problems.append(f"{lt}: pano store {pano_store} has no .pano-store marker")
            else:
                import data_generation as dg
                pairs = sorted(set(zip(ok["city"], ok["pano_id"])))
                gone = [(c, p) for c, p in pairs if not os.path.exists(dg.pano_store_path(pano_store, c, p))]
                if gone:
                    listing = os.path.join(d, "stage_files.txt")
                    with open(listing, "w") as f:
                        f.writelines(f"{c}/{p[:2]}/{p}.jpg\n" for c, p in gone)
                    problems.append(
                        f"{lt}: {len(gone):,} of {len(pairs):,} panoramas are not in {pano_store} (about "
                        f"{len(gone) * PANO_MB / 1e3:.0f} GB at {PANO_MB} MB each); list in {listing}; restage with "
                        f"RUN={d} DEST={pano_store} sbatch analysis/ai_labels_stage.slurm (makelab2; Jon's call)")
            raw = (rec or {}).get("rawlabels")
            if not raw or not os.path.isdir(raw):
                problems.append(f"{lt}: the shadow's rawLabels snapshot {raw} is not a directory")
            else:
                cities = sorted(set(ok["city"]))
                no_csv = [c for c in cities if not os.path.isfile(os.path.join(raw, f"{c}.csv"))]
                if no_csv:
                    problems.append(f"{lt}: {raw} has no <city>.csv for {len(no_csv)} manifest cities")
    versions = installed_versions() if versions is None else versions
    pandas_v = versions.get("pandas")
    if not pandas_v or pandas_v.split(".")[0] != "2":
        problems.append(f"pandas {pandas_v} in the job env; the klone env must keep pandas 2.x")
    pin = transformers_pin if transformers_pin is not None else environment_pin("transformers")
    if versions.get("transformers") != pin:
        problems.append(f"transformers {versions.get('transformers')} in the job env, environment.yml pins {pin}")
    if not hf_home or not glob.glob(os.path.join(hf_home, BACKBONE_CACHE)):
        problems.append(f"facebook/dinov2-large is not in the HF cache {hf_home} (jobs run with HF_HUB_OFFLINE=1)")
    if write_test:
        for directory in (runs, data):
            err = write_probe(directory)
            if err:
                problems.append(f"a real {WRITE_TEST_BYTES >> 20} MiB write to {directory} failed: {err}")
    if keep:
        probe = keep if os.path.isdir(keep) else os.path.dirname(os.path.abspath(keep))
        try:
            st = os.statvfs(probe)
            out(f"KEEP {keep}: {st.f_bavail * st.f_frsize / 1e9:.0f} GB and {st.f_favail:,} files free "
                "(filesystem view; group quotas may be tighter)")
        except OSError as e:
            out(f"KEEP {keep}: statvfs failed ({e})")
        if shutil.which("hyakstorage"):
            import subprocess
            try:
                r = subprocess.run(["hyakstorage", probe], capture_output=True, text=True, timeout=60)
                out(r.stdout.strip())
            except (OSError, subprocess.SubprocessError) as e:
                out(f"hyakstorage failed ({e}); skipped")
        else:
            out("hyakstorage not installed here; quota not checked")
    for p in problems:
        out(f"PREFLIGHT FAILED: {p}")
    out(f"preflight {stage or ''}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return problems


def run_record(data_dir, label_type, tag):
    """What a run's durable record says: ``{"status": "ok" | "missing" | "incomplete", "seed", "split_sha256",
    "files_sha256", "path"}``. Read from ``<data>/<type>/runs/<tag>/`` (this chain's layout) or, failing that,
    ``training_run_<tag>.json`` beside the metrics (the backbone stream's layout)."""
    d = os.path.join(data_dir, label_type, "runs", tag)
    flat = os.path.join(data_dir, label_type, f"training_run_{tag}.json")
    if os.path.isdir(d):
        files = {n: sha256_file(os.path.join(d, n)) for n in RECORD_FILES if os.path.isfile(os.path.join(d, n))}
        key = {}
        if "run_key.json" in files:
            with open(os.path.join(d, "run_key.json")) as f:
                key = json.load(f)
        status = "ok" if {"training_run.json", "run_key.json"} <= set(files) else "incomplete"
        return {"status": status, "path": os.path.relpath(d, data_dir), "seed": key.get("seed"),
                "split_sha256": key.get("split_sha256"), "files_sha256": files}
    if os.path.isfile(flat):
        with open(flat) as f:
            run = json.load(f)
        return {"status": "ok", "path": os.path.relpath(flat, data_dir), "seed": run.get("seed"),
                "split_sha256": run.get("split_sha256"), "files_sha256": {os.path.basename(flat): sha256_file(flat)}}
    return {"status": "missing", "path": None, "seed": None, "split_sha256": None, "files_sha256": {}}


def check_run(data_dir, label_type, tag, seed=None):
    """The checks a finished training job must pass (the smoke stage's criterion): ``predictions_<tag>.csv`` holds
    exactly the split's ``EVAL_SPLITS`` rows, one per ``label_uid``, each in its own split; ``metrics_<tag>.json``
    and ``report_<tag>.txt`` exist; the run record is complete, with this ``seed`` (when given) and the hash of this
    ``split.csv``. Returns the list of problems (empty: pass)."""
    d = os.path.join(data_dir, label_type)
    problems = []
    split = pd.read_csv(os.path.join(d, "split.csv"), dtype=str, keep_default_na=False)
    pred_path = os.path.join(d, f"predictions_{tag}.csv")
    if not os.path.exists(pred_path):
        return [f"{label_type}/{tag}: no predictions_{tag}.csv"]
    pred = pd.read_csv(pred_path, dtype={"label_uid": str, "split": str})
    if not pred["label_uid"].is_unique:
        problems.append(f"{label_type}/{tag}: {int(pred['label_uid'].duplicated().sum())} duplicate label_uids")
    want = split[split["split"].isin(EVAL_SPLITS)]
    if set(pred["label_uid"]) != set(want["label_uid"]) or len(pred) != len(want):
        problems.append(f"{label_type}/{tag}: {len(pred):,} predictions for {len(want):,} val/test/held-out rows "
                        f"({len(set(want['label_uid']) - set(pred['label_uid'])):,} missing, "
                        f"{len(set(pred['label_uid']) - set(want['label_uid'])):,} extra)")
    m = pred.merge(want[["label_uid", "split"]], on="label_uid", how="inner", suffixes=("", "_split"),
                   validate="one_to_one") if pred["label_uid"].is_unique else None
    if m is not None and (m["split"] != m["split_split"]).any():
        problems.append(f"{label_type}/{tag}: {int((m['split'] != m['split_split']).sum())} rows in the wrong split")
    for n in (f"metrics_{tag}.json", f"report_{tag}.txt"):
        if not os.path.exists(os.path.join(d, n)):
            problems.append(f"{label_type}/{tag}: no {n}")
    rec = run_record(data_dir, label_type, tag)
    if rec["status"] != "ok":
        problems.append(f"{label_type}/{tag}: run record {rec['status']}")
    else:
        if seed is not None and rec["seed"] != seed:
            problems.append(f"{label_type}/{tag}: run record seed {rec['seed']}, expected {seed}")
        if rec["split_sha256"] != sha256_file(os.path.join(d, "split.csv")):
            problems.append(f"{label_type}/{tag}: run record split hash differs from this split.csv")
    return problems


def cmd_shadow(args):
    shadow(args.src, args.dest, _list(args.types), args.only_staged)
    return 0


def cmd_preflight(args):
    problems = preflight(args.data, args.runs, _list(args.types), _list(args.arms), args.stage, args.keep,
                         args.pano_store, args.backfill, args.skip_backfill, args.pair_october, args.hf_home,
                         write_test=not args.no_write_test)
    return 1 if problems else 0


def cmd_check_run(args):
    problems = []
    for tag in _list(args.tags):
        p = check_run(args.data, args.label_type, tag, seed_of_tag(tag) if args.seed_from_tag else None)
        print(f"{args.label_type}/{tag}: " + ("ok" if not p else "; ".join(p)))
        problems += p
    return 1 if problems else 0


def _list(text):
    return [t.strip() for t in str(text).replace(" ", ",").split(",") if t.strip()]


def seed_of_tag(tag, base_seed=42):
    """The training seed a tag stands for: ``<variant>-s<seed>``, else ``base_seed``.

    >>> seed_of_tag("view_heatmap-r448-s43"), seed_of_tag("baseline-rerun")
    (43, 42)
    """
    m = cl.SEED_RUN_RE.match(f"metrics_{tag}.json")
    return int(m["seed"]) if m else base_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shadow", help="a shadow data directory of the round-3 dataset on scratch (see shadow())")
    s.add_argument("--src", required=True, help="the round-3 data directory (read only)")
    s.add_argument("--dest", required=True, help="the shadow to create or extend (not inside --src)")
    s.add_argument("--types", required=True, help="comma list of label types")
    s.add_argument("--only-staged", default=None,
                   help="a pano store: keep only rows whose panorama is in it (smoke tests; never decidable)")
    s.set_defaults(func=cmd_shadow)
    f = sub.add_parser("preflight", help="check a stage before it takes a GPU (login node; see preflight())")
    f.add_argument("--data", required=True, help="the shadow data directory the jobs use")
    f.add_argument("--runs", required=True, help="the checkpoint root the jobs write")
    f.add_argument("--types", required=True)
    f.add_argument("--arms", required=True, help="comma list of the stage's arms (none = the baseline rerun)")
    f.add_argument("--stage", default="")
    f.add_argument("--keep", default=None)
    f.add_argument("--pano-store", default=None)
    f.add_argument("--backfill", action="store_true", help="the stage submits a backfill")
    f.add_argument("--skip-backfill", action="store_true", help="image arms run on an existing context index")
    f.add_argument("--pair-october", action="store_true", help="require October's baseline files (drift row)")
    f.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    f.add_argument("--no-write-test", action="store_true", help="skip the 8 MiB write probes (tests only)")
    f.set_defaults(func=cmd_preflight)
    c = sub.add_parser("check-run", help="did a finished training job write everything (smoke criterion)")
    c.add_argument("--data", required=True)
    c.add_argument("--label-type", required=True)
    c.add_argument("--tags", required=True, help="comma list of evaluate.py tags")
    c.add_argument("--seed-from-tag", action="store_true", help="also check the record's seed (-s<seed>, else 42)")
    c.set_defaults(func=cmd_check_run)
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
