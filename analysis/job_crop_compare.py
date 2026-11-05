"""Job-window crops against depth-sized crops on the same labels (issue #53): counts, restrict, compare.

The question is whether a CurbRamp model trained on the windows SidewalkWebpage's nightly crop job cuts
(``data_generation.py --job-window-crops``, ``job_crop.py``) does as well as the round-3 model trained on depth-sized
views, overall and per imagery provider. The protocol is PR #51's: the two models are scored on identical rows and
compared with a paired panorama bootstrap. Three subcommands, run in this order:

``counts``  Before any GPU time: of the round-3 split's rows, how many WOULD get a job crop, per ``pano_source`` x
            split x class, and whether each provider's held-out AUC would be measurable. A row gets a job crop when
            it was cut from the pano store (``pano_path`` set), its recorded ``pano_width``/``pano_height`` (from the
            rawLabels exports, ``--rawlabels-cache``) equal the stored file's (``pano_native_w``/``h`` in
            ``split.csv``) or are missing, and ``0 <= pano_y < height`` (job_crop.cut_job_crop's rules, which are the
            backend's). With ``--store`` the file must also still exist. Writes ``counts.csv`` and ``counts.md``.
``restrict`` After ``data_generation.py --job-window-crops``: the round-3 split restricted to the rows that got a
            job crop, every split kept and each row's split inherited (``split_data.py`` is not re-run, so nothing
            drifts), with the job manifest's crop fields. Writes the new ``split.csv`` (what
            ``fine_tune_classifier.py --data-dir`` reads), ``restricted_uids.txt`` and its sha256.
``compare`` After training and ``evaluate.py``: job minus depth AUC on the restricted rows, pooled and per split
            (``test``, ``test_heldout``), per provider (only above the gates), within-city AUC for both, and the
            severity x tag slices (``analysis/slices.paired_delta_table``, the standing rule). Writes
            ``headline.csv``, ``per_provider.csv``, ``slices.csv`` and ``tables.md``.

No output names a deployment: every table is per provider or pooled (``max_city_share`` in the slice table is a
number), so private deployments never appear. The depth model is NOT retrained on the restricted rows (#51 did not
either): it is the round-3 model, scored on the subset.

Usage (paths are examples)::

    python analysis/job_crop_compare.py counts --split <round3>/CurbRamp/split.csv \\
        --rawlabels-cache <round3 rawlabels> --out-dir docs/job-window-crops-2026-11
    python analysis/job_crop_compare.py restrict --split <round3>/CurbRamp/split.csv \\
        --manifest <job out>/CurbRamp/manifest.csv --out <job out>/CurbRamp/split.csv
    python analysis/job_crop_compare.py compare --depth-pred <round3>/CurbRamp/predictions.csv \\
        --job-pred <job out>/CurbRamp/predictions.csv --split <job out>/CurbRamp/split.csv \\
        --out-dir docs/job-window-crops-2026-11
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import retrain_report as rr  # noqa: E402
from analysis import slices as sl  # noqa: E402

EVAL_SPLITS = ("test", "test_heldout")
SPLITS = ("train", "val", "test", "test_heldout")
MIN_N = sl.MIN_N        # 30 rows, as the slice tables
MIN_CLASS = 10          # each class, for a per-provider AUC to be called measurable
N_BOOT = 1000
RAW_COLS = ["label_id", "label_type", "pano_x", "pano_y", "pano_width", "pano_height"]


def _provider(s):
    return s.fillna("").astype(str).str.strip().str.lower().replace("", "(unknown)")


def _given(s):
    return pd.to_numeric(s, errors="coerce").notna()


# ---------------------------------------------------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------------------------------------------------

def recorded_geometry(cache_dir, cities, label_type):
    """``pano_x, pano_y, pano_width, pano_height`` per ``label_uid`` from the rawLabels exports
    ``<cache_dir>/<city>.csv`` (the frame each label was placed on, as data_generation.py read it)."""
    frames = []
    for city in sorted(cities):
        path = os.path.join(cache_dir, f"{city}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: no rawLabels export for {city}")
        d = pd.read_csv(path, usecols=RAW_COLS, low_memory=False)
        d = d[d.label_type == label_type].drop(columns="label_type")
        d["label_uid"] = city + ":" + d.label_id.astype(str)
        frames.append(d.drop(columns="label_id"))
    out = pd.concat(frames, ignore_index=True)
    assert out.label_uid.is_unique, "label_uid is not unique in the rawLabels exports"
    return out


def job_crop_status(df, store_exists=None):
    """Per row, what the job-crop path would do: ``ok``, ``no_store`` (no ``pano_path``, or with ``store_exists``
    the file is gone), ``dims_mismatch`` or ``out_of_frame``, checked in that order (the backend's: the pano is
    opened, its dims checked, then each label's y). ``df`` needs ``pano_path, pano_native_w, pano_native_h, pano_y,
    pano_width, pano_height``.

    >>> d = pd.DataFrame({"pano_path": ["/a", None, "/c", "/d", "/e"], "pano_native_w": [100, 100, 100, 100, 100],
    ...     "pano_native_h": [50] * 5, "pano_y": [10, 10, 10, 50, 10], "pano_width": [100, 100, 200, 100, None],
    ...     "pano_height": [50, 50, 100, 50, None]})
    >>> job_crop_status(d).tolist()
    ['ok', 'no_store', 'dims_mismatch', 'out_of_frame', 'ok']
    """
    nw, nh = pd.to_numeric(df.pano_native_w, errors="coerce"), pd.to_numeric(df.pano_native_h, errors="coerce")
    rw, rh = pd.to_numeric(df.pano_width, errors="coerce"), pd.to_numeric(df.pano_height, errors="coerce")
    y = pd.to_numeric(df.pano_y, errors="coerce")
    has_path = df.pano_path.notna() & (df.pano_path.astype(str).str.strip() != "")
    if store_exists is not None:
        has_path &= df.pano_path.map(lambda p: isinstance(p, str) and store_exists(p)).astype(bool)
    mismatch = (rw.notna() & (rw != nw)) | (rh.notna() & (rh != nh))
    out_of_frame = ~((y >= 0) & (y < nh))
    status = np.select([~has_path, mismatch, out_of_frame], ["no_store", "dims_mismatch", "out_of_frame"], "ok")
    return pd.Series(status, index=df.index)


def counts_table(split, min_n=MIN_N, min_class=MIN_CLASS):
    """``split`` (with ``job_status`` from :func:`job_crop_status`) -> one row per provider x split x class with the
    row counts by status, plus one row per provider and split with both classes (``label_class`` = ``all``) carrying
    the verdict: ``AUC measurable`` on ``test_heldout`` (and ``test``) when the job-crop rows number at least
    ``min_n`` and each class at least ``min_class``, else ``too few``."""
    d = split.assign(pano_source=_provider(split.pano_source))
    statuses = ["ok", "no_store", "dims_mismatch", "out_of_frame"]
    rows = []
    for (src, sp), g in d.groupby(["pano_source", "split"]):
        for cls, gg in list(g.groupby("label_class")) + [("all", g)]:
            row = {"pano_source": src, "split": sp, "label_class": cls, "n": len(gg)}
            row.update({f"n_{s}": int((gg.job_status == s).sum()) for s in statuses})
            row["n_unverified_dims"] = int((gg.job_status.eq("ok") & ~gg.dims_recorded).sum())
            if cls == "all":
                ok = gg[gg.job_status == "ok"]
                n_inc = int((ok.label_class == "incorrect").sum())
                n_cor = int((ok.label_class == "correct").sum())
                row["verdict"] = ("AUC measurable" if len(ok) >= min_n and min(n_inc, n_cor) >= min_class
                                  else "too few") if sp in EVAL_SPLITS else ""
            else:
                row["verdict"] = ""
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SPLITS)}
    out["_o"] = out.split.map(lambda s: order.get(s, 99))
    out["_c"] = out.label_class.map({"correct": 0, "incorrect": 1, "all": 2})
    return out.sort_values(["pano_source", "_o", "_c"]).drop(columns=["_o", "_c"]).reset_index(drop=True)


def counts(args):
    split = pd.read_csv(args.split, dtype={"label_uid": str}, low_memory=False)
    assert split.label_uid.is_unique, f"{args.split}: label_uid is not unique"
    label_type = split.label_type.iloc[0]
    geo = recorded_geometry(args.rawlabels_cache, split.city.unique(), label_type)
    d = split.merge(geo, on="label_uid", how="left", validate="one_to_one")
    missing = int(d.pano_y.isna().sum())
    if missing:
        raise ValueError(f"{missing} split rows have no rawLabels row in {args.rawlabels_cache} (another snapshot?)")
    d["dims_recorded"] = _given(d.pano_width) & _given(d.pano_height)
    exists = os.path.exists if args.store else None
    d["job_status"] = job_crop_status(d, exists)
    table = counts_table(d)
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "counts.csv"), index=False)
    md = counts_markdown(table, label_type, os.path.basename(args.split))
    with open(os.path.join(args.out_dir, "counts.md"), "w") as f:
        f.write(md)
    print(md)
    return table


def counts_markdown(table, label_type, source):
    ev = table[(table.label_class == "all")]
    lines = [f"# {label_type}: rows that would get a job-window crop", "",
             f"From `{source}` (round 3) and the rawLabels exports it was built from. `ok` = would be cut; "
             "`dims_mismatch` = recorded frame differs from the stored file (the backend skips these); "
             "`out_of_frame` = `pano_y` outside the image; `no_store` = not cut from the pano store.", "",
             "| provider | split | n | ok | dims_mismatch | out_of_frame | no_store | ok, dims unrecorded | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ev.itertuples():
        lines.append(f"| {r.pano_source} | {r.split} | {r.n} | {r.n_ok} | {r.n_dims_mismatch} | {r.n_out_of_frame} "
                     f"| {r.n_no_store} | {r.n_unverified_dims} | {r.verdict} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# restrict
# ---------------------------------------------------------------------------------------------------------------------

def restrict_split(split, manifest):
    """The rows of ``split`` (round 3) whose label has an ``ok`` row in ``manifest`` (the job-crop run), with the
    manifest's columns (its ``crop_path``, ``crop_half_px``, ...), the round-3 ``split`` and ``hard`` columns, and
    the label's class checked to agree. Every split is kept."""
    ok = manifest[manifest.status.astype(str) == "ok"]
    assert ok.label_uid.is_unique, "the job manifest repeats a label_uid"
    keep = [c for c in ("split", "hard") if c in split.columns]
    out = ok.merge(split[["label_uid", "label_class"] + keep].rename(columns={"label_class": "_round3_class"}),
                   on="label_uid", how="inner", validate="one_to_one")
    bad = out[out.label_class != out._round3_class]
    if len(bad):
        raise ValueError(f"{len(bad)} labels changed class since round 3 (e.g. {bad.label_uid.iloc[0]}); the job run "
                         "must use round 3's selection")
    return out.drop(columns="_round3_class").sort_values("label_uid").reset_index(drop=True)


def restrict(args):
    split = pd.read_csv(args.split, dtype={"label_uid": str}, low_memory=False)
    manifest = pd.read_csv(args.manifest, dtype={"label_uid": str}, low_memory=False)
    out = restrict_split(split, manifest)
    if os.path.abspath(args.out) == os.path.abspath(args.split):
        raise SystemExit("--out must not overwrite the round-3 split")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    uids_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "restricted_uids.txt")
    text = "\n".join(out.label_uid) + "\n"
    with open(uids_path, "w") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    with open(uids_path + ".sha256", "w") as f:
        f.write(f"{digest}  restricted_uids.txt\n")
    summary = restrict_summary(split, manifest, out)
    summary["restricted_uids_sha256"] = digest
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "restrict_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"restricted split -> {args.out}: " + ", ".join(
        f"{s} {summary['kept_by_split'].get(s, 0)}/{summary['round3_by_split'].get(s, 0)}" for s in SPLITS
        if s in summary["round3_by_split"]))
    print(f"round-3 rows not cut: {summary['dropped_by_reason'] or 'none'}; ok job rows outside round 3's split: "
          f"{summary['job_rows_not_in_round3']}")
    print(f"{uids_path}: sha256 {digest}")
    return out, digest


def restrict_summary(split, manifest, out):
    """Why each round-3 row did or did not make the restricted split: kept per split, and the dropped rows by the job
    manifest's status (``not_selected`` when the job run did not select the label at all, which happens when the
    selection code or votes moved since round 3, e.g. #72 now keeps Mapillary labels and so shifts the downsample)."""
    status = manifest.drop_duplicates("label_uid", keep="last").set_index("label_uid").status.astype(str)
    dropped = split.loc[~split.label_uid.isin(set(out.label_uid)), "label_uid"]
    reasons = dropped.map(lambda u: status.get(u, "not_selected")).value_counts().to_dict()
    ok = set(status.index[status == "ok"])
    return {"round3_by_split": {k: int(v) for k, v in split.split.value_counts().items()},
            "kept_by_split": {k: int(v) for k, v in out.split.value_counts().items()},
            "dropped_by_reason": {k: int(v) for k, v in reasons.items()},
            "job_rows_not_in_round3": len(ok - set(split.label_uid))}


# ---------------------------------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------------------------------

def paired_frame(job_pred, depth_pred, split):
    """One row per restricted eval label: ``p_job``, ``p_depth`` (the incorrect probability), ``y``, and the split's
    ``split, city, pano_id, pano_source, severity, tags, true``. Raises if either model misses a restricted row."""
    ev = split[split.split.isin(EVAL_SPLITS)]
    cols = ["label_uid", "split", "city", "pano_id", "pano_source", "label_class", "severity", "tags"]
    ev = ev[[c for c in cols if c in ev.columns]]
    out = ev
    for name, pred in (("job", job_pred), ("depth", depth_pred)):
        p = pred[["label_uid", "true", "p_incorrect"]].rename(columns={"p_incorrect": f"p_{name}",
                                                                       "true": f"true_{name}"})
        out = out.merge(p, on="label_uid", how="left", validate="one_to_one")
        missing = int(out[f"p_{name}"].isna().sum())
        if missing:
            raise ValueError(f"the {name} predictions have no score for {missing} restricted eval rows; score both "
                             "on the restricted split")
        if (out[f"true_{name}"] != out.label_class).any():
            raise ValueError(f"the {name} predictions disagree with the split on some labels' class")
    out["y"] = (out.label_class == "incorrect").astype(int)
    out["true"] = out.label_class
    out["pano_source"] = _provider(out.pano_source)
    return out.drop(columns=["true_job", "true_depth"])


def delta_row(g, n_boot, seed, min_n=MIN_N, min_class=MIN_CLASS):
    """AUC of both models on ``g``, job minus depth with a paired panorama-bootstrap 95% interval, and within-city
    AUC of both. Blank AUCs when ``g`` is under ``min_n`` rows or a class under ``min_class``."""
    n_inc = int(g.y.sum())
    row = {"n": len(g), "n_incorrect": n_inc, "n_panos": int(g.pano_id.nunique()), "auc_job": np.nan,
           "auc_depth": np.nan, "delta": np.nan, "delta_lo": np.nan, "delta_hi": np.nan,
           "within_city_auc_job": np.nan, "within_city_auc_depth": np.nan, "within_city_n_cities": 0}
    if len(g) < min_n or min(n_inc, len(g) - n_inc) < min_class:
        return row
    a, b = roc_auc_score(g.y, g.p_job), roc_auc_score(g.y, g.p_depth)
    row.update(auc_job=a, auc_depth=b, delta=a - b)
    if n_boot:
        boot = rr.cluster_boot(g.y.values, g[["p_job", "p_depth"]].values, g.pano_id.values, n_boot, seed)
        row["delta_lo"], row["delta_hi"] = rr.ci(boot[:, 0] - boot[:, 1])
    row["within_city_auc_job"], row["within_city_n_cities"] = rr.within_city_auc(g.y, g.p_job, g.city)
    row["within_city_auc_depth"], _ = rr.within_city_auc(g.y, g.p_depth, g.city)
    return row


def compare_tables(paired, n_boot=N_BOOT, seed=0):
    """``(headline, per_provider, slices)``: pooled and per eval split; per provider x split (rows only where the
    AUC is measurable); severity and tag slices per split from ``slices.paired_delta_table``."""
    head = []
    for name, g in [("pooled (test + test_heldout)", paired)] + [(s, paired[paired.split == s]) for s in EVAL_SPLITS]:
        if len(g):
            head.append({"split": name, **delta_row(g, n_boot, sl.slice_seed(seed, name))})
    prov = []
    for (sp, src), g in paired.groupby(["split", "pano_source"]):
        r = delta_row(g, n_boot, sl.slice_seed(seed, f"{sp}:{src}"))
        if not np.isnan(r["auc_job"]):
            prov.append({"split": sp, "pano_source": src, **r})
    sl_rows = []
    for sp in EVAL_SPLITS:
        g = paired[paired.split == sp]
        if not len(g):
            continue
        a = g.rename(columns={"p_job": "p_incorrect"})[["label_uid", "true", "p_incorrect", "pano_id", "city",
                                                        "severity", "tags"]]
        b = g.rename(columns={"p_depth": "p_incorrect"})[["label_uid", "true", "p_incorrect"]]
        for by in sl.BY:
            t = sl.paired_delta_table(a, b, by, n_boot=n_boot, seed=seed)
            sl_rows.append(t.assign(split=sp, by=by).rename(columns={"auc_a": "auc_job", "auc_b": "auc_depth"}))
    slices = pd.concat(sl_rows, ignore_index=True) if sl_rows else pd.DataFrame()
    if len(slices):
        slices = slices[["split", "by"] + [c for c in slices.columns if c not in ("split", "by")]]
    return pd.DataFrame(head), pd.DataFrame(prov), slices


def compare(args):
    split = pd.read_csv(args.split, dtype={"label_uid": str}, low_memory=False)
    paired = paired_frame(pd.read_csv(args.job_pred, dtype={"label_uid": str}),
                          pd.read_csv(args.depth_pred, dtype={"label_uid": str}), split)
    head, prov, slices = compare_tables(paired, args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    head.to_csv(os.path.join(args.out_dir, "headline.csv"), index=False)
    prov.to_csv(os.path.join(args.out_dir, "per_provider.csv"), index=False)
    slices.to_csv(os.path.join(args.out_dir, "slices.csv"), index=False)
    md = ["# Job-window crops minus depth-sized crops (AUC, same rows)", "",
          f"Paired panorama bootstrap, {args.n_boot} resamples, seed {args.seed}. Positive delta = the job-crop "
          "model is better.", "", "## Headline", "", rr.md(head, list(head.columns)), "",
          "## Per provider (only where both classes have at least 10 rows and n >= 30)", "",
          rr.md(prov, list(prov.columns)) if len(prov) else "(none measurable)", ""]
    for sp in EVAL_SPLITS:
        for by in sl.BY:
            t = slices[(slices.split == sp) & (slices.by == by)] if len(slices) else slices
            if len(t):
                md += [f"## {sp}: by {by}", "", rr.md(t.drop(columns=["split", "by"]),
                                                       [c for c in t.columns if c not in ("split", "by")]), ""]
    with open(os.path.join(args.out_dir, "tables.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump({"job_pred": args.job_pred, "depth_pred": args.depth_pred, "split": args.split,
                   "n_boot": args.n_boot, "seed": args.seed}, f, indent=2)
    print("\n".join(md))
    return head, prov, slices


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counts", help="rows that would get a job crop, per provider x split x class")
    c.add_argument("--split", required=True, help="round-3 <LabelType>/split.csv")
    c.add_argument("--rawlabels-cache", required=True, help="the rawLabels exports round 3 was built from")
    c.add_argument("--store", action="store_true", help="also require each pano_path to exist (needs the store)")
    c.add_argument("--out-dir", required=True)
    r = sub.add_parser("restrict", help="the round-3 split restricted to rows with a job crop")
    r.add_argument("--split", required=True, help="round-3 <LabelType>/split.csv")
    r.add_argument("--manifest", required=True, help="the --job-window-crops run's manifest.csv")
    r.add_argument("--out", required=True, help="new split.csv, beside the job run's manifest")
    m = sub.add_parser("compare", help="paired job-minus-depth AUC on the restricted rows")
    m.add_argument("--job-pred", required=True)
    m.add_argument("--depth-pred", required=True, help="round 3's predictions.csv (a superset is fine)")
    m.add_argument("--split", required=True, help="the restricted split.csv")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--n-boot", type=int, default=N_BOOT)
    m.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"counts": counts, "restrict": restrict, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
