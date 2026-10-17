"""Assign one label type's crops to train/val/test by panorama, as a manifest (``split.csv``); no files are touched.

Input is ``<out>/<label_type>/manifest.csv`` written by ``data_generation.py``. Output is
``<out>/<label_type>/split.csv``: the manifest rows with ``status == ok``, sorted by ``label_uid``, plus ``hard``
(1 if the label is in ``--hard-examples``, else 0; inserted before ``crop_path``) and a ``split`` column in
{``train``, ``val``, ``test``, ``test_heldout``}. ``hard`` is joined here rather than at generation time so a new
hard-example list never needs the crops regenerated. ``fine_tune_classifier.py`` trains on it and
``evaluate.py`` / ``calibration/calibrate.py`` evaluate on it. Image files are never deleted, moved or copied, so
the step is idempotent: re-running with the same seed writes a byte-identical file
(https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/10).

Method (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3):

1. **Held-out cities.** Every row of a ``--holdout-cities`` city goes to ``test_heldout`` (Duan et al. ASSETS'22:
   held-out-city numbers are the ones that predict deployment). A panorama that also carries labels of another city
   (overlapping deployments) goes to ``test_heldout`` whole, so the no-shared-panorama guarantee holds.
2. **Groups.** The remaining rows are grouped by ``pano_id``: labels cluster on panoramas (2.8 per panorama across
   all types in Teaneck; within one type's selected set CurbRamp 1.66, SurfaceProblem 1.19), and two crops
   of one corner must not straddle the split (Weld et al. ASSETS'19 split panoramas for the same reason).
3. **Approximate stratification.** Each group gets a class-ratio bin, ``round(4 * share_incorrect)`` (0 = all
   correct ... 4 = all incorrect). Bins are processed in order; within a bin the groups (sorted by ``pano_id``) are
   shuffled with a seeded generator, then filled greedily: each group goes to the split with the largest deficit
   against ``TARGET`` (70/15/15) of that bin's ROWS. Every bin is thus split about 70/15/15, so the splits end up with
   similar correct:incorrect shares as well as the target row shares.

Usage::

    python split_data.py --label_type CurbRamp                              # ./CurbRamp/manifest.csv
    python split_data.py --label_type CurbRamp --holdout-cities teaneck-nj,taipei --seed 7
    python split_data.py --label_type CurbRamp --hard-examples analysis/hard_examples_2026-10-17.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

TARGET = (("train", 0.70), ("val", 0.15), ("test", 0.15))
HELDOUT = "test_heldout"
SPLITS = tuple(s for s, _ in TARGET) + (HELDOUT,)
N_BINS = 4  # class-ratio bins are round(N_BINS * share_incorrect), i.e. 0..4


def assign_splits(manifest, seed=42, holdout_cities=()):
    """Return the ``status == ok`` rows of ``manifest`` (sorted by ``label_uid``) with a ``split`` column.

    Pure function of its inputs: same manifest and seed, same assignment. See the module docstring for the method.
    """
    df = manifest[manifest["status"].astype(str) == "ok"].copy()
    df = df.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid is not unique in the manifest")
    # A row without a pano_id is its own group.
    group = df["pano_id"].astype(str).where(df["pano_id"].astype(str).str.len() > 0, "uid:" + df["label_uid"])
    df["split"] = ""

    holdout = set(holdout_cities)
    if holdout:
        held_groups = set(group[df["city"].isin(holdout)])
        df.loc[group.isin(held_groups), "split"] = HELDOUT

    rest = df["split"] == ""
    g = pd.DataFrame({"group": group[rest], "incorrect": (df.loc[rest, "label_class"] == "incorrect").astype(int)})
    groups = g.groupby("group", sort=True).agg(n=("incorrect", "size"), k=("incorrect", "sum"))
    groups["bin"] = np.rint(N_BINS * groups["k"] / groups["n"]).astype(int)

    rng = np.random.default_rng(seed)
    assignment = {}
    for _, members in groups.groupby("bin", sort=True):
        order = members.index.to_numpy()[rng.permutation(len(members))]
        total = members["n"].sum()
        filled = {s: 0 for s, _ in TARGET}
        for gid in order:
            # largest deficit against the target row count; ties go to the earlier split in TARGET
            best = max(TARGET, key=lambda t: t[1] * total - filled[t[0]])[0]
            assignment[gid] = best
            filled[best] += members.at[gid, "n"]
    df.loc[rest, "split"] = group[rest].map(assignment)
    return df


def add_hard(df, hard_uids):
    """Set ``hard`` (``"1"``/``"0"``) from a set of ``label_uid``s, replacing any ``hard`` column the manifest had,
    and order the columns: ``hard`` just before ``crop_path`` (the place evaluate.py's split.csv always had it),
    ``split`` last."""
    df = df.drop(columns=[c for c in ("hard",) if c in df])
    df["hard"] = df["label_uid"].isin(hard_uids).astype(int).astype(str)
    cols = [c for c in df.columns if c not in ("hard", "split")]
    at = cols.index("crop_path") if "crop_path" in cols else len(cols)
    return df[cols[:at] + ["hard"] + cols[at:] + ["split"]]


def check_no_shared_panoramas(df):
    """Assert that no ``pano_id`` appears in two splits."""
    with_pano = df[df["pano_id"].astype(str).str.len() > 0]
    per_pano = with_pano.groupby(with_pano["pano_id"].astype(str))["split"].nunique()
    shared = per_pano[per_pano > 1]
    assert shared.empty, f"{len(shared)} panoramas appear in more than one split, e.g. {list(shared.index[:3])}"


def summary(df):
    """Per-split row counts by class, panorama counts and incorrect share, as a DataFrame."""
    rows = []
    for s in SPLITS:
        part = df[df["split"] == s]
        if part.empty:
            continue
        n_inc = int((part["label_class"] == "incorrect").sum())
        rows.append({"split": s, "rows": len(part), "correct": len(part) - n_inc, "incorrect": n_inc,
                     "share_incorrect": round(n_inc / len(part), 3), "panoramas": part["pano_id"].nunique(),
                     "share_rows": round(len(part) / len(df), 3)})
    return pd.DataFrame(rows)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label_type", required=True)
    p.add_argument("--out", default=".", help="output root; reads and writes under <out>/<label_type>/")
    p.add_argument("--manifest", default=None, help="default <out>/<label_type>/manifest.csv")
    p.add_argument("--seed", type=int, default=42, help="seed for the group shuffle")
    p.add_argument("--holdout-cities", default="", help="comma list of city ids sent whole to test_heldout")
    p.add_argument("--hard-examples", default=None,
                   help="CSV with a label_uid column (e.g. analysis/hard_examples_2026-10-17.csv); those rows get hard = 1")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = os.path.join(args.out, args.label_type)
    manifest_path = args.manifest or os.path.join(base, "manifest.csv")
    if not os.path.isfile(manifest_path):
        print(f"Error: manifest '{manifest_path}' not found. Run data_generation.py first; split_data.py splits a "
              f"manifest by panorama and will not split bare image folders.", file=sys.stderr)
        return 2

    # Strings throughout, so every manifest cell is written back exactly as read.
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    holdout = [c.strip() for c in args.holdout_cities.split(",") if c.strip()]
    unknown = sorted(set(holdout) - set(manifest["city"]))
    if unknown:
        print(f"Warning: --holdout-cities {unknown} have no rows in the manifest.", file=sys.stderr)

    df = assign_splits(manifest, seed=args.seed, holdout_cities=holdout)
    hard = set(pd.read_csv(args.hard_examples, dtype=str)["label_uid"]) if args.hard_examples else set()
    df = add_hard(df, hard)
    check_no_shared_panoramas(df)
    missing = [p for p in df["crop_path"] if not os.path.exists(os.path.join(base, p))]
    if missing:
        print(f"Warning: {len(missing)} ok rows have no crop file under {base}/, e.g. {missing[:3]}", file=sys.stderr)

    out_path = os.path.join(base, "split.csv")
    df.to_csv(out_path, index=False, lineterminator="\n")
    print(f"Wrote {out_path}: {len(df)} rows of {len(manifest)} in the manifest (seed {args.seed}, "
          f"held-out cities {holdout or 'none'}, {int((df['hard'] == '1').sum())} hard)")
    print(summary(df).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
