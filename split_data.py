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
2. **Groups.** Rows are grouped by ``pano_id``: labels cluster on panoramas (2.8 per panorama across
   all types in Teaneck; within one type's selected set CurbRamp 1.66, SurfaceProblem 1.19), and two crops
   of one corner must not straddle the split (Weld et al. ASSETS'19 split panoramas for the same reason).
3. **Merge nearby groups** (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/27). Panorama groups
   whose labels lie within ``--merge-radius-m`` (default 3 m, haversine) of each other become one group (connected
   components over "same panorama or within the radius"), so one physical curb ramp or obstacle, labelled from two
   adjacent panoramas or by two users, is on one side of the split only. Without it, on the 19 training cities,
   3.4% (CurbRamp), 8.2% (NoCurbRamp) and 2.3% (Obstacle) of test labels had a same-type train label within 2 m.
   The manifest holds one label type, so "same type" is implicit. Coordinates come from the manifest's
   ``latitude``/``longitude`` columns if it has them, otherwise from the cached rawLabels exports
   (``--rawlabels DIR``, ``<DIR>/<city>.csv``), joined on ``label_uid`` (city + label_id; label_id alone is per-city).
   A missing export or a label without coordinates is an error, never a silent skip. ``--merge-radius-m 0`` turns
   merging off and reproduces the panorama-only split byte for byte. Steps 1 and 4 run on the merged groups, so a
   merged group that touches a held-out city goes to ``test_heldout`` whole.
4. **Approximate stratification.** Each group gets a class-ratio bin, ``round(4 * share_incorrect)`` (0 = all
   correct ... 4 = all incorrect). Bins are processed in order; within a bin the groups (sorted by key, the
   smallest ``pano_id`` in the group) are
   shuffled with a seeded generator, then filled greedily: each group goes to the split with the largest deficit
   against ``TARGET`` (70/15/15) of that bin's ROWS. Every bin is thus split about 70/15/15, so the splits end up with
   similar correct:incorrect shares as well as the target row shares.

The printed summary adds, whatever the radius, the share of val/test labels with a train label within 2, 5 and
10 m (and within the merge radius, which is 0 by construction), plus how many groups were merged and how large the
merged groups got. The proximity report needs coordinates; with ``--merge-radius-m 0`` and none available it is
skipped with a note.

Usage::

    python split_data.py --label_type CurbRamp --rawlabels rawlabels        # ./CurbRamp/manifest.csv
    python split_data.py --label_type CurbRamp --merge-radius-m 0           # panorama grouping only
    python split_data.py --label_type CurbRamp --rawlabels rawlabels --holdout-cities teaneck-nj,taipei --seed 7
    python split_data.py --label_type CurbRamp --rawlabels rawlabels --hard-examples analysis/hard_examples_2026-10-17.csv
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
EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius; at a few metres the spherical error is far below GPS noise
PROXIMITY_RADII_M = (2.0, 5.0, 10.0)


def load_coordinates(manifest, rawlabels_dir=None):
    """``latitude``/``longitude`` (floats) for every ``status == ok`` row of ``manifest``, indexed by ``label_uid``.

    Taken from the manifest's own ``latitude``/``longitude`` columns when it has them; otherwise joined from the
    cached rawLabels exports ``<rawlabels_dir>/<city>.csv`` (columns ``label_id``, ``latitude``, ``longitude``, ...)
    on the composite ``label_uid`` = ``<city>:<label_id>``, never on ``label_id`` alone (it is a per-city serial).

    Raises ``ValueError`` when no source is given, a city's export is missing, an export has a duplicated
    ``label_id`` with different coordinates, or any ok row ends up without finite coordinates.

    >>> m = pd.DataFrame({"label_uid": ["a:1"], "city": ["a"], "label_id": ["1"], "status": ["ok"],
    ...                   "latitude": ["47.6"], "longitude": ["-122.3"]})
    >>> load_coordinates(m).loc["a:1"].tolist()
    [47.6, -122.3]
    """
    ok = manifest[manifest["status"].astype(str) == "ok"]
    if {"latitude", "longitude"} <= set(ok.columns):
        coords = pd.DataFrame({"latitude": pd.to_numeric(ok["latitude"], errors="coerce").to_numpy(),
                               "longitude": pd.to_numeric(ok["longitude"], errors="coerce").to_numpy()},
                              index=ok["label_uid"].to_numpy())
        source = "the manifest's latitude/longitude columns"
    else:
        if not rawlabels_dir:
            raise ValueError("the manifest has no latitude/longitude columns and no --rawlabels directory was given")
        parts = []
        for city in sorted(ok["city"].astype(str).unique()):
            path = os.path.join(rawlabels_dir, f"{city}.csv")
            if not os.path.isfile(path):
                raise ValueError(f"no rawLabels export for city '{city}' at {path}")
            raw = pd.read_csv(path, usecols=["label_id", "latitude", "longitude"], dtype={"label_id": str},
                              low_memory=False).drop_duplicates()
            if not raw["label_id"].is_unique:
                dup = raw.loc[raw["label_id"].duplicated(), "label_id"].iloc[0]
                raise ValueError(f"{path} has label_id {city}:{dup} with conflicting coordinates")
            raw.index = city + ":" + raw["label_id"].str.strip()
            parts.append(raw[["latitude", "longitude"]])
        raw = pd.concat(parts) if parts else pd.DataFrame(columns=["latitude", "longitude"], dtype=float)
        coords = raw.reindex(ok["label_uid"].to_numpy()).astype(float)
        source = f"rawLabels exports in {rawlabels_dir}"
    bad = ~np.isfinite(coords.to_numpy()).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} ok labels have no coordinates in {source}, "
                         f"e.g. {list(coords.index[bad][:3])}")
    return coords


def _radians(lat, lon):
    return np.radians(np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)]))


def merge_nearby_groups(group, lat, lon, radius_m):
    """Merge groups that have labels within ``radius_m`` metres of each other (haversine).

    ``group`` is a Series of group keys (one per label), ``lat``/``lon`` the labels' coordinates in degrees. Returns
    ``(merged, stats)``: ``merged`` is a Series like ``group`` whose value is the smallest original key in each
    connected component of "same group, or within the radius" (transitively: a chain of labels 2 m apart is one
    group), and ``stats`` a dict with ``groups``, ``merged_groups``, ``groups_merged`` (original groups absorbed),
    ``max_panoramas`` and ``max_labels`` per merged group. The result does not depend on row order: components are
    computed by ``scipy.sparse.csgraph.connected_components`` and named by their smallest key.

    >>> g = pd.Series(["p1", "p2", "p3"])
    >>> m, st = merge_nearby_groups(g, [47.0, 47.0 + 2 / 111195, 47.1], [-122.0] * 3, 3.0)
    >>> m.tolist(), st["groups_merged"]
    (['p1', 'p1', 'p3'], 1)
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keys = np.array(sorted(group.unique()), dtype=object)
    code = pd.Index(keys).get_indexer(group)
    n = len(keys)
    rows, cols = [code], [code]  # every label joins its own group
    if radius_m > 0 and len(group):
        from sklearn.neighbors import BallTree
        pts = _radians(lat, lon)
        neighbours = BallTree(pts, metric="haversine").query_radius(pts, r=radius_m / EARTH_RADIUS_M)
        src = np.repeat(np.arange(len(pts)), [len(nb) for nb in neighbours])
        dst = np.concatenate(neighbours) if len(neighbours) else np.array([], dtype=int)
        rows.append(code[src])
        cols.append(code[dst])
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, comp = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    # name each component by its smallest key (keys are sorted, so that is the first index seen per component)
    first = pd.Series(np.arange(n)).groupby(comp).transform("min").to_numpy()
    merged = pd.Series(keys[first][code], index=group.index)
    sizes = pd.Series(comp).value_counts()
    labels_per = merged.value_counts()
    stats = {"groups": n, "merged_groups": len(sizes), "groups_merged": n - len(sizes),
             "multi_panorama_groups": int((sizes > 1).sum()),
             "max_panoramas": int(sizes.max()) if n else 0, "max_labels": int(labels_per.max()) if n else 0}
    return merged, stats


def proximity_report(df, coords, radii_m=PROXIMITY_RADII_M):
    """Share of each non-train split's labels with a ``train`` label within each radius (metres), as a DataFrame.

    ``df`` has ``label_uid`` and ``split``; ``coords`` is indexed by ``label_uid`` (see ``load_coordinates``). One
    row per split present (``val``, ``test``, ``test_heldout``), columns ``labels`` and ``within_<r>m`` (a share).

    >>> d = pd.DataFrame({"label_uid": ["a", "b"], "split": ["train", "test"]})
    >>> c = pd.DataFrame({"latitude": [0.0, 4 / 111195], "longitude": [0.0, 0.0]}, index=["a", "b"])
    >>> proximity_report(d, c, (2, 5)).to_dict("records")
    [{'split': 'test', 'labels': 1, 'within_2m': 0.0, 'within_5m': 1.0}]
    """
    from sklearn.neighbors import BallTree
    xy = coords.loc[df["label_uid"]]
    is_train = (df["split"] == "train").to_numpy()
    tree = BallTree(_radians(xy["latitude"][is_train], xy["longitude"][is_train]), metric="haversine") \
        if is_train.any() else None
    rows = []
    for s in SPLITS[1:]:
        mask = (df["split"] == s).to_numpy()
        if not mask.any():
            continue
        pts = _radians(xy["latitude"][mask], xy["longitude"][mask])
        row = {"split": s, "labels": int(mask.sum())}
        for r in radii_m:
            hit = tree.query_radius(pts, r=r / EARTH_RADIUS_M, count_only=True) > 0 if tree is not None \
                else np.zeros(len(pts), dtype=bool)
            row[f"within_{r:g}m"] = round(float(hit.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_splits(manifest, seed=42, holdout_cities=(), merge_radius_m=0.0, coords=None, return_stats=False):
    """Return the ``status == ok`` rows of ``manifest`` (sorted by ``label_uid``) with a ``split`` column.

    With ``merge_radius_m > 0``, ``coords`` (``latitude``/``longitude`` indexed by ``label_uid``, see
    ``load_coordinates``) is required and panorama groups within the radius are merged first. With
    ``return_stats``, returns ``(df, stats)`` where ``stats`` is ``merge_nearby_groups``'s dict.

    Pure function of its inputs: same manifest and seed, same assignment. See the module docstring for the method.
    """
    df = manifest[manifest["status"].astype(str) == "ok"].copy()
    df = df.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid is not unique in the manifest")
    # A row without a pano_id is its own group.
    group = df["pano_id"].astype(str).where(df["pano_id"].astype(str).str.len() > 0, "uid:" + df["label_uid"])
    if merge_radius_m > 0:
        if coords is None:
            raise ValueError("merge_radius_m > 0 needs coords")
        xy = coords.loc[df["label_uid"]]
        group, stats = merge_nearby_groups(group, xy["latitude"].to_numpy(), xy["longitude"].to_numpy(),
                                           merge_radius_m)
    else:
        group, stats = merge_nearby_groups(group, None, None, 0)
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
    return (df, stats) if return_stats else df


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
    p.add_argument("--merge-radius-m", type=float, default=3.0,
                   help="merge panorama groups with labels within this many metres (0 = panorama grouping only)")
    p.add_argument("--rawlabels", default=None,
                   help="directory of cached rawLabels exports <city>.csv (data_generation.slurm's rawlabels/), "
                        "for label coordinates; needed unless the manifest has latitude/longitude")
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

    if args.merge_radius_m < 0:
        print("Error: --merge-radius-m must be >= 0.", file=sys.stderr)
        return 2
    coords = None
    try:
        coords = load_coordinates(manifest, args.rawlabels)
    except ValueError as e:
        if args.merge_radius_m > 0:
            print(f"Error: --merge-radius-m {args.merge_radius_m:g} needs label coordinates, but {e}. Pass "
                  f"--rawlabels DIR (the cached rawLabels exports, e.g. rawlabels/) or --merge-radius-m 0.",
                  file=sys.stderr)
            return 2
        print(f"Note: no proximity report ({e}).", file=sys.stderr)

    df, stats = assign_splits(manifest, seed=args.seed, holdout_cities=holdout, merge_radius_m=args.merge_radius_m,
                              coords=coords, return_stats=True)
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
    print(f"Merge radius {args.merge_radius_m:g} m: {stats['groups']} panorama groups -> {stats['merged_groups']} "
          f"groups ({stats['groups_merged']} merged away; {stats['multi_panorama_groups']} groups span more than one "
          f"panorama; largest {stats['max_panoramas']} panoramas / {stats['max_labels']} labels)")
    if coords is not None:
        radii = sorted(set(PROXIMITY_RADII_M) | ({args.merge_radius_m} if args.merge_radius_m > 0 else set()))
        print("Share of labels with a train label within r metres:")
        print(proximity_report(df, coords, radii).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
