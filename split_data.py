"""Assign one label type's crops to train/val/test by panorama, as a manifest (``split.csv``); no files are touched.

Input is ``<out>/<label_type>/manifest.csv`` written by ``data_generation.py``. Output is
``<out>/<label_type>/split.csv``: the manifest rows with ``status == ok``, sorted by ``label_uid``, plus ``hard``
(1 if the label is in ``--hard-examples``, else 0; inserted before ``crop_path``) and a ``split`` column in
{``train``, ``val``, ``test``, ``test_heldout``}. ``hard`` is joined here rather than at generation time so a new
hard-example list never needs the crops regenerated. ``fine_tune_classifier.py`` trains on it and
``evaluate.py`` / ``calibration/calibrate.py`` evaluate on it. Image files are never deleted, moved or copied, so
the step is idempotent: re-running with the same seed writes a byte-identical file
(https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/10). ``split.csv`` is written to a temporary file
and moved into place. Once the manifest has loaded and a new split is being computed, any error (a refused split or
an exception) removes the ``split.csv`` of an earlier run and the temporary file, so a stale split is never trained on
by mistake; errors before that point (bad arguments, a missing manifest or hard-example file) leave it alone.

Method (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3):

1. **Held-out cities.** Every row of a ``--holdout-cities`` city goes to ``test_heldout`` (Duan et al. ASSETS'22:
   held-out-city numbers are the ones that predict deployment). A group (step 2/3) that also carries labels of
   another city (overlapping deployments) goes to ``test_heldout`` whole, so the no-shared-panorama guarantee holds;
   the number of other-city rows pulled in this way is printed.
2. **Groups.** Rows are grouped by ``pano_id``: labels cluster on panoramas (2.8 per panorama across
   all types in Teaneck; within one type's selected set CurbRamp 1.66, SurfaceProblem 1.19), and two crops
   of one corner must not straddle the split (Weld et al. ASSETS'19 split panoramas for the same reason). A row
   without a ``pano_id`` is its own group.
3. **Merge nearby groups** (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/27). Groups whose labels
   lie within ``--merge-radius-m`` (default 3 m; haversine distance ``<=`` the radius) of each other become one group
   (connected components over "same panorama or within the radius", so chains merge transitively), so one physical
   curb ramp or obstacle, labelled from two adjacent panoramas or by two users, is on one side of the split only.
   Without it, on the 19 training cities, 3.4% (CurbRamp), 8.2% (NoCurbRamp) and 2.3% (Obstacle) of test labels had
   a same-type train label within 2 m. The manifest holds one label type, so "same type" is implicit.

   Coordinates come from the manifest's ``latitude``/``longitude`` columns if it has them, otherwise from the cached
   rawLabels exports ``<DIR>/<city>.csv`` (``--rawlabels DIR``, default ``<out>/rawlabels`` when it exists), joined
   on ``label_uid`` (city + label_id; label_id alone is a per-city serial). No coordinate source, or a missing
   city export, is an error. A label without coordinates stays in its panorama group (it is never merged by
   distance) and is left out of the proximity report; their count is printed, and more than
   ``--max-missing-coords-frac`` (default 1%) of ok labels without coordinates is an error.

   Transitive merging can chain a dense stretch into one big group, which the greedy filler cannot balance. With
   merging on, and counting only the train/val/test rows (not ``test_heldout``), the run warns when the largest
   group holds more than 1% of them, and stops when it holds more than ``--max-group-share`` (default 5%) of them
   AND more than 50 rows (the floor keeps small and smoke-test manifests from tripping it). A split missing its
   target share by more than 3 points is warned about at any radius. ``--merge-radius-m 0`` turns merging off and
   reproduces the panorama-only split byte for byte, with no guard.
4. **Approximate stratification.** Each group gets a class-ratio bin, ``round(4 * share_incorrect)`` (0 = all
   correct ... 4 = all incorrect). Bins are processed in order; within a bin the groups (sorted by key, the
   smallest original group key, normally a ``pano_id``) are shuffled with a seeded generator, then filled greedily:
   each group goes to the split with the largest deficit against ``TARGET`` (70/15/15) of that bin's ROWS. Every bin
   is thus split about 70/15/15, so the splits end up with similar correct:incorrect shares as well as the target
   row shares.

The printed summary adds, whatever the radius, the share of val/test labels with a train label within 2, 5 and
10 m (and within the merge radius, which is 0 by construction), plus how many groups were merged and how large the
merged groups got. The proximity report needs coordinates; with ``--merge-radius-m 0`` and none available it is
skipped with a note.

Usage::

    python split_data.py --label_type CurbRamp                              # ./CurbRamp/manifest.csv, ./rawlabels/
    python split_data.py --label_type CurbRamp --rawlabels ~/rawlabels      # coordinates from elsewhere
    python split_data.py --label_type CurbRamp --merge-radius-m 0           # panorama grouping only
    python split_data.py --label_type CurbRamp --holdout-cities teaneck-nj,chicago-il --seed 7
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
EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius; at a few metres the spherical error is far below GPS noise
PROXIMITY_RADII_M = (2.0, 5.0, 10.0)
WARN_GROUP_SHARE = 0.01   # warn when the largest group holds more than this share of the rows
WARN_SHARE_MISS = 0.03    # warn when train/val/test misses its TARGET share by more than this
MIN_GUARD_ROWS = 50       # --max-group-share only fires when the largest group also exceeds this many rows


def load_coordinates(manifest, rawlabels_dir=None):
    """``latitude``/``longitude`` (floats, NaN where unknown) for every ``status == ok`` row, indexed by ``label_uid``.

    Taken from the manifest's own ``latitude``/``longitude`` columns when it has them; otherwise joined from the
    cached rawLabels exports ``<rawlabels_dir>/<city>.csv`` (columns ``label_id``, ``latitude``, ``longitude``, ...)
    on the composite ``label_uid`` = ``<city>:<label_id>``, never on ``label_id`` alone (it is a per-city serial).
    Each export is cut to the label_ids the manifest needs (after stripping whitespace) before duplicates are
    checked, so an unrelated bad row in an export does not matter.

    Raises ``ValueError`` when no source is given, a city's export is missing, or an export has a needed
    ``label_id`` twice with different coordinates. Labels without coordinates come back as NaN; the caller decides.

    >>> m = pd.DataFrame({"label_uid": ["a:1"], "city": ["a"], "label_id": ["1"], "status": ["ok"],
    ...                   "latitude": ["47.6"], "longitude": ["-122.3"]})
    >>> load_coordinates(m).loc["a:1"].tolist()
    [47.6, -122.3]
    """
    ok = manifest[manifest["status"].astype(str) == "ok"]
    if {"latitude", "longitude"} <= set(ok.columns):
        return pd.DataFrame({"latitude": pd.to_numeric(ok["latitude"], errors="coerce").to_numpy(),
                             "longitude": pd.to_numeric(ok["longitude"], errors="coerce").to_numpy()},
                            index=ok["label_uid"].to_numpy())
    if not rawlabels_dir:
        raise ValueError("the manifest has no latitude/longitude columns and no rawLabels directory was found")
    parts = []
    for city in sorted(ok["city"].astype(str).unique()):
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        if not os.path.isfile(path):
            raise ValueError(f"no rawLabels export for city '{city}' at {path}")
        raw = pd.read_csv(path, usecols=["label_id", "latitude", "longitude"], dtype={"label_id": str},
                          low_memory=False)
        # label_id is matched as a stripped string: an export spelling it "8.0" or "007" matches nothing, and those
        # labels then count as missing coordinates, which --max-missing-coords-frac turns into a loud failure.
        raw["label_id"] = raw["label_id"].str.strip()
        needed = set(ok.loc[ok["city"].astype(str) == city, "label_id"].astype(str).str.strip())
        raw = raw[raw["label_id"].isin(needed)].drop_duplicates()
        if not raw["label_id"].is_unique:
            dup = raw.loc[raw["label_id"].duplicated(), "label_id"].iloc[0]
            raise ValueError(f"{path} has label_id {city}:{dup} with conflicting coordinates")
        raw.index = city + ":" + raw["label_id"]
        parts.append(raw[["latitude", "longitude"]])
    raw = pd.concat(parts) if parts else pd.DataFrame(columns=["latitude", "longitude"], dtype=float)
    return raw.reindex(ok["label_uid"].to_numpy()).astype(float)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between points given in degrees (NumPy-vectorised, symmetric).

    This is THE distance of the split: merging keeps pairs with ``haversine_m <= radius`` and the proximity report
    counts the same way, which is why the report's ``within_<merge radius>m`` is exactly 0.

    >>> round(float(haversine_m(47.0, -122.0, 47.0 + 3 / 111195, -122.0)), 3)
    3.0
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def pairs_within(lat_q, lon_q, lat_t, lon_t, radius_m):
    """Index pairs ``(i, j)`` with ``haversine_m(query i, target j) <= radius_m``, as two int arrays.

    A BallTree (haversine) on the targets finds candidates with a slightly padded radius; the exact ``<=`` test is
    then done with ``haversine_m``, so the boundary does not depend on the tree's internal arithmetic.
    """
    from sklearn.neighbors import BallTree
    lat_q, lon_q, lat_t, lon_t = (np.asarray(v, dtype=float) for v in (lat_q, lon_q, lat_t, lon_t))
    if not len(lat_q) or not len(lat_t):
        return np.array([], dtype=int), np.array([], dtype=int)
    tree = BallTree(np.radians(np.column_stack([lat_t, lon_t])), metric="haversine")
    pad = radius_m * (1 + 1e-6) + 1e-6
    cand = tree.query_radius(np.radians(np.column_stack([lat_q, lon_q])), r=pad / EARTH_RADIUS_M)
    i = np.repeat(np.arange(len(lat_q)), [len(c) for c in cand])
    j = np.concatenate(cand).astype(int)
    keep = haversine_m(lat_q[i], lon_q[i], lat_t[j], lon_t[j]) <= radius_m
    return i[keep], j[keep]


def merge_nearby_groups(group, lat, lon, radius_m):
    """Merge groups that have labels within ``radius_m`` metres of each other (``haversine_m <= radius_m``).

    ``group`` is a Series of group keys (one per label; a ``pano_id`` or ``uid:<label_uid>``), ``lat``/``lon`` the
    labels' coordinates in degrees. A label with non-finite coordinates keeps its group and is never merged by
    distance. Returns ``(merged, stats)``: ``merged`` is a Series like ``group`` whose value is the smallest original
    key in each connected component of "same group, or within the radius" (transitively: a chain of labels 2 m apart
    is one group). ``stats`` is a dict with

    - ``groups``: original groups (panoramas, plus one per row without a ``pano_id``);
    - ``merged_groups``: groups after merging; ``groups_merged``: original groups absorbed into another;
    - ``multi_panorama_groups``: merged groups made of more than one original group;
    - ``max_original_groups``: original groups in the largest (by that count) merged group;
    - ``max_labels``: labels in the largest merged group.

    The result does not depend on row order: components come from ``scipy.sparse.csgraph.connected_components`` and
    are named by their smallest key.

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
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        has = np.flatnonzero(np.isfinite(lat) & np.isfinite(lon))
        i, j = pairs_within(lat[has], lon[has], lat[has], lon[has], radius_m)
        rows.append(code[has[i]])
        cols.append(code[has[j]])
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, comp = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    # name each component by its smallest key (keys are sorted, so that is the smallest index in the component)
    first = pd.Series(np.arange(n)).groupby(comp).transform("min").to_numpy()
    merged = pd.Series(keys[first][code], index=group.index)
    sizes = pd.Series(comp).value_counts()
    stats = {"groups": n, "merged_groups": len(sizes), "groups_merged": n - len(sizes),
             "multi_panorama_groups": int((sizes > 1).sum()),
             "max_original_groups": int(sizes.max()) if n else 0,
             "max_labels": int(merged.value_counts().max()) if n else 0}
    return merged, stats


def proximity_report(df, coords, radii_m=PROXIMITY_RADII_M):
    """Share of each non-train split's labels with a ``train`` label within each radius (metres), as a DataFrame.

    ``df`` has ``label_uid`` and ``split``; ``coords`` is indexed by ``label_uid`` (see ``load_coordinates``). Labels
    without finite coordinates are left out on both sides (the caller reports how many). One row per split present
    (``val``, ``test``, ``test_heldout``), columns ``labels`` (with coordinates) and ``within_<r>m`` (a share; "within"
    is ``haversine_m <= r``, the merge's own test).

    >>> d = pd.DataFrame({"label_uid": ["a", "b"], "split": ["train", "test"]})
    >>> c = pd.DataFrame({"latitude": [0.0, 4 / 111195], "longitude": [0.0, 0.0]}, index=["a", "b"])
    >>> proximity_report(d, c, (2, 5)).to_dict("records")
    [{'split': 'test', 'labels': 1, 'within_2m': 0.0, 'within_5m': 1.0}]
    """
    xy = coords.loc[df["label_uid"]]
    lat, lon = xy["latitude"].to_numpy(dtype=float), xy["longitude"].to_numpy(dtype=float)
    has = np.isfinite(lat) & np.isfinite(lon)
    split = df["split"].to_numpy()
    train = has & (split == "train")
    rows = []
    for s in SPLITS[1:]:
        mask = has & (split == s)
        if not (split == s).any():
            continue
        row = {"split": s, "labels": int(mask.sum())}
        for r in radii_m:
            i, _ = pairs_within(lat[mask], lon[mask], lat[train], lon[train], r)
            row[f"within_{r:g}m"] = round(len(np.unique(i)) / mask.sum(), 4) if mask.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def assign_splits(manifest, seed=42, holdout_cities=(), merge_radius_m=0.0, coords=None, return_stats=False):
    """Return the ``status == ok`` rows of ``manifest`` (sorted by ``label_uid``) with a ``split`` column.

    With ``merge_radius_m > 0``, ``coords`` (``latitude``/``longitude`` indexed by ``label_uid``, see
    ``load_coordinates``) is required and groups within the radius are merged first. With ``return_stats``, returns
    ``(df, stats)``: ``merge_nearby_groups``'s dict plus ``split_rows`` / ``max_split_group`` (rows split
    train/val/test, and the largest group among them), ``heldout_pulled_in`` (rows of non-held-out cities sent to
    ``test_heldout`` because their group touches a held-out city) and ``heldout_pulled_in_by_merge`` (the part of
    those that a shared panorama alone would not have pulled in).

    Pure function of its inputs: same manifest and seed, same assignment. See the module docstring for the method.
    """
    df = manifest[manifest["status"].astype(str) == "ok"].copy()
    df = df.sort_values("label_uid", kind="mergesort").reset_index(drop=True)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid is not unique in the manifest")
    pano_group = df["pano_id"].astype(str).where(df["pano_id"].astype(str).str.len() > 0, "uid:" + df["label_uid"])
    if merge_radius_m > 0:
        if coords is None:
            raise ValueError("merge_radius_m > 0 needs coords")
        xy = coords.loc[df["label_uid"]]
        group, stats = merge_nearby_groups(pano_group, xy["latitude"].to_numpy(), xy["longitude"].to_numpy(),
                                           merge_radius_m)
    else:
        group, stats = merge_nearby_groups(pano_group, None, None, 0)
    df["split"] = ""

    holdout = set(holdout_cities)
    in_holdout_city = df["city"].isin(holdout)
    if holdout:
        df.loc[group.isin(set(group[in_holdout_city])), "split"] = HELDOUT
    pulled = (df["split"] == HELDOUT) & ~in_holdout_city
    by_pano = pano_group.isin(set(pano_group[in_holdout_city])) & ~in_holdout_city
    stats["heldout_pulled_in"] = int(pulled.sum())
    stats["heldout_pulled_in_by_merge"] = int((pulled & ~by_pano).sum())

    rest = df["split"] == ""
    g = pd.DataFrame({"group": group[rest], "incorrect": (df.loc[rest, "label_class"] == "incorrect").astype(int)})
    groups = g.groupby("group", sort=True).agg(n=("incorrect", "size"), k=("incorrect", "sum"))
    groups["bin"] = np.rint(N_BINS * groups["k"] / groups["n"]).astype(int)
    stats["split_rows"] = int(groups["n"].sum())  # rows split train/val/test (test_heldout excluded)
    stats["max_split_group"] = int(groups["n"].max()) if len(groups) else 0  # largest group among them

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


def target_shares(df):
    """Achieved share of each TARGET split among the train/val/test rows (test_heldout excluded), as a dict."""
    split = df.loc[df["split"].isin([s for s, _ in TARGET]), "split"]
    return {s: (float((split == s).mean()) if len(split) else 0.0) for s, _ in TARGET}


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
                   help="directory of cached rawLabels exports <city>.csv, for label coordinates; default "
                        "<out>/rawlabels when it exists (data_generation.slurm's cache). Not needed if the manifest "
                        "has latitude/longitude")
    p.add_argument("--max-group-share", type=float, default=0.05,
                   help="stop when the largest merged group holds more than this share of the rows (0 = no limit)")
    p.add_argument("--max-missing-coords-frac", type=float, default=0.01,
                   help="with merging on, stop when more than this share of ok labels have no coordinates")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = os.path.join(args.out, args.label_type)
    out_path = os.path.join(base, "split.csv")
    tmp_path = out_path + ".tmp"

    def fail(message, computing=True):
        """Print the error and return 2. Once a new split is being computed (``computing``), also remove the
        split.csv of an earlier run, which no longer matches the inputs."""
        print(f"Error: {message}", file=sys.stderr)
        if computing and os.path.exists(out_path):
            os.remove(out_path)
            print(f"Removed the stale {out_path} from an earlier run.", file=sys.stderr)
        return 2

    # Before the manifest has loaded, errors leave an existing split.csv alone.
    manifest_path = args.manifest or os.path.join(base, "manifest.csv")
    if not os.path.isfile(manifest_path):
        return fail(f"manifest '{manifest_path}' not found. Run data_generation.py first; split_data.py splits a "
                    f"manifest by panorama and will not split bare image folders.", computing=False)
    if args.merge_radius_m < 0:
        return fail("--merge-radius-m must be >= 0.", computing=False)
    if args.hard_examples and not os.path.isfile(args.hard_examples):
        return fail(f"--hard-examples '{args.hard_examples}' not found.", computing=False)
    # Strings throughout, so every manifest cell is written back exactly as read.
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    hard = set(pd.read_csv(args.hard_examples, dtype=str)["label_uid"]) if args.hard_examples else set()

    try:
        return _split(args, base, manifest, hard, out_path, tmp_path, fail)
    except BaseException:
        for path in (out_path, tmp_path):
            if os.path.exists(path):
                os.remove(path)
        print(f"Error: split_data.py stopped; removed any split.csv under {base}/ so no stale split is left.",
              file=sys.stderr)
        raise


def _split(args, base, manifest, hard, out_path, tmp_path, fail):
    """The part of ``main`` that computes and writes a new split; ``main`` cleans up if it raises."""
    holdout = [c.strip() for c in args.holdout_cities.split(",") if c.strip()]
    unknown = sorted(set(holdout) - set(manifest["city"]))
    if unknown:
        print(f"Warning: --holdout-cities {unknown} have no rows in the manifest.", file=sys.stderr)

    rawlabels = args.rawlabels
    if rawlabels is None and os.path.isdir(os.path.join(args.out, "rawlabels")):
        rawlabels = os.path.join(args.out, "rawlabels")
    coords = None
    try:
        coords = load_coordinates(manifest, rawlabels)
    except ValueError as e:
        if args.merge_radius_m > 0:
            return fail(f"--merge-radius-m {args.merge_radius_m:g} needs label coordinates, but {e}. Pass "
                        f"--rawlabels DIR (the cached rawLabels exports, e.g. rawlabels/) or --merge-radius-m 0.")
        print(f"Note: no proximity report ({e}).", file=sys.stderr)
    n_missing = 0
    if coords is not None:
        no_xy = ~np.isfinite(coords.to_numpy()).all(axis=1)
        n_missing = int(no_xy.sum())
        frac = n_missing / max(len(coords), 1)
        if args.merge_radius_m > 0 and frac > args.max_missing_coords_frac:
            return fail(f"{n_missing} of {len(coords)} ok labels ({frac:.1%}) have no coordinates, more than "
                        f"--max-missing-coords-frac {args.max_missing_coords_frac:g}; e.g. "
                        f"{list(coords.index[no_xy][:3])}. Raise the limit or use --merge-radius-m 0.")

    df, stats = assign_splits(manifest, seed=args.seed, holdout_cities=holdout, merge_radius_m=args.merge_radius_m,
                              coords=coords, return_stats=True)
    # Guards on the rows actually split train/val/test; only with merging on, so radius 0 behaves as before merging.
    big, n_split = stats["max_split_group"], stats["split_rows"]
    group_share = big / max(n_split, 1)
    if args.merge_radius_m > 0 and args.max_group_share > 0 \
            and big > max(args.max_group_share * n_split, MIN_GUARD_ROWS):
        return fail(f"the largest merged group holds {big} of {n_split} train/val/test rows ({group_share:.1%}), "
                    f"more than --max-group-share {args.max_group_share:g} (and more than {MIN_GUARD_ROWS} rows): "
                    f"merging at {args.merge_radius_m:g} m chained a dense area into one group, and the split cannot "
                    f"be balanced around it. Use a smaller --merge-radius-m, or a larger --max-group-share "
                    f"(0 = no limit) to accept it.")
    shares = target_shares(df)
    off = n_split and any(abs(shares[s] - t) > WARN_SHARE_MISS for s, t in TARGET)
    if (args.merge_radius_m > 0 and group_share > WARN_GROUP_SHARE) or off:
        print(f"Warning: the largest group holds {big} rows ({group_share:.1%} of {n_split} train/val/test rows); "
              f"achieved train/val/test shares " + " / ".join(f"{shares[s]:.3f}" for s, _ in TARGET)
              + " against " + " / ".join(f"{t:.2f}" for _, t in TARGET) + ".", file=sys.stderr)

    df = add_hard(df, hard)
    check_no_shared_panoramas(df)
    missing = [p for p in df["crop_path"] if not os.path.exists(os.path.join(base, p))]
    if missing:
        print(f"Warning: {len(missing)} ok rows have no crop file under {base}/, e.g. {missing[:3]}", file=sys.stderr)

    df.to_csv(tmp_path, index=False, lineterminator="\n")
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}: {len(df)} rows of {len(manifest)} in the manifest (seed {args.seed}, "
          f"held-out cities {holdout or 'none'}, {int((df['hard'] == '1').sum())} hard)")
    print(summary(df).to_string(index=False))
    print(f"Merge radius {args.merge_radius_m:g} m: {stats['groups']} original groups (panoramas) -> "
          f"{stats['merged_groups']} groups ({stats['groups_merged']} merged away; {stats['multi_panorama_groups']} "
          f"groups span more than one panorama; largest {stats['max_original_groups']} panoramas, "
          f"{stats['max_labels']} labels)")
    if holdout:
        print(f"Rows of other cities pulled into {HELDOUT}: {stats['heldout_pulled_in']} "
              f"({stats['heldout_pulled_in_by_merge']} by a merged group, the rest by a shared panorama)")
    if coords is not None:
        if n_missing:
            print(f"{n_missing} ok labels have no coordinates: kept in their panorama group, never merged by "
                  f"distance, and left out of the proximity report.")
        radii = sorted(set(PROXIMITY_RADII_M) | ({args.merge_radius_m} if args.merge_radius_m > 0 else set()))
        print("Share of labels with a train label within r metres:")
        print(proximity_report(df, coords, radii).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
