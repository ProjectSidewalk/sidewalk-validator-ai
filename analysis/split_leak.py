"""The panorama leak measured in isolation (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3).

The 2025-08 models were trained on a file-level split, so labels of one panorama (and two labels of one physical
feature, seen from adjacent panoramas) could sit in train and test. Round 3 splits by panorama with a 3 m merge
(``split_data.py``). This module measures what the file-level split was worth, holding everything else fixed.
The design, the pre-registered rule and its limits are in ``docs/split-leak-2026-11.md``.

**The leaky split is a coupled permutation of the production split** (``make-split``): the grouped split is
recomputed with ``split_data.assign_splits`` from the same manifest, coordinates, seed 42, held-out deployments and
3 m merge, and must equal the production ``split.csv`` byte for byte. Then the split labels of the rows of
multi-member groups (a shared panorama, or a 3 m chain) are permuted within each class across the whole type, with
the generator ``default_rng(SeedSequence([seed, SCATTER_TAG]))``. ``test_heldout`` and every single-label group keep
their split, and per (class, split) row counts are exactly equal. The file adds ``split_mode`` (``label_level``),
``split_grouped`` and ``split_group`` before ``split``; ``fine_tune_classifier.py`` records the mode and refuses the
production output directory, and ``publish_to_hf.py`` refuses the model.

**Exposure** (``check``): a val or test row is *exposed* when a direct companion (same ``pano_id``, or within 3 m of
the row itself by ``split_data.haversine_m``) is in the leaky train split; ``k`` is its number of direct companions.
A row whose only train companions are further along a merged chain is *chain-only*. Because the permutation
randomises which multi-member rows get a train companion, exposed against unexposed rows inside one model is a
causal estimate of the leak.

Subcommands (paths follow one layout: ``<data>/<T>/{manifest,split}.csv`` is round 3, ``<runs>/<T>/splits/`` holds
the leaky splits and their exposure, ``<runs>/<T>/<arm>-s<seed>/`` one training run)::

    python analysis/split_leak.py make-split --data DATA --runs RUNS --types CurbRamp ... --seeds 42 43 44 \
        --rawlabels DATA/rawlabels                       # CPU; stops unless the grouped split reproduces
    python analysis/split_leak.py check --data DATA --runs RUNS --types ... --seeds ... --rawlabels ... --out REPORT
    python analysis/split_leak.py bound --data DATA --runs RUNS --types ... --rawlabels ... --out REPORT
    python analysis/split_leak.py predict --data DATA --label_type CurbRamp --split FILE --run-dir RUN \
        --out RUN/predictions.csv                        # GPU
    python analysis/split_leak.py report --data DATA --runs RUNS --types ... --seeds 42 43 44 --g-seeds 42 43 44 \
        --g-source stored --out REPORT

Nothing written for ``docs/`` names a deployment: there is no deployment column, held-out deployments are pooled,
the held-out list is recorded as a count, and the per-deployment class-share shift is an aggregate.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import split_data  # noqa: E402
from analysis import slices  # noqa: E402

# --- the pre-registered constants (docs/split-leak-2026-11.md quotes every one; a test checks it) ------------------
SPLIT_MODE = "label_level"
EXTRA_COLUMNS = ("split_mode", "split_grouped", "split_group")
SCATTER_TAG = 3_000_311          # issue 3, split-leak scatter: its own stream, never assign_splits's default_rng(seed)
GROUPED_SEED = 42                # the production split's seed
MERGE_RADIUS_M = 3.0             # the production split's merge radius, and the direct-companion distance
TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
EVAL_SPLITS = ("val", "test", "test_heldout")
K_STRATA = (1, 2, 3)             # k = 1, 2, 3+ direct companions
MIN_UNEXPOSED = slices.MIN_N     # a stratum with fewer unexposed rows is left blank (30)
INFLATION_MARGIN = 0.006         # round 3's own single-seed caveat (docs/retrain-2026-10.md, "What this does not establish")
CONFIRMATORY_SEEDS = 3
N_BOOT = 2000
BOOT_SEED = 3
CI_LEVEL = 0.95
CENSUS_MIN_FRACTION = 0.90       # check stops when exposed rows fall under this share of the iid expectation
TRAIN_SHARE = dict(split_data.TARGET)["train"]
IID_UNEXPOSED = 1 - TRAIN_SHARE  # a companion is outside train with probability about 0.3
PRECISION_TARGET = 0.92          # the operational threshold: precision of Agree (correct) on each arm's own val
POWER = 0.80                     # for the minimum detectable difference written by bound
MEMO_DRAWS = 200                 # Monte Carlo draws of the memorisation bound
EXPOSURE_CLASSES = ("exposed", "chain_only", "unexposed", "singleton", "train")
DESCRIPTIVE = ("via_pano", "via_nearby_only", "companion_same", "companion_opposite_only", "chain_only")


class SplitLeakError(Exception):
    """A refused step: the message says what to fix."""


# ---------------------------------------------------------------------------------------------------------------------
# Paths and small helpers
# ---------------------------------------------------------------------------------------------------------------------

def leaky_split_path(runs, label_type, seed):
    """``<runs>/<T>/splits/label_level_s<seed>.csv``: one leaky split per training seed (scatter seed = training
    seed). Outside the run directory, because fine_tune_classifier.py copies its split into the run directory."""
    return Path(runs) / label_type / "splits" / f"label_level_s{seed}.csv"


def exposure_path(runs, label_type, seed):
    """``<runs>/<T>/splits/exposure_s<seed>.csv``, written by ``check`` (label_uids: never copied to docs/)."""
    return Path(runs) / label_type / "splits" / f"exposure_s{seed}.csv"


def run_dir(runs, label_type, arm, seed):
    """``<runs>/<T>/<arm>-s<seed>`` (arm ``L`` or ``G``): one training run, its predictions and training record."""
    return Path(runs) / label_type / f"{arm}-s{seed}"


def stored_g_predictions(data, label_type, seed):
    """Where round 3's stored grouped-arm predictions are: ``predictions.csv`` for seed 42 (the production run) and
    ``predictions_baseline-s<seed>.csv`` for the seed replicates (``evaluate.py --tag baseline-s<seed>``). That the
    files exist and cover the production eval rows is checked by ``bound`` at Stage 0, before any GPU job."""
    name = "predictions.csv" if seed == GROUPED_SEED else f"predictions_baseline-s{seed}.csv"
    return os.path.join(data, label_type, name)


def g_prediction_path(data, runs, label_type, seed, g_source):
    if g_source == "stored":
        return stored_g_predictions(data, label_type, seed)
    if g_source == "fresh":
        return str(run_dir(runs, label_type, "G", seed) / "predictions.csv")
    raise ValueError(f"g_source must be stored or fresh, not {g_source!r}")


def csv_bytes(df):
    """The bytes ``split_data.py`` writes for ``df`` (no index, ``\\n`` line ends)."""
    return df.to_csv(index=False, lineterminator="\n").encode()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def read_str_csv(path):
    """Every cell as written (strings, empty for blank), as split_data.py reads its manifest."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _inside(path, directory):
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return os.path.commonpath([path, directory]) == directory


def kbin(k):
    """The k stratum: 1, 2 or 3 (= 3+); 0 for a singleton.

    >>> kbin([0, 1, 2, 3, 7]).tolist()
    [0, 1, 2, 3, 3]
    """
    return np.minimum(np.asarray(k), K_STRATA[-1])


# ---------------------------------------------------------------------------------------------------------------------
# make-split
# ---------------------------------------------------------------------------------------------------------------------

def heldout_deployments(split_df):
    """The deployments every row of which is ``test_heldout`` in the production split (the ``--holdout-cities`` it
    was made with, which this module never writes down; the recomputation proves the list right)."""
    held = split_df.groupby("city")["split"].agg(lambda s: bool((s == split_data.HELDOUT).all()))
    return sorted(held.index[held])


def recompute_grouped(manifest, coords, holdout, hard_uids, seed=GROUPED_SEED, radius_m=MERGE_RADIUS_M):
    """split_data.py's split of ``manifest`` (``assign_splits`` then ``add_hard``) and the merged group key per row."""
    df, group = split_data.assign_splits(manifest, seed=seed, holdout_cities=holdout, merge_radius_m=radius_m,
                                         coords=coords, return_group=True)
    return split_data.add_hard(df, hard_uids), group


def scatter_rng(seed):
    """The scatter's own generator, ``default_rng(SeedSequence([seed, SCATTER_TAG]))``."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), SCATTER_TAG]))


def scatter_splits(df, group, seed, rng=None):
    """The label-level split: ``df.split`` with the labels of multi-member-group rows (not ``test_heldout``)
    permuted within each class across the whole frame. Rows are taken in ``df``'s order (sorted by ``label_uid``
    in a split file), classes in sorted order, so the result is a pure function of ``df``, ``group`` and ``seed``.

    >>> d = pd.DataFrame({"label_class": ["correct"] * 4, "split": ["train", "train", "test", "val"]})
    >>> sorted(scatter_splits(d, pd.Series(["a", "a", "b", "b"]), 0)) == sorted(d.split)
    True
    """
    rng = rng or scatter_rng(seed)
    split = df["split"].to_numpy().astype(object).copy()
    size = pd.Series(np.asarray(group)).groupby(np.asarray(group)).transform("size").to_numpy()
    movable = (size > 1) & (split != split_data.HELDOUT)
    cls = df["label_class"].to_numpy()
    for c in sorted(set(cls[movable])):
        idx = np.flatnonzero(movable & (cls == c))
        split[idx] = split[idx][rng.permutation(len(idx))]
    return split


def leaky_frame(df, group, seed):
    """``df`` (a production split) with ``split_mode``, ``split_grouped`` and ``split_group`` before a scattered
    ``split``."""
    out = df.copy()
    at = list(out.columns).index("split")
    out.insert(at, "split_mode", SPLIT_MODE)
    out.insert(at + 1, "split_grouped", df["split"].to_numpy())
    out.insert(at + 2, "split_group", np.asarray(group))
    out["split"] = scatter_splits(df, group, seed)
    return out


def strip_leaky(leaky):
    """The production split a leaky file was made from: the extra columns dropped, ``split`` = ``split_grouped``."""
    out = leaky.drop(columns=list(EXTRA_COLUMNS))
    out["split"] = leaky["split_grouped"].to_numpy()
    return out


def make_split(manifest_path, split_path, rawlabels, out_path, seed, holdout=None, hard_examples=None,
               radius_m=MERGE_RADIUS_M, grouped_seed=GROUPED_SEED):
    """Recompute the grouped split, assert it equals ``split_path`` byte for byte, scatter, write ``out_path``.

    Refuses (``SplitLeakError``, nothing written) when ``out_path`` is under the data directory (the parent of
    ``<T>/split.csv``) or the recomputation differs. ``holdout`` defaults to :func:`heldout_deployments` of the
    production split, ``hard_examples`` (a CSV with ``label_uid``) to its ``hard == "1"`` rows. Returns a summary
    dict (counts only; no deployment names).
    """
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(split_path)))
    if _inside(out_path, data_dir):
        raise SplitLeakError(f"{out_path} is under the data directory {data_dir}; a leaky split is written under the "
                             f"run directory only")
    production = read_str_csv(split_path)
    manifest = read_str_csv(manifest_path)
    if holdout is None:
        holdout = heldout_deployments(production)
    hard = (set(read_str_csv(hard_examples)["label_uid"]) if hard_examples
            else set(production.loc[production["hard"] == "1", "label_uid"]))
    try:
        coords = split_data.load_coordinates(manifest, rawlabels)
    except ValueError as e:
        raise SplitLeakError(f"no coordinates for the 3 m merge: {e}") from e
    grouped, group = recompute_grouped(manifest, coords, holdout, hard, grouped_seed, radius_m)
    with open(split_path, "rb") as f:
        if csv_bytes(grouped) != f.read():
            raise SplitLeakError(f"the recomputed grouped split does not reproduce {split_path} byte for byte "
                                 f"(manifest, rawLabels coordinates, held-out deployments or hard examples differ "
                                 f"from the ones it was made with); nothing is trained on an uncoupled split")
    leaky = leaky_frame(grouped, group, seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp"
    with open(tmp, "wb") as f:
        f.write(csv_bytes(leaky))
    os.replace(tmp, out_path)
    moved = leaky["split"] != leaky["split_grouped"]
    spans = leaky[leaky["pano_id"] != ""].groupby("pano_id")["split"].agg(lambda s: {"train", "test"} <= set(s))
    return {"rows": len(leaky), "multi_member_rows": int(
        (leaky.groupby("split_group")["label_uid"].transform("size") > 1).sum()),
        "rows_moved": int(moved.sum()), "panoramas_spanning_train_and_test": int(spans.sum()),
        "heldout_deployments": len(holdout)}


# ---------------------------------------------------------------------------------------------------------------------
# Exposure and census
# ---------------------------------------------------------------------------------------------------------------------

def direct_pairs(df, coords=None, radius_m=MERGE_RADIUS_M):
    """Ordered index pairs ``(i, j)``, ``i != j``, of direct companions in ``df`` (positional): the same non-empty
    ``pano_id``, or within ``radius_m`` by ``split_data.haversine_m`` (labels without coordinates only by panorama).
    Also returns a bool array, True where the pair shares a panorama.

    >>> i, j, pano = direct_pairs(pd.DataFrame({"pano_id": ["a", "a", "b"]}))
    >>> list(zip(i.tolist(), j.tolist())), pano.tolist()
    ([(0, 1), (1, 0)], [True, True])
    """
    n = len(df)
    pano = df["pano_id"].astype(str).to_numpy()
    has = pano != ""
    codes = pd.Series(pano[has]).astype("category").cat.codes.to_numpy()
    pos = np.flatnonzero(has)
    order = np.argsort(codes, kind="mergesort")
    pi, pj = [], []
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for members in np.split(pos[order], bounds):
        if len(members) > 1:
            a, b = np.meshgrid(members, members, indexing="ij")
            keep = a != b
            pi.append(a[keep])
            pj.append(b[keep])
    i_p = np.concatenate(pi) if pi else np.array([], int)
    j_p = np.concatenate(pj) if pj else np.array([], int)
    i_n = j_n = np.array([], int)
    if coords is not None and n:
        xy = coords.reindex(df["label_uid"].to_numpy())
        lat, lon = xy["latitude"].to_numpy(float), xy["longitude"].to_numpy(float)
        ok = np.flatnonzero(np.isfinite(lat) & np.isfinite(lon))
        a, b = split_data.pairs_within(lat[ok], lon[ok], lat[ok], lon[ok], radius_m)
        keep = a != b
        i_n, j_n = ok[a[keep]], ok[b[keep]]
    code_p = i_p.astype(np.int64) * n + j_p
    code_all = np.concatenate([code_p, i_n.astype(np.int64) * n + j_n])
    uniq = np.unique(code_all)
    return (uniq // n).astype(int), (uniq % n).astype(int), np.isin(uniq, code_p)


def exposure_table(df, coords=None, radius_m=MERGE_RADIUS_M, group=None):
    """Per row of a (leaky) split: ``k`` (direct companions), ``exposure`` (``exposed``: a direct companion in train;
    ``chain_only``: a train row only further along its merged group; ``unexposed``; ``singleton``: k = 0; ``train``
    for train rows), ``via`` (``pano`` if a train companion shares the panorama, else ``nearby``; exposed rows only)
    and ``companion`` (``same`` if a train companion has the row's class, else ``opposite``; exposed rows only).

    ``df`` has ``label_uid, pano_id, label_class, split`` and ``split_group`` (or pass ``group``); ``coords`` as in
    ``split_data.load_coordinates`` (None: panorama companions only).
    """
    df = df.reset_index(drop=True)
    group = np.asarray(df["split_group"] if group is None else group)
    split = df["split"].to_numpy()
    cls = df["label_class"].to_numpy()
    i, j, by_pano = direct_pairs(df, coords, radius_m)
    n = len(df)
    k = np.bincount(i, minlength=n)
    tj = split[j] == "train"
    exposed = np.bincount(i[tj], minlength=n) > 0
    pano_train = np.bincount(i[tj & by_pano], minlength=n) > 0
    same_train = np.bincount(i[tj & (cls[i] == cls[j])], minlength=n) > 0
    group_train = pd.Series(split == "train").groupby(group).transform("any").to_numpy()
    size = pd.Series(group).groupby(group).transform("size").to_numpy()
    expo = np.where(exposed, "exposed", np.where(group_train, "chain_only",
                                                 np.where((k == 0) & (size == 1), "singleton", "unexposed")))
    expo = np.where(split == "train", "train", expo)
    out_exposed = expo == "exposed"
    return pd.DataFrame({"label_uid": df["label_uid"].to_numpy(), "split": split, "label_class": cls,
                         "group": group, "k": k, "exposure": expo,
                         "via": np.where(out_exposed, np.where(pano_train, "pano", "nearby"), ""),
                         "companion": np.where(out_exposed, np.where(same_train, "same", "opposite"), "")})


def census(expo, unexposed_share=IID_UNEXPOSED, min_fraction=CENSUS_MIN_FRACTION):
    """Exposed val/test rows against the iid expectation ``sum(1 - unexposed_share ** k)`` over val/test rows with
    k >= 1. ``ok`` is False under ``min_fraction`` of it (``check`` then stops: the scatter did not expose rows as
    the design assumes)."""
    rows = expo[expo["split"].isin(["val", "test"]) & (expo["k"].astype(int) >= 1)]
    expected = float((1 - unexposed_share ** rows["k"].astype(int)).sum())
    observed = int((rows["exposure"] == "exposed").sum())
    fraction = observed / expected if expected else float("nan")
    return {"rows": len(rows), "observed": observed, "expected": expected,
            "fraction": fraction if expected else None, "ok": bool(expected and fraction >= min_fraction)}


def deployment_shift(grouped, leaky):
    """How the scatter moved per-deployment class shares, as an aggregate with no deployment column: per split, the
    largest and median absolute change in incorrect share over deployments with at least ``slices.MIN_N`` rows in
    that split in both arms; per class, a chi-squared test of deployment x split in each arm."""
    from scipy.stats import chi2_contingency
    out = {"shares": {}, "chi2": {}}
    for s in ("train", "val", "test"):
        def share(df):
            part = df[df["split"] == s]
            g = part.groupby("city")["label_class"]
            return g.size(), g.agg(lambda c: float((c == "incorrect").mean()))
        n_g, sh_g = share(grouped)
        n_l, sh_l = share(leaky)
        both = [c for c in n_g.index if n_g[c] >= slices.MIN_N and n_l.get(c, 0) >= slices.MIN_N]
        d = np.abs(sh_l.reindex(both).to_numpy() - sh_g.reindex(both).to_numpy())
        out["shares"][s] = {"deployments": len(both), "max_abs_change": round(float(d.max()), 4) if len(d) else None,
                            "median_abs_change": round(float(np.median(d)), 4) if len(d) else None}
    for c in ("correct", "incorrect"):
        out["chi2"][c] = {}
        for arm, df in (("grouped", grouped), ("label_level", leaky)):
            part = df[(df["label_class"] == c) & df["split"].isin(["train", "val", "test"])]
            tab = pd.crosstab(part["city"], part["split"])
            tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                stat, p, dof, _ = chi2_contingency(tab.to_numpy())
                out["chi2"][c][arm] = {"chi2": round(float(stat), 3), "dof": int(dof), "p": float(p)}
            else:
                out["chi2"][c][arm] = None
    return out


def _coordinates(split_df, rawlabels):
    frame = split_df if "status" in split_df.columns else split_df.assign(status="ok")
    try:
        return split_data.load_coordinates(frame, rawlabels)
    except ValueError as e:
        raise SplitLeakError(f"no coordinates for the direct-companion test: {e}") from e


def _rawlabels_hashes(split_df, rawlabels):
    """sha256 of each rawLabels export the coordinates came from, sorted, with no file (deployment) names."""
    out = []
    for city in sorted(split_df["city"].unique()):
        path = os.path.join(rawlabels, f"{city}.csv")
        if os.path.isfile(path):
            out.append(sha256_file(path))
    return sorted(out)


def git_commit():
    import fine_tune_classifier as ft
    return ft.git_state()


def check(data, runs, types, seeds, rawlabels, out):
    """Stage 0's audit of every leaky split: coupling with the production split by sha256 and per (class, split)
    counts, the exposure table (written to :func:`exposure_path`), the census against the iid expectation, and the
    deployment shift aggregate; then ``<out>/design.json`` with the sha256 of every input. Raises
    ``SplitLeakError`` (and writes no design.json) on the first failure."""
    design = {"issue": 3, "constants": constants(), "commit": git_commit(), "coordinate_source": "rawlabels",
              "types": {}}
    heldout_counts = set()
    pending = []
    for t in types:
        split_path = os.path.join(data, t, "split.csv")
        with open(split_path, "rb") as f:
            prod_bytes = f.read()
        production = read_str_csv(split_path)
        held = heldout_deployments(production)
        heldout_counts.add(len(held))
        coords = _coordinates(production, rawlabels)
        manifest = os.path.join(data, t, "manifest.csv")
        rec = {"manifest_sha256": sha256_file(manifest) if os.path.exists(manifest) else None,
               "production_split_sha256": sha256_bytes(prod_bytes),
               "hard_label_uids_sha256": sha256_bytes("\n".join(sorted(
                   production.loc[production["hard"] == "1", "label_uid"])).encode()),
               "rawlabels_sha256": _rawlabels_hashes(production, rawlabels),
               "leaky_split_sha256": {}, "census": {}, "deployment_shift": {}, "scatter": {}}
        for seed in seeds:
            path = leaky_split_path(runs, t, seed)
            if not os.path.isfile(path):
                raise SplitLeakError(f"no {path}; run make-split first")
            leaky = read_str_csv(path)
            if set(leaky["split_mode"]) != {SPLIT_MODE}:
                raise SplitLeakError(f"{path}: split_mode is not {SPLIT_MODE} on every row")
            if csv_bytes(strip_leaky(leaky)) != prod_bytes:
                raise SplitLeakError(f"{path}: coupling broken (without its extra columns it is not {split_path} "
                                     f"byte for byte)")
            if not leaky.groupby(["label_class", "split"]).size().equals(
                    production.groupby(["label_class", "split"]).size()):
                raise SplitLeakError(f"{path}: coupling broken (per class and split counts differ)")
            expo = exposure_table(leaky, coords)
            c = census(expo)
            if not c["ok"]:
                raise SplitLeakError(f"{path}: exposure census {c['observed']} exposed val/test rows, "
                                     f"{_f(c['fraction'], '{:.1%}')} of the iid expectation {c['expected']:.1f} (needs "
                                     f"{CENSUS_MIN_FRACTION:.0%}); the scatter did not expose rows as designed")
            pending.append((exposure_path(runs, t, seed), expo))
            rec["leaky_split_sha256"][str(seed)] = sha256_file(path)
            rec["census"][str(seed)] = c
            rec["deployment_shift"][str(seed)] = deployment_shift(production, leaky)
            spans = leaky[leaky["pano_id"] != ""].groupby("pano_id")["split"].agg(
                lambda s: {"train", "test"} <= set(s))
            rec["scatter"][str(seed)] = {"rows_moved": int((leaky["split"] != leaky["split_grouped"]).sum()),
                                         "panoramas_spanning_train_and_test": int(spans.sum())}
        design["types"][t] = rec
    design["heldout_deployments"] = {"count": heldout_counts.pop() if len(heldout_counts) == 1
                                     else sorted(heldout_counts)}
    for path, expo in pending:
        expo.to_csv(path, index=False, lineterminator="\n")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "design.json"), "w") as f:
        json.dump(design, f, indent=2, sort_keys=True)
        f.write("\n")
    return design


def constants():
    return {"split_mode": SPLIT_MODE, "scatter_tag": SCATTER_TAG, "grouped_seed": GROUPED_SEED,
            "merge_radius_m": MERGE_RADIUS_M, "k_strata": "1, 2, 3+", "min_unexposed": MIN_UNEXPOSED,
            "inflation_margin": INFLATION_MARGIN, "confirmatory_seeds": CONFIRMATORY_SEEDS, "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED, "ci_level": CI_LEVEL, "census_min_fraction": CENSUS_MIN_FRACTION,
            "iid_unexposed": IID_UNEXPOSED, "precision_target": PRECISION_TARGET, "power": POWER,
            "memo_draws": MEMO_DRAWS}


# ---------------------------------------------------------------------------------------------------------------------
# Estimation: weighted AUC, CMH pooling, the group-and-seed bootstrap
# ---------------------------------------------------------------------------------------------------------------------

def weighted_auc(rw, y, score):
    """AUC of ``score`` for ``y`` (True = positive = incorrect) under each row of the weight matrix ``rw`` (B x n;
    integer weights = a bootstrap's multiplicities), ties counted half. Returns a length-B array (NaN where a
    weighting leaves one class).

    >>> weighted_auc(np.ones((1, 4)), np.array([0, 0, 1, 1], bool), np.array([0.1, 0.4, 0.35, 0.8])).tolist()
    [0.75]
    """
    from scipy.sparse import csr_matrix
    rw = np.atleast_2d(np.asarray(rw, float))
    y, score = np.asarray(y, bool), np.asarray(score, float)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    tie = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]) if len(s) else np.array([], int)
    n_tie = int(tie[-1]) + 1 if len(s) else 0
    t_mat = csr_matrix((np.ones(len(s)), (np.arange(len(s)), tie)), shape=(len(s), n_tie))
    w = rw[:, order]
    pos = np.asarray((t_mat.T @ (w * y[order]).T).T)
    neg = np.asarray((t_mat.T @ (w * ~y[order]).T).T)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def cmh_pool(strata, min_unexposed=MIN_UNEXPOSED):
    """Pool per-stratum differences ``err_exp - err_unexp`` with Cochran-Mantel-Haenszel weights
    ``n_exp * n_unexp / (n_exp + n_unexp)``. A stratum with fewer than ``min_unexposed`` unexposed rows (or no
    exposed row) is blank. Returns ``(pooled, blank)``, ``blank`` a bool Series aligned with ``strata``.

    >>> st = pd.DataFrame({"n_exp": [100], "n_unexp": [100], "err_exp": [0.1], "err_unexp": [0.2]})
    >>> round(cmh_pool(st)[0], 3)
    -0.1
    """
    blank = (strata["n_unexp"] < min_unexposed) | (strata["n_exp"] <= 0)
    use = strata[~blank]
    w = use["n_exp"] * use["n_unexp"] / (use["n_exp"] + use["n_unexp"])
    d = use["err_exp"] - use["err_unexp"]
    pooled = float((w * d).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return pooled, blank


@dataclass
class TypeData:
    """One label type's study inputs, aligned with ``rows`` (a RangeIndex frame with ``label_uid, label_class,
    split`` (the grouped arm) and ``group`` (the merged group key, the bootstrap cluster))."""
    name: str
    rows: pd.DataFrame
    l_split: dict = field(default_factory=dict)   # seed -> leaky split labels
    l_expo: dict = field(default_factory=dict)    # seed -> exposure_table frame
    l_pred: dict = field(default_factory=dict)    # seed -> p_incorrect (NaN where unscored)
    g_pred: dict = field(default_factory=dict)    # seed -> p_incorrect (NaN where unscored)

    def add_l(self, seed, split, expo, p_incorrect):
        self.l_split[seed] = np.asarray(split, dtype=object)
        self.l_expo[seed] = expo.reset_index(drop=True)
        self.l_pred[seed] = np.asarray(p_incorrect, float)

    def add_g(self, seed, p_incorrect):
        self.g_pred[seed] = np.asarray(p_incorrect, float)


def _nanmean(a, axis=None):
    """``np.nanmean`` without the empty-slice warning (an all-NaN slice is NaN, which the summaries blank)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _name_tag(name):
    return int(hashlib.sha256(str(name).encode()).hexdigest()[:8], 16)


def _multiplicities(n_groups, n_boot, rng):
    """(n_boot + 1) x n_groups: row 0 all ones (the point estimate), then draws of n_groups whole groups."""
    if n_groups == 0:
        return np.ones((n_boot + 1, 0), np.uint16)
    draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot) if n_boot else \
        np.empty((0, n_groups), int)
    return np.vstack([np.ones((1, n_groups), int), draws]).astype(np.uint16)


def _summ(vec, t_ci=None):
    vec = np.asarray(vec, float)
    boot = vec[1:]
    lo = hi = None
    if len(boot) and np.isfinite(boot).any():
        lo, hi = (float(v) for v in np.nanpercentile(boot, [100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2]))
    t_lo, t_hi = t_ci if t_ci is not None else (None, None)
    return {"point": float(vec[0]) if np.isfinite(vec[0]) else None, "lo": lo, "hi": hi, "t_lo": t_lo,
            "t_hi": t_hi}


def t_interval(values):
    """``CI_LEVEL`` t interval on seed-level values, or None with fewer than two."""
    from scipy.stats import t
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) < 2:
        return None
    half = t.ppf((1 + CI_LEVEL) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - half), float(v.mean() + half)


def welch_interval(a, b):
    """``CI_LEVEL`` Welch interval on mean(a) - mean(b) (seed-level values of two arms), or None if either arm has
    fewer than two values."""
    from scipy.stats import t
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    b = np.asarray([x for x in b if np.isfinite(x)], float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    if va + vb == 0:
        return float(d), float(d)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t.ppf((1 + CI_LEVEL) / 2, df) * np.sqrt(va + vb)
    return float(d - half), float(d + half)


def _select(per_seed, sel):
    """``per_seed`` (n_seeds x (B+1)) at each replicate's drawn seeds ``sel`` ((B+1) x n_seeds) -> (B+1) x n_seeds."""
    return np.take_along_axis(per_seed.T, sel, axis=1)


def _seed_draws(n_seeds, n_boot, rng):
    draws = rng.integers(0, n_seeds, size=(n_boot, n_seeds)) if n_boot else np.empty((0, n_seeds), int)
    return np.vstack([np.arange(n_seeds)[None, :], draws])


def _cmh_parts(rw, err, treated, control, kb, min_unexposed, record):
    """Numerator and denominator (length B+1) of the CMH-pooled ``treated - control`` difference of ``err`` over k
    strata; ``record`` collects the point-estimate strata (blank ones too) as dicts."""
    num = np.zeros(rw.shape[0])
    den = np.zeros(rw.shape[0])
    for k in K_STRATA:
        e, u = treated & (kb == k), control & (kb == k)
        n_e, n_u = int(e.sum()), int(u.sum())
        if n_e == 0 and n_u == 0:
            continue
        blank = n_u < min_unexposed or n_e == 0
        rec = {"k": k, "n_exp": n_e, "n_unexp": n_u,
               "err_exp": float(err[e].mean()) if n_e else None, "err_unexp": float(err[u].mean()) if n_u else None,
               "blank": blank}
        record.append(rec)
        if blank:
            continue
        we, wu = rw[:, e].sum(1), rw[:, u].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (rw[:, e] @ err[e]) / we - (rw[:, u] @ err[u]) / wu
            w = we * wu / (we + wu)
        ok = (we > 0) & (wu > 0)
        num += np.where(ok, w * np.where(ok, d, 0), 0)
        den += np.where(ok, w, 0)
    return num, den


def _type_stats(td, seeds_l, seeds_g, n_boot, boot_seed, min_unexposed):
    """Every per-seed statistic of one type under the shared group draws (row 0 = point estimate)."""
    rows = td.rows
    y = (rows["label_class"] == "incorrect").to_numpy()
    gcode = pd.factorize(rows["group"])[0]
    size = np.bincount(gcode)[gcode]
    multi = size > 1
    gsplit = rows["split"].to_numpy()

    def universe(name, mask):
        members = np.unique(gcode[mask])
        idx = np.full(gcode.max() + 1 if len(gcode) else 0, -1)
        idx[members] = np.arange(len(members))
        rng = np.random.default_rng(np.random.SeedSequence([boot_seed, _name_tag(td.name), _name_tag(name)]))
        return _multiplicities(len(members), n_boot, rng), idx

    in_split = {s: (gsplit == s) | np.any([td.l_split[x] == s for x in seeds_l], axis=0) for s in ("test", "val")}
    universes = {"test": universe("test", in_split["test"]), "val": universe("val", in_split["val"]),
                 "test_heldout": universe("test_heldout", gsplit == split_data.HELDOUT)}

    def rw(uname, mask):
        m, idx = universes[uname]
        cols = idx[gcode[mask]]
        if (cols < 0).any():
            raise ValueError(f"{td.name}: rows outside the {uname} bootstrap universe")
        return m[:, cols].astype(float)

    nb = n_boot + 1
    stats = {"l": {}, "g": {}}
    strata = []
    for arm, seeds, preds, splits in (("l", seeds_l, td.l_pred, td.l_split),
                                      ("g", seeds_g, td.g_pred, {s: gsplit for s in seeds_g})):
        for s in seeds:
            p, sp = preds[s], splits[s]
            out = {}
            for name, uname, mask in (("test", "test", sp == "test"), ("val", "val", sp == "val"),
                                      ("heldout", "test_heldout", sp == split_data.HELDOUT),
                                      ("multi", "test", (sp == "test") & multi),
                                      ("single", "test", (sp == "test") & ~multi)):
                if mask.any() and np.isnan(p[mask]).any():
                    raise ValueError(f"{td.name} {arm.upper()} seed {s}: unscored {name} rows")
                out[f"auc_{name}"] = weighted_auc(rw(uname, mask), y[mask], p[mask]) if mask.any() \
                    else np.full(nb, np.nan)
            if arm == "l":
                ex = td.l_expo[s]
                expo, k = ex["exposure"].to_numpy(), kbin(ex["k"].to_numpy(int))
                via, comp = ex["via"].to_numpy(), ex["companion"].to_numpy()
                err = ((p >= 0.5) != y).astype(float)
                ptrue = np.where(y, p, 1 - p)
                defs = {"exposed": expo == "exposed", "via_pano": (expo == "exposed") & (via == "pano"),
                        "via_nearby_only": (expo == "exposed") & (via == "nearby"),
                        "companion_same": (expo == "exposed") & (comp == "same"),
                        "companion_opposite_only": (expo == "exposed") & (comp == "opposite"),
                        "chain_only": expo == "chain_only"}
                control = expo == "unexposed"
                for where, uname, sub in (("test", "test", sp == "test"), ("val", "val", sp == "val")):
                    w_sub = rw(uname, sub)
                    names = defs if where == "test" else {"exposed": defs["exposed"]}
                    for dname, treated in names.items():
                        for metric, values, sign in (("error", err, 1.0), ("p_true", ptrue, 1.0)):
                            rec = []
                            num, den = _cmh_parts(w_sub, values[sub] * sign, treated[sub], control[sub], k[sub],
                                                  min_unexposed, rec)
                            out[f"{where}:{dname}:{metric}:num"] = num
                            out[f"{where}:{dname}:{metric}:den"] = den
                            if where == "test" and dname == "exposed" and metric == "error":
                                strata += [{"type": td.name, "seed": s, **r} for r in rec]
                aucs = {}
                for kk in K_STRATA:
                    for cname, cmask in (("exposed", defs["exposed"]), ("unexposed", control)):
                        m = (sp == "test") & cmask & (k == kk)
                        ok = m.sum() >= min_unexposed and min(y[m].sum(), (~y[m]).sum()) >= slices.MIN_CLASS_FOR_CI
                        aucs[f"k{kk}_{cname}"] = float(weighted_auc(np.ones((1, m.sum())), y[m], p[m])[0]) if ok \
                            else None
                out["stratum_auc"] = aucs
            stats[arm][s] = out
    return stats, strata


def estimate(types, n_boot=N_BOOT, boot_seed=BOOT_SEED, min_unexposed=MIN_UNEXPOSED):
    """Every estimate of the study, with 95% percentile intervals from ``n_boot`` replicates. Per type, whole merged
    groups are drawn from the union of groups with a row in G test or any L test (val and test_heldout: their own
    universes), one draw shared by both arms and every seed; seeds are resampled independently per arm, the same
    seed draw for every type. Types are drawn independently. Row 0 of every replicate matrix is the point estimate.

    Returns a dict: ``contrast`` (pooled exposed - unexposed, ``error`` at 0.5 and ``p_true``, with the t interval
    on per-seed values), ``contrast_by_type``, ``contrast_descriptive`` (subsets of the exposed rows, chain-only
    rows, and val), ``strata`` (a DataFrame of the point-estimate strata, ``blank`` marked), ``stratum_auc``,
    ``inflation`` (``test`` and ``val``: ``pooled`` (macro over types) and ``by_type``; Welch t interval on per-seed
    macro values), ``multi_member``, ``heldout_delta``, ``singleton_delta``, ``gap_of_gaps``, and the seeds.
    """
    seeds_l = sorted(types[0].l_pred)
    seeds_g = sorted(types[0].g_pred)
    for td in types:
        if sorted(td.l_pred) != seeds_l or sorted(td.g_pred) != seeds_g:
            raise ValueError(f"{td.name}: seeds differ from {types[0].name}'s")
    per_type, strata = {}, []
    for td in types:
        per_type[td.name], st = _type_stats(td, seeds_l, seeds_g, n_boot, boot_seed, min_unexposed)
        strata += st
    sel_l = _seed_draws(len(seeds_l), n_boot, np.random.default_rng(np.random.SeedSequence([boot_seed, 1])))
    sel_g = _seed_draws(len(seeds_g), n_boot, np.random.default_rng(np.random.SeedSequence([boot_seed, 2])))
    names = [td.name for td in types]

    def stack(arm, key, t):
        seeds = seeds_l if arm == "l" else seeds_g
        return np.vstack([per_type[t][arm][s][key] for s in seeds])

    def pooled_ratio(key, ts):
        num = sum(stack("l", f"{key}:num", t) for t in ts)
        den = sum(stack("l", f"{key}:den", t) for t in ts)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = _select(num, sel_l).sum(1) / _select(den, sel_l).sum(1)
            seed_values = num[:, 0] / den[:, 0]
        return _summ(vec, t_interval(seed_values)), seed_values

    res = {"seeds_l": seeds_l, "seeds_g": seeds_g, "contrast": {}, "contrast_by_type": {},
           "contrast_descriptive": {}}
    for metric in ("error", "p_true"):
        res["contrast"][metric], seed_values = pooled_ratio(f"test:exposed:{metric}", names)
        res["contrast"][metric]["per_seed"] = {str(s): (float(v) if np.isfinite(v) else None)
                                               for s, v in zip(seeds_l, seed_values)}
    for t in names:
        res["contrast_by_type"][t] = {m: pooled_ratio(f"test:exposed:{m}", [t])[0] for m in ("error", "p_true")}
    for d in DESCRIPTIVE:
        res["contrast_descriptive"][d] = {m: pooled_ratio(f"test:{d}:{m}", names)[0] for m in ("error", "p_true")}
    res["contrast_descriptive"]["val_exposed"] = {m: pooled_ratio(f"val:exposed:{m}", names)[0]
                                                  for m in ("error", "p_true")}
    res["strata"] = pd.DataFrame(strata, columns=["type", "seed", "k", "n_exp", "n_unexp", "err_exp", "err_unexp",
                                                  "blank"])
    res["stratum_auc"] = {t: {str(s): per_type[t]["l"][s]["stratum_auc"] for s in seeds_l} for t in names}

    def arm_mean(arm, key, t):
        sel = sel_l if arm == "l" else sel_g
        return _nanmean(_select(stack(arm, key, t), sel), axis=1)

    def delta(key):
        by_type = {t: arm_mean("l", key, t) - arm_mean("g", key, t) for t in names}
        pooled = np.mean([by_type[t] for t in names], axis=0)
        macro_l = np.mean([stack("l", key, t)[:, 0] for t in names], axis=0)
        macro_g = np.mean([stack("g", key, t)[:, 0] for t in names], axis=0)
        welch = welch_interval(macro_l, macro_g)
        out = {"pooled": _summ(pooled, welch), "by_type": {t: _summ(by_type[t]) for t in names}}
        return out, pooled

    res["inflation"] = {}
    res["inflation"]["test"], infl_test = delta("auc_test")
    res["inflation"]["val"], _ = delta("auc_val")
    res["heldout_delta"], ho = delta("auc_heldout")
    res["singleton_delta"], _ = delta("auc_single")
    res["gap_of_gaps"] = {"pooled": _summ(infl_test - ho)}
    res["multi_member"] = {}
    for td in types:
        t = td.name
        d = float(_nanmean(stack("l", "auc_multi", t)[:, 0]) - _nanmean(stack("g", "auc_multi", t)[:, 0]))
        gs = td.rows["split"].to_numpy()
        size = td.rows.groupby("group")["label_uid"].transform("size").to_numpy()
        share = float(((gs == "test") & (size > 1)).sum() / max((gs == "test").sum(), 1))
        res["multi_member"][t] = {"delta_auc": d, "share_of_test": share, "product": d * share}
    return res


# ---------------------------------------------------------------------------------------------------------------------
# The pre-registered rule
# ---------------------------------------------------------------------------------------------------------------------

def _contrast_class(lo, hi):
    return "leak" if hi < 0 else ("hurts" if lo > 0 else "none")


def _inflation_class(lo, hi, margin):
    return "shown" if lo > 0 else ("none" if lo > -margin and hi < margin else "unresolved")


def rule_outcome(contrast, inflation, n_seeds, g_source, margin=INFLATION_MARGIN):
    """The pre-registered reading (docs/split-leak-2026-11.md, "Pre-registered rule").

    ``contrast`` and ``inflation`` are dicts with the deciding bootstrap interval ``lo``/``hi`` and the sensitivity
    t interval ``t_lo``/``t_hi`` (None when unavailable). The contrast is exposed minus unexposed error, so a leak is
    negative: "leak shown" when the upper bound is below 0, "exposure hurts" when the lower bound is above 0, "not
    shown" otherwise. Inflation: "shown" when the interval lies above 0, "no material inflation" inside
    ±``margin``, otherwise "unresolved, upper bound X". When the two intervals give different outcomes the reading is
    "unresolved". Against stored baselines (``g_source="stored"``) the inflation label is descriptive; with fewer
    than ``CONFIRMATORY_SEEDS`` seeds both labels are provisional.

    >>> rule_outcome({"lo": -0.03, "hi": -0.01}, {"lo": 0.001, "hi": 0.004}, 3, "fresh")["contrast"]
    'leak shown'
    """
    if g_source not in ("stored", "fresh"):
        raise ValueError(f"g_source must be stored or fresh, not {g_source!r}")
    lo, hi = contrast["lo"], contrast["hi"]
    t_lo, t_hi = contrast.get("t_lo"), contrast.get("t_hi")
    c = _contrast_class(lo, hi) if lo is not None else None
    if c is None:
        c_text = f"not estimable (every stratum has fewer than {MIN_UNEXPOSED} unexposed rows)"
    elif t_lo is not None and _contrast_class(t_lo, t_hi) != c:
        c_text = "unresolved (the bootstrap and t intervals disagree)"
    else:
        c_text = {"leak": "leak shown", "hurts": "exposure hurts"}.get(
            c, f"not shown (95% interval [{lo:+.4f}, {hi:+.4f}])")
    lo, hi = inflation["lo"], inflation["hi"]
    t_lo, t_hi = inflation.get("t_lo"), inflation.get("t_hi")
    i = _inflation_class(lo, hi, margin) if lo is not None else None
    if i is None:
        i_text = "not estimable (no bootstrap interval)"
    elif t_lo is not None and _inflation_class(t_lo, t_hi, margin) != i:
        i_text = f"unresolved, upper bound {max(hi, t_hi):+.4f} (the bootstrap and t intervals disagree)"
    else:
        i_text = {"shown": "shown", "none": "no material inflation"}.get(i, f"unresolved, upper bound {hi:+.4f}")
    if g_source == "stored":
        i_text += " (descriptive: runs from different dates and environments)"
    provisional = n_seeds < CONFIRMATORY_SEEDS
    if provisional:
        tail = f" (provisional: {n_seeds} seed{'s' if n_seeds != 1 else ''})"
        c_text += tail
        i_text += tail
    return {"contrast": c_text, "inflation": i_text, "provisional": provisional,
            "confirmatory_inflation": g_source == "fresh" and not provisional}


# ---------------------------------------------------------------------------------------------------------------------
# Stage 0 bound
# ---------------------------------------------------------------------------------------------------------------------

def memorisation_bound(y, score, companions, p_train=TRAIN_SHARE, n_draws=MEMO_DRAWS, rng=None):
    """What the grouped model's test AUC would become if every multi-member test row it scored had instead been
    scattered and memorised: each direct companion is in train with probability ``p_train``; an exposed row gets a
    perfect score (``perfect``) or its train companions' class share (``companion``: memorising a label of the
    opposite class is wrong). Returns ``auc_grouped`` and the mean memorised AUC minus it, over ``n_draws`` draws, for
    both. Not a strict bound: it leaves out the effect of selecting the checkpoint on a leaky val.

    ``companions`` holds, per row, the classes (True = incorrect) of its direct companions (empty for a singleton).
    """
    rng = rng or np.random.default_rng(0)
    y, score = np.asarray(y, bool), np.asarray(score, float)
    base = float(weighted_auc(np.ones((1, len(y))), y, score)[0])
    multi = [i for i, c in enumerate(companions) if len(c)]
    perfect, comp = [], []
    for _ in range(n_draws):
        s_p, s_c = score.copy(), score.copy()
        for i in multi:
            c = np.asarray(companions[i], bool)
            in_train = rng.random(len(c)) < p_train
            if in_train.any():
                s_p[i] = 1.0 if y[i] else 0.0
                s_c[i] = float(c[in_train].mean())
        perfect.append(weighted_auc(np.ones((1, len(y))), y, s_p)[0])
        comp.append(weighted_auc(np.ones((1, len(y))), y, s_c)[0])
    return {"auc_grouped": base, "perfect": float(np.mean(perfect)) - base, "companion": float(np.mean(comp)) - base}


def min_detectable_difference(strata, p_err, n_seeds=CONFIRMATORY_SEEDS, alpha=1 - CI_LEVEL, power=POWER,
                              min_unexposed=MIN_UNEXPOSED):
    """The smallest exposed - unexposed error difference the CMH-pooled contrast detects with ``power`` at two-sided
    ``alpha``, from the census's per-stratum counts (``type, k, n_exp, n_unexp`` for one seed, repeated
    ``n_seeds`` times) and each type's baseline error rate ``p_err[type]``: ``(z_{1-alpha/2} + z_power) *
    sqrt(sum w p (1 - p)) / sum w`` with CMH weights ``w``. Ignores clustering, so it is optimistic. NaN when every
    stratum is blank."""
    from scipy.stats import norm
    use = strata[(strata["n_unexp"] >= min_unexposed) & (strata["n_exp"] > 0)]
    if use.empty:
        return float("nan")
    w = (use["n_exp"] * use["n_unexp"] / (use["n_exp"] + use["n_unexp"])).to_numpy() * n_seeds
    p = use["type"].map(p_err).to_numpy(float)
    se = np.sqrt((w * p * (1 - p)).sum()) / w.sum()
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def expected_halfwidth(sd_seed, sd_boot, multi_share, n_seeds=CONFIRMATORY_SEEDS, level=CI_LEVEL):
    """Planning half-width of one type's inflation interval: ``z * sqrt(2 sd_seed^2 / n_seeds + 2 multi_share
    sd_boot^2)``: two arms' seed means, plus sampling noise from the multi-member part of test, the only rows on which
    the two test sets differ (``sd_boot`` is the group-bootstrap sd of one test AUC)."""
    from scipy.stats import norm
    return float(norm.ppf((1 + level) / 2) * np.sqrt(2 * sd_seed ** 2 / n_seeds + 2 * multi_share * sd_boot ** 2))


def _read_predictions(path, split_df, arm_split, what):
    """``p_incorrect`` aligned with ``split_df``, NaN outside the eval splits; refuses predictions that do not cover
    exactly the arm's val/test/test_heldout rows with the same splits."""
    if not os.path.isfile(path):
        raise SplitLeakError(f"{what}: no predictions at {path}")
    pred = pd.read_csv(path, dtype={"label_uid": str, "split": str})
    want = pd.DataFrame({"label_uid": split_df["label_uid"].to_numpy(), "split": np.asarray(arm_split)})
    want = want[want["split"].isin(EVAL_SPLITS)]
    if not pred["label_uid"].is_unique or set(pred["label_uid"]) != set(want["label_uid"]):
        raise SplitLeakError(f"{what}: {path} does not cover exactly the arm's val/test/test_heldout rows")
    m = want.merge(pred[["label_uid", "split", "p_incorrect"]], on="label_uid", how="left", suffixes=("", "_pred"),
                   validate="one_to_one")
    if (m["split"] != m["split_pred"]).any():
        raise SplitLeakError(f"{what}: {path} does not cover the arm's rows with the same splits")
    out = pd.Series(np.nan, index=split_df["label_uid"].to_numpy())
    out.loc[m["label_uid"].to_numpy()] = m["p_incorrect"].to_numpy(float)
    return out.to_numpy()


def bound(data, runs, types, rawlabels, out, g_seeds=(42, 43, 44), n_boot=N_BOOT, boot_seed=BOOT_SEED,
          seed=GROUPED_SEED):
    """Stage 0's three planning figures, from the stored grouped-arm predictions and the seed-``seed`` exposure
    census: the memorisation bound on the inflation (per type and pooled), the expected inflation interval
    half-width at three seeds (and whether the ±``INFLATION_MARGIN`` band can be reached), and the contrast's minimum
    detectable difference at three seeds. Writes ``<out>/bound.json`` and ``<out>/bound.md``."""
    res = {"constants": constants(), "commit": git_commit(), "types": {}}
    strata_all, p_err = [], {}
    for t in types:
        split_path = os.path.join(data, t, "split.csv")
        prod = read_str_csv(split_path)
        coords = _coordinates(prod, rawlabels)
        y_all = (prod["label_class"] == "incorrect").to_numpy()
        preds, hashes = {}, {}
        for s in g_seeds:
            path = stored_g_predictions(data, t, s)
            if os.path.isfile(path):
                preds[s] = _read_predictions(path, prod, prod["split"], f"{t} stored grouped seed {s}")
                hashes[str(s)] = sha256_file(path)
        if seed not in preds:
            raise SplitLeakError(f"{t}: no stored grouped seed-{seed} predictions at "
                                 f"{stored_g_predictions(data, t, seed)}")
        test = (prod["split"] == "test").to_numpy()
        i, j, _ = direct_pairs(prod, coords)
        comp = [[] for _ in range(len(prod))]
        for a, b in zip(i, j):
            comp[a].append(y_all[b])
        idx = np.flatnonzero(test)
        p42 = preds[seed]
        rng = np.random.default_rng(np.random.SeedSequence([boot_seed, _name_tag(t), _name_tag("memo")]))
        memo = memorisation_bound(y_all[idx], p42[idx], [np.array(comp[x], bool) for x in idx], rng=rng)
        gcode = pd.factorize(read_str_csv(leaky_split_path(runs, t, seed))["split_group"])[0] \
            if os.path.isfile(leaky_split_path(runs, t, seed)) else pd.factorize(prod["pano_id"])[0]
        members = np.unique(gcode[idx])
        col = {g: c for c, g in enumerate(members)}
        m = _multiplicities(len(members), n_boot,
                            np.random.default_rng(np.random.SeedSequence([boot_seed, _name_tag(t), 7])))
        boot = weighted_auc(m[:, [col[g] for g in gcode[idx]]].astype(float), y_all[idx], p42[idx])
        sd_boot = float(np.nanstd(boot[1:], ddof=1)) if n_boot > 1 else float("nan")
        seed_aucs = [float(weighted_auc(np.ones((1, len(idx))), y_all[idx], preds[s][idx])[0]) for s in sorted(preds)]
        sd_seed = float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else float("nan")
        size = np.bincount(gcode)[gcode]
        multi_share = float((size[idx] > 1).mean())
        hw = expected_halfwidth(sd_seed, sd_boot, multi_share)
        expo_path = exposure_path(runs, t, seed)
        if not os.path.isfile(expo_path):
            raise SplitLeakError(f"{t}: no {expo_path}; run check first")
        expo = read_str_csv(expo_path)
        expo = expo[expo["split"] == "test"].assign(k=lambda d: kbin(d["k"].astype(int)))
        st = []
        for k in K_STRATA:
            part = expo[expo["k"] == k]
            st.append({"type": t, "k": k, "n_exp": int((part["exposure"] == "exposed").sum()),
                       "n_unexp": int((part["exposure"] == "unexposed").sum())})
        st = pd.DataFrame(st)
        strata_all.append(st)
        multi_test = idx[size[idx] > 1]
        p_err[t] = float(((p42[multi_test] >= 0.5) != y_all[multi_test]).mean()) if len(multi_test) else float("nan")
        res["types"][t] = {
            "memorisation_bound": memo["perfect"], "companion_bound": memo["companion"],
            "auc_grouped_test": memo["auc_grouped"], "sd_seed": sd_seed, "sd_boot": sd_boot, "seeds_found": sorted(preds),
            "multi_member_share_of_test": multi_share, "expected_halfwidth": hw,
            "band_reachable": bool(hw < INFLATION_MARGIN) if np.isfinite(hw) else None,
            "baseline_error_multi_member": p_err[t], "census_strata": st.drop(columns="type").to_dict("records"),
            "min_detectable_difference": min_detectable_difference(st, p_err, n_seeds=CONFIRMATORY_SEEDS),
            "predictions_sha256": hashes}
    tt = res["types"]
    pooled_hw = float(np.sqrt(sum(tt[t]["expected_halfwidth"] ** 2 for t in types)) / len(types))
    res["pooled"] = {
        "memorisation_bound": float(np.mean([tt[t]["memorisation_bound"] for t in types])),
        "companion_bound": float(np.mean([tt[t]["companion_bound"] for t in types])),
        "expected_halfwidth": pooled_hw,
        "band_reachable": bool(pooled_hw < INFLATION_MARGIN) if np.isfinite(pooled_hw) else None,
        "min_detectable_difference": min_detectable_difference(pd.concat(strata_all), p_err),
    }
    res["memorisation_bound_under_margin_everywhere"] = bool(
        all(tt[t]["memorisation_bound"] < INFLATION_MARGIN for t in types)
        and res["pooled"]["memorisation_bound"] < INFLATION_MARGIN)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bound.json"), "w") as f:
        json.dump(_jsonable(res), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out, "bound.md"), "w") as f:
        f.write(bound_markdown(res, types))
    return res


def _f(v, fmt="{:.4f}"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def bound_markdown(res, types):
    lines = ["# Stage 0: planning figures (issue #3)", "",
             f"Memorisation bound: the stored seed-{GROUPED_SEED} grouped model's test AUC with each multi-member test "
             f"row perfect-scored with its exposure probability (each direct companion in train with probability "
             f"{TRAIN_SHARE}), minus the AUC itself; the companion column memorises the train companions' class "
             f"instead. Not a strict bound: it leaves out the effect of selecting the checkpoint on a leaky val. "
             f"Half-width: the expected 95% half-width of the inflation at {CONFIRMATORY_SEEDS} seeds; the "
             f"±{INFLATION_MARGIN} band is reachable when it is smaller. MDD: the smallest exposed - unexposed error "
             f"difference the pooled contrast detects with power {POWER} at {CONFIRMATORY_SEEDS} seeds (strata with "
             f"fewer than {MIN_UNEXPOSED} unexposed rows left out; clustering ignored, so optimistic).", "",
             "| Type | grouped test AUC | memorisation bound | companion bound | half-width | band reachable | MDD |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for t in types:
        r = res["types"][t]
        lines.append(f"| {t} | {_f(r['auc_grouped_test'])} | {_f(r['memorisation_bound'], '{:+.4f}')} | "
                     f"{_f(r['companion_bound'], '{:+.4f}')} | {_f(r['expected_halfwidth'])} | "
                     f"{'' if r['band_reachable'] is None else ('yes' if r['band_reachable'] else 'no')} | "
                     f"{_f(r['min_detectable_difference'])} |")
    p = res["pooled"]
    lines.append(f"| pooled (macro) | | {_f(p['memorisation_bound'], '{:+.4f}')} | "
                 f"{_f(p['companion_bound'], '{:+.4f}')} | {_f(p['expected_halfwidth'])} | "
                 f"{'' if p['band_reachable'] is None else ('yes' if p['band_reachable'] else 'no')} | "
                 f"{_f(p['min_detectable_difference'])} |")
    lines += ["", f"Memorisation bound under {INFLATION_MARGIN} for every type and pooled: "
              f"{'yes' if res['memorisation_bound_under_margin_everywhere'] else 'no'}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# Loading a finished run, and the report
# ---------------------------------------------------------------------------------------------------------------------

def load_study(data, runs, types, seeds_l, seeds_g, g_source):
    """``TypeData`` per type from the files of a finished run, plus the frames the report needs (``extras``)."""
    studies, extras = [], {}
    for t in types:
        split_path = os.path.join(data, t, "split.csv")
        prod = read_str_csv(split_path)
        leaky = {s: read_str_csv(leaky_split_path(runs, t, s)) for s in seeds_l}
        group = leaky[seeds_l[0]]["split_group"]
        for s, lf in leaky.items():
            if not lf["label_uid"].equals(prod["label_uid"]) or not lf["split_group"].equals(group):
                raise SplitLeakError(f"{t}: the seed-{s} leaky split is not aligned with the production split")
        rows = prod.assign(group=group.to_numpy())
        td = TypeData(t, rows)
        for s in seeds_l:
            path = exposure_path(runs, t, s)
            if not os.path.isfile(path):
                raise SplitLeakError(f"{t}: no {path}; run check first")
            expo = read_str_csv(path)
            if not expo["label_uid"].equals(prod["label_uid"]):
                raise SplitLeakError(f"{path} is not aligned with the production split")
            expo["k"] = expo["k"].astype(int)
            p = _read_predictions(str(run_dir(runs, t, "L", s) / "predictions.csv"), prod, leaky[s]["split"],
                                  f"{t} L seed {s}")
            td.add_l(s, leaky[s]["split"].to_numpy(), expo, p)
        for s in seeds_g:
            path = g_prediction_path(data, runs, t, s, g_source)
            td.add_g(s, _read_predictions(path, prod, prod["split"], f"{t} G seed {s}"))
        studies.append(td)
        extras[t] = {"prod": prod, "leaky": leaky}
    return studies, extras


def _training_record(data, runs, t, arm, seed, g_source):
    if arm == "L" or g_source == "fresh":
        path = run_dir(runs, t, arm, seed) / "training_run.json"
    elif seed == GROUPED_SEED:
        path = os.path.join(data, t, "dinov2", "training_run.json")
    else:
        return None
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def secondary(td, data, runs, g_source):
    """Per type, arm and seed: AUC on test, val and test_heldout; macro-F1 and balanced accuracy at 0.5 on the arm's
    own test; best-epoch val macro-F1 from the training record; the threshold reaching precision
    ``PRECISION_TARGET`` for Agree (correct) on the arm's own val, and its realised precision and coverage on
    test_heldout."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from evaluate import recall_at_precision
    y = (td.rows["label_class"] == "incorrect").to_numpy()
    out = []
    for arm, seeds, preds, splits in (("L", sorted(td.l_pred), td.l_pred, td.l_split),
                                      ("G", sorted(td.g_pred), td.g_pred,
                                       {s: td.rows["split"].to_numpy() for s in td.g_pred})):
        for s in seeds:
            p, sp = preds[s], splits[s]
            row = {"type": td.name, "arm": arm, "seed": s}
            for name, split in (("test", "test"), ("val", "val"), ("heldout", split_data.HELDOUT)):
                m = sp == split
                row[f"auc_{name}"] = float(weighted_auc(np.ones((1, m.sum())), y[m], p[m])[0]) if m.any() else None
            m = sp == "test"
            pred = p[m] >= 0.5
            row["macro_f1_test"] = float(f1_score(y[m], pred, average="macro"))
            row["balanced_accuracy_test"] = float(balanced_accuracy_score(y[m], pred))
            rec = _training_record(data, runs, td.name, arm, s, g_source)
            row["best_val_macro_f1"] = (rec or {}).get("best_metric_value") \
                if (rec or {}).get("best_metric", "macro_f1") == "macro_f1" else None
            v = sp == "val"
            _, thr = recall_at_precision((~y[v]).astype(int), 1 - p[v], PRECISION_TARGET)
            row["agree_threshold"] = thr
            h = sp == split_data.HELDOUT
            if thr is not None and h.any():
                agree = (1 - p[h]) >= thr
                row["heldout_agree_precision"] = float((~y[h][agree]).mean()) if agree.any() else None
                row["heldout_agree_coverage"] = float(agree[~y[h]].mean()) if (~y[h]).any() else None
            else:
                row["heldout_agree_precision"] = row["heldout_agree_coverage"] = None
            out.append(row)
    return pd.DataFrame(out)


def slice_section(td, extra, min_n=slices.MIN_N, n_boot=slices.N_BOOT):
    """Type x severity and type x tag for each arm on its own test rows (the first seed of each arm) with that arm's
    training counts, and the paired deltas (L - G) on the identical test_heldout rows."""
    prod = extra["prod"]
    cols = ["label_uid", "pano_id", "city", "severity", "tags"]
    y_text = prod["label_class"].to_numpy()

    def frame(p, split, mask):
        d = prod.loc[mask, cols].copy()
        d["true"] = y_text[mask]
        d["p_incorrect"] = p[mask]
        d["split"] = split
        return d

    s_l, s_g = sorted(td.l_pred)[0], sorted(td.g_pred)[0]
    sp_l, sp_g = td.l_split[s_l], prod["split"].to_numpy()
    parts = [f"## {td.name}", ""]
    for arm, seed, p, sp in (("label-level (L)", s_l, td.l_pred[s_l], sp_l), ("grouped (G)", s_g, td.g_pred[s_g], sp_g)):
        test = frame(p, "test", sp == "test")
        train = prod.loc[sp == "train", cols]
        for by in slices.BY:
            parts.append(slices.slice_markdown(slices.slice_table(test, by, train, min_n, n_boot),
                                               by, f"{td.name}, {arm} seed {seed}, own test, by {by}"))
    ho = sp_g == split_data.HELDOUT
    a, b = frame(td.l_pred[s_l], "test_heldout", ho), frame(td.g_pred[s_g], "test_heldout", ho)
    for by in slices.BY:
        parts.append(slices.delta_markdown(slices.paired_delta_table(a, b, by, min_n, n_boot), by,
                                           f"{td.name}, test_heldout (identical rows), L - G, by {by}",
                                           a="L", b="G"))
    return "\n".join(parts)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict("records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _ci(d, fmt="{:+.4f}"):
    if d is None or d.get("point") is None:
        return ""
    s = fmt.format(d["point"])
    if d.get("lo") is not None:
        s += f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]"
    if d.get("t_lo") is not None:
        s += f"; t [{fmt.format(d['t_lo'])}, {fmt.format(d['t_hi'])}]"
    return s


def tables_markdown(est, rule, g_source, min_unexposed=MIN_UNEXPOSED):
    seeds = ", ".join(map(str, est["seeds_l"]))
    lines = ["# Split leak (issue #3): results", "",
             f"Label-level arm seeds: {seeds}. Grouped arm: {g_source} runs, seeds "
             f"{', '.join(map(str, est['seeds_g']))}. Intervals: 95% percentile bootstrap over whole merged groups "
             f"and seeds; after the semicolon, the t interval on seed-level values (the sensitivity check).", "",
             "## Pre-registered reading", "", f"- Exposure contrast: **{rule['contrast']}**",
             f"- Inflation: **{rule['inflation']}**", "",
             "## Exposure contrast (test rows of the label-level models)", "",
             "| Metric | exposed - unexposed |", "|---|---:|",
             f"| error at 0.5 (negative = leak) | {_ci(est['contrast']['error'])} |",
             f"| mean true-class probability (positive = leak) | {_ci(est['contrast']['p_true'])} |", "",
             f"Strata (k = 1, 2, 3+ direct companions; a stratum with fewer than {min_unexposed} unexposed rows is "
             f"blank and left out of the pooled number):", "",
             "| Type | seed | k | exposed | unexposed | error, exposed | error, unexposed | |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in est["strata"].itertuples():
        k = "3+" if r.k == K_STRATA[-1] else str(r.k)
        lines.append(f"| {r.type} | {r.seed} | {k} | {r.n_exp:,} | {r.n_unexp:,} | {_f(r.err_exp, '{:.3f}')} | "
                     f"{_f(r.err_unexp, '{:.3f}')} | {'blank' if r.blank else ''} |")
    lines += ["", "Per type (descriptive):", "", "| Type | error | true-class probability |", "|---|---:|---:|"]
    for t, d in est["contrast_by_type"].items():
        lines.append(f"| {t} | {_ci(d['error'])} | {_ci(d['p_true'])} |")
    lines += ["", "Descriptive extensions (each against the same unexposed rows):", "",
              "| Rows | error | true-class probability |", "|---|---:|---:|"]
    labels = {"via_pano": "exposed through the same panorama", "via_nearby_only": "exposed only by a nearby label",
              "companion_same": "a train companion of the same class",
              "companion_opposite_only": "train companions of the opposite class only",
              "chain_only": "chain-only (train companion further along the merged group)",
              "val_exposed": "val rows (checkpoint selection used val)"}
    for name, d in est["contrast_descriptive"].items():
        lines.append(f"| {labels[name]} | {_ci(d['error'])} | {_ci(d['p_true'])} |")
    lines += ["", "## Inflation: AUC(L on its test) - AUC(G on its test)", "",
              "The multi-member part of test is a different row set in each arm, so the comparison is unpaired "
              "where it matters.", "", "| Type | test | val |", "|---|---:|---:|"]
    for t in est["inflation"]["test"]["by_type"]:
        lines.append(f"| {t} | {_ci(est['inflation']['test']['by_type'][t])} | "
                     f"{_ci(est['inflation']['val']['by_type'][t])} |")
    lines.append(f"| pooled (macro) | {_ci(est['inflation']['test']['pooled'])} | "
                 f"{_ci(est['inflation']['val']['pooled'])} |")
    lines += ["", "| Type | Δ AUC on the multi-member part | its share of test | product |", "|---|---:|---:|---:|"]
    for t, d in est["multi_member"].items():
        lines.append(f"| {t} | {d['delta_auc']:+.4f} | {d['share_of_test']:.3f} | {d['product']:+.4f} |")
    lines += ["", "## Secondary", "", "| Quantity | pooled (macro over types) |", "|---|---:|",
              f"| held-out Δ, L - G on identical test_heldout rows | {_ci(est['heldout_delta']['pooled'])} |",
              f"| singleton test rows, L - G (identical rows) | {_ci(est['singleton_delta']['pooled'])} |",
              f"| gap of gaps, (L test - L held-out) - (G test - G held-out) | {_ci(est['gap_of_gaps']['pooled'])} |",
              ""]
    return "\n".join(lines)


def seeds_markdown(sec):
    lines = ["# Split leak (issue #3): per seed", "",
             f"AUC on each arm's own rows; macro-F1 and balanced accuracy at 0.5 on its own test; best-epoch val "
             f"macro-F1 from the training record; the Agree threshold reaching precision {PRECISION_TARGET} on its "
             f"own val, with the precision and coverage it realises on test_heldout.", "",
             "| Type | arm | seed | AUC test | AUC val | AUC held-out | macro-F1 | balanced acc. | best val F1 | "
             "threshold | held-out precision | held-out coverage |", "|---|---|---:|" + "---:|" * 9]
    for r in sec.itertuples():
        lines.append(f"| {r.type} | {r.arm} | {r.seed} | {_f(r.auc_test)} | {_f(r.auc_val)} | {_f(r.auc_heldout)} | "
                     f"{_f(r.macro_f1_test)} | {_f(r.balanced_accuracy_test)} | {_f(r.best_val_macro_f1)} | "
                     f"{_f(r.agree_threshold, '{:.6f}')} | {_f(r.heldout_agree_precision)} | "
                     f"{_f(r.heldout_agree_coverage)} |")
    return "\n".join(lines) + "\n"


def report(data, runs, types, seeds_l, seeds_g, g_source, out, n_boot=N_BOOT, boot_seed=BOOT_SEED,
           slice_boot=slices.N_BOOT):
    """Write ``tables.md``, ``seeds.md``, ``slices.md`` and ``results.json`` to ``out`` (nothing names a
    deployment)."""
    studies, extras = load_study(data, runs, types, seeds_l, seeds_g, g_source)
    est = estimate(studies, n_boot=n_boot, boot_seed=boot_seed)
    rule = rule_outcome(est["contrast"]["error"], est["inflation"]["test"]["pooled"], n_seeds=len(seeds_l),
                        g_source=g_source)
    sec = pd.concat([secondary(td, data, runs, g_source) for td in studies], ignore_index=True)
    hashes = {}
    for t in types:
        hashes[t] = {"production_split": sha256_file(os.path.join(data, t, "split.csv")),
                     **{f"L-s{s}": sha256_file(str(run_dir(runs, t, "L", s) / "predictions.csv")) for s in seeds_l},
                     **{f"G-s{s}": sha256_file(g_prediction_path(data, runs, t, s, g_source)) for s in seeds_g}}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(tables_markdown(est, rule, g_source))
    with open(os.path.join(out, "seeds.md"), "w") as f:
        f.write(seeds_markdown(sec))
    with open(os.path.join(out, "slices.md"), "w") as f:
        f.write("# Split leak (issue #3): by severity and tag\n\n" + "\n".join(
            slice_section(td, extras[td.name], n_boot=slice_boot) for td in studies))
    results = {"rule": rule, "g_source": g_source, "constants": constants(), "commit": git_commit(),
               "inputs_sha256": hashes, "secondary": sec, "n_boot": n_boot, "boot_seed": boot_seed,
               **{k: v for k, v in est.items()}}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")
    return results


# ---------------------------------------------------------------------------------------------------------------------
# predict (GPU)
# ---------------------------------------------------------------------------------------------------------------------

def predict(data, label_type, split, run, out):
    """Score an arm's own val, test and test_heldout rows (``evaluate.run_inference``) with the best checkpoint in
    ``run``; write ``out`` atomically. Crops resolve under ``<data>/<label_type>/``."""
    import evaluate
    df = pd.read_csv(split, dtype={"pano_id": str})
    base = os.path.join(data, label_type)
    df["crop_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df["crop_path"]]
    pred = evaluate.run_inference(df, evaluate.best_checkpoint_in(run), splits=EVAL_SPLITS)
    tmp = f"{out}.tmp"
    pred.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return pred


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, seeds=True):
        q.add_argument("--data", required=True, help="round 3's data directory (<data>/<T>/split.csv)")
        q.add_argument("--runs", required=True, help="run directory on scratch (never under --data)")
        q.add_argument("--types", nargs="+", default=list(TYPES))
        if seeds:
            q.add_argument("--seeds", nargs="+", type=int, default=[42])

    q = sub.add_parser("make-split", help="recompute, assert equality with production, scatter (CPU)")
    common(q)
    q.add_argument("--rawlabels", required=True, help="the rawLabels cache the production split was made with")
    q.add_argument("--holdout-cities", default=None,
                   help="comma list; default: the deployments wholly in test_heldout (never written to a file)")
    q.add_argument("--hard-examples", default=None, help="default: the production split's hard column")
    q = sub.add_parser("check", help="coupling, exposure census, design.json (CPU)")
    common(q)
    q.add_argument("--rawlabels", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("bound", help="Stage 0's planning figures (CPU)")
    common(q, seeds=False)
    q.add_argument("--rawlabels", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--g-seeds", nargs="+", type=int, default=[42, 43, 44])
    q.add_argument("--n-boot", type=int, default=N_BOOT)
    q = sub.add_parser("predict", help="score an arm's eval rows (GPU)")
    q.add_argument("--data", required=True)
    q.add_argument("--label_type", required=True)
    q.add_argument("--split", required=True)
    q.add_argument("--run-dir", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("report", help="tables.md, seeds.md, slices.md, results.json (CPU)")
    common(q)
    q.add_argument("--g-seeds", nargs="+", type=int, default=[42, 43, 44])
    q.add_argument("--g-source", choices=["stored", "fresh"], default="stored")
    q.add_argument("--out", required=True)
    q.add_argument("--n-boot", type=int, default=N_BOOT)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "make-split":
            holdout = None if args.holdout_cities is None else \
                [c.strip() for c in args.holdout_cities.split(",") if c.strip()]
            if _inside(args.runs, args.data):
                raise SplitLeakError(f"--runs {args.runs} is under the data directory {args.data}")
            for t in args.types:
                for s in args.seeds:
                    r = make_split(os.path.join(args.data, t, "manifest.csv"), os.path.join(args.data, t, "split.csv"),
                                   args.rawlabels, leaky_split_path(args.runs, t, s), s, holdout, args.hard_examples)
                    print(f"{t} seed {s}: {json.dumps(r)}")
        elif args.cmd == "check":
            design = check(args.data, args.runs, args.types, args.seeds, args.rawlabels, args.out)
            for t, rec in design["types"].items():
                print(t, json.dumps(rec["census"]))
        elif args.cmd == "bound":
            res = bound(args.data, args.runs, args.types, args.rawlabels, args.out, args.g_seeds, args.n_boot)
            print(bound_markdown(res, args.types))
        elif args.cmd == "predict":
            pred = predict(args.data, args.label_type, args.split, args.run_dir, args.out)
            print(f"wrote {args.out} ({len(pred)} rows)")
        elif args.cmd == "report":
            res = report(args.data, args.runs, args.types, args.seeds, args.g_seeds, args.g_source, args.out,
                         n_boot=args.n_boot)
            print(json.dumps(res["rule"], indent=2))
    except SplitLeakError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
