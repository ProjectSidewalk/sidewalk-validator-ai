"""The crop-constant sweep (issue #21): re-cut round 3's labels at 4000, 6100 and 8000, train each arm, and read the
pre-registered rule. Design, gates and rule: ``docs/crop-constant-2026-11.md``.

Every crop is ``int(C / depth_m)`` px on each side of the label, with ``C = 6100`` inherited from the first version.
This module tests whether another constant validates better. One re-cut pass renders each label's view once (round
3's render: ``read_store_panorama``, ``crop_pose``'s pose, the torch renderer) and cuts three crops from it:

* ``c4000`` and ``c8000``: ``data_generation.depth_sized_crop(view, depth_m, constant=C)`` from the **recorded**
  ``depth_m`` (no depth model runs, and ``depth_m`` is never rewritten);
* ``c6100``: ``data_generation.centre_crop`` at the **recorded** ``crop_half_px`` (17 round-3 rows differ from
  ``int(6100 / depth_m)`` by 1 px), trained again beside the arms as the baseline.

==============  ======  ===========================================================================================
subcommand      where   what
==============  ======  ===========================================================================================
``stage-list``  Mac     the deduplicated panorama list for the chosen types (``analysis/ai_labels_stage.slurm``)
``chunks``      login   how many array tasks ``recrop`` needs so that a chunk fits ``--time`` at 3.5 s per label
``preflight``   CPU     G0: round 3's pins, the panorama-tools commit, the environment's versions; the store
``recrop``      GPU     one chunk of panoramas: one read and one render per label, three crops; resumable
``gates``       CPU     G2 (row loss) and G3 (re-cut parity); writes the three arm directories when both pass
``compare``     Mac     the family check, the paired seed-paired bootstrap, the rule read-out, reported slices
``report``      Mac     ``tables.md``, the figure and ``provenance.json``; non-public deployments are not named
==============  ======  ===========================================================================================

Nothing here edits a hot file: ``data_generation``, ``crop_pose`` and ``analysis/depth_agreement`` are imported.

Example (the doc's "Reproduce" section)::

    R3=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; D=~/Data/sidewalk-validator-ai-crop-constant-2026-11
    python analysis/crop_constant_sweep.py stage-list --root $R3 --types CurbRamp,SurfaceProblem \\
        --out $D/stage_files.txt
    # klone: analysis/crop_constant_submit.sh (stage -> preflight -> recrop -> gates -> trainings)
    python analysis/crop_constant_sweep.py compare --work $D/w --types CurbRamp,SurfaceProblem \\
        --seeds 42,43,44,45,46 --control-root $R3 --out $D/compare.json
    python analysis/crop_constant_sweep.py report --compare $D/compare.json --out-dir docs/crop-constant-2026-11
"""
import argparse
import copy
import csv
import datetime
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import crop_pose  # noqa: E402  (imported, never edited: a hot file)
import data_generation as dg  # noqa: E402  (imported, never edited: a hot file)
from analysis import depth_agreement as da  # noqa: E402
from analysis import slices as sl  # noqa: E402

BASELINE = 6100                             # the served constant; a test pins it to data_generation's default
ARMS = (4000, BASELINE, 8000)
SWEPT = (4000, 8000)
STAGE1_TYPES = ("CurbRamp", "SurfaceProblem")
STAGE2_TYPES = ("NoCurbRamp", "Obstacle", "Crosswalk")
ALL_TYPES = STAGE1_TYPES + STAGE2_TYPES
REGISTERED_SEEDS = (42, 43, 44, 45, 46)     # PROVISIONAL (decision 2 for Jon): five seeds recommended, three possible.
# The registration itself: the submit script trains exactly these (``registered-seeds``) and ``compare`` reads exactly
# these, so changing the seed count is a commit, made before the first training job. There is no CLI override.
M_ARMS = 2                                  # m in the level 1 - 0.05 / m, fixed at 2 for every type and stage
ALPHA = 0.05
N_BOOT = 2000
SEED = 20261123
MAX_TRIES = 20                              # a resample with one class is redrawn, as slices.boot_indices does
G2_MAX_LOSS = Fraction(5, 1000)             # at most 0.5 % of a type's rows lost
G2_MAX_SPLIT_LOSS = Fraction(1, 100)        # and at most 1 % of its test or test_heldout rows
G2_SPLITS = ("test", "test_heldout")
G3_MAX_MAD = 3.0                            # per-row mean absolute difference, grey levels, after the q95 round trip
G3_MIN_SHARE = Fraction(995, 1000)          # on at least 99.5 % of rows
G3_HARD_MAX = 12.0                          # and no row above 12
MIN_CLASS = 10                              # fewer test rows of a class: unestimable
MIN_CITY_N = 10                             # the within-city AUC's per-city minimum (retrain_report.within_city_auc)
RECALL_PRECISION = 0.92
SLOW_SECONDS_PER_LABEL = 3.5                # the slow end of the measured re-cut rate (seg-prior job, decode-bound)
TIME_MARGIN = 1.5
GSV_BETA = dg.DEFAULT_TILT_BETA
WORK_NAME = "crop-constant-2026-11"
POOLED_NAME = "non-public deployment {}"

RECROP_FINAL = ("ok", "no_pano", "bad_pano", "native_mismatch", crop_pose.OVER_TILT_STATUS,
                crop_pose.PAST_POLE_STATUS)
"""Statuses a resumed ``recrop`` does not retry; ``error:*`` rows are retried and the later row wins."""
RECROP_FIELDS = (["label_uid", "label_type", "status", "renderer", "tilt_applied", "store_root", "native_match",
                  "half_recorded", "half_6100_recut"]
                 + [f"{k}_{c}" for c in ARMS for k in ("half", "clipped")] + ["parity_mad"])
VERSION_MODULES = (("torch", "torch"), ("torchvision", "torchvision"), ("transformers", "transformers"),
                   ("opencv", "cv2"), ("pillow", "PIL"), ("numpy", "numpy"))
REQUIRED_VERSIONS = ("torch", "torchvision", "transformers", "opencv", "pillow")

load_frame = da.load_frame
sha256_file = da.sha256_file
write_json = da.write_json


def arm_name(constant):
    """The arm directory name of a constant.

    >>> arm_name(4000)
    'c4000'
    """
    return f"c{int(constant)}"


def arm_dir(work, constant):
    """``<work>/c<constant>``: the arm's data root (``<type>/split.csv``, crops, records), the ``DATA`` of a training."""
    return os.path.join(work, arm_name(constant))


def split_types(text):
    """Label types from ``a,b``, ``a:b`` or ``a b`` (the job files pass ``:``, since ``sbatch --export`` splits on
    commas).

    >>> split_types("CurbRamp:SurfaceProblem"), split_types("CurbRamp, Obstacle")
    (('CurbRamp', 'SurfaceProblem'), ('CurbRamp', 'Obstacle'))
    """
    return tuple(t for t in re.split(r"[,:\s]+", text.strip()) if t)


def split_ints(text):
    """``"42,43"`` -> ``(42, 43)``."""
    return tuple(int(s) for s in split_types(text))


# ---------------------------------------------------------------------------------------------------------------------
# Chunks and the stage list
# ---------------------------------------------------------------------------------------------------------------------

def parse_time(text):
    """Seconds in a Slurm time (``M``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``).

    >>> parse_time("8:00:00"), parse_time("90"), parse_time("2-00")
    (28800, 5400, 172800)
    """
    days, _, rest = text.rpartition("-") if "-" in text else ("0", "", text)
    parts = [int(p) for p in rest.split(":")]
    if "-" in text:
        h, m, s = (parts + [0, 0])[:3]
    elif len(parts) == 1:
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    else:
        h, m, s = parts
    return int(days) * 86400 + h * 3600 + m * 60 + s


def max_labels_for(time_limit):
    """The most labels one ``recrop`` chunk may hold so that labels x 3.5 s x 1.5 fits ``time_limit``.

    >>> max_labels_for("8:00:00")
    5485
    """
    return int(parse_time(time_limit) // (SLOW_SECONDS_PER_LABEL * TIME_MARGIN))


def chunk_plan(frame, max_labels):
    """The array task of every row of ``frame`` (columns ``city``, ``pano_id``), as a Series on ``frame``'s index.

    Panoramas sorted by ``(city, pano_id)`` fill chunks in order, and a panorama is never split, so each is read
    once. Depends only on the set of rows, so ``chunks`` (split files) and ``recrop`` (split joined to rawLabels)
    agree. Raises ValueError when one panorama holds more than ``max_labels`` labels.
    """
    counts = frame.groupby(["city", "pano_id"], sort=True).size()
    if len(counts) and counts.max() > max_labels:
        raise ValueError(f"a panorama holds {counts.max()} labels, more labels than the chunk cap {max_labels}; "
                         f"raise --time")
    ids, cur, n = [], 0, 0
    for c in counts.to_numpy():
        if n and n + c > max_labels:
            cur, n = cur + 1, 0
        ids.append(cur)
        n += c
    per = dict(zip(counts.index, ids))
    return pd.Series([per[k] for k in zip(frame.city, frame.pano_id)], index=frame.index, name="chunk")


def stage_files(frame):
    """The sorted, deduplicated store-relative panorama paths ``<city>/<pano_id[:2]>/<pano_id>.jpg`` of ``frame``."""
    return sorted({f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(frame.city, frame.pano_id)})


def present_in_store(files, store):
    """The members of ``files`` already present under ``store`` (stage 2 against the stage-1 store)."""
    return {f for f in files if os.path.exists(os.path.join(store, f))}


def read_splits(root, types):
    """Round 3's ``split.csv`` rows of ``types`` (ids as strings), concatenated."""
    return pd.concat([pd.read_csv(os.path.join(root, t, "split.csv"), dtype=da.SPLIT_DTYPES, low_memory=False)
                      for t in types], ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
# preflight (G0)
# ---------------------------------------------------------------------------------------------------------------------

def environment_versions():
    """The versions of the packages a crop or a training depends on (None when a package is absent)."""
    out = {}
    for name, mod in VERSION_MODULES:
        try:
            out[name] = __import__(mod).__version__
        except Exception:
            out[name] = None
    out["sidewalk-panorama-tools"] = dg.package_source()
    return out


def round3_package_sources(root, types):
    """``{type: package_source}`` from round 3's ``generation_run.json`` (``settings`` first, then top level)."""
    out = {}
    for t in types:
        with open(os.path.join(root, t, "generation_run.json")) as f:
            run = json.load(f)
        out[t] = (run.get("settings") or {}).get("package_source") or run.get("package_source")
    return out


def preflight(root, rawlabels_dir, types, pano_store, package=dg.package_source, versions=None,
              expected_pins=da.ROUND3_PINS):
    """G0 and the store check, as a record (``preflight.json``). ``g0.passed`` needs: every input pin of
    ``expected_pins`` matching, round 3's ``package_source`` equal to this environment's, and the versions of
    :data:`REQUIRED_VERSIONS` recorded. The store must carry its marker, and the panoramas missing from it must not
    break G2 on their own (a missing panorama loses its rows; G2 itself is read after the re-cut)."""
    failures = []
    pins, _ = da.input_pins(root, rawlabels_dir, types)
    bad = da.check_pins(pins, expected_pins)
    if bad:
        failures.append(f"input pins differ from round 3's: {', '.join(bad)}")
    r3 = round3_package_sources(root, types)
    now = package()
    if any(v != now for v in r3.values()):
        failures.append(f"sidewalk-panorama-tools differs from round 3's: {now} vs {sorted(set(r3.values()))}")
    vers = environment_versions() if versions is None else versions
    absent = [k for k in REQUIRED_VERSIONS if not vers.get(k)]
    if absent:
        failures.append(f"versions not recorded: {', '.join(absent)}")
    store_failures = []
    marker = None
    try:
        marker = dg.check_pano_store_marker(pano_store)
    except SystemExit as e:
        store_failures.append(str(e))
    frame = read_splits(root, types)
    files = {(c, p): os.path.exists(dg.pano_store_path(pano_store, c, p))
             for c, p in frame[["city", "pano_id"]].drop_duplicates().itertuples(index=False)}
    status = pd.Series(["ok" if files[(c, p)] else "no_pano" for c, p in zip(frame.city, frame.pano_id)])
    g2 = gate_g2(frame, frame[["label_uid"]].assign(status=status.to_numpy()))
    if not g2["passed"]:
        store_failures.append("the panoramas missing from the store already break G2")
    commit, dirty = da.git_commit()
    return {"stage": "preflight", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
            "types": list(types), "g0": {"passed": not failures, "failures": failures},
            "store": {"passed": not store_failures, "failures": store_failures, "pano_store": pano_store,
                      "marker": marker, "panoramas": len(files), "missing_panoramas": int(sum(not v for v in
                                                                                             files.values())),
                      "missing_rows": g2["by_type"]},
            "pins": pins, "package_source": {"now": now, "round3": r3}, "versions": vers}


# ---------------------------------------------------------------------------------------------------------------------
# recrop (GPU)
# ---------------------------------------------------------------------------------------------------------------------

def chunk_path(work, chunk):
    return os.path.join(work, "recrop", f"chunk_{int(chunk):04d}.csv")


def _final_uids(path):
    """The label_uids ``path`` already holds with a final status (:data:`RECROP_FINAL`); a cut-off last line (a
    preempted write) is removed first."""
    if not os.path.exists(path) or not os.path.getsize(path):
        return set()
    if da._drop_partial_last_line(path):
        print(f"recrop: removed a cut-off last line from {path}")
    if not os.path.getsize(path):
        return set()
    prev = pd.read_csv(path, dtype={"label_uid": str, "status": str})
    return set(prev.label_uid[prev.status.isin(RECROP_FINAL)])


def parity_mad(new_path, stored_path):
    """Mean absolute difference (grey levels, all channels) between two decoded WebP crops; ``inf`` when the stored
    crop is missing or the sizes differ, so an unjudged row can never pass G3."""
    from PIL import Image
    if not os.path.exists(stored_path):
        return float("inf")
    with Image.open(new_path) as a, Image.open(stored_path) as b:
        x, y = np.asarray(a.convert("RGB"), np.int16), np.asarray(b.convert("RGB"), np.int16)
    if x.shape != y.shape:
        return float("inf")
    return round(float(np.abs(x - y).mean()), 4)


def _safe_store_root(row):
    try:
        return da.store_root(row["pano_path"], row["city"], row["pano_id"])
    except (ValueError, TypeError):
        return None


def _native_match(info, row):
    try:
        return int(int(info.get("pano_native_w")) == int(row["pano_native_w"])
                   and int(info.get("pano_native_h")) == int(row["pano_native_h"]))
    except (TypeError, ValueError):
        return 0


def recrop(frame, work, chunk, max_labels, pano_store=None, stored_root=None, device=None, gsv_beta=GSV_BETA,
           pano_reader=dg.read_store_panorama, render=dg.render_label_view, limit=None):
    """Re-cut one chunk (:func:`chunk_plan`) of ``frame`` (:func:`load_frame`): append-only and resumable.

    Per panorama, one ``pano_reader(store, city, pano_id)`` (``read_store_panorama``; ``store`` is the row's own
    recorded store root unless ``pano_store`` overrides it); a native size other than the row's ``pano_native_w/h``
    is ``native_mismatch`` (lost). Per label, the pose of ``crop_pose.pose_for_row(row, gsv_beta)``, the centre row
    of ``crop_pose.rig_y_norm`` (a Panoramax skip is decided first, as data_generation does), and one ``render``;
    then the three crops into ``<work>/c<C>/<type>/<crop_path>``, and the c6100 crop's mean absolute difference from
    round 3's stored crop at ``<stored_root>/<type>/<crop_path>`` after the same q95 WebP round trip. Returns
    ``{"processed", "seconds", "statuses"}``.
    """
    plan = chunk_plan(frame, max_labels)
    rows = frame[plan == chunk]
    out = chunk_path(work, chunk)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _final_uids(out)
    todo = rows[~rows.label_uid.isin(done)].sort_values(["city", "pano_id"], kind="stable")
    if limit:
        todo = todo.head(limit)
    statuses = Counter()
    started = time.time()
    if not len(todo):
        return {"processed": 0, "seconds": 0.0, "statuses": {}}
    renderer = "torch" if device is not None else "numpy"
    new = not os.path.exists(out) or not os.path.getsize(out)
    last = (None, None, None)
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=RECROP_FIELDS)
        if new:
            wr.writeheader()
        for r in todo.to_dict("records"):
            rec = {"label_uid": r["label_uid"], "label_type": r["label_type"], "renderer": renderer,
                   "store_root": _safe_store_root(r), "half_recorded": int(r["crop_half_px"])}
            try:
                pose = crop_pose.pose_for_row(r, gsv_beta)
                y_norm = crop_pose.rig_y_norm(r["pano_y"] / r["pano_height"], pose)
                skip = crop_pose.skip_status(pose, y_norm)
                if skip is not None:
                    rec["status"] = skip
                else:
                    key = (r["city"], r["pano_id"])
                    if last[0] != key:
                        store = pano_store or rec["store_root"]
                        if store is None:
                            raise ValueError(f"no store root in pano_path {r['pano_path']!r}; pass --pano-store")
                        read, info = pano_reader(store, r["city"], r["pano_id"])
                        rgb = None if isinstance(read, str) else np.ascontiguousarray(read[..., ::-1])
                        last = (key, read if isinstance(read, str) else rgb, info)
                    pano, info = last[1], last[2]
                    if isinstance(pano, str):
                        rec["status"] = pano
                    elif not _native_match(info, r):
                        rec.update(status="native_mismatch", native_match=0)
                    else:
                        rec["native_match"] = 1
                        view, tilt = render(pano, r["pano_x"] / r["pano_width"], y_norm, pose.camera_pitch,
                                            pose.camera_roll, apply_tilt=pose.beta is not None,
                                            beta=pose.beta or 0.0, device=device)
                        rec["tilt_applied"] = int(tilt)
                        depth = float(r["depth_m"])
                        for c in ARMS:
                            if c == BASELINE:
                                crop, half, clipped = dg.centre_crop(view, int(r["crop_half_px"]))
                            else:
                                crop, half, clipped = dg.depth_sized_crop(view, depth, constant=c)
                            dg.save_webp(crop, os.path.join(arm_dir(work, c), r["label_type"], r["crop_path"]))
                            rec[f"half_{c}"], rec[f"clipped_{c}"] = int(half), int(clipped)
                        rec["half_6100_recut"] = int(BASELINE / depth)
                        rec["parity_mad"] = parity_mad(
                            os.path.join(arm_dir(work, BASELINE), r["label_type"], r["crop_path"]),
                            os.path.join(stored_root, r["label_type"], r["crop_path"])) if stored_root else float(
                            "inf")
                        rec["status"] = "ok"
            except Exception as e:
                if dg.is_fatal(e):
                    raise
                rec["status"] = f"error:{type(e).__name__}"
                print(f"recrop {r['label_uid']}: {type(e).__name__}: {e}")
            wr.writerow(rec)
            f.flush()
            statuses[rec["status"]] += 1
    return {"processed": int(sum(statuses.values())), "seconds": time.time() - started, "statuses": dict(statuses)}


def load_recrop(work):
    """Every chunk's rows, one per label: an ``ok`` row wins over any other, then the later row (a retried label;
    files read in name order, so a stale ``error:*`` row in a later chunk file never beats an ``ok`` one)."""
    paths = sorted(glob.glob(os.path.join(work, "recrop", "chunk_*.csv")))
    if not paths:
        return pd.DataFrame(columns=RECROP_FIELDS)
    parts = []
    for p in paths:
        _final_uids(p)                      # drops a cut-off last line
        if os.path.getsize(p):
            parts.append(pd.read_csv(p, dtype={"label_uid": str, "status": str, "store_root": str,
                                               "label_type": str, "renderer": str}))
    rec = pd.concat(parts, ignore_index=True)
    rec = rec.assign(_ok=rec.status.eq("ok"), _order=np.arange(len(rec)))
    rec = rec.sort_values(["_ok", "_order"], kind="stable").drop_duplicates("label_uid", keep="last")
    return rec.sort_values("_order").drop(columns=["_ok", "_order"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# Gates and the arm directories
# ---------------------------------------------------------------------------------------------------------------------

def _share(num, den):
    return float(num) / den if den else None


def gate_g2(frame, rec):
    """G2, row loss: a row whose re-cut status is not ``ok`` (or that was never attempted) is lost from every arm.
    Passes when, for every type, at most 0.5 % of its rows and at most 1 % of its ``test`` and ``test_heldout`` rows
    are lost (exact fractions, so the bounds themselves pass), and every row was attempted."""
    m = frame[["label_uid", "label_type", "split"]].merge(rec[["label_uid", "status"]], on="label_uid", how="left",
                                                          validate="one_to_one")
    lost = m.status.ne("ok")
    by_type = {}
    for t, g in m.groupby("label_type"):
        n, n_lost = len(g), int(lost[g.index].sum())
        row = {"n": n, "lost": n_lost, "share": _share(n_lost, n),
               "lost_by_status": {str(k): int(v) for k, v in g.status.fillna("not_attempted")[lost[g.index]]
                                  .value_counts().items()},
               "passed": Fraction(n_lost, n) <= G2_MAX_LOSS}
        for s in G2_SPLITS:
            gs = g[g.split == s]
            if len(gs):
                ls = int(lost[gs.index].sum())
                row[s] = {"n": len(gs), "lost": ls, "share": _share(ls, len(gs)),
                          "passed": Fraction(ls, len(gs)) <= G2_MAX_SPLIT_LOSS}
                row["passed"] = row["passed"] and row[s]["passed"]
        by_type[t] = row
    not_attempted = int(m.status.isna().sum())
    return {"passed": not_attempted == 0 and bool(by_type) and all(r["passed"] for r in by_type.values()),
            "not_attempted": not_attempted, "by_type": by_type}


def _parity_stats(g):
    mad = pd.to_numeric(g.parity_mad, errors="coerce").astype(float)
    within = int((mad <= G3_MAX_MAD).sum())
    above = int((~(mad <= G3_HARD_MAX)).sum())       # NaN and inf are above: an unjudged row fails
    n = len(g)
    finite = mad[np.isfinite(mad)]
    return {"n": n, "within": within, "share_within": _share(within, n), "above_hard": above,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(mad.max()) if n else None,
            "passed": n > 0 and Fraction(within, n) >= G3_MIN_SHARE and above == 0}


def gate_g3(rec):
    """G3, re-cut parity on the ``ok`` rows: per type, the c6100 crop's mean absolute difference from round 3's stored
    crop is at most 3.0 on at least 99.5 % of rows and no row is above 12 (a missing stored crop or another size is
    ``inf``). Reported per source store as well; the rule is read per type."""
    ok = rec[rec.status == "ok"]
    by_type = {t: _parity_stats(g) for t, g in ok.groupby("label_type")}
    by_store = {str(s): _parity_stats(g) for s, g in ok.groupby(ok.store_root.fillna("(unknown)"))}
    return {"passed": bool(by_type) and all(v["passed"] for v in by_type.values()), "by_type": by_type,
            "by_store": by_store}


def arm_record(src_run, constant, source_split_sha256, recrop_info):
    """The arm's ``generation_run.json``: round 3's record (so ``tilt_beta`` and ``depth_encoder`` still reach
    ``training_run.json``) plus ``crop_depth_constant`` (top level and under ``settings``), ``derived_from`` (the
    source split's sha256) and the ``recrop`` provenance block.

    >>> r = arm_record({"tilt_beta": 0.95, "settings": {"depth_encoder": "vitb"}}, 4000, "ab", {"tool": "x"})
    >>> r["crop_depth_constant"], r["settings"]["crop_depth_constant"], r["derived_from"]["split_sha256"]
    (4000, 4000, 'ab')
    """
    out = copy.deepcopy(src_run)
    out["crop_depth_constant"] = int(constant)
    out.setdefault("settings", {})["crop_depth_constant"] = int(constant)
    out["derived_from"] = {"split_sha256": source_split_sha256, "what": "round 3's split.csv, generation_run.json"}
    out["recrop"] = recrop_info
    return out


def write_arms(work, root, frame, rec, types, recrop_info):
    """Write ``<work>/c<C>/<type>/{split,manifest}.csv`` and ``generation_run.json`` for every arm, plus each arm's
    ``deployment_priors.json``. Each split is round 3's, read as text: same rows (less every lost row, removed from
    every arm), same order, same ``split``, every column byte for byte except ``crop_half_px`` and ``clipped``.
    ``crop_path`` stays relative, so it resolves to the arm's own crops. Returns ``{arm: {type: rows}}``."""
    ok = rec[rec.status == "ok"]
    lost = set(frame.label_uid) - set(ok.label_uid)
    cols = ["label_uid"] + [f"{k}_{c}" for c in ARMS for k in ("half", "clipped")]
    counts = {arm_name(c): {} for c in ARMS}
    for t in types:
        src_path = os.path.join(root, t, "split.csv")
        src = pd.read_csv(src_path, dtype=str, keep_default_na=False)
        kept = src[~src.label_uid.isin(lost)].reset_index(drop=True)
        sizes = kept[["label_uid"]].merge(ok[cols], on="label_uid", how="left", validate="one_to_one")
        if sizes[cols[1:]].isna().any().any():
            raise ValueError(f"{t}: {int(sizes[cols[1]].isna().sum())} kept rows have no re-cut sizes")
        with open(os.path.join(root, t, "generation_run.json")) as f:
            src_run = json.load(f)
        src_sha = sha256_file(src_path)
        for c in ARMS:
            out = kept.copy()
            out["crop_half_px"] = sizes[f"half_{c}"].astype(int).astype(str).to_numpy()
            out["clipped"] = sizes[f"clipped_{c}"].astype(int).astype(str).to_numpy()
            d = os.path.join(arm_dir(work, c), t)
            os.makedirs(d, exist_ok=True)
            out.to_csv(os.path.join(d, "split.csv"), index=False)
            out.drop(columns=[x for x in ("split", "hard") if x in out.columns]).to_csv(
                os.path.join(d, "manifest.csv"), index=False)
            write_json(os.path.join(d, "generation_run.json"), arm_record(src_run, c, src_sha, recrop_info))
            counts[arm_name(c)][t] = len(out)
    priors = os.path.join(root, "deployment_priors.json")
    if os.path.exists(priors):
        import shutil
        for c in ARMS:
            shutil.copy(priors, os.path.join(arm_dir(work, c), "deployment_priors.json"))
    return counts


# ---------------------------------------------------------------------------------------------------------------------
# Statistics: weighted AUCs and the paired, seed-paired panorama bootstrap
# ---------------------------------------------------------------------------------------------------------------------

class RankedScores:
    """One score vector, sorted once, scored under any row weights: the Mann-Whitney AUC with ties counted half.
    Integer weights equal repeating rows, so a panorama resample is a weight vector (counts of each panorama).

    >>> r = RankedScores(np.array([0.1, 0.4, 0.35, 0.8]))
    >>> r.auc(np.array([False, False, True, True]), np.ones(4))
    0.75
    """

    def __init__(self, scores):
        s = np.asarray(scores, float)
        self.order = np.argsort(s, kind="mergesort")
        ss = s[self.order]
        self.group = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]) if len(ss) else np.zeros(0, int)
        self.n_groups = int(self.group[-1]) + 1 if len(ss) else 0

    def auc(self, y, w):
        y = np.asarray(y, bool)[self.order]
        w = np.asarray(w, float)[self.order]
        pos = np.bincount(self.group, weights=w * y, minlength=self.n_groups)
        neg = np.bincount(self.group, weights=w * ~y, minlength=self.n_groups)
        p, n = pos.sum(), neg.sum()
        if p <= 0 or n <= 0:
            return float("nan")
        below = np.cumsum(neg) - neg
        return float(np.dot(pos, below + 0.5 * neg) / (p * n))


class WithinCityScores:
    """The within-city AUC under row weights: per-deployment AUCs averaged with weights n_incorrect x n_correct, over
    deployments with at least ``min_n`` (weighted) rows and both classes; the definition of
    ``docs/retrain-2026-10.md``, which a score that only separates deployments cannot raise."""

    def __init__(self, scores, city, min_n=MIN_CITY_N):
        s, city = np.asarray(scores, float), np.asarray(city)
        self.min_n = min_n
        self.parts = [(idx, RankedScores(s[idx])) for idx in (np.flatnonzero(city == c) for c in np.unique(city))]

    def auc(self, y, w):
        y, w = np.asarray(y, bool), np.asarray(w, float)
        num = den = 0.0
        for idx, r in self.parts:
            yy, ww = y[idx], w[idx]
            if ww.sum() < self.min_n:
                continue
            p, n = float((ww * yy).sum()), float((ww * ~yy).sum())
            if p <= 0 or n <= 0:
                continue
            num += r.auc(yy, ww) * p * n
            den += p * n
        return num / den if den else float("nan")


def _scorer(stat, city):
    if stat == "pooled":
        return RankedScores
    return lambda s: WithinCityScores(s, city)


def replicates(y, arm, base, clusters, n_boot=N_BOOT, seed=SEED, stat="pooled", city=None):
    """``(D, draws)``: ``D[b, s]`` is seed pair ``s``'s ΔAUC (arm minus baseline) in replicate ``b``, and ``draws[b]``
    the ``k`` seed indices replicate ``b`` resamples with replacement (the hierarchical interval).

    Each replicate draws whole panoramas (``clusters``) with replacement **once**, and that one draw scores both
    models of every seed pair, so the arm and the baseline, and every seed, see the same rows. ``arm`` and ``base``
    are ``k x n`` score arrays (row ``s`` is seed ``s``), aligned with ``y`` and ``clusters``."""
    arm, base = np.atleast_2d(np.asarray(arm, float)), np.atleast_2d(np.asarray(base, float))
    y = np.asarray(y, bool)
    k = arm.shape[0]
    make = _scorer(stat, city)
    ra, rb = [make(a) for a in arm], [make(b) for b in base]
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    rng = np.random.default_rng(seed)
    D = np.empty((n_boot, k))
    draws = np.empty((n_boot, k), int)
    for b in range(n_boot):
        for _ in range(MAX_TRIES):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
            if (w * y).sum() > 0 and (w * ~y).sum() > 0:
                break
        D[b] = [ra[s].auc(y, w) - rb[s].auc(y, w) for s in range(k)]
        draws[b] = rng.integers(0, k, k)
    return D, draws


def paired_stats(y, arm, base, clusters, k, m=M_ARMS, n_boot=N_BOOT, seed=SEED, stat="pooled", city=None):
    """The registered statistic and both intervals for one type and arm: ``point`` is the mean over the ``k`` seed
    pairs of the paired ΔAUC (``seed_deltas`` lists them); ``reg_lo``/``reg_hi`` the registered panorama-only
    interval (every seed in every replicate) and ``hier_lo``/``hier_hi`` the hierarchical one (seed pairs resampled
    as well), both at level ``1 - 0.05 / m`` from the same replicates (:func:`replicates`). ``mdd`` is the Δ that
    both lower bounds would have needed to clear 0 with the same interval widths."""
    arm, base = np.atleast_2d(np.asarray(arm, float)), np.atleast_2d(np.asarray(base, float))
    y = np.asarray(y, bool)
    make = _scorer(stat, city)
    ones = np.ones(len(y))
    deltas = [make(a).auc(y, ones) - make(b).auc(y, ones) for a, b in zip(arm, base)]
    point = float(np.mean(deltas))
    D, draws = replicates(y, arm, base, clusters, n_boot, seed, stat, city)
    reg = D.mean(axis=1)
    hier = np.take_along_axis(D, draws, axis=1).mean(axis=1)
    tail = 100 * ALPHA / m / 2
    reg_lo, reg_hi = (float(v) for v in np.nanpercentile(reg, [tail, 100 - tail]))
    hier_lo, hier_hi = (float(v) for v in np.nanpercentile(hier, [tail, 100 - tail]))
    return {"point": point, "seed_deltas": [float(d) for d in deltas], "k": int(k), "m": int(m),
            "level": 1 - ALPHA / m, "n_boot": int(n_boot), "reg_lo": reg_lo, "reg_hi": reg_hi, "hier_lo": hier_lo,
            "hier_hi": hier_hi, "mdd": point - min(reg_lo, hier_lo), "n": int(len(y)),
            "n_incorrect": int(y.sum()), "n_correct": int((~y).sum()),
            "n_panoramas": int(len(np.unique(np.asarray(clusters))))}


def seed_guard(seed_deltas, k):
    """The seed guard on ``test``, by the registered seed count ``k``: every seed pair ahead (Δ > 0) for k < 5, at
    least k - 1 of k for k >= 5. A list of another length than ``k`` fails (no reading on a subset).

    >>> seed_guard([0.01, 0.02, -0.01], 3), seed_guard([0.01, 0.02, 0.03, 0.01, -0.02], 5)
    (False, True)
    """
    if len(seed_deltas) != k:
        return False
    ahead = sum(1 for d in seed_deltas if d > 0)
    return ahead >= (k - 1 if k >= 5 else k)


def arm_passes(cell):
    """``(passes, why)`` for one type and arm under R3 as amended: estimable, both lower bounds strictly above 0,
    the seed guard, and the mean within-city ΔAUC on ``test_heldout`` at least 0."""
    if cell.get("unestimable"):
        return False, f"unestimable: {cell['unestimable']}"
    if not cell["reg_lo"] > 0:
        return False, f"registered lower bound {cell['reg_lo']:+.4f} is not above 0"
    if not cell["hier_lo"] > 0:
        return False, f"hierarchical lower bound {cell['hier_lo']:+.4f} is not above 0"
    if not seed_guard(cell["seed_deltas"], cell["k"]):
        ahead = sum(1 for d in cell["seed_deltas"] if d > 0)
        return False, f"seed guard: {ahead} of {cell['k']} seed pairs ahead"
    w = cell.get("heldout_within")
    if w is None or not np.isfinite(w):
        return False, "no held-out within-city reading"
    if not w >= 0:
        return False, f"held-out within-city Δ {w:+.4f} is below 0"
    return True, "passes"


def verdict(cell):
    """``pass``, ``fail``, ``worse`` (both upper bounds below 0: significantly worse than 6100, reported as such) or
    ``unestimable``."""
    if cell.get("unestimable"):
        return "unestimable"
    if arm_passes(cell)[0]:
        return "pass"
    if cell.get("reg_hi") is not None and cell["reg_hi"] < 0 and cell.get("hier_hi") is not None and \
            cell["hier_hi"] < 0:
        return "worse"
    return "fail"


def rule_readout(results, m=M_ARMS):
    """The R3 read-out over ``results`` = ``{type: {arm: cell}}`` (an arm absent for a type was not run there).

    Outcomes: ``unestimable`` (some cell is: nothing changes until it is fixed and the whole family re-read),
    ``no_change`` (no arm passes every swept type: 6100 stays), ``both_pass`` (more than one does: not a constant
    effect, nothing changes, stage 2 not triggered), ``candidate`` (exactly one passes, but not all five types were
    swept: the default does not change; ``stage2_triggered`` when both stage-1 types were) or ``change`` (exactly one
    arm passes all five types)."""
    types = list(results)
    arms = sorted({int(a) for t in types for a in results[t]})
    verdicts = {t: {int(a): verdict(c) for a, c in results[t].items()} for t in types}
    why = {t: {int(a): arm_passes(c)[1] for a, c in results[t].items()} for t in types}
    unest = [f"{t} {arm_name(a)}: {c['unestimable']}" for t in types for a, c in results[t].items()
             if c.get("unestimable")]
    passing = [a for a in arms if all(verdicts[t].get(a) == "pass" for t in types)]
    stage1 = set(STAGE1_TYPES) <= set(types)
    complete = set(ALL_TYPES) <= set(types)
    ks = [c.get("k") for t in types for c in results[t].values() if c.get("k")]
    k = max(ks) if ks else None
    if unest:
        outcome, passing = "unestimable", []
        reading = "No reading: " + "; ".join(unest) + ". Nothing changes; fix the cause and re-read the whole family."
    elif not passing:
        outcome = "no_change"
        reading = "No arm passes every swept type: 6100 stays, and the curve is the deliverable."
    elif len(passing) > 1:
        outcome = "both_pass"
        reading = ("More than one arm passes: the result does not read as a constant effect. Nothing changes and "
                   "stage 2 is not triggered.")
    elif complete:
        outcome = "change"
        reading = (f"{arm_name(passing[0])} passes all five types: the default changes, with a retrain of all five "
                   f"types, refitted maps and the serving change, landing with a round.")
    else:
        outcome = "candidate"
        reading = (f"{arm_name(passing[0])} passes {', '.join(types)}: a candidate only; the default does not change"
                   + (" until stage 2 (NoCurbRamp, Obstacle, Crosswalk; that arm and c6100) has been read."
                      if stage1 else "."))
    caveat = None
    if k:
        caveat = (f"With {k} registered seeds, (1/{k})^{k} = {100 * (1 / k) ** k:.2g} % of hierarchical replicates "
                  f"draw one given seed only (for example the worst), so the hierarchical level is nominal.")
    mdd = {t: {int(a): c.get("mdd") for a, c in results[t].items() if not c.get("unestimable")} for t in types}
    return {"outcome": outcome, "passing_arms": passing, "stage2_triggered": outcome == "candidate" and stage1,
            "verdicts": verdicts, "why": why, "unestimable": unest, "m": m, "k": k, "caveat": caveat, "mdd": mdd,
            "reading": reading, "types": types}


# ---------------------------------------------------------------------------------------------------------------------
# compare (Mac, after the klone runs)
# ---------------------------------------------------------------------------------------------------------------------

def family_problems(runs):
    """Why a family of runs (``{(arm, seed): training_run}``, every arm and seed of one type, c6100 included) cannot
    be read together: differing ``versions`` or ``git_commit``, a missing value, ``git_dirty`` not false, or a
    checkout that moved during a run (``git_state_at_finish``). Empty when the family is consistent."""
    probs = []
    for key in ("git_commit", "versions"):
        vals = {json.dumps(r.get(key), sort_keys=True) for r in runs.values()}
        if len(vals) > 1:
            probs.append(f"runs differ in {key} ({len(vals)} distinct values)")
        missing = sorted(f"{arm_name(c)}/s{s}" for (c, s), r in runs.items() if not r.get(key))
        if missing:
            probs.append(f"no {key} recorded in {', '.join(missing)}")
    dirty = sorted(f"{arm_name(c)}/s{s}" for (c, s), r in runs.items() if r.get("git_dirty") is not False)
    if dirty:
        probs.append(f"git_dirty is not false in {', '.join(dirty)}")
    moved = sorted(f"{arm_name(c)}/s{s}" for (c, s), r in runs.items() if r.get("git_state_at_finish"))
    if moved:
        probs.append(f"git_state_at_finish recorded (the checkout moved during the run) in {', '.join(moved)}")
    return probs


def _read_arm(work, t, c, seeds, runs_root):
    d = os.path.join(arm_dir(work, c), t)
    split_p = os.path.join(d, "split.csv")
    split = pd.read_csv(split_p, dtype={"label_uid": str, "pano_id": str, "city": str}, low_memory=False) \
        if os.path.exists(split_p) else None
    preds, runs = {}, {}
    on_disk = {int(m.group(1)) for x in glob.glob(os.path.join(d, "predictions_s*.csv"))
               if (m := re.fullmatch(r"predictions_s(\d+)\.csv", os.path.basename(x)))}
    on_disk |= {int(m.group(1)) for x in glob.glob(os.path.join(runs_root, arm_name(c), t, "s*", "training_run.json"))
                if (m := re.fullmatch(r"s(\d+)", os.path.basename(os.path.dirname(x))))}
    inputs = {f"{arm_name(c)}/{t}/split.csv": sha256_file(split_p)} if split is not None else {}
    for s in seeds:
        pp = os.path.join(d, f"predictions_s{s}.csv")
        rp = os.path.join(runs_root, arm_name(c), t, f"s{s}", "training_run.json")
        if os.path.exists(pp) and os.path.exists(rp):
            preds[s] = pd.read_csv(pp, dtype={"label_uid": str})
            with open(rp) as f:
                runs[s] = json.load(f)
            inputs[f"{arm_name(c)}/{t}/predictions_s{s}.csv"] = sha256_file(pp)
            inputs[f"runs/{arm_name(c)}/{t}/s{s}/training_run.json"] = sha256_file(rp)
    return {"split": split, "preds": preds, "runs": runs, "inputs": inputs,
            "unregistered": sorted(on_disk - {int(x) for x in seeds})}


class Unestimable(Exception):
    """A cell (or the negative control) that cannot be read. Its text goes into ``compare.json`` and ``tables.md``, so
    it is built only from seeds, counts, arm and split names: never a label id, city id or other row identifier."""


def _matrix(preds, seeds, uids, what):
    """``k x n`` p_incorrect scores in ``uids`` order; raises :class:`Unestimable` with the count of missing rows
    (never which rows: a ``label_uid`` carries its city id)."""
    out = []
    for s in seeds:
        p = preds[s].drop_duplicates("label_uid").set_index("label_uid").p_incorrect
        miss = uids[~uids.isin(p.index)]
        if len(miss):
            raise Unestimable(f"{what} seed {s} has no prediction for {len(miss)} of {len(uids)} rows")
        out.append(p.loc[uids].to_numpy(float))
    return np.stack(out)


def _year(v):
    return da.year_band(v)


def _quartile_labels(values, ref):
    """Quartile bins of ``values`` by the edges of ``ref`` (the baseline's ``crop_half_px`` on the same split); the
    outer bins are open, so training rows beyond the split's range fall in Q1 or Q4.

    >>> _quartile_labels(pd.Series([100, 250, 900]), [200, 300, 400, 500, 600]).tolist()
    ['Q1 (200, 300]', 'Q1 (200, 300]', 'Q4 (500, 600]']
    """
    edges = np.unique(np.quantile(np.asarray(ref, float), [0, 0.25, 0.5, 0.75, 1]))
    if len(edges) < 2:
        return pd.Series(["all"] * len(values), index=values.index)
    labels = [f"Q{i + 1} ({edges[i]:.0f}, {edges[i + 1]:.0f}]" for i in range(len(edges) - 1)]
    bins = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])   # training rows beyond the split's range still count
    return pd.cut(values.astype(float), bins, labels=labels).astype(str)


def _slice_rows(part, train, A, B, by, split, arm):
    """One row per slice: n, the baseline's and the arm's seed-mean AUC, the seed-mean Δ, and training counts."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for name, idx in part.groupby("slice").indices.items():
        g = part.iloc[idx]
        y = (g.label_class == "incorrect").to_numpy()
        row = {"split": split, "by": by, "arm": arm, "slice": str(name), "n": int(len(g)),
               "n_incorrect": int(y.sum()), "base_auc": None, "arm_auc": None, "delta": None}
        if len(g) >= sl.MIN_N and min(y.sum(), (~y).sum()) >= sl.MIN_CLASS_FOR_CI:
            rows_idx = g["_row"].to_numpy()
            ba = [roc_auc_score(y, B[s, rows_idx]) for s in range(B.shape[0])]
            aa = [roc_auc_score(y, A[s, rows_idx]) for s in range(A.shape[0])]
            row.update(base_auc=float(np.mean(ba)), arm_auc=float(np.mean(aa)),
                       delta=float(np.mean(np.subtract(aa, ba))))
        tn = int((train["slice"] == name).sum()) if train is not None else 0
        row.update(train_n=tn, needs_data=tn < sl.THIN_TRAIN)
        rows.append(row)
    return rows


def _with_mechanism(df, by, ref_half):
    if by == "crop_half_px_quartile":
        return df.assign(slice=_quartile_labels(df.crop_half_px, ref_half))
    return df.assign(slice=df.image_capture_date.map(_year))


def slice_table(split_rows, train_rows, A, B, arm, split):
    """Severity, tag and the two mechanism slices (baseline ``crop_half_px`` quartile, capture-year band) of one
    split: seed-mean Δ beside the baseline AUC, with training counts and the thin-slice flag."""
    part = split_rows.assign(_row=np.arange(len(split_rows)))
    out = []
    for by in sl.BY:
        out += _slice_rows(sl.with_slice(part, by), sl.with_slice(train_rows, by), A, B, by, split, arm)
    ref = split_rows.crop_half_px
    for by in ("crop_half_px_quartile", "capture_year"):
        out += _slice_rows(_with_mechanism(part, by, ref), _with_mechanism(train_rows, by, ref), A, B, by, split,
                           arm)
    return out


def recall_at_precision(y, A, level=RECALL_PRECISION):
    """Seed-mean recall at precision ``level`` for each class (``evaluate.recall_at_precision``)."""
    import evaluate
    return {"incorrect": float(np.mean([evaluate.recall_at_precision(y, a, level)[0] for a in A])),
            "correct": float(np.mean([evaluate.recall_at_precision(~y, 1 - a, level)[0] for a in A]))}


def gpu_models(logs_dir, t, c, seeds):
    """The GPU each run trained on, read from its log (``nvidia-smi`` prints ``name, memory`` first)."""
    out = {}
    for s in seeds:
        paths = sorted(glob.glob(os.path.join(logs_dir, f"cc-{t}-{arm_name(c)}-s{s}-*.out")))
        for p in paths[-1:]:
            with open(p, errors="replace") as f:
                for line in f:
                    if "MiB" in line:
                        out[s] = line.split(",")[0].strip()
                        break
    return out


def control_predictions(control_root, t, seeds=(42, 43, 44)):
    """Round 3's stored 6100 predictions: seed 42 is ``<t>/predictions.csv`` (``fine_tune.slurm``, 4 workers), 43 and
    44 ``<t>/predictions_baseline-s<seed>.csv`` (``confident_learning.slurm``, 7 workers)."""
    out = {}
    for s in seeds:
        name = "predictions.csv" if s == 42 else f"predictions_baseline-s{s}.csv"
        p = os.path.join(control_root, t, name)
        if os.path.exists(p):
            out[s] = pd.read_csv(p, dtype={"label_uid": str})
            out[s].attrs["sha256"] = sha256_file(p)
    return out


def compare_type(work, t, seeds=REGISTERED_SEEDS, arms=SWEPT, n_boot=N_BOOT, seed=SEED, runs_root=None,
                 control_root=None, logs_dir=None):
    """Everything ``compare`` computes for one type: the R3 cells and every reported table.

    A cell is unestimable, and never passes, when: the family differs (:func:`family_problems`); either side has
    fewer runs with predictions than the registered seeds (``len(seeds)``), or holds predictions or runs for a seed
    outside them (a reading on a subset); a run records no ``crop_depth_constant`` or another one than its arm; the
    arms' rows differ; a prediction is missing; or ``test`` has fewer than 10 rows of a class. The family check is per
    label type: pairing is within a type, and stage 2 is its own family."""
    runs_root = runs_root or os.path.join(work, "runs")
    k = len(seeds)
    data = {c: _read_arm(work, t, c, seeds, runs_root) for c in (BASELINE,) + tuple(arms)}
    allruns = {(c, s): r for c, d in data.items() for s, r in d["runs"].items()}
    fam = family_problems(allruns) if allruns else []
    res = {"type": t, "seeds": list(seeds), "k": k, "arms": [int(a) for a in arms], "family_problems": fam,
           "cells": {}, "heldout": {}, "lodo": {}, "slices": [], "recall_at_precision": {}, "auc_test": {},
           "gpu": {}, "negative_control": None,
           "inputs": {name: h for d in data.values() for name, h in d["inputs"].items()},
           "cities": sorted({str(x) for d in data.values() if d["split"] is not None for x in d["split"].city})}

    def problem(c):
        d = data[c]
        if d["split"] is None:
            return f"no {arm_name(c)} split.csv"
        if d["unregistered"]:
            return (f"{arm_name(c)} holds predictions or runs for seeds outside the registered ones "
                    f"({', '.join(f's{s}' for s in d['unregistered'])}): no reading on a subset")
        if len(d["runs"]) < k:
            return (f"{arm_name(c)} has {len(d['runs'])} of the {k} registered seeds (no reading on a subset; "
                    f"missing: {', '.join(f's{s}' for s in seeds if s not in d['runs'])})")
        absent = [f"s{s}" for s, r in sorted(d["runs"].items()) if r.get("crop_depth_constant") is None]
        if absent:                         # every arm's generation_run.json records it, and training copies it
            return f"{arm_name(c)} runs {', '.join(absent)} record no crop_depth_constant"
        wrong = [f"s{s}" for s, r in sorted(d["runs"].items()) if not _is_constant(r["crop_depth_constant"], c)]
        if wrong:
            return f"{arm_name(c)} runs {', '.join(wrong)} record another crop_depth_constant"
        return None

    base_problem = problem(BASELINE)
    base = data[BASELINE]
    for c in arms:
        why = "; ".join(p for p in (("family differs: " + "; ".join(fam)) if fam else None,
                                    base_problem and f"baseline: {base_problem}", problem(c)) if p)
        if not why and set(data[c]["split"].label_uid) != set(base["split"].label_uid):
            why = f"{arm_name(c)} and c6100 hold different rows"
        cell = {"unestimable": why or None, "k": k}
        if not why:
            try:
                cell.update(_score_arm(data, c, t, seeds, k, n_boot, seed, res))
            except Unestimable as e:                  # anything else is a bug and stops compare
                cell = {"unestimable": str(e), "k": k}
        res["cells"][int(c)] = cell
        if logs_dir:
            res["gpu"][arm_name(c)] = gpu_models(logs_dir, t, c, seeds)
    if logs_dir:
        res["gpu"][arm_name(BASELINE)] = gpu_models(logs_dir, t, BASELINE, seeds)
    if control_root and base_problem is None:
        res["negative_control"] = _control(control_root, t, base, n_boot, seed)
    return res


def _is_constant(value, c):
    try:
        return float(value) == c
    except (TypeError, ValueError):
        return False


def _rows(split_df, name):
    r = split_df[split_df.split == name].reset_index(drop=True)
    return r, (r.label_class == "incorrect").to_numpy(), (r.city + ":" + r.pano_id).to_numpy()


def _score_arm(data, c, t, seeds, k, n_boot, seed, res):
    base, arm = data[BASELINE], data[c]
    split = base["split"]
    test, y, clusters = _rows(split, "test")
    if min(y.sum(), (~y).sum()) < MIN_CLASS:
        raise Unestimable(f"fewer than {MIN_CLASS} test rows of a class ({int(y.sum())} incorrect, "
                         f"{int((~y).sum())} correct)")
    A = _matrix(arm["preds"], seeds, test.label_uid, f"{arm_name(c)} test")
    B = _matrix(base["preds"], seeds, test.label_uid, "c6100 test")
    s0 = sl.slice_seed(seed, f"{t}:{c}")
    cell = paired_stats(y, A, B, clusters, k, n_boot=n_boot, seed=s0)
    held, yh, ch = _rows(split, "test_heldout")
    within = None
    if len(held):
        Ah = _matrix(arm["preds"], seeds, held.label_uid, f"{arm_name(c)} test_heldout")
        Bh = _matrix(base["preds"], seeds, held.label_uid, "c6100 test_heldout")
        ones = np.ones(len(held))
        w = [WithinCityScores(a, held.city).auc(yh, ones) - WithinCityScores(b, held.city).auc(yh, ones)
             for a, b in zip(Ah, Bh)]
        within = float(np.mean(w)) if np.all(np.isfinite(w)) else None
        if min(yh.sum(), (~yh).sum()) >= MIN_CLASS:
            res["heldout"][int(c)] = {
                "pooled": paired_stats(yh, Ah, Bh, ch, k, n_boot=n_boot, seed=s0 + 1),
                "within": paired_stats(yh, Ah, Bh, ch, k, n_boot=n_boot, seed=s0 + 2, stat="within",
                                       city=held.city.to_numpy())}
        res["lodo"][int(c)] = _lodo(held, yh, Ah, Bh)
        res["slices"] += slice_table(held, split[split.split == "train"], Ah, Bh, int(c), "test_heldout")
    cell["heldout_within"] = within
    res["slices"] += slice_table(test, split[split.split == "train"], A, B, int(c), "test")
    res["recall_at_precision"][int(c)] = recall_at_precision(y, A)
    res["recall_at_precision"].setdefault(BASELINE, recall_at_precision(y, B))
    from sklearn.metrics import roc_auc_score
    res["auc_test"][int(c)] = {f"s{s}": float(roc_auc_score(y, A[i])) for i, s in enumerate(seeds)}
    res["auc_test"].setdefault(BASELINE, {f"s{s}": float(roc_auc_score(y, B[i])) for i, s in enumerate(seeds)})
    res["train_counts"] = {"train": int((split.split == "train").sum()), "test": len(test),
                           "test_heldout": len(held)}
    return cell


def _lodo(held, y, A, B):
    """Leave one held-out deployment out: the seed-mean pooled ΔAUC on the other held-out rows (keyed by city id;
    ``report`` pools the names of non-public deployments)."""
    out = []
    for city in sorted(held.city.unique()):
        keep = (held.city != city).to_numpy()
        yy = y[keep]
        rec = {"left_out": city, "n_left_out": int((~keep).sum()), "n": int(keep.sum()), "delta": None}
        if min(yy.sum(), (~yy).sum()) >= MIN_CLASS:
            ones = np.ones(int(keep.sum()))
            rec["delta"] = float(np.mean([RankedScores(a[keep]).auc(yy, ones) - RankedScores(b[keep]).auc(yy, ones)
                                          for a, b in zip(A, B)]))
        out.append(rec)
    return out


def _control(control_root, t, base, n_boot, seed):
    """The negative control, deciding nothing: round 3's stored 6100 predictions against the regenerated c6100 at
    seeds 42 to 44 (both intervals). The stored seed 42 came from ``fine_tune.slurm`` with 4 workers, 43 and 44 from
    ``confident_learning.slurm`` with 7, so a seed-42 outlier is not a pipeline effect."""
    seeds = [s for s in (42, 43, 44) if s in base["preds"]]
    stored = control_predictions(control_root, t, seeds)
    if len(stored) < 3 or len(seeds) < 3:
        return {"seeds": sorted(stored), "note": "fewer than three stored or regenerated seeds; not read"}
    test, y, clusters = _rows(base["split"], "test")
    try:
        A = _matrix(stored, seeds, test.label_uid, "stored round-3 test")
        B = _matrix(base["preds"], seeds, test.label_uid, "c6100 test")
    except Unestimable as e:
        return {"seeds": seeds, "note": str(e)}
    st = paired_stats(y, A, B, clusters, len(seeds), n_boot=n_boot, seed=sl.slice_seed(seed, f"{t}:control"))
    return {**st, "seeds": seeds, "what": "stored round-3 6100 minus regenerated c6100",
            "inputs_sha256": {f"s{s}": stored[s].attrs.get("sha256") for s in seeds}}


def context_auc(work, t, path):
    """The test AUC of another run's predictions (#51's fixed 238 px CurbRamp run) on c6100's test rows, beside the
    arms as context only: it comes from another pipeline and decides nothing. None when it misses a row."""
    from sklearn.metrics import roc_auc_score
    split = pd.read_csv(os.path.join(arm_dir(work, BASELINE), t, "split.csv"), dtype={"label_uid": str})
    test = split[split.split == "test"]
    p = pd.read_csv(path, dtype={"label_uid": str}).drop_duplicates("label_uid").set_index("label_uid")
    if not test.label_uid.isin(p.index).all():
        return {"path_sha256": sha256_file(path), "auc": None, "note": "misses some c6100 test rows"}
    y = (test.label_class == "incorrect").to_numpy()
    return {"path_sha256": sha256_file(path), "n": int(len(test)),
            "auc": float(roc_auc_score(y, p.loc[test.label_uid, "p_incorrect"].to_numpy(float)))}


def compare_output(results, seeds):
    """The ``compare.json`` record: every type's tables and the read-out over all of them."""
    commit, dirty = da.git_commit()
    readout = rule_readout({t: r["cells"] for t, r in results.items()})
    return {"stage": "compare", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
            "seeds": list(seeds), "registration": "REGISTERED_SEEDS in analysis/crop_constant_sweep.py at git_commit",
            "m": M_ARMS, "alpha": ALPHA, "types": results, "readout": readout}


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _f(v, fmt="{:+.4f}"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def _ci(lo, hi):
    return "" if lo is None or hi is None else f"[{_f(lo)}, {_f(hi)}]"


def report_tables(out, public):
    """``tables.md`` from a ``compare.json`` record. Held-out deployments outside ``public`` are named
    ``non-public deployment A``, ``B``, ... in order of row count, never by id."""
    lines = ["# Crop-constant sweep: generated tables", "",
             f"Generated from `compare.json` (code at `{out.get('git_commit')}`, dirty {out.get('git_dirty')}); "
             f"seeds {', '.join(str(s) for s in out['seeds'])}; level 1 - {out['alpha']} / {out['m']}.", ""]
    ro = out["readout"]
    lines += ["## Rule read-out (R3 as amended)", "", f"**Outcome: {ro['outcome']}.** {ro['reading']}", ""]
    if ro.get("caveat"):
        lines += [f"Caveat: {ro['caveat']}", ""]
    lines += ["| type | arm | Δ test (seed mean) | registered interval | hierarchical interval | seed Δs | "
              "held-out within-city Δ | verdict | why | MDD |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in out["types"].items():
        for a, c in r["cells"].items():
            v = ro["verdicts"].get(t, {}).get(str(a), ro["verdicts"].get(t, {}).get(a, ""))
            w = ro["why"].get(t, {}).get(str(a), ro["why"].get(t, {}).get(a, ""))
            if c.get("unestimable"):
                lines.append(f"| {t} | c{a} | | | | | | unestimable | {c['unestimable']} | |")
                continue
            lines.append(f"| {t} | c{a} | {_f(c['point'])} | {_ci(c['reg_lo'], c['reg_hi'])} | "
                         f"{_ci(c['hier_lo'], c['hier_hi'])} | {', '.join(_f(d) for d in c['seed_deltas'])} | "
                         f"{_f(c.get('heldout_within'))} | {v} | {w} | {_f(c.get('mdd'))} |")
    lines += ["", "MDD: the Δ that both lower bounds would have needed to clear 0, at the same interval widths.", ""]
    lines += ["## Test AUC per seed", "", "| type | arm | " + " | ".join(f"s{s}" for s in out["seeds"]) + " |",
              "|---|---|" + "---|" * len(out["seeds"])]
    for t, r in out["types"].items():
        for a, aucs in r.get("auc_test", {}).items():
            lines.append(f"| {t} | c{a} | " + " | ".join(_f(aucs.get(f's{s}'), "{:.4f}") for s in out["seeds"])
                         + " |")
    ctx = [(t, r["context_fixed238"]) for t, r in out["types"].items() if r.get("context_fixed238")]
    for t, c in ctx:
        lines += ["", f"Context only (another pipeline, decides nothing): #51's fixed 238 px {t} run, test AUC "
                  f"{_f(c.get('auc'), '{:.4f}') or c.get('note', '')} on the same test rows."]
    lines += ["", "## Negative control (decides nothing)", "",
              "Stored round-3 6100 predictions minus the regenerated c6100, seeds 42 to 44. Stored seed 42 came "
              "from `fine_tune.slurm` (4 workers), 43 and 44 from `confident_learning.slurm` (7 workers).", "",
              "| type | Δ | registered interval | hierarchical interval | seed Δs | note |", "|---|---|---|---|---|---|"]
    for t, r in out["types"].items():
        c = r.get("negative_control")
        if c and "point" in c:
            lines.append(f"| {t} | {_f(c['point'])} | {_ci(c['reg_lo'], c['reg_hi'])} | "
                         f"{_ci(c['hier_lo'], c['hier_hi'])} | {', '.join(_f(d) for d in c['seed_deltas'])} | |")
        elif c:
            lines.append(f"| {t} | | | | | {c.get('note', '')} |")
    lines += ["", "## Held-out deployments (reported, not deciding)", "",
              "| type | arm | pooled Δ | registered | hierarchical | within-city Δ | registered | hierarchical |",
              "|---|---|---|---|---|---|---|---|"]
    for t, r in out["types"].items():
        for a, h in r.get("heldout", {}).items():
            p, w = h["pooled"], h["within"]
            lines.append(f"| {t} | c{a} | {_f(p['point'])} | {_ci(p['reg_lo'], p['reg_hi'])} | "
                         f"{_ci(p['hier_lo'], p['hier_hi'])} | {_f(w['point'])} | {_ci(w['reg_lo'], w['reg_hi'])} | "
                         f"{_ci(w['hier_lo'], w['hier_hi'])} |")
    lines += ["", "Leave one held-out deployment out (pooled Δ on the remaining held-out rows):", "",
              "| type | arm | left out | rows left out | rows kept | Δ |", "|---|---|---|---|---|---|"]
    for t, r in out["types"].items():
        for a, rows in r.get("lodo", {}).items():
            names = _deployment_names(rows, public)
            for row in rows:
                lines.append(f"| {t} | c{a} | {names[row['left_out']]} | {row['n_left_out']} | {row['n']} | "
                             f"{_f(row['delta'])} |")
    lines += ["", "## Slices (seed-mean Δ beside the baseline AUC; training counts from c6100's train rows)", "",
              "| type | split | by | slice | arm | n | n incorrect | baseline AUC | arm AUC | Δ | train n | thin |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in out["types"].items():
        for s in r.get("slices", []):
            lines.append(f"| {t} | {s['split']} | {s['by']} | {s['slice']} | c{s['arm']} | {s['n']} | "
                         f"{s['n_incorrect']} | {_f(s['base_auc'], '{:.4f}')} | {_f(s['arm_auc'], '{:.4f}')} | "
                         f"{_f(s['delta'])} | {s['train_n']} | {'yes' if s['needs_data'] else ''} |")
    lines += ["", f"## Recall at precision {RECALL_PRECISION} (seed mean, test)", "",
              "| type | arm | incorrect | correct |", "|---|---|---|---|"]
    for t, r in out["types"].items():
        for a, v in r.get("recall_at_precision", {}).items():
            lines.append(f"| {t} | c{a} | {_f(v['incorrect'], '{:.4f}')} | {_f(v['correct'], '{:.4f}')} |")
    gpus = [(t, a, g) for t, r in out["types"].items() for a, g in r.get("gpu", {}).items() if g]
    if gpus:
        lines += ["", "## GPU per run (from the logs)", ""]
        lines += [f"- {t} {a}: " + ", ".join(f"s{s} {n}" for s, n in sorted(g.items())) for t, a, g in gpus]
    return "\n".join(lines) + "\n"


def _deployment_names(rows, public):
    """City id -> printed name: public ids as they are, the others ``non-public deployment A``, ``B``, ... by row
    count (descending), so no non-public id reaches a table."""
    hidden = sorted((r for r in rows if r["left_out"] not in public), key=lambda r: (-r["n_left_out"], r["left_out"]))
    names = {r["left_out"]: r["left_out"] for r in rows if r["left_out"] in public}
    names.update({r["left_out"]: POOLED_NAME.format(chr(ord("A") + i)) for i, r in enumerate(hidden)})
    return names


def report_figure(out, path):
    """The sensitivity curve per type (seed-mean ΔAUC against 6100 with both intervals, at each constant) and Δ by
    the baseline's crop_half_px quartile on ``test``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = list(out["types"])
    fig, axes = plt.subplots(2, max(len(types), 1), figsize=(4.2 * max(len(types), 1), 6.4), squeeze=False)
    for j, t in enumerate(types):
        r = out["types"][t]
        ax = axes[0][j]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot([BASELINE], [0], "o", color="0.3")
        for a, c in r["cells"].items():
            if c.get("unestimable"):
                continue
            x = int(a)
            ax.errorbar([x - 80], [c["point"]], yerr=[[c["point"] - c["reg_lo"]], [c["reg_hi"] - c["point"]]],
                        fmt="o", color="#1f5f99", capsize=3, label="registered" if x == min(map(int, r["cells"])) else None)
            ax.errorbar([x + 80], [c["point"]], yerr=[[c["point"] - c["hier_lo"]], [c["hier_hi"] - c["point"]]],
                        fmt="s", color="#b35900", capsize=3,
                        label="hierarchical" if x == min(map(int, r["cells"])) else None)
        ax.set_xticks(list(ARMS))
        ax.set_title(t)
        ax.set_xlabel("crop constant C")
        ax.set_ylabel("ΔAUC vs 6100 (test)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax2 = axes[1][j]
        ax2.axhline(0, color="0.6", lw=0.8)
        q = [s for s in r.get("slices", []) if s["by"] == "crop_half_px_quartile" and s["split"] == "test"]
        for a in sorted({s["arm"] for s in q}):
            pts = [(s["slice"], s["delta"]) for s in q if s["arm"] == a and s["delta"] is not None]
            if pts:
                ax2.plot(range(len(pts)), [p[1] for p in pts], "o-", label=f"c{a}")
                ax2.set_xticks(range(len(pts)))
                ax2.set_xticklabels([p[0] for p in pts], rotation=30, fontsize=7)
        ax2.set_xlabel("baseline crop_half_px quartile")
        ax2.set_ylabel("seed-mean ΔAUC")
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def _gsv_beta(root, types):
    betas = set()
    for t in types:
        with open(os.path.join(root, t, "generation_run.json")) as f:
            run = json.load(f)
        betas.add(run.get("tilt_beta", (run.get("settings") or {}).get("tilt_beta", GSV_BETA)))
    if len(betas) != 1:
        sys.exit(f"round 3's records disagree on tilt_beta: {sorted(betas)}")
    return float(betas.pop())


def cmd_stage_list(args):
    types = split_types(args.types)
    frame = read_splits(args.root, types)
    files = stage_files(frame)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(files) + "\n")
    present = present_in_store(files, args.exclude_store) if args.exclude_store else set()
    rec = {"stage": "stage-list", "date": datetime.date.today().isoformat(), "types": list(types),
           "labels": int(len(frame)), "panoramas": len(files), "already_in_store": len(present),
           "to_transfer": len(files) - len(present), "stage_files_sha256": sha256_file(args.out),
           "inputs": {f"{t}/split.csv": sha256_file(os.path.join(args.root, t, "split.csv")) for t in types}}
    write_json(os.path.splitext(args.out)[0] + ".json", rec)
    print(f"{len(files)} panoramas for {len(frame)} labels -> {args.out}"
          + (f"; {len(present)} already in {args.exclude_store}" if args.exclude_store else ""))


def cmd_chunks(args):
    types = split_types(args.types)
    frame = read_splits(args.root, types)
    cap = max_labels_for(args.time_limit)
    plan = chunk_plan(frame, cap)
    sizes = frame.groupby(plan).size()
    print(f"{len(frame)} labels in {plan.nunique()} chunks of at most {cap} labels (largest {sizes.max()}) for "
          f"--time {args.time_limit}", file=sys.stderr)
    print(plan.nunique())


def cmd_registered_seeds(args):
    print(",".join(str(x) for x in REGISTERED_SEEDS))


def cmd_preflight(args):
    types = split_types(args.types)
    rec = preflight(args.root, args.rawlabels_dir, types, args.pano_store)
    write_json(args.out, rec)
    print(json.dumps({"g0": rec["g0"], "store": {k: rec["store"][k] for k in ("passed", "failures",
                                                                              "missing_panoramas")}}, indent=1))
    if not (rec["g0"]["passed"] and rec["store"]["passed"]):
        sys.exit("preflight failed; see " + args.out)


def cmd_recrop(args):
    types = split_types(args.types)
    if args.pano_store:
        dg.check_pano_store_marker(args.pano_store)
    frame = load_frame(args.root, args.rawlabels_dir, types)
    device = dg.resolve_device(args.allow_cpu)
    out = recrop(frame, args.work, args.chunk, max_labels_for(args.time_limit), pano_store=args.pano_store,
                 stored_root=args.stored_root or args.root, device=device, gsv_beta=_gsv_beta(args.root, types),
                 limit=args.limit)
    per = out["seconds"] / out["processed"] if out["processed"] else float("nan")
    print(f"recrop chunk {args.chunk}: {out['processed']} labels in {out['seconds']:.0f} s ({per:.2f} s per label); "
          f"statuses {out['statuses']}")


def cmd_gates(args):
    types = split_types(args.types)
    pre_path = os.path.join(args.work, "preflight.json")
    if not os.path.exists(pre_path):
        sys.exit(f"no {pre_path}: run preflight first")
    with open(pre_path) as f:
        pre = json.load(f)
    frame = load_frame(args.root, args.rawlabels_dir, types)
    rec = load_recrop(args.work)
    rec = rec[rec.label_uid.isin(frame.label_uid)]
    g2, g3 = gate_g2(frame, rec), gate_g3(rec)
    parity_path = os.path.join(args.work, "recrop_parity.csv")
    rec.to_csv(parity_path, index=False)
    commit, dirty = da.git_commit()
    out = {"stage": "gates", "date": datetime.date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
           "G0": pre["g0"], "G2": g2, "G3": g3, "recrop_parity_sha256": sha256_file(parity_path),
           "passed": bool(pre["g0"]["passed"] and g2["passed"] and g3["passed"])}
    if out["passed"]:
        info = {"tool": "analysis/crop_constant_sweep.py recrop", "git_commit": commit, "git_dirty": dirty,
                "date": out["date"], "pano_store": pre.get("store", {}).get("pano_store"),
                "renderer": sorted(rec.renderer.dropna().unique().tolist()),
                "recrop_parity_sha256": out["recrop_parity_sha256"], "n_source_rows": int(len(frame)),
                "lost_by_status": {str(k): int(v) for k, v in rec.status[rec.status != "ok"].value_counts().items()},
                "sizing": {"4000": "int(4000 / depth_m), recorded depth_m", "8000": "int(8000 / depth_m), recorded "
                           "depth_m", "6100": "recorded crop_half_px"}}
        out["arms"] = write_arms(args.work, args.root, frame, rec, types, info)
    write_json(os.path.join(args.work, "gates.json"), out)
    print(json.dumps({"G0": pre["g0"]["passed"], "G2": g2["passed"], "G3": g3["passed"]}))
    if not out["passed"]:
        sys.exit("a gate failed; no arm directory was written. See gates.json")


def cmd_compare(args):
    types = split_types(args.types)
    seeds = REGISTERED_SEEDS                         # the committed registration; never a command-line choice
    arms, arms2 = split_ints(args.arms), split_ints(args.stage2_arms or args.arms)
    res = {t: compare_type(args.work, t, seeds, arms2 if t in STAGE2_TYPES else arms, args.n_boot,
                           runs_root=args.runs_root, control_root=args.control_root, logs_dir=args.logs_dir)
           for t in types}
    if args.fixed238 and "CurbRamp" in res:
        res["CurbRamp"]["context_fixed238"] = context_auc(args.work, "CurbRamp", args.fixed238)
    out = compare_output(res, seeds)
    write_json(args.out, out)
    ro = out["readout"]
    print(f"Outcome: {ro['outcome']}. {ro['reading']}")
    for t in types:
        for a, v in ro["verdicts"][t].items():
            print(f"  {t} {arm_name(a)}: {v} ({ro['why'][t][a]}); MDD {ro['mdd'][t].get(a)}")
    if ro["caveat"]:
        print(ro["caveat"])


def report_leaks(text, cities, public):
    """Why ``text`` must not be written: a non-public city id anywhere (case-insensitive), or anything shaped like a
    label id (``<city>:<digits>``) for any city. Empty when the text is clean. The last check before ``report``
    writes, whatever the code that built the text."""
    low = text.lower()
    hidden = sorted(c for c in cities if c not in public)
    found = [f"a non-public deployment id ({len([c for c in hidden if c.lower() in low])} of {len(hidden)})"] \
        if any(c.lower() in low for c in hidden) else []
    if any(re.search(re.escape(c.lower()) + r":\d", low) for c in cities):
        found.append("a label id")
    return found


def cmd_report(args):
    with open(args.compare) as f:
        out = json.load(f)
    public, listing = da.resolve_public(args.public_cities, args.cities_json)
    text = report_tables(out, public)
    cities = {c for r in out["types"].values() for c in r.get("cities", [])}
    leaks = report_leaks(text, cities, public)
    if leaks:
        sys.exit(f"refusing to write the report: the tables would contain {' and '.join(leaks)}")
    os.makedirs(args.out_dir, exist_ok=True)
    tables = os.path.join(args.out_dir, "tables.md")
    with open(tables, "w") as f:
        f.write(text)
    fig = os.path.join(args.out_dir, "crop_constant.png")
    report_figure(out, fig)
    write_json(os.path.join(args.out_dir, "provenance.json"), {
        "stage": "report", "date": datetime.date.today().isoformat(), "compare_sha256": sha256_file(args.compare),
        "public_listing": {k: v for k, v in listing.items() if k != "path"},
        "outputs": {"tables.md": sha256_file(tables), "crop_constant.png": sha256_file(fig)}})
    print(f"wrote {tables}, {fig}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage-list", help="the deduplicated panorama list for ai_labels_stage.slurm")
    s.add_argument("--root", required=True, help="round 3's data root (<type>/split.csv)")
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--out", required=True, help="stage_files.txt (a stage_list.json record is written beside it)")
    s.add_argument("--exclude-store", default=None, help="count the files already in this store (stage 2)")
    s.set_defaults(func=cmd_stage_list)

    s = sub.add_parser("chunks", help="print the number of recrop array tasks for --time-limit")
    s.add_argument("--root", required=True)
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--time-limit", required=True, help="the recrop job's --time (Slurm format)")
    s.set_defaults(func=cmd_chunks)

    s = sub.add_parser("registered-seeds", help="print the registered seeds (the submit script trains these)")
    s.set_defaults(func=cmd_registered_seeds)

    s = sub.add_parser("preflight", help="G0 and the store check")
    s.add_argument("--root", required=True)
    s.add_argument("--rawlabels-dir", required=True)
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--pano-store", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("recrop", help="re-cut one chunk at 4000 / 6100 / 8000 (GPU)")
    s.add_argument("--root", required=True)
    s.add_argument("--rawlabels-dir", required=True)
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--work", required=True)
    s.add_argument("--chunk", type=int, required=True)
    s.add_argument("--time-limit", required=True, help="the job's --time; fixes the chunking")
    s.add_argument("--pano-store", default=None, help="override every row's recorded store root (a fresh restage)")
    s.add_argument("--stored-root", default=None, help="round 3's crops (<type>/<crop_path>); default --root")
    s.add_argument("--limit", type=int, default=None, help="at most this many labels (SMOKE=1: 20)")
    s.add_argument("--allow-cpu", action="store_true", help="numpy render on a CPU (smoke tests only)")
    s.set_defaults(func=cmd_recrop)

    s = sub.add_parser("gates", help="G2 and G3; writes the arm directories when every gate passes")
    s.add_argument("--root", required=True)
    s.add_argument("--rawlabels-dir", required=True)
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--work", required=True)
    s.set_defaults(func=cmd_gates)

    s = sub.add_parser("compare", help="the R3 read-out and the reported tables")
    s.add_argument("--work", required=True, help="a copy of the klone working directory (arms, runs)")
    s.add_argument("--types", default=",".join(STAGE1_TYPES))
    s.add_argument("--arms", default="4000,8000")
    s.add_argument("--stage2-arms", default=None, help="the arm trained for the stage-2 types (default --arms)")
    s.add_argument("--runs-root", default=None, help="default <work>/runs")
    s.add_argument("--control-root", default=None, help="round 3's data root, for the negative control")
    s.add_argument("--logs-dir", default=None, help="the training logs, for the GPU model of each run")
    s.add_argument("--fixed238", default=None, help="#51's fixed 238 px CurbRamp predictions (a context row)")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("report", help="tables.md, the figure and provenance.json")
    s.add_argument("--compare", required=True)
    s.add_argument("--public-cities", default=None, help="a JSON list of public city ids (tests, re-runs)")
    s.add_argument("--cities-json", default=None, help="a saved /v3/api/cities listing")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
