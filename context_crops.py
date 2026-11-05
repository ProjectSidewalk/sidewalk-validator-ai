"""Context images around a label (issue #14): a crop ``WIDE_FACTOR`` times wider than the depth-sized one, and the
whole label-centred view downsampled to ``VIEW_PX`` px. Both are cut from the same view ``data_generation.py``
renders for the training crop, so the three images share one geometry.

Why it exists: the validator sees only the tight crop, and for some types the question is about the surroundings
("is this obstacle actually in the path?", "is there a ramp a metre to the left?"). ``context_model.py`` trains three
"more context" arms on these files; ``analysis/context_arms.py`` compares them with the round-3 model.

The API is deliberately generic (peer code, e.g. the tagger, calls it too): it takes a pano, a label point, a crop
half-width in view pixels and output paths. It never sees a label type, a manifest row or a split.

Files, under ``<out_dir>`` (``context_dir(<data>/<label_type>)`` = ``<data>/<label_type>/context``)::

    wide/<stem>.webp   the wide crop: dg.centre_crop at wide_half_px, long side dg.CROP_LONG_SIDE, WebP q95
    view/<stem>.webp   the 2048 px view, INTER_AREA-resized to VIEW_PX x VIEW_PX, WebP q95
    index.csv          one row per label_uid: INDEX_COLUMNS (paths relative to out_dir)
    context_run.json   how a backfill made them (factor, view_px, renderer, package source, generation settings)

``stem`` is the training crop's path without ``.webp`` (``<label_class>/<city>_<label_id>``), so the three images of
a label have the same name.

**The wide crop is often the whole view.** The median tight half-width is about 238 px of a 1024 px half-view, so at
4x roughly half the labels ask for more than the view holds; ``wide_half_px`` is capped at ``VIEW_SIZE // 2`` and
``wide_clipped`` says the requested context did not fit. ``analysis/context_arms.py`` reports the share per type.

Two ways to make them:

* new datasets: ``data_generation.py --context-crops`` (a resume setting; ~3 lines in the main loop call
  :func:`write_context` after each crop is saved);
* an existing dataset (round 3): ``python context_crops.py backfill``, which re-renders each ok row's view from the
  pano store with the pose the crop was cut under and writes only ``<label_type>/context/``. The manifest, crops and
  split.csv are not touched. ``--verify N`` re-cuts N tight crops from the re-rendered views and compares them with
  the stored WebPs, so a backfill that drifted from the original geometry fails loudly.

Usage::

    python context_crops.py backfill --data-dir /data/retrain-2026-10 --label_type NoCurbRamp \\
        --pano-store /path/to/Panoramas --rawlabels-cache /data/rawlabels --verify 50
    # in code (the tagger's two-scale input, for example):
    info = context_crops.write_context(pano_rgb, x_norm, y_norm, crop_half_px, out_dir, "correct/seattle-wa_123")
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

import data_generation as dg

WIDE_FACTOR = 4   # issue #14: "a 4x wider crop"
VIEW_PX = 448     # 2 x the 224 model input (32 x 32 DINOv2 patches of 14 px), so a 448 run needs no upsampling
INDEX_FILE = "index.csv"
RUN_FILE = "context_run.json"
INDEX_COLUMNS = ["label_uid", "wide_path", "view_path", "wide_half_px", "wide_clipped"]
VERIFY_MAX_MEAN_ABS = 12.0  # grey levels: a re-cut tight crop further than this from the stored one is a different crop


# ---------------------------------------------------------------------------------------------------------------------
# Images (pure functions of their inputs)
# ---------------------------------------------------------------------------------------------------------------------

def wide_half_px(crop_half_px, factor=WIDE_FACTOR, view_size=dg.VIEW_SIZE):
    """The wide crop's half-width: ``factor`` times the tight one, capped at the whole view; and whether the cap
    applied (the requested context did not fit). ``(half, clipped)``.

    >>> wide_half_px(100), wide_half_px(300), wide_half_px(300, factor=2), wide_half_px(256)
    ((400, False), (1024, True), (600, False), (1024, False))
    """
    want = int(int(crop_half_px) * factor)
    if want < 1:
        raise ValueError(f"wide half-width must be positive (crop_half_px {crop_half_px}, factor {factor})")
    cap = view_size // 2
    return min(want, cap), want > cap


def downsample_view(view_rgb, px=VIEW_PX):
    """The square view resized to ``px`` x ``px`` with ``cv2.INTER_AREA`` (an area filter: no aliasing on a 4.6x
    reduction, and a flat image stays flat).

    >>> downsample_view(np.full((2048, 2048, 3), 7, np.uint8)).shape
    (448, 448, 3)
    """
    import cv2
    h, w = view_rgb.shape[:2]
    if h != w:
        raise ValueError(f"the view must be square, got {w}x{h}")
    return cv2.resize(view_rgb, (px, px), interpolation=cv2.INTER_AREA)


def context_images(view_rgb, crop_half_px, factor=WIDE_FACTOR, view_px=VIEW_PX):
    """``(wide_rgb, small_view_rgb, wide_half_px, wide_clipped)`` from a rendered label-centred view."""
    half, clipped = wide_half_px(crop_half_px, factor, view_rgb.shape[0])
    wide, _, _ = dg.centre_crop(view_rgb, half)
    return wide, downsample_view(view_rgb, view_px), half, clipped


def save_context(view_rgb, crop_half_px, out_dir, stem, factor=WIDE_FACTOR, view_px=VIEW_PX):
    """Write ``<out_dir>/wide/<stem>.webp`` and ``<out_dir>/view/<stem>.webp`` from a rendered view; returns the index
    fields (paths relative to ``out_dir``)."""
    wide, small, half, clipped = context_images(view_rgb, crop_half_px, factor, view_px)
    rel = {"wide_path": f"wide/{stem}.webp", "view_path": f"view/{stem}.webp"}
    dg.save_webp(wide, os.path.join(out_dir, rel["wide_path"]))
    dg.save_webp(small, os.path.join(out_dir, rel["view_path"]))
    return {**rel, "wide_half_px": int(half), "wide_clipped": int(clipped)}


def write_context(pano_rgb, x_norm, y_norm, crop_half_px, out_dir, stem, *, camera_pitch=None, camera_roll=None,
                  apply_tilt=True, beta=dg.DEFAULT_TILT_BETA, device=None, factor=WIDE_FACTOR, view_px=VIEW_PX):
    """Render the label view (``data_generation.render_label_view``, the same call the training crop came from) and
    write its wide crop and downsampled view. Returns ``{"wide_path", "view_path", "wide_half_px", "wide_clipped"}``
    with paths relative to ``out_dir``. A pure function of its inputs: both renderers are deterministic.

    The view is re-rendered rather than handed over by ``create_perspective_crop``: the render is milliseconds on a
    GPU next to the depth inference, and the alternative threads a ``return_view`` through a function the serving
    parity tests pin. The backfill has to re-render anyway, so both paths share this one.
    """
    view, _ = dg.render_label_view(pano_rgb, x_norm, y_norm, camera_pitch, camera_roll, apply_tilt=apply_tilt,
                                   beta=beta, device=device)
    return save_context(view, crop_half_px, out_dir, stem, factor, view_px)


# ---------------------------------------------------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------------------------------------------------

def context_dir(base_dir):
    """Where a dataset's context images live: ``<base_dir>/context`` (``base_dir`` = ``<data>/<label_type>``)."""
    return os.path.join(base_dir, "context")


class ContextIndex:
    """Append-only, flushed-per-row ``<out_dir>/index.csv``, resumable; one row per ``label_uid``.

    On open an existing index is read; a row is kept only if both its files exist and, when ``keep`` is given, its
    uid is in ``keep``. ``data_generation.py`` passes the manifest's kept ok rows, so a label whose crop is about to be
    cut again (a changed class, a forced re-run, a crash between the two writes) loses its context row too and gets a
    fresh one: the context never outlives the crop it belongs to. The kept rows are rewritten atomically. Writing a
    uid that is already in the index raises: there are never two rows for one label.

    Example::

        with ContextIndex("CurbRamp/context") as ctx:
            if not ctx.has("seattle-wa:1"):
                ctx.write("seattle-wa:1", write_context(pano, x, y, 238, ctx.out_dir, "correct/seattle-wa_1"))
    """

    def __init__(self, out_dir, keep=None):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, INDEX_FILE)
        self.dropped = 0
        rows = []
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            old = pd.read_csv(self.path, dtype=str, keep_default_na=False)
            old = old.drop_duplicates("label_uid", keep="last")
            for row in old.to_dict("records"):
                complete = all(row.get(c) for c in INDEX_COLUMNS)  # a torn last line after a crash
                ok = complete and all(os.path.exists(os.path.join(out_dir, row[c])) for c in ("wide_path", "view_path"))
                if ok and (keep is None or row["label_uid"] in keep):
                    rows.append(row)
                else:
                    self.dropped += 1
        os.makedirs(out_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)
        self.uids = {r["label_uid"] for r in rows}
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")

    def has(self, label_uid):
        return label_uid in self.uids

    def __len__(self):
        return len(self.uids)

    def write(self, label_uid, info):
        if label_uid in self.uids:
            raise ValueError(f"{label_uid} is already in {self.path}")
        self._w.writerow({**info, "label_uid": label_uid})
        self._f.flush()
        self.uids.add(label_uid)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def index_sha256(base_dir):
    """Hex sha256 of ``<base_dir>/context/index.csv`` (provenance for a model trained on it)."""
    with open(os.path.join(context_dir(base_dir), INDEX_FILE), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_index(base_dir, require_uids=None):
    """``<base_dir>/context/index.csv`` as a DataFrame with absolute ``wide_path``/``view_path``. With
    ``require_uids``, raise ``ValueError`` naming the first missing ones when any uid has no row or a file of its
    row is gone (a training run must never silently drop rows)."""
    out_dir = context_dir(base_dir)
    path = os.path.join(out_dir, INDEX_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {path}: run `python context_crops.py backfill` (or data_generation.py "
                                "--context-crops) for this label type first")
    df = pd.read_csv(path, dtype={"label_uid": str, "wide_path": str, "view_path": str})
    if not df["label_uid"].is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    for c in ("wide_path", "view_path"):
        df[c] = [os.path.abspath(os.path.join(out_dir, p)) for p in df[c]]
    if require_uids is not None:
        want = list(dict.fromkeys(str(u) for u in require_uids))
        have = set(df["label_uid"])
        missing = [u for u in want if u not in have]
        sub = df[df["label_uid"].isin(set(want))]
        gone = [u for u, a, b in zip(sub["label_uid"], sub["wide_path"], sub["view_path"])
                if not (os.path.exists(a) and os.path.exists(b))]
        if missing or gone:
            raise ValueError(f"{path}: {len(missing)} of {len(want)} required labels have no context row "
                             f"(e.g. {missing[:5]}) and {len(gone)} have a missing file (e.g. {gone[:5]}); run "
                             "context_crops.py backfill for this label type")
    return df


def write_run_file(out_dir, settings):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, RUN_FILE), "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)
        f.write("\n")


def read_run_file(out_dir):
    path = os.path.join(out_dir, RUN_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# Backfill (an existing dataset)
# ---------------------------------------------------------------------------------------------------------------------

#: Manifest pose fields the backfill cross-checks against the pose it recomputes (present from #72 on).
POSE_CHECK = ("pose_source", "pose_pitch", "pose_roll", "click_lat_offset_deg")


def _num(v):
    """A manifest cell as a float, None when blank.

    >>> _num(""), _num("2.5"), _num(None)
    (None, 2.5, None)
    """
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    f = float(v)
    return None if np.isnan(f) else f


def pose_mismatch(row, pose, tilt_beta):
    """Why ``pose`` (``crop_pose.pose_for_row`` now) differs from the pose ``row`` (a manifest row, strings) was
    cut under, or None. Rows from before #72 have no ``pose_source``: then only the provider (GSV, the only one
    then) and the recorded ``tilt_beta`` are checked, and the caller also compares ``tilt_applied``."""
    if row.get("pose_source"):
        if row["pose_source"] != pose.pose_source:
            return f"pose_source {row['pose_source']} -> {pose.pose_source}"
        for col, now in (("pose_pitch", pose.camera_pitch), ("pose_roll", pose.camera_roll),
                         ("click_lat_offset_deg", pose.click_lat_offset_deg)):
            old = _num(row.get(col))
            now = None if now is None else float(now)
            if col == "click_lat_offset_deg":
                old, now = old or 0.0, now or 0.0
            if (old is None) != (now is None) or (old is not None and abs(old - now) > 1e-6):
                return f"{col} {old} -> {now}"
    elif str(row.get("pano_source") or "gsv").strip().lower() != "gsv":
        return f"no recorded pose for a {row.get('pano_source')} row (made before #72?)"
    old_beta = _num(row.get("tilt_beta"))
    now_beta = pose.beta if pose.beta is not None else dg.row_beta(row, tilt_beta)
    if old_beta is not None and now_beta is not None and abs(old_beta - float(now_beta)) > 1e-9:
        return f"tilt_beta {old_beta} -> {now_beta}"
    return None


def label_points(manifest, cache_dir):
    """``manifest`` (ok rows) joined with ``pano_x, pano_y, pano_width, pano_height`` from the rawLabels cache
    (``<cache_dir>/<city>.csv``; never downloaded), on ``label_uid``. Raises on a missing CSV or label."""
    parts = []
    for city in sorted(manifest["city"].unique()):
        path = os.path.join(cache_dir, f"{city}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"--rawlabels-cache {cache_dir} has no {city}.csv; the backfill never downloads")
        raw = dg.load_city(city, None, cache_dir)
        raw = raw[["label_id", "pano_x", "pano_y", "pano_width", "pano_height"]].copy()
        raw["label_uid"] = city + ":" + raw["label_id"].astype(str)
        parts.append(raw.drop(columns="label_id"))
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["label_uid"].isin(set(manifest["label_uid"]))]
    out = manifest.merge(raw, on="label_uid", how="left", validate="one_to_one")
    missing = out["pano_x"].isna() | out["pano_width"].isna()
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} manifest rows have no point in the rawLabels cache (e.g. "
                         f"{out.loc[missing, 'label_uid'].head(3).tolist()}): is it the snapshot the dataset was "
                         "built from?")
    return out


def _settings_sha256(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def backfill(args):
    """``context_crops.py backfill``: the context images of every ok manifest row not yet in the index."""
    import crop_pose
    from PIL import Image
    base_dir = os.path.join(args.data_dir, args.label_type)
    out_dir = context_dir(base_dir)
    with open(os.path.join(base_dir, "generation_run.json")) as f:
        run = json.load(f)
    gen_settings = run.get("settings", {})
    tilt_beta = run.get("tilt_beta", gen_settings.get("tilt_beta", dg.DEFAULT_TILT_BETA))
    recorded = gen_settings.get("context_crops")
    if recorded and (recorded.get("factor"), recorded.get("view_px")) != (args.factor, VIEW_PX):
        raise SystemExit(f"generation_run.json records context crops at {recorded}; this backfill would write "
                         f"factor {args.factor}, view_px {VIEW_PX}")
    device = dg.resolve_device(args.allow_cpu)
    renderer = "torch" if device else "numpy"
    if gen_settings.get("renderer") and gen_settings["renderer"] != renderer:
        print(f"WARNING: the crops were rendered with {gen_settings['renderer']}, this backfill renders with "
              f"{renderer}; the views differ by rounding (use --verify)")
    settings = {"factor": args.factor, "view_px": VIEW_PX, "renderer": renderer,
                "generation_settings_sha256": _settings_sha256(gen_settings), "package_source": dg.package_source()}
    old = read_run_file(out_dir)
    if old is not None:
        changed = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if changed:
            raise SystemExit(f"{os.path.join(out_dir, RUN_FILE)} was written under other settings {changed}; "
                             f"move {out_dir} aside to start over (one index never mixes two geometries)")
    rotations = None
    if gen_settings.get("mapillary_api"):
        cache = os.path.join(base_dir, crop_pose.MAPILLARY_CACHE_FILE)
        with open(cache) as f:
            rotations = {k: v.get("computed_rotation") for k, v in json.load(f).items()}

    manifest = pd.read_csv(os.path.join(base_dir, "manifest.csv"), dtype=str, keep_default_na=False)
    manifest = manifest[manifest["status"] == "ok"]
    ctx = ContextIndex(out_dir)
    todo = manifest[~manifest["label_uid"].isin(ctx.uids)]
    todo = todo.sort_values("pano_id", kind="mergesort")  # the main loop's order: one pano read per pano
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(manifest)} ok rows, {len(ctx)} already in {ctx.path}, {len(todo)} to do"
          + (f" (--limit {args.limit})" if args.limit is not None else ""))
    write_run_file(out_dir, {**settings, "written": (old or {}).get("written", date.today().isoformat())})
    if todo.empty:
        ctx.close()
        return 0
    todo = label_points(todo, args.rawlabels_cache)
    verify = []
    failures = []
    start = time.time()
    last = (None, None)
    with ctx:
        for n, row in enumerate(todo.to_dict("records"), start=1):
            key = (row["city"], row["pano_id"])
            if last[0] != key:
                read, _ = dg.read_store_panorama(args.pano_store, *key)
                last = (key, read if isinstance(read, str) else np.ascontiguousarray(read[..., ::-1]))
            if isinstance(last[1], str):  # no_pano / bad_pano now, although the crop was cut from it
                failures.append((row["label_uid"], last[1]))
                continue
            pose = crop_pose.pose_for_row(row, tilt_beta, rotations)
            why = pose_mismatch(row, pose, tilt_beta)
            if why:
                raise SystemExit(f"{row['label_uid']}: the manifest's pose differs from the one recomputed now "
                                 f"({why}); the crops were cut under another pose, so a backfill would not match them")
            x_norm = float(row["pano_x"]) / float(row["pano_width"])
            y_norm = float(row["pano_y"]) / float(row["pano_height"]) + (pose.click_lat_offset_deg or 0.0) / 180.0
            view, tilt_applied = dg.render_label_view(last[1], x_norm, y_norm, pose.camera_pitch, pose.camera_roll,
                                                      apply_tilt=pose.beta is not None, beta=pose.beta or 0.0,
                                                      device=device)
            if row.get("tilt_applied") not in (None, "") and int(float(row["tilt_applied"])) != int(tilt_applied):
                raise SystemExit(f"{row['label_uid']}: tilt_applied {row['tilt_applied']} in the manifest, "
                                 f"{int(tilt_applied)} now; the crops were cut under another pose")
            crop_half = int(float(row["crop_half_px"]))
            stem = os.path.splitext(row["crop_path"])[0]
            ctx.write(row["label_uid"], save_context(view, crop_half, out_dir, stem, args.factor, VIEW_PX))
            if len(verify) < args.verify:
                tight, _, _ = dg.centre_crop(view, crop_half)
                stored = np.asarray(Image.open(os.path.join(base_dir, row["crop_path"])).convert("RGB"))
                if stored.shape != tight.shape:
                    verify.append((row["label_uid"], float("inf"), float("inf")))
                else:
                    d = np.abs(stored.astype(np.int16) - tight.astype(np.int16))
                    verify.append((row["label_uid"], float(d.max()), float(d.mean())))
            if n % 500 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {time.time() - start:.0f} s")
    print(f"index: {len(ctx)} rows in {ctx.path}")
    status = 0
    if verify:
        worst = max(v[2] for v in verify)
        print(f"verify: {len(verify)} re-cut tight crops vs the stored WebPs: mean abs diff "
              f"{np.mean([v[2] for v in verify]):.2f} (worst {worst:.2f}), max abs diff "
              f"{max(v[1] for v in verify):.0f} grey levels")
        if worst > VERIFY_MAX_MEAN_ABS:
            bad = [v[0] for v in verify if v[2] > VERIFY_MAX_MEAN_ABS]
            print(f"verify FAILED: {len(bad)} crops differ by more than {VERIFY_MAX_MEAN_ABS} grey levels on "
                  f"average (e.g. {bad[:3]}): the re-rendered views are not the ones the crops came from",
                  file=sys.stderr)
            status = 1
    if failures:
        print(f"{len(failures)} labels skipped, pano unreadable now (e.g. {failures[:3]}); re-run when the store "
              "is complete", file=sys.stderr)
        status = 1
    return status


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="write the context images of an existing dataset (crops untouched)")
    b.add_argument("--data-dir", required=True, help="data_generation.py's --out (holds <label_type>/manifest.csv)")
    b.add_argument("--label_type", required=True)
    b.add_argument("--pano-store", required=True, help="the pano store the crops were cut from (or a staged copy)")
    b.add_argument("--rawlabels-cache", required=True,
                   help="the rawLabels snapshot the dataset was built from (<city_id>.csv; the manifest has no "
                        "pano_x/pano_y); never downloaded")
    b.add_argument("--factor", type=int, default=WIDE_FACTOR, help="wide crop = factor x the tight half-width")
    b.add_argument("--limit", type=int, default=None, help="process at most N rows (smoke tests)")
    b.add_argument("--verify", type=int, default=0,
                   help=f"re-cut the tight crop of the first N rows from the re-rendered view and compare with the "
                        f"stored WebP; exit 1 above {VERIFY_MAX_MEAN_ABS} grey levels mean abs difference")
    b.add_argument("--allow-cpu", action="store_true", help=dg.DEPTH_DEVICE_HELP)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "backfill":
        args.pano_store = os.path.abspath(args.pano_store)
        dg.check_pano_store_marker(args.pano_store)
        return backfill(args)
    return 2  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":
    sys.exit(main())
