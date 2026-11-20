"""Backbone comparison (issue #119): the read-only data view the klone runs train on, and the report that reads their
records against the pre-registered rule of docs/backbone-2026-11.md.

``view`` (klone login node) builds ``<dst>/<LT>/`` from the round-3 data without writing under it: the small files
(``split.csv``, ``manifest.csv``, ``generation_run.json``, the round-3 ``metrics*.json`` / ``predictions*.csv``, and
``deployment_priors.json``) are copied as real files and their sha256 checked against ``analysis/backbone_inputs.json``
before and after the copy; only the crop directories ``correct/`` and ``incorrect/`` are symlinks. ``evaluate.py``
then writes its tagged outputs into the view, never into the round-3 directory. ``view --smoke`` writes a seeded,
class-balanced subsample of CurbRamp for the GPU smoke test, and ``backbone_inputs_smoke.json`` with its hash.

``report`` (laptop, after ``scp -r`` of the klone ``$KEEP``) reads every run under ``<runs-dir>/<LT>/runs/<TAG>/`` whose
tag starts with ``--tag-prefix`` (``bb-<key>`` at seed 42, ``bb-<key>-s<seed>``, ``bb-<key>-lr3e5``; ``smoke-<key>`` /
``smoke-rand-<key>`` in the smoke job). Each run must pass the run-consistency check (split hash equal to the pinned
one, seed / backbone / learning rate equal to its tag, real weights, loss divided by the accumulation steps), or it is
listed as INCOMPLETE and left out. Every arm is compared with the in-env baseline (``<prefix>dinov2-large``) of the
same seed on the same rows: paired AUC differences on test and held-out with panorama-bootstrap intervals
(``confident_learning.overall_deltas``), the paired within-city difference with a panorama bootstrap stratified by
deployment (``within_city_delta``), the 0.92 gate under the ``majority2`` prior (and ``margin`` as a reference, never
in the same table; ``prior_rules.gate_rows``), each held-out deployment, and every type x severity and type x tag
slice with its training count (``slices.paired_delta_table``). ``decide`` applies the rule mechanically.

Outputs (``--out``): ``arms.md/json``, ``seeds.md``, ``gate.md``, ``slices.md``, ``cost.md``, ``env.md``, ``lr.md``,
``rampnet.md``, ``decision.md/json``, ``null.md``, ``provenance.json``. Only aggregate numbers and held-out deployment
names that the public-city list allows are written (private ones lettered), so ``--out`` can be in the repo.

Usage::

    python analysis/backbone_comparison.py view --src $R --dst $OUT/data --inputs analysis/backbone_inputs.json \\
        --label-types CurbRamp,SurfaceProblem
    python analysis/backbone_comparison.py view --src $R --dst $OUT/smoke/data --inputs analysis/backbone_inputs.json \\
        --smoke --smoke-inputs-out $OUT/smoke/backbone_inputs_smoke.json
    python analysis/backbone_comparison.py report --split-dir ~/Data/.../round3 --runs-dir ~/Data/backbone-keep \\
        --shares docs/deployment-prior-2026-11/shares.json --public-cities cities.json \\
        --label-types CurbRamp,SurfaceProblem --out docs/backbone-2026-11
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backbones as bb  # noqa: E402

PILOT_TYPES = ("CurbRamp", "SurfaceProblem")
SEEDS = (42, 43, 44)
N_SEEDS = len(SEEDS)
DEFAULT_LR = 1e-5
ALT_LR = 3e-5
# The pre-registered rule (docs/backbone-2026-11.md): two standard errors of a three-seed mean, from the round-3
# baseline's seed sd of held-out AUC (0.006, 0.010) times sqrt(2) for a paired difference, floored at 0.010.
BASE_SEED_SD = {"CurbRamp": 0.006, "SurfaceProblem": 0.010}
DELTA_SD = {t: round(math.sqrt(2) * s, 4) for t, s in BASE_SEED_SD.items()}
THRESHOLDS = {"CurbRamp": 0.010, "SurfaceProblem": 0.016}
NON_INFERIOR = -0.005
GATE_PRECISION = 0.92
GATE_PRECISION_DROP = 0.01
GATE_COVERAGE_DROP = 0.02
DECISION_RULE = "majority2"
REFERENCE_RULE = "margin"
WITHIN_MIN_N = 10
BASELINE_KEY = bb.DEFAULT
CONTROL_KEY = "convnextv2-base"
RAMPNET_KEY = "rampnet-convnextv2-base"
VIEW_FILES = ("split.csv", "manifest.csv", "generation_run.json", "metrics.json", "predictions.csv",
              "metrics_baseline-s43.json", "predictions_baseline-s43.csv", "metrics_baseline-s44.json",
              "predictions_baseline-s44.csv")
CROP_DIRS = ("correct", "incorrect")
ROUND3_REF = {42: "", 43: "_baseline-s43", 44: "_baseline-s44"}   # round-3 baseline file suffix per seed
KEEP_FILES = ("metrics_{tag}.json", "predictions_{tag}.csv", "training_run.json", "run_key.json")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(path):
    """``analysis/backbone_inputs.json``: ``files`` (path under the round-3 directory -> sha256) and
    ``split_sha256`` (label type -> the split hash every run's ``run_key.json`` must carry)."""
    with open(path) as f:
        d = json.load(f)
    if "files" not in d or "split_sha256" not in d:
        raise ValueError(f"{path}: needs 'files' and 'split_sha256'")
    return d


# ---------------------------------------------------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------------------------------------------------

TAG_RE = re.compile(r"^(?P<rand>rand-)?(?P<key>.+?)(?:-lr(?P<lr>\d+e\d+))?(?:-s(?P<seed>\d+))?$")


def make_tag(prefix, key, seed=42, lr=DEFAULT_LR, rand=False):
    """The run tag: ``<prefix>[rand-]<key>[-lr3e5][-s<seed>]`` (seed 42 and lr 1e-5 carry no suffix).

    >>> make_tag("bb-", "convnextv2-base", 43), make_tag("bb-", "dinov2-large", lr=3e-5), make_tag("smoke-", "dinov3-vitl16", rand=True)
    ('bb-convnextv2-base-s43', 'bb-dinov2-large-lr3e5', 'smoke-rand-dinov3-vitl16')
    """
    tag = f"{prefix}{'rand-' if rand else ''}{key}"
    if not math.isclose(lr, DEFAULT_LR):
        mant, exp = f"{lr:.0e}".split("e")
        tag += f"-lr{int(mant)}e{abs(int(exp))}"
    if seed != 42:
        tag += f"-s{seed}"
    return tag


def parse_tag(tag, prefix):
    """``{"key", "seed", "lr", "random"}`` of a tag made by ``make_tag``, or None when it does not start with
    ``prefix`` or names no registry key.

    >>> parse_tag("bb-rampnet-convnextv2-base-lr3e5", "bb-")
    {'key': 'rampnet-convnextv2-base', 'seed': 42, 'lr': 3e-05, 'random': False}
    >>> parse_tag("bb-dinov2-reg-large-s44", "bb-")["seed"], parse_tag("bb-nope", "bb-")
    (44, None)
    """
    if not tag.startswith(prefix):
        return None
    m = TAG_RE.match(tag[len(prefix):])
    if not m or m["key"] not in bb.REGISTRY:
        return None
    lr = DEFAULT_LR
    if m["lr"]:
        mant, exp = m["lr"].split("e")
        lr = float(f"{mant}e-{exp}")
    return {"key": m["key"], "seed": int(m["seed"]) if m["seed"] else 42, "lr": lr, "random": bool(m["rand"])}


# ---------------------------------------------------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------------------------------------------------

def _inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def copy_verified(src, dst, sha):
    """Copy ``src`` to ``dst`` as a real file after checking ``src``'s sha256 against ``sha``, then check the copy;
    a ``dst`` that already holds the right bytes is left alone. ``ValueError`` on any mismatch, and on a ``dst``
    that is a symlink (a write through it would land in the source)."""
    if os.path.islink(dst):
        raise ValueError(f"{dst} is a symlink; the view holds real copies only (remove it and rebuild)")
    if os.path.isfile(dst) and sha256_of(dst) == sha:
        return False
    got = sha256_of(src)
    if got != sha:
        raise ValueError(f"{src}: sha256 {got} != pinned {sha} (analysis/backbone_inputs.json); refusing to build the view")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if sha256_of(dst) != sha:
        raise ValueError(f"{dst}: the copy's sha256 differs from {src}'s")
    return True


def link_crops(src_lt, dst_lt, split):
    """Symlink ``correct/`` and ``incorrect/`` of ``src_lt`` into ``dst_lt``; refuses a split whose ``crop_path``
    is absolute or lives elsewhere."""
    tops = {str(p).split("/")[0] for p in split["crop_path"]}
    if not tops <= set(CROP_DIRS) or any(os.path.isabs(str(p)) for p in split["crop_path"]):
        raise ValueError(f"{src_lt}/split.csv has crop paths outside {CROP_DIRS}: {sorted(tops - set(CROP_DIRS))[:3]}")
    for d in CROP_DIRS:
        link, target = os.path.join(dst_lt, d), os.path.join(src_lt, d)
        if os.path.islink(link):
            if os.path.realpath(link) != os.path.realpath(target):
                raise ValueError(f"{link} points to {os.readlink(link)}, not {target}")
            continue
        if os.path.exists(link):
            raise ValueError(f"{link} exists and is not a symlink")
        os.symlink(os.path.abspath(target), link)


def smoke_subsample(split, counts=None, seed=0):
    """A seeded subsample of ``split`` with ``counts`` rows per split (default train 320, val 160, test 160,
    test_heldout 160), half of each class, in the original row order.

    >>> s = pd.DataFrame({"split": ["train"] * 8 + ["val"] * 4, "label_class": ["correct", "incorrect"] * 6})
    >>> smoke_subsample(s, {"train": 4, "val": 2}).groupby(["split", "label_class"]).size().tolist()
    [2, 2, 1, 1]
    """
    counts = counts or {"train": 320, "val": 160, "test": 160, "test_heldout": 160}
    rng = np.random.default_rng(seed)
    keep = []
    for sp, n in counts.items():
        for cls in ("correct", "incorrect"):
            idx = split.index[(split["split"] == sp) & (split["label_class"] == cls)].to_numpy()
            if len(idx) < n // 2:
                raise ValueError(f"{sp}/{cls}: {len(idx)} rows, need {n // 2}")
            keep.append(rng.choice(idx, n // 2, replace=False))
    return split.loc[np.sort(np.concatenate(keep))]


def cmd_view(args):
    inputs = load_inputs(args.inputs)
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if _inside(dst, src):
        raise SystemExit(f"--dst {dst} is inside --src {src}: the view must not write under the round-3 data")
    files = inputs["files"]
    types = [t for t in args.label_types.split(",") if t]
    copy_verified(os.path.join(src, "deployment_priors.json"), os.path.join(dst, "deployment_priors.json"),
                  files["deployment_priors.json"])
    for lt in types:
        s_lt, d_lt = os.path.join(src, lt), os.path.join(dst, lt)
        if f"{lt}/split.csv" not in files:
            raise SystemExit(f"{lt}/split.csv is not pinned in {args.inputs}")
        if args.smoke:
            copy_verified(os.path.join(s_lt, "split.csv"), os.path.join(args.scratch or dst, ".src_split", lt, "split.csv"),
                          files[f"{lt}/split.csv"])
            full = pd.read_csv(os.path.join(s_lt, "split.csv"), dtype={"pano_id": str})
            sub = smoke_subsample(full, seed=args.seed)
            os.makedirs(d_lt, exist_ok=True)
            out = os.path.join(d_lt, "split.csv")
            if os.path.islink(out):
                raise SystemExit(f"{out} is a symlink")
            sub.to_csv(out, index=False)
            copy_verified(os.path.join(s_lt, "generation_run.json"), os.path.join(d_lt, "generation_run.json"),
                          files[f"{lt}/generation_run.json"])
            smoke = {"files": {}, "split_sha256": {lt: sha256_of(out)}, "source_split_sha256": files[f"{lt}/split.csv"],
                     "seed": args.seed, "counts": sub.groupby(["split", "label_class"]).size().rename("n").reset_index()
                     .to_dict("records")}
            with open(args.smoke_inputs_out, "w") as f:
                json.dump(smoke, f, indent=2)
                f.write("\n")
            print(f"smoke view {d_lt}: {len(sub)} rows, split sha256 {smoke['split_sha256'][lt][:12]}")
        else:
            for name in VIEW_FILES:
                rel = f"{lt}/{name}"
                if rel in files:
                    copy_verified(os.path.join(s_lt, name), os.path.join(d_lt, name), files[rel])
            sub = pd.read_csv(os.path.join(d_lt, "split.csv"), usecols=["crop_path"])
            print(f"view {d_lt}: {sum(f.startswith(lt + '/') for f in files)} files checked")
        link_crops(s_lt, d_lt, sub)
    bad = [p for p in _all_links(dst) if os.path.basename(p) not in CROP_DIRS]
    if bad:
        raise SystemExit(f"the view holds symlinks other than the crop directories: {bad[:3]}")
    return 0


def _all_links(root):
    out = []
    for d, dirs, files in os.walk(root):
        out += [os.path.join(d, n) for n in dirs + files if os.path.islink(os.path.join(d, n))]
        dirs[:] = [n for n in dirs if not os.path.islink(os.path.join(d, n))]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------------

def fast_auc(y, s):
    """ROC AUC via ranks (ties averaged); NaN with one class. Equal to ``sklearn.metrics.roc_auc_score``.

    >>> fast_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    0.75
    """
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _within(y, a, b, codes, groups, min_n):
    """Within-city AUC of scores ``a`` and ``b`` over the index arrays in ``groups`` (one per city): per-city AUCs
    weighted by n_incorrect x n_correct over cities with both classes and at least ``min_n`` rows."""
    num_a = num_b = den = 0.0
    for idx in groups:
        yy = y[idx]
        n1 = int(yy.sum())
        if len(idx) < min_n or n1 == 0 or n1 == len(idx):
            continue
        w = n1 * (len(idx) - n1)
        num_a += w * fast_auc(yy, a[idx])
        num_b += w * fast_auc(yy, b[idx])
        den += w
    return (num_a / den, num_b / den) if den else (float("nan"), float("nan"))


def within_city_delta(pa, pb, split, n_boot=1000, seed=0, min_n=WITHIN_MIN_N):
    """Paired within-city AUC difference of model A (``pa``) minus model B (``pb``) on ``split``'s rows, with a
    95% interval from a panorama bootstrap stratified by deployment: panoramas are resampled with replacement within
    each deployment and both models are scored on the same resample. ``pa`` carries ``city`` and ``pano_id``; both
    carry ``label_uid, true, p_incorrect``. ``{"a", "b", "delta", "delta_lo", "delta_hi", "n_cities"}``."""
    from analysis import retrain_report as rr
    a = pa[pa["split"] == split]
    if a.empty:
        return {"a": None, "b": None, "delta": None, "delta_lo": None, "delta_hi": None, "n_cities": 0}
    m = a[["label_uid", "true", "p_incorrect", "city", "pano_id"]].merge(
        pb[["label_uid", "p_incorrect"]], on="label_uid", how="left", suffixes=("", "_b"), validate="one_to_one")
    if m["p_incorrect_b"].isna().any():
        raise ValueError(f"model B has no score for {int(m['p_incorrect_b'].isna().sum())} of model A's {split} rows")
    y = (m["true"] == "incorrect").to_numpy()
    sa, sb = m["p_incorrect"].to_numpy(float), m["p_incorrect_b"].to_numpy(float)
    wa, n = rr.within_city_auc(y, sa, m["city"], min_n)
    wb, _ = rr.within_city_auc(y, sb, m["city"], min_n)
    out = {"a": _num(wa), "b": _num(wb), "delta": _num(wa - wb), "delta_lo": None, "delta_hi": None, "n_cities": int(n)}
    if n_boot and n:
        rng = np.random.default_rng(seed)
        cities = m["city"].to_numpy()
        panos = m["pano_id"].astype(str).to_numpy()
        per_city = []
        for c in np.unique(cities):
            rows = np.flatnonzero(cities == c)
            u, inv = np.unique(panos[rows], return_inverse=True)
            per_city.append([rows[inv == g] for g in range(len(u))])
        vals = []
        for _ in range(n_boot):
            groups = []
            for members in per_city:
                pick = rng.integers(0, len(members), len(members))
                groups.append(np.concatenate([members[i] for i in pick]))
            ba, bb_ = _within(y, sa, sb, None, groups, min_n)
            if not (np.isnan(ba) or np.isnan(bb_)):
                vals.append(ba - bb_)
        if vals:
            out["delta_lo"], out["delta_hi"] = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return out


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def run_metrics(pred):
    """Test and held-out AUC, within-city AUC (round-3 definition), macro F1 on test, and the incorrect class's
    recall at precision 0.92 and 0.95 on test, from one joined predictions frame."""
    from sklearn.metrics import f1_score

    from analysis import retrain_report as rr
    from evaluate import recall_at_precision
    out = {}
    for split, key in (("test", "test"), ("test_heldout", "heldout")):
        p = pred[pred["split"] == split]
        if p.empty:
            continue
        y = (p["true"] == "incorrect").to_numpy()
        out[f"{key}_auc"] = _num(fast_auc(y, p["p_incorrect"].to_numpy(float)))
        wc, n = rr.within_city_auc(y, p["p_incorrect"], p["city"], WITHIN_MIN_N)
        out[f"wc_{key}_auc"], out[f"{key}_cities"] = _num(wc), int(n)
        if split == "test":
            yhat = np.where(p["p_incorrect"] > p["p_correct"], "incorrect", "correct")
            out["macro_f1"] = float(f1_score(p["true"], yhat, average="macro"))
            for level in (0.92, 0.95):
                out[f"recall_incorrect_p{int(level * 100)}"] = float(recall_at_precision(y, p["p_incorrect"], level)[0])
    return out


def gate_stats(pred, shares_types, label_type, rule):
    """Agree ('correct') and Disagree ('incorrect') coverage and precision at the 0.92 gate on test and held-out,
    under ``rule``'s prior (``prior_rules.gate_rows``: the uncapped map fitted on val)."""
    from analysis import prior_rules as pr
    cols = ["label_uid", "split", "true", "p_correct", "p_incorrect"]
    return pr.gate_rows(pred[cols], pr.rule_prior(shares_types, label_type, rule), splits=("test", "test_heldout"))


# ---------------------------------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------------------------------

def _read_json(path):
    with open(path) as f:
        return json.load(f)


class Tracker:
    """Records the sha256 of every file the report reads (``provenance.json``)."""

    def __init__(self):
        self.files = {}

    def track(self, path):
        self.files[os.path.abspath(path)] = sha256_of(path)
        return path


def consistency(run_dir, tag, parsed, split_sha, epochs=None):
    """Reasons a run cannot be used (empty when it can): files missing, ``run_key.json``'s split hash differs from
    the pinned one, the recorded seed / backbone / learning rate differ from the tag, the recorded epochs differ from
    ``epochs`` (when given; a shell with ``EPOCHS=1`` exported once submitted one-epoch runs under main tags), random
    weights, or a loss not divided by the accumulation steps."""
    missing = [n.format(tag=tag) for n in KEEP_FILES if not os.path.isfile(os.path.join(run_dir, n.format(tag=tag)))]
    if missing:
        return [f"missing {', '.join(missing)}"]
    key, run = _read_json(os.path.join(run_dir, "run_key.json")), _read_json(os.path.join(run_dir, "training_run.json"))
    why = []
    if key.get("split_sha256") != split_sha:
        why.append(f"split_sha256 {str(key.get('split_sha256'))[:12]} is not the pinned {str(split_sha)[:12]}")
    if run.get("seed") != parsed["seed"]:
        why.append(f"seed {run.get('seed')} != tag's {parsed['seed']}")
    backbone = run.get("backbone") or {}
    if backbone.get("key") != parsed["key"]:
        why.append(f"backbone {backbone.get('key')} != tag's {parsed['key']}")
    lr = run.get("learning_rate")
    if lr is None or not math.isclose(lr, parsed["lr"]):
        why.append(f"learning_rate {lr} != tag's {parsed['lr']}")
    if epochs is not None and run.get("epochs") != epochs:
        why.append(f"epochs {run.get('epochs')} != {epochs}")
    if backbone.get("random_init") is not False:
        why.append("random_init (smoke test weights)")
    if run.get("model_accepts_loss_kwargs") is not False:
        why.append(f"model_accepts_loss_kwargs {run.get('model_accepts_loss_kwargs')} (loss not divided by the steps)")
    return why


def load_split(split_dir, label_type, inputs, tracker):
    """``split.csv`` of ``label_type`` after checking its hash against the pinned one."""
    path = os.path.join(split_dir, label_type, "split.csv")
    got = sha256_of(path)
    want = inputs["split_sha256"].get(label_type)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got[:12]} is not the pinned {str(want)[:12]}; the report must read the "
                         "split the runs trained on")
    tracker.track(path)
    return pd.read_csv(path, dtype={"pano_id": str, "label_uid": str})


JOIN = ["label_uid", "city", "pano_id", "severity", "tags"]


def join_split(pred, split):
    """Predictions joined with the split's city, panorama and slice columns on ``label_uid`` (one to one)."""
    m = pred.merge(split[JOIN], on="label_uid", how="left", validate="one_to_one")
    if m["city"].isna().any():
        raise ValueError(f"{int(m['city'].isna().sum())} predictions have no split.csv row")
    return m


def load_runs(runs_dir, label_type, prefix, split, split_sha, tracker, epochs=None):
    """Every run of ``label_type`` under ``<runs_dir>/<LT>/runs/`` whose tag starts with ``prefix``: a dict per tag
    with ``parsed``, ``status`` (``complete`` / ``INCOMPLETE``), ``why``, and for complete runs the joined
    predictions and run records."""
    base = os.path.join(runs_dir, label_type, "runs")
    out = {}
    eval_rows = int(split["split"].isin(["val", "test", "test_heldout"]).sum())
    for tag in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        parsed = parse_tag(tag, prefix)
        if parsed is None:
            continue
        d = os.path.join(base, tag)
        why = consistency(d, tag, parsed, split_sha, epochs)
        rec = {"tag": tag, "parsed": parsed, "dir": d, "why": why}
        if not why:
            pred = pd.read_csv(tracker.track(os.path.join(d, f"predictions_{tag}.csv")),
                               dtype={"label_uid": str, "split": str, "true": str})
            if len(pred) != eval_rows:
                why.append(f"{len(pred)} predictions, the split has {eval_rows} evaluation rows")
            elif not np.isfinite(pred[["p_correct", "p_incorrect"]].to_numpy(float)).all():
                why.append("non-finite scores")
            else:
                rec["pred"] = join_split(pred, split)
                rec["run"] = _read_json(tracker.track(os.path.join(d, "training_run.json")))
                ts = os.path.join(d, "trainer_state.json")
                rec["state"] = _read_json(tracker.track(ts)) if os.path.isfile(ts) else {}
                gpu = os.path.join(d, "gpu.txt")
                rec["gpu"] = sorted({ln.strip() for ln in open(gpu)} - {""}) if os.path.isfile(gpu) else []
        rec["status"] = "INCOMPLETE" if why else "complete"
        out[tag] = rec
    return out


def round3_reference(split_dir, label_type, split, tracker):
    """The round-3 baseline's predictions per seed (``predictions.csv`` = seed 42, ``predictions_baseline-s4x.csv``),
    joined with the split; seeds without a file are absent."""
    out = {}
    for seed, suffix in ROUND3_REF.items():
        path = os.path.join(split_dir, label_type, f"predictions{suffix}.csv")
        if os.path.isfile(path):
            out[seed] = join_split(pd.read_csv(tracker.track(path), dtype={"label_uid": str, "split": str, "true": str}),
                                   split)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# comparison rows
# ---------------------------------------------------------------------------------------------------------------------

def paired(arm_pred, base_pred, name, n_boot, seed):
    """Paired differences of an arm against a baseline on the same rows: AUC (test, held-out) via
    ``confident_learning.overall_deltas`` and within-city AUC via ``within_city_delta``."""
    from analysis import confident_learning as cl
    d = cl.overall_deltas({"baseline": base_pred, name: arm_pred}, [name], n_boot, seed)
    out = {}
    for split, key in (("test", "test"), ("test_heldout", "heldout")):
        r = d.get((name, split)) or {}
        out[f"d_{key}"], out[f"d_{key}_lo"], out[f"d_{key}_hi"] = r.get("delta"), r.get("delta_lo"), r.get("delta_hi")
        w = within_city_delta(arm_pred, base_pred, split, n_boot, seed)
        out[f"wc_d_{key}"], out[f"wc_d_{key}_lo"], out[f"wc_d_{key}_hi"] = w["delta"], w["delta_lo"], w["delta_hi"]
    return out


def seconds_per_step(rec):
    """Wall seconds per optimizer step of the run's last attempt (loading and evaluation included):
    ``finished_at - started_at`` over the steps that attempt ran (``max_steps`` minus the step it resumed from)."""
    from datetime import datetime
    run, state = rec.get("run") or {}, rec.get("state") or {}
    try:
        t0 = datetime.strptime(run["started_at"], "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.strptime(run["finished_at"], "%Y-%m-%dT%H:%M:%SZ")
        steps = int(state["max_steps"]) - (int(str(run.get("resumed_from") or "checkpoint-0").split("-")[1]))
    except (KeyError, ValueError, TypeError):
        return None
    return (t1 - t0).total_seconds() / steps if steps > 0 else None


def compare_type(label_type, runs, ref, shares_types, prefix, public, n_boot, seed, slice_seeds, train):
    """All rows for one label type: per-run metrics and deltas, gate stats, held-out deployments, slices."""
    from analysis import confident_learning as cl
    from analysis import slices as sl
    complete = {t: r for t, r in runs.items() if r["status"] == "complete"}
    by = {(r["parsed"]["key"], r["parsed"]["seed"], r["parsed"]["lr"]): r for r in complete.values()}
    rows, gates, heldout, slice_rows = [], [], [], []
    for (key, s, lr), rec in sorted(by.items(), key=lambda kv: (kv[0][0] != BASELINE_KEY, kv[0])):
        row = {"type": label_type, "arm": key, "seed": s, "lr": lr, "tag": rec["tag"], **run_metrics(rec["pred"])}
        base = by.get((BASELINE_KEY, s, DEFAULT_LR))
        if key != BASELINE_KEY or not math.isclose(lr, DEFAULT_LR):
            row["baseline_tag"] = base["tag"] if base else None
            if base:
                row.update(paired(rec["pred"], base["pred"], rec["tag"], n_boot, seed))
                if s in ref:   # the round-3 file is joined but the deltas use the in-env baseline
                    pass
                if not math.isclose(lr, DEFAULT_LR):
                    own = by.get((key, s, DEFAULT_LR))
                    if own:
                        row.update({f"vs_own_lr_{k}": v for k, v in paired(rec["pred"], own["pred"], rec["tag"],
                                                                           n_boot, seed).items()})
                    alt = by.get((BASELINE_KEY, s, lr))
                    if alt and key != BASELINE_KEY:
                        row.update({f"vs_base_same_lr_{k}": v for k, v in paired(rec["pred"], alt["pred"], rec["tag"],
                                                                                 n_boot, seed).items()})
                if key == RAMPNET_KEY:
                    ctl = by.get((CONTROL_KEY, s, lr))
                    if ctl:
                        row.update({f"vs_control_{k}": v for k, v in paired(rec["pred"], ctl["pred"], rec["tag"],
                                                                           n_boot, seed).items()})
                for h in cl.heldout_rows({"baseline": base["pred"], rec["tag"]: rec["pred"]}, public):
                    a, b_ = h.get(rec["tag"]), h.get("baseline")
                    heldout.append({"type": label_type, "arm": key, "seed": s, "lr": lr, "deployment": h["deployment"],
                                    "n": h["n"], "n_incorrect": h["n_incorrect"], "auc": a, "auc_baseline": b_,
                                    "delta": None if a is None or b_ is None else a - b_})
        backbone = rec["run"].get("backbone") or {}
        row.update(params=backbone.get("params"), tokens_per_image=backbone.get("tokens_per_image"),
                   model_class=backbone.get("model_class"), family=backbone.get("family"),
                   gpu=", ".join(rec["gpu"]), seconds_per_step=seconds_per_step(rec),
                   peak_gpu_memory_gb=(rec["run"].get("peak_gpu_memory_bytes") or 0) / 2**30 or None)
        for rule in (DECISION_RULE, REFERENCE_RULE):
            g = gate_stats(rec["pred"], shares_types, label_type, rule)
            for split, classes in g.items():
                for verdict, st in classes.items():
                    if isinstance(st, dict):
                        gates.append({"type": label_type, "arm": key, "seed": s, "lr": lr, "rule": rule, "split": split,
                                      "verdict": verdict, **st})
        if key != BASELINE_KEY and base and s in slice_seeds and math.isclose(lr, DEFAULT_LR):
            for split in ("test", "test_heldout"):
                part = rec["pred"][rec["pred"]["split"] == split]
                r3 = ref.get(s)
                for b_ in sl.BY:
                    t = sl.paired_delta_table(part, base["pred"], b_, sl.MIN_N, n_boot, seed)
                    tr = sl.with_slice(train, b_).groupby("slice").size()
                    r3auc = {}
                    if r3 is not None:
                        rt = sl.slice_table(r3[r3["split"] == split], b_, None, sl.MIN_N, 0, seed)
                        r3auc = dict(zip(rt["slice"], rt["auc"]))
                    for r in t.to_dict("records"):
                        slice_rows.append({"type": label_type, "arm": key, "seed": s, "split": split, "by": b_, **r,
                                           "train_n": int(tr.get(r["slice"], 0)),
                                           "thin": int(tr.get(r["slice"], 0)) < sl.THIN_TRAIN,
                                           "round3_auc": _num(r3auc.get(r["slice"]))})
        rows.append(row)
    return rows, gates, heldout, slice_rows


def env_rows(label_type, runs, ref, n_boot, seed):
    """The in-env baseline (``<prefix>dinov2-large``, transformers 4.56.2) against round 3's own baseline (4.55.0) at
    the same seed: AUC on test and held-out, and their paired difference."""
    out = []
    for rec in runs.values():
        p = rec["parsed"]
        if rec["status"] != "complete" or p["key"] != BASELINE_KEY or not math.isclose(p["lr"], DEFAULT_LR):
            continue
        if p["seed"] not in ref:
            continue
        d = paired(rec["pred"], ref[p["seed"]], rec["tag"], n_boot, seed)
        m, r3 = run_metrics(rec["pred"]), run_metrics(ref[p["seed"]])
        out.append({"type": label_type, "seed": p["seed"], "test_auc": m.get("test_auc"), "heldout_auc": m.get("heldout_auc"),
                    "round3_test_auc": r3.get("test_auc"), "round3_heldout_auc": r3.get("heldout_auc"), **d})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# summary and the rule
# ---------------------------------------------------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def _sd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


def summarise(rows, gates):
    """Per arm and type over the seeds at lr 1e-5: the inputs of ``decide``."""
    out = {}
    main = [r for r in rows if math.isclose(r["lr"], DEFAULT_LR)]
    agree = {(g["type"], g["arm"], g["seed"]): g for g in gates
             if g["rule"] == DECISION_RULE and g["split"] == "test" and g["verdict"] == "correct" and math.isclose(g["lr"], DEFAULT_LR)}
    for arm in sorted({r["arm"] for r in main}):
        out[arm] = {}
        for t in sorted({r["type"] for r in main}):
            rs = [r for r in main if r["arm"] == arm and r["type"] == t]
            bs = [r for r in main if r["arm"] == BASELINE_KEY and r["type"] == t]
            seeds = sorted(r["seed"] for r in rs if arm == BASELINE_KEY or r.get("d_heldout") is not None)
            gs = [agree.get((t, arm, s)) or {} for s in seeds]
            gb = [agree.get((t, BASELINE_KEY, s)) or {} for s in seeds]
            out[arm][t] = {
                "n_seeds": len(seeds), "seeds": seeds,
                "heldout_auc_mean": _mean([r.get("heldout_auc") for r in rs]), "heldout_auc_sd": _sd([r.get("heldout_auc") for r in rs]),
                "test_auc_mean": _mean([r.get("test_auc") for r in rs]),
                "d_heldout_mean": _mean([r.get("d_heldout") for r in rs]),
                "n_pos_heldout": sum(1 for r in rs if (r.get("d_heldout") or 0) > 0),
                "wc_d_heldout_mean": _mean([r.get("wc_d_heldout") for r in rs]),
                "d_test_mean": _mean([r.get("d_test") for r in rs]),
                "wc_d_test_mean": _mean([r.get("wc_d_test") for r in rs]),
                "agree_prec_mean": _mean([g.get("gate_precision") for g in gs]),
                "agree_cov_mean": _mean([g.get("gate_coverage") for g in gs]),
                "base_agree_prec_mean": _mean([g.get("gate_precision") for g in gb]),
                "base_agree_cov_mean": _mean([g.get("gate_coverage") for g in gb]),
                "params": next((r.get("params") for r in rs if r.get("params")), None),
                "base_params": next((r.get("params") for r in bs if r.get("params")), None),
            }
    return out


def _gate_ok(r):
    p, c, bp, bc = (r.get(k) for k in ("agree_prec_mean", "agree_cov_mean", "base_agree_prec_mean", "base_agree_cov_mean"))
    if None in (p, c, bp, bc):
        return False
    return p >= GATE_PRECISION and p >= bp - GATE_PRECISION_DROP and c >= bc - GATE_COVERAGE_DROP


def arm_status(arm, per_type, submitted=False, types=PILOT_TYPES):
    """The rule of docs/backbone-2026-11.md for one arm. ``per_type`` is ``summarise``'s entry for the arm.
    Returns ``(status, detail)``; statuses: ``winner``, ``meets the doubled rule`` (an arm outside ``ELIGIBLE``),
    ``between-deployment only``, ``worse``, ``non-inferior small``, ``null``, ``running``, ``not yet run``."""
    have = [(per_type.get(t) or {}).get("n_seeds", 0) for t in types]
    if any(h < N_SEEDS for h in have):
        return ("running" if submitted else "not yet run"), {"seeds": dict(zip(types, have))}
    eligible = arm in bb.ELIGIBLE
    thr = {t: THRESHOLDS[t] * (1 if eligible else 2) for t in types}
    pooled = [t for t in types if per_type[t]["d_heldout_mean"] is not None
              and per_type[t]["d_heldout_mean"] >= thr[t] and per_type[t]["n_pos_heldout"] == N_SEEDS]
    c1 = [t for t in pooled if (per_type[t]["wc_d_heldout_mean"] or 0) > 0]
    c2 = all(per_type[t][k] is not None and per_type[t][k] >= NON_INFERIOR for t in types
             for k in ("d_heldout_mean", "wc_d_heldout_mean", "d_test_mean"))
    c3 = all(_gate_ok(per_type[t]) for t in types)
    params, base_params = per_type[types[0]].get("params"), per_type[types[0]].get("base_params")
    small = bool(params and base_params and params < 0.5 * base_params)
    detail = {"thresholds": thr, "condition_1_types": c1, "pooled_bar_types": pooled, "condition_2": c2,
              "condition_3": c3, "non_inferior_small": small and c2 and c3, "eligible": eligible}
    if c1 and c2 and c3:
        return ("winner" if eligible else "meets the doubled rule"), detail
    if pooled and not c1:
        return "between-deployment only", detail
    if not c2:
        return "worse", detail
    if small and c3:
        return "non-inferior small", detail
    return "null", detail


def decide(summary, submitted=False, types=PILOT_TYPES):
    """Every arm's status and the recommendation: the eligible winner with the larger sum of mean Δ held-out AUC
    (within 0.005: one ``AutoModelForImageClassification`` loads with no custom code, then fewer parameters); an
    explanatory arm only if it meets the doubled rule and no eligible arm wins; else keep ``dinov2-large``."""
    arms = {}
    for arm, per_type in summary.items():
        if arm == BASELINE_KEY:
            continue
        status, detail = arm_status(arm, per_type, submitted, types)
        arms[arm] = {"status": status, **detail}
    for arm in bb.REGISTRY:
        if arm != BASELINE_KEY and arm not in arms and arm in bb.ELIGIBLE:
            arms[arm] = {"status": "running" if submitted and not bb.get(arm).gated else "not yet run"}

    def score(a):
        return sum(summary[a][t]["d_heldout_mean"] for t in types)

    def pick(cands):
        if not cands:
            return None
        best = max(score(a) for a in cands)
        close = [a for a in cands if score(a) >= best - 0.005]
        return sorted(close, key=lambda a: (bb.get(a).family == "dinov3_vit", bb.get(a).params_m, -score(a)))[0]
    winners = [a for a, r in arms.items() if r["status"] == "winner"]
    promoted = [a for a, r in arms.items() if r["status"] == "meets the doubled rule"]
    choice = pick(winners) or pick(promoted)
    pending = any(r["status"] in ("running", "not yet run") for a, r in arms.items() if a in summary or a in bb.ELIGIBLE)
    if choice:
        rec = f"{choice} (pending confirmation on the other three types)"
    elif pending:
        rec = "running" if submitted else "not yet run"
    else:
        rec = f"keep {BASELINE_KEY} (no arm met the rule)"
    return {"arms": arms, "winners": winners, "promoted": promoted, "recommendation": rec}


def null_winner_rate(delta_sd=None, thresholds=None, n_arms=len(bb.ELIGIBLE), n_seeds=N_SEEDS, n_sims=20000, seed=0,
                     types=PILOT_TYPES):
    """Monte Carlo of the rule under no true difference: each eligible arm's per-seed Δ held-out, Δ within-city
    held-out and Δ test are independent N(0, ``delta_sd[type]``) draws (the gate condition assumed met).
    ``{"expected_winners", "p_any_winner"}`` among ``n_arms`` arms."""
    delta_sd = delta_sd or DELTA_SD
    thresholds = thresholds or THRESHOLDS
    rng = np.random.default_rng(seed)
    wins = np.zeros(n_sims)
    for _ in range(n_arms):
        c1 = np.zeros(n_sims, bool)
        c2 = np.ones(n_sims, bool)
        for t in types:
            d = rng.normal(0, delta_sd[t], (n_sims, n_seeds))
            wc = rng.normal(0, delta_sd[t], (n_sims, n_seeds)).mean(1)
            te = rng.normal(0, delta_sd[t], (n_sims, n_seeds)).mean(1)
            c1 |= (d.mean(1) >= thresholds[t]) & (d > 0).all(1) & (wc > 0)
            c2 &= (d.mean(1) >= NON_INFERIOR) & (wc >= NON_INFERIOR) & (te >= NON_INFERIOR)
        wins += c1 & c2
    return {"expected_winners": float(wins.mean()), "p_any_winner": float((wins > 0).mean()), "n_arms": n_arms,
            "n_sims": n_sims, "seed": seed, "delta_sd": delta_sd, "thresholds": thresholds}


# ---------------------------------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------------------------------

def _f(v, fmt="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _d(r, k):
    v = r.get(k)
    if v is None:
        return ""
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    return f"{v:+.3f}" + ("" if lo is None else f" [{lo:+.3f}, {hi:+.3f}]")


def _table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |"
                                                                                 for r in rows]) + "\n"


def arms_markdown(rows, heldout, incomplete, round3):
    parts = ["# Backbone comparison: every run against the same-seed in-env baseline", "",
             "Δ = arm − in-env `dinov2-large` of the same seed on the same rows, with 95% panorama-bootstrap intervals "
             "(within-city: panoramas resampled within each deployment). Within-city AUC: per-deployment AUCs weighted "
             "by n_incorrect × n_correct over deployments with ≥ 10 labels of both classes.", ""]
    for t in sorted({r["type"] for r in rows} | {r["type"] for r in round3}):
        parts += [f"## {t}", ""]
        body = [[f"round-3 baseline (4.55.0)", r["seed"], _f(r.get("test_auc")), _f(r.get("wc_test_auc")),
                 _f(r.get("heldout_auc")), _f(r.get("wc_heldout_auc")), "", "", "", "", _f(r.get("macro_f1")),
                 _f(r.get("recall_incorrect_p92")), _f(r.get("recall_incorrect_p95"))] for r in round3 if r["type"] == t]
        for r in rows:
            if r["type"] != t or not math.isclose(r["lr"], DEFAULT_LR):
                continue
            body.append([f"`{r['arm']}`", r["seed"], _f(r.get("test_auc")), _f(r.get("wc_test_auc")),
                         _f(r.get("heldout_auc")), _f(r.get("wc_heldout_auc")), _d(r, "d_test"), _d(r, "wc_d_test"),
                         _d(r, "d_heldout"), _d(r, "wc_d_heldout"), _f(r.get("macro_f1")),
                         _f(r.get("recall_incorrect_p92")), _f(r.get("recall_incorrect_p95"))])
        parts.append(_table(["arm", "seed", "test AUC", "within-city test", "held-out AUC", "within-city held-out",
                             "Δ test", "Δ within-city test", "Δ held-out", "Δ within-city held-out", "macro F1 (test)",
                             "incorrect recall @ P 0.92", "@ P 0.95"], body))
        hb = [[f"`{h['arm']}`", h["seed"], h["deployment"], h["n"], h["n_incorrect"], _f(h["auc_baseline"]), _f(h["auc"]),
               _f(h["delta"], "{:+.3f}")] for h in heldout if h["type"] == t and math.isclose(h["lr"], DEFAULT_LR)]
        if hb:
            parts += ["### Each held-out deployment", "", _table(["arm", "seed", "deployment", "n", "incorrect",
                                                                   "baseline AUC", "arm AUC", "Δ"], hb)]
    if incomplete:
        parts += ["## INCOMPLETE runs (left out)", "", _table(["type", "tag", "why"], incomplete)]
    return "\n".join(parts)


def seeds_markdown(summary):
    body = []
    for arm, per in summary.items():
        for t, r in per.items():
            body.append([f"`{arm}`", t, r["n_seeds"], f"{_f(r['heldout_auc_mean'])} ± {_f(r['heldout_auc_sd'])}",
                         _f(r["test_auc_mean"]), _f(r["d_heldout_mean"], "{:+.3f}"),
                         f"{r['n_pos_heldout']} of {r['n_seeds']}" if arm != BASELINE_KEY else "",
                         _f(r["wc_d_heldout_mean"], "{:+.3f}"), _f(r["d_test_mean"], "{:+.3f}")])
    return "# Seeds: mean ± sample sd over seeds (lr 1e-5)\n\n" + _table(
        ["arm", "type", "seeds", "held-out AUC", "test AUC", "mean Δ held-out", "seeds with Δ > 0",
         "mean Δ within-city held-out", "mean Δ test"], body)


def gate_markdown(gates):
    parts = ["# The 0.92 gate", "", "Coverage and precision of each verdict at the 0.92 gate, uncapped map fitted on "
             "val under each rule's deployment prior (`analysis/prior_rules.py gate_rows`). One table per rule; "
             f"`{DECISION_RULE}` is the rule's input, `{REFERENCE_RULE}` a reference.", ""]
    for rule in (DECISION_RULE, REFERENCE_RULE):
        rs = [g for g in gates if g["rule"] == rule]
        body = [[g["type"], f"`{g['arm']}`", g["seed"], f"{g['lr']:g}", g["split"],
                 "Agree" if g["verdict"] == "correct" else "Disagree", g.get("n"), _f(g.get("gate_coverage")),
                 _f(g.get("gate_precision")), g.get("gate_n")] for g in rs]
        parts += [f"## prior rule `{rule}`", "", _table(["type", "arm", "seed", "lr", "split", "verdict", "n",
                                                         "coverage", "precision", "gated n"], body)]
    return "\n".join(parts)


def slices_markdown(slice_rows):
    from analysis import slices as sl
    parts = ["# Slices: type × severity and type × tag against the in-env baseline", "",
             f"Same rows for both models; Δ with a 95% panorama-bootstrap interval; blank under {sl.MIN_N} labels; "
             f"⚠ thin = fewer than {sl.THIN_TRAIN} training rows; round-3 AUC is round 3's own seed-42 baseline on the "
             "slice. Intervals are not corrected for the number of slices.", ""]
    keys = sorted({(r["type"], r["arm"], r["seed"], r["split"], r["by"]) for r in slice_rows})
    for t, arm, s, split, by in keys:
        rs = [r for r in slice_rows if (r["type"], r["arm"], r["seed"], r["split"], r["by"]) == (t, arm, s, split, by)]
        body = [[r["slice"], r["n"], r["n_incorrect"], f"{r['train_n']:,}{' ⚠ thin' if r['thin'] else ''}",
                 _f(r.get("round3_auc")), _f(r.get("auc_b")), _f(r.get("auc_a")),
                 "" if r.get("delta") is None or np.isnan(r["delta"]) else _d({k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}, "delta")]
                for r in rs]
        parts += [f"## {t} / `{arm}` seed {s} / {split} / by {by}", "",
                  _table([by, "n", "incorrect", "training rows", "round-3 AUC", "baseline AUC", "arm AUC", "Δ [95% CI]"], body)]
    return "\n".join(parts)


def cost_markdown(rows):
    seen = {}
    for r in rows:
        k = (r["arm"], r["type"])
        seen.setdefault(k, []).append(r)
    body = []
    for (arm, t), rs in sorted(seen.items()):
        b = bb.get(arm)
        body.append([f"`{arm}`", t, _f(rs[0].get("params") and rs[0]["params"] / 1e6, "{:.1f}"), b.tokens_per_image,
                     rs[0].get("model_class") or bb.MODEL_CLASS[b.family], "no" if b.family == "dinov3_vit" else "yes",
                     ", ".join(sorted({g for r in rs for g in (r.get("gpu") or "").split(", ") if g})),
                     _f(_mean([r.get("seconds_per_step") for r in rs]), "{:.2f}"),
                     _f(max([r.get("peak_gpu_memory_gb") or 0 for r in rs]) or None, "{:.1f}")])
    return ("# Cost\n\nParameters counted from the built model; seconds per optimizer step are the last attempt's wall "
            "time (loading and evaluation included) over the steps it ran, mean over runs; GPU models from each run's "
            "`gpu.txt` (appended on every attempt). ONNX rows join when that later stage has run.\n\n"
            + _table(["arm", "type", "params (M)", "tokens / image", "model class",
                      "AutoModelForImageClassification, no custom code", "GPUs", "s / step", "peak GPU GB"], body))


def env_markdown(env):
    parts = ["# Environment: in-env baseline (transformers 4.56.2) against round 3 (4.55.0)", "",
             "Same recipe, split and seed; a mean difference beyond twice the round-3 seed sd "
             f"({', '.join(f'{t} {s}' for t, s in BASE_SEED_SD.items())}) is called out at the top of "
             "docs/backbone-2026-11.md.", ""]
    body = [[r["type"], r["seed"], _f(r["round3_test_auc"]), _f(r["test_auc"]), _d(r, "d_test"),
             _f(r["round3_heldout_auc"]), _f(r["heldout_auc"]), _d(r, "d_heldout")] for r in env]
    parts.append(_table(["type", "seed", "round-3 test AUC", "in-env test AUC", "Δ test", "round-3 held-out AUC",
                         "in-env held-out AUC", "Δ held-out"], body))
    flags = []
    for t in sorted({r["type"] for r in env}):
        m = _mean([r.get("d_heldout") for r in env if r["type"] == t])
        if m is not None and abs(m) > 2 * BASE_SEED_SD.get(t, float("inf")):
            flags.append(f"**{t}: mean Δ held-out {m:+.3f} exceeds twice the seed sd.**")
    return "\n".join(parts + flags) + "\n"


def lr_markdown(rows):
    body = []
    for r in rows:
        if math.isclose(r["lr"], DEFAULT_LR):
            continue
        thr = THRESHOLDS.get(r["type"])
        own = r.get("vs_own_lr_d_heldout")
        limited = own is not None and thr is not None and own >= thr
        body.append([r["type"], f"`{r['arm']}`", r["seed"], f"{r['lr']:g}", _f(r.get("heldout_auc")),
                     _d({k[len('vs_own_lr_'):]: v for k, v in r.items() if k.startswith("vs_own_lr_")}, "d_heldout"),
                     _d({k[len('vs_base_same_lr_'):]: v for k, v in r.items() if k.startswith("vs_base_same_lr_")}, "d_heldout"),
                     "recipe-limited" if limited else ""])
    return ("# Learning rate (explanatory; cannot win)\n\nEach 3e-5 run against the same arm at 1e-5 (seed 42) and "
            "against the baseline at 3e-5. An arm whose 3e-5 run beats its 1e-5 run by at least the type's threshold "
            f"({', '.join(f'{t} +{v:.3f}' for t, v in THRESHOLDS.items())}) is called recipe-limited.\n\n"
            + _table(["type", "arm", "seed", "lr", "held-out AUC", "Δ held-out vs own 1e-5",
                      "Δ held-out vs baseline at the same lr", "reading"], body))


def rampnet_markdown(rows, exposure):
    body = []
    for r in rows:
        if r["arm"] != RAMPNET_KEY or not math.isclose(r["lr"], DEFAULT_LR):
            continue
        ctl = {k[len("vs_control_"):]: v for k, v in r.items() if k.startswith("vs_control_")}
        body.append([r["type"], r["seed"], _d(ctl, "d_heldout"), _d(ctl, "wc_d_heldout"),
                     _d(ctl, "d_test") + " (indirectly exposed)" if ctl.get("d_test") is not None else "",
                     _d(r, "d_heldout"), _d(r, "wc_d_heldout")])
    exp = [[e["city_id"], e["stage"], e["round3_splits"]] for e in exposure.get("deployments", [])]
    return ("# RampNet's backbone\n\nThe converted RampNet ConvNeXt V2-B against the `convnextv2-base` control (the "
            "same architecture and final layernorm, ImageNet stem and stages) and against the in-env baseline. Its "
            "`test` numbers are indirectly exposed: RampNet's crop model trained on CurbRamp labels of deployments now "
            "in round 3's `test` split. Conclusions about it rest on held-out only.\n\n"
            + _table(["type", "seed", "Δ held-out vs control", "Δ within-city held-out vs control", "Δ test vs control",
                      "Δ held-out vs baseline", "Δ within-city held-out vs baseline"], body)
            + f"\n## Exposure (`analysis/backbone_rampnet_exposure.json`)\n\n{exposure.get('summary', '')}\n\n"
            + _table(["deployment", "RampNet stage", "round-3 splits"], exp))


def decision_markdown(decision, n_compared):
    body = [[f"`{a}`", r["status"], ", ".join(r.get("condition_1_types") or []), r.get("condition_2", ""),
             r.get("condition_3", ""), "yes" if r.get("eligible") else ("" if "eligible" not in r else "no")]
            for a, r in decision["arms"].items()]
    return (f"# Decision\n\nRecommendation: **{decision['recommendation']}**\n\nArms compared: {n_compared}; the family "
            "is not corrected for multiplicity (null.md gives the expected number of null winners under this rule).\n\n"
            + _table(["arm", "status", "condition 1 met on", "condition 2", "condition 3", "eligible"], body))


def null_markdown(res, res_doubled):
    return ("# Expected winners under no true difference\n\n"
            f"Seeded Monte Carlo ({res['n_sims']:,} draws, seed {res['seed']}): each eligible arm's per-seed paired Δ "
            f"held-out, Δ within-city held-out and Δ test are independent N(0, sd) with sd "
            f"{', '.join(f'{t} {s}' for t, s in res['delta_sd'].items())}; the gate condition is assumed met.\n\n"
            + _table(["thresholds", "arms", "expected winners", "P(at least one)"],
                     [[", ".join(f"{t} +{v:.3f}" for t, v in res["thresholds"].items()), res["n_arms"],
                       f"{res['expected_winners']:.4f}", f"{res['p_any_winner']:.4f}"],
                      [", ".join(f"{t} +{v:.3f}" for t, v in res_doubled["thresholds"].items()) + " (doubled)",
                       res_doubled["n_arms"], f"{res_doubled['expected_winners']:.4f}",
                       f"{res_doubled['p_any_winner']:.4f}"]]))


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def _clean(obj):
    """JSON-safe: NaN to None, numpy scalars to Python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def cmd_report(args):
    from analysis import retrain_report as rr
    from fine_tune_classifier import git_state
    tracker = Tracker()
    inputs = load_inputs(tracker.track(args.inputs))
    with open(tracker.track(args.shares)) as f:
        shares_types = json.load(f)["types"]
    public, public_source = rr.public_cities(args.public_cities)
    if args.public_cities:
        tracker.track(args.public_cities)
    with open(tracker.track(args.exposure)) as f:
        exposure = json.load(f)
    types = [t for t in args.label_types.split(",") if t]
    slice_seeds = {int(s) for s in args.slice_seeds.split(",") if s}
    jobs = os.path.join(args.runs_dir, "backbone_jobs.txt")
    submitted = os.path.isfile(jobs)
    rows, gates, heldout, slice_rows, env, incomplete, round3 = [], [], [], [], [], [], []
    for t in types:
        split = load_split(args.split_dir, t, inputs, tracker)
        runs = load_runs(args.runs_dir, t, args.tag_prefix, split, inputs["split_sha256"][t], tracker, args.epochs)
        incomplete += [[t, f"`{tag}`", "; ".join(r["why"])] for tag, r in runs.items() if r["status"] != "complete"]
        ref = round3_reference(args.split_dir, t, split, tracker)
        round3 += [{"type": t, "seed": s, **run_metrics(p)} for s, p in sorted(ref.items())]
        r, g, h, sr = compare_type(t, runs, ref, shares_types, args.tag_prefix, public, args.n_boot, args.seed,
                                   slice_seeds, split[split["split"] == "train"])
        rows += r
        gates += g
        heldout += h
        slice_rows += sr
        env += env_rows(t, runs, ref, args.n_boot, args.seed)
    summary = summarise(rows, gates)
    decision = decide(summary, submitted, tuple(types))
    null = null_winner_rate(seed=0)
    null2 = null_winner_rate(thresholds={t: 2 * v for t, v in THRESHOLDS.items()}, seed=0)
    os.makedirs(args.out, exist_ok=True)
    outputs = {
        "arms.md": arms_markdown(rows, heldout, incomplete, round3),
        "seeds.md": seeds_markdown(summary), "gate.md": gate_markdown(gates), "slices.md": slices_markdown(slice_rows),
        "cost.md": cost_markdown(rows), "env.md": env_markdown(env), "lr.md": lr_markdown(rows),
        "rampnet.md": rampnet_markdown(rows, exposure),
        "decision.md": decision_markdown(decision, len({r["arm"] for r in rows} - {BASELINE_KEY})),
        "null.md": null_markdown(null, null2),
    }
    for name, text in outputs.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    json_out = {"arms.json": {"rows": rows, "heldout": heldout, "summary": summary, "round3": round3,
                              "incomplete": incomplete},
                "decision.json": {"decision": decision, "null": null, "null_doubled": null2,
                                  "thresholds": THRESHOLDS, "delta_sd": DELTA_SD}}
    for name, obj in json_out.items():
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(_clean(obj), f, indent=2, default=str)
            f.write("\n")
    extra = [p for p in (args.manifest, args.pip_freeze, jobs) if p and os.path.isfile(p)]
    extra += [os.path.join(args.runs_dir, n) for n in sorted(os.listdir(args.runs_dir)) if n.startswith("tasks_")] \
        if os.path.isdir(args.runs_dir) else []
    for p in extra:
        tracker.track(p)
    prov = {"files_read": {os.path.relpath(p, os.path.commonpath([os.path.abspath(args.runs_dir), p]))
                           if p.startswith(os.path.abspath(args.runs_dir)) else os.path.basename(p): h
                           for p, h in sorted(tracker.files.items())},
            "outputs": {n: sha256_of(os.path.join(args.out, n)) for n in sorted(list(outputs) + list(json_out))},
            "report_git": git_state(), "n_boot": args.n_boot, "bootstrap_seed": args.seed, "tag_prefix": args.tag_prefix,
            "public_cities_source": public_source, "label_types": types, "slice_seeds": sorted(slice_seeds)}
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"wrote {len(outputs) + len(json_out) + 1} files to {args.out}; recommendation: {decision['recommendation']}")
    for t, tag, why in incomplete:
        print(f"INCOMPLETE {t} {tag}: {why}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view", help="a read-only view of the round-3 data (copies checked against the pins)")
    v.add_argument("--src", required=True, help="round-3 data directory (read only)")
    v.add_argument("--dst", required=True)
    v.add_argument("--inputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbone_inputs.json"))
    v.add_argument("--label-types", default="CurbRamp,SurfaceProblem")
    v.add_argument("--smoke", action="store_true", help="a seeded class-balanced subsample of the first label type")
    v.add_argument("--smoke-inputs-out", default=None, help="with --smoke: where to write the subsample's pins")
    v.add_argument("--scratch", default=None, help="with --smoke: where the verified source split copy goes")
    v.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report", help="the comparison tables from the runs' records")
    r.add_argument("--split-dir", required=True, help="holds <LT>/split.csv and round 3's reference predictions")
    r.add_argument("--runs-dir", required=True, help="the $KEEP layout: <LT>/runs/<TAG>/...")
    r.add_argument("--inputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbone_inputs.json"))
    r.add_argument("--shares", required=True, help="docs/deployment-prior-2026-11/shares.json")
    r.add_argument("--public-cities", default=None, help="a saved /v3/api/cities listing (default: the live API)")
    r.add_argument("--exposure", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "backbone_rampnet_exposure.json"))
    r.add_argument("--label-types", default=",".join(PILOT_TYPES))
    r.add_argument("--tag-prefix", default="bb-")
    r.add_argument("--slice-seeds", default="42")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--epochs", type=int, default=25, help="every counted run must have trained this many epochs")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--manifest", default=None, help="backbone_manifest.json (hashed into provenance)")
    r.add_argument("--pip-freeze", default=None, help="pip_freeze.txt (hashed into provenance)")
    r.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "view":
        if args.smoke and not args.smoke_inputs_out:
            raise SystemExit("--smoke needs --smoke-inputs-out")
        if args.smoke:
            args.label_types = args.label_types.split(",")[0]
        return cmd_view(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
