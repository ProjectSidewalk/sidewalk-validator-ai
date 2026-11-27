"""Input-resolution study (docs/resolution-2026-11.md): does DINOv2-L at 448 px (518 as a later, optional arm) beat the
round-3 recipe's 224 px on within-city held-out AUC, without harming the guard type or the fine-texture slices?

Every validator so far, and the backbone (docs/backbone-2026-11.md) and context-arm (docs/context-arms-2026-11.md)
nulls, trained at 224 px while the crops are stored at 640 px. This module holds the pre-registered rule
(``DECISION_RULE``; the doc's fenced json block must equal it, a test checks), the seeds, the tags, and four commands:

- ``power`` (laptop, before any run): the rule's operating characteristics from October's round-3 seed sds, through
  ``context_arms.simulate_rule`` (the same seed-mean interval the rule uses).
- ``bench`` (one GPU job, or a laptop CPU): per-crop latency of DINOv2-L at each study size, preprocessing and forward
  pass timed apart, at batch 1 and 16. Random weights from the registry's architecture, so no hub access; latency does
  not depend on the weights.
- ``report`` (laptop, after ``scp -r`` of the klone ``$KEEP``): reads every run under ``<runs-dir>/<LT>/runs/<TAG>/``,
  checks each against the recipe (``consistency``), and writes the tables and the decision. It reuses the backbone
  study's scoring (``backbone_comparison.run_metrics``, ``paired``, ``load_split``, ``join_split``, the run loader's
  checks) and the context-arm study's seed-mean interval (``context_arms.within_city_boot``, ``boot_var_of_delta``,
  ``seed_mean_delta``).
- the data view is ``backbone_comparison.py view --inputs analysis/resolution_inputs.json`` (pins for all five types).

Tags: ``res-r<px>`` at seed 42, ``res-r<px>-s<seed>`` otherwise (``smoke-res-r<px>`` in the smoke job). Label
references are ``label_uid`` (``city:label_id``); private held-out deployments are lettered, never named.

Usage::

    python analysis/resolution_study.py power --out docs/resolution-2026-11
    python analysis/resolution_study.py bench --out bench.json [--device cpu] [--resolutions 224,448,518]
    python analysis/resolution_study.py report --split-dir <retrain-2026-10 copy> --runs-dir <keep copy> \\
        --public-cities cities.json --bench <keep>/bench_a40.json --out docs/resolution-2026-11
"""
import argparse
import itertools
import json
import math
import os
import re
import statistics
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backbones as bb  # noqa: E402
from analysis import backbone_comparison as bc  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(REPO_ROOT, "analysis", "resolution_inputs.json")
SHARES = os.path.join(REPO_ROOT, "docs", "deployment-prior-2026-11", "shares.json")

#: Registered before any run (docs/resolution-2026-11.md). Changing any value after a result exists is a deviation.
#: Amended 2026-11-27, before any result (PR #133 review; doc, "Amendments"): five seeds (J1), the all-terms
#: Satterthwaite interval, interval-based guard and test floors, a 99% texture screen, a minimum bootstrap count.
SEEDS = (42, 43, 44, 45, 46)
TARGET_TYPES = ("Obstacle", "SurfaceProblem", "NoCurbRamp")
GUARD_TYPE = "CurbRamp"
TEXTURE_TYPE = "Crosswalk"   # in the study only for its "paint fading" slice; its type-level numbers are descriptive
TYPES = TARGET_TYPES + (GUARD_TYPE, TEXTURE_TYPE)
BASELINE_RES = 224
ARMS = (448,)
LATER_ARMS = (518,)          # registered; submitted only on Jon's word (cost)
TEXTURE_SLICES = (("SurfaceProblem", "tag", "brick/cobblestone"), ("CurbRamp", "tag", "missing tactile warning"),
                  ("Crosswalk", "tag", "paint fading"))
WATCH_SLICES = (("Obstacle", "tag", "stairs"),)   # descriptive only: too few rows to judge (docs, "Slices")
RECIPE = {"epochs": 25, "learning_rate": 1e-5, "best_metric": "macro_f1", "batch_size": 8, "target": "hard"}
NIGHTLY_LABELS_PER_CITY = 800   # labels per city per nightly run, all types (docs/deploy-checklist-2026-10.md, F6)
#: Per run, what the report needs in ``$KEEP``: backbone_comparison's four, plus the processor that scored it.
KEEP_FILES = bc.KEEP_FILES + ("preprocessor_config.json",)

#: The rule's constants, written verbatim into decision.json; the doc's fenced json block must equal it (tested).
DECISION_RULE = {
    "endpoint": "within-city AUC on test_heldout",
    "statistic": "seed mean of the arm minus seed mean of the in-env 224 rerun",
    "interval": "t_seed_variances_plus_panorama_bootstrap_satterthwaite_all_terms",
    "level": 0.95, "min_delta": 0.015, "min_target_wins": 2,
    "target_types": list(TARGET_TYPES),
    "target_test_auc_floor": -0.005, "guard_type": GUARD_TYPE, "guard_floor": -0.005,
    "floors_fail_when": "interval_wholly_below_floor",
    "texture_slices": [[t, by, s] for t, by, s in TEXTURE_SLICES], "texture_splits": ["test", "test_heldout"],
    "texture_level": 0.99, "texture_fails_when": "interval_wholly_below_zero",
    "seeds": list(SEEDS), "baseline_res": BASELINE_RES, "arms": list(ARMS), "later_arms": list(LATER_ARMS),
    "prefer_larger_res_margin": 0.01, "within_min_n": 10, "slice_min_n": 30, "slice_min_class": 8,
    "min_boot": 200, "amended": "2026-11-27",
}
#: ``(seed sd, bootstrap sd of a paired difference)`` per check, for ``power`` only (assumptions of the power table,
#: not of the rule). Targets' within-city held-out: October's round-3 trio (docs/context-arms-2026-11.md, "Operating
#: characteristics", klone job 43403551). Pooled test AUC: the context study's five-seed rerun (seed sd) and its
#: rerun-against-round-3 paired interval (bootstrap sd), docs/context-arms-2026-11/arms.json. CurbRamp: the backbone
#: study's in-env baseline over three seeds (docs/backbone-2026-11/arms.json) and the bootstrap sd within the PR
#: review's measured 0.006-0.010. Texture slices: 0.7 x one run's panorama-bootstrap sd of the slice's pooled AUC on
#: the laptop copy of round 3 (seed 42), 0.7 being about the type-level ratio of paired to single-run sd; the seed sd
#: is set equal to it, since no per-slice seed trio exists.
POWER_INPUTS = {"NoCurbRamp": (0.024, 0.012), "Obstacle": (0.009, 0.010), "SurfaceProblem": (0.021, 0.015)}
POWER_TEST_INPUTS = {"NoCurbRamp": (0.0023, 0.0032), "Obstacle": (0.0021, 0.0038), "SurfaceProblem": (0.0040, 0.0040),
                     "CurbRamp": (0.0019, 0.0039)}
POWER_GUARD_INPUTS = (0.011, 0.010)
POWER_TEXTURE_INPUTS = {("SurfaceProblem", "test"): 0.028, ("SurfaceProblem", "test_heldout"): 0.024,
                        ("CurbRamp", "test"): 0.010, ("CurbRamp", "test_heldout"): 0.024,
                        ("Crosswalk", "test"): 0.018, ("Crosswalk", "test_heldout"): 0.025}
STUDY_VERSIONS = bc.STUDY_VERSIONS
PREFIX = "res-"
TAG_RE = re.compile(r"^r(?P<res>\d+)(?:-s(?P<seed>\d+))?$")


# ---------------------------------------------------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------------------------------------------------

def make_tag(res, seed=42, prefix=PREFIX):
    """``<prefix>r<res>[-s<seed>]`` (seed 42 carries no suffix).

    >>> make_tag(448), make_tag(224, 43), make_tag(518, 44, "smoke-res-")
    ('res-r448', 'res-r224-s43', 'smoke-res-r518-s44')
    """
    return f"{prefix}r{int(res)}" + ("" if int(seed) == 42 else f"-s{int(seed)}")


def parse_tag(tag, prefix=PREFIX):
    """``{"res", "seed"}`` of a tag ``make_tag`` wrote, or None.

    >>> parse_tag("res-r448-s43"), parse_tag("res-r224"), parse_tag("bb-dinov2-large"), parse_tag("res-r999")
    ({'res': 448, 'seed': 43}, {'res': 224, 'seed': 42}, None, None)
    """
    if not tag.startswith(prefix):
        return None
    m = TAG_RE.match(tag[len(prefix):])
    if not m:
        return None
    res = int(m["res"])
    if res not in (BASELINE_RES, *ARMS, *LATER_ARMS):
        return None
    return {"res": res, "seed": int(m["seed"]) if m["seed"] else 42}


# ---------------------------------------------------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------------------------------------------------

def consistency(run_dir, tag, parsed, split_sha, epochs=RECIPE["epochs"]):
    """Reasons a run cannot be counted (empty when it can): files missing; ``run_key.json``'s split hash is not the
    pinned one; the recorded seed, input size (run key, record, the processor the record describes, and the saved
    ``preprocessor_config.json`` of the checkpoint that scored the run), backbone, learning rate, epochs, best metric,
    target, per-device batch or gradient checkpointing differ from the tag and the round-3 recipe; the study env's
    package versions differ; random weights; loss not divided by the accumulation steps."""
    missing = [n.format(tag=tag) for n in KEEP_FILES if not os.path.isfile(os.path.join(run_dir, n.format(tag=tag)))]
    if missing:
        return [f"missing {', '.join(missing)}"]
    key = bc._read_json(os.path.join(run_dir, "run_key.json"))
    run = bc._read_json(os.path.join(run_dir, "training_run.json"))
    saved = (bc._read_json(os.path.join(run_dir, "preprocessor_config.json")) or {}).get("crop_size") or {}
    why = []
    if key.get("split_sha256") != split_sha:
        why.append(f"split_sha256 {str(key.get('split_sha256'))[:12]} is not the pinned {str(split_sha)[:12]}")
    if run.get("seed") != parsed["seed"]:
        why.append(f"seed {run.get('seed')} != tag's {parsed['seed']}")
    res = parsed["res"]
    key_res = key.get("input_res", BASELINE_RES)   # the default size adds nothing to the key
    if key_res != res or run.get("input_res") != res:
        why.append(f"input_res (key {key_res}, record {run.get('input_res')}) != tag's {res}")
    backbone = run.get("backbone") or {}
    crop = ((backbone.get("preprocess") or {}).get("crop_size") or {}).get("height")
    if crop != res:
        why.append(f"recorded preprocessing crops {crop}, not {res}")
    if not (saved.get("height") == res == saved.get("width")):
        why.append(f"the saved processor that scored the run crops {saved.get('height')}x{saved.get('width')}, not {res}")
    if backbone.get("key") != bb.DEFAULT:
        why.append(f"backbone {backbone.get('key')} is not {bb.DEFAULT}")
    lr = run.get("learning_rate")
    if lr is None or not math.isclose(lr, RECIPE["learning_rate"]):
        why.append(f"learning_rate {lr} != {RECIPE['learning_rate']}")
    if run.get("epochs") != epochs:
        why.append(f"epochs {run.get('epochs')} != {epochs}")
    if run.get("best_metric") != RECIPE["best_metric"]:
        why.append(f"best_metric {run.get('best_metric')} != {RECIPE['best_metric']}")
    if key.get("target", "hard") != RECIPE["target"]:
        why.append(f"target {key.get('target')} != {RECIPE['target']}")
    if key.get("batch_size", RECIPE["batch_size"]) != RECIPE["batch_size"]:
        # a smaller per-device batch keeps the effective batch of 32 but is not what the plan registered; it is
        # allowed only through BATCH= in the job, which the doc lists as a deviation to record
        why.append(f"per-device batch {key.get('batch_size')} != {RECIPE['batch_size']} (record it as a deviation and "
                   "pass --allow-batch to count it)")
    if key.get("gradient_checkpointing") or run.get("gradient_checkpointing"):
        why.append("gradient checkpointing (GC=1; record it as a deviation and pass --allow-gc to count it)")
    versions = run.get("versions") or {}
    for mod, want in STUDY_VERSIONS.items():
        got = versions.get(mod)
        if got is not None and str(got).split("+")[0] != want:
            why.append(f"{mod} {got} != the study env's {want}")
    if backbone.get("random_init") is not False:
        why.append("random_init (smoke test weights)")
    if run.get("model_accepts_loss_kwargs") is not False:
        why.append(f"model_accepts_loss_kwargs {run.get('model_accepts_loss_kwargs')} (loss not divided by the steps)")
    return why


def load_runs(runs_dir, label_type, prefix, split, split_sha, tracker, epochs=RECIPE["epochs"], allow_batch=False,
              allow_gc=False):
    """Every run of ``label_type`` under ``<runs_dir>/<LT>/runs/`` whose tag parses: ``{tag: record}`` with
    ``parsed``, ``status`` (``complete`` / ``INCOMPLETE``), ``why``, and for a complete run its predictions joined
    with the split (``backbone_comparison.join_split``, one to one on ``label_uid``), record and trainer state.
    ``allow_batch`` / ``allow_gc`` count runs that used the recorded memory fallbacks (doc, J7)."""
    base = os.path.join(runs_dir, label_type, "runs")
    out = {}
    eval_rows = int(split["split"].isin(["val", "test", "test_heldout"]).sum())
    for tag in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        parsed = parse_tag(tag, prefix)
        if parsed is None:
            continue
        d = os.path.join(base, tag)
        why = consistency(d, tag, parsed, split_sha, epochs)
        if allow_batch:
            why = [w for w in why if not w.startswith("per-device batch")]
        if allow_gc:
            why = [w for w in why if not w.startswith("gradient checkpointing")]
        for name in ("run_key.json", "training_run.json", "preprocessor_config.json"):
            if os.path.isfile(os.path.join(d, name)):
                tracker.track(os.path.join(d, name))
        rec = {"tag": tag, "parsed": parsed, "dir": d, "why": why}
        if not why:
            pred = pd.read_csv(tracker.track(os.path.join(d, f"predictions_{tag}.csv")),
                               dtype={"label_uid": str, "split": str, "true": str})
            if len(pred) != eval_rows:
                why.append(f"{len(pred)} predictions, the split has {eval_rows} evaluation rows")
            elif not pred["label_uid"].is_unique:
                why.append("duplicate label_uid in the predictions")
            elif not np.isfinite(pred[["p_correct", "p_incorrect"]].to_numpy(float)).all():
                why.append("non-finite scores")
            else:
                rec["pred"] = bc.join_split(pred, split)
                rec["run"] = bc._read_json(os.path.join(d, "training_run.json"))
                ts = os.path.join(d, "trainer_state.json")
                rec["state"] = bc._read_json(tracker.track(ts)) if os.path.isfile(ts) else {}
                gpu = os.path.join(d, "gpu.txt")
                rec["gpu"] = sorted({ln.strip() for ln in open(gpu)} - {""}) if os.path.isfile(gpu) else []
        rec["status"] = "INCOMPLETE" if why else "complete"
        out[tag] = rec
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the seed-mean comparison
# ---------------------------------------------------------------------------------------------------------------------

def seed_mean(preds, arm_names, base_names, split, n_boot, seed, min_n=DECISION_RULE["within_min_n"],
              level=DECISION_RULE["level"], min_boot=DECISION_RULE["min_boot"]):
    """The rule's statistic on ``split``: within-city AUC of each run (``context_arms.within_city_boot``, one joint
    panorama bootstrap stratified by deployment for every run, so differences are paired on the evaluation rows), then
    ``context_arms.seed_mean_delta(..., df="all_terms")`` of the arm's seeds against the baseline's, with the
    bootstrap variance of that difference added to the seed variances and the t quantile's degrees of freedom from
    Satterthwaite over all three terms (2026-11-27 amendment). With fewer than ``min_boot`` kept resamples the
    interval is withheld (``lo`` / ``hi`` None, ``boot_short`` True): a verdict on seed variance alone would understate
    the spread (PR #133 review S1). Returns the ``seed_mean_delta`` record plus ``point`` (per run),
    ``per_deployment`` and the bootstrap's ``kept`` / ``dropped`` counts."""
    from analysis import context_arms as ca
    names = list(arm_names) + list(base_names)
    b = ca.within_city_boot(preds, split, names, min_n, n_boot, seed)
    kept = int(len(b["boot"]))
    var_boot = ca.boot_var_of_delta(b["boot"], names, list(arm_names), list(base_names)) if kept else None
    r = ca.seed_mean_delta([b["point"][n] for n in arm_names], [b["point"][n] for n in base_names], var_boot or 0.0,
                           level, df="all_terms")
    short = var_boot is None or kept < min_boot
    if short:
        r.update(lo=None, hi=None)
    r.update(point=b["point"], per_deployment=b["per_deployment"], kept=kept, dropped=b["dropped"],
             boot_short=short, n_deployments=len(b["deployments"]))
    return r


def pooled_frames(preds):
    """``preds`` with one constant ``city``, so ``within_city_boot`` reads the pooled AUC and resamples panoramas."""
    return {run: p.assign(city="pooled") for run, p in preds.items()}


def slice_frames(preds, by, name):
    """``preds`` restricted to the labels in slice ``name`` of ``by`` (``slices.with_slice``), with one constant
    ``city`` so that ``within_city_boot`` reads the slice's pooled AUC and resamples its panoramas."""
    from analysis import slices as sl
    out = {}
    for run, p in preds.items():
        s = sl.with_slice(p, by)
        s = s[s["slice"].astype(str) == str(name)].drop(columns="slice").drop_duplicates("label_uid")
        out[run] = s.assign(city="slice")
    return out


def slice_seed_mean(preds, arm_names, base_names, split, by, name, n_boot, seed, level=DECISION_RULE["level"],
                    min_boot=DECISION_RULE["min_boot"]):
    """``seed_mean`` on one slice (its pooled AUC) at ``level``, with its own bootstrap seed
    (``slices.slice_seed``), or None when the slice has fewer than ``slice_min_n`` rows or fewer than
    ``slice_min_class`` of either class on ``split`` (then it is not evaluable and never fails)."""
    from analysis import slices as sl
    frames = slice_frames(preds, by, name)
    first = frames[arm_names[0]] if arm_names else None
    if first is None:
        return None
    part = first[first["split"] == split]
    y = (part["true"] == "incorrect").to_numpy()
    if len(part) < DECISION_RULE["slice_min_n"] or min(int(y.sum()), int((~y).sum())) < DECISION_RULE["slice_min_class"]:
        return None
    r = seed_mean(frames, arm_names, base_names, split, n_boot, sl.slice_seed(seed, f"{by}:{name}:{split}"),
                  min_n=DECISION_RULE["slice_min_n"], level=level, min_boot=min_boot)
    r.update(n=int(len(part)), n_incorrect=int(y.sum()))
    return r


def _complete(runs, res):
    """``{seed: record}`` of the complete runs at ``res`` on the registered seeds."""
    return {r["parsed"]["seed"]: r for r in runs.values()
            if r["status"] == "complete" and r["parsed"]["res"] == res and r["parsed"]["seed"] in SEEDS}


def compare_type(label_type, runs, arm_res, train, public, n_boot, seed, min_boot=DECISION_RULE["min_boot"]):
    """Everything for one type and one arm against the 224 rerun: per-seed metrics and paired deltas; the seed-mean
    statistic with its interval on within-city held-out (the endpoint), within-city test and pooled test AUC (the test
    floor); per held-out deployment; every severity and tag slice (95%, descriptive); the texture slices of this type
    at the screen's level (99%)."""
    from analysis import context_arms as ca
    from analysis import slices as sl
    arm, base = _complete(runs, arm_res), _complete(runs, BASELINE_RES)
    shared = sorted(set(arm) & set(base))
    out = {"type": label_type, "arm_res": arm_res, "seeds_arm": sorted(arm), "seeds_base": sorted(base),
           "complete": len(arm) == len(SEEDS) and len(base) == len(SEEDS), "per_seed": [], "slices": [],
           "texture": [], "heldout": None, "test": None, "test_auc": None, "deployments": []}
    for s in shared:
        a, b_ = arm[s], base[s]
        row = {"seed": s, "tag": a["tag"], "base_tag": b_["tag"], **{f"arm_{k}": v for k, v in bc.run_metrics(a["pred"]).items()},
               **{f"base_{k}": v for k, v in bc.run_metrics(b_["pred"]).items()},
               **bc.paired(a["pred"], b_["pred"], a["tag"], n_boot, seed)}
        out["per_seed"].append(row)
    if not (arm and base):
        return out
    preds = {r["tag"]: r["pred"] for r in list(arm.values()) + list(base.values())}
    an, bn = [arm[s]["tag"] for s in sorted(arm)], [base[s]["tag"] for s in sorted(base)]
    out["heldout"] = seed_mean(preds, an, bn, "test_heldout", n_boot, seed, min_boot=min_boot)
    out["test"] = seed_mean(preds, an, bn, "test", n_boot, seed, min_boot=min_boot)
    out["test_auc"] = seed_mean(pooled_frames(preds), an, bn, "test", n_boot, seed, min_boot=min_boot)
    out["test_auc"].pop("per_deployment", None)
    d_test = [r["d_test"] for r in out["per_seed"] if r.get("d_test") is not None]
    out["mean_d_test_auc"] = float(np.mean(d_test)) if d_test else None
    d_held = [r["d_heldout"] for r in out["per_seed"] if r.get("d_heldout") is not None]
    out["mean_d_heldout_auc"] = float(np.mean(d_held)) if d_held else None
    names = ca.heldout_display_names(preds, public)
    for city, aucs in (out["heldout"]["per_deployment"] or {}).items():
        a_m = float(np.mean([aucs[n] for n in an])) if an else None
        b_m = float(np.mean([aucs[n] for n in bn])) if bn else None
        out["deployments"].append({"deployment": names.get(city, city), "auc_arm": a_m, "auc_base": b_m,
                                   "delta": None if a_m is None or b_m is None else a_m - b_m})
    # raw deployment ids stop here: only the display names (private ones lettered) leave this function
    out["heldout"].pop("per_deployment", None)
    out["test"].pop("per_deployment", None)
    textures = [(by, s) for t, by, s in TEXTURE_SLICES if t == label_type]
    for by in sl.BY:
        tr = sl.with_slice(train, by).groupby("slice").size()
        for split in ("test", "test_heldout"):
            part = arm[sorted(arm)[0]]["pred"]
            present = sl.with_slice(part[part["split"] == split], by)["slice"].astype(str).unique()
            for name in sorted(present):
                r = slice_seed_mean(preds, an, bn, split, by, name, n_boot, seed, min_boot=min_boot)
                n_train = int(tr.get(name, 0))
                row = {"type": label_type, "by": by, "slice": name, "split": split, "train_n": n_train,
                       "thin": n_train < sl.THIN_TRAIN}
                if r is not None:
                    row.update(n=r["n"], n_incorrect=r["n_incorrect"], delta=r["delta"], lo=r["lo"], hi=r["hi"],
                               n_above=r["n_above"], n_arm=r["n_arm"],
                               auc_arm=float(np.mean(r["arm_values"])), auc_base=float(np.mean(r["base_values"])))
                out["slices"].append(row)
            for by_, name in textures:   # the screen, at its own level; always listed, evaluable or not
                if by_ != by:
                    continue
                r = (slice_seed_mean(preds, an, bn, split, by, name, n_boot, seed, level=DECISION_RULE["texture_level"],
                                     min_boot=min_boot) if name in present else None)
                row = {"type": label_type, "by": by, "slice": name, "split": split, "train_n": int(tr.get(name, 0)),
                       "level": DECISION_RULE["texture_level"]}
                if r is not None:
                    row.update(n=r["n"], n_incorrect=r["n_incorrect"], delta=r["delta"], lo=r["lo"], hi=r["hi"],
                               boot_short=r["boot_short"])
                evaluable = r is not None and r.get("hi") is not None
                out["texture"].append({**row, "evaluable": evaluable, "falls": bool(evaluable and r["hi"] < 0)})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------------------------------------

def below_floor(iv, floor):
    """True when interval ``iv`` (``{"hi": ...}``) lies wholly below ``floor``; None when it has no interval.

    >>> below_floor({"hi": -0.006}, -0.005), below_floor({"hi": -0.004}, -0.005), below_floor({"hi": None}, -0.005)
    (True, False, None)
    """
    if not iv or iv.get("hi") is None:
        return None
    return bool(iv["hi"] < floor)


def target_verdict(cmp, rule=DECISION_RULE):
    """``(verdict, criteria)`` of one target type: ``WIN`` when the seed-mean within-city held-out D >= min_delta, its
    interval's lower end > 0 and the seed-mean pooled test AUC delta's interval does not lie wholly below the test
    floor (non-inferiority, 2026-11-27 amendment); ``REGRESSION`` when the held-out interval's upper end < 0;
    ``NO GAIN`` otherwise; ``INCOMPLETE`` without every registered seed of both models or with too few bootstrap
    resamples for either interval.

    >>> c = {"complete": True, "heldout": {"delta": 0.03, "lo": 0.005, "hi": 0.05}, "test_auc": {"hi": 0.002}}
    >>> target_verdict(c)[0], target_verdict({**c, "test_auc": {"hi": -0.006}})[0]
    ('WIN', 'NO GAIN')
    >>> target_verdict({**c, "test_auc": {"hi": None}})[0]
    'INCOMPLETE'
    >>> target_verdict({**c, "heldout": {"delta": -0.03, "lo": -0.05, "hi": -0.01}})[0]
    'REGRESSION'
    """
    h = cmp.get("heldout") or {}
    floor = below_floor(cmp.get("test_auc"), rule["target_test_auc_floor"])
    if not cmp.get("complete") or h.get("lo") is None or floor is None:
        return "INCOMPLETE", []
    crit = [{"criterion": f"D >= {rule['min_delta']}", "value": h["delta"], "pass": h["delta"] >= rule["min_delta"]},
            {"criterion": "interval lower end > 0", "value": h["lo"], "pass": h["lo"] > 0},
            {"criterion": f"test AUC delta's interval not wholly below {rule['target_test_auc_floor']}",
             "value": cmp["test_auc"]["hi"], "pass": not floor}]
    if h["hi"] < 0:
        return "REGRESSION", crit
    return ("WIN" if all(c["pass"] for c in crit) else "NO GAIN"), crit


def guard_verdict(cmp, rule=DECISION_RULE):
    """``(passes, criteria)`` for the guard type, interval-based non-inferiority (2026-11-27 amendment): it fails only
    when the interval of its within-city held-out D, or of its pooled test AUC delta, lies wholly below the floor.
    None when incomplete or an interval is withheld (too few bootstrap resamples).

    >>> guard_verdict({"complete": True, "heldout": {"delta": -0.01, "hi": 0.004}, "test_auc": {"hi": 0.003}})[0]
    True
    >>> guard_verdict({"complete": True, "heldout": {"delta": -0.02, "hi": -0.006}, "test_auc": {"hi": 0.003}})[0]
    False
    """
    held = below_floor(cmp.get("heldout"), rule["guard_floor"])
    test = below_floor(cmp.get("test_auc"), rule["guard_floor"])
    if not cmp.get("complete") or held is None or test is None:
        return None, []
    crit = [{"criterion": f"guard within-city held-out interval not wholly below {rule['guard_floor']}",
             "value": cmp["heldout"]["hi"], "pass": not held},
            {"criterion": f"guard test AUC interval not wholly below {rule['guard_floor']}",
             "value": cmp["test_auc"]["hi"], "pass": not test}]
    return all(c["pass"] for c in crit), crit


def decide_arm(per_type, rule=DECISION_RULE):
    """The study's verdict for one arm from ``{type: compare_type output}``: ``WIN`` when at least ``min_target_wins``
    target types win, the guard passes and no texture slice falls; ``HELD`` while a needed type is incomplete (a
    missing run could change the verdict); otherwise ``NO WIN`` with the reasons. On a WIN, the per-type proposal:
    the arm's size for every type except one whose within-city held-out interval lies wholly below zero or whose test
    AUC interval lies wholly below the target floor (that type keeps 224). Crosswalk is judged by the same two
    interval tests, so its proposal is not a coin flip on its 445 test rows."""
    targets = {t: target_verdict(per_type.get(t, {})) for t in rule["target_types"]}
    guard, guard_crit = guard_verdict(per_type.get(rule["guard_type"], {}))
    # a slice of a type still short of seeds is shown but cannot fail the rule yet
    texture = [x for t in per_type.values() for x in t.get("texture", [])]
    texture_types = {t for t, _, _ in rule["texture_slices"]}
    incomplete = [t for t in rule["target_types"] if targets[t][0] == "INCOMPLETE"]
    incomplete += [t for t in sorted(texture_types | {rule["guard_type"]}) if not per_type.get(t, {}).get("complete")]
    if guard is None and rule["guard_type"] not in incomplete:
        incomplete.append(rule["guard_type"])
    wins = [t for t, (v, _) in targets.items() if v == "WIN"]
    falls = [f"{x['type']} {x['slice']} on {x['split']}" for x in texture
             if x.get("falls") and per_type.get(x["type"], {}).get("complete")]
    reasons = []
    if len(wins) < rule["min_target_wins"]:
        reasons.append(f"{len(wins)} of {len(rule['target_types'])} target types win (needs {rule['min_target_wins']})")
    if guard is False:
        reasons.append(f"the guard ({rule['guard_type']}) is below the {rule['guard_floor']} floor")
    if falls:
        reasons.append("texture slice(s) fall: " + "; ".join(falls))
    could_win = len(wins) + len(set(incomplete) & set(rule["target_types"])) >= rule["min_target_wins"]
    if guard is False or falls or not could_win:
        status = "NO WIN"   # decided: no missing run can turn it into a win
    elif incomplete:
        status = "HELD"
    else:
        status = "WIN"
    proposal = {}
    if status == "WIN":
        for t, c in per_type.items():
            keep = below_floor(c.get("heldout"), 0.0) or below_floor(c.get("test_auc"), rule["target_test_auc_floor"])
            proposal[t] = BASELINE_RES if keep else per_type[t]["arm_res"]
    return {"status": status, "wins": wins, "targets": {t: {"verdict": v, "criteria": c} for t, (v, c) in targets.items()},
            "guard": {"passes": guard, "criteria": guard_crit},
            "texture": [{k: x.get(k) for k in ("type", "slice", "split", "evaluable", "falls", "delta", "lo", "hi", "n",
                                                "train_n", "level")} for x in texture],
            "incomplete": sorted(set(incomplete)), "reasons": reasons, "proposal": proposal}


def recommend(decisions, rule=DECISION_RULE):
    """The study's recommendation from ``{arm_res: decide_arm output}``. One winning arm: that arm. 448 and 518 both
    winning: 448 unless 518's summed target-type D exceeds 448's by at least ``prefer_larger_res_margin`` (518 costs
    about 1.5x 448 to serve: docs, "Latency"). Held while any submitted arm is held. None winning: keep 224."""
    held = [r for r, d in decisions.items() if d["status"] == "HELD"]
    if held:
        return f"held: arm(s) {', '.join(map(str, held))} incomplete"
    winners = [r for r, d in decisions.items() if d["status"] == "WIN"]
    if not winners:
        return f"keep {BASELINE_RES} px (no arm met the rule)"
    if len(winners) == 1:
        return f"adopt {winners[0]} px (per-type proposal in decision.json)"

    def total(r):
        return sum(t["criteria"][0]["value"] for t in decisions[r]["targets"].values() if t["criteria"])
    lo, hi = min(winners), max(winners)
    if total(hi) - total(lo) >= rule["prefer_larger_res_margin"]:
        return f"adopt {hi} px (summed D exceeds {lo} px by at least {rule['prefer_larger_res_margin']})"
    return f"adopt {lo} px (the larger size's summed D is within {rule['prefer_larger_res_margin']})"


# ---------------------------------------------------------------------------------------------------------------------
# power (before any run)
# ---------------------------------------------------------------------------------------------------------------------

def at_least(probs, k):
    """P(at least ``k`` of independent events with probabilities ``probs``).

    >>> round(at_least([0.5, 0.5, 0.5], 2), 3), at_least([1.0, 0.0, 0.0], 2)
    (0.5, 0.0)
    """
    total = 0.0
    for hits in itertools.product([0, 1], repeat=len(probs)):
        if sum(hits) >= k:
            total += math.prod(p if h else 1 - p for p, h in zip(probs, hits))
    return total


def simulate_check(effect, sd_seed, sd_boot, n, n_sim, rng, level):
    """``(D, lo, hi)`` arrays of ``n_sim`` simulated seed-mean comparisons with ``n`` seeds per model, as the rule
    computes them: baseline seeds ``N(0, sd_seed)``, arm seeds ``N(effect, sd_seed)`` shifted together by one draw of
    the evaluation-sample noise of the difference ``N(0, sd_boot)``; the interval is ``seed_mean_delta``'s
    ``all_terms`` form with ``var_boot = sd_boot^2``."""
    from scipy.stats import t as student_t
    b = rng.normal(0.0, sd_seed, (n_sim, n))
    a = rng.normal(effect, sd_seed, (n_sim, n)) + rng.normal(0.0, sd_boot, (n_sim, 1))
    d = a.mean(axis=1) - b.mean(axis=1)
    ta, tb = a.var(axis=1, ddof=1) / n, b.var(axis=1, ddof=1) / n
    nu = (ta + tb + sd_boot ** 2) ** 2 / (ta ** 2 / (n - 1) + tb ** 2 / (n - 1))
    half = student_t.ppf(0.5 + level / 2, nu) * np.sqrt(ta + tb + sd_boot ** 2)
    return d, d - half, d + half


def simulate_full_rule(effect, n, n_sim=20000, seed=0, rule=DECISION_RULE, test_effect=0.0, guard_effect=0.0,
                       texture_effect=0.0):
    """Shares of ``n_sim`` simulated studies with ``n`` seeds per model in which each part of the rule passes, at a
    true within-city held-out effect ``effect`` on every target type (``test_effect`` on their pooled test AUC,
    ``guard_effect`` on both guard checks, ``texture_effect`` on every texture slice). Every check is simulated
    independently from the ``POWER_*`` inputs. Returns per-target ``crit12`` (criteria 1 and 2) and ``win`` (all
    three), ``guard``, ``texture``, the study's ``win_crit12`` (at least two targets on criteria 1 and 2 only, the old
    table's quantity) and ``win`` (the full rule)."""
    rng = np.random.default_rng(seed)
    out, wins, wins12 = {"targets": {}}, [], []
    for t in rule["target_types"]:
        d, lo, hi = simulate_check(effect, *POWER_INPUTS[t], n, n_sim, rng, rule["level"])
        _, _, thi = simulate_check(test_effect, *POWER_TEST_INPUTS[t], n, n_sim, rng, rule["level"])
        c12 = (d >= rule["min_delta"]) & (lo > 0)
        w = c12 & ~(thi < rule["target_test_auc_floor"])
        wins.append(w)
        wins12.append(c12)
        out["targets"][t] = {"crit12": float(c12.mean()), "win": float(w.mean())}
    _, _, ghi = simulate_check(guard_effect, *POWER_GUARD_INPUTS, n, n_sim, rng, rule["level"])
    _, _, gthi = simulate_check(guard_effect, *POWER_TEST_INPUTS[rule["guard_type"]], n, n_sim, rng, rule["level"])
    guard = ~(ghi < rule["guard_floor"]) & ~(gthi < rule["guard_floor"])
    texture = np.ones(n_sim, bool)
    for (t, split), sd in POWER_TEXTURE_INPUTS.items():
        _, _, xhi = simulate_check(texture_effect, sd, sd, n, n_sim, rng, rule["texture_level"])
        texture &= ~(xhi < 0)
    k = rule["min_target_wins"]
    out.update(guard=float(guard.mean()), texture=float(texture.mean()),
               win_crit12=float((np.sum(wins12, axis=0) >= k).mean()),
               win=float(((np.sum(wins, axis=0) >= k) & guard & texture).mean()))
    return out


def power_table(n_seeds=(3, 5), effects=(0.0, 0.015, 0.03, 0.05), n_sim=20000, seed=0, rule=DECISION_RULE):
    """The full rule's operating characteristics (2026-11-27 amendment; the PR #133 review's B1.3): per seed count and
    true effect on every target, each target's chance to win (criteria 1-2, and all three), the chance the guard and
    the texture screen pass when 448 changes nothing on them, and the study's chance to WIN under the full rule,
    beside the criteria-1-2-only rate the first table reported."""
    rows = []
    for n in n_seeds:
        for e in effects:
            s = simulate_full_rule(e, n, n_sim, seed, rule)
            rows.append({"seeds": n, "effect": e, **{f"{t}_crit12": v["crit12"] for t, v in s["targets"].items()},
                         **{f"{t}_win": v["win"] for t, v in s["targets"].items()},
                         "guard": s["guard"], "texture": s["texture"], "study_crit12": s["win_crit12"],
                         "study_win": s["win"]})
    return {"rows": rows, "effects": list(effects), "n_sim": n_sim, "seed": seed,
            "inputs": {"targets_within_city_heldout": POWER_INPUTS, "test_auc": POWER_TEST_INPUTS,
                       "guard_within_city_heldout": POWER_GUARD_INPUTS,
                       "texture": {f"{t}:{s}": v for (t, s), v in POWER_TEXTURE_INPUTS.items()}},
            "rule": rule}


def power_markdown(p):
    tt = list(p["rule"]["target_types"])
    head = (["seeds per model", "true effect on every target"] + [f"{t} win" for t in tt]
            + ["guard passes", "texture screen passes", "study: criteria 1-2 only", "**study: full rule**"])
    lines = ["# Power of the pre-registered rule (computed before any run; amended 2026-11-27)", "",
             f"From `analysis/resolution_study.py power` (`simulate_full_rule`, {p['n_sim']:,} simulated studies per "
             f"row, seed {p['seed']}). Every check of the rule is simulated with the interval the report uses "
             "(Satterthwaite over the two seed variances and the bootstrap term): each target's criteria 1-3, the "
             "guard's two non-inferiority checks, and the six texture checks at 99%. The guard, the test floors and the "
             "texture slices are simulated at a true change of 0, so the guard and texture columns are the chance "
             "that a clause vetoes a real win by chance. Checks are independent draws. Inputs (`POWER_*` in the "
             "module): the targets' within-city held-out seed and bootstrap sds from October (context study, klone "
             "job 43403551); pooled test AUC from the context study's five-seed rerun; the CurbRamp guard from the "
             "backbone study's in-env baseline; texture slices scaled from one run's slice bootstrap. "
             "`study: criteria 1-2 only` is the quantity the first table reported.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in p["rows"]:
        cells = [str(r["seeds"]), f"+{r['effect']:.3f}"] + [f"{r[f'{t}_win']:.3f}" for t in tt]
        cells += [f"{r['guard']:.3f}", f"{r['texture']:.3f}", f"{r['study_crit12']:.3f}", f"**{r['study_win']:.3f}**"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Inputs, (seed sd, bootstrap sd of a paired difference):", ""]
    for k, v in p["inputs"].items():
        lines.append(f"- {k}: {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def cmd_power(args):
    p = power_table(n_sim=args.n_sim, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "power.md"), "w") as f:
        f.write(power_markdown(p))
    with open(os.path.join(args.out, "power.json"), "w") as f:
        json.dump(bc._clean(p), f, indent=2)
        f.write("\n")
    print(power_markdown(p))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------------------------------------------------

def _median_ms(fn, warmup, reps, sync):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench(resolutions=(BASELINE_RES, *ARMS, *LATER_ARMS), device=None, batches=(1, 16), warmup=3, reps=10,
          arch=None, image_px=640):
    """Median milliseconds per crop at each size: preprocessing (the slow ``BitImageProcessor`` on a 640 x 640 RGB
    image, as serving and evaluate.py run it) and the fp32 forward pass under ``torch.no_grad`` at each batch size.
    Weights are random (``backbones.build_classifier(..., random_init=True)``; ``arch`` overrides the registry's
    DINOv2-L config, e.g. a tiny one in tests), so no hub access is needed and the time is the weights-independent
    cost of the architecture at that size."""
    import platform

    import torch
    from PIL import Image
    from transformers import BitImageProcessor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    torch.manual_seed(0)
    model = bb.build_classifier(bb.DEFAULT, {0: "correct", 1: "incorrect"}, {"correct": 0, "incorrect": 1},
                                random_init=True, arch=arch).to(device).eval()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (image_px, image_px, 3), dtype=np.uint8))
    out = {"device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
           "torch": torch.__version__, "threads": torch.get_num_threads(), "warmup": warmup, "reps": reps,
           "attn_implementation": getattr(model.config, "_attn_implementation", None), "rows": []}
    for res in resolutions:
        proc = BitImageProcessor(**bb.preprocess_spec(bb.DEFAULT, res))
        pre = _median_ms(lambda: proc(img, return_tensors="pt"), warmup, reps, lambda: None)
        x1 = proc(img, return_tensors="pt")["pixel_values"]
        for batch in batches:
            x = x1.repeat(batch, 1, 1, 1).to(device)
            with torch.no_grad():
                ms = _median_ms(lambda: model(pixel_values=x), warmup, reps, sync)
            out["rows"].append({"res": res, "batch": batch, "tokens": bb.tokens_at(bb.DEFAULT, res),
                                "preprocess_ms": pre, "forward_ms_per_crop": ms / batch,
                                "total_ms_per_crop": pre + ms / batch})
    return out


def cmd_bench(args):
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    res = [int(r) for r in args.resolutions.split(",") if r]
    out = bench(res, args.device, tuple(int(b) for b in args.batches.split(",")), args.warmup, args.reps)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for r in out["rows"]:
        print(f"{out['device_name']} res {r['res']} batch {r['batch']}: {r['total_ms_per_crop']:.1f} ms per crop "
              f"(preprocess {r['preprocess_ms']:.1f}, forward {r['forward_ms_per_crop']:.1f})")
    return 0


def nightly_rows(benches, n_cities):
    """Per bench file and size, the per-crop time at batch 1 (the API's pattern) and the hours one nightly sweep at
    the actor's cap (``NIGHTLY_LABELS_PER_CITY`` x ``n_cities``) would take on the classifier alone (depth and
    cropping do not change with the size), if every city filled it."""
    rows = []
    crops = NIGHTLY_LABELS_PER_CITY * n_cities if n_cities else None
    for b in benches:
        for r in b["rows"]:
            if r["batch"] != 1:
                continue
            rows.append({"device": b["device_name"], "res": r["res"], "ms_per_crop": r["total_ms_per_crop"],
                         "crops_per_night": crops,
                         "hours_per_night": None if crops is None else crops * r["total_ms_per_crop"] / 3.6e6})
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------------------------------

def _iv(r):
    if not r or r.get("delta") is None:
        return ""
    if r.get("lo") is None:
        return f"{r['delta']:+.3f}"
    return f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def decision_markdown(decisions, comparisons, recommendation):
    lines = ["# Decision (pre-registered rule, docs/resolution-2026-11.md)", "", f"**Recommendation: {recommendation}.**",
             ""]
    for res, d in sorted(decisions.items()):
        lines += [f"## {res} px against the in-env 224 px rerun: {d['status']}", ""]
        if d["reasons"]:
            lines += ["Reasons: " + "; ".join(d["reasons"]) + ".", ""]
        if d["incomplete"]:
            lines += ["Incomplete: " + ", ".join(d["incomplete"]) + ".", ""]
        lines += ["| type | role | seeds (arm/224) | D within-city held-out [95%] | seeds above the 224 mean | "
                  "Δ test AUC, seed mean [95%] | mean Δ held-out AUC | D within-city test [95%] | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for t in TYPES:
            c = comparisons.get((t, res), {})
            role = "target" if t in TARGET_TYPES else ("guard" if t == GUARD_TYPE else "texture guard")
            if t in d["targets"]:
                verdict = d["targets"][t]["verdict"]
            elif t == GUARD_TYPE:
                verdict = {True: "pass", False: "FAIL", None: "INCOMPLETE"}[d["guard"]["passes"]]
            else:
                verdict = "descriptive"
            h = c.get("heldout") or {}
            lines.append(f"| {t} | {role} | {len(c.get('seeds_arm', []))}/{len(c.get('seeds_base', []))} | {_iv(h)} | "
                         f"{'' if h.get('n_above') is None else str(h['n_above']) + ' of ' + str(h['n_arm'])} | "
                         f"{_iv(c.get('test_auc'))} | {bc._f(c.get('mean_d_heldout_auc'), '{:+.3f}')} | "
                         f"{_iv(c.get('test'))} | {verdict} |")
        lines += ["", "Texture slices (harm screen at 99%: a slice falls when its interval lies wholly below zero):", "",
                  "| type | slice | split | n | train n | D [99%] | evaluable | falls |", "|---|---|---|---|---|---|---|---|"]
        for x in d["texture"]:
            lines.append(f"| {x['type']} | {x['slice']} | {x['split']} | {x.get('n') or ''} | {x.get('train_n') or ''} | "
                         f"{_iv(x)} | {'yes' if x['evaluable'] else 'no'} | {'YES' if x['falls'] else 'no'} |")
        if d["proposal"]:
            lines += ["", "Per-type proposal: " + ", ".join(f"{t} {r} px" for t, r in sorted(d["proposal"].items())) + "."]
        lines.append("")
    return "\n".join(lines)


def runs_markdown(comparisons, incomplete):
    lines = ["# Runs: every seed against the same-seed 224 px rerun", "",
             "Δ = arm − 224 on the same rows; intervals are per-seed panorama bootstraps (`backbone_comparison.paired`).",
             "", "| type | arm | seed | held-out AUC (arm / 224) | Δ held-out [95%] | within-city held-out (arm / 224) | "
             "Δ within-city held-out [95%] | test AUC (arm / 224) | Δ test [95%] |", "|---|---|---|---|---|---|---|---|---|"]
    for (t, res), c in sorted(comparisons.items()):
        for r in c["per_seed"]:
            lines.append(
                f"| {t} | {res} | {r['seed']} | {bc._f(r.get('arm_heldout_auc'))} / {bc._f(r.get('base_heldout_auc'))} | "
                f"{_iv({'delta': r.get('d_heldout'), 'lo': r.get('d_heldout_lo'), 'hi': r.get('d_heldout_hi')})} | "
                f"{bc._f(r.get('arm_wc_heldout_auc'))} / {bc._f(r.get('base_wc_heldout_auc'))} | "
                f"{_iv({'delta': r.get('wc_d_heldout'), 'lo': r.get('wc_d_heldout_lo'), 'hi': r.get('wc_d_heldout_hi')})} | "
                f"{bc._f(r.get('arm_test_auc'))} / {bc._f(r.get('base_test_auc'))} | "
                f"{_iv({'delta': r.get('d_test'), 'lo': r.get('d_test_lo'), 'hi': r.get('d_test_hi')})} |")
    lines += ["", "## Held-out deployments (seed means; private deployments lettered)", "",
              "| type | arm | deployment | AUC arm | AUC 224 | Δ |", "|---|---|---|---|---|---|"]
    for (t, res), c in sorted(comparisons.items()):
        for d in c["deployments"]:
            lines.append(f"| {t} | {res} | {d['deployment']} | {bc._f(d['auc_arm'])} | {bc._f(d['auc_base'])} | "
                         f"{bc._f(d['delta'], '{:+.3f}')} |")
    if incomplete:
        lines += ["", "## Not counted", "", "| type | tag | why |", "|---|---|---|"]
        lines += [f"| {t} | `{tag}` | {why} |" for t, tag, why in incomplete]
    return "\n".join(lines) + "\n"


def slices_markdown(comparisons):
    lines = ["# Slices: type x severity and type x tag (seed means, arm − 224)", "",
             "Every slice with its training rows; `thin` = under 100 training rows. D and its interval as the primary "
             "endpoint's (seed variance plus panorama bootstrap), on the slice's pooled AUC; blank under 30 rows or 8 "
             "of a class. Not corrected for the number of slices. Texture slices are marked; only they enter the rule.",
             "", "| type | arm | by | slice | split | train n | thin | n | AUC 224 | AUC arm | D [95%] | seeds above |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = {(t, s) for t, _, s in TEXTURE_SLICES} | {(t, s) for t, _, s in WATCH_SLICES}
    for (t, res), c in sorted(comparisons.items()):
        for r in c["slices"]:
            mark = " (texture)" if (t, r["slice"]) in {(a, s) for a, _, s in TEXTURE_SLICES} else (
                " (watch)" if (t, r["slice"]) in tex else "")
            lines.append(f"| {t} | {res} | {r['by']} | {r['slice']}{mark} | {r['split']} | {r['train_n']} | "
                         f"{'yes' if r['thin'] else ''} | {r.get('n', '')} | {bc._f(r.get('auc_base'))} | "
                         f"{bc._f(r.get('auc_arm'))} | {_iv(r)} | "
                         f"{'' if r.get('n_above') is None else str(r['n_above']) + ' of ' + str(r['n_arm'])} |")
    return "\n".join(lines) + "\n"


def cost_markdown(cost_rows, nightly):
    lines = ["# Cost: training and serving", "",
             "Training: seconds per optimizer step over each run's last attempt (loading and per-epoch evaluation "
             "included), GPU-hours ≈ that rate × the run's steps, peak GPU memory from `training_run.json`.", "",
             "| type | res | seed | GPUs | s / step | steps | GPU-hours | peak GPU GB | gradient checkpointing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in cost_rows:
        lines.append(f"| {r['type']} | {r['res']} | {r['seed']} | {r['gpu']} | {bc._f(r['s_per_step'], '{:.2f}')} | "
                     f"{r['steps'] or ''} | {bc._f(r['gpu_hours'], '{:.1f}')} | {bc._f(r['peak_gb'], '{:.1f}')} | "
                     f"{r['gc']} |")
    lines += ["", "Serving: per-crop latency at batch 1 (the API's pattern; preprocessing included) and one nightly "
              f"sweep at the actor's cap ({NIGHTLY_LABELS_PER_CITY} labels per city, all types) across the cities listed. "
              "Descriptive: no latency threshold is pre-registered (docs, \"Latency\").", "",
              "| device | res | ms per crop | crops per night (cap) | classifier hours per night |", "|---|---|---|---|---|"]
    for r in nightly:
        lines.append(f"| {r['device']} | {r['res']} | {r['ms_per_crop']:.1f} | {r['crops_per_night'] or ''} | "
                     f"{bc._f(r['hours_per_night'], '{:.2f}')} |")
    return "\n".join(lines) + "\n"


def env_markdown(env):
    lines = ["# Environment readout (not a gate)", "",
             "The in-env 224 px rerun against round 3's own baseline at the same seed (where round 3's predictions are "
             "pinned), on the same rows.", "",
             "| type | seed | held-out AUC (rerun / round 3) | Δ held-out [95%] | test AUC (rerun / round 3) | Δ test [95%] |",
             "|---|---|---|---|---|---|"]
    for r in env:
        lines.append(f"| {r['type']} | {r['seed']} | {bc._f(r.get('heldout_auc'))} / {bc._f(r.get('round3_heldout_auc'))} | "
                     f"{_iv({'delta': r.get('d_heldout'), 'lo': r.get('d_heldout_lo'), 'hi': r.get('d_heldout_hi')})} | "
                     f"{bc._f(r.get('test_auc'))} / {bc._f(r.get('round3_test_auc'))} | "
                     f"{_iv({'delta': r.get('d_test'), 'lo': r.get('d_test_lo'), 'hi': r.get('d_test_hi')})} |")
    return "\n".join(lines) + "\n"


def gate_markdown(gates):
    lines = ["# The 0.92 gate under `majority2` (descriptive)", "",
             "Agree ('correct') and Disagree ('incorrect') coverage and precision at the gate, val-fitted uncapped map "
             "(`prior_rules.gate_rows`), seed means.", "",
             "| type | res | split | verdict | seeds | coverage | precision | gated n |", "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for g in gates:
        agg.setdefault((g["type"], g["res"], g["split"], g["verdict"]), []).append(g)
    for (t, res, split, v), rows in sorted(agg.items()):
        def m(k):
            vals = [r.get(k) for r in rows if r.get(k) is not None and not (isinstance(r.get(k), float) and np.isnan(r[k]))]
            return float(np.mean(vals)) if vals else None
        lines.append(f"| {t} | {res} | {split} | {v} | {len(rows)} | {bc._f(m('gate_coverage'))} | "
                     f"{bc._f(m('gate_precision'))} | {bc._f(m('gate_n'), '{:.0f}')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def cost_rows(label_type, runs):
    out = []
    for r in sorted(runs.values(), key=lambda r: (r["parsed"]["res"], r["parsed"]["seed"])):
        if r["status"] != "complete":
            continue
        sps = bc.seconds_per_step(r)
        steps = (r.get("state") or {}).get("max_steps")
        out.append({"type": label_type, "res": r["parsed"]["res"], "seed": r["parsed"]["seed"], "gpu": "; ".join(r["gpu"]),
                    "s_per_step": sps, "steps": steps,
                    "gpu_hours": sps * steps / 3600 if sps and steps else None,
                    "peak_gb": (r["run"].get("peak_gpu_memory_bytes") or 0) / 2**30 or None,
                    "gc": "yes" if r["run"].get("gradient_checkpointing") else "no"})
    return out


def env_rows(label_type, runs, ref, n_boot, seed):
    out = []
    for s, rec in sorted(_complete(runs, BASELINE_RES).items()):
        if s not in ref:
            continue
        m, r3 = bc.run_metrics(rec["pred"]), bc.run_metrics(ref[s])
        out.append({"type": label_type, "seed": s, "test_auc": m.get("test_auc"), "heldout_auc": m.get("heldout_auc"),
                    "round3_test_auc": r3.get("test_auc"), "round3_heldout_auc": r3.get("heldout_auc"),
                    **bc.paired(rec["pred"], ref[s], rec["tag"], n_boot, seed)})
    return out


def cmd_report(args):
    from analysis import retrain_report as rr
    from fine_tune_classifier import git_state
    tracker = bc.Tracker()
    inputs = bc.load_inputs(tracker.track(args.inputs))
    with open(tracker.track(args.shares)) as f:
        shares_types = json.load(f)["types"]
    public, public_source = rr.public_cities(args.public_cities)
    n_cities = None
    if args.public_cities:
        tracker.track(args.public_cities)
        try:
            with open(args.public_cities) as f:
                n_cities = len(json.load(f)["cities"])
        except (ValueError, KeyError):
            n_cities = None
    types = [t for t in args.label_types.split(",") if t]
    unknown = [t for t in types if t not in TYPES]
    if unknown:
        raise SystemExit(f"--label-types {unknown}: not study types {TYPES}")
    arms = [int(a) for a in args.arms.split(",") if a]
    comparisons, incomplete, env, costs, gates = {}, [], [], [], []
    for t in types:
        split = bc.load_split(args.split_dir, t, inputs, tracker)
        runs = load_runs(args.runs_dir, t, args.tag_prefix, split, inputs["split_sha256"][t], tracker, args.epochs,
                         args.allow_batch, args.allow_gc)
        incomplete += [[t, tag, "; ".join(r["why"])] for tag, r in runs.items() if r["status"] != "complete"]
        train = split[split["split"] == "train"]
        for res in arms:
            comparisons[(t, res)] = compare_type(t, runs, res, train, public, args.n_boot, args.seed, args.min_boot)
        env += env_rows(t, runs, bc.round3_reference(args.split_dir, t, split, tracker), args.n_boot, args.seed)
        costs += cost_rows(t, runs)
        for r in runs.values():
            if r["status"] != "complete":
                continue
            g = bc.gate_stats(r["pred"], shares_types, t, "majority2")
            for split_name, classes in g.items():
                for verdict, st in classes.items():
                    if isinstance(st, dict):
                        gates.append({"type": t, "res": r["parsed"]["res"], "seed": r["parsed"]["seed"],
                                      "split": split_name, "verdict": verdict, **st})
    decisions = {res: decide_arm({t: comparisons[(t, res)] for t in types}) for res in arms}
    recommendation = recommend(decisions)
    benches = []
    for p in args.bench or []:
        with open(tracker.track(p)) as f:
            benches.append(json.load(f))
    os.makedirs(args.out, exist_ok=True)
    outputs = {"decision.md": decision_markdown(decisions, comparisons, recommendation),
               "runs.md": runs_markdown(comparisons, incomplete), "slices.md": slices_markdown(comparisons),
               "cost.md": cost_markdown(costs, nightly_rows(benches, n_cities)), "env.md": env_markdown(env),
               "gate.md": gate_markdown(gates)}
    for name, text in outputs.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    decision_json = {"rule": DECISION_RULE, "recommendation": recommendation,
                     "decisions": {str(k): v for k, v in decisions.items()},
                     "comparisons": {f"{t}:{res}": {k: v for k, v in c.items() if k != "slices"}
                                     for (t, res), c in comparisons.items()},
                     "incomplete": incomplete}
    with open(os.path.join(args.out, "decision.json"), "w") as f:
        json.dump(bc._clean(decision_json), f, indent=2, default=str)
        f.write("\n")
    prov = {"files_read": bc.provenance_keys(tracker.files, {"runs-dir": args.runs_dir, "split-dir": args.split_dir}),
            "outputs": {n: bc.sha256_of(os.path.join(args.out, n)) for n in sorted([*outputs, "decision.json"])},
            "report_git": git_state(), "n_boot": args.n_boot, "bootstrap_seed": args.seed, "tag_prefix": args.tag_prefix,
            "public_cities_source": os.path.basename(public_source) if args.public_cities else public_source,
            "label_types": types, "arms": arms}
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"wrote {len(outputs) + 2} files to {args.out}; recommendation: {recommendation}")
    for t, tag, why in incomplete:
        print(f"INCOMPLETE {t} {tag}: {why}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("power", help="the rule's operating characteristics (before any run)")
    w.add_argument("--out", required=True)
    w.add_argument("--n-sim", type=int, default=20000)
    w.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bench", help="per-crop latency at each size (random weights, no hub access)")
    b.add_argument("--out", required=True)
    b.add_argument("--device", default=None)
    b.add_argument("--resolutions", default=",".join(str(r) for r in (BASELINE_RES, *ARMS, *LATER_ARMS)))
    b.add_argument("--batches", default="1,16")
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--reps", type=int, default=10)
    b.add_argument("--threads", type=int, default=None, help="CPU threads (torch.set_num_threads)")
    r = sub.add_parser("report", help="the tables and the decision from the runs' records")
    r.add_argument("--split-dir", required=True, help="holds <LT>/split.csv and round 3's reference predictions")
    r.add_argument("--runs-dir", required=True, help="the $KEEP layout: <LT>/runs/<TAG>/...")
    r.add_argument("--inputs", default=INPUTS)
    r.add_argument("--shares", default=SHARES)
    r.add_argument("--public-cities", default=None, help="a saved /v3/api/cities listing (default: the live API)")
    r.add_argument("--label-types", default=",".join(TYPES))
    r.add_argument("--arms", default=",".join(str(a) for a in ARMS), help="arm sizes to judge (518 once it has run)")
    r.add_argument("--tag-prefix", default=PREFIX)
    r.add_argument("--bench", action="append", help="a bench JSON (repeatable)")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    r.add_argument("--allow-gc", action="store_true",
                   help="count runs trained with gradient checkpointing (same maths); a recorded deviation")
    r.add_argument("--min-boot", type=int, default=DECISION_RULE["min_boot"],
                   help="kept bootstrap resamples an interval needs (below: no verdict); lower only in tests")
    r.add_argument("--allow-batch", action="store_true",
                   help="count runs trained at another per-device batch (same effective batch); a recorded deviation")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"power": cmd_power, "bench": cmd_bench, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
